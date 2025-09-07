import argparse
import os
import sys
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd

# Ensure Django is configured when this module is executed directly
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "mysite.settings")
try:
    import django  # type: ignore

    django.setup()
except Exception:
    # If imported by Django (e.g., management command context), setup may be redundant
    pass

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    balanced_accuracy_score,
)

from dashboard.models import CsvData
from dashboard.logic import cache
from dashboard.logic.machine_learning.settings import prediction_name, scale_name


def _load_df_for_csv(csv: CsvData) -> Optional[pd.DataFrame]:
    """Load prediction DataFrame for a given CsvData.

    Preference order:
    - Pickled DataFrame at `cached_prediction_path` (fast, includes predictions)
    - Excel at `excel_prediction_path` (slower)
    - Excel at `x_data_path` (features/labels only; may be missing predictions)
    """
    if os.path.exists(csv.cached_prediction_path):
        try:
            return cache.load_obj(csv.cached_prediction_path)
        except Exception:
            pass

    if os.path.exists(csv.excel_prediction_path):
        try:
            return pd.read_excel(csv.excel_prediction_path, index_col=0)
        except Exception:
            pass

    if os.path.exists(csv.x_data_path):
        try:
            return pd.read_excel(csv.x_data_path, index_col=0)
        except Exception:
            pass

    return None


def _sanitize_labels(y: pd.Series) -> np.ndarray:
    y = pd.to_numeric(y, errors="coerce").fillna(-1).astype(int)
    # Clamp to {0,1}; any negative or >1 become -1 (ignored later)
    y = y.where((y == 0) | (y == 1), other=-1)
    return y.values


def validate_dreamt_predictions(
        threshold: float = 0.5,
        save_report: Optional[str] = None,
        include_per_file: bool = True,
) -> Tuple[pd.DataFrame, dict]:
    """Validate model predictions against DREAMT labels.

    - Finds all `CsvData` with `dreamt_data=True`.
    - Loads their prediction DataFrames.
    - Compares `{scale_name}` vs `{prediction_name}` with a threshold.

    Returns a tuple of (per_file_df, overall_metrics_dict).
    Optionally writes a CSV report to `save_report`.
    """
    rows: List[dict] = []
    y_true_all: List[int] = []
    y_pred_all: List[int] = []
    y_score_all: List[float] = []

    qs = CsvData.objects.filter(dreamt_data=True)
    for csv in qs:
        df = _load_df_for_csv(csv)
        if df is None:
            rows.append({
                "file": csv.filename,
                "n_samples": 0,
                "status": "missing_df",
            })
            continue

        if scale_name not in df.columns:
            rows.append({
                "file": csv.filename,
                "n_samples": 0,
                "status": f"missing_label_col:{scale_name}",
            })
            continue

        if prediction_name not in df.columns:
            rows.append({
                "file": csv.filename,
                "n_samples": 0,
                "status": f"missing_pred_col:{prediction_name}",
            })
            continue

        y_true = _sanitize_labels(df[scale_name])
        y_score = pd.to_numeric(df[prediction_name], errors="coerce").astype(float).values
        valid_mask = (y_true >= 0) & np.isfinite(y_score)

        if not np.any(valid_mask):
            rows.append({
                "file": csv.filename,
                "n_samples": 0,
                "status": "no_valid_rows",
            })
            continue

        y_true_v = y_true[valid_mask]
        y_score_v = y_score[valid_mask]
        y_pred_v = (y_score_v >= threshold).astype(int)

        # Collect for global metrics
        y_true_all.append(y_true_v)
        y_pred_all.append(y_pred_v)
        y_score_all.append(y_score_v)

        # Per-file metrics
        try:
            acc = accuracy_score(y_true_v, y_pred_v)
            prec = precision_score(y_true_v, y_pred_v, zero_division=0)
            rec = recall_score(y_true_v, y_pred_v, zero_division=0)
            f1 = f1_score(y_true_v, y_pred_v, zero_division=0)
            auc = roc_auc_score(y_true_v, y_score_v) if len(np.unique(y_true_v)) == 2 else np.nan
            bal_acc = balanced_accuracy_score(y_true_v, y_pred_v)
            tn, fp, fn, tp = confusion_matrix(y_true_v, y_pred_v).ravel()
            spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan
            status = "ok"
        except Exception as e:
            acc = prec = rec = f1 = auc = bal_acc = spec = np.nan
            tp = tn = fp = fn = 0
            status = f"error:{type(e).__name__}"

        if include_per_file:
            rows.append({
                "file": csv.filename,
                "n_samples": int(valid_mask.sum()),
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1": f1,
                "auc": auc,
                "balanced_accuracy": bal_acc,
                "specificity": spec,
                "tp": int(tp),
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "status": status,
            })

    # Aggregate metrics
    if y_true_all:
        y_true_cat = np.concatenate(y_true_all)
        y_pred_cat = np.concatenate(y_pred_all)
        y_score_cat = np.concatenate(y_score_all)
        overall = {
            "files": len(qs),
            "evaluated_files": sum(1 for r in rows if r.get("status") in ("ok",) or r.get("n_samples", 0) > 0),
            "samples": int(len(y_true_cat)),
            "accuracy": float(accuracy_score(y_true_cat, y_pred_cat)),
            "precision": float(precision_score(y_true_cat, y_pred_cat, zero_division=0)),
            "recall": float(recall_score(y_true_cat, y_pred_cat, zero_division=0)),
            "f1": float(f1_score(y_true_cat, y_pred_cat, zero_division=0)),
            "auc": float(roc_auc_score(y_true_cat, y_score_cat)) if len(np.unique(y_true_cat)) == 2 else np.nan,
            "balanced_accuracy": float(balanced_accuracy_score(y_true_cat, y_pred_cat)),
        }
        cm = confusion_matrix(y_true_cat, y_pred_cat)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            overall.update({
                "tp": int(tp),
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else np.nan,
            })
    else:
        overall = {
            "files": len(qs),
            "evaluated_files": 0,
            "samples": 0,
            "accuracy": np.nan,
            "precision": np.nan,
            "recall": np.nan,
            "f1": np.nan,
            "auc": np.nan,
            "balanced_accuracy": np.nan,
            "tp": 0,
            "tn": 0,
            "fp": 0,
            "fn": 0,
            "specificity": np.nan,
        }

    per_file_df = pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=[
            "file",
            "n_samples",
            "accuracy",
            "precision",
            "recall",
            "f1",
            "auc",
            "balanced_accuracy",
            "specificity",
            "tp",
            "tn",
            "fp",
            "fn",
            "status",
        ]
    )

    if save_report:
        out_dir = os.path.dirname(save_report)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)
        # Write an overall summary row at top via a small CSV with two sections
        with open(save_report, "w", encoding="utf-8") as f:
            # Overall block
            f.write("metric,value\n")
            for k, v in overall.items():
                f.write(f"{k},{v}\n")
            f.write("\n")
        # Append per-file block
        per_file_path = save_report.replace(".csv", "_per_file.csv")
        per_file_df.to_csv(per_file_path, index=False)

    return per_file_df, overall


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Validate DREAMT predictions against labels.")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Decision threshold for positive class (default 0.5)")
    parser.add_argument(
        "--report",
        type=str,
        default=os.path.join("media", "predictions", "validation_report.csv"),
        help="Path to write overall metrics CSV and per-file CSV (suffix _per_file.csv)",
    )
    parser.add_argument("--no-per-file", action="store_true", help="Do not include per-file rows in memory return")

    args = parser.parse_args(argv)

    per_file_df, overall = validate_dreamt_predictions(
        threshold=args.threshold,
        save_report=args.report,
        include_per_file=not args.no_per_file,
    )

    # Print concise summary to stdout for quick checks
    print("Overall metrics:")
    for k in [
        "files",
        "evaluated_files",
        "samples",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "auc",
        "balanced_accuracy",
        "specificity",
        "tp",
        "tn",
        "fp",
        "fn",
    ]:
        print(f"  {k}: {overall.get(k)}")

    print(f"\nReport written to: {args.report} (overall) and {args.report.replace('.csv', '_per_file.csv')} (per-file)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
