"""Separate, exploratory paired feature-block classification experiment.

Reuses the established model pipelines/search spaces without re-running data
preparation or changing the canonical manuscript classification runs.
"""

import json
import logging
import platform
import re
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone

from dashboard.logic import classification_person_grouped_validation as engine
from dashboard.logic.state_activity_classification_data import (
    ADDED_FEATURES, ADJUSTMENT_COLUMNS, AUGMENTED, BASELINE, FEATURE_SETS, load_experiment_inputs,
)
from dashboard.logic.state_activity_export import _fingerprint
from mysite.settings import BASE_DIR, MEDIA_ROOT

logger = logging.getLogger(__name__)
PROTOCOL = Path(BASE_DIR) / "report/state_activity_classification.md"


def _write_json(path, value):
    def encode(item):
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, (Path, set, frozenset, tuple)):
            return list(item) if not isinstance(item, Path) else str(item)
        raise TypeError(f"Unsupported JSON value: {type(item)}")

    path.write_text(json.dumps(value, indent=2, default=encode, allow_nan=False) + "\n")


def checked_splits(y, groups, maximum):
    cv = engine._build_stratified_group_cv(y, groups, maximum)
    splits = list(cv.split(np.zeros((len(y), 1)), y, groups))
    for train, test in splits:
        if set(groups[train]) & set(groups[test]):
            raise ValueError("Participant leakage in a validation split")
        if set(y[train]) != {0, 1} or set(y[test]) != {0, 1}:
            raise ValueError("A grouped fold lacks one diagnosis group")
    return splits


def split_plan(prepared):
    """Persist both outer and inner memberships before fitting any model."""
    rows = []
    for fold, (train, test) in enumerate(checked_splits(
            prepared["y"], prepared["groups"], engine.PERSON_GROUPED_OUTER_CV_SPLITS), 1):
        rows.extend({"outer_fold": fold, "inner_fold": 0, "partition": role,
                     "#Subject": str(prepared["subjects"][index]),
                     "person_group": str(prepared["groups"][index])}
                    for role, indices in (("train", train), ("test", test)) for index in indices)
        for inner, (inner_train, inner_test) in enumerate(checked_splits(
                prepared["y"][train], prepared["groups"][train], engine.PERSON_GROUPED_INNER_CV_SPLITS), 1):
            rows.extend({"outer_fold": fold, "inner_fold": inner, "partition": role,
                         "#Subject": str(prepared["subjects"][train[index]]),
                         "person_group": str(prepared["groups"][train[index]])}
                        for role, indices in (("train", inner_train), ("test", inner_test)) for index in indices)
    return pd.DataFrame(rows)


def model_input(data, feature_set):
    table = data["table"]
    features = list(data["baseline_features"])
    if feature_set == AUGMENTED:
        features.extend(ADDED_FEATURES)
    elif feature_set != BASELINE:
        raise ValueError(f"Unknown feature set: {feature_set}")
    return {"X": table[features + list(ADJUSTMENT_COLUMNS)].apply(pd.to_numeric, errors="raise").replace(
        [np.inf, -np.inf], np.nan).to_numpy(dtype=float),
            "y": table.y_true.to_numpy(dtype=int), "subjects": table["#Subject"].to_numpy(dtype=str),
            "groups": table.person_group.to_numpy(dtype=str),
            "source_cohorts": table.source_cohort.to_numpy(dtype=str),
            "visit_indices": table.visit_index.to_numpy(dtype=int),
            "feature_columns": features, "adjustment_columns": list(ADJUSTMENT_COLUMNS)}


def _spec(model):
    return {"run_key": model, "run_label": model, "model_family": model,
            "feature_selector_mode": (engine.FEATURE_SELECTOR_MODE_KBEST if model == engine.MODEL_FAMILY_XGBOOST
                                      else engine.FEATURE_SELECTOR_MODE_EMBEDDED)}


def _fit_nested(prepared, spec, plan):
    predictions, folds, coefficients = engine._run_grouped_nested_cv(
        X=prepared["X"], y=prepared["y"], groups=prepared["groups"], subjects=prepared["subjects"],
        source_cohorts=prepared["source_cohorts"], visit_indices=prepared["visit_indices"],
        model_family=spec["model_family"], feature_selector_mode=spec["feature_selector_mode"],
        n_covariates=len(ADJUSTMENT_COLUMNS), feature_names=prepared["feature_columns"],
        outer_splits=engine.PERSON_GROUPED_OUTER_CV_SPLITS, inner_splits=engine.PERSON_GROUPED_INNER_CV_SPLITS)
    predictions = predictions.set_index("#Subject").loc[prepared["subjects"]].reset_index()
    expected = plan.loc[(plan.inner_fold == 0) & plan.partition.eq("test")].set_index("#Subject").outer_fold
    if not np.array_equal(predictions["fold"], predictions["#Subject"].map(expected)):
        raise ValueError("Fitted outer folds differ from the saved common split plan")
    predictions["diagnosis_label"] = predictions.y_true.map({0: "HC", 1: "MCI-LB"})
    return predictions, folds, coefficients


def paired_feature_comparison(baseline, augmented, n_bootstrap):
    """Percentile CIs, participant-resampled within diagnosis; not p-values.

    All visits of a sampled person are included together, with multiplicity.
    As in the original audit, this conditions on saved fitted OOF predictions.
    """
    identity = ["#Subject", "person_group", "source_cohort", "visit_index", "fold", "y_true"]
    for frame in (baseline, augmented):
        if frame["#Subject"].duplicated().any():
            raise ValueError("Duplicate held-out visits")
        engine._validate_person_groups(frame.person_group, frame.y_true)
    columns = identity + ["y_pred_default", "pred_probability_positive"]
    paired = baseline[columns].merge(augmented[columns], on=identity, suffixes=("_base", "_added"),
                                     how="inner", validate="one_to_one")
    if len(paired) != len(baseline) or len(paired) != len(augmented):
        raise ValueError("Paired comparison requires identical visits, labels, participants and folds")
    if n_bootstrap < 1 or set(paired.y_true) != {0, 1}:
        raise ValueError("Bootstrap requires positive resample count and two diagnosis groups")

    def values(sample, suffix):
        return engine._comparison_metric_values(sample.y_true, sample[f"y_pred_default_{suffix}"],
                                                sample[f"pred_probability_positive_{suffix}"])

    base, added = values(paired, "base"), values(paired, "added")
    indices = {person: group.index.to_numpy() for person, group in paired.groupby("person_group", sort=False)}
    people = paired[["person_group", "y_true"]].drop_duplicates()
    strata = [people.loc[people.y_true.eq(label), "person_group"].to_numpy() for label in (0, 1)]
    rng = np.random.default_rng(engine.SEED)
    differences = {metric: [] for metric in base}
    for _ in range(n_bootstrap):
        sampled = [person for stratum in strata for person in rng.choice(stratum, len(stratum), replace=True)]
        sample = paired.loc[np.concatenate([indices[person] for person in sampled])]
        a, b = values(sample, "base"), values(sample, "added")
        for metric in base:
            differences[metric].append(b[metric] - a[metric])
    return pd.DataFrame([{
        "metric": metric, "baseline_value": base[metric], "augmented_value": added[metric],
        "difference_augmented_minus_baseline": added[metric] - base[metric],
        "difference_ci_low": float(np.quantile(differences[metric], .025)),
        "difference_ci_high": float(np.quantile(differences[metric], .975)),
        "higher_is_better": metric not in ("BRIER", "LOG_LOSS"), "visits": len(paired),
        "people": len(people), "bootstrap_replicates": n_bootstrap,
        "bootstrap_unit": "person, stratified by diagnosis; all visits retained",
    } for metric in base])


def _metrics(predictions, context, prediction_column="y_pred_default"):
    return {"context": context, "visits": len(predictions), "people": predictions.person_group.nunique(),
            **engine._binary_metrics(predictions.y_true, predictions[prediction_column],
                                     predictions.pred_probability_positive)}


def _source_checks(prepared, spec, destination):
    """Source-stratified OOF summaries are separate from transportability fits."""
    rows = []
    for source in sorted(set(prepared["source_cohorts"])):
        # Established source labels include slashes (e.g. HC/HC2); never use
        # them verbatim as directory paths. Keep the original label in CSVs.
        source_key = re.sub(r"[^A-Za-z0-9_-]+", "_", str(source))
        mask = prepared["source_cohorts"] == source
        subset = {key: value[mask] if isinstance(value, np.ndarray) else value for key, value in prepared.items()}
        try:
            plan = split_plan(subset)
        except ValueError as error:
            rows.append({"source": source, "validation": "within_source", "status": "skipped", "reason": str(error)})
        else:
            predictions, folds, coefficients = _fit_nested(subset, spec, plan)
            directory = destination / f"within_{source_key}"
            directory.mkdir()
            plan.to_csv(directory / "split_plan.csv", index=False)
            predictions.to_csv(directory / "predictions.csv", index=False)
            folds.to_csv(directory / "folds.csv", index=False)
            if not coefficients.empty:
                coefficients.to_csv(directory / "coefficients.csv", index=False)
            rows.append({"source": source, "validation": "within_source", "status": "complete",
                         **_metrics(predictions, "default_threshold")})
        # Holding out a whole collection is participant-disjoint (checked in preparation).
        train = ~mask
        try:
            checked_splits(prepared["y"][train], prepared["groups"][train], engine.PERSON_GROUPED_INNER_CV_SPLITS)
            if set(prepared["y"][mask]) != {0, 1}:
                raise ValueError("Held-out source does not contain both diagnosis groups")
        except ValueError as error:
            rows.append({"source": source, "validation": "leave_source_out", "status": "skipped", "reason": str(error)})
            continue
        cv = engine._build_stratified_group_cv(prepared["y"][train], prepared["groups"][train],
                                               engine.PERSON_GROUPED_INNER_CV_SPLITS)
        model, search = engine._run_grouped_search(
            X=prepared["X"][train], y=prepared["y"][train], groups=prepared["groups"][train], cv=cv,
            model_family=spec["model_family"], feature_selector_mode=spec["feature_selector_mode"],
            n_covariates=len(ADJUSTMENT_COLUMNS))
        model = engine._fit_with_device_fallback(clone(model), prepared["X"][train], prepared["y"][train])
        predictions = pd.DataFrame({"#Subject": prepared["subjects"][mask],
                                    "person_group": prepared["groups"][mask], "y_true": prepared["y"][mask],
                                    "y_pred_default": model.predict(prepared["X"][mask]).astype(int),
                                    "pred_probability_positive": engine._predict_positive_probability(model,
                                                                                                      prepared["X"][
                                                                                                          mask])})
        predictions.to_csv(destination / f"leave_out_{source_key}_predictions.csv", index=False)
        _write_json(destination / f"leave_out_{source_key}_search.json", {
            "best_params": search.best_params_, "best_inner_score": float(search.best_score_)})
        rows.append({"source": source, "validation": "leave_source_out", "status": "complete",
                     **_metrics(predictions, "default_threshold")})
    pd.DataFrame(rows).to_csv(destination / "source_validation_metrics.csv", index=False)


def run_state_activity_classification(state_dir, target_dir, output_dir=None, prepare_only=False,
                                      models=engine.MODEL_FAMILIES, n_bootstrap=10000):
    models = tuple(models)
    if not models or len(set(models)) != len(models) or set(models) - set(engine.MODEL_FAMILIES):
        raise ValueError("Select each supported model family at most once")
    if n_bootstrap < 100:
        raise ValueError("At least 100 paired bootstrap resamples are required")
    data, fingerprints, config = load_experiment_inputs(state_dir, target_dir)
    prepared = model_input(data, BASELINE)
    plan = split_plan(prepared)
    destination = Path(output_dir).resolve() if output_dir else Path(
        MEDIA_ROOT) / "classification/state-activity" / datetime.now(
        timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    destination.mkdir(parents=True, exist_ok=False)
    manifest = {
        "status": "preparing", "started_utc": datetime.now(timezone.utc).isoformat(),
        "scenario_codes": {"negative": 0, "positive": 3}, "scenario": "HC vs MCI-LB", "seed": engine.SEED,
        "visits": len(data["table"]), "people": data["table"].person_group.nunique(),
        "diagnosis_visit_counts": data["table"].diagnosis_code.value_counts().to_dict(),
        "source_by_diagnosis": pd.crosstab(data["table"].source_cohort, data["table"].diagnosis_code).to_dict(),
        "adjustment_columns": list(ADJUSTMENT_COLUMNS),
        "adjustment_scope": "training fold only; not appended as predictors",
        "feature_sets": {BASELINE: data["baseline_features"],
                         AUGMENTED: data["baseline_features"] + list(ADDED_FEATURES)},
        "input_files": fingerprints, "state_config": config, "models": models,
        "outer_max_splits": engine.PERSON_GROUPED_OUTER_CV_SPLITS,
        "inner_max_splits": engine.PERSON_GROUPED_INNER_CV_SPLITS,
        "search_settings": {model: engine._search_settings_for_model(model, _spec(model)["feature_selector_mode"])
                            for model in models},
        "bootstrap_replicates": n_bootstrap,
        "implementation": [_fingerprint(Path(BASE_DIR) / path) for path in (
            "dashboard/logic/state_activity_classification.py", "dashboard/logic/state_activity_classification_data.py",
            "dashboard/logic/classification_person_grouped_validation.py",
            "dashboard/logic/classification_grouped_statistics.py",
            "dashboard/logic/classification_grouped_statistics_strict.py",
            "dashboard/logic/classification_validity_checks.py",
            "dashboard/logic/feature_families.py", "dashboard/logic/group_data.py",
            "dashboard/logic/state_activity_features.py", "dashboard/logic/state_activity_targeted_features.py",
            "dashboard/logic/state_activity_rbdq.py", "dashboard/logic/state_activity_export.py",
            "dashboard/logic/xgboost_runtime.py",
            "dashboard/management/commands/run_state_activity_classification.py")],
        "xgboost_runtime": engine.xgboost_runtime_metadata(),
        "versions": {name: version(name) for name in ("numpy", "pandas", "scipy", "scikit-learn", "xgboost", "Django")},
        "python": platform.python_version(), "protocol": PROTOCOL.read_text(),
        "interpretation": "Post hoc exploratory feature-block comparison, not independent or external validation. "
                          "Bootstrap CIs condition on saved OOF predictions; models are not retrained in the bootstrap.",
    }
    _write_json(destination / "manifest.json", manifest)
    try:
        for key in ("table", "cohort_audit", "legacy_audit", "family_audit", "matched_nights"):
            data[key].to_csv(destination / f"{key}.csv", index=False)
        plan.to_csv(destination / "split_plan.csv", index=False)
        manifest["prepared_files"] = [_fingerprint(destination / f"{key}.csv") for key in
                                      ("table", "cohort_audit", "legacy_audit", "family_audit", "matched_nights",
                                       "split_plan")]
        if prepare_only:
            manifest["status"] = "prepared_only"
        else:
            metrics, comparisons = [], []
            for model in models:
                pair = {}
                for feature_set in FEATURE_SETS:
                    logger.info("State activity classifier: %s / %s", model, feature_set)
                    run_dir = destination / model / feature_set
                    run_dir.mkdir(parents=True)
                    prepared = model_input(data, feature_set)
                    predictions, folds, coefficients = _fit_nested(prepared, _spec(model), plan)
                    predictions.to_csv(run_dir / "predictions.csv", index=False)
                    folds.to_csv(run_dir / "folds.csv", index=False)
                    if not coefficients.empty:
                        coefficients.to_csv(run_dir / "coefficients.csv", index=False)
                    pair[feature_set] = predictions
                    metrics.append(
                        {"model": model, "feature_set": feature_set, **_metrics(predictions, "primary_default")})
                    secondary = [_metrics(predictions, "tuned_threshold", "y_pred_tuned"),
                                 _metrics(predictions.loc[predictions.is_first_visit], "first_visit_default")]
                    secondary.extend(_metrics(group, f"source_{source}_default") for source, group in
                                     predictions.groupby("source_cohort", sort=True))
                    pd.DataFrame(secondary).to_csv(run_dir / "secondary_metrics.csv", index=False)
                    if model == models[0] and feature_set == BASELINE:
                        source_metrics, source_predictions = engine._source_only_ascertainment_benchmark(
                            predictions, engine.PERSON_GROUPED_OUTER_CV_SPLITS)
                        source_predictions["fold"] = predictions["fold"].to_numpy()
                        source_predictions.to_csv(destination / "source_only_predictions.csv", index=False)
                        pd.DataFrame([source_metrics]).to_csv(destination / "source_only_metrics.csv", index=False)
                    _source_checks(prepared, _spec(model), run_dir)
                logger.info("State activity classifier: %s paired bootstrap (%s resamples)", model, n_bootstrap)
                comparison = paired_feature_comparison(pair[BASELINE], pair[AUGMENTED], n_bootstrap)
                comparison.insert(0, "model", model)
                comparisons.append(comparison)
                # Checkpoint each completed pair; never lose it if a later model fails.
                pd.DataFrame(metrics).to_csv(destination / "primary_metrics.csv", index=False)
                pd.concat(comparisons, ignore_index=True).to_csv(destination / "paired_feature_differences.csv",
                                                                 index=False)
            manifest["status"] = "complete"
    except Exception as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
        _write_json(destination / "manifest.json", manifest)
    return {"output_dir": str(destination), "status": manifest["status"],
            "visits": manifest["visits"], "people": manifest["people"]}
