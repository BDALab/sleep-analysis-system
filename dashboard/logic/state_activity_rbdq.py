"""Targeted RBDq extension; never alters the canonical manuscript analyses."""

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import statsmodels
from scipy.stats import rankdata

from dashboard.logic.association_sensitivity_analysis import (
    MODEL_VARIANTS, WITHIN_COLLECTION_VARIANTS, _attach_source_variables, _fit_candidate,
)
from dashboard.logic.feature_family_followup_analysis import (
    _benjamini_hochberg, _fit_gee, _person_id, _standardize,
)
from dashboard.logic.state_activity_export import _fingerprint
from dashboard.logic.state_activity_features import (
    METRICS, STATES, VISIT_FEATURES, StateActivityConfig, aggregate_visits,
)
from mysite.settings import BASE_DIR, MEDIA_ROOT

CORE_RUN = Path(MEDIA_ROOT) / "analysis-preparation/dataset-clinical/20260629_132528_560853"
EXTENDED_RUN = Path(MEDIA_ROOT) / "analysis-preparation/dataset-clinical-acc/20260629_132533_149540"
COHORT_RELATIVE_PATH = "scenarios/predlb-vs-hc/data/grouped_clinical_matrix_with_stats.xlsx"
PROTOCOL_PATH = Path(BASE_DIR) / "report/state_activity_rbdq_protocol.md"
PRIMARY_COVARIATES = ("#Gender", "#Education", "#Age")


def paired_night_aggregates(nightly, config):
    paired = nightly.copy()
    valid = (paired["status"] == "ok") & (paired["sleep_status"] == "ok") & (paired["wake_status"] == "ok")
    for state in STATES:
        paired.loc[~valid, f"{state}_status"] = "not_paired"
        for metric in METRICS:
            paired.loc[~valid, f"{state}_{metric}"] = np.nan
    output = aggregate_visits(paired, config)
    output["paired_valid_nights"] = output["sleep_valid_nights"]
    return output


def validate_exported_aggregates(nightly, visits, config):
    if visits["#Subject"].duplicated().any():
        raise ValueError("Duplicate visit rows in the extraction")
    reproduced = aggregate_visits(nightly, config).set_index("#Subject").sort_index()
    saved = visits.set_index("#Subject").sort_index()
    if not reproduced.index.equals(saved.index):
        raise ValueError("Nightly and visit tables contain different visits")
    for column in (*VISIT_FEATURES, "sleep_valid_nights", "wake_valid_nights", "nights_selected", "nights_processed",
                   "#DiagnosisCode", *PRIMARY_COVARIATES, "RBDq"):
        a = pd.to_numeric(reproduced[column], errors="raise").to_numpy(dtype=float, na_value=np.nan)
        b = pd.to_numeric(saved[column], errors="raise").to_numpy(dtype=float, na_value=np.nan)
        if not np.allclose(a, b, rtol=1e-10, atol=1e-12, equal_nan=True):
            raise ValueError(f"Saved visit aggregates do not reproduce: {column}")


def prepare_analysis(nightly, visits, core_metadata, extended_codes, diagnosis_by_subject, config):
    """Validate cohort identity before any outcome-feature association is fitted."""
    validate_exported_aggregates(nightly, visits, config)
    if config.min_nights != 5:
        raise ValueError("This manuscript-aligned protocol requires five usable nights")
    if (visits["nights_selected"] > 7).any():
        raise ValueError("More than seven nights in a visit; reconcile with the manuscript first")
    if core_metadata["#Subject"].duplicated().any():
        raise ValueError("Duplicate frozen cohort visit identifiers")
    if not set(extended_codes) <= set(core_metadata["#Subject"]):
        raise ValueError("Frozen extended cohort is not a subset of the core cohort")
    codes = set(visits["#Subject"]) | set(core_metadata["#Subject"])
    if codes - set(diagnosis_by_subject):
        raise ValueError("Missing numeric diagnosis codes for cohort reconciliation")
    if any(diagnosis_by_subject[code] not in (0, 3) for code in codes):
        raise ValueError("Non-HC/MCI-LB diagnosis in the selected/frozen cohort")
    expected = visits["#Subject"].map(diagnosis_by_subject)
    if not np.array_equal(expected, visits["#DiagnosisCode"]):
        raise ValueError("Diagnosis codes changed since the state extraction")

    joined = visits.merge(core_metadata, on="#Subject", suffixes=("", "_frozen"), validate="one_to_one")
    for column in PRIMARY_COVARIATES:
        if not np.allclose(joined[column], joined[f"{column}_frozen"], equal_nan=True):
            raise ValueError(f"Demographics differ from the manuscript snapshot: {column}")
    sleep_codes = set(visits.loc[visits["sleep_valid_nights"] >= 5, "#Subject"])
    if sleep_codes != set(extended_codes):
        raise ValueError("State sleep eligibility differs from the frozen extended cohort; review before modeling")

    paired = paired_night_aggregates(nightly, config)
    audit = pd.DataFrame({"#Subject": sorted(codes)})
    audit["diagnosis_code"] = audit["#Subject"].map(diagnosis_by_subject)
    audit["Diagnosis"] = audit["diagnosis_code"].map({0: "HC", 3: "MCI-LB"})
    audit["Person ID"] = audit["#Subject"].map(_person_id)
    audit["in_manuscript_cohort"] = audit["#Subject"].isin(core_metadata["#Subject"])
    audit["in_frozen_extended_cohort"] = audit["#Subject"].isin(extended_codes)
    audit["in_state_extraction"] = audit["#Subject"].isin(visits["#Subject"])
    audit = audit.merge(paired.drop(columns="#DiagnosisCode"), on="#Subject", how="left", validate="one_to_one")
    for state in STATES:
        original_counts = visits.set_index("#Subject")[f"{state}_valid_nights"]
        audit[f"original_{state}_valid_nights"] = audit["#Subject"].map(original_counts)
        audit[f"{state}_eligible"] = audit[f"original_{state}_valid_nights"].fillna(0) >= 5
    missing_clinical = audit[["RBDq", *PRIMARY_COVARIATES]].replace([np.inf, -np.inf], np.nan).isna().any(axis=1)
    missing_features = audit[list(VISIT_FEATURES)].replace([np.inf, -np.inf], np.nan).isna().any(axis=1)
    audit["common_model_eligible"] = audit["in_manuscript_cohort"] & ~missing_clinical & ~missing_features
    reasons = []
    for row in audit.to_dict("records"):
        reason = []
        if not row["in_manuscript_cohort"]:
            reason.append("outside frozen manuscript cohort")
        if not row["in_state_extraction"]:
            reason.append("absent from extraction")
        elif row["paired_valid_nights"] < 5:
            reason.append("fewer than five paired valid nights")
        for field in ("RBDq", *PRIMARY_COVARIATES):
            if not np.isfinite(row[field]):
                reason.append(f"missing {field}")
        reasons.append("; ".join(reason))
    audit["exclusion_reason"] = reasons
    audit = _attach_source_variables(audit)
    primary = audit.loc[audit["common_model_eligible"]].copy().reset_index(drop=True)
    if primary.empty or set(primary["diagnosis_code"]) != {0, 3}:
        raise ValueError("Common cohort must contain both HC and MCI-LB")
    primary["rbdq"] = primary["RBDq"]
    original = visits.set_index("#Subject").loc[primary["#Subject"], list(VISIT_FEATURES)]
    measurement_sensitivity = primary.copy()
    measurement_sensitivity.loc[:, list(VISIT_FEATURES)] = original.to_numpy()
    return primary, measurement_sensitivity, audit


def planned_fdr(values):
    """16 planned tests, including failed/constant tests in the denominator."""
    numeric = pd.to_numeric(values, errors="coerce")
    if len(numeric) != len(VISIT_FEATURES):
        raise ValueError("FDR family must contain all 16 planned state features")
    output = _benjamini_hochberg(numeric.fillna(1.0))
    output.loc[numeric.isna()] = np.nan
    return output


def candidate(feature):
    return {"Clinical outcome": "RBDq", "Feature family": "Predicted-state activity", "Redundancy cluster": feature,
            "Representative feature": feature}


def fit_models(primary, measurement_sensitivity):
    rows, slopes = [], []
    jobs = [("paired-nights", "All collections", primary, variant) for variant in MODEL_VARIANTS]
    jobs.extend(
        ("paired-nights", collection, primary.loc[primary["Clinical collection"] == collection], variant)
        for collection in ("NINR", "NU20") for variant in WITHIN_COLLECTION_VARIANTS
    )
    jobs.append(("original-state-nights", "All collections", measurement_sensitivity, MODEL_VARIANTS[0]))
    for dataset, subset, data, variant in jobs:
        for feature in VISIT_FEATURES:
            result, group_slopes = _fit_candidate(data, candidate(feature), variant)
            context = {"Dataset": dataset, "Analysis subset": subset, "Model": variant["key"]}
            if result["Main model status"] != "ok":
                result["Feature p"] = np.nan
            if result["Interaction model status"] != "ok":
                result["Interaction p"] = np.nan
            if result["Main model status"] == "ok" and result["n"] != len(data):
                raise ValueError("Unexpected model-specific row deletion in the matched cohort")
            for diagnosis in ("HC", "MCI-LB"):
                result[f"{diagnosis} visits"] = int((data["Diagnosis"] == diagnosis).sum())
                result[f"{diagnosis} people"] = int(data.loc[data["Diagnosis"] == diagnosis, "Person ID"].nunique())
                matches = [row for row in group_slopes if row["Diagnosis"] == diagnosis]
                row = matches[0] if matches else {
                    **candidate(feature), "Diagnosis": diagnosis, "n": len(data),
                    "Person clusters": data["Person ID"].nunique(), "Status": result["Interaction model status"],
                    "Feature beta per SD": np.nan, "Robust SE": np.nan,
                    "95% CI lower": np.nan, "95% CI upper": np.nan, "p": np.nan,
                }
                if row["Status"] != "ok":
                    row["p"] = np.nan
                slopes.append({**context, **row})
            rows.append({**context, **result})
    models = pd.DataFrame(rows)
    slope_frame = pd.DataFrame(slopes)
    groups = ["Dataset", "Analysis subset", "Model"]
    for p, q in (("Feature p", "Feature FDR p"), ("Interaction p", "Interaction FDR p")):
        models[q] = models.groupby(groups)[p].transform(planned_fdr)
    slope_frame["FDR p"] = slope_frame.groupby([*groups, "Diagnosis"])["p"].transform(planned_fdr)
    return models, slope_frame


def rank_correlations(data):
    rows = []
    for diagnosis in ("Pooled", "HC", "MCI-LB"):
        subset = data if diagnosis == "Pooled" else data.loc[data["Diagnosis"] == diagnosis]
        for feature in VISIT_FEATURES:
            row = {"Diagnosis": diagnosis, "Feature": feature, "n": len(subset),
                   "people": subset["Person ID"].nunique()}
            if len(subset) < 5 or subset[feature].nunique() < 2 or subset["rbdq"].nunique() < 2:
                row.update(status="insufficient or constant data", rho=np.nan, p=np.nan, ci_lower=np.nan,
                           ci_upper=np.nan)
            else:
                design = pd.DataFrame({"rho": _standardize(rankdata(subset[feature], method="average"))})
                model = _fit_gee(_standardize(rankdata(subset["rbdq"], method="average")), design,
                                 subset["Person ID"].to_numpy())
                row.update(status=model["status"], rho=model["coefficients"]["rho"],
                           p=model["p_values"]["rho"] if model["status"] == "ok" else np.nan,
                           ci_lower=model["ci_lower"]["rho"], ci_upper=model["ci_upper"]["rho"])
            rows.append(row)
    result = pd.DataFrame(rows)
    result["FDR p"] = result.groupby("Diagnosis")["p"].transform(planned_fdr)
    return result


def coverage_tables(audit, nightly, primary):
    rows = []
    for collection in ("All collections", "NINR", "NU20"):
        for diagnosis in ("HC", "MCI-LB"):
            group = audit.loc[audit["in_manuscript_cohort"] & (audit["Diagnosis"] == diagnosis)]
            if collection != "All collections":
                group = group.loc[group["Clinical collection"] == collection]
            chosen = group.loc[group["common_model_eligible"]]
            rows.append({"Collection": collection, "Diagnosis": diagnosis, "manuscript_visits": len(group),
                         "manuscript_people": group["Person ID"].nunique(),
                         "frozen_extended_visits": int(group["in_frozen_extended_cohort"].sum()),
                         "extraction_visits": int(group["in_state_extraction"].sum()),
                         "paired_state_visits": int((group["paired_valid_nights"].fillna(0) >= 5).sum()),
                         "model_visits": len(chosen), "model_people": chosen["Person ID"].nunique()})
    selected = nightly.loc[nightly["#Subject"].isin(primary["#Subject"])].copy()
    selected["Diagnosis"] = selected["#DiagnosisCode"].map({0: "HC", 3: "MCI-LB"})
    selected["paired_valid"] = (selected["status"] == "ok") & (selected["sleep_status"] == "ok") & (
            selected["wake_status"] == "ok")
    columns = ["sleep_minutes", "wake_minutes", "mixed_minutes", "missing_prediction_minutes",
               "invalid_activity_minutes", "usable_signal_coverage_fraction"]
    support = selected.groupby("Diagnosis")[columns].agg(["median", "min", "max"])
    support.columns = [f"{name}_{stat}" for name, stat in support.columns]
    support = support.reset_index()
    support["nights"] = support["Diagnosis"].map(selected.groupby("Diagnosis").size())
    support["paired_valid_nights"] = support["Diagnosis"].map(selected.groupby("Diagnosis")["paired_valid"].sum())
    return pd.DataFrame(rows), support


def _markdown_table(frame, columns):
    if frame.empty:
        return "No rows."

    def display(value):
        if pd.isna(value):
            return "NA"
        if isinstance(value, (float, np.floating)):
            return f"{value:.5g}"
        return str(value).replace("|", "/")

    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    lines.extend("| " + " | ".join(display(row[col]) for col in columns) + " |" for row in frame.to_dict("records"))
    return "\n".join(lines)


def write_report(output, coverage, support, models, slopes, correlations):
    main = models.loc[(models.Dataset == "paired-nights") & (models.Model == "primary")]
    robust = models.loc[(models.Dataset == "paired-nights") & (models.Model == "age-collection-adjusted")]
    counts = {
        "primary_main_fdr": int((main["Feature FDR p"] < .05).sum()),
        "primary_interaction_fdr": int((main["Interaction FDR p"] < .05).sum()),
        "age_collection_main_fdr": int((robust["Feature FDR p"] < .05).sum()),
        "age_collection_interaction_fdr": int((robust["Interaction FDR p"] < .05).sum()),
    }
    reading = ["## Plain-language reading", ""]
    hits = main.loc[main["Feature FDR p"] < .05]
    if hits.empty:
        reading += [
            "None of the 16 new features passed main-effect FDR correction in the primary model. This does not establish absence of an association, but it provides no FDR-supported new main-effect candidate in this sample.",
            ""]
    for row in hits.to_dict("records"):
        feature = row["Representative feature"]
        reading += [
            f"Primary candidate: `{feature}`. Beta {row['Feature beta per SD']:.3f} RBDq points per feature SD, 95% CI {row['Feature 95% CI lower']:.3f} to {row['Feature 95% CI upper']:.3f}, FDR q={row['Feature FDR p']:.4f}.",
            "",
        ]
        sensitivity = robust.loc[robust["Representative feature"] == feature].iloc[0]
        reading += [
            f"With additional age and collection adjustment: beta {sensitivity['Feature beta per SD']:.3f}, 95% CI {sensitivity['Feature 95% CI lower']:.3f} to {sensitivity['Feature 95% CI upper']:.3f}, FDR q={sensitivity['Feature FDR p']:.4f}. "
            + ("It also passes this sensitivity FDR threshold." if sensitivity[
                                                                       "Feature FDR p"] < .05 else "It does not pass the sensitivity FDR threshold; do not describe it as fully robust to collection adjustment."),
            "",
        ]
        for collection in ("NINR", "NU20"):
            within = models.loc[(models.Dataset == "paired-nights") & (models["Analysis subset"] == collection) & (
                    models.Model == "within-collection-primary") & (
                                        models["Representative feature"] == feature)].iloc[0]
            reading += [
                f"Within {collection}: beta {within['Feature beta per SD']:.3f}, 95% CI {within['Feature 95% CI lower']:.3f} to {within['Feature 95% CI upper']:.3f}, FDR q={within['Feature FDR p']:.4f}.",
                ""]
    reading += [
        "For `Median.activity_sleep.p90`, take the 90th percentile of minute-level activity in each night's predicted-sleep minutes, then the median of those nightly percentiles. It describes the upper part of the movement-intensity distribution, not a count of RBD events or REM-specific activity.",
        "",
        "Source-specific estimates are descriptive robustness checks. A smaller estimate or nonsignificant p value in one source does not itself establish source heterogeneity. Likewise, raw within-diagnosis rank associations and adjusted diagnosis interactions answer different questions.",
        "",
    ]
    lines = ["# Exploratory RBDq analysis of activity during predicted sleep and wake", "",
             "This is a targeted post hoc extension of the manuscript, not independent validation. The original article and canonical results have not been changed.",
             "",
             "## Cohort and coverage", "",
             _markdown_table(coverage, ["Collection", "Diagnosis", "manuscript_visits", "frozen_extended_visits",
                                        "paired_state_visits", "model_visits", "model_people"]), "",
             "All main models use the same complete-case visits. Both states are aggregated over the same jointly valid nights. The original independent-state-night aggregation is retained only as measurement sensitivity on the same visits. Repeat visits are clustered by the article's person mapping.",
             "",
             _markdown_table(support, ["Diagnosis", "nights", "paired_valid_nights", "wake_minutes_median",
                                       "mixed_minutes_median", "usable_signal_coverage_fraction_median"]), "",
             "## Primary adjusted results", "",
             "Diagnosis, sex, and education adjustment; Gaussian GEE with independence working correlation and robust person-clustered covariance. Betas are RBDq points per one feature SD. CIs are ordinary robust 95% intervals. Each main-effect and interaction FDR family contains 16 planned tests.",
             "",
             _markdown_table(main, ["Representative feature", "n", "Person clusters", "Feature beta per SD",
                                    "Feature 95% CI lower", "Feature 95% CI upper", "Feature FDR p",
                                    "Interaction FDR p"]), "",
             f"Primary FDR-significant main effects: {counts['primary_main_fdr']}/16. Primary FDR-significant diagnosis interactions: {counts['primary_interaction_fdr']}/16.",
             "",
             "## Age and collection sensitivity", "",
             _markdown_table(robust, ["Representative feature", "Feature beta per SD", "Feature 95% CI lower",
                                      "Feature 95% CI upper", "Feature FDR p", "Interaction FDR p"]), "",
             f"Age-plus-collection FDR-significant main effects: {counts['age_collection_main_fdr']}/16; interactions: {counts['age_collection_interaction_fdr']}/16.",
             "",
             *reading,
             "All prespecified age/source/within-collection model results, diagnosis-specific slopes, robust rank associations, and measurement-sensitivity models are in the accompanying CSV files; no significance filter was used to choose features.",
             "",
             "## Interpretation limits", "",
             "A significant result for one state and a nonsignificant result for the other is not evidence of a state difference. No formal sleep-versus-wake coefficient contrast was tested. The new 16-feature FDR families differ from the original broad screen, so significance is not evidence of improvement over the original whole-interval features. These labels are movement-derived predictions, not PSG-confirmed sleep, wake, REM, or RBD. This analysis does not establish causality, diagnostic validity, or independent replication.",
             "",
             "Coverage-based selection can preferentially exclude people with little detectable wake; inference is conditional on sufficient paired-night data. Small diagnosis counts in source-specific analyses can make interactions unstable. The raw extraction, its exclusions, and its support thresholds are retained, not tuned to these results.",
             ""]
    (output / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    return counts


def run_state_activity_rbdq(input_dir, output_dir=None):
    from dashboard.models import Subject

    input_dir = Path(input_dir).resolve()
    manifest_path = input_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["status"] != "complete" or manifest["summary"]["failed_recordings"]:
        raise ValueError("Only complete extraction runs may be analyzed")
    config = StateActivityConfig(**manifest["config"])
    nightly_path, visit_path = input_dir / "nightly_state_activity.csv", input_dir / "visit_state_activity.csv"
    nightly, visits = pd.read_csv(nightly_path), pd.read_csv(visit_path)
    core_path, extended_path = CORE_RUN / COHORT_RELATIVE_PATH, EXTENDED_RUN / COHORT_RELATIVE_PATH
    core = pd.read_excel(core_path, usecols=["#Subject", *PRIMARY_COVARIATES])
    extended_codes = pd.read_excel(extended_path, usecols=["#Subject"])["#Subject"]
    codes = set(core["#Subject"]) | set(visits["#Subject"])
    diagnoses = dict(Subject.objects.filter(code__in=codes).values_list("code", "diagnosis_code"))
    dates_path = CORE_RUN / "scenarios/predlb-vs-hc/data/clinical_data_adjusted.xlsx"
    dates = pd.read_excel(dates_path, usecols=["#Subject", "#Date"])
    dates["#Date"] = pd.to_datetime(dates["#Date"]).dt.strftime("%Y-%m-%d")
    frozen_pairs = set(map(tuple, dates.to_numpy()))
    for subject, day in nightly[["#Subject", "#Date"]].itertuples(index=False, name=None):
        if subject in set(core["#Subject"]) and (subject, day) not in frozen_pairs:
            raise ValueError("Night dates differ from the frozen manuscript source")
    primary, measurement, audit = prepare_analysis(nightly, visits, core, extended_codes, diagnoses, config)
    output = Path(output_dir).resolve() if output_dir else Path(MEDIA_ROOT) / "state-activity-rbdq" / datetime.now(
        timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=False)
    protocol = {
        "status": "prepared_before_fitting", "created_utc": datetime.now(timezone.utc).isoformat(),
        "input_files": [_fingerprint(p) for p in
                        (manifest_path, nightly_path, visit_path, core_path, extended_path, dates_path)],
        "protocol": PROTOCOL_PATH.read_text(), "features": list(VISIT_FEATURES),
        "config": config.to_dict(), "main_covariates": ["diagnosis", "sex", "education"],
        "person_mapping": "feature_family_followup_analysis._person_id",
        "common_visits": len(primary), "common_people": primary["Person ID"].nunique(),
        "fdr_family_size": len(VISIT_FEATURES),
        "versions": {"numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__,
                     "statsmodels": statsmodels.__version__},
        "diagnosis_source": "numeric database diagnosis codes, checked against extraction; legacy #Disease is not used",
        "implementation": [_fingerprint(p) for p in (
            Path(__file__), Path(BASE_DIR) / "dashboard/logic/association_sensitivity_analysis.py",
            Path(BASE_DIR) / "dashboard/logic/feature_family_followup_analysis.py",
            Path(BASE_DIR) / "dashboard/logic/state_activity_features.py",
            Path(BASE_DIR) / "dashboard/logic/classification_validity_checks.py",
        )],
    }
    protocol_file = output / "protocol.json"
    protocol_file.write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    audit.to_csv(output / "cohort_audit.csv", index=False)
    primary.to_csv(output / "matched_visit_features.csv", index=False)
    coverage, support = coverage_tables(audit, nightly, primary)
    coverage.to_csv(output / "coverage_by_diagnosis_collection.csv", index=False)
    support.to_csv(output / "nightly_support_by_diagnosis.csv", index=False)
    models, slopes = fit_models(primary, measurement)
    correlations = rank_correlations(primary)
    models.to_csv(output / "adjusted_models.csv", index=False)
    slopes.to_csv(output / "diagnosis_specific_slopes.csv", index=False)
    correlations.to_csv(output / "rank_correlations.csv", index=False)
    counts = write_report(output, coverage, support, models, slopes, correlations)
    protocol.update(status="complete", finished_utc=datetime.now(timezone.utc).isoformat(),
                    main_model_statuses=models["Main model status"].value_counts().to_dict(),
                    interaction_model_statuses=models["Interaction model status"].value_counts().to_dict(),
                    summary=counts)
    # Keep the pre-fit protocol immutable; completion metadata is a separate file.
    (output / "analysis_manifest.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    return {"output_dir": str(output), "visits": len(primary), "people": int(primary["Person ID"].nunique()), **counts}
