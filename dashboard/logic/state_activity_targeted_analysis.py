"""Two manuscript-selected follow-ups, with matched whole/sleep/wake data."""

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
from dashboard.logic.state_activity_features import StateActivityConfig
from dashboard.logic.state_activity_rbdq import (
    CORE_RUN, EXTENDED_RUN, COHORT_RELATIVE_PATH, PRIMARY_COVARIATES,
    _markdown_table, validate_exported_aggregates,
)
from dashboard.logic.state_activity_targeted_features import (
    CONTEXTS, TARGETS, aggregate_target_nights, audit_legacy_nightly,
    extract_targeted_nights, target_feature,
)
from mysite.settings import BASE_DIR, MEDIA_ROOT

PROTOCOL = Path(BASE_DIR) / "report/state_activity_targeted_protocol.md"


def planned_fdr(values, size=3):
    values = pd.to_numeric(values, errors="coerce")
    if len(values) != size:
        raise ValueError(f"Expected {size} planned tests in the targeted FDR family")
    q = _benjamini_hochberg(values.fillna(1.0))
    q.loc[values.isna()] = np.nan
    return q


def candidate(target, feature):
    return {"Clinical outcome": target["outcome"], "Feature family": "Manuscript-targeted state activity",
            "Redundancy cluster": feature, "Representative feature": feature}


def clinical_snapshot(rows):
    data = pd.DataFrame(rows).rename(columns={"code": "#Subject", "age": "#Age", "education_years": "#Education"})
    if data.empty or data["#Subject"].duplicated().any():
        raise ValueError("Missing or duplicate clinical snapshot rows")
    data["#Gender"] = data["sex"].map({"F": 1, "M": 2})
    data = data.drop(columns="sex")
    data["Diagnosis"] = data["diagnosis_code"].map({0: "HC", 3: "MCI-LB"})
    if data["Diagnosis"].isna().any():
        raise ValueError("Numeric clinical diagnoses must be HC=0 or MCI-LB=3")
    data["Person ID"] = data["#Subject"].map(_person_id)
    return _attach_source_variables(data)


def validate_cohort(snapshot, core, extended_codes, source_visits):
    core_codes = set(core["#Subject"])
    required = core_codes | set(source_visits["#Subject"])
    if set(snapshot["#Subject"]) != required or not set(extended_codes) <= core_codes:
        raise ValueError("Clinical snapshot does not match the frozen/source cohort")
    for table in (source_visits, core):
        merged = table.merge(snapshot, on="#Subject", suffixes=("", "_snapshot"), validate="one_to_one")
        for field in PRIMARY_COVARIATES:
            if not np.allclose(merged[field], merged[f"{field}_snapshot"], equal_nan=True):
                raise ValueError(f"Demographic drift since the frozen extraction: {field}")
        if "#DiagnosisCode" in merged and not np.array_equal(merged["#DiagnosisCode"], merged["diagnosis_code"]):
            raise ValueError("Numeric diagnoses changed since the extraction")
    if set(source_visits.loc[source_visits.sleep_valid_nights >= 5, "#Subject"]) != set(extended_codes):
        raise ValueError("Source sleep eligibility does not reproduce the manuscript extended cohort")


def reproduce_original_models(raw, reference, snapshot, extended_codes):
    # Raw feature values, never the covariate-residualized screening matrix.
    data = raw.loc[raw["#Subject"].isin(extended_codes)].merge(
        snapshot[["#Subject", "diagnosis_code", "Diagnosis", "Person ID", "updrs", "attention"]],
        on="#Subject", validate="one_to_one")
    checks = []
    for target in TARGETS:
        feature = target["canonical_feature"]
        result, _ = _fit_candidate(data, candidate(target, feature), MODEL_VARIANTS[0])
        expected = reference.loc[
            (reference["Clinical outcome"] == target["outcome"]) & (reference["Representative feature"] == feature)]
        if len(expected) != 1:
            raise ValueError("The frozen reference must contain exactly one row per targeted finding")
        expected = expected.iloc[0]
        mapping = {"n": "n", "Person clusters": "Person clusters", "Feature beta per SD": "Feature beta per SD",
                   "Feature robust SE": "Robust SE", "Feature 95% CI lower": "95% CI lower",
                   "Feature 95% CI upper": "95% CI upper", "Feature p": "p"}
        for actual_key, expected_key in mapping.items():
            actual, frozen = result.get(actual_key, np.nan), expected[expected_key]
            passed = result["Main model status"] == "ok" and np.isclose(actual, frozen, rtol=1e-8, atol=1e-10)
            checks.append({"Outcome": target["outcome"], "Feature": feature, "Quantity": actual_key,
                           "recomputed": actual, "frozen": frozen, "passed": bool(passed)})
    return pd.DataFrame(checks)


def prepare_target_samples(nightly, snapshot, core_codes, config):
    if nightly.groupby("#Subject").size().max() > 7:
        raise ValueError("More than seven nights in a visit")
    samples, audits = {}, []
    for target in TARGETS:
        features = aggregate_target_nights(nightly, target, config)
        data = snapshot.merge(features, on="#Subject", how="left", validate="one_to_one")
        data["Clinical outcome"] = target["outcome"]
        data["in_manuscript_cohort"] = data["#Subject"].isin(core_codes)
        fields = [target_feature(target, context) for context in (*CONTEXTS, "legacy_whole")]
        required = [target["field"], *PRIMARY_COVARIATES, *fields]
        complete = np.isfinite(data[required].to_numpy(dtype=float)).all(axis=1)
        data["common_model_eligible"] = data["in_manuscript_cohort"] & complete & (
                    data.target_valid_nights.fillna(0) >= config.min_nights)
        reasons = []
        for row in data.to_dict("records"):
            reason = []
            if not row["in_manuscript_cohort"]:
                reason.append("outside frozen manuscript cohort")
            if not np.isfinite(row["target_valid_nights"]) or row["target_valid_nights"] < config.min_nights:
                reason.append("fewer than five jointly valid target nights")
            for field in [target["field"], *PRIMARY_COVARIATES]:
                if not np.isfinite(row[field]):
                    reason.append(f"missing {field}")
            reasons.append("; ".join(reason))
        data["exclusion_reason"] = reasons
        common = data.loc[data.common_model_eligible].copy().reset_index(drop=True)
        if common.empty or set(common.diagnosis_code) != {0, 3}:
            raise ValueError(f"No estimable matched two-diagnosis sample for {target['outcome']}")
        samples[target["outcome"]] = common
        audits.append(data)
    return samples, pd.concat(audits, ignore_index=True)


def fit_target_models(samples):
    rows, slopes = [], []
    for target in TARGETS:
        data = samples[target["outcome"]]
        jobs = [("aligned", "All collections", data, variant) for variant in MODEL_VARIANTS]
        jobs.extend(("aligned", collection, data.loc[data["Clinical collection"] == collection], variant)
                    for collection in ("NINR", "NU20") for variant in WITHIN_COLLECTION_VARIANTS)
        legacy = data.copy()
        legacy[target_feature(target, "whole")] = legacy[target_feature(target, "legacy_whole")]
        jobs.extend(("legacy-whole-endpoints", "All collections", legacy, variant)
                    for variant in MODEL_VARIANTS if variant["key"] in ("primary", "age-collection-adjusted"))
        for window, subset, selected, variant in jobs:
            for context in CONTEXTS:
                feature = target_feature(target, context)
                result, diagnosis_slopes = _fit_candidate(selected, candidate(target, feature), variant)
                identity = {"Window definition": window, "Analysis subset": subset, "Model": variant["key"],
                            "Context": context, "Metric": target["metric"], "Aggregation": target["aggregation"]}
                if result["Main model status"] != "ok":
                    result["Feature p"] = np.nan
                elif result["n"] != len(selected):
                    raise ValueError("Unexpected context-specific deletion from the common outcome sample")
                if result["Interaction model status"] != "ok":
                    result["Interaction p"] = np.nan
                for diagnosis in ("HC", "MCI-LB"):
                    result[f"{diagnosis} visits"] = int((selected.Diagnosis == diagnosis).sum())
                    result[f"{diagnosis} people"] = int(
                        selected.loc[selected.Diagnosis == diagnosis, "Person ID"].nunique())
                    found = [row for row in diagnosis_slopes if row["Diagnosis"] == diagnosis]
                    row = found[0] if found else {
                        **candidate(target, feature), "Diagnosis": diagnosis, "n": len(selected),
                        "Person clusters": selected["Person ID"].nunique(),
                        "Status": result["Interaction model status"],
                        "Feature beta per SD": np.nan, "Robust SE": np.nan, "95% CI lower": np.nan,
                        "95% CI upper": np.nan, "p": np.nan,
                    }
                    if row["Status"] != "ok":
                        row["p"] = np.nan
                    slopes.append({**identity, **row})
                rows.append({**identity, **result})
    models, slope_frame = pd.DataFrame(rows), pd.DataFrame(slopes)
    grouping = ["Clinical outcome", "Window definition", "Analysis subset", "Model"]
    for p, q in (("Feature p", "Feature FDR p"), ("Interaction p", "Interaction FDR p")):
        models[q] = models.groupby(grouping)[p].transform(planned_fdr)
        models[q + " across outcomes"] = models.groupby(grouping[1:])[p].transform(
            lambda values: planned_fdr(values, 6))
    slope_frame["FDR p"] = slope_frame.groupby([*grouping, "Diagnosis"])["p"].transform(planned_fdr)
    return models, slope_frame


def rank_correlations(samples):
    rows = []
    for target in TARGETS:
        for diagnosis in ("Pooled", "HC", "MCI-LB"):
            data = samples[target["outcome"]]
            selected = data if diagnosis == "Pooled" else data.loc[data.Diagnosis == diagnosis]
            for context in CONTEXTS:
                feature = target_feature(target, context)
                row = {"Clinical outcome": target["outcome"], "Context": context, "Diagnosis": diagnosis,
                       "n": len(selected), "people": selected["Person ID"].nunique()}
                if len(selected) < 5 or selected[feature].nunique() < 2 or selected[target["field"]].nunique() < 2:
                    row.update(rho=np.nan, p=np.nan, ci_lower=np.nan, ci_upper=np.nan,
                               status="insufficient or constant data")
                else:
                    design = pd.DataFrame({"rho": _standardize(rankdata(selected[feature], method="average"))})
                    model = _fit_gee(_standardize(rankdata(selected[target["field"]], method="average")), design,
                                     selected["Person ID"].to_numpy())
                    row.update(rho=model["coefficients"]["rho"],
                               p=model["p_values"]["rho"] if model["status"] == "ok" else np.nan,
                               ci_lower=model["ci_lower"]["rho"], ci_upper=model["ci_upper"]["rho"],
                               status=model["status"])
                rows.append(row)
    output = pd.DataFrame(rows)
    output["FDR p"] = output.groupby(["Clinical outcome", "Diagnosis"])["p"].transform(planned_fdr)
    return output


def coverage_table(audit):
    rows = []
    for outcome in audit["Clinical outcome"].unique():
        for collection in ("All collections", "NINR", "NU20"):
            for diagnosis in ("HC", "MCI-LB"):
                group = audit.loc[(audit["Clinical outcome"] == outcome) & audit.in_manuscript_cohort & (
                            audit.Diagnosis == diagnosis)]
                if collection != "All collections":
                    group = group.loc[group["Clinical collection"] == collection]
                matched = group.loc[group.common_model_eligible]
                rows.append({"Outcome": outcome, "Collection": collection, "Diagnosis": diagnosis,
                             "manuscript_visits": len(group),
                             "paired_target_visits": int((group.target_valid_nights.fillna(0) >= 5).sum()),
                             "model_visits": len(matched), "model_people": matched["Person ID"].nunique()})
    return pd.DataFrame(rows)


def write_report(output, models, coverage, baseline_audit, nightly_audit):
    primary = models.loc[(models["Window definition"] == "aligned") & (models.Model == "primary")]
    sensitivity = models.loc[(models["Window definition"] == "aligned") & (models.Model == "age-collection-adjusted")]
    endpoint = models.loc[(models["Window definition"] == "legacy-whole-endpoints") & (models.Model == "primary") & (
                models.Context == "whole")]
    columns = ["Clinical outcome", "Context", "n", "Person clusters", "Feature beta per SD", "Feature 95% CI lower",
               "Feature 95% CI upper", "Feature FDR p", "Feature FDR p across outcomes", "Interaction FDR p"]
    lines = ["# Targeted UPDRS and attention activity follow-ups", "",
             "These are post hoc extensions of the two existing activity findings, not independent validation. No classifier was trained, no raw accelerometry was reprocessed, and the manuscript and earlier RBDq outputs were not edited.",
             "",
             "## What was compared", "",
             "UPDRS uses IQR across nights of nightly activity MAD. Attention uses population SD across nights of nightly relative interdecile range, (P90-P10)/maximum. For each outcome, whole interval, predicted sleep, and predicted wake use exactly the same nights and complete-case visits. Each state needs sufficient minutes; each visit needs at least five jointly valid nights. The two outcomes are not restricted by RBDq availability.",
             "",
             "## Validation and coverage", "",
             f"Frozen whole-interval model reproduction: {int(baseline_audit.passed.sum())}/{len(baseline_audit)} checks passed. Legacy nightly metric comparisons: {int((nightly_audit.status == 'pass').sum())} passed, {int((nightly_audit.status == 'mismatch').sum())} mismatches, {int((nightly_audit.status == 'no_frozen_reference').sum())} without a frozen reference (listed, not treated as reproduction passes).",
             "",
             _markdown_table(coverage,
                             ["Outcome", "Collection", "Diagnosis", "manuscript_visits", "paired_target_visits",
                              "model_visits", "model_people"]), "",
             "## Primary adjusted associations", "",
             "Gaussian GEE with independence working correlation, robust person-clustered covariance, diagnosis, sex and education. Betas are outcome points per one SD of the corresponding feature. CIs are ordinary robust 95% intervals. BH correction is across three comparator tests per outcome/model, with a six-test companion correction across both targeted outcomes.",
             "",
             _markdown_table(primary, columns), "",
             "## Additional age and collection adjustment", "", _markdown_table(sensitivity, columns), "",
             "## Original whole-interval endpoint sensitivity", "",
             "Only the whole-interval endpoint convention changes below; the selected nights, visits and sleep/wake values remain identical. The main whole comparator uses aligned complete minutes and includes mixed-state minutes. The legacy comparator uses the original inclusive onset/offset timestamp slice.",
             "",
             _markdown_table(endpoint, columns), "",
             "## Interpretation limits", "",
             "A significant whole/sleep/wake estimate and a nonsignificant alternative do not establish that their coefficients differ. No formal state-coefficient contrast was tested. Feature standardization is context-specific, so betas describe a one-SD change in each context, not identical raw movement increments. A smaller q value than in the original broad screening does not establish improvement: the test families and matched samples differ.",
             "",
             "These movement-derived labels do not identify REM sleep or confirm physiological sleep/wake. Mixed minutes are excluded from state-specific metrics but retained in the whole comparator. There is no additional nonwear mask. Requiring measurable nocturnal wake selects a subset of the original cohort. Source-specific estimates may be imprecise where few MCI-LB participants are available.",
             "",
             "All prespecified source/age/stratum models, diagnosis-specific slopes, descriptive rank associations, matched visit features, nightly metrics, audits, frozen clinical snapshot, protocol and provenance are in the accompanying files. No outcome-driven feature or threshold selection was performed.",
             ""]
    (output / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def run_targeted_followups(input_dir, output_dir=None):
    from dashboard.models import Subject

    input_dir = Path(input_dir).resolve()
    source_manifest_path = input_dir / "manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text())
    if source_manifest["status"] != "complete" or source_manifest["summary"]["failed_recordings"]:
        raise ValueError("A complete state extraction is required")
    config = StateActivityConfig(**source_manifest["config"])
    if config.min_nights != 5:
        raise ValueError("The targeted manuscript protocol requires five paired nights")
    source_nightly_path, source_visits_path = input_dir / "nightly_state_activity.csv", input_dir / "visit_state_activity.csv"
    source_nightly, source_visits = pd.read_csv(source_nightly_path), pd.read_csv(source_visits_path)
    validate_exported_aggregates(source_nightly, source_visits, config)
    core_path, extended_path = CORE_RUN / COHORT_RELATIVE_PATH, EXTENDED_RUN / COHORT_RELATIVE_PATH
    raw_path, frozen_nightly_path = EXTENDED_RUN / "raw/grouped_clinical_matrix_with_stats.xlsx", EXTENDED_RUN / "raw/clinical_data.xlsx"
    reference_path = EXTENDED_RUN / "scenarios/predlb-vs-hc/correlation/feature_family_followup_analysis.xlsx"
    core = pd.read_excel(core_path, usecols=["#Subject", *PRIMARY_COVARIATES])
    extended_codes = pd.read_excel(extended_path, usecols=["#Subject"])["#Subject"]
    raw = pd.read_excel(raw_path, usecols=["#Subject", *PRIMARY_COVARIATES, *[t["canonical_feature"] for t in TARGETS]])
    frozen_nightly = pd.read_excel(frozen_nightly_path,
                                   usecols=["#Subject", "#Date", *[t["canonical_nightly"] for t in TARGETS]])
    reference = pd.read_excel(reference_path, sheet_name="adjusted_associations")
    codes = set(core["#Subject"]) | set(source_visits["#Subject"])
    snapshot = clinical_snapshot(list(Subject.objects.filter(code__in=codes).values(
        "code", "diagnosis_code", "age", "sex", "education_years", "updrs", "attention")))
    validate_cohort(snapshot, core, extended_codes, source_visits)
    output = Path(output_dir).resolve() if output_dir else Path(MEDIA_ROOT) / "state-activity-targeted" / datetime.now(
        timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "status": "prepared_before_fitting", "started_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": PROTOCOL.read_text(), "targets": list(TARGETS), "config": config.to_dict(),
        "input_files": [_fingerprint(p) for p in
                        (source_manifest_path, source_nightly_path, source_visits_path, core_path, extended_path,
                         raw_path, frozen_nightly_path, reference_path)],
        "versions": {"numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__,
                     "statsmodels": statsmodels.__version__},
        "implementation": [_fingerprint(p) for p in (
            Path(__file__), Path(__file__).with_name("state_activity_targeted_features.py"),
            Path(__file__).with_name("state_activity_features.py"),
            Path(__file__).with_name("state_activity_export.py"),
            Path(__file__).with_name("state_activity_rbdq.py"),
            Path(__file__).with_name("association_sensitivity_analysis.py"),
            Path(__file__).with_name("classification_validity_checks.py"),
            Path(__file__).with_name("feature_family_followup_analysis.py"),
        )],
    }
    snapshot.to_csv(output / "clinical_snapshot.csv", index=False)
    manifest["clinical_snapshot"] = _fingerprint(output / "clinical_snapshot.csv")
    (output / "protocol.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    try:
        baseline_audit = reproduce_original_models(raw, reference, snapshot, extended_codes)
        baseline_audit.to_csv(output / "original_model_reproduction.csv", index=False)
        if not baseline_audit.passed.all():
            raise ValueError("Original manuscript models did not reproduce; do not interpret new associations")
        nightly, sources = extract_targeted_nights(source_nightly, source_manifest, config)
        nightly.to_csv(output / "nightly_target_features.csv", index=False)
        nightly_audit = audit_legacy_nightly(nightly, frozen_nightly)
        nightly_audit.to_csv(output / "legacy_nightly_reproduction.csv", index=False)
        if (nightly_audit.status == "mismatch").any():
            raise ValueError(
                "Legacy nightly metrics do not reproduce the frozen export; do not interpret new associations")
        paired = nightly.loc[
            (nightly.sleep_status == "ok") & (nightly.wake_status == "ok") & nightly["#Subject"].isin(extended_codes)]
        paired_keys = set(paired[["#Subject", "#Date"]].itertuples(index=False, name=None))
        no_reference = nightly_audit.loc[nightly_audit.status == "no_frozen_reference"]
        if any(key in paired_keys for key in no_reference[["#Subject", "#Date"]].itertuples(index=False, name=None)):
            raise ValueError(
                "A usable paired night is outside the frozen nightly source; reconcile dates before modeling")
        samples, audit = prepare_target_samples(nightly, snapshot, set(core["#Subject"]), config)
        audit.to_csv(output / "cohort_audit.csv", index=False)
        pd.concat(samples.values(), ignore_index=True).to_csv(output / "matched_visit_features.csv", index=False)
        coverage = coverage_table(audit)
        coverage.to_csv(output / "coverage_by_outcome_diagnosis_collection.csv", index=False)
        models, slopes = fit_target_models(samples)
        models.to_csv(output / "adjusted_models.csv", index=False)
        slopes.to_csv(output / "diagnosis_specific_slopes.csv", index=False)
        rank_correlations(samples).to_csv(output / "rank_correlations.csv", index=False)
        write_report(output, models, coverage, baseline_audit, nightly_audit)
        manifest.update(status="complete", finished_utc=datetime.now(timezone.utc).isoformat(), sources=sources,
                        common_cohorts={outcome: {"visits": len(data), "people": int(data["Person ID"].nunique())} for
                                        outcome, data in samples.items()},
                        main_model_statuses=models["Main model status"].value_counts().to_dict(),
                        interaction_model_statuses=models["Interaction model status"].value_counts().to_dict())
        result = {"output_dir": str(output), "cohorts": manifest["common_cohorts"]}
    except Exception as error:
        manifest.update(status="failed_validation_or_analysis", error=f"{type(error).__name__}: {error}",
                        finished_utc=datetime.now(timezone.utc).isoformat())
        raise ValueError(f"{error}. Audit directory: {output}") from error
    finally:
        (output / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return result
