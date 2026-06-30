# GeneActiv / preDLB Project Update Report

Prepared: 2026-04-22  
Project path: `E:\geneactiv-processing-data`

This report is an updated version of the previous summary (`phd_supervisor_experiment_report_2026-04-08.md`) and focuses on progress and new findings made after that date.

## 1) Major Progress Since 2026-04-08

### 1.1 Clinical label source was corrected based on expert feedback

We replaced the temporary Identity-sheet label mapping with the expert-confirmed source:

- Excel file: `preDLB_shared.xlsx`
- Sheet: `PSY_RAW`
- Column: `HC0_nHC1_MCI2_MCILB3_baseline`
- Mapping: direct `0..3` to `diagnosis_code` (`HC, NonHC, MCI-AD, preDLB`)

Implementation details:

- IDs were matched by `ID_1.meranie` to DB `Subject.code`
- `pre-LBD2-XXX` was normalized to canonical `pre-LBD-XXX` (same patient, second visit)
- update was executed with full backup/change/rollback artifacts

Result:

- Relevant DB rows checked: `169`
- Rows updated in this PSY-based relabeling pass: `48`
- Post-update mismatches vs `PSY_RAW` baseline: `0`

### 1.2 Full audit trail was produced before and during updates

Created files:

- [psyraw_db_backup_before_update_20260422_194835.csv](/E:/geneactiv-processing-data/doc/psyraw_db_backup_before_update_20260422_194835.csv)
- [psyraw_diagnosis_code_changes_20260422_194835.csv](/E:/geneactiv-processing-data/doc/psyraw_diagnosis_code_changes_20260422_194835.csv)
- [psyraw_rollback_20260422_194835.sql](/E:/geneactiv-processing-data/doc/psyraw_rollback_20260422_194835.sql)
- [psyraw_comparison_vs_original_20260422_194835.csv](/E:/geneactiv-processing-data/doc/psyraw_comparison_vs_original_20260422_194835.csv)
- [psyraw_update_summary_20260422_194835.txt](/E:/geneactiv-processing-data/doc/psyraw_update_summary_20260422_194835.txt)

Comparison against the original snapshot from before yesterday's relabeling:

- Difference vs original **before** PSY update: `47` rows
- Difference vs original **after** PSY update: `0` rows (for rows with non-null original labels)
- Note: `9` rows had null original diagnosis in that old snapshot, so exact original comparison is not defined for those rows

### 1.3 Storage cleanup and operational maintenance

- `media/data` cleanup completed by deleting unreferenced files only (checked against `dashboard_csvdata.data`)
- Deleted: `233` files
- Freed space: `~79.9 GB`
- Validation after cleanup: `0` missing referenced files

Sleeppy storage assessment:

- `media/sleeppy` total: `~177.15 GB`
- `raw_days` alone: `~175.56 GB`
- `results` + `reports`: `~0.38 GB`

This confirms that almost all Sleeppy space usage is temporary/intermediate raw day files.

## 2) New Modeling Diagnostic (Most Important Scenario)

Target scenario analyzed:

- `scenario-preDLB_vs_HC`
- File: [subject_predictions.xlsx](/E:/geneactiv-processing-data/media/classification/grouped-statistics/dataset-clinical/all/20260422_200038/scenario-preDLB_vs_HC/subject_predictions.xlsx)

### 2.1 Error rates

- Subjects analyzed: `136`
- Default prediction errors: `50 / 136` (`36.8%`)
- Tuned-threshold errors: `48 / 136` (`35.3%`)

### 2.2 Misclassification pattern by cohort/prefix

Default model error rates:

- `pre-LBD`: `21 / 45` = `46.7%`
- `pre-LBD2`: `2 / 15` = `13.3%`
- `COBEN`: `18 / 55` = `32.7%`
- `HC`: `6 / 16` = `37.5%`

Key observation:

> `pre-LBD2-*` subjects had much lower error than `pre-LBD-*` subjects (`13.3%` vs `46.7%`).

This is a very relevant finding and consistent with your interpretation that later disease stage might be easier to identify.

### 2.3 Interpretation of the pre-LBD2 finding

Most likely interpretation:

- Second visit (`pre-LBD2`) may indeed show stronger/clearer disease signal due to progression, making classification easier.

Important caveat:

- Because `pre-LBD2` is a repeated measurement of an already-seen patient, there is also a possible patient-level dependence/leakage risk in standard CV if first and second visit can influence each other across folds.

So the signal is promising, but must be validated with patient-grouped evaluation.

## 3) Covariate Exploration for Misclassified Subjects

We tested whether errors correlate with metadata not currently used as model features, especially these `SleepDiaryDay` fields:

- `day_sleep_count`, `day_sleep_time`
- `alcohol_count`, `alcohol_time`
- `caffeine_count`, `caffeine_time`
- `sleeping_pill`
- `sleep_quality`, `rest_quality`

Main result:

- No strong standalone covariate signal yet.
- Several weak trends appeared:
  - higher `sleeping_pill_rate` in misclassified group
  - slightly lower `rest_quality` in misclassified group
  - later `caffeine_time` in preDLB false negatives

These effects are not strong enough yet to claim a robust confounder, but they are reasonable candidates for controlled ablation.

## 4) Practical Conclusions at This Stage

1. Label source is now aligned with expert guidance (`PSY_RAW` baseline), and DB consistency is clean.
2. Pipeline reliability and traceability improved (backups, rollback scripts, explicit change logs).
3. The most clinically interesting new finding is the large error gap between `pre-LBD` and `pre-LBD2`.
4. This supports the progression hypothesis, but requires stricter patient-level validation to separate true progression signal from repeated-subject effects.

## 5) Recommended Next Experimental Steps

### 5.1 Methodological step (highest priority)

Run evaluation with grouped CV by canonical patient ID:

- group key: canonical patient code (`pre-LBD2-XXX` grouped with `pre-LBD-XXX`)
- prevents fold leakage between first and second visit of the same patient

### 5.2 Clinical/progression analysis

Run explicit first-vs-second visit experiment:

- train/test performance separately on first and second visits
- compare score distribution shifts between visits for same patients
- quantify progression-sensitive features

### 5.3 Covariate step

Test adding these diary covariates as an optional block:

- `sleeping_pill_rate`
- `rest_quality_mean`
- `caffeine_time_mean`
- optionally `alcohol_time_mean`

Do this as a controlled ablation (`with_covariates` vs `without_covariates`), not mixed blindly into all features.

## 6) Final Note for Supervisor

The project has moved from “pipeline construction and unstable labels” to a much stronger state:

- labels now reflect the expert-specified baseline source,
- data lineage is auditable,
- and we identified a potentially important clinical/progression signal (`pre-LBD2` easier than `pre-LBD`).

The next milestone should be a publication-grade patient-grouped validation to confirm whether this effect reflects true disease progression signal.
