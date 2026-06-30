# GeneActiv / preDLB Project Update Report

Prepared: 2026-04-23  
Project path: `E:\geneactiv-processing-data`

This report extends the previous update (`phd_supervisor_experiment_report_2026-04-22.md`) with the newest classification run after adding covariates.

## 1) Newly Completed Experiment (Most Recent)

### 1.1 Pipeline variant

- Variant: grouped-statistics classification with covariates
- Dataset: `dataset-clinical`
- Run folder: `media/classification/grouped-statistics-with-covariates/dataset-clinical/all/20260422_215810`
- Summary file: `classification_summary.xlsx`

### 1.2 Covariates included

- `sleeping_pill_rate`
- `rest_quality_mean`
- `caffeine_time_mean`
- `alcohol_time_mean`

## 2) Main Results (Classify + Covariates)

### 2.1 Dataset overview in this run

- HC: `73`
- NonHC: `79`
- MCI-AD: `30`
- preDLB: `59`
- Missing diagnosis_code excluded: `4` subjects (`MY-HC-AU5`, `MY-HC-AU6`, `MY-HC-AU7`, `MY-HC-JM`)

### 2.2 Scenario metrics (default vs tuned threshold)

| Scenario | ROC-AUC | PR-AUC | BACC default | BACC tuned | MCC tuned | SEN tuned | SPE tuned |
|---|---:|---:|---:|---:|---:|---:|---:|
| preDLB vs HC | 0.8194 | 0.8182 | 0.7235 | **0.7535** | 0.5241 | 0.6441 | 0.8630 |
| preDLB+MCI-AD vs HC | 0.7186 | 0.7492 | 0.6548 | **0.6931** | 0.3966 | 0.5506 | 0.8356 |
| MCI-AD vs HC | 0.4534 | 0.3285 | 0.4954 | 0.5167 | 0.1545 | 0.0333 | 1.0000 |

Most promising result remains:

- `scenario-preDLB_vs_HC` with tuned `BACC = 0.7535`

## 3) Direct Comparison to Previous Non-Covariate Baseline

Previous reference run:

- `media/classification/grouped-statistics/dataset-clinical/all/20260422_200038`
- Scenario used for direct comparison: `scenario-preDLB_vs_HC`

Baseline (`without` covariates, tuned threshold):

- BACC: `0.6447`
- MCC: `0.2877`
- SEN: `0.6271`
- SPE: `0.6623`
- Cohort size: `136` (HC `77`, preDLB `59`)

Newest (`with` covariates, tuned threshold):

- BACC: `0.7535`
- MCC: `0.5241`
- SEN: `0.6441`
- SPE: `0.8630`
- Cohort size: `132` (HC `73`, preDLB `59`)

Observed change (new minus baseline):

- BACC: `+0.1088`
- MCC: `+0.2364`
- SEN: `+0.0170`
- SPE: `+0.2007`

Interpretation:

- This is a strong improvement in the main target scenario (`preDLB vs HC`), especially in specificity and MCC.
- Comparison is still favorable despite a slightly smaller HC cohort in the latest run (4 HC subjects were excluded due to missing diagnosis code).

## 4) Feature Signal from the New Run

In `scenario-preDLB_vs_HC`, added covariates are not only present but highly ranked:

- `rest_quality_mean` appears as the top feature (model importance and SHAP summary)
- `alcohol_time_mean` appears among top features (model importance and SHAP summary)

This supports that diary-derived behavioral context carries useful disease-separation signal when combined with grouped sleep statistics.

## 5) Practical Conclusion

The latest `Classify + covariates` run is currently the most promising result in the project for the main endpoint (`preDLB vs HC`).

Recommended next step before publication-facing claims:

1. Re-run this same variant under strict patient-grouped validation (to control repeated-visit effects).
2. Keep the same covariates and feature-selection setup, so we test robustness of the observed gain rather than changing multiple factors at once.
