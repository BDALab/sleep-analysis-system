# GeneActiv / Sleep Pipeline Experiment Report

Prepared: 2026-04-08

## Purpose

This document summarizes the work completed on the `geneactiv-processing-data` project during the recent development cycle, including:

- data-processing and pipeline engineering changes,
- data quality investigations,
- exploratory classification experiments,
- feature ablation experiments,
- the current interpretation of results,
- and recommended next steps before any publication-grade analysis.

The overall goal of this work was to build a reproducible Django-based pipeline that starts from raw GeneActiv / diary / Sleeppy outputs and ends with grouped, covariate-adjusted subject-level features for classification of diagnostic groups.

## High-Level Summary

The main outcome is that the full pipeline is now operational inside Django, including preprocessing, Sleeppy processing, covariate correction, grouping by subject, grouped-statistics feature generation, and exploratory classification analysis.

From the modeling perspective, the results are currently weak to modest rather than strong. The most promising signal so far is for `preDLB vs HC`, but even there the performance is still exploratory. The stricter nested cross-validation pipeline that is more suitable for publication has been implemented, but has not yet been run.

## 1. Engineering and Pipeline Work Completed

### 1.1 Preprocessing Hardening

The original preprocessing occasionally stopped the whole batch when one file failed. This was changed so that:

- `preprocess_all_data()` continues after per-file failures and logs the exception,
- malformed CSV timestamp rows are skipped with warnings instead of crashing the batch,
- preprocessing no longer rewrites raw source CSVs in place,
- time parsing was made more tolerant to null bytes and malformed timestamp formatting,
- and tolerant reading was added for additional raw file types.

This significantly improved robustness when processing heterogeneous historical files.

### 1.2 Investigation of "No data to preprocess" Cases

A detailed audit of repeated preprocessing failures showed that these cases were usually real metadata/data mismatches rather than code failures. The recurring causes were:

- no sleep diary for the subject,
- sleep diary dates not overlapping the recording period,
- or missing/corrupted cached/generated files.

In other words, many "no data" cases were correct behavior and exposed a data curation issue rather than a software bug.

### 1.3 Sleeppy Pipeline Fixes

Multiple issues in the Sleeppy integration were resolved:

- the report stage was fixed after endpoint-summary format changes,
- large GeneActiv CSV files are now streamed in chunks instead of being loaded entirely into memory,
- incomplete Sleeppy output folders are removed before reruns,
- failed Sleeppy runs no longer permanently poison future reruns,
- and corrupted Sleeppy result CSVs are now skipped safely when transferring outputs back into Django models.

This resolved several recurring failures, including large-file crashes and "skip because directory exists" behavior after incomplete runs.

### 1.4 High-Level Error Reporting in Django

The `utils` view was wrapped in a high-level `try/except`, so when a utility action fails:

- the full traceback is written into `debug.log`,
- and the user sees a visible failure message in the webpage instead of a silent crash.

### 1.5 Covariates Utility

The original Colab notebook for covariate correction was converted into Django utilities. Two variants were created:

- clinical dataset variant,
- clinical + activity variant.

The utilities generate the covariate-corrected outputs used by later steps:

- `feature_matrix.xlsx`
- `feature_matrix_controlled_gender_age.xlsx`
- `clinical_data.xlsx`
- `corr_covars_age.xlsx`
- `corr_covars_gender.xlsx`

The current implementation keeps the Excel-based outputs because that matches the original notebook workflow closely and is easy to inspect.

### 1.6 Grouping by Subject and Statistical Aggregation

The grouping notebook was also converted into a Django utility. The grouped output is now generated automatically from the covariate-corrected `clinical_data.xlsx` files.

For each subject, nightly rows are grouped and the following statistics are calculated per feature:

- `Mean`
- `Median`
- `Min`
- `Max`
- `Slope`
- `SD`
- `MAD`
- `Range`
- `IQR`
- `CV`

This step originally used only variability features. Later, level and trend features were added because variability-only summaries likely discarded too much clinically relevant signal.

### 1.7 Minimum-Night Filter

Subjects with fewer than 5 valid nights are now excluded before grouped-statistics generation. This was introduced because short recordings create unstable variability and trend estimates.

Current exclusions:

- `dataset-clinical`: 20 subjects excluded for `<5` nights
- `dataset-clinical-acc`: 22 subjects excluded for `<5` nights

### 1.8 Exploratory Classification Pipeline

The grouped-statistics classification notebook was converted into a Django utility pipeline. Important choices in the current exploratory pipeline are:

- no SMOTE,
- labels are taken from `Subject.diagnosis_code`,
- grouped statistics are used instead of all raw nightly features,
- `LeaveOneOut` is used for outer evaluation,
- `StratifiedKFold` is used for hyperparameter tuning,
- PCA projection is generated before scenario runs,
- SHAP outputs are generated with a fallback explainer if TreeExplainer fails.

The diagnostic label mapping is:

- `0 = HC`
- `1 = NonHC`
- `2 = MCI-AD`
- `3 = preDLB`

### 1.9 Publication-Oriented Strict Pipeline

A second, stricter pipeline was implemented for publication-oriented evaluation. It uses nested cross-validation:

- outer `LeaveOneOut` for evaluation,
- inner CV for hyperparameter tuning,
- and threshold tuning only inside training data.

This strict pipeline is methodologically preferable for publication, but it has not yet been run because it is computationally expensive.

### 1.10 Feature Ablation Pipeline

An additional feature-block ablation pipeline was implemented to understand whether some signal is diluted when all feature families are combined.

The following blocks were defined:

- `all`
- `diary-only`
- `actigraphy-only`
- `activity-only`
- `norm-only`
- `non-norm-only`
- `level-only`
- `trend-only`
- `variability-only`

## 2. Data Quality Checks and Diagnostics

### 2.1 Verification of COBEN Labels Against `KARDIOVIZE.xlsx`

For all subjects with IDs of the form `COBEN-XXXX`, the new label mapping was verified against `KARDIOVIZE.xlsx`:

- `HC -> HC`
- `at risk -> NonHC`
- `MCI -> MCI-AD`
- `MCI-LB -> preDLB`

The verification showed that:

- `diagnosis_code` in the Django database matches the KARDIOVIZE file,
- `AGE` and `SEX` also matched,
- the old `#Disease` column in exported Excel files contained stale values for some subjects,
- but the modeling pipeline does **not** use the old `#Disease` column.

Therefore, the current model labels are based on the correct source.

### 2.2 PCA Diagnostic

PCA projection was added to the classification pipeline as a quick visual diagnostic before scenario runs.

From the dedicated diagnostic workbook:

- `dataset-clinical`: silhouette by diagnosis = `-0.0895`
- `dataset-clinical-acc`: silhouette by diagnosis = `-0.0962`

These values indicate strong class overlap rather than clear class separation in the current feature space.

Other diagnostic findings from the same check:

- `dataset-clinical`: 257 subjects, 130 features in the diagnostic snapshot
- `dataset-clinical-acc`: 254 subjects, 345 features in the diagnostic snapshot
- 73 stale Excel-vs-DB label mismatches were found in metadata exports, but these were not the labels used for training
- 3 age outliers were detected at that stage and were later corrected

Overall, the PCA suggests that the current classification difficulty is at least partly a real data-separation issue, not just a modeling bug.

## 3. Classification Scenarios

The following binary scenarios were used throughout the exploratory runs:

1. `preDLB vs HC`
2. `preDLB + MCI-AD vs HC`
3. `MCI-AD vs HC`
4. `preDLB + MCI-AD + NonHC vs HC`
5. `HC vs NonHC`

Unless stated otherwise, the main interpretation should focus on the default metrics. The tuned-threshold metrics are informative, but they are secondary and more optimistic.

## 4. Exploratory Results: Initial Grouped-Statistics Pipeline

### 4.1 `dataset-clinical` Baseline

Run folder:

- `media/classification/grouped-statistics/dataset-clinical/20260330_110830`

Cohort summary:

- `HC = 78`
- `NonHC = 83`
- `MCI-AD = 30`
- `preDLB = 60`
- `missing diagnosis_code = 14`

#### Default Metrics

| Scenario | BACC | MCC | ROC AUC | PR AUC |
| --- | ---: | ---: | ---: | ---: |
| `HC vs NonHC` | 0.5504 | 0.1021 | 0.5712 | 0.5494 |
| `MCI-AD vs HC` | 0.4962 | -0.0100 | 0.6021 | 0.3714 |
| `preDLB + MCI-AD + NonHC vs HC` | 0.5387 | 0.1267 | 0.6147 | 0.7831 |
| `preDLB + MCI-AD vs HC` | 0.6043 | 0.2302 | 0.6489 | 0.6403 |
| `preDLB vs HC` | 0.5731 | 0.1457 | 0.5923 | 0.5113 |

#### Tuned-Threshold Metrics

| Scenario | BACC | MCC | Threshold |
| --- | ---: | ---: | ---: |
| `HC vs NonHC` | 0.5470 | 0.1668 | 0.3300 |
| `MCI-AD vs HC` | 0.6269 | 0.2306 | 0.1818 |
| `preDLB + MCI-AD + NonHC vs HC` | 0.5762 | 0.1967 | 0.6509 |
| `preDLB + MCI-AD vs HC` | 0.6440 | 0.2908 | 0.6569 |
| `preDLB vs HC` | 0.5622 | 0.2188 | 0.0550 |

Interpretation:

- the best baseline signal was `preDLB + MCI-AD vs HC`,
- `preDLB vs HC` was modest,
- `MCI-AD vs HC` was very weak,
- and overall performance was not yet compelling.

### 4.2 `dataset-clinical-acc` Baseline

Run folder:

- `media/classification/grouped-statistics/dataset-clinical-acc/20260330_165409`

Cohort summary:

- `HC = 82`
- `NonHC = 83`
- `MCI-AD = 30`
- `preDLB = 59`
- `missing diagnosis_code = 8`

#### Default Metrics

| Scenario | BACC | MCC | ROC AUC | PR AUC |
| --- | ---: | ---: | ---: | ---: |
| `HC vs NonHC` | 0.4783 | -0.0442 | 0.4644 | 0.4871 |
| `MCI-AD vs HC` | 0.5102 | 0.0189 | 0.5179 | 0.2692 |
| `preDLB + MCI-AD + NonHC vs HC` | 0.5434 | 0.0880 | 0.5072 | 0.6718 |
| `preDLB + MCI-AD vs HC` | 0.5569 | 0.1322 | 0.5499 | 0.5778 |
| `preDLB vs HC` | 0.4897 | -0.0204 | 0.5360 | 0.4304 |

#### Tuned-Threshold Metrics

| Scenario | BACC | MCC | Threshold |
| --- | ---: | ---: | ---: |
| `HC vs NonHC` | 0.5320 | 0.0710 | 0.5028 |
| `MCI-AD vs HC` | 0.5659 | 0.1468 | 0.4716 |
| `preDLB + MCI-AD + NonHC vs HC` | 0.5632 | 0.1354 | 0.3205 |
| `preDLB + MCI-AD vs HC` | 0.5543 | 0.1767 | 0.9728 |
| `preDLB vs HC` | 0.5677 | 0.1546 | 0.2581 |

Interpretation:

- adding the activity feature block in this first version did **not** improve results,
- in fact, the clinical + activity dataset was mostly worse than clinical-only,
- which suggested that the combined feature space might contain too much noise or redundancy.

## 5. Improved Exploratory Run After Pipeline Changes

After the baseline results, the following changes were introduced:

- additional grouped statistics: `Mean`, `Median`, `Min`, `Max`, `Slope`
- exclusion of subjects with fewer than 5 nights
- constant-feature removal
- supervised feature selection inside the pipeline

### 5.1 `dataset-clinical` Improved Exploratory Run

Run folder:

- `media/classification/grouped-statistics/dataset-clinical/20260403_213148`

Cohort summary after filtering:

- `HC = 77`
- `NonHC = 79`
- `MCI-AD = 30`
- `preDLB = 59`
- `missing diagnosis_code = 0`

#### Default Metrics

| Scenario | BACC | MCC | ROC AUC | PR AUC |
| --- | ---: | ---: | ---: | ---: |
| `preDLB vs HC` | 0.6258 | 0.2516 | 0.6300 | 0.5849 |
| `preDLB + MCI-AD vs HC` | 0.5561 | 0.1169 | 0.6015 | 0.6396 |
| `MCI-AD vs HC` | 0.5275 | 0.0526 | 0.5658 | 0.4124 |
| `preDLB + MCI-AD + NonHC vs HC` | 0.5419 | 0.1053 | 0.6180 | 0.7823 |
| `HC vs NonHC` | 0.5858 | 0.1859 | 0.6454 | 0.6520 |

#### Tuned-Threshold Metrics

| Scenario | BACC | MCC | Threshold |
| --- | ---: | ---: | ---: |
| `preDLB vs HC` | 0.6447 | 0.2877 | 0.4504 |
| `preDLB + MCI-AD vs HC` | 0.5737 | 0.1910 | 0.2235 |
| `MCI-AD vs HC` | 0.5768 | 0.3000 | 0.8148 |
| `preDLB + MCI-AD + NonHC vs HC` | 0.6063 | 0.2034 | 0.9592 |
| `HC vs NonHC` | 0.6247 | 0.2989 | 0.8653 |

Interpretation:

- `preDLB vs HC` improved and became the most promising individual scenario,
- `HC vs NonHC` also improved,
- mixed scenarios remained only modest,
- `MCI-AD vs HC` remained weak.

### 5.2 Feature Selection Behavior in the Improved Run

Feature selection was active in this run, but it did not prune all scenarios equally:

| Scenario | Candidate Features | Selected Features | Best `k` |
| --- | ---: | ---: | --- |
| `preDLB vs HC` | 259 | 259 | `all` |
| `preDLB + MCI-AD vs HC` | 259 | 259 | `all` |
| `MCI-AD vs HC` | 258 | 20 | `20` |
| `preDLB + MCI-AD + NonHC vs HC` | 259 | 259 | `all` |
| `HC vs NonHC` | 259 | 80 | `80` |

Interpretation:

- feature selection was useful for some scenarios,
- but the preDLB-oriented scenarios often preferred to keep the full candidate set,
- suggesting that the added features were not universally harmful and that the signal may be distributed across many correlated variables.

## 6. Feature-Block Ablation Experiment

An ablation experiment was then run to determine whether specific feature families were more informative than the full combined set.

Run folder:

- `media/classification/grouped-statistics-ablation/dataset-clinical-acc/20260404_100750`

This ablation was performed on `dataset-clinical-acc`.

### 6.1 Best Default-Metric Block per Scenario

| Scenario | Best Feature Block | BACC | MCC | ROC AUC | PR AUC |
| --- | --- | ---: | ---: | ---: | ---: |
| `HC vs NonHC` | `activity-only` | 0.6416 | 0.2989 | 0.7056 | 0.7054 |
| `MCI-AD vs HC` | `trend-only` | 0.5478 | 0.1081 | 0.5825 | 0.4068 |
| `preDLB + MCI-AD + NonHC vs HC` | `norm-only` | 0.5960 | 0.2445 | 0.5983 | 0.7298 |
| `preDLB + MCI-AD vs HC` | `non-norm-only` | 0.5586 | 0.1255 | 0.5653 | 0.6024 |
| `preDLB vs HC` | `norm-only` | 0.6140 | 0.2292 | 0.6604 | 0.5852 |

### 6.2 Best Tuned-Metric Block per Scenario

| Scenario | Best Feature Block | BACC | MCC | Threshold |
| --- | --- | ---: | ---: | ---: |
| `HC vs NonHC` | `activity-only` | 0.6737 | 0.3514 | 0.5854 |
| `MCI-AD vs HC` | `trend-only` | 0.5803 | 0.2594 | 0.6859 |
| `preDLB + MCI-AD + NonHC vs HC` | `activity-only` | 0.6285 | 0.2529 | 0.6791 |
| `preDLB + MCI-AD vs HC` | `level-only` | 0.5766 | 0.1842 | 0.1648 |
| `preDLB vs HC` | `norm-only` | 0.6404 | 0.2906 | 0.1605 |

### 6.3 Mean Default Performance Across Scenarios by Block

| Feature Block | Mean BACC | Mean MCC | Mean ROC AUC | Mean PR AUC |
| --- | ---: | ---: | ---: | ---: |
| `non-norm-only` | 0.5717 | 0.1563 | 0.6038 | 0.5963 |
| `norm-only` | 0.5602 | 0.1353 | 0.5773 | 0.5620 |
| `actigraphy-only` | 0.5552 | 0.1191 | 0.5915 | 0.5763 |
| `variability-only` | 0.5468 | 0.1005 | 0.5638 | 0.5491 |
| `trend-only` | 0.5468 | 0.1123 | 0.5677 | 0.5647 |
| `diary-only` | 0.5466 | 0.1005 | 0.5782 | 0.5618 |
| `all` | 0.5466 | 0.1005 | 0.5782 | 0.5618 |
| `activity-only` | 0.5422 | 0.0911 | 0.5794 | 0.5726 |
| `level-only` | 0.5282 | 0.0607 | 0.5256 | 0.5085 |

Interpretation:

- the full combined feature set was **not** the best-performing block on average,
- `non-norm-only` and `norm-only` were more competitive than expected,
- `activity-only` was especially promising for `HC vs NonHC`,
- `norm-only` was the best block for `preDLB vs HC`,
- `trend-only` was the strongest block for `MCI-AD vs HC`,
- and this suggests that scenario-specific signal may be diluted when all feature families are merged together.

This was an important result because earlier there was concern that the `*_norm` features were too sparse. The ablation indicates they should not be discarded blindly.

## 7. Current Scientific Interpretation

### 7.1 What Looks Promising

- `preDLB vs HC` is the most promising scenario so far.
- `HC vs NonHC` may also contain signal, especially when focusing on activity-only features in the ablation run.
- Some signal seems to live in specific feature blocks rather than in the full combined feature set.

### 7.2 What Looks Weak

- `MCI-AD vs HC` remains weak across runs.
- `preDLB + MCI-AD vs HC` and `preDLB + MCI-AD + NonHC vs HC` are only modest.
- The combined clinical + activity feature space did not automatically improve performance and may currently add noise unless it is pruned or split more carefully.

### 7.3 Why Results Are Still Exploratory

The currently executed classification runs are exploratory rather than publication-grade because hyperparameter tuning was not nested inside the outer leave-one-out loop. The stricter nested pipeline has been implemented but not yet executed.

Therefore:

- the current results are useful for comparing directions,
- but they should not yet be treated as the final unbiased performance estimates for a paper.

## 8. Main Lessons Learned So Far

1. The project is now in a much stronger engineering state: the end-to-end utilities exist and the pipeline is reproducible in Django.
2. A substantial fraction of early failures were caused by data/date mismatches or corrupted intermediate outputs, not by the modeling code itself.
3. The current diagnostic feature space shows heavy overlap between diagnostic groups.
4. Variability-only features were probably too restrictive.
5. Adding level and trend statistics improved some scenarios, especially `preDLB vs HC`.
6. Feature selection helped in some scenarios, but not enough to create a clear breakthrough.
7. Feature-block ablation suggests that specific groups of features may carry more signal than the union of all features.
8. The `norm` features, despite being sparse, appear to contain useful information and should be retained for now.

## 9. Recommended Next Steps

### 9.1 Recommended Before Running the Strict Pipeline

The ablation results suggest that the next steps should focus on smarter model inputs rather than immediately running the strict nested pipeline on everything.

Recommended next experiments:

1. Run the strict pipeline only on the most promising scenario(s), especially `preDLB vs HC`.
2. Compare at least these blocks in the strict setting:
   - `all`
   - `norm-only`
   - `non-norm-only`
   - optionally `activity-only` for `HC vs NonHC`
3. Consider simpler baselines such as regularized logistic regression or linear SVM.
4. Investigate whether cohort effects or site effects are contributing to the overlap.
5. Consider whether some scenarios should remain separated rather than merged.

### 9.2 Publication Strategy

If the next strict runs remain similar to the exploratory results, the most realistic publication position would be:

- present this work as an exploratory feasibility study,
- emphasize the pipeline development and data harmonization effort,
- report that some modest signal was observed for `preDLB vs HC`,
- and explain that stronger performance will likely require either richer features, cleaner cohorts, or a larger sample.

## 10. Main Output Locations

### Key Exploratory Classification Runs

- `media/classification/grouped-statistics/dataset-clinical/20260330_110830`
- `media/classification/grouped-statistics/dataset-clinical-acc/20260330_165409`
- `media/classification/grouped-statistics/dataset-clinical/20260403_213148`

### Key Ablation Run

- `media/classification/grouped-statistics-ablation/dataset-clinical-acc/20260404_100750`

### PCA / Diagnostics

- `media/classification/grouped-statistics-diagnostics/20260403_103712`

### Label Verification

- `media/classification/kardiovize-label-verification/20260403_200900`

## Closing Note

From a software and data-engineering perspective, substantial progress was made: the notebooks were converted into reusable utilities, the pipeline became much more robust, and the project now supports systematic comparison experiments. From a scientific perspective, the current results are encouraging enough to justify more focused follow-up experiments, but not yet strong enough to claim a clinically useful classifier.
