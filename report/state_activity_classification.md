# Classifier comparison for separate sleep and wake activity

This is a post hoc exploratory extension of the secondary classification audit.
It does not replace the primary clinical association analysis, constitute external
validation, or change any existing manuscript results. The feature block is fixed
before this experiment is run; it must not be repeatedly changed to improve AUC.

## Question and fixed comparison

Does adding activity summaries from predicted sleep and nocturnal wake improve
HC (numeric diagnosis 0) versus MCI-LB (numeric diagnosis 3) discrimination?

- Baseline: the existing stable sleep and whole-interval activity-variability
  feature families, with their ten established across-night summaries.
- Augmented: exactly that baseline plus the 16 state activity features (median
  and IQR across nights of mean, median, P90 and IQR within each state), and four
  targeted summaries (across-night IQR of MAD, and SD of relative interdecile
  range, separately for sleep and wake). No new whole-interval copy is added.

These are the already extracted features, not a new screen of clinical scales.
No UPDRS, RBDq, MFS or cognitive score is a predictor or eligibility requirement.
The target extraction's outcome-filtered visit tables are deliberately not used.
Source and visit identifiers are audit metadata, not wearable predictors.

Require at least five jointly valid nights, within the frozen extended manuscript
cohort. Apply the completed primary extraction's coverage, state support and
mixed-minute rules unchanged. Recompute **both** the baseline and state summaries
on the same nights. Consequently baseline performance need not reproduce the
older 129-visit classifier: the new matched sample is smaller and some visits
have fewer retained nights. This is the comparator needed to isolate the added
feature block, not evidence that an older result was wrong.

State labels come from the existing sleep/wake model, not PSG or REM labels.
Wake means wake within the estimated nocturnal sleep interval, not daytime wake.
The baseline retains its original whole-interval minute-boundary convention;
state summaries retain the validated fully covered minute convention.

## Validation

Both feature sets use the same participants, nights, row order, person grouping,
outer and inner folds, random seed, and tuning budget **within each algorithm**.
All visits of a person stay together. Models are XGBoost with the existing
foldwise KBest selector and elastic-net logistic regression with embedded
shrinkage. Search spaces and tuning scoring (balanced accuracy) are unchanged.
Sex and education adjustment is mandatory and fitted inside training folds.
Missing predictors and covariates are imputed in training folds. Age is not
added, consistent with the existing primary classifier adjustment decision.
There is no global clinical-score feature ranking or global missingness-based
feature screen in this experiment.

Primary performance uses pooled outer-held-out probabilities and the default
decision threshold. AUC is the primary comparison; PR-AUC, balanced accuracy,
MCC, Brier score and log loss are secondary descriptive checks. The paired
difference is **augmented minus baseline**. For AUC/PR-AUC/BACC/MCC positive is
better; for Brier/log loss negative is better. The 95% percentile intervals use
10,000 participant-level resamples, stratified by diagnosis, preserving every
visit and its multiplicity. They condition on saved fitted OOF predictions, do
not include retraining uncertainty, and are not p-values. No multiplicity-adjusted
confirmatory significance claim is planned.

Threshold-tuned and first-visit results are secondary. Source-stratified OOF
metrics, a separate source-only ascertainment benchmark, within-source nested
CV, and leave-one-source-out fitting are retained. Infeasible source validation
is explicitly reported as skipped. Source is diagnosis-enriched by recruitment;
source-only performance is not evidence of acquisition artefacts by itself.
Transportability claims require the actual source holdouts, not just stratified
metrics. Source holdouts fail if a participant crosses sources.

## Run

From the repository root, activate the working environment:

```sh
source /Volumes/Portable/geneactiv-processing-data/.venv/bin/activate
```

Input validation only (no training):

```sh
python manage.py run_state_activity_classification \
  --state-input-dir media/state-activity/20261026_130149_395288 \
  --target-input-dir media/state-activity-targeted/20261026_162604_438589 \
  --prepare-only
```

Full experiment (remove `--prepare-only`):

```sh
python manage.py run_state_activity_classification \
  --state-input-dir media/state-activity/20261026_130149_395288 \
  --target-input-dir media/state-activity-targeted/20261026_162604_438589
```

Defaults: both algorithms, up to five outer and five inner person-grouped folds,
20 XGBoost and 30 elastic-net search draws per fit, 10,000 paired bootstrap
resamples per algorithm, plus source validation fits. This is not a quick
four-fit run. Existing environment variables can override the budgets:
`GENEACTIV_PERSON_GROUPED_OUTER_CV_SPLITS`,
`GENEACTIV_PERSON_GROUPED_INNER_CV_SPLITS`, `GENEACTIV_STRICT_SEARCH_ITER`, and
`GENEACTIV_ELASTIC_NET_SEARCH_ITER`. Set these before starting Python. Unset
them to use the documented defaults. The actual values are saved in each run.
Use `--models xgboost` or `--models elastic_net` only for an explicitly partial
experiment. `--bootstrap-resamples` changes the saved interval resample count.
Reduced-budget runs are for development, not interchangeable with the full run.

Each invocation creates a new directory under `media/classification/state-activity`.
An explicit `--output-dir` must not already exist. Preparation-only runs cannot
be resumed as training runs; run the full command to create a separate result.
Existing predictions, extracted signals, database and manuscript files are not
modified. Full runs fit validation models only, not a final deployable model.

## Outputs

- `manifest.json`: status, input/code fingerprints, versions, exact feature lists,
  covariates, search settings and protocol. `prepared_only` is not a result.
- `table.csv`, `matched_nights.csv`, `cohort_audit.csv`: exact matched inputs and
  eligibility. No clinical scores are copied into the classifier input table.
- `split_plan.csv`: outer and inner train/test membership, common to all branches.
- `primary_metrics.csv`: one row per feature set and algorithm.
- `paired_feature_differences.csv`: observed differences and confidence intervals.
- Per branch: held-out `predictions.csv`, fold parameters, elastic-net coefficients,
  `secondary_metrics.csv`, and source-validation results/predictions. The source-only
  benchmark is saved once at the top level.

Branches are saved as they finish and completed pairs are checkpointed. A failure
sets the run status to `failed`; inspect it before interpreting partial output.
No automatic resume or overwriting of existing runs is performed. The manifest
must say `complete` before treating the full experiment as finished.
