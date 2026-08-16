# GeneActiv / preDLB Project Update Report

Prepared: 2026-06-07
Updated after feature-family reduction, diagnosis-adjusted GEE follow-up, HC-versus-preDLB focused visualization, dual reporting of correlation and adjusted effect estimates, strict nested-CV classification, comparison with the previous global-covariate classifier runs, all-diagnosis clinical-scale regression, focused HC-versus-preDLB regression on 2026-06-27, WASO-corrected correlation, strict RFE classification, focused HC-versus-preDLB regression, all-diagnosis clinical regression reruns on 2026-06-29, non-strict classification sensitivity runs on 2026-06-30, strict RFE probability diagnostic plots, official HC-versus-preDLB feature-family stability analysis, stable-family restricted HC-versus-preDLB association follow-up, strict stable-family HC-versus-preDLB classification, HC-versus-preDLB classification validity checks, and thesis-level person-grouped HC-versus-preDLB classification
Project path: `/Volumes/Portable/geneactiv-processing-data`

This report updates `phd_supervisor_experiment_report_2026-05-06.md`. It covers clinical-variable integration, education harmonization, scenario-specific covariate verification, feature-family reduction, the diagnosis-adjusted follow-up of exploratory feature-outcome associations, strict classification, classification validity checks, thesis-level person-grouped classifier validation, and first-pass regression prediction of clinical scores.

> **Interpretation correction, 2026-08-14:** Earlier versions used the terms
> "source-cohort confounding" and "source-only negative control" too strongly.
> The source cohorts were consortium-defined recruitment strata: HC/HC2 was
> recruited with an expectation of health, pre-LBD/pre-LBD2 included people
> with an expected clinical problem, and COBEN was not recruited using the same
> binary division. Source was therefore expected to be associated with the
> eventual label. The source-only analysis is an **ascertainment benchmark**,
> not proof of technical acquisition or protocol confounding. It still shows
> that pooled classification performance cannot be interpreted independently
> of this recruitment design. Within-source estimates are additionally limited
> by very small minority classes.

## 1) Executive Summary

1. Six clinical variables and education were integrated into `dashboard.models.Subject`.
2. Education was harmonized as years of education using `Delka_vzdelani` in preDLB and `EDU` in KARDIOVIZE.
3. Covariates are selected independently for each diagnostic scenario:

| Scenario | Covariates selected in both feature datasets |
|---|---|
| preDLB vs HC | gender, education |
| preDLB + MCI-AD vs HC | education |
| MCI-AD vs HC | education |

4. The pooled exploratory analysis tested 260 sleep/diary features and 690 sleep/diary + activity features against six clinical outcomes.
5. Redundant significant variants were reduced to interpretable feature-family representatives before follow-up.
6. Results are reported using both Spearman correlation coefficients (`rho`) and adjusted GEE regression coefficients (`beta`). Person-clustered robust 95% confidence intervals are included for both estimates.
7. The primary interpretation figures now focus exclusively on HC versus preDLB. The most convincing adjusted patterns are:
   - greater activity variability associated with higher UPDRS;
   - activity-distribution measures associated with attention and visuospatial performance;
   - greater variability in nocturnal awakenings associated with poorer executive performance;
   - the previously highlighted normalized wake-after-sleep-onset association
     with MFS became weaker after the seconds-versus-minutes WASO correction
     and is no longer retained as a focused candidate;
   - four associations in preDLB vs HC with evidence that the feature-outcome slope differs by diagnosis.
8. The isolated MCI-AD vs HC scenario produced no FDR-significant pooled candidates and therefore no second-stage models. This is an absence of detected evidence in the present sample, not evidence of equivalence.
9. The previously highlighted diary-normalized sleep-onset-latency association remains a strong pooled discovery, but its raw grouped feature is constant and cannot be estimated in the GEE follow-up. It should not be treated as the principal confirmed result.
10. The current WASO-corrected strict nested-CV classification analysis with
    inner-fold RFE provides the most interpretable classifier summary so far:
    - preDLB vs HC: ROC AUC `0.764`, balanced accuracy `0.668`;
    - preDLB + MCI-AD vs HC: ROC AUC `0.718`, balanced accuracy `0.634`;
    - MCI-AD vs HC: ROC AUC `0.669`, balanced accuracy `0.648`.
    The HC-versus-preDLB model therefore shows moderate, not strong,
    discrimination after leakage-resistant hyperparameter tuning and feature
    selection.
11. Inner-CV threshold optimization did not consistently improve the RFE
    classifiers. It worsened preDLB vs HC, improved the combined scenario, and
    changed MCI-AD only slightly. Because thresholds were unstable across
    outer folds, the default probability threshold of `0.5` remains the more
    defensible primary result.
12. Two validation risks were identified for diagnostic classification:
    repeated visits can be separated by database subject code rather than
    underlying person, and diagnostic group is strongly associated with source
    cohort. These risks are now explicitly tested in the 2026-07-01 audit and
    2026-07-02 person-grouped thesis validation.
13. Compared with the previous global-covariate strict classifier, the
    WASO-corrected RFE run improves all three scenarios in balanced accuracy,
    but the practical interpretation remains conservative. MCI-AD vs HC
    increased from AUC `0.482` to `0.669` and balanced accuracy from `0.463`
    to `0.648`. The preDLB vs HC strict result improved from AUC `0.710` to
    `0.764` and balanced accuracy from `0.640` to `0.668`.
14. Four non-strict classification sensitivity runs completed on 2026-06-30
    show higher apparent performance than the strict nested estimates, but
    should not replace them as the primary result. The best non-strict
    HC-versus-preDLB run was clinical non-RFE (`AUC = 0.804`, balanced accuracy
    `0.744`). The best combined preDLB + MCI-AD vs HC run was activity-enhanced
    non-RFE (`AUC = 0.774`, balanced accuracy `0.730`). The best MCI-AD vs HC
    run remained clinical RFE (`AUC = 0.689`, balanced accuracy `0.674`). These
    runs support the presence of signal but are less conservative than the
    strict nested RFE analysis.
15. The strict HC-versus-preDLB RFE probability diagnostics show a moderate
    separation signal even though default-threshold classification is
    imperfect. Median predicted preDLB probability is `0.289` for HC and
    `0.844` for preDLB, ROC AUC is `0.764`, and average precision is `0.753`.
    However, the distributions overlap and the calibration curve is not close
    enough to treat the scores as absolute clinical risk estimates.
16. The official HC-versus-preDLB feature-family stability analysis now
    aggregates strict classification, non-strict classification, correlation,
    and GEE evidence using fixed family definitions. The strongest primary
    cross-method families are long awakenings, sleep onset latency, sleep
    efficiency, and wake-bout frequency. WASO remains cross-method supported
    after correction, but the old normalized-WASO/MFS focused candidate remains
    downgraded. Activity variability is supported only in the activity-enhanced
    analysis stream. Alcohol, caffeine, rest quality, day sleep, and
    sleeping-pill variables are retained as secondary, confounding-sensitive
    predictors.
17. Stable-family restricted HC-versus-preDLB confirmation was run on
    2026-07-01 using only the fixed families above. In the sleep-only primary
    family set, retained adjusted associations are concentrated in executive
    performance and long-awakening variability. In the sleep-plus-activity
    set, activity variability adds retained associations with UPDRS and
    attention, supporting its role as a separate activity-enhanced extension.
    Most visuospatial and sleep-onset-latency pooled findings weaken after
    diagnosis/covariate adjustment or cannot be estimated from the raw grouped
    feature.
18. Corrected strict stable-family HC-versus-preDLB classifiers preserve a
    moderate signal but do not improve diagnostic performance over the broader
    strict RFE model. Primary sleep families give ROC AUC `0.691` and balanced
    accuracy `0.664`; primary sleep plus activity variability gives ROC AUC
    `0.690` and balanced accuracy `0.650`.
19. A dedicated HC-versus-preDLB classification validity audit was added on
    2026-07-01. First-visit-only sensitivity reduced apparent performance in
    all strict runs: broad strict RFE decreased from AUC `0.764` / BACC
    `0.668` to AUC `0.736` / BACC `0.655`; stable sleep decreased from AUC
    `0.691` / BACC `0.664` to AUC `0.647` / BACC `0.637`; stable sleep plus
    activity decreased from AUC `0.690` / BACC `0.650` to AUC `0.649` / BACC
    `0.619`.
20. The same validity audit confirmed strong source-label enrichment. Diagnosis
    and source were strongly associated (`Cramer's V` approximately
    `0.68-0.69`, `p < 5e-14`), as expected from the consortium-defined
    recruitment strata. A source-only ascertainment benchmark reached BACC
    approximately `0.84`. This does not prove acquisition/protocol confounding
    or that the wearable classifier used source identity, but it prevents the
    pooled diagnostic performance from being interpreted independently of the
    recruitment design. Classification should therefore remain exploratory and
    secondary.
21. A full thesis-level person-grouped nested-CV HC-versus-preDLB validation
    was run on 2026-07-02. The broad RFE model preserved moderate ranking
    signal (`AUC = 0.746`, default `BACC = 0.649`, tuned `BACC = 0.666`), but
    the source-only ascertainment benchmark was higher (`AUC = 0.814`,
    `BACC = 0.840`). Stable primary sleep families gave similar balanced
    accuracy (`AUC = 0.706`, `BACC = 0.663`), while adding activity variability
    weakened diagnostic classification (`AUC = 0.620`, `BACC = 0.604`).
22. Source-sensitive validation does not yet establish transportability beyond
    the pooled recruitment design. Within-source nested-CV point estimates were
    weak in COBEN and pre-LBD/pre-LBD2, but both analyses had severe class
    imbalance; HC/HC2 could not be estimated because it contained only one
    preDLB subject. These results do not prove technical source confounding or
    absence of a disease signal. They do argue against article-level diagnostic
    claims until validation includes adequate numbers of both groups within
    sources or an independent cohort.
23. The WASO-corrected all-diagnosis clinical-scale regression remains
    exploratory prediction, not a ready clinical estimator. In the sleep/diary
    plus lifestyle dataset, only visuospatial and executive scores improved
    over a foldwise median baseline. RBDq, UPDRS, MFS, and attention had worse
    MAE than the baseline despite non-zero prediction correlations. The
    strongest point estimate was visuospatial performance (`R2 = 0.147`,
    Pearson `r = 0.392`), followed by executive performance (`R2 = 0.061`,
    Pearson `r = 0.255`).
24. The WASO-corrected focused HC-versus-preDLB regression reruns keep
    the same main conclusion: individual clinical-scale prediction remains
    exploratory. The strongest corrected focused result is visuospatial
    performance in the clinical-core model (`MAE = 0.71`, `R2 = 0.177`,
    Pearson `r = 0.426`, `9.8%` MAE improvement over baseline). RBDq still
    improves modestly in the activity-core model (`MAE = 1.66`, `6.3%` better
    than baseline). Executive performance and attention improve only modestly
    (`5.5%` and `4.2%` MAE improvement in their best variants). UPDRS and MFS
    still do not beat the foldwise median baseline in MAE.

### 1.1 WASO Correction Update

The actigraphy WASO normalization issue identified during interpretation has
now been corrected and the source Excel exports have been regenerated:

- `dataset-clinical.xlsx`, regenerated on 2026-06-29 at 13:25;
- `dataset-clinical-acc.xlsx`, regenerated on 2026-06-29 at 13:25.

All correlation and feature-family interpretation should now use only these
canonical WASO-corrected analysis runs:

- sleep/diary: `media/analysis-preparation/dataset-clinical/20260629_132528_560853/`
- sleep/diary + activity: `media/analysis-preparation/dataset-clinical-acc/20260629_132533_149540/`

After this correction, normalized-WASO findings are weaker. In particular,
`MFS ~ Range.actigraphy_norm.Wake after sleep onset` is no longer a focused
candidate and should not be presented as a retained finding. The HC vs preDLB
analysis still shows meaningful associations, especially in the activity-
enhanced dataset, where UPDRS, attention, visuospatial performance, and
executive outcomes retain interpretable candidate families.

## 2) Clinical Data Integration

### 2.1 Variables added to the Subject model

The following nullable subject-level variables are available:

- `rbdq`
- `updrs`
- `mfs`
- `visuospatial`
- `attention`
- `executive`
- `education_years`

The visit is represented by the database subject code:

- first visit: `pre-LBD-XXX` or `HC-XXX`;
- second visit: `pre-LBD2-XXX` or `HC2-XXX`.

Second-visit subjects receive second-visit clinical values from the preDLB workbook. Education is treated as a person-level characteristic and is copied consistently to both visits.

### 2.2 Import coverage

The database contains `412` subjects.

| Variable | Subjects with a non-null value |
|---|---:|
| RBDq | 268 |
| UPDRS | 265 |
| MFS | 266 |
| Visuospatial | 268 |
| Attention | 269 |
| Executive | 269 |
| Education (years) | 266 |

Four subjects from the target cohorts remain without education because the source workbooks contain no usable value:

- `COBEN-45`
- `COBEN-1087`
- `pre-LBD-22`
- `pre-LBD2-22`

## 3) Education Harmonization

The preDLB workbook contains:

- `Vzdelani`: categorical education level;
- `Delka_vzdelani`: duration of education in years.

KARDIOVIZE `EDU` is also expressed as years of education. The common quantitative variable therefore uses:

- preDLB/HC: `preDLB_shared.xlsx`, sheet `PSY_RAW`, column `Delka_vzdelani`;
- KARDIOVIZE/COBEN: `KARDIOVIZE.xlsx`, sheet `ID_Clinical_Cognitive`, column `EDU`.

This avoids assigning one approximate duration to every person with the same Czech degree category.

## 4) Scenario-Specific Covariate Verification

### 4.1 Unit of analysis and tests

All covariate tests use one observation per database subject:

- age: Welch independent-samples t-test;
- education: Welch independent-samples t-test;
- gender: Pearson chi-squared test;
- exploratory threshold: `p < 0.05`.

### 4.2 Results

The decisions were consistent across the two feature datasets:

| Scenario | Age | Education | Gender |
|---|---|---|---|
| preDLB vs HC | do not select | **select** | **select** |
| preDLB + MCI-AD vs HC | do not select | **select** | do not select |
| MCI-AD vs HC | do not select | **select** | do not select |

For the main sleep/diary dataset:

| Scenario | Age p | Education p | Gender p |
|---|---:|---:|---:|
| preDLB vs HC | 0.1799 | 0.0000066 | 0.0102 |
| preDLB + MCI-AD vs HC | 0.1637 | 0.0000289 | 0.1110 |
| MCI-AD vs HC | 0.3927 | 0.0368 | 0.5346 |

In preDLB vs HC:

- HC age: `68.92 +/- 5.87` years;
- preDLB age: `70.35 +/- 6.38` years;
- HC education: `15.71 +/- 3.15` years;
- preDLB education: `13.47 +/- 2.38` years;
- HC gender distribution: `39 F / 39 M`;
- preDLB gender distribution: `43 F / 17 M`.

The principal measured imbalances are therefore education and, specifically for preDLB vs HC, gender.

## 5) Revised Analysis Architecture

Each analysis creates a timestamped, non-destructive run under:

`media/analysis-preparation/<dataset>/<timestamp>/`

For each scenario, the pipeline:

1. re-runs covariate verification;
2. selects the scenario-specific covariates;
3. attaches subject-level clinical outcomes;
4. adjusts nightly features for exploratory pooled testing;
5. aggregates nightly measurements to subject/visit-level summaries;
6. performs normality, Mann-Whitney U, and feature-outcome correlation analyses;
7. reduces significant correlated variants into interpretable feature families;
8. fits diagnosis-aware GEE follow-up models for the retained family representatives.

Previous runs and source data are retained.

For classification, covariate residualization is fitted inside each cross-validation training fold and then applied to held-out observations. This avoids leakage from global pre-adjustment.

The covariate adjustment code now also detects constant features and leaves them unchanged rather than generating invalid residuals.

The strict classifier additionally nests hyperparameter and feature-count
selection inside each outer evaluation fold. Threshold selection, when used,
is based only on predictions generated inside the corresponding outer
training fold.

## 6) Statistical Methods

### 6.1 Diagnostic scenarios

1. preDLB vs HC;
2. preDLB + MCI-AD vs HC;
3. MCI-AD vs HC.

### 6.2 Feature sets

- `dataset-clinical`: 260 aggregated sleep, actigraphy, diary, and normalized features;
- `dataset-clinical-acc`: 690 features, adding activity-distribution and activity-variability measures.

### 6.3 Pooled exploratory stage

For each scenario and feature set:

1. Shapiro-Wilk normality tests were calculated separately in each diagnostic group.
2. Group differences were tested using two-sided Mann-Whitney U tests.
3. Group-test p-values were corrected across all tested features using Benjamini-Hochberg FDR.
4. Spearman correlations were calculated between every feature and RBDq, UPDRS, MFS, visuospatial, attention, and executive scores.
5. Correlation p-values were FDR-corrected separately across features for each clinical outcome.
6. The significance threshold was FDR-adjusted `p < 0.05`.

For the selected associations presented in this report, 95% confidence
intervals around pooled Spearman `rho` were additionally estimated by fitting
a Gaussian GEE model to standardized ranks of the feature and outcome. The
point estimate from this model is the Spearman correlation coefficient, while
the robust sandwich covariance is clustered by underlying person. This
prevents repeated visits from being treated as fully independent when
estimating uncertainty. The original discovery FDR p-values remain the values
used for feature screening; these robust confidence intervals were added to
quantify uncertainty for the selected report results and did not change
feature selection.

At least one diagnostic group was non-normal for approximately `94%` to `97%` of testable sleep/diary features and `87%` to `91%` of extended features. This supports the robust, rank-based exploratory tests.

### 6.4 Feature-family reduction

FDR-significant correlation candidates were grouped using:

- conceptual feature families based on their source measurement and summary statistic;
- complete-linkage clustering of highly correlated variants using absolute Spearman correlation `|rho| >= 0.85`;
- one representative feature from each resulting cluster.

This reduces multiple near-duplicate mean, median, SD, MAD, IQR, range, percentile, and normalized variants without pretending that each row is an independent biological discovery.

### 6.5 Diagnosis-adjusted follow-up

For each retained feature-outcome pair:

- within-diagnosis associations were estimated using rank-based slopes with GEE robust inference;
- adjusted associations used the raw grouped feature, standardized to one standard deviation;
- models included diagnosis and the scenario-specific covariates;
- feature-by-diagnosis interactions tested whether slopes differed between diagnostic groups;
- diagnosis-specific slopes and 95% confidence intervals were derived from the interaction model;
- repeated visits were clustered by underlying person, so `HC2/HC3` and `pre-LBD2/pre-LBD3` visits were not treated as independent people;
- an independence working correlation was used with robust sandwich covariance;
- FDR correction was applied separately by clinical outcome for stratified, adjusted, interaction, and diagnosis-specific slope results.

Adjusted GEE coefficients are expressed as change in the clinical outcome for a one-standard-deviation increase in the raw grouped feature.

### 6.6 Why both `rho` and `beta` are reported

The two estimates answer related but different questions:

- **Spearman `rho`** describes the direction and strength of the pooled
  monotonic association after scenario-specific feature preprocessing. It is
  unitless and always ranges from `-1` to `+1`. A value farther from zero
  indicates a stronger relationship. The correlation model does not include
  diagnosis directly; selected covariates may already have been removed from
  the nightly feature values during preprocessing.
- **Adjusted `beta`** estimates how much the clinical outcome changes for a
  one-standard-deviation increase in the feature after diagnosis and the
  scenario-specific covariates have been included in the model. It therefore
  tests whether the association remains after accounting for these potential
  alternative explanations.

`Beta` was introduced as a second-stage robustness and interpretation
estimate, not as a replacement for the correlation coefficient. A significant
`rho` establishes an exploratory association; a retained adjusted `beta`
provides stronger evidence that this association is not explained solely by
diagnostic-group separation or the selected covariates. Conversely, a
significant `rho` with a non-significant adjusted `beta` suggests that group
composition or covariates may contribute to the pooled relationship.

The numerical values of `rho` and `beta` are not directly comparable. `Rho` is
unitless, whereas `beta` is expressed in units of the clinical outcome per
feature standard deviation. In addition, the pooled discovery used nightly
covariate residualization before aggregation, while the GEE follow-up models
the raw grouped feature and covariates directly. Their magnitudes, and
occasionally their signs, can therefore differ.

## 7) Pooled Discovery Results

### 7.1 FDR-significant counts

| Feature set | Scenario | Features tested | Group differences | Feature-outcome correlations | Family representatives |
|---|---|---:|---:|---:|---:|
| Sleep/diary | preDLB vs HC | 260 | 19 | 36 | 22 |
| Sleep/diary | preDLB + MCI-AD vs HC | 260 | 22 | 35 | 25 |
| Sleep/diary | MCI-AD vs HC | 260 | 0 | 0 | 0 |
| Sleep/diary + activity | preDLB vs HC | 690 | 27 | 83 | 46 |
| Sleep/diary + activity | preDLB + MCI-AD vs HC | 690 | 32 | 59 | 36 |
| Sleep/diary + activity | MCI-AD vs HC | 690 | 0 | 0 | 0 |

### 7.2 Significant correlations by outcome

| Feature set | Scenario | RBDq | UPDRS | MFS | Visuospatial | Attention | Executive | Total |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Sleep/diary | preDLB vs HC | 15 | 0 | 0 | 10 | 0 | 11 | 36 |
| Sleep/diary | preDLB + MCI-AD vs HC | 0 | 0 | 0 | 1 | 19 | 15 | 35 |
| Sleep/diary | MCI-AD vs HC | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Extended | preDLB vs HC | 0 | 56 | 0 | 12 | 10 | 5 | 83 |
| Extended | preDLB + MCI-AD vs HC | 0 | 27 | 0 | 0 | 28 | 4 | 59 |
| Extended | MCI-AD vs HC | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

### 7.3 Main pooled patterns

The recurring feature families were:

- sleep-onset timing;
- wake after sleep onset or offset;
- awakenings longer than five minutes;
- activity distribution, dispersion, and robust variability.

The strongest pooled association was diary-normalized sleep-onset latency versus visuospatial performance in preDLB vs HC:

- sleep/diary set: `rho = -0.588`, FDR p `5.15e-11`;
- extended set: `rho = -0.590`, FDR p `1.65e-10`.

However, the raw subject-level maximum of this diary-normalized feature is constant. Its pooled variation arose because adjustment occurred at the nightly level before aggregation. The association cannot therefore be estimated using the raw grouped GEE model and should remain an exploratory transformation-specific observation.

The extended feature set also revealed broad pooled clusters involving activity variability and UPDRS, and activity-distribution features with attention.

## 8) Diagnosis-Adjusted Follow-up Results

### 8.1 Candidate disposition

| Feature set | Scenario | Candidates | Adjusted only | Within-group + adjusted | Diagnosis interaction | Within-group only | Not retained | Not estimable |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Sleep/diary | preDLB vs HC | 22 | 5 | 2 | 4 | 1 | 9 | 1 |
| Sleep/diary | preDLB + MCI-AD vs HC | 25 | 9 | 2 | 1 | 2 | 10 | 1 |
| Extended | preDLB vs HC | 46 | 9 | 5 | 0 | 3 | 28 | 1 |
| Extended | preDLB + MCI-AD vs HC | 36 | 9 | 7 | 2 | 2 | 15 | 1 |

The MCI-AD vs HC scenarios had no pooled FDR-significant candidates to carry into follow-up.

The “not retained” category means that the selected pooled association did not survive FDR in the within-group, diagnosis-adjusted, or interaction follow-up. This is expected in a deliberately stricter second stage.

### 8.2 Strong diagnosis-adjusted associations

Selected robust results are shown below. The correlation column reports the
pooled Spearman coefficient with a person-cluster-robust 95% confidence
interval. `Beta` is the adjusted change in outcome per one-standard-deviation
increase in the raw grouped feature. The discovery and adjusted FDR p-values
belong to their respective analysis stages.

| Feature set and scenario | Outcome | Feature-family representative | Spearman rho (95% robust CI) | Discovery FDR p | Adjusted beta per SD (95% CI) | Adjusted FDR p |
|---|---|---|---:|---:|---:|---:|
| Extended, preDLB vs HC | UPDRS | IQR of activity median absolute deviation | 0.331 (0.171 to 0.490) | 0.01197 | 1.322 (0.807 to 1.837) | 0.000014 |
| Extended, combined | UPDRS | IQR of activity median absolute deviation | 0.319 (0.175 to 0.462) | 0.01152 | 1.208 (0.747 to 1.669) | 0.000004 |
| Extended, preDLB vs HC | Visuospatial | Minimum activity 80th percentile | -0.295 (-0.487 to -0.102) | 0.04432 | 0.163 (0.093 to 0.233) | 0.000042 |
| Extended, preDLB vs HC | Attention | SD of relative interdecile range | -0.327 (-0.506 to -0.149) | 0.02613 | -0.198 (-0.311 to -0.084) | 0.00379 |
| Extended, combined | Attention | IQR of relative interdecile range | -0.254 (-0.408 to -0.101) | 0.04321 | -0.199 (-0.292 to -0.106) | 0.000457 |
| Extended, combined | Attention | SD of relative interdecile range | -0.313 (-0.462 to -0.163) | 0.01737 | -0.189 (-0.290 to -0.089) | 0.00189 |
| Sleep/diary, combined | Executive | MAD of normalized awakenings over 5 minutes | -0.273 (-0.389 to -0.157) | 0.01820 | -0.129 (-0.213 to -0.045) | 0.00933 |
| Extended, combined | Executive | SD of normalized awakenings over 5 minutes | -0.279 (-0.422 to -0.137) | 0.04140 | -0.165 (-0.260 to -0.069) | 0.00435 |

The normalized-WASO/MFS rows from the previous report version have been
removed from the retained-results table. After the WASO seconds-to-minutes
correction and recomputation, `MFS ~ Range.actigraphy_norm.Wake after sleep
onset` is no longer a focused candidate.

The opposite signs for the pooled visuospatial `rho` and adjusted `beta` do
not indicate a calculation error. This is the clearest example of why both
stages must be retained: the pooled coefficient uses a nightly residualized
feature before aggregation, whereas the GEE coefficient is estimated from the
raw grouped feature with diagnosis and covariates entered directly.

Additional adjusted UPDRS results in the extended set included median activity IQR and robust mean-excluding-outlier summaries. Their convergence supports an activity-variability construct rather than one isolated engineered feature.

Attention results similarly converged across relative interdecile range, relative variation range, percentiles, and other activity-distribution summaries. Executive results repeatedly involved the dispersion of awakenings longer than five minutes.

### 8.3 Diagnosis-specific interactions

Four feature-outcome pairs showed an FDR-significant feature-by-diagnosis
interaction, all in the sleep/diary preDLB vs HC scenario. Correlations and
adjusted slopes are presented separately within each diagnosis:

| Outcome and feature | Diagnosis | Spearman rho (95% robust CI) | Correlation FDR p | Adjusted beta per SD (95% CI) | Slope FDR p | Interaction FDR p |
|---|---|---:|---:|---:|---:|---:|
| RBDq: maximum actigraphy wake bouts | HC | -0.025 (-0.262 to 0.213) | 0.83890 | -0.032 (-0.361 to 0.296) | 0.98460 | 0.00227 |
| RBDq: maximum actigraphy wake bouts | preDLB | 0.586 (0.374 to 0.797) | <0.000001 | 1.360 (0.662 to 2.058) | 0.00047 | 0.00227 |
| RBDq: median normalized sleep efficiency | HC | -0.070 (-0.303 to 0.163) | 0.64803 | -0.133 (-0.455 to 0.190) | 0.73676 | 0.01965 |
| RBDq: median normalized sleep efficiency | preDLB | -0.415 (-0.667 to -0.164) | 0.00423 | -1.099 (-1.792 to -0.406) | 0.00330 | 0.01965 |
| RBDq: median actigraphy wake after sleep onset | HC | -0.032 (-0.267 to 0.203) | 0.83890 | -0.004 (-0.405 to 0.397) | 0.98460 | 0.01436 |
| RBDq: median actigraphy wake after sleep onset | preDLB | 0.460 (0.246 to 0.674) | 0.00018 | 1.076 (0.401 to 1.750) | 0.00330 | 0.01436 |
| Executive: CV of normalized awakenings over 5 minutes | HC | -0.196 (-0.404 to 0.013) | 0.13540 | -0.239 (-0.341 to -0.137) | 0.000036 | 0.02733 |
| Executive: CV of normalized awakenings over 5 minutes | preDLB | -0.199 (-0.435 to 0.036) | 0.16538 | -0.025 (-0.129 to 0.079) | 0.63967 | 0.02733 |

The first three indicate RBDq associations that are present primarily in preDLB and essentially absent in HC. The executive association shows the opposite pattern: a negative slope in HC but little evidence of an association in preDLB.

No interaction survived FDR in either combined scenario or in either extended-feature scenario. Most retained extended-feature findings are therefore better interpreted as diagnosis-adjusted associations shared across the modeled groups rather than evidence of different slopes by diagnosis.

### 8.4 Within-diagnosis support

Examples with both within-diagnosis and adjusted support include:

- executive performance versus MAD of normalized awakenings in preDLB;
- attention versus relative interdecile-range variability in preDLB and/or MCI-AD;
- visuospatial performance versus minimum sleep fragmentation in preDLB and MCI-AD.

Some candidates were significant within one diagnosis but not in the adjusted pooled model. These should be considered weaker subgroup hypotheses because the subgroup estimates have smaller sample sizes.

### 8.5 Primary HC-versus-preDLB visualizations

Because the primary thesis comparison is HC versus preDLB, the focused figure
set now uses only these two groups. After WASO correction, seven focused
HC-versus-preDLB figures remain: three activity-enhanced shared adjusted
associations, one sleep/diary executive association, and three RBDq
associations with an FDR-significant feature-by-diagnosis interaction. The
previous normalized-WASO/MFS plot has been retired because that pair is no
longer a focused candidate.

In each figure:

- colored points are raw subject/visit observations;
- thin lines connect repeated visits from the same underlying person;
- thick lines are predictions from the diagnosis-adjusted GEE model;
- shaded regions are model-based 95% confidence bands;
- covariates are held at their reference or mean values for prediction;
- slightly separated points for discrete features are a display aid only; model estimates use the original values.

For associations without an FDR-significant interaction, parallel fitted lines
show the common adjusted feature slope while allowing different diagnostic
intercepts. Different slopes are drawn only when the formal interaction test
survives FDR correction.

#### How to read the engineered feature names

Each feature combines a **nightly measurement** with an **across-night
summary**. Only subjects/visits with at least five recorded nights enter the
grouped matrix.

For example, `Max.actigraphy.Wake bouts` is constructed as follows:

1. The sleep/wake classifier labels consecutive 30-second epochs during each
   sleep period.
2. A wake bout is inferred when the classification changes from sleep to wake.
3. The number of wake bouts is calculated separately for every night.
4. `Max` retains the largest nightly count for that subject/visit.

It therefore represents the subject's **most fragmented recorded night**, not
the average number of awakenings. Other prefixes have similarly specific
meanings:

- `Median`: the typical nightly value;
- `Max` or `Min`: the most extreme recorded nightly value;
- `SD`: conventional night-to-night dispersion;
- `MAD`: median absolute deviation across nights, a robust dispersion measure;
- `IQR`: the difference between the 75th and 25th percentiles across nights;
- `Range`: the largest minus the smallest nightly value.

The clinical outcomes also have different meanings and directions:

- higher UPDRS means greater motor symptom burden;
- higher attention, executive, and visuospatial z-scores are interpreted as
  better performance, according to the source variable labels and conventional
  z-score direction; the exact normative construction should still be checked
  before manuscript submission;
- higher MFS means more features of cognitive fluctuation, such as daytime
  drowsiness, prolonged daytime sleep, staring episodes, or disorganized
  speech;
- higher RBDq means more self-reported symptoms compatible with REM sleep
  behavior disorder, including vivid/action dreams, vocalization, limb or
  complex movements, dream enactment, and disturbed sleep.

**RBDq is not a count of nighttime awakenings.** One questionnaire item asks
whether movements wake the person, but the total score is a broader
13-point RBD screening score. Actigraphy-derived wake bouts and WASO measure
sleep continuity from movement-based sleep/wake classification. An association
between them and RBDq means that the two phenomena co-occur statistically; it
does not show that every inferred awakening is an RBD episode or establish an
RBD diagnosis.

#### Activity variability and UPDRS

![HC versus preDLB activity variability and motor impairment](figures/hc-vs-predlb/01_activity_variability_updrs.png)

**Figure 1.** Activity variability is positively correlated with UPDRS in the HC-versus-preDLB sample (`rho = 0.331`, robust 95% CI `0.171` to `0.490`; discovery FDR p `0.01197`). The association remains after adjustment for diagnosis, gender, and education (`beta = 1.322` UPDRS points per feature SD, 95% CI `0.807` to `1.837`; adjusted FDR p `0.000014`). The interaction is not significant (FDR p `0.637`), supporting a common positive adjusted slope rather than different HC and preDLB slopes.

**How the feature is created.** The activity index is first calculated for
each 60-second window within the detected sleep period. It is the square root
of the mean variance of the band-pass-filtered X, Y, and Z accelerometer
channels. For each night, the median absolute deviation (MAD) describes how
far the minute-level activity values typically lie from that night's median.
The final `IQR` then measures how much those nightly MAD values differ across
the recorded nights.

This is therefore a second-order variability feature: **night-to-night
variability of within-night movement variability**. A high value can arise
when some nights are consistently quiet but other nights contain much more
variable movement. It does not simply mean that the person moves more on
average.

**Clinical interpretation.** Higher values are associated with higher UPDRS,
that is, greater motor symptom burden. A defensible interpretation is that
unstable nocturnal motor activity accompanies greater motor impairment.
However, the activity index does not directly measure bradykinesia, rigidity,
tremor, or a specific parasomnia. The graph supports an association between
two measures, not a mechanism in which nocturnal movement causes motor
impairment.

#### Activity variability and attention

![HC versus preDLB activity variability and attention](figures/hc-vs-predlb/02_activity_variability_attention.png)

**Figure 2.** Greater variability in the relative interdecile-range activity measure correlates with lower attention scores (`rho = -0.327`, robust 95% CI `-0.506` to `-0.149`; discovery FDR p `0.02613`) and remains negatively associated after adjustment (`beta = -0.198`, 95% CI `-0.311` to `-0.084`; adjusted FDR p `0.00379`). The interaction is not significant (FDR p `0.937`).

**How the feature is created.** Within each night, the 10th percentile of the
minute-level activity index is subtracted from the 90th percentile. This
interdecile range captures the spread of the central 80% of activity values
while reducing the influence of the most extreme 10% at either end. The result
is divided by that night's maximum activity, making it relative to the night's
largest movement. `SD` then describes how much this relative spread changes
from night to night.

A high value means that the **shape and spread of nocturnal activity are
inconsistent across nights**. It is not the same as a consistently restless
subject: a person with similarly high activity every night could have a low
across-night SD.

**Clinical interpretation.** Greater night-to-night instability is associated
with a lower attention/processing z-score. This is compatible with the
hypothesis that unstable sleep or nocturnal motor behavior accompanies poorer
daytime attention. It is also compatible with a shared underlying disease
process affecting both sleep and cognition. The cross-sectional graph cannot
distinguish these explanations or show that improving this activity feature
would improve attention.

#### Activity level and visuospatial performance

![HC versus preDLB activity level and visuospatial performance](figures/hc-vs-predlb/03_activity_level_visuospatial.png)

**Figure 3.** The pooled preprocessed feature has a negative correlation with visuospatial performance (`rho = -0.295`, robust 95% CI `-0.487` to `-0.102`; discovery FDR p `0.04432`), whereas the direct raw-feature GEE model has a positive adjusted coefficient (`beta = 0.163`, 95% CI `0.093` to `0.233`; adjusted FDR p `0.000042`). This sign reversal reflects the different preprocessing and model definitions, not two interchangeable estimates. The figure displays the direct adjusted raw-feature relationship. This result should therefore be treated as a sensitivity finding requiring further verification rather than as a straightforward replicated direction of association.

**How the feature is created.** For every night, the 80th percentile is the
activity level below which 80% of the minute-level values fall. It describes
the upper part of usual activity without depending only on the single maximum
movement. `Min` then selects the smallest nightly 80th percentile across the
recorded nights.

This can be read as the upper-activity level on the subject's **quietest
recorded night**. If the value remains high, even the quietest recorded night
contained a relatively elevated upper range of activity.

**Clinical interpretation.** The outcome is a visuospatial z-score, for which
higher values are provisionally interpreted as better performance. Because
the unadjusted/preprocessed correlation and direct adjusted model point in
opposite directions, no stable clinical narrative should be assigned to this
figure. It is a useful sensitivity result, but it has lower interpretive
priority than results whose direction is consistent across analysis stages.

#### Awakening variability and executive performance

![HC versus preDLB nocturnal awakening variability and executive performance](figures/hc-vs-predlb/04_awakening_variability_executive.png)

**Figure 4.** Greater variability in normalized awakenings longer than five minutes correlates with lower executive scores (`rho = -0.324`, robust 95% CI `-0.459` to `-0.188`; discovery FDR p `0.00705`) and remains negatively associated after adjustment (`beta = -0.155`, 95% CI `-0.254` to `-0.056`; adjusted FDR p `0.01677`). The interaction is not significant (FDR p `0.969`).

**How the feature is created.** The classifier searches the 30-second
sleep/wake sequence for sustained wake periods of at least ten consecutive
epochs, corresponding to five minutes, after at least five minutes of
classified sleep. The nightly number of these long awakenings is converted to
an age-dependent category: appropriate (`+1`), uncertain (`0`), or
inappropriate (`-1`). The final across-night MAD measures how much these
categories vary around the subject's median category.

A high value therefore means that the subject alternates between nights with
different clinical categories of long-awakening burden. It means
**inconsistent sleep continuity**, not necessarily a high number of long
awakenings on every night.

**Clinical interpretation.** Greater inconsistency is associated with lower
executive-function scores. A possible clinical hypothesis is that variable
sleep continuity accompanies poorer executive control, but the association
may also reflect disease burden, medication, sleep apnea, nocturia, periodic
limb movements, or classification error. The graph does not identify which
cause generated the long wake periods.

#### Retired normalized-WASO/MFS candidate

**Retired Figure 5.** The previous HC-versus-preDLB figure for
`MFS ~ Range.actigraphy_norm.Wake
after sleep onset` has been removed from the canonical figure set. After
correcting WASO normalization from seconds to minutes, regenerating
`dataset-clinical.xlsx` and `dataset-clinical-acc.xlsx`, and rerunning the
correlation and family-follow-up analyses, this feature-outcome pair is no
longer selected as a focused candidate.

The historical result remains useful only as an audit trail showing why the
WASO unit correction was necessary. It should not be used as evidence that
MFS is associated with normalized WASO variability in the corrected analysis.
MFS is a cognitive-fluctuation scale, not a direct nighttime-waking scale.

#### preDLB-specific RBDq associations

![Wake bouts and RBD symptoms](figures/hc-vs-predlb/06_wake_bouts_rbdq.png)

**Figure 6.** Maximum nightly wake bouts show the strongest diagnosis interaction (FDR p `0.00227`). In preDLB, both the correlation (`rho = 0.586`, robust 95% CI `0.374` to `0.797`) and adjusted slope (`beta = 1.360`, 95% CI `0.662` to `2.058`) are positive. In HC, both estimates are approximately null (`rho = -0.025`, 95% CI `-0.262` to `0.213`; `beta = -0.032`, 95% CI `-0.361` to `0.296`).

**How the feature is created.** A wake bout is an inferred transition from a
classified sleep period into a classified wake period. The count is calculated
for each night, and `Max` selects the largest nightly count. The x-axis
therefore answers: “On this subject's most fragmented recorded night, how many
separate sleep-to-wake transitions were detected?”

**Clinical interpretation.** The positive preDLB slope means that subjects
with a more fragmented worst night tend to have higher RBDq scores. A careful
wording is:

> In preDLB, a greater maximum number of actigraphy-inferred wake bouts was
> associated with a greater burden of self-reported RBD-compatible symptoms.

It is **not** correct to say that RBDq describes the number of times the person
wakes. RBD-related movements or vocalizations could cause movement algorithms
to classify an epoch as wake, so the association is biologically plausible.
However, insomnia, sleep apnea, periodic limb movements, nocturia, medication,
or algorithmic misclassification can also increase wake bouts. Actigraphy does
not identify REM sleep or demonstrate REM sleep without atonia, so it cannot
determine which wake bouts are true dream-enactment events.

![Sleep efficiency and RBD symptoms](figures/hc-vs-predlb/07_sleep_efficiency_rbdq.png)

**Figure 7.** Median normalized sleep efficiency is negatively correlated with RBDq in preDLB (`rho = -0.415`, robust 95% CI `-0.667` to `-0.164`) and remains negative after adjustment (`beta = -1.099`, 95% CI `-1.792` to `-0.406`). Neither estimate supports an association in HC (`rho = -0.070`, 95% CI `-0.303` to `0.163`; `beta = -0.133`, 95% CI `-0.455` to `0.190`; interaction FDR p `0.01965`).

**How the feature is created.** For every night, total sleep time is estimated
as time in bed minus sleep-onset latency, WASO, and wake after the final sleep
epoch. Sleep efficiency is then total sleep time divided by time in bed,
multiplied by 100. Each night is categorized as appropriate (`+1`, generally
at least 85%), uncertain (`0`), or inappropriate (`-1`) using age-dependent
rules. `Median` retains the subject's typical nightly category.

The x-axis is therefore an ordinal summary of typical sleep efficiency, not
the raw percentage. Moving right generally means a more favorable typical
sleep-efficiency category.

**Clinical interpretation.** In preDLB, poorer typical sleep efficiency is
associated with higher RBDq. This supports co-occurrence of disrupted sleep
continuity and RBD-compatible symptoms. It remains nonspecific: reduced
efficiency can arise from many sleep disorders and from actigraphy's tendency
to classify quiet wakefulness as sleep. The model also treats the three
categories numerically, so the distance from `-1` to `0` is assumed to be
comparable with the distance from `0` to `+1`; this should be checked in an
ordinal or raw-percentage sensitivity analysis.

![Wake after sleep onset and RBD symptoms](figures/hc-vs-predlb/08_waso_rbdq.png)

**Figure 8.** Median actigraphy wake after sleep onset is positively correlated with RBDq in preDLB (`rho = 0.460`, robust 95% CI `0.246` to `0.674`) and remains positive after adjustment (`beta = 1.076`, 95% CI `0.401` to `1.750`). Both estimates are approximately null in HC (`rho = -0.032`, 95% CI `-0.267` to `0.203`; `beta = -0.004`, 95% CI `-0.405` to `0.397`; interaction FDR p `0.01436`).

**How the feature is created.** For each night, WASO sums all 30-second epochs
classified as wake between the first and last detected sleep epochs. `Median`
is the typical nightly total across the recorded nights. The raw database and
plot use seconds; dividing the x-axis value by 60 gives the clinically familiar
number of minutes awake after sleep onset. This raw feature is not affected by
the normalized-WASO category correction; it uses the raw WASO duration.

**Clinical interpretation.** In preDLB, subjects who typically spend more time
awake after initially falling asleep tend to have higher RBDq scores. This is
the closest of the three RBDq plots to the proposed “waking often” explanation,
but it measures **total wake duration**, whereas Figure 6 measures the
**number of wake episodes**. Neither is what RBDq directly measures. The result
supports an association between sleep fragmentation and RBD-compatible symptom
burden within preDLB, not proof that RBD symptoms caused the wake time.

The HC-versus-preDLB figures support two different interpretations. UPDRS,
attention, and executive performance show common adjusted associations without
evidence that the slopes differ by diagnosis. The three RBDq findings show
diagnosis-dependent associations concentrated in preDLB. The visuospatial
result is statistically retained but less stable in direction across
preprocessing stages and should receive lower interpretive priority. The
previous normalized-WASO/MFS finding was not retained after correction and is
not part of the canonical focused figure set.

The clinical interpretation above is based on the original
[RBDSQ validation](https://pubmed.ncbi.nlm.nih.gov/17894337/), the
[Mayo Fluctuations Scale development study](https://pubmed.ncbi.nlm.nih.gov/14745051/),
an [actigraphy sleep-parameter review](https://pmc.ncbi.nlm.nih.gov/articles/PMC7191872/),
and the standard interpretation that
[higher UPDRS scores indicate greater severity](https://www.apta.org/patient-care/evidence-based-practice-resources/test-measures/unified-parkinsons-disease-rating-scale-updrs-movement-disorders-society-mds-modified-unified-parkinsons-disease-rating-scale-mds-updrs).

## 9) Strict Nested-CV Classification

### 9.1 Validation design

The strict classification pipeline was rerun after the WASO correction using
`dataset-clinical` with diary/lifestyle covariates and recursive feature
elimination (RFE). This is the current strict classification run:

`media/classification/grouped-statistics-strict-with-covariates/dataset-clinical-rfe/20260629_173137/`

The run used the WASO-corrected grouped source matrix:

`media/analysis-preparation/dataset-clinical/20260629_173132_047541/raw/grouped_clinical_matrix_with_stats.xlsx`

The updated strict RFE analysis used:

- outer leave-one-database-subject-out cross-validation;
- five-fold stratified inner cross-validation within each outer training set;
- `12` randomized hyperparameter configurations per outer fold;
- median imputation, constant-feature removal, scaling, covariate
  residualization, RFE, and XGBoost fitting within training data only;
- RFE with XGBoost inside inner CV, testing `20`, `40`, `80`, and `120`
  selected-feature options;
- diary/lifestyle covariates: sleep/rest quality, day sleep, caffeine,
  alcohol, and sleeping-pill summary variables;
- scenario-specific covariates:
  - gender and education for preDLB vs HC;
  - education for the combined and MCI-AD scenarios;
- a default classification threshold of `0.5`;
- an exploratory threshold selected independently within each outer training
  fold from inner-CV predictions.

The ROC AUC and PR AUC are calculated from the nested out-of-fold
probabilities. The approximate 95% confidence intervals below were obtained
by class-stratified bootstrap resampling of the saved nested out-of-fold
predictions. They quantify uncertainty in the fixed saved predictions but do
not repeat the complete nested model-training procedure, so they may
understate total model-selection uncertainty.

### 9.2 Primary nested RFE results

The default threshold is used for the primary classification metrics.

| Scenario | Visits, positive/HC | ROC AUC (approx. 95% CI) | PR AUC (approx. 95% CI) | Balanced accuracy (approx. 95% CI) | MCC | Sensitivity | Specificity |
|---|---:|---:|---:|---:|---:|---:|---:|
| preDLB vs HC | 132, 59/73 | 0.764 (0.677 to 0.844) | 0.753 (0.657 to 0.850) | 0.668 (0.588 to 0.748) | 0.334 | 0.678 | 0.658 |
| preDLB + MCI-AD vs HC | 162, 89/73 | 0.718 (0.636 to 0.793) | 0.764 (0.690 to 0.839) | 0.634 (0.561 to 0.705) | 0.281 | 0.775 | 0.493 |
| MCI-AD vs HC | 103, 30/73 | 0.669 (0.540 to 0.782) | 0.517 (0.387 to 0.675) | 0.648 (0.551 to 0.745) | 0.323 | 0.433 | 0.863 |

The HC-versus-preDLB RFE model is the strongest current strict classifier. A
randomly selected preDLB observation receives a higher predicted probability
than a randomly selected HC observation approximately `76%` of the time.
Classification at `0.5` remains moderate rather than strong, with `40` of `59`
preDLB visits and `48` of `73` HC visits classified correctly.

The combined preDLB + MCI-AD model has moderate ranking performance but weak
specificity at the default threshold. It correctly identifies `69` of `89`
positive visits, but only `36` of `73` HC visits. This is consistent with a
broad non-HC screening signal, not a disease-specific classifier.

The isolated MCI-AD model is still exploratory. It has high specificity
(`63` of `73` HC visits correctly classified) but low sensitivity (`13` of
`30` MCI-AD visits). The wide AUC confidence interval reflects the small
MCI-AD sample.

### 9.3 Threshold optimization

| Scenario | Mean inner threshold | Median inner threshold | Tuned balanced accuracy | Tuned MCC | Tuned sensitivity | Tuned specificity |
|---|---:|---:|---:|---:|---:|---:|
| preDLB vs HC | 0.730 | 0.767 | 0.638 | 0.286 | 0.508 | 0.767 |
| preDLB + MCI-AD vs HC | 0.644 | 0.618 | 0.702 | 0.403 | 0.719 | 0.685 |
| MCI-AD vs HC | 0.560 | 0.526 | 0.655 | 0.344 | 0.433 | 0.877 |

Inner-CV threshold selection did not consistently improve the RFE models. It
reduced preDLB vs HC balanced accuracy by shifting toward specificity at the
cost of sensitivity. It improved the combined model by balancing sensitivity
and specificity better, and it changed MCI-AD only slightly. Thresholds still
varied substantially between outer folds; for preDLB vs HC, the 5th to 95th
percentile range was approximately `0.47` to `0.92`. The default threshold
therefore remains the primary analysis, while the tuned-threshold combined
result can be reported as a sensitivity analysis.

### 9.4 RFE versus non-RFE strict classification

The table below compares the current strict RFE run with the previous
WASO-corrected strict non-RFE run from `20260629_145843`.

| Scenario | Non-RFE AUC | RFE AUC | Change | Non-RFE BACC | RFE BACC | Change | Non-RFE MCC | RFE MCC | Change |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| preDLB vs HC | 0.718 | 0.764 | +0.046 | 0.646 | 0.668 | +0.022 | 0.290 | 0.334 | +0.044 |
| preDLB + MCI-AD vs HC | 0.700 | 0.718 | +0.018 | 0.664 | 0.634 | -0.029 | 0.347 | 0.281 | -0.066 |
| MCI-AD vs HC | 0.672 | 0.669 | -0.003 | 0.651 | 0.648 | -0.003 | 0.316 | 0.323 | +0.007 |

RFE clearly helps the HC-versus-preDLB model: ranking, balanced accuracy, and
MCC all improve, and the final model is reduced to `80` selected features from
`267` post-filter candidates. The combined model improves in ranking but loses
default-threshold balanced accuracy because specificity falls. MCI-AD changes
very little and remains limited by the small positive sample.

### 9.5 Comparison with previous global-covariate classifier runs

The previous April/May classifier runs used the old
`media/covariates/dataset-clinical` prepared matrix. That preparation selected
the union of recommended covariates (`gender, education`) across planned
scenarios. The corrected June preparation instead selects covariates within
each diagnostic scenario:

- preDLB vs HC: gender and education;
- preDLB + MCI-AD vs HC: education only;
- MCI-AD vs HC: education only.

The corrected classifier pipelines also fit covariate residualization within
training folds. Therefore the comparison below is an empirical run comparison,
not a pure estimate of the isolated causal effect of changing one covariate
column. The non-strict runs also differ in XGBoost configuration after the
Mac/runtime update and predate the WASO correction, so the WASO-corrected
strict RFE comparison is the more defensible current summary.

Default-threshold non-strict comparison, retained as historical context:

| Scenario | Old AUC | Corrected AUC | Change | Old BACC | Corrected BACC | Change | Old MCC | Corrected MCC | Change |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| preDLB vs HC | 0.819 | 0.798 | -0.021 | 0.724 | 0.727 | +0.004 | 0.451 | 0.454 | +0.003 |
| preDLB + MCI-AD vs HC | 0.719 | 0.753 | +0.035 | 0.655 | 0.677 | +0.022 | 0.320 | 0.353 | +0.033 |
| MCI-AD vs HC | 0.453 | 0.667 | +0.214 | 0.495 | 0.606 | +0.111 | -0.009 | 0.202 | +0.211 |

Default-threshold strict nested-CV comparison, using the current WASO-corrected
strict RFE run:

| Scenario | Old strict AUC | Current RFE AUC | Change | Old strict BACC | Current RFE BACC | Change | Old strict MCC | Current RFE MCC | Change |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| preDLB vs HC | 0.710 | 0.764 | +0.054 | 0.640 | 0.668 | +0.028 | 0.283 | 0.334 | +0.051 |
| preDLB + MCI-AD vs HC | 0.687 | 0.718 | +0.031 | 0.599 | 0.634 | +0.035 | 0.214 | 0.281 | +0.067 |
| MCI-AD vs HC | 0.482 | 0.669 | +0.187 | 0.463 | 0.648 | +0.185 | -0.077 | 0.323 | +0.400 |

Relative to the old global-covariate strict run, the current RFE analysis
improves all three default-threshold scenarios. The biggest relative gain is
still MCI-AD vs HC, which moves from near-random to moderate exploratory
performance. However, the RFE analysis should not be interpreted as validated
biomarker discovery: feature selection still occurs inside a small internal
dataset, and the final selected features are not independent stability
estimates.

### 9.6 Feature interpretation

The final models fitted to each complete scenario dataset were generated only
after nested evaluation. Their selected features, feature importances, and
SHAP values are useful for generating hypotheses, but they are not nested
measures of feature-selection stability.

RFE selected `80` final features in each scenario from `267` post-filter
candidate variables. This is more interpretable than the non-RFE model, but
`80` features is still large relative to the sample size.

For preDLB vs HC, final-model feature importance emphasizes awakenings longer
than five minutes, diary-normalized awakening variability, sleep-efficiency
variability, subjective rest quality, alcohol count/timing, diary-normalized
wake after sleep onset, and sleep-onset-latency slopes. SHAP summaries also
highlight rest quality, alcohol timing/count, total-sleep-time variability,
and diary wake-after-sleep-onset slope. The sleep-continuity signal is
plausible, but the lifestyle/diary terms should be treated cautiously.

For preDLB + MCI-AD vs HC, the most influential final-model features include
diary-normalized long-awakening variability, diary-normalized sleep-onset
latency slope, diary wake-after-sleep-onset slope, actigraphy long-awakening
variability, diary sleep-fragmentation slope, alcohol timing, and actigraphy
sleep-onset latency. This is consistent with a broad non-HC signal, but it is
not diagnostically specific because the positive class combines preDLB and
MCI-AD.

For MCI-AD vs HC, final-model features emphasize diary long-awakening range,
diary-normalized sleep-onset-latency variability, normalized actigraphy
sleep-onset-latency variability, sleep fragmentation, caffeine/alcohol timing
variability, total-sleep-time variability, and wake-bout variability. Given the
small MCI-AD sample and low sensitivity, these should be treated as exploratory
candidate families only.

These patterns support the broader sleep-continuity hypothesis, but the
importance of diary, alcohol, and caffeine variables also raises concern that
the model may partly learn recruitment, reporting, or source-protocol
differences. Feature importance does not establish direction, causality, or
clinical specificity.

### 9.7 Non-strict classification sensitivity runs from 2026-06-30

Four additional non-strict classification pipelines were rerun after the WASO
correction. These runs use foldwise covariate residualization and internal CV
for feature/model tuning, but they are not the strict nested leave-one-subject
analyses used above. They are therefore useful as sensitivity and engineering
comparisons, not as the primary estimate of classifier generalization.

The four non-strict runs were:

| Run | Dataset | Feature selection | Run directory |
|---|---|---|---|
| Clinical RFE | `dataset-clinical` | XGBoost RFE | `media/classification/grouped-statistics-with-covariates/dataset-clinical-rfe/all/20260630_085208/` |
| Clinical-acc RFE | `dataset-clinical-acc` | XGBoost RFE | `media/classification/grouped-statistics-with-covariates/dataset-clinical-acc-rfe/all/20260630_091208/` |
| Clinical-acc non-RFE | `dataset-clinical-acc` | ANOVA SelectKBest / all | `media/classification/grouped-statistics-with-covariates/dataset-clinical-acc/all/20260630_095927/` |
| Clinical non-RFE | `dataset-clinical` | ANOVA SelectKBest / all | `media/classification/grouped-statistics-with-covariates/dataset-clinical/all/20260630_103329/` |

Default-threshold performance was:

| Scenario | Run | n, positive/HC | AUC | PR AUC | BACC | MCC | Sensitivity | Specificity |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| preDLB vs HC | clinical RFE | 132, 59/73 | 0.767 | 0.774 | 0.702 | 0.413 | 0.610 | 0.795 |
| preDLB vs HC | clinical-acc RFE | 129, 57/72 | 0.797 | 0.758 | 0.721 | 0.441 | 0.789 | 0.653 |
| preDLB vs HC | clinical-acc non-RFE | 129, 57/72 | 0.785 | 0.763 | 0.711 | 0.419 | 0.754 | 0.667 |
| preDLB vs HC | clinical non-RFE | 132, 59/73 | 0.804 | 0.786 | 0.744 | 0.486 | 0.763 | 0.726 |
| preDLB + MCI-AD vs HC | clinical RFE | 162, 89/73 | 0.697 | 0.727 | 0.650 | 0.300 | 0.697 | 0.603 |
| preDLB + MCI-AD vs HC | clinical-acc RFE | 159, 87/72 | 0.679 | 0.716 | 0.619 | 0.244 | 0.724 | 0.514 |
| preDLB + MCI-AD vs HC | clinical-acc non-RFE | 159, 87/72 | 0.774 | 0.786 | 0.730 | 0.481 | 0.862 | 0.597 |
| preDLB + MCI-AD vs HC | clinical non-RFE | 162, 89/73 | 0.730 | 0.749 | 0.693 | 0.397 | 0.798 | 0.589 |
| MCI-AD vs HC | clinical RFE | 103, 30/73 | 0.689 | 0.526 | 0.674 | 0.338 | 0.567 | 0.781 |
| MCI-AD vs HC | clinical-acc RFE | 102, 30/72 | 0.491 | 0.302 | 0.532 | 0.067 | 0.300 | 0.764 |
| MCI-AD vs HC | clinical-acc non-RFE | 102, 30/72 | 0.581 | 0.365 | 0.554 | 0.106 | 0.400 | 0.708 |
| MCI-AD vs HC | clinical non-RFE | 103, 30/73 | 0.673 | 0.507 | 0.607 | 0.216 | 0.433 | 0.781 |

The best non-strict default-threshold result by scenario was:

| Scenario | Best non-strict run | AUC | BACC | Sensitivity | Specificity | Interpretation |
|---|---|---:|---:|---:|---:|---|
| preDLB vs HC | clinical non-RFE | 0.804 | 0.744 | 0.763 | 0.726 | strongest fast-run performance; still optimistic relative to strict nested CV |
| preDLB + MCI-AD vs HC | clinical-acc non-RFE | 0.774 | 0.730 | 0.862 | 0.597 | activity-enhanced features help the broad non-HC screen |
| MCI-AD vs HC | clinical RFE | 0.689 | 0.674 | 0.567 | 0.781 | best MCI fast-run result, but still modest and sample-limited |

Threshold tuning produced mixed results. The largest tuned balanced accuracy
was `0.753` for the activity-enhanced non-RFE combined scenario, with
sensitivity `0.908` and specificity `0.597`. For HC-versus-preDLB, tuning did
not improve over the best default-threshold clinical non-RFE run. For MCI-AD,
the best tuned result remained clinical RFE (`BACC = 0.685`), again indicating
that MCI-AD discrimination remains much weaker than HC-versus-preDLB.

The feature-selection pattern is also informative:

| Run | preDLB vs HC selected features | preDLB + MCI-AD vs HC selected features | MCI-AD vs HC selected features |
|---|---:|---:|---:|
| Clinical RFE | 40 / 269 | 20 / 269 | 80 / 269 |
| Clinical-acc RFE | 40 / 699 | 80 / 699 | 20 / 699 |
| Clinical-acc non-RFE | 696 / 699 | 40 / 699 | 20 / 699 |
| Clinical non-RFE | 267 / 269 | 20 / 269 | 80 / 269 |

The non-strict results are useful for deciding what to rerun strictly. They
suggest that a strict nested clinical non-RFE model is worth running for
HC-versus-preDLB, and a strict nested activity-enhanced non-RFE model is worth
running for the combined preDLB + MCI-AD vs HC scenario. They do not support
prioritizing activity-enhanced RFE for MCI-AD, where performance was near
random.

### 9.8 Remaining classification limitations

1. **The outer folds are visit-level, not person-level.** The preDLB vs HC
   scenario contains `132` visits from `118` underlying people; `28` visits
   belong to `14` people represented more than once. The combined scenario
   contains `162` visits from `145` people, and MCI-AD vs HC contains `103`
   visits from `98` people. A second visit can therefore occur in training
   while another visit from the same person is held out.
2. **Diagnosis is associated with source cohort.** Approximately `83%` of
   preDLB observations come from the pre-LBD source cohort, while `63%` of HC
   observations come from COBEN. Device handling, recruitment, diary
   completion, or protocol differences can therefore imitate diagnostic
   signal even though the subject prefix is not supplied directly to the
   classifier.
3. **The feature-to-sample ratio remains high.** The final preDLB RFE model selected `80` features from
   `267` post-filter candidate variables. RFE reduces but does not remove
   instability in a dataset of `132` visits.
4. **There is no external test cohort.** Nested cross-validation improves the
   internal performance estimate, but it cannot demonstrate transportability
   to a newly recruited cohort.
5. **Final-model SHAP values are not fold-stability estimates.** Repeated
   grouped validation or bootstrap selection frequencies are needed before
   presenting individual features as robust classifier biomarkers.

### 9.9 Strict RFE HC-vs-preDLB error profile

The strict RFE HC-versus-preDLB run was profiled at subject level to understand
why strict validation gives only moderate classification performance. The full
error-profile note and subject-level CSV are saved here:

- `doc/strict_rfe_hc_vs_predlb_error_profile_2026-06-30.md`
- `doc/strict_rfe_hc_vs_predlb_error_profile_subjects_2026-06-30.csv`

The default-threshold confusion matrix is:

| Group | Meaning | n | Median predicted preDLB probability | Near-threshold n | Confident n |
|---|---|---:|---:|---:|---:|
| TP | preDLB correctly classified | 40 | 0.933 | 4 | 30 |
| FN | preDLB classified as HC | 19 | 0.318 | 5 | 8 |
| TN | HC correctly classified | 48 | 0.098 | 4 | 35 |
| FP | HC classified as preDLB | 25 | 0.718 | 4 | 8 |

The error-group clinical profile is:

| Group | n | Median age | F/M | Median education | Median RBDq | Median UPDRS | Median MFS | Median visuospatial | Median attention | Median executive |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| TP | 40 | 70.0 | 36/4 | 12.25 | 4.0 | 5.0 | 1.0 | -1.341 | -0.167 | -0.460 |
| FN | 19 | 71.0 | 7/12 | 13.00 | 4.0 | 6.0 | 0.0 | 0.150 | -0.333 | -0.500 |
| TN | 48 | 69.5 | 22/26 | 17.00 | 2.0 | 1.0 | 0.0 | 0.499 | 0.333 | 0.150 |
| FP | 25 | 71.0 | 14/11 | 13.00 | 2.0 | 1.0 | 0.0 | 0.458 | 0.333 | 0.007 |

Source-prefix and visit-pattern summaries are:

| Source prefix | n | TP | FN | TN | FP | Error rate |
|---|---:|---:|---:|---:|---:|---:|
| COBEN | 55 | 5 | 4 | 30 | 16 | 0.364 |
| HC | 16 | 0 | 0 | 11 | 5 | 0.312 |
| HC2 | 1 | 0 | 1 | 0 | 0 | 1.000 |
| pre-LBD | 45 | 25 | 12 | 5 | 3 | 0.333 |
| pre-LBD2 | 15 | 10 | 2 | 2 | 1 | 0.200 |

| Visit type | n | TP | FN | TN | FP | Sensitivity | Specificity | BACC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| first visit | 61 | 25 | 12 | 16 | 8 | 0.676 | 0.667 | 0.671 |
| second visit | 16 | 10 | 3 | 2 | 1 | 0.769 | 0.667 | 0.718 |
| single visit | 55 | 5 | 4 | 30 | 16 | 0.556 | 0.652 | 0.604 |

For repeated people, `8` people were correct at both/all visits, `5` had mixed
visit-level correctness, and `1` person was wrong at both visits
(`pre-LBD-5`, classified as HC at both visits despite true preDLB label). This
supports the existing concern that database-subject-level leave-one-out is not
as conservative as person-grouped validation.

This profile shows that the strict performance drop is not explained only by
borderline probabilities. Several errors are confident. False-positive HC
subjects do not look globally more clinically affected than true negatives by
median RBDq/UPDRS/MFS, so they should not automatically be interpreted as
prodromal-like HC cases. False-negative preDLB subjects are also not simply
clinically mild by RBDq or UPDRS: their median RBDq equals true positives and
median UPDRS is slightly higher. The error pattern therefore points toward
feature instability, source/visit effects, and limited sample size rather than
a clean clinical severity gradient.

### 9.10 Strict RFE HC-vs-preDLB probability diagnostics

Additional diagnostic plots were generated from the saved nested out-of-fold
probabilities for the strict RFE HC-versus-preDLB run. These plots are useful
because ROC AUC, PR AUC, and probability distributions describe the ranking
signal even when a single hard threshold produces imperfect classification.

![Strict RFE HC vs preDLB probability diagnostics](/Volumes/Portable/geneactiv-processing-data/media/classification/grouped-statistics-strict-with-covariates/dataset-clinical-rfe/20260629_173137/scenario-preDLB_vs_HC/diagnostic_plots/classification_diagnostic_panel.png)

The plot files are saved here:

`media/classification/grouped-statistics-strict-with-covariates/dataset-clinical-rfe/20260629_173137/scenario-preDLB_vs_HC/diagnostic_plots/`

This directory contains PNG and PDF versions of:

- predicted probability distribution by diagnosis;
- ROC curve;
- precision-recall curve;
- calibration curve;
- confusion matrix;
- a combined diagnostic panel.

The diagnostic summary is:

| Metric | Value |
|---|---:|
| Visits | 132 |
| HC / preDLB | 73 / 59 |
| ROC AUC | 0.764 |
| Average precision / PR AUC | 0.753 |
| Positive-class prevalence | 0.447 |
| Balanced accuracy at 0.5 threshold | 0.668 |
| Sensitivity at 0.5 threshold | 0.678 |
| Specificity at 0.5 threshold | 0.658 |
| MCC at 0.5 threshold | 0.334 |
| Brier score | 0.218 |
| HC median predicted preDLB probability | 0.289 |
| preDLB median predicted preDLB probability | 0.844 |
| Visits with probability 0.4 to 0.6 | 17 |
| Confident wrong classifications | 16 |

Interpretation:

- The probability distribution supports a real but moderate separation signal.
  HC visits are shifted toward low predicted preDLB probabilities, while
  preDLB visits are shifted toward high probabilities.
- The distributions still overlap substantially. This explains why the model
  can have a useful ROC AUC while still making many hard-threshold errors.
- The ROC curve shows better-than-chance ranking. The AUC of `0.764` means
  that a randomly selected preDLB visit receives a higher predicted preDLB
  probability than a randomly selected HC visit about `76%` of the time.
- The precision-recall curve is above the positive-class prevalence baseline
  (`0.753` average precision vs `0.447` prevalence), supporting enrichment of
  preDLB among high-scoring observations.
- The calibration curve is imperfect. The XGBoost probabilities should
  therefore be interpreted primarily as ranking or screening scores, not as
  calibrated absolute clinical risks.
- The confusion matrix at the default threshold remains `48` true negatives,
  `25` false positives, `19` false negatives, and `40` true positives. This
  supports the conservative conclusion: the model detects signal, but it is
  not yet reliable enough for diagnostic classification.

Overall, these diagnostics make the classification result easier to present:
the strict RFE model is not a strong binary classifier yet, but it does show a
moderate HC-versus-preDLB probability-separation signal that is consistent
with the association analyses.

### 9.11 Official HC-vs-preDLB feature-family stability

An official feature-family stability analysis was run for the primary
HC-versus-preDLB scenario. The purpose is to avoid over-interpreting individual
feature variants such as `MAD.diary_norm.Awakening > 5 minutes` and instead
ask whether broader, fixed physiological families recur across analysis
streams.

The analysis uses the shared fixed feature-family mapper in
`dashboard/logic/feature_families.py` and aggregates evidence from:

- strict RFE classification, using selected features, top-30 final-model
  importances, and top-30 SHAP summaries;
- non-strict clinical classification, using selected features, top-30
  importances, and top-30 SHAP summaries;
- non-strict activity-enhanced classification, using selected features,
  top-30 importances, and top-30 SHAP summaries;
- canonical WASO-corrected sleep/diary group differences, clinical
  correlations, and GEE follow-up;
- canonical WASO-corrected sleep/diary + activity group differences, clinical
  correlations, and GEE follow-up.

The run output is:

`media/feature-family-stability/hc-vs-predlb/20260630_135657/`

The main workbook is:

`media/feature-family-stability/hc-vs-predlb/20260630_135657/feature_family_stability.xlsx`

![HC vs preDLB feature-family stability heatmap](/Volumes/Portable/geneactiv-processing-data/media/feature-family-stability/hc-vs-predlb/20260630_135657/feature_family_stability_heatmap.png)

The main interpretation table is:

| Feature family | Stability class | Evidence sources | Classification run support | Association method support | Interpretation |
|---|---|---:|---:|---:|---|
| Long awakenings | primary cross-method stable | 18 | 3 | 5 | strongest sleep-continuity candidate |
| Sleep onset latency | primary cross-method stable | 15 | 3 | 3 | stable sleep-timing candidate |
| Sleep efficiency | primary cross-method stable | 14 | 3 | 4 | supportive sleep-continuity candidate |
| Wake-bout frequency | primary cross-method stable | 13 | 3 | 5 | supportive awakening-frequency candidate |
| Wake after sleep onset | WASO-corrected cross-method | 15 | 3 | 5 | retained after correction, but old normalized-WASO/MFS candidate remains downgraded |
| Activity variability/dispersion | activity-enhanced cross-method stable | 7 | 1 | 4 | supported only when activity features are included |
| Subjective sleep/rest quality | secondary lifestyle signal | 8 | 3 | 0 | useful sensitivity predictor, not primary disease physiology |
| Alcohol exposure/timing | secondary lifestyle signal | 7 | 3 | 0 | possible behavior/cohort/reporting signal |
| Caffeine exposure/timing | secondary lifestyle signal | 7 | 3 | 0 | possible behavior/cohort/reporting signal |

Interpretation:

- The most defensible family-level story is sleep continuity, especially long
  awakenings, sleep onset latency, sleep efficiency, and wake-bout frequency.
- WASO remains present across methods after the seconds-versus-minutes
  correction. However, the previous focused MFS association with normalized
  WASO should remain retired because it did not survive the corrected
  analysis as a focused candidate.
- Activity variability is a credible activity-enhanced candidate, but it
  should be reported separately from sleep/diary-only models because it cannot
  appear in the clinical-only classifier.
- Diary/lifestyle predictors improve or recur in classification outputs, but
  they do not have matching association/GEE support in this stability analysis.
  They should therefore remain secondary sensitivity predictors and should not
  be presented as primary disease-physiology biomarkers.
- This is cross-analysis family consensus, not true nested fold-level feature
  stability. Strict RFE evidence still comes from final full-data model
  selection/importances after nested evaluation. A stronger future analysis
  would save selected families inside each outer fold and report selection
  frequency.

### 9.12 Stable-family restricted HC-vs-preDLB confirmation

After defining the fixed feature families, a restricted confirmation analysis
was run for HC versus preDLB only. The purpose was to test whether the
previous conclusions survive when the feature space is reduced to
pre-specified interpretable families before the association workflow.

The run output is:

`media/feature-family-restricted-analysis/hc-vs-predlb/20260701_094747/`

The summary workbook is:

`media/feature-family-restricted-analysis/hc-vs-predlb/20260701_094747/feature_family_restricted_analysis_summary.xlsx`

Design:

- source data: the two canonical WASO-corrected analysis-preparation runs from
  2026-06-29;
- scenario: HC versus preDLB only;
- controlled covariates: gender and education;
- primary sleep families: long awakenings, sleep-onset latency, sleep
  efficiency, wake-bout frequency, and corrected WASO;
- activity-enhanced extension: the same primary sleep families plus activity
  variability/dispersion;
- statistics: Mann-Whitney/FDR group comparison, Spearman/FDR clinical
  correlations, and diagnosis/covariate-adjusted GEE follow-up.

Summary:

| Restricted analysis | Features tested | Group FDR < 0.05 | Correlation FDR < 0.05 | Follow-up candidates | Adjusted GEE models |
|---|---:|---:|---:|---:|---:|
| Primary sleep families | 180 | 19 | 22 | 15 | 15 |
| Primary sleep + activity variability | 340 | 25 | 50 | 27 | 27 |

Main retained findings:

| Feature family | Clinical outcome | Restricted stream | Interpretation |
|---|---|---|---|
| Long awakenings | Executive | primary sleep | Several long-awakening variability summaries remain associated with executive score after diagnosis, gender, and education adjustment. Adjusted FDR values are approximately `0.019` to `0.027`; one representative is also supported within preDLB. |
| Long awakenings | Executive | primary sleep + activity | The same executive/awakening pattern remains present after adding activity features, supporting it as a sleep-continuity signal rather than only an activity-feature artifact. |
| Activity variability/dispersion | UPDRS | primary sleep + activity | Activity variability is the clearest activity-enhanced signal. Representative activity-dispersion features show pooled Spearman `rho` about `0.315` to `0.344` and adjusted FDR values from `<0.001` to `0.030`. Higher activity variability is associated with higher motor score. |
| Activity variability/dispersion | Attention | primary sleep + activity | Several activity-variability measures remain associated with attention after adjustment. The sign depends on the exact variability metric, so the family-level interpretation should be "activity-distribution variability relates to attention" rather than a single directional statement for every metric. |
| Activity variability/dispersion | Visuospatial | primary sleep + activity | Some visuospatial associations are supported within HC or preDLB separately, but the common adjusted effect is not retained. This should remain secondary. |
| Sleep onset latency | Visuospatial | both streams | The strong pooled sleep-onset-latency/visuospatial association weakens after diagnosis/covariate adjustment or is not estimable from the raw grouped feature. It should not be used as a primary retained result. |

Interpretation:

- The stable-family restriction makes the primary story narrower and more
  defensible. The strongest retained sleep-only finding is long-awakening
  variability versus executive performance.
- Activity variability adds a distinct and stronger signal for UPDRS and
  attention. This supports reporting an activity-enhanced extension rather
  than mixing activity features into the primary sleep-continuity claim.
- Corrected WASO remains part of the stable family set, but it does not restore
  the old normalized-WASO/MFS candidate. The MFS/WASO finding remains retired.
- Sleep-onset latency remains useful as a stable family in the broader
  evidence table, but the restricted GEE follow-up does not support it as the
  strongest confirmed HC-versus-preDLB clinical association.

### 9.13 Strict stable-family HC-vs-preDLB classification

The strict nested-CV stable-family classifiers were then run using the same
fixed family definitions. These models are a classification sensitivity
analysis: they ask whether the smaller, interpretable family set preserves the
HC-versus-preDLB discrimination seen in the broader strict RFE classifier.

Only the corrected runs below should be interpreted:

- primary sleep families:
  `media/classification/grouped-statistics-strict-with-covariates/dataset-clinical-stable-primary-sleep-hc-predlb/20260701_120433/`
- primary sleep + activity variability:
  `media/classification/grouped-statistics-strict-with-covariates/dataset-clinical-acc-stable-primary-sleep-activity-hc-predlb/20260701_125206/`

An earlier same-day exploratory rerun was discarded because excluded feature
columns were still present as dataframe metadata and could re-enter the model.
The code was corrected to drop all non-allowed feature columns before coverage
filtering and nested-CV feature selection. The corrected output feature maps
contain only the intended families:

- primary sleep run: `178` retained mapped features after coverage filtering;
- activity-enhanced run: `338` retained mapped features after coverage
  filtering;
- no retained feature outside the intended family set was detected.

Corrected strict nested-CV results:

| Model | n | Features/families | ROC AUC | PR AUC | Default BACC | Sensitivity | Specificity | Default confusion matrix |
|---|---:|---|---:|---:|---:|---:|---:|---|
| Stable primary sleep | 132 | long awakenings, sleep-onset latency, sleep efficiency, wake-bout frequency, WASO | 0.691 | 0.661 | 0.664 | 0.712 | 0.616 | TN 45, FP 28, FN 17, TP 42 |
| Stable primary sleep + activity variability | 129 | primary sleep families plus activity variability | 0.690 | 0.600 | 0.650 | 0.702 | 0.597 | TN 43, FP 29, FN 17, TP 40 |

Tuned-threshold results:

| Model | Tuned BACC | Tuned sensitivity | Tuned specificity | Tuned confusion matrix |
|---|---:|---:|---:|---|
| Stable primary sleep | 0.656 | 0.627 | 0.685 | TN 50, FP 23, FN 22, TP 37 |
| Stable primary sleep + activity variability | 0.630 | 0.649 | 0.611 | TN 44, FP 28, FN 20, TP 37 |

Interpretation:

- The stable-family classifiers preserve a moderate HC-versus-preDLB signal,
  but they do not outperform the broader strict RFE run (`ROC AUC = 0.764`,
  `BACC = 0.668`).
- The primary sleep model is the cleaner classifier sensitivity result. It
  uses only interpretable sleep-continuity families and reaches almost the
  same balanced accuracy as the broader strict RFE run, but with lower ROC AUC.
- Adding activity variability does not improve strict classification in this
  restricted setting. This differs from the association analyses, where
  activity variability remains important for UPDRS and attention. The practical
  interpretation is that activity variability is useful for clinical-outcome
  association, but not necessarily for a smaller strict diagnostic classifier.
- Inner-CV threshold tuning again does not improve the result. The default
  `0.5` threshold remains the primary classification summary.

### 9.14 HC-vs-preDLB classification validity checks

A dedicated validity audit was added for the current HC-versus-preDLB strict
classification results. The purpose was to test three practical risks before
using the classifier in article-level interpretation:

- whether second visits (`HC2-*`, `pre-LBD2-*`) can inflate performance when
  they are not grouped with the same underlying person;
- whether performance remains similar when only first visits are retained;
- how much diagnostic-label information is built into the consortium-defined
  recruitment source. This is an ascertainment benchmark, not a negative
  control for acquisition or protocol artefacts.

The run output is:

`media/classification/validity-checks/hc-vs-predlb/20260701_180935/`

The workbook is:

`media/classification/validity-checks/hc-vs-predlb/20260701_180935/hc_vs_predlb_classification_validity_checks.xlsx`

Summary:

| Run | n / persons | Repeated persons | All-visits AUC | All-visits BACC | First-visit AUC | First-visit BACC | Source-only AUC | Source-only BACC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Broad strict RFE | 132 / 118 | 14 | 0.764 | 0.668 | 0.736 | 0.655 | 0.739 | 0.840 |
| Stable primary sleep | 132 / 118 | 14 | 0.691 | 0.664 | 0.647 | 0.637 | 0.739 | 0.840 |
| Stable primary sleep + activity variability | 129 / 116 | 13 | 0.690 | 0.650 | 0.649 | 0.619 | 0.745 | 0.845 |

The cohort composition was strongly imbalanced:

| Source cohort | Broad/stable sleep HC | Broad/stable sleep preDLB | Activity HC | Activity preDLB |
|---|---:|---:|---:|---:|
| COBEN | 46 | 9 | 45 | 8 |
| HC/HC2 | 16 | 1 | 16 | 1 |
| pre-LBD/pre-LBD2 | 11 | 49 | 11 | 48 |

Interpretation:

- Repeated visits are a real but not catastrophic issue in the existing
  strict prediction files. There are `14` repeated people (`28` subject rows)
  in the broad RFE and stable-sleep runs, and `13` repeated people (`26`
  subject rows) in the activity-enhanced stable-family run. Removing second
  visits lowers AUC and balanced accuracy in all three models, so repeated
  visits probably inflate performance slightly.
- Diagnosis and source are strongly associated (`Cramer's V` approximately
  `0.68-0.69`, `p < 5e-14`). This is expected from the consortium design:
  HC/HC2 was recruited with an expectation of health, pre-LBD/pre-LBD2
  included people with an expected clinical problem, and COBEN did not use the
  same binary recruitment division.
- The source-only ascertainment benchmark gives BACC approximately `0.84`.
  Because source was expected to contain label information, this is not a valid
  negative control and does not demonstrate technical acquisition or protocol
  confounding. Source identity was not supplied to the wearable classifier, so
  the result also does not prove that the classifier learned source artefacts.
  It does show that pooled diagnostic performance cannot be separated cleanly
  from the source-associated recruitment design using this dataset alone.
- The practical conclusion is that the classifier should remain a secondary
  exploratory analysis. It can support the statement that there is signal in
  the data, but it should not be presented as a reliable diagnostic model
  until the result survives person-grouped retraining and validation with
  adequate representation of both diagnoses within each source or in an
  independent cohort.
- For the article, the more defensible primary statistical story remains the
  association/regression framework: clinically interpretable sleep-continuity
  and activity-variability families related to RBDq, UPDRS, attention,
  visuospatial, and executive outcomes, with diagnosis/covariate-adjusted
  estimates.

### 9.15 Thesis-level person-grouped HC-vs-preDLB classification

The full person-grouped thesis validation was then run on 2026-07-02. This
analysis retrains the HC-versus-preDLB classifiers with true person-grouped
outer and inner cross-validation, so repeated visits such as `HC2-*` and
`pre-LBD2-*` cannot be split between training and test folds.

The run output is:

`media/classification/person-grouped-thesis/hc-vs-predlb/20260702_111118/`

The summary workbook is:

`media/classification/person-grouped-thesis/hc-vs-predlb/20260702_111118/hc_vs_predlb_person_grouped_classification_summary.xlsx`

Design:

- scenario: HC versus preDLB only;
- outer validation: `5`-fold `StratifiedGroupKFold` by underlying person;
- inner tuning: `5`-fold `StratifiedGroupKFold` by underlying person;
- covariates residualized inside training folds: gender and education;
- model variants: broad RFE, stable primary sleep families, and stable primary
  sleep plus activity variability;
- sensitivity outputs: first-visit-only metrics, cohort-stratified
  performance, source-only ascertainment benchmark, leave-one-cohort-out validation,
  and within-cohort nested CV where sample size allowed.

Main person-grouped nested-CV results:

| Model | n / persons | AUC | PR AUC | Default BACC | Tuned BACC | First-visit AUC | First-visit BACC | Source-only AUC | Source-only BACC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Broad strict RFE | 132 / 118 | 0.746 | 0.744 | 0.649 | 0.666 | 0.740 | 0.648 | 0.814 | 0.840 |
| Stable primary sleep | 132 / 118 | 0.706 | 0.634 | 0.663 | 0.642 | 0.677 | 0.644 | 0.814 | 0.840 |
| Stable primary sleep + activity variability | 129 / 116 | 0.620 | 0.575 | 0.604 | 0.620 | 0.589 | 0.586 | 0.814 | 0.845 |

Interpretation of the main table:

- Person grouping does not fully remove the broad RFE signal. The broad model
  still ranks HC and preDLB above chance (`AUC = 0.746`) and reaches tuned
  balanced accuracy `0.666`.
- The result is slightly weaker than the earlier visit-level strict RFE
  classifier (`AUC = 0.764`, default `BACC = 0.668`), which supports the
  previous conclusion that repeated visits inflated performance modestly, not
  catastrophically.
- The stable primary sleep family model remains close to the broad model in
  balanced accuracy (`0.663`) despite lower AUC (`0.706`). This is useful for
  thesis interpretation because it shows that interpretable sleep-continuity
  families preserve some diagnostic signal.
- Adding activity variability weakens strict diagnostic classification in this
  person-grouped validation. This does not contradict the association results
  where activity variability relates to UPDRS and attention. It means activity
  variability is more convincing as a clinical-outcome association family than
  as a diagnostic classifier feature family.
- The source-only ascertainment benchmark gives BACC approximately `0.84`.
  This is expected from diagnosis-enriched recruitment and must not be read as
  proof that the wearable classifier is technically source-confounded. The
  limitation is that pooled performance cannot establish discrimination that
  is independent of the recruitment design.

Cohort composition remained highly imbalanced:

| Source cohort | HC | preDLB | Interpretation |
|---|---:|---:|---|
| COBEN | 46 | 9 | mostly HC |
| HC/HC2 | 16 | 1 | almost entirely HC; within-cohort classifier not estimable |
| pre-LBD/pre-LBD2 | 11 | 49 | mostly preDLB |

Within-cohort nested-CV results:

| Model | Cohort | Status | AUC | BACC | Sensitivity | Specificity | Interpretation |
|---|---|---|---:|---:|---:|---:|---|
| Broad strict RFE | COBEN | completed | 0.579 | 0.557 | 0.222 | 0.891 | weak estimate; only 9 preDLB visits |
| Broad strict RFE | pre-LBD/pre-LBD2 | completed | 0.501 | 0.469 | 0.939 | 0.000 | weak estimate; only 11 HC visits |
| Stable primary sleep | COBEN | completed | 0.522 | 0.524 | 0.222 | 0.826 | weak, imbalanced estimate |
| Stable primary sleep | pre-LBD/pre-LBD2 | completed | 0.510 | 0.490 | 0.980 | 0.000 | weak, imbalanced estimate; poor HC recognition |
| Stable sleep + activity | COBEN | completed | 0.471 | 0.463 | 0.125 | 0.800 | imbalanced estimate below chance |
| Stable sleep + activity | pre-LBD/pre-LBD2 | completed | 0.383 | 0.469 | 0.938 | 0.000 | imbalanced estimate; poor HC recognition |
| All models | HC/HC2 | skipped | - | - | - | - | only one preDLB subject, so grouped CV is not estimable |

Leave-one-cohort-out validation was also informative:

- When COBEN was held out, balanced accuracy was only moderate (`0.595` to
  `0.626`), with low precision for preDLB because COBEN contains few preDLB
  subjects.
- When pre-LBD/pre-LBD2 was held out, AUC looked moderate to good (`0.638` to
  `0.718`), but default-threshold sensitivity was very low (`0.143` to
  `0.208`). The models mostly failed to label the held-out pre-LBD/pre-LBD2
  preDLB subjects as preDLB.
- HC/HC2 leave-one-cohort-out results are not clinically stable because this
  cohort contains only one preDLB subject.

Overall interpretation:

- This is a useful thesis-level validation result, even though it is not a
  strong classifier result. It demonstrates that careful person grouping and
  cohort sensitivity analysis are necessary and materially change the
  interpretation.
- The broad RFE and stable sleep-family models show a moderate pooled internal
  signal. The recruitment enrichment and weak, imbalanced within-source
  estimates mean that source-independent transportability has not been
  established; they do not prove that acquisition artefacts caused the signal.
- The thesis can present this as a methodological finding: diagnostic
  classification from the current pooled dataset is vulnerable to cohort
  composition, while feature-outcome association analyses remain the more
  defensible primary scientific output.
- For an article, the diagnostic classifier should be omitted or kept as a
  short exploratory supplement unless an external or better balanced cohort is
  available.


## 10) Clinical-Scale Regression

### 10.1 Regression design

The all-diagnosis clinical-scale regression workflow has now been rerun after
regenerating the source Excel exports with corrected WASO normalization. This
analysis estimates the six clinical scales directly from the grouped
sleep/diary feature matrix plus lifestyle diary predictors, rather than
classifying diagnosis. The current all-diagnosis clinical run is:

`media/regression/clinical-scales/dataset-clinical/20260629_143305/`

The run uses the WASO-corrected grouped source matrix:

`media/analysis-preparation/dataset-clinical/20260629_143300_182369/raw/grouped_clinical_matrix_with_stats.xlsx`

The source overview contains 73 HC, 59 preDLB, 30 MCI-AD, 79 NonHC, and four
records with missing diagnosis code. Outcome-specific sample sizes differ
slightly because some subjects have missing clinical-scale values.

The model design was:

- outcomes: RBDq, UPDRS, MFS, visuospatial, attention, and executive scores;
- predictors: `269` grouped sleep/diary features and lifestyle diary
  predictors after coverage and non-zero filtering;
- lifestyle predictors: sleep/rest quality, day sleep, caffeine, alcohol, and
  sleeping-pill summary variables;
- algorithm: XGBoost regression with squared-error objective;
- validation: five-fold outer `GroupKFold` by underlying person/visit group;
- tuning: five-fold inner `GroupKFold`, `20` randomized hyperparameter
  configurations;
- covariate handling: age, gender, and education residualized inside each
  training fold;
- baseline: the median clinical score in each training fold;
- main metrics: mean absolute error (MAE), root mean squared error (RMSE), and
  `estimation_error_rate = MAE / observed score range`.

This is stricter than the earlier visit-level classifier with respect to
repeated visits: second visits such as `HC2-*` and `pre-LBD2-*` were grouped
with the corresponding first-visit person when assigning folds.

The `estimation_error_rate` uses the observed range in this dataset, not the
theoretical instrument range. It is therefore useful for comparing outcomes
inside this run, but it should not be presented as a universal clinical
percentage error without this definition.

### 10.2 Regression performance

| Outcome | Visits / person groups | Observed range | MAE | RMSE | MAE/range | R2 | Pearson r | Baseline MAE | MAE change vs baseline |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| RBDq | 240 / 208 | 0.00 to 10.00 | 1.85 | 2.28 | 18.5% | -0.004 | 0.138 | 1.82 | -1.6% |
| UPDRS | 238 / 206 | 0.00 to 25.00 | 2.73 | 3.90 | 10.9% | 0.034 | 0.191 | 2.66 | -2.6% |
| MFS | 238 / 207 | 0.00 to 4.00 | 0.70 | 0.89 | 17.6% | 0.017 | 0.227 | 0.60 | -17.7% |
| Visuospatial | 239 / 207 | -2.41 to 1.21 | 0.68 | 0.84 | 18.9% | 0.147 | 0.392 | 0.72 | +4.6% |
| Attention | 240 / 208 | -2.67 to 1.67 | 0.52 | 0.67 | 12.0% | 0.009 | 0.185 | 0.52 | -0.3% |
| Executive | 240 / 208 | -1.94 to 2.00 | 0.54 | 0.70 | 13.6% | 0.061 | 0.255 | 0.56 | +4.7% |

The main result is unchanged after the corrected all-diagnosis rerun:
prediction of individual clinical scale values remains weak to modest. Positive
Pearson correlations show that the out-of-fold predictions are not random for
several outcomes, but the models often do not beat a simple foldwise median
baseline in absolute error.

Only visuospatial and executive performance beat the median baseline in this
all-diagnosis run, and both improvements are small (`+4.6%` and `+4.7%`).
Visuospatial performance remains the strongest all-diagnosis regression target,
with the highest `R2` (`0.147`) and prediction-observation correlation
(`r = 0.392`). Executive performance is second, but its explained variance is
low (`R2 = 0.061`).

RBDq, UPDRS, MFS, and attention should not currently be described as
quantitatively predictable from this all-diagnosis feature set because their
MAE is worse than the foldwise median baseline. This does not contradict the
correlation and GEE findings. It means that a feature can show a statistically
detectable association while still being insufficient for accurate
individual-level prediction.

Compared with the focused HC-versus-preDLB reruns in Section 10.5, the
all-diagnosis model is less useful for RBDq and attention. This is expected:
pooled all-diagnosis regression mixes HC, preDLB, MCI-AD, and NonHC subjects,
so diagnosis-specific relationships can be diluted. UPDRS and MFS remain
negative against the median baseline in both pooled and focused regression,
so they should remain association/effect-estimation outcomes rather than
individual score-prediction targets for now.

### 10.3 Outcome interpretation

For RBDq, the corrected all-diagnosis model selected diary sleep-onset latency,
subjective sleep/rest quality, awakenings longer than five minutes, sleep
efficiency, raw actigraphy WASO, and wake bouts among its most important
features. These families are clinically consistent with disrupted sleep
continuity and the earlier RBDq interaction results. However, the pooled
regression model does not beat the baseline MAE, suggesting that RBDq may
require diagnosis-stratified or interaction-aware models rather than a single
pooled predictor across all diagnoses.

For UPDRS, the strongest final-model features include sleep-onset latency,
wake bouts, sleep efficiency, awakenings longer than five minutes, diary WASO,
and wake after sleep offset. The model has a positive out-of-fold prediction
correlation (`r = 0.191`) but worse MAE than baseline. This is important: the
previously identified activity-variability association is an interpretable
group-level signal, but this sleep/diary/lifestyle regression run does not yet
provide useful individual UPDRS estimation.

For MFS, the scale is sparse and strongly affected by many zero values. The
foldwise median baseline is therefore difficult to beat. The corrected model
produced a positive prediction correlation (`r = 0.227`) but a substantially
worse MAE than baseline (`-17.7%`). Top features emphasize wake after sleep
offset, sleep-onset latency, awakenings longer than five minutes, time in bed,
and raw WASO. The previous normalized-WASO/MFS candidate remains retired and
should not be reintroduced based on this regression run.

For visuospatial performance, top final-model features emphasize diary
sleep-onset latency, sleep-efficiency variability, diary WASO variability, and
awakenings longer than five minutes. This is the strongest all-diagnosis
regression result after correction, with small but real improvement over the
median baseline. The result supports that sleep-continuity features carry some
predictive structure for visuospatial performance, not that the model is ready
for individual clinical estimation.

For attention, top final-model features include variability in long awakenings,
wake after sleep onset, sleep fragmentation, wake bouts, and sleep-onset
latency. The corrected all-diagnosis model is slightly worse than baseline in
MAE, so these features should remain hypothesis-generating rather than
presented as useful attention predictors.

For executive performance, the top final-model features are clinically coherent
with the earlier GEE result: sleep fragmentation, awakenings longer than five
minutes, and normalized/diary WASO variability dominate the model. Executive
performance beats the median baseline modestly, but the explained variance is
still small. This is supportive evidence for the sleep-continuity hypothesis,
not a strong predictive model.

### 10.4 Interpretation of feature importances

The regression feature importances were calculated from final models fitted
after validation on the complete labelled dataset. They are useful for
interpretation and hypothesis generation, but they are not nested
feature-stability estimates and should not be treated as independently
validated biomarkers.

Across outcomes, the recurring feature families in the corrected all-diagnosis
run are:

| Outcome | Dominant final-model feature families |
|---|---|
| RBDq | sleep-onset latency, subjective sleep/rest quality, long awakenings, sleep efficiency, raw WASO, wake bouts |
| UPDRS | sleep-onset latency, wake bouts, sleep efficiency, long awakenings, diary WASO, wake after sleep offset |
| MFS | wake after sleep offset, sleep-onset latency, long awakenings, time in bed, raw WASO |
| Visuospatial | sleep-onset latency, sleep-efficiency variability, long awakenings, diary WASO |
| Attention | long-awakening variability, diary WASO, sleep fragmentation, wake bouts, sleep-onset latency |
| Executive | sleep fragmentation, long awakenings, normalized/diary WASO variability |

The repeated appearance of sleep continuity and fragmentation measures is
consistent with the correlation/GEE analyses. The corrected regression analysis
adds a more conservative point: these features carry some information about
clinical scores, but only a small fraction of individual score variability is
currently explained.

### 10.5 Focused HC-versus-preDLB regression after WASO correction

The four focused HC-versus-preDLB regression variants were rerun after
regenerating the source Excel exports with corrected WASO normalization. These
are the current focused regression runs for HC-versus-preDLB interpretation:

| Run | Dataset | Feature set | Run directory | Visits / person groups |
|---|---|---|---|---:|
| Clinical core | `dataset-clinical` | grouped sleep/diary features only | `media/regression/clinical-scales/dataset-clinical/hc-vs-predlb-core/20260629_141420/` | 132 / 118 |
| Clinical extended | `dataset-clinical` | grouped sleep/diary features plus lifestyle diary predictors | `media/regression/clinical-scales/dataset-clinical/hc-vs-predlb-extended/20260629_140501/` | 132 / 118 |
| Clinical-acc core | `dataset-clinical-acc` | grouped sleep/diary/activity features only | `media/regression/clinical-scales/dataset-clinical-acc/hc-vs-predlb-core/20260629_135250/` | 129 / 116 |
| Clinical-acc extended | `dataset-clinical-acc` | grouped sleep/diary/activity features plus lifestyle diary predictors | `media/regression/clinical-scales/dataset-clinical-acc/hc-vs-predlb-extended/20260629_134118/` | 129 / 116 |

All four focused reruns used person-grouped outer and inner cross-validation.
They also used foldwise residualization for age, gender, and education. This
is slightly more conservative than the scenario-specific correlation pipeline,
where HC-versus-preDLB selected gender and education; the regression models
therefore test prediction after also removing linear age effects inside each
training fold.

The best focused model for each outcome, selected by lowest MAE, was:

| Outcome | Best focused run | n / person groups | MAE | RMSE | MAE/range | R2 | Pearson r | Baseline MAE | MAE change vs baseline |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| RBDq | clinical-acc core | 128 / 116 | 1.66 | 2.13 | 16.6% | -0.004 | 0.133 | 1.77 | +6.3% |
| UPDRS | clinical-acc core | 127 / 115 | 2.99 | 4.42 | 12.0% | 0.096 | 0.321 | 2.85 | -4.8% |
| MFS | clinical-acc core | 128 / 116 | 0.68 | 0.90 | 16.9% | 0.037 | 0.304 | 0.57 | -18.5% |
| Visuospatial | clinical core | 130 / 117 | 0.71 | 0.90 | 19.7% | 0.177 | 0.426 | 0.79 | +9.8% |
| Attention | clinical-acc extended | 129 / 116 | 0.51 | 0.65 | 14.4% | 0.057 | 0.241 | 0.53 | +4.2% |
| Executive | clinical extended | 131 / 118 | 0.51 | 0.68 | 13.6% | 0.056 | 0.242 | 0.54 | +5.5% |

The complete corrected focused metrics were:

| Outcome | Clinical core MAE / change | Clinical extended MAE / change | Clinical-acc core MAE / change | Clinical-acc extended MAE / change |
|---|---:|---:|---:|---:|
| RBDq | 1.69 / +1.8% | 1.71 / +0.2% | 1.66 / +6.3% | 1.69 / +4.8% |
| UPDRS | 3.13 / -7.1% | 3.07 / -5.2% | 2.99 / -4.8% | 3.01 / -5.8% |
| MFS | 0.73 / -22.5% | 0.73 / -22.2% | 0.68 / -18.5% | 0.69 / -20.3% |
| Visuospatial | 0.71 / +9.8% | 0.74 / +6.1% | 0.82 / -3.3% | 0.80 / -1.3% |
| Attention | 0.53 / +1.3% | 0.53 / +0.3% | 0.51 / +3.5% | 0.51 / +4.2% |
| Executive | 0.52 / +4.2% | 0.51 / +5.5% | 0.52 / +4.8% | 0.52 / +4.9% |

The corrected focused regression interpretation is:

- **Visuospatial performance** is the strongest focused regression target after
  correction. The best model is now the clinical-core sleep/diary model, not
  the activity-augmented or lifestyle-extended model. It has the largest MAE
  improvement over baseline (`9.8%`) and the highest `R2` (`0.177`). Top final
  model features include wake-after-sleep-onset and sleep-efficiency
  variability, so this remains broadly consistent with a sleep-continuity
  signal.
- **RBDq** has the clearest clinically useful MAE improvement among the
  symptom-oriented scales. The best model is the activity-core model, with a
  `6.3%` MAE improvement over baseline. However, `R2` is approximately zero,
  so this is a small absolute predictive gain rather than a robust individual
  RBDq estimator. Top features include raw actigraphy WASO, diary WASO, and
  activity-distribution summaries, aligning with the sleep-fragmentation
  interpretation.
- **Executive performance** remains a modest but coherent target. The best
  model is the clinical extended model, with a `5.5%` MAE improvement. Feature
  importance is dominated by awakenings longer than five minutes, matching the
  correlation/GEE finding that executive performance is linked to nocturnal
  awakening variability.
- **Attention** improves only modestly. The best model is the activity plus
  lifestyle extended model, with a `4.2%` MAE improvement. Top features combine
  activity variability, diary WASO, sleep-efficiency summaries, and alcohol
  count, so this should be interpreted as an exploratory mixed sleep/activity/
  diary-context signal rather than a clean biomarker.
- **UPDRS** has the highest prediction-observation correlation among the best
  focused models (`r = 0.321`) and the activity-feature model uses plausible
  activity-variability predictors, but MAE remains worse than the foldwise
  median baseline. This supports UPDRS as an association/effect-estimation
  outcome, not as a reliable individual-level regression target in the current
  sample.
- **MFS** still does not beat baseline after the WASO correction. The best MAE
  is in the activity-core model, but it remains `18.5%` worse than the
  baseline. This is important because the corrected correlation analysis also
  retired the normalized-WASO/MFS focused candidate. MFS should therefore not
  be presented as a successful prediction target at this stage.

The role of lifestyle diary predictors remains secondary. In the clinical-only
feature set, lifestyle predictors improve UPDRS and executive performance
slightly but worsen RBDq, visuospatial performance, and attention. In the
activity-enhanced feature set, lifestyle predictors help attention and
executive performance modestly but do not improve RBDq, UPDRS, MFS, or
visuospatial performance. Alcohol/caffeine and related diary variables should
therefore be reported as an extended sensitivity model, not as primary disease
biomarkers.

Overall, focusing on HC vs preDLB remains useful for interpretability, but it
still does not create a clinically strong score estimator. The focused
regression results are best reported as exploratory predictive support for the
feature-outcome associations, with visuospatial performance, RBDq, executive
performance, and attention as the only outcomes that beat the median baseline
in their best variants. UPDRS and MFS remain association-level outcomes rather
than reliable regression targets.

## 11) Interpretation

### 11.1 Motor impairment

The most coherent adjusted result is the relationship between activity variability and UPDRS. Several robust summaries of activity dispersion remain associated with UPDRS after diagnosis and scenario-specific covariate adjustment, with the strongest representative corresponding to approximately `1.2-1.3` UPDRS points per feature standard deviation.

This is currently the clearest candidate family for focused motor-outcome analysis.

### 11.2 Attention and executive performance

Attention is associated with several activity-distribution and variability measures. Negative coefficients for relative interdecile-range variability suggest poorer attention scores with greater variability, subject to confirmation of the score direction.

Executive performance is repeatedly associated with the variability of awakenings longer than five minutes. These associations remain after diagnosis and covariate adjustment in both feature sets.

### 11.3 Sleep continuity, cognitive fluctuations, and RBD symptoms

The previous range-of-normalized-WASO association with MFS did not remain a
focused candidate after correcting the WASO seconds-versus-minutes mismatch
and rerunning the source exports, grouped datasets, correlations, and
feature-family follow-up. It should therefore be removed from the main
clinical interpretation.

RBDq shows the clearest evidence of diagnosis-dependent effects: wake bouts, sleep efficiency, and wake after sleep onset are associated with RBDq primarily within preDLB. These interactions are scientifically interesting because they are more specific than a pooled correlation, but they require replication.

### 11.4 Visuospatial performance

The spectacular pooled diary-normalized sleep-onset association is not estimable in the raw-feature GEE model and must be interpreted cautiously. More defensible adjusted visuospatial candidates include activity percentiles and minimum wake-after-sleep-offset measures, although their clinical magnitude is modest.

### 11.5 MCI-AD vs HC

No candidate survived the first pooled FDR screen in the isolated MCI-AD vs HC analysis. With approximately 30 MCI-AD participants and hundreds of correlated features, the analysis is likely underpowered. The correct conclusion is that no association was detected under the present multiplicity control.

## 12) Methodological Caveats

1. **The work remains exploratory.** Feature selection and follow-up use the same dataset, so the GEE stage is a robustness and interpretation analysis, not independent validation.
2. **Raw and exploratory adjusted features are related but not identical.** Pooled discovery adjusted nightly values before aggregation; GEE follow-up models raw grouped features with covariates directly. Coefficient magnitudes and signs can therefore differ between pooled Spearman `rho` and adjusted `beta`. They answer different questions and neither should be relabeled as the other.
3. **Feature families remain correlated.** Family reduction substantially improves interpretability but does not create statistically independent biological constructs.
4. **FDR is outcome-specific.** Correction is performed across features separately for each clinical outcome and analysis type, not globally across every outcome and model.
5. **The two feature sets are not independent replications.** They contain overlapping subjects and features.
6. **Repeated visits are handled by person-level GEE clustering.** This improves inference but does not substitute for a dedicated longitudinal model of change over time.
7. **An independence working correlation was used.** Robust sandwich covariance protects inference against correlation misspecification, but estimates should be checked in sensitivity analyses.
8. **Constant raw summaries are not estimable.** Maximum diary-normalized sleep-onset latency and maximum diary-normalized awakenings over five minutes vary after nightly residualization but are constant in their raw grouped form.
9. **Covariate selection by preliminary p-value is pragmatic, not causal.** Publication models should include prespecified age, gender, and education sensitivity analyses.
10. **Clinical score direction must be verified before clinical wording is finalized.** Statistical signs are reported accurately, but “better” and “worse” interpretations depend on each score’s coding.
11. **The figures combine raw observations and adjusted predictions.** Vertical group separation in the raw points is not itself a diagnosis-specific slope effect. Only the formal FDR-corrected interaction test supports such an interpretation.
12. **Normalized actigraphy WASO was corrected and rerun.** Raw `SleepNight.waso`
    is stored in seconds, while the age-dependent normalization thresholds are
    expressed in minutes. This mismatch has now been corrected and the source
    Excel exports plus correlation/follow-up analyses were regenerated on
    2026-06-29. Historical results using the old
    `actigraphy_norm.Wake after sleep onset` values should not be interpreted.
13. **Earlier strict classification prediction files were visit-level, but the
    thesis validation is now person-grouped.** The 2026-07-01 validity audit
    found that first-visit-only evaluation lowered HC-versus-preDLB
    performance. The 2026-07-02 thesis validation then retrained the models
    with person-grouped outer and inner CV, so repeated visits no longer leak
    across folds in that run.
14. **Source-label ascertainment is confirmed, but technical source confounding
    is not.** Diagnostic composition differs substantially among COBEN, HC/HC2,
    and pre-LBD sources because the consortium recruitment strata were defined
    using prior expectations of health or clinical concern. In the
    person-grouped thesis run, source alone reached balanced accuracy around
    `0.84`, as expected under that design. Within-source nested-CV estimates
    were weak or not estimable, but had very small minority groups. Diagnostic
    classification therefore cannot currently be interpreted independently of
    recruitment design; neither the source-only benchmark nor these imbalanced
    sensitivities prove acquisition/protocol confounding.
15. **The association models have not yet included recruitment source.** This
    does not invalidate the current diagnosis-, sex-, and education-adjusted
    GEE findings, but source may be related to both wearable measurements and
    clinical outcomes through recruitment or protocol differences. Before
    publication, primary associations should receive source-adjusted and, where
    estimable, source-stratified sensitivity analyses. Diagnosis-source
    collinearity and small minority groups must be reported rather than hidden.
16. **Clinical-scale regression is internally person-grouped but still not
    externally validated.** The regression folds keep repeated visits from the
    same person together, but this does not solve source-associated
    ascertainment or transportability to a new cohort.
17. **Regression error rates use observed score ranges.** The reported
    `MAE/range` values divide by the minimum-to-maximum span observed in this
    dataset. They should not be confused with error as a percentage of the
    theoretical questionnaire maximum.
18. **Final regression feature importances are not stability estimates.** They
    come from final all-data models after validation. Bootstrap or repeated
    grouped resampling is needed before presenting individual predictors as
    stable score-estimation biomarkers.
19. **Focused HC-versus-preDLB regression improves interpretability, not all
    outcomes.** Removing MCI-AD and NonHC observations reduces clinical
    heterogeneity but also changes the score distributions and sample size.
    Focused regression should therefore be compared against its own baseline
    and not interpreted as automatically superior.
20. **Lifestyle diary predictors require cautious interpretation.** Alcohol
    and caffeine timing sometimes improve prediction, but they can also encode
    cohort, lifestyle, diary-completion, or reporting-pattern differences.
    They should be reported as an extended model, not as primary disease
    biomarkers.

## 13) Recommended Next Steps

1. Treat the 2026-06-29 WASO-corrected correlation outputs as canonical and exclude the retired normalized-WASO/MFS candidate from the primary interpretation.
2. Use the retained focused plots and effect estimates to define a short, clinically interpretable candidate list.
3. Treat the strict and person-grouped classifiers as exploratory thesis
   analyses, not as diagnostic models. The 2026-07-02 person-grouped run
   confirms moderate pooled internal signal but does not establish
   source-independent transportability.
4. Fit prespecified sensitivity models with age, gender, and education
   regardless of preliminary covariate-test significance, then add
   source-adjusted and source-stratified GEE sensitivities while reporting
   diagnosis-source collinearity and small strata.
5. Compare the current GEE results with alternative working correlations and, where appropriate, mixed-effects models.
6. Do not spend more effort tuning the current diagnostic classifier unless a
   better balanced validation design is available. Person-grouped validation
   has already been run and shows that recruitment-source enrichment prevents
   a clean source-independent interpretation.
7. If classification is included in the thesis, present the full validation
   ladder: visit-level strict model, first-visit audit, person-grouped nested
   CV, source-only ascertainment benchmark, leave-one-cohort-out, and within-cohort
   nested CV.
8. Compare actigraphy-only, diary-only, and combined classifiers. A signal
   confined to diary variables would be especially vulnerable to reporting
   and protocol differences.
9. If activity-enhanced all-diagnosis regression remains relevant, rerun that
   broader activity analysis using the WASO-corrected source exports. The
   all-diagnosis clinical regression and four focused HC-versus-preDLB
   regression variants have already been regenerated.
10. Keep alcohol/caffeine and other diary-lifestyle predictors as a secondary
   extended feature set and report their incremental value against the core
   model.
11. For RBDq, test diagnosis-stratified or interaction-aware regression models
   rather than relying on a single pooled score predictor.
12. For UPDRS and MFS, prioritize association and effect-estimation analyses
    over individual-level regression until a model beats the foldwise median
    baseline.
13. Report regression results only together with the foldwise median baseline,
    because several outcomes show non-zero prediction correlation without
    improving MAE.
14. For the article, prioritize the association/GEE/regression story over
    diagnostic classification. Classification can be mentioned as exploratory
    or reserved for the thesis methodology chapter.
15. Estimate feature-selection stability across grouped resamples before
    identifying classifier biomarkers.
16. Treat all present effects as hypothesis-generating until confirmed in
    held-out or external data.
17. Consider bootstrap stability analysis for feature-family selection and
    effect estimates.

## 14) Reproducibility

The updated runs are:

- canonical WASO-corrected sleep/diary correlation run:
  `media/analysis-preparation/dataset-clinical/20260629_132528_560853/`
- canonical WASO-corrected sleep/diary + activity correlation run:
  `media/analysis-preparation/dataset-clinical-acc/20260629_132533_149540/`
- refreshed source exports:
  `dataset-clinical.xlsx` and `dataset-clinical-acc.xlsx`, regenerated on
  2026-06-29 at 13:25
- previous non-strict global-covariate classifier:
  `media/classification/grouped-statistics-with-covariates/dataset-clinical/all/20260422_215810/`
- corrected non-strict scenario-specific classifier:
  `media/classification/grouped-statistics-with-covariates/dataset-clinical/all/20260614_203725/`
- WASO-corrected non-strict classifier, clinical RFE:
  `media/classification/grouped-statistics-with-covariates/dataset-clinical-rfe/all/20260630_085208/`
- WASO-corrected non-strict classifier, clinical-acc RFE:
  `media/classification/grouped-statistics-with-covariates/dataset-clinical-acc-rfe/all/20260630_091208/`
- WASO-corrected non-strict classifier, clinical-acc non-RFE:
  `media/classification/grouped-statistics-with-covariates/dataset-clinical-acc/all/20260630_095927/`
- WASO-corrected non-strict classifier, clinical non-RFE:
  `media/classification/grouped-statistics-with-covariates/dataset-clinical/all/20260630_103329/`
- previous strict nested global-covariate classifier:
  `media/classification/grouped-statistics-strict-with-covariates/dataset-clinical/20260504_175456/`
- WASO-corrected strict nested-CV classification with diary covariates and RFE:
  `media/classification/grouped-statistics-strict-with-covariates/dataset-clinical-rfe/20260629_173137/`
- strict RFE HC-vs-preDLB probability diagnostic plots:
  `media/classification/grouped-statistics-strict-with-covariates/dataset-clinical-rfe/20260629_173137/scenario-preDLB_vs_HC/diagnostic_plots/`
- official HC-vs-preDLB feature-family stability analysis:
  `media/feature-family-stability/hc-vs-predlb/20260630_135657/`
- stable-family restricted HC-vs-preDLB association follow-up:
  `media/feature-family-restricted-analysis/hc-vs-predlb/20260701_094747/`
- corrected strict stable-family HC-vs-preDLB classifier, primary sleep:
  `media/classification/grouped-statistics-strict-with-covariates/dataset-clinical-stable-primary-sleep-hc-predlb/20260701_120433/`
- corrected strict stable-family HC-vs-preDLB classifier, primary sleep + activity variability:
  `media/classification/grouped-statistics-strict-with-covariates/dataset-clinical-acc-stable-primary-sleep-activity-hc-predlb/20260701_125206/`
- HC-vs-preDLB classification validity checks:
  `media/classification/validity-checks/hc-vs-predlb/20260701_180935/`
- thesis-level person-grouped HC-vs-preDLB classification and cohort sensitivity:
  `media/classification/person-grouped-thesis/hc-vs-predlb/20260702_111118/`
- WASO-corrected strict nested-CV classification with diary covariates, non-RFE comparison:
  `media/classification/grouped-statistics-strict-with-covariates/dataset-clinical/20260629_145843/`
- WASO-corrected all-diagnosis clinical-scale regression:
  `media/regression/clinical-scales/dataset-clinical/20260629_143305/`
- WASO-corrected focused HC-vs-preDLB regression, clinical core:
  `media/regression/clinical-scales/dataset-clinical/hc-vs-predlb-core/20260629_141420/`
- WASO-corrected focused HC-vs-preDLB regression, clinical extended:
  `media/regression/clinical-scales/dataset-clinical/hc-vs-predlb-extended/20260629_140501/`
- WASO-corrected focused HC-vs-preDLB regression, clinical-acc core:
  `media/regression/clinical-scales/dataset-clinical-acc/hc-vs-predlb-core/20260629_135250/`
- WASO-corrected focused HC-vs-preDLB regression, clinical-acc extended:
  `media/regression/clinical-scales/dataset-clinical-acc/hc-vs-predlb-extended/20260629_134118/`

Each scenario contains:

- `correlation/feature_clinical_correlation_matrix.xlsx`
- `correlation/feature_family_followup_analysis.xlsx`
- `correlation/focused_plots/*.png`
- `correlation/focused_plots/*.pdf`

The follow-up workbook includes:

- `interpretation_summary`
- `stratified_correlations`
- `adjusted_associations`
- `diagnosis_interactions`
- `diagnosis_specific_slopes`
- `candidate_pairs`
- `focused_plot_index`
- `settings`

The PNG files are intended for reports and presentations. Matching vector PDF versions are available for publication-quality export.

The fresh analyses contained:

- sleep/diary preDLB vs HC: `132` visits from `118` people;
- sleep/diary combined: `162` visits from `145` people;
- extended preDLB vs HC: `129` visits from `116` people;
- extended combined: `159` visits from `143` people.

## 15) Current Conclusion

After reducing redundant feature variants and explicitly modeling diagnosis, covariates, and repeated visits, the project retains several interpretable exploratory signals.

The strongest shared pattern links greater activity variability with higher
UPDRS. Attention is associated with activity-distribution variability and
executive performance with variability in nocturnal awakenings. After WASO
correction, the apparent MFS association with normalized wake-after-sleep-onset
variability is no longer retained as a focused candidate. In preDLB vs HC,
three sleep-continuity measures show preDLB-specific relationships with RBDq,
while one awakening-variability association with executive performance is
stronger in HC.

The previously strongest pooled sleep-onset-latency result cannot be estimated from the corresponding raw grouped feature and is therefore downgraded from the main conclusion.

No FDR-significant candidate was detected for MCI-AD vs HC alone. Focused
visualization is now complete. Person-grouped thesis validation of the
HC-versus-preDLB classifiers has also been completed and confirms the same
practical boundary: diagnostic classification contains moderate internal
signal, but source-independent transportability is not established by the
available imbalanced sensitivity checks.

WASO-corrected strict nested classification with RFE confirms a moderate
internal signal but still produces conservative results. The primary
HC-versus-preDLB estimate is ROC AUC `0.764` and balanced accuracy `0.668`.
This is better and more interpretable than the non-RFE strict run, but it is
not strong enough to claim diagnostic performance. The combined and isolated
MCI-AD models also show moderate point estimates, with limited specificity or
sensitivity depending on the scenario.

The corrected strict stable-family HC-versus-preDLB classifiers preserve part
of this signal but do not improve diagnostic performance. The primary sleep
family classifier gives ROC AUC `0.691` and balanced accuracy `0.664`; the
sleep-plus-activity-variability classifier gives ROC AUC `0.690` and balanced
accuracy `0.650`. These results support the stable families as interpretable
signal carriers, but not as a superior diagnostic classifier.

The new HC-versus-preDLB probability diagnostics clarify this classification
result. The model separates the probability distributions in the expected
direction, with median predicted preDLB probability `0.289` in HC and `0.844`
in preDLB, and the precision-recall curve stays above the prevalence baseline.
However, the overlapping distributions, imperfect calibration, and confident
wrong classifications mean that the probabilities should be treated as
ranking/screening scores rather than calibrated individual diagnostic risks.

The official HC-versus-preDLB feature-family stability analysis strengthens
the interpretation by moving away from individual feature importance. The
families with the clearest cross-method support are long awakenings, sleep
onset latency, sleep efficiency, and wake-bout frequency. Corrected WASO
remains supported as a family, but not as the old normalized-WASO/MFS focused
candidate. Activity variability is supported in the activity-enhanced stream
and should be reported as a separate secondary extension. Lifestyle variables
remain useful sensitivity predictors but are not primary physiological
claims.

The stable-family restricted HC-versus-preDLB confirmation makes this more
specific. When the analysis is limited to the primary sleep families, the
retained adjusted clinical signal is mainly long-awakening variability versus
executive performance. When activity variability is added, the clearest
additional signals are activity variability versus UPDRS and attention.
Visuospatial and sleep-onset-latency findings remain exploratory because they
weaken after diagnosis/covariate adjustment or are not estimable in the raw
GEE follow-up.

Relative to the previous global-covariate classifier runs, corrected
scenario-specific handling plus RFE strengthens all three strict comparisons,
with the largest relative gain in MCI-AD vs HC. For preDLB vs HC, the RFE
model improves AUC and balanced accuracy, but the practical classification
conclusion remains conservative because performance is still moderate and not
externally validated.

The 2026-06-30 non-strict sensitivity runs show higher apparent performance,
particularly clinical non-RFE for HC-versus-preDLB (`AUC = 0.804`, balanced
accuracy `0.744`) and activity-enhanced non-RFE for the combined scenario
(`AUC = 0.774`, balanced accuracy `0.730`). These results identify useful
candidate configurations for future strict reruns, but they should not replace
the strict nested RFE estimates as the primary performance claims.

The first clinical-scale regression analysis provides a useful negative and
calibrating result. Person-grouped prediction of continuous clinical scores is
not yet strong. Visuospatial and executive scores show the most promising
internal signal, but even these explain only a small fraction of score
variance. RBDq, UPDRS, and MFS do not currently beat the foldwise median
baseline in MAE. Therefore, the regression output supports the same biological
theme as the association analyses - sleep continuity and fragmentation relate
to clinical status - but it does not yet support accurate individual clinical
score estimation.

The focused HC-versus-preDLB regression round partially improves this picture.
It strengthens visuospatial prediction (`R2 = 0.175`, Pearson `r = 0.433`),
keeps executive prediction modestly above baseline, improves attention only in
the activity-plus-lifestyle extended model, and gives RBDq a small baseline
improvement in the activity-core model. However, UPDRS and MFS still do not
beat the median baseline in MAE. Therefore, the assumption that intermediate
states are the only barrier is incomplete. Removing MCI-AD and NonHC helps
interpretability and some targets, but score noise, feature overlap,
source-associated recruitment or measurement heterogeneity, and sparse scales
still limit individual prediction.

Alcohol/caffeine timing and related diary-lifestyle predictors should remain
in an extended secondary model. They sometimes improve MAE, especially for
attention and visuospatial performance, but their clinical meaning is weaker
than actigraphy/sleep-continuity features because they may reflect behavior,
reporting, or cohort differences rather than disease-specific physiology.

The classification findings should remain secondary to the current
association analyses. The 2026-07-02 thesis validation has now grouped
validation by underlying person and tested cohort/source transportability; it
shows moderate pooled internal signal, while source-independent performance
remains unresolved because the recruitment strata are diagnosis-enriched and
the within-source minority classes are small. The most important next step is
therefore not additional hyperparameter tuning, but either acquiring a better
balanced validation cohort or focusing the article on the association, GEE,
and regression evidence.
