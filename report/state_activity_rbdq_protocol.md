# Targeted exploratory RBDq analysis of activity during predicted sleep and wake

Written on 26 October 2026 before inspecting the new association estimates.
This is a post hoc extension motivated by the internal meeting, not a
preregistered or independent validation study. The manuscript remains unchanged.

## Question and relationship to the article

Do the level and variability of wrist activity during predicted sleep and
nocturnal wake relate to RBDq, and do these associations differ between HC and
MCI-LB? This extends the article's primary clinical association analysis, in
particular its diagnosis-dependent RBDq findings. It is not a new diagnostic
classification experiment and does not address REM-specific motor activity.

Use the same diagnosis-code-defined cohort as the frozen manuscript analyses.
Use the existing Gaussian GEE implementation, independence working correlation,
robust sandwich covariance, and person mapping. Diagnosis comes from numeric
database codes, not subject-name prefixes or the legacy `#Disease` export field.
Subject-name prefixes are used only for the established person/source mapping.

## Input and validation

The requested extraction is `media/state-activity/20261026_130149_395288`.
Validate the manifest, unique visit/night keys, reproduction of all 16 exported
visit features from the nightly CSV, and agreement of available demographics
with the frozen source. Restrict to the frozen core HC/MCI-LB cohort; reconcile
sleep eligibility with the frozen extended activity cohort. Check that included
night dates are in the frozen source and that no visit contains more than seven
nights. Keep the original extraction and canonical outputs unchanged.

Report coverage separately by diagnosis and clinical collection, including
excluded visits, insufficient wake nights, missing clinical/covariate data,
mixed minutes, and missing/invalid signal. These are descriptive checks, not
an unplanned battery of missingness significance tests.

## Features and common sample

Retain all 16 predefined features: four nightly distribution summaries (mean,
median, 90th percentile, IQR), two predicted states, and two across-night
aggregations (median, IQR). No screening, clustering, outcome-driven feature
selection, threshold tuning, or imputation is performed.

For the main comparison, aggregate both states using exactly the same nights:
a paired night must satisfy the extraction's minimum support and coverage rules
for **both** states. Require at least five paired nights per visit. Restrict all
16 models to the same complete-case visits with RBDq, diagnosis, sex, education,
and age (age is complete here, allowing a fixed sensitivity-ready sample).
Retain visits as observations; cluster repeat visits by the existing person ID.

As a measurement sensitivity analysis, refit the primary model using the
original independent-state-night aggregates, on exactly the same visits as the
paired-night analysis. This tests the consequence of matching nights rather
than changing the analyzed participants.

## Estimates and adjustment

- Descriptive pooled and within-diagnosis Spearman estimates use standardized
  rank regression with the same person-clustered Gaussian GEE uncertainty as
  the article. They do not select which adjusted models are fitted.
- Main adjusted models: RBDq ~ standardized feature + diagnosis + sex + education.
- Interaction models add feature x diagnosis and estimate a slope within each
  diagnosis. Report all interactions, not only the significant ones.
- The feature SD is calculated on the complete-case sample for that model.
  RBDq remains in its original score units. CIs are ordinary 95% robust intervals,
  not simultaneous or FDR-adjusted intervals. All tests are two-sided.
- Apply the article's existing sensitivity variants: additionally age,
  collection, both, ascertainment stratum, age plus stratum, and within each
  collection with/without age. Keep the selected features fixed.

## Multiplicity and interpretation

For the single RBDq outcome, define separate BH families of **16 planned tests**
for main effects and diagnosis interactions, within each analysis dataset,
model variant, and analysis subset. Diagnosis-specific slopes have 16 tests per
diagnosis; rank associations have 16 per pooled/diagnosis scope. Unestimable
planned tests count conservatively as p=1 for correction, but their own output
p/q values remain missing. Nonconverged/invalid model results are not interpreted.
These families are separate from the manuscript's older screening families;
smaller q values do not establish improvement over the original feature set.

A significant result in one state and a nonsignificant result in another is **not** a test that the state-specific
associations differ. This analysis does
not fit a formal sleep-versus-wake coefficient contrast or claim incremental
prediction. Interpret the two state estimates descriptively on a matched
sample. Associations cannot establish that RBD causes movement, diagnose RBD,
identify REM, or remove the circularity of movement-informed state estimation.

No change to the paper's primary adjustment model, original conclusions, or
classification pipeline is authorized or required by this extension.
