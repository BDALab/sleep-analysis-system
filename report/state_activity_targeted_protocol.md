# Targeted UPDRS and attention follow-ups

Written before inspecting the new state-specific outcome associations on
26 October 2026. This is a post hoc extension selected from the article's
reported activity findings, not independent validation or preregistration.

## Two fixed questions

1. UPDRS: IQR across nights of nightly activity median absolute deviation (MAD).
2. Attention: SD across nights of nightly relative interdecile activity range.

For each question, compare whole interval, predicted sleep, and predicted
nocturnal wake on the same nights and complete-case visits. Do not screen
other metrics, outcomes, thresholds, or feature combinations. RBDq results and
the manuscript remain unchanged. There is no classifier fitting or raw-signal
reprocessing.

## Definitions and validation

Nightly MAD is median (abs (activity - median (activity))), without a consistency
scaling factor. Nightly relative interdecile range is (P90 - P10) / max (activity),
using linear quantiles; a zero maximum gives a missing ratio, not zero. Across
nights use linear-quantile IQR for UPDRS and population SD (ddof=0) for attention,
matching the original extraction/grouping code.

Read the existing minute-level activity index and cached sleep/wake predictions
referenced by the completed extraction. Check their SHA-256 fingerprints against
that extraction. Reuse the existing UTC/interval-overlap alignment, probability
threshold, mixed-minute policy, coverage and support rules. Do not regenerate
predictions, change source files, or write to the database.

Before interpreting new associations:

- Reproduce the article's original whole-interval primary GEE estimates from
  the frozen extended dataset, for both fixed outcome-feature pairs. Check n,
  people, beta, robust SE, CI, and unadjusted p, not its older FDR family.
- Recompute the legacy nightly metrics using the original inclusive
  onset/offset timestamp slice and compare with the frozen nightly export.
  Stop interpretation on a substantive discrepancy.
- Match database night/recording/visit identities and interval boundaries to
  the supplied extraction. Diagnosis is numeric code 0/3, never inferred from
  identifier prefixes or the legacy #Disease field. Use the established person
  and collection/ascertainment mappings.

The main whole-interval comparator uses the same fully covered, finite minute
windows as the state extraction, including mixed minutes. Predicted sleep and
wake exclude mixed minutes under the primary policy. Thus sleep and wake alone
do not exhaust all whole-interval minutes. The first/last epoch edges are
recovered from centre timestamps as in the existing state analysis.

As an endpoint-convention sensitivity, replace only the whole-interval
comparator with the legacy whole-interval metric, retaining identical selected
nights and visits and the sleep/wake values. This is separate from the primary
aligned-window comparison.

## Common samples

Restrict to the frozen manuscript HC/MCI-LB cohort. A usable paired night must
meet the extraction's coverage and minimum-minute rules for both states and
have finite values for all comparator versions of the target metric. Require
at least five paired nights and no more than seven selected nights per visit.
Use a separate complete-case sample for each outcome: its score, the three
features, sex, education, age, and source must be available. Do not restrict
UPDRS/attention by RBDq availability or require all clinical outcomes at once.
Retain repeat visits as observations clustered by the existing person ID.

Snapshot the clinical scores used in this run; validate diagnoses and available
demographics against the extraction/frozen source. Keep all exclusions in the
cohort audit and describe coverage by diagnosis and collection.

## Models and multiplicity

Reuse Gaussian GEE, independence working correlation, robust person-clustered
covariance, standardized predictors and original-scale outcomes. Primary
adjustment is diagnosis, sex, education. Also fit diagnosis interactions and
diagnosis-specific slopes. Retain the article's six pooled age/source/stratum
variants and primary/age-adjusted models within each collection. Endpoint
sensitivity uses primary and age-plus-collection models only.

Within each outcome, window definition, analysis subset and model variant,
there are three planned main-effect tests (whole/sleep/wake), three interaction
tests, and three slope tests per diagnosis. BH correction is separate for each
family. Failed/unestimable tests count as p=1 in the correction denominator but
retain missing p/q outputs. Also report a broader-family companion BH correction
across all six main-effect/interaction tests from both targeted outcomes within
each window/subset/model. These corrections do not replace the article's older
screening FDR values. A broader BH family need not increase every individual
adjusted p value. CIs are ordinary robust 95% intervals, not simultaneous.

Compute descriptive pooled and within-diagnosis rank associations with robust
person-clustered rank regression; correct three variants per outcome/scope.
Do not use these results to select adjusted models.

Comparisons are descriptive: no formal sleep-versus-wake coefficient contrast
or incremental-prediction test is planned. Significance for one state but not
another does not establish a state difference. These are movement-derived
predictions, not PSG-confirmed sleep/wake or REM, and no additional nonwear
exclusion is introduced. Selection for sufficient wake may limit generality.
