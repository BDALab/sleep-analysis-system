# Targeted activity follow-ups: results and interpretation

Completed
run: [full results and tables](/Volumes/Portable/geneactiv-processing-data/media/state-activity-targeted/20261026_162604_438589/summary.md).

These are post hoc follow-ups of the two activity associations already in the
manuscript. They do not replace the original findings or provide independent
validation. The manuscript, canonical outputs, and RBDq analysis remain
unchanged. No classifier was trained and no raw accelerometry was reprocessed.

## What was tested

- UPDRS: IQR across nights of nightly activity median absolute deviation (MAD).
- Attention: population SD across nights of nightly relative interdecile range,
  defined as (P90 - P10) / maximum activity index.

Each predictor was calculated for the whole sleep-onset-to-offset interval,
predicted-sleep minutes, and predicted nocturnal-wake minutes. Within an
outcome, the three versions use the same nights and complete-case visits.

UPDRS used **110 visits from 99 people** (62 HC and 48 MCI-LB visits). Attention
used **111 visits from 100 people** (63 HC and 48 MCI-LB visits). The samples
differ because outcome availability differs; neither is restricted by RBDq.

Models retain diagnosis, sex and education adjustment, Gaussian GEE with an
independence working correlation, and robust uncertainty clustered by person.
Each predictor is standardized within its complete-case model sample. Betas
are outcome points per one feature SD. Confidence intervals are ordinary robust
95% intervals, not multiplicity-adjusted intervals.

## Primary results

FDR correction below is across the three planned comparator tests within each
outcome. It is not the older, larger screening family used in the manuscript.

| Outcome   | Activity context | Adjusted beta (95% CI)    | FDR-adjusted p |
|-----------|------------------|---------------------------|----------------|
| UPDRS     | Whole interval   | 1.455 (0.907 to 2.002)    | 5.69e-7        |
| UPDRS     | Predicted sleep  | 0.473 (0.144 to 0.802)    | 0.00722        |
| UPDRS     | Predicted wake   | -0.190 (-0.881 to 0.501)  | 0.591          |
| Attention | Whole interval   | -0.222 (-0.330 to -0.114) | 0.000166       |
| Attention | Predicted sleep  | -0.032 (-0.138 to 0.074)  | 0.771          |
| Attention | Predicted wake   | 0.016 (-0.089 to 0.120)   | 0.771          |

No primary diagnosis interaction passed FDR correction. This does not prove
identical associations in HC and MCI-LB.

## Age and source sensitivity

Both whole-interval findings remain supported after additional age and
collection adjustment:

- UPDRS: beta 1.375, 95% CI 0.796 to 1.953; FDR-adjusted p approximately 0.000010.
- Attention: beta -0.224, 95% CI -0.329 to -0.118; FDR-adjusted p 0.000102.

For the predicted-sleep UPDRS feature, the age/collection-adjusted estimate is **0.458 (95% CI 0.056 to 0.859)**. Its
per-outcome FDR-adjusted p is **0.0384**.
The prespecified companion correction across all six targeted tests gives **0.0511**. Thus this is a positive
exploratory signal whose threshold-based
interpretation is sensitive to the multiple-testing family; it should not be
presented as unconditionally robust. The ordinary 95% CI and the broader
adjusted p value answer different uncertainty/multiplicity questions.

The predicted-sleep UPDRS estimates are positive within both collections,
but support is less precise in NINR (FDR-adjusted p 0.113) than in NU20 (0.0135).
Only eight MCI-LB visits are available in the NINR matched samples. Differences
in significance across collections are not a formal test of heterogeneity.
Neither separate state shows a clear attention association after adjustment.

## What this means for the article

1. **Retain the original whole-interval analyses.** Both findings remain clear
   in this matched-sample follow-up; state splitting is not a general
   improvement over the original features.
2. **UPDRS has an additional predicted-sleep association.** This makes the
   follow-up worth documenting, with the broader-FDR caveat above. It does not
   establish that the whole-interval signal originates specifically in sleep.
3. **The attention finding is not clearly localized to either predicted state.**
   The whole feature captures the combined distribution, mixed minutes, and a
   context-specific normalization. These results cannot identify which of
   those components explains the association.
4. **Do not infer a state difference from significance versus nonsignificance.**
   No formal sleep-versus-wake coefficient contrast was fitted. Context-specific
   betas also represent different one-SD raw feature increments.

If included after collaborator review, a short secondary results paragraph or
small table would be appropriate. Do not replace the main findings or describe
these results as new validated biomarkers, REM-specific effects, or causal
evidence. State estimation itself uses movement; mixed minutes are excluded
from state summaries but included in the whole comparator. No new nonwear mask
was introduced, and sufficient-wake eligibility selects a subset of the cohort.

## Validation performed

- All **14** numerical reproduction checks for the two original manuscript
  models passed, including sample sizes, person counts, beta, robust SE, CI,
  and unadjusted p values.
- All **1,750** available legacy nightly metric comparisons matched the frozen
  export. Eight comparisons had no frozen reference, corresponding to four
  unusable nights; none entered a matched model.
- All **72 main and 72 interaction models** converged successfully.
- Whole-interval endpoint sensitivity changed the primary beta only slightly:
  UPDRS 1.461 versus 1.455, attention -0.221 versus -0.222. Conclusions did not
  depend on the endpoint convention.
- All **55** related regression tests passed.
- All six primary coefficients and robust SE/p values matched independently
  specified cluster-robust OLS calculations (maximum numerical discrepancy
  approximately 5.3e-14). Three-test and six-test BH corrections were checked
  independently. Numerical reruns matched exactly.
- Frozen input, cached signal, implementation and clinical-snapshot fingerprints
  were verified. Original files were not overwritten.

See the [fixed protocol](/Volumes/Portable/geneactiv-processing-data/report/state_activity_targeted_protocol.md)
and [reproduction command](/Volumes/Portable/geneactiv-processing-data/report/state_activity_targeted_followups.md).
