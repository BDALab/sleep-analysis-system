# Simple Guide to the Statistics and Graphs

Updated: 2026-06-09

This document explains the statistical analysis in the GeneActiv project in
plain language. It is intended as a practical guide for reading the results,
not as a complete statistics textbook.

The detailed scientific report is:

[phd_supervisor_experiment_report_2026-06-07.md](phd_supervisor_experiment_report_2026-06-07.md)

## One-Minute Summary

- We first check whether the diagnostic groups differ in age, gender, or
  education.
- We use rank-based tests because most sleep and activity features are not
  normally distributed.
- We test both group differences and relationships with clinical scores.
- Because we perform hundreds of tests, we correct the p-values using FDR.
- We combine very similar features into families to avoid counting the same
  underlying pattern many times.
- We use GEE regression to adjust for diagnosis and covariates and to account
  for repeated visits from the same person.
- A significant **association** means two variables are related in the data. It
  does not prove that one causes the other.
- A significant **interaction** means the relationship appears different
  between diagnostic groups.
- The strongest shared pattern concerns activity variability and UPDRS. Three
  sleep-related relationships with RBDq appear specifically in preDLB.

## 1. What Are We Trying to Learn?

The project asks two main questions:

1. **Do diagnostic groups differ?**

   For example, is an actigraphy feature generally higher in preDLB than in
   healthy controls?

2. **Is an actigraphy or sleep feature related to a clinical score?**

   For example, do people with greater activity variability also tend to have
   higher UPDRS scores?

The diagnostic groups are:

- **HC**: healthy controls;
- **preDLB**: participants with possible early signs related to dementia with
  Lewy bodies;
- **MCI-AD**: participants with mild cognitive impairment related to
  Alzheimer's disease.

The clinical outcomes are:

- **RBDq**: questionnaire score for REM sleep behavior disorder symptoms;
- **UPDRS**: motor impairment score;
- **MFS**: Mayo Fluctuations Scale, a score describing cognitive fluctuations;
- **Visuospatial**: visuospatial performance;
- **Attention**: attention performance;
- **Executive**: executive-function performance.

Before saying that a higher cognitive score is "better" or "worse," the coding
direction of the particular score must be confirmed.

## 2. Basic Vocabulary

### Feature

A **feature** is a number calculated from actigraphy, a sleep diary, or another
measurement.

Examples:

- median wake after sleep onset;
- variability in the number of long awakenings;
- average activity;
- range of sleep efficiency;
- IQR of nightly activity variability.

The project initially has hundreds of features because the same underlying
measurement may be summarized using a mean, median, minimum, maximum, standard
deviation, IQR, MAD, and other statistics.

### Common feature summaries

| Term | Simple meaning |
|---|---|
| Mean | ordinary arithmetic average |
| Median | middle value after sorting the observations |
| Minimum / maximum | smallest / largest observed value |
| Range | maximum minus minimum |
| SD | typical spread around the mean |
| IQR | spread of the middle 50% of observations |
| MAD | robust spread around the median |
| CV | variability relative to the mean |

IQR and MAD are called **robust** because a few extreme observations usually
affect them less than they affect the range or standard deviation.

A feature containing `_norm` has been transformed to a normalized scale. Its
exact numerical units are therefore not the original minutes or counts. The
direction and relative ordering remain useful, but its values must be
interpreted according to the transformation used by the pipeline.

### Outcome

An **outcome** is the clinical score that we want to relate to a feature, such
as UPDRS or RBDq.

### Covariate

A **covariate** is another variable that might influence both diagnosis and the
outcome.

In this project the main covariates are:

- age;
- gender;
- years of education.

For example, education may be related to cognitive-test performance. If the
groups have different education levels, an apparent diagnosis effect could
partly be an education effect. Statistical adjustment tries to separate these
effects.

### Visit and person

One person may have more than one visit, for example `HC-10` and `HC2-10`.
These are two observations but not two independent people. The follow-up model
therefore clusters visits by the underlying person.

## 3. Overview of the Analysis

The analysis follows this sequence:

1. Check whether age, gender, and education differ between diagnostic groups.
2. Test whether each feature has an approximately normal distribution.
3. Compare feature values between diagnostic groups.
4. Correlate features with clinical outcomes.
5. Correct p-values because hundreds of tests are performed.
6. Combine strongly related feature variants into interpretable families.
7. Fit adjusted GEE models that include diagnosis and covariates.
8. Test whether the feature-outcome relationship differs by diagnosis.
9. Visualize selected associations.

Each step answers a different question. A feature passing one step does not
automatically prove a clinically useful or causal relationship.

## 4. Covariate Tests

### Welch t-test for age and education

The Welch t-test compares the **average value** of a continuous variable
between two groups.

Example question:

> Is the average number of education years different between preDLB and HC?

The test is called "Welch" because it does not require the two groups to have
the same variance or the same sample size.

In the current exploratory rule:

- `p < 0.05`: the groups show evidence of a difference, so the variable is
  selected for adjustment;
- `p >= 0.05`: there is not enough evidence of a difference under this rule.

This selected:

| Scenario | Selected covariates |
|---|---|
| preDLB vs HC | gender and education |
| preDLB + MCI-AD vs HC | education |
| MCI-AD vs HC | education |

Important: a non-significant age test does **not** prove that age is irrelevant.
It only means that the current sample did not show a statistically significant
group difference. Publication sensitivity models should still consider age,
gender, and education based on clinical knowledge.

### Chi-squared test for gender

The chi-squared test compares **category proportions**.

Example question:

> Is the female/male distribution different between preDLB and HC?

It does not test whether gender causes the diagnosis. It only tests whether the
observed proportions differ more than expected from random sampling
variability.

## 5. Normality: Shapiro-Wilk Test

A normal distribution has the familiar symmetric bell shape. Many classical
statistical tests assume that data or model residuals are approximately normal.

The Shapiro-Wilk test asks:

> Is there evidence that this feature is not normally distributed?

Interpretation:

- `p < 0.05`: evidence against normality;
- `p >= 0.05`: normality was not rejected.

Most project features were non-normal. This is unsurprising because sleep and
activity measures can be skewed, bounded, discrete, or contain outliers.

The normality result is one reason we used rank-based methods such as
Mann-Whitney U and Spearman correlation.

Important: with larger samples, a normality test can detect small and
unimportant deviations. It should be considered together with histograms,
scatter plots, and scientific knowledge.

## 6. Comparing Groups: Mann-Whitney U Test

The Mann-Whitney U test compares the **ranks** of values in two independent
groups.

Instead of relying directly on the numerical distances, it orders values from
smallest to largest and asks whether one group tends to occur higher in that
ordering.

Example:

> Do preDLB participants generally have higher wake-after-sleep-onset values
> than healthy controls?

Why we use it:

- it does not require a normal distribution;
- it is less sensitive to extreme values than an ordinary t-test;
- it works with skewed features.

It is often described as a test of medians, but that is only strictly
appropriate when the two distributions have similar shapes. More generally,
it tests whether values in one group tend to be larger than values in the
other group.

The test tells us whether groups differ. It does not tell us whether the
feature is related to a clinical score.

## 7. Association: Spearman Correlation

Spearman correlation measures whether two variables generally move in the same
or opposite direction.

It uses ranks, so it can detect a monotonic relationship even when the
relationship is not perfectly linear.

The result is called **rho**, written as `rho`.

| Rho | Plain-language meaning |
|---:|---|
| close to `+1` | higher feature values strongly accompany higher outcomes |
| close to `0` | little monotonic relationship |
| close to `-1` | higher feature values strongly accompany lower outcomes |

A rough descriptive guide is:

| Absolute rho | Approximate description |
|---:|---|
| below `0.10` | very weak |
| `0.10-0.29` | weak |
| `0.30-0.49` | moderate |
| `0.50` or more | strong |

These boundaries are not universal. A small association may still be
clinically relevant, and a large association may be unstable in a small
sample.

Correlation does **not** prove causation. For example, activity variability
could be associated with UPDRS because:

- activity variability influences motor symptoms;
- motor symptoms influence daily activity;
- diagnosis or another variable influences both;
- the result is partly due to sampling variation.

## 8. What Is a p-value?

A p-value answers a narrow question:

> If there were no true effect under the statistical model, how surprising
> would results at least this extreme be?

A small p-value is evidence against the no-effect model. It is **not**:

- the probability that the result is false;
- the probability that the hypothesis is true;
- a measure of clinical importance;
- proof that the relationship is causal;
- proof that the result will replicate.

The usual threshold `p < 0.05` is a convention. Results close to the threshold
should not be treated as fundamentally different from results just above it.
Effect size, confidence intervals, sample size, study design, and replication
are also important.

## 9. Why FDR Correction Is Necessary

If we perform one test with a 5% threshold, a false-positive result is possible.
If we perform hundreds of tests, false positives become much more likely.

For example, testing 690 features against six outcomes creates thousands of
opportunities for apparently significant results.

The Benjamini-Hochberg false-discovery-rate correction adjusts the p-values to
limit the expected proportion of false discoveries among the results called
significant.

In this project:

- raw `p` is the uncorrected result;
- `FDR p` or adjusted `p` is the result after correction;
- we normally require `FDR p < 0.05`.

FDR is less strict than trying to prevent even one false positive, but it is
appropriate for exploratory work with many related features.

## 10. Why Features Were Grouped into Families

Many feature names describe nearly the same underlying behavior.

For example:

- mean activity variability;
- median activity variability;
- IQR of activity variability;
- MAD of activity variability.

These are not four independent biological discoveries. They may all represent
one broader idea: **activity variability**.

The pipeline therefore:

1. groups features by their clinical or measurement meaning;
2. measures how strongly variants correlate with each other;
3. clusters variants when their absolute Spearman correlation is at least
   `0.85`;
4. selects a representative feature.

This makes the results easier to interpret and reduces the risk of presenting
many technical variants as separate findings.

## 11. Adjusted Follow-up: GEE

GEE means **generalized estimating equations**.

In this project, GEE is a regression model used to answer:

> Is the feature associated with the clinical outcome after accounting for
> diagnosis, selected covariates, and repeated visits from the same person?

This is stronger than a simple pooled correlation because it can include
several explanatory variables at once.

### Why repeated visits matter

Measurements from the same person tend to be more similar than measurements
from unrelated people. Treating two visits as two fully independent people
would make the amount of independent information look larger than it really
is.

GEE clusters visits by person and uses robust standard errors. This allows both
visits to contribute while accounting for their dependence.

### What "adjusted" means

Suppose education differs between diagnostic groups. An adjusted model asks
about the feature-outcome association while comparing observations at the same
modeled education level.

Adjustment reduces measured confounding, but it cannot remove:

- variables that were not measured;
- measurement error;
- all possible selection bias;
- uncertainty about causal direction.

Therefore, "adjusted association" still does not mean "causal effect."

## 12. Beta per Standard Deviation

The GEE output reports a coefficient called **beta**.

Features use different units, so the feature is standardized. A one-unit
increase in the model means a **one-standard-deviation increase** in that
feature.

Example:

> `beta = 1.208` for activity variability and UPDRS.

Plain-language interpretation:

> After adjustment, increasing the activity-variability feature by one standard
> deviation is associated with an average increase of about 1.21 UPDRS points.

For a negative beta:

> `beta = -0.199` for activity variability and attention.

Plain-language interpretation:

> A one-standard-deviation increase in the feature is associated with an
> average decrease of about 0.20 units in the attention score.

Whether a decrease means poorer performance depends on how the clinical score
is coded.

## 13. Confidence Intervals

A 95% confidence interval describes uncertainty around the estimated beta or
slope.

Example:

> preDLB wake-bout slope: `1.360`, 95% CI `0.662 to 2.058`.

The estimated slope is 1.360, but the data are also reasonably compatible with
values from approximately 0.662 to 2.058 under the model.

Practical reading:

- narrower interval: more precise estimate;
- wider interval: less precise estimate;
- interval crossing zero: the direction is uncertain at the corresponding
  unadjusted 5% level;
- interval fully above or below zero: evidence for a positive or negative
  association.

A confidence interval is not the range containing 95% of individual people,
and it is not a guarantee that the true value has a 95% probability of being
inside this particular interval.

## 14. Interaction: Does the Slope Differ by Diagnosis?

An interaction tests whether the feature-outcome relationship is different
between diagnoses.

Example:

- HC line is approximately flat;
- preDLB line rises clearly;
- the interaction test asks whether this difference in slopes is larger than
  expected from sampling uncertainty.

Interpretation:

- interaction FDR p `< 0.05`: evidence that slopes differ by diagnosis;
- interaction FDR p `>= 0.05`: no sufficient evidence that slopes differ.

Lines can look different in a graph while the formal interaction is
non-significant. Visual appearance alone is not enough.

## 15. How to Read the Focused Graphs

Every graph uses the same basic visual language:

- **x-axis**: the actigraphy or sleep feature;
- **y-axis**: the clinical outcome;
- **colored point**: one subject visit;
- **color**: diagnosis;
- **thin line**: connects repeated visits from the same person;
- **thick line**: adjusted prediction from the GEE model;
- **shaded area**: 95% confidence band around the predicted line;
- **n in the legend**: number of observations with complete data in that group.

For discrete features, points are moved very slightly sideways so overlapping
diagnoses remain visible. The statistical model still uses the original
values.

### A useful reading order

1. Read the x-axis and y-axis labels.
2. Look at the overall direction of each thick line.
3. Check how widely the raw points are scattered.
4. Look at the width of the confidence band.
5. Read the adjusted FDR p-value.
6. Read the interaction FDR p-value before claiming group-specific slopes.
7. Consider whether a few extreme observations may influence the visual
   impression.

## 16. Graph 1: Activity Variability and UPDRS

![Activity variability and motor impairment](../media/analysis-preparation/dataset-clinical-acc/20260607_215625_828823/scenarios/predlb-mci-vs-hc/correlation/focused_plots/01_updrs_iqr_activity_median_absolute_deviation.png)

### What the axes mean

- x-axis: variability of nightly activity variability;
- y-axis: UPDRS motor score.

### What the model found

- adjusted beta: `1.208`;
- adjusted FDR p: `0.000004`;
- interaction FDR p: `0.300`.

### Simple interpretation

People with greater activity variability tend to have higher UPDRS scores,
after adjusting for diagnosis and education.

The association is statistically strong. However, the interaction is not
significant. Although the colored lines look different, we do not have enough
evidence to claim that the slope is genuinely different among HC, MCI-AD, and
preDLB.

The high-UPDRS preDLB observations also remind us that raw data are variable and
that a regression line is only a summary.

## 17. Graph 2: Activity Variability and Attention

![Activity variability and attention](../media/analysis-preparation/dataset-clinical-acc/20260607_215625_828823/scenarios/predlb-mci-vs-hc/correlation/focused_plots/02_attention_iqr_activity_relative_interdencile_range.png)

### What the model found

- adjusted beta: `-0.199`;
- adjusted FDR p: `0.000457`;
- interaction FDR p: `0.771`.

### Simple interpretation

Greater activity variability is associated with a lower attention score after
adjusting for diagnosis and education.

There is no evidence that the relationship differs by diagnosis. The most
defensible conclusion is a shared adjusted association across the modeled
groups, not three distinct diagnosis-specific effects.

The clinical meaning of "lower attention" must be checked against the score
coding before calling it better or worse performance.

## 18. Graph 3: Long Awakenings and Executive Performance

![Nocturnal awakening variability and executive performance](../media/analysis-preparation/dataset-clinical-acc/20260607_215625_828823/scenarios/predlb-mci-vs-hc/correlation/focused_plots/03_executive_sd_actigraphy_norm_awakening_5_minutes.png)

### What the model found

- adjusted beta: `-0.165`;
- adjusted FDR p: `0.00435`;
- interaction FDR p: `0.806`.

### Simple interpretation

Greater night-to-night variability in awakenings longer than five minutes is
associated with a lower executive score.

The interaction is not significant, so there is no evidence that this slope
differs by diagnosis. Again, the safe interpretation is a shared adjusted
association.

## 19. Graph 4: Wake After Sleep Onset and Cognitive Fluctuations

![Wake-after-sleep-onset variability and Mayo Fluctuations Scale](../media/analysis-preparation/dataset-clinical/20260607_215611_803349/scenarios/predlb-mci-vs-hc/correlation/focused_plots/04_mfs_range_actigraphy_norm_wake_after_sleep_onset.png)

### What the model found

- adjusted beta: `0.270`;
- adjusted FDR p: `0.000684`;
- interaction: not estimable.

### Simple interpretation

A greater range of normalized wake after sleep onset is associated with a
higher Mayo Fluctuations Scale score after adjusting for diagnosis and
education. A higher score indicates more cognitive-fluctuation features.

The feature takes only a small number of distinct values and is unevenly
distributed. The interaction model could not reliably estimate separate
diagnosis slopes. The graph therefore shows a common adjusted slope.

We can discuss the overall association, but we should not claim that it is
stronger in one diagnostic group.

## 20. Graph 5: Wake Bouts and RBDq

![Wake bouts and RBD symptoms](../media/analysis-preparation/dataset-clinical/20260607_215611_803349/scenarios/predlb-vs-hc/correlation/focused_plots/05_rbdq_max_actigraphy_wake_bouts.png)

### What the model found

- interaction FDR p: `0.00227`;
- HC slope: `-0.032`, 95% CI `-0.361 to 0.296`;
- preDLB slope: `1.360`, 95% CI `0.662 to 2.058`.

### Simple interpretation

In HC, the line is approximately flat and the confidence interval includes
zero. In preDLB, more wake bouts are associated with a higher RBDq score.

The significant interaction supports the conclusion that the relationship is
different between HC and preDLB. This is the strongest of the three plotted
RBDq interactions.

It remains an exploratory association and should be replicated before being
treated as a biomarker.

## 21. Graph 6: Sleep Efficiency and RBDq

![Sleep efficiency and RBD symptoms](../media/analysis-preparation/dataset-clinical/20260607_215611_803349/scenarios/predlb-vs-hc/correlation/focused_plots/06_rbdq_median_actigraphy_norm_sleep_efficiency.png)

### What the model found

- interaction FDR p: `0.01965`;
- HC slope: `-0.133`, 95% CI `-0.455 to 0.190`;
- preDLB slope: `-1.099`, 95% CI `-1.792 to -0.406`.

### Simple interpretation

The HC relationship is weak and uncertain. In preDLB, higher normalized sleep
efficiency is associated with a lower RBDq score.

The significant interaction supports different slopes between HC and preDLB.

Because the normalized feature has only a few possible values, the graph shows
many overlapping points. Their slight horizontal separation is only for
visibility.

## 22. Graph 7: Wake After Sleep Onset and RBDq

![Wake after sleep onset and RBD symptoms](../media/analysis-preparation/dataset-clinical/20260607_215611_803349/scenarios/predlb-vs-hc/correlation/focused_plots/07_rbdq_median_actigraphy_wake_after_sleep_onset.png)

### What the model found

- interaction FDR p: `0.01436`;
- HC slope: `-0.004`, 95% CI `-0.405 to 0.397`;
- preDLB slope: `1.076`, 95% CI `0.401 to 1.750`.

### Simple interpretation

The HC line is essentially flat. In preDLB, more wake after sleep onset is
associated with a higher RBDq score.

The significant interaction supports a diagnosis-dependent relationship.

## 23. Shared Associations Versus Interactions

The seven graphs fall into two categories:

| Graphs | Best interpretation |
|---|---|
| UPDRS, attention, executive, MFS | adjusted association across modeled groups |
| three RBDq graphs | evidence that the slope differs between HC and preDLB |

This distinction matters.

For the first category, diagnosis is included in the model, but the evidence
does not support different slopes. For the RBDq category, the formal
interaction test supports different slopes.

## 24. What Does "No Significant Result" Mean?

The MCI-AD vs HC analysis did not produce an FDR-significant candidate.

This means:

> Under the current sample size, feature set, variability, and multiple-testing
> correction, we did not detect sufficient evidence.

It does **not** mean:

- the groups are identical;
- no biological difference exists;
- actigraphy is useless for MCI-AD;
- the true effect is exactly zero.

The MCI-AD sample is relatively small, while hundreds of features are tested.
This reduces statistical power.

## 25. Important Limitations

1. This is exploratory analysis, not final confirmation.
2. Candidate selection and follow-up used the same dataset.
3. The results have not yet been replicated in an independent cohort.
4. FDR reduces false discoveries but cannot guarantee that every retained
   result is true.
5. Covariate adjustment handles only measured variables included in the model.
6. Association does not establish causal direction.
7. Some diagnostic groups are small, especially MCI-AD.
8. Several features remain related even after family reduction.
9. Repeated visits are handled statistically but do not create the same amount
   of information as independent participants.
10. Clinical importance must be considered separately from statistical
    significance.

## 26. A Short Template for Explaining a Result

For a shared adjusted association:

> After adjusting for diagnosis and education and accounting for repeated
> visits, higher activity variability was associated with higher UPDRS. The
> association survived FDR correction. There was no evidence that the slope
> differed by diagnosis. This is an exploratory association and does not prove
> causation.

For a diagnosis interaction:

> The relationship between wake bouts and RBDq differed between HC and preDLB.
> The association was positive in preDLB and approximately absent in HC, and
> the interaction survived FDR correction. This suggests a preDLB-specific
> relationship that requires independent replication.

For a null result:

> No association survived FDR correction in the MCI-AD vs HC analysis. This
> indicates insufficient detected evidence in the current sample, not proof
> that no association exists.

## 27. Quick Reference

| Term | Simple meaning |
|---|---|
| t-test | compares averages between groups |
| chi-squared test | compares category proportions |
| Shapiro-Wilk | checks for evidence against a normal distribution |
| Mann-Whitney U | compares ranked values between two groups |
| Spearman rho | direction and strength of a ranked association |
| p-value | compatibility of the result with a no-effect model |
| FDR p | p-value adjusted for performing many tests |
| beta | expected outcome change per one feature SD in the adjusted model |
| 95% CI | uncertainty range around an estimate |
| GEE | regression accounting for covariates and repeated visits |
| interaction | tests whether slopes differ by diagnosis |
| significant | passes the chosen statistical threshold |
| non-significant | insufficient evidence under the current analysis |

## 28. Main Takeaway

The analysis does not simply search for visually attractive correlations. It
uses several filters:

1. robust tests for non-normal features;
2. correction for hundreds of comparisons;
3. reduction of redundant feature variants;
4. adjustment for diagnosis and selected covariates;
5. person-level handling of repeated visits;
6. formal interaction tests before claiming diagnosis-specific effects.

The current evidence most clearly supports:

- an association between activity variability and UPDRS;
- associations between activity or awakening variability and cognitive scores;
- an association between wake-after-sleep-onset variability and MFS;
- three sleep-related RBDq associations that appear specific to preDLB.

These are promising hypotheses for sensitivity analysis and validation, not
final causal conclusions or validated diagnostic biomarkers.
