# HC-versus-preDLB association sensitivity analysis

Date: 2026-08-24

## Purpose

This experiment addresses two reviewer-facing robustness questions in the
Cognitive Computation manuscript:

1. Do the reported feature--clinical associations persist after adjustment for
   age?
2. Do they persist after accounting for clinical collection and when estimated
   separately within NINR and NU20?

## Frozen analysis inputs

The experiment uses the canonical WASO-corrected runs listed in
`report/canonical_correlation_outputs.md`:

- core sleep/diary: `media/analysis-preparation/dataset-clinical/20260629_132528_560853/`
- extended activity: `media/analysis-preparation/dataset-clinical-acc/20260629_132533_149540/`

Representative feature--outcome pairs were frozen from the canonical
`candidate_pairs` sheets. Screening, FDR filtering, feature-family clustering,
and representative selection were not repeated.

## Models

All models were Gaussian GEE with an independence working correlation, robust
sandwich covariance, and visits clustered by underlying person. The feature was
standardized within each model's complete-case sample.

- Primary reproduction: feature + diagnosis + sex + education
- Age-adjusted: primary + age
- Collection-adjusted: primary + NINR/NU20
- Age- and collection-adjusted: primary + age + NINR/NU20
- Ascertainment-stratum audit: primary + COBEN/HC-HC2/pre-LBD-pre-LBD2, with
  and without age
- Within-collection: primary and age-adjusted models repeated separately in
  NINR and NU20

Benjamini--Hochberg FDR correction was applied separately by dataset, analysis
subset, model variant, and clinical outcome. These are post hoc sensitivity
analyses, not independent confirmation.

## Reproduction audit

Before interpreting the new models, the primary variant was compared with the
canonical follow-up workbooks. All 68 candidate models reproduced successfully.

- maximum absolute beta difference: `2.22e-16`
- maximum absolute interaction-p difference: `5.55e-17`

## Pooled reported findings

| Finding | Age-adjusted | Age + collection adjusted | Interpretation |
|---|---:|---:|---|
| UPDRS ~ activity MAD variability | beta 1.186 (0.438, 1.934), FDR p 0.05108 | beta 1.154 (0.449, 1.860), FDR p 0.03622 | Direction and CI robust; age-only FDR narrowly exceeds 0.05 |
| Attention ~ activity-range variability | beta -0.195 (-0.314, -0.075), FDR p 0.00687 | beta -0.194 (-0.312, -0.075), FDR p 0.00699 | Robust |
| Executive ~ long-awakening variability | beta -0.158 (-0.254, -0.062), FDR p 0.00981 | beta -0.158 (-0.254, -0.062), FDR p 0.01046 | Robust |
| Wake bouts x diagnosis for RBDq | interaction FDR p 0.00271 | interaction FDR p 0.00117 | Robust |
| Sleep efficiency x diagnosis for RBDq | interaction FDR p 0.02582 | interaction FDR p 0.02175 | Robust |
| WASO x diagnosis for RBDq | interaction FDR p 0.02152 | interaction FDR p 0.01155 | Robust |

The age- and collection-adjusted preDLB slopes for the RBDq interaction
findings were:

- maximum wake bouts: beta 1.419 (0.751, 2.087), slope FDR p 0.00022
- median normalized sleep efficiency: beta -1.105 (-1.785, -0.426), slope FDR p 0.00250
- median raw WASO: beta 1.113 (0.465, 1.760), slope FDR p 0.00177

Corresponding HC slopes were approximately null.

## Within-collection results

All three shared associations retained the same direction in both NINR and
NU20, but precision differed:

- age-adjusted UPDRS and attention associations remained FDR-significant in
  NINR, not NU20;
- the age-adjusted executive association remained FDR-significant in NU20, not
  NINR.

For the age-adjusted RBDq interactions:

- none was FDR-significant within NINR;
- wake bouts (FDR p 0.00002) and WASO (FDR p 0.01222) remained significant
  within NU20;
- sleep efficiency did not survive within-NU20 FDR correction (FDR p 0.08303).

The NINR clinical data contained only 8--9 preDLB visits depending on feature
availability. NU20 also has diagnosis-enriched recruitment strata. These
within-collection analyses therefore assess consistency and precision, not
independent replication.

## Reproducible command and outputs

```bash
python manage.py run_association_sensitivity
```

Validated output:

- `media/association-sensitivity/hc-vs-predlb/20260824_155751/association_sensitivity_analysis.xlsx`
- `media/association-sensitivity/hc-vs-predlb/20260824_155751/association_sensitivity_summary.json`
- formatted and visually verified review copy:
  `outputs/association-sensitivity-20260824/association_sensitivity_analysis.xlsx`

Implementation and tests:

- `dashboard/logic/association_sensitivity_analysis.py`
- `dashboard/management/commands/run_association_sensitivity.py`
- `dashboard/test_association_sensitivity_analysis.py`
