# Canonical Correlation Outputs

Updated: 2026-06-29

Use only the following WASO-corrected correlation/follow-up runs for current interpretation and reporting.

## Source Exports

- `dataset-clinical.xlsx`, regenerated 2026-06-29 13:25
- `dataset-clinical-acc.xlsx`, regenerated 2026-06-29 13:25

The previous April 2026 source exports were backed up under:

- `media/analysis-preparation/source-backups/20260629_1322_before_waso_export_refresh/`

## Canonical Runs

| Dataset | Canonical run directory | Notes |
|---|---|---|
| Sleep/diary | `media/analysis-preparation/dataset-clinical/20260629_132528_560853/` | WASO-corrected source export; all three diagnostic scenarios rerun |
| Sleep/diary + activity | `media/analysis-preparation/dataset-clinical-acc/20260629_132533_149540/` | WASO-corrected source export; all three diagnostic scenarios rerun |

Each scenario contains:

- `correlation/feature_clinical_correlation_matrix.xlsx`
- `correlation/feature_family_followup_analysis.xlsx`
- `correlation/focused_plots/`

## WASO Correction Interpretation

The previously highlighted `MFS ~ Range.actigraphy_norm.Wake after sleep onset` association is no longer a focused candidate after correcting WASO normalization from seconds to minutes. Do not use that historical result as a retained finding.

HC vs preDLB still shows meaningful candidate associations, especially in the activity-enhanced dataset for UPDRS, attention, visuospatial performance, and executive outcomes.

## Age and source sensitivity analysis

The frozen canonical HC-versus-preDLB candidate pairs were refitted with age,
clinical-collection, ascertainment-stratum, and within-collection sensitivity
models on 2026-08-24. The validated output is:

- `media/association-sensitivity/hc-vs-predlb/20260824_155751/association_sensitivity_analysis.xlsx`
- `outputs/association-sensitivity-20260824/association_sensitivity_analysis.xlsx`
  (formatted and visually verified review copy)
- `report/association_sensitivity_hc_vs_predlb_2026-08-24.md`

The primary-model reproduction audit passed for all 68 candidate models before
the new sensitivity variants were interpreted.
