# Run the two targeted activity follow-ups

The analysis is restricted to the two activity predictors already reported in
the manuscript: UPDRS with IQR of nightly activity MAD, and attention with SD of
nightly relative interdecile range. It does not scan all outcomes/features.

From `/Volumes/Portable/geneactiv-processing-data`, activate the project's
working environment and run:

```bash
source /Volumes/Portable/geneactiv-processing-data/.venv/bin/activate
python manage.py run_state_activity_targets --input-dir /Volumes/Portable/geneactiv-processing-data/media/state-activity/20261026_130149_395288
```

Use a full project environment from `requirements.txt`, not the minimal
extraction-only environment. The command reads existing cached activity indices
and sleep/wake predictions; it neither retrains models nor reprocesses raw
accelerometry. It writes only to a new timestamped directory under
`media/state-activity-targeted/`. Optional `--output-dir` must name a directory
that does not exist.

The supplied completed state extraction fixes the cache fingerprints, selected
recordings, nights, intervals and support rules. Its explicit exclusions of
recordings 672 and 679 are retained; no additional recordings are silently
dropped. Clinical UPDRS/attention scores are snapshotted from the database for
this run. RBDq availability does not select either outcome's sample.

The command first verifies the article's original primary GEE estimates and the
legacy nightly feature calculations. A failed prerequisite saves an audit and
stops before interpretation of new associations. Whole-interval, predicted-sleep
and predicted-wake features then use the same eligible nights and visits within
each outcome. Across-night SD uses ddof=0, as in the article's code.

Results include:

- `summary.md`: primary and age/collection-adjusted estimates and endpoint
  sensitivity, including three-test per-outcome and six-test companion FDR.
- `nightly_target_features.csv` and `matched_visit_features.csv`: the new
  targeted features, without modifying earlier feature tables.
- `adjusted_models.csv`, `diagnosis_specific_slopes.csv`, `rank_correlations.csv`:
  all planned estimates, not a significance-selected subset.
- `original_model_reproduction.csv`, `legacy_nightly_reproduction.csv`,
  `cohort_audit.csv`, and coverage tables: prerequisite and missingness checks.
- `clinical_snapshot.csv`, immutable pre-fit `protocol.json`, and
  `analysis_manifest.json`: scores, design, fingerprints, versions, completion
  and model statuses.

Read `report/state_activity_targeted_protocol.md` for the fixed questions,
covariates, paired-night criteria, FDR families, and interpretation limits.
Inspect `analysis_manifest.json` for `status: complete` and model statuses before
using results. The manuscript, canonical outputs, and RBDq analysis are not
edited by this command. Never reuse an old output directory for a rerun.

Regression tests can be run without creating a test database:

```bash
DJANGO_SETTINGS_MODULE=mysite.settings python -c 'import django,unittest; django.setup(); s=unittest.defaultTestLoader.loadTestsFromNames(["dashboard.test_state_activity_targeted", "dashboard.test_state_activity_rbdq", "dashboard.test_state_activity_features", "dashboard.test_association_sensitivity_analysis"]); r=unittest.TextTestRunner(verbosity=2).run(s); raise SystemExit(not r.wasSuccessful())'
```
