# TRIPOD+AI working audit

This is an internal, paraphrased audit of the secondary HC-versus-preDLB
classification analysis. It is not the journal's official checklist and should
be revisited immediately before submission. The official 27-item TRIPOD+AI
statement is Collins et al., *BMJ* 2024;385:e078378:
https://doi.org/10.1136/bmj-2023-078378.

The manuscript treats classification as a secondary validity analysis, not as
a clinically deployable prediction model. Items about intended clinical use or
deployment should therefore be answered explicitly as not applicable rather
than implying readiness for clinical use.

## Already reported or substantially addressed

- The title identifies the population, data type, multi-night setting, and
  source-aware computational framing.
- The Introduction states the clinical context, study objectives, and the
  computational contribution.
- The four data sources and the sleep/wake model-development sources are
  separated from the two clinical collections.
- The outcome is HC versus preDLB, with repeated visits linked to the underlying
  person before splitting.
- The three candidate predictor pipelines and feature-family restrictions are
  described.
- Preprocessing, residualisation, imputation, scaling, feature selection,
  hyperparameter tuning, and threshold tuning are confined to training folds.
- Nested five-fold stratified person-grouped validation, seed 17, evaluated
  metrics, and first-visit sensitivity analysis are reported.
- Recruitment-source dependence is audited with an ascertainment-stratum-only
  benchmark and within-stratum nested validation.
- Per-model visit/person counts and discrimination metrics are reported.
- Code repositories and principal software versions are cited.
- Limitations state that this is internal validation and that the small,
  diagnosis-enriched sample does not support deployment claims.

## High-priority items before submission

- [x] Report the analytical record flow into each classification pipeline,
  including missing-label, non-target-label, and missing-activity exclusions.
- [ ] Add the clinical recruitment flow by source, including eligibility and
  clinical exclusion reasons, after confirmation by the clinical team.
- [ ] Confirm recruitment dates, eligibility criteria, diagnostic procedures,
  clinical assessors, and ethics/consent identifiers with the clinical team.
- [x] State that the sample used all analytically eligible available HC/preDLB
  visits and that no prospective prediction-model sample-size calculation was
  performed.
- [x] Report predictor and demographic missingness, the 90% coverage filter, and
  fold-confined median imputation.
- [x] Report that no synthetic resampling was used and that positive-class weight
  was selected inside the hyperparameter search.
- [x] Add 95% confidence intervals from 10,000 diagnosis-stratified person-level
  bootstrap samples for all classification metrics.
- [x] Add descriptive calibration curves, Brier scores, intercepts, and slopes,
  while explaining that the analysis is not an individual-risk model.
- [x] Export the full search space and foldwise selected hyperparameters in the
  internal classification-audit materials (not planned for journal submission).
- [x] Explain the default 0.5 threshold and inner-fold MCC-based threshold
  selection.
- [ ] Freeze the analysis commit, environment/lock file, exact run identifiers,
  and internal audit outputs only after collaborator review is complete.
- [ ] State whether patients or the public were involved in study design,
  analysis, or reporting.

## Interpretation guardrails

- Do not describe the pooled sleep/wake test partition as external validation;
  participants from both development datasets occur on both sides of the split.
- Do not describe source or ascertainment stratum as a negative control: the
  recruitment design deliberately makes stratum informative about diagnosis.
- Do not claim diagnostic transportability from the current within-stratum
  results; the AUC values are near chance and minority-class counts are small.
- Do not freeze or publish restricted clinical data. The reproducibility release
  should contain code, environment metadata, de-identified aggregate outputs,
  and explicit access conditions for non-public data.
