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

- [ ] Add a participant-flow diagram or table giving records assessed, excluded,
  and analysed for each clinical source, with reasons for exclusion.
- [ ] Confirm recruitment dates, eligibility criteria, diagnostic procedures,
  clinical assessors, and ethics/consent identifiers with the clinical team.
- [ ] State whether all eligible available visits were used and add a sample-size
  rationale; if there was no prospective calculation, say so transparently.
- [ ] Report missingness for candidate predictors and outcomes, and distinguish
  unavailable clinical measurements from values imputed inside model folds.
- [ ] Report how class imbalance was handled. If no weighting or resampling was
  used, state this explicitly.
- [ ] Add confidence intervals for ROC AUC, PR AUC, balanced accuracy,
  sensitivity, and specificity using a person-level bootstrap or another method
  that respects repeated visits.
- [ ] Add calibration assessment (at minimum a calibration plot and Brier score)
  if the classification analysis is presented as estimating individual risk.
  If it remains a ranking/validity analysis, explicitly explain why calibration
  is not interpreted clinically.
- [ ] Provide the final search space and foldwise selected hyperparameters or a
  machine-readable supplement sufficient to reproduce each pipeline.
- [ ] Explain how the reported threshold was selected and identify clearly which
  metrics use the default 0.5 threshold and which use inner-fold tuning.
- [ ] Freeze the analysis commit, environment/lock file, exact run identifiers,
  and supplementary outputs only after collaborator review is complete.
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
