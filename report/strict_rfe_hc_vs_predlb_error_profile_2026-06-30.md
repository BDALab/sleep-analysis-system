# Strict RFE HC-vs-preDLB Error Profile

Source run: `media/classification/grouped-statistics-strict-with-covariates/dataset-clinical-rfe/20260629_173137/scenario-preDLB_vs_HC/`

## Headline

- Default-threshold confusion matrix: TP `40`, FN `19`, TN `48`, FP `25`.
- Sensitivity `0.678`, specificity `0.658`, balanced accuracy `0.668`, accuracy `0.667`.
- Misclassifications: `44` of `132` visits. Of these, `9` are near-threshold and `16` are confident errors.
- There are `14` people with repeated visits in this scenario; repeated-visit error status is shown below.

## Confusion Groups And Probability Strength

| Group | Meaning | n | median probability preDLB | median distance from 0.5 | near threshold n | confident n |
| --- | --- | --- | --- | --- | --- | --- |
| TP | preDLB correctly classified | 40 | 0.933 | 0.433 | 4 | 30 |
| FN | preDLB classified as HC | 19 | 0.318 | 0.182 | 5 | 8 |
| TN | HC correctly classified | 48 | 0.098 | 0.402 | 4 | 35 |
| FP | HC classified as preDLB | 25 | 0.718 | 0.218 | 4 | 8 |

## Error-Group Clinical Profile

| Group | n | median age | F/M | median education | median RBDq | median UPDRS | median MFS | median visuospatial | median attention | median executive |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| TP | 40 | 70.000 | 36/4 | 12.250 | 4.000 | 5.000 | 1.000 | -1.341 | -0.167 | -0.460 |
| FN | 19 | 71.000 | 7/12 | 13.000 | 4.000 | 6.000 | 0.000 | 0.150 | -0.333 | -0.500 |
| TN | 48 | 69.500 | 22/26 | 17.000 | 2.000 | 1.000 | 0.000 | 0.499 | 0.333 | 0.150 |
| FP | 25 | 71.000 | 14/11 | 13.000 | 2.000 | 1.000 | 0.000 | 0.458 | 0.333 | 0.007 |

## Source Prefix Concentration

| source_prefix | n | TP | FN | TN | FP | error rate |
| --- | --- | --- | --- | --- | --- | --- |
| COBEN | 55 | 5 | 4 | 30 | 16 | 0.364 |
| HC | 16 | 0 | 0 | 11 | 5 | 0.312 |
| HC2 | 1 | 0 | 1 | 0 | 0 | 1.000 |
| pre-LBD | 45 | 25 | 12 | 5 | 3 | 0.333 |
| pre-LBD2 | 15 | 10 | 2 | 2 | 1 | 0.200 |

## First vs Second Visit

| visit_label | n | TP | FN | TN | FP | sensitivity | specificity | BACC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| first visit | 61 | 25 | 12 | 16 | 8 | 0.676 | 0.667 | 0.671 |
| second visit | 16 | 10 | 3 | 2 | 1 | 0.769 | 0.667 | 0.718 |
| single visit | 55 | 5 | 4 | 30 | 16 | 0.556 | 0.652 | 0.604 |

## Repeated-Person Error Summary

| repeat_error_status | person_count |
| --- | --- |
| both/all correct | 8 |
| mixed | 5 |
| both/all wrong | 1 |

### Repeated Persons With At Least One Error

| person_id | n_visits | subjects | true_labels | error_groups | n_errors | mean_prob | repeat_error_status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| pre-LBD-5 | 2 | pre-LBD-5, pre-LBD2-5 | preDLB, preDLB | FN, FN | 2 | 0.351 | both/all wrong |
| pre-LBD-2 | 2 | pre-LBD-2, pre-LBD2-2 | preDLB, preDLB | FN, TP | 1 | 0.646 | mixed |
| pre-LBD-49 | 2 | pre-LBD-49, pre-LBD2-49 | preDLB, preDLB | FN, TP | 1 | 0.506 | mixed |
| pre-LBD-53 | 2 | pre-LBD-53, pre-LBD2-53 | HC, HC | FP, TN | 1 | 0.469 | mixed |
| pre-LBD-68 | 2 | pre-LBD-68, pre-LBD2-68 | HC, HC | TN, FP | 1 | 0.343 | mixed |
| pre-LBD-92 | 2 | pre-LBD-92, pre-LBD2-92 | preDLB, preDLB | TP, FN | 1 | 0.557 | mixed |

## Probability Distribution By Error Group

| Group | n | p10 | median | p90 | near threshold n | moderate n | confident n |
| --- | --- | --- | --- | --- | --- | --- | --- |
| TP | 40 | 0.630 | 0.933 | 0.993 | 4 | 6 | 30 |
| FN | 19 | 0.011 | 0.318 | 0.427 | 5 | 6 | 8 |
| TN | 48 | 0.011 | 0.098 | 0.384 | 4 | 9 | 35 |
| FP | 25 | 0.551 | 0.718 | 0.886 | 4 | 13 | 8 |

## Most Confident Errors

| #Subject | source_prefix | visit_label | true_label | pred_label | error_group | pred_probability_positive | probability_margin_from_0_5 | age | sex_label | education | rbdq | updrs | mfs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pre-LBD-91 | pre-LBD | first visit | preDLB | HC | FN | 0.001 | 0.499 | 60 | M | 19.000 | 6.000 | 7.000 | 1.000 |
| COBEN-2129 | COBEN | single visit | preDLB | HC | FN | 0.001 | 0.499 | 68 | M | 18.000 | 3.000 | 4.000 | 1.000 |
| HC-17 | HC | first visit | HC | preDLB | FP | 0.995 | 0.495 | 76 | F | 14.000 | 3.000 | 0.000 | 0.000 |
| COBEN-1889 | COBEN | single visit | preDLB | HC | FN | 0.013 | 0.487 | 67 | F | 18.000 | 2.000 | 4.000 | 0.000 |
| pre-LBD-28 | pre-LBD | first visit | preDLB | HC | FN | 0.035 | 0.465 | 68 | M | 12.000 | 3.000 | 8.000 | 1.000 |
| COBEN-822 | COBEN | single visit | HC | preDLB | FP | 0.938 | 0.438 | 62 | F | 13.000 | 3.000 | 0.000 | 0.000 |
| HC-29 | HC | first visit | HC | preDLB | FP | 0.895 | 0.395 | 76 | F | 17.000 | 0.000 | 1.000 | 0.000 |
| pre-LBD2-92 | pre-LBD2 | second visit | preDLB | HC | FN | 0.119 | 0.381 | 77 | M | 13.000 | 6.000 | 3.000 | 1.000 |
| COBEN-45 | COBEN | single visit | HC | preDLB | FP | 0.873 | 0.373 | 75 | M |  | 1.000 | 2.000 | 1.000 |
| COBEN-1609 | COBEN | single visit | HC | preDLB | FP | 0.857 | 0.357 | 74 | M | 12.000 | 0.000 | 0.000 | 0.000 |
| pre-LBD-110 | pre-LBD | first visit | preDLB | HC | FN | 0.145 | 0.355 | 76 | F | 17.000 | 1.000 | 6.000 | 0.000 |
| HC-7 | HC | first visit | HC | preDLB | FP | 0.855 | 0.355 | 83 | F | 20.000 | 4.000 |  | 0.000 |
| COBEN-111 | COBEN | single visit | HC | preDLB | FP | 0.821 | 0.321 | 61 | F | 13.000 | 1.000 | 0.000 | 0.000 |
| COBEN-277 | COBEN | single visit | HC | preDLB | FP | 0.810 | 0.310 | 63 | M | 12.000 | 3.000 | 0.000 | 0.000 |
| pre-LBD-71 | pre-LBD | first visit | preDLB | HC | FN | 0.192 | 0.308 | 71 | M | 17.000 | 8.000 | 5.000 | 0.000 |
| pre-LBD-112 | pre-LBD | first visit | preDLB | HC | FN | 0.193 | 0.307 | 87 | F | 13.000 | 3.000 | 25.000 | 0.000 |
| pre-LBD-53 | pre-LBD | first visit | HC | preDLB | FP | 0.795 | 0.295 | 61 | M | 12.000 | 0.000 | 0.000 | 0.000 |
| HC-21 | HC | first visit | HC | preDLB | FP | 0.795 | 0.295 | 76 | F | 17.000 | 2.000 | 2.000 | 0.000 |
| COBEN-1705 | COBEN | single visit | HC | preDLB | FP | 0.792 | 0.292 | 62 | F | 13.000 | 2.000 | 1.000 | 0.000 |
| pre-LBD2-5 | pre-LBD2 | second visit | preDLB | HC | FN | 0.242 | 0.258 | 76 | M | 17.000 | 4.000 | 11.000 | 0.000 |

## HC False Positives With Clinical Scores

| #Subject | source_prefix | visit_label | pred_probability_positive | age | sex_label | education | rbdq | updrs | mfs | visuospatial | attention | executive |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| HC-17 | HC | first visit | 0.995 | 76 | F | 14.000 | 3.000 | 0.000 | 0.000 | 0.583 | 0.333 | -0.333 |
| COBEN-822 | COBEN | single visit | 0.938 | 62 | F | 13.000 | 3.000 | 0.000 | 0.000 | -0.719 | 0.840 | -0.093 |
| HC-29 | HC | first visit | 0.895 | 76 | F | 17.000 | 0.000 | 1.000 | 0.000 | 1.080 | 0.833 | 0.333 |
| COBEN-45 | COBEN | single visit | 0.873 | 75 | M |  | 1.000 | 2.000 | 1.000 | 0.458 | -0.020 | -0.033 |
| COBEN-1609 | COBEN | single visit | 0.857 | 74 | M | 12.000 | 0.000 | 0.000 | 0.000 | 0.583 | 1.000 | 0.111 |
| HC-7 | HC | first visit | 0.855 | 83 | F | 20.000 | 4.000 |  | 0.000 | 1.080 | 1.333 | 1.111 |
| COBEN-111 | COBEN | single visit | 0.821 | 61 | F | 13.000 | 1.000 | 0.000 | 0.000 | -0.253 | 0.333 | 0.111 |
| COBEN-277 | COBEN | single visit | 0.810 | 63 | M | 12.000 | 3.000 | 0.000 | 0.000 | 0.707 | 0.187 | 0.007 |
| pre-LBD-53 | pre-LBD | first visit | 0.795 | 61 | M | 12.000 | 0.000 | 0.000 | 0.000 | 1.080 | 0.333 | 1.833 |
| HC-21 | HC | first visit | 0.795 | 76 | F | 17.000 | 2.000 | 2.000 | 0.000 | 0.583 | 0.333 | 0.500 |
| COBEN-1705 | COBEN | single visit | 0.792 | 62 | F | 13.000 | 2.000 | 1.000 | 0.000 | -0.042 | 0.907 | 0.993 |
| pre-LBD-90 | pre-LBD | first visit | 0.756 | 78 | M | 18.000 | 0.000 | 0.000 | 0.000 | 0.151 | 0.167 | -0.167 |
| COBEN-1797 | COBEN | single visit | 0.718 | 64 | M | 18.000 | 0.000 | 2.000 | 2.000 | 0.583 | -1.000 | 0.540 |
| HC-13 | HC | first visit | 0.718 | 58 | F | 12.000 | 0.000 | 0.000 | 0.000 | -0.253 | -0.667 | -0.167 |
| COBEN-803 | COBEN | single visit | 0.718 | 74 | M | 18.000 | 3.000 | 2.000 | 0.000 | 0.583 | 0.500 | -0.056 |

## preDLB False Negatives With Clinical Scores

| #Subject | source_prefix | visit_label | pred_probability_positive | age | sex_label | education | rbdq | updrs | mfs | visuospatial | attention | executive |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pre-LBD-91 | pre-LBD | first visit | 0.001 | 60 | M | 19.000 | 6.000 | 7.000 | 1.000 | 0.583 | 1.000 | 1.000 |
| COBEN-2129 | COBEN | single visit | 0.001 | 68 | M | 18.000 | 3.000 | 4.000 | 1.000 | 0.458 | 0.400 | -0.467 |
| COBEN-1889 | COBEN | single visit | 0.013 | 67 | F | 18.000 | 2.000 | 4.000 | 0.000 | -0.170 | -0.460 | -0.267 |
| pre-LBD-28 | pre-LBD | first visit | 0.035 | 68 | M | 12.000 | 3.000 | 8.000 | 1.000 | 0.151 | 0.000 | 0.111 |
| pre-LBD2-92 | pre-LBD2 | second visit | 0.119 | 77 | M | 13.000 | 6.000 | 3.000 | 1.000 | 0.580 | 1.170 | 0.277 |
| pre-LBD-110 | pre-LBD | first visit | 0.145 | 76 | F | 17.000 | 1.000 | 6.000 | 0.000 | -1.341 | 0.833 | -0.444 |
| pre-LBD-71 | pre-LBD | first visit | 0.192 | 71 | M | 17.000 | 8.000 | 5.000 | 0.000 | -0.772 | -1.167 | -1.278 |
| pre-LBD-112 | pre-LBD | first visit | 0.193 | 87 | F | 13.000 | 3.000 | 25.000 | 0.000 |  | -2.000 | -1.000 |
| pre-LBD2-5 | pre-LBD2 | second visit | 0.242 | 76 | M | 17.000 | 4.000 | 11.000 | 0.000 | 0.150 | -1.165 | -0.390 |
| pre-LBD-44 | pre-LBD | first visit | 0.318 | 69 | M | 14.000 | 4.000 | 1.000 | 1.000 | -0.772 | 0.000 | -1.000 |
| COBEN-2488 | COBEN | single visit | 0.331 | 71 | F | 13.000 | 5.000 | 13.000 | 0.000 | 0.373 | -0.647 | -0.287 |
| pre-LBD-49 | pre-LBD | first visit | 0.338 | 67 | M | 13.000 | 5.000 | 1.000 | 0.000 | 1.080 | 0.333 | -0.611 |
| pre-LBD-51 | pre-LBD | first visit | 0.348 | 69 | M | 12.000 | 1.000 | 6.000 | 1.000 | 0.151 | -0.500 | -0.667 |
| HC2-19 | HC2 | second visit | 0.389 | 78 | M | 12.000 | 5.000 | 4.000 | 0.000 | 1.080 | 0.000 | -0.057 |
| pre-LBD-2 | pre-LBD | first visit | 0.416 | 76 | F | 13.000 | 2.000 | 7.000 | 0.000 | 0.583 | -0.333 | -1.389 |

## Interpretation

- The strict model is not failing only because many predictions are near 0.5. Several errors are confident, especially some HC false positives and preDLB false negatives.
- Errors are not evenly distributed by source prefix. The source-prefix table should be treated as a cohort-confounding warning because diagnosis and recruitment source are coupled.
- Repeated visits are a visible issue: if a person has two visits, the model can classify one visit correctly and the other incorrectly, meaning visit-level validation is still optimistic relative to person-level validation.
- HC false positives do not look globally more clinically affected than true negatives by median RBDq/UPDRS/MFS: both groups have median RBDq `2`, UPDRS `1`, and MFS `0`. Individual false positives with higher RBDq still deserve manual review, but the group-level profile does not prove they are prodromal-like HC cases.
- preDLB false negatives are not simply clinically mild by RBDq or UPDRS. Their median RBDq is the same as true positives (`4`) and median UPDRS is slightly higher (`6` vs `5`). The model is therefore missing some clinically affected preDLB subjects, which points more toward feature instability, source effects, visit effects, or insufficient sample size than toward only mild disease.
