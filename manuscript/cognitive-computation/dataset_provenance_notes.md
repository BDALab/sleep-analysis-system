# Dataset provenance notes

Checked 16 August 2026. This is a working provenance audit for the manuscript,
not a substitute for confirmation by the investigators or the ethics office.

## Best-supported attribution of the local `COBEN-*` collection

The best-supported attribution is the **National Institute for Neurological
Research (NINR; NPO-NEURO-D), project LX22NPO5107, WP2.1**, particularly
sub-objective WP2.1/2 concerning cardiovascular risk factors, cognitive decline,
and identification of people at risk of degenerative dementia in a Kardiovize
population-based recruitment pool aged 60 years or older.

This should be described as a **closed NINR cognitive-risk follow-up substudy
using Kardiovize participants and legacy identifiers**, not as the original
Kardiovize Brno 2030 baseline dataset. `COBEN` appears to be an internal database
prefix and should not, by itself, be expanded to the older CoBeN grant name.

### Evidence from the local project

- `KARDIOVIZE.xlsx` was created on 28 February 2023; its comments identify Luboš
  Brabenec as an author/contributor.
- Its neurological/cognitive sheet classifies older participants as HC, at risk,
  MCI, or MCI-LB using cognition and core Lewy-body features.
- Its `FNUSA` sheet contains legacy cardiovascular variables and identifiers;
  this explains the KARDIOVIZE label without establishing that the new multimodal
  collection is the original cardiovascular cohort dataset.
- The workbook contains modality placeholders for VUT acoustics, handwriting,
  and actigraphy, consistent with a CEITEC--FNUSA--VUT collaboration.
- Local analysis code maps the numeric workbook identifiers to `COBEN-<ID>`.
- The 55 analysed `COBEN-*` recordings date from 8 March 2023 to 20 February
  2025, overlapping NINR (July 2022--December 2025).

### External evidence

- The [NINR 2023 scientific report](https://ta-service.cz/ninr2023/downloads/Scientific_Report_NINR_2023.pdf)
  names Irena Rektorová as WP2.1 lead. WP2.1/2 explicitly concerns a Kardiovize
  population-based cohort aged 60 years or older and risk of degenerative
  dementia; WP2.1/5, led by Jiří Mekyska, develops speech/voice and
  handwriting/drawing biomarkers for prodromal DLB.
- The same report describes Brno participation by CEITEC MU, St. Anne's
  University Hospital, and its International Clinical Research Center.
- The [Masaryk University project record](https://www.muni.cz/en/research/projects/67338)
  identifies project LX22NPO5107 and its July 2022--December 2025 period.
- The [VUT BDALab project page](https://bdalab.utko.fekt.vut.cz/) lists
  LX22NPO5107 among its CEITEC/FNUSA collaborations and explicitly lists sleep
  actigraphy and prodromal-LBD diagnosis from actigraphy among its research
  topics.

### Alternatives considered

| Candidate | Why it initially looked plausible | Why it is less likely |
|---|---|---|
| CoBeN, grant 734718 | Exact match to the internal `COBEN` prefix; involved Rektorová, Mekyska, and Brabenec | Formal project period ended in 2021/2022 and its documented focus was chiefly speech, handwriting, visual processing, and behavioral neurology across languages. The local 2023--2025 collection and Kardiovize 60+ clinical structure match NINR more directly. |
| GF21-13462L | Healthy seniors/MCI, 2021--2024, Rektorová and Brabenec | It was a brain-stimulation/working-memory trial; the public team record does not include Mekyska, and no Kardiovize recruitment link was found. |
| NU20-04-00294 | Multimodal prodromal-LBD project with actigraphy and the same collaborators | This is already the other closed clinical collection in the manuscript, represented locally by `HC/HC2` and `pre-LBD/pre-LBD2`; the `COBEN-*` collection has different files, dates, and ascertainment. |
| Original Kardiovize Brno 2030 cohort | Workbook name, legacy identifiers, and cardiovascular variables | The local data add neurological, cognitive, actigraphy, and risk-classification assessments collected later under a different research objective. It is a Kardiovize-derived follow-up/substudy, not the original baseline dataset. |

## Relevant CEITEC--FNUSA--VUT dataset/cohort families

This is a focused list of datasets and project cohorts connected to Jiří
Mekyska, Irena Rektorová, and/or Luboš Brabenec and relevant to the manuscript.
It is not a claim that all three investigators collected every dataset.

| Dataset or project cohort | Main content | Access | Relationship to this study |
|---|---|---|---|
| PaHaW | Online handwriting from 37 people with Parkinson's disease and 38 matched controls | Available under a licence agreement from VUT BDALab | Earlier public/controlled VUT--FNUSA handwriting resource; not the local `COBEN-*` actigraphy cohort |
| CoBeN (734718) | Multilingual speech, handwriting, imaging, and behavioral-neurology cohorts involving PD, stroke, dementia, and controls | Mostly controlled/project datasets; some derived publications and resources are public | Explains a historical collaboration and possibly the internal naming convention, but is not the best match for the 2023--2025 collection |
| NV16-30805A / 16-30805A | PD hypokinetic-dysarthria, micrographia, brain-plasticity, and rTMS data | Controlled clinical data | Source of several joint speech/handwriting publications; not the current older cognitive-risk cohort |
| NU20-04-00294 | Longitudinal multimodal prodromal-LBD cohort: clinical assessment, EEG, MRI, acoustics, handwriting, actigraphy, and related biomarkers | Closed/controlled | The manuscript's `HC/HC2` and `pre-LBD/pre-LBD2` clinical collection |
| NINR LX22NPO5107, WP2.1/1 | Longitudinal multimodal LBD-risk, LBD, AD, and HC data | Closed/controlled | Broader NINR neurodegeneration program overlapping the study's diagnostic topic |
| NINR LX22NPO5107, WP2.1/2 | Neurological/cognitive and biomarker follow-up of a Kardiovize recruitment pool aged 60+ to study cardiovascular risk and risk of AD/DLB | Closed/controlled | **Best match for the local `COBEN-*` collection** |
| NINR LX22NPO5107, WP2.1/5 | Speech/voice and handwriting/drawing biomarkers for prodromal DLB, led by Mekyska | Closed/controlled | Explains the VUT modality placeholders and cross-institutional analysis plan |
| GF21-13462L | EEG/MRI and non-invasive stimulation in healthy older adults and people with MCI | Closed; trial record says individual data are not publicly shared | Relevant neighboring MCI cohort, but a weaker provenance match |
| NU22J-04-00074 | Home-based NIBS plus LSVT for PD hypokinetic dysarthria | Closed clinical trial data | Later joint speech cohort, not the current cohort |
| NU23J-04-00005 | Sentence comprehension and NIBS in Lewy-body diseases | Closed clinical trial data | Later joint language/LBD cohort, not the current cohort |

## Confirmation still needed from the supervisors

1. Is `COBEN-*` the NINR LX22NPO5107 WP2.1/2 cohort, and is there an internal
   protocol/study name preferred over “NINR cognitive-risk follow-up”?
2. Were all subjects recruited from Kardiovize Parental 65+, the main follow-up,
   or a mixture of Kardiovize phases?
3. Which ethics committee approval and protocol number cover the actigraphy
   visit and this secondary analysis?
4. Was actigraphy funded only by LX22NPO5107, or also by another grant?
5. Should the manuscript retain `COBEN` only as a reproducibility label, or omit
   it from the public cohort name?
