# Separate activity summaries during predicted sleep and wake

This is a new exploratory extraction, not a replacement for the manuscript's
canonical analyses. It reads existing cached sleep/wake predictions and the
existing minute-level activity index. It does not train models, regenerate
signals, write to the database, or change the original exports.

## Run

From `/Volumes/Portable/geneactiv-processing-data`, in an environment with the
project requirements installed:

```bash
python3 -m unittest dashboard.test_state_activity_features -v
python3 manage.py extract_state_activity --check-inputs
python3 manage.py extract_state_activity --limit-recordings 2
python3 manage.py extract_state_activity
```

The first extraction is a small smoke test. The second processes all eligible
HC (diagnosis code 0) and MCI-LB (code 3) visits, using left-wrist recordings.
Training and DREAMT recordings are always excluded. Each run gets a new directory
under `media/state-activity/`; its path and usable-visit counts are printed.
Do not interpret smoke-test visit counts as a complete cohort.

An optional mixed-window sensitivity run is separate:

```bash
python3 manage.py extract_state_activity --mixed-policy majority
```

Other options:

```bash
python3 manage.py extract_state_activity --help
python3 manage.py extract_state_activity --subject HC-13
python3 manage.py extract_state_activity --diagnosis-codes 0 2 3
python3 manage.py extract_state_activity --min-state-minutes 5 --min-nights 5
```

An explicit `--output-dir` must name a directory that does not already exist.
Missing/corrupt inputs are never automatically regenerated. A partial extraction
saves its audit trail but exits with an error. Review `manifest.json`, rather
than treating the existence of CSVs as proof of a successful run.

If a recording is intentionally excluded after reviewing missing inputs, use
`--exclude-recording ID` (repeat for multiple IDs). Those exclusions are recorded
in the manifest and change the extraction cohort; they are never automatic.

## Definitions

- Activity-index timestamps mark the start of nonoverlapping 60-second windows.
- Prediction timestamps mark the centre of 30-second epochs. Probabilities
  greater than or equal to 0.5 mean predicted sleep, matching the existing code.
- Naive cached timestamps are interpreted as UTC, not local wall-clock time.
- The nightly interval runs from the beginning of the first sleep epoch to the
  end of the last sleep epoch. It excludes sleep latency, final time awake in
  bed, and daytime wake. Only complete activity windows within it are used.
- Prediction epochs are matched by exact duration of overlap, not nearest
  timestamps. Missing epochs are never interpolated or forward-filled.
- The default requires all 60 seconds to have the same predicted state. Minutes
  containing both states are counted as mixed and excluded from state features.
- Under `majority`, fully covered mixed minutes can be assigned to the state
  covering strictly more than 30 seconds. Ties remain mixed. Because the grids
  can be offset, actual overlap durations matter. A minute is never used in both
  states.
- Nonfinite/negative activity is invalid. Zero is a valid activity value.
  Invalid probabilities, duplicate timestamps, and overlapping source windows
  fail that recording rather than silently selecting a label.

## Features and minimum support

Each state has four nightly features: mean, median, 90th percentile, and IQR of
the activity-index values. A state needs at least five assigned full minutes per
night by default. Below this, all four features for that state are missing.
At least 90% of the entire nightly interval must also be covered by valid
activity windows with complete predictions; otherwise both states are missing.
Mixed minutes count toward coverage, but not toward the pure-state summaries.
The coverage threshold can be set with `--min-night-coverage` and is recorded.

Each database `Subject` is treated as a visit. Repeated visits remain separate.
For each of the eight nightly features, retain the median and IQR across nights:
16 visit features named, for example, `Median.activity_sleep.mean` and
`IQR.activity_wake.p90`. Each state separately needs five usable nights by
default. Thus a visit may have sleep features but no wake features. Missing
features are not imputed; excluded visits remain in the output for auditing.
Duplicate subject/date nights are rejected to avoid counting a night twice.

The coverage, five-minute, and five-night settings are explicit exploratory support rules,
not validated biological cutoffs. Fix them before examining associations and
record any planned sensitivity settings; do not lower them until significance
appears. Nightly coverage and per-state usable-night counts must be inspected
before deciding whether the wake summaries are sufficiently supported.

## Files

- `nightly_state_activity.csv`: eight nightly features, state/mixed/missing
  minute counts, prediction coverage, processing status, clinical metadata,
  recording/night identifiers, and the existing SleepPy broken flag. Whole
  interval distribution summaries are included as diagnostic references; they
  use complete minute windows and do not exactly reproduce the original
  endpoint conventions or replace canonical features.
- `visit_state_activity.csv`: the 16 summaries plus state-valid-night counts,
  diagnosis **codes**, age, sex (F=1, M=2, matching existing preparation),
  education, RBDq, and the visit code. No covariate adjustment is applied here.
- `manifest.json`: thresholds, filters, definitions, runtime versions, summary,
  errors, implementation fingerprints, and SHA-256 fingerprints of all loaded
  cache files.

## Interpretation and next analysis

These are **predicted** sleep/wake states, not PSG-confirmed wake, REM, or NREM.
Movement affects the sleep/wake classifier itself, so the split is conditional
on an imperfect movement-derived label. Keep the original whole-interval
results alongside it. No additional nonwear mask is applied by this command;
finite low activity alone cannot establish wear. The existing SleepPy broken
flag is exported but does not automatically exclude a night.

Inspect coverage and errors first. Then specify the small RBDq association
analysis and its FDR family before running it, retaining the current covariate,
person-clustering, and source/age sensitivity conventions. This command does **not** calculate correlations, fit GEE
models, or rerun classification. The CSV
visit code is not an independent-person ID; use the established person mapping
when modeling repeated visits. Do not send this dataset through the canonical
screening command, which would create a different multiple-testing problem.

## Environment

Python, Django, NumPy, pandas, and PyTables (`tables`) are required. Use the
project's existing working environment. `MACOS_SETUP.md` describes installation;
keep virtual environments on the internal disk. If the older Python 3.13
environment no longer launches after a Homebrew update, repair/recreate it
before activating it. A broken activation can silently leave another Python on
the PATH. Check `python3 --version` and `python3 -c 'import django, pandas, tables'`.

If you need a fresh environment only for this extraction, the following keeps
it on the internal disk and avoids installing the full ML stack. It uses the
Python 3.11 executable currently available on this Mac:

```bash
/opt/homebrew/bin/python3.11 -m venv /Users/marekmikulec/.virtualenvs/geneactiv-state-activity-py311
source /Users/marekmikulec/.virtualenvs/geneactiv-state-activity-py311/bin/activate
python3 -m pip install 'Django>=5.2,<6.1' 'numpy>=2,<3' 'pandas>=2.2,<3.1' 'tables>=3.10,<4' 'django-extensions>=4.1,<5' 'django-nested-inline>=0.4.6,<0.5' pytz
```

Do not recreate an existing environment if it contains unrelated work. These
instructions are not executed automatically by the extraction command.
