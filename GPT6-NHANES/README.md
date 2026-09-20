# GPT6-NHANES

Explicit demographic and physiological factors generate measurements, disease histories, diagnosis reporting, and treatment-related measurement shifts. A reported previous diagnosis can coexist with improved current measurements. These are cross-sectional diagnosis-history targets, not future-event labels.

## Run

From the repository root:

```bash
python GPT6-NHANES/generate.py --n 1000 --seed 20260831 --output-dir generated/GPT6-NHANES
python GPT6-NHANES/generate.py --help
```

Add `--diseases CODE1 CODE2` to select endpoints from the table below. Omit it to generate all 14. `--n` specifies rows **per endpoint**. Repeating the command with the same inputs, seed, and dependency versions reproduces the measurements and labels; runtime timestamps in audit files can differ. Output paths must be new.

## Files and interpretation

- `allowed_schema.csv`: ordered 52-measurement interface, units, category codes, and range or missingness constraints. Empty CSV cells represent missing measurements.
- `disease_identities.csv`: endpoint identifiers and definitions.
- `generate_nhanes14.py`: frozen sampling engine; inspect this file for equations and random sampling rules.
- `generate.py`: convenience entry point resolving bundled metadata relative to the script, so generation does not depend on the current working directory.

Each disease directory contains `synthetic_<CODE>_<N>.csv` with 52 predictors followed by the binary outcome named `<CODE>`. Codes in the first column of the table below are also the values accepted by `--diseases`. Output audits are supplementary metadata; use the CSV to train a classifier.

## Endpoints

| Code | Endpoint |
|---|---|
| I9_HYPTENSESS | Reported hypertension |
| E4_HYPERCHOL | Reported high cholesterol |
| E4_DM2 | Reported diagnosed diabetes, type unspecified |
| N14_CHRONKIDNEYDIS | Reported weak or failing kidneys |
| I9_ISCHHEART | Reported coronary heart disease, angina or myocardial infarction |
| I9_HEARTFAIL | Reported congestive heart failure |
| K11_LIVER | Reported liver condition |
| M13_OSTEOPOROSIS | Reported osteoporosis |
| N14_PROSTHYPERPLA | Reported benign prostatic enlargement |
| C3_PROSTATE_EXALLC | Reported prostate cancer |
| J10_ASTHMA_MAIN_EXMORE | Reported asthma |
| I9_STR_SAH | Reported stroke |
| M13_ARTHROSIS | Reported osteoarthritis |
| M13_RHEUMA | Reported rheumatoid arthritis |
