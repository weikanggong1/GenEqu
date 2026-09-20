# GPT6-KNHANES

Shared factors, diagnosis reporting, and treatment assumptions generate a Korean survey-style population. The target age range is 40–70 years. Endpoint definitions and their relationship to UKB definitions are given in `disease_identities.csv`; dyslipidaemia, for example, is broader than hypercholesterolaemia.

## Run

From the repository root:

```bash
python GPT6-KNHANES/generate.py --n 1000 --seed 20260831 --output-dir generated/GPT6-KNHANES
python GPT6-KNHANES/generate.py --help
```

Add `--diseases CODE1 CODE2` to select endpoints from the table below. Omit it to generate all 10. `--n` specifies rows **per endpoint**. Repeating the command with the same inputs, seed, and dependency versions reproduces the measurements and labels; runtime timestamps in audit files can differ. Output paths must be new.

## Files and interpretation

- `allowed_schema.csv`: ordered 44-measurement interface, units, category codes, and range or missingness constraints. Empty CSV cells represent missing measurements.
- `disease_identities.csv`: endpoint identifiers and definitions.
- `generate_knhanes.py`: frozen sampling engine; inspect this file for equations and random sampling rules.
- `schema_provenance.json`: schema origin and release hash metadata.
- `generate.py`: convenience entry point resolving bundled metadata relative to the script, so generation does not depend on the current working directory.

Each disease directory contains `synthetic_<CODE>_<N>.csv` with 44 predictors followed by the binary outcome named `<CODE>`. Codes in the first column of the table below are also the values accepted by `--diseases`. Output audits are supplementary metadata; use the CSV to train a classifier.

## Endpoints

| Code | Endpoint |
|---|---|
| I9_HYPTENSESS | Hypertension |
| E4_HYPERCHOL | Dyslipidaemia |
| E4_DM2 | Diabetes, type unspecified |
| N14_CHRONKIDNEYDIS | Kidney disease, chronicity unspecified |
| J10_ASTHMA_MAIN_EXMORE | Asthma |
| I9_STR_SAH | Stroke |
| I9_ISCHHEART | Myocardial infarction or angina |
| M13_OSTEOPOROSIS | Osteoporosis |
| G6_SLEEPAPNO | Obstructive sleep apnoea |
| COPD_MODE | Physician-diagnosed COPD |
