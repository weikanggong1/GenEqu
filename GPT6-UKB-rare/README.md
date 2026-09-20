# GPT6-UKB-rare

Endpoint-specific risk weights couple Gaussian factors to case status. Frozen marginal quantiles and categorical rules map factors to measurements. Case proportions are prescribed by the rules. Aggregate schema summaries are part of this generator; they are not participant records.

## Run

From the repository root:

```bash
python GPT6-UKB-rare/generate.py --n 1000 --seed 20260831 --output-dir generated/GPT6-UKB-rare
python GPT6-UKB-rare/generate.py --help
```

Add `--diseases CODE1 CODE2` to select endpoints from the table below. Omit it to generate all 15. `--n` specifies rows **per endpoint**. Repeating the command with the same inputs, seed, and dependency versions reproduces the measurements and labels; runtime timestamps in audit files can differ. Output paths must be new.

## Files and interpretation

- `allowed_schema.csv`: ordered 201-measurement interface, units, category codes, and range or missingness constraints. Empty CSV cells represent missing measurements.
- `disease_identities.csv`: endpoint identifiers and definitions.
- `generate_gpt6_rare_v1.py`: frozen sampling engine; inspect this file for equations and random sampling rules.
- `schema_provenance.json`: schema origin and release hash metadata.
- `generate.py`: convenience entry point resolving bundled metadata relative to the script, so generation does not depend on the current working directory.

Each disease directory contains `synthetic_<CODE>_<N>.csv` with 201 predictors followed by the binary outcome named `<CODE>`. Codes in the first column of the table below are also the values accepted by `--diseases`. Output audits are supplementary metadata; use the CSV to train a classifier.

## Endpoints

| Code | Endpoint |
|---|---|
| E4_HYPERPARA | Hyperparathyroidism |
| F5_PHOBANX | Phobic anxiety disorders |
| K11_COELIAC | Coeliac disease |
| F5_ALCOHOL_DEPENDENCE | Alcohol dependence |
| I9_ABAORTANEUR | Abdominal aortic aneurysm (AAA) |
| K11_ACUTPANC | Acute pancreatitis |
| F5_ALZHDEMENT | Dementia in Alzheimer disease |
| K11_ULCER | Ulcerative colitis |
| I9_CARDMYO | Cardiomyopathy |
| C3_NONHODGKIN_EXALLC | Non-Hodgkin lymphoma, all, excluding all cancers (controls excluding all cancers) |
| M13_POLYMYALGIA | Polymyalgia rheumatica |
| C3_MELANOMA_SKIN_EXALLC | Malignant melanoma of skin, excluding all cancers (controls excluding all cancers) |
| N14_ENDOMETRIOSIS | Endometriosis |
| C3_BLADDER_EXALLC | Malignant neoplasm of bladder, excluding all cancers (controls excluding all cancers) |
| M13_FIBROMYALGIA | Fibromyalgia |
