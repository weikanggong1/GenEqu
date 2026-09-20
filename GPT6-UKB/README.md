# GPT6-UKB

Shared physiological factors generate the measurement panel and enter endpoint-specific logistic risk equations. Numeric measurement equations, category probabilities, missingness, rounding, and range constraints are explicit in the engine and `rulebook.py`. Each endpoint represents a separately generated population; it is not a joint 30-label dataset.

## Run

From the repository root:

```bash
python GPT6-UKB/generate.py --n 1000 --seed 20260831 --output-dir generated/GPT6-UKB
python GPT6-UKB/generate.py --help
```

Add `--diseases CODE1 CODE2` to select endpoints from the table below. Omit it to generate all 30. `--n` specifies rows **per endpoint**. Repeating the command with the same inputs, seed, and dependency versions reproduces the measurements and labels; runtime timestamps in audit files can differ. Output paths must be new.

## Files and interpretation

- `allowed_schema.csv`: ordered 201-measurement interface, units, category codes, and range or missingness constraints. Empty CSV cells represent missing measurements.
- `disease_identities.csv`: endpoint identifiers and definitions.
- `generate_ukb_common30.py`: frozen sampling engine; inspect this file for equations and random sampling rules.
- `schema_provenance.json`: schema origin and release hash metadata.
- `generate.py`: convenience entry point resolving bundled metadata relative to the script, so generation does not depend on the current working directory.

Each disease directory contains `synthetic_<CODE>_<N>.csv` with 201 predictors followed by the binary outcome named `<CODE>`. Codes in the first column of the table below are also the values accepted by `--diseases`. Output audits are supplementary metadata; use the CSV to train a classifier.

## Endpoints

| Code | Endpoint |
|---|---|
| I9_HYPTENSESS | Hypertension, essential |
| M13_ARTHROSIS | Arthrosis |
| K11_DIVERTIC | Diverticular disease of intestine |
| E4_HYPERCHOL | Pure hypercholesterolaemia |
| N14_PROSTHYPERPLA | Hyperplasia of prostate |
| K11_REFLUX | Gastro-oesophageal reflux disease |
| I9_AF | Atrial fibrillation and flutter |
| J10_ASTHMA_MAIN_EXMORE | Asthma (only as main-diagnosis) (more control exclusions) |
| I9_ISCHHEART | Ischaemic heart diseases |
| H7_CATARACTSENILE | Senile cataract |
| E4_OBESITY | Obesity |
| E4_DM2 | Type 2 diabetes |
| C3_PROSTATE_EXALLC | Malignant neoplasm of prostate, excluding all cancers (controls excluding all cancers) |
| F5_DEPRESSIO | Depression |
| COPD_MODE | COPD (mode) |
| E4_HYTHYNAS | Hypothyroidism, other/unspecified |
| N14_CHRONKIDNEYDIS | Chronic kidney disease |
| F5_ANXIETY | Other anxiety disorders |
| D3_ANAEMIA_IRONDEF_NAS | Other and unspecified iron deficiency |
| M13_OSTEOPOROSIS | Osteoporosis |
| K11_CHOLELITH | Cholelithiasis |
| I9_HEARTFAIL | Heart failure, strict |
| K11_LIVER | Diseases of liver |
| K11_IBS | Irritable bowel syndrome |
| M13_GOUT | Gout |
| H7_GLAUCOMA | Glaucoma |
| I9_STR_SAH | Stroke, including SAH |
| M13_RHEUMA | Rheumatoid arthritis |
| G6_CARPTU | Carpal tunnel syndrome |
| G6_SLEEPAPNO | Sleep apnoea |

## Optional label-free population

The `label_free` directory supplies the frozen 201-variable background population generator used for representation-learning inputs. It returns measurements without disease targets. Its background engine is invoked in an inert, zero-disease mode; the legacy disease-specific interfaces in that dependency are not used here.

```bash
python GPT6-UKB/label_free/generate_label_free_population_v4.py --n 1000 --audit-rows 500 --seed 20260831 --output-dir generated/label_free
```

Use `--write-csv` to additionally write CSV output. This mode uses the aggregate marginal constraints in its bundled schema. Its English translation preserves the original category-selection rule and numerical values.
