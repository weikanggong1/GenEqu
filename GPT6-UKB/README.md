# GPT6-UKB

Shared physiological factors generate the measurement panel and enter endpoint-specific 15-year event equations. The event probability is 1 − exp[−0.004 × 15 × exp(clip(η,−12,12))], where η combines the endpoint's factor weights, applicable subtype effects and independently sampled susceptibility/progression terms. The annual hazard prior of 0.004 is an authored modeling assumption. Numeric measurement equations, category probabilities, missingness, rounding, and range constraints are explicit in the engine and `rulebook.py`. Each endpoint represents a separately generated population; it is not a joint 30-label dataset.

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

The [label-free generator](label_free/README.md) constructs 201 baseline measurements for representation learning. Its dictionary, shared physiological factors, algebraic identities, categorical rules and missingness assumptions are explicit; it generates no disease or prediction targets.

    python GPT6-UKB/label_free/generate.py --n 1000 --seed 20260831 --output-dir generated/label_free

The output contains synthetic_features_float32.npy, feature_columns.json and generation_parameters.json. Rows follow the dictionary's exact Field-ID order. All numeric priors and dependence mechanisms are authored assumptions. See [label_free/core/MODEL_SPEC.md](label_free/core/MODEL_SPEC.md) and the 201-row [measurement rule table](label_free/core/measurement_rules.csv).
