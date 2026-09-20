# GenEqu

**Biomedical knowledge expressed as executable population models.**

GenEqu contains six GPT-6-authored generator families used in the disease-learning study. They sample measurements and disease labels through explicit equations, probability distributions, shared factors, and frozen parameter tables. Sampling runs locally: no LLM API, neural generator, API key, GPU, or fitted model is required.

| Generator | Measurements | Outcomes |
|---|---|---|
| [GPT6-UKB](GPT6-UKB/README.md) | 201 UK Biobank variables | 30 selected disease endpoints |
| [GPT6-UKB-rare](GPT6-UKB-rare/README.md) | The same 201-variable panel | 15 additional lower-frequency endpoints |
| [GPT6-NHANES](GPT6-NHANES/README.md) | 52 NHANES measurements | 14 reported diagnosis endpoints |
| [GPT6-KNHANES](GPT6-KNHANES/README.md) | 44 harmonized measurements | 10 reported diagnosis endpoints |
| [GPT6-Brain](GPT6-Brain/README.md) | 15 FIRST regional brain volumes | AD versus NC; NC/MCI/AD; SCZ versus HC |
| [GPT6-protein](GPT6-protein/README.md) | 648 protein measurements | Six binary diagnostic-status tasks |

UKB endpoint events, NHANES diagnosis histories, and KNHANES diagnosis histories have different ascertainment. Matching endpoint codes do not make their definitions identical. Each clinical directory includes the complete measurement schema and endpoint definitions. The `rare` name identifies the frozen 15-endpoint generator; it does not assert a uniform clinical rarity threshold.

## Installation

Use Python 3.10 or later. The release is tested with Python 3.10 and the versions in `requirements.txt`.

```bash
git clone https://github.com/weikanggong1/GenEqu.git
cd GenEqu
python -m venv .venv
# Linux / macOS:
source .venv/bin/activate
# Windows PowerShell instead:
# .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Run commands below from the repository root. Use a new output path for each run; the generators protect completed outputs from accidental replacement. All paths and counts are user-configurable.

## Generate data

The four clinical commands generate **one population per endpoint**. `--n` is the number of rows per endpoint, not the combined total. Without `--diseases`, all endpoints in that family are generated.

```bash
python GPT6-UKB/generate.py --n 1000 --seed 20260831 --output-dir generated/ukb
python GPT6-UKB-rare/generate.py --n 1000 --seed 20260831 --output-dir generated/ukb_rare
python GPT6-NHANES/generate.py --n 1000 --seed 20260831 --output-dir generated/nhanes
python GPT6-KNHANES/generate.py --n 1000 --seed 20260831 --output-dir generated/knhanes

python GPT6-Brain/generate.py --task ad-binary --n-per-class 1000 --output generated/ad_binary.npz
python GPT6-Brain/generate.py --task ad-multiclass --n-per-class 1000 --output generated/ad_multiclass.npz
python GPT6-Brain/generate.py --task scz --n-per-class 1000 --output generated/scz.npz
python GPT6-protein/generate.py --n 6000 --output generated/protein.npz
```

For example, generate only type 2 diabetes:

```bash
python GPT6-UKB/generate.py --diseases E4_DM2 --n 10000 --seed 42 --output-dir generated/ukb_diabetes
```

Use `python <generator-directory>/generate.py --help` for all options. Clinical defaults produce 360,000 rows **per endpoint**; choose a smaller explicit `--n` for a quick trial. The rare generator needs enough rows to represent its prescribed case proportion; 1,000 rows works for all 15 endpoints.

## Read the output

Clinical CSV files contain predictors followed by a binary outcome column. Find the CSV files recursively beneath the chosen output directory; adjacent audit files describe seeds, schemas, and generation checks. Disease-specific files are separate simulated populations. Do not join rows from separate disease files and treat the resulting labels as a jointly observed 45-disease record.

```python
from pathlib import Path
import pandas as pd
import numpy as np

path = next(Path('generated/ukb_diabetes').rglob('synthetic_*.csv'))
cohort = pd.read_csv(path)
X, y = cohort.iloc[:, :-1], cohort.iloc[:, -1]

brain = np.load('generated/ad_binary.npz', allow_pickle=False)
X_brain, y_brain = brain['X'], brain['y']
print(brain['feature_names'], brain['class_names'])

protein = np.load('generated/protein.npz', allow_pickle=False)
X_protein, Y_protein = protein['X'], protein['Y']
print(protein['label_names'])
```

Brain volumes are positive regional volumes in the model's assumed volumetric scale. Protein measurements are **natural-log abundance**, not raw concentrations. Protein labels are six binary columns; comorbidity can produce more than one positive disease label. Read the family-specific documentation before fitting a model.

## Train a prediction model

[`examples/train_logistic_regression.py`](examples/train_logistic_regression.py) demonstrates training, validation-based regularization and threshold selection, and held-out evaluation on one synthetic clinical CSV. It uses the supplied schema to distinguish categorical and numeric measurements. All preprocessing is fitted on the training partition. The reported synthetic AUC and F1 are a software demonstration, not the study's real-cohort transfer results.

```bash
python examples/train_logistic_regression.py --csv generated/ukb_diabetes/E4_DM2/synthetic_E4_DM2_10000.csv --schema GPT6-UKB/allowed_schema.csv
```

For external validation, harmonize measurement definitions, units, category codes, missing-value conventions, and endpoint definitions first. Fit preprocessing on the designated training data and select hyperparameters and decision thresholds using validation data; keep test outcomes out of these steps. AUC uses continuous scores; F1 uses predictions at the frozen validation-selected threshold. Population priors and effect magnitudes in these generators are modelling assumptions, not patient-level risk calibration.

## Reproducibility and scope

- [`tests/smoke_test.py`](tests/smoke_test.py) generates every supported endpoint, checks output shapes and labels, and verifies reproducibility of the public brain and protein interfaces.
- [`tests/check_release.py`](tests/check_release.py) verifies the file manifest and checks published text for untranslated CJK characters.
- `release_source_hashes.json` records hashes of the original source files. `release_manifest.json` records the packaged files. English schema metadata translations do not change numeric constraints or generation equations.
- The checks completed successfully; see `validation_report.json` and `parity_report.json`. Run `python tests/smoke_test.py` and `python tests/check_release.py` to repeat the packaged checks.
- Small files under [`examples/synthetic`](examples/synthetic) are generated examples. **No real participant records, private train/test splits, credentials, or participant identifiers are included.**
- Frozen research engines preserve historical auxiliary branches and manifests for provenance. The supported brain interface exposes FIRST knowledge-based generation only; the supported protein interface exposes the six binary tasks. Historical control and AAL3 branches are not the current release workflow.
- UKB schemas include aggregate constraints, and the rare/background generators use supplied aggregate marginal summaries. Inspect the schema and source rather than interpreting “knowledge-generated” as “without any schema-level distribution information”.
- This release provides generation code, not the complete real-data benchmark pipeline or observed-measurement posterior risk inference implementation. Synthetic labels and internal risk equations do not by themselves provide a validated clinical prediction service.

The optional label-free UKB population mode is documented in [GPT6-UKB](GPT6-UKB/README.md). All code comments, help text, documentation, and descriptive metadata in this release are in English.
