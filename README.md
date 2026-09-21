# GenEqu

**Biomedical knowledge expressed as executable population models.**

GenEqu contains six generator families. They sample measurements and disease labels through explicit equations, probability distributions, shared factors and fixed parameter tables. Sampling runs locally without an LLM API, API key, GPU or fitted neural generator.

| Generator | Measurements | Outcomes |
|---|---|---|
| [GPT6-UKB](GPT6-UKB/README.md) | 201 UK Biobank variables | 30 selected 15-year event endpoints |
| [GPT6-UKB-rare](GPT6-UKB-rare/README.md) | The same 201-variable panel | 15 selected 15-year event endpoints |
| [GPT6-NHANES](GPT6-NHANES/README.md) | 52 NHANES measurements | 14 reported diagnosis endpoints |
| [GPT6-KNHANES](GPT6-KNHANES/README.md) | 44 harmonized measurements | 10 reported diagnosis endpoints |
| [GPT6-Brain](GPT6-Brain/README.md) | 15 FIRST regional brain volumes | AD versus NC; NC/MCI/AD; SCZ versus HC |
| [GPT6-protein](GPT6-protein/README.md) | 648 protein measurements | Six binary diagnostic-status tasks |

UKB endpoint events and survey diagnosis histories have different ascertainment. Matching codes do not make endpoint definitions identical. The rare family name identifies the selected 15-endpoint model; it does not assert a uniform clinical rarity threshold.

## Installation

Use Python 3.10 or later and the versions in requirements.txt.

    git clone https://github.com/weikanggong1/GenEqu.git
    cd GenEqu
    python -m venv .venv
    # Linux / macOS:
    source .venv/bin/activate
    # Windows PowerShell:
    # .venv\Scripts\Activate.ps1
    python -m pip install -r requirements.txt

Run the commands below from the repository root. Use a new output path for each run.

## Generate data

Clinical --n values are rows **per endpoint**. Omitting --diseases generates all endpoints in that family. Clinical defaults are 360,000 rows per endpoint; choose an explicit smaller number for a trial.

    python GPT6-UKB/generate.py --n 1000 --seed 20260831 --output-dir generated/ukb
    python GPT6-UKB-rare/generate.py --n 1000 --seed 20260831 --output-dir generated/rare
    python GPT6-NHANES/generate.py --n 1000 --seed 20260831 --output-dir generated/nhanes
    python GPT6-KNHANES/generate.py --n 1000 --seed 20260831 --output-dir generated/knhanes

    python GPT6-Brain/generate.py --task ad-binary --n-per-class 1000 --output generated/ad_binary.npz
    python GPT6-Brain/generate.py --task ad-multiclass --n-per-class 1000 --output generated/ad_multiclass.npz
    python GPT6-Brain/generate.py --task scz --n-per-class 1000 --output generated/scz.npz
    python GPT6-protein/generate.py --n 6000 --output generated/protein.npz

Select endpoints with --diseases:

    python GPT6-UKB/generate.py --diseases E4_DM2 --n 10000 --seed 42 --output-dir generated/ukb_diabetes

[Label-free UKB generation](GPT6-UKB/label_free/README.md) produces a joint 201-measurement population for representation learning:

    python GPT6-UKB/label_free/generate.py --n 1000 --seed 20260831 --output-dir generated/label_free

Its matrix contains no disease targets. Physiological factors, body/blood identities, categorical rules and missingness are specified in its model documentation.

## Read the output

Clinical CSV files contain predictors followed by one binary outcome. Find them recursively beneath the selected output directory. Adjacent audit files record generation metadata. Each endpoint file represents an independently generated population; rows from separate files do not form jointly observed multi-disease records.

    from pathlib import Path
    import pandas as pd
    import numpy as np

    path = next(Path("generated/ukb_diabetes").rglob("synthetic_*.csv"))
    cohort = pd.read_csv(path)
    X, y = cohort.iloc[:, :-1], cohort.iloc[:, -1]

    brain = np.load("generated/ad_binary.npz", allow_pickle=False)
    X_brain, y_brain = brain["X"], brain["y"]

    protein = np.load("generated/protein.npz", allow_pickle=False)
    X_protein, Y_protein = protein["X"], protein["Y"]

    label_free = np.load("generated/label_free/synthetic_features_float32.npy", allow_pickle=False)

Brain volumes use the model's assumed volumetric scale. Protein measurements are natural-log abundance; six binary labels can represent comorbidity. Family READMEs specify units, categories and assumptions. Missing values need explicit downstream handling.

## Observed-measurement risk

The rare-endpoint family includes a fixed observed-measurement risk interface:

    python GPT6-UKB-rare/predict.py --input predictors.csv --disease E4_HYPERPARA --output generated/rare_risk.csv

Input columns are dictionary Field IDs, without outcome labels. The conditioning set comprises [38 documented measurements](GPT6-UKB-rare/observed_risk_inputs.csv); a full 201-field predictor table or a named subset is accepted. Missing selected measurements are integrated out. The equations produce continuous 15-year model probabilities without fitting or outcome calibration. See the [rare-family documentation](GPT6-UKB-rare/README.md) for supports, censoring and interpretation.

## Train a prediction model

[examples/train_logistic_regression.py](examples/train_logistic_regression.py) demonstrates preprocessing fitted on the training partition, validation-based regularization and threshold selection, and held-out synthetic evaluation.

    python examples/train_logistic_regression.py --csv generated/ukb_diabetes/E4_DM2/synthetic_E4_DM2_10000.csv --schema GPT6-UKB/allowed_schema.csv

For external validation, harmonize definitions, units, category codes, missing values and endpoints first. Keep test outcomes separate from preprocessing, tuning and threshold selection. AUC uses continuous scores; F1 uses the validation-selected threshold. Synthetic example metrics demonstrate software behavior and are not evidence of clinical calibration or real-cohort transfer.
