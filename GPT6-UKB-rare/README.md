# GPT6-UKB-rare

Seventeen shared baseline factors generate a 201-measurement population. Endpoint-specific probit equations then sample binary events over an assumed 15-year horizon. The measurement dictionary supplies names, units, bounds and category codes; all population distributions, factor loadings, event coefficients and missingness mechanisms are explicit modeling assumptions. Disease labels do not enter measurement generation.

## Generate populations

From the repository root:

    python GPT6-UKB-rare/generate.py --n 1000 --seed 20260831 --output-dir generated/rare
    python GPT6-UKB-rare/generate.py --diseases E4_HYPERPARA K11_COELIAC --n 1000 --seed 42 --output-dir generated/rare_selected

Omitting --diseases generates all 15 endpoints. --n is the number of rows per endpoint, with a default of 360,000. Each endpoint receives an independent population and stochastic event labels; case counts are not fixed. Small samples may contain no positive cases. The output directory must not exist.

Each synthetic_CODE_N.csv contains 201 ordered Field-ID predictors followed by a binary outcome named CODE. The audit.json file records seeds, source/input hashes, row counts, missingness and output hashes; its status becomes complete after every requested file is written. Identical inputs, seed, row count and dependency versions reproduce the generated values. Row prefixes across different requested sample sizes are not guaranteed.

## Observed-measurement risk

    python GPT6-UKB-rare/predict.py --input predictors.csv --disease E4_HYPERPARA --output generated/rare_risk.csv

The input is a numeric CSV with unique Field-ID column names and no outcome column. The output is one continuous probability per row. NaN and blank cells are accepted. Remove the synthetic target before supplying generated data:

    import pandas as pd
    frame = pd.read_csv("generated/rare/synthetic_E4_HYPERPARA_1000.csv")
    frame.iloc[:, :201].to_csv("predictors.csv", index=False)

The interface accepts any named subset of the 201 dictionary fields. Its conditioning set consists of exactly **38 measurements**: age, sex, pack-years and 35 continuous observation equations. [observed_risk_inputs.csv](observed_risk_inputs.csv) lists every input, description, unit, model support and missing-value rule. Other dictionary fields are accepted but do not enter the risk calculation. Unknown columns, including outcomes, are rejected.

Available observations update the shared-factor distribution using fixed Gaussian conditioning. The event equation integrates the remaining uncertainty; missing sex is integrated over its two model components. Pack-years at zero or its upper bound use censored-observation integration with a fixed 48-node rule. Missing or out-of-support values contribute no observation likelihood, except for the explicitly handled age boundaries and smoking censoring. No fitting, batch statistics or outcome calibration occurs. The result is P(15-year event | the available declared measurement subset), an assumed-model probability rather than a clinically calibrated risk estimate.

The Python function is predict_risk(X, feature_names, disease) in [core/predict.py](core/predict.py). [core/MODEL_SPEC.md](core/MODEL_SPEC.md) provides its equations and an executable import example. [core/test_model.py](core/test_model.py) verifies prior integration, censoring quadrature, missingness, sex gating, row/batch invariance and rejection of outcome columns.

## Model files

- [core/measurement_dictionary.csv](core/measurement_dictionary.csv): the ordered 201-field measurement dictionary.
- [core/disease_identities.csv](core/disease_identities.csv): the 15 endpoint identities and coding patterns.
- [core/model.py](core/model.py), [core/generate.py](core/generate.py) and [core/predict.py](core/predict.py): fixed scientific runtime equations.
- [core/MODEL_SPEC.md](core/MODEL_SPEC.md): population assumptions, observation transforms and event equations.
- [core/feature_disease_rules.csv](core/feature_disease_rules.csv): all 3,015 measurement–endpoint pairs, their shared-factor paths and conditioning-set membership.

The thin public entry points resolve bundled files independently of the working directory. Runtime hashes are recorded in the repository source-verification report. Endpoint names describe model targets; registry-specific control exclusions and actual disease-free cohort eligibility are not reconstructed.

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
