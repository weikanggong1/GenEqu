# Label-free UKB measurement population

This mode generates the 201 dictionary-defined baseline measurements without diagnosis or prediction targets. Shared demographic, lifestyle and physiological factors connect the measurements. Body composition and blood counts obey explicit algebraic identities. Categorical values use declared probability and ordering assumptions. Structural applicability, reporting limits and fixed nonresponse mechanisms produce NaNs.

    python GPT6-UKB/label_free/generate.py --n 1000 --seed 20260831 --output-dir generated/label_free

Run from the repository root with a new output directory. The default is 300,000 rows and seed 20260831. Every distributional parameter, covariance mechanism and missingness probability is an authored modeling assumption. The permitted dictionary fixes names, units, ranges and category codes; its exact input hash is checked.

Outputs:

- synthetic_features_float32.npy: an n × 201 float32 matrix.
- feature_columns.json: ordered string Field IDs.
- generation_parameters.json: seed, model version, runtime/input/output hashes, numerical environment, provenance and field rules.

The same seed, code and numerical environment reproduce the matrix. Initial rows remain identical when n changes. No latent factors or extra identifiers are exported. Dictionary-defined broad health and family-context self-reports remain baseline measurements.

[core/MODEL_SPEC.md](core/MODEL_SPEC.md) defines every shared factor and equation. [core/measurement_rules.csv](core/measurement_rules.csv) lists all 201 generation and missingness rules. [core/test_generator.py](core/test_generator.py) provides synthetic-only consistency tests:

    python GPT6-UKB/label_free/core/test_generator.py

The source records the requested model configuration without claiming independent verification of a hidden backend. Synthetic consistency checks establish software and mathematical behavior, not cohort representativeness or prediction performance.
