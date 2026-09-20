# Independent GPT6-KNHANES knowledge-to-population construction

Build a new executable KNHANES generator with requested tool-visible configuration gpt-6-astra, ultra reasoning, fork_turns=none. This is an independent generator for South Korean adults aged 40–70 and the 10 clinical questionnaire identities in disease_identities.csv, using the 44 measurements in allowed_schema.csv.

## Input boundary

Read only this contract, allowed_schema.csv, disease_identities.csv, and schema_provenance.json in THIS builder directory as pre-existing project inputs. Do not read memory, parent context, adjacent directories, previous generators, real or synthetic datasets, cohort summaries, empirical case/missing counts, disease effects, classifiers, or performance results. Do not access remote sessions, APIs, or other agents. This is an input-handling boundary, not OS isolation. Use stored biomedical knowledge and the supplied public measurement semantics; no external retrieval is part of this construction. The provided schema has no empirical marginal ranges or disease statistics. Write only inside this builder directory.

Author numerical priors, correlations, disease associations, treatment/ascertainment mechanisms, and missingness independently. State which parameters are construction choices rather than measured Korean population estimates. Do not invent citations. No target prevalence, AUC, F1, feature importance, or separation objective is provided. Never fit classifiers or tune to empirical feedback.

## Clinical interpretation

Generate a plausible Korean survey population at the current measurement visit, with a binary self-reported physician-diagnosis history label. The exact clinical target is the KNHANES target_name/target_field, while the phenotype code is only the existing task's stable identifier. Dyslipidaemia is broader than pure hypercholesterolaemia; diabetes is type-unspecified; kidney disease does not establish chronicity; MI/angina and obstructive sleep apnoea follow their stated definitions. Model each exact target as given, without substituting a lab cutoff, spirometric obstruction, low bone density, or a prospective incident-risk endpoint.

Create overlapping case/non-case populations, coherent physiology, latent unmeasured disease/diagnostic uncertainty, measurement noise, and age/sex/lifestyle structure. Current measurements may reflect duration, diagnosis, management, and residual risk; author any such mechanisms without using observed performance. Do not place labels, disease probabilities, treatment flags, diagnostic codes, identifiers, or latent variables in predictors. Missingness should reflect measurement processes and not encode the outcome as an artificial shortcut. Consider direct, indirect, and neutral disease relationships across all 44 features. The dependence and exact numerical choices must be fully executable and documented.

Output uses the 44 field_id columns and units in allowed_schema.csv, already harmonized to the shared analysis representation. Native Korean units are contextual: do not convert an already-harmonized value twice. Preserve documented smoking/alcohol codes and coherence, top-code household size at six, and respect age 40–70 and walking days 0–7. Physiology must respect units and mass/count identities within documented measurement variation. Do not leave unspecified numeric fields to a generic normal fallback.

## Interface and scale

Implement generate_knhanes.py using standard Python, NumPy, and pandas only:

`python3 generate_knhanes.py --schema allowed_schema.csv --disease-identities disease_identities.csv --output-dir OUTPUT --n 360000 --seed 20260831 --diseases EXACT_CODE [EXACT_CODE ...]`

Default to all 10 diseases. Each task writes `OUTPUT/<phenotype>/synthetic_<phenotype>_<n>.csv`, exactly the 44 predictor columns in schema order plus one binary target column named as the phenotype. Generate one task at a time or in fixed blocks to avoid retaining all populations in memory. Use stable disease-specific random streams independent of task order, subset selection, and Python hash randomization. Expose a small in-memory generation function. Refuse overwriting existing task outputs. Write input/code/output hashes and counts to separate audit JSON, never to predictors.

Downstream, a fixed label-independent permutation splits each disease's 360000 synthetic participants into 300000 training and 60000 validation. This matches the existing synthetic training budget. Standardization, feature selection, model fitting, validation selection/calibration/thresholding happen outside the generator. No real evaluation feedback will be returned to the builder.

## Deliverables and verification

Provide generate_knhanes.py, GENERATOR_RULES.md, feature_disease_rules.csv covering every 10×44 pair, and test_generator.py. Any local constants module must be included. Tests must verify schema/codes/units, binary targets, all-disease coverage, determinism, physiological relationships, missingness integrity, no forbidden input reads, no overwrite, and one full 360000-row write/read with bounded memory. Do not test predictive performance or optimize class separation. If scientific libraries are absent locally, finish source and tell root the synthetic-only test command; root will run it on gpucw1 and return implementation errors only. Record model configuration as requested, not a verified backend build identity. After actual implementation tests pass, all inputs/code/rules/tests and test evidence will be hashed and frozen BEFORE generation/evaluation against real data. Do not self-claim unrun tests passed.
