# Protein population model

The supported public interface generates 648 natural-log protein-abundance measurements and six binary status labels: CU, AD, PD, FTD, ALS and StrokeTIA. Sampling uses fixed identity and effect tables, explicit background distributions and individual disease-severity variation. It reads no participant abundance records and fits no predictive model.

## Run and output

From the repository root:

    python GPT6-protein/generate.py --n 6000 --seed 20260908 --output generated/protein.npz

The output path must not exist. The default sample count is 60,000 and the default seed is 20260908. The sample count must be a multiple of six and at least 12.

The NPZ file contains:

| Key | Contents |
|---|---|
| X | n × 648 float32 natural-log abundance |
| Y | n × 6 binary status indicators |
| primary_label | Primary assigned group; audit metadata, not a predictor |
| feature_names | Ordered aptamer identifiers |
| label_names | CU, AD, PD, FTD, ALS, StrokeTIA |

Each binary prediction task uses one column of Y. CU represents an assigned cognitively unimpaired reference state in the model and cannot co-occur with a disease label.

## Shipped identities, effects and provenance

The packaged scientific inputs are:

- identity_only.csv: 648 ordered assay identities, aptamer identifiers and protein annotations.
- feature_effects.csv: 648 × 6 rows containing assay identity, disease, background family, evidence status, literature direction, model_effect_sd, public_reference_ids, literature_estimates_json and modeling notes.
- EFFECTS_DESIGN.json: effect-tier assumptions, nonzero-effect counts, source-exclusion decisions and the reference hash for the association curation.
- EVIDENCE_FREEZE.json: scientific design provenance and integrity hashes checked at runtime.

The row-level reference identifiers and literature-estimate fields document the evidence attached to the shipped effects. Disease directions derived from literature are distinguished from the numerical model amplitudes. Prospective risk associations used as directions are explicitly marked as proxies for abundance shifts. An unsupported, ambiguous, indirect or conflicting direction produces zero imposed shift; this is not a claim of biological absence.

All numerical effect amplitudes are assumptions in units of each feature's residual standard deviation. They do not directly substitute an odds ratio, hazard ratio or measured fold change into synthetic abundance:

| Retained evidence category | Absolute model amplitude |
|---|---:|
| Concordant clinical sources | 0.55 |
| A clinical source with multiplicity-controlled support | 0.40 |
| A targeted clinical source | 0.30 |
| Prospective disease-risk direction proxy | 0.25 |
| Genetic FTD subtype direction | 0.20 |
| No eligible imposed effect | 0 |

The fixed table has 152 nonzero assay–disease pairs across 120 assays: AD 10, PD 24, FTD 13, ALS 23 and StrokeTIA 82. CU has zero added disease effect. The executable reads model_effect_sd directly; it does not re-estimate these values from the generated batch or target observations.

## Labels and measurement equations

Primary groups are balanced exactly, with n/6 individuals in each group. Each individual whose primary group is not CU receives a second disease with probability 0.20, chosen uniformly among the other four diseases. The six binary positive counts therefore need not be balanced.

For feature j, its fixed-within-generation background location μj is sampled uniformly from 9.4–10.6, and its residual scale sj uniformly from 0.65–1.05. These are arbitrary natural-log abundance assumptions. Each individual receives an independent feature residual with scale sj, a shared sample factor with standard deviation 0.35, a shared protein-family factor with standard deviation 0.22, and additional measurement noise with standard deviation 0.10. Family assignments are read from the shipped effect table. All background components are independent of the sampled labels.

For each individual and disease, severity is sampled from a normal distribution with mean 1 and standard deviation 0.20, then clipped to 0.4–1.6. CU severity is set to zero. Let Yid be a binary disease indicator, Vid its severity and Ejd the signed model effect. The generated log abundance is:

    Xij = background_ij + sj × sum_d(Yid × Vid × Ejd)

The result is stored as float32. Co-occurring disease shifts add in this equation. No cohort mean, variance, covariance or outcome rate is fitted.

## Reproducibility and checks

The public interface uses seed+1000 for its multilabel generation stream. Labels and background noise use distinct deterministic streams. The same sample count, seed, files and numerical environment reproduce the output.

Runtime integrity checks verify the executable, identity table, effect table, design metadata and this specification against EVIDENCE_FREEZE.json. Run the scientific self-test with:

    python GPT6-protein/generate_protein_v2.py --self-test

The self-test checks deterministic arrays, binary-label constraints, finite values, effect permutations and reversal identities using synthetic inputs. The repository smoke test also verifies the supported six-binary public interface. These checks establish software and equation consistency, not empirical disease frequencies, clinical calibration or predictive performance.
