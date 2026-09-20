# Protein evidence expansion v2 — frozen design specification

This is a new synthetic design dated 2026-09-08, not a reproduction of the earlier 200,000-sample experiment. No target participant values, disease effects, trained models, historical blocklist or historical effect rules were used. All 648 allowed assay identities remain in every matrix, in their original order.

## Evidence and identity

`feature_coverage.csv` has one row per allowed assay. `associations.csv` separates reviewed measured blood results, CSF results, curated genetic disease annotations and unreviewed title/abstract candidates. `references.csv`, `gene_search_log.csv` and the downloaded public files preserve source provenance. A combined five-disease query was completed for each of 630 unique genes using the original gene symbol and available original full protein name; the first 25 indexed records were retrieved. Hit counts and truncation are explicit. UniProt was additionally queried for all 630 unique accessions to obtain gene synonyms, curated disease annotations and background functional categories. This is a documented evidence screen, not an exhaustive review or a claim that every retrieved association has been manually adjudicated.

Only reviewed direct blood evidence with a reported supported direction can enter `feature_effects.csv`. Unreviewed candidates, tissue/CSF findings and genetic/pathway evidence never create a plasma direction. Non-significant measurements remain in the coverage table. All-cause dementia and vascular dementia are separate outcomes, not relabeled AD or StrokeTIA. Incident risk estimates are explicitly marked as direction proxies when used to model a disease-associated abundance shift. Acute stroke, historical stroke/TIA, genetic FTD and unselected FTD are not treated as identical observed phenotypes.

The final bulk expansion includes six author-published UKB Cell Atlas endpoint tables, each containing all 2,920 proteins: incident AD, incident PD, incident and prevalent ischemic stroke, and incident and prevalent TIA. Public ratios and confidence intervals are rounded; the original p values and protein-specific sample counts are retained. We apply a prespecified Bonferroni threshold of 0.05/2,920 within each retrieved endpoint, explicitly separate from the paper's broader phenome-wide analysis. Prevalent-disease OR direction is retained as a measured blood association, not a mean concentration difference; incident HR direction remains a prospective risk proxy. The six exports add 1,901 mapped assay-result rows, including nonsignificant results and three C3 fragment mappings per endpoint marked indirect. A missing OR in one mapped prevalent-stroke row creates no direction. Generic Atlas NPPB does not establish equivalence to either the BNP-32 or NT-proBNP assay and is excluded from primary directions.

All 15 multi-target assays receive zero disease shift because evidence for a single component does not identify the complex assay's direction. Multiple assays for the same gene remain separate identities. Where a publication specifies a SomaScan SeqId, exact aptamer matching is required. NT-proBNP evidence maps to the N-terminal pro-BNP assay, not the separate BNP-32 assay. Total C3 evidence is not transferred to the C3adesArg, C3b or C3d assays; these mappings are indirect and create no primary shift. Other cross-platform gene matches retain epitope/proteoform uncertainty.

GNPC papers are excluded as effect priors. Imam et al. 2025 Table 1 is used only as a list of 23 contributing cohorts. Sources containing GNPC results are separated by cohort; only EPIC4PD columns of the PD preprint were extracted. Chia 2025 includes BLSA controls, bPRIDE includes BioFINDER/Amsterdam, and GENFI has recruiting-center overlap that cannot be separated: these three sources are excluded from the main prior. UKB, ARIC, CHS, EPIC4PD, Sydney FRONTIER/ForeFront, Brescia and Miami Pre-fALS/CRiALS/CReATe are not named as cohorts in that public table. This comparison cannot establish absence of individual overlap. The optional `exclude_uncertain_clinical` policy omits all clinical case-control directions and retains only the population-cohort prospective directions; it is a conservative source-exclusion sensitivity, not a separately optimized model.

## Frozen disease shifts

`feature_effects.csv` has 648 × 6 rows. It keeps original literature estimates and their units in `literature_estimates_json`, the adjudicated literature direction in `literature_direction`, and the separate assumed generative amplitude in `model_effect_sd`.

Case-control directions take priority over prospective risk proxies. Discordant eligible case-control directions yield zero shift, including the acute-stroke FCN2 conflict. CU receives zero added disease shift. No eligible direction also yields zero shift; zero here means no imposed knowledge effect, not proof of biological absence.

The following magnitudes are design assumptions in units of feature residual SD. They do not copy a log-HR, odds ratio, NPX beta or measured fold change into synthetic log abundance:

| Evidence retained | Absolute model effect |
|---|---:|
| Concordant named clinical studies from at least two references | 0.55 |
| One clinical study with multiplicity-controlled support | 0.40 |
| One targeted clinical study or reported direction | 0.30 |
| Prospective disease-risk direction proxy | 0.25 |
| Genetic FTD subtype-only evidence | 0.20 |
| Conflict, ambiguous multi-target identity, indirect, unreviewed or unsupported | 0 |

References from the same named study population are not regarded as independent replication. Current effect selection has no tier promotion from multiple publications of UKB: prospective effects remain 0.25, and the six Cell Atlas tables share one reference. The 2020 serum paper's `LogFC` column lists positive magnitudes in both increased/decreased blocks; its explicit block heading determines direction, and its magnitudes are never transferred. Serum evidence is labeled as serum even when retained for a general blood-based prior. The FAP stroke study compares 47 acute cases with 22 cardiology patients, not healthy CU; its small contextual direction is an assumed transfer, not an observed stroke-versus-CU effect. IL1R2 separately compares 48 PD cases with 50 healthy controls.

The frozen main prior has 152 nonzero assay-disease pairs across 120 distinct assays: AD 10, PD 24, FTD 13, ALS 23 and StrokeTIA 82. These counts describe imposed model effects, not all supported literature associations. Fifty-five pairs use prospective risk proxies; the remaining 97 use context-specific clinical blood directions. The evidence table preserves conflicts and excluded sources even when the modeled shift is zero.

## Labels and sample counts

Default seed: `20260908`. Each interface contains 60,000 synthetic samples. Primary classes are exactly balanced: 10,000 each in the order `CU, AD, PD, FTD, ALS, StrokeTIA`.

For the exclusive multiclass interface, one class is assigned to each sample. For the multilabel interface, a secondary diagnosis is added independently to 20% of non-CU samples, uniformly over the other four diseases. The six-column target contains binary indicators. CU is an assigned cognitively unimpaired reference state, mutually exclusive with the five modeled diseases in this simulation. Absence of these five diagnoses does not establish cognitive normality in real GNPC participants: MCI and other impairments can lie outside this label set. This is a limitation of the synthetic label coverage, not a diagnostic rule inferred from target labels.

## Background, heterogeneity and arms

Each feature has a fixed prior mean drawn uniformly from 9.4–10.6 in arbitrary natural-log abundance units and a residual SD drawn uniformly from 0.65–1.05. Individual residuals are Gaussian. Added nuisance terms are a shared sample factor with SD 0.35, a functional-family factor with SD 0.22, and independent measurement noise with SD 0.10. Functional families are assigned deterministically from public UniProt function/GO annotations; they describe background covariance and carry no disease direction.

Disease severity is independently drawn from Normal(1, 0.20), clipped to 0.4–1.6. Active diagnoses contribute additive signed feature shifts multiplied by feature residual SD. Labels are independent of every background nuisance factor. The design does not estimate means, variances or covariance from target data.

Four arms share labels, background, measurement noise and severity within each interface:

- `knowledge`: the frozen effect matrix.
- `shuffled_identity`: one fixed permutation of its 648 feature rows, shared across disease columns, preserving standardized effect values and cross-disease structure.
- `reversed_direction`: every knowledge effect multiplied by −1.
- `no_effect`: zero effects with the same background.

Multilabel and multiclass interfaces use separate deterministic random streams. Arrays are float32 log abundance. No normalizer or predictive model is fitted in this generator. The downstream comparison of raw log abundance, within-person log ratio and within-person z score, each followed by feature normalization, is a separate evaluation task.

## Interface, audit and execution

```
common/Y_multilabel.npy                 (60000, 6), int8
common/y_multiclass.npy                 (60000,), int8
common/label_schema.json                semantic label/class/feature order
arms/<arm>/X_multilabel.npy             (60000, 648), float32
arms/<arm>/X_multiclass.npy             (60000, 648), float32
feature_names.json
label_names.json
synthetic_audit.csv                     aggregate label counts, moments and SMD norms
GENERATOR_FREEZE.json                   design and generated-array hashes
```

Rebuild the evidence/effect design only before freezing an experiment:

```bash
python build_evidence.py
python build_feature_effects.py
python finalize_design.py
python generate_protein_v2.py --self-test
python generate_protein_v2.py --output-root /path/to/new/synthetic_root --n-multilabel 60000 --n-multiclass 60000
```

The two sample-count options override the shared `--n-samples` default of 60,000. Each count must be a multiple of six and at least 12. The self-test uses small artificial arrays and checks determinism, label constraints, paired background, effect permutation, reversal symmetry and finiteness. Full generation and its synthetic audit must be run on the authorized computation server. The generator checks every required runtime file against `EVIDENCE_FREEZE.json` and refuses any nonempty output directory. `finalize_design.py` is a pre-experiment sealing step; changing the evidence afterward creates a different design, which must receive a new experimental freeze. All fitted-model and real-data analysis belongs to the parent evaluation task; descriptive associations do not establish causation.
