# GPT6-protein

This generator samples 648 protein measurements and six binary labels: cognitively unimpaired status (CU), Alzheimer disease (AD), Parkinson disease (PD), frontotemporal dementia (FTD), amyotrophic lateral sclerosis (ALS), and stroke/transient ischaemic attack (StrokeTIA).

## Run

```bash
python GPT6-protein/generate.py --n 6000 --seed 20260908 --output generated/protein.npz
```

`--n` must be a multiple of six and at least 12. The default is 60,000. Primary labels are balanced across the six groups. Some participants with a disease receive a second disease label, so positive counts in the six binary tasks need not be equal. CU does not co-occur with a disease. The multilabel random stream uses the supplied seed plus 1,000 internally.

## Output

```python
import numpy as np
cohort = np.load('generated/protein.npz', allow_pickle=False)
X, Y = cohort['X'], cohort['Y']
```

| Key | Shape / meaning |
|---|---|
| `X` | N × 648, natural-log abundance |
| `Y` | N × 6, binary status for each task |
| `feature_names` | Aptamer identifiers in column order |
| `label_names` | CU, AD, PD, FTD, ALS, StrokeTIA |
| `primary_label` | Primary sampled group, supplied for auditing only |

Train each binary task using the corresponding `Y` column. `primary_label` is **not an input feature** and does not define a six-class benchmark in this release. Do not exponentiate `X` unless a downstream procedure explicitly requires the corresponding model-scale abundance.

The model adds global and protein-family variation to protein-specific background abundance, then adds disease-specific shifts weighted by participant severity. Inspect `feature_effects.csv` for all 648 × 6 effects, including zero effects. `identity_only.csv` provides protein identities. All values are synthetic; no participant abundance table is packaged.

[GENERATOR_SPEC.md](GENERATOR_SPEC.md) documents the supported six-binary interface, the shipped identity and effect tables, and the measurement equations. EVIDENCE_FREEZE.json records design provenance and verifies runtime integrity.
