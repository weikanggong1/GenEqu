# GPT6-Brain

The supported interface generates **15 FIRST regional brain volumes** for AD/MCI or schizophrenia. Background anatomy and disease-related shifts are explicit lognormal models with global, anatomical-family, homologous-region, and individual components. Disease effects are frozen in the supplied `feature_effects.csv` files; source evidence and model assumptions can be inspected independently.

## Run

```bash
python GPT6-Brain/generate.py --task ad-binary --n-per-class 1000 --output generated/ad_binary.npz
python GPT6-Brain/generate.py --task ad-multiclass --n-per-class 1000 --output generated/ad_multiclass.npz
python GPT6-Brain/generate.py --task scz --n-per-class 1000 --output generated/scz.npz
```

`--n-per-class` is the size of **each class**: 1,000 gives 2,000 rows for either binary task and 3,000 for the AD three-class task. Seeds default to 20260908 for AD and 20260918 for SCZ; override with `--seed`. Output files must not exist.

## Output

Load the NPZ with `numpy.load(path, allow_pickle=False)`:

| Key | Meaning |
|---|---|
| `X` | Positive volume matrix, participants by 15 FIRST regions |
| `y` | Integer class index for each participant |
| `feature_names` | Column labels in exact matrix order |
| `class_names` | Class labels corresponding to integer indices |

The AD binary task contains NC and AD; the three-class task contains NC, MCI, and AD. SCZ contains HC and SCZ. Use the stored class labels rather than guessing index meanings. Model baselines represent assumed regional volumes in mm³; they are not resampled participant scans. No MRI image synthesis or registration is performed.

The AD model adds disease-specific shifts on the log-volume scale with individual severity variation. The SCZ model converts literature-standardized effects to log-volume shifts under its specified lognormal assumption. Neither model fits target participants during generation.

## Provenance

The ad and scz directories contain the fixed sampling engines, FIRST effect tables, reference records and feature-coverage tables. The public generate.py interface exposes the FIRST knowledge-based populations described above. Evidence-derived effect values and their modeling conversions are documented in the packaged effect tables; source hashes record the scientific inputs.
