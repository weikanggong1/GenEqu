"""New evidence-expanded GNPC synthetic design; no target data are read or fitted."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
LABELS = ["CU", "AD", "PD", "FTD", "ALS", "StrokeTIA"]
ARMS = ["knowledge", "shuffled_identity", "reversed_direction", "no_effect"]
DEFAULT_SEED = 20260908
DEFAULT_N = 60000


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_design(policy="main"):
    frozen = json.loads((ROOT/"EVIDENCE_FREEZE.json").read_text())
    for filename in frozen["required_runtime_files"]:
        if sha256(ROOT/filename) != frozen["design_sha256"][filename]:
            raise ValueError("Frozen design hash mismatch: "+filename)
    rows = list(csv.DictReader((ROOT/"feature_effects.csv").open(encoding="utf-8-sig")))
    identity = list(csv.DictReader((ROOT/"identity_only.csv").open()))
    assert len(identity) == 648 and len(rows) == 648*6
    effects = np.zeros((648, 6), dtype=np.float64)
    families = [""]*648
    for row in rows:
        i, j = int(row["feature_index"]), LABELS.index(row["disease"])
        assert identity[i]["Aptamer"] == row["Aptamer"]
        effects[i, j] = float(row["model_effect_sd"])
        families[i] = row["background_family"]
        if policy == "exclude_uncertain_clinical" and row["evidence_status"] not in {
                "prospective_risk_direction_proxy", "reference_no_added_disease_shift"}:
            effects[i, j] = 0
    assert not np.any(effects[:, 0])
    return identity, families, effects


def make_labels(n, rng, interface):
    if n < 12 or n % 6:
        raise ValueError("n must be a multiple of six and at least12")
    primary = np.tile(np.arange(6, dtype=np.int8), n//6)
    rng.shuffle(primary)
    labels = np.eye(6, dtype=np.int8)[primary]
    if interface == "multilabel":
        cases = np.flatnonzero((primary != 0) & (rng.random(n) < .20))
        second = rng.integers(1, 5, size=len(cases))
        second += second >= primary[cases]
        labels[cases, second] = 1
    return primary, labels


def arm_effects(effects, seed):
    permutation = np.random.default_rng(seed+400).permutation(len(effects))
    return {
        "knowledge": effects,
        "shuffled_identity": effects[permutation],
        "reversed_direction": -effects,
        "no_effect": np.zeros_like(effects),
    }, permutation


def generate_interface(n, seed, interface, families, effects):
    label_rng = np.random.default_rng(seed+101)
    primary, labels = make_labels(n, label_rng, interface)
    # All nuisance factors are drawn independently of disease labels.
    rng = np.random.default_rng(seed+201)
    unique_families = sorted(set(families))
    family_index = np.array([unique_families.index(f) for f in families])
    baseline = rng.uniform(9.4, 10.6, len(families))
    residual_sd = rng.uniform(.65, 1.05, len(families))
    background = rng.normal(size=(n, len(families))) * residual_sd
    background += baseline
    background += rng.normal(0, .35, (n, 1))
    family_factors = rng.normal(0, .22, (n, len(unique_families)))
    background += family_factors[:, family_index]
    background += rng.normal(0, .10, background.shape)
    severity = np.clip(rng.normal(1, .20, (n, 6)), .4, 1.6)
    severity[:, 0] = 0
    active = labels * severity
    return primary, labels, background, active, residual_sd


def audit_rows(interface, arm, x, labels):
    rows = []
    median_variance = float(np.median(x.var(axis=0, dtype=np.float64)))
    for j, disease in enumerate(LABELS):
        positive = labels[:, j] == 1
        a, b = x[positive], x[~positive]
        ma, mb = a.mean(axis=0, dtype=np.float64), b.mean(axis=0, dtype=np.float64)
        va, vb = a.var(axis=0, ddof=1, dtype=np.float64), b.var(axis=0, ddof=1, dtype=np.float64)
        pooled = np.sqrt((va+vb)/2)
        smd = (ma-mb)/np.maximum(pooled, 1e-12)
        rows.append(dict(interface=interface, arm=arm, label=disease, n_positive=int(positive.sum()),
            n_negative=int((~positive).sum()), positive_mean_log_abundance=float(ma.mean()),
            negative_mean_log_abundance=float(mb.mean()), mean_difference_norm=float(np.linalg.norm(ma-mb)),
            smd_norm=float(np.linalg.norm(smd)), median_feature_variance=median_variance,
            finite=bool(np.isfinite(x).all())))
    return rows


def self_test():
    _, families, effects = read_design()
    arms, permutation = arm_effects(effects, DEFAULT_SEED)
    assert np.array_equal(np.sort(arms["shuffled_identity"], axis=0), np.sort(effects, axis=0))
    assert sorted(permutation.tolist()) == list(range(648))
    for interface in ["multilabel", "multiclass"]:
        a = generate_interface(120, DEFAULT_SEED, interface, families, effects)
        b = generate_interface(120, DEFAULT_SEED, interface, families, effects)
        assert all(np.array_equal(x, y) for x, y in zip(a, b))
        primary, labels, background, active, sd = a
        assert labels.shape == (120, 6) and background.shape == (120, 648)
        assert np.all((labels == 0) | (labels == 1))
        assert np.all(labels[primary == 0, 1:] == 0)
        if interface == "multiclass": assert np.all(labels.sum(axis=1) == 1)
        shifted = background + (active @ effects.T)*sd
        reverse = background - (active @ effects.T)*sd
        assert np.max(np.abs((shifted+reverse)/2-background)) < 1e-12
        assert np.isfinite(shifted).all()
    print(json.dumps(dict(self_test="PASS", features=648, labels=LABELS, arms=ARMS,
        checks=["deterministic", "paired_background", "binary_labels", "exclusive_multiclass", "shuffle_preserves_effects", "reversal_identity", "finite"]), indent=2))


def run(args):
    identity, families, effects = read_design(args.evidence_policy)
    counts = {"multilabel": args.n_samples if args.n_multilabel is None else args.n_multilabel,
        "multiclass": args.n_samples if args.n_multiclass is None else args.n_multiclass}
    if any(n < 12 or n % 6 for n in counts.values()):
        raise ValueError("Sample counts must be multiples of six and at least12")
    out = Path(args.output)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Output directory must be empty")
    (out/"common").mkdir(parents=True, exist_ok=True)
    matrix_by_arm, permutation = arm_effects(effects, args.seed)
    audit = []; paths = []
    for interface, offset in [("multilabel", 1000), ("multiclass", 2000)]:
        primary, labels, background, active, sd = generate_interface(counts[interface], args.seed+offset, interface, families, effects)
        label_path = out/"common"/("Y_multilabel.npy" if interface == "multilabel" else "y_multiclass.npy")
        np.save(label_path, labels if interface == "multilabel" else primary, allow_pickle=False); paths.append(label_path)
        for arm in ARMS:
            x = (background + (active @ matrix_by_arm[arm].T)*sd).astype(np.float32)
            arm_dir = out/"arms"/arm; arm_dir.mkdir(parents=True, exist_ok=True)
            path = arm_dir/("X_"+interface+".npy")
            np.save(path, x, allow_pickle=False); paths.append(path)
            audit.extend(audit_rows(interface, arm, x, labels))
            del x
        del background, active
    with (out/"synthetic_audit.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(audit[0]));writer.writeheader();writer.writerows(audit)
    (out/"feature_names.json").write_text(json.dumps([r["Feature"] for r in identity])+"\n")
    (out/"label_names.json").write_text(json.dumps(LABELS)+"\n")
    schema_path = out/"common"/"label_schema.json"
    schema_path.write_text(json.dumps(dict(label_names=LABELS, multiclass_names=LABELS,
        feature_names=[r["Feature"] for r in identity]), indent=2)+"\n")
    paths.append(schema_path)
    design_files = [Path(__file__), ROOT/"feature_effects.csv", ROOT/"identity_only.csv", ROOT/"GENERATOR_SPEC.md", ROOT/"EFFECTS_DESIGN.json", ROOT/"EVIDENCE_FREEZE.json"]
    freeze = dict(generator="protein_evidence_expansion_v2_20260908", new_design_not_old_experiment_reproduction=True,
        seed=args.seed, n_samples_per_interface=counts, evidence_policy=args.evidence_policy,
        features=648, labels=LABELS, arms=ARMS, input_scale="natural_log_abundance_arbitrary_prior_baseline",
        effect_scale="fixed_model_residual_SD_units_not_literature_effect_scale", secondary_diagnosis_probability=.20,
        CU_definition="assigned_cognitively_unimpaired_reference_exclusive_with_five_diseases_not_real_diagnostic_rule", paired_nuisance_across_arms=True,
        nuisance_independent_of_labels=True, shuffled_identity_permutation=permutation.tolist(),
        residual_sd_prior=[.65, 1.05], baseline_log_abundance_prior=[9.4, 10.6], global_sd=.35, family_sd=.22,
        measurement_sd=.10, severity_normal=[1, .20], severity_clip=[.4, 1.6],
        nonzero_effect_counts={d:int(np.count_nonzero(effects[:, j])) for j,d in enumerate(LABELS)},
        source_design_sha256={p.name:sha256(p) for p in design_files},
        generated_files={str(p.relative_to(out)):dict(sha256=sha256(p), bytes=p.stat().st_size) for p in paths},
        target_data_read=False, real_model_fitted=False, numpy_version=np.__version__)
    (out/"GENERATOR_FREEZE.json").write_text(json.dumps(freeze, indent=2)+"\n")
    print(json.dumps(dict(output=str(out.resolve()), n_samples_per_interface=counts, features=648, arrays=10,
        audit_rows=len(audit), nonzero_effect_counts=freeze["nonzero_effect_counts"]), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", "--output", dest="output")
    parser.add_argument("--n-samples", type=int, default=DEFAULT_N)
    parser.add_argument("--n-multilabel", type=int)
    parser.add_argument("--n-multiclass", type=int)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--evidence-policy", choices=["main", "exclude_uncertain_clinical"], default="main")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        if not args.output: parser.error("--output is required")
        run(args)
