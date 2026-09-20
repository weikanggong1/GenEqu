#!/usr/bin/env python3
"""Generate a label-free synthetic population for representation learning.

Version P0.1.0 (protocol name: POP-JOINT-P0.1.0) deliberately generates no
disease target, diagnosis, incidence time, severity, or treatment variable.
It reuses the frozen G0 semantic/schema engine only in an inert baseline mode,
then adds continuous correlated multi-system excursions.  Anonymous sparse
programs create many overlapping physiological patterns and comorbidity-like
combinations without assigning any program to a named disease.

The generator reads only the aggregate 201-variable schema.  It must not read
participant-level real data, real labels, or predictive-performance results.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


VERSION = "P0.1.0"
PROTOCOL = "POP-JOINT-P0.1.0"
DEFAULT_N = 300_000
DEFAULT_SEED = 20260831
DEFAULT_G0 = "generate_all_12_synthetic_diseases_v2.py"
DEFAULT_SCHEMA = "all_variable_to_generate_with_constraints.csv"

# The domain names describe physiology, not diseases.  They are latent and are
# written only to an audit sample, never to the representation-training matrix.
DOMAINS = [
    "metabolic", "vascular", "renal", "hepatic", "inflammatory",
    "hematologic", "pulmonary", "functional", "neurobehavioral",
    "endocrine", "nutritional", "systemic_frailty",
]

# Frozen anonymous program loadings.  Every row is one unnamed physiological
# program and every column follows DOMAINS.  Sparse overlap permits many joint
# patterns without constructing a disease label.  These numbers are versioned
# constants and must not be tuned against real outcome performance.
PROGRAM_LOADINGS = np.asarray([
    [1.00, .55, .15, .45, .35, 0, 0, .20, .20, .25, -.15, .25],
    [.20, 1.00, .45, 0, .25, 0, .10, .35, .15, 0, 0, .40],
    [.10, .35, 1.00, .20, .45, .45, 0, .50, .10, 0, .25, .55],
    [.35, .10, .15, 1.00, .50, .20, 0, .25, .15, .20, -.30, .35],
    [.20, .25, .30, .35, 1.00, .35, .25, .45, .35, .10, -.20, .55],
    [0, .10, .30, .20, .35, 1.00, 0, .35, .10, .15, .35, .30],
    [0, .15, .10, 0, .35, 0, 1.00, .55, .20, 0, 0, .40],
    [.10, .30, .35, .15, .35, .15, .30, 1.00, .45, .15, .25, .70],
    [.15, .15, .10, .10, .30, .10, .15, .45, 1.00, .35, .20, .40],
    [.25, 0, .10, .25, .20, .25, 0, .20, .25, 1.00, .35, .25],
    [-.10, 0, .20, -.15, .25, .30, 0, .35, .20, .25, 1.00, .30],
    [.35, .50, .55, .40, .60, .35, .35, .75, .45, .25, .20, 1.00],
    [.75, .70, .30, .55, .50, 0, 0, .25, .25, .15, -.20, .45],
    [0, .55, .65, .10, .45, .20, .20, .65, .35, 0, .20, .75],
    [.20, 0, .20, .70, .65, .45, .10, .50, .25, .15, -.15, .65],
    [.10, .25, .15, .10, .65, .15, .70, .70, .45, .10, .10, .75],
    [.35, .15, .20, .20, .25, .35, .10, .35, .50, .75, .55, .45],
    [.45, .35, .50, .35, .55, .50, .40, .80, .60, .40, .35, 1.00],
], dtype=np.float64)


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, Path):
        return str(value)
    return value


def load_g0(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("metaworld_g0_frozen", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load frozen G0 module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def positive_tail(x: np.ndarray, threshold: float = 0.70,
                  temperature: float = 0.55) -> np.ndarray:
    """Smooth heavy positive tail, centered to preserve population location."""
    y = temperature * np.logaddexp(0.0, (x - threshold) / temperature)
    return y - np.median(y, axis=0, keepdims=x.ndim == 2)


def generate_states(n: int, rng: np.random.Generator) -> tuple[np.ndarray, dict[str, Any]]:
    """Generate continuous heavy-tailed, correlated multi-system states."""
    n_domains = len(DOMAINS)
    global_raw = rng.standard_t(df=5, size=n)
    block_raw = rng.standard_t(df=6, size=(n, 4))
    domain_raw = rng.standard_t(df=7, size=(n, n_domains))
    program_raw = rng.standard_t(df=4, size=(n, PROGRAM_LOADINGS.shape[0]))

    # Four broad dependency blocks: cardio-metabolic, renal/hepatic/immune,
    # pulmonary/function/neurobehavioral, and endocrine/nutrition/frailty.
    block_map = np.asarray([0, 0, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3])
    base = (
        0.30 * global_raw[:, None]
        + 0.32 * block_raw[:, block_map]
        + 0.70 * domain_raw
    )
    anonymous = positive_tail(program_raw, threshold=0.95, temperature=0.60)
    anonymous = anonymous @ PROGRAM_LOADINGS
    anonymous /= np.sqrt(np.maximum((PROGRAM_LOADINGS ** 2).sum(axis=0), 1.0))[None, :]
    state = 0.72 * positive_tail(base) + 0.38 * anonymous
    state += 0.16 * global_raw[:, None]
    state = np.clip(state, -2.5, 5.0)
    state -= np.median(state, axis=0, keepdims=True)

    audit = {
        "distribution": "correlated Student-t blocks + smooth positive tails + 18 frozen anonymous sparse programs",
        "domains": DOMAINS,
        "program_count": int(PROGRAM_LOADINGS.shape[0]),
        "program_loadings": PROGRAM_LOADINGS,
        "state_clip": [-2.5, 5.0],
        "no_discrete_case_indicator": True,
        "no_named_disease_program": True,
    }
    return state.astype(np.float32), audit


def apply_continuous_physiology(values: dict[str, np.ndarray], norm: Any,
                                state: np.ndarray) -> list[dict[str, Any]]:
    """Apply version-frozen continuous organ-state effects to semantic values."""
    s = {name: state[:, i].astype(np.float64) for i, name in enumerate(DOMAINS)}
    changes: list[dict[str, Any]] = []

    def get(description: str) -> np.ndarray | None:
        return values.get(norm(description))

    def add(description: str, terms: dict[str, float]) -> None:
        x = get(description)
        if x is None:
            changes.append({"variable": description, "operation": "add", "terms": terms, "status": "not_present"})
            return
        delta = sum(float(coef) * s[name] for name, coef in terms.items())
        values[norm(description)] = np.asarray(x, dtype=np.float64) + delta
        changes.append({"variable": description, "operation": "add", "terms": terms, "status": "applied"})

    def log_multiply(description: str, terms: dict[str, float]) -> None:
        x = get(description)
        if x is None:
            changes.append({"variable": description, "operation": "log_multiply", "terms": terms, "status": "not_present"})
            return
        log_delta = sum(float(coef) * s[name] for name, coef in terms.items())
        values[norm(description)] = np.asarray(x, dtype=np.float64) * np.exp(np.clip(log_delta, -0.8, 1.2))
        changes.append({"variable": description, "operation": "log_multiply", "terms": terms, "status": "applied"})

    # Glycaemic/lipid continuum.  Effects are deliberately smaller than G0's
    # case-control disease consequences and remain smooth across all subjects.
    add("Glucose", {"metabolic": .48, "systemic_frailty": .08})
    add("Glycated haemoglobin (HbA1c)", {"metabolic": 3.20, "systemic_frailty": .35})
    log_multiply("Triglycerides", {"metabolic": .11, "hepatic": .04})
    log_multiply("HDL cholesterol", {"metabolic": -.055, "inflammatory": -.025})
    log_multiply("Apolipoprotein B", {"metabolic": .045, "vascular": .035})
    add("LDL direct", {"metabolic": .10, "vascular": .08})

    # Vascular continuum.
    add("Systolic blood pressure, automated reading", {"vascular": 5.2, "renal": 1.2})
    add("Diastolic blood pressure, automated reading", {"vascular": 3.1, "renal": .55})
    add("Pulse rate, automated reading", {"vascular": 1.1, "inflammatory": 1.0, "neurobehavioral": .7})

    # Renal/mineral continuum with coherent anaemia tendency.
    add("Creatinine", {"renal": 10.0, "systemic_frailty": 1.2})
    log_multiply("Cystatin C", {"renal": .085, "inflammatory": .018})
    add("Urea", {"renal": .48, "nutritional": .10})
    add("Phosphate", {"renal": .026})
    add("Albumin", {"renal": -.18, "hepatic": -.25, "inflammatory": -.24, "nutritional": .24})

    # Hepatic continuum.
    log_multiply("Alanine aminotransferase", {"hepatic": .14, "metabolic": .035})
    log_multiply("Aspartate aminotransferase", {"hepatic": .11, "systemic_frailty": .025})
    log_multiply("Gamma glutamyltransferase", {"hepatic": .15, "metabolic": .025})
    log_multiply("Alkaline phosphatase", {"hepatic": .065, "nutritional": -.015})
    log_multiply("Direct bilirubin", {"hepatic": .085})
    log_multiply("Total bilirubin", {"hepatic": .060})

    # Inflammatory/immune continuum.
    log_multiply("C-reactive protein", {"inflammatory": .24, "systemic_frailty": .07})
    log_multiply("White blood cell (leukocyte) count", {"inflammatory": .045})
    log_multiply("Neutrophill count", {"inflammatory": .055})
    log_multiply("Lymphocyte count", {"inflammatory": .025, "systemic_frailty": -.012})
    log_multiply("Monocyte count", {"inflammatory": .040})
    log_multiply("Rheumatoid factor", {"inflammatory": .075})

    # Correlated haematopoietic continuum.  Hemoglobin/haematocrit/RBC move
    # together, while RDW and reticulocyte measures provide compensatory axes.
    add("Red blood cell (erythrocyte) count", {"hematologic": .065, "renal": -.025, "inflammatory": -.018})
    add("Haemoglobin concentration", {"hematologic": 1.55, "renal": -.70, "inflammatory": -.35})
    add("Haematocrit percentage", {"hematologic": .0048, "renal": -.0022, "inflammatory": -.0012})
    add("Red blood cell (erythrocyte) distribution width", {"hematologic": .13, "inflammatory": .07, "nutritional": .06})
    add("Platelet count", {"hematologic": 7.0, "inflammatory": 4.0, "hepatic": -3.0})
    log_multiply("Reticulocyte percentage", {"hematologic": .045})
    log_multiply("Immature reticulocyte fraction", {"hematologic": .050, "inflammatory": .020})

    # Pulmonary and physical-function continua.
    add("Forced vital capacity (FVC)", {"pulmonary": -.11, "functional": -.035})
    add("Forced expiratory volume in 1-second (FEV1)", {"pulmonary": -.13, "functional": -.035})
    add("Peak expiratory flow (PEF)", {"pulmonary": -10.5, "functional": -2.5})
    add("FEV1/ FVC ratio Z-score", {"pulmonary": -.10})
    add("Hand grip strength (left)", {"functional": -1.15, "systemic_frailty": -.55, "nutritional": .30})
    add("Hand grip strength (right)", {"functional": -1.15, "systemic_frailty": -.55, "nutritional": .30})

    # Endocrine/nutritional continua.
    log_multiply("IGF-1", {"endocrine": .045, "nutritional": .025, "systemic_frailty": -.025})
    log_multiply("SHBG", {"endocrine": .060, "metabolic": -.040})
    log_multiply("Testosterone", {"endocrine": .055, "systemic_frailty": -.020})
    log_multiply("Oestradiol", {"endocrine": .050})
    add("Vitamin D", {"nutritional": 4.5, "functional": -1.0})
    add("Total protein", {"nutritional": .55, "inflammatory": .35, "hepatic": -.20})
    add("Calcium", {"nutritional": .018, "renal": -.008})
    add("Urate", {"renal": 10.0, "metabolic": 8.0, "nutritional": 2.0})

    return changes


def state_summary(state: np.ndarray) -> pd.DataFrame:
    rows = []
    for i, name in enumerate(DOMAINS):
        x = state[:, i].astype(np.float64)
        rows.append({
            "domain": name, "mean": float(x.mean()), "std": float(x.std()),
            "p01": float(np.quantile(x, .01)), "p25": float(np.quantile(x, .25)),
            "median": float(np.median(x)), "p75": float(np.quantile(x, .75)),
            "p95": float(np.quantile(x, .95)), "p99": float(np.quantile(x, .99)),
            "fraction_gt_1": float(np.mean(x > 1.0)),
            "fraction_gt_2": float(np.mean(x > 2.0)),
        })
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve().parent
    parser.add_argument("--g0-script", type=Path, default=here / DEFAULT_G0)
    parser.add_argument("--schema", type=Path, default=here / DEFAULT_SCHEMA)
    parser.add_argument("--output-dir", type=Path, default=Path(f"label_free_population_{VERSION}"))
    parser.add_argument("--n", type=int, default=DEFAULT_N)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--schema-name-column-index", type=int, default=0)
    parser.add_argument("--audit-rows", type=int, default=50_000)
    parser.add_argument("--write-csv", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.n <= 0:
        raise ValueError("--n must be positive")
    for path in [args.g0_script, args.schema]:
        if not path.exists():
            raise FileNotFoundError(path)
    if args.output_dir.exists():
        if not args.overwrite and any(args.output_dir.iterdir()):
            raise FileExistsError(f"Non-empty output exists: {args.output_dir}; pass --overwrite")
        if args.overwrite:
            shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    g0 = load_g0(args.g0_script)
    rng = np.random.default_rng(args.seed)

    # The frozen G0 function requires a DiseaseSpec argument.  This adapter has
    # effectively zero prevalence, no risk terms, no effects, no treatment, and
    # a non-disease key.  Runtime assertions require that its disease-related
    # arrays are identically zero before they are discarded.
    inert = g0.DiseaseSpec(
        name="Inert population baseline adapter", abbr="POPULATION_BASE",
        prevalence=1e-12, sensitivity=0.0, false_positive_rate=0.0,
        treatment_logit=-30.0, risk_terms={}, interactions={},
        distinction="No disease construct; adapter only for frozen G0 baseline semantics.",
        manifest_effects={},
    )
    generated = g0.generate_population(args.n, rng, inert)
    if int(np.sum(generated["true_disease"])) != 0:
        raise RuntimeError("Inert adapter unexpectedly sampled a disease state")
    if int(np.sum(generated["target"])) != 0 or float(np.sum(generated["treated"])) != 0.0:
        raise RuntimeError("Inert adapter produced a target/treatment state")

    state, state_protocol = generate_states(args.n, rng)
    physiology_changes = apply_continuous_physiology(generated["values"], g0.norm, state)

    schema, schema_names, metadata = g0.load_schema(args.schema, args.schema_name_column_index)
    feature_names = g0.ordered_unique(g0.EXTRA_REQUIRED_COLUMNS + schema_names)
    data: dict[str, np.ndarray] = {}
    map_rows: list[dict[str, Any]] = []
    constraint_rows: list[dict[str, Any]] = []
    canonical_by_output: dict[str, str | None] = {}
    unresolved: list[str] = []
    fallback_base = g0.z(generated["values"][g0.norm("Age at recruitment")])

    for output_name in feature_names:
        row_meta = metadata.get(output_name, {})
        canonical, proposed, recognition = g0.identify_variable(output_name, row_meta)
        info = generated["meta"].get(canonical or "")
        if info is None or canonical not in generated["values"]:
            canonical_by_output[output_name] = None
            raw = .12 * fallback_base + rng.normal(0, 1, args.n)
            info = {
                "interpreted_name": proposed, "variable_type": "continuous",
                "unit": "unknown", "distribution": "schema-constrained fallback",
                "physiological_group": "unresolved", "decimals": 3,
                "notes": "No disease effect; weak age-correlated fallback.",
            }
            unresolved.append(output_name)
        else:
            canonical_by_output[output_name] = canonical
            raw = generated["values"][canonical].copy()
        constrained, constraint_info = g0.apply_v1_schema_constraints(raw, row_meta, info, rng)
        if not np.isfinite(constrained).all():
            raise RuntimeError(f"Non-finite final values for {output_name}")
        data[output_name] = constrained.astype(np.float32)
        map_rows.append({
            "output_column": output_name,
            "interpreted_name": info.get("interpreted_name", proposed),
            "variable_type": info.get("variable_type", "unknown"),
            "unit": info.get("unit", "unknown"),
            "physiological_group": info.get("physiological_group", "unresolved"),
            "recognition_source": recognition,
            "continuous_program_effect": any(
                row["status"] == "applied" and g0.norm(row["variable"]) == canonical
                for row in physiology_changes
            ),
        })
        constraint_rows.append({"output_column": output_name, **constraint_info})

    dataset = pd.DataFrame(data, columns=feature_names)
    matrix = dataset.to_numpy(dtype=np.float32, copy=True)
    if dataset.shape != (args.n, len(feature_names)):
        raise RuntimeError(f"Unexpected shape: {dataset.shape}")
    if not np.isfinite(matrix).all() or dataset.isna().any().any():
        raise RuntimeError("Final matrix must be finite and complete")
    forbidden_exact = set(g0.DISEASES) | {spec.abbr for spec in g0.DISEASES.values()}
    leaked = [name for name in feature_names if name in forbidden_exact]
    if leaked:
        raise RuntimeError(f"Disease label columns leaked into output: {leaked}")

    # Frozen training artifacts.
    np.save(args.output_dir / "synthetic_features_float32.npy", matrix, allow_pickle=False)
    (args.output_dir / "feature_columns.json").write_text(
        json.dumps(feature_names, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if args.write_csv:
        dataset.to_csv(args.output_dir / "synthetic_population.csv", index=False, encoding="utf-8-sig")

    pd.DataFrame(map_rows).to_csv(args.output_dir / "variable_manifest.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(constraint_rows).to_csv(args.output_dir / "schema_constraint_audit.csv", index=False, encoding="utf-8-sig")
    summary = state_summary(state)
    summary.to_csv(args.output_dir / "latent_state_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(np.corrcoef(state, rowvar=False), index=DOMAINS, columns=DOMAINS).to_csv(
        args.output_dir / "latent_state_correlation.csv", encoding="utf-8-sig"
    )
    audit_n = min(args.audit_rows, args.n)
    audit_rng = np.random.default_rng(args.seed + 991)
    audit_idx = np.sort(audit_rng.choice(args.n, size=audit_n, replace=False))
    pd.DataFrame(state[audit_idx], columns=DOMAINS).to_csv(
        args.output_dir / "latent_state_audit_sample.csv", index=False, encoding="utf-8-sig"
    )
    feature_audit = dataset.iloc[audit_idx].describe(percentiles=[.01, .05, .25, .5, .75, .95, .99]).T
    feature_audit.to_csv(args.output_dir / "synthetic_feature_descriptives.csv", encoding="utf-8-sig")
    dataset.iloc[audit_idx].corr(method="spearman").to_csv(
        args.output_dir / "synthetic_feature_spearman_audit.csv", encoding="utf-8-sig"
    )

    shutil.copy2(args.g0_script, args.output_dir / args.g0_script.name)
    shutil.copy2(args.schema, args.output_dir / args.schema.name)
    shutil.copy2(Path(__file__).resolve(), args.output_dir / Path(__file__).name)

    parameters = {
        "version": VERSION,
        "protocol": PROTOCOL,
        "purpose": "Unlabeled synthetic population for representation learning",
        "knowledge_boundary": {
            "participant_level_real_data_read": False,
            "real_outcome_labels_read": False,
            "predictive_metrics_used_for_tuning": False,
            "aggregate_schema_constraints_used": True,
        },
        "sample": {"n": args.n, "seed": args.seed, "feature_count": len(feature_names)},
        "label_policy": {
            "disease_columns": [], "diagnosis_columns": [], "incidence_time_columns": [],
            "severity_columns": [], "treatment_columns": [],
            "inert_g0_adapter_asserted_all_zero_then_discarded": True,
        },
        "joint_population_model": state_protocol,
        "continuous_physiology_changes": physiology_changes,
        "schema": {
            "path": str(args.schema), "sha256": sha256_file(args.schema),
            "unresolved_columns": unresolved,
            "no_missing": True, "schema_constraint_engine": "Frozen G0/V1 final-scale constraints",
        },
        "source": {"g0_path": str(args.g0_script), "g0_sha256": sha256_file(args.g0_script)},
        "outputs": {
            "matrix": "synthetic_features_float32.npy",
            "columns": "feature_columns.json",
            "csv": "synthetic_population.csv" if args.write_csv else None,
        },
    }
    (args.output_dir / "generation_parameters.json").write_text(
        json.dumps(json_ready(parameters), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    result = {
        "version": VERSION, "n": args.n, "feature_count": len(feature_names),
        "output_dir": str(args.output_dir), "unresolved_count": len(unresolved),
        "disease_label_column_count": 0,
        "matrix_sha256": sha256_file(args.output_dir / "synthetic_features_float32.npy"),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
