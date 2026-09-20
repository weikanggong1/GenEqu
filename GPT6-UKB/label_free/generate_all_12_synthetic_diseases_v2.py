#!/usr/bin/env python3
"""Generate synthetic cohorts for all 12 prespecified diseases from one dynamic CSV schema.

This is the V2 hierarchical latent-physiology generator with three V1 safeguards
ported into the final output layer: (1) sex is consistently coded 0=male,
1=female; (2) all stochastic and structural missingness is disabled and every
predictor must be finite; and (3) schema hard bounds, allowed values, decimal
precision, and anti-collapse regeneration are enforced before serialization.

By default one population-based cohort is generated per disease. Disease
liability is driven by latent physiology, true disease is followed by severity
and treatment, and observed biomarkers are then generated. No predictive metric
is used to tune generation parameters.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import re
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import ndtr
from scipy.stats import chi2


SCHEMA_CSV = "all_variable_to_generate_with_constraints.csv"
SCHEMA_NAME_COLUMN_INDEX = 0
# Latest task-level override: age and sex already exist in the schema as
# Field_ID 21022 and 31, so no duplicate convenience columns are added.
EXTRA_REQUIRED_COLUMNS: list[str] = []
REQUIRE_NO_MULTIVARIATE_SCHEMA_ROWS = True
N_SYNTHETIC = 10000
RANDOM_SEED = 2026
COHORT_MODE = "population_based"
TARGET_DEFINITION = "0=control, 1=disease"
SEX_DEFINITION = "0=male, 1=female"


@dataclass(frozen=True)
class DiseaseSpec:
    name: str
    abbr: str
    prevalence: float
    sensitivity: float
    false_positive_rate: float
    treatment_logit: float
    risk_terms: dict[str, float]
    interactions: dict[str, tuple[str, str, float]]
    distinction: str
    manifest_effects: dict[str, tuple[str, str, str]]


DISEASES: dict[str, DiseaseSpec] = {
    "T2D": DiseaseSpec(
        "Type 2 diabetes", "T2D", 0.105, 0.95, 0.004, 0.85,
        {"age_z": 0.58, "insulin_resistance": 0.92, "adiposity": 0.30,
         "hepatic_fat": 0.23, "inflammation": 0.16, "deprivation": 0.14,
         "sleep_disruption": 0.14, "low_activity": 0.17,
         "family_metabolic": 0.24, "unmeasured": 0.55, "male": 0.12},
        {"age_x_ir": ("age_z", "insulin_resistance", 0.14)},
        "Established T2D with imperfect ascertainment and heterogeneous glucose-lowering treatment.",
        {"glycaemic": ("higher", "strong", "near-diagnostic marker/disease consequence"),
         "adiposity": ("higher", "moderate", "risk/correlated marker"),
         "lipid": ("mixed dyslipidaemia", "moderate", "correlated metabolic marker"),
         "liver": ("higher enzymes", "weak-to-moderate", "correlated metabolic marker"),
         "renal": ("worse renal profile", "weak", "possible consequence/correlate"),
         "activity": ("lower", "weak-to-moderate", "risk/correlated behaviour"),
         "family_metabolic": ("higher burden", "weak", "risk marker")}),
    "T2D_broad": DiseaseSpec(
        "Type 2 diabetes, broad definition", "T2D_broad", 0.145, 0.97, 0.008, 0.35,
        {"age_z": 0.48, "insulin_resistance": 0.80, "adiposity": 0.28,
         "hepatic_fat": 0.18, "inflammation": 0.13, "deprivation": 0.13,
         "sleep_disruption": 0.12, "low_activity": 0.16,
         "family_metabolic": 0.22, "unmeasured": 0.58, "male": 0.10},
        {"age_x_ir": ("age_z", "insulin_resistance", 0.10)},
        "Broader case construct including milder/earlier glycaemic disease and less frequently treated cases.",
        {"glycaemic": ("higher", "strong", "near-diagnostic marker/disease consequence"),
         "adiposity": ("higher", "moderate", "risk/correlated marker"),
         "lipid": ("mixed dyslipidaemia", "moderate", "correlated metabolic marker"),
         "activity": ("lower", "weak-to-moderate", "risk/correlated behaviour")}),
    "CKD": DiseaseSpec(
        "Chronic kidney disease", "CKD", 0.085, 0.92, 0.006, 0.10,
        {"age_z": 0.75, "renal_impairment": 0.78, "bp_load": 0.34,
         "insulin_resistance": 0.25, "inflammation": 0.18, "smoking": 0.12,
         "family_renal": 0.22, "unmeasured": 0.58},
        {"age_x_renal": ("age_z", "renal_impairment", 0.15)},
        "Chronic renal impairment with heterogeneous stage and incomplete diagnosis; no eGFR field is added unless present in the schema.",
        {"renal": ("worse renal profile", "strong", "disease consequence/near-diagnostic marker"),
         "blood_count": ("anaemia tendency", "weak-to-moderate", "possible disease consequence"),
         "blood_pressure": ("higher/mixed under treatment", "moderate", "risk factor/correlate"),
         "mineral": ("mildly altered", "weak", "possible consequence")}),
    "Hypercholesterolaemia": DiseaseSpec(
        "Hypercholesterolaemia", "Hypercholesterolaemia", 0.170, 0.93, 0.008, 0.35,
        {"age_z": 0.30, "lipid_load": 0.92, "adiposity": 0.18,
         "poor_diet": 0.16, "family_lipid": 0.38, "unmeasured": 0.50},
        {}, "Predominantly elevated LDL/ApoB phenotype with lipid-lowering treatment producing current-value overlap.",
        {"lipid": ("higher LDL/ApoB; mixed under treatment", "strong", "defining marker/risk phenotype"),
         "adiposity": ("slightly higher", "weak", "correlated risk marker"),
         "diet": ("less favourable", "weak", "correlated behaviour")}),
    "Hyperlipidaemia": DiseaseSpec(
        "Hyperlipidaemia", "Hyperlipidaemia", 0.220, 0.94, 0.010, 0.20,
        {"age_z": 0.27, "lipid_load": 0.65, "insulin_resistance": 0.42,
         "adiposity": 0.30, "poor_diet": 0.16, "family_lipid": 0.28,
         "unmeasured": 0.53},
        {}, "Broad dyslipidaemia combining cholesterol- and triglyceride-predominant phenotypes.",
        {"lipid": ("atherogenic/mixed dyslipidaemia", "strong", "defining marker/risk phenotype"),
         "glycaemic": ("slightly higher", "weak", "correlated metabolic marker"),
         "adiposity": ("higher", "moderate", "risk/correlated marker")}),
    "Gout": DiseaseSpec(
        "Gout", "Gout", 0.045, 0.94, 0.003, 0.25,
        {"age_z": 0.48, "male": 0.85, "urate_burden": 1.00,
         "renal_impairment": 0.30, "adiposity": 0.20, "alcohol": 0.18,
         "purine_diet": 0.14, "family_gout": 0.25, "unmeasured": 0.52},
        {"urate_x_renal": ("urate_burden", "renal_impairment", 0.13)},
        "Crystal arthritis liability driven by chronic urate burden; current urate is treatment- and timing-sensitive.",
        {"urate": ("higher/mixed under treatment", "strong", "near-diagnostic correlated marker"),
         "renal": ("worse renal profile", "moderate", "risk/correlated marker"),
         "alcohol": ("higher", "weak", "risk behaviour"),
         "adiposity": ("higher", "weak-to-moderate", "risk/correlated marker")}),
    "Liver_disease": DiseaseSpec(
        "Liver disease", "Liver_disease", 0.055, 0.90, 0.006, -0.45,
        {"age_z": 0.20, "hepatic_fat": 0.72, "alcohol": 0.42,
         "adiposity": 0.28, "insulin_resistance": 0.25,
         "viral_toxic_liver": 0.38, "unmeasured": 0.58},
        {"alcohol_x_liver": ("alcohol", "hepatic_fat", 0.12)},
        "Mixed metabolic, alcohol-related and viral/toxic liver disease rather than a single-enzyme threshold.",
        {"liver": ("higher enzymes/bilirubin; lower albumin", "strong", "disease consequence/diagnostic correlate"),
         "alcohol": ("higher", "moderate", "risk behaviour"),
         "adiposity": ("higher", "moderate", "risk/correlated marker"),
         "blood_count": ("mixed; platelets lower in advanced disease", "weak", "possible consequence")}),
    "IHD": DiseaseSpec(
        "Ischaemic heart disease", "IHD", 0.075, 0.92, 0.006, 0.70,
        {"age_z": 0.88, "male": 0.44, "lipid_load": 0.48,
         "bp_load": 0.42, "smoking": 0.32, "insulin_resistance": 0.26,
         "inflammation": 0.20, "family_cardiac": 0.28,
         "cardiac_susceptibility": 0.35, "unmeasured": 0.55},
        {"age_x_cardiac": ("age_z", "cardiac_susceptibility", 0.11)},
        "Clinically manifest myocardial ischaemia; preventive treatment attenuates current lipid and blood-pressure differences.",
        {"lipid": ("mixed/lower under treatment", "moderate", "risk factor/treatment-influenced correlate"),
         "blood_pressure": ("higher/mixed under treatment", "moderate", "risk factor/treatment-influenced correlate"),
         "smoking": ("higher burden", "moderate", "risk behaviour"),
         "inflammation": ("higher", "weak-to-moderate", "correlated marker")}),
    "CHD": DiseaseSpec(
        "Coronary heart disease", "CHD", 0.085, 0.94, 0.007, 0.62,
        {"age_z": 0.80, "male": 0.40, "lipid_load": 0.44,
         "bp_load": 0.38, "smoking": 0.29, "insulin_resistance": 0.24,
         "inflammation": 0.18, "family_cardiac": 0.28,
         "cardiac_susceptibility": 0.31, "unmeasured": 0.57},
        {}, "Broader documented coronary disease construct, including stable coronary disease and prior revascularisation.",
        {"lipid": ("mixed/lower under treatment", "moderate", "risk factor/treatment-influenced correlate"),
         "blood_pressure": ("higher/mixed under treatment", "moderate", "risk factor/treatment-influenced correlate"),
         "smoking": ("higher burden", "moderate", "risk behaviour")}),
    "Hypertension": DiseaseSpec(
        "Hypertension", "Hypertension", 0.300, 0.95, 0.012, 0.45,
        {"age_z": 0.95, "bp_load": 0.82, "adiposity": 0.38,
         "renal_impairment": 0.22, "salt": 0.18, "alcohol": 0.12,
         "family_bp": 0.28, "unmeasured": 0.45},
        {"age_x_bp": ("age_z", "bp_load", 0.10)},
        "Underlying hypertensive liability with antihypertensive treatment and current blood-pressure overlap.",
        {"blood_pressure": ("higher/mixed under treatment", "strong", "defining marker/treatment-influenced consequence"),
         "adiposity": ("higher", "moderate", "risk marker"),
         "renal": ("slightly worse", "weak", "risk/correlated marker"),
         "diet": ("higher salt pattern", "weak", "risk behaviour")}),
    "Heart_failure": DiseaseSpec(
        "Heart failure", "Heart_failure", 0.035, 0.90, 0.004, 1.05,
        {"age_z": 1.05, "cardiac_susceptibility": 0.62, "bp_load": 0.32,
         "renal_impairment": 0.32, "insulin_resistance": 0.24,
         "adiposity": 0.20, "smoking": 0.18, "inflammation": 0.24,
         "family_cardiac": 0.18, "unmeasured": 0.58},
        {"age_x_cardiac": ("age_z", "cardiac_susceptibility", 0.16)},
        "Heterogeneous heart failure without BNP/echocardiography; functional, renal and inflammatory consequences remain overlapping.",
        {"functional": ("lower", "moderate", "disease consequence"),
         "renal": ("worse", "moderate", "disease consequence/correlate"),
         "inflammation": ("higher", "weak-to-moderate", "disease consequence/correlate"),
         "blood_pressure": ("mixed under treatment", "weak-to-moderate", "risk/treatment-influenced marker"),
         "activity": ("lower", "moderate", "disease consequence/correlate")}),
    "Osteoporosis": DiseaseSpec(
        "Osteoporosis", "Osteoporosis", 0.065, 0.90, 0.005, -0.15,
        {"age_z": 1.02, "female": 0.74, "bone_loss": 0.72,
         "low_body_mass": 0.26, "smoking": 0.18, "low_activity": 0.18,
         "low_vitamin_d": 0.18, "family_bone": 0.28, "unmeasured": 0.55},
        {"age_x_female": ("age_z", "female", 0.42)},
        "Low-bone-strength phenotype; serum calcium is weak because homeostatic regulation is preserved.",
        {"demographic": ("older and more often female", "strong", "risk predictor"),
         "bone": ("lower bone-strength profile", "strong", "latent risk construct"),
         "functional": ("lower grip/more falls", "moderate", "consequence/correlated marker"),
         "vitamin_d": ("lower", "weak-to-moderate", "risk/correlated marker"),
         "mineral": ("near-zero/mixed", "near-zero", "homeostatically regulated marker")}),
}


def norm(value: Any) -> str:
    text = str(value).strip().lower().replace("haemoglobin", "hemoglobin")
    return re.sub(r"[^a-z0-9]+", "", text)


def z(x: np.ndarray) -> np.ndarray:
    sd = float(np.nanstd(x))
    return (x - float(np.nanmean(x))) / (sd if sd else 1.0)


def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -35.0, 35.0)))


def ordered_unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def calibrate_intercept(score: np.ndarray, prevalence: float) -> float:
    low, high = -18.0, 8.0
    for _ in range(100):
        mid = (low + high) / 2.0
        if float(np.mean(sigmoid(mid + score))) < prevalence:
            low = mid
        else:
            high = mid
    return (low + high) / 2.0


def extract_unit(prompt: str) -> str:
    if "***" not in prompt:
        return "coded category" if prompt else "unspecified"
    suffix = prompt.split("***", 1)[1].strip(" .:%")
    return suffix if suffix else ("%" if "%" in prompt else "coded category")


def load_schema(path: Path, index: int) -> tuple[pd.DataFrame, list[str], dict[str, dict[str, str]]]:
    schema = pd.read_csv(path, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    if index < 0 or index >= schema.shape[1]:
        raise ValueError(f"SCHEMA_NAME_COLUMN_INDEX={index} unavailable for {schema.shape[1]} columns")
    if REQUIRE_NO_MULTIVARIATE_SCHEMA_ROWS and "Variable_Type" in schema.columns:
        multivariate = schema["Variable_Type"].astype(str).str.contains("multivariate", regex=False, na=False)
        if bool(multivariate.any()):
            bad = schema.loc[multivariate, schema.columns[index]].astype(str).tolist()
            raise ValueError(f"Schema still contains multivariate predictors, contrary to the current task: {bad}")
    names: list[str] = []
    metadata: dict[str, dict[str, str]] = {}
    name_col = schema.columns[index]
    for _, row in schema.iterrows():
        raw = str(row.iloc[index])
        if not raw.strip():
            continue
        names.append(raw)
        if raw not in metadata:
            metadata[raw] = {str(k): str(v) for k, v in row.to_dict().items() if k != name_col}
    return schema, ordered_unique(names), metadata


def infer_disease_key(script_path: Path) -> str:
    stem = script_path.stem
    prefix = "generate_synthetic_"
    candidate = stem[len(prefix):] if stem.startswith(prefix) else ""
    if candidate in DISEASES:
        return candidate
    return "T2D"


def safe_float_array(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=float)


def categorical_from_score(score: np.ndarray, cuts: list[float]) -> np.ndarray:
    return np.digitize(score, cuts).astype(float)


def clamp_round(x: np.ndarray, low: float, high: float, decimals: int) -> np.ndarray:
    return np.round(np.clip(x, low, high), decimals)


def _safe_float(value: Any, default: float = math.nan) -> float:
    try:
        if value is None or str(value).strip() == "":
            return default
        return float(str(value).replace(",", ""))
    except Exception:
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(round(float(value)))
    except Exception:
        return default


def parse_allowed_values(value: Any) -> list[float]:
    """Parse V1-style Recommended/Observed allowed-value metadata."""
    if value is None or str(value).strip() == "":
        return []
    raw = str(value).strip()
    try:
        parsed = ast.literal_eval(raw)
        if isinstance(parsed, (list, tuple, set)):
            return sorted({float(x) for x in parsed})
    except Exception:
        pass
    # Robust fallback for simple comma/semicolon-separated numeric metadata.
    stripped = raw.strip("[](){}")
    pieces = re.split(r"[;,|]", stripped)
    out: list[float] = []
    for piece in pieces:
        try:
            if piece.strip() != "":
                out.append(float(piece.strip()))
        except Exception:
            return []
    return sorted(set(out))


def _meta_first_float(meta: dict[str, str], keys: tuple[str, ...], default: float = math.nan) -> float:
    for key in keys:
        value = _safe_float(meta.get(key), math.nan)
        if np.isfinite(value):
            return value
    return default


def schema_allowed(meta: dict[str, str]) -> list[float]:
    for key in ("Recommended_Allowed_Values", "Observed_Allowed_Values"):
        values = parse_allowed_values(meta.get(key))
        if values:
            return values
    return []


def schema_bounds(meta: dict[str, str]) -> tuple[float, float]:
    lo = _meta_first_float(meta, ("Recommended_Hard_Min", "Observed_Min"), math.nan)
    hi = _meta_first_float(meta, ("Recommended_Hard_Max", "Observed_Max"), math.nan)
    if np.isfinite(lo) and np.isfinite(hi) and hi < lo:
        lo, hi = hi, lo
    return lo, hi


def schema_quantiles(meta: dict[str, str]) -> tuple[float, float, float, float, float]:
    lo, hi = schema_bounds(meta)
    p01 = _meta_first_float(meta, ("Observed_P01",), lo)
    med = _meta_first_float(meta, ("Observed_Median",), (lo + hi) / 2 if np.isfinite(lo) and np.isfinite(hi) else math.nan)
    p99 = _meta_first_float(meta, ("Observed_P99",), hi)
    vals = np.asarray([lo, p01, med, p99, hi], dtype=float)
    finite = vals[np.isfinite(vals)]
    if finite.size == 0:
        return (math.nan,) * 5
    # Fill missing anchors conservatively from neighbouring/central support.
    fallback = float(np.median(finite))
    vals = np.where(np.isfinite(vals), vals, fallback)
    vals = np.maximum.accumulate(vals)
    return tuple(float(x) for x in vals)


def resolve_schema_decimals(meta: dict[str, str], semantic_default: int) -> tuple[int, str]:
    """Port V1 precision resolution: increase decimals if integer rounding collapses real support."""
    raw = meta.get("Recommended_Decimal_Places", "")
    dec = _safe_int(raw, semantic_default) if str(raw).strip() != "" else int(semantic_default)
    dec = max(0, min(dec, 8))
    lo, hi = schema_bounds(meta)
    omin = _safe_float(meta.get("Observed_Min"), math.nan)
    omax = _safe_float(meta.get("Observed_Max"), math.nan)
    chosen = dec
    reason = "schema/default precision retained"
    if dec == 0 and np.isfinite(lo) and np.isfinite(hi) and lo < hi and math.floor(lo) == math.floor(hi):
        demonstrated = (np.isfinite(omin) and np.isfinite(omax) and omax > omin) or hi > lo
        if demonstrated:
            chosen = 1
            while chosen < 8 and math.floor((hi - lo) * (10 ** chosen) + 1e-9) < 2:
                chosen += 1
            reason = "anti-collapse precision increased from 0 decimals to preserve multiple representable values"
    return chosen, reason


def metadata_supports_variability(meta: dict[str, str]) -> bool:
    allowed = schema_allowed(meta)
    if len(allowed) > 1:
        return True
    omin = _safe_float(meta.get("Observed_Min"), math.nan)
    omax = _safe_float(meta.get("Observed_Max"), math.nan)
    if np.isfinite(omin) and np.isfinite(omax) and omax > omin:
        return True
    lo, hi = schema_bounds(meta)
    return bool(np.isfinite(lo) and np.isfinite(hi) and hi > lo)


def _fill_nonfinite(values: np.ndarray, meta: dict[str, str]) -> tuple[np.ndarray, int, float]:
    """Disable structural/stochastic missingness without inventing out-of-range sentinels."""
    x = np.asarray(values, dtype=float).copy()
    bad = ~np.isfinite(x)
    n_bad = int(bad.sum())
    if n_bad == 0:
        return x, 0, math.nan
    allowed = schema_allowed(meta)
    med = _meta_first_float(meta, ("Observed_Median",), math.nan)
    if allowed:
        if np.isfinite(med):
            fill = min(allowed, key=lambda a: abs(a - med))
        elif 0.0 in allowed:
            fill = 0.0
        else:
            fill = float(allowed[len(allowed) // 2])
    elif np.isfinite(med):
        fill = med
    else:
        finite = x[np.isfinite(x)]
        if finite.size:
            fill = float(np.median(finite))
        else:
            lo, hi = schema_bounds(meta)
            if np.isfinite(lo) and np.isfinite(hi):
                fill = float((lo + hi) / 2)
            elif np.isfinite(lo):
                fill = float(lo)
            elif np.isfinite(hi):
                fill = float(hi)
            else:
                fill = 0.0
    x[bad] = fill
    return x, n_bad, float(fill)


def _metadata_quantile_regenerate(score: np.ndarray, meta: dict[str, str], rng: np.random.Generator) -> np.ndarray:
    """V1-style metadata-grounded regeneration used only when final-scale collapse is detected."""
    s = np.asarray(score, dtype=float)
    if not np.isfinite(s).all() or float(np.std(s)) < 1e-12:
        s = rng.normal(size=len(s))
    else:
        s = z(s)
    allowed = schema_allowed(meta)
    q = np.clip(ndtr(s), 1e-12, 1 - 1e-12)
    if allowed:
        a = np.asarray(allowed, dtype=float)
        med = _meta_first_float(meta, ("Observed_Median",), float(a[len(a)//2]))
        p01 = _meta_first_float(meta, ("Observed_P01",), float(a[0]))
        p99 = _meta_first_float(meta, ("Observed_P99",), float(a[-1]))
        mid = int(np.argmin(np.abs(a - med)))
        i01 = int(np.argmin(np.abs(a - p01)))
        i99 = int(np.argmin(np.abs(a - p99)))
        sigma = max((i99 - i01) / 4.652, 0.60)
        ranks = np.arange(len(a))
        weights = np.exp(-0.5 * ((ranks - mid) / sigma) ** 2)
        weights = np.maximum(weights, 0.002 * np.max(weights))
        cdf = np.cumsum(weights / np.sum(weights))
        idx = np.minimum(np.searchsorted(cdf, q, side="right"), len(a) - 1)
        return a[idx]
    lo, p01, med, p99, hi = schema_quantiles(meta)
    if not np.all(np.isfinite([lo, p01, med, p99, hi])) or hi <= lo:
        return s
    knots = np.array([0.0, 0.01, 0.5, 0.99, 1.0])
    vals = np.array([lo, p01, med, p99, hi], dtype=float)
    dist_text = " ".join(str(meta.get(k, "")) for k in ("Suggested_Marginal_Distribution", "Distribution", "Prompts")).lower()
    loglike = any(token in dist_text for token in ("right-skewed", "log-normal", "lognormal", "gamma")) and np.all(vals > 0)
    if loglike:
        return np.exp(np.interp(q, knots, np.log(vals)))
    return np.interp(q, knots, vals)


def apply_v1_schema_constraints(values: np.ndarray, row_meta: dict[str, str], info: dict[str, Any],
                                rng: np.random.Generator) -> tuple[np.ndarray, dict[str, Any]]:
    """Final V1-style constraint + anti-collapse layer applied to V2 semantic values."""
    x, filled_nonfinite, fill_value = _fill_nonfinite(values, row_meta)
    raw_for_regeneration = x.copy()
    semantic_decimals = int(info.get("decimals", 3))
    decimals, precision_reason = resolve_schema_decimals(row_meta, semantic_decimals)
    allowed = schema_allowed(row_meta)
    lo, hi = schema_bounds(row_meta)

    if allowed:
        arr = np.asarray(allowed, dtype=float)
        x = arr[np.abs(x[:, None] - arr[None, :]).argmin(axis=1)]
    else:
        if np.isfinite(lo):
            x = np.maximum(x, lo)
        if np.isfinite(hi):
            x = np.minimum(x, hi)
        x = np.round(x, decimals)

    unique_before_repair = int(np.unique(x).size)
    p01 = _safe_float(row_meta.get("Observed_P01"), math.nan)
    p99 = _safe_float(row_meta.get("Observed_P99"), math.nan)
    supports = metadata_supports_variability(row_meta)
    near_collapse = unique_before_repair == 1 or (
        not allowed and np.isfinite(p01) and np.isfinite(p99) and p99 > p01 and unique_before_repair < 3
    )
    repaired = False

    if supports and near_collapse:
        repaired = True
        x = _metadata_quantile_regenerate(raw_for_regeneration, row_meta, rng)
        if allowed:
            arr = np.asarray(allowed, dtype=float)
            x = arr[np.abs(x[:, None] - arr[None, :]).argmin(axis=1)]
        else:
            if np.isfinite(lo):
                x = np.maximum(x, lo)
            if np.isfinite(hi):
                x = np.minimum(x, hi)
            x = np.round(x, decimals)

    final_unique = int(np.unique(x).size)
    still_collapsed = supports and (final_unique == 1 or (
        not allowed and np.isfinite(p01) and np.isfinite(p99) and p99 > p01 and final_unique < 3
    ))
    if still_collapsed:
        raise RuntimeError(
            f"Anti-collapse failed: metadata supports variability but only {final_unique} final value(s) remain"
        )
    if not np.isfinite(x).all():
        raise RuntimeError("Non-finite values remain after no-missing/schema-constraint processing")

    return x.astype(float), {
        "filled_nonfinite_count": filled_nonfinite,
        "fill_value_used": None if not np.isfinite(fill_value) else fill_value,
        "schema_allowed_value_count": len(allowed),
        "schema_hard_min": None if not np.isfinite(lo) else lo,
        "schema_hard_max": None if not np.isfinite(hi) else hi,
        "selected_decimal_places": decimals,
        "precision_resolution": precision_reason,
        "metadata_supports_variability": supports,
        "unique_before_anti_collapse": unique_before_repair,
        "anti_collapse_regenerated": repaired,
        "final_unique_values": final_unique,
    }


def stable_disease_seed(base_seed: int, disease_key: str) -> int:
    digest = hashlib.sha256(disease_key.encode("utf-8")).digest()
    return int((base_seed + int.from_bytes(digest[:8], "little")) % (2**32 - 1))


def generate_population(n: int, rng: np.random.Generator, disease: DiseaseSpec) -> dict[str, Any]:
    """Generate latent physiology, disease, treatment, and 225 semantic variables."""
    e = rng.normal(size=(n, 48))
    older_component = rng.random(n) < 0.30
    age = np.where(older_component, rng.normal(69.0, 6.4, n), rng.normal(57.0, 8.0, n))
    age = clamp_round(age, 40.0, 82.0, 1)
    sex = rng.binomial(1, 0.52, n).astype(float)  # 0=male, 1=female
    age_z = z(age)
    male = 1.0 - sex
    female = sex
    male_z = z(male)

    deprivation = z(e[:, 0] + 0.10 * age_z)
    education = z(-0.58 * deprivation - 0.10 * age_z + 0.72 * e[:, 1])
    activity = z(0.34 * education - 0.25 * deprivation - 0.18 * age_z + 0.75 * e[:, 2])
    sedentary = z(-0.42 * activity + 0.22 * deprivation + 0.12 * age_z + 0.72 * e[:, 3])
    diet_quality = z(0.35 * education - 0.25 * deprivation + 0.18 * activity + 0.76 * e[:, 4])
    salt = z(-0.18 * diet_quality + 0.83 * e[:, 5])
    smoking = z(0.34 * deprivation - 0.22 * education + 0.15 * male_z + 0.78 * e[:, 6])
    alcohol = z(0.22 * male_z + 0.10 * education + 0.84 * e[:, 7])
    sleep_disruption = z(0.22 * deprivation + 0.18 * sedentary + 0.76 * e[:, 8])
    adiposity = z(0.30 * deprivation - 0.27 * activity - 0.18 * diet_quality + 0.20 * age_z + 0.72 * e[:, 9])
    muscle = z(0.48 * male_z + 0.28 * activity - 0.20 * age_z + 0.72 * e[:, 10])
    inflammation = z(0.36 * adiposity + 0.18 * smoking + 0.16 * sleep_disruption + 0.64 * e[:, 11])
    insulin_resistance = z(0.58 * adiposity + 0.20 * age_z - 0.18 * activity + 0.18 * e[:, 12] + 0.48 * e[:, 13])
    hepatic_fat = z(0.46 * insulin_resistance + 0.28 * alcohol + 0.20 * adiposity + 0.56 * e[:, 14])
    renal_impairment = z(0.40 * age_z + 0.18 * insulin_resistance + 0.15 * inflammation + 0.62 * e[:, 15])
    poor_diet = -diet_quality
    low_activity = -activity
    lipid_load = z(0.32 * insulin_resistance + 0.16 * poor_diet + 0.68 * e[:, 16])
    bp_load = z(0.38 * age_z + 0.28 * adiposity + 0.18 * salt + 0.12 * alcohol + 0.58 * e[:, 17])
    bone_loss = z(0.45 * age_z + 0.28 * female * np.maximum(age_z, 0) - 0.22 * activity - 0.16 * e[:, 18] + 0.58 * e[:, 19])
    lung_impairment = z(0.36 * smoking + 0.22 * age_z - 0.18 * activity + 0.62 * e[:, 20])
    cardiac_susceptibility = z(0.35 * lipid_load + 0.30 * bp_load + 0.24 * smoking + 0.20 * insulin_resistance + 0.58 * e[:, 21])
    urate_burden = z(0.42 * renal_impairment + 0.32 * male_z + 0.25 * insulin_resistance + 0.18 * alcohol + 0.58 * e[:, 22])
    sun_exposure = z(0.25 * activity - 0.20 * adiposity + 0.78 * e[:, 23])
    viral_toxic_liver = e[:, 24]
    immune = e[:, 25]
    bilirubin_genetic = e[:, 26]
    lpa_genetic = e[:, 27]
    family_metabolic = e[:, 28]
    family_renal = 0.35 * family_metabolic + 0.94 * e[:, 29]
    family_lipid = 0.40 * family_metabolic + 0.92 * e[:, 30]
    family_gout = 0.30 * family_renal + 0.95 * e[:, 31]
    family_cardiac = 0.36 * family_lipid + 0.30 * family_metabolic + 0.88 * e[:, 32]
    family_bp = 0.35 * family_cardiac + 0.94 * e[:, 33]
    family_bone = e[:, 34]
    unmeasured = e[:, 35]
    missingness = z(0.20 * deprivation + 0.16 * age_z + 0.70 * e[:, 36])
    reproductive = e[:, 37]
    allergy = e[:, 38]
    hematology = e[:, 39]
    platelet_factor = e[:, 40]
    erythrocyte_factor = e[:, 41]
    nutrition = z(0.22 * diet_quality - 0.14 * deprivation + 0.82 * e[:, 42])
    occupational = e[:, 43]
    social = z(-0.35 * deprivation + 0.22 * education + 0.80 * e[:, 44])
    purine_diet = z(-0.22 * diet_quality + 0.18 * alcohol + 0.82 * e[:, 45])
    low_body_mass = z(-adiposity)
    low_vitamin_d = z(-sun_exposure + 0.25 * adiposity)

    factors = locals().copy()
    score = np.zeros(n)
    for key, coefficient in disease.risk_terms.items():
        score += coefficient * safe_float_array(factors[key])
    for _, (left, right, coefficient) in disease.interactions.items():
        score += coefficient * safe_float_array(factors[left]) * safe_float_array(factors[right])
    intercept = calibrate_intercept(score, disease.prevalence)
    probability = sigmoid(intercept + score)
    true_disease = rng.binomial(1, probability, n).astype(int)
    severity_latent = z(0.35 * score + 0.70 * e[:, 46])
    severity = true_disease * np.clip(0.72 + 0.34 * severity_latent, 0.08, 1.80)
    treatment_probability = sigmoid(disease.treatment_logit + 0.32 * age_z + 0.35 * severity_latent)
    treated = ((true_disease == 1) & (rng.random(n) < treatment_probability)).astype(float)
    treatment_intensity = treated * np.clip(rng.beta(2.3, 2.0, n), 0.05, 0.98)
    observed = np.where(
        true_disease == 1,
        rng.random(n) < disease.sensitivity,
        rng.random(n) < disease.false_positive_rate,
    ).astype(int)

    # Consequence multipliers are continuous, heterogeneous, and attenuated by
    # treatment where clinically appropriate.  They never determine the label.
    dsev = severity
    key = disease.abbr
    glucose_shift = hba1c_shift = 0.0
    renal_shift = liver_shift = inflammation_shift = 0.0
    lipid_ldl_shift = lipid_tg_shift = hdl_shift = 0.0
    bp_systolic_shift = bp_diastolic_shift = 0.0
    urate_shift = functional_shift = anemia_shift = platelet_shift = 0.0
    if key == "T2D":
        glucose_shift = dsev * 1.60 - treated * (0.80 + 0.85 * treatment_intensity)
        hba1c_shift = dsev * 12.0 - treated * (4.8 + 5.5 * treatment_intensity)
        renal_shift, liver_shift, inflammation_shift = 0.08 * dsev, 0.10 * dsev, 0.08 * dsev
        lipid_tg_shift, hdl_shift = 0.20 * dsev, -0.08 * dsev
    elif key == "T2D_broad":
        glucose_shift = dsev * 1.10 - treated * (0.55 + 0.55 * treatment_intensity)
        hba1c_shift = dsev * 7.2 - treated * (3.0 + 3.5 * treatment_intensity)
        renal_shift, liver_shift = 0.05 * dsev, 0.07 * dsev
        lipid_tg_shift, hdl_shift = 0.15 * dsev, -0.06 * dsev
    elif key == "CKD":
        renal_shift, anemia_shift = 0.34 * dsev, -0.12 * dsev
        inflammation_shift = 0.10 * dsev
    elif key == "Hypercholesterolaemia":
        lipid_ldl_shift = 0.52 * dsev - treated * (0.38 + 0.35 * treatment_intensity)
        lipid_tg_shift = 0.06 * dsev
    elif key == "Hyperlipidaemia":
        lipid_ldl_shift = 0.34 * dsev - treated * (0.28 + 0.28 * treatment_intensity)
        lipid_tg_shift = 0.34 * dsev - treated * (0.10 + 0.12 * treatment_intensity)
        hdl_shift = -0.10 * dsev
    elif key == "Gout":
        urate_shift = 48.0 * dsev - treated * (30.0 + 42.0 * treatment_intensity)
        inflammation_shift = 0.12 * dsev * (rng.random(n) < 0.12)
    elif key == "Liver_disease":
        liver_shift = 0.52 * dsev
        inflammation_shift, platelet_shift, anemia_shift = 0.10 * dsev, -0.10 * dsev, -0.05 * dsev
    elif key in {"IHD", "CHD"}:
        lipid_ldl_shift = -treated * (0.30 + 0.34 * treatment_intensity)
        bp_systolic_shift = 3.0 * dsev - treated * (4.5 + 5.0 * treatment_intensity)
        bp_diastolic_shift = 1.5 * dsev - treated * (2.5 + 3.0 * treatment_intensity)
        inflammation_shift, functional_shift = 0.08 * dsev, -0.08 * dsev
    elif key == "Hypertension":
        bp_systolic_shift = 12.0 * dsev - treated * (7.0 + 8.0 * treatment_intensity)
        bp_diastolic_shift = 6.5 * dsev - treated * (4.0 + 4.5 * treatment_intensity)
        renal_shift = 0.04 * dsev
    elif key == "Heart_failure":
        renal_shift, inflammation_shift = 0.20 * dsev, 0.20 * dsev
        functional_shift = -0.32 * dsev
        anemia_shift = -0.10 * dsev
        bp_systolic_shift = 2.0 * dsev - treated * (4.0 + 4.0 * treatment_intensity)
    elif key == "Osteoporosis":
        functional_shift = -0.24 * dsev

    values: dict[str, np.ndarray] = {}
    meta: dict[str, dict[str, Any]] = {}

    def put(description: str, value: np.ndarray, variable_type: str, unit: str,
            distribution: str, group: str, decimals: int = 3,
            missing_rate: float = 0.01, levels: str = "",
            derivation: str = "", notes: str = "") -> None:
        canonical = norm(description)
        values[canonical] = safe_float_array(value)
        meta[canonical] = {
            "interpreted_name": description, "variable_type": variable_type,
            "unit": unit, "distribution": distribution, "physiological_group": group,
            "decimals": decimals, "missing_rate": 0.0,
            "category_levels": levels, "derivation_rule": derivation, "notes": notes,
        }

    # Demographics and socioeconomic structure.
    ethnicity = rng.choice(5, n, p=[0.75, 0.07, 0.06, 0.04, 0.08]).astype(float)
    income_score = 0.75 * education - 0.70 * deprivation - 0.22 * np.maximum(age_z, 0) + 0.55 * social + rng.normal(0, 0.70, n)
    income = categorical_from_score(income_score, [-1.0, -0.25, 0.45, 1.20])
    retired = age > (64 + rng.normal(0, 3.0, n))
    disabled = rng.random(n) < sigmoid(-3.7 + 0.55 * age_z + 0.40 * inflammation + 0.35 * dsev)
    unemployed = (~retired) & (~disabled) & (rng.random(n) < sigmoid(-2.4 + 0.45 * deprivation))
    employment = np.where(retired, 1, np.where(disabled, 3, np.where(unemployed, 2, 0))).astype(float)
    accommodation_type = categorical_from_score(-0.40 * deprivation + 0.20 * social + rng.normal(size=n), [-0.8, 0.4, 1.2])
    tenure = categorical_from_score(-0.62 * deprivation + 0.25 * age_z + rng.normal(0, 0.9, n), [-0.7, 0.4])
    household_n = clamp_round(1.4 + 0.55 * (age < 62) + 0.30 * social + rng.poisson(0.65, n), 1, 7, 0)
    live_with = np.where(household_n <= 1, 0, np.where(household_n >= 3, 2, 1)).astype(float)
    vehicles = clamp_round(np.maximum(0, 0.8 + 0.55 * income - 0.32 * deprivation + rng.normal(0, 0.7, n)), 0, 5, 0)
    address_years = clamp_round(np.exp(np.log(10.0) + 0.33 * age_z + 0.18 * tenure + rng.normal(0, 0.65, n)), 0.1, 60, 1)
    solid_fuel = (rng.random(n) < sigmoid(-2.8 + 0.55 * deprivation + 0.18 * age_z)).astype(float)
    commute = np.where(employment == 0, np.clip(rng.lognormal(np.log(8.0), 0.85, n), 0.2, 85), np.nan)
    job_years = np.where(employment == 0, np.clip(age - 19 - np.abs(rng.normal(8, 7, n)), 0.2, 50), np.nan)
    work_hours = np.where(employment == 0, np.clip(rng.normal(37 + 2 * male + 2 * occupational, 9, n), 4, 75), np.nan)
    commute_freq = np.where(employment == 0, np.clip(np.round(rng.normal(4.2, 1.2, n)), 0, 7), np.nan)
    job_walk = np.where(employment == 0, categorical_from_score(0.45 * activity + 0.45 * occupational + rng.normal(size=n), [-0.6, 0.5]), np.nan)
    job_heavy = np.where(employment == 0, categorical_from_score(0.35 * activity + 0.70 * occupational + rng.normal(size=n), [-0.7, 0.7]), np.nan)
    shift_work = np.where(employment == 0, categorical_from_score(0.30 * deprivation + rng.normal(size=n), [0.45, 1.35]), np.nan)
    qualifications = categorical_from_score(education + rng.normal(0, 0.45, n), [-1.0, -0.25, 0.55, 1.25])
    education_age = clamp_round(16.5 + 2.1 * education + rng.normal(0, 1.2, n), 12, 30, 0)
    disability_allowance = (disabled & (rng.random(n) < 0.72)).astype(float)
    private_health = (rng.random(n) < sigmoid(-1.0 + 0.45 * income + 0.30 * education)).astype(float)
    put("Age at recruitment", age, "continuous", "years", "truncated two-component normal mixture", "demographic", 1, 0.0, derivation="Same construct as extra predictor age.")
    put("Gender", sex, "binary", SEX_DEFINITION, "Bernoulli", "demographic", 0, 0.0, "0=male; 1=female", "Same construct as schema sex predictor; coding is fixed globally.")
    put("Ethnic background", ethnicity, "categorical", "coded category", "multinomial", "demographic", 0, 0.01, "0=majority European; 1=South Asian; 2=Black; 3=East Asian; 4=mixed/other", notes="Synthetic broad categories; not calibrated to a named cohort.")
    put("Average total household income before tax", income, "categorical", "coded income band", "ordinal latent-score categories", "socioeconomic", 0, 0.03, "0=<18k; 1=18-30k; 2=30-52k; 3=52-100k; 4=>100k")
    put("Type of accommodation lived in", accommodation_type, "categorical", "coded category", "multinomial/ordinal approximation", "socioeconomic", 0, 0.02, "0=flat/other; 1=terraced/semi-detached; 2=detached; 3=institution/other")
    put("Own or rent accommodation lived in", tenure, "categorical", "coded category", "ordinal latent-score categories", "socioeconomic", 0, 0.02, "0=rent/other; 1=mortgage; 2=own outright")
    put("Length of time at current address", address_years, "continuous", "years", "bounded log-normal", "socioeconomic", 1, 0.02)
    put("Number in household", household_n, "count", "people", "bounded Poisson mixture", "socioeconomic", 0, 0.02)
    put("How are people in household related to participant", live_with, "categorical", "coded category", "structural categorical summary", "socioeconomic", 0, 0.03, "0=lives alone; 1=partner/one other; 2=family/multiple", notes="A single summary code approximates a potentially multi-select field.")
    put("Gas or solid-fuel cooking/heating", solid_fuel, "binary", "0=no, 1=yes", "Bernoulli", "environment", 0, 0.02)
    put("Number of vehicles in household", vehicles, "count", "vehicles", "bounded count", "socioeconomic", 0, 0.02)
    put("Current employment status", employment, "categorical", "coded category", "age- and health-dependent multinomial", "socioeconomic", 0, 0.01, "0=employed; 1=retired; 2=not employed; 3=disabled")
    put("Distance between home and job workplace", commute, "continuous", "miles", "structurally missing bounded log-normal", "socioeconomic", 2, 0.01, notes="Missing when not currently employed.")
    put("Time employed in main current job", job_years, "continuous", "years", "structurally missing bounded distribution", "socioeconomic", 1, 0.01, notes="Missing when not currently employed.")
    put("Length of working week for main job", work_hours, "continuous", "hours", "structurally missing bounded normal", "socioeconomic", 1, 0.01, notes="Missing when not currently employed.")
    put("Frequency of travelling from home to job workplace", commute_freq, "count", "times/week", "structurally missing bounded count", "socioeconomic", 0, 0.01, notes="Missing when not currently employed.")
    put("Job involves mainly walking or standing", job_walk, "categorical", "coded category", "structurally missing ordinal", "activity", 0, 0.02, "0=never/rarely; 1=sometimes; 2=usually", notes="Missing when not currently employed.")
    put("Job involves heavy manual or physical work", job_heavy, "categorical", "coded category", "structurally missing ordinal", "activity", 0, 0.02, "0=never/rarely; 1=sometimes; 2=usually", notes="Missing when not currently employed.")
    put("Job involves shift work", shift_work, "categorical", "coded category", "structurally missing ordinal", "sleep", 0, 0.02, "0=never/rarely; 1=sometimes; 2=usually", notes="Missing when not currently employed.")
    put("Qualifications", qualifications, "categorical", "coded category", "ordinal latent-score categories", "socioeconomic", 0, 0.02, "0=none; 1=school; 2=vocational; 3=college; 4=degree")
    put("Age completed full time education", education_age, "continuous", "years", "bounded normal", "socioeconomic", 0, 0.04)
    put("Attendance/disability/mobility allowance", disability_allowance, "binary", "0=no, 1=yes", "Bernoulli", "general_health", 0, 0.02)
    put("Private healthcare", private_health, "binary", "0=no, 1=yes", "Bernoulli", "socioeconomic", 0, 0.04)

    # Physical measurements and internally coherent body composition.
    ethnicity_height = np.choose(ethnicity.astype(int), [0.0, -2.0, 0.5, -1.0, -0.5])
    height = clamp_round(163.5 + 13.0 * male + ethnicity_height - 1.1 * age_z + 5.7 * e[:, 47], 143, 201, 1)
    bmi = np.clip(25.8 + 4.2 * adiposity + 0.8 * age_z + rng.normal(0, 1.25, n), 16.0, 47.0)
    weight = np.clip(bmi * (height / 100.0) ** 2 + rng.normal(0, 0.45, n), 42, 165)
    waist = np.clip(48.0 + 1.62 * bmi + 7.0 * male + 2.0 * insulin_resistance + rng.normal(0, 4.5, n), 58, 155)
    hip = np.clip(69.0 + 1.37 * bmi + 2.0 * female + rng.normal(0, 3.8, n), 70, 150)
    sitting_height = np.clip(0.52 * height + 1.0 * male + rng.normal(0, 2.2, n), 72, 105)
    body_fat_pct = np.clip(27.5 - 8.5 * male + 0.92 * (bmi - 25) + 0.85 * age_z + rng.normal(0, 2.0, n), 7, 52)
    whole_fat = weight * body_fat_pct / 100.0
    whole_ffm = np.maximum(weight - whole_fat, 28)
    body_water = np.clip(0.73 * whole_ffm + rng.normal(0, 0.75, n), 24, 75)
    trunk_total = weight * np.clip(0.49 + 0.018 * adiposity + rng.normal(0, 0.009, n), 0.43, 0.56)
    leg_total_each = weight * np.clip(0.155 - 0.009 * adiposity + rng.normal(0, 0.005, n), 0.125, 0.19)
    arm_total_each = weight * np.clip(0.052 + 0.006 * male + rng.normal(0, 0.003, n), 0.038, 0.072)
    trunk_fat = trunk_total * np.clip((body_fat_pct + 4.0 + 1.3 * adiposity) / 100.0, 0.06, 0.60)
    trunk_ffm = trunk_total - trunk_fat
    asym_leg = rng.normal(0, 0.015, n)
    leg_left_total, leg_right_total = leg_total_each * (1 + asym_leg), leg_total_each * (1 - asym_leg)
    leg_fat_fraction = np.clip((body_fat_pct + 1.5 * female - 1.0 * male) / 100.0, 0.05, 0.58)
    leg_fat_left = leg_left_total * leg_fat_fraction
    leg_fat_right = leg_right_total * leg_fat_fraction
    leg_ffm_left, leg_ffm_right = leg_left_total - leg_fat_left, leg_right_total - leg_fat_right
    asym_arm = rng.normal(0, 0.02, n)
    arm_left_total, arm_right_total = arm_total_each * (1 + asym_arm), arm_total_each * (1 - asym_arm)
    arm_fat_fraction = np.clip((body_fat_pct - 2.0) / 100.0, 0.04, 0.52)
    arm_fat_left = arm_left_total * arm_fat_fraction
    arm_fat_right = arm_right_total * arm_fat_fraction
    arm_ffm_left, arm_ffm_right = arm_left_total - arm_fat_left, arm_right_total - arm_fat_right
    bmr = np.clip(4.184 * (370 + 21.6 * whole_ffm), 4100, 10500)
    whole_impedance = np.clip(790 - 7.2 * body_water - 1.2 * weight + rng.normal(0, 28, n), 260, 850)
    arm_imp_base = np.clip(390 + 0.42 * whole_impedance - 4.0 * arm_total_each + rng.normal(0, 22, n), 250, 850)
    leg_imp_base = np.clip(340 + 0.38 * whole_impedance - 2.0 * leg_total_each + rng.normal(0, 20, n), 220, 750)
    systolic = np.clip(121 + 10.0 * bp_load + 0.28 * (age - 60) + 2.0 * male + bp_systolic_shift + rng.normal(0, 8.5, n), 82, 225)
    diastolic = np.clip(76 + 6.8 * bp_load - 0.10 * np.maximum(age - 65, 0) + 1.2 * male + bp_diastolic_shift + rng.normal(0, 5.4, n), 48, 128)
    systolic = np.maximum(systolic, diastolic + 15)
    pulse = np.clip(68 + 4.5 * inflammation + 3.0 * sleep_disruption - 2.5 * activity + 4.0 * (-functional_shift) + rng.normal(0, 7.5, n), 40, 135)
    grip_base = 25.0 + 15.0 * male + 4.8 * muscle - 2.4 * age_z + 6.0 * functional_shift
    grip_left = np.clip(grip_base + rng.normal(0, 2.5, n), 7, 70)
    grip_right = np.clip(grip_base + 1.2 + rng.normal(0, 2.5, n), 7, 72)
    fvc = np.clip(3.35 + 1.05 * male + 0.025 * (height - 165) - 0.30 * age_z - 0.32 * lung_impairment + 0.15 * functional_shift + rng.normal(0, 0.34, n), 1.0, 7.5)
    fev1 = np.clip(fvc * np.clip(0.79 - 0.035 * lung_impairment - 0.015 * age_z + rng.normal(0, 0.025, n), 0.42, 0.94), 0.55, 6.5)
    ratio_z = np.clip(((fev1 / fvc) - (0.79 - 0.012 * age_z)) / 0.065, -5.0, 3.5)
    pef = np.clip(90 + 95 * fev1 + 18 * male - 13 * lung_impairment + rng.normal(0, 28, n), 90, 850)
    birth_weight = np.clip(3.35 + 0.12 * male - 0.18 * (ethnicity == 1) + 0.10 * nutrition + rng.normal(0, 0.42, n), 1.2, 5.5)

    put("Diastolic blood pressure, automated reading", diastolic, "continuous", "mmHg", "bounded normal with shared blood-pressure factor", "blood_pressure", 1, 0.01)
    put("Systolic blood pressure, automated reading", systolic, "continuous", "mmHg", "bounded normal with shared blood-pressure factor", "blood_pressure", 1, 0.01)
    put("Pulse rate, automated reading", pulse, "continuous", "bpm", "bounded normal", "cardiovascular", 1, 0.01)
    put("Hand grip strength (left)", grip_left, "continuous", "kg", "bounded correlated normal", "functional", 1, 0.01)
    put("Hand grip strength (right)", grip_right, "continuous", "kg", "bounded correlated normal", "functional", 1, 0.01)
    put("Waist circumference", waist, "continuous", "cm", "bounded normal", "adiposity", 1, 0.01)
    put("Weight", weight, "continuous", "kg", "BMI/height-conditioned bounded distribution", "adiposity", 1, 0.005)
    put("Body mass index (BMI)", bmi, "continuous", "kg/m^2", "bounded normal mixture", "adiposity", 2, 0.005, derivation="Approximately weight/height^2 with measurement variation.")
    put("Hip circumference", hip, "continuous", "cm", "bounded normal", "adiposity", 1, 0.01)
    put("Standing height", height, "continuous", "cm", "sex- and ancestry-conditioned bounded normal", "body_size", 1, 0.005)
    put("Sitting height", sitting_height, "continuous", "cm", "height-conditioned normal", "body_size", 1, 0.01)

    segment_specs = [
        ("Leg fat percentage (left)", 100 * leg_fat_left / leg_left_total, "%", "body_composition"),
        ("Leg fat percentage (right)", 100 * leg_fat_right / leg_right_total, "%", "body_composition"),
        ("Leg fat mass (left)", leg_fat_left, "kg", "body_composition"),
        ("Leg fat mass (right)", leg_fat_right, "kg", "body_composition"),
        ("Leg fat-free mass (left)", leg_ffm_left, "kg", "body_composition"),
        ("Leg fat-free mass (right)", leg_ffm_right, "kg", "body_composition"),
        ("Leg predicted mass (left)", leg_left_total, "kg", "body_composition"),
        ("Leg predicted mass (right)", leg_right_total, "kg", "body_composition"),
        ("Arm fat percentage (left)", 100 * arm_fat_left / arm_left_total, "%", "body_composition"),
        ("Arm fat percentage (right)", 100 * arm_fat_right / arm_right_total, "%", "body_composition"),
        ("Arm fat mass (left)", arm_fat_left, "kg", "body_composition"),
        ("Arm fat mass (right)", arm_fat_right, "kg", "body_composition"),
        ("Arm fat-free mass (right)", arm_ffm_right, "kg", "body_composition"),
        ("Arm fat-free mass (left)", arm_ffm_left, "kg", "body_composition"),
        ("Arm predicted mass (left)", arm_left_total, "kg", "body_composition"),
        ("Arm predicted mass (right)", arm_right_total, "kg", "body_composition"),
        ("Trunk fat percentage", 100 * trunk_fat / trunk_total, "%", "body_composition"),
        ("Trunk fat mass", trunk_fat, "kg", "body_composition"),
        ("Trunk fat-free mass", trunk_ffm, "kg", "body_composition"),
        ("Trunk predicted mass", trunk_total, "kg", "body_composition"),
        ("Basal metabolic rate", bmr, "kJ/day", "body_composition"),
        ("Body fat percentage", body_fat_pct, "%", "adiposity"),
        ("Whole body fat mass", whole_fat, "kg", "adiposity"),
        ("Whole body fat-free mass", whole_ffm, "kg", "body_composition"),
        ("Whole body water mass", body_water, "kg", "body_composition"),
        ("Impedance of whole body", whole_impedance, "ohms", "body_composition"),
        ("Impedance of arm (left)", arm_imp_base * (1 + asym_arm), "ohms", "body_composition"),
        ("Impedance of arm (right)", arm_imp_base * (1 - asym_arm), "ohms", "body_composition"),
        ("Impedance of leg (left)", leg_imp_base * (1 + asym_leg), "ohms", "body_composition"),
        ("Impedance of leg (right)", leg_imp_base * (1 - asym_leg), "ohms", "body_composition"),
    ]
    for desc, val, unit, group in segment_specs:
        derivation = ""
        if "predicted mass" in desc:
            derivation = "Segment fat mass + segment fat-free mass."
        elif "fat percentage" in desc:
            derivation = "100 × segment fat mass / segment predicted mass."
        elif desc == "Body fat percentage":
            derivation = "100 × whole body fat mass / weight."
        elif desc == "Whole body fat-free mass":
            derivation = "Weight - whole body fat mass."
        notes = "Metadata prompt listed KJ, but mass is generated in kg for clinical dimensional consistency." if desc == "Whole body fat mass" else ""
        put(desc, val, "continuous", unit, "physiologically constrained correlated distribution", group, 2, 0.02, derivation=derivation, notes=notes)
    put("Forced vital capacity (FVC)", fvc, "continuous", "litres", "bounded height/sex/smoking-conditioned normal", "lung", 3, 0.04)
    put("Forced expiratory volume in 1-second (FEV1)", fev1, "continuous", "litres", "FVC-conditioned bounded distribution", "lung", 3, 0.04)
    put("FEV1/ FVC ratio Z-score", ratio_z, "continuous", "z-score", "derived standardized ratio", "lung", 3, 0.04, derivation="Standardized FEV1/FVC ratio with age-dependent expected value.")
    put("Peak expiratory flow (PEF)", pef, "continuous", "litres/min", "FEV1-conditioned bounded normal", "lung", 1, 0.04)
    put("Birth weight", birth_weight, "continuous", "kg", "bounded normal", "early_life", 2, 0.08)

    # Lifestyle variables share activity, sedentary, diet, sleep, smoking,
    # alcohol and sun-exposure factors rather than being independent draws.
    moderate_minutes = np.clip(rng.gamma(2.0, 18.0, n) * np.exp(0.22 * activity), 0, 240)
    vigorous_minutes = np.clip(rng.gamma(1.35, 12.0, n) * np.exp(0.30 * activity - 0.12 * age_z), 0, 180)
    walk_minutes = np.clip(rng.gamma(2.3, 20.0, n) * np.exp(0.22 * activity), 0, 300)
    moderate_days = np.clip(np.round(3.3 + 1.15 * activity + rng.normal(0, 1.5, n)), 0, 7)
    vigorous_days = np.clip(np.round(1.8 + 1.05 * activity - 0.35 * age_z + rng.normal(0, 1.25, n)), 0, 7)
    walk_days = np.clip(np.round(4.7 + 0.90 * activity + rng.normal(0, 1.35, n)), 0, 7)
    heavy_diy_freq = categorical_from_score(0.55 * activity - 0.18 * age_z + rng.normal(size=n), [-0.9, -0.1, 0.8])
    light_diy_freq = categorical_from_score(0.42 * activity + rng.normal(size=n), [-0.9, -0.1, 0.8])
    exercise_freq = categorical_from_score(0.70 * activity - 0.12 * age_z + rng.normal(size=n), [-0.9, -0.1, 0.8])
    pleasure_walk_freq = categorical_from_score(0.52 * activity + rng.normal(size=n), [-0.9, -0.1, 0.8])
    duration_category = lambda latent: categorical_from_score(latent + rng.normal(0, 0.7, n), [-0.8, 0.0, 0.8])
    stair_freq = categorical_from_score(0.45 * activity - 0.25 * age_z + rng.normal(size=n), [-0.8, 0.1, 0.9])
    driving = np.clip(rng.gamma(1.5, 0.60, n) * np.exp(0.26 * sedentary + 0.10 * income), 0, 8)
    computer = np.clip(rng.gamma(1.8, 0.75, n) * np.exp(0.30 * sedentary + 0.12 * education), 0, 12)
    tv = np.clip(rng.gamma(2.2, 0.75, n) * np.exp(0.32 * sedentary + 0.14 * deprivation), 0, 12)
    walking_pace = categorical_from_score(0.70 * activity - 0.30 * age_z - 0.15 * adiposity + 0.35 * functional_shift + rng.normal(0, 0.7, n), [-0.55, 0.55])
    phone_weekly = categorical_from_score(-0.15 * age_z + 0.20 * social + rng.normal(size=n), [-0.8, 0.2, 1.0])
    handsfree = categorical_from_score(0.18 * education - 0.12 * age_z + rng.normal(size=n), [-0.7, 0.4])
    games = categorical_from_score(-0.55 * age_z + 0.20 * male_z + rng.normal(size=n), [-0.2, 0.8])
    sleep_hours = np.clip(7.2 - 0.35 * sleep_disruption - 0.12 * age_z + rng.normal(0, 0.65, n), 3.5, 11.5)
    getting_up = categorical_from_score(-0.18 * sleep_disruption + 0.12 * age_z + rng.normal(size=n), [-0.8, 0.2, 1.0])
    chronotype = categorical_from_score(-0.14 * age_z + rng.normal(size=n), [-0.9, -0.15, 0.65])
    nap = categorical_from_score(0.36 * age_z + 0.40 * sleep_disruption + rng.normal(size=n), [0.1, 1.1])
    insomnia = categorical_from_score(0.70 * sleep_disruption + 0.12 * female + rng.normal(size=n), [-0.2, 0.9])
    snoring = categorical_from_score(0.44 * adiposity + 0.22 * male_z + 0.25 * sleep_disruption + rng.normal(size=n), [0.0, 1.0])
    daytime_dozing = categorical_from_score(0.55 * sleep_disruption + 0.26 * age_z + 0.18 * (-functional_shift) + rng.normal(size=n), [0.0, 1.0])
    current_smoker_prob = sigmoid(-2.35 + 0.68 * smoking - 0.18 * age_z)
    former_smoker_prob = sigmoid(-0.55 + 0.52 * smoking + 0.38 * age_z) * (1 - current_smoker_prob)
    draw = rng.random(n)
    smoking_status = np.where(draw < current_smoker_prob, 2, np.where(draw < current_smoker_prob + former_smoker_prob, 1, 0)).astype(float)
    current_smoking = np.where(smoking_status == 2, categorical_from_score(smoking + rng.normal(size=n), [-0.2, 0.8]) + 1, 0)
    past_smoking = np.where(smoking_status == 1, categorical_from_score(smoking + rng.normal(size=n), [-0.2, 0.8]) + 1, 0)
    pack_years = np.where(smoking_status == 0, 0.0, np.clip(np.exp(np.log(11.0) + 0.48 * smoking + 0.28 * age_z + rng.normal(0, 0.55, n)), 0.2, 120))
    household_smoker = (rng.random(n) < sigmoid(-1.5 + 0.42 * smoking + 0.35 * deprivation)).astype(float)
    smoke_home = np.where(household_smoker == 1, np.clip(rng.gamma(1.3, 2.2, n), 0, 35), 0)
    smoke_outside = np.clip(rng.gamma(1.1, 1.2, n) * np.exp(0.24 * deprivation), 0, 30)

    put("Duration of heavy DIY", duration_category(0.35 * activity), "categorical", "coded duration", "ordinal latent-score categories", "activity", 0, 0.03, "0=<15 min; 1=15-30; 2=30-60; 3=>60")
    put("Frequency of heavy DIY in last 4 weeks", heavy_diy_freq, "categorical", "coded frequency", "ordinal latent-score categories", "activity", 0, 0.03, "0=never; 1=1-2; 2=3-10; 3=>10 occasions")
    put("Duration of light DIY", duration_category(0.28 * activity), "categorical", "coded duration", "ordinal latent-score categories", "activity", 0, 0.03, "0=<15 min; 1=15-30; 2=30-60; 3=>60")
    put("Frequency of light DIY in last 4 weeks", light_diy_freq, "categorical", "coded frequency", "ordinal latent-score categories", "activity", 0, 0.03, "0=never; 1=1-2; 2=3-10; 3=>10 occasions")
    put("Duration of moderate activity", moderate_minutes, "continuous", "minutes/day", "bounded gamma", "activity", 1, 0.025)
    put("Duration of other exercises", duration_category(0.55 * activity), "categorical", "coded duration", "ordinal latent-score categories", "activity", 0, 0.03, "0=<15 min; 1=15-30; 2=30-60; 3=>60")
    put("Frequency of other exercises in last 4 weeks", exercise_freq, "categorical", "coded frequency", "ordinal latent-score categories", "activity", 0, 0.03, "0=never; 1=1-2; 2=3-10; 3=>10 occasions")
    put("Duration of vigorous activity", vigorous_minutes, "continuous", "minutes/day", "bounded gamma", "activity", 1, 0.025)
    put("Duration of walks", walk_minutes, "continuous", "minutes/day", "bounded gamma", "activity", 1, 0.025)
    put("Duration walking for pleasure", duration_category(0.45 * activity), "categorical", "coded duration", "ordinal latent-score categories", "activity", 0, 0.03, "0=<15 min; 1=15-30; 2=30-60; 3=>60")
    put("Frequency of walking for pleasure in last 4 weeks", pleasure_walk_freq, "categorical", "coded frequency", "ordinal latent-score categories", "activity", 0, 0.03, "0=never; 1=1-2; 2=3-10; 3=>10 occasions")
    put("Frequency of stair climbing in last 4 weeks", stair_freq, "categorical", "coded frequency", "ordinal latent-score categories", "activity", 0, 0.03, "0=none; 1=low; 2=moderate; 3=high")
    put("Number of days/week of moderate physical activity 10+ minutes", moderate_days, "count", "days/week", "bounded count", "activity", 0, 0.02)
    put("Number of days/week of vigorous physical activity 10+ minutes", vigorous_days, "count", "days/week", "bounded count", "activity", 0, 0.02)
    put("Number of days/week walked 10+ minutes", walk_days, "count", "days/week", "bounded count", "activity", 0, 0.02)
    put("Time spent driving", driving, "continuous", "hours/day", "bounded gamma", "sedentary", 2, 0.025)
    put("Time spent using computer", computer, "continuous", "hours/day", "bounded gamma", "sedentary", 2, 0.025)
    put("Time spent watching television (TV)", tv, "continuous", "hours/day", "bounded gamma", "sedentary", 2, 0.025)
    put("Usual walking pace", walking_pace, "categorical", "coded category", "ordinal latent-score categories", "functional", 0, 0.02, "0=slow; 1=steady; 2=brisk")
    put("Weekly usage of mobile phone in last 3 months", phone_weekly, "categorical", "coded frequency", "ordinal latent-score categories", "electronic_use", 0, 0.04, "0=rare; 1=1-5; 2=6-20; 3=>20 calls/week")
    put("Hands-free device/speakerphone use with mobile phone in last 3 month", handsfree, "categorical", "coded category", "ordinal latent-score categories", "electronic_use", 0, 0.05, "0=never; 1=sometimes; 2=usually")
    put("Plays computer games", games, "categorical", "coded category", "ordinal latent-score categories", "electronic_use", 0, 0.04, "0=never; 1=sometimes; 2=often")
    put("Sleep duration", sleep_hours, "continuous", "hours/day", "bounded normal", "sleep", 2, 0.02)
    put("Getting up in morning", getting_up, "categorical", "coded ease", "ordinal latent-score categories", "sleep", 0, 0.025, "0=very difficult; 1=difficult; 2=fairly easy; 3=very easy")
    put("Morning/evening person (chronotype)", chronotype, "categorical", "coded chronotype", "ordinal latent-score categories", "sleep", 0, 0.03, "0=definite evening; 1=more evening; 2=more morning; 3=definite morning")
    put("Nap during day", nap, "categorical", "coded frequency", "ordinal latent-score categories", "sleep", 0, 0.02, "0=never/rarely; 1=sometimes; 2=usually")
    put("Sleeplessness / insomnia", insomnia, "categorical", "coded frequency", "ordinal latent-score categories", "sleep", 0, 0.02, "0=never/rarely; 1=sometimes; 2=usually")
    put("Snoring", snoring, "categorical", "coded category", "ordinal latent-score categories", "sleep", 0, 0.04, "0=no; 1=possible; 2=yes")
    put("Daytime dozing / sleeping", daytime_dozing, "categorical", "coded frequency", "ordinal latent-score categories", "sleep", 0, 0.02, "0=never/rarely; 1=sometimes; 2=often")
    put("Smoking status", smoking_status, "categorical", "coded category", "multinomial from shared smoking propensity", "smoking", 0, 0.01, "0=never; 1=previous; 2=current")
    put("Current tobacco smoking", current_smoking, "categorical", "coded frequency", "structural ordinal", "smoking", 0, 0.015, "0=not current; 1=occasional; 2=most days; 3=daily")
    put("Past tobacco smoking", past_smoking, "categorical", "coded frequency", "structural ordinal", "smoking", 0, 0.02, "0=not former; 1=occasional; 2=most days; 3=daily")
    put("Pack years of smoking", pack_years, "continuous", "pack-years", "zero-inflated bounded log-normal", "smoking", 2, 0.03)
    put("Smoking/smokers in household", household_smoker, "binary", "0=no, 1=yes", "Bernoulli", "smoking", 0, 0.02)
    put("Exposure to tobacco smoke at home", smoke_home, "continuous", "hours/week", "zero-inflated gamma", "smoking", 2, 0.03)
    put("Exposure to tobacco smoke outside home", smoke_outside, "continuous", "hours/week", "bounded gamma", "smoking", 2, 0.03)

    vegetable = np.clip(rng.gamma(2.2, 1.25, n) * np.exp(0.15 * diet_quality), 0, 15)
    raw_veg = np.clip(rng.gamma(1.8, 1.15, n) * np.exp(0.17 * diet_quality), 0, 14)
    fresh_fruit = np.clip(rng.gamma(1.9, 1.0, n) * np.exp(0.18 * diet_quality), 0, 12)
    dried_fruit = np.clip(rng.gamma(1.1, 0.55, n) * np.exp(0.12 * diet_quality), 0, 7)
    bread = np.clip(rng.gamma(2.1, 7.0, n) * np.exp(-0.05 * diet_quality), 0, 70)
    cereal = np.clip(rng.gamma(1.5, 2.4, n) * np.exp(0.08 * diet_quality), 0, 21)
    tea = np.clip(rng.gamma(2.0, 1.25, n), 0, 12)
    coffee = np.clip(rng.gamma(1.6, 0.95, n) * np.exp(0.08 * sedentary), 0, 10)
    water = np.clip(rng.gamma(2.2, 1.45, n) * np.exp(0.10 * activity), 0, 15)
    food_freq = lambda score: categorical_from_score(score + rng.normal(size=n), [-1.0, -0.25, 0.55, 1.25])
    hot_temp = categorical_from_score(rng.normal(size=n), [-0.65, 0.70])
    oily_fish = food_freq(0.36 * diet_quality)
    nonoily_fish = food_freq(0.24 * diet_quality)
    processed_meat = food_freq(-0.50 * diet_quality + 0.20 * male_z)
    poultry = food_freq(0.12 * diet_quality)
    beef = food_freq(-0.10 * diet_quality + 0.18 * male_z)
    lamb = food_freq(-0.05 * diet_quality)
    pork = food_freq(-0.10 * diet_quality)
    cheese = food_freq(0.05 * diet_quality)
    salt_added = categorical_from_score(0.70 * salt - 0.18 * diet_quality + rng.normal(size=n), [-0.8, 0.0, 0.8])
    never_eat = categorical_from_score(-0.15 * diet_quality + rng.normal(size=n), [0.9, 1.6])
    diet_cont = [
        ("Cooked vegetable intake", vegetable, "tablespoons/day"), ("Salad / raw vegetable intake", raw_veg, "tablespoons/day"),
        ("Fresh fruit intake", fresh_fruit, "pieces/day"), ("Dried fruit intake", dried_fruit, "pieces/day"),
        ("Bread intake", bread, "slices/week"), ("Cereal intake", cereal, "bowls/week"),
        ("Tea intake", tea, "cups/day"), ("Coffee intake", coffee, "cups/day"), ("Water intake", water, "glasses/day"),
    ]
    for desc, val, unit in diet_cont:
        put(desc, val, "continuous", unit, "bounded gamma", "diet", 2, 0.025)
    put("Hot drink temperature", hot_temp, "categorical", "coded category", "ordinal categorical", "diet", 0, 0.03, "0=warm; 1=hot; 2=very hot")
    for desc, val in [("Oily fish intake", oily_fish), ("Non-oily fish intake", nonoily_fish),
                      ("Processed meat intake", processed_meat), ("Poultry intake", poultry),
                      ("Beef intake", beef), ("Lamb/mutton intake", lamb), ("Pork intake", pork),
                      ("Cheese intake", cheese)]:
        put(desc, val, "categorical", "coded frequency", "ordinal latent-score categories", "diet", 0, 0.025, "0=never; 1=<1/week; 2=1/week; 3=2-4/week; 4=most days")
    put("Salt added to food", salt_added, "categorical", "coded frequency", "ordinal latent-score categories", "diet", 0, 0.025, "0=never; 1=sometimes; 2=usually; 3=always")
    put("Never eat eggs, dairy, wheat, sugar", never_eat, "categorical", "coded summary", "categorical summary", "diet", 0, 0.04, "0=none; 1=one group; 2=multiple groups", notes="A single summary code approximates a potentially multi-select field.")

    alcohol_current = rng.random(n) < sigmoid(1.55 + 0.45 * alcohol - 0.10 * age_z)
    alcohol_former = (~alcohol_current) & (rng.random(n) < sigmoid(-1.4 + 0.30 * alcohol + 0.18 * age_z))
    alcohol_status = np.where(alcohol_current, 2, np.where(alcohol_former, 1, 0)).astype(float)
    alcohol_freq = np.where(alcohol_current, categorical_from_score(0.78 * alcohol + rng.normal(size=n), [-1.1, -0.45, 0.15, 0.75]), 0)
    total_units = np.where(alcohol_current, np.clip(rng.gamma(1.5, 5.0, n) * np.exp(0.30 * alcohol), 0, 70), 0)
    mix = rng.dirichlet([1.7, 1.4, 1.8, 1.0, 0.35], n)
    wine_red, wine_white, beer, spirits, fortified = [total_units * mix[:, i] for i in range(5)]
    alcohol_meals = np.where(alcohol_current, categorical_from_score(0.20 * education + rng.normal(size=n), [-0.45, 0.65]), 0)
    put("Alcohol drinker status", alcohol_status, "categorical", "coded category", "multinomial", "alcohol", 0, 0.01, "0=never; 1=former; 2=current")
    put("Alcohol intake frequency.", alcohol_freq, "categorical", "coded frequency", "structural ordinal", "alcohol", 0, 0.02, "0=none; 1=special occasions; 2=monthly; 3=weekly; 4=several/week; 5=daily")
    for desc, val, unit in [
        ("Average weekly red wine intake", wine_red, "glasses/week"),
        ("Average weekly champagne plus white wine intake", wine_white, "glasses/week"),
        ("Average weekly beer plus cider intake", beer, "pints/week"),
        ("Average weekly spirits intake", spirits, "measures/week"),
        ("Average weekly fortified wine intake", fortified, "glasses/week"),
    ]:
        put(desc, val, "continuous", unit, "zero-inflated gamma/Dirichlet composition", "alcohol", 2, 0.025)
    put("Alcohol usually taken with meals", alcohol_meals, "categorical", "coded category", "structural ordinal", "alcohol", 0, 0.03, "0=not current/never; 1=sometimes; 2=usually")

    summer_out = np.clip(2.4 + 0.65 * sun_exposure + 0.35 * activity + rng.normal(0, 0.8, n), 0, 9)
    winter_out = np.clip(1.3 + 0.45 * sun_exposure + 0.28 * activity + rng.normal(0, 0.6, n), 0, 6)
    skin = categorical_from_score(0.20 * ethnicity + rng.normal(size=n), [-0.8, -0.1, 0.6, 1.3])
    tanning = categorical_from_score(-0.30 * skin + rng.normal(size=n), [-0.8, 0.0, 0.8])
    childhood_sunburn = np.clip(rng.poisson(np.exp(0.35 + 0.22 * sun_exposure - 0.15 * skin), n), 0, 20)
    hair = categorical_from_score(0.15 * ethnicity + rng.normal(size=n), [-1.0, -0.25, 0.5, 1.2])
    facial_age = categorical_from_score(0.55 * age_z + 0.18 * smoking + rng.normal(size=n), [-0.7, 0.45])
    sun_protection = categorical_from_score(0.28 * female + 0.18 * education + rng.normal(size=n), [-0.8, 0.0, 0.8])
    solarium = np.where(female == 1, np.clip(rng.poisson(np.exp(-1.0 + 0.20 * reproductive), n), 0, 15), np.clip(rng.poisson(0.15, n), 0, 8))
    put("Time spend outdoors in summer", summer_out, "continuous", "hours/day", "bounded normal", "sun", 2, 0.025)
    put("Time spent outdoors in winter", winter_out, "continuous", "hours/day", "bounded normal", "sun", 2, 0.025)
    put("Skin colour", skin, "categorical", "coded category", "multinomial approximation", "sun", 0, 0.03, "0=very fair; 1=fair; 2=light/medium; 3=brown; 4=dark")
    put("Ease of skin tanning", tanning, "categorical", "coded category", "ordinal categorical", "sun", 0, 0.03, "0=never; 1=mild; 2=moderate; 3=deep")
    put("Childhood sunburn occasions", childhood_sunburn, "count", "occasions", "bounded Poisson", "sun", 0, 0.04)
    put("Hair colour (natural, before greying)", hair, "categorical", "coded category", "multinomial approximation", "sun", 0, 0.04, "0=blonde; 1=light brown; 2=dark brown; 3=black; 4=red/other")
    put("Facial ageing", facial_age, "categorical", "coded category", "ordinal age/smoking-conditioned", "general_health", 0, 0.035, "0=younger; 1=about age; 2=older")
    put("Use of sun/uv protection", sun_protection, "categorical", "coded frequency", "ordinal categorical", "sun", 0, 0.035, "0=never; 1=sometimes; 2=often; 3=always")
    put("Frequency of solarium/sunlamp use", solarium, "count", "times/year", "zero-inflated bounded Poisson", "sun", 0, 0.05)

    first_sex_age = np.clip(18.5 - 0.35 * social + rng.normal(0, 2.8, n), 12, 40)
    partners = np.clip(np.exp(np.log(3.2) + 0.35 * male_z + 0.25 * social + rng.normal(0, 0.85, n)), 0, 200)
    same_sex = (rng.random(n) < sigmoid(-2.8 + 0.25 * social + 0.15 * reproductive)).astype(float)
    put("Age first had sexual intercourse", first_sex_age, "continuous", "years", "bounded normal", "sexual", 1, 0.07)
    put("Lifetime number of sexual partners", partners, "count", "partners", "bounded log-normal count", "sexual", 0, 0.07)
    put("Ever had same-sex intercourse", same_sex, "binary", "0=no, 1=yes", "Bernoulli", "sexual", 0, 0.07)

    # Early life, family history and general health.
    born_country = np.where(rng.random(n) < 0.82, 0, np.where(rng.random(n) < 0.55, 1, 2)).astype(float)
    breastfed = (rng.random(n) < sigmoid(0.75 + 0.18 * education)).astype(float)
    body10 = categorical_from_score(0.42 * adiposity + 0.38 * e[:, 9] + rng.normal(size=n), [-0.55, 0.55])
    height10 = categorical_from_score((height - np.mean(height)) / np.std(height) + rng.normal(0, 0.65, n), [-0.65, 0.65])
    handedness = rng.choice(3, n, p=[0.88, 0.10, 0.02]).astype(float)
    multiple_birth = (rng.random(n) < 0.025).astype(float)
    maternal_smoking = (rng.random(n) < sigmoid(-1.65 + 0.38 * deprivation + 0.18 * smoking)).astype(float)
    family_score = 0.38 * family_cardiac + 0.25 * family_metabolic + 0.20 * family_bone
    family_illness = categorical_from_score(family_score + rng.normal(size=n), [-0.75, 0.15, 0.95])
    father_alive = rng.random(n) < sigmoid(-0.85 - 0.95 * age_z)
    mother_alive = rng.random(n) < sigmoid(-0.35 - 0.85 * age_z)
    father_lifespan = np.clip(76 + 4.5 * family_cardiac * -0.12 + rng.normal(0, 9.0, n), 45, 103)
    mother_lifespan = np.clip(81 + 4.0 * family_cardiac * -0.10 + rng.normal(0, 8.5, n), 48, 106)
    father_current_age = np.where(father_alive, np.clip(age + 27 + rng.normal(0, 4.0, n), 60, 108), np.nan)
    mother_current_age = np.where(mother_alive, np.clip(age + 25 + rng.normal(0, 4.0, n), 58, 108), np.nan)
    father_death_age = np.where(~father_alive, father_lifespan, np.nan)
    mother_death_age = np.where(~mother_alive, mother_lifespan, np.nan)
    siblings = np.clip(rng.poisson(np.exp(0.52 - 0.14 * (age - 60) / 10), n), 0, 12).astype(float)
    family_nonaccidental_death = (rng.random(n) < sigmoid(-2.0 + 0.38 * family_cardiac + 0.22 * age_z)).astype(float)
    longstanding = (rng.random(n) < sigmoid(-1.25 + 0.58 * age_z + 0.30 * inflammation + 0.78 * dsev)).astype(float)
    falls_prob = sigmoid(-2.1 + 0.55 * age_z + 0.28 * bone_loss + 0.30 * (-functional_shift))
    falls = np.where(rng.random(n) < falls_prob, np.where(rng.random(n) < 0.25 + 0.15 * dsev, 2, 1), 0).astype(float)
    weight_change_score = 0.28 * adiposity + 0.18 * true_disease * severity + rng.normal(size=n)
    weight_change = categorical_from_score(weight_change_score, [-0.75, 0.75])
    put("Country of birth (UK/elsewhere)", born_country, "categorical", "coded category", "multinomial", "early_life", 0, 0.01, "0=country of assessment; 1=other high-income; 2=other", notes="Generic synthetic coding; not tied to a named country cohort.")
    put("Breastfed as a baby", breastfed, "binary", "0=no, 1=yes", "Bernoulli", "early_life", 0, 0.06)
    put("Comparative body size at age 10", body10, "categorical", "coded category", "ordinal latent-score categories", "early_life", 0, 0.05, "0=thinner; 1=average; 2=plumper")
    put("Comparative height size at age 10", height10, "categorical", "coded category", "ordinal latent-score categories", "early_life", 0, 0.05, "0=shorter; 1=average; 2=taller")
    put("Handedness (chirality/laterality)", handedness, "categorical", "coded category", "multinomial", "early_life", 0, 0.015, "0=right; 1=left; 2=ambidextrous")
    put("Part of a multiple birth", multiple_birth, "binary", "0=no, 1=yes", "Bernoulli", "early_life", 0, 0.03)
    put("Maternal smoking around birth", maternal_smoking, "binary", "0=no, 1=yes", "Bernoulli", "early_life", 0, 0.06)
    for desc, val, group in [("Illnesses of father", family_illness, "family_history"),
                             ("Illnesses of mother", np.clip(family_illness + rng.integers(-1, 2, n), 0, 3), "family_history"),
                             ("Illnesses of siblings", np.where(siblings > 0, np.clip(family_illness + rng.integers(-1, 2, n), 0, 3), np.nan), "family_history")]:
        put(desc, val, "categorical", "coded summary", "familial latent-score category", group, 0, 0.03, "0=none reported; 1=cardiometabolic; 2=cancer/neurologic; 3=multiple/other", notes="A single summary code approximates a potentially multi-select field.")
    put("Father's age", father_current_age, "continuous", "years", "structurally missing bounded normal", "family_history", 1, 0.02, notes="Missing when father is deceased or unknown.")
    put("Father's age at death", father_death_age, "continuous", "years", "structurally missing bounded normal", "family_history", 1, 0.02, notes="Missing when father is alive or unknown.")
    put("Mother's age", mother_current_age, "continuous", "years", "structurally missing bounded normal", "family_history", 1, 0.02, notes="Missing when mother is deceased or unknown.")
    put("Mother's age at death", mother_death_age, "continuous", "years", "structurally missing bounded normal", "family_history", 1, 0.02, notes="Missing when mother is alive or unknown.")
    put("Number of siblings", siblings, "count", "siblings", "bounded Poisson", "family_history", 0, 0.02, derivation="Synthetic count representing the schema note's combined sibling fields.")
    put("Non-accidental death in close genetic family", family_nonaccidental_death, "binary", "0=no, 1=yes", "Bernoulli", "family_history", 0, 0.03)
    put("Long-standing illness, disability or infirmity", longstanding, "binary", "0=no, 1=yes", "Bernoulli", "general_health", 0, 0.01)
    put("Falls in the last year", falls, "categorical", "coded category", "ordinal Bernoulli mixture", "functional", 0, 0.02, "0=none; 1=one; 2=multiple")
    put("Weight change compared with 1 year ago", weight_change, "categorical", "coded category", "ordinal latent-score categories", "general_health", 0, 0.02, "0=lost weight; 1=about the same; 2=gained weight")

    # Haematology: cell counts, percentages and indices share conserved totals.
    inflammatory_multiplier = np.exp(0.08 * inflammation_shift)
    wbc = np.clip(np.exp(np.log(6.4) + 0.10 * inflammation + 0.06 * smoking + 0.05 * dsev + rng.normal(0, 0.17, n)) * inflammatory_multiplier, 2.0, 20.0)
    logits = np.column_stack([
        1.15 + 0.28 * inflammation + rng.normal(0, 0.22, n),
        0.65 - 0.16 * inflammation + rng.normal(0, 0.20, n),
        -0.35 + 0.30 * allergy + rng.normal(0, 0.23, n),
        -0.25 + 0.08 * inflammation + rng.normal(0, 0.20, n),
        -1.55 + 0.15 * allergy + rng.normal(0, 0.20, n),
    ])
    exp_logits = np.exp(logits - logits.max(axis=1, keepdims=True))
    leukocyte_props = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    neut_pct, lymph_pct, eos_pct, mono_pct, baso_pct = [100 * leukocyte_props[:, i] for i in range(5)]
    neut_count, lymph_count, eos_count, mono_count, baso_count = [wbc * leukocyte_props[:, i] for i in range(5)]
    mcv = np.clip(89.5 + 2.1 * alcohol + 1.8 * liver_shift + 1.4 * hematology + rng.normal(0, 2.7, n), 72, 112)
    mch = np.clip(29.7 + 0.19 * (mcv - 90) + rng.normal(0, 0.8, n), 22, 36)
    rbc = np.clip(4.45 + 0.48 * male + 0.22 * erythrocyte_factor - 0.12 * age_z + anemia_shift + rng.normal(0, 0.20, n), 2.8, 6.6)
    hemoglobin = np.clip(rbc * mch / 10.0, 8.0, 19.5)
    hematocrit = np.clip(rbc * mcv / 10.0, 25, 58)
    mchc = np.clip(100 * hemoglobin / hematocrit, 28, 38)
    rdw = np.clip(13.1 + 0.42 * inflammation + 0.20 * age_z - 0.35 * anemia_shift + rng.normal(0, 0.55, n), 10.5, 20)
    platelet_count = np.clip(245 + 25 * platelet_factor + 13 * inflammation + 70 * platelet_shift + rng.normal(0, 22, n), 75, 600)
    mpv = np.clip(9.7 - 0.004 * (platelet_count - 245) + rng.normal(0, 0.75, n), 6.5, 14.5)
    plateletcrit = np.clip(platelet_count * mpv / 10000.0, 0.08, 0.60)
    pdw = np.clip(13.0 + 0.55 * platelet_factor + 0.25 * inflammation + rng.normal(0, 0.9, n), 8, 22)
    retic_pct = np.clip(np.exp(np.log(1.35) + 0.12 * erythrocyte_factor - 0.18 * anemia_shift + rng.normal(0, 0.22, n)), 0.3, 4.5)
    retic_count = rbc * retic_pct / 100.0
    irf = np.clip(0.28 + 0.05 * inflammation - 0.04 * anemia_shift + rng.normal(0, 0.055, n), 0.08, 0.65)
    hl_pct = np.clip(100 * (0.18 + 0.30 * irf + rng.normal(0, 0.025, n)), 3, 50)
    hl_count = retic_count * hl_pct / 100.0
    mean_retic_vol = np.clip(mcv + 9.0 + 4.5 * irf + rng.normal(0, 2.0, n), 82, 135)
    mean_sphered_vol = np.clip(mcv - 4.5 + rng.normal(0, 1.8, n), 70, 108)
    nrbc_present = rng.random(n) < sigmoid(-6.0 + 0.55 * inflammation + 0.30 * dsev)
    nrbc_count = np.where(nrbc_present, np.clip(rng.lognormal(-3.8, 0.65, n), 0.005, 0.3), 0)
    nrbc_pct = 100 * nrbc_count / np.maximum(wbc, 0.1)
    blood_specs = [
        ("Basophill count", baso_count, "10^9 cells/L", "continuous", "blood_count", 3, "WBC × basophil percentage / 100."),
        ("Basophill percentage", baso_pct, "%", "continuous", "blood_count", 2, "100 × basophil count / WBC."),
        ("Eosinophill count", eos_count, "10^9 cells/L", "continuous", "blood_count", 3, "WBC × eosinophil percentage / 100."),
        ("Eosinophill percentage", eos_pct, "%", "continuous", "blood_count", 2, "100 × eosinophil count / WBC."),
        ("Haematocrit percentage", hematocrit, "%", "continuous", "blood_count", 2, "RBC count × MCV / 10."),
        ("Haemoglobin concentration", hemoglobin, "g/dL", "continuous", "blood_count", 2, "RBC count × MCH / 10."),
        ("High light scatter reticulocyte count", hl_count, "10^12 cells/L", "continuous", "blood_count", 4, "Reticulocyte count × high-light-scatter percentage / 100."),
        ("High light scatter reticulocyte percentage", hl_pct, "%", "continuous", "blood_count", 2, "100 × high-light-scatter reticulocyte count / reticulocyte count."),
        ("Immature reticulocyte fraction", irf, "ratio", "continuous", "blood_count", 3, ""),
        ("Lymphocyte count", lymph_count, "10^9 cells/L", "continuous", "blood_count", 3, "WBC × lymphocyte percentage / 100."),
        ("Lymphocyte percentage", lymph_pct, "%", "continuous", "blood_count", 2, "100 × lymphocyte count / WBC."),
        ("Mean corpuscular haemoglobin", mch, "pg", "continuous", "blood_count", 2, "10 × haemoglobin / RBC count."),
        ("Mean corpuscular haemoglobin concentration", mchc, "g/dL", "continuous", "blood_count", 2, "100 × haemoglobin / haematocrit."),
        ("Mean corpuscular volume", mcv, "fL", "continuous", "blood_count", 2, "10 × haematocrit / RBC count."),
        ("Mean platelet (thrombocyte) volume", mpv, "fL", "continuous", "blood_count", 2, ""),
        ("Mean reticulocyte volume", mean_retic_vol, "fL", "continuous", "blood_count", 2, ""),
        ("Mean sphered cell volume", mean_sphered_vol, "fL", "continuous", "blood_count", 2, ""),
        ("Monocyte count", mono_count, "10^9 cells/L", "continuous", "blood_count", 3, "WBC × monocyte percentage / 100."),
        ("Monocyte percentage", mono_pct, "%", "continuous", "blood_count", 2, "100 × monocyte count / WBC."),
        ("Neutrophill count", neut_count, "10^9 cells/L", "continuous", "blood_count", 3, "WBC × neutrophil percentage / 100."),
        ("Neutrophill percentage", neut_pct, "%", "continuous", "blood_count", 2, "100 × neutrophil count / WBC."),
        ("Nucleated red blood cell count", nrbc_count, "10^9 cells/L", "continuous", "blood_count", 4, "WBC × nucleated-RBC percentage / 100."),
        ("Nucleated red blood cell percentage", nrbc_pct, "%", "continuous", "blood_count", 4, "100 × nucleated-RBC count / WBC."),
        ("Platelet count", platelet_count, "10^9 cells/L", "continuous", "blood_count", 1, ""),
        ("Platelet crit", plateletcrit, "%", "continuous", "blood_count", 3, "Platelet count × mean platelet volume / 10,000."),
        ("Platelet distribution width", pdw, "%", "continuous", "blood_count", 2, ""),
        ("Red blood cell (erythrocyte) count", rbc, "10^12 cells/L", "continuous", "blood_count", 3, ""),
        ("Red blood cell (erythrocyte) distribution width", rdw, "%", "continuous", "blood_count", 2, ""),
        ("Reticulocyte count", retic_count, "10^12 cells/L", "continuous", "blood_count", 4, "RBC count × reticulocyte percentage / 100."),
        ("Reticulocyte percentage", retic_pct, "%", "continuous", "blood_count", 3, "100 × reticulocyte count / RBC count."),
        ("White blood cell (leukocyte) count", wbc, "10^9 cells/L", "continuous", "blood_count", 3, "Sum of major leukocyte subtype counts."),
    ]
    for desc, val, unit, vtype, group, decimals, derivation in blood_specs:
        distribution = "physiologically constrained correlated distribution"
        if "Nucleated" in desc:
            distribution = "zero-inflated log-normal/derived"
        put(desc, val, vtype, unit, distribution, group, decimals, 0.012, derivation=derivation)

    # Blood biochemistry: shared hepatic, renal, metabolic, inflammatory,
    # nutritional, genetic and hormonal factors create joint structure.
    albumin = np.clip(45.0 + 0.85 * nutrition - 0.45 * inflammation - 0.18 * age_z - 0.55 * liver_shift - 0.25 * renal_shift + rng.normal(0, 1.8, n), 29, 54)
    alt = np.clip(np.exp(np.log(22.0) + 0.23 * hepatic_fat + 0.15 * insulin_resistance + 0.09 * male + 0.46 * liver_shift + rng.normal(0, 0.33, n)), 4, 350)
    ast = np.clip(np.exp(np.log(23.5) + 0.14 * hepatic_fat + 0.16 * alcohol + 0.36 * liver_shift + rng.normal(0, 0.27, n)), 7, 320)
    alp = np.clip(np.exp(np.log(73.0) + 0.13 * bone_loss + 0.10 * hepatic_fat + 0.18 * liver_shift + 0.05 * age_z + rng.normal(0, 0.22, n)), 24, 420)
    hdl = np.clip(np.exp(np.log(1.38) - 0.18 * insulin_resistance - 0.12 * male - 0.07 * hepatic_fat + hdl_shift + rng.normal(0, 0.17, n)), 0.30, 3.4)
    apoa = np.clip(0.88 + 0.43 * hdl + 0.05 * female + 0.06 * nutrition - 0.03 * inflammation + rng.normal(0, 0.18, n), 0.50, 2.7)
    apob = np.clip(np.exp(np.log(0.98) + 0.17 * lipid_load + 0.07 * insulin_resistance + 0.46 * lipid_ldl_shift + rng.normal(0, 0.14, n)), 0.30, 2.4)
    triglycerides = np.clip(np.exp(np.log(1.35) + 0.29 * insulin_resistance + 0.15 * hepatic_fat + 0.09 * alcohol + lipid_tg_shift + rng.normal(0, 0.32, n)), 0.25, 12)
    ldl = np.clip(0.90 + 1.95 * apob + 0.15 * lipid_load + 1.05 * lipid_ldl_shift + rng.normal(0, 0.43, n), 0.35, 8.5)
    cholesterol = np.clip(ldl + hdl + triglycerides / 2.2 + rng.normal(0, 0.20, n), 2.0, 12.5)
    direct_bilirubin = np.clip(np.exp(np.log(2.1) + 0.10 * hepatic_fat + 0.07 * male + 0.35 * liver_shift + rng.normal(0, 0.35, n)), 0.3, 28)
    indirect_bilirubin = np.clip(np.exp(np.log(6.2) + 0.42 * bilirubin_genetic + 0.08 * male + 0.20 * liver_shift + rng.normal(0, 0.25, n)), 0.7, 55)
    total_bilirubin = np.clip(direct_bilirubin + indirect_bilirubin, 2, 75)
    creatinine = np.clip(66 + 13.5 * male + 7.0 * muscle + 9.5 * renal_impairment + 38 * renal_shift + rng.normal(0, 6.4, n), 32, 420)
    urea = np.clip(5.0 + 0.62 * renal_impairment + 0.28 * nutrition + 0.25 * age_z + 2.6 * renal_shift + rng.normal(0, 0.72, n), 1.3, 28)
    cystatin = np.clip(np.exp(np.log(0.86) + 0.17 * renal_impairment + 0.07 * age_z + 0.05 * inflammation + 0.72 * renal_shift + rng.normal(0, 0.11, n)), 0.40, 5.0)
    calcium = np.clip(2.35 + 0.014 * (albumin - 45) + 0.018 * bone_loss * -1 + rng.normal(0, 0.047, n), 1.90, 2.85)
    phosphate = np.clip(1.14 - 0.025 * age_z + 0.03 * renal_impairment + 0.12 * renal_shift + 0.018 * female + rng.normal(0, 0.105, n), 0.50, 2.2)
    crp = np.clip(np.exp(np.log(1.45) + 0.43 * inflammation + 0.14 * adiposity + 0.55 * inflammation_shift + rng.normal(0, 0.62, n)), 0.03, 120)
    ggt = np.clip(np.exp(np.log(27.0) + 0.24 * hepatic_fat + 0.25 * alcohol + 0.13 * insulin_resistance + 0.48 * liver_shift + 0.10 * male + rng.normal(0, 0.38, n)), 4, 650)
    meal_effect = rng.gamma(1.6, 0.22, n)
    glucose = np.clip(4.65 + 0.44 * insulin_resistance + 0.10 * age_z + 0.22 * e[:, 12] + meal_effect + glucose_shift + rng.normal(0, 0.46 + 0.12 * true_disease, n), 2.5, 22)
    hba1c = np.clip(34.0 + 2.35 * insulin_resistance + 0.80 * age_z + 1.35 * e[:, 12] + hba1c_shift + rng.normal(0, 3.5 + 0.9 * true_disease, n), 19, 125)
    igf1 = np.clip(22.0 - 1.7 * age_z - 0.45 * inflammation + rng.normal(0, 2.9, n), 4.5, 46)
    lpa = np.clip(np.exp(np.log(20.0) + 1.12 * lpa_genetic + rng.normal(0, 0.18, n)), 0.8, 420)
    shbg = np.clip(np.exp(np.log(42.0) + 0.31 * female + 0.08 * age_z - 0.23 * insulin_resistance - 0.07 * hepatic_fat + rng.normal(0, 0.32, n)), 6, 240)
    testosterone = np.where(
        male == 1,
        np.exp(np.log(13.5) - 0.15 * age_z - 0.10 * insulin_resistance + rng.normal(0, 0.26, n)),
        np.exp(np.log(0.72) - 0.07 * age_z - 0.08 * insulin_resistance + rng.normal(0, 0.35, n)),
    )
    testosterone = np.where(male == 1, np.clip(testosterone, 2.5, 36), np.clip(testosterone, 0.06, 4.2))
    menopause = (female == 1) & (age > (51 + rng.normal(0, 3.0, n)))
    oestradiol = np.where(
        male == 1,
        np.exp(np.log(75) - 0.05 * age_z + rng.normal(0, 0.28, n)),
        np.where(menopause,
                 np.exp(np.log(45) + rng.normal(0, 0.50, n)),
                 np.exp(np.log(190) + 0.30 * reproductive + rng.normal(0, 0.65, n))),
    )
    oestradiol = np.clip(oestradiol, 15, 1600)
    rheumatoid_factor = np.clip(np.where(rng.random(n) < sigmoid(-2.5 + 0.55 * immune + 0.15 * age_z),
                                         rng.lognormal(np.log(32), 0.8, n),
                                         rng.lognormal(np.log(6), 0.45, n)), 0.5, 650)
    non_albumin = np.clip(27.0 + 0.85 * inflammation + 0.58 * nutrition + 0.45 * liver_shift + rng.normal(0, 1.7, n), 18, 43)
    total_protein = np.clip(albumin + non_albumin, 52, 94)
    urate = np.clip(270 + 48 * male + 21 * insulin_resistance + 14 * renal_impairment + 11 * adiposity + urate_shift + rng.normal(0, 40, n), 90, 760)
    season = rng.uniform(0, 2 * np.pi, n)
    vitamin_d = np.clip(58 + 13 * sun_exposure + 8 * np.sin(season) - 5.5 * adiposity - 1.5 * age_z - 5.0 * (key == "Osteoporosis") * dsev + rng.normal(0, 10, n), 7, 190)

    chemistry_specs = [
        ("Alanine aminotransferase", alt, "U/L", "bounded log-normal", "liver", 2, 0.008, ""),
        ("Albumin", albumin, "g/L", "bounded normal", "liver", 2, 0.006, ""),
        ("Alkaline phosphatase", alp, "U/L", "bounded log-normal", "liver", 2, 0.009, ""),
        ("Apolipoprotein A", apoa, "g/L", "HDL-conditioned bounded normal", "lipid", 3, 0.014, "Generated conditionally on HDL with residual variation."),
        ("Apolipoprotein B", apob, "g/L", "bounded log-normal", "lipid", 3, 0.014, "Generated from atherogenic lipid physiology with treatment effect."),
        ("Aspartate aminotransferase", ast, "U/L", "bounded log-normal", "liver", 2, 0.009, ""),
        ("C-reactive protein", crp, "mg/L", "bounded log-normal with long tail", "inflammation", 3, 0.012, ""),
        ("Calcium", calcium, "mmol/L", "albumin-conditioned bounded normal", "mineral", 3, 0.008, "Conditioned on albumin; homeostatic regulation limits disease differences."),
        ("Cholesterol", cholesterol, "mmol/L", "lipoprotein-conditioned bounded distribution", "lipid", 3, 0.007, "Approximately LDL + HDL + triglycerides/2.2 with assay residual."),
        ("Creatinine", creatinine, "umol/L", "muscle/renal-conditioned bounded normal", "renal", 2, 0.007, ""),
        ("Cystatin C", cystatin, "mg/L", "bounded log-normal", "renal", 3, 0.014, ""),
        ("Direct bilirubin", direct_bilirubin, "umol/L", "bounded log-normal", "liver", 3, 0.022, ""),
        ("Gamma glutamyltransferase", ggt, "U/L", "bounded log-normal with long tail", "liver", 2, 0.009, ""),
        ("Glucose", glucose, "mmol/L", "bounded mixture with meal/treatment effects", "glycaemic", 3, 0.008, "Current measurement includes meal variation, treatment, assay noise and disease heterogeneity."),
        ("Glycated haemoglobin (HbA1c)", hba1c, "mmol/mol", "bounded conditional mixture", "glycaemic", 2, 0.010, "Treatment response and severity heterogeneity preserve overlap."),
        ("HDL cholesterol", hdl, "mmol/L", "bounded log-normal", "lipid", 3, 0.008, ""),
        ("IGF-1", igf1, "nmol/L", "bounded normal", "hormonal", 2, 0.020, ""),
        ("LDL direct", ldl, "mmol/L", "ApoB-conditioned bounded normal", "lipid", 3, 0.009, "Generated conditionally on ApoB with treatment and assay residual."),
        ("Lipoprotein A", lpa, "nmol/L", "heavy-tailed log-normal", "lipid", 2, 0.055, "Predominantly genetically driven."),
        ("Oestradiol", oestradiol, "pmol/L", "sex/menopause-stratified log-normal mixture", "hormonal", 2, 0.070, "Menstrual phase is unobserved and represented by broad variation."),
        ("Phosphate", phosphate, "mmol/L", "bounded normal", "mineral", 3, 0.011, ""),
        ("Rheumatoid factor", rheumatoid_factor, "IU/mL", "two-component heavy-tailed mixture", "immune", 2, 0.045, "Most participants have low values; a minority have an elevated immune-related component."),
        ("SHBG", shbg, "nmol/L", "sex/metabolic-conditioned log-normal", "hormonal", 2, 0.045, ""),
        ("Testosterone", testosterone, "nmol/L", "sex-stratified log-normal mixture", "hormonal", 3, 0.050, ""),
        ("Total bilirubin", total_bilirubin, "umol/L", "direct + indirect bilirubin", "liver", 3, 0.008, "Sum of direct and latent indirect fractions before assay residual/bounds."),
        ("Total protein", total_protein, "g/L", "albumin + non-albumin protein", "protein", 2, 0.008, "Generated conditionally on albumin and inflammatory protein fraction."),
        ("Triglycerides", triglycerides, "mmol/L", "bounded log-normal with long tail", "lipid", 3, 0.009, ""),
        ("Urate", urate, "umol/L", "bounded renal/metabolic-conditioned normal", "urate", 2, 0.008, "Current value can be lowered by treatment in gout."),
        ("Urea", urea, "mmol/L", "bounded renal/protein-conditioned normal", "renal", 3, 0.008, ""),
        ("Vitamin D", vitamin_d, "nmol/L", "seasonal bounded mixture", "vitamin_d", 2, 0.028, "Includes unobserved season and sun-exposure variation."),
    ]
    for desc, val, unit, distribution, group, decimals, miss, note in chemistry_specs:
        put(desc, val, "continuous", unit, distribution, group, decimals, miss, notes=note)

    # Missingness is intentionally disabled. Any structural non-finite values
    # are deterministically filled at the final schema-constraint layer.
    masks: dict[str, np.ndarray] = {canonical: np.zeros(n, dtype=bool) for canonical in meta}

    return {
        "values": values, "meta": meta, "masks": masks,
        "target": observed, "true_disease": true_disease,
        "severity": severity, "treated": treated,
        "treatment_intensity": treatment_intensity,
        "disease_probability": probability,
        "liability_intercept": float(intercept),
        "sampled_true_prevalence": float(np.mean(true_disease)),
        "sampled_observed_prevalence": float(np.mean(observed)),
        "sampled_treatment_fraction_true_cases": float(np.mean(treated[true_disease == 1])) if np.any(true_disease == 1) else None,
        "factors": {name: safe_float_array(factors[name]) for name in disease.risk_terms},
    }


def identify_variable(output_name: str, metadata: dict[str, str]) -> tuple[str | None, str, str]:
    if output_name == "age":
        return norm("Age at recruitment"), "Age at recruitment", "extra required column"
    if output_name == "sex":
        return norm("Gender"), "Gender", "extra required column"
    prioritized = ["Description", "Prompts", "Detail_Category", "Category"]
    for field in prioritized:
        text = str(metadata.get(field, "")).strip()
        if text:
            candidate = text.split(":", 1)[0].strip() if field == "Prompts" else text
            canonical = norm(candidate)
            if canonical:
                return canonical, candidate, f"schema metadata: {field}"
    return None, output_name, "unresolved"


def disease_effect_for(group: str, canonical: str, disease: DiseaseSpec) -> tuple[str, str, str]:
    alias = {
        "body_composition": "adiposity", "body_size": "adiposity",
        "cardiovascular": "blood_pressure", "general_health": "functional",
        "sun": "vitamin_d", "family_history": "family_metabolic",
        "sedentary": "activity", "protein": "liver",
    }.get(group, group)
    if group == "demographic":
        if canonical == norm("Age at recruitment"):
            return "higher age", "moderate-to-strong", "risk predictor"
        if canonical == norm("Gender"):
            direction = "male more common" if disease.abbr not in {"Osteoporosis"} else "female more common"
            return direction, "moderate" if disease.abbr in {"Gout", "Osteoporosis"} else "weak", "risk modifier"
    if alias in disease.manifest_effects:
        direction, strength, role = disease.manifest_effects[alias]
        return direction, strength, role
    general: dict[str, tuple[str, str, str]] = {
        "socioeconomic": ("weak/mixed", "weak", "distal risk correlate"),
        "environment": ("weak/mixed", "weak", "environmental correlate"),
        "early_life": ("weak/mixed", "weak", "distal risk correlate"),
        "sexual": ("near-zero/mixed", "near-zero", "weak/unrelated predictor"),
        "electronic_use": ("near-zero/mixed", "near-zero", "weak/unrelated predictor"),
        "hormonal": ("sex-dependent/mixed", "weak", "correlated marker"),
        "immune": ("near-zero/mixed", "near-zero", "weak/unrelated predictor"),
        "mineral": ("near-zero/mixed", "near-zero", "homeostatically regulated marker"),
        "blood_count": ("weak/mixed", "weak", "correlated physiological marker"),
        "lung": ("lower", "weak", "correlated health marker"),
        "sleep": ("less favourable", "weak", "risk/correlated behaviour"),
        "diet": ("less favourable", "weak", "risk/correlated behaviour"),
        "activity": ("lower", "weak", "risk/correlated behaviour"),
        "smoking": ("higher burden", "weak", "risk behaviour"),
        "alcohol": ("mixed", "weak", "risk/correlated behaviour"),
        "lipid": ("mixed", "weak", "correlated biomarker"),
        "liver": ("mixed", "weak", "correlated biomarker"),
        "renal": ("worse", "weak", "correlated biomarker"),
        "glycaemic": ("slightly higher", "weak", "correlated metabolic marker"),
        "inflammation": ("higher", "weak", "correlated marker"),
        "urate": ("slightly higher", "weak", "correlated metabolic marker"),
        "functional": ("lower function", "weak", "correlated health marker"),
        "vitamin_d": ("slightly lower", "weak", "correlated lifestyle marker"),
    }
    return general.get(group, ("near-zero/mixed", "near-zero", "weak/unrelated predictor"))


def quantiles(x: np.ndarray) -> dict[str, float]:
    valid = x[np.isfinite(x)]
    if len(valid) == 0:
        return {k: math.nan for k in ("min", "p01", "q1", "median", "mean", "q3", "p99", "max", "sd")}
    qv = np.quantile(valid, [0.01, 0.25, 0.50, 0.75, 0.99])
    return {
        "min": float(np.min(valid)), "p01": float(qv[0]), "q1": float(qv[1]),
        "median": float(qv[2]), "mean": float(np.mean(valid)), "q3": float(qv[3]),
        "p99": float(qv[4]), "max": float(np.max(valid)),
        "sd": float(np.std(valid, ddof=1)) if len(valid) > 1 else math.nan,
    }


def formatted_summary(summary: dict[str, float]) -> tuple[str, str]:
    vals = [summary[k] for k in ("mean", "sd", "median", "q1", "q3")]
    if not all(np.isfinite(v) for v in vals):
        return "", ""
    return f"{summary['mean']:.3f} ± {summary['sd']:.3f}", f"{summary['median']:.3f} [{summary['q1']:.3f}, {summary['q3']:.3f}]"


def rank_auc(x: np.ndarray, y: np.ndarray) -> float:
    valid = np.isfinite(x) & np.isfinite(y)
    xv, yv = x[valid], y[valid].astype(int)
    n1, n0 = int(np.sum(yv == 1)), int(np.sum(yv == 0))
    if n1 == 0 or n0 == 0 or np.std(xv) == 0:
        return math.nan
    ranks = pd.Series(xv).rank(method="average").to_numpy()
    return float((np.sum(ranks[yv == 1]) - n1 * (n1 + 1) / 2) / (n1 * n0))


def fit_logistic_design(design: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    prevalence = float(np.clip(np.mean(y), 1e-6, 1 - 1e-6))
    beta = np.zeros(design.shape[1])
    beta[0] = math.log(prevalence / (1 - prevalence))
    ridge = np.eye(design.shape[1]) * 1e-7
    ridge[0, 0] = 0
    for _ in range(80):
        fitted = sigmoid(design @ beta)
        weights = np.maximum(fitted * (1 - fitted), 1e-7)
        gradient = design.T @ (fitted - y) + ridge @ beta
        hessian = design.T @ (weights[:, None] * design) + ridge
        step = np.linalg.pinv(hessian) @ gradient
        beta_new = beta - step
        if np.max(np.abs(beta_new - beta)) < 1e-9:
            beta = beta_new
            break
        beta = beta_new
    fitted = np.clip(sigmoid(design @ beta), 1e-12, 1 - 1e-12)
    weights = np.maximum(fitted * (1 - fitted), 1e-7)
    covariance = np.linalg.pinv(design.T @ (weights[:, None] * design) + ridge)
    loglik = float(np.sum(y * np.log(fitted) + (1 - y) * np.log(1 - fitted)))
    return beta, covariance, loglik


def logistic_continuous_or_binary(x: np.ndarray, y: np.ndarray, standardize: bool) -> dict[str, float]:
    valid = np.isfinite(x) & np.isfinite(y)
    xv, yv = x[valid].astype(float), y[valid].astype(float)
    if len(xv) < 30 or np.std(xv) == 0 or len(np.unique(yv)) < 2:
        return {k: math.nan for k in ("beta", "se", "or", "ci_low", "ci_high", "p")}
    if standardize:
        xv = (xv - np.mean(xv)) / np.std(xv, ddof=1)
    design = np.column_stack([np.ones(len(xv)), xv])
    beta, cov, _ = fit_logistic_design(design, yv)
    slope, se = float(beta[1]), float(np.sqrt(max(cov[1, 1], 0)))
    z_value = slope / se if se > 0 else math.nan
    return {
        "beta": slope, "se": se,
        "or": float(math.exp(np.clip(slope, -20, 20))),
        "ci_low": float(math.exp(np.clip(slope - 1.96 * se, -20, 20))),
        "ci_high": float(math.exp(np.clip(slope + 1.96 * se, -20, 20))),
        "p": float(math.erfc(abs(z_value) / math.sqrt(2))) if np.isfinite(z_value) else math.nan,
    }


def categorical_logistic(x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    valid = np.isfinite(x) & np.isfinite(y)
    xv, yv = x[valid].astype(float), y[valid].astype(float)
    levels = np.sort(np.unique(xv))
    if len(xv) < 30 or len(levels) < 2 or len(np.unique(yv)) < 2:
        return {"reference": None, "or_json": "{}", "lr_chi2": math.nan, "df": 0, "p": math.nan}
    reference = float(levels[0])
    dummies = np.column_stack([(xv == level).astype(float) for level in levels[1:]])
    design = np.column_stack([np.ones(len(xv)), dummies])
    beta, _, ll_full = fit_logistic_design(design, yv)
    p0 = float(np.clip(np.mean(yv), 1e-12, 1 - 1e-12))
    ll_null = float(np.sum(yv * math.log(p0) + (1 - yv) * math.log(1 - p0)))
    lr = max(0.0, 2 * (ll_full - ll_null))
    df = len(levels) - 1
    odds = {str(int(level) if level.is_integer() else level): float(math.exp(np.clip(beta[i + 1], -20, 20))) for i, level in enumerate(levels[1:])}
    return {"reference": reference, "or_json": json.dumps(odds, sort_keys=True), "lr_chi2": lr, "df": df, "p": float(chi2.sf(lr, df))}


def smd_continuous(x: np.ndarray, y: np.ndarray) -> float:
    x0, x1 = x[(y == 0) & np.isfinite(x)], x[(y == 1) & np.isfinite(x)]
    if len(x0) < 2 or len(x1) < 2:
        return math.nan
    pooled = math.sqrt((float(np.var(x0, ddof=1)) + float(np.var(x1, ddof=1))) / 2)
    return float((np.mean(x1) - np.mean(x0)) / pooled) if pooled else math.nan


def smd_categorical(x: np.ndarray, y: np.ndarray) -> float:
    valid = np.isfinite(x)
    results: list[float] = []
    for level in np.unique(x[valid]):
        p0 = float(np.mean(x[(y == 0) & valid] == level))
        p1 = float(np.mean(x[(y == 1) & valid] == level))
        denom = math.sqrt(max((p0 * (1 - p0) + p1 * (1 - p1)) / 2, 1e-12))
        results.append((p1 - p0) / denom)
    return max(results, key=abs) if results else math.nan


def category_json(x: np.ndarray, y: np.ndarray, label: int, proportions: bool) -> str:
    selected = x[(y == label) & np.isfinite(x)]
    if len(selected) == 0:
        return "{}"
    levels, counts = np.unique(selected, return_counts=True)
    result = {}
    for level, count in zip(levels, counts):
        key = str(int(level) if float(level).is_integer() else float(level))
        result[key] = float(count / len(selected)) if proportions else int(count)
    return json.dumps(result, sort_keys=True)


def expected_high_pair(left: str, right: str, canonical_by_output: dict[str, str | None], info_by_canonical: dict[str, dict[str, Any]]) -> bool:
    lc, rc = canonical_by_output[left], canonical_by_output[right]
    if lc is None or rc is None:
        return False
    if lc == rc:
        return True
    ln = info_by_canonical.get(lc, {}).get("interpreted_name", lc).lower()
    rn = info_by_canonical.get(rc, {}).get("interpreted_name", rc).lower()
    if re.sub(r"\((left|right)\)", "", ln) == re.sub(r"\((left|right)\)", "", rn):
        return True
    related_sets = [
        ("weight", "body mass index", "fat mass", "fat-free mass", "predicted mass", "fat percentage", "water mass", "basal metabolic rate"),
        ("white blood cell", "neutroph", "lymphocyte", "monocyte", "eosinoph", "basoph"),
        ("red blood cell", "haemoglobin", "hemoglobin", "haematocrit", "hematocrit", "corpuscular"),
        ("reticulocyte count", "reticulocyte percentage", "high light scatter"),
        ("platelet count", "platelet crit", "mean platelet"),
        ("fvc", "fev1", "forced vital capacity", "forced expiratory", "peak expiratory flow"),
        ("number in household", "people in household related"),
        ("smokers in household", "exposure to tobacco smoke at home"),
        ("apolipoprotein a", "hdl cholesterol"),
        ("apolipoprotein b", "ldl direct", "cholesterol"),
    ]
    for tokens in related_sets:
        if any(t in ln for t in tokens) and any(t in rn for t in tokens):
            return True
    return False


def create_qc(dataset: pd.DataFrame, predictors: list[str], target: str,
              canonical_by_output: dict[str, str | None], info_by_canonical: dict[str, dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    y = dataset[target].to_numpy(float)
    numeric = dataset[predictors].apply(pd.to_numeric, errors="coerce")
    pearson = numeric.corr(method="pearson")
    spearman = numeric.corr(method="spearman")
    rows: list[dict[str, Any]] = []
    for column in predictors:
        x = numeric[column].to_numpy(float)
        canonical = canonical_by_output[column]
        info = info_by_canonical[canonical] if canonical in info_by_canonical else {}
        variable_type = str(info.get("variable_type", "continuous"))
        is_categorical = variable_type in {"categorical", "binary"}
        is_binary = variable_type == "binary"
        valid = np.isfinite(x)
        overall, control, case = quantiles(x), quantiles(x[y == 0]), quantiles(x[y == 1])
        control_ms, control_mi = formatted_summary(control)
        case_ms, case_mi = formatted_summary(case)
        if is_categorical and not is_binary:
            logistic_cat = categorical_logistic(x, y)
            logistic = {k: math.nan for k in ("beta", "se", "or", "ci_low", "ci_high", "p")}
            auc = math.nan
            smd = smd_categorical(x, y)
            scale = "dummy-coded categories; global likelihood-ratio test"
        else:
            logistic = logistic_continuous_or_binary(x, y, standardize=not is_binary)
            logistic_cat = {"reference": None, "or_json": "{}", "lr_chi2": math.nan, "df": 0, "p": math.nan}
            auc = rank_auc(x, y)
            smd = smd_continuous(x, y)
            scale = "per 1-unit increase" if is_binary else "per SD increase"
        pabs = pearson[column].abs().drop(labels=[column], errors="ignore").dropna()
        sabs = spearman[column].abs().drop(labels=[column], errors="ignore").dropna()
        rows.append({
            "variable": column, "interpreted_name": info.get("interpreted_name", "Unresolved variable"),
            "variable_type": variable_type, "unit": info.get("unit", "unknown"),
            "n_total": len(x), "missing_n": int(np.sum(~valid)), "missing_pct": float(np.mean(~valid) * 100),
            **{f"overall_{k}": v for k, v in overall.items()},
            **{f"control_{k}": v for k, v in control.items()},
            **{f"disease_{k}": v for k, v in case.items()},
            "control_mean_sd": "" if is_categorical else control_ms,
            "control_median_iqr": "" if is_categorical else control_mi,
            "disease_mean_sd": "" if is_categorical else case_ms,
            "disease_median_iqr": "" if is_categorical else case_mi,
            "control_category_counts_json": category_json(x, y, 0, False) if is_categorical else "",
            "control_category_proportions_json": category_json(x, y, 0, True) if is_categorical else "",
            "disease_category_counts_json": category_json(x, y, 1, False) if is_categorical else "",
            "disease_category_proportions_json": category_json(x, y, 1, True) if is_categorical else "",
            "smd_disease_minus_control": smd,
            "univariable_logistic_scale": scale,
            "univariable_log_odds": logistic["beta"], "univariable_log_odds_se": logistic["se"],
            "univariable_or": logistic["or"], "univariable_or_ci_low": logistic["ci_low"],
            "univariable_or_ci_high": logistic["ci_high"], "univariable_p_value": logistic["p"],
            "categorical_reference_code": logistic_cat["reference"],
            "categorical_level_or_vs_reference_json": logistic_cat["or_json"],
            "categorical_global_lr_chi2": logistic_cat["lr_chi2"], "categorical_global_df": logistic_cat["df"],
            "categorical_global_p_value": logistic_cat["p"],
            "univariable_rank_auc": auc,
            "univariable_auc_discrimination": max(auc, 1 - auc) if np.isfinite(auc) else math.nan,
            "near_perfect_single_variable_flag": bool(np.isfinite(auc) and max(auc, 1 - auc) >= 0.95),
            "max_abs_pearson": float(pabs.max()) if len(pabs) else math.nan,
            "max_abs_pearson_with": str(pabs.idxmax()) if len(pabs) else "",
            "max_abs_spearman": float(sabs.max()) if len(sabs) else math.nan,
            "max_abs_spearman_with": str(sabs.idxmax()) if len(sabs) else "",
        })
    descriptive = pd.DataFrame(rows)
    high_pairs: list[dict[str, Any]] = []
    unexpected: list[dict[str, Any]] = []
    for i, left in enumerate(predictors):
        for right in predictors[i + 1:]:
            p = float(pearson.loc[left, right]) if pd.notna(pearson.loc[left, right]) else math.nan
            s = float(spearman.loc[left, right]) if pd.notna(spearman.loc[left, right]) else math.nan
            if (np.isfinite(p) and abs(p) >= 0.90) or (np.isfinite(s) and abs(s) >= 0.90):
                item = {"variable_1": left, "variable_2": right, "pearson": p, "spearman": s,
                        "expected_semantic_or_derived_relationship": expected_high_pair(left, right, canonical_by_output, info_by_canonical)}
                high_pairs.append(item)
                if not item["expected_semantic_or_derived_relationship"] and ((np.isfinite(p) and abs(p) >= 0.95) or (np.isfinite(s) and abs(s) >= 0.95)):
                    unexpected.append(item)
    age_rows = descriptive[descriptive["interpreted_name"] == "Age at recruitment"]
    age_warning = bool(len(age_rows) and pd.to_numeric(age_rows["univariable_auc_discrimination"], errors="coerce").max() >= 0.90)
    near_perfect = descriptive.loc[descriptive["near_perfect_single_variable_flag"], "variable"].tolist()
    summary = {
        "observed_case_count": int(np.sum(y == 1)), "observed_control_count": int(np.sum(y == 0)),
        "observed_case_prevalence": float(np.mean(y)),
        "near_perfect_single_variables_auc_ge_0.95": near_perfect,
        "age_label_near_deterministic_auc_ge_0.90": age_warning,
        "unexpected_abs_correlation_ge_0.95": unexpected,
        "all_abs_correlation_ge_0.90": high_pairs,
        "extreme_value_policy": "Long tails/mixtures retained; only broad physiological or instrument-plausibility bounds applied; no participant rows deleted.",
        "auc_use_statement": "Univariable rank AUC is a post-generation QC diagnostic only and never feeds back into generation parameters.",
        "categorical_logistic_note": "Nominal/ordinal categorical predictors use dummy coding with the lowest code as reference and a global likelihood-ratio test; continuous/count predictors report per-SD OR.",
    }
    return descriptive, pearson, spearman, summary


def json_ready(value: Any) -> Any:
    """Recursively convert numpy/pandas scalars to strict JSON values."""
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return [json_ready(v) for v in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(float(value)) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if pd.isna(value):
        return None
    return value


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def semantic_output(canonical_by_output: dict[str, str | None], description: str) -> str | None:
    wanted = norm(description)
    for output, canonical in canonical_by_output.items():
        if canonical == wanted:
            return output
    return None


def group_rates(mask: np.ndarray, y: np.ndarray) -> dict[str, float | None]:
    result: dict[str, float | None] = {}
    for label, code in (("control", 0), ("disease", 1)):
        eligible = y == code
        result[label] = float(np.mean(mask[eligible])) if np.any(eligible) else None
    return result


def disease_overlap_checks(dataset: pd.DataFrame, target: str, disease: DiseaseSpec,
                           canonical_by_output: dict[str, str | None]) -> dict[str, Any]:
    """Clinically interpretable overlap checks; never used to tune generation."""
    y = dataset[target].to_numpy(dtype=int)

    def arr(description: str) -> np.ndarray | None:
        column = semantic_output(canonical_by_output, description)
        return None if column is None else pd.to_numeric(dataset[column], errors="coerce").to_numpy(float)

    checks: dict[str, Any] = {
        "purpose": "Post-generation biological plausibility and case-control overlap checks only; results never feed back into effect sizes.",
        "disease_specific_interpretation": disease.distinction,
    }
    glucose, hba1c = arr("Glucose"), arr("Glycated haemoglobin (HbA1c)")
    creatinine, cystatin = arr("Creatinine"), arr("Cystatin C")
    ldl, triglycerides = arr("Direct LDL cholesterol"), arr("Triglycerides")
    urate, sex = arr("Urate"), arr("Gender")
    alt, ggt = arr("Alanine aminotransferase"), arr("Gamma glutamyltransferase")
    sbp, dbp = arr("Systolic blood pressure"), arr("Diastolic blood pressure")
    crp, grip, calcium = arr("C-reactive protein"), arr("Hand grip strength (left)"), arr("Calcium")

    if disease.abbr in {"T2D", "T2D_broad"} and glucose is not None and hba1c is not None:
        valid = np.isfinite(glucose) & np.isfinite(hba1c)
        checks["both_glucose_ge_7_mmol_L_and_HbA1c_ge_48_mmol_mol"] = group_rates(valid & (glucose >= 7.0) & (hba1c >= 48.0), y)
        checks["both_below_those_marker_thresholds"] = group_rates(valid & (glucose < 7.0) & (hba1c < 48.0), y)
    elif disease.abbr == "CKD" and creatinine is not None and cystatin is not None:
        valid = np.isfinite(creatinine) & np.isfinite(cystatin)
        checks["creatinine_ge_100_umol_L_or_cystatin_C_ge_1_0_mg_L"] = group_rates(valid & ((creatinine >= 100) | (cystatin >= 1.0)), y)
        checks["both_below_those_marker_thresholds"] = group_rates(valid & (creatinine < 100) & (cystatin < 1.0), y)
    elif disease.abbr in {"Hypercholesterolaemia", "Hyperlipidaemia"} and ldl is not None and triglycerides is not None:
        valid = np.isfinite(ldl) & np.isfinite(triglycerides)
        checks["LDL_ge_4_0_mmol_L_or_triglycerides_ge_2_3_mmol_L"] = group_rates(valid & ((ldl >= 4.0) | (triglycerides >= 2.3)), y)
        checks["both_below_those_marker_thresholds"] = group_rates(valid & (ldl < 4.0) & (triglycerides < 2.3), y)
    elif disease.abbr == "Gout" and urate is not None and sex is not None:
        threshold = np.where(sex == 1, 420.0, 360.0)
        valid = np.isfinite(urate) & np.isfinite(sex)
        checks["urate_above_sex_specific_screening_level"] = group_rates(valid & (urate >= threshold), y)
        checks["urate_below_sex_specific_screening_level"] = group_rates(valid & (urate < threshold), y)
    elif disease.abbr == "Liver_disease" and alt is not None and ggt is not None:
        threshold = np.where(sex == 1, 40.0, 35.0) if sex is not None else np.full(len(y), 38.0)
        valid = np.isfinite(alt) & np.isfinite(ggt)
        checks["ALT_above_sex_specific_level_or_GGT_ge_60_U_L"] = group_rates(valid & ((alt >= threshold) | (ggt >= 60.0)), y)
        checks["both_below_those_marker_levels"] = group_rates(valid & (alt < threshold) & (ggt < 60.0), y)
    elif disease.abbr == "Hypertension" and sbp is not None and dbp is not None:
        valid = np.isfinite(sbp) & np.isfinite(dbp)
        checks["current_SBP_ge_140_or_DBP_ge_90_mmHg"] = group_rates(valid & ((sbp >= 140) | (dbp >= 90)), y)
        checks["both_current_values_below_those_levels"] = group_rates(valid & (sbp < 140) & (dbp < 90), y)
    elif disease.abbr in {"IHD", "CHD"}:
        checks["limitation"] = "The schema has risk factors but no coronary imaging/troponin variable; current LDL and blood pressure are treatment-influenced and intentionally overlap."
    elif disease.abbr == "Heart_failure":
        checks["limitation"] = "The schema has no BNP/NT-proBNP or echocardiography; functional, renal and inflammatory correlates intentionally overlap."
        if creatinine is not None and crp is not None:
            valid = np.isfinite(creatinine) & np.isfinite(crp)
            checks["creatinine_ge_100_or_CRP_ge_5"] = group_rates(valid & ((creatinine >= 100) | (crp >= 5)), y)
    elif disease.abbr == "Osteoporosis":
        checks["limitation"] = "No BMD/DXA variable is present; liability uses latent bone strength. Serum calcium is kept weak because of homeostatic regulation."
        if grip is not None and calcium is not None:
            valid = np.isfinite(grip) & np.isfinite(calcium)
            checks["lower_grip_but_normal_calcium_pattern"] = group_rates(valid & (grip < 24) & (calcium >= 2.15) & (calcium <= 2.55), y)
    return checks


def derived_consistency_checks(dataset: pd.DataFrame,
                               canonical_by_output: dict[str, str | None]) -> dict[str, Any]:
    def get(description: str) -> np.ndarray | None:
        column = semantic_output(canonical_by_output, description)
        return None if column is None else pd.to_numeric(dataset[column], errors="coerce").to_numpy(float)

    result: dict[str, Any] = {}
    for extra, description in (("age", "Age at recruitment"), ("sex", "Gender")):
        matches = [column for column, canonical in canonical_by_output.items() if canonical == norm(description) and column != extra]
        canonical_column = matches[0] if matches else None
        if extra in dataset and canonical_column:
            a = pd.to_numeric(dataset[extra], errors="coerce").to_numpy(float)
            b = pd.to_numeric(dataset[canonical_column], errors="coerce").to_numpy(float)
            valid = np.isfinite(a) & np.isfinite(b)
            result[f"{extra}_duplicate_{canonical_column}"] = {
                "n_comparable": int(valid.sum()),
                "max_absolute_difference": float(np.max(np.abs(a[valid] - b[valid]))) if valid.any() else None,
                "expected_exact_duplicate": True,
            }
    body, fat, ffm = get("Weight"), get("Whole body fat mass"), get("Whole body fat-free mass")
    if body is not None and fat is not None and ffm is not None:
        valid = np.isfinite(body) & np.isfinite(fat) & np.isfinite(ffm)
        error = body[valid] - fat[valid] - ffm[valid]
        result["whole_body_mass_equals_fat_plus_fat_free_mass"] = {
            "n_comparable": int(valid.sum()), "mean_absolute_rounding_error_kg": float(np.mean(np.abs(error))) if valid.any() else None,
            "max_absolute_rounding_error_kg": float(np.max(np.abs(error))) if valid.any() else None,
        }
    wbc = get("White blood cell (leukocyte) count")
    component_names = ["Neutrophill count", "Lymphocyte count", "Monocyte count", "Eosinophill count", "Basophill count"]
    components = [get(name) for name in component_names]
    if wbc is not None and all(x is not None for x in components):
        comp = np.vstack([x for x in components if x is not None])
        valid = np.isfinite(wbc) & np.all(np.isfinite(comp), axis=0)
        error = wbc[valid] - np.sum(comp[:, valid], axis=0)
        result["white_cell_count_equals_component_counts"] = {
            "n_comparable": int(valid.sum()), "mean_absolute_rounding_error_10e9_L": float(np.mean(np.abs(error))) if valid.any() else None,
            "max_absolute_rounding_error_10e9_L": float(np.max(np.abs(error))) if valid.any() else None,
        }
    return result


def build_dataset(schema_path: Path, output_dir: Path, disease_key: str, n: int,
                  seed: int, cohort_mode: str, schema_index: int) -> dict[str, Any]:
    disease = DISEASES[disease_key]
    schema, schema_names, metadata = load_schema(schema_path, schema_index)
    predictors = ordered_unique(EXTRA_REQUIRED_COLUMNS + schema_names)
    predictors = [name for name in predictors if name != disease.abbr]
    if not predictors:
        raise ValueError("Predictor list is empty after applying schema rules")

    rng = np.random.default_rng(seed)
    working_n = n if cohort_mode == "population_based" else max(50000, n * 6)
    generated = generate_population(working_n, rng, disease)
    selected = np.arange(working_n)
    if cohort_mode == "case_control":
        y_working = generated["target"]
        cases = np.flatnonzero(y_working == 1)
        controls = np.flatnonzero(y_working == 0)
        n_cases = min(n // 2, len(cases))
        n_controls = min(n - n_cases, len(controls))
        if n_cases + n_controls < n:
            raise RuntimeError("Insufficient synthetic working population for requested case-control sample")
        selected = np.concatenate([rng.choice(cases, n_cases, False), rng.choice(controls, n_controls, False)])
        rng.shuffle(selected)

    data: dict[str, np.ndarray] = {}
    manifest_rows: list[dict[str, Any]] = []
    constraint_audit_rows: list[dict[str, Any]] = []
    canonical_by_output: dict[str, str | None] = {}
    unresolved: list[dict[str, str]] = []
    fallback_base = z(generated["values"][norm("Age at recruitment")])

    for output_name in predictors:
        row_meta = metadata.get(output_name, {})
        canonical, proposed_name, recognized_via = identify_variable(output_name, row_meta)
        info = generated["meta"].get(canonical or "")
        if info is None or canonical not in generated["values"]:
            canonical_by_output[output_name] = None
            values = 0.12 * fallback_base + rng.normal(0, 1, working_n)
            info = {
                "interpreted_name": "Unresolved variable", "variable_type": "continuous", "unit": "unknown",
                "distribution": "metadata-constrained weakly age-correlated fallback",
                "physiological_group": "unresolved", "decimals": 3, "missing_rate": 0.0,
                "category_levels": "", "derivation_rule": "",
                "notes": "No reliable semantic identification; no disease-specific effect assigned. V1 schema constraints and anti-collapse remain active.",
            }
            unresolved.append({"output_column": output_name, "proposed_interpretation": proposed_name,
                               "reason": "Metadata did not match a supported semantic variable; generated as a weak/unrelated fallback, then constrained by schema."})
        else:
            canonical_by_output[output_name] = canonical
            values = generated["values"][canonical].copy()

        # V1 final-scale constraint engine + no-missing fill + anti-collapse repair.
        constrained, constraint_info = apply_v1_schema_constraints(values, row_meta, info, rng)
        final_values = constrained[selected]
        if not np.isfinite(final_values).all():
            raise RuntimeError(f"{disease_key}/{output_name}: non-finite values after final processing")
        data[output_name] = final_values

        direction, strength, role = disease_effect_for(str(info.get("physiological_group", "unresolved")), canonical or "", disease)
        source_notes = str(info.get("notes", ""))
        if output_name not in EXTRA_REQUIRED_COLUMNS and row_meta:
            prompt_unit = extract_unit(str(row_meta.get("Prompts", "")))
            if prompt_unit not in {"unspecified", "coded category"} and prompt_unit.lower() != str(info.get("unit", "")).lower():
                source_notes = (source_notes + " " if source_notes else "") + f"Schema prompt unit text was '{prompt_unit}'; generated unit uses semantic/dimensional consistency."
        manifest_rows.append({
            "output_column": output_name, "interpreted_name": info.get("interpreted_name", proposed_name),
            "unit": info.get("unit", "unknown"), "variable_type": info.get("variable_type", "continuous"),
            "distribution": info.get("distribution", "unknown"), "physiological_group": info.get("physiological_group", "unresolved"),
            "role_in_disease_model": role, "effect_direction": direction,
            "effect_strength_category": strength, "derivation_rule": info.get("derivation_rule", ""),
            "category_levels": info.get("category_levels", ""), "planned_missing_rate": 0.0,
            "recognition_source": recognized_via, "metadata_interpretation_text": proposed_name,
            "notes": source_notes,
        })
        constraint_audit_rows.append({"output_column": output_name, **constraint_info})

    data[disease.abbr] = generated["target"][selected].astype(int)
    dataset = pd.DataFrame(data, columns=predictors + [disease.abbr])
    manifest = pd.DataFrame(manifest_rows)
    constraint_audit = pd.DataFrame(constraint_audit_rows)

    # Hard postconditions requested for this version.
    numeric_predictors = dataset[predictors].apply(pd.to_numeric, errors="coerce")
    missing_cells = int(numeric_predictors.isna().sum().sum())
    nonfinite_cells = int((~np.isfinite(numeric_predictors.to_numpy(float))).sum())
    if missing_cells != 0 or nonfinite_cells != 0:
        raise RuntimeError(f"{disease_key}: no-missing postcondition failed: missing={missing_cells}, nonfinite={nonfinite_cells}")
    anti_collapse_failures: list[str] = []
    for audit in constraint_audit_rows:
        if audit["metadata_supports_variability"] and int(audit["final_unique_values"]) < 2:
            anti_collapse_failures.append(str(audit["output_column"]))
    if anti_collapse_failures:
        raise RuntimeError(f"{disease_key}: anti-collapse failed for {anti_collapse_failures}")

    info_by_output = {row["output_column"]: row for row in manifest_rows}
    info_by_canonical: dict[str, dict[str, Any]] = {}
    for output_name, canonical in canonical_by_output.items():
        if canonical is not None:
            info_by_canonical[canonical] = info_by_output[output_name]
    descriptive, pearson, spearman, qc = create_qc(
        dataset, predictors, disease.abbr, canonical_by_output, info_by_canonical
    )
    qc["disease_overlap_checks"] = disease_overlap_checks(dataset, disease.abbr, disease, canonical_by_output)
    qc["derived_variable_consistency_checks"] = derived_consistency_checks(dataset, canonical_by_output)
    qc["column_order_exact"] = list(dataset.columns) == predictors + [disease.abbr]
    qc["target_is_last_column"] = dataset.columns[-1] == disease.abbr
    qc["target_unique_occurrence"] = list(dataset.columns).count(disease.abbr) == 1
    qc["row_count_exact"] = len(dataset) == n
    qc["missing_cells"] = missing_cells
    qc["nonfinite_cells"] = nonfinite_cells
    qc["no_missing_postcondition"] = True
    qc["sex_definition"] = SEX_DEFINITION
    qc["schema_constraint_columns_with_precision_resolution"] = [
        r["output_column"] for r in constraint_audit_rows if str(r["precision_resolution"]).startswith("anti-collapse")
    ]
    qc["schema_constraint_columns_regenerated_for_anti_collapse"] = [
        r["output_column"] for r in constraint_audit_rows if bool(r["anti_collapse_regenerated"])
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = output_dir / f"synthetic_{disease.abbr}_{n}.csv"
    descriptive_path = output_dir / "synthetic_descriptive_statistics.csv"
    manifest_path = output_dir / "variable_generation_manifest.csv"
    constraint_path = output_dir / "schema_constraint_audit.csv"
    pearson_path = output_dir / "synthetic_Pearson_correlation.csv"
    spearman_path = output_dir / "synthetic_Spearman_correlation.csv"
    parameters_path = output_dir / "synthetic_generation_parameters.json"
    dataset.to_csv(dataset_path, index=False, encoding="utf-8-sig")
    descriptive.to_csv(descriptive_path, index=False, encoding="utf-8-sig")
    manifest.to_csv(manifest_path, index=False, encoding="utf-8-sig")
    constraint_audit.to_csv(constraint_path, index=False, encoding="utf-8-sig")
    pearson.to_csv(pearson_path, encoding="utf-8-sig")
    spearman.to_csv(spearman_path, encoding="utf-8-sig")

    # Reload the serialized dataset and require the same no-missing/shape conditions.
    reloaded = pd.read_csv(dataset_path)
    if list(reloaded.columns) != predictors + [disease.abbr] or len(reloaded) != n:
        raise RuntimeError(f"{disease_key}: post-serialization schema/row-count verification failed")
    reloaded_numeric = reloaded.apply(pd.to_numeric, errors="coerce")
    if reloaded_numeric.isna().any().any() or not np.isfinite(reloaded_numeric.to_numpy(float)).all():
        raise RuntimeError(f"{disease_key}: post-serialization no-missing verification failed")

    parameters = {
        "generation_scope": {
            "purpose": "Synthetic adult cohort generation only; not model training, model selection, or validation.",
            "knowledge_boundary": "General biomedical, clinical, and epidemiologic knowledge only. No participant-level external data are read.",
            "predictive_performance_statement": "No AUC/PR-AUC/accuracy target and no classifier-guided parameter tuning were used.",
        },
        "schema": {
            "input_path": str(schema_path), "input_sha256": file_sha256(schema_path),
            "name_column_index": schema_index, "name_column": str(schema.columns[schema_index]),
            "metadata_columns": [str(c) for c in schema.columns if c != schema.columns[schema_index]],
            "schema_nonempty_name_count": len(schema_names), "predictor_count": len(predictors),
            "extra_required_columns": EXTRA_REQUIRED_COLUMNS, "unresolved_variables": unresolved,
            "require_no_multivariate_schema_rows": REQUIRE_NO_MULTIVARIATE_SCHEMA_ROWS,
            "constraint_policy": "V2 semantic/physiological formulas first; V1 Recommended/Observed hard bounds, allowed values and decimal precision enforced on final scale; metadata-quantile regeneration is used only when final-scale collapse is detected.",
            "anti_collapse_policy": "If metadata demonstrates variability, final constant/near-constant output triggers V1-style metadata-grounded regeneration; unresolved collapse is fatal.",
            "missingness_policy": "Disabled. Structural non-finite values are filled with a schema-valid central value before final constraints. Serialized predictors must contain zero missing/non-finite cells.",
        },
        "disease": {
            "name": disease.name, "abbreviation_and_target_column": disease.abbr,
            "target_definition": TARGET_DEFINITION, "cohort_mode": cohort_mode,
            "intended_population_prevalence": disease.prevalence,
            "ascertainment_sensitivity": disease.sensitivity, "ascertainment_false_positive_rate": disease.false_positive_rate,
            "clinical_scope": disease.distinction,
        },
        "sample": {
            "n_synthetic": n, "random_seed": seed,
            "sampled_true_disease_prevalence_before_optional_sampling": generated["sampled_true_prevalence"],
            "sampled_observed_prevalence_before_optional_sampling": generated["sampled_observed_prevalence"],
            "observed_output_prevalence": float(dataset[disease.abbr].mean()),
            "sampled_treatment_fraction_among_true_cases": generated["sampled_treatment_fraction_true_cases"],
            "sex_definition": SEX_DEFINITION,
        },
        "joint_generation": {
            "architecture": "Hierarchical latent-factor structural generator with shared demographic, social, behavioural and physiological factors plus variable-specific residuals and disease/treatment consequences.",
            "risk_terms_log_odds_coefficients": disease.risk_terms,
            "interactions": disease.interactions,
            "calibrated_liability_intercept": generated["liability_intercept"],
            "calibration_note": "Only the intercept was calibrated to the prespecified population prevalence; effect sizes were not tuned to predictive metrics.",
            "unmeasured_heterogeneity": "Explicit unmeasured susceptibility enters liability; labels are Bernoulli draws with imperfect ascertainment.",
            "treatment": "Disease-specific treatment probability and intensity attenuate selected current biomarkers, creating clinically plausible overlap.",
            "missingness": "Completely disabled in final output.",
            "extremes": qc["extreme_value_policy"],
        },
        "disease_specificity_guard": {
            "active_disease_key": disease_key,
            "active_target_column": disease.abbr,
            "active_clinical_scope": disease.distinction,
            "active_risk_terms_only": disease.risk_terms,
            "active_interactions_only": disease.interactions,
            "active_manifest_effects_only": disease.manifest_effects,
            "inactive_disease_keys": [key for key in DISEASES if key != disease_key],
            "guard_statement": "Exactly one DiseaseSpec contributes to each disease-specific cohort.",
        },
        "variables": manifest_rows,
        "schema_constraint_audit": constraint_audit_rows,
        "quality_control": qc,
        "outputs": {
            "dataset": dataset_path.name, "descriptive_statistics": descriptive_path.name,
            "manifest": manifest_path.name, "schema_constraint_audit": constraint_path.name,
            "pearson_correlation": pearson_path.name, "spearman_correlation": spearman_path.name,
        },
    }
    with parameters_path.open("w", encoding="utf-8") as handle:
        json.dump(json_ready(parameters), handle, ensure_ascii=False, indent=2, allow_nan=False)

    return {
        "disease_key": disease_key, "disease_name": disease.name, "abbr": disease.abbr,
        "seed": seed, "n": len(dataset), "predictor_count": len(predictors), "column_count": dataset.shape[1],
        "observed_case_count": int(dataset[disease.abbr].sum()), "observed_prevalence": float(dataset[disease.abbr].mean()),
        "true_prevalence_before_sampling": generated["sampled_true_prevalence"],
        "target_last": bool(dataset.columns[-1] == disease.abbr), "unresolved_variable_count": len(unresolved),
        "missing_cells": 0, "nonfinite_cells": 0,
        "anti_collapse_regenerated_count": int(sum(bool(r["anti_collapse_regenerated"]) for r in constraint_audit_rows)),
        "dataset_path": str(dataset_path), "parameters_path": str(parameters_path),
        "dataset_sha256": file_sha256(dataset_path),
    }


def resolve_default_schema(script_path: Path) -> Path:
    candidates = [
        script_path.parent / SCHEMA_CSV,
        script_path.parent / "all_variable_to_generate_with_constraints.csv",
        script_path.parent / "all_variable_to_generate.csv",
        script_path.parent.parent / SCHEMA_CSV,
        script_path.parent.parent / "all_variable_to_generate_with_constraints.csv",
        script_path.parent.parent / "all_variable_to_generate.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def build_all_diseases(schema_path: Path, output_root: Path, disease_keys: list[str], n: int,
                       base_seed: int, cohort_mode: str, schema_index: int,
                       overwrite: bool, make_zip: bool) -> dict[str, Any]:
    if not schema_path.exists():
        raise FileNotFoundError(
            f"Schema CSV not found: {schema_path}. Pass --schema /path/to/all_variable_to_generate_with_constraints.csv"
        )
    if output_root.exists():
        if overwrite:
            shutil.rmtree(output_root)
        elif any(output_root.iterdir()):
            raise FileExistsError(f"Output directory already exists and is not empty: {output_root}; use --overwrite")
    output_root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(schema_path, output_root / schema_path.name)
    try:
        shutil.copy2(Path(__file__).resolve(), output_root / Path(__file__).name)
    except Exception:
        pass

    results: list[dict[str, Any]] = []
    for i, disease_key in enumerate(disease_keys, 1):
        disease = DISEASES[disease_key]
        disease_seed = stable_disease_seed(base_seed, disease_key)
        disease_dir = output_root / disease.abbr
        result = build_dataset(schema_path, disease_dir, disease_key, n, disease_seed, cohort_mode, schema_index)
        results.append(result)
        print(f"[{i}/{len(disease_keys)}] {disease.name}: PASS | observed prevalence={result['observed_prevalence']:.4f} | seed={disease_seed}", flush=True)

    manifest = pd.DataFrame(results)
    manifest_path = output_root / "all_diseases_manifest.csv"
    manifest.to_csv(manifest_path, index=False, encoding="utf-8-sig")
    summary = {
        "disease_count": len(disease_keys), "disease_keys": disease_keys,
        "n_per_disease": n, "base_seed": base_seed, "cohort_mode": cohort_mode,
        "sex_definition": SEX_DEFINITION, "missingness": "disabled",
        "schema_constraint_engine": "V1 hard bounds/allowed values/precision + anti-collapse ported to V2 final output layer",
        "manifest": manifest_path.name,
    }
    (output_root / "all_diseases_summary.json").write_text(
        json.dumps(json_ready(summary), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )

    zip_path: str | None = None
    if make_zip:
        archive = output_root.with_suffix(".zip")
        if archive.exists():
            archive.unlink()
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
            for path in sorted(output_root.rglob("*")):
                if path.is_file():
                    zf.write(path, arcname=path.relative_to(output_root))
        zip_path = str(archive)
        print(f"ZIP: {archive}", flush=True)

    return {"output_root": str(output_root), "manifest": str(manifest_path), "zip": zip_path, "results": results}


def main() -> None:
    script_path = Path(__file__).resolve()
    default_schema = resolve_default_schema(script_path)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schema", type=Path, default=default_schema,
                        help="Constraint schema CSV. Prefer all_variable_to_generate_with_constraints.csv")
    parser.add_argument("--schema-name-column-index", type=int, default=SCHEMA_NAME_COLUMN_INDEX)
    parser.add_argument("--output-dir", type=Path, default=Path("synthetic_12_diseases_v2"))
    parser.add_argument("--n", type=int, default=N_SYNTHETIC, help="Participants per disease")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Base seed; each disease receives a stable derived seed")
    parser.add_argument("--cohort-mode", choices=["population_based", "case_control"], default=COHORT_MODE)
    parser.add_argument("--diseases", nargs="*", choices=list(DISEASES), default=list(DISEASES),
                        help="Disease keys to generate; default is all 12")
    parser.add_argument("--overwrite", action="store_true", help="Delete an existing non-empty output directory before generation")
    parser.add_argument("--zip", action="store_true", help="Also create a ZIP archive of the complete output directory")
    args = parser.parse_args()
    if args.n <= 0:
        parser.error("--n must be positive")
    result = build_all_diseases(args.schema, args.output_dir, list(args.diseases), args.n, args.seed,
                                args.cohort_mode, args.schema_name_column_index, args.overwrite, args.zip)
    compact = {k: v for k, v in result.items() if k != "results"}
    compact["disease_count"] = len(result["results"])
    print(json.dumps(json_ready(compact), ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
