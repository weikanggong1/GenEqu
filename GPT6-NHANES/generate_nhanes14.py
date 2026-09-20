#!/usr/bin/env python3
"""Knowledge-only NHANES native52 populations; no empirical inputs or fitting."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time

try:
    import numpy as np
    import pandas as pd
except ModuleNotFoundError:
    np = pd = None

VERSION = "nhanes14-knowledge-native52-v1"
DEFAULT_SEED = 20260831
DEFAULT_N = 360000
BLOCK_SIZE = 8192
REQUESTED_MODEL = {"model": "gpt-6-astra", "reasoning_effort": "ultra",
                   "backend_build_identifier": None}

# Descriptions and sets below are used directly to produce the full rule ledger.
# Dependencies are transitive: derived body/CBC/protein measurements inherit the
# dependencies of their components. Neutral means no specified endpoint pathway.
FEATURE_SPEC = {
    "RIDAGEYR": ("Eligible integer age; latent ages 85--95 are topcoded to 85.", "age", ""),
    "RIAGENDR": ("1 male, 2 female; equal sex sampling except male-only endpoints.", "sex", ""),
    "DMDHHSIZ": ("Independent 1 + Poisson(1.8), topcoded at 7.", "", ""),
    "BIDFAT": ("Weight times fat fraction, sharing BMI, age and sex physiology.", "age sex adiposity inactive frailty", "adipose wasting lean_loss height_loss congestion disability steroid"),
    "BIDFFM": ("Weight minus fat mass; shared body-composition identity.", "age sex adiposity inactive frailty", "adipose wasting lean_loss height_loss congestion disability steroid"),
    "BIDPFAT": ("100 times smooth fat fraction from BMI, age and sex.", "age sex adiposity inactive frailty", "adipose wasting lean_loss congestion disability steroid"),
    "BIDTBW": ("Fat-free mass times bounded hydration fraction, in litres.", "age sex adiposity inactive frailty", "adipose wasting lean_loss height_loss congestion disability steroid"),
    "BMXBMI": ("Age-dependent BMI with shared adiposity, activity and frailty.", "age adiposity inactive frailty", "adipose wasting lean_loss congestion disability steroid"),
    "BMXHT": ("Sex-specific assumed growth/ageing curve with individual variation.", "age sex", "height_loss"),
    "BMXWAIST": ("Age/sex body size and BMI determine a noisy waist circumference.", "age sex adiposity inactive frailty", "adipose wasting lean_loss congestion disability steroid"),
    "BMXWT": ("BMI times squared height in metres; measured mass identity.", "age sex adiposity inactive frailty", "adipose wasting lean_loss height_loss congestion disability steroid"),
    "BPXCHR": ("Age-dependent pulse with smoking, inactivity, inflammation and anaemia.", "age smoke inactive inflammation renal", "tachy inflammation anemia renal disability steroid"),
    "BPXDI1": ("Diastolic pressure from age, pressure liability and renal burden.", "age pressure renal", "bp renal"),
    "BPXSY1": ("Systolic pressure from age, sex, liability and renal burden; above diastolic.", "age sex pressure renal", "bp renal"),
    "LBXSAPSI": ("Alkaline phosphatase includes a broad childhood growth component.", "age", "bone_turnover cholestasis"),
    "LBXSATSI": ("Lognormal ALT responds to shared hepatic stress and hepatic activity.", "age metabolic alcohol liver", "hepatic"),
    "LBXSASSI": ("Lognormal AST responds to hepatic stress and alcohol exposure.", "age metabolic alcohol liver", "hepatic"),
    "LBXSGTSI": ("GGT shares hepatic/alcohol variation and cholestatic activity.", "age metabolic alcohol liver", "hepatic cholestasis"),
    "LBDBANO": ("WBC times basophil percentage / 100.", "age adiposity smoke inflammation immune atopy", "inflammation steroid eos"),
    "LBXBAPCT": ("Softmax differential; changes with competing leukocyte populations.", "age adiposity smoke inflammation immune atopy", "inflammation steroid eos"),
    "LBDEONO": ("WBC times eosinophil percentage / 100.", "age adiposity smoke inflammation immune atopy", "inflammation steroid eos"),
    "LBXEOPCT": ("Differential eosinophils respond to atopy and heterogeneous asthma activity.", "age adiposity smoke inflammation immune atopy", "inflammation steroid eos"),
    "LBXHCT": ("RBC count times latent MCV / 10; coherent with haemoglobin.", "age sex alcohol iron renal inflammation", "anemia renal inflammation"),
    "LBXHGB": ("RBC count times MCH / 10, in g/dL.", "age sex alcohol iron renal inflammation", "anemia renal inflammation"),
    "LBDLYMNO": ("WBC times lymphocyte percentage / 100.", "age adiposity smoke inflammation immune atopy", "inflammation steroid eos"),
    "LBXLYPCT": ("Age-sensitive softmax lymphocytes compete with other cell classes.", "age adiposity smoke inflammation immune atopy", "inflammation steroid eos"),
    "LBXMC": ("MCHC depends on iron limitation with independent cellular variation.", "age sex iron", ""),
    "LBXMCHSI": ("MCV times MCHC / 100; age, iron and alcohol influence cell size.", "age sex iron alcohol", ""),
    "LBXMPSI": ("Platelet volume includes weak inverse count dependence and metabolic variation.", "age metabolic inflammation", "inflammation platelet_suppression"),
    "LBDMONO": ("WBC times monocyte percentage / 100.", "age adiposity smoke inflammation immune atopy", "inflammation steroid eos"),
    "LBXMOPCT": ("Softmax monocytes increase modestly with inflammation.", "age adiposity smoke inflammation immune atopy", "inflammation steroid eos"),
    "LBXPLTSI": ("Platelets vary with inflammatory stimulation and selected hepatic suppression.", "age inflammation", "inflammation platelet_suppression"),
    "LBXRBCSI": ("RBC number includes sex, iron, renal and inflammatory influences.", "age sex iron renal inflammation", "anemia renal inflammation"),
    "LBXRDW": ("Cell-size heterogeneity includes iron, frailty, renal burden and anaemia.", "age sex iron frailty renal", "anemia renal"),
    "LBDNENO": ("WBC times neutrophil percentage / 100.", "age adiposity smoke inflammation immune atopy", "inflammation steroid eos"),
    "LBXNEPCT": ("Softmax neutrophils respond to inflammation and systemic steroid exposure.", "age adiposity smoke inflammation immune atopy", "inflammation steroid eos"),
    "LBXWBCSI": ("Positive WBC count shares inflammatory, smoking and steroid influences.", "age adiposity smoke inflammation immune", "inflammation steroid"),
    "SSCYPC": ("Cystatin C increases with renal burden and weakly with inflammation.", "age renal inflammation", "renal inflammation"),
    "LBXGH": ("Glycohaemoglobin shares metabolic/glycaemic burden, iron and renal variation.", "age sex metabolic iron renal", "glyco steroid renal"),
    "LBXGLU": ("Plasma glucose shares metabolic burden with independent short-term noise.", "age metabolic", "glucose steroid"),
    "LBXCRP": ("CRP is lognormal in mg/dL and responds to shared inflammatory activity.", "age adiposity smoke immune inflammation", "inflammation"),
    "LBDHDL": ("HDL depends on sex, metabolic liability and alcohol with multiplicative noise.", "age sex metabolic alcohol", "hdl"),
    "LBXTC": ("Total cholesterol sums HDL, a positive LDL-like component and 0.18 TG.", "age sex metabolic lipid alcohol", "hdl lipid tg"),
    "LBXSAL": ("Albumin reflects inflammation, renal burden, protein loss and synthetic function.", "age renal inflammation", "inflammation renal synthetic_failure protein_loss"),
    "LBXSTB": ("Independent bilirubin variation plus specified cholestatic/hepatic dysfunction.", "", "cholestasis synthetic_failure"),
    "LBXSBU": ("Urea nitrogen depends on age, renal burden and congestion.", "age renal", "renal congestion"),
    "LBXSCA": ("Total calcium includes an albumin-bound component and small renal/bone terms.", "age renal inflammation", "inflammation renal synthetic_failure protein_loss bone_turnover"),
    "LBXSCR": ("Creatinine combines age/sex, relative lean mass and reduced clearance.", "age sex adiposity inactive frailty renal", "renal adipose wasting lean_loss congestion disability steroid"),
    "LBXSPH": ("Phosphate has childhood growth, renal and small bone-turnover components.", "age renal", "renal bone_turnover"),
    "LBXSTP": ("Total protein equals albumin plus a positive globulin component.", "age renal inflammation", "inflammation renal synthetic_failure protein_loss"),
    "LBXSTR": ("Lognormal triglycerides share metabolic, lipid and alcohol influences.", "age metabolic lipid alcohol", "tg"),
    "LBXSUA": ("Urate shares sex, metabolic, renal and selected diuretic influences.", "age sex metabolic renal", "renal diuretic"),
}

# Tuple: required minimum age, sex, risk dependencies, possible direct pathways,
# assumed probability of a currently active episode given history, clinical rule.
ENDPOINT_SPEC = {
    "I9_HYPTENSESS": (16, "both", "age sex pressure metabolic renal", "bp renal diuretic", .85,
        "Stochastic hypertension history from pressure/metabolic liability; current pressure and heterogeneous antihypertensive/diuretic effects."),
    "E4_HYPERCHOL": (20, "both", "age sex lipid metabolic", "lipid tg hdl", .85,
        "Broad reported high cholesterol from lipid/metabolic liability; LDL-like and triglyceride mechanisms plus variable treatment."),
    "E4_DM2": (1, "both", "age sex metabolic adiposity", "glucose glyco renal wasting", .85,
        "Type-unspecified diagnosed diabetes is the union of insulin-resistant and insulin-deficient processes; no type-2-only restriction."),
    "N14_CHRONKIDNEYDIS": (20, "both", "age sex renal pressure metabolic", "renal anemia protein_loss", .85,
        "Reported weak/failing kidneys from latent renal burden, with clearance, anaemia and occasional protein-loss mechanisms."),
    "I9_ISCHHEART": (20, "both", "age sex pressure lipid smoke metabolic", "inflammation tachy lipid tg disability bp", .35,
        "Union concept of coronary heart disease, angina or myocardial infarction; broad history with often low current activity and secondary prevention."),
    "I9_HEARTFAIL": (20, "both", "age sex pressure renal metabolic", "congestion renal inflammation tachy bp diuretic disability hepatic lean_loss", .75,
        "Reported congestive heart failure with heterogeneous congestion, cardiorenal burden, reduced mobility and treatment."),
    "K11_LIVER": (20, "both", "age sex liver metabolic alcohol", "hepatic cholestasis synthetic_failure platelet_suppression anemia wasting", .55,
        "Broad reported liver condition includes metabolic, alcohol-associated and other inflammatory/cholestatic mechanisms; synthetic dysfunction is a subset."),
    "M13_OSTEOPOROSIS": (20, "both", "age sex bone frailty adiposity smoke", "bone_turnover disability lean_loss height_loss", .65,
        "Reported osteoporosis follows age/sex/bone liability; many routine blood measures remain normal, with only occasional remodeling/vertebral effects."),
    "N14_PROSTHYPERPLA": (30, "male", "age metabolic", "renal", .75,
        "Reported benign prostatic enlargement in men age 30+; routine blood features have no prostate-specific marker, with rare modeled obstruction."),
    "C3_PROSTATE_EXALLC": (30, "male", "age", "bone_turnover anemia inflammation wasting lean_loss adipose", .45,
        "Reported prostate cancer in men age 30+; systemic/bone changes are limited to a hypothetical advanced subset and androgen-treatment subset."),
    "J10_ASTHMA_MAIN_EXMORE": (1, "both", "age sex atopy adiposity", "eos inflammation steroid tachy", .55,
        "Reported asthma history without a hospital/main-diagnosis restriction; eosinophilic and non-eosinophilic activity plus uncommon systemic steroids."),
    "I9_STR_SAH": (20, "both", "age sex pressure smoke lipid", "disability lean_loss bp lipid", .50,
        "Broad reported stroke, not subarachnoid hemorrhage only; ischemic/nonischemic mixture with residual disability and heterogeneous prevention."),
    "M13_ARTHROSIS": (20, "both", "age sex adiposity inactive", "disability inflammation bone_turnover", .70,
        "Reported osteoarthritis subtype with mechanical/age/adiposity liability and mobility consequences; low inflammatory activity."),
    "M13_RHEUMA": (20, "both", "age sex smoke immune", "inflammation anemia steroid lean_loss disability bone_turnover hepatic", .70,
        "Reported rheumatoid arthritis subtype with immune liability, variable inflammatory activity, normocytic anaemia and treatment effects."),
}

PATHWAY_TEXT = {
    "bp": "disease/current treatment modifies pressure",
    "renal": "disease-related clearance impairment propagates to renal/body-fluid/CBC measures",
    "diuretic": "selected treated individuals have increased urate",
    "lipid": "lipid burden or lipid-lowering treatment changes the LDL-like component",
    "tg": "broad dyslipidaemia or treatment modifies triglycerides",
    "hdl": "heterogeneous lipid treatment modestly modifies HDL",
    "glucose": "heterogeneous diabetes activity/treatment modifies glucose",
    "glyco": "heterogeneous diabetes activity/treatment modifies glycohaemoglobin",
    "wasting": "selected active disease causes wasting and derived body-composition changes",
    "anemia": "active disease reduces RBC number and modifies related indices",
    "protein_loss": "a subset has renal protein loss",
    "inflammation": "current inflammatory burden propagates through blood cells and chemistry",
    "tachy": "active disease or treatment changes pulse",
    "disability": "reduced mobility changes body composition and pulse",
    "congestion": "congestion changes hydration/body mass and nitrogen handling",
    "hepatic": "hepatic injury or a selected treatment effect changes enzymes",
    "lean_loss": "active disease or selected treatment reduces relative lean mass",
    "cholestasis": "selected liver conditions increase cholestatic markers",
    "synthetic_failure": "a liver-dysfunction subset alters albumin and related chemistry",
    "platelet_suppression": "a liver-dysfunction subset reduces platelets",
    "bone_turnover": "selected remodeling/bone involvement affects ALP and small mineral terms",
    "height_loss": "selected vertebral consequences reduce stature",
    "adipose": "selected androgen treatment increases relative adiposity",
    "eos": "the eosinophilic asthma subset alters leukocyte composition",
    "steroid": "selected systemic steroid exposure affects metabolism and leukocytes",
}


def load_inputs(schema_path, disease_path):
    """Read only the two explicitly supplied interface files, never extra paths."""
    with open(schema_path, newline="", encoding="utf-8-sig") as handle:
        schema = list(csv.DictReader(handle))
    with open(disease_path, newline="", encoding="utf-8-sig") as handle:
        identities = list(csv.DictReader(handle))
    names = [row["native_name"] for row in schema]
    if names != list(FEATURE_SPEC):
        raise ValueError("Schema must contain the supplied 52 native names in their supplied order")
    codes = [row["endpoint"] for row in identities]
    if len(codes) != len(set(codes)) or set(codes) != set(ENDPOINT_SPEC):
        raise ValueError("Disease identities must contain exactly the 14 supported endpoints")
    for row in identities:
        expected = ENDPOINT_SPEC[row["endpoint"]]
        if int(row["eligible_age_min"]) != expected[0] or row["eligible_sex"] != expected[1]:
            raise ValueError("Endpoint eligibility differs from the construction contract")
    return schema, identities


def stable_seed(seed, endpoint, block=0, stream="physiology"):
    """Stable identity seeds; independent of invocation subset/order and hash salt."""
    payload = f"{VERSION}|{int(seed)}|{endpoint}|{int(block)}|{stream}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:16], "little")


def _require_numeric():
    if np is None or pd is None:
        raise RuntimeError("NumPy and pandas are required for generation and numeric tests")


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -35, 35)))


def _softplus(x):
    return np.logaddexp(0, x)


def _background(rng, n, endpoint):
    minimum, sex_rule = ENDPOINT_SPEC[endpoint][:2]
    ages = np.arange(minimum, 96)
    # This smooth sampling law is an author-chosen design distribution, not a
    # survey-weighted age distribution or an estimate of the US population.
    weights = np.exp(-ages / 110) * (1 + .4 * np.exp(-.5 * ((ages - 35) / 22) ** 2))
    age = rng.choice(ages, n, p=weights / weights.sum()).astype(float)
    male = np.ones(n) if sex_rule == "male" else rng.integers(0, 2, n).astype(float)
    old = (age - 45) / 20
    adiposity = rng.normal(0, 1, n) + .20 * np.clip(old, -1, 1)
    inactive = rng.normal(0, 1, n)
    smoke = (rng.random(n) < _sigmoid(-1.1 + .15 * old)) * rng.uniform(.3, 1.5, n) * (age >= 16)
    alcohol = np.minimum(rng.lognormal(-.9, .7, n), 4) * (age >= 18)
    metabolic = .75 * adiposity + .30 * inactive + .20 * old + rng.normal(0, .65, n)
    lipid = .45 * metabolic + .20 * alcohol + rng.normal(0, .8, n)
    pressure = .45 * metabolic + .40 * old + .20 * smoke + rng.normal(0, .75, n)
    renal = _softplus(-2.4 + .7 * old + .4 * pressure + .3 * metabolic + rng.normal(0, .65, n))
    immune = rng.normal(0, 1, n)
    inflammation = _softplus(-.6 + .3 * adiposity + .2 * smoke + .2 * immune + rng.normal(0, .55, n))
    inflammation += (rng.random(n) < .02) * np.minimum(rng.lognormal(0, .6, n), 4)
    atopy = rng.lognormal(-.25, .8, n)
    frailty = _softplus(-1.6 + .65 * old + .25 * inactive + rng.normal(0, .6, n))
    bone = .65 * old + .85 * (1 - male) * (age > 50) + .3 * frailty - .25 * adiposity + .2 * smoke + rng.normal(0, .65, n)
    liver = _softplus(-1.6 + .55 * metabolic + .50 * alcohol + rng.normal(0, .7, n))
    iron = (rng.random(n) < _sigmoid(-2.5 + .8 * (1 - male) * (age < 50))) * rng.gamma(2, .4, n)
    return dict(age=age, male=male, old=old, adiposity=adiposity, inactive=inactive,
                smoke=smoke, alcohol=alcohol, metabolic=metabolic, lipid=lipid,
                pressure=pressure, renal=renal, immune=immune, inflammation=inflammation,
                atopy=atopy, frailty=frailty, bone=bone, liver=liver, iron=iron)


def _disease(rng, b, endpoint):
    """Reported diagnosis history, distinct from noisy current measurements."""
    n = len(b["age"])
    a, m, old = b["age"], b["male"], b["old"]
    met, lip, pressure, renal = (b[k] for k in ("metabolic", "lipid", "pressure", "renal"))
    logits = {
        "I9_HYPTENSESS": -1.5 + 1.35 * pressure + .35 * met + .25 * old + .2 * m,
        "E4_HYPERCHOL": -1.45 + 1.1 * lip + .5 * old + .25 * met + .2 * m,
        "N14_CHRONKIDNEYDIS": -3.1 + 1.25 * renal + .45 * old + .35 * pressure + .1 * m,
        "I9_ISCHHEART": -3.0 + .7 * old + .5 * pressure + .5 * lip + .45 * b["smoke"] + .3 * met + .45 * m,
        "I9_HEARTFAIL": -3.5 + .85 * old + .7 * pressure + .55 * renal + .25 * met + .18 * m,
        "K11_LIVER": -2.8 + b["liver"] + .2 * met + .35 * b["alcohol"] + .2 * m,
        "M13_OSTEOPOROSIS": -3.2 + 1.3 * b["bone"] + .4 * old,
        "N14_PROSTHYPERPLA": -2.6 + 1.05 * old + .2 * met,
        "C3_PROSTATE_EXALLC": -3.7 + 1.35 * old,
        "J10_ASTHMA_MAIN_EXMORE": -2.4 + .75 * np.log1p(b["atopy"]) + .15 * b["adiposity"] + .2 * (1 - m) * (a > 20) + .3 * (a < 18),
        "I9_STR_SAH": -3.5 + .8 * old + .7 * pressure + .35 * b["smoke"] + .2 * lip + .1 * m,
        "M13_ARTHROSIS": -2.5 + old + .45 * b["adiposity"] + .35 * b["inactive"] + .3 * (1 - m),
        "M13_RHEUMA": -3.3 + .3 * old + .6 * (1 - m) + .3 * b["smoke"] + .7 * b["immune"],
    }
    deficient = np.zeros(n)
    if endpoint == "E4_DM2":
        deficient = (rng.random(n) < _sigmoid(-4.1 - .015 * (a - 20))).astype(float)
        resistant = rng.random(n) < (_sigmoid(-3.1 + 1.1 * met + .55 * old + .3 * b["adiposity"] + .12 * m) * np.clip(a / 20, .05, 1))
        present = (deficient > 0) | resistant
    else:
        present = rng.random(n) < _sigmoid(logits[endpoint])
    care = rng.normal(0, 1, n)
    diagnosed = present & (rng.random(n) < _sigmoid(1.15 + .2 * old + .4 * care))
    recalled = rng.random(n) < _sigmoid(2.3 + .15 * care)
    # Small independent misreporting is an explicit design assumption, not a
    # measured sensitivity/specificity estimate.
    label = ((diagnosed & recalled) | (~present & (rng.random(n) < .003))).astype(np.int8)
    treatment = diagnosed * (rng.random(n) < .75) * rng.uniform(.4, 1, n)
    active = present * (.15 + .85 * (rng.random(n) < ENDPOINT_SPEC[endpoint][4])) * np.minimum(rng.lognormal(-.25, .65, n), 4)
    x = active
    t = treatment
    effects = {key: np.zeros(n) for key in PATHWAY_TEXT}
    e = effects
    if endpoint == "I9_HYPTENSESS":
        e["bp"] = 16 * x * (1 - .5 * t) - 10 * t
        e["renal"] = .12 * x
        e["diuretic"] = .4 * t * (rng.random(n) < .35)
    elif endpoint == "E4_HYPERCHOL":
        e["lipid"] = 45 * x * (1 - .65 * t) - 20 * t
        e["tg"] = .20 * x * (1 - .5 * t) - .10 * t
        e["hdl"] = 2 * t * rng.uniform(0, 1, n)
    elif endpoint == "E4_DM2":
        e["glucose"] = 75 * x * (1 + deficient) * (1 - .7 * t)
        e["glyco"] = 2.6 * x * (1 + .5 * deficient) * (1 - .7 * t)
        e["renal"] = .15 * x
        e["wasting"] = .9 * x * deficient
    elif endpoint == "N14_CHRONKIDNEYDIS":
        e["renal"] = 1.15 * x * (1 - .2 * t)
        e["anemia"] = .35 * x
        e["protein_loss"] = .35 * x * (rng.random(n) < .3)
    elif endpoint == "I9_ISCHHEART":
        e["inflammation"], e["tachy"] = .12 * x, -6 * t
        e["lipid"], e["tg"] = -28 * t, -.1 * t
        e["disability"], e["bp"] = .25 * x, -6 * t
    elif endpoint == "I9_HEARTFAIL":
        e["congestion"] = .9 * x * (1 - .6 * t)
        e["renal"], e["inflammation"] = .3 * x, .15 * x
        e["tachy"], e["bp"], e["diuretic"] = 8 * x - 8 * t, -8 * t, .8 * t
        e["disability"], e["hepatic"], e["lean_loss"] = .5 * x, .12 * e["congestion"], .4 * x
    elif endpoint == "K11_LIVER":
        chronic = rng.random(n) < .3
        e["hepatic"] = 1.2 * x * (1 - .4 * t)
        e["cholestasis"] = .8 * x * (rng.random(n) < .25)
        e["synthetic_failure"], e["platelet_suppression"] = .35 * x * chronic, .5 * x * chronic
        e["anemia"], e["wasting"] = .25 * x * chronic, .4 * x * chronic
    elif endpoint == "M13_OSTEOPOROSIS":
        e["bone_turnover"] = .25 * x * (1 - .5 * t) * (rng.random(n) < .35)
        e["disability"], e["lean_loss"] = .2 * x, .15 * x
        e["height_loss"] = .8 * x * (rng.random(n) < .25)
    elif endpoint == "N14_PROSTHYPERPLA":
        e["renal"] = .15 * x * (rng.random(n) < .12)
    elif endpoint == "C3_PROSTATE_EXALLC":
        advanced = rng.random(n) < .10
        androgen = t * (rng.random(n) < .20)
        e["bone_turnover"], e["anemia"] = .8 * x * advanced, .5 * x * advanced
        e["inflammation"], e["wasting"] = .4 * x * advanced, .8 * x * advanced
        e["lean_loss"], e["adipose"] = .5 * androgen, .35 * androgen
    elif endpoint == "J10_ASTHMA_MAIN_EXMORE":
        e["eos"] = .9 * x * (1 - .55 * t) * (rng.random(n) < .65)
        e["inflammation"], e["tachy"] = .12 * x, 3 * x
        e["steroid"] = .25 * t * (rng.random(n) < .25)
    elif endpoint == "I9_STR_SAH":
        e["disability"], e["lean_loss"] = .5 * x, .3 * x
        e["bp"], e["lipid"] = -7 * t, -20 * t * (rng.random(n) < .8)
    elif endpoint == "M13_ARTHROSIS":
        e["disability"], e["inflammation"], e["bone_turnover"] = .35 * x, .05 * x, .08 * x
    elif endpoint == "M13_RHEUMA":
        e["inflammation"], e["anemia"] = 1.1 * x * (1 - .7 * t), .4 * x
        e["steroid"] = .3 * t * (rng.random(n) < .4)
        e["lean_loss"], e["disability"], e["bone_turnover"] = .4 * x, .3 * x, .1 * x
        e["hepatic"] = .1 * t * (rng.random(n) < .1)
    return label, effects


def _measurements(rng, b, e):
    n = len(b["age"])
    age, male = b["age"], b["male"]
    female = 1 - male
    met = b["metabolic"]
    renal = 6 * np.tanh((b["renal"] + e["renal"]) / 6)
    infl = b["inflammation"] + e["inflammation"]
    noise = lambda sd: rng.normal(0, sd, n)
    mult = lambda sd: np.exp(noise(sd))
    height_m = np.interp(age, [1, 2, 5, 10, 15, 20, 40, 60, 80, 95], [78, 88, 110, 139, 170, 178, 178, 177, 174, 171])
    height_f = np.interp(age, [1, 2, 5, 10, 15, 20, 40, 60, 80, 95], [76, 86, 109, 138, 161, 164, 164, 162, 159, 156])
    height = np.clip(male * height_m + female * height_f + noise(.9 + 4 * (1 - np.exp(-age / 5))) - e["height_loss"], 60, 220)
    maturity = np.clip((age - 8) / 14, 0, 1)
    bmi = 15.5 + 11 * maturity + (1.7 + 2.1 * maturity) * b["adiposity"] + .45 * b["inactive"]
    bmi += -.3 * b["frailty"] + e["adipose"] - .9 * e["wasting"] - .6 * e["lean_loss"] + .65 * e["congestion"] + .4 * e["disability"] + .4 * e["steroid"] + noise(.8)
    bmi = np.clip(bmi, 10, 65)
    weight = bmi * (height / 100) ** 2
    fraction = _sigmoid(-1.3 + .52 * female + .045 * (bmi - 24) + .008 * np.maximum(age - 35, 0) + .08 * e["wasting"] + .10 * e["lean_loss"] + noise(.18))
    fat = weight * fraction
    lean = weight - fat
    hydration = np.clip(.732 + noise(.012) + .02 * e["congestion"], .64, .81)
    waist = np.maximum(30, 38 + .85 * age.clip(0, 20) + 1.45 * bmi + 3 * male + noise(5))
    dia = np.clip(57 + .23 * age - .18 * np.maximum(age - 60, 0) + 5 * b["pressure"] + .35 * e["bp"] + 1.2 * renal + noise(5), 30, 160)
    sys = np.maximum(dia + 18, 95 + .55 * age + 8 * b["pressure"] + 3 * male + e["bp"] + 2 * renal + noise(8))
    iron = b["iron"]
    mchc = np.clip(33.8 - .65 * iron + noise(.65), 28, 37)
    mcv = np.clip(88 - 12 * np.exp(-age / 8) - 5 * iron + 3 * b["alcohol"] + noise(3.5), 55, 115)
    rbc = np.clip(4.45 + .5 * male - .25 * np.exp(-age / 10) - .10 * iron - .22 * renal - .12 * infl - .28 * e["anemia"] + noise(.32), 1.2, 7.5)
    mch = mcv * mchc / 100
    hgb = rbc * mch / 10
    hct = rbc * mcv / 10
    wbc = np.clip(6.2 * np.exp(.15 * infl + .07 * b["smoke"] + .12 * e["steroid"]) * mult(.23), 1, 80)
    logits = np.column_stack([
        np.log(.55) + .20 * infl + .22 * e["steroid"] + noise(.22),
        np.log(.34) + .4 * np.exp(-age / 10) + noise(.22),
        np.log(.065) + .08 * infl + noise(.20),
        np.log(.035) + .35 * np.log1p(b["atopy"]) + e["eos"] - .25 * e["steroid"] + noise(.38),
        np.log(.01) + noise(.30),
    ])
    exp_logits = np.exp(logits - logits.max(axis=1, keepdims=True))
    pct = 100 * exp_logits / exp_logits.sum(axis=1, keepdims=True)
    platelets = np.clip((245 + 25 * infl - 35 * e["platelet_suppression"]) * mult(.20), 20, 950)
    albumin = np.clip(4.3 - .08 * infl - .18 * e["synthetic_failure"] - .10 * e["protein_loss"] - .04 * renal + noise(.18), 1.3, 5.7)
    hdl = np.maximum(12, (52 + 7 * female - 4.5 * met + 2 * b["alcohol"] + e["hdl"]) * mult(.15))
    tg = np.maximum(15, np.exp(np.log(100) + .25 * met + .20 * b["lipid"] + .15 * b["alcohol"] + e["tg"] + noise(.35)))
    ldl_like = np.maximum(15, (100 + .5 * age + 22 * b["lipid"] + e["lipid"]) * mult(.16))
    growth = 100 * np.exp(-((age - 12) / 4) ** 2) * (age < 20) + 35 * (age < 6)
    values = {
        "RIDAGEYR": np.minimum(age, 85), "RIAGENDR": 2 - male,
        "DMDHHSIZ": np.minimum(1 + rng.poisson(1.8, n), 7).astype(float),
        "BIDFAT": fat, "BIDFFM": lean, "BIDPFAT": 100 * fraction,
        "BIDTBW": lean * hydration, "BMXBMI": bmi, "BMXHT": height,
        "BMXWAIST": waist, "BMXWT": weight,
        "BPXCHR": np.clip(69 + 35 * np.exp(-age / 8) + 2 * b["inactive"] + 2 * b["smoke"] + 2 * infl + 1.2 * renal + 2 * e["anemia"] + e["tachy"] + e["disability"] + e["steroid"] + noise(8), 35, 190),
        "BPXDI1": dia, "BPXSY1": sys,
        "LBXSAPSI": np.maximum(10, (65 + growth + 12 * e["bone_turnover"] + 30 * e["cholestasis"]) * mult(.22)),
        "LBXSATSI": np.exp(np.log(20) + .25 * b["liver"] + .35 * e["hepatic"] + noise(.35)),
        "LBXSASSI": np.exp(np.log(21) + .15 * b["liver"] + .22 * e["hepatic"] + .20 * b["alcohol"] + noise(.28)),
        "LBXSGTSI": np.exp(np.log(22) + .40 * b["liver"] + .50 * b["alcohol"] + .40 * e["hepatic"] + .45 * e["cholestasis"] + noise(.40)),
        "LBDBANO": wbc * pct[:, 4] / 100, "LBXBAPCT": pct[:, 4],
        "LBDEONO": wbc * pct[:, 3] / 100, "LBXEOPCT": pct[:, 3],
        "LBXHCT": hct, "LBXHGB": hgb,
        "LBDLYMNO": wbc * pct[:, 1] / 100, "LBXLYPCT": pct[:, 1],
        "LBXMC": mchc, "LBXMCHSI": mch,
        "LBXMPSI": np.clip(9.4 - .002 * (platelets - 245) + .15 * met + noise(.7), 5, 17),
        "LBDMONO": wbc * pct[:, 2] / 100, "LBXMOPCT": pct[:, 2],
        "LBXPLTSI": platelets, "LBXRBCSI": rbc,
        "LBXRDW": np.clip(12.7 + .9 * iron + .25 * b["frailty"] + .3 * renal + .45 * e["anemia"] + noise(.6), 8, 40),
        "LBDNENO": wbc * pct[:, 0] / 100, "LBXNEPCT": pct[:, 0], "LBXWBCSI": wbc,
        "SSCYPC": (.65 + .003 * age) * np.exp(.43 * renal + .06 * infl) * mult(.10),
        "LBXGH": np.clip((5.1 + .23 * met + e["glyco"] + .10 * e["steroid"] + .06 * iron - .06 * renal) * mult(.025), 3, 25),
        "LBXGLU": np.clip((86 + 7 * met + 5 * e["steroid"] + e["glucose"]) * mult(.09), 40, 900),
        "LBXCRP": np.exp(np.log(.08) + .8 * infl + .12 * b["adiposity"] + noise(.65)),
        "LBDHDL": hdl, "LBXTC": hdl + ldl_like + .18 * tg,
        "LBXSAL": albumin,
        "LBXSTB": .6 * mult(.35) + .45 * e["cholestasis"] + .12 * e["synthetic_failure"],
        "LBXSBU": np.maximum(2, (11 + .07 * age + .5 * e["congestion"]) * np.exp(.40 * renal) * mult(.16)),
        "LBXSCA": np.maximum(4, 9.4 + .6 * (albumin - 4.3) - .10 * renal + .08 * e["bone_turnover"] + noise(.18)),
        "LBXSCR": np.maximum(.1, (.65 + .18 * male + .002 * age + .10 * (lean / (height / 100) ** 2 - 18) / 5) * np.exp(.45 * renal) * mult(.10)),
        "LBXSPH": np.maximum(1, 3.5 + 1.5 * np.exp(-age / 8) + .30 * renal + .05 * e["bone_turnover"] + noise(.35)),
        "LBXSTP": albumin + np.maximum(1, 2.6 + .15 * infl + noise(.22)),
        "LBXSTR": tg,
        "LBXSUA": np.maximum(.5, (4.5 + male + .25 * met + .30 * renal + .30 * e["diuretic"]) * mult(.15)),
    }
    return values


def missingness_mask(n, rng):
    """Assumed panel attendance, independent of labels AND all physiology."""
    mask = np.zeros((n, len(FEATURE_SPEC)), dtype=bool)
    names = list(FEATURE_SPEC)
    blood = rng.random(n) < .04
    bioimpedance = rng.random(n) < .16
    anthropometry = rng.random(n) < .025
    pressure = rng.random(n) < .035
    cystatin = rng.random(n) < .55
    fasting = rng.random(n) < .50
    for j, name in enumerate(names):
        if j < 3:
            continue
        group = bioimpedance if name.startswith("BID") else anthropometry if name.startswith("BMX") else pressure if name.startswith("BPX") else blood
        mask[:, j] = group | (rng.random(n) < .006)
        if name == "SSCYPC":
            mask[:, j] |= cystatin
        elif name == "LBXGLU":
            mask[:, j] |= fasting
    return mask


def validate_frame(frame, schema, endpoint):
    if list(frame.columns) != [s["native_name"] for s in schema] + [endpoint]:
        raise ValueError("Output columns or order differ from contract")
    for row in schema:
        data = frame[row["native_name"]].to_numpy()
        observed = data[~np.isnan(data)]
        if not np.isfinite(observed).all():
            raise ValueError("Nonfinite measurement: " + row["native_name"])
        for bound, op in (("hard_min", np.less), ("hard_max", np.greater)):
            if row[bound] and op(observed, float(row[bound])).any():
                raise ValueError("Measurement violates " + bound + ": " + row["native_name"])
        if row["type"] == "integer" and not np.equal(observed, np.floor(observed)).all():
            raise ValueError("Noninteger coded measurement")
        if row["valid_codes"]:
            codes = [int(item.split("=")[0]) for item in row["valid_codes"].split(";")]
            if not np.isin(observed, codes).all():
                raise ValueError("Invalid categorical code")
    label = frame[endpoint].to_numpy()
    if not np.isin(label, [0, 1]).all():
        raise ValueError("Nonbinary label")
    if frame["RIDAGEYR"].isna().any() or (frame["RIDAGEYR"] < ENDPOINT_SPEC[endpoint][0]).any():
        raise ValueError("Age eligibility violated")
    if frame["RIAGENDR"].isna().any() or (ENDPOINT_SPEC[endpoint][1] == "male" and (frame["RIAGENDR"] != 1).any()):
        raise ValueError("Sex eligibility violated")


def iter_population(endpoint, n, seed, schema, *, missing=True):
    """Yield fixed canonical blocks; all random draws use bounded 8192-row arrays."""
    _require_numeric()
    if endpoint not in ENDPOINT_SPEC or not isinstance(n, int) or n < 1:
        raise ValueError("A supported endpoint and positive integer n are required")
    for block, start in enumerate(range(0, n, BLOCK_SIZE)):
        rng = np.random.default_rng(stable_seed(seed, endpoint, block))
        background = _background(rng, BLOCK_SIZE, endpoint)
        label, effects = _disease(rng, background, endpoint)
        values = _measurements(rng, background, effects)
        frame = pd.DataFrame(values, columns=list(FEATURE_SPEC))
        if missing:
            mask_rng = np.random.default_rng(stable_seed(seed, endpoint, block, "missingness"))
            frame = frame.mask(missingness_mask(BLOCK_SIZE, mask_rng))
        frame[endpoint] = label
        frame = frame.iloc[:min(BLOCK_SIZE, n - start)].copy()
        validate_frame(frame, schema, endpoint)
        yield frame


def generate_population(endpoint, n, seed, schema, *, missing=True):
    """In-memory interface for small fixtures; CLI uses iter_population instead."""
    _require_numeric()
    return pd.concat(iter_population(endpoint, n, seed, schema, missing=missing), ignore_index=True)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_rule_ledger(path, schema, identities):
    fields = ["endpoint", "clinical_name", "native_name", "relationship", "implemented_rationale", "common_dependencies", "direct_pathways", "status"]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for identity in identities:
            endpoint = identity["endpoint"]
            spec = ENDPOINT_SPEC[endpoint]
            for row in schema:
                feature = row["native_name"]
                base, risks, paths = FEATURE_SPEC[feature]
                common = sorted(set(risks.split()) & set(spec[2].split()))
                direct = sorted(set(paths.split()) & set(spec[3].split()))
                if feature in ("RIDAGEYR", "RIAGENDR"):
                    relation = "direct"
                    why = "Eligibility/diagnostic-liability or ascertainment rule: " + (f"age >= {spec[0]}, latent age enters diagnosis/ascertainment" if feature == "RIDAGEYR" else ("male-only eligibility; constant male code" if spec[1] == "male" else "sex enters endpoint liability directly or through bone liability"))
                elif direct:
                    relation = "direct"
                    why = "; ".join(PATHWAY_TEXT[p] for p in direct)
                elif common:
                    relation = "indirect"
                    why = "Shared prediagnosis background: " + ", ".join(common) + "; no endpoint-specific term in this measurement"
                else:
                    relation = "neutral"
                    why = "No specified endpoint pathway/shared risk term; retained background variation, with no forced case-control shift"
                writer.writerow(dict(endpoint=endpoint, clinical_name=identity["clinical_name"], native_name=feature,
                    relationship=relation, implemented_rationale=base + " " + why + ". " + spec[5],
                    common_dependencies=";".join(common), direct_pathways=";".join(direct),
                    status="authored knowledge assumption; not empirically fitted or validated"))


def write_population(endpoint, n, seed, schema, output_dir, input_hashes):
    """Atomically publish one CSV; exclusive claim prevents concurrent overwrite."""
    folder = Path(output_dir) / endpoint
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / f"synthetic_{endpoint}_{n}.csv"
    audit_path = folder / f"synthetic_{endpoint}_{n}.audit.json"
    partial = folder / (target.name + ".partial")
    lock = folder / (target.name + ".lock")
    if target.exists() or audit_path.exists() or partial.exists():
        raise FileExistsError(f"Refusing to overwrite completed or partial output: {target}")
    lock_fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.close(lock_fd)
    published = False
    started = time.monotonic()
    count = positives = 0
    max_rows = 0
    try:
        # Recheck under the exclusive claim in case a prior writer just finished.
        if target.exists() or audit_path.exists() or partial.exists():
            raise FileExistsError(f"Refusing to overwrite output: {target}")
        with open(partial, "x", newline="", encoding="utf-8") as handle:
            for frame in iter_population(endpoint, n, seed, schema):
                frame.to_csv(handle, index=False, header=(count == 0), float_format="%.8g", na_rep="")
                count += len(frame)
                positives += int(frame[endpoint].sum())
                max_rows = max(max_rows, len(frame))
        # link is an atomic no-replace publication on this filesystem.
        os.link(partial, target)
        published = True
        partial.unlink()
        audit = dict(generator_version=VERSION, endpoint=endpoint, rows=count, predictor_count=52,
            label=endpoint, seed=int(seed), disease_seed_hex=hex(stable_seed(seed, endpoint)),
            canonical_block_rows=BLOCK_SIZE, maximum_emitted_block_rows=max_rows,
            sha256=sha256_file(target), input_sha256=input_hashes,
            synthetic_positive_count=positives, elapsed_seconds=time.monotonic() - started,
            numeric_versions={"numpy": np.__version__, "pandas": pd.__version__},
            model_configuration=REQUESTED_MODEL,
            source_split="External label-independent 300000 TRAIN / 60000 CAL when n=360000; no split column is emitted.",
            evidence_scope="Generated synthetic data only; no empirical frequency, target feedback, predictive performance, or external validation.")
        with open(audit_path, "x", encoding="utf-8") as handle:
            json.dump(audit, handle, indent=2, sort_keys=True)
            handle.write("\n")
        return audit
    finally:
        if not published and partial.exists():
            partial.unlink()
        lock.unlink()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schema", default="allowed_schema.csv")
    parser.add_argument("--disease-identities", default="disease_identities.csv")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--n", type=int, default=DEFAULT_N)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--diseases", nargs="+", choices=list(ENDPOINT_SPEC))
    args = parser.parse_args(argv)
    if args.n < 1:
        parser.error("--n must be positive")
    schema, identities = load_inputs(args.schema, args.disease_identities)
    diseases = args.diseases or [row["endpoint"] for row in identities]
    if len(diseases) != len(set(diseases)):
        parser.error("Duplicate disease codes are not allowed")
    _require_numeric()
    # Preflight all completed/partial destinations before any generation.
    for disease in diseases:
        folder = Path(args.output_dir) / disease
        filename = f"synthetic_{disease}_{args.n}"
        for suffix in (".csv", ".csv.partial", ".csv.lock", ".audit.json"):
            candidate = folder / (filename + suffix)
            if candidate.exists():
                raise FileExistsError(f"Refusing to overwrite output: {candidate}")
    hashes = {"schema": sha256_file(args.schema), "disease_identities": sha256_file(args.disease_identities)}
    for disease in diseases:
        audit = write_population(disease, args.n, args.seed, schema, args.output_dir, hashes)
        print(json.dumps({key: audit[key] for key in ("endpoint", "rows", "sha256", "elapsed_seconds")}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
