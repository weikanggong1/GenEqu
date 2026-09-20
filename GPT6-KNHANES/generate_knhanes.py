#!/usr/bin/env python3
"""Independently authored knowledge-to-population KNHANES construction.

All numerical priors are construction choices, not fitted population estimates.
See GENERATOR_RULES.md and feature_disease_rules.csv for the measurement model.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
BLOCK_SIZE = 10000
MODEL_REQUEST = {"model": "gpt-6-astra", "reasoning": "ultra",
                 "context_fork": "none", "backend_identity_verified": False}
FIELDS = tuple("21022 31 709 4079 4080 102 48 21002 21001 50 23116 23112 "
               "23124 23120 23128 23099 23100 23101 23102 874 864 20116 "
               "1239 20117 30030 30020 30050 30060 30040 30080 30010 30000 "
               "30620 30650 30710 30690 30700 30740 30750 30760 30780 "
               "30870 30880 30670".split())
UNITS = dict(zip(FIELDS, ["years", "binary code", "persons", "mmHg", "mmHg", "bpm",
    "cm", "kg", "kg/m^2", "cm", "kg", "kg", "kg", "kg", "kg", "%", "kg", "kg",
    "kg", "minutes/day", "days/week", "category code", "category code", "category code",
    "%", "g/dL", "pg", "g/dL", "fL", "10^9 cells/L", "10^12 cells/L", "10^9 cells/L",
    "U/L", "U/L", "mg/L", "mmol/L", "umol/L", "mmol/L", "mmol/mol", "mmol/L",
    "mmol/L", "mmol/L", "umol/L", "mmol/L"]))
CATEGORY_CODES = {"31": "0=female;1=male", "20116": "0=never;1=previous;2=current",
                  "1239": "0=no;1=daily;2=occasional",
                  "20117": "0=never;1=previous;2=current"}

# Risk coefficients multiply the named prediagnosis latent traits below. Effects
# multiply active disease burden; management effects multiply diagnosed treatment.
# Diagnosis is not a laboratory cutoff. These are authored generative parameters.
DISEASES = {
    "I9_HYPTENSESS": dict(field="DI1_dg", years=(2022, 2023, 2024), intercept=-1.45,
        risk={"age": .65, "bmi": .65, "vascular": .90, "renal": .20},
        ascertain=-.10, active=.95, treat=.76,
        effect={"sbp": 23., "dbp": 11., "gfr": -3.},
        management={"sbp": -18., "dbp": -8., "pulse": -2.5, "urate": 8.}),
    "E4_HYPERCHOL": dict(field="DI2_dg", years=(2022, 2023, 2024), intercept=-1.40,
        risk={"age": .32, "bmi": .48, "lipid": .95, "insulin": .35},
        ascertain=-.25, active=.93, treat=.64,
        effect={"ldl": .80, "log_tg": .43, "hdl": -.10},
        management={"ldl": -.95, "log_tg": -.22, "hdl": .025}),
    "E4_DM2": dict(field="DE1_dg", years=(2022, 2023, 2024), intercept=-2.45,
        risk={"age": .55, "bmi": .70, "insulin": 1.05, "diabetes_other": 1.50},
        ascertain=.00, active=.94, treat=.76,
        effect={"glucose": 3.8, "log_tg": .19, "hdl": -.07, "gfr": -4.},
        management={"glucose": -2.4, "bmi": -.30, "ldl": -.20}),
    "N14_CHRONKIDNEYDIS": dict(field="DN1_dg", years=(2022, 2023, 2024), intercept=-3.60,
        risk={"age": .45, "renal": 1.10, "vascular": .30, "insulin": .25},
        ascertain=-.55, active=.69, treat=.64,
        effect={"gfr": -33., "sbp": 5., "inflammation": .25, "rbc": -.16},
        management={"sbp": -4., "gfr": 3.}),
    "J10_ASTHMA_MAIN_EXMORE": dict(field="DJ4_dg", years=(2022, 2023, 2024), intercept=-2.85,
        risk={"atopy": 1.15, "bmi": .22, "female": .18, "smoke": .12},
        ascertain=.05, active=.63, treat=.63,
        effect={"inflammation": .38, "pulse": 2., "mobility": -.30},
        management={"inflammation": -.20, "pulse": 1., "bmi": .10}),
    "I9_STR_SAH": dict(field="DI3_dg", years=(2024,), intercept=-4.05,
        risk={"age": .95, "vascular": .70, "smoke": .40, "insulin": .30},
        ascertain=1.55, active=.74, treat=.81,
        effect={"mobility": -1., "inflammation": .15},
        management={"sbp": -5., "ldl": -.38}),
    "I9_ISCHHEART": dict(field="DI4_dg", years=(2024,), intercept=-3.55,
        risk={"age": .80, "vascular": .60, "lipid": .50, "smoke": .50, "male": .35},
        ascertain=1.00, active=.81, treat=.85,
        effect={"mobility": -.65, "inflammation": .20, "pulse": 1.},
        management={"sbp": -4., "ldl": -.50, "pulse": -6.}),
    "M13_OSTEOPOROSIS": dict(field="DM4_dg", years=(2024,), intercept=-2.65,
        risk={"age": .75, "female": .85, "menopause": .80, "bmi": -.55, "bone": .85},
        ascertain=-.65, active=.96, treat=.56,
        effect={"mobility": -.22, "height": -.30}, management={}),
    "G6_SLEEPAPNO": dict(field="BP17_dg", years=(2022, 2023), intercept=-2.15,
        risk={"age": .22, "bmi": .90, "male": .65, "airway": 1.00},
        ascertain=.20, active=.90, treat=.52,
        effect={"sbp": 5., "dbp": 3., "inflammation": .25, "mobility": -.22},
        management={"sbp": -2., "dbp": -1., "mobility": .12}),
    "COPD_MODE": dict(field="HE_PFTdr", years=(2024,), intercept=-3.45,
        risk={"age": .72, "smoke": 1.00, "lung": .85, "male": .22},
        ascertain=-.60, active=.96, treat=.64,
        effect={"inflammation": .50, "pulse": 3., "mobility": -.60, "bmi": -1.05},
        management={"inflammation": -.13, "pulse": 1.2}),
}
PROCESS_FIELDS = {
    "sbp": ("4080",), "dbp": ("4079", "4080"), "pulse": ("102",),
    "bmi": ("21001", "21002", "48", "23099", "23100", "23101", "23102",
            "23116", "23112", "23124", "23120", "23128", "30620", "30650", "30710", "30700"),
    "height": ("50", "21002", "23100", "23101", "23102", "23116", "23112",
               "23124", "23120", "23128", "30700"),
    "glucose": ("30740", "30750"), "ldl": ("30780", "30690"),
    "log_tg": ("30870", "30690"), "hdl": ("30760", "30690"),
    "gfr": ("30700", "30670", "30880", "30010", "30020", "30030", "30750", "23102"),
    "urate": ("30880",), "rbc": ("30010", "30020", "30030"),
    "inflammation": ("30000", "30080", "30710", "30620", "30650"),
    "mobility": ("864", "874", "102"),
}


def sigmoid(x):
    return 1. / (1. + np.exp(-np.clip(x, -35., 35.)))


def stream(seed, phenotype, block_index, purpose):
    """No Python hash, task-order dependence, global RNG, or feedback."""
    identity = hashlib.sha256((phenotype + ":" + purpose).encode()).digest()
    words = [int.from_bytes(identity[i:i + 4], "little") for i in range(0, 16, 4)]
    return np.random.default_rng(np.random.SeedSequence([seed, block_index] + words))


def read_inputs(schema_path, identities_path):
    with Path(schema_path).open(newline="", encoding="utf-8-sig") as handle:
        schema = list(csv.DictReader(handle))
    fields = [row["field_id"] for row in schema]
    if len(fields) != 44 or len(set(fields)) != 44 or set(fields) != set(FIELDS):
        raise ValueError("Schema must contain exactly the 44 contracted measurements")
    for row in schema:
        f = row["field_id"]
        if row["unit"] != UNITS[f] or row["category_codes"] != CATEGORY_CODES.get(f, ""):
            raise ValueError("Unit/category semantics disagree for " + f)
        if row["type"] != ("categorical" if f in CATEGORY_CODES else "numeric"):
            raise ValueError("Measurement type disagrees for " + f)
    with Path(identities_path).open(newline="", encoding="utf-8-sig") as handle:
        identities = list(csv.DictReader(handle))
    if len(identities) != 10 or {r["phenotype"] for r in identities} != set(DISEASES):
        raise ValueError("Disease identities must contain exactly the 10 contracted tasks")
    for row in identities:
        spec = DISEASES[row["phenotype"]]
        if (row["target_field"] != spec["field"] or row["case_value"] != "1"
                or row["noncase_value"] != "0" or row["analysis_age_min"] != "40"
                or row["analysis_age_max"] != "70"
                or tuple(map(int, row["available_years"].split(";"))) != spec["years"]):
            raise ValueError("Target identity semantics disagree for " + row["phenotype"])
    return fields, identities


def apply_missingness(frame, rng, age_z, access, deprivation, short_fast):
    """Measurement-only interface: target/history/treatment never enter this function."""
    n = len(frame)
    missed_visit = rng.random(n) < sigmoid(-4.0 + .16 * age_z - .30 * access + .20 * deprivation)
    blood = missed_visit | (rng.random(n) < sigmoid(-3.7 - .25 * access + .13 * deprivation))
    anthro = missed_visit | (rng.random(n) < .009)
    vital = missed_visit | (rng.random(n) < .012)
    bia = anthro | (rng.random(n) < sigmoid(-2.7 + .12 * age_z - .12 * access))
    blood_fields = list(FIELDS[24:])
    masks = {"blood_panel": blood, "anthropometry": anthro, "vitals": vital, "BIA": bia}
    frame.loc[blood, blood_fields] = np.nan
    frame.loc[anthro, ["50", "21002", "21001", "48"]] = np.nan
    frame.loc[vital, ["4079", "4080", "102"]] = np.nan
    frame.loc[bia, list(FIELDS[10:19])] = np.nan
    for field in blood_fields:
        frame.loc[rng.random(n) < .004, field] = np.nan
    frame.loc[rng.random(n) < (.10 + .06 * short_fast), "30780"] = np.nan
    frame.loc[rng.random(n) < .025, "30750"] = np.nan
    frame.loc[rng.random(n) < .008, "709"] = np.nan
    smoking_missing = rng.random(n) < sigmoid(-4.0 - .15 * access)
    frame.loc[smoking_missing, ["20116", "1239"]] = np.nan
    # Conservative KNHANES mapping: a small authored share of reported former
    # smokers have <100 lifetime cigarettes; current tobacco remains known as no.
    low_lifetime_former = (frame["20116"].to_numpy() == 1) & (rng.random(n) < .035)
    frame.loc[low_lifetime_former, "20116"] = np.nan
    frame.loc[rng.random(n) < sigmoid(-4.1 - .15 * access), "20117"] = np.nan
    walk_missing = rng.random(n) < sigmoid(-3.7 - .15 * access + .08 * age_z)
    frame.loc[walk_missing, ["864", "874"]] = np.nan
    return {name: int(mask.sum()) for name, mask in masks.items()} | {
        "former_low_lifetime_smoking_mapping": int(low_lifetime_former.sum())}


def generate_population(n, phenotype, seed=20260831, block_index=0, fields=FIELDS,
                        with_audit=False):
    """Return at most one fixed 10,000-row block, containing predictors and target."""
    if phenotype not in DISEASES:
        raise ValueError("Unknown phenotype: " + str(phenotype))
    if not 1 <= n <= BLOCK_SIZE or seed < 0 or block_index < 0:
        raise ValueError("n must be 1..10000; seed and block_index must be nonnegative")
    if len(fields) != 44 or set(fields) != set(FIELDS):
        raise ValueError("Exactly the contracted fields are required")
    rng = stream(seed, phenotype, block_index, "physiology")
    normal = lambda: rng.standard_normal(n)
    age = rng.integers(40, 71, n)
    az = (age - 55.) / 10.
    male = rng.binomial(1, .49, n)
    deprivation, access_base, inherited = normal(), normal(), normal()
    access = .80 * access_base - .30 * deprivation + .10 * az
    household = np.minimum(1 + rng.poisson(np.exp(.52 - .16 * az), n), 6)
    menopause = (1 - male) * (rng.random(n) < sigmoid((age - 50.) / 2.4))
    ever = rng.random(n) < sigmoid(-1.90 + 2.25 * male + .14 * az + .25 * deprivation)
    current = ever & (rng.random(n) < sigmoid(.30 - .44 * az + .18 * deprivation))
    smoking = np.where(current, 2, np.where(ever, 1, 0))
    smoke_load = np.where(ever, np.clip(.65 + .38 * az + .42 * rng.gamma(1.8, .6, n), .2, 3.5), 0.)
    alcohol_ever = rng.random(n) < sigmoid(1.25 + .50 * male - .12 * az)
    alcohol = np.where(alcohol_ever, np.where(rng.random(n) < sigmoid(1.70 - .33 * az), 2, 1), 0)
    alcohol_load = (alcohol == 2) * rng.gamma(1.3, .65, n)
    fitness = normal() - .18 * deprivation - .17 * az
    stature = 158.8 + 12.2 * male - .105 * (age - 50.) + 5.1 * normal()
    bmi0 = np.clip(23.7 + .65 * male + .34 * az + .45 * deprivation - .44 * fitness + 2.6 * normal(), 16., 39.)
    bz = (bmi0 - 24.) / 3.
    insulin = .32 * bz + .18 * inherited + .92 * normal()
    lipid = .22 * inherited + .22 * alcohol_load + .90 * normal()
    vascular = .22 * inherited + .18 * deprivation + .16 * smoke_load + .90 * normal()
    renal = .25 * inherited + .15 * vascular + .90 * normal()
    traits = dict(age=az, bmi=bz, vascular=vascular, renal=renal, insulin=insulin,
        lipid=lipid, atopy=normal(), female=1-male, male=male, menopause=menopause,
        smoke=smoke_load, bone=normal(), airway=normal(), lung=normal(),
        diabetes_other=(rng.random(n) < .025).astype(float))
    lipid_subtype = rng.choice(3, n, p=[.42, .25, .33])
    survey_year = rng.choice(DISEASES[phenotype]["years"], n)
    processes = {key: np.zeros(n) for key in PROCESS_FIELDS}
    labels = {}
    cardiopulmonary_care = np.zeros(n)
    any_care = np.zeros(n)
    for code, spec in DISEASES.items():
        liability = np.full(n, spec["intercept"])
        for trait, coefficient in spec["risk"].items():
            liability += coefficient * traits[trait]
        history = rng.random(n) < sigmoid(liability)
        severity = .35 + 1.25 * rng.beta(2., 2.8, n)
        duration = np.minimum(rng.gamma(1.6, 4., n), age - 18.)
        active = history * (rng.random(n) < spec["active"])
        burden = active * severity * (.82 + .18 * (1 - np.exp(-duration / 6.)))
        # Kidney disease history includes resolved episodes and varied severity;
        # no chronicity or eGFR stage defines this target.
        if code == "N14_CHRONKIDNEYDIS":
            burden *= np.where(rng.random(n) < .085, 1.8, 1.)
        ascertain = sigmoid(spec["ascertain"] + .58 * access + .70 * (severity - .8)
                            + .10 * az + .10 * np.log1p(duration))
        opportunity = np.ones(n, dtype=bool)
        if code == "G6_SLEEPAPNO":
            opportunity = rng.random(n) < sigmoid(-.85 + .70 * access + .55 * (severity - .8))
        diagnosed = ((history & (rng.random(n) < ascertain))
                     | (~history & (rng.random(n) < .001))) & opportunity
        treated = diagnosed * (rng.random(n) < sigmoid(np.log(spec["treat"] / (1-spec["treat"]))
                                                       + .28 * access))
        adherence = .35 + .65 * rng.beta(3., 1.8, n)
        management = treated * adherence
        remembered = rng.random(n) < sigmoid(3.55 - .055 * duration + .25 * active)
        # A false recalled sleep diagnosis still requires a latent testing opportunity.
        report = (diagnosed & remembered) | (~diagnosed & opportunity & (rng.random(n) < .0006))
        for process, coefficient in spec["effect"].items():
            modifier = 1.
            if code == "E4_HYPERCHOL":
                # High-LDL, high-TG, and mixed histories all map to dyslipidaemia.
                modifier = (np.where(lipid_subtype == 1, .15, 1.) if process == "ldl"
                            else np.where(lipid_subtype == 0, .20, 1.))
            processes[process] += coefficient * burden * modifier
        for process, coefficient in spec["management"].items():
            processes[process] += coefficient * management
        labels[code] = report.astype(np.int8)
        any_care += diagnosed
        if code in ("I9_STR_SAH", "I9_ISCHHEART", "COPD_MODE", "J10_ASTHMA_MAIN_EXMORE"):
            cardiopulmonary_care += diagnosed

    # Diagnostic-history management changes current reported behavior, even if a
    # diagnosis is forgotten. Labels themselves never drive a measurement.
    quit_smoking = (smoking == 2) & (rng.random(n) < (1 - np.exp(-.20 * cardiopulmonary_care - .035 * any_care)))
    smoking[quit_smoking] = 1
    tobacco = np.where(smoking == 2, np.where(rng.random(n) < .86, 1, 2), 0)
    stop_alcohol = (alcohol == 2) & (rng.random(n) < 1 - np.exp(-.045 * any_care))
    alcohol[stop_alcohol] = 1
    alcohol_load *= alcohol == 2
    mobility = fitness + processes["mobility"]
    walking_days = rng.binomial(7, sigmoid(.40 + .50 * mobility - .14 * bz))
    walking_minutes = np.where(walking_days > 0, np.rint(np.clip(np.exp(3.35 + .20 * mobility + .57 * normal()), 10., 240.)), 0.)
    height = np.clip(stature + processes["height"] + .15 * normal(), 140., 192.)
    bmi = np.clip(bmi0 + processes["bmi"] + .30 * normal(), 15.5, 42.)
    weight = bmi * (height / 100.) ** 2
    bf_pct = np.clip(26.0 - 7.2 * male + 1.15 * (bmi - 23.) + .085 * (age - 50.) + 2.6 * normal(), 8., 53.)
    fat = weight * bf_pct / 100.
    ffm = weight - fat
    fractions = np.column_stack([.175-.012*male, .175-.012*male, .063+.002*male,
                                 .063+.002*male, .524+.020*male])
    fractions *= np.exp(rng.normal(0., .075, (n, 5)))
    fractions /= fractions.sum(axis=1, keepdims=True)
    regions = fractions * (fat * (.955 + .025 * rng.beta(2., 2., n)))[:, None]
    water = ffm * np.clip(.735 + .007 * normal() - .00006 * processes["gfr"], .70, .77)
    waist = np.clip(76. + 6.0 * male + 2.20 * (bmi - 23.) + .08 * (age - 50.) + 3.3 * normal(), 55., 147.)
    dbp = np.clip(74. + .13 * (age - 50.) + 2.1 * bz + 3.6 * vascular
                  + processes["dbp"] + 5.0 * normal(), 45., 126.)
    raw_sbp = 116. + .49 * (age - 50.) + 3.1 * bz + 5.0 * vascular + processes["sbp"] + 7.2 * normal()
    sbp = dbp + np.clip(raw_sbp - dbp, 22., 112.)
    pulse = np.clip(71. - 2.6 * mobility + 1.1 * (smoking == 2) + processes["pulse"] + 7.0 * normal(), 43., 128.)
    pulse = np.where(rng.random(n) < .85, 2. * np.rint(pulse / 2.), np.rint(pulse))
    inflammation = np.maximum(-1.2, .18 * bz + .17 * (smoking == 2) + .65 * normal() + processes["inflammation"])
    iron_deficit = (rng.random(n) < sigmoid(-3.0 + .65 * (1-male) * (1-menopause))) * rng.uniform(.3, 1., n)
    gfr = np.clip(99. - .45 * (age - 50.) - 6. * renal + processes["gfr"] + 6.0 * normal(), 12., 135.)
    rbc = np.clip(4.42 + .48 * male - .035 * az - .48 * iron_deficit
                  - .006 * np.maximum(60.-gfr, 0.) + processes["rbc"] + .29 * normal(), 2.4, 6.4)
    mcv = np.clip(89.5 + 1.5 * alcohol_load - 10. * iron_deficit + 3.2 * normal(), 62., 112.)
    mchc = np.clip(33.6 - .65 * iron_deficit + .63 * normal(), 29., 37.)
    hct = rbc * mcv / 10.
    hb = hct * mchc / 100.
    mch = mcv * mchc / 100.
    platelets = np.clip(np.exp(np.log(246.) - .025 * az + .075 * inflammation + .12 * iron_deficit + .22 * normal()), 65., 780.)
    wbc = np.clip(np.exp(np.log(5.9) + .09 * (smoking == 2) + .12 * inflammation + .22 * normal()), 2., 22.)
    liver = .30 * normal()
    alt = np.clip(np.exp(np.log(21.) + .14 * (bmi-24.)/3. + .13 * alcohol_load + .11 * inflammation + liver + .34 * normal()), .5, 380.)
    ast = np.clip(np.exp(np.log(23.) + .055 * (bmi-24.)/3. + .10 * alcohol_load + .07 * inflammation + .70 * liver + .23 * normal()), 5., 300.)
    crp = np.clip(np.exp(np.log(.85) + .25 * (bmi-24.)/3. + .12 * az + .50 * inflammation + .74 * normal()), .015, 90.)
    alt_censored, crp_censored = alt < 5., crp < .2
    alt[alt_censored] = 5. / np.sqrt(2.)
    crp[crp_censored] = .2 / np.sqrt(2.)
    short_fast = rng.random(n) < .04
    long_glucose = np.clip(5.20 + .24 * bz + .12 * az + .32 * insulin + processes["glucose"], 3.2, 19.)
    glucose = np.clip(long_glucose + .38 * normal() + short_fast * rng.uniform(.3, 1.5, n), 2.8, 24.)
    hba1c_ngsp = np.clip(5.35 + .47 * (long_glucose-5.2) + .11 * iron_deficit
                         - .002 * np.maximum(45.-gfr, 0.) + .20 * normal(), 3.5, 14.)
    hba1c = (hba1c_ngsp - 2.152) / .09148
    ldl = np.clip(3.05 + .22 * bz + .13 * az + .43 * lipid + processes["ldl"] + .27 * normal(), .35, 8.)
    hdl = np.clip(1.42 - .13 * male - .10 * bz + .035 * alcohol_load + processes["hdl"] + .18 * normal(), .40, 3.0)
    tg = np.clip(np.exp(np.log(1.25) + .21 * bz + .13 * insulin + .13 * alcohol_load
                        + .13 * lipid + processes["log_tg"] + .34 * normal() + .15 * short_fast), .25, 12.)
    remnant = np.maximum(.06, .43 * np.minimum(tg, 4.5) + .12 * np.maximum(tg-4.5, 0.) + .07 * normal())
    total_chol = ldl + hdl + remnant
    creatinine = np.clip(78. * (100./gfr)**.85 * (.90+.14*male)
                         * (ffm/(44.+13.*male))**.30 * np.exp(.10 * normal()), 28., 800.)
    urate = np.clip(290. + 48. * male + 19. * bz + .85 * (95.-gfr) + 9. * alcohol_load
                    + processes["urate"] + 39. * normal(), 105., 790.)
    urea = np.clip((4.7 + .039 * (95.-gfr) + .16 * male) * np.exp(.18 * normal()), 1.5, 26.)
    data = {
        "21022": age, "31": male, "709": household, "4079": dbp, "4080": sbp,
        "102": pulse, "48": waist, "21002": weight, "21001": bmi, "50": height,
        "23116": regions[:, 0], "23112": regions[:, 1], "23124": regions[:, 2],
        "23120": regions[:, 3], "23128": regions[:, 4], "23099": bf_pct, "23100": fat,
        "23101": ffm, "23102": water, "874": walking_minutes, "864": walking_days,
        "20116": smoking, "1239": tobacco, "20117": alcohol, "30030": hct, "30020": hb,
        "30050": mch, "30060": mchc, "30040": mcv, "30080": platelets, "30010": rbc,
        "30000": wbc, "30620": alt, "30650": ast, "30710": crp, "30690": total_chol,
        "30700": creatinine, "30740": glucose, "30750": hba1c, "30760": hdl,
        "30780": ldl, "30870": tg, "30880": urate, "30670": urea,
    }
    frame = pd.DataFrame(data, columns=list(fields)).astype(float)
    process_counts = apply_missingness(frame, stream(seed, phenotype, block_index, "measurement_missingness"),
                                      az, access, deprivation, short_fast)
    frame[phenotype] = labels[phenotype]
    audit = {"rows": n, "cases": int(labels[phenotype].sum()),
             "missing": {f: int(frame[f].isna().sum()) for f in fields},
             "measurement_process_counts": process_counts,
             "censoring_before_panel_missingness": {"ALT_lt5": int(alt_censored.sum()),
                                                      "CRP_lt0_2": int(crp_censored.sum())},
             "censoring_observed_replacements": {"ALT_lt5": int((frame["30620"] == 5./np.sqrt(2.)).sum()),
                                                  "CRP_lt0_2": int((frame["30710"] == .2/np.sqrt(2.)).sum())},
             "survey_year_counts": {str(y): int((survey_year == y).sum()) for y in DISEASES[phenotype]["years"]},
             "censoring_by_year": {}}
    for year in DISEASES[phenotype]["years"]:
        for name, field, censored in (("ALT_lt5", "30620", alt_censored), ("CRP_lt0_2", "30710", crp_censored)):
            year_mask = survey_year == year
            audit["censoring_by_year"][str(year) + ":" + name + ":before_missingness"] = int((year_mask & censored).sum())
            audit["censoring_by_year"][str(year) + ":" + name + ":observed"] = int((year_mask & censored & frame[field].notna().to_numpy()).sum())
    return (frame, audit) if with_audit else frame


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_task(output_dir, phenotype, n, seed, fields, provenance):
    destination = Path(output_dir) / phenotype
    destination.mkdir(parents=True, exist_ok=True)
    output = destination / ("synthetic_" + phenotype + "_" + str(n) + ".csv")
    audit_path = output.with_suffix(".audit.json")
    partial = output.with_suffix(".csv.part")
    if output.exists() or audit_path.exists() or partial.exists():
        raise FileExistsError("Refusing to overwrite existing task output: " + str(output))
    counts = {"rows": 0, "cases": 0, "missing": {f: 0 for f in fields},
              "measurement_process_counts": {}, "censoring_before_panel_missingness": {},
              "censoring_observed_replacements": {}, "survey_year_counts": {}, "censoring_by_year": {}}
    created_output = False
    created_audit = False
    # An exclusive partial reserves the task. Hard-link publication cannot overwrite.
    with partial.open("x", encoding="utf-8", newline="") as handle:
        try:
            for block_index, start in enumerate(range(0, n, BLOCK_SIZE)):
                frame, audit = generate_population(min(BLOCK_SIZE, n-start), phenotype, seed,
                    block_index, fields, with_audit=True)
                frame.to_csv(handle, index=False, header=(start == 0), float_format="%.8g", na_rep="")
                for key in ("rows", "cases"):
                    counts[key] += audit[key]
                for section in ("missing", "measurement_process_counts", "censoring_before_panel_missingness",
                                "censoring_observed_replacements", "survey_year_counts", "censoring_by_year"):
                    for key, value in audit[section].items():
                        counts[section][key] = counts[section].get(key, 0) + value
                del frame
            handle.flush()
            os.fsync(handle.fileno())
            result = dict(provenance, phenotype=phenotype, n=n, seed=seed, block_size=BLOCK_SIZE,
                          output_file=output.name, output_sha256=sha256_file(partial), counts=counts,
                          target_field=DISEASES[phenotype]["field"],
                          available_years=list(DISEASES[phenotype]["years"]))
            os.link(partial, output)
            created_output = True
            with audit_path.open("x", encoding="utf-8") as audit_handle:
                created_audit = True
                json.dump(result, audit_handle, indent=2, sort_keys=True)
                audit_handle.write("\n")
        except BaseException:
            if created_output:
                output.unlink(missing_ok=True)
            if created_audit:
                audit_path.unlink(missing_ok=True)
            raise
        finally:
            partial.unlink(missing_ok=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schema", type=Path, required=True)
    parser.add_argument("--disease-identities", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n", type=int, default=360000)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--diseases", nargs="+", choices=tuple(DISEASES), default=list(DISEASES))
    args = parser.parse_args()
    if args.n < 1 or args.seed < 0 or len(set(args.diseases)) != len(args.diseases):
        parser.error("n must be positive, seed nonnegative, and diseases unique")
    fields, identities = read_inputs(args.schema, args.disease_identities)
    inputs = {"allowed_schema.csv": args.schema, "disease_identities.csv": args.disease_identities,
              "CONSTRUCTION_CONTRACT.md": HERE / "CONSTRUCTION_CONTRACT.md",
              "schema_provenance.json": HERE / "schema_provenance.json"}
    authored = ("generate_knhanes.py", "GENERATOR_RULES.md", "feature_disease_rules.csv", "test_generator.py")
    provenance = {"construction": "independent_biomedical_priors_v1", "model_configuration_requested": MODEL_REQUEST,
                  "input_sha256": {name: sha256_file(path) for name, path in inputs.items()},
                  "code_rules_tests_sha256": {name: sha256_file(HERE / name) for name in authored},
                  "libraries": {"numpy": np.__version__, "pandas": pd.__version__},
                  "numerical_priors": "Authored construction choices; no Korean empirical estimates or performance feedback",
                  "target_identities": identities}
    # Preflight all requested tasks before writing any of them.
    for code in args.diseases:
        path = args.output_dir / code / ("synthetic_" + code + "_" + str(args.n) + ".csv")
        if any(p.exists() for p in (path, path.with_suffix(".audit.json"), path.with_suffix(".csv.part"))):
            raise FileExistsError("Refusing to overwrite existing task output: " + str(path))
    for code in args.diseases:
        print(write_task(args.output_dir, code, args.n, args.seed, fields, provenance), flush=True)


if __name__ == "__main__":
    main()
