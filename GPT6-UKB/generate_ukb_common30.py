#!/usr/bin/env python3
"""Independent, knowledge-authored baseline measurements and 15-year outcomes."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd

from rulebook import (ANNUAL_HAZARD_PRIOR, CHUNK_SIZE, DEFAULT_N, DEFAULT_SEED,
                      FEATURE_META, HORIZON_YEARS, MALE_ONLY,
                      MODEL_CONFIGURATION, RISK_WEIGHTS, SUBTYPES)


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def stable_seed(seed, phenotype, chunk_index=0, stream="baseline"):
    text = f"ukb-common30-independent-v1|{int(seed)}|{phenotype}|{int(chunk_index)}|{stream}"
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:16], "big")


def load_schema(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    ids = [r["Field_ID"] for r in rows]
    if len(ids) != 201 or len(set(ids)) != 201 or set(ids) != set(FEATURE_META):
        raise ValueError("Schema must contain exactly the 201 authored measurement identities")
    for r in rows:
        r["lo"] = float(r["Recommended_Hard_Min"])
        r["hi"] = float(r["Recommended_Hard_Max"])
        r["decimals"] = int(r["Recommended_Decimal_Places"])
        r["codes"] = json.loads(r["Recommended_Allowed_Values"]) if r["Recommended_Allowed_Values"] else None
        if not (math.isfinite(r["lo"]) and math.isfinite(r["hi"]) and r["lo"] <= r["hi"]):
            raise ValueError("Invalid bound for " + r["Field_ID"])
        if r["codes"] and any(v < r["lo"] or v > r["hi"] for v in r["codes"]):
            raise ValueError("Category outside hard bounds")
    return rows


def load_identities(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    ids = [r["phenotype"] for r in rows]
    if len(ids) != 30 or len(set(ids)) != 30 or set(ids) != set(RISK_WEIGHTS):
        raise ValueError("Identity file must contain the exact 30 authored endpoints")
    return {r["phenotype"]: r for r in rows}


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -35, 35)))


def _baseline(schema, phenotype, n, rng):
    """No outcome or event status is an input to baseline physiology."""
    z = lambda: rng.normal(size=n)
    x = {}
    d = {}
    age = rng.integers(39, 71, size=n).astype(float)
    sex = np.ones(n) if phenotype in MALE_ONLY else rng.binomial(1, .5, n).astype(float)
    d["age"], d["male"] = (age - 54.5) / 9.0, sex - .5
    a, m = d["age"], d["male"]
    d["height"] = np.clip(163 + 13 * sex + 6.4 * z(), 142, 196)
    d["activity"] = z() - .20 * a
    d["diet"] = z()
    smoking = rng.choice(3, n, p=[.55, .25, .20])
    pack = (smoking > 0) * np.minimum(rng.gamma(1.8, 8.0 + .22 * (age - 39)), (age - 18) * 2.0)
    d["smoking"] = smoking.astype(float)
    d["tobacco"] = pack / 20.0
    d["current"] = (smoking == 2).astype(float)
    drinking = rng.choice(3, n, p=[.20, .15, .65])
    units = (drinking == 2) * rng.gamma(1.3, 9.0, n)
    d["alcohol"] = np.log1p(units / 7.0) - .5
    d["distress"] = .85*z() - .12*a + .15*d["current"] - .15*d["activity"]
    d["adiposity"] = .70*z() + .20*a + .08*m - .30*d["activity"] - .18*d["diet"]
    d["lean"] = .65*z() + .45*m + .20*d["activity"] - .15*a
    d["pressure"] = .80*z() + .38*a + .28*d["adiposity"] + .16*d["alcohol"] + .12*d["tobacco"]
    d["thyroid"] = .85*z() + .15*a + .20*(1-sex)
    d["lipids"] = .80*z() + .22*d["adiposity"] + .15*a - .13*d["diet"] + .12*d["thyroid"] + 1.8*rng.binomial(1, .02, n)
    d["hepatic"] = .75*z() + .45*d["adiposity"] + .45*d["alcohol"]
    d["glycaemia"] = .75*z() + .45*d["adiposity"] + .20*a + .15*d["hepatic"]
    d["renal"] = .80*z() + .35*a + .20*d["pressure"] + .15*d["glycaemia"] + 1.5*rng.binomial(1, .06, n)
    d["atopy"] = z()
    d["obstruction"] = .75*z() + .50*d["tobacco"] + .25*d["current"] + .18*a + .20*d["atopy"]
    d["autoimmune"] = .80*z() + .18*d["current"] + .10*(1-sex)
    d["inflammation"] = .55*z() + .45*d["adiposity"] + .30*d["current"] + .32*d["autoimmune"] + .25*d["hepatic"]
    d["gut"] = .80*z() + .30*d["distress"]
    d["iron"] = .80*z() + .40*(1-sex)*(age < 52)
    d["sun"] = .70*z() + .50*d["activity"]
    d["bone"] = .75*z() + .35*a + .50*(1-sex)*(age > 50) - .25*d["adiposity"] - .18*d["activity"] - .15*d["sun"]
    d["urate"] = .65*z() + .30*d["renal"] + .30*d["adiposity"] + .25*d["alcohol"] + .22*m
    d["apnoea"] = .75*z() + .60*d["adiposity"] + .35*m + .12*a
    d["insomnia"] = .75*z() + .55*d["distress"] + .20*d["apnoea"]
    d["reflux"] = .70*z() + .35*d["adiposity"] + .15*d["tobacco"] + .15*d["alcohol"]
    d["mechanical"] = .70*z() + .30*d["adiposity"] + .20*a
    d["prostate"] = .80*z() + .40*a + .10*d["adiposity"]
    d["prostate_cancer"], d["gallstone"], d["cardiac"] = z(), z(), z()
    d["ocular"] = .85*z() + .25*a
    d["lpa"], d["social"], d["pigment"], d["family"] = z(), z(), z(), z()
    put = lambda fid, values: x.__setitem__(str(fid), np.asarray(values, dtype=float))
    put(21022, age); put(31, sex)
    soc = d["social"]
    put(699, np.minimum(age - 18, rng.gamma(1.6, 8, n)))
    put(709, 1 + rng.poisson(np.exp(.35 + .12*soc), n))
    put(796, rng.gamma(1.4, 6, n)*np.exp(.15*soc))
    put(757, np.minimum(age - 18, rng.gamma(1.3, 7, n)))
    employed = rng.random(n) < sigmoid(2.0 - 1.1*a)
    put(767, employed*np.clip(35 + 8*z() + 2*soc, 4, 70))
    put(777, employed*rng.binomial(7, .68, n))
    put(845, np.clip(18 + 2.2*z() + soc, 12, 30))
    put(4080, 123 + 15*d["pressure"] + 5*z())
    put(4079, 77 + 8*d["pressure"] - 3*a + 3*z())
    put(102, 69 + 5*d["distress"] + 3*d["apnoea"] - 3*d["activity"] + 5*z())
    grip = 28 + 13*sex + 5*d["lean"] - 2*a - 1.3*d["mechanical"]
    put(46, grip + 2.3*z()); put(47, grip + 1 + 2.3*z())
    height = d["height"]
    bmi = np.clip(26 + 4.7*d["adiposity"], 16, 51)
    weight = bmi*(height/100)**2
    fatpct = np.clip(29 - 8*sex + 4.1*d["adiposity"] - .8*d["lean"] + 1.1*z(), 8, 54)
    fat = weight*fatpct/100
    lean = weight-fat
    put(21001, bmi); put(21002, weight); put(50, height)
    put(20015, .52*height + 1.3*z())
    put(48, 87 + 7*sex + 9*d["adiposity"] + .08*(height-169) + 2*z())
    put(49, 99 + 6*d["adiposity"] + .06*(height-169) + 2*z())
    put(23099, fatpct); put(23100, fat); put(23101, lean)
    water = lean*np.clip(.73 + .015*z(), .68, .78)
    put(23102, water)
    put(23105, (370 + 21.6*lean)*4.184)
    # Five mutually exhaustive segments share total fat/lean mass before small
    # device error. Predicted muscle mass is a proportion of segment lean mass.
    fat_share = np.array([.185, .185, .052, .052, .526])
    lean_share = np.array([.175, .175, .065, .065, .520])
    fs = rng.lognormal(0, .05, (n, 5))*fat_share
    ls = rng.lognormal(0, .035, (n, 5))*lean_share
    fs /= fs.sum(axis=1, keepdims=True); ls /= ls.sum(axis=1, keepdims=True)
    segments = [(23115,23116,23117,23118), (23111,23112,23113,23114),
                (23123,23124,23125,23126), (23119,23120,23121,23122),
                (23127,23128,23129,23130)]
    for j, (pct_id, fat_id, lean_id, muscle_id) in enumerate(segments):
        sf, sl = fat*fs[:,j], lean*ls[:,j]
        put(pct_id, 100*sf/(sf+sl)); put(fat_id, sf); put(lean_id, sl)
        put(muscle_id, .955*sl)
    impedance = 540*(height/169)**2*(38/np.maximum(water, 15))
    put(23106, impedance + 20*z())
    for fid, fraction in [(23110,.72),(23109,.72),(23108,.40),(23107,.40)]:
        put(fid, fraction*impedance + 13*z())
    fvc = np.maximum(.9, (3.6 + .65*m + .035*(height-169) - .25*a - .15*d["adiposity"])*np.exp(.075*z()))
    ratio = sigmoid(1.325 - .27*d["obstruction"] - .075*a + .06*z())
    put(3062, fvc); put(3063, fvc*ratio)
    put(20258, (ratio - (.79-.012*a))/.055)
    put(3064, (100*fvc+30)*np.exp(-.10*d["obstruction"]+.12*z()))
    put(20022, 3.35 + .10*sex + .48*z())
    # Levels without supplied semantic labels are deliberately unordered.
    for r in schema:
        if FEATURE_META[r["Field_ID"]]["family"].startswith("unordered_"):
            put(r["Field_ID"], rng.choice(r["codes"], n))
    act = d["activity"]
    for minutes, days, mean, offset in [(894,884,35,0),(914,904,25,-.65),(874,864,45,.85)]:
        frequency = rng.binomial(7, sigmoid(.65*act+offset), n)
        duration = (frequency > 0)*rng.gamma(2, mean/2, n)*np.exp(.15*act)
        put(minutes, duration); put(days, frequency)
    put(924, 1 + np.digitize(act - .20*d["adiposity"] - .12*a + .7*z(), [-.85,.85]))
    put(1090, rng.gamma(1.3,.7,n)*np.exp(.1*soc))
    put(1080, rng.gamma(1.5,1.2,n)*np.exp(.12*soc-.1*a))
    put(1070, rng.gamma(2,1.3,n)*np.exp(-.15*act+.12*a))
    ins, apn = d["insomnia"], d["apnoea"]
    put(1160, 7.2 - .45*ins + .15*apn + .85*z())
    put(1170, 4 - np.digitize(ins + .6*z(), [-.9,0,.9]))
    put(1190, 1 + np.digitize(.6*apn + .5*ins + z(), [0,1.4]))
    put(1200, 1 + np.digitize(ins + .6*z(), [-.4,1.0]))
    put(1210, np.where(rng.random(n) < sigmoid(-.3 + .9*apn), 1, 2))
    put(1220, np.digitize(.8*apn + .25*ins + .8*z(), [-.5,.5,1.5]))
    put(20116, smoking)
    put(1239, np.where(smoking == 2, rng.choice([1,2],n,p=[.8,.2]), 0))
    put(1249, np.where(smoking == 0, rng.choice([3,4],n,p=[.2,.8]), rng.choice([1,2],n,p=[.8,.2])))
    put(20161, pack)
    household_smoking = rng.binomial(1, .12+.30*d["current"], n)
    put(1259, household_smoking*(1+rng.binomial(1,.12,n)))
    put(1269, household_smoking*rng.gamma(1.5,3,n))
    put(1279, rng.binomial(1,.20+.20*d["current"],n)*rng.gamma(1.3,2,n))
    for fid, mean, slope in [(1289,3,.20),(1299,2,.25),(1309,2,.25),(1319,1,.12),(1438,14,.06),(1458,4,.15)]:
        put(fid, rng.gamma(2, mean/2, n)*np.exp(slope*d["diet"]))
    for fid, mean in [(1488,2.7),(1498,1.8),(1528,4.0)]:
        put(fid, rng.gamma(2,mean/2,n))
    put(20117, drinking)
    put(1558, np.where(drinking != 2, 6, 5-np.digitize(units,[2,6,14,28])))
    shares = rng.dirichlet([1.4,1.4,1.5,.7,.25],n)
    # Units are only an internal common intake scale; outputs retain their
    # requested glasses, pints and measures with explicit conversion factors.
    for j, fid in enumerate([1568,1578,1588,1598,1608]):
        put(fid, units*shares[:,j]/[1.5,1.5,2.0,1.0,1.0][j])
    put(1618, (drinking == 2)*rng.binomial(1,.55,n))
    put(1050, rng.gamma(2,1.4,n)*np.exp(.18*d["sun"]))
    put(1060, rng.gamma(2,.8,n)*np.exp(.18*d["sun"]))
    pigment = d["pigment"]
    put(1717, 1+np.digitize(pigment,[-1.4,-.6,.1,.8,1.5]))
    put(1727, 1+np.digitize(pigment+.6*z(),[-.7,.2,1]))
    put(1747, rng.choice([1,2,3,4,5,6],n))
    put(1737, rng.poisson(np.exp(.7+.2*d["sun"]-.25*pigment),n))
    put(2277, rng.binomial(1,.08,n)*rng.gamma(1.2,4,n)*np.exp(.1*d["sun"]))
    put(1757, 1+np.digitize(.3*a+.2*d["sun"]+.15*d["current"]+z(),[-.7,.7]))
    put(2139, np.clip(19+3*z(),12,35))
    put(2149, 1+rng.lognormal(1.1,1.0,n))
    put(2159, rng.binomial(1,.05,n))
    for fid, prob in [(1677,.55),(1777,.025),(1787,.22),(4501,.10)]:
        put(fid, rng.binomial(1,prob,n))
    family = d["family"]
    father_age = age+np.clip(29+5*z(),18,48)
    mother_age = age+np.clip(26+4*z(),18,44)
    father_death = np.clip(76+7*family+9*z(),25,106)
    mother_death = np.clip(81+7*family+8*z(),28,104)
    father_alive, mother_alive = father_death > father_age, mother_death > mother_age
    put(2946, father_age); put(1807, np.minimum(father_age, father_death))
    put(1845, mother_age); put(3526, np.minimum(mother_age, mother_death))
    put(2188, rng.binomial(1,sigmoid(-1+.25*a+.2*d["adiposity"]+.3*d["distress"]+.2*d["mechanical"])))
    inf, at = d["inflammation"], d["atopy"]
    counts = {
        "30160": .04*np.exp(.3*at+.30*z()),
        "30150": .15*np.exp(.65*at+.30*z()),
        "30120": 1.8*np.exp(.05*at+.20*z()),
        "30130": .45*np.exp(.15*inf+.22*z()),
        "30140": 3.5*np.exp(.22*inf+.16*d["current"]+.24*z()),
    }
    wbc = sum(counts.values())
    for fid, pct in [("30160",30220),("30150",30210),("30120",30180),("30130",30190),("30140",30200)]:
        put(fid, counts[fid]); put(pct, 100*counts[fid]/wbc)
    put(30000,wbc)
    iron, renal, immune, alc = d["iron"], d["renal"], d["autoimmune"], d["alcohol"]
    rbc = np.maximum(2.2,4.55+.35*sex-.22*iron-.10*renal-.06*immune+.20*z())
    mcv = 89-4*iron+1.5*alc+1.8*z()
    mchc = 33.5-.55*iron+.6*z()
    mch = mcv*mchc/100
    hct = rbc*mcv/10
    put(30010,rbc); put(30040,mcv); put(30060,mchc); put(30050,mch)
    put(30030,hct); put(30020,rbc*mch/10)
    put(30070,13+.65*np.maximum(iron,0)+.35*np.maximum(renal,0)+.45*np.abs(z()))
    retic_pct = 1.1*np.exp(-.15*iron-.12*renal+.20*z())
    scatter = np.clip(.18+.025*z(),.08,.28)
    put(30240,retic_pct); put(30250,rbc*retic_pct/100)
    put(30290,retic_pct*scatter); put(30300,rbc*retic_pct*scatter/100)
    put(30280,.25*sigmoid(-.25*iron+.35*z())+.04)
    put(30260,mcv+12+2*z()); put(30270,mcv-5+1.2*z())
    # Field 30170 has 0 decimal places and a 0.8 maximum, so the only feasible
    # nonnegative representable integer is 0. Both rare nucleated-cell fields
    # are neutral zero-inflated background, not disease flags.
    put(30170,np.zeros(n)); put(30230,rng.binomial(1,.001,n))
    platelet = 245+25*iron+15*immune+42*z()
    mpv = 10.1+.65*z()-.001*(platelet-245)
    put(30080,platelet); put(30100,mpv); put(30090,platelet*mpv/10000)
    put(30110,16.6+.35*(mpv-10)+.35*z())
    liver = d["hepatic"]
    put(30620,23*np.exp(.32*liver+.20*z()))
    put(30650,24*np.exp(.24*liver+.12*alc+.17*z()))
    put(30730,28*np.exp(.38*liver+.26*alc+.25*z()))
    albumin = 44-.8*liver-.5*renal-.5*inf+1.8*z()
    put(30600,albumin)
    put(30610,75*np.exp(.12*liver+.10*d["bone"]+.18*z()))
    ldl = 3.2+.80*d["lipids"]+.12*z()
    hdl = 1.45-.15*d["adiposity"]+.08*alc+.16*z()
    triglyceride = 1.35*np.exp(.28*d["adiposity"]+.18*d["lipids"]+.12*alc+.24*z())
    put(30780,ldl); put(30760,hdl); put(30870,triglyceride)
    put(30690,ldl+hdl+.45*triglyceride+.12*z())
    put(30640,.88+.19*(ldl-3.2)+.035*(triglyceride-1.35)+.04*z())
    put(30630,1.45+.30*(hdl-1.45)+.06*z())
    put(30710,1.2*np.exp(.65*inf+.45*z()))
    put(30680,2.40+.016*(albumin-44)-.025*renal+.012*d["bone"]+.06*z())
    put(30700,(69+15*sex+5*d["lean"])*np.exp(.20*renal+.08*z()))
    put(30720,.86*np.exp(.20*renal+.07*z()))
    put(30670,5.1*np.exp(.18*renal+.08*d["lean"]+.12*z()))
    bilirubin = 10*np.exp(.14*liver+.30*z())
    put(30840,bilirubin); put(30660,bilirubin*np.clip(.24+.035*z(),.10,.40))
    put(30740,5.15*np.exp(.095*d["glycaemia"]+.055*z()))
    put(30750,35+4.5*d["glycaemia"]+1.3*z())
    put(30770,22*np.exp(-.16*a+.06*d["lean"]+.20*z()))
    put(30790,28*np.exp(.90*d["lpa"]))
    menopausal = (sex == 0) & (age >= 51)
    estradiol = np.where(sex == 1, 100*np.exp(.22*z()),
                        np.where(menopausal,65*np.exp(.4*z()),400*np.exp(.7*z())))
    put(30800,estradiol)
    put(30850,np.where(sex == 1,16*np.exp(-.10*a-.12*d["adiposity"]+.23*z()),1.15*np.exp(-.06*a+.12*d["adiposity"]+.30*z())))
    put(30830,(42+20*(1-sex))*np.exp(-.23*d["adiposity"]+.10*a-.10*d["thyroid"]+.22*z()))
    put(30810,1.12+.045*renal+.025*d["bone"]+.12*z())
    put(30820,10*np.exp(.60*immune+.35*z()))
    put(30860,albumin+27+.45*immune+.4*inf+1.8*z())
    put(30880,325+65*d["urate"]+30*m+12*z())
    put(30890,55*np.exp(.30*d["sun"]-.12*d["adiposity"]+.20*z()))
    if set(x) != set(FEATURE_META):
        raise AssertionError("Missing/unexpected implemented fields: " + str(set(x)^set(FEATURE_META)))
    context = {"father_alive": father_alive, "mother_alive": mother_alive,
               "employed": employed}
    return x, d, context


def _draw_outcome(phenotype, drivers, rng):
    n = len(drivers["age"])
    lp = np.zeros(n)
    for key, weight in RISK_WEIGHTS[phenotype].items():
        lp += weight*drivers[key]
    if phenotype in SUBTYPES:
        components = SUBTYPES[phenotype]
        subtype = rng.choice(len(components), n, p=[p for p,_ in components])
        for j, (_, adjustment) in enumerate(components):
            mask = subtype == j
            for key, weight in adjustment.items():
                lp[mask] += weight*drivers[key][mask]
    # Independent omitted susceptibility and heterogeneous progression, never
    # written as predictors. This is a stochastic time-to-first-event model.
    lp += .9*rng.normal(size=n)
    lp += .18*rng.normal(size=n)*np.maximum(drivers["adiposity"],0)
    lp += .15*rng.normal(size=n)*drivers["age"]
    cumulative_hazard = ANNUAL_HAZARD_PRIOR*HORIZON_YEARS*np.exp(np.clip(lp,-12,12))
    probability = -np.expm1(-cumulative_hazard)
    return (rng.random(n) < probability).astype(np.int8)


def _bounded_array(values, row):
    if row["codes"] is not None:
        if not np.isin(values, row["codes"]).all():
            raise AssertionError("Invalid generated category in " + row["Field_ID"])
        return values
    decimals = row["decimals"]
    if row["Recommended_Data_Type"] == "bounded_integer_or_count":
        decimals = 0
    scale = 10**decimals
    lo = math.ceil(row["lo"]*scale-1e-8)/scale
    hi = math.floor(row["hi"]*scale+1e-8)/scale
    return np.clip(np.round(values,decimals),lo,hi)


def _apply_missingness(frame, schema, context, rng):
    """Mask construction never accesses phenotype labels or physiological values."""
    n = len(frame)
    # Technical failures are participant-independent Bernoulli draws. Panel
    # sharing gives realistic within-panel absence without outcome conditioning.
    panel_fail = {k: rng.random(n)<p for k,p in
                  [("Blood count",.015),("Blood biochemistry",.025),("impedance",.025),("spirometry",.06)]}
    questionnaire_nonresponse = rng.random(n)<.01
    for r in schema:
        fid = r["Field_ID"]
        if fid in {"21022","31"}:
            continue
        category, detail = r["Category"], r["Detail_Category"]
        mask = np.zeros(n,dtype=bool)
        if category in panel_fail:
            mask |= panel_fail[category]
        if detail == "Body composition by impedance":
            mask |= panel_fail["impedance"]
        if detail == "Spirometry":
            mask |= panel_fail["spirometry"]
        if category in {"Lifestyle","Environment","Basic information"}:
            mask |= questionnaire_nonresponse
        mask |= rng.random(n) < (.018 if category in {"Lifestyle","Environment"} else .006)
        if fid in {"796","757","767","777"}:
            mask |= ~context["employed"]
        if fid == "2946": mask |= ~context["father_alive"]
        if fid == "1807": mask |= context["father_alive"]
        if fid == "1845": mask |= ~context["mother_alive"]
        if fid == "3526": mask |= context["mother_alive"]
        frame.loc[mask,fid] = np.nan
    return frame


def generate_population(schema, phenotype, n, seed=DEFAULT_SEED, chunk_index=0,
                        apply_missingness=True):
    """Generate one in-memory chunk. Seed is independent of disease order/subset.

    n changes the sequence of random draws; prefix invariance across different n
    is not claimed. The CLI uses a fixed 12000-row block size.
    """
    if phenotype not in RISK_WEIGHTS:
        raise ValueError("Unknown phenotype: " + phenotype)
    if not isinstance(n,int) or n < 1 or chunk_index < 0:
        raise ValueError("n must be positive and chunk_index nonnegative")
    rng = np.random.default_rng(stable_seed(seed,phenotype,chunk_index,"baseline"))
    raw, drivers, context = _baseline(schema,phenotype,n,rng)
    columns = {}
    for row in schema:
        fid = row["Field_ID"]
        columns[fid] = _bounded_array(raw.pop(fid),row)
    frame = pd.DataFrame(columns,columns=[r["Field_ID"] for r in schema])
    outcome_rng = np.random.default_rng(stable_seed(seed,phenotype,chunk_index,"outcome"))
    outcome = _draw_outcome(phenotype,drivers,outcome_rng)
    del drivers, raw, columns
    if apply_missingness:
        missing_rng = np.random.default_rng(stable_seed(seed,phenotype,chunk_index,"missingness"))
        _apply_missingness(frame,schema,context,missing_rng)
    frame[phenotype] = outcome
    validate_frame(frame,schema,phenotype)
    return frame


def validate_frame(frame,schema,phenotype):
    if list(frame.columns) != [r["Field_ID"] for r in schema]+[phenotype]:
        raise AssertionError("Incorrect column order or hidden predictor")
    if not frame[phenotype].isin([0,1]).all() or frame[phenotype].isna().any():
        raise AssertionError("Invalid binary outcome")
    for row in schema:
        values = frame[row["Field_ID"]].dropna().to_numpy()
        if not np.isfinite(values).all() or (values<row["lo"]-1e-9).any() or (values>row["hi"]+1e-9).any():
            raise AssertionError("Nonfinite/out-of-bound field " + row["Field_ID"])
        if row["codes"] is not None and not np.isin(values,row["codes"]).all():
            raise AssertionError("Invalid categorical code")
        if row["Recommended_Data_Type"] == "bounded_integer_or_count" and not np.equal(values,np.round(values)).all():
            raise AssertionError("Noninteger count")
    if phenotype in MALE_ONLY and not (frame["31"]==1).all():
        raise AssertionError("Prostate population must be male")


def write_dataset(schema, identity, output_dir, n, seed, input_hashes=None):
    phenotype = identity["phenotype"]
    out = Path(output_dir)/phenotype
    # Atomic reservation also protects incomplete runs. Recovery is explicit:
    # choose a new destination; this program never replaces an existing task.
    out.mkdir(parents=True,exist_ok=False)
    final = out/f"synthetic_{phenotype}_{n}.csv"
    partial = out/(final.name+".partial")
    t0 = time.monotonic()
    counts = {"0":0,"1":0}
    missing = np.zeros(len(schema),dtype=np.int64)
    chunks = 0
    for start in range(0,n,CHUNK_SIZE):
        count = min(CHUNK_SIZE,n-start)
        frame = generate_population(schema,phenotype,count,seed,chunks)
        for label in [0,1]:
            counts[str(label)] += int((frame[phenotype]==label).sum())
        missing += frame.iloc[:,:201].isna().sum().to_numpy(dtype=np.int64)
        frame.to_csv(partial,mode="x" if chunks==0 else "a",header=chunks==0,index=False,
                     na_rep="",float_format="%.10g",lineterminator="\n")
        del frame
        chunks += 1
    os.rename(partial,final)
    audit = {
        "phenotype":phenotype,"identity":identity,"n":n,"n_predictors":201,
        "target":phenotype,"seed":seed,"disease_seed":str(stable_seed(seed,phenotype)),
        "chunk_size":CHUNK_SIZE,"chunks":chunks,"maximum_chunk_rows":min(n,CHUNK_SIZE),
        "horizon_years":HORIZON_YEARS,"annual_hazard_prior":ANNUAL_HAZARD_PRIOR,
        "outcome_interpretation":"Knowledge-authored stochastic incident-diagnosis risk approximation; registry control exclusions and true baseline disease-free eligibility are not reconstructed.",
        "label_counts":counts,"missing_counts":dict(zip([r["Field_ID"] for r in schema],map(int,missing))),
        "input_sha256":input_hashes or {},
        "source_sha256":{p:sha256_file(Path(__file__).resolve().parent/p) for p in ["generate_ukb_common30.py","rulebook.py"]},
        "output_sha256":sha256_file(final),"output_bytes":final.stat().st_size,
        "elapsed_seconds":round(time.monotonic()-t0,3),
        "runtime":{"python":platform.python_version(),"numpy":np.__version__,"pandas":pd.__version__},
        "model_configuration":MODEL_CONFIGURATION,
        "provenance_caveat":"Hard bounds are inherited, including disclosed historical aggregate-derived bounds; no empirical distributions or evaluation feedback used.",
    }
    with (out/"generation_audit.json").open("x",encoding="utf-8") as f:
        json.dump(audit,f,ensure_ascii=False,indent=2)
    return final,audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schema",required=True,type=Path)
    parser.add_argument("--disease-identities",required=True,type=Path)
    parser.add_argument("--output-dir",required=True,type=Path)
    parser.add_argument("--n",type=int,default=DEFAULT_N)
    parser.add_argument("--seed",type=int,default=DEFAULT_SEED)
    parser.add_argument("--diseases",nargs="+")
    args = parser.parse_args()
    if args.n < 1:
        parser.error("--n must be positive")
    schema = load_schema(args.schema)
    identities = load_identities(args.disease_identities)
    codes = args.diseases or list(identities)
    if len(codes) != len(set(codes)) or any(code not in identities for code in codes):
        parser.error("--diseases must contain distinct exact declared phenotype codes")
    # Preflight every requested destination before creating any dataset.
    for code in codes:
        if (args.output_dir/code).exists():
            raise FileExistsError("Refusing to overwrite existing destination: " + str(args.output_dir/code))
    provenance = args.schema.resolve().parent/"schema_provenance.json"
    with provenance.open(encoding="utf-8") as f:
        provenance_data = json.load(f)
    schema_hash = sha256_file(args.schema)
    if provenance_data["allowed_schema_sha256"] != schema_hash:
        raise ValueError("Schema SHA256 does not match the supplied provenance")
    hashes = {"allowed_schema.csv":schema_hash,
              "disease_identities.csv":sha256_file(args.disease_identities),
              "schema_provenance.json":sha256_file(provenance)}
    for code in codes:
        path,audit = write_dataset(schema,identities[code],args.output_dir,args.n,args.seed,hashes)
        print(json.dumps({"path":str(path),"n":audit["n"],"sha256":audit["output_sha256"]}),flush=True)


if __name__ == "__main__":
    main()
