"""Authored construction constants; no cohort estimates or retrieved references."""
import csv
import json
from pathlib import Path

MODEL_CONFIGURATION = {
    "requested_model": "gpt-6-astra",
    "requested_reasoning_effort": "ultra",
    "requested_fork_turns": "none",
    "configuration_basis": "parent task explicitly records requested tool-visible invocation",
    "backend_build_identifier": None,
    "os_level_isolation_claimed": False,
}
DEFAULT_SEED = 20260831
DEFAULT_N = 360000
CHUNK_SIZE = 12000
HORIZON_YEARS = 15
ANNUAL_HAZARD_PRIOR = 0.004

# Parents are the actual inputs to each constructed physiological driver. A
# driver also has its own independent random innovation unless documented below.
DRIVER_PARENTS = {
    "age": (), "male": (), "height": ("male",),
    "activity": ("age",), "diet": (), "smoking": (),
    "tobacco": ("smoking", "age"), "current": ("smoking",),
    "alcohol": (), "distress": ("age", "current", "activity"),
    "adiposity": ("age", "male", "activity", "diet"),
    "lean": ("male", "activity", "age"),
    "pressure": ("age", "adiposity", "alcohol", "tobacco"),
    "thyroid": ("age", "male"),
    "lipids": ("adiposity", "age", "diet", "thyroid"),
    "hepatic": ("adiposity", "alcohol"),
    "glycaemia": ("adiposity", "age", "hepatic"),
    "renal": ("age", "pressure", "glycaemia"),
    "atopy": (), "obstruction": ("tobacco", "current", "age", "atopy"),
    "autoimmune": ("current", "male"),
    "inflammation": ("adiposity", "current", "autoimmune", "hepatic"),
    "gut": ("distress",), "iron": ("male", "age"),
    "sun": ("activity",),
    "bone": ("age", "male", "adiposity", "activity", "sun"),
    "urate": ("renal", "adiposity", "alcohol", "male"),
    "apnoea": ("adiposity", "male", "age"),
    "insomnia": ("distress", "apnoea"),
    "reflux": ("adiposity", "tobacco", "alcohol"),
    "mechanical": ("adiposity", "age"),
    "prostate": ("age", "adiposity"),
    "prostate_cancer": (), "ocular": ("age",),
    "gallstone": (), "cardiac": (), "lpa": (),
    "social": (), "pigment": (), "family": (),
}

# (family, Field_IDs, exact physiological drivers used by that family).
# Category codes lacking supplied level descriptions receive symmetric unordered
# priors and no disease direction. Their family has an empty driver set.
FEATURE_FAMILIES = [
    ("age", "21022", "age"), ("sex", "31", "male"),
    ("household_employment", "699 709 796 757 767 777 845", "age social"),
    ("blood_pressure", "4079 4080", "pressure age"),
    ("pulse", "102", "distress apnoea activity"),
    ("grip", "46 47", "male lean age mechanical"),
    ("body_size", "48 21002 21001 49", "adiposity height male lean"),
    ("height", "50 20015", "height"),
    ("body_composition", "23115 23111 23116 23112 23117 23113 23118 23114 23123 23119 23124 23120 23121 23125 23126 23122 23127 23128 23129 23130 23105 23099 23100 23101 23102 23106 23110 23109 23108 23107", "adiposity height male lean age"),
    ("spirometry", "3062 3063 20258 3064", "height male age adiposity obstruction"),
    ("birth_weight", "20022", "male"),
    ("unordered_activity", "2634 2624 1021 1011 3647 3637 981 971 943", ""),
    ("activity_minutes", "894 914 874 884 904 864 924", "activity adiposity age"),
    ("sedentary_time", "1090 1080 1070", "activity social age"),
    ("unordered_electronics", "1120 1130 2237", ""),
    ("sleep", "1160 1170 1190 1200 1210 1220", "insomnia apnoea"),
    ("unordered_chronotype", "1180", ""),
    ("smoking", "20116 1239 1249 20161 1259 1269 1279", "smoking tobacco current"),
    ("plant_food", "1289 1299 1309 1319 1438 1458", "diet"),
    ("beverages", "1488 1498 1528", ""),
    ("unordered_food", "1518 1329 1339 1349 1359 1369 1379 1389 1408 1478 6144", ""),
    ("alcohol", "20117 1558 1568 1578 1588 1598 1608 1618", "alcohol"),
    ("outdoors", "1050 1060", "sun"),
    ("pigmentation", "1717 1727 1747", "pigment"),
    ("sun_history", "1737 2277", "sun pigment"),
    ("facial_ageing", "1757", "age sun current"),
    ("unordered_sun_protection", "2267", ""),
    ("sexual_factors", "2139 2149 2159", ""),
    ("early_life", "1677 1777 1787", ""),
    ("parental_age", "2946 1807 1845 3526", "age family"),
    ("family_accident", "4501", ""),
    ("general_health", "2188", "age adiposity distress mechanical"),
    ("basophils", "30160 30220", "atopy inflammation current"),
    ("eosinophils", "30150 30210", "atopy inflammation current"),
    ("erythrocytes", "30030 30020 30050 30060 30040 30260 30270 30010 30070", "male iron renal autoimmune alcohol"),
    ("reticulocytes", "30300 30290 30280 30250 30240", "male iron renal autoimmune alcohol"),
    ("lymphocytes", "30120 30180", "atopy inflammation current"),
    ("monocytes", "30130 30190", "atopy inflammation current"),
    ("neutrophils", "30140 30200", "atopy inflammation current"),
    ("nucleated_erythrocytes", "30170 30230", ""),
    ("platelets", "30100 30080 30090 30110", "iron autoimmune"),
    ("leukocytes", "30000", "atopy inflammation current"),
    ("transaminases", "30620 30650 30730", "hepatic alcohol"),
    ("albumin", "30600", "hepatic renal inflammation"),
    ("alkaline_phosphatase", "30610", "hepatic bone"),
    ("lipid_panel", "30630 30640 30690 30760 30780 30870", "lipids adiposity alcohol"),
    ("crp", "30710", "inflammation"),
    ("calcium", "30680", "hepatic renal inflammation bone"),
    ("renal_markers", "30700 30720 30670", "renal lean male"),
    ("bilirubin", "30660 30840", "hepatic"),
    ("glycaemic_markers", "30740 30750", "glycaemia"),
    ("igf1", "30770", "age lean"),
    ("lipoprotein_a", "30790", "lpa"),
    ("sex_hormones", "30800 30830 30850", "age male adiposity thyroid"),
    ("phosphate", "30810", "renal bone"),
    ("rheumatoid_factor", "30820", "autoimmune"),
    ("protein", "30860", "hepatic renal inflammation autoimmune"),
    ("urate", "30880", "urate male"),
    ("vitamin_d", "30890", "sun adiposity"),
]
FEATURE_META = {}
for family, ids, drivers in FEATURE_FAMILIES:
    for fid in ids.split():
        if fid in FEATURE_META:
            raise ValueError("Duplicate authored field " + fid)
        FEATURE_META[fid] = {"family": family, "drivers": drivers.split()}


# Exact field-level driver refinements for heterogeneous measurement families.
# Structural employment/parental absence is included in the corresponding
# family dependency, because it is part of the observed measurement process.
FIELD_DRIVER_OVERRIDES = {
    "699": "age", "709": "social", "796": "social age", "757": "age",
    "767": "social age", "777": "age", "845": "social",
    "4080": "pressure", "21001": "adiposity", "21002": "adiposity height",
    "48": "adiposity height male", "49": "adiposity height",
    "3062": "height male age adiposity", "20258": "obstruction age",
    "894": "activity", "914": "activity", "874": "activity",
    "884": "activity", "904": "activity", "864": "activity",
    "1090": "social", "1080": "social age", "1070": "activity age",
    "1170": "insomnia", "1200": "insomnia", "1210": "apnoea",
    "20116": "smoking", "1239": "smoking", "1249": "smoking",
    "20161": "tobacco", "1259": "current", "1269": "current", "1279": "current",
    "1747": "", "2277": "sun",
    "30160": "atopy", "30150": "atopy", "30120": "atopy",
    "30130": "inflammation", "30140": "inflammation current",
    "30010": "male iron renal autoimmune", "30040": "iron alcohol",
    "30060": "iron", "30050": "iron alcohol", "30070": "iron renal",
    "30260": "iron alcohol", "30270": "iron alcohol",
    "30240": "iron renal", "30250": "male iron renal autoimmune",
    "30290": "iron renal", "30300": "male iron renal autoimmune", "30280": "iron",
    "30620": "hepatic", "30780": "lipids", "30760": "adiposity alcohol",
    "30630": "adiposity alcohol", "30720": "renal", "30670": "renal lean",
    "30800": "age male", "30850": "age male adiposity",
}
for fid, drivers in FIELD_DRIVER_OVERRIDES.items():
    FEATURE_META[fid]["drivers"] = drivers.split()

# Coefficients are subjective log-hazard contributions per authored driver unit.
# They are neither published effect estimates nor empirically fitted quantities.
RISK_WEIGHTS = {
    "I9_HYPTENSESS": dict(pressure=.72, age=.25, adiposity=.25, renal=.12, alcohol=.12),
    "M13_ARTHROSIS": dict(mechanical=.75, age=.55, adiposity=.38, male=-.12),
    "K11_DIVERTIC": dict(age=.70, diet=-.22, adiposity=.15, tobacco=.12),
    "E4_HYPERCHOL": dict(lipids=.85, thyroid=.18, age=.12),
    "N14_PROSTHYPERPLA": dict(prostate=.70, age=.65, adiposity=.12),
    "K11_REFLUX": dict(reflux=.70, adiposity=.32, tobacco=.20, alcohol=.12),
    "I9_AF": dict(age=.75, pressure=.33, adiposity=.22, alcohol=.30, apnoea=.22, renal=.10),
    "J10_ASTHMA_MAIN_EXMORE": dict(atopy=.85, obstruction=.32, adiposity=.20, tobacco=.12, male=-.15),
    "I9_ISCHHEART": dict(lipids=.55, pressure=.40, tobacco=.45, glycaemia=.30, age=.55, male=.40, inflammation=.18, lpa=.20),
    "H7_CATARACTSENILE": dict(age=.95, glycaemia=.28, tobacco=.22, sun=.18),
    "E4_OBESITY": dict(adiposity=.90, activity=-.18, thyroid=.15, distress=.12),
    "E4_DM2": dict(glycaemia=.75, adiposity=.55, age=.25, activity=-.18, hepatic=.12),
    "C3_PROSTATE_EXALLC": dict(age=.90, prostate_cancer=.80),
    "F5_DEPRESSIO": dict(distress=.85, age=-.18, male=-.22, insomnia=.25, activity=-.12, thyroid=.12),
    "COPD_MODE": dict(obstruction=.75, tobacco=.70, current=.20, age=.35),
    "E4_HYTHYNAS": dict(thyroid=.90, autoimmune=.25, age=.20, male=-.45),
    "N14_CHRONKIDNEYDIS": dict(renal=.80, glycaemia=.30, pressure=.30, age=.25),
    "F5_ANXIETY": dict(distress=.80, insomnia=.25, male=-.25, age=-.18),
    "D3_ANAEMIA_IRONDEF_NAS": dict(iron=.90, age=.12, male=-.25),
    "M13_OSTEOPOROSIS": dict(bone=.85, age=.50, male=-.50, tobacco=.20, adiposity=-.15),
    "K11_CHOLELITH": dict(gallstone=.60, adiposity=.40, glycaemia=.15, age=.30, male=-.35),
    "I9_HEARTFAIL": dict(age=.55, pressure=.40, glycaemia=.20, adiposity=.30, renal=.30, lipids=.20, current=.15, cardiac=.65),
    "K11_LIVER": dict(hepatic=.55, alcohol=.35, adiposity=.25, autoimmune=.12),
    "K11_IBS": dict(gut=.75, distress=.40, male=-.30, age=-.10),
    "M13_GOUT": dict(urate=.90, renal=.20, alcohol=.20, male=.40, adiposity=.20),
    "H7_GLAUCOMA": dict(ocular=.80, age=.70, pressure=.10, glycaemia=.12),
    "I9_STR_SAH": dict(pressure=.60, age=.60, tobacco=.35, glycaemia=.15, lipids=.15, alcohol=.10),
    "M13_RHEUMA": dict(autoimmune=.90, current=.30, male=-.35, age=.25),
    "G6_CARPTU": dict(mechanical=.50, adiposity=.40, thyroid=.25, glycaemia=.25, male=-.35),
    "G6_SLEEPAPNO": dict(apnoea=.90, adiposity=.50, male=.35, age=.20),
}
# Subtype adjustments are added to the above coefficients. Mixture probabilities
# are deliberately authored construction priors, not observed subtype shares.
SUBTYPES = {
    "K11_LIVER": [(.45, dict(adiposity=.25, alcohol=-.25)),
                  (.30, dict(alcohol=.40, adiposity=-.15)),
                  (.25, dict(autoimmune=.35, alcohol=-.25))],
    "I9_STR_SAH": [(.60, dict(lipids=.15, glycaemia=.10)),
                   (.25, dict(pressure=.25, alcohol=.20, lipids=-.15)),
                   (.15, dict(tobacco=.20, age=-.35, glycaemia=-.15, lipids=-.15))],
    "I9_HEARTFAIL": [(.50, dict(adiposity=.20, pressure=.15, lipids=-.15)),
                     (.50, dict(lipids=.20, current=.15, cardiac=.20))],
}
MALE_ONLY = {"N14_PROSTHYPERPLA", "C3_PROSTATE_EXALLC"}

# Direct means a measurement of an included risk process, not a causal regression
# coefficient on that assay. Indirect follows shared baseline-driver ancestry.
DIRECT_FAMILIES = {
    "pressure": {"blood_pressure"}, "adiposity": {"body_size", "body_composition"},
    "activity": {"activity_minutes"}, "diet": {"plant_food"},
    "tobacco": {"smoking"}, "current": {"smoking"}, "alcohol": {"alcohol"},
    "lipids": {"lipid_panel"}, "hepatic": {"transaminases", "bilirubin", "albumin"},
    "glycaemia": {"glycaemic_markers"}, "renal": {"renal_markers"},
    "atopy": {"eosinophils", "basophils"}, "obstruction": {"spirometry"},
    "autoimmune": {"rheumatoid_factor"}, "inflammation": {"crp", "leukocytes"},
    "iron": {"erythrocytes", "reticulocytes", "platelets"},
    "sun": {"outdoors"}, "bone": set(), "urate": {"urate"},
    "apnoea": {"sleep"}, "insomnia": {"sleep"}, "lpa": {"lipoprotein_a"},
    "age": {"age"}, "male": {"sex"},
}


def ancestors(driver):
    result = {driver}
    for parent in DRIVER_PARENTS[driver]:
        result.update(ancestors(parent))
    return result


def coverage_rows(schema_rows, identities):
    for disease in identities:
        code = disease["phenotype"]
        risk = set(RISK_WEIGHTS[code])
        for _, adjustment in SUBTYPES.get(code, []):
            risk.update(adjustment)
        risk_anc = set().union(*(ancestors(k) for k in risk))
        for field in schema_rows:
            fid = field["Field_ID"]
            meta = FEATURE_META[fid]
            feature_anc = set().union(*(ancestors(k) for k in meta["drivers"]))
            shared = sorted(feature_anc & risk_anc)
            direct = sorted(k for k in risk if k in meta["drivers"] and meta["family"] in DIRECT_FAMILIES.get(k, set()))
            if fid == "31" and code in MALE_ONLY:
                relation = "neutral"
                rationale = "Eligibility only: male code 1 for every row; no within-task variation."
            elif direct:
                relation = "direct"
                rationale = "Baseline measurement of included risk process(es): " + "; ".join(direct) + ". Measurement error remains independent of the future outcome draw."
            elif shared:
                relation = "indirect"
                rationale = "No dedicated field coefficient; association arises only through shared baseline ancestors: " + "; ".join(shared) + "."
            else:
                relation = "neutral"
                rationale = "No implemented shared physiological driver or dedicated endpoint term; preserved with an independent construction prior."
            yield {"phenotype": code, "Field_ID": fid, "Description": field["Description"],
                   "Unit": field["Unit"], "relationship": relation,
                   "implementation_family": meta["family"],
                   "feature_drivers": json.dumps(meta["drivers"]),
                   "outcome_drivers": json.dumps(sorted(risk)),
                   "shared_baseline_ancestors": json.dumps(shared), "rationale": rationale}


def write_coverage(directory):
    directory = Path(directory)
    with (directory / "allowed_schema.csv").open(newline="", encoding="utf-8-sig") as f:
        schema = list(csv.DictReader(f))
    with (directory / "disease_identities.csv").open(newline="", encoding="utf-8-sig") as f:
        identities = list(csv.DictReader(f))
    rows = list(coverage_rows(schema, identities))
    with (directory / "feature_disease_rules.csv").open("x", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


if __name__ == "__main__":
    print(write_coverage(Path(__file__).resolve().parent))
