#!/usr/bin/env python3
"""Synthetic-only implementation verification; never fits or evaluates a classifier."""
import argparse
import ast
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import traceback

import numpy as np
import pandas as pd

import generate_knhanes as g


HERE = Path(__file__).resolve().parent
SOURCE_FILES = ("CONSTRUCTION_CONTRACT.md", "allowed_schema.csv", "disease_identities.csv",
                "schema_provenance.json", "generate_knhanes.py", "GENERATOR_RULES.md",
                "feature_disease_rules.csv", "test_generator.py")

# Imports complete before the hook. Every subsequent input content read by the
# actual CLI must be one of the four supplied inputs, four authored deliverables,
# or a generated artifact under this invocation's explicit output directory.
GUARDED_CLI = r'''
import encodings.utf_8_sig, json, pathlib, resource, sys
base = pathlib.Path(sys.argv[1]).resolve()
sys.path.insert(0, str(base))
import generate_knhanes as g
# Exercise lazy numerical/CSV module imports before enforcing project-input reads.
g.generate_population(2, next(iter(g.DISEASES))).to_csv(index=False, float_format="%.8g")
output = pathlib.Path(sys.argv[2]).resolve()
allowed = {base / p for p in ("CONSTRUCTION_CONTRACT.md", "allowed_schema.csv", "disease_identities.csv", "schema_provenance.json", "generate_knhanes.py", "GENERATOR_RULES.md", "feature_disease_rules.csv", "test_generator.py")}
reads = set()
def guard(event, args):
    if event == "open" and isinstance(args[0], (str, bytes)):
        path = pathlib.Path(args[0]).resolve()
        mode = args[1]
        if mode is None or "r" in mode or "+" in mode:
            if path not in allowed and output not in path.parents:
                raise RuntimeError("Forbidden input read: " + str(path))
            reads.add(str(path))
sys.addaudithook(guard)
sys.argv = [str(base / "generate_knhanes.py"), "--schema", str(base / "allowed_schema.csv"), "--disease-identities", str(base / "disease_identities.csv"), "--output-dir", str(output)] + sys.argv[3:]
g.main()
print(json.dumps({"input_boundary_checked": True, "reads": sorted(reads), "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}))
'''


def invoke(destination, n, codes, seed=20260831, hash_seed="0", expected_success=True):
    args = [sys.executable, "-c", GUARDED_CLI, str(HERE), str(destination),
            "--n", str(n), "--seed", str(seed), "--diseases"] + list(codes)
    env = dict(os.environ, PYTHONHASHSEED=hash_seed)
    result = subprocess.run(args, capture_output=True, text=True, env=env, timeout=600)
    if expected_success:
        if result.returncode:
            raise AssertionError(result.stdout + "\n" + result.stderr)
        return json.loads(result.stdout.strip().splitlines()[-1])
    assert result.returncode != 0, "Expected existing-output rejection"
    assert "Refusing to overwrite" in result.stderr, result.stderr
    return {"returncode": result.returncode, "reason": "existing output rejected"}


def artifact(directory, code, n):
    return directory / code / ("synthetic_" + code + "_" + str(n) + ".csv")


def pair(frame, *fields):
    return frame.loc[:, list(fields)].dropna()


def physiology(frame, target):
    assert list(frame.columns) == list(g.FIELDS) + [target]
    assert set(frame[target].unique()) <= {0, 1} and not frame[target].isna().any()
    features = frame.loc[:, list(g.FIELDS)]
    values = features.to_numpy()
    assert not np.isinf(values).any()
    assert np.nanmin(values) >= 0
    assert frame["21022"].between(40, 70).all()
    for f, allowed in {"31": {0, 1}, "20116": {0, 1, 2}, "1239": {0, 1, 2},
                       "20117": {0, 1, 2}, "709": set(range(1, 7)), "864": set(range(8))}.items():
        assert set(frame[f].dropna().unique()) <= allowed, f
    smoke = pair(frame, "20116", "1239")
    assert ((smoke["20116"] == 2) == smoke["1239"].isin([1, 2])).all()
    walk = pair(frame, "864", "874")
    assert ((walk["864"] == 0) == (walk["874"] == 0)).all()
    assert walk.loc[walk["864"] > 0, "874"].between(10, 240).all()
    mass = pair(frame, "21002", "21001", "50", "23100", "23101", "23099", "23102")
    np.testing.assert_allclose(mass["21002"], mass["21001"]*(mass["50"]/100)**2, rtol=5e-7, atol=2e-5)
    np.testing.assert_allclose(mass["23100"] + mass["23101"], mass["21002"], rtol=5e-7, atol=2e-5)
    np.testing.assert_allclose(100*mass["23100"]/mass["21002"], mass["23099"], rtol=5e-7, atol=2e-5)
    assert (mass["23102"]/mass["23101"]).between(.69999, .77001).all()
    regional = pair(frame, "23116", "23112", "23124", "23120", "23128", "23100")
    ratio = regional.iloc[:, :5].sum(axis=1)/regional["23100"]
    assert ratio.between(.95499, .98001).all()
    assert (regional["23116"]/regional["23112"]).between(.45, 2.2).all()
    bp = pair(frame, "4079", "4080")
    assert (bp["4080"]-bp["4079"]).between(21.9999, 112.0001).all()
    assert (frame["102"].dropna() % 1 == 0).all()
    cbc = pair(frame, "30010", "30040", "30030", "30020", "30050", "30060")
    np.testing.assert_allclose(cbc["30030"], cbc["30010"]*cbc["30040"]/10, rtol=5e-7, atol=2e-5)
    np.testing.assert_allclose(cbc["30020"], cbc["30010"]*cbc["30050"]/10, rtol=5e-7, atol=2e-5)
    np.testing.assert_allclose(cbc["30060"], 100*cbc["30020"]/cbc["30030"], rtol=5e-7, atol=2e-5)
    lipids = pair(frame, "30690", "30760", "30780")
    assert ((lipids["30690"]-lipids["30760"]-lipids["30780"]) > 0).all()
    # Unit ranges come from authored executable supports, not a cohort reference.
    for field, lower, upper in (("30700", 28, 800), ("30740", 2.8, 24), ("30880", 105, 790),
                                ("30670", 1.5, 26), ("30870", .25, 12), ("30760", .4, 3)):
        assert frame[field].dropna().between(lower-1e-5, upper+1e-5).all(), field
    ngsp = frame["30750"].dropna()*.09148 + 2.152
    assert ngsp.between(3.49999, 14.00001).all()
    crp = frame["30710"].dropna()
    assert ((crp >= .19999) | np.isclose(crp, .2/np.sqrt(2), rtol=1e-6)).all()
    alt = frame["30620"].dropna()
    assert ((alt >= 4.99999) | np.isclose(alt, 5/np.sqrt(2), rtol=1e-6)).all()


def test_inputs_and_coverage(work):
    fields, identities = g.read_inputs(HERE/"allowed_schema.csv", HERE/"disease_identities.csv")
    assert fields == list(g.FIELDS)
    with (HERE/"feature_disease_rules.csv").open(newline="", encoding="utf-8") as handle:
        rules = list(csv.DictReader(handle))
    assert len(rules) == 440
    assert {(r["phenotype"], r["field_id"]) for r in rules} == {(d, f) for d in g.DISEASES for f in g.FIELDS}
    for row in rules:
        assert row["relationship"] in ("direct", "indirect", "neutral")
        assert row["rule"] and row["unit"] == g.UNITS[row["field_id"]]
    for code in g.DISEASES:
        frame, audit = g.generate_population(4096, code, with_audit=True)
        physiology(frame, code)
        assert set(frame[code].unique()) == {0, 1}
        assert audit["rows"] == 4096 and sum(audit["survey_year_counts"].values()) == 4096
        assert set(map(int, audit["survey_year_counts"])) == set(g.DISEASES[code]["years"])
        assert audit["missing"] == {f: int(frame[f].isna().sum()) for f in fields}
    return {"diseases": len(identities), "features": len(fields), "disease_feature_pairs": len(rules)}


def test_determinism_and_no_overwrite(work):
    first, second = list(g.DISEASES)[:2]
    a = g.generate_population(768, first)
    _ = g.generate_population(768, second)
    pd.testing.assert_frame_equal(a, g.generate_population(768, first), check_exact=True)
    assert not a.equals(g.generate_population(768, first, seed=20260832))
    assert not a.equals(g.generate_population(768, first, block_index=1))
    both, subset, reverse = work/"both", work/"subset", work/"reverse"
    guarded = invoke(both, 256, [first, second], hash_seed="17")
    invoke(subset, 256, [first], hash_seed="932")
    invoke(reverse, 256, [second, first], hash_seed="7")
    hashes = [g.sha256_file(artifact(d, first, 256)) for d in (both, subset, reverse)]
    assert len(set(hashes)) == 1
    assert g.sha256_file(artifact(both, second, 256)) == g.sha256_file(artifact(reverse, second, 256))
    invoke(both, 256, [first], expected_success=False)
    assert g.sha256_file(artifact(both, first, 256)) == hashes[0]
    return {"task_order_subset_hash_seed_invariant": True, "overwrite_refused": True,
            "input_boundary_checked": guarded["input_boundary_checked"], "opened_inputs": guarded["reads"]}


def test_missingness_and_input_boundary(work):
    code = next(iter(g.DISEASES))
    frame = g.generate_population(4096, code)
    bia = frame.loc[:, list(g.FIELDS[10:19])].isna()
    assert (bia.nunique(axis=1) == 1).all()
    assert frame["864"].isna().equals(frame["874"].isna())
    assert not frame[["21022", "31", code]].isna().any().any()
    # Counterfactual target replacement has no effect on any measurement mask.
    x = frame.copy()
    y = frame.copy()
    x[code], y[code] = 0, 1
    zeros = np.zeros(len(frame))
    g.apply_missingness(x, np.random.default_rng(500), zeros, zeros, zeros, zeros)
    g.apply_missingness(y, np.random.default_rng(500), zeros, zeros, zeros, zeros)
    pd.testing.assert_frame_equal(x.drop(columns=code), y.drop(columns=code), check_exact=True)
    assert (x[code] == 0).all() and (y[code] == 1).all()
    tree = ast.parse((HERE/"generate_knhanes.py").read_text(encoding="utf-8"))
    imports = {node.names[0].name.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.Import)}
    imports |= {node.module.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    assert imports <= {"argparse", "csv", "hashlib", "json", "os", "pathlib", "numpy", "pandas"}
    calls = {node.func.id for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
    assert not (calls & {"eval", "exec", "compile", "__import__"})
    # A wrong harmonization unit must be rejected, not silently re-converted.
    bad = work/"bad_schema.csv"
    bad.write_text((HERE/"allowed_schema.csv").read_text(encoding="utf-8").replace("mg/L,,HE_hsCRP", "mg/dL,,HE_hsCRP"), encoding="utf-8")
    try:
        g.read_inputs(bad, HERE/"disease_identities.csv")
    except ValueError:
        pass
    else:
        raise AssertionError("Wrong CRP unit was not rejected")
    return {"label_invariant_measurement_masks": True, "generator_import_allowlist_checked": True,
            "wrong_unit_rejected": True, "joint_BIA_and_walking_missingness": True}


def test_full_scale(work):
    code = next(iter(g.DISEASES))
    destination = work/"full_scale"
    metrics = invoke(destination, 360000, [code])
    path = artifact(destination, code, 360000)
    with path.with_suffix(".audit.json").open(encoding="utf-8") as handle:
        audit = json.load(handle)
    rows, cases = 0, 0
    missing = {f: 0 for f in g.FIELDS}
    for frame in pd.read_csv(path, chunksize=20000):
        physiology(frame, code)
        rows += len(frame)
        cases += int(frame[code].sum())
        for field in g.FIELDS:
            missing[field] += int(frame[field].isna().sum())
    assert rows == 360000 == audit["counts"]["rows"]
    assert cases == audit["counts"]["cases"]
    assert missing == audit["counts"]["missing"]
    digest = g.sha256_file(path)
    assert digest == audit["output_sha256"]
    assert audit["block_size"] == 10000
    assert metrics["peak_rss_kib"] < 512 * 1024, metrics
    import resource
    reader_peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    assert reader_peak < 768 * 1024
    return {"rows": rows, "cases": cases, "output_sha256": digest, "bytes": path.stat().st_size,
            "generator_peak_rss_kib": metrics["peak_rss_kib"], "reader_test_peak_rss_kib": reader_peak,
            "generator_limit_mib": 512, "reader_limit_mib": 768, "read_chunk_rows": 20000,
            "generation_block_rows": 10000, "output_removed_after_test": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=HERE/"synthetic_test_report.json")
    parser.add_argument("--skip-full-scale", action="store_true", help="Implementation smoke run only; not final acceptance")
    args = parser.parse_args()
    tests = [test_inputs_and_coverage, test_determinism_and_no_overwrite, test_missingness_and_input_boundary]
    if not args.skip_full_scale:
        tests.append(test_full_scale)
    report = {"schema_version": 1, "status": "running", "synthetic_only": True,
              "full_scale_requested": not args.skip_full_scale, "tests": [],
              "model_configuration_requested": g.MODEL_REQUEST,
              "file_sha256": {name: g.sha256_file(HERE/name) for name in SOURCE_FILES},
              "libraries": {"python": sys.version, "numpy": np.__version__, "pandas": pd.__version__}}
    with tempfile.TemporaryDirectory(prefix="synthetic_test_", dir=HERE) as directory:
        for test in tests:
            started = time.monotonic()
            item = {"name": test.__name__}
            try:
                item.update(status="passed", evidence=test(Path(directory)))
            except Exception as error:
                item.update(status="failed", error=str(error), traceback=traceback.format_exc())
            item["elapsed_seconds"] = round(time.monotonic()-started, 3)
            report["tests"].append(item)
            print(json.dumps(item, ensure_ascii=False), flush=True)
    passed = all(item["status"] == "passed" for item in report["tests"])
    report["status"] = ("passed" if not args.skip_full_scale else "smoke_passed_full_scale_unrun") if passed else "failed"
    with args.report.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print("Report: " + str(args.report), flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
