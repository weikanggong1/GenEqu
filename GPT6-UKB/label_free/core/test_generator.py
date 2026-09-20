"""Synthetic-only consistency tests; no external observations or outcome evaluation."""
import json
import subprocess
import sys
import tempfile
from pathlib import Path
import numpy as np
from generate import Population, read_schema, validate, sha

BASE=Path(__file__).resolve().parent
SCHEMA=read_schema(BASE/"measurement_dictionary.csv")
N=16384

def run():
    model=Population(SCHEMA,N,20260831)
    x=model.generate()
    validate(x,SCHEMA)
    # Exported float32 is tested, not only the unrounded internal values.
    ix={r["Field_ID"]:i for i,r in enumerate(SCHEMA)}
    def v(fid): return x[:,ix[str(fid)]]
    checks=[]
    def identity(name,left,right,rtol=3e-6,atol=3e-6):
        mask=np.isfinite(left)&np.isfinite(right)
        assert mask.sum()>N//20,(name,"too few observed rows")
        np.testing.assert_allclose(left[mask],right[mask],rtol=rtol,atol=atol,err_msg=name)
        checks.append(name)
    identity("body total mass",v(21002),v(23100)+v(23101))
    identity("BMI height weight",v(21001),v(21002)/(v(50)/100)**2)
    identity("body fat percent",v(23099),100*v(23100)/v(21002))
    for pct,fat,lean,pred in [(23115,23116,23117,23118),(23111,23112,23113,23114),(23123,23124,23125,23126),(23119,23120,23121,23122),(23127,23128,23129,23130)]:
        identity(f"segment {pct} fat percent",v(pct),100*v(fat)/(v(fat)+v(lean)))
        identity(f"segment {pred} predicted mass",v(pred),.96*v(lean))
    for total,parts in [(23100,[23116,23112,23124,23120,23128]),(23101,[23117,23113,23125,23121,23129])]:
        remainder=v(total)-sum(v(fid) for fid in parts)
        assert np.nanmin(remainder)>0,"Negative residual head/other compartment"
    checks.append("positive residual body compartments")
    identity("haematocrit",v(30030),v(30010)*v(30040)/10)
    identity("haemoglobin",v(30020),v(30010)*v(30050)/10)
    identity("MCHC relation",v(30050),v(30040)*v(30060)/100)
    white_counts=[30160,30150,30120,30130,30140]
    white_pcts=[30220,30210,30180,30190,30200]
    identity("white differential counts sum",v(30000),sum(v(i) for i in white_counts))
    identity("white differential percent sum",np.full(N,100.),sum(v(i) for i in white_pcts))
    for count,pct in zip(white_counts,white_pcts):
        identity(f"white count {count} percent",v(count),v(30000)*v(pct)/100)
    identity("platelet crit",v(30090),v(30080)*v(30100)/10000)
    identity("reticulocyte count",v(30250),v(30010)*v(30240)/100)
    identity("high scatter reticulocyte count",v(30300),v(30010)*v(30290)/100)
    identity("nucleated RBC percent",v(30230),100*v(30170)/v(30000))
    identity("assumed lipid accounting",v(30690),v(30780)+v(30760)+v(30870)/2.2)
    assert np.nanmin(v(4080)-v(4079))>=25-1e-5
    assert np.nanmax(v(3063)-v(3062))<0
    assert np.nanmax(v(30660)-v(30840))<0
    assert np.nanmin(v(30860)-v(30600))>0
    assert np.nanmax(v(1090)+v(1080)+v(1070)+v(1160))<=24
    checks.append("physiological inequalities and time budget")
    for days,duration in [(884,894),(904,914),(864,874)]:
        mask=(v(days)==0)&np.isfinite(v(duration)); assert np.all(v(duration)[mask]==0)
    mask=(v(20116)==0)&np.isfinite(v(20161)); assert np.all(v(20161)[mask]==0)
    for living,dead in [(2946,1807),(1845,3526)]:
        assert not np.any(np.isfinite(v(living))&np.isfinite(v(dead)))
    checks.append("questionnaire applicability")
    for i,row in enumerate(SCHEMA):
        values=x[np.isfinite(x[:,i]),i]
        assert len(values)>0 and len(np.unique(values))>1,row["Field_ID"]
    checks.append("all 201 measurements observed and variable")
    # These broad directions are prescribed model properties, not empirical estimates.
    for left,right,label in [(46,47,"bilateral grip"),(30700,30720,"shared filtration"),(30740,30750,"shared glucose regulation"),(30620,30650,"shared hepatic turnover")]:
        mask=np.isfinite(v(left))&np.isfinite(v(right))
        assert np.corrcoef(v(left)[mask],v(right)[mask])[0,1]>.2,label
    checks.append("prespecified joint-physiology directions")
    smaller=Population(SCHEMA,257,20260831).generate()
    np.testing.assert_array_equal(x[:257],smaller)
    np.testing.assert_array_equal(smaller,Population(SCHEMA,257,20260831).generate())
    assert not np.array_equal(smaller,Population(SCHEMA,257,20260832).generate(),equal_nan=True)
    checks.append("fixed-seed repeatability, different-seed change and row-prefix stability")
    without=Population(SCHEMA,257,20260831).generate(apply_missingness=False)
    # Disabling random nonresponse preserves all already observed values.
    np.testing.assert_array_equal(smaller[np.isfinite(smaller)],without[np.isfinite(smaller)])
    checks.append("missingness does not perturb latent measured values")
    with tempfile.TemporaryDirectory(prefix="ssl_dictionary_cli_") as temp:
        output=Path(temp)/"sample"
        command=[sys.executable,str(BASE/"generate.py"),"--schema",str(BASE/"measurement_dictionary.csv"),
                 "--output-dir",str(output),"--n","257","--seed","20260831"]
        result=subprocess.run(command,capture_output=True,text=True,check=True)
        y=np.load(output/"synthetic_features_float32.npy",allow_pickle=False)
        np.testing.assert_array_equal(y,smaller)
        columns=json.loads((output/"feature_columns.json").read_text())
        assert columns==[r["Field_ID"] for r in SCHEMA]
        parameters=json.loads((output/"generation_parameters.json").read_text())
        assert parameters["seed"]==20260831 and parameters["n"]==257
        assert len(parameters["parameters_and_rules"])==201
        for name,digest in parameters["output_sha256"].items(): assert sha(output/name)==digest
        assert subprocess.run(command,capture_output=True).returncode!=0,"Output overwrite accepted"
    checks.append("CLI save reload, exact schema order, output hashes and overwrite refusal")
    return {"status":"passed","synthetic_sample_n":N,"feature_count":201,
            "production_generation_run":False,"empirical_data_used":False,
            "checks":checks,"number_of_checks":len(checks),
            "report_note":"Synthetic mathematical consistency only; no realism, predictive utility or outcome claims."}

if __name__=="__main__":
    report=run()
    (BASE/"TEST_REPORT.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))
