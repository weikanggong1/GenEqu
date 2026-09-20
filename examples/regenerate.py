"""Regenerate the small public synthetic examples using the packaged interfaces."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/"examples/synthetic"

def run(*args):
    return subprocess.run([sys.executable,*map(str,args)],cwd=ROOT,capture_output=True,text=True,check=True)

def main():
    records=[]
    DEST.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="genequ_examples_") as temp:
        temp=Path(temp)
        for family,disease in [("GPT6-UKB","E4_DM2"),("GPT6-UKB-rare","E4_HYPERPARA"),("GPT6-NHANES","E4_DM2"),("GPT6-KNHANES","E4_DM2")]:
            output=temp/family
            run(ROOT/family/"generate.py","--diseases",disease,"--n",1000,"--seed",20260831,"--output-dir",output)
            source=next(output.rglob("synthetic_*.csv"))
            name=f"{family}_{disease}.csv";shutil.copyfile(source,DEST/name)
            records.append({"file":name,"generator":family,"endpoint":disease,"rows":1000,"seed":20260831,"synthetic":True})
        for task,seed in [("ad-binary",20260908),("ad-multiclass",20260908),("scz",20260918)]:
            name=f"GPT6-Brain_{task}.npz";output=temp/name
            run(ROOT/"GPT6-Brain/generate.py","--task",task,"--n-per-class",50,"--seed",seed,"--output",output)
            shutil.copyfile(output,DEST/name)
            records.append({"file":name,"generator":"GPT6-Brain","task":task,"n_per_class":50,"seed":seed,"synthetic":True})
        name="GPT6-protein.npz";output=temp/name
        run(ROOT/"GPT6-protein/generate.py","--n",120,"--seed",20260908,"--output",output)
        shutil.copyfile(output,DEST/name)
        records.append({"file":name,"generator":"GPT6-protein","rows":120,"seed":20260908,"synthetic":True})
        output=temp/"label_free"
        run(ROOT/"GPT6-UKB/label_free/generate.py","--n",256,"--seed",20260831,"--output-dir",output)
        for source,name in [("synthetic_features_float32.npy","GPT6-UKB_label_free.npy"),("feature_columns.json","GPT6-UKB_label_free_columns.json")]:
            shutil.copyfile(output/source,DEST/name)
            records.append({"file":name,"generator":"GPT6-UKB","mode":"label_free","rows":256,"measurements":201,"seed":20260831,"synthetic":True})
        import pandas as pd
        frame=pd.read_csv(DEST/"GPT6-UKB-rare_E4_HYPERPARA.csv")
        names=pd.read_csv(ROOT/"GPT6-UKB-rare/observed_risk_inputs.csv",dtype={"Field_ID":str}).Field_ID.tolist()
        name="GPT6-UKB-rare_predictors.csv";frame[names].to_csv(DEST/name,index=False)
        records.append({"file":name,"generator":"GPT6-UKB-rare","mode":"observed_risk_inputs","rows":1000,"measurements":38,"seed":20260831,"synthetic":True})
        name="GPT6-UKB-rare_probabilities.csv";output=temp/name
        run(ROOT/"GPT6-UKB-rare/predict.py","--input",DEST/"GPT6-UKB-rare_predictors.csv","--disease","E4_HYPERPARA","--output",output)
        shutil.copyfile(output,DEST/name)
        records.append({"file":name,"generator":"GPT6-UKB-rare","mode":"observed_risk","endpoint":"E4_HYPERPARA","rows":1000,"synthetic":True})
    for item in records:
        item["sha256"]=hashlib.sha256((DEST/item["file"]).read_bytes()).hexdigest()
    (DEST/"manifest.json").write_text(json.dumps(records,indent=2)+"\n")
    result=run(ROOT/"examples/train_logistic_regression.py","--csv",DEST/"GPT6-UKB_E4_DM2.csv","--schema",ROOT/"GPT6-UKB/allowed_schema.csv")
    (ROOT/"examples/logistic_demo_report.json").write_text(json.dumps(json.loads(result.stdout),indent=2)+"\n")
    print(json.dumps({"status":"passed","synthetic_artifacts":len(records)},indent=2))

if __name__=="__main__":
    main()
