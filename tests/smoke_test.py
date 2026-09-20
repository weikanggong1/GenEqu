"""Small all-endpoint runtime checks; no real participant data are used."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

def run(*args):
    result = subprocess.run([sys.executable, *map(str, args)], cwd=ROOT, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(result.stdout + result.stderr)
    return result

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='Optional JSON validation report.')
    args = parser.parse_args()
    report = {'synthetic_only': True, 'clinical': {}, 'brain': {}, 'protein': {}, 'label_free': {}, 'rare_risk': {}}
    with tempfile.TemporaryDirectory(prefix='genequ_smoke_') as tmp:
        tmp = Path(tmp)
        for family, count, features in [('GPT6-UKB',30,201), ('GPT6-UKB-rare',15,201), ('GPT6-NHANES',14,52), ('GPT6-KNHANES',10,44)]:
            out = tmp / family
            run(ROOT/family/'generate.py', '--n', 1000, '--seed', 20260831, '--output-dir', out)
            files = sorted(out.rglob('synthetic_*.csv'))
            assert len(files) == count, (family, len(files))
            for path in files:
                frame = pd.read_csv(path)
                assert frame.shape == (1000, features + 1), (path, frame.shape)
                assert set(frame.iloc[:,-1].unique()) <= {0,1}
                assert not np.isinf(frame.to_numpy(dtype=float)).any()
            report['clinical'][family] = {'endpoints': count, 'rows_per_endpoint': 1000, 'measurements': features}
            print(f'{family}: {count} endpoints passed', flush=True)
        correctness = run(ROOT/'GPT6-UKB-rare/core/test_model.py')
        print(correctness.stdout + correctness.stderr, flush=True)
        rare = pd.read_csv(next((tmp/'GPT6-UKB-rare').glob('synthetic_E4_HYPERPARA_*.csv')))
        names = pd.read_csv(ROOT/'GPT6-UKB-rare/observed_risk_inputs.csv',dtype={'Field_ID':str}).Field_ID.tolist()
        assert len(names) == 38 and len(set(names)) == 38
        full = rare.iloc[:, :-1]
        results = []
        for label, frame in [('full',full), ('selected',full[names])]:
            source = tmp/f'risk_{label}_input.csv'
            destination = tmp/f'risk_{label}_output.csv'
            frame.to_csv(source,index=False)
            run(ROOT/'GPT6-UKB-rare/predict.py','--input',source,'--disease','E4_HYPERPARA','--output',destination)
            probability = pd.read_csv(destination)['probability'].to_numpy()
            assert len(probability) == 1000 and np.isfinite(probability).all()
            assert np.all((probability >= 0) & (probability <= 1))
            results.append(probability)
        np.testing.assert_allclose(*results,rtol=0,atol=1e-13)
        report['rare_risk'] = {'correctness_tests':9,'conditioning_fields':38,
                              'cli_rows':1000,'full_and_selected_input_agree':True}
        for task, classes in [('ad-binary',2), ('ad-multiclass',3), ('scz',2)]:
            outputs=[]
            for replicate in range(2):
                path=tmp/f'{task}_{replicate}.npz'
                run(ROOT/'GPT6-Brain/generate.py','--task',task,'--n-per-class',50,'--output',path)
                outputs.append(np.load(path,allow_pickle=False))
            a,b=outputs
            assert a['X'].shape == (classes*50,15)
            assert np.all(np.isfinite(a['X'])) and np.all(a['X']>0)
            assert np.array_equal(np.bincount(a['y']),np.full(classes,50))
            assert all(np.array_equal(a[k],b[k]) for k in a.files)
            report['brain'][task] = {'rows': classes*50, 'measurements':15, 'deterministic':True}
        for i in range(2):
            run(ROOT/'GPT6-protein/generate.py','--n',120,'--output',tmp/f'protein_{i}.npz')
        a,b=[np.load(tmp/f'protein_{i}.npz',allow_pickle=False) for i in range(2)]
        assert a['X'].shape==(120,648) and a['Y'].shape==(120,6)
        assert np.isfinite(a['X']).all()
        assert set(np.unique(a['Y'])) == {0,1}
        assert not (a['Y'][:,0].astype(bool) & a['Y'][:,1:].any(axis=1)).any()
        assert all(np.array_equal(a[k],b[k]) for k in a.files)
        report['protein']={'rows':120,'measurements':648,'binary_tasks':6,'deterministic':True}
        label_free_runs=[]
        for n in [256,113]:
            output=tmp/f'label_free_{n}'
            run(ROOT/'GPT6-UKB/label_free/generate.py','--n',n,'--seed',20260831,'--output-dir',output)
            x=np.load(output/'synthetic_features_float32.npy',allow_pickle=False)
            assert x.shape==(n,201) and x.dtype==np.float32
            columns=json.loads((output/'feature_columns.json').read_text())
            schema=pd.read_csv(ROOT/'GPT6-UKB/allowed_schema.csv',dtype={'Field_ID':str})
            assert columns==schema.Field_ID.tolist()
            assert not np.isinf(x).any()
            label_free_runs.append(x)
        np.testing.assert_array_equal(label_free_runs[0][:113],label_free_runs[1])
        index={fid:i for i,fid in enumerate(columns)}
        left=label_free_runs[0][:,index['21002']]
        right=label_free_runs[0][:,index['23100']]+label_free_runs[0][:,index['23101']]
        valid=np.isfinite(left)&np.isfinite(right)
        np.testing.assert_allclose(left[valid],right[valid],rtol=3e-6)
        report['label_free']={'rows':256,'measurements':201,'prefix_rows_checked':113,
                              'dtype':'float32','body_mass_identity':True}
    report['status']='passed'
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':
    main()
