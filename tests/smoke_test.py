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

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='Optional JSON validation report.')
    args = parser.parse_args()
    report = {'synthetic_only': True, 'clinical': {}, 'brain': {}, 'protein': {}, 'label_free': {}}
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
        run(ROOT/'GPT6-UKB/label_free/generate_label_free_population_v4.py','--n',256,'--audit-rows',128,'--output-dir',tmp/'label_free')
        x=np.load(tmp/'label_free/synthetic_features_float32.npy',allow_pickle=False)
        assert x.shape==(256,201)
        report['label_free']={'rows':256,'measurements':201}
    report['status']='passed'
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':
    main()
