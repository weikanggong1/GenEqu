"""Public-evidence SCZ generator. No real inputs, estimators, or evaluation code."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

SEED=20260918
ARMS=('knowledge','shuffled_identity','reversed_direction','no_effect')
CLASSES=np.asarray(['HC','SCZ'])
P=Path(__file__).resolve().parent

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def spec(task):
    with (P/'feature_effects.csv').open(newline='') as f:
        rows=[r for r in csv.DictReader(f) if r['space']=={'first':'FIRST','aal3':'AAL3'}[task]]
    assert [int(r['column_index']) for r in rows]==list(range({'first':15,'aal3':166}[task]))
    assert len({r['feature_name'] for r in rows})==len(rows)
    return rows

def log_shift(d,sigma):
    """Equal-log-variance lognormal contrast with the requested pooled-volume d.

    This is an assumed source-family conversion; heterogeneous case amplitudes
    make the realized pooled d approximate. Never calibrate it to target data.
    """
    k=d*np.sqrt(np.expm1(sigma*sigma)/2.)
    assert np.all(np.abs(k)<1)
    ratio=(1+k*np.sqrt(2-k*k))/(1-k*k)
    return np.log(ratio)

def background(rows,n):
    task_id=0 if rows[0]['space']=='FIRST' else 1
    streams=np.random.SeedSequence([SEED,task_id]).spawn(3)
    rng=np.random.default_rng(streams[0])
    names=np.asarray([r['feature_name'] for r in rows]);p=len(rows)
    family_names=sorted({r['factor_family'] for r in rows})
    pair_names=sorted({r['homolog_group'] for r in rows})
    fi=np.asarray([family_names.index(r['factor_family']) for r in rows])
    pi=np.asarray([pair_names.index(r['homolog_group']) for r in rows])
    # Variance fractions .16 global + .20 family + .24 homolog + .40 residual =1.
    z=(.4*rng.standard_normal((2*n,1))+np.sqrt(.20)*rng.standard_normal((2*n,len(family_names)))[:,fi]+np.sqrt(.24)*rng.standard_normal((2*n,len(pair_names)))[:,pi]+np.sqrt(.40)*rng.standard_normal((2*n,p)))
    median=np.asarray([float(r['baseline_median_assumed']) for r in rows])
    sigma=np.asarray([float(r['background_log_sd_assumed']) for r in rows])
    base=np.log(median)[None,:]+sigma[None,:]*z
    amp_rng=np.random.default_rng(streams[1])
    amplitude=amp_rng.lognormal(-.5*.35**2,.35,size=n) # mean1; heterogeneity is an assumption.
    perm=np.random.default_rng(streams[2]).permutation(p)
    effects=np.asarray([float(r['central_standardized_effect']) for r in rows])
    return base,amplitude,perm,effects,sigma,names

def arm_array(base,amplitude,permutation,effects,sigma,arm):
    n=len(amplitude)
    if arm=='knowledge': current=effects
    elif arm=='shuffled_identity': current=effects[permutation]
    elif arm=='reversed_direction': current=-effects
    elif arm=='no_effect': current=np.zeros_like(effects)
    else: raise ValueError(arm)
    out=base.copy()
    out[n:]+=amplitude[:,None]*log_shift(current,sigma)[None,:]
    x=np.exp(out).astype(np.float32)
    assert np.all(np.isfinite(x)) and np.all(x>0)
    return x

def smd(x,n):
    a=x[:n].astype(np.float64);b=x[n:].astype(np.float64)
    return (b.mean(0)-a.mean(0))/np.sqrt((a.var(0,ddof=1)+b.var(0,ddof=1))/2)

def self_test():
    for task in ['first','aal3']:
        rows=spec(task);values=background(rows,512);base,amp,perm,effect,sigma,names=values
        arrays={a:arm_array(base,amp,perm,effect,sigma,a) for a in ARMS}
        assert sorted(perm.tolist())==list(range(len(rows)))
        assert not np.array_equal(effect,effect[perm])
        assert sorted(effect.tolist())==sorted(effect[perm].tolist())
        for a in ARMS:
            assert np.array_equal(arrays[a][:512],arrays['no_effect'][:512])
            assert arrays[a].shape==(1024,len(rows))
        # Reversal is antisymmetric in log shift; each participant shares the same background.
        assert np.allclose(np.log(arrays['knowledge'][512:])+np.log(arrays['reversed_direction'][512:]),2*np.log(arrays['no_effect'][512:]),atol=5e-6,rtol=0)
        again=background(rows,512)
        assert all(np.array_equal(a,b) for a,b in zip(values,again))
        assert np.array_equal(arrays['knowledge'],arm_array(*values[:5],arm='knowledge'))
    return {'status':'PASS','tests':['identity_axes','positive_float32','shared_HC','shared_case_background_and_antisymmetric_log_shifts','one_to_one_identity_shuffle','deterministic_fixed_seed'],'participant_data_read':False,'fitted_models':0}

def generate(task,n,out):
    dest=out/task
    if dest.exists() and any(dest.iterdir()):
        raise FileExistsError('Refusing nonempty synthetic output directory: '+str(dest))
    rows=spec(task);base,amp,perm,effect,sigma,names=background(rows,n)
    y=np.repeat(np.asarray([0,1],dtype=np.int8),n)
    dest.mkdir(parents=True,exist_ok=True)
    manifest={'task':task,'seed':SEED,'n_per_class':n,'n_features':len(rows),'class_names':CLASSES.tolist(),'code_sha256':sha(Path(__file__)),'effect_spec_sha256':sha(P/'feature_effects.csv'),'evidence_sha256':sha(P/'evidence_records.csv'),'coverage_sha256':sha(P/'feature_coverage_181.csv'),'identity_shuffle_destination_to_source':perm.tolist(),'background_sha256':hashlib.sha256(base.tobytes()).hexdigest(),'files':{},'metrics':{},'target_access':False,'model_fits':0}
    hc_hash=None
    for arm in ARMS:
        x=arm_array(base,amp,perm,effect,sigma,arm)
        h=hashlib.sha256(x[:n].tobytes()).hexdigest()
        if hc_hash is None: hc_hash=h
        assert hc_hash==h
        filename=dest/(arm+'.npz')
        np.savez_compressed(filename,X=x,y=y,feature_names=names,class_names=CLASSES)
        measured=smd(x,n)
        central=effect if arm=='knowledge' else effect[perm] if arm=='shuffled_identity' else -effect if arm=='reversed_direction' else np.zeros_like(effect)
        manifest['files'][filename.name]={'sha256':sha(filename),'bytes':filename.stat().st_size}
        manifest['metrics'][arm]={'shape':list(x.shape),'dtype':str(x.dtype),'y_dtype':str(y.dtype),'minimum':float(x.min()),'maximum':float(x.max()),'HC_sha256':h,'central_d_before_heterogeneity_by_feature':dict(zip(names.tolist(),central.tolist())),'mean_abs_smd':float(np.abs(measured).mean()),'smd_by_feature':dict(zip(names.tolist(),measured.tolist()))}
    manifest['status']='PASS_GENERATED_SYNTHETIC_CHECKS'
    (dest/'SYNTHETIC_AUDIT.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return {'task':task,'status':manifest['status'],'directory':str(dest),'n':2*n,'n_features':len(rows)}

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--task',choices=['first','aal3','both'],default='both')
    parser.add_argument('--n-per-class',type=int,default=40000)
    parser.add_argument('--out',type=Path)
    parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args()
    if args.self_test:
        if args.out and args.out.exists() and any(args.out.iterdir()):
            parser.error('Refusing nonempty self-test output directory: '+str(args.out))
        result=self_test()
        if args.out:
            args.out.mkdir(parents=True,exist_ok=True);(args.out/'SYNTHETIC_SELF_TEST.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result));return
    if args.out is None: parser.error('--out is required for generation')
    if args.n_per_class<2: parser.error('--n-per-class must be >=2')
    tasks=['first','aal3'] if args.task=='both' else [args.task]
    for task in tasks:
        dest=args.out/task
        if dest.exists() and any(dest.iterdir()):
            parser.error('Refusing nonempty synthetic output directory: '+str(dest))
    for task in tasks:
        print(json.dumps(generate(task,args.n_per_class,args.out)),flush=True)
if __name__=='__main__': main()
