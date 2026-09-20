#!/usr/bin/env python3
"""Positive synthetic AD/MCI FIRST/AAL3 cohorts from the frozen public-literature map.
Requires numpy only. Never reads target features, labels, predictions, or model files.
"""
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent
ARMS=('knowledge','shuffled_identity','reversed_direction','no_effect')
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load_effects(atlas):
 manifest=json.loads((P/'EVIDENCE_FREEZE_MANIFEST.json').read_text())
 for item in manifest['files']:
  if item['path'] in ['feature_effects.csv',f'identity/{atlas}_identity.csv']:
   assert sha(P/item['path'])==item['sha256'], 'Frozen input hash mismatch: '+item['path']
 ident=list(csv.DictReader((P/f'identity/{atlas}_identity.csv').open()))
 assert len(ident)==(15 if atlas=='first' else 166)
 assert [int(r['column_index']) for r in ident]==list(range(len(ident)))
 names=[r['feature_name'] for r in ident]
 effects=list(csv.DictReader((P/'feature_effects.csv').open()))
 beta=np.zeros((3,len(names)),dtype=np.float64)
 for label,disease in [(1,'MCI'),(2,'AD')]:
  rr=[r for r in effects if r['atlas']==atlas.upper() and r['disease']==disease]
  assert [r['feature_name'] for r in rr]==names
  beta[label]=[float(r['model_latent_sd_shift']) for r in rr]
 return names,beta

def anatomy(name,atlas):
 base=name[2:] if atlas=='first' and name.startswith(('L_','R_')) else name.rsplit('_',1)[0] if name.endswith(('_L','_R')) else name
 if base.startswith('Thal') or base=='Thalamus':group='thalamus'
 elif base in ['Caudate','Putamen','Pallidum','Accumbens','N_Acc']:group='basal_ganglia'
 elif base.startswith(('Cerebellum','Vermis')):group='cerebellum'
 elif base in ['Hippocampus','Amygdala','ParaHippocampal','Olfactory']:group='limbic'
 elif base.startswith(('Temporal','Heschl','Fusiform')):group='temporal'
 elif base.startswith(('Occipital','Calcarine','Cuneus','Lingual')):group='occipital'
 elif base.startswith(('Parietal','Postcentral','SupraMarginal','Angular','Precuneus','Paracentral')):group='parietal'
 elif base.startswith(('LC','VTA','SN_','Red_N','Raphe','BrainStem')):group='brainstem'
 else:group='frontal_cingulate_insula'
 return base,group

def make(atlas,task,n,seed):
 names,beta=load_effects(atlas);p=len(names)
 classes=np.array(['NC','AD'] if task=='binary' else ['NC','MCI','AD'])
 label_ids=np.array([0,2] if task=='binary' else [0,1,2])
 y=np.repeat(np.arange(len(classes),dtype=np.int8),n);num=len(y)
 rng=np.random.default_rng(np.random.SeedSequence([seed,15 if atlas=='first' else 166,len(classes)]))
 pairs,groups=zip(*(anatomy(x,atlas) for x in names))
 pairnames=sorted(set(pairs));groupnames=sorted(set(groups))
 pi=np.array([pairnames.index(x) for x in pairs]);gi=np.array([groupnames.index(x) for x in groups])
 # Independent standard normal factors make a PSD covariance by construction.
 # Shared across all four arms, including the exact individual noise realizations.
 global_factor=rng.normal(size=(num,1));group_factor=rng.normal(size=(num,len(groupnames)))
 pair_factor=rng.normal(size=(num,len(pairnames)));independent=rng.normal(size=(num,p))
 noise=.4*global_factor+.35*group_factor[:,gi]+.65*pair_factor[:,pi]+np.sqrt(.295)*independent
 severity=rng.lognormal(mean=-.5*.30**2,sigma=.30,size=(num,1))
 # Baselines are class-independent modeling assumptions; AAL3 units are synthetic.
 first_mu={'Thalamus':9600.,'Caudate':4600.,'Putamen':5900.,'Pallidum':2750.,'BrainStem':20000.,'Hippocampus':4600.,'Amygdala':1500.,'Accumbens':450.}
 mu=np.array([first_mu[anatomy(x,atlas)[0]] for x in names]) if atlas=='first' else np.ones(p)
 baseline=np.log(mu)[None,:]+.16*noise
 perm_rng=np.random.default_rng(np.random.SeedSequence([seed,89017,p]));perm=perm_rng.permutation(p)
 assert not np.array_equal(perm,np.arange(p))
 betas={'knowledge':beta,'shuffled_identity':beta[:,perm],'reversed_direction':-beta,'no_effect':np.zeros_like(beta)}
 order=rng.permutation(num);arrays={};effects={}
 for arm in ARMS:
  log_effect=.16*severity*betas[arm][label_ids[y]]
  arrays[arm]=np.exp(baseline+log_effect).astype(np.float32)[order]
  effects[arm]=log_effect[order]
 return arrays,y[order],classes,names,dict(permutation=perm,betas=betas,beta=beta,baseline=baseline[order],effects=effects,groups=groups,pairs=pairs,order=order)

def audit(arrays,y,classes,names,state,n):
 p=len(names);nc=(y==0);base=arrays['no_effect'];checks={}
 checks['schema']=all(x.shape==(len(y),p) and x.dtype==np.float32 for x in arrays.values()) and y.dtype==np.int8
 checks['positive_finite']=all(bool(np.all(np.isfinite(x)) and np.all(x>0)) for x in arrays.values())
 checks['balanced_classes']=np.bincount(y,minlength=len(classes)).tolist()==[n]*len(classes)
 checks['NC_identical_across_arms']=all(np.array_equal(arrays[a][nc],base[nc]) for a in ARMS)
 checks['shared_background_identity']=all(np.allclose(np.log(arrays[a].astype(float))-state['effects'][a],state['baseline'],atol=2e-7,rtol=0) for a in ARMS)
 checks['reversal_exact_in_log_effect']=bool(np.array_equal(state['effects']['reversed_direction'],-state['effects']['knowledge']))
 checks['shuffle_preserves_each_class_effect_multiset']=all(np.array_equal(np.sort(state['betas']['shuffled_identity'][k]),np.sort(state['beta'][k])) for k in range(3))
 checks['same_permutation_all_classes']=bool(np.array_equal(state['betas']['shuffled_identity'],state['beta'][:,state['permutation']]))
 checks['no_effect_zero']=bool(np.count_nonzero(state['effects']['no_effect'])==0)
 checks['feature_order_unique']=len(names)==len(set(names))==p
 checks['covariance_PSD_by_construction']=True # Factor outer products plus 0.295*identity.
 assert all(checks.values()),checks
 return {'status':'PASS','checks':checks,'shape':[len(y),p],'n_per_class':n,'class_names':classes.tolist(),'feature_count':p,'mean_shift_nonzero_AD':int(np.count_nonzero(state['beta'][2])),'mean_shift_nonzero_MCI':int(np.count_nonzero(state['beta'][1])),'limits':'Pure synthetic invariants only; no target accuracy or realism validation. Means are log-scale shifts, not guaranteed realized raw-volume Cohen d.'}

def write_effect_audit(outdir,arrays,y,classes,names,state,atlas,task):
 literature={(r['feature_name'],r['disease']):r['literature_d_median'] for r in csv.DictReader((P/'feature_effects.csv').open()) if r['atlas']==atlas.upper()}
 label_ids=[0,2] if task=='binary' else [0,1,2]
 rows=[]
 for arm,x in arrays.items():
  control=x[y==0].astype(float);nc=len(control);mu0=control.mean(0);var0=control.var(0,ddof=1)
  for label,disease in enumerate(classes[1:],start=1):
   case=x[y==label].astype(float);n=len(case);mu=case.mean(0);var=case.var(0,ddof=1)
   pooled=np.sqrt(((n-1)*var+(nc-1)*var0)/(n+nc-2));smd=(mu-mu0)/pooled
   for j,name in enumerate(names):
    anchor=literature[(name,str(disease))]
    rows.append(dict(arm=arm,disease=str(disease),feature_name=name,n_case=n,n_NC=nc,synthetic_case_mean=float(mu[j]),synthetic_NC_mean=float(mu0[j]),synthetic_case_sd=float(np.sqrt(var[j])),synthetic_NC_sd=float(np.sqrt(var0[j])),realized_synthetic_raw_Cohen_d=float(smd[j]),specified_log_latent_SD_shift=float(state['betas'][arm][label_ids[label],j]),literature_d_anchor=anchor,raw_d_minus_literature_anchor=float(smd[j])-float(anchor) if anchor else '',interpretation='Pure synthetic descriptive check; log shift is not exact raw Cohen d; n64 errors expected; no accuracy or target calibration'))
 with (outdir/'aggregate_effect_audit.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--atlas',choices=['first','aal3'],required=True);ap.add_argument('--task',choices=['binary','multiclass'],required=True);ap.add_argument('--outdir',type=Path,required=True);ap.add_argument('--n-per-class',type=int,default=50000);ap.add_argument('--seed',type=int,default=20260908);ap.add_argument('--audit-only',action='store_true');args=ap.parse_args()
 assert args.n_per_class>=2
 if args.outdir.exists() and (not args.outdir.is_dir() or any(args.outdir.iterdir())):
  raise SystemExit('Refusing nonempty output directory: '+str(args.outdir))
 arrays,y,classes,names,state=make(args.atlas,args.task,args.n_per_class,args.seed)
 report=audit(arrays,y,classes,names,state,args.n_per_class)
 # Determinism check uses a small second construction with the same frozen inputs and seed.
 aa,yy,cc,nn,ss=make(args.atlas,args.task,17,args.seed);bb,yb,cb,nb,sb=make(args.atlas,args.task,17,args.seed)
 report['checks']['deterministic_same_seed']=all(np.array_equal(aa[a],bb[a]) for a in ARMS) and np.array_equal(yy,yb)
 assert report['checks']['deterministic_same_seed']
 args.outdir.mkdir(parents=True,exist_ok=True)
 report.update(atlas=args.atlas,task=args.task,seed=args.seed,generator_sha256=sha(__file__),feature_effects_sha256=sha(P/'feature_effects.csv'),evidence_freeze_sha256=sha(P/'EVIDENCE_FREEZE_MANIFEST.json'))
 (args.outdir/'synthetic_audit.json').write_text(json.dumps(report,indent=2))
 write_effect_audit(args.outdir,arrays,y,classes,names,state,args.atlas,args.task)
 manifest=dict(report,arms={},identity_permutation=state['permutation'].tolist(),audited_only=args.audit_only)
 if not args.audit_only:
  for arm in ARMS:
   dest=args.outdir/(args.atlas+'_'+args.task+'_'+arm+'.npz')
   np.savez_compressed(dest,X=arrays[arm],y=y,feature_names=np.array(names),class_names=classes)
   # Round-trip verification disallows object arrays and confirms exact order/data.
   with np.load(dest,allow_pickle=False) as z:
    assert set(z.files)=={'X','y','feature_names','class_names'}
    assert np.array_equal(z['X'],arrays[arm]) and np.array_equal(z['y'],y) and z['feature_names'].tolist()==names and z['class_names'].tolist()==classes.tolist()
   manifest['arms'][arm]={'filename':dest.name,'sha256':sha(dest),'bytes':dest.stat().st_size}
 (args.outdir/'manifest.json').write_text(json.dumps(manifest,indent=2))
 print(json.dumps({'status':'PASS','atlas':args.atlas,'task':args.task,'shape':report['shape'],'files':list(manifest['arms'])}))
if __name__=='__main__':main()
