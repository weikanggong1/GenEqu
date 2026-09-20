"""Generate per-disease independent populations into a new directory."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from model import CHUNK_SIZE, MODEL_VERSION, RISKS, disease_seed, generate, population_block, read_schema, sample_target

def sha256_file(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()

def main():
    parser=argparse.ArgumentParser(description='Dictionary-only 15-year synthetic disease generator.')
    parser.add_argument('--schema',required=True)
    parser.add_argument('--disease-identities',required=True)
    parser.add_argument('--output-dir',required=True,help='New directory; existing directories are rejected.')
    parser.add_argument('--n',type=int,default=360000,help='Independent rows per disease.')
    parser.add_argument('--seed',type=int,default=20260831)
    parser.add_argument('--diseases',nargs='+',required=True,choices=sorted(RISKS))
    args=parser.parse_args()
    schema=read_schema(args.schema)
    identities=pd.read_csv(args.disease_identities)
    if args.n<1 or len(set(args.diseases))!=len(args.diseases):
        raise ValueError('Positive n and distinct disease codes are required.')
    if not set(args.diseases)<=set(identities.phenotype):
        raise ValueError('Requested diseases are absent from disease identities.')
    output=Path(args.output_dir)
    output.mkdir(parents=True,exist_ok=False)
    audit={'model_version':MODEL_VERSION,'status':'generating','horizon_years':15,
           'seed':args.seed,'rows_per_disease':args.n,'chunk_size':CHUNK_SIZE,
           'seed_derivation':'SeedSequence([seed, four little-endian uint32 SHA256(disease) words, chunk_index])',
           'inputs':{str(Path(p).name):sha256_file(p) for p in [args.schema,args.disease_identities]},
           'sources':{name:sha256_file(Path(__file__).with_name(name)) for name in ['model.py','generate.py','predict.py']},
           'predictor_fields':list(schema.Field_ID),'outcomes':{}}
    (output/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    for disease in args.diseases:
        path=output/f'synthetic_{disease}_{args.n}.csv'
        positive=0
        missing={fid:0 for fid in schema.Field_ID}
        with path.open('x') as handle:
            for chunk,start in enumerate(range(0,args.n,CHUNK_SIZE)):
                rng=np.random.default_rng(disease_seed(args.seed,disease,chunk))
                frame,z,male=population_block(schema,min(CHUNK_SIZE,args.n-start),rng)
                labels=sample_target(z,male,disease,rng)
                positive+=int(labels.sum())
                for fid,count in frame.isna().sum().items():
                    missing[fid]+=int(count)
                frame[disease]=labels
                frame.to_csv(handle,index=False,header=start==0,float_format='%.9g')
        audit['outcomes'][disease]={'path':path.name,'n':args.n,'positive':positive,'negative':args.n-positive,
                                    'target_column':disease,'sha256':sha256_file(path),'missing_counts':missing}
        (output/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
        print(json.dumps({'disease':disease,'rows':args.n,'positive':positive,'file':path.name}),flush=True)
    audit['status']='complete'
    (output/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')

if __name__=='__main__':
    main()
