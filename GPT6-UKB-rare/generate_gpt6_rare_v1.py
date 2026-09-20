#!/usr/bin/env python3
"""Independent knowledge-authored prevalent-cohort simulator; see GENERATOR_RULES.md."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd

VERSION = 'gpt6-rare-v1.0.0'
# prior, case age shift (latent SD), case male probability, treatment probability,
# direct shifts on latent marginal-normal scale. Numerical values are assumptions.
RULES = {
'E4_HYPERPARA': (.008, .45, .25, .45, {'30680':1.5,'30810':-.7,'30610':.45,'30890':-.3,'30700':.2,'46':-.15,'47':-.15}),
'F5_PHOBANX': (.035,-.3,.35,.4, {'1200':.55,'102':.2,'1160':-.15,'924':-.15}),
'K11_COELIAC': (.012,-.15,.35,.65, {'30020':-.45,'30040':-.3,'30070':.3,'30600':-.3,'30890':-.4,'21001':-.25,'1438':-.65,'1458':-.2}),
'F5_ALCOHOL_DEPENDENCE': (.018,-.1,.72,.35, {'1568':.3,'1578':.25,'1588':1.0,'1598':1.2,'1608':.4,'1558':-.8,'30730':1.0,'30650':.55,'30620':.25,'30040':.65,'30080':-.25,'30600':-.2,'1200':.35}),
'I9_ABAORTANEUR': (.009,.9,.82,.55, {'20161':.7,'4080':.25,'30710':.15,'30780':.1,'3063':-.15}),
'K11_ACUTPANC': (.007,.05,.55,.65, {'30710':1.1,'30140':.6,'30600':-.4,'30620':.3,'30730':.35,'30870':.35,'30740':.25,'30680':-.25,'102':.35}),
'F5_ALZHDEMENT': (.006,1.3,.43,.35, {'46':-.45,'47':-.45,'924':-.5,'21001':-.35,'1070':.2,'1080':-.4,'864':-.3}),
'K11_ULCER': (.009,-.15,.5,.65, {'30710':.65,'30020':-.4,'30080':.35,'30600':-.35,'30140':.3,'21001':-.2}),
'I9_CARDMYO': (.008,.45,.66,.7, {'102':.35,'4080':-.25,'30700':.3,'30720':.35,'30670':.3,'924':-.5,'3062':-.25,'3063':-.25,'46':-.2,'47':-.2}),
'C3_NONHODGKIN_EXALLC': (.007,.6,.57,.55, {'30120':.35,'30020':-.4,'30080':-.15,'30710':.5,'30600':-.3,'30860':.2,'21001':-.15}),
'M13_POLYMYALGIA': (.012,.95,.35,.7, {'30710':1.1,'30080':.35,'30020':-.3,'30600':-.25,'46':-.4,'47':-.4,'924':-.4}),
'C3_MELANOMA_SKIN_EXALLC': (.01,.25,.5,.65, {'1717':-.45,'1737':.55,'1050':.15,'2277':.25}),
'N14_ENDOMETRIOSIS': (.035,-.85,0.,.55, {'30020':-.2,'30070':.15,'1200':.25,'924':-.15}),
'C3_BLADDER_EXALLC': (.008,.75,.78,.6, {'20161':.75,'30020':-.15,'30700':.1}),
'M13_FIBROMYALGIA': (.025,-.05,.18,.55, {'1200':.7,'1220':.35,'1160':-.15,'924':-.4,'46':-.35,'47':-.35,'21001':.15})}

# Shared unit-variance Gaussian physiological factors; remaining variance is noise.
GROUPS = {
'body_fat': ('21001 48 49 23099 23100 23115 23111 23116 23112 23123 23119 23124 23120 23127 23128', .75),
'lean': ('50 20015 23101 23102 23117 23113 23118 23114 23121 23125 23126 23122 23129 23130 23105 46 47', .7),
'impedance': ('23106 23110 23109 23108 23107', .8),
'lung': ('3062 3063 3064 20258', .75),
'pressure': ('4079 4080', .7),
'activity': ('894 914 874 884 904 864 924', .5),
'sleep': ('1200 1220 1190', .45),
'alcohol': ('1568 1578 1588 1598 1608', .65),
'red_cells': ('30010 30020 30030', .75),
'red_size': ('30040 30050 30260 30270', .75),
'retic': ('30250 30240 30300 30290 30280', .6),
'platelet': ('30080 30090', .75),
'inflammation': ('30710 30140 30130 30000', .5),
'lymph': ('30120 30180', .65),
'eosin': ('30150 30210', .7),
'baso': ('30160 30220', .7),
'liver': ('30620 30650 30730 30660 30840', .5),
'renal': ('30700 30720 30670 30880', .65),
'lipid': ('30690 30780 30640', .8),
'hdl': ('30760 30630', .85),
'glycemia': ('30740 30750', .8),
'nutrition': ('30600 30860', .5),
}
FIELD_GROUP = {f:(g,w) for g,(fields,w) in GROUPS.items() for f in fields.split()}
DERIVED = {'21002','30030','30050','30060','30090','30180','30190','30200','30210','30220','3063'}


def seed_for(seed, disease):
    return int.from_bytes(hashlib.sha256(f'{seed}:{disease}'.encode()).digest()[:8], 'little')


def load_schema(path):
    with open(path, encoding='utf-8-sig') as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 201 or len({r['Field_ID'] for r in rows}) != 201:
        raise ValueError('Expected exactly 201 unique fields')
    return rows


def normal_cdf(z):
    # Vectorized Abramowitz-Stegun approximation; no scipy dependency.
    x = np.abs(z); t = 1/(1+.2316419*x)
    tail = np.exp(-x*x/2)/np.sqrt(2*np.pi)*t*(.319381530+t*(-.356563782+t*(1.781477937+t*(-1.821255978+t*1.330274429))))
    return np.where(z >= 0, 1-tail, tail)


def marginal(row, z):
    allowed = row['Recommended_Allowed_Values']
    med = float(row['Observed_Median'])
    if allowed:
        codes = np.array(json.loads(allowed), dtype=float)
        # Unknown category frequencies: maximum-entropy prior with median preference.
        weights = np.exp(-.65*np.abs(codes-med)); weights /= weights.sum()
        return codes[np.minimum(np.searchsorted(np.cumsum(weights), normal_cdf(z)),len(codes)-1)]
    knots = np.array([float(row[k]) for k in ('Recommended_Hard_Min','Observed_P01','Observed_Median','Observed_P99','Recommended_Hard_Max')])
    return np.interp(normal_cdf(z), [0,.01,.5,.99,1], knots)


def generate(schema, disease, n=360000, seed=20260831):
    if disease not in RULES or n < 2:
        raise ValueError('Unknown disease or n < 2')
    prior, age_shift, male_case, treatment_p, effects = RULES[disease]
    if n*prior < 1 or n*(1-prior) < 1:
        raise ValueError('n too small for prespecified prior to supply both classes')
    rng = np.random.default_rng(seed_for(seed,disease))
    # Systematic stratified sampling directly from specified mixture; no retries or label edits.
    y = ((np.arange(n)+rng.random())/n < prior).astype(np.int8)
    rng.shuffle(y)
    sex = (rng.random(n) < np.where(y==1,male_case,.48)).astype(float)
    if disease == 'N14_ENDOMETRIOSIS': sex[:] = 0
    age_z = rng.normal(size=n)+age_shift*y
    treated = (rng.random(n)<treatment_p)*y
    severity = rng.lognormal(-.5*.35**2,.35,n)
    activity = y*severity*(1-.65*treated)
    latent = {g:rng.normal(size=n) for g in GROUPS}
    latent['lean'] += .65*(2*sex-1)-.15*age_z
    latent['body_fat'] += .15*age_z-.2*(2*sex-1)
    latent['impedance'] += -.4*latent['lean']
    latent['lung'] += .4*latent['lean']-.2*age_z
    latent['pressure'] += .2*age_z+.2*latent['body_fat']
    latent['glycemia'] += .25*latent['body_fat']
    latent['renal'] += .15*age_z
    values = {}
    for row in schema:
        f = row['Field_ID']; z = rng.normal(size=n)
        if f in FIELD_GROUP:
            group,w = FIELD_GROUP[f]; z = w*latent[group]+np.sqrt(1-w*w)*z
        if f in effects:
            # Risk-history features do not disappear with treatment.
            risk_history = f in {'20161','1717','1737','1050','2277'}
            z += effects[f]*(y if risk_history else activity)
        if f == '21022': z=age_z
        if f == '30850': z=.85*(2*sex-1)+.35*z
        if f == '30800': z += .3*(1-sex)-.3*age_z
        if f == '30830': z += -.3*(2*sex-1)-.15*latent['body_fat']
        values[f] = sex.copy() if f=='31' else marginal(row,z)
    # Physiological algebra before bounds/precision/missingness.
    values['21002'] = values['21001']*(values['50']/100)**2
    values['30030'] = values['30010']*values['30040']/10
    values['30050'] = values['30020']*10/np.maximum(values['30010'],.1)
    values['30060'] = values['30020']*100/np.maximum(values['30030'],1)
    values['30090'] = values['30080']*values['30100']/10000
    counts=['30140','30120','30130','30150','30160']
    total=sum(values[f] for f in counts)
    values['30000']=total
    for count,pct in zip(counts,['30200','30180','30190','30210','30220']):
        values[pct]=100*values[count]/np.maximum(total,.01)
    values['3063']=np.minimum(values['3063'],.96*values['3062'])
    for f in ['699','757','845','2139']: values[f]=np.minimum(values[f],values['21022'])
    # Independent masks have exactly the same mechanism in cases and controls.
    for row in schema:
        f=row['Field_ID']; dp=int(row['Recommended_Decimal_Places']); scale=10.**dp
        lo=np.ceil(float(row['Recommended_Hard_Min'])*scale)/scale
        hi=np.floor(float(row['Recommended_Hard_Max'])*scale)/scale
        x=np.clip(np.round(values[f],dp),lo,hi)
        x[rng.random(n)<float(row['Recommended_Missing_Rate'])]=np.nan
        values[f]=x
    frame=pd.DataFrame(values,columns=[r['Field_ID'] for r in schema]); frame[disease]=y
    return frame


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--schema',required=True); p.add_argument('--output-dir',required=True)
    p.add_argument('--n',type=int,default=360000); p.add_argument('--seed',type=int,default=20260831)
    p.add_argument('--cohort-mode',choices=['population_based'],default='population_based')
    p.add_argument('--diseases',nargs='+',choices=list(RULES),default=list(RULES))
    p.add_argument('--disease-identities'); p.add_argument('--overwrite',action='store_true')
    args=p.parse_args(); schema=load_schema(args.schema)
    if args.disease_identities:
        with open(args.disease_identities,encoding='utf-8-sig') as handle:
            ids=list(csv.DictReader(handle))
        if {r['phenotype'] for r in ids} != set(RULES): raise ValueError('Disease identity manifest mismatch')
    for disease in dict.fromkeys(args.diseases):
        directory=Path(args.output_dir)/disease; directory.mkdir(parents=True,exist_ok=True)
        dest=directory/f'synthetic_{disease}_{args.n}.csv'; audit=directory/f'synthetic_{disease}_{args.n}.audit.json'
        if not args.overwrite and (dest.exists() or audit.exists()): raise FileExistsError(dest)
        partial=dest.with_suffix('.csv.partial')
        if partial.exists(): raise FileExistsError(partial)
        frame=generate(schema,disease,args.n,args.seed)
        try:
            with open(partial,'x') as handle: frame.to_csv(handle,index=False)
            if not args.overwrite and dest.exists(): raise FileExistsError(dest)
            os.replace(partial,dest)
        finally:
            if partial.exists(): partial.unlink()
        report={'version':VERSION,'disease':disease,'n':args.n,'seed':args.seed,'prior':RULES[disease][0],
                'synthetic_class_counts':frame[disease].value_counts().to_dict(),'n_features':201,
                'domain':'current/prevalent diagnosis; not future incident outcomes',
                'schema_sha256':hashlib.sha256(Path(args.schema).read_bytes()).hexdigest()}
        audit.write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report),flush=True)
        del frame

if __name__ == '__main__': main()
