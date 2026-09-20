"""Analytic Gaussian conditioning and quadrature; never fits any model."""
from __future__ import annotations

import argparse
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import ndtr, ndtri, logsumexp

from model import FACTORS, FACTOR_INDEX, OBSERVATIONS, RISKS, encode_observation, vector, read_schema

OBS_IDS = tuple(OBSERVATIONS) + ('20161',)
LOADINGS = np.vstack([vector(s[7]) for s in OBSERVATIONS.values()] + [vector({'smoking':1})])
NOISE = np.array([s[6] for s in OBSERVATIONS.values()] + [.55])
MALE_SHIFTS = np.array([s[5] for s in OBSERVATIONS.values()] + [0.])
SMOKING = FACTOR_INDEX['smoking']
AGE = FACTOR_INDEX['age']
QUADRATURE_ORDER = 48
_nodes, _weights = np.polynomial.legendre.leggauss(QUADRATURE_ORDER)
QUAD_U, QUAD_W = (_nodes+1)/2, _weights/2
SMOKE_CAP = np.log1p(222/np.exp(2.2))/.75

@lru_cache(maxsize=4096)
def _conditioning_parameters(indices, age_known):
    rows = np.array(indices, dtype=int)
    free = np.array([j for j in range(len(FACTORS)) if not(age_known and j==AGE)])
    lmat = LOADINGS[rows][:,free]
    precision = np.eye(len(free)) + (lmat.T / NOISE[rows]**2) @ lmat
    cov_free = np.linalg.inv(precision)
    covariance = np.zeros((len(FACTORS),len(FACTORS)))
    covariance[np.ix_(free,free)] = cov_free
    logdet = np.linalg.slogdet(precision)[1] + np.sum(2*np.log(NOISE[rows]))
    return rows, free, lmat, covariance, cov_free, logdet

def _conditional_event(mean, covariance, disease, sex, smoke_censor):
    intercept, sex_shift, coefficients = RISKS[disease]
    b = vector(coefficients)
    event_mean = intercept+sex_shift*sex+mean@b
    event_variance = 1+b@covariance@b
    if not smoke_censor:
        probability=ndtr(event_mean/np.sqrt(event_variance))
        log_evidence=np.zeros(len(mean))
    else:
        wmean=mean[:,SMOKING]
        wvariance=covariance[SMOKING,SMOKING]+.55**2
        threshold=(0 if smoke_censor==-1 else SMOKE_CAP)
        cdf=ndtr((threshold-wmean)/np.sqrt(wvariance))
        evidence=np.clip(cdf if smoke_censor==-1 else 1-cdf,1e-14,1.)
        # Integrate W over its truncated Gaussian CDF interval. Node set is fixed.
        u = evidence[:,None]*QUAD_U[None,:]
        if smoke_censor==1:
            u=1-evidence[:,None]+u
        standardized_w=ndtri(np.clip(u,1e-15,1-1e-15))
        cross=covariance[SMOKING]@b
        conditional_sd=np.sqrt(max(event_variance-cross*cross/wvariance,1e-12))
        conditional_mean=event_mean[:,None]+cross/np.sqrt(wvariance)*standardized_w
        probability=ndtr(conditional_mean/conditional_sd)@QUAD_W
        log_evidence=np.log(evidence)
    if disease=='N14_ENDOMETRIOSIS' and sex==1:
        probability=np.zeros(len(mean))
    return probability,log_evidence

def _predict_block(values, index, disease):
    n=len(values)
    y=np.full((n,len(OBS_IDS)),np.nan)
    for j,fid in enumerate(OBS_IDS[:-1]):
        if fid not in index:
            continue
        raw=values[:,index[fid]]
        spec=OBSERVATIONS[fid]
        valid=np.isfinite(raw)&(raw>spec[2])&(raw<spec[3])
        y[valid,j]=encode_observation(raw[valid],spec)
    censor=np.zeros(n,dtype=np.int8)
    if '20161' in index:
        raw=values[:,index['20161']]
        valid=np.isfinite(raw)&(raw>0)&(raw<222)
        y[valid,-1]=np.log1p(raw[valid]/np.exp(2.2))/.75
        censor[raw==0]=-1
        censor[raw==222]=1
    age=np.full(n,np.nan)
    if '21022' in index:
        raw=values[:,index['21022']]
        valid=np.isfinite(raw)&(raw>=39)&(raw<=70)
        age[valid]=ndtri(np.clip((raw[valid]-39)/31,1e-10,1-1e-10))
    sex=np.full(n,-1,dtype=np.int8)
    if '31' in index:
        raw=values[:,index['31']]
        sex[raw==0]=0
        sex[raw==1]=1
    mask=np.isfinite(y)
    keys=np.column_stack([np.packbits(mask,axis=1),np.isfinite(age),sex+1,censor+1]).astype(np.uint8)
    groups,inverse=np.unique(keys,axis=0,return_inverse=True)
    answer=np.empty(n)
    for group_id in range(len(groups)):
        selected=np.flatnonzero(inverse==group_id)
        first=selected[0]
        age_known=bool(np.isfinite(age[first]))
        indices=tuple(np.flatnonzero(mask[first]).tolist())
        rows,free,lmat,covariance,cov_free,logdet=_conditioning_parameters(indices,age_known)
        sexes=(int(sex[first]),) if sex[first]>=0 else (0,1)
        probabilities,log_weights=[],[]
        for sex_value in sexes:
            residual=y[selected][:,rows]-sex_value*MALE_SHIFTS[rows]
            if age_known:
                residual=residual-age[selected,None]*LOADINGS[rows,AGE]
            natural=(residual/NOISE[rows]**2)@lmat
            mean=np.zeros((len(selected),len(FACTORS)))
            mean[:,free]=natural@cov_free
            if age_known:
                mean[:,AGE]=age[selected]
            quadratic=np.sum(residual**2/NOISE[rows]**2,axis=1)-np.sum(natural*(natural@cov_free),axis=1)
            likelihood=-.5*(quadratic+logdet+len(rows)*np.log(2*np.pi))
            p,extra_likelihood=_conditional_event(mean,covariance,disease,sex_value,int(censor[first]))
            probabilities.append(p)
            log_weights.append(likelihood+extra_likelihood)
        if len(sexes)==1:
            answer[selected]=probabilities[0]
        else:
            log_weights=np.asarray(log_weights)
            weights=np.exp(log_weights-logsumexp(log_weights,axis=0))
            answer[selected]=np.sum(weights*np.asarray(probabilities),axis=0)
    return np.clip(answer,0,1)

def predict_risk(X, feature_names, disease):
    """Return P(15-year event | declared measured subset), with no fitting.

    X: numeric N by P array, NaN permitted. feature_names: P unique Field_IDs.
    All 201 ordered fields are accepted; only OBS_IDS, age and sex are used.
    Missing/unselected measurements are marginalized, not mean-imputed.
    The result is invariant to row ordering and external batching.
    """
    if disease not in RISKS:
        raise ValueError('Unknown disease code: '+str(disease))
    values=np.asarray(X,dtype=np.float64)
    names=list(map(str,feature_names))
    if values.ndim!=2 or values.shape[1]!=len(names) or len(set(names))!=len(names):
        raise ValueError('X must be N by P with P unique feature names.')
    allowed=set(read_schema(Path(__file__).with_name('measurement_dictionary.csv')).Field_ID)
    if set(names)-allowed:
        raise ValueError('Unknown predictor names; outcome columns are not accepted.')
    index={name:j for j,name in enumerate(names)}
    result=np.empty(len(values))
    for start in range(0,len(values),8192):
        result[start:start+8192]=_predict_block(values[start:start+8192],index,disease)
    return result

def main():
    parser=argparse.ArgumentParser(description='Apply fixed equations; no fitting or calibration.')
    parser.add_argument('--input',required=True,help='Predictor CSV; disease/outcome columns are rejected.')
    parser.add_argument('--disease',required=True,choices=sorted(RISKS))
    parser.add_argument('--output',required=True,help='New probability CSV; must not exist.')
    args=parser.parse_args()
    output=Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    frame=pd.read_csv(args.input)
    probabilities=predict_risk(frame.to_numpy(),frame.columns,args.disease)
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as handle:
        pd.DataFrame({'probability':probabilities}).to_csv(handle,index=False)

if __name__=='__main__':
    main()
