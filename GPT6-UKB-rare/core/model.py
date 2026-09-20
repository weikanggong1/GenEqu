"""Fixed, dictionary-only generative assumptions; no learned parameters."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit, ndtr, ndtri

FACTORS = ('age', 'adiposity', 'smoking', 'alcohol', 'activity', 'anxiety',
           'inflammation', 'renal', 'calcium', 'immune', 'lipids', 'glycemia',
           'pigmentation', 'uv', 'reproduction', 'education', 'sleep')
FACTOR_INDEX = {name: i for i, name in enumerate(FACTORS)}
CHUNK_SIZE = 10000
MODEL_VERSION = 'dictionary_probit_15y_1'

# Each tuple: center, dispersion, support low/high, log scale, sex shift,
# residual SD, factor loadings. Dispersion is a local scale, NOT an empirical SD.
OBSERVATIONS = {
 '845': (18, 3, 5, 35, False, 0, .35, {'education': 1}),
 '21001': (26, 4, 14.8744, 58.5937, False, .08, .45, {'adiposity': 1, 'age': .12, 'activity': -.12}),
 '4080': (124, 13, 80, 225.5, False, .15, .8, {'age': .65, 'adiposity': .35, 'renal': .25, 'smoking': .15}),
 '102': (68, 9, 35, 128.5, False, 0, .85, {'anxiety': .4, 'activity': -.4, 'smoking': .15, 'sleep': .2}),
 '20258': (0, 1, -3.687, 4.931, False, 0, .65, {'smoking': -.8, 'age': -.1, 'activity': .15}),
 '30010': (4.5, .4, 2.5, 5.4, False, .65, .65, {'immune': -.25, 'renal': -.2, 'smoking': .1}),
 '30040': (90, 5, 68, 103, False, 0, .8, {'immune': -.4, 'alcohol': .25}),
 '30060': (32.5, .6, 30, 34, False, 0, .85, {'immune': -.15}),
 '30000': (6.4, .28, 1.2, 23.15, True, .05, .7, {'inflammation': .55, 'immune': .2, 'smoking': .2}),
 '30620': (23, .38, 3.77, 384.73, True, .2, .8, {'adiposity': .45, 'alcohol': .35, 'inflammation': .15}),
 '30600': (44, 2.5, 25.83, 57.75, False, 0, .8, {'inflammation': -.45, 'immune': -.2, 'renal': -.15}),
 '30610': (76, .25, 19.3, 384.3, True, 0, .85, {'age': .15, 'calcium': .35, 'alcohol': .15}),
 '30630': (1.5, .19, .771, 2.498, False, -.35, .7, {'lipids': -.4, 'adiposity': -.3, 'activity': .2}),
 '30640': (1.0, .2, .412, 1.966, False, .1, .65, {'lipids': .8, 'adiposity': .2}),
 '30650': (24, .25, 8.2, 459.9, True, .1, .8, {'alcohol': .5, 'adiposity': .2}),
 '30710': (1.5, .9, .08, 71.1, True, 0, .6, {'inflammation': .9, 'immune': .3, 'adiposity': .3, 'smoking': .12}),
 '30680': (2.35, .1, 1.87, 3.452, False, 0, .65, {'calcium': 1, 'renal': -.12, 'immune': -.1}),
 '30700': (75, .22, 10.8, 819, True, .75, .55, {'renal': .8, 'age': .15, 'activity': .1}),
 '30720': (.85, .18, .295, 5.581, True, .1, .6, {'renal': .9, 'age': .25, 'inflammation': .1}),
 '30730': (27, .5, 5.8, 1092.9, True, .35, .65, {'alcohol': .8, 'adiposity': .3}),
 '30740': (5.2, .12, 1.585, 32.746, True, .1, .65, {'glycemia': .9, 'adiposity': .25}),
 '30750': (35, 5, 15.6, 126.6, False, 0, .55, {'glycemia': .95, 'age': .15, 'adiposity': .2}),
 '30760': (1.45, .3, .569, 4.107, False, -.35, .55, {'lipids': -.7, 'activity': .3, 'adiposity': -.3}),
 '30770': (20, .22, 3.559, 68.832, True, .1, .85, {'age': -.45, 'reproduction': .15}),
 '30780': (3.2, .7, 1.128, 7.418, False, .1, .55, {'lipids': .85, 'adiposity': .15}),
 '30790': (28, .85, 3.8, 189, True, 0, .98, {'lipids': .15}),
 '30800': (240, .7, 175, 12647.8, True, -.8, .9, {'reproduction': .7, 'age': -.4}),
 '30810': (1.1, .14, .425, 1.844, False, -.15, .7, {'calcium': -.6, 'renal': .3}),
 '30820': (15, .55, 10, 118.6, True, 0, .8, {'immune': .75, 'age': .15}),
 '30830': (48, .4, .47, 236.68, True, -.35, .7, {'adiposity': -.5, 'reproduction': .2, 'age': .15}),
 '30850': (1.5, .65, .35, 45.795, True, 3.4, .85, {'reproduction': .2, 'age': -.2, 'adiposity': -.15}),
 '30870': (1.4, .4, .356, 10.969, True, .15, .7, {'lipids': .65, 'adiposity': .3, 'alcohol': .3, 'glycemia': .2}),
 '30880': (290, 55, 101.3, 765.5, False, .7, .7, {'renal': .4, 'adiposity': .4, 'alcohol': .2}),
 '30670': (5, .22, 1.48, 26.19, True, .15, .7, {'renal': .6, 'age': .15}),
 '30890': (50, .35, 10, 169, True, 0, .8, {'uv': .45, 'activity': .2, 'adiposity': -.3, 'pigmentation': .15}),
}

# Probit intercept, male shift, factor coefficients. All are author assumptions.
RISKS = {
 'E4_HYPERPARA': (-2.8, -.18, {'age': .35, 'calcium': .8, 'renal': .25}),
 'F5_PHOBANX': (-2.8, -.15, {'age': -.12, 'anxiety': .85, 'sleep': .25, 'education': -.08}),
 'K11_COELIAC': (-3.0, -.1, {'immune': .85, 'age': -.08}),
 'F5_ALCOHOL_DEPENDENCE': (-2.8, .3, {'age': -.12, 'alcohol': .9, 'anxiety': .25, 'smoking': .18, 'education': -.12}),
 'I9_ABAORTANEUR': (-3.4, .6, {'age': .7, 'smoking': .75, 'lipids': .2, 'inflammation': .12}),
 'K11_ACUTPANC': (-3.1, .08, {'age': .12, 'alcohol': .65, 'lipids': .25, 'adiposity': .25, 'calcium': .18, 'smoking': .2}),
 'F5_ALZHDEMENT': (-3.6, -.05, {'age': 1.05, 'education': -.3, 'activity': -.18, 'glycemia': .16, 'smoking': .12}),
 'K11_ULCER': (-3.0, 0, {'immune': .75, 'inflammation': .25, 'age': -.08, 'smoking': -.12}),
 'I9_CARDMYO': (-3.1, .25, {'age': .3, 'alcohol': .5, 'adiposity': .22, 'inflammation': .2, 'renal': .12}),
 'C3_NONHODGKIN_EXALLC': (-3.2, .15, {'age': .55, 'immune': .45, 'inflammation': .2}),
 'M13_POLYMYALGIA': (-3.4, -.15, {'age': .85, 'immune': .35, 'inflammation': .5}),
 'C3_MELANOMA_SKIN_EXALLC': (-3.0, .1, {'age': .25, 'pigmentation': .55, 'uv': .6}),
 'N14_ENDOMETRIOSIS': (-3.0, 0, {'age': -.65, 'reproduction': .75, 'inflammation': .2}),
 'C3_BLADDER_EXALLC': (-3.5, .5, {'age': .65, 'smoking': .75}),
 'M13_FIBROMYALGIA': (-2.8, -.35, {'anxiety': .5, 'sleep': .65, 'activity': -.22, 'adiposity': .18}),
}

def vector(coefficients):
    result = np.zeros(len(FACTORS))
    for name, value in coefficients.items():
        result[FACTOR_INDEX[name]] = value
    return result

def observation_parameters(spec):
    center, dispersion, low, high, logarithmic, male, noise, load = spec
    if logarithmic:
        low, high, center = np.log([low, high, center])
    fraction = (center - low) / (high - low)
    return low, high, logit(fraction), dispersion / ((high - low) * fraction * (1-fraction))

def encode_observation(value, spec):
    low, high, offset, scale = observation_parameters(spec)
    if spec[4]:
        value = np.log(value)
    fraction = (value - low) / (high - low)
    return (logit(fraction) - offset) / scale

def decode_observation(score, spec):
    low, high, offset, scale = observation_parameters(spec)
    value = low + (high - low) * expit(offset + scale * score)
    return np.exp(value) if spec[4] else value

def read_schema(schema):
    result = pd.read_csv(schema, dtype={'Field_ID': str}) if not isinstance(schema, pd.DataFrame) else schema.copy()
    result['Field_ID'] = result['Field_ID'].astype(str)
    if len(result) != 201 or result['Field_ID'].duplicated().any():
        raise ValueError('The interface requires exactly 201 distinct ordered fields.')
    return result

def disease_seed(seed, disease, chunk):
    digest = hashlib.sha256(disease.encode('utf-8')).digest()
    return np.random.SeedSequence([int(seed), *np.frombuffer(digest[:16], dtype='<u4').tolist(), int(chunk)])

def population_block(schema, n, rng, with_metadata=False):
    """Baseline measurements are constructed without any disease argument."""
    schema = read_schema(schema)
    z = rng.normal(size=(n, len(FACTORS)))
    male = rng.binomial(1, .5, n)
    f = {name: z[:, j] for j, name in enumerate(FACTORS)}
    x, meta = {}, {}
    limits = schema.set_index('Field_ID')

    def normal(sd=1):
        return rng.normal(0, sd, n)

    def put(fid, value, formula, factors=()):
        fid = str(fid)
        if fid in x:
            raise RuntimeError('Duplicate field: ' + fid)
        row = limits.loc[fid]
        value = np.broadcast_to(np.asarray(value, dtype=float), (n,)).copy()
        low, high = float(row.Recommended_Hard_Min), float(row.Recommended_Hard_Max)
        value = np.clip(value, low, high)
        if row.Recommended_Data_Type in ('bounded_integer_or_count', 'categorical_integer_code', 'binary_integer_code'):
            value = np.rint(value)
        x[fid] = value
        meta[fid] = {'formula': formula, 'factors': sorted(set(factors))}
        return value

    def categorical(fid, score, factors, formula):
        row = limits.loc[str(fid)]
        allowed = np.asarray(json.loads(row.Recommended_Allowed_Values), dtype=float)
        # Uniform normal-score thresholds are explicit assumptions, not code semantics.
        k = np.minimum((ndtr(score) * len(allowed)).astype(int), len(allowed)-1)
        return put(fid, allowed[k], formula + '; ordinal bins at Phi^-1(k/K)', factors)

    age = put('21022', 39 + 31*ndtr(f['age']), '39 + 31 Phi(age); age factor N(0,1)', ['age'])
    put('31', male, 'Bernoulli(0.5); 0 female, 1 male', ['sex'])
    for fid, spec in OBSERVATIONS.items():
        score = z @ vector(spec[7]) + spec[5]*male + spec[6]*normal()
        put(fid, decode_observation(score, spec), 'bounded Gaussian observation; see OBSERVATIONS table', list(spec[7]) + (['sex'] if spec[5] else []))

    education = x['845']
    put('699', (age-18)*expit(.4*f['age'] + normal()), '(age-18) sigmoid(0.4 age_factor + N)', ['age'])
    put('709', 1+rng.poisson(np.clip(1.3-.15*f['age'], .3, 3)), '1 + Poisson(clip(1.3-0.15 age_factor,0.3,3))', ['age'])
    employed = rng.random(n) < expit(2-.13*(age-55))
    put('796', employed*np.exp(1.7+.55*normal()+.15*f['education']), 'employed exp(1.7+0.55 N+0.15 education_factor)', ['age','education'])
    put('757', employed*np.maximum(age-education,0)*rng.beta(2,2,n), 'employed max(age-education_years,0) Beta(2,2)', ['age','education'])
    put('767', employed*np.clip(37+8*normal(),4,80), 'employed clip(37+8N,4,80); employed~Bernoulli(sigmoid(2-0.13(age-55)))', ['age'])
    put('777', employed*rng.binomial(6,.75,n), 'employed Binomial(6,0.75)', ['age'])
    put('4079', np.minimum(x['4080']-12, 77+7*normal()+4*f['adiposity']-2*f['age']), 'min(systolic-12,77+7N+4 adiposity-2 age_factor)', ['age','adiposity','renal','smoking','sex'])

    bmi = x['21001']
    height = 163+13*male-1.4*f['age']+5.5*normal()
    height = np.clip(height, np.maximum(138,100*np.sqrt(35.4/bmi)), np.minimum(201,100*np.sqrt(181.7/bmi)))
    put('50', height, 'clip(163+13 male-1.4 age_factor+5.5N, max(138,100 sqrt(35.4/BMI)), min(201,100 sqrt(181.7/BMI)))', ['age','sex','adiposity','activity'])
    weight = put('21002', bmi*(height/100)**2, 'BMI (height_cm/100)^2', ['age','sex','adiposity','activity'])
    put('20015', .525*height+1.5*normal(), '0.525 height + 1.5N', ['age','sex','adiposity','activity'])
    put('48', 82+2.3*(bmi-25)+7*male+3*normal(), '82+2.3(BMI-25)+7male+3N', ['age','sex','adiposity','activity'])
    put('49', 98+1.7*(bmi-25)-2*male+3*normal(), '98+1.7(BMI-25)-2male+3N', ['age','sex','adiposity','activity'])
    grip = 26+16*male-3*f['age']+2.5*f['activity']+4*normal()
    put('46', grip+normal(), '26+16male-3age_factor+2.5activity+4N_shared+N_left', ['age','sex','activity'])
    put('47', grip+2+normal(), 'left grip generating base+2+N_right', ['age','sex','activity'])
    fraction = expit(-.7-.55*male+.28*f['adiposity']+.12*f['age']+.12*normal())
    fraction = np.clip(fraction, np.maximum.reduce([np.full(n,.07),5.1/weight,1-97.5/weight]), np.minimum.reduce([np.full(n,.56),87.9/weight,1-28.6/weight]))
    fat, lean = weight*fraction, weight*(1-fraction)
    body_factors = ['age','sex','adiposity','activity']
    put('23099', 100*fraction, '100*feasible sigmoid(-0.7-0.55male+0.28adiposity+0.12age_factor+0.12N); feasible fat fraction enforces total mass bounds', body_factors)
    put('23100', fat, 'weight * body_fat_percentage/100', body_factors)
    put('23101', lean, 'weight - whole_body_fat_mass', body_factors)
    put('23102', .732*lean, '0.732 whole_body_fat_free_mass', body_factors)
    put('23105', 4.184*(370+21.6*lean), '4.184*(370+21.6*whole_body_fat_free_mass) kJ/day', body_factors)
    regions = [('leg_left','23115','23116','23117','23118',.16,.17), ('leg_right','23111','23112','23113','23114',.16,.17), ('arm_left','23123','23124','23125','23126',.045,.05), ('arm_right','23119','23120','23121','23122',.045,.05), ('trunk','23127','23128','23129','23130',.52,.52)]
    for region,pct,fm,ffm,pm,a,b in regions:
        localfat, locallean = a*fat, b*lean
        if region == 'trunk':
            locallean = np.clip(locallean,16,52.3)
        put(fm,localfat,f'{a} whole_body_fat_mass',body_factors)
        put(ffm,locallean,f'{b} whole_body_fat_free_mass' + (' clipped [16,52.3]' if region=='trunk' else ''),body_factors)
        put(pct,100*localfat/(localfat+locallean),f'100 field{fm}/(field{fm}+field{ffm})',body_factors)
        put(pm,.96*locallean,f'0.96 field{ffm}',body_factors)
    for fid,center in [('23106',530),('23110',350),('23109',350),('23108',250),('23107',250)]:
        put(fid, center*(50/lean)**.7*np.exp(.08*normal()), f'{center}*(50/whole_body_fat_free_mass)^0.7 exp(0.08N)',body_factors)
    fvc = put('3062', np.clip(3.4+.9*male+.045*(height-165)-.025*(age-55)+.25*f['activity']+.35*normal(),.9,6.1), 'clip(3.4+0.9male+0.045(height-165)-0.025(age-55)+0.25activity+0.35N,0.9,6.1)',body_factors)
    ratio = np.clip(.8-.0015*(age-55)+.05*x['20258'],.35,.97)
    put('3063', fvc*ratio, 'FVC clip(0.8-0.0015(age-55)+0.05 ratio_Z,0.35,0.97)', body_factors+['smoking'])
    put('3064', 110*x['3063']*np.exp(.12*normal()), '110 FEV1 exp(0.12N) L/min',body_factors+['smoking'])
    multiple = put('1777', rng.binomial(1,.025,n), 'Bernoulli(0.025)', [])
    maternal = put('1787', rng.random(n)<expit(-1+.25*f['smoking']-.2*f['education']), 'Bernoulli(sigmoid(-1+0.25smoking-0.2education))',['smoking','education'])
    put('20022', 3.35-.45*multiple-.15*maternal+.45*normal(), '3.35-0.45multiple_birth-0.15maternal_smoking+0.45N',['smoking','education'])
    put('1677', rng.random(n)<expit(.4+.2*f['education']), 'Bernoulli(sigmoid(0.4+0.2education))',['education'])

    for fid in ['2634','2624','1021','1011','3647','3637','981','971','943','924']:
        categorical(fid,.65*f['activity']-.15*f['age']+.7*normal(),['activity','age'],'0.65activity-0.15age_factor+0.7N ordinal score; numeric order is a declared coding convention')
    for duration,days,base in [('894','884',35),('914','904',22),('874','864',45)]:
        d=put(days,rng.binomial(7,expit(.3+.65*f['activity']),n),'Binomial(7,sigmoid(0.3+0.65activity))',['activity'])
        put(duration,np.where(d>0,base*np.exp(.35*f['activity']+.45*normal()),0),f'if corresponding days>0: {base} exp(0.35activity+0.45N), else 0',['activity'])
    for fid,base,loading in [('1090',1.2,-.1),('1080',2,.1),('1070',2.5,-.5)]:
        put(fid,np.maximum(0,base+loading*f['activity']+.7*normal()),f'round(max(0,{base}+{loading}activity+0.7N))',['activity'])
    for fid in ['1120','1130','2237']:
        categorical(fid,-.2*f['age']+.2*f['education']+normal(),['age','education'],'-0.2age_factor+0.2education+N ordinal score')
    put('1160',np.clip(7.2-.45*f['sleep']+.7*normal(),3,12),'round(clip(7.2-0.45sleep+0.7N,3,12))',['sleep'])
    for fid in ['1170','1180','1190','1200','1220']:
        categorical(fid,.6*f['sleep']+.35*f['anxiety']+.7*normal(),['sleep','anxiety'],'0.6sleep+0.35anxiety+0.7N ordinal score')
    put('1210',np.where(rng.random(n)<expit(-.6+.5*f['adiposity']+.35*male),1,2),'code 1 yes, 2 no; Bernoulli(sigmoid(-0.6+0.5adiposity+0.35male))',['adiposity','sex'])
    smoke = f['smoking']+.55*normal()
    pack = np.where(smoke>0, np.exp(2.2)*np.expm1(.75*np.maximum(smoke,0)),0)
    put('20161',pack,'min(222,exp(2.2)*expm1(0.75W)) if W>0 else 0; W=smoking+0.55N',['smoking'])
    current = (smoke>0)&(rng.random(n)<expit(-.25+.25*smoke-.15*f['age']))
    status = np.where(smoke<=0,0,np.where(current,2,1))
    put('20116',status,'0 never if W<=0; else 2 current with sigmoid(-0.25+0.25W-0.15age_factor), otherwise 1 former',['smoking','age'])
    put('1239',np.where(current,np.where(rng.random(n)<.15,2,1),0),'0 not current; current: 1 on most days with probability 0.85, otherwise 2',['smoking','age'])
    put('1249',np.where(status==0,4,np.where(status==1,1,3)),'declared code convention: never 4, former 1, current 3',['smoking','age'])
    household = put('1259',rng.binomial(2,expit(-1+.35*f['smoking']),n),'Binomial(2,sigmoid(-1+0.35smoking))',['smoking'])
    put('1269',household*np.exp(.5+.7*normal()),'household_smoking_code exp(0.5+0.7N) hours/week',['smoking'])
    put('1279',np.maximum(0,2+.6*f['smoking']+2*normal()),'max(0,2+0.6smoking+2N) hours/week',['smoking'])

    for fid,base in [('1289',3),('1299',2),('1309',2),('1319',1),('1438',12),('1458',5),('1488',3),('1498',2),('1528',5)]:
        put(fid,base*np.exp(.12*f['education']+.12*f['activity']+.45*normal()),f'{base} exp(0.12education+0.12activity+0.45N)',['education','activity'])
    for fid in ['1329','1339','1349','1359','1369','1379','1389','1408','1478','6144','1518']:
        sign = -1 if fid in ['1349','1369','1379','1389','1478'] else 1
        categorical(fid,sign*.2*f['education']+.95*normal(),['education'],f'{sign*.2}education+0.95N ordinal score; 6144 is a single nominal response, not a multi-response bitmask')
    drinks = rng.random(n)<expit(1+.45*f['alcohol'])
    alcstatus = np.where(drinks,2,np.where(rng.random(n)<.35,1,0))
    put('20117',alcstatus,'2 current with sigmoid(1+0.45alcohol), else 1 former with probability 0.35, else 0 never',['alcohol'])
    categorical('1558',-.7*f['alcohol']+.7*normal(),['alcohol'],'-0.7alcohol+0.7N ordinal score; higher numeric frequency code means less frequent drinking')
    # Beverage quantities share a burden, but all are baseline reports preceding outcomes.
    for fid,base in [('1568',3),('1578',2),('1588',3),('1598',1),('1608',.4)]:
        put(fid,drinks*base*np.exp(.7*f['alcohol']+.6*normal()),f'current_drinker * {base} exp(0.7alcohol+0.6N)',['alcohol'])
    put('1618',drinks*(rng.random(n)<expit(.25+.2*f['education'])),'current_drinker Bernoulli(sigmoid(0.25+0.2education))',['alcohol','education'])
    # Unknown code labels are not silently treated as validated UKB semantic labels.
    for fid,base in [('1050',3),('1060',1.5)]:
        put(fid,np.maximum(0,base+.6*f['uv']+.4*f['activity']+.7*normal()),f'round(max(0,{base}+0.6uv+0.4activity+0.7N))',['uv','activity'])
    categorical('1717',-f['pigmentation']+.4*normal(),['pigmentation'],'-pigmentation+0.4N ordinal score; model convention: smaller code is fairer')
    categorical('1727',f['pigmentation']+.5*normal(),['pigmentation'],'pigmentation+0.5N ordinal score; model convention: larger code is harder to tan')
    put('1737',rng.poisson(np.exp(.8+.35*f['uv']+.3*f['pigmentation'])),'Poisson(exp(0.8+0.35uv+0.3pigmentation))',['uv','pigmentation'])
    categorical('1747',-f['pigmentation']+normal(),['pigmentation'],'-pigmentation+N ordinal score; unverified synthetic code-order assumption')
    categorical('1757',.35*f['age']+.3*f['uv']+normal(),['age','uv'],'0.35age_factor+0.3uv+N ordinal score')
    categorical('2267',.3*f['education']+.3*f['pigmentation']+normal(),['education','pigmentation'],'0.3education+0.3pigmentation+N ordinal score')
    put('2277',rng.poisson(np.exp(-.2+.35*f['uv'])),'Poisson(exp(-0.2+0.35uv))',['uv'])
    put('2139',np.minimum(age,18+.7*f['education']+2*normal()),'min(age,18+0.7education+2N)',['age','education'])
    put('2149',1+rng.poisson(np.exp(1+.2*f['alcohol']+.65*normal())),'1+Poisson(exp(1+0.2alcohol+0.65N))',['alcohol'])
    put('2159',rng.binomial(1,.06,n),'Bernoulli(0.06); independent of disease risk',[])
    for live,dead,offset in [('2946','1807',29),('1845','3526',26)]:
        potential = age+offset+3*normal()
        deathage = np.clip((77 if live=='2946' else 81)+10*normal(),float(limits.loc[dead].Recommended_Hard_Min),float(limits.loc[dead].Recommended_Hard_Max))
        deceased = potential>=deathage
        living_age = put(live,potential,f'age+{offset}+3N when potential age < independently sampled death age; otherwise missing',['age'])
        put(dead,deathage,('77' if live=='2946' else '81')+'+10N clipped to dictionary bounds; reported only if deceased',['age'])
        x[live][deceased]=np.nan
        x[dead][~deceased]=np.nan
    put('4501',rng.binomial(1,.08,n),'Bernoulli(0.08); unspecified family cause, no inherited target label',[])
    put('2188',rng.random(n)<expit(-.6+.3*f['age']+.35*f['inflammation']+.2*f['anxiety']+.2*f['renal']),'Bernoulli(sigmoid(-0.6+0.3age_factor+0.35inflammation+0.2anxiety+0.2renal))',['age','inflammation','anxiety','renal'])

    wbc = x['30000']
    baso = .002+.012*expit(normal())
    eos = .005+.055*expit(.3*f['immune']+normal())
    mono = .025+.065*expit(.25*f['inflammation']+normal())
    neut = np.minimum(.4+.3*expit(.3*f['inflammation']+normal()),14.9/wbc)
    lymph = 1-baso-eos-mono-neut
    for count,pct,share,name in [('30160','30220',baso,'basophil'),('30150','30210',eos,'eosinophil'),('30130','30190',mono,'monocyte'),('30140','30200',neut,'neutrophil'),('30120','30180',lymph,'lymphocyte')]:
        put(count,wbc*share,f'white_cell_count * {name}_fraction; fractions partition unity',['inflammation','immune','smoking','sex'])
        put(pct,100*share,f'100*{name}_fraction; see explicit fraction equations in model.py',['inflammation','immune','smoking','sex'])
    rbc,mcv,mchc = x['30010'],x['30040'],x['30060']
    blood_factors=['immune','renal','smoking','alcohol','sex']
    put('30030',rbc*mcv/10,'RBC MCV /10',blood_factors)
    put('30050',mcv*mchc/100,'MCV MCHC /100',blood_factors)
    put('30020',rbc*mcv*mchc/1000,'RBC MCV MCHC /1000',blood_factors)
    put('30070',13+.5*f['immune']+.55*normal(),'13+0.5immune+0.55N',['immune'])
    retpct=put('30240',np.clip(1.3*np.exp(.2*normal()+.1*f['immune']),.15,4),'clip(1.3exp(0.2N+0.1immune),0.15,4)',['immune'])
    put('30250',rbc*retpct/100,'RBC reticulocyte_percentage/100',blood_factors)
    irf=put('30280',.06+.2*expit(normal()),'0.06+0.2sigmoid(N)',[])
    highpct=put('30290',np.maximum(.006,.25*retpct*irf),'max(0.006,0.25 reticulocyte_percentage immature_reticulocyte_fraction)',['immune'])
    put('30300',rbc*highpct/100,'RBC high_light_scatter_reticulocyte_percentage/100',blood_factors)
    put('30260',mcv+12+3*normal(),'MCV+12+3N',['immune','alcohol'])
    put('30270',mcv-8+2*normal(),'MCV-8+2N',['immune','alcohol'])
    put('30170',0,'0; mature peripheral-blood population assumption',[])
    put('30230',0,'0; corresponds to zero nucleated RBC count',[])
    platelets=put('30080',np.clip(240+35*f['inflammation']+30*normal(),70,450),'clip(240+35inflammation+30N,70,450)',['inflammation'])
    mpv=put('30100',np.clip(10-.003*(platelets-240)+.5*normal(),7,12),'clip(10-0.003(platelets-240)+0.5N,7,12)',['inflammation'])
    put('30090',platelets*mpv/10000,'platelet_count*mean_platelet_volume/10000 (percent)',['inflammation'])
    put('30110',17+.3*normal(),'17+0.3N',[])
    put('30690',x['30780']+x['30760']+.45*x['30870'],'LDL + HDL + 0.45 triglycerides, clipped at dictionary total cholesterol bounds',['lipids','adiposity','activity','alcohol','glycemia','sex'])
    bilirubin=put('30840',np.clip(11*np.exp(.3*normal()+.12*f['alcohol']),3,55),'clip(11exp(0.3N+0.12alcohol),3,55)',['alcohol'])
    put('30660',np.clip(.24*bilirubin,1,20.42),'clip(0.24 total_bilirubin,1,20.42)',['alcohol'])
    put('30860',x['30600']+np.clip(27+2*f['immune']+2*normal(),25.3,40),'albumin+clip(27+2immune+2N,25.3,40), then dictionary bounds',['inflammation','immune','renal'])

    expected = list(schema.Field_ID)
    if set(x) != set(expected):
        raise RuntimeError(f'Field mismatch: missing={set(expected)-set(x)}, extra={set(x)-set(expected)}')
    frame = pd.DataFrame({fid:x[fid] for fid in expected})
    if with_metadata:
        return frame,z,male,meta
    return frame,z,male

def sample_target(z, male, disease, rng):
    intercept, male_shift, coefficients = RISKS[disease]
    event_score = intercept + male_shift*male + z @ vector(coefficients) + rng.normal(size=len(z))
    y = (event_score>0).astype(np.int8)
    if disease == 'N14_ENDOMETRIOSIS':
        y[male==1] = 0
    return y

def generate(schema, disease, n, seed):
    """Return 201 predictors followed by a binary column named by disease code."""
    schema = read_schema(schema)
    if disease not in RISKS or n < 1:
        raise ValueError('A supported disease and positive n are required.')
    blocks=[]
    for chunk, start in enumerate(range(0,n,CHUNK_SIZE)):
        rng=np.random.default_rng(disease_seed(seed,disease,chunk))
        frame,z,male=population_block(schema,min(CHUNK_SIZE,n-start),rng)
        frame[disease]=sample_target(z,male,disease,rng)
        blocks.append(frame)
    return pd.concat(blocks,ignore_index=True)
