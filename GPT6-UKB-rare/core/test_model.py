"""Correctness checks using only small, locally constructed synthetic arrays."""
import hashlib
import inspect
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.special import ndtr

from model import FACTORS, OBSERVATIONS, RISKS, generate, population_block, read_schema, vector
from predict import _conditional_event, predict_risk, SMOKING, SMOKE_CAP

HERE=Path(__file__).resolve().parent

class ModelCorrectness(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.schema=read_schema(HERE/'measurement_dictionary.csv')
        cls.frame,cls.z,cls.sex=population_block(cls.schema,2048,np.random.default_rng(4091))
        cls.names=list(cls.frame.columns)

    def test_input_hashes(self):
        manifest=json.loads((HERE/'input_manifest.json').read_text())
        for name,digest in manifest['permitted_input_files'].items():
            self.assertEqual(hashlib.sha256((HERE/name).read_bytes()).hexdigest(),digest)

    def test_determinism_shape_and_binary_targets(self):
        for disease in RISKS:
            a=generate(self.schema,disease,192,2201)
            b=generate(self.schema,disease,192,2201)
            pd.testing.assert_frame_equal(a,b)
            self.assertEqual(a.shape,(192,202))
            self.assertEqual(list(a.columns),self.names+[disease])
            self.assertTrue(set(a[disease].unique())<={0,1})
        a=generate(self.schema,'E4_HYPERPARA',192,2201)
        b=generate(self.schema,'K11_COELIAC',192,2201)
        self.assertFalse(np.array_equal(a['21022'],b['21022']))
        self.assertNotIn('disease',inspect.signature(population_block).parameters)

    def test_bounds_codes_and_missingness(self):
        for row in self.schema.itertuples():
            values=self.frame[row.Field_ID].dropna().to_numpy()
            self.assertTrue(np.all(values>=row.Recommended_Hard_Min),row.Field_ID)
            self.assertTrue(np.all(values<=row.Recommended_Hard_Max),row.Field_ID)
            if isinstance(row.Recommended_Allowed_Values,str):
                self.assertTrue(set(values)<=set(json.loads(row.Recommended_Allowed_Values)),row.Field_ID)
            if row.Recommended_Data_Type=='bounded_integer_or_count':
                np.testing.assert_array_equal(values,np.round(values))
        self.assertEqual(set(self.frame.columns[self.frame.isna().any()]),{'2946','1807','1845','3526'})
        for live,dead in [('2946','1807'),('1845','3526')]:
            self.assertTrue(np.all(self.frame[live].isna()!=self.frame[dead].isna()))

    def test_algebra_and_physically_ordered_measurements(self):
        x=self.frame
        np.testing.assert_allclose(x['21002'],x['21001']*(x['50']/100)**2,rtol=1e-12)
        np.testing.assert_allclose(x['21002'],x['23100']+x['23101'],rtol=1e-12)
        np.testing.assert_allclose(x['23099'],100*x['23100']/x['21002'],rtol=1e-12)
        for pct,fm,ffm in [('23115','23116','23117'),('23111','23112','23113'),('23123','23124','23125'),('23119','23120','23121'),('23127','23128','23129')]:
            np.testing.assert_allclose(x[pct],100*x[fm]/(x[fm]+x[ffm]),rtol=1e-12)
        np.testing.assert_allclose(x['30000'],x[['30160','30150','30130','30140','30120']].sum(axis=1),rtol=1e-12)
        np.testing.assert_allclose(x[['30220','30210','30190','30200','30180']].sum(axis=1),100,rtol=1e-12)
        np.testing.assert_allclose(x['30020'],x['30010']*x['30040']*x['30060']/1000,rtol=1e-12)
        np.testing.assert_allclose(x['30030'],x['30010']*x['30040']/10,rtol=1e-12)
        np.testing.assert_allclose(x['30090'],x['30080']*x['30100']/10000,rtol=1e-12)
        self.assertTrue(np.all(x['4080']>x['4079']))
        self.assertTrue(np.all(x['3062']>=x['3063']))
        self.assertTrue(np.all(x['30840']>=x['30660']))
        self.assertTrue(np.all(x.loc[x['20116']==0,'20161']==0))

    def test_all_missing_is_integrated_prior(self):
        x=np.full((3,201),np.nan)
        for disease,(intercept,shift,coefficients) in RISKS.items():
            denominator=np.sqrt(1+sum(v*v for v in coefficients.values()))
            expected=.5*ndtr(intercept/denominator)
            if disease!='N14_ENDOMETRIOSIS':
                expected+=.5*ndtr((intercept+shift)/denominator)
            np.testing.assert_allclose(predict_risk(x,self.names,disease),expected,atol=1e-13)

    def test_batch_and_permutation_invariance(self):
        rng=np.random.default_rng(74)
        x=self.frame.iloc[:91].to_numpy().copy()
        x[rng.random(x.shape)<.15]=np.nan
        permutation=rng.permutation(len(x))
        for disease in RISKS:
            p=predict_risk(x,self.names,disease)
            chunks=np.concatenate([predict_risk(x[i:i+13],self.names,disease) for i in range(0,len(x),13)])
            np.testing.assert_allclose(p,chunks,atol=2e-13,rtol=2e-13)
            np.testing.assert_allclose(p[permutation],predict_risk(x[permutation],self.names,disease),atol=2e-13,rtol=2e-13)
            self.assertTrue(np.all(np.isfinite(p)&(p>=0)&(p<=1)))

    def test_censored_smoking_quadrature(self):
        covariance=np.eye(len(FACTORS))
        mean=np.zeros((1,len(FACTORS)))
        disease='I9_ABAORTANEUR'
        a,shift,coeff=RISKS[disease]
        b=vector(coeff)
        variance=1+np.dot(b,b)
        wvar=1+.55**2
        cross=b[SMOKING]
        for censor in [-1,1]:
            threshold=(0 if censor==-1 else SMOKE_CAP)/np.sqrt(wvar)
            low,high=(-np.inf,threshold) if censor==-1 else (threshold,np.inf)
            evidence=ndtr(threshold) if censor==-1 else ndtr(-threshold)
            integrand=lambda t: ndtr((a+cross/np.sqrt(wvar)*t)/np.sqrt(variance-cross**2/wvar))*np.exp(-t*t/2)/np.sqrt(2*np.pi)
            exact=quad(integrand,low,high,epsabs=1e-12)[0]/evidence
            actual,_=_conditional_event(mean,covariance,disease,0,censor)
            np.testing.assert_allclose(actual,exact,atol=8e-5,rtol=0)

    def test_sex_gate_and_risk_monotonicity(self):
        x=np.full((2,201),np.nan)
        x[:,self.names.index('31')]=[0,1]
        p=predict_risk(x,self.names,'N14_ENDOMETRIOSIS')
        self.assertGreater(p[0],0)
        self.assertEqual(p[1],0)
        x[:]=np.nan
        x[:,self.names.index('30680')]=[2.2,2.7]
        p=predict_risk(x,self.names,'E4_HYPERPARA')
        self.assertGreater(p[1],p[0])

    def test_no_outcome_columns_or_data_reads(self):
        with self.assertRaises(ValueError):
            predict_risk(np.ones((3,202)),self.names+['E4_HYPERPARA'],'E4_HYPERPARA')
        real_read=pd.read_csv
        observed=[]
        def dictionary_only(path,*args,**kwargs):
            self.assertEqual(Path(path).resolve(),HERE/'measurement_dictionary.csv')
            observed.append(str(path))
            return real_read(path,*args,**kwargs)
        with patch('pandas.read_csv',dictionary_only),patch('numpy.load',side_effect=AssertionError('Unexpected model/data loading')):
            predict_risk(self.frame.iloc[:5].to_numpy(),self.names,'K11_COELIAC')
        self.assertTrue(observed)

if __name__=='__main__':
    unittest.main(verbosity=2)
