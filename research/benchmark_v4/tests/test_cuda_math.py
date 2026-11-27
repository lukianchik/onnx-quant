import unittest,os
from unittest.mock import patch
import numpy as np
import torch
from algorithms import activation_int8 as v1,spectral_rounding as sr
from cuda_math import v1_torch,sdr_torch

class TorchMath(unittest.TestCase):
    def devices(self):return ['cpu']+(['cuda'] if torch.cuda.is_available() else [])
    def test_v1_against_original(self):
        for seed in (11,23,47):
            rng=np.random.default_rng(seed);x=rng.normal(size=(40,24)).astype('f');w=rng.normal(size=(24,35)).astype('f');x[:,3]=0;w[:,0]=0
            for paired in (False,True):
                if paired:q,s=v1.quantize_int8(x,w)
                else:
                    with patch.object(v1,'_pairs',return_value=[]):q,s=v1.quantize_int8(x,w)
                for device in self.devices():
                    qq,ss=v1_torch(x,w,paired,device=device)
                    np.testing.assert_array_equal(qq,q);np.testing.assert_allclose(ss,s,rtol=1e-6,atol=0)
    def test_sdr_against_original(self):
        for seed in (11,23,47):
            rng=np.random.default_rng(seed);x=rng.normal(size=(40,24)).astype('f');w=rng.normal(size=(24,9)).astype('f');x[:,3]=0;w[:,0]=0
            q=sr.quantize(w,x);base=sr.quantize(w,None)
            for device in self.devices():
                (a,b),info=sdr_torch(x,w,device=device)
                np.testing.assert_array_equal(b,q.scales)
                np.testing.assert_array_equal(a,q.codes)
                self.assertEqual(info['accepted'],q.diagnostics['accepted'])
                loss=lambda z,s:np.sum((x.astype(float)@(w-(z*s)))**2,axis=0)
                self.assertTrue(np.all(loss(a,b)<=loss(base.codes,base.scales)))
    def test_sdr_budget(self):
        x=np.eye(3,dtype='f');w=x.copy();(a,b),i=sdr_torch(x,w,sr.Config(work_limit=0),device='cpu');self.assertEqual(i['reason'],'budget_or_width')

if __name__=='__main__':unittest.main()
