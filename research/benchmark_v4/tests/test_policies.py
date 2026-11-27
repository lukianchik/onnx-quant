import unittest
import numpy as np
from adaptive_policies import policy, check, aggregate, Loss, Portfolio
class Policies(unittest.TestCase):
    def test_progressive_early_stop(self):
        l={'minmax':Loss(1,1,1),'single':Loss(.5,.5,1),'paired':Loss(.2,.2,1),'sdr':Loss(.1,.1,1)}
        self.assertEqual(policy('Progressive',l,np.zeros(4),Portfolio([]))[:2],('single',('single',)))
        l['single']=Loss(.1,1.1,1)
        self.assertEqual(policy('Progressive',l,np.zeros(4),Portfolio([]))[:2],('paired',('single','paired')))
    def test_global_check(self):
        r={'a':np.zeros(2),'b':np.zeros(2)};b={k:np.ones(2) for k in r}
        self.assertFalse(check(r,b,{'a':np.zeros(2),'b':np.full(2,2.)})[0])
        self.assertTrue(check(r,b,{'a':np.zeros(2),'b':np.ones(2)})[0])
    def test_group_energy(self):
        g=dict(channels=2,fit_rows=3,dev_rows=2,shape=[4,2],fit_mse=.5,dev_mse=.6,weight_mse=.25,profile_sampled_features=4,stable_rank_sample=2,mean_abs_corr_sample=.1,weight_abs_max=1,weight_rms=.5)
        l,f=aggregate({'groups':[g,g]});self.assertAlmostEqual(l.energy,1.);self.assertAlmostEqual(l.dev,.6)
    def test_astra_outside(self):
        l={'minmax':Loss(1,1,1),'single':Loss(.1,.1,1)}
        self.assertEqual(policy('Astra-Adaptive',l,np.zeros(4),Portfolio([]))[:2],('minmax',()))
