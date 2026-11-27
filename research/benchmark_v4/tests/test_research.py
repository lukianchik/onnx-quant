import unittest
from dataclasses import replace
import numpy as np
from benchmark import quant,PROFILES,fidelity
from pretrained import indices,IMAGENETTE
from graph import Target,conv_rows

class ResearchTests(unittest.TestCase):
    def test_caps_leave_other_channels_at_minmax(self):
        rng=np.random.default_rng(52);w=rng.normal(size=(16,13)).astype(np.float32);x=rng.normal(size=(48,16)).astype(np.float32);x[:,1]=x[:,0]
        cfg=replace(PROFILES['smoke'],channel_cap=5,sdr_work=10**9)
        base,_=quant(w,x,'MinMax',cfg);chosen=np.linspace(0,12,5,dtype=int);other=np.setdiff1d(np.arange(13),chosen)
        for name in ('Single-3','Paired-3','SDR'):
            (q,s),info=quant(w,x,name,cfg);np.testing.assert_array_equal(q[:,other],base[0][:,other]);np.testing.assert_array_equal(s[other],base[1][other]);self.assertEqual(info['selected_channels'],chosen.tolist());self.assertEqual(q.nbytes+s.nbytes,w.size+4*w.shape[1])
    def test_research_splits(self):
        a=indices(1000,192,42);b=indices(1000,192,123)
        self.assertEqual(len(set(a)),192);self.assertFalse(set(a[:128])&set(a[128:]));self.assertFalse(np.array_equal(a,b));np.testing.assert_array_equal(indices(500,100,731),indices(500,100,731));self.assertEqual(len(set(IMAGENETTE.values())),10)
    def test_top5_full_class_space(self):
        y=np.array([1,5]);r=np.array([[0,9,8,7,6,5],[0,1,2,3,4,5]],np.float32);p=r.copy();m,_=fidelity({'o':r},{'o':p},'classification',y);self.assertEqual(m['top5_accuracy'],1);self.assertEqual(m['accuracy'],1)
    def test_conv3d_selected_patches(self):
        x=np.arange(64,dtype=np.float32).reshape(1,1,4,4,4);w=np.ones((1,1,2,2,2),np.float32);y=np.array([x[0,0,i:i+2,j:j+2,k:k+2].sum() for i in range(3) for j in range(3) for k in range(3)],np.float32).reshape(1,1,3,3,3);t=Target('w','n','Conv','x','y',w,{})
        a=conv_rows(x,y,t,np.array([0,12,26]));self.assertEqual(a[0].shape,(3,8));np.testing.assert_array_equal(a[0].sum(1),y.ravel()[[0,12,26]])
if __name__=='__main__':unittest.main()
