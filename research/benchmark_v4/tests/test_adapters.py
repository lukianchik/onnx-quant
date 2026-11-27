import importlib.util,unittest
import numpy as np
from graph import Target,sample_rows,conv_rows
from models import conv_numpy,split_indices
from benchmark import quant,PROFILES,fidelity

class AdapterTests(unittest.TestCase):
    def test_gemm_transpose_alpha_beta(self):
        r=np.random.default_rng(2);x=r.normal(size=(5,4)).astype(np.float32);w=r.normal(size=(3,4)).astype(np.float32);b=r.normal(size=3).astype(np.float32)
        y=2*(x@w.T)+.5*b;t=Target('w','n','Gemm','x','y',w,{'transA':1,'transB':1,'alpha':2.,'beta':.5},b)
        rows,n=sample_rows(x.T,y,t,5,r);self.assertEqual(n,5);self.assertEqual(rows[0].shape,(5,4))
        q=np.arange(12,dtype=np.int8).reshape(4,3);s=np.ones(3,np.float32)
        a,b=t.pack([(q,s)]);np.testing.assert_array_equal(a,q.T);np.testing.assert_array_equal(t.stored(t.canonical(a,b))[0],q.T)
    def test_conv_groups_depthwise_and_padding(self):
        r=np.random.default_rng(2);x=r.normal(size=(2,4,5,6)).astype(np.float32)
        for groups,out in [(1,6),(2,6),(4,8)]:
            w=r.normal(size=(out,4//groups,3,3)).astype(np.float32);y=conv_numpy(x,w,groups)
            t=Target('w','n','Conv','x','y',w,{'group':groups,'auto_pad':b'SAME_UPPER'})
            rows=conv_rows(x,y,t,np.arange(2*5*6));self.assertEqual(len(rows),groups)
            self.assertEqual(rows[0].shape,(60,4//groups*9))
            pairs=[(np.rint(m).astype(np.int8),np.ones(m.shape[1],np.float32)) for m in t.matrices()]
            q,s=t.pack(pairs);np.testing.assert_array_equal(q,np.rint(w));np.testing.assert_array_equal(t.stored(t.canonical(q,s))[0],q)
    def test_conv_stride_dilation(self):
        x=np.arange(49,dtype=np.float32).reshape(1,1,7,7);w=np.array([[[[1,2],[3,4]]]],np.float32)
        y=np.array([[[[np.sum(x[0,0,i:i+3:2,j:j+3:2]*w[0,0]) for j in (0,2,4)] for i in (0,2,4)]]],np.float32)
        t=Target('w','n','Conv','x','y',w,{'strides':[2,2],'dilations':[2,2]});rows=conv_rows(x,y,t,np.arange(9));self.assertEqual(rows[0].shape,(9,4))
    def test_matmul_mask(self):
        r=np.random.default_rng(1);x=r.normal(size=(2,3,4)).astype(np.float32);w=r.normal(size=(4,2)).astype(np.float32);mask=np.array([[1,1,0],[1,0,0]])
        t=Target('w','n','MatMul','x','y',w,{})
        rows,n=sample_rows(x,x@w,t,99,r,mask);self.assertEqual(n,3);self.assertEqual(rows[0].shape,(3,4))
    def test_split_and_metrics(self):
        tr,a,b,c=split_indices(100,42,12,6,20);_,a2,b2,c2=split_indices(100,123,12,6,20)
        self.assertEqual(len(set(np.concatenate([tr,a,b,c]))),sum(map(len,[tr,a,b,c])));np.testing.assert_array_equal(c,c2)
        r={'out':np.array([[0.,1.],[1.,0.]])};p={'out':np.array([[0.,1.],[0.,1.]])};m,e=fidelity(r,p,'classification',np.array([1,0]));self.assertEqual(m['changed'],1);self.assertEqual(m['accuracy'],.5);self.assertEqual(m['mse_equal_output_mean'],.5)
    def test_bank_formats(self):
        r=np.random.default_rng(2);x=r.normal(size=(32,12)).astype(np.float32);w=r.normal(size=(12,5)).astype(np.float32);base,_=quant(w,x,'MinMax',PROFILES['smoke']);loss=lambda q,s:np.sum((x.astype(float)@((q*s).astype(float)-w))**2)
        for name in ('Single-3','Paired-3','SDR'):
            (q,s),info=quant(w,x,name,PROFILES['smoke']);self.assertEqual(q.dtype,np.int8);self.assertEqual(s.dtype,np.float32);self.assertLessEqual(loss(q,s),loss(*base)+1e-10)

@unittest.skipUnless(importlib.util.find_spec('onnx') and importlib.util.find_spec('onnxruntime'),'ONNX/ORT недоступны')
class RuntimeTests(unittest.TestCase):
    def test_real_graph(self):
        import tempfile
        from pathlib import Path
        from models import make_case
        from benchmark import session,infer,capture,quant
        from graph import discover,export
        cfg=PROFILES['smoke']
        for name in ('cnn','depthwise_cnn','regression','rnn','autoencoder','multi_output'):
            case=make_case(name,42,cfg,Path('.'));targets,skipped=discover(case.model);self.assertTrue(targets)
            bank={}
            for i,t in enumerate(targets):
                xs,_=capture(case.model,t,case.fit,cfg,42+i);pairs=[quant(w,x,'SDR',cfg)[0] for w,x in zip(t.matrices(),xs)];bank[t.weight]=t.canonical(*t.pack(pairs))
            model=export(case.model,targets,bank);a=infer(session(case.model,1),case.test,cfg.batch);b=infer(session(model,1),case.test,cfg.batch)
            self.assertEqual(set(a),set(b));self.assertEqual(sum(n.op_type=='DequantizeLinear' for n in model.graph.node),len(targets))
            for k in a:self.assertEqual(a[k].shape,b[k].shape);self.assertTrue(np.isfinite(b[k]).all())
if __name__=='__main__':unittest.main()
