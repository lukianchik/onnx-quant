"""Проверки математической границы, разграничения этапов и лимитов инференса."""

import unittest
import numpy as np
from adaptive_selector import (BASE, Loss, Portfolio, choose, confirm_outputs, finalize,
                               global_plan, paired_upper, profile, search_priority, select)


class SelectorTests(unittest.TestCase):
    def test_bound_covers_stress_set(self):
        rng=np.random.default_rng(73)
        x,y=rng.normal(size=(2,8,3))
        hf,hd=x.T@x/8,y.T@y/8
        ea,eb=rng.normal(size=(2,3,2))
        def loss(e):
            return Loss(*(np.trace(e.T@h@e)/2 for h in (hf,hd)),np.sum(e**2)/2)
        rho=.13
        l={BASE:loss(eb),"single":loss(ea)}
        _,diag=choose(l,rho)
        direction=np.linalg.eigh(ea@ea.T-eb@eb.T)
        delta=(direction[1]*np.sign(direction[0]))@direction[1].T*rho
        for t in (-1,0,.5,1):
            h=hd+t*(hd-hf)+delta
            diff=(np.trace(ea.T@h@ea)-np.trace(eb.T@h@eb))/2
            self.assertLessEqual(diff,diag["upper"]["single"]+1e-12)
        self.assertEqual(choose(l,rho,confidence={"single":1.})[0],BASE)
        self.assertEqual(paired_upper([-1.,-1.],0.,1),-1.)
        self.assertGreaterEqual(search_priority(l),0.)

    def test_only_reserved_builder_runs(self):
        w=np.array([[.25],[0],[127]],np.float32)
        x=np.array([[4,5,0],[0,0,1]],np.float32)
        f,diag=profile(w,x)
        self.assertEqual(diag["sampled"],3)
        prior=Portfolio([(f[None,:],[[.1,1.,.2]])])
        def variant(values):
            return np.array(values,np.int8)[:,None],np.ones(1,np.float32)
        base=lambda w,x:variant([0,0,127])
        calls=[]
        builders={"single":lambda *_:self.fail(),"sdr":lambda *_:self.fail()}
        def paired(*args):
            calls.append(1)
            return variant([-1,1,127])
        builders["paired"]=paired
        for budget,expected in ((0.,BASE),(1.,"paired")):
            name,_,bank,info=select(w,x,x.copy(),base,builders,prior,
                                    dict.fromkeys(builders,1.),budget)
            self.assertEqual(name,expected)
            self.assertEqual(len(bank),1+int(budget))
        self.assertEqual(calls,[1])
        builders["paired"]=lambda *_:variant([-128,1,127])
        self.assertEqual(select(w,x,x.copy(),base,builders,prior,{"paired":1})[0],BASE)
        self.assertEqual(prior.plan(f+50,{"paired":1},1)[0],())
        self.assertEqual(Portfolio([]).plan(f,{"paired":1},1)[0],())

    def test_global_budget_and_multioutput_rollback(self):
        local={"a":"single"}; options={"a":[BASE,"single","sdr"]}
        def infer(c):
            values=(0.,0.) if c is None else {BASE:(2.,2.),"single":(1.,3.),"sdr":(.5,1.)}[c["a"]]
            return {k:np.array([v]) for k,v in zip(("signal","aux"),values)}
        self.assertEqual(finalize(local,options,{},infer,infer,2)[1]["calls"],0)
        self.assertEqual(finalize(local,options,{},infer,infer,3)[0]["a"],BASE)
        probes=[]
        def propose(anchor,active,probe):
            for c in (None,anchor,{"a":BASE},{"a":"sdr"}):
                probes.append(probe(c))
            return {"a":"sdr"}
        name,diag=finalize(local,options,{"a":1},infer,infer,8,propose)
        self.assertEqual((name["a"],diag["calls"],len(probes)),("sdr",8,4))
        def excessive(anchor,active,probe):
            for _ in range(20):
                probe(anchor)
        self.assertEqual(finalize(local,options,{"a":1},infer,infer,8,excessive)[1]["calls"],7)
        self.assertEqual(global_plan(options,{"a":1},7),())
        z=np.array([2**60],np.uint64)
        a,b=confirm_outputs({"v":z},{"v":z+1},{"v":z})
        self.assertEqual((a[0],b[0]),(1.,0.))


if __name__=="__main__":
    unittest.main()