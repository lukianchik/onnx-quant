"""Проверки ленивой политики, бюджета и совпадения с исследованием."""
import numpy as np
import pytest
from quantlab.weight_only import progressive as p
from quantlab.weight_only.robust import Loss
from quantlab.weight_only.fast import select_single


def inputs():
    """Создать небольшую воспроизводимую задачу."""
    rng = np.random.default_rng(42)
    return ([rng.normal(size=(8,4)).astype(np.float32)],
            [rng.normal(size=(16,8)).astype(np.float32)],
            [rng.normal(size=(12,8)).astype(np.float32)])


@pytest.mark.parametrize('losses,expected,stop', [
    ([Loss(1,1,1),Loss(.5,.5,1)],['single'],'robust_target_reached'),
    ([Loss(1,1,1),Loss(.9,.9,1),Loss(.6,.6,1)],['single','paired'],'robust_target_reached'),
    ([Loss(1,1,1),Loss(.9,.9,1),Loss(.8,.8,1),Loss(.6,.6,1)],['single','paired','sdr'],'robust_target_reached'),
    ([Loss(1,1,1),Loss(.9,.9,1),Loss(.8,.8,1),Loss(.85,.85,1)],['single','paired','sdr'],'all_candidates_considered'),
    ([Loss(0,0,0)],[],'zero_baseline_dev'),
])
def test_lazy(monkeypatch,losses,expected,stop):
    """Доказать отсутствие вызовов дорогостоящих методов после остановки."""
    calls=[]
    candidates={}
    it=iter(losses)
    monkeypatch.setattr(p,'aggregate',lambda *args:next(it))
    def build(method,matrices,fit,base,budget):
        """Подменить вычисление кандидата управляемой заглушкой."""
        calls.append(method)
        assert budget.reserve(1)
        candidates[method]=[(q.copy(),s.copy()) for q,s in base]
        return candidates[method],[]
    monkeypatch.setattr(p,'build',build)
    base,best,work,info=p.select_progressive(*inputs(),100)
    assert calls==expected and info['stop_reason']==stop and work==len(expected)
    if expected:
        winner=next(k for k,v in p.METHODS.items() if v==info['method'])
        assert best is candidates[winner]
    else:
        assert best is base


def test_no_budget(monkeypatch):
    """Не выполнять ни одного оптимизатора при исчерпанном бюджете."""
    def forbidden(*args,**kwargs):
        """Обнаружить незапланированный запуск."""
        pytest.fail('optimizer must not run')
    monkeypatch.setattr(p,'build',forbidden)
    base,best,work,info=p.select_progressive(*inputs(),0)
    assert best is base and work==0 and info['stop_reason']=='work_budget'


def test_single_equivalent():
    """Первый этап совпадает с действующей реализацией Fast."""
    w,x,d=inputs()
    cost=len(x[0])*w[0].size
    b1,p1,_,r1=select_single(w,x,d,cost)
    b2,p2,used,r2=p.select_progressive(w,x,d,cost)
    assert used==cost and r1['method']==r2['method']
    for a,b in zip(b1+p1,b2+p2,strict=True):
        for c,e in zip(a,b,strict=True):
            np.testing.assert_array_equal(c,e)


def test_real_candidates():
    """Проверить форматы и фактический учёт Single, Paired и SDR."""
    w,x,d=inputs()
    base=[p.minmax(v) for v in w]
    budget=p.WorkBudget(10_000_000)
    for method in ('single','paired','sdr'):
        before=budget.used
        pairs,groups=p.build(method,w,x,base,budget)
        assert budget.used>before and budget.used<=budget.limit
        assert sum(g['work'] for g in groups)==budget.used-before
        assert pairs[0][0].dtype==np.int8 and pairs[0][1].dtype==np.float32
        assert np.isfinite(p.aggregate(w,pairs,x,d).dev)
    assert not budget.reserve(budget.remaining+1)


def test_real_progressive():
    """Проверить выбор и конечность метрик без подмен алгоритмов."""
    b,q,used,diag=p.select_progressive(*inputs(),10_000_000)
    assert 0<used<=10_000_000
    assert diag['method'] in p.METHODS.values()
    assert sum(s['work'] for s in diag['steps'])==used
    assert np.isfinite(diag['robust_gain_fraction'])
