"""Проверки робастного выбора, бюджета и разделения данных Fast."""
import numpy as np
import pytest
from quantlab.weight_only.fast import select_single, split_feeds, guard_decision
from quantlab.weight_only.minmax import minmax
from quantlab.weight_only.robust import Loss, choose


def test_split():
    """Не допустить повторного использования пакета в разных ролях."""
    a,b,c = split_feeds(list(range(16)))
    assert a+b+c == list(range(16)) and all((a,b,c))
    with pytest.raises(ValueError):
        split_feeds([1,2,3,4])


def test_budget():
    """Нулевой бюджет возвращает исходное квантование без Single."""
    rng = np.random.default_rng(42)
    w = rng.normal(size=(8,4)).astype(np.float32)
    x = rng.normal(size=(12,8)).astype(np.float32)
    b,p,work,d = select_single([w],[x],[x.copy()],0)
    assert work == 0 and d['method']=='MinMax'
    for a,c in zip(p[0],minmax(w)):
        np.testing.assert_array_equal(a,c)


def test_single_and_guard():
    """Проверить формат результата и запрет ухудшения одного выхода."""
    rng = np.random.default_rng(12)
    w = rng.normal(size=(8,4)).astype(np.float32)
    x = rng.normal(size=(32,8)).astype(np.float32)
    d = rng.normal(size=(32,8)).astype(np.float32)
    b,p,work,info = select_single([w],[x],[d],100000)
    assert work == 32*8*4 and p[0][0].dtype == np.int8
    assert p[0][1].dtype == np.float32
    assert choose({'minmax':Loss(1,1,1),'single':Loss(.5,.5,.5)})[0] == 'single'
    old = {'mse_by_output':{'a':1.,'b':2.}}
    assert guard_decision(old,{'mse_by_output':{'a':.8,'b':2.}})
    assert not guard_decision(old,{'mse_by_output':{'a':.1,'b':2.01}})
    assert not guard_decision(old,old)
