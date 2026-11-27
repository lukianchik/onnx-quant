"""Проверки глобальной политики, независимых пакетов и ограничений."""
import numpy as np
import pytest
from quantlab.weight_only import global_search as g
from quantlab.weight_only.robust import Loss
from quantlab.weight_only import progressive as p


def measure(a,b):
    """Подготовить две численные ошибки."""
    return {'mse_by_output':{'a':a,'b':b}}


def pair(code):
    """Создать различимый корректный INT8 payload."""
    return np.full((3,2),code,np.int8),np.ones(2,np.float32)


def test_split():
    """Раздельные пакеты и отсутствие повторного использования."""
    for n in (6,7,16,224):
        groups=g.split_quality(list(range(n)))
        assert all(groups) and sum(groups,[])==list(range(n))
    with pytest.raises(ValueError,match='6'):
        g.split_quality(list(range(5)))


def test_greedy_cap_and_all_outputs():
    """Проверить предел девяти вызовов и отклонение ухудшения второго выхода."""
    options={str(k):{str(i):pair(i) for i in range(4)} for k in range(4)}
    local=dict.fromkeys(options,'0')
    seen=[]
    def evaluate(c):
        """Первый выход улучшается за счёт второго: запрещено."""
        seen.append(c)
        x=sum(map(int,c.values()))
        return measure(100-x,1+x)
    selected,info=g.search(local,options,evaluate)
    assert selected==local and info['calls']==len(seen)==9
    assert len(info['active'])==4


def test_greedy_accept():
    """Выбрать реально улучшающее сочетание."""
    local={'a':'minmax','b':'minmax'}
    opts={k:{'minmax':pair(0),'single':pair(1)} for k in local}
    selected,info=g.search(local,opts,lambda c:measure(3-sum(v=='single' for v in c.values()),1))
    assert selected==dict.fromkeys(local,'single') and info['calls']==3


def test_v2_real():
    """Запустить оригинальный V2 на контролируемых логитах и учесть вызовы."""
    local={'a':'minmax','fixed':'paired'}
    opts={'a':{'minmax':pair(0),'single':pair(1)}}
    calls=[]
    def infer(weights,split):
        """Альтернативный кандидат точно воспроизводит FP32."""
        calls.append(split)
        return np.array([[2.,0.],[0.,2.]]) if weights is None or weights['a'][0][0,0]==1 else np.array([[1.,0.],[0.,1.]])
    chosen,info=g.search(local,opts,None,classification=True,infer=infer)
    assert chosen=={'a':'single','fixed':'paired'}
    assert len(calls)==info['calls']==info['cap']==6
    assert set(calls)=={'fit','search'}


def test_v2_error_falls_back():
    """Некорректные логиты не нарушают исходный локальный выбор."""
    local={'a':'single'}
    opts={'a':{'single':pair(1),'minmax':pair(0)}}
    selected,info=g.search(local,opts,None,classification=True,infer=lambda *a:np.zeros(3))
    assert selected==local and info['calls']==1 and 'error' in info


def test_pool_limits_and_duplicates():
    """Ограничить банк четырьмя операторами, удалить одинаковые веса."""
    pool=g.CandidatePool()
    losses={'minmax':Loss(1,1,1),'single':Loss(.5,.6,1)}
    for k in reversed(range(6)):
        pool.add(str(k),{'minmax':pair(0),'single':pair(1)},losses,'single')
    assert list(pool.options())==['0','1','2','3'] and pool.dropped==2
    pool.add('same',{'minmax':pair(0),'single':pair(0)},losses,'single')
    assert 'same' not in pool.options()
    small=g.CandidatePool(cap=1)
    small.add('a',{'minmax':pair(0),'single':pair(1)},losses,'single')
    assert not small.options() and small.bytes==0


def test_guard_score():
    """Выбирать только улучшившие все выходы варианты, включая нулевой MSE."""
    assert g.check_score(measure(1,0),measure(.5,0))==.25
    assert g.check_score(measure(1,0),measure(.5,.01)) is None
    assert g.check_score(measure(1,0),measure(1,0)) is None


def test_retain_only_built(monkeypatch):
    """Quality не строит дополнительных кандидатов после ранней остановки."""
    rng=np.random.default_rng(42)
    w=[rng.normal(size=(8,4)).astype('float32')]
    x=[rng.normal(size=(16,8)).astype('float32')]
    d=[rng.normal(size=(12,8)).astype('float32')]
    losses=iter([Loss(1,1,1),Loss(.5,.5,1)])
    monkeypatch.setattr(p,'aggregate',lambda *a:next(losses))
    retained=[]
    _,_,_,info=p.select_progressive(w,x,d,100000,retain=lambda name,*a:retained.append(name))
    assert retained==['minmax','single'] and info['stop_reason']=='robust_target_reached'
