"""Single-Robust v4: ограниченное построение одного кандидата на оператор."""
import numpy as np
from quantlab.weight_only.activation_int8 import quantize_int8
from quantlab.weight_only.minmax import minmax
from quantlab.weight_only.robust import BASE, Loss, choose, _loss

CHANNEL_CAP = 128
GROUP_WORK = 500_000_000
TOTAL_WORK = 1_000_000_000


def split_feeds(feeds):
    """Разделить целые входные пакеты без повторного использования в проверке."""
    n = len(feeds)
    if n < 5:
        raise ValueError('Нужно минимум 5 независимых пакетов калибровки')
    a = max(2, n*3//5)
    b = min(n-1, a+max(1,n//5))
    return feeds[:a], feeds[a:b], feeds[b:]


def select_single(matrices, fit, dev, remaining):
    """Сохранить формулу групповой агрегации и выбора из v4."""
    base, trial, records = [], [], {BASE: [], 'single': []}
    work = 0
    for w,x,d in zip(matrices, fit, dev, strict=True):
        q,s = minmax(w)
        base.append((q,s))
        idx = np.linspace(0,w.shape[1]-1,min(CHANNEL_CAP,w.shape[1]),dtype=int)
        cost = int(len(x))*int(w.shape[0])*len(idx)
        qq,ss = q.copy(),s.copy()
        if cost <= GROUP_WORK and cost <= remaining-work:
            work += cost
            qq[:,idx],ss[idx] = quantize_int8(x,w[:,idx],sweeps=3,paired=False)
        trial.append((qq,ss))
        for name,pair in ((BASE,(q,s)),('single',(qq,ss))):
            records[name].append((_loss(w,pair,x,d),w.shape[1],len(x),len(d)))
    losses = {}
    for name, rows in records.items():
        losses[name] = Loss(*(float(np.average([getattr(r[0],field) for r in rows],
            weights=[r[1]*(r[k] if k else 1) for r in rows]))
            for field,k in [('fit',2),('dev',3),('energy',0)]))
    winner,diag = choose(losses)
    return base, trial if winner == 'single' else base, work, {
        'method': 'Single-3' if winner == 'single' else 'MinMax',
        'reason': diag['reason'] if work else 'work_budget',
        'robust_upper': {k:float(v) for k,v in diag['upper'].items()},
        'radius': float(diag['radius']), 'work': work,
    }


def guard_decision(old, new):
    """Ни один численный выход не хуже, хотя бы один строго лучше."""
    a,b = old['mse_by_output'],new['mse_by_output']
    return bool(a and a.keys()==b.keys() and all(b[k]<=a[k] for k in a)
                and any(b[k]<a[k] for k in a))
