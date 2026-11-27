"""Глобальная политика v4: только уже построенные варианты, отдельный Search."""
import numpy as np
from quantlab.weight_only.robust import BASE, choose
from quantlab.weight_only.fast import guard_decision
from quantlab.weight_only import logit_int8

MAX_ACTIVE = 4
GENERIC_CAP = 9
# Дополнительный банк, не общий лимит RAM процесса.
BANK_BYTES = 64 * 1024**2


def split_quality(feeds):
    """Разделить независимые пакеты приблизительно 57/21/11/11, минимум 6."""
    n = len(feeds)
    if n < 6:
        raise ValueError('Quality: нужно минимум 6 независимых пакетов калибровки')
    a = min(n-3, max(2, n*128//224))
    b = min(n-2, a+max(1,n*48//224))
    c = min(n-1, b+max(1,n*24//224))
    return feeds[:a], feeds[a:b], feeds[b:c], feeds[c:]


def priority(losses):
    """Ширина интервала относительного выигрыша из search_priority v4."""
    _, info = choose(losses)
    b = losses[BASE]
    if not b.dev:
        return 0.
    widths = []
    for k,l in losses.items():
        if k != BASE:
            center = l.dev-b.dev
            half = info['upper'][k]-center
            if center-half < 0:
                widths.append(min(1.,half/b.dev))
    return max(widths,default=0.)


class CandidatePool:
    """Хранить до четырёх приоритетных банков, ограничивая дополнительные payload."""
    def __init__(self, cap=BANK_BYTES):
        self.cap = cap
        self.entries = {}
        self.dropped = 0

    @property
    def bytes(self):
        """Размер удерживаемых кодов и масштабов (без накладных расходов)."""
        return sum(e[2] for e in self.entries.values())

    def add(self, key, variants, losses, local):
        """Сохранить только реально построенные отличающиеся варианты."""
        options = {local:variants[local]}
        for name,pair in variants.items():
            if not any(all(np.array_equal(a,b) for a,b in zip(pair,p)) for p in options.values()):
                options[name] = pair
        if len(options) < 2:
            return
        size = sum(q.nbytes+s.nbytes for q,s in options.values())
        self.entries[key] = (priority(losses),options,size)
        while len(self.entries)>MAX_ACTIVE or self.bytes>self.cap:
            worst = sorted(self.entries,key=lambda k:(-self.entries[k][0],k))[-1]
            del self.entries[worst]
            self.dropped += 1

    def options(self):
        """Вернуть банки в порядке оригинального приоритета."""
        return {k:self.entries[k][1] for k in sorted(self.entries,key=lambda k:(-self.entries[k][0],k))}


def search(local, options, evaluate, *, classification=False, infer=None):
    """Поиск v4: generic greedy либо оригинальный V2 с ограничением вызовов.

    evaluate принимает выбор имён кандидатов; infer принимает полные веса либо None.
    """
    if not options:
        return dict(local), {'kind':'none','calls':0,'cap':0,'active':[]}
    if len(options)>MAX_ACTIVE:
        raise ValueError('Слишком много активных операторов')
    calls = 0
    if classification:
        cap = 5+sum(len(v)-1 for v in options.values())
        def callback(weights, feeds):
            """Учитывать каждый реальный вызов V2, включая FP32."""
            nonlocal calls
            if calls>=cap:
                raise ValueError('V2 budget')
            calls+=1
            return infer(weights,feeds)
        bank = {k:list(v.values()) for k,v in options.items()}
        try:
            idx = logit_int8.select_int8(bank,callback,'fit','search',width=4)
            result = {**local, **{k:list(options[k])[i] for k,i in idx.items()}}
            return result, {'kind':'original_V2','calls':calls,'cap':cap,'active':list(options)}
        except Exception as exc:
            return dict(local), {'kind':'original_V2','calls':calls,'cap':cap,'active':list(options),'error':str(exc)[:300]}
    cap=GENERIC_CAP
    cur=dict(local)
    try:
        calls+=1
        old=evaluate(cur)
        for k, variants in options.items():
            for name in variants:
                if calls>=cap:
                    break
                if name==cur[k]:
                    continue
                trial={**cur,k:name}
                calls+=1
                try:
                    new=evaluate(trial)
                except Exception:
                    continue
                if guard_decision(old,new):
                    cur,old=trial,new
    except Exception as exc:
        return dict(local), {'kind':'exact_multioutput_greedy','calls':calls,'cap':cap,'active':list(options),'error':str(exc)[:300]}
    return cur, {'kind':'exact_multioutput_greedy','calls':calls,'cap':cap,'active':list(options)}


def check_score(baseline, candidate):
    """Среднее отношение MSE к MinMax; недопустимый кандидат не оценивается."""
    if not guard_decision(baseline,candidate):
        return None
    old,new=baseline['mse_by_output'],candidate['mse_by_output']
    return float(np.mean([new[k]/v if v else 0. for k,v in old.items()]))
