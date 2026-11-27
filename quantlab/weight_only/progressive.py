"""Ленивое построение кандидатов Progressive по замороженной политике v4."""
from dataclasses import dataclass
import numpy as np
from quantlab.weight_only.activation_int8 import quantize_int8
from quantlab.weight_only.fast import CHANNEL_CAP, GROUP_WORK, TOTAL_WORK
from quantlab.weight_only.minmax import minmax
from quantlab.weight_only.robust import BASE, Loss, choose, _loss
from quantlab.weight_only import spectral_rounding as sdr

TARGET = .25
METHODS = {'minmax': 'MinMax', 'single': 'Single-3', 'paired': 'Paired-3', 'sdr': 'SDR'}
BUDGETS = (250_000_000, TOTAL_WORK, 4_000_000_000)


@dataclass
class WorkBudget:
    """Учёт оценочной работы поиска; не лимит времени или всей памяти процесса."""
    limit: int
    used: int = 0

    @property
    def remaining(self):
        """Вернуть остаток бюджета."""
        return self.limit - self.used

    def reserve(self, cost):
        """Зарезервировать работу до запуска алгоритма."""
        if cost < 0 or cost > self.remaining:
            return False
        self.used += cost
        return True


def aggregate(matrices, pairs, fit, dev):
    """Объединить ошибки групп с весами из политики v4."""
    rows = [(_loss(w,p,x,d),w.shape[1],len(x),len(d))
            for w,p,x,d in zip(matrices,pairs,fit,dev,strict=True)]
    return Loss(*(float(np.average([getattr(r[0],field) for r in rows],
                weights=[r[1]*(r[k] if k else 1) for r in rows]))
                for field,k in [('fit',2),('dev',3),('energy',0)]))


def build(method, matrices, fit, base, budget):
    """Построить один вариант из MinMax, не изменяя предыдущих кандидатов."""
    result, groups = [], []
    for w,x,(q,s) in zip(matrices,fit,base,strict=True):
        idx = np.linspace(0,w.shape[1]-1,min(CHANNEL_CAP,w.shape[1]),dtype=int)
        qq,ss = q.copy(),s.copy()
        if method == 'sdr':
            cfg = sdr.Config(work_limit=min(sdr.Config().work_limit,budget.remaining))
            plan = sdr._plan(len(x),w.shape[0],len(idx),cfg)
            if plan is None:
                groups.append({'reason':'budget_or_width','work':0})
                result.append((qq,ss))
                continue
            cost = int(plan[1])
        else:
            # Та же оценка n*d*m, что и v4/Fast; Paired резервирует двойной объём.
            cost = int(len(x))*int(w.shape[0])*len(idx)*(2 if method=='paired' else 1)
            if cost > GROUP_WORK:
                groups.append({'reason':'group_work_budget','work':0})
                result.append((qq,ss))
                continue
        if not budget.reserve(cost):
            groups.append({'reason':'work_budget','work':0})
            result.append((qq,ss))
            continue
        try:
            if method == 'sdr':
                r = sdr.quantize(w[:,idx],x,cfg)
                codes,scales = r.codes,r.scales
                reason = r.diagnostics['reason']
            else:
                codes,scales = quantize_int8(x,w[:,idx],sweeps=3,paired=method=='paired')
                reason = 'computed'
            qq[:,idx],ss[idx] = codes,scales
            groups.append({'reason':reason,'work':cost})
        except (ValueError,ArithmeticError,np.linalg.LinAlgError) as exc:
            qq,ss = q.copy(),s.copy()
            groups.append({'reason':'candidate_invalid','error':str(exc)[:200],'work':cost})
        result.append((qq,ss))
    return result,groups


def select_progressive(matrices,fit,dev,remaining,retain=None):
    """Вычислять Single, Paired, SDR последовательно до робастного выигрыша 25%."""
    budget = WorkBudget(remaining)
    base = [minmax(w) for w in matrices]
    losses = {BASE:aggregate(matrices,base,fit,dev)}
    best = base
    if retain is not None:
        retain(BASE, base, losses[BASE])
    steps = []
    stop = 'all_candidates_considered'
    if losses[BASE].dev == 0:
        stop = 'zero_baseline_dev'
    else:
        for method in ('single','paired','sdr'):
            if budget.remaining == 0:
                stop = 'work_budget'
                break
            before = budget.used
            candidate,groups = build(method,matrices,fit,base,budget)
            step = {'method':METHODS[method],'groups':groups,'work':budget.used-before}
            steps.append(step)
            if budget.used == before:
                step['status'] = 'skipped'
                continue
            try:
                losses[method] = aggregate(matrices,candidate,fit,dev)
                winner,diag = choose(losses)
            except (ValueError,ArithmeticError) as exc:
                losses.pop(method,None)
                step.update(status='invalid',error=str(exc)[:200])
                continue
            if retain is not None:
                retain(method, candidate, losses[method])
            if winner == method:
                best = candidate
            step.update(status='evaluated',robust_upper=float(diag['upper'][method]),
                        dev_mse=float(losses[method].dev))
            if diag['upper'][winner] <= -TARGET*losses[BASE].dev:
                stop = 'robust_target_reached'
                break
    winner,diag = choose(losses)
    return base,best,budget.used,{
        'method':METHODS[winner],'reason':diag['reason'],'stop_reason':stop,
        'robust_upper':{k:float(v) for k,v in diag['upper'].items()},
        'radius':float(diag['radius']),'work':budget.used,'steps':steps,
        'baseline_dev_mse':float(losses[BASE].dev),'target':TARGET,
        'robust_gain_fraction':float(-diag['upper'][winner]/losses[BASE].dev) if losses[BASE].dev else 0.,
    }
