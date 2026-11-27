"""Замороженные политики v4; выбор по fit/local-dev, без доступа к test."""
from algorithms.adaptive_selector import BASE, Loss, Portfolio, choose, features, confirm_outputs
import numpy as np

VARIANTS={'minmax':'MinMax','single':'Single-3','paired':'Paired-3','sdr':'SDR'}
POLICIES=['Single-Dev','Single-Robust','Paired-Robust','SDR-Robust','Astra-Adaptive','Progressive','FullBank-Robust','Progressive-Global']

def aggregate(layers):
    """Объединяет группы оператора с весами по числу выходов и строк."""
    gs=layers['groups']; mf=np.array([g['channels']*g['fit_rows'] for g in gs],float); md=np.array([g['channels']*g['dev_rows'] for g in gs],float)
    m=np.array([g['channels'] for g in gs],float)
    f=float(np.average([g['fit_mse'] for g in gs],weights=mf));d=float(np.average([g['dev_mse'] for g in gs],weights=md))
    e=float(np.average([g['shape'][0]*g['weight_mse'] for g in gs],weights=m))
    med=lambda k:float(np.median([g[k] for g in gs]))
    vec=features(max(g['shape'][0] for g in gs),min(g['fit_rows'] for g in gs),med('profile_sampled_features'),med('stable_rank_sample'),med('mean_abs_corr_sample'),med('weight_abs_max'),med('weight_rms'))
    return Loss(f,d,e),vec

def policy(name,losses,vec,prior,target=.25):
    """Replay банка: политика читает только результаты разрешённых построений.
    Progressive: Single, затем Paired, затем SDR до робастного выигрыша 25%.
    Порог заморожен до независимого теста, не оптимизирован по архиву.
    """
    seen={BASE:losses[BASE]}; planned=[]; pre=None
    def add(k):
        if k in losses and k not in seen:seen[k]=losses[k];planned.append(k)
    if name=='Single-Dev':
        add('single'); winner=min(seen,key=lambda k:seen[k].dev);info={'reason':'dev_only'}
    elif name=='Astra-Adaptive':
        selected,pre=prior.plan(vec,{k:1. for k in ('single','paired','sdr') if k in losses},1.)
        for k in selected:add(k)
        winner,info=choose(seen)
    else:
        if name in ('Single-Robust','Paired-Robust','SDR-Robust'):
            add(name.split('-')[0].lower())
        elif name=='FullBank-Robust':
            for k in ('single','paired','sdr'):add(k)
        elif name in ('Progressive','Progressive-Global'):
            if seen[BASE].dev>0:
                for k in ('single','paired','sdr'):
                    add(k); win,diag=choose(seen)
                    if diag['upper'][win] <= -target*seen[BASE].dev:break
        else:raise ValueError(name)
        winner,info=choose(seen)
    return winner,tuple(planned),dict(info,pre=pre)

def check(reference,baseline,proposal):
    old,new=confirm_outputs(reference,baseline,proposal)
    accept=bool(np.all(new<=old) and np.any(new<old))
    ratios=[float(n/o) if o>0 else (0. if n==0 else None) for o,n in zip(old,new)]
    return accept,dict(outputs=sorted(reference),old=old.tolist(),new=new.tolist(),ratios=ratios,accepted=accept)
