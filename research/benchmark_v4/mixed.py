"""Политики из одного банка; измерения реальной ONNX-модели, не сумма локальных MSE."""
from pathlib import Path
import json,time,gc
import numpy as np
from adaptive_policies import POLICIES,VARIANTS,aggregate,policy,check,Portfolio
from algorithms.adaptive_selector import search_priority
from algorithms import logit_int8 as v2

EXTRA=POLICIES+['Progressive-Local','Single-Robust-Local','Single-Dev-Local']

def mixed_runs(case,cfg,root,valid,banks,requested,evaluate,search_feeds,check_feeds,base_sess,cal_time):
    from benchmark import session,infer,export,save_bank
    if not requested:return
    prior=Portfolio(json.loads((Path(__file__).parent/'prior.json').read_text()))
    records={a:{l['weight']:l for l in json.loads((root/(m+'.json')).read_text())['layers']} for a,m in VARIANTS.items()}
    stats={};vectors={};costs={}
    for t in valid:
        k=t.weight;stats[k]={};costs[k]={}
        for a in VARIANTS:
            l=records[a][k];loss,vec=aggregate(l);stats[k][a]=loss;costs[k][a]=l['layer_wall_seconds']
            if a=='minmax':vectors[k]=vec
    baseline=dict.fromkeys(stats,'minmax')
    def payload(c):return {k:banks[VARIANTS[a]][k] for k,a in c.items()}
    def run(c,feeds):
        if c is None:return infer(base_sess,feeds,cfg.batch)
        p=export(case.model,valid,payload(c));s=session(p,cfg.threads);z=infer(s,feeds,cfg.batch);del s,p;return z
    ref_check=run(None,check_feeds);base_check=run(baseline,check_feeds)
    ref_search=run(None,search_feeds)
    check_cache={tuple(baseline.items()):base_check}
    # Cache only a few small outputs, not model graphs.
    def measured(c):
        key=tuple(c.items())
        if key not in check_cache:check_cache[key]=run(c,check_feeds)
        return check_cache[key]
    def choose_global(local,plans):
        """Original V2 on classifiers, exact greedy on generic outputs. Only built candidates."""
        keys=[k for k in local if any(a!=local[k] for a in ('minmax',)+plans[k])]
        keys=sorted(keys,key=lambda k:(-search_priority({a:stats[k][a] for a in ('minmax',)+plans[k]}),k))
        active=keys[:min(cfg.v2_layers or 4,4)];calls=0
        if not active:return local,{'kind':'none','calls':0,'active':[]}
        opts={k:tuple(dict.fromkeys((local[k],'minmax')+plans[k])) if k in active else (local[k],) for k in local}
        if case.task=='classification' and len(ref_search)==1 and next(iter(ref_search.values())).ndim==2 and next(iter(ref_search.values())).shape[1]>=2:
            b={k:[banks[VARIANTS[a]][k] for a in opts[k]] for k in local}
            cap=5+sum(len(v)-1 for v in opts.values())
            def cb(weights,feeds):
                nonlocal calls
                if calls>=cap:raise ValueError('V2 budget')
                calls+=1
                if weights is None:return next(iter(run(None,feeds).values()))
                p=export(case.model,valid,weights);s=session(p,cfg.threads);z=next(iter(infer(s,feeds,cfg.batch).values()));del s,p;return z
            idx=v2.select_int8(b,cb,case.fit,search_feeds,width=cfg.beam)
            return {k:opts[k][i] for k,i in idx.items()},{'kind':'original_V2','calls':calls,'cap':cap,'active':active}
        cur=dict(local);cur_z=run(cur,search_feeds);calls=1;cap=9
        for k in active:
            for a in opts[k]:
                if calls>=cap:break
                if a==cur[k]:continue
                trial={**cur,k:a};z=run(trial,search_feeds);calls+=1
                if check(ref_search,cur_z,z)[0]:cur,cur_z=trial,z
        return cur,{'kind':'exact_multioutput_greedy','calls':calls,'cap':cap,'active':active}
    # Map exact selections to measured model outputs. Same bank/selection means same exported graph.
    seen={tuple(dict.fromkeys(stats,a).items()):m for a,m in VARIANTS.items()}
    for name in requested:
        if (root/(name+'.json')).exists():
            previous=json.loads((root/(name+'.json')).read_text())
            if 'selection' in previous:seen[tuple((k,previous['selection'][k]) for k in stats)]=name
            continue
        print('    policy',name,flush=True);t0=time.perf_counter();local={};plans={};details=[];build_seconds=0.;guard_seconds=0.
        local_only=name.endswith('-Local');kind=name.removesuffix('-Local')
        for k in stats:
            start=time.perf_counter();winner,planned,info=policy(kind,stats[k],vectors[k],prior);guard_seconds+=time.perf_counter()-start
            local[k]=winner;plans[k]=planned;build_seconds+=costs[k]['minmax']+sum(costs[k][a] for a in planned)
            details.append(dict(weight=k,local=winner,attempted=list(planned),bounds=info.get('upper'),radius=info.get('radius'),pre=info.get('pre'),dev_ratio=stats[k][winner].dev/stats[k]['minmax'].dev if stats[k]['minmax'].dev else None))
        candidate=local;global_info={'kind':'none','calls':0};global_seconds=0.
        if kind=='Progressive-Global':
            tt=time.perf_counter();candidate,global_info=choose_global(local,plans);global_seconds=time.perf_counter()-tt
        checks=[];final=local if local_only else baseline
        # Independent model check: comparison only to all-MinMax, every output must be non-worse.
        # Local-only variants deliberately expose unguarded test outcomes for ablation.
        best=None
        for c in [local]+([candidate] if candidate!=local else []):
            ok,diag=check(ref_check,base_check,measured(c));checks.append(dict(diag,selection='local' if c==local else 'global'))
            if ok and not local_only:
                score=float(np.mean([r for r in diag['ratios'] if r is not None]))
                if best is None or score<best:best=score;final=c
        if local_only:final=local
        elapsed=time.perf_counter()-t0;bank=payload(final);p=export(case.model,valid,bank)
        # Actual policy replay time excludes all-bank generation. Estimated standalone cost uses measured layer builds.
        extra=dict(calibration_seconds=cal_time,policy_replay=True,policy_build_seconds_estimate=build_seconds,policy_replay_seconds=elapsed,policy_guard_decision_seconds=guard_seconds,global_search_seconds=global_seconds,global_search=global_info,global_checks=checks,global_guard_enabled=not local_only,selection=final,local_selection=local,policy_decisions=details,expensive_builds=sum(len(v) for v in plans.values()),fallback_to_all_minmax=final==baseline,payload_bytes=sum(q.nbytes+s.nbytes for q,s in bank.values()),quantized_parameters=sum(q.size for q,s in bank.values()))
        # Compact operator-level coverage, no duplicate per-group diagnostics.
        layers=[]
        for k,a in final.items():
            l=records[a][k];gs=l['groups'];layers.append({'weight':k,'operator':l['operator'],'groups':[{'channels':sum(g['channels'] for g in gs),'accepted':sum(g.get('accepted',0) for g in gs),'attempted_channels':sum(g.get('attempted_channels',0) for g in gs)}]})
        key=tuple((k,final[k]) for k in stats)
        if key in seen:extra['reuse_from']=seen[key]
        evaluate(name,p,build_seconds+elapsed,layers,extra);seen[key]=name;del p,bank;gc.collect()
