"""ONNX-бенчмарк с изолированными GPU-воркерами и checkpoint по слоям."""
import argparse, hashlib, json, os, sys, time, subprocess, gc, traceback
from pathlib import Path
from dataclasses import dataclass,asdict
from collections import Counter
from unittest.mock import patch
import numpy as np
from threadpoolctl import threadpool_limits
from algorithms import activation_int8 as v1, spectral_rounding as sdr, logit_int8 as v2
from graph import discover, with_outputs, sample_rows, export
from models import make_case,custom_case

from mixed import EXTRA, mixed_runs
VERSION='4.0.0-kaggle'
METHODS=['FP32','MinMax','Single-3','Paired-3','SDR','V2']+EXTRA
@dataclass(frozen=True)
class Profile:
    models: tuple
    seeds: tuple
    fit: int
    dev: int
    test: int
    rows: int
    threads: int
    batch: int
    sdr_work: int=150_000_000
    v1_work: int=100_000_000
    beam: int=4
    channel_cap: int=0
    v2_layers: int=0
    cache_mib: int=128
    provider: str="cpu"
    math_backend: str='numpy'
    capture_group: int=4
PROFILES={
 'final_smoke':Profile(('efficientnet_b0','minilm_embeddings','multi_output'),(42,),12,12,16,32,2,4,500_000_000,100_000_000,2,4,2,256,'cuda'),
 'final':Profile(('efficientnet_b0','vit_tiny','minilm_embeddings','mlp','regression','autoencoder','multi_output','lstm_head'),(42,123,731),128,96,512,512,2,8,2_000_000_000,500_000_000,4,128,4,768,'cuda'),
 'final_full':Profile(('efficientnet_b0','vit_tiny','minilm_embeddings','mlp','regression','autoencoder','multi_output','lstm_head'),(42,123,731,2026,91),256,144,1024,1024,2,8,6_000_000_000,2_000_000_000,4,256,4,1024,'cuda'),
 'gpu_smoke':Profile(('resnet18','distilbert_sst2'),(42,),8,8,16,32,2,4,500_000_000,100_000_000,2,4,2,256,'cuda'),
 'research':Profile(('resnet18','mobilenet_v2','squeezenet1_1','distilbert_sst2','mlp','regression','autoencoder','multi_output'),(42,123,731),128,64,512,512,2,8,2_000_000_000,500_000_000,4,128,8,768,'cuda'),
 'research_full':Profile(('resnet18','mobilenet_v2','squeezenet1_1','distilbert_sst2','mlp','regression','autoencoder','multi_output','rnn'),(42,123,731,2026,91),256,128,1024,1024,2,8,6_000_000_000,2_000_000_000,8,256,12,1024,'cuda'),
 'smoke':Profile(('cnn','depthwise_cnn','regression'),(42,),8,8,12,24,1,4),
 'quick':Profile(('bert_tiny','mlp','cnn','regression'),(42,),64,32,128,256,2,8),
 'full':Profile(('bert_tiny','mlp','cnn','depthwise_cnn','regression','autoencoder'),(42,123,731),128,64,360,512,2,8),
 'extended':Profile(('bert_tiny','mlp','cnn','depthwise_cnn','regression','autoencoder','multi_output','rnn'),(42,123,731,2026,91),192,64,872,2048,4,8,350_000_000,200_000_000,8)}

def atomic(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');os.replace(tmp,path)

def hash_feeds(h,feeds):
    for k,v in sorted(feeds.items()):
        a=np.ascontiguousarray(v);h.update(k.encode());h.update(str((a.shape,a.dtype)).encode());h.update(memoryview(a).cast('B'))

def session(model,threads,profiling=False):
    import onnxruntime as ort
    opt=ort.SessionOptions();opt.intra_op_num_threads=threads;opt.inter_op_num_threads=1;opt.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL
    gpu=os.environ.get('QUANT_INFER_PROVIDER',os.environ.get('QUANT_PROVIDER','cpu'))=='cuda'
    if profiling:opt.enable_profiling=True
    if gpu:
        import torch  # Загружает совместимые CUDA/cuDNN библиотеки до ORT.
        if not torch.cuda.is_available() or 'CUDAExecutionProvider' not in ort.get_available_providers():
            raise RuntimeError('CUDA недоступна. Проверьте Kaggle GPU / onnxruntime-gpu.')
        if hasattr(ort,'preload_dlls'):ort.preload_dlls()
        providers=[('CUDAExecutionProvider',{'device_id':0,'use_tf32':0,'cudnn_conv_algo_search':os.environ.get('QUANT_CONV_ALGO','HEURISTIC'),'cudnn_conv_use_max_workspace':1,'prefer_nhwc':int(os.environ.get('QUANT_NHWC','0'))}),'CPUExecutionProvider']
    else:providers=['CPUExecutionProvider']
    sess=ort.InferenceSession(model.SerializeToString(),sess_options=opt,providers=providers)
    if gpu and sess.get_providers()[0]!='CUDAExecutionProvider':raise RuntimeError('ORT silently fell back to CPU')
    sess.disable_fallback()
    return sess

def batches(feeds,batch):
    if not feeds or any(v.ndim==0 for v in feeds.values()): raise ValueError('Все входы должны иметь общую первую batch-ось.')
    lengths={len(v) for v in feeds.values()}
    if len(lengths)!=1 or not next(iter(lengths)): raise ValueError('Несогласованные batch-размеры входов.')
    n=next(iter(lengths))
    for i in range(0,n,batch): yield {k:v[i:i+batch] for k,v in feeds.items()}

def infer(sess,feeds,batch):
    names=[o.name for o in sess.get_outputs()]; out={n:[] for n in names}
    for f in batches(feeds,batch):
        for n,v in zip(names,sess.run(names,f)):
            if v.ndim==0 or v.shape[0]!=len(next(iter(f.values()))) or v.dtype.kind not in 'fiu b'.replace(' ',''):
                raise ValueError(f'Выход {n}: нужен числовой тензор с первой batch-осью.')
            if not np.isfinite(v).all(): raise ValueError(f'Неконечный выход {n}')
            out[n].append(v)
    return {k:np.concatenate(v,axis=0) for k,v in out.items()}

def capture_many(model, indexed, feeds, cfg, seed, dev=False):
    """Одна ORT-сессия на группу, независимый reservoir на каждый слой."""
    names=list(dict.fromkeys(v for _,t in indexed for v in (t.input,t.output)))
    sess=session(with_outputs(model,names),cfg.threads)
    states={i:[np.random.default_rng(seed+i*101+int(dev)),None,np.empty(0),0] for i,t in indexed}
    errors={}
    for feed in batches(feeds,cfg.batch):
        outputs=dict(zip(names,sess.run(names,feed)))
        for i,t in indexed:
            if i in errors:continue
            rng,pools,keys,total=states[i]
            try:
                rows,n=sample_rows(outputs[t.input],outputs[t.output],t,cfg.rows,rng,feed.get('attention_mask'))
                if not n:continue
                k=len(rows[0]);priorities=[];prev=0.
                for j in range(k):
                    prev+=(1-prev)*rng.beta(1,n-j);priorities.append(prev)
                priorities=np.array(priorities);rng.shuffle(priorities)
                merged=np.concatenate([keys,priorities]);keep=np.argsort(merged,kind='stable')[:cfg.rows]
                pools=[np.concatenate([a,b],axis=0)[keep] for a,b in zip(pools,rows)] if pools is not None else [a[keep] for a in rows]
                states[i]=[rng,pools,merged[keep],total+n]
            except (ValueError,ArithmeticError) as exc:errors[i]=str(exc)
        del outputs
    del sess
    return {i:(v[1],v[3]) for i,v in states.items() if i not in errors and v[1] is not None},errors


def capture(model,target,feeds,cfg,seed):
    data,errors=capture_many(model,[(0,target)],feeds,cfg,seed)
    if 0 not in data:raise ValueError(errors.get(0,'Нет допустимых строк активаций'))
    return data[0]


def matrix_profile(w,x):
    # Диагностика на фиксированной подвыборке признаков, не полный спектр слоя.
    idx=np.linspace(0,x.shape[1]-1,min(64,x.shape[1]),dtype=int)
    a=x[:,idx].astype(float);g=a.T@a;ev=np.maximum(np.linalg.eigvalsh(g),0);total=ev.sum()
    h=np.sqrt(np.diag(g));c=np.divide(g,h[:,None]*h[None,:],out=np.zeros_like(g),where=h[:,None]*h[None,:]>0)
    np.fill_diagonal(c,0.)
    return {'profile_sampled_features':len(idx),'stable_rank_sample':float(total/ev[-1]) if ev[-1]>0 else 0.,'top_energy_sample':float(ev[-1]/total) if total>0 else 0.,'mean_abs_corr_sample':float(np.abs(c).sum()/max(1,len(idx)*(len(idx)-1))),'weight_abs_max':float(np.abs(w).max()),'weight_rms':float(np.sqrt(np.mean(w.astype(float)**2)))}

def local_metrics(w,x,q,s):
    err=(q*s).astype(np.float64)-w; residual=x.astype(np.float64)@err
    return float(np.mean(err**2)),float(np.mean(residual**2))

def _quant_core(w,x,method,cfg):
    base=v1.quantize_int8(x,w,sweeps=0); info={'reason':'baseline','accepted':0}
    if method=='MinMax':return base,{**info,'math_backend':'numpy'}
    if cfg.math_backend=='torch':
        from cuda_math import v1_torch,sdr_torch
        device='cuda' if cfg.provider=='cuda' else 'cpu'
        if method in ('Single-3','Paired-3'):
            if int(x.shape[0])*int(w.size)>cfg.v1_work:return base,{'reason':'budget','accepted':0,'math_backend':'numpy'}
            q,s=v1_torch(x,w,paired=method=='Paired-3',device=device)
            changed=np.any(q!=base[0],axis=0)|(s!=base[1]);return (q,s),{'reason':'accepted' if changed.any() else 'no_gain','accepted':int(changed.sum()),'math_backend':'torch_'+device}
        if method=='SDR':
            pair,info=sdr_torch(x,w,sdr.Config(work_limit=cfg.sdr_work),device=device)
            return pair,{**info,'math_backend':'torch_'+device}
    if method in ('Single-3','Paired-3'):
        if int(x.shape[0])*int(w.size)>cfg.v1_work:
            return base,{'reason':'budget','accepted':0}
        if method=='Single-3':
            with patch.object(v1,'_pairs',return_value=[]): q,s=v1.quantize_int8(x,w,sweeps=3)
        else:q,s=v1.quantize_int8(x,w,sweeps=3)
        changed=np.any(q!=base[0],axis=0)|(s!=base[1]);return (q,s),{'reason':'accepted' if changed.any() else 'no_gain','accepted':int(changed.sum())}
    if method=='SDR':
        r=sdr.quantize(w.astype(np.float32,copy=False),x.astype(np.float32,copy=False),sdr.Config(work_limit=cfg.sdr_work))
        return (r.codes,r.scales),r.diagnostics
    raise ValueError(method)

def quant(w,x,method,cfg):
    # Одинаковый детерминированный набор каналов для всех методов; остальные MinMax.
    cap=cfg.channel_cap
    if method=='MinMax' or not cap or w.shape[1]<=cap:
        pair,info=_quant_core(w,x,method,cfg)
        return pair,{**info,'attempted_channels':0 if method=='MinMax' or info['reason'] in ('budget','budget_or_width','flat_blocks') else w.shape[1],'channel_cap':cap}
    (q,s),_=_quant_core(w,x,'MinMax',cfg)
    idx=np.linspace(0,w.shape[1]-1,cap,dtype=int)
    (qq,ss),info=_quant_core(w[:,idx],x,method,cfg)
    q[:,idx]=qq;s[idx]=ss
    return (q,s),{**info,'attempted_channels':0 if info['reason'] in ('budget','budget_or_width','flat_blocks') else len(idx),'channel_cap':cap,'selected_channels':idx.tolist()}

def fidelity(ref,pred,task,labels):
    if set(ref)!=set(pred):raise ValueError('Набор выходов отличается')
    metrics={}; sample=[]
    for k,r in ref.items():
        p=pred[k]
        if r.shape!=p.shape:raise ValueError(f'Размерность выхода {k} изменилась')
        r=r.astype(float);p=p.astype(float); e=p-r; energy=float(np.mean(r*r)); mse=float(np.mean(e*e))
        sample.append(np.mean(e.reshape(len(e),-1)**2,axis=1))
        metrics[k]={'mse':mse,'mae':float(np.mean(abs(e))),'max_abs':float(np.max(abs(e))),'nmse':mse/energy if energy>1e-20 else None}
    result={'outputs':metrics,'mse_equal_output_mean':float(np.mean([m['mse'] for m in metrics.values()])), 'samples':len(next(iter(ref.values())))}
    if len(ref)==1:
        r,p=next(iter(ref.values())),next(iter(pred.values()))
        if task=='classification' and r.ndim==2 and r.shape[1]>=2:
            a=r.argmax(1);b=p.argmax(1);result.update(agreement=float(np.mean(a==b)),changed=int(np.sum(a!=b)))
            if labels is not None:
                from sklearn.metrics import f1_score
                y=np.asarray(labels).reshape(-1);result['top5_accuracy']=float(np.mean(np.any(np.argsort(p,axis=1)[:,-min(5,p.shape[1]):]==y[:,None],axis=1))) if p.shape[1]>=5 else None;result.update(accuracy=float(np.mean(b==y)),macro_f1=float(f1_score(y,b,average='macro',zero_division=0)),lost_correct=int(np.sum((a==y)&(b!=y))),restored_correct=int(np.sum((a!=y)&(b==y))))
        if task in ('regression','reconstruction') and labels is not None:
            y=np.asarray(labels).reshape(p.shape);e=p.astype(float)-y;result.update(task_rmse=float(np.sqrt(np.mean(e*e))),task_mae=float(np.mean(abs(e))))
        if task=='generic' and r.ndim==2 and r.shape[1]>1:
            a=r.astype(float);b=p.astype(float);den=np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1)
            if np.all(den>0):result['mean_cosine_to_fp32']=float(np.mean(np.sum(a*b,axis=1)/den))
    elif task=='multi':
        if 'logits' in ref:result['logits_agreement']=float(np.mean(ref['logits'].argmax(1)==pred['logits'].argmax(1)))
    return result,np.mean(np.stack(sample),axis=0)

def save_bank(path,bank):
    data={}
    for i,(k,(q,s)) in enumerate(bank.items()):data[f'q{i}']=q;data[f's{i}']=s
    data['names']=np.array(list(bank))
    tmp=path.with_suffix('.tmp.npz');np.savez_compressed(tmp,**data);os.replace(tmp,path)
def load_bank(path):
    with np.load(path,allow_pickle=False) as z:return {str(k):(z[f'q{i}'],z[f's{i}']) for i,k in enumerate(z['names'])}

def benchmark_one(case,cfg,seed,out,methods,save_models):
    import onnx,onnxruntime as ort
    h=hashlib.sha256(case.model.SerializeToString())
    for f in (case.fit,case.dev,case.test,case.shift or {}):hash_feeds(h,f)
    if case.labels is not None:h.update(np.ascontiguousarray(case.labels).tobytes())
    if case.shift_labels is not None:h.update(np.ascontiguousarray(case.shift_labels).tobytes())
    device={'provider':cfg.provider,'inference_provider':os.environ.get('QUANT_INFER_PROVIDER',cfg.provider),'conv_algo':os.environ.get('QUANT_CONV_ALGO','HEURISTIC'),'nhwc':os.environ.get('QUANT_NHWC','0'),'math_backend':cfg.math_backend}
    if cfg.provider=='cuda':
        import torch
        device.update(gpu=torch.cuda.get_device_name(0),torch=torch.__version__,cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version())
    h.update(json.dumps(device,sort_keys=True).encode());h.update(json.dumps(asdict(cfg),sort_keys=True).encode());h.update(str((VERSION,np.__version__,onnx.__version__,ort.__version__,save_models,case.task)).encode())
    for p in sorted(Path(__file__).parent.glob('*.py'))+sorted((Path(__file__).parent/'algorithms').glob('*.py')):h.update(p.read_bytes())
    h.update((Path(__file__).parent/'prior.json').read_bytes());key=h.hexdigest()[:16];root=out/case.name/f'seed_{seed}'/key;root.mkdir(parents=True,exist_ok=True)
    atomic(root/'manifest.json',{'model':case.name,'seed':seed,'task':case.task,'historical_test':case.historical,'note':case.note,'config':asdict(cfg),'fingerprint':key,'versions':{'numpy':np.__version__,'onnx':onnx.__version__,'ort':ort.__version__},'provider_requested':cfg.provider,'device':device,'provenance':getattr(case,'provenance',{}),'counts':{k:len(next(iter(v.values()))) for k,v in [('fit',case.fit),('dev',case.dev),('test',case.test)]}})
    if all((root/(m+'.json')).exists() for m in methods): print(f'CACHE {case.name} seed={seed}',flush=True);return root
    start=time.perf_counter();onnx.checker.check_model(case.model);base_sess=session(case.model,cfg.threads)
    # Local selection, global search, final check use disjoint subsets of original dev.
    ndev=len(next(iter(case.dev.values())))
    if ndev<3:raise ValueError('Need at least 3 development examples for v4 splits')
    a=max(1,ndev//2);b=a+max(1,(ndev-a)//2)
    local_feeds={k:v[:a] for k,v in case.dev.items()};search_feeds={k:v[a:b] for k,v in case.dev.items()};check_feeds={k:v[b:] for k,v in case.dev.items()}
    atomic(root/'splits_v4.json',{'fit':len(next(iter(case.fit.values()))),'local_dev':a,'global_search':b-a,'global_check':ndev-b,'test':len(next(iter(case.test.values()))),'note':'Original dev split by position; all selectors frozen before test metrics are used. No test values used in selection.'})
    ref=infer(base_sess,case.test,cfg.batch);ref_dev=infer(base_sess,search_feeds,cfg.batch);ref_shift=infer(base_sess,case.shift,cfg.batch) if case.shift else None
    fit_ref=None
    def evaluate(method,model,prep,layer_data,extra):
        reuse=extra.pop('reuse_from',None)
        if reuse and (root/(reuse+'.json')).exists() and (root/(reuse+'.mse.npy')).exists():
            import shutil
            source=json.loads((root/(reuse+'.json')).read_text())
            # Only inference metrics/latency are reused for byte-identical candidate selections.
            keys=('metrics','development','shift','model_bytes','initializer_bytes','nodes','dq_nodes','median_ms','p95_ms','latency_batch','providers','inference_provider','math_backend_requested','latency_warmup','latency_repetitions')
            result={k:source[k] for k in keys};result.update(method=method,model=case.name,seed=seed,status='ok',historical_test=case.historical,prepare_seconds=prep,calibration_seconds=extra.pop('calibration_seconds',0.),layers=layer_data,measurement_reused_from=reuse,**extra)
            if save_models:onnx.save_model(model,str(root/(method+'.onnx')))
            shutil.copyfile(root/(reuse+'.mse.npy'),root/(method+'.mse.npy'));atomic(root/(method+'.json'),result)
            print(f'{case.name} seed={seed} {method}: MSE={result["metrics"]["mse_equal_output_mean"]:.6g} prep={prep:.2f}s (identical selection: {reuse})',flush=True);return
        sess=base_sess if method=='FP32' else session(model,cfg.threads)
        pred=infer(sess,case.test,cfg.batch);dev=infer(sess,search_feeds,cfg.batch)
        metrics,per=fidelity(ref,pred,case.task,case.labels);dev_metrics,_=fidelity(ref_dev,dev,case.task,None)
        shift_metrics=None
        if case.shift:shift_metrics,_=fidelity(ref_shift,infer(sess,case.shift,cfg.batch),case.task,case.shift_labels)
        sample=next(batches(case.test,cfg.batch));times=[]
        for i in range(25):
            t=time.perf_counter();sess.run(None,sample);dt=(time.perf_counter()-t)*1000
            if i>=5:times.append(dt)
        if save_models:onnx.save_model(model,str(root/(method+'.onnx')))
        # Per-example MSE позволяет вычислять парное сравнение, без повторного инференса.
        np.save(root/(method+'.mse.npy'),per)
        result={'method':method,'model':case.name,'seed':seed,'status':'ok','historical_test':case.historical,'metrics':metrics,'development':dev_metrics,'shift':shift_metrics,'prepare_seconds':prep,'calibration_seconds':extra.pop('calibration_seconds',0.),'model_bytes':len(model.SerializeToString()),'initializer_bytes':sum(t.raw_data.__len__() or len(t.SerializeToString()) for t in model.graph.initializer),'nodes':len(model.graph.node),'dq_nodes':sum(n.op_type=='DequantizeLinear' for n in model.graph.node),'median_ms':float(np.median(times)),'p95_ms':float(np.percentile(times,95)),'latency_batch':len(next(iter(sample.values()))),'providers':sess.get_providers(),'inference_provider':os.environ.get('QUANT_INFER_PROVIDER',cfg.provider),'math_backend_requested':cfg.math_backend,'latency_warmup':5,'latency_repetitions':20,'layers':layer_data,**extra}
        atomic(root/(method+'.json'),result);print(f'{case.name} seed={seed} {method}: MSE={metrics["mse_equal_output_mean"]:.6g} prep={prep:.2f}s',flush=True)
        if method!='FP32':del sess
    if 'FP32' in methods and not (root/'FP32.json').exists():evaluate('FP32',case.model,0,[],{})
    if set(methods)=={'FP32'}:return root
    start=time.perf_counter();targets,skipped=discover(case.model);data={};valid=[]
    # Группируем выходы, Fit и Dev используют одну сессию на группу каждый.
    for start_i in range(0,len(targets),cfg.capture_group):
        indexed=list(enumerate(targets[start_i:start_i+cfg.capture_group],start_i))
        pending=[(i,t) for i,t in indexed if not (root/f'activation_{i:04d}.npz').exists()]
        if pending:
            print(f'  capture {start_i+1}–{start_i+len(indexed)}/{len(targets)}',flush=True)
            try:
                fit,ef=capture_many(case.model,pending,case.fit,cfg,seed)
                dev,ed=capture_many(case.model,pending,local_feeds,cfg,seed,True)
            except Exception as exc:
                # Изолируем неисправный адаптер; ошибки CUDA не скрываем.
                print('  group capture failed, isolating:',str(exc)[:180],flush=True)
                fit={};dev={};ef={};ed={}
                for i,t in pending:
                    try:
                        f,_=capture_many(case.model,[(i,t)],case.fit,cfg,seed);d,_=capture_many(case.model,[(i,t)],local_feeds,cfg,seed,True);fit.update(f);dev.update(d)
                    except Exception as exc:
                        if os.environ.get('QUANT_INFER_PROVIDER')=='cuda' and any(k in str(exc).lower() for k in ('cudnn','ep_fail','cudaexecutionprovider')):raise
                        ef[i]=str(exc)
            for i,t in pending:
                if i not in fit or i not in dev:
                    skipped.append({'node':t.node,'weight':t.weight,'reason':ef.get(i,ed.get(i,'no_valid_rows'))[:400]});continue
                x,n=fit[i];d,nd=dev[i];ap=root/f'activation_{i:04d}.npz';tmp=ap.with_suffix('.tmp.npz')
                np.savez(tmp,counts=np.array([n,nd]),**{f'x{j}':v for j,v in enumerate(x)},**{f'd{j}':v for j,v in enumerate(d)});os.replace(tmp,ap)
            del fit,dev
        for i,t in indexed:
            ap=root/f'activation_{i:04d}.npz'
            if not ap.exists():continue
            # Активации загружаются с диска по слою, не отбрасываем большие модели по размеру RAM.
            data[t.weight]=ap;valid.append(t)
            print(f'  {t.op} {t.node}: cached',flush=True)
    cal_time=time.perf_counter()-start
    atomic(root/'coverage.json',{'eligible':len(targets),'validated':len(valid),'graph_operator_counts':dict(Counter(n.op_type for n in case.model.graph.node)),'eligible_parameters':sum(t.original.size for t in targets),'fp32_initializer_parameters':sum(int(np.prod(t.dims)) for t in case.model.graph.initializer if t.data_type==onnx.TensorProto.FLOAT),'skipped':skipped,'quantized_parameters':sum(t.original.size for t in valid),'calibration_seconds':cal_time})
    if not valid:
        for m in methods:
            if m!='FP32':atomic(root/(m+'.json'),{'model':case.name,'seed':seed,'method':m,'status':'skipped','reason':'no_validated_targets'})
        return root
    banks={}
    required=set(methods)
    v2_eligible=case.task=='classification' and len(ref)==1 and next(iter(ref.values())).ndim==2 and next(iter(ref.values())).shape[1]>=2
    if any(m in EXTRA for m in methods):required.update(['MinMax','Single-3','Paired-3','SDR'])
    if 'V2' in methods and v2_eligible:required.update(['MinMax','Single-3','Paired-3','SDR'])
    for method in ('MinMax','Single-3','Paired-3','SDR'):
        if method not in required:continue
        bp=root/(method+'.bank.npz');jp=root/(method+'.json')
        if bp.exists() and jp.exists():banks[method]=load_bank(bp);continue
        bank={};layers=[];t0=time.perf_counter()
        for li,t in enumerate(valid):
            print(f'    {method} {li+1}/{len(valid)} {t.node}',flush=True)
            layer_started=time.perf_counter()
            cp=root/f'{method}.layer_{li:04d}.npz';cj=cp.with_suffix('.json')
            if cp.exists() and cj.exists():
                with np.load(cp,allow_pickle=False) as z:bank[t.weight]=(z['q'],z['s'])
                layers.append(json.loads(cj.read_text(encoding='utf-8')));continue
            with np.load(data[t.weight],allow_pickle=False) as z:
                n,nd=map(int,z['counts']);xs=[z[f'x{j}'] for j in range(t.groups)];ds=[z[f'd{j}'] for j in range(t.groups)]
            pairs=[];details=[]
            for group,(w,x,d) in enumerate(zip(t.matrices(),xs,ds)):
                try:pair,info=quant(w,x,method,cfg)
                except (ValueError,ArithmeticError,np.linalg.LinAlgError,RuntimeError) as exc:
                    pair=v1.quantize_int8(x,w,sweeps=0);info={'reason':'algorithm_error','error':str(exc)[:400],'accepted':0,'math_backend':'numpy_fallback','attempted_channels':0}
                q,s=pair;wm,fm=local_metrics(w,x,q,s);_,dm=local_metrics(w,d,q,s)
                details.append({'group':group,'shape':list(w.shape),'fit_rows':len(x),'dev_rows':len(d),'fit_available':n,'dev_available':nd,'weight_mse':wm,'fit_mse':fm,'dev_mse':dm,'channels':w.shape[1],**matrix_profile(w,x),**info});pairs.append(pair)
            q,s=t.pack(pairs);bank[t.weight]=t.canonical(q,s);layer={'node':t.node,'weight':t.weight,'operator':t.op,'axis':t.axis,'groups':details,'layer_wall_seconds':time.perf_counter()-layer_started};layers.append(layer)
            tmp=cp.with_suffix('.tmp.npz');np.savez(tmp,q=bank[t.weight][0],s=bank[t.weight][1]);os.replace(tmp,cp);atomic(cj,layer)
        prep=sum(l.get('layer_wall_seconds',0.) for l in layers);model=export(case.model,valid,bank);save_bank(bp,bank);banks[method]=bank
        evaluate(method,model,prep,layers,{'calibration_seconds':cal_time,'payload_bytes':sum(q.nbytes+s.nbytes for q,s in bank.values()),'quantized_parameters':sum(q.size for q,s in bank.values())});del model;gc.collect()
    if 'V2' in methods and not (root/'V2.json').exists():
        if not v2_eligible:
            atomic(root/'V2.json',{'model':case.name,'seed':seed,'method':'V2','status':'skipped','reason':'original V2 requires one classification logits tensor'})
        else:
            variant_names=['MinMax','Single-3','Paired-3','SDR']
            scores={m:json.loads((root/(m+'.json')).read_text(encoding='utf-8'))['development']['mse_equal_output_mean'] for m in variant_names}
            anchor=min(variant_names,key=lambda m:scores[m]);variant_names=[anchor]+[m for m in variant_names if m!=anchor]
            bank={t.weight:[banks[m][t.weight] for m in variant_names] for t in valid}
            # Неизменённый V2 на ограниченном банке: остальные тензоры фиксированы на лучшем по dev MSE anchor.
            variable=[t for t in valid if any(not (np.array_equal(bank[t.weight][0][0],p[0]) and np.array_equal(bank[t.weight][0][1],p[1])) for p in bank[t.weight][1:])]
            active={t.weight for t in sorted(variable,key=lambda t:(-t.original.size,t.weight))[:cfg.v2_layers]} if cfg.v2_layers else {t.weight for t in variable}
            bank={k:(v if k in active else v[:1]) for k,v in bank.items()}
            calls=0; t0=time.perf_counter()
            def callback(weights,feeds):
                nonlocal calls
                calls+=1
                p=case.model if weights is None else export(case.model,valid,weights)
                s=session(p,cfg.threads);z=next(iter(infer(s,feeds,cfg.batch).values()));del s;return z
            indices=v2.select_int8(bank,callback,case.fit,search_feeds,width=cfg.beam)
            chosen={k:bank[k][i] for k,i in indices.items()};prep=time.perf_counter()-t0
            p=export(case.model,valid,chosen);save_bank(root/'V2.bank.npz',chosen)
            evaluate('V2',p,prep,[],{'calibration_seconds':cal_time,'candidate_bank':variant_names,'anchor':anchor,'anchor_dev_mse':scores,'v2_variable_layers':sorted(active),'v2_layer_cap':cfg.v2_layers,'variant':'budgeted V2' if cfg.v2_layers else 'V2','selection':{k:variant_names[i] for k,i in indices.items()},'forward_dataset_calls':calls,'payload_bytes':sum(q.nbytes+s.nbytes for q,s in chosen.values()),'quantized_parameters':sum(q.size for q,s in chosen.values()),'bank_build_seconds':sum(json.loads((root/(m+'.json')).read_text(encoding='utf-8'))['prepare_seconds'] for m in variant_names)})
    mixed_runs(case,cfg,root,valid,banks,[m for m in methods if m in EXTRA],evaluate,search_feeds,check_feeds,base_sess,cal_time)
    return root


def summarize(out):
    import csv
    all_runs=[]; manifests=[]
    # Только последний fingerprint каждого model/seed, чтобы не смешивать настройки.
    for seed_dir in sorted(out.glob('*/seed_*')):
        roots=[p for p in seed_dir.iterdir() if (p/'manifest.json').is_file()]
        if not roots:continue
        root=max(roots,key=lambda p:(p/'manifest.json').stat().st_mtime)
        manifest=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
        rows=[json.loads((root/(m+'.json')).read_text(encoding='utf-8')) for m in METHODS if (root/(m+'.json')).is_file()]
        base=next((r for r in rows if r['method']=='MinMax' and r['status']=='ok'),None)
        for r in rows:
            if base and r['status']=='ok':
                b=base['metrics']['mse_equal_output_mean'];r['ratio_to_minmax']=r['metrics']['mse_equal_output_mean']/b if b else None
                if r['method'] not in ('FP32','MinMax') and (root/(r['method']+'.mse.npy')).exists():
                    diff=np.load(root/(r['method']+'.mse.npy'))-np.load(root/'MinMax.mse.npy');rng=np.random.default_rng(42)
                    means=np.array([rng.choice(diff,len(diff),replace=True).mean() for _ in range(500)])
                    r['paired_bootstrap_delta_ci95']=[float(i) for i in np.percentile(means,[2.5,97.5])]
            all_runs.append(r)
        manifest['coverage']=json.loads((root/'coverage.json').read_text(encoding='utf-8')) if (root/'coverage.json').exists() else None
        manifest['process_stats']=json.loads((root/'process_stats.json').read_text(encoding='utf-8')) if (root/'process_stats.json').exists() else None
        manifests.append(manifest)
    atomic(out/'research_report.json',{'version':VERSION,'runs':all_runs,'manifests':manifests,'note':'Test-выборки между seed фиксированы. Bootstrap внутри одного seed, не независимый CI по всем seed; BERT test исторический.'})
    flat=[]
    for r in all_runs:
        if r['status']!='ok':flat.append({k:r.get(k) for k in ('model','seed','method','status','reason')});continue
        groups=[g for l in r['layers'] for g in l['groups']];channels=sum(g['channels'] for g in groups)
        m=r['metrics'];flat.append({'model':r['model'],'seed':r['seed'],'method':r['method'],'status':'ok','mse':m['mse_equal_output_mean'],'ratio_to_minmax':r.get('ratio_to_minmax'),'accuracy':m.get('accuracy'),'agreement':m.get('agreement'),'task_rmse':m.get('task_rmse'),'shift_mse':(r.get('shift') or {}).get('mse_equal_output_mean'),'prepare_s':r['prepare_seconds'],'median_ms':r['median_ms'],'model_bytes':r['model_bytes'],'channels_applied':sum(g.get('accepted',0) for g in groups),'channels_total':channels,'channels_attempted':sum(g.get('attempted_channels',0) for g in groups),'fallback_groups':sum(g.get('accepted',0)==0 for g in groups),'historical_test':r['historical_test'],'policy_build_seconds_estimate':r.get('policy_build_seconds_estimate'),'policy_replay_seconds':r.get('policy_replay_seconds'),'expensive_builds':r.get('expensive_builds'),'fallback_to_all_minmax':r.get('fallback_to_all_minmax')})
    cols=list(dict.fromkeys(k for row in flat for k in row));out.mkdir(parents=True,exist_ok=True)
    with (out/'summary.csv').open('w',encoding='utf-8-sig',newline='') as f:
        if cols:w=csv.DictWriter(f,fieldnames=cols);w.writeheader();w.writerows(flat)
    aggregates=[]
    for model,method in sorted({(r['model'],r['method']) for r in flat if r['status']=='ok'}):
        selected=[r for r in flat if r['model']==model and r['method']==method and r['status']=='ok']
        ratios=[r['ratio_to_minmax'] for r in selected if r.get('ratio_to_minmax') is not None]
        aggregates.append({'model':model,'method':method,'seeds':len(selected),'median_ratio_to_minmax':float(np.median(ratios)) if ratios else None,'median_prepare_s':float(np.median([r['prepare_s'] for r in selected]))})
    atomic(out/'aggregate.json',aggregates)
    print('Report:',out/'research_report.json',flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--provider',choices=['cpu','cuda']);p.add_argument('--math-backend',choices=['numpy','torch'],default='torch');p.add_argument('--capture-group',type=int,default=4);p.add_argument('--gpus',type=int,default=2);p.add_argument('--conv-cpu-fallback',action='store_true');p.add_argument('--channel-cap',type=int);p.add_argument('--v2-layers',type=int);p.add_argument('--profile',choices=PROFILES,default='quick');p.add_argument('--models',nargs='+');p.add_argument('--seeds',nargs='+',type=int);p.add_argument('--methods',nargs='+',choices=METHODS,default=METHODS);p.add_argument('--archive');p.add_argument('--output');p.add_argument('--threads',type=int);p.add_argument('--rows',type=int);p.add_argument('--sdr-work',type=int);p.add_argument('--save-models',action='store_true');p.add_argument('--worker',action='store_true');p.add_argument('--custom-model');p.add_argument('--fit-npz');p.add_argument('--dev-npz');p.add_argument('--test-npz');p.add_argument('--shift-npz');p.add_argument('--task',choices=['generic','classification','regression','reconstruction','multi'],default='generic');a=p.parse_args()
    from dataclasses import replace
    cfg=PROFILES[a.profile];cfg=replace(cfg,threads=a.threads or cfg.threads,rows=a.rows or cfg.rows,sdr_work=a.sdr_work if a.sdr_work is not None else cfg.sdr_work)
    cfg=replace(cfg,math_backend=a.math_backend,capture_group=a.capture_group,provider=a.provider or cfg.provider,channel_cap=a.channel_cap if a.channel_cap is not None else cfg.channel_cap,v2_layers=a.v2_layers if a.v2_layers is not None else cfg.v2_layers)
    if min(cfg.channel_cap,cfg.v2_layers)<0:p.error('negative cap')
    os.environ['QUANT_PROVIDER']=cfg.provider
    if cfg.rows<1 or cfg.threads<1 or cfg.sdr_work<0 or a.capture_group<1 or a.gpus<1:p.error('Недопустимый бюджет')
    root=Path(__file__).resolve().parent;out=Path(a.output).resolve() if a.output else root/'results'/a.profile;out.mkdir(parents=True,exist_ok=True)
    archive=a.archive
    if archive is None:
        archive=next((str(z) for z in [root/'onnx-quant-lab-handoff-v0.5.zip',root.parent/'onnx-quant-lab-handoff-v0.5.zip',root/'onnx-quant-lab',root.parent/'onnx-quant-lab'] if z.exists()),None)
    models=['custom'] if a.custom_model else (a.models or cfg.models);seeds=a.seeds or cfg.seeds
    if a.custom_model and not all([a.fit_npz,a.dev_npz,a.test_npz]):p.error('custom требует fit/dev/test NPZ')
    if a.worker:
        name=models[0];seed=seeds[0]
        if cfg.provider=='cuda':
            from gpu_preflight import configure
            configure(name,a.conv_cpu_fallback)
            print('Inference:',os.environ.get('QUANT_INFER_PROVIDER'), 'Math:',cfg.math_backend, 'GPU:',os.environ.get('CUDA_VISIBLE_DEVICES'),flush=True)
        import threading
        import psutil
        stop=threading.Event(); proc=psutil.Process(); peak=[proc.memory_info().rss]; t0=time.perf_counter()
        def monitor():
            while not stop.wait(.1):
                peak[0]=max(peak[0],proc.memory_info().rss)
        thread=threading.Thread(target=monitor,daemon=True);thread.start()
        try:
            with threadpool_limits(cfg.threads):
                from pretrained import VISION,NLP,make_pretrained
                from independent import VISION_NEW,NLP_NEW,build,lstm_case
                def load_case():
                    if a.custom_model:return custom_case(a.custom_model,a.fit_npz,a.dev_npz,a.test_npz,a.task,a.shift_npz)
                    if name in VISION+NLP+VISION_NEW+NLP_NEW+('lstm_head',):
                        import fcntl
                        # Одна загрузка/экспорт общих файлов одновременно; квантование параллельно.
                        with (root/'pretrained_cache.lock').open('a') as lock:
                            fcntl.flock(lock,fcntl.LOCK_EX)
                            if name in VISION_NEW+NLP_NEW:return build(name,seed,cfg,root)
                            if name=='lstm_head':return lstm_case(seed,cfg,root)
                            return make_pretrained(name,seed,cfg,root)
                    return make_case(name,seed,cfg,root,archive)
                try:
                    case=load_case();run_root=benchmark_one(case,cfg,seed,out,a.methods,a.save_models)
                except Exception as exc:
                    if not (a.conv_cpu_fallback and name in VISION+VISION_NEW+('cnn','depthwise_cnn','custom') and os.environ.get('QUANT_INFER_PROVIDER')=='cuda' and any(k in str(exc).lower() for k in ('cudnn','ep_fail','cudaexecutionprovider'))):raise
                    os.environ['QUANT_INFER_PROVIDER']='cpu'
                    print('EXPLICIT FALLBACK: actual CNN CUDA error; restarting with CPU inference and CUDA math:',str(exc)[:500],flush=True)
                    case=load_case();run_root=benchmark_one(case,cfg,seed,out,a.methods,a.save_models)
            if not a.save_models and all((run_root/(m+'.json')).exists() for m in a.methods):
                for pattern in ('activation_*.npz','*.bank.npz','*.layer_*.npz'):
                    for artifact in run_root.glob(pattern):artifact.unlink(missing_ok=True)
            if not (run_root/'process_stats.json').exists():
                atomic(run_root/'process_stats.json',{'sampled_peak_rss_mib':peak[0]/2**20,'worker_wall_seconds':time.perf_counter()-t0,'rss_sample_interval_seconds':.1})
        finally:
            stop.set();thread.join()
        return
    # Отдельный процесс на физический GPU, один experiment/worker. Кэш загрузок защищён flock.
    from concurrent.futures import ThreadPoolExecutor,as_completed
    import queue
    if cfg.provider=='cuda':
        import torch
        devices=list(range(min(a.gpus,torch.cuda.device_count())))
        if not devices:raise RuntimeError('No CUDA devices')
        visible=os.environ.get('CUDA_VISIBLE_DEVICES')
        if visible and len(visible.split(','))>=len(devices):devices=[visible.split(',')[i] for i in devices]
    else:devices=[None]
    gpu_configs={}
    if cfg.provider=='cuda':
        for device in devices:
            env={**os.environ,'CUDA_VISIBLE_DEVICES':str(device)}
            proc=subprocess.run([sys.executable,'-u',str(root/'gpu_preflight.py')],env=env,capture_output=True,text=True)
            log=out/f'gpu_preflight_{device}.log';log.write_text(proc.stdout+'\n'+proc.stderr,encoding='utf-8')
            if proc.returncode:raise RuntimeError(f'GPU {device} preflight failed: {log}')
            gpu_configs[device]=next(line[len('GPU_PROBE '):] for line in proc.stdout.splitlines() if line.startswith('GPU_PROBE '))
            print('GPU',device,gpu_configs[device],flush=True)
    slots=queue.Queue()
    for device in devices:slots.put(device)
    failures=[]
    def run_job(name,seed):
        device=slots.get()
        try:
            args=[sys.executable,'-u',__file__,'--worker','--profile',a.profile,'--models',name,'--seeds',str(seed),'--methods',*a.methods,'--output',str(out),'--threads',str(cfg.threads),'--rows',str(cfg.rows),'--sdr-work',str(cfg.sdr_work),'--provider',cfg.provider,'--channel-cap',str(cfg.channel_cap),'--v2-layers',str(cfg.v2_layers),'--math-backend',cfg.math_backend,'--capture-group',str(cfg.capture_group)]
            if a.conv_cpu_fallback:args+=['--conv-cpu-fallback']
            if archive:args+=['--archive',archive]
            if a.save_models:args+=['--save-models']
            if a.custom_model:
                for arg,val in [('custom-model',a.custom_model),('fit-npz',a.fit_npz),('dev-npz',a.dev_npz),('test-npz',a.test_npz),('shift-npz',a.shift_npz),('task',a.task)]:
                    if val:args+=['--'+arg,val]
            print(f'\n=== {name}, seed={seed}, GPU={device} ===',flush=True)
            env={**os.environ,'PYTHONUTF8':'1',**{k:str(cfg.threads) for k in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','BLIS_NUM_THREADS')}}
            if device is not None:env.update(CUDA_VISIBLE_DEVICES=str(device),QUANT_GPU_CONFIG=gpu_configs[device])
            log=out/'logs'/f'{name}_{seed}.log';log.parent.mkdir(exist_ok=True)
            with log.open('w',encoding='utf-8') as f:
                proc=subprocess.Popen(args,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
                for line in proc.stdout:
                    f.write(line);f.flush()
                    if (' MSE=' in line or line.startswith('    policy') or 'EXPLICIT FALLBACK' in line or 'CACHE ' in line):print(f'[{name}:{seed}:GPU{device}] {line}',end='',flush=True)
                rc=proc.wait()
            return name,seed,rc
        finally:slots.put(device)
    with ThreadPoolExecutor(max_workers=len(devices)) as pool:
        jobs=[pool.submit(run_job,name,seed) for seed in seeds for name in models]
        for done,f in enumerate(as_completed(jobs),1):
            name,seed,rc=f.result();print(f'COMPLETED {done}/{len(jobs)} {name} seed={seed} rc={rc}',flush=True)
            if rc:failures.append({'model':name,'seed':seed,'returncode':rc})
            summarize(out)
    atomic(out/'failures.json',failures)
    if failures:print('Ошибки отдельных запусков:',failures,flush=True);sys.exit(1)

if __name__=='__main__':main()
