"""Захват и валидация активаций операторов на исходном FP32/FP16-графе."""
import copy
import math
import numpy as np

def conv_rows(x, y, target, matrices, idx):
    """Извлечение только выбранных патчей, в порядке C,k1,...; без полного im2col."""
    a=target.attrs; w=target.original; k=w.shape[2:]; dim=len(k)
    if x.ndim!=dim+2 or y.ndim!=dim+2: raise ValueError('Conv rank')
    stride=tuple(a.get('strides',[1]*dim)); dil=tuple(a.get('dilations',[1]*dim))
    eff=tuple((z-1)*d+1 for z,d in zip(k,dil))
    auto=a.get('auto_pad',b'NOTSET'); auto=auto.decode() if isinstance(auto,bytes) else auto
    if auto in ('SAME_UPPER','SAME_LOWER'):
        total=[max(0,(o-1)*s+e-i) for o,s,e,i in zip(y.shape[2:],stride,eff,x.shape[2:])]
        left=[(p+int(auto=='SAME_LOWER'))//2 for p in total]; right=[p-l for p,l in zip(total,left)]
    elif auto=='VALID': left=right=[0]*dim
    elif auto=='NOTSET':
        pads=list(a.get('pads',[0]*(2*dim))); left,right=pads[:dim],pads[dim:]
    else: raise ValueError('auto_pad')
    if any(p<0 for p in left+right): raise ValueError('negative pad')
    xp=np.pad(x,[(0,0),(0,0)]+list(zip(left,right))) if any(left+right) else x
    coords=np.array(np.unravel_index(idx,(y.shape[0],*y.shape[2:]))).T
    cin=w.shape[1]; cout=w.shape[0]//target.groups
    if x.shape[1]!=cin*target.groups: raise ValueError('Conv channels')
    result=[]
    # Векторизация выбранных патчей, без полного im2col и копии полного y.
    rank=2+dim
    batch_idx=coords[:,0].reshape((len(idx),)+(1,)*(dim+1))
    spatial=[]
    for j,(size,st,di) in enumerate(zip(k,stride,dil)):
        shape=[1]*rank;shape[0]=len(idx);base=(coords[:,j+1]*st).reshape(shape)
        shape=[1]*rank;shape[2+j]=size;spatial.append(base+np.arange(size).reshape(shape)*di)
    for group,mat in enumerate(matrices):
        channels=np.arange(group*cin,(group+1)*cin).reshape((1,cin)+(1,)*dim)
        rows=xp[(batch_idx,channels,*spatial)].reshape(len(idx),-1).astype(np.float32, copy=False)
        ref=y[(coords[:,0,None],np.arange(group*cout,(group+1)*cout)[None,:],*[coords[:,j+1,None] for j in range(dim)])]
        calc=rows@mat
        if target.bias is not None: calc=calc+target.bias[group*cout:(group+1)*cout]
        check_local(calc,ref)
        result.append(rows)
    return result


def check_local(calc, ref):
    """Проверить согласованность адаптера с ORT с учётом типа выхода."""
    if not np.isfinite(calc).all() or not np.isfinite(ref).all(): raise ValueError('nonfinite activations')
    # Для FP16 дополнительно учитывается округление результата в half.
    tolerance = 2e-3 if ref.dtype == np.float16 else 2e-4
    scale=max(1.0,float(np.max(np.abs(ref))))
    error = float(np.max(np.abs(calc.astype(np.float32, copy=False)-ref.astype(np.float32, copy=False))))
    if error > tolerance*scale:
        raise ValueError(f'{ref.dtype} adapter mismatch: max={error:.3g}, scale={scale:.3g}')


def sample_rows(x, y, t, matrices, cap, rng, mask=None):
    """Выбрать строки активаций и проверить локальный выход."""
    if x.dtype not in (np.float32, np.float16) or y.dtype != x.dtype or x.dtype != t.original.dtype:
        raise ValueError('Matching FP32/FP16 activations and weights required')
    if t.op=='Conv':
        n=y.shape[0]*math.prod(y.shape[2:]); idx=rng.choice(n,min(cap,n),replace=False)
        return (conv_rows(x, y, t, matrices, idx) if n else []), n
    if t.op=='Gemm' and t.attrs.get('transA',0): x=x.T
    m=matrices[0]; rows=x.reshape(-1,m.shape[0]); ref=y.reshape(-1,m.shape[1])
    if len(rows)!=len(ref) or x.shape[-1]!=m.shape[0]: raise ValueError('linear dimensions')
    valid=np.arange(len(rows))
    if mask is not None and x.ndim>=3 and tuple(x.shape[:-1])==tuple(mask.shape):
        valid=valid[np.asarray(mask).reshape(-1)!=0]
    if not len(valid): return [np.empty((0,m.shape[0]),np.float32)],0
    idx=rng.choice(valid,min(cap,len(valid)),replace=False)
    selected = rows[idx].astype(np.float32, copy=False)
    calc = selected @ m
    if t.op=='Gemm':
        calc=calc*float(t.attrs.get('alpha',1.))
        if t.bias is not None: calc=calc+float(t.attrs.get('beta',1.))*np.broadcast_to(t.bias,y.shape).reshape(ref.shape)[idx].astype(np.float32, copy=False)
    check_local(calc,ref[idx]); return [selected],len(valid)


def activation_info(model, targets):
    """Определить метаданные наблюдаемых тензоров один раз для всех слоёв."""
    import onnx
    try:
        inferred = onnx.shape_inference.infer_shapes(model)
    except Exception:
        inferred = model
    names = {name for target in targets for name in (target.input, target.output)}
    graph = inferred.graph
    return {v.name: copy.deepcopy(v) for values in (graph.value_info, graph.input, graph.output)
            for v in values if v.name in names}


def with_outputs(model, names, info, dtype=None):
    """Сериализовать граф с дополнительными выходами и восстановить исходный список."""
    import onnx
    from onnx import helper
    outputs = model.graph.output
    known = {v.name for v in outputs}
    count = len(outputs)
    try:
        for name in names:
            if name not in known:
                outputs.append(info.get(name, helper.make_tensor_value_info(name, dtype or onnx.TensorProto.FLOAT, None)))
        return model.SerializeToString()
    finally:
        del outputs[count:]


def capture(model, target, matrices, info, fit, dev, session_factory, seed, cap=512):
    """Собрать независимые ограниченные пулы строк по алгоритму reservoir v4."""
    names = list(dict.fromkeys((target.input, target.output)))
    import onnx
    dtype = onnx.TensorProto.FLOAT16 if target.original.dtype == np.float16 else onnx.TensorProto.FLOAT
    session = session_factory(with_outputs(model, names, info, dtype))
    def collect(feeds, offset):
        """Сохранить не более cap строк на группу."""
        rng = np.random.default_rng(seed + offset)
        pools, keys = None, np.empty(0)
        for feed in feeds:
            values = dict(zip(names, session.run(names, feed), strict=True))
            rows, n = sample_rows(values[target.input], values[target.output], target, matrices,
                                  cap, rng, feed.get('attention_mask'))
            if not n:
                continue
            priorities, prev = [], 0.
            for j in range(len(rows[0])):
                prev += (1-prev)*rng.beta(1,n-j)
                priorities.append(prev)
            priorities = np.array(priorities)
            rng.shuffle(priorities)
            merged = np.concatenate([keys, priorities])
            keep = np.argsort(merged, kind='stable')[:cap]
            pools = ([np.concatenate([a,b], axis=0)[keep] for a,b in zip(pools,rows,strict=True)]
                     if pools is not None else [a[keep] for a in rows])
            keys = merged[keep]
        if pools is None or len(keys) < 2:
            raise ValueError('Недостаточно валидных строк активаций')
        return pools
    return collect(fit, 0), collect(dev, 1)
