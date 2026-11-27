"""Адаптеры постоянных FP32-весов ONNX; без зависимости от архитектуры модели."""
from dataclasses import dataclass
from collections import Counter
import copy, math
import numpy as np


@dataclass
class Target:
    weight: str
    node: str
    op: str
    input: str
    output: str
    original: np.ndarray
    attrs: dict
    bias: np.ndarray | None = None

    @property
    def axis(self):
        return 0 if self.op == 'Conv' or self.attrs.get('transB', 0) else 1

    @property
    def groups(self):
        return int(self.attrs.get('group', 1)) if self.op == 'Conv' else 1

    def matrices(self):
        w = self.original
        if self.op == 'Conv':
            return [a.reshape(a.shape[0], -1).T.copy() for a in np.split(w, self.groups, axis=0)]
        return [w.T.copy() if self.axis == 0 else w.copy()]

    def pack(self, pairs):
        q = np.concatenate([a for a, _ in pairs], axis=1)
        s = np.concatenate([b for _, b in pairs]).astype(np.float32)
        if self.op == 'Conv': q = q.T.reshape(self.original.shape)
        elif self.axis == 0: q = q.T
        return np.ascontiguousarray(q), s

    def canonical(self, q, s):
        """Общий двумерный интерфейс для V2, включая групповые Conv."""
        if self.op == 'Conv': return q.reshape(q.shape[0], -1).T.copy(), s
        return (q.T.copy() if self.axis == 0 else q.copy()), s

    def stored(self, pair):
        q,s = pair
        if self.op == 'Conv': q=q.T.reshape(self.original.shape)
        elif self.axis == 0: q=q.T
        return np.ascontiguousarray(q),s


def discover(model):
    import onnx
    from onnx import numpy_helper, helper
    if not any(o.domain in ('','ai.onnx') and o.version >= 13 for o in model.opset_import):
        raise ValueError('Нужен стандартный ONNX opset >= 13 для per-axis DQ.')
    g=model.graph
    init={t.name:t for t in g.initializer}
    uses=Counter(i for n in g.node for i in n.input if i)
    # Вложенные графы могут ссылаться на внешние инициализаторы.
    def nested_refs(graph):
        for n in graph.node:
            for a in n.attribute:
                if a.type == onnx.AttributeProto.GRAPH:
                    uses.update(i for nn in a.g.node for i in nn.input if i)
                    uses.update(o.name for o in a.g.output)
                    nested_refs(a.g)
                elif a.type == onnx.AttributeProto.GRAPHS:
                    for sub in a.graphs:
                        uses.update(i for nn in sub.node for i in nn.input if i)
                        uses.update(o.name for o in sub.output)
                        nested_refs(sub)
    nested_refs(g)
    excluded={v.name for v in list(g.input)+list(g.output)}
    aliases={n.output[0]:n.input[0] for n in g.node if n.op_type=='Identity' and n.domain in ('','ai.onnx') and len(n.input)==len(n.output)==1}
    def constant(name):
        seen=set()
        while name in aliases and name not in seen and name not in excluded:
            seen.add(name);name=aliases[name]
        return init.get(name) if name not in {v.name for v in g.input} else None
    targets=[]; skipped=[]
    for index,n in enumerate(g.node):
        if n.op_type not in ('MatMul','Gemm','Conv') or n.domain not in ('','ai.onnx'): continue
        name=n.name or f'{n.op_type}_{index}'
        reason=None
        if len(n.input)<2 or n.input[1] not in init: reason='nonconstant_rhs'
        else:
            t=init[n.input[1]]
            if t.name in excluded or uses[t.name]!=1: reason='shared_or_public_weight'
            elif t.data_type!=onnx.TensorProto.FLOAT: reason='not_fp32'
            elif n.op_type in ('MatMul','Gemm') and len(t.dims)!=2: reason='not_rank2'
            elif n.op_type=='Conv' and len(t.dims) not in (3,4,5): reason='unsupported_conv_rank'
            elif not all(t.dims): reason='empty_weight'
        if reason:
            skipped.append({'node':name,'reason':reason}); continue
        attrs={a.name:helper.get_attribute_value(a) for a in n.attribute}
        bias=None
        if n.op_type in ('Gemm','Conv') and len(n.input)>2 and n.input[2]:
            bt=constant(n.input[2])
            if bt is None:
                skipped.append({'node':name,'reason':'nonconstant_bias'}); continue
            bias=numpy_helper.to_array(bt)
        w=numpy_helper.to_array(t).copy()
        if not np.isfinite(w).all():
            skipped.append({'node':name,'reason':'nonfinite_weight'}); continue
        target=Target(t.name,name,n.op_type,n.input[0],n.output[0],w,attrs,bias)
        if target.groups<1 or (n.op_type=='Conv' and w.shape[0]%target.groups):
            skipped.append({'node':name,'reason':'invalid_groups'}); continue
        targets.append(target)
    return targets,skipped


def conv_rows(x, y, target, idx):
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
    for group,mat in enumerate(target.matrices()):
        channels=np.arange(group*cin,(group+1)*cin).reshape((1,cin)+(1,)*dim)
        rows=xp[(batch_idx,channels,*spatial)].reshape(len(idx),-1)
        ref=y[(coords[:,0,None],np.arange(group*cout,(group+1)*cout)[None,:],*[coords[:,j+1,None] for j in range(dim)])]
        calc=rows@mat
        if target.bias is not None: calc=calc+target.bias[group*cout:(group+1)*cout]
        check_local(calc,ref)
        result.append(rows)
    return result


def check_local(calc, ref):
    if not np.isfinite(calc).all() or not np.isfinite(ref).all(): raise ValueError('nonfinite activations')
    # Различный порядок FP32-накоплений ORT и NumPy допустим.
    scale=max(1.0,float(np.max(np.abs(ref))))
    if float(np.max(np.abs(calc-ref)))>2e-4*scale:
        raise ValueError(f'FP32 adapter mismatch: max={np.max(np.abs(calc-ref)):.3g}, scale={scale:.3g}')


def sample_rows(x,y,t,cap,rng,mask=None):
    if x.dtype!=np.float32 or y.dtype!=np.float32: raise ValueError('FP32 activations required')
    if t.op=='Conv':
        n=y.shape[0]*math.prod(y.shape[2:]); idx=rng.choice(n,min(cap,n),replace=False)
        return conv_rows(x,y,t,idx),n
    if t.op=='Gemm' and t.attrs.get('transA',0): x=x.T
    m=t.matrices()[0]; rows=x.reshape(-1,m.shape[0]); ref=y.reshape(-1,m.shape[1])
    if len(rows)!=len(ref) or x.shape[-1]!=m.shape[0]: raise ValueError('linear dimensions')
    valid=np.arange(len(rows))
    if mask is not None and x.ndim>=3 and tuple(x.shape[:-1])==tuple(mask.shape):
        valid=valid[np.asarray(mask).reshape(-1)!=0]
    if not len(valid): return [np.empty((0,m.shape[0]),np.float32)],0
    idx=rng.choice(valid,min(cap,len(valid)),replace=False)
    calc=rows[idx]@m
    if t.op=='Gemm':
        calc=calc*float(t.attrs.get('alpha',1.))
        if t.bias is not None: calc=calc+float(t.attrs.get('beta',1.))*np.broadcast_to(t.bias,y.shape).reshape(ref.shape)[idx]
    check_local(calc,ref[idx]); return [rows[idx].copy()],len(valid)


def with_outputs(model, names):
    import onnx
    from onnx import helper
    p=copy.deepcopy(model)
    try: inferred=onnx.shape_inference.infer_shapes(p)
    except Exception: inferred=p
    info={v.name:v for v in list(inferred.graph.value_info)+list(inferred.graph.input)+list(inferred.graph.output)}
    known={v.name for v in p.graph.output}
    for name in dict.fromkeys(names):
        if name not in known:
            p.graph.output.append(info.get(name,helper.make_tensor_value_info(name,onnx.TensorProto.FLOAT,None)))
    return p


def export(model, targets, bank):
    """Одна DQ на исходный тензор, первоначальные имена выхода и порядок узлов сохранены."""
    import onnx
    from onnx import helper,numpy_helper
    p=copy.deepcopy(model); g=p.graph
    known={v.name for v in list(g.input)+list(g.output)+list(g.value_info)} | {i.name for i in g.initializer} | {v for n in g.node for v in list(n.input)+list(n.output)}
    def unique(name):
        while name in known: name+='_'
        known.add(name);return name
    dq=[]; originals=set()
    for t in targets:
        if t.weight not in bank: continue
        q,s=t.stored(bank[t.weight]); assert q.shape==t.original.shape and s.shape==(q.shape[t.axis],)
        if q.dtype!=np.int8 or s.dtype!=np.float32 or not np.all(np.isfinite(s)&(s>0)) or np.any(q==-128): raise ValueError('invalid INT8 representation')
        qname=unique(t.weight+'__int8'); sname=unique(t.weight+'__scale')
        g.initializer.extend([numpy_helper.from_array(q,qname),numpy_helper.from_array(s,sname)])
        dq.append(helper.make_node('DequantizeLinear',[qname,sname],[t.weight],axis=t.axis,name=unique(t.weight+'__DQ')))
        originals.add(t.weight)
    keep=[i for i in g.initializer if i.name not in originals]; del g.initializer[:]; g.initializer.extend(keep)
    nodes=list(g.node); del g.node[:];g.node.extend(dq+nodes)
    onnx.checker.check_model(p)
    return p
