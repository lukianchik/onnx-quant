"""Адаптеры постоянных FP32/FP16-весов ONNX; без зависимости от архитектуры модели."""
from dataclasses import dataclass
from collections import Counter
import copy
import numpy as np


def combine_groups(pairs):
    """Объединить группы в формате [вход, выход] для поиска и экспорта."""
    if len(pairs) == 1:
        return pairs[0]
    codes, scales = zip(*pairs, strict=True)
    return np.concatenate(codes, axis=1), np.concatenate(scales)


@dataclass
class Target:
    """Адаптер одного постоянного весового тензора."""
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
        """Ось выходных каналов в исходном тензоре."""
        return 0 if self.op == 'Conv' or self.attrs.get('transB', 0) else 1

    @property
    def groups(self):
        """Число групп свёртки."""
        return int(self.attrs.get('group', 1)) if self.op == 'Conv' else 1

    def matrices(self):
        """Представить веса матрицами входы на выходы."""
        w = self.original.astype(np.float32, copy=False)
        if self.op == 'Conv':
            return [a.reshape(a.shape[0], -1).T.copy() for a in np.split(w, self.groups, axis=0)]
        return [np.ascontiguousarray(w.T) if self.axis == 0 else w]

    def pack(self, pairs):
        """Собрать коды групп и масштабы в исходном формате."""
        return self.stored(combine_groups(pairs))

    def canonical(self, q, s):
        """Общий двумерный интерфейс для V2, включая групповые Conv."""
        if self.op == 'Conv': return q.reshape(q.shape[0], -1).T.copy(), s
        return (q.T.copy() if self.axis == 0 else q.copy()), s

    def stored(self, pair):
        """Восстановить расположение кодов в исходном тензоре."""
        q,s = pair
        if self.op == 'Conv': q=q.T.reshape(self.original.shape)
        elif self.axis == 0: q=q.T
        return np.ascontiguousarray(q),s


def supports_weight_only(model):
    """Проверить версию стандартного opset для поканальной деквантизации."""
    return any(o.domain in ('', 'ai.onnx') and o.version >= 13 for o in model.opset_import)


def discover(model):
    """Подготовить адаптеры весов для квантования."""
    if not supports_weight_only(model):
        raise ValueError('Нужен стандартный ONNX opset >= 13 для per-axis DQ.')
    skipped = []
    return list(iter_targets(model, skipped)), skipped


def iter_targets(model, skipped):
    """Выдавать поддерживаемые веса по одному, записывая причины пропуска."""
    import onnx
    from onnx import numpy_helper, helper
    g=model.graph
    init={t.name:t for t in g.initializer}
    uses=Counter(i for n in g.node for i in n.input if i)
    # Вложенные графы могут ссылаться на внешние инициализаторы.
    def nested_refs(graph):
        """Учесть ссылки из вложенных графов."""
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
    inputs = {v.name for v in g.input}
    excluded = inputs | {v.name for v in g.output}
    aliases={n.output[0]:n.input[0] for n in g.node if n.op_type=='Identity' and n.domain in ('','ai.onnx') and len(n.input)==len(n.output)==1}
    def constant(name):
        """Разрешить безопасный Identity-псевдоним константы."""
        seen=set()
        while name in aliases and name not in seen and name not in excluded:
            seen.add(name);name=aliases[name]
        return init.get(name) if name not in inputs else None
    for index,n in enumerate(g.node):
        if n.op_type not in ('MatMul','Gemm','Conv') or n.domain not in ('','ai.onnx'): continue
        name=n.name or f'{n.op_type}_{index}'
        reason=None
        if len(n.input)<2 or n.input[1] not in init: reason='nonconstant_rhs'
        else:
            t=init[n.input[1]]
            if t.name in excluded or uses[t.name]!=1: reason='shared_or_public_weight'
            elif t.data_type not in (onnx.TensorProto.FLOAT, onnx.TensorProto.FLOAT16): reason='not_fp32_or_fp16'
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
        w=numpy_helper.to_array(t)
        if not np.isfinite(w).all():
            skipped.append({'node':name,'reason':'nonfinite_weight'}); continue
        target=Target(t.name,name,n.op_type,n.input[0],n.output[0],w,attrs,bias)
        if target.groups<1 or (n.op_type=='Conv' and w.shape[0]%target.groups):
            skipped.append({'node':name,'reason':'invalid_groups'}); continue
        yield target


def fp16_safe_scales(q, s, axis):
    """Ограничить масштаб так, чтобы фактические коды не переполняли FP16."""
    axes = tuple(i for i in range(q.ndim) if i != axis)
    peak = np.maximum(q.max(axis=axes), -q.min(axis=axes)).astype(np.float64)
    cap = float(np.finfo(np.float16).max) / np.maximum(peak, 1.)
    # Нулевые каналы сохраняют исходный положительный масштаб.
    limited = np.where(peak > 0, np.minimum(s, cap), s).astype(np.float32)
    too_high = (peak > 0) & (peak * limited.astype(np.float64) > float(np.finfo(np.float16).max))
    limited[too_high] = np.nextafter(limited[too_high], np.float32(0))
    return limited


def export(model, targets, bank):
    """Одна DQ на исходный тензор, первоначальные имена выхода и порядок узлов сохранены."""
    import onnx
    from onnx import helper,numpy_helper
    p=copy.deepcopy(model); g=p.graph
    known={v.name for v in list(g.input)+list(g.output)+list(g.value_info)} | {i.name for i in g.initializer} | {v for n in g.node for v in list(n.input)+list(n.output)}
    def unique(name):
        """Выделить свободное имя значения или узла."""
        while name in known: name+='_'
        known.add(name);return name
    known.update(n.name for n in g.node if n.name)
    known.update(t.values.name for t in g.sparse_initializer)
    dq=[]; originals=set()
    for t in targets:
        if t.weight not in bank: continue
        q,s=t.stored(bank[t.weight])
        if q.shape!=t.original.shape or s.shape!=(q.shape[t.axis],):
            raise ValueError("invalid INT8 shape")
        if q.dtype!=np.int8 or s.dtype!=np.float32 or not np.all(np.isfinite(s)&(s>0)) or np.any(q==-128): raise ValueError('invalid INT8 representation')
        qname=unique(t.weight+'__int8'); sname=unique(t.weight+'__scale')
        half = t.original.dtype == np.float16
        if half:
            s = fp16_safe_scales(q, s, t.axis)
        g.initializer.extend([numpy_helper.from_array(q,qname),numpy_helper.from_array(s,sname)])
        restored = unique(t.weight+'__dq_fp32') if half else t.weight
        dq.append(helper.make_node('DequantizeLinear',[qname,sname],[restored],axis=t.axis,name=unique(t.weight+'__DQ')))
        if half:
            dq.append(helper.make_node('Cast', [restored], [t.weight], to=onnx.TensorProto.FLOAT16,
                                       name=unique(t.weight+'__CastFP16')))
        originals.add(t.weight)
    keep=[i for i in g.initializer if i.name not in originals]; del g.initializer[:]; g.initializer.extend(keep)
    nodes=list(g.node); del g.node[:];g.node.extend(dq+nodes)
    onnx.checker.check_model(p)
    return p
