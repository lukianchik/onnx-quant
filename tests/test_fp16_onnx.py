"""Интеграционные проверки FP16 на реальном CPU ONNX Runtime."""
import json

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from quantlab.core import analyze
from quantlab.engine import quantize
from quantlab.evaluation import evaluate_dataset
from quantlab.runtime import session
from quantlab.weight_only.graph import discover, export
from quantlab.weight_only.minmax import minmax


def make_model(path, op='MatMul', opset=13):
    """Построить независимую FP16-модель с постоянными весами."""
    rng = np.random.default_rng(42)
    attrs = {}
    if op == 'Conv':
        shape, xshape, yshape = (4, 2, 2, 2), ['batch', 4, 4, 4], ['batch', 4, 3, 3]
        attrs = {'group': 2}
    elif op == 'Gemm':
        shape, xshape, yshape = (4, 3), ['batch', 3], ['batch', 4]
        attrs = {'transB': 1, 'alpha': 0.5, 'beta': 2.0}
    else:
        shape, xshape, yshape = (3, 4), ['batch', 3], ['batch', 4]
    w = rng.uniform(-0.5, 0.5, shape).astype(np.float16)
    initializers = [numpy_helper.from_array(w, 'W')]
    inputs = ['X', 'W']
    if op != 'MatMul':
        initializers.append(numpy_helper.from_array(np.full(4, 0.125, np.float16), 'B'))
        inputs.append('B')
    graph = helper.make_graph([helper.make_node(op, inputs, ['Y'], **attrs)], 'half',
        [helper.make_tensor_value_info('X', TensorProto.FLOAT16, xshape)],
        [helper.make_tensor_value_info('Y', TensorProto.FLOAT16, yshape)], initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', opset)], ir_version=10)
    onnx.checker.check_model(model)
    onnx.save(model, path)
    return tuple([2, *xshape[1:]])


def check_graph(path, axis):
    """Проверить типы payload, ось и сохранение входного/выходного FP16."""
    onnx.checker.check_model(str(path), full_check=True)
    graph = onnx.load(path).graph
    assert [n.op_type for n in graph.node[:2]] == ['DequantizeLinear', 'Cast']
    dq, cast = graph.node[:2]
    assert cast.input[0] == dq.output[0] and cast.output[0] == 'W'
    assert helper.get_attribute_value(dq.attribute[0]) == axis
    assert helper.get_attribute_value(cast.attribute[0]) == TensorProto.FLOAT16
    tensors = {i.name: i for i in graph.initializer}
    assert 'W' not in tensors
    assert tensors[dq.input[0]].data_type == TensorProto.INT8
    assert tensors[dq.input[1]].data_type == TensorProto.FLOAT
    assert graph.input[0].type.tensor_type.elem_type == TensorProto.FLOAT16
    assert graph.output[0].type.tensor_type.elem_type == TensorProto.FLOAT16


@pytest.mark.parametrize('mode,task', [('standard','generic'), ('fast','generic'),
    ('balanced','generic'), ('quality','generic'), ('quality','classification')])
def test_fp16_modes(tmp_path, mode, task):
    """Проверить четыре режима, фактический инференс и глобальный guard."""
    path = tmp_path / 'half.onnx'
    shape = make_model(path)
    original = path.read_bytes()
    rng = np.random.default_rng(11)
    feeds = [{'X': rng.uniform(-1, 1, shape).tolist()} for _ in range(16)]
    cal = tmp_path / 'cal.json'
    cal.write_text(json.dumps(feeds), encoding='utf-8')
    assert analyze(path)['candidates'][0]['source_dtype'] == 'float16'
    out, report = quantize(path, mode, cal, task=task)
    check_graph(out, 1)
    assert path.read_bytes() == original
    assert report['layers'][0]['source_dtype'] == 'float16'
    assert report['fp32_initializer_parameters'] == 0
    assert report['fp16_initializer_parameters'] == 12
    assert report['coverage_percent'] == 100
    assert report['quantized_operators'] == {'DequantizeLinear': 1, 'Cast': 1}
    assert report['output_fidelity']['mse_by_output']['Y'] < 1e-4
    a = session(path).run(None, {'X': np.asarray(feeds[0]['X'], dtype=np.float16)})[0]
    b = session(out).run(None, {'X': np.asarray(feeds[0]['X'], dtype=np.float16)})[0]
    assert a.dtype == b.dtype == np.float16
    assert np.max(np.abs(a.astype(np.float32)-b.astype(np.float32))) < 0.03
    if mode != 'standard':
        assert report['output_fidelity']['mse_by_output']['Y'] <= report['global_guard']['baseline']['mse_by_output']['Y']
    dataset = tmp_path / 'evaluation.json'
    xs = np.eye(3, dtype=np.float16)
    labels = session(path).run(None, {'X': xs})[0].argmax(axis=1)
    dataset.write_text(json.dumps([{'inputs': {'X': [x.tolist()]}, 'label': int(label)}
        for x, label in zip(xs, labels, strict=True)]), encoding='utf-8')
    metrics = evaluate_dataset(path, out, dataset, is_demo=False)
    assert metrics['fp32']['accuracy'] == 1.0 and metrics['int8']['accuracy'] >= 2/3


@pytest.mark.parametrize('op', ['Gemm', 'Conv'])
def test_fp16_layout_export(tmp_path, op):
    """Проверить FP16 Gemm и grouped Conv, включая транспонирование и bias."""
    path = tmp_path / 'half.onnx'
    make_model(path, op, opset=17)
    model = onnx.load(path)
    targets, skipped = discover(model)
    assert not skipped and len(targets) == 1
    t = targets[0]
    q, s = t.pack([minmax(w) for w in t.matrices()])
    output = tmp_path / 'out.onnx'
    onnx.save(export(model, targets, {t.weight: t.canonical(q, s)}), output)
    check_graph(output, 0)
    # CPU execution of these operators depends on the provider's FP16 kernels.
    # The structural contract is tested independently; MatMul is exercised end-to-end above.


def test_mixed_precision_export(tmp_path):
    """Не добавлять FP16 Cast к FP32-весам в смешанном графе."""
    path = tmp_path / 'mixed.onnx'
    make_model(path)
    model = onnx.load(path)
    graph = model.graph
    graph.initializer.append(numpy_helper.from_array(np.ones((4, 2), np.float32), 'W2'))
    graph.node.extend([helper.make_node('Cast', ['Y'], ['Z'], to=TensorProto.FLOAT),
                       helper.make_node('MatMul', ['Z', 'W2'], ['O'])])
    graph.output.append(helper.make_tensor_value_info('O', TensorProto.FLOAT, ['batch', 2]))
    onnx.save(model, path)
    output, report = quantize(path, 'standard')
    assert report['selected_count'] == 2 and report['coverage_percent'] == 100
    assert report['fp32_initializer_parameters'] == 8 and report['fp16_initializer_parameters'] == 12
    assert report['quantized_operators'] == {'DequantizeLinear': 2, 'Cast': 1}
    actual = session(output).run(None, {'X': np.ones((2, 3), np.float16)})
    assert [v.dtype for v in actual] == [np.float16, np.float32]
