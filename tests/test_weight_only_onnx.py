"""Интеграционные проверки экспорта и инференса Standard."""

import numpy as np
import onnx
import pytest
from onnx import helper, TensorProto, numpy_helper

from quantlab.core import ModelError, reject_external_data
from quantlab.weight_only.engine import quantize_standard
from quantlab.engine import quantize
from tests.test_core import create_tiny_model


def test_standard_matmul_and_unique_runs(tmp_path):
    """Проверить реальные выходы, сохранение исходника и отдельные отчёты."""
    path = create_tiny_model(tmp_path / 'model.onnx')
    original = path.read_bytes()
    qpath, report = quantize_standard(path, [{'input': np.ones((2, 3), np.float32)}])
    assert report['output_fidelity']['status'] == 'measured'
    assert report['output_fidelity']['mse_by_output']['output'] < 1e-4
    assert report['coverage_percent'] == 100
    graph = onnx.load(qpath).graph
    assert [n.op_type for n in graph.node] == ['DequantizeLinear', 'MatMul']
    assert not any(t.name == 'W' for t in graph.initializer)
    assert any(t.data_type == TensorProto.INT8 for t in graph.initializer)
    second, report2 = quantize(path, 'standard')
    assert second.parent != qpath.parent and qpath.exists()
    assert (qpath.parent / 'report.json').is_file()
    assert report2['output_fidelity']['status'] == 'not_run_no_inputs'
    assert path.read_bytes() == original


@pytest.mark.parametrize('op,shape,xshape,attrs', [
    ('Gemm', (4, 3), (3, 2), {'transA': 1, 'transB': 1, 'alpha': 0.5, 'beta': 2.0}),
    ('Conv', (4, 2, 2), (1, 4, 4), {'group': 2}),
    ('Conv', (4, 1, 2, 2), (1, 4, 4, 4), {'group': 4}),
    ('Conv', (4, 2, 2, 2, 2), (1, 4, 4, 4, 4), {'group': 2}),
])
def test_standard_layout_inference(tmp_path, op, shape, xshape, attrs):
    """Проверить Gemm с транспонированием и групповые Conv разных рангов."""
    rng = np.random.default_rng(123)
    w = rng.uniform(-0.5, 0.5, shape).astype(np.float32)
    x = rng.uniform(-0.5, 0.5, xshape).astype(np.float32)
    yshape = [2, 4] if op == 'Gemm' else [1, 4, *([3] * (len(shape) - 2))]
    graph = helper.make_graph([helper.make_node(op, ['X', 'W', 'B'], ['Y'], **attrs)],
        'layout', [helper.make_tensor_value_info('X', TensorProto.FLOAT, xshape)],
        [helper.make_tensor_value_info('Y', TensorProto.FLOAT, yshape)],
        [numpy_helper.from_array(w, 'W'), numpy_helper.from_array(np.ones(4, np.float32), 'B')])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 17)], ir_version=10)
    path = tmp_path / 'layout.onnx'
    onnx.save(model, path)
    output, report = quantize_standard(path, [{'X': x}])
    assert report['layers'][0]['axis'] == 0
    assert report['output_fidelity']['mse_by_output']['Y'] < 1e-4
    assert onnx.load(output).graph.node[1].op_type == op


def test_shared_weights_rejected(tmp_path):
    """Не переписывать веса с несколькими потребителями."""
    path = create_tiny_model(tmp_path / 'shared.onnx')
    m = onnx.load(path)
    m.graph.node.append(helper.make_node('MatMul', ['input', 'W'], ['other']))
    onnx.save(m, path)
    with pytest.raises(ModelError, match='shared_or_public_weight'):
        quantize_standard(path)


def test_external_constant_rejected():
    """Внешние данные запрещены и внутри атрибута Constant."""
    t = numpy_helper.from_array(np.ones(2, np.float32))
    t.data_location = TensorProto.EXTERNAL
    t.external_data.add(key='location', value='../weights.bin')
    n = helper.make_node('Constant', [], ['Y'], value=t)
    with pytest.raises(ModelError):
        reject_external_data(n)
