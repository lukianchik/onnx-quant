"""Быстрые тесты на небольшой синтетической линейной ONNX-модели."""

import json
from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from quantlab.core import ModelError, analyze, stage_model
from quantlab.engine import quantize
from quantlab.inference import compare_inference


def create_tiny_model(path: Path):
    weight = np.array([[0.2, -0.5], [0.7, 0.4], [0.1, 0.8]], dtype=np.float32)
    graph = helper.make_graph(
        [helper.make_node('MatMul', ['input', 'W'], ['output'], name='dense')],
        'tiny',
        [helper.make_tensor_value_info('input', TensorProto.FLOAT, ['batch', 3])],
        [helper.make_tensor_value_info('output', TensorProto.FLOAT, ['batch', 2])],
        [numpy_helper.from_array(weight, 'W')],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('', 17)])
    model.ir_version = 10
    onnx.save(model, str(path))
    return path


def test_staging_analysis_and_standard(tmp_path):
    path = create_tiny_model(tmp_path / 'model.onnx')
    prepared = stage_model(path, tmp_path / 'runs')
    report = analyze(prepared)
    assert [c['node'] for c in report['candidates']] == ['dense']
    assert 'dynamic_candidates' not in report and 'static_candidates' not in report
    assert report['candidates'][0]['axis'] == 1
    qpath, metrics = quantize(prepared, 'standard')
    assert qpath.is_file()
    assert metrics['onnx_valid'] and metrics['selected_count'] == 1
    outputs = compare_inference(prepared, qpath, 'JSON-тензоры', '', '{"input": [[1, 2, 3]]}')
    assert outputs['comparison']['max_absolute_output_difference'] < 0.05


def test_standard_with_numeric_calibration(tmp_path):
    path = stage_model(create_tiny_model(tmp_path / 'model.onnx'), tmp_path / 'runs')
    samples = [{'input': [[float(i), 0.5, -1.0]]} for i in range(1, 12)]
    calibration = tmp_path / 'calibration.json'
    calibration.write_text(json.dumps(samples), encoding='utf-8')
    qpath, metrics = quantize(path, 'standard', calibration)
    assert metrics['onnxruntime_load_ok'] and qpath.is_file()
    assert metrics['output_fidelity']['status'] == 'measured'
    assert metrics['output_fidelity']['batches'] == len(samples)


def test_invalid_upload_rejected(tmp_path):
    bad = tmp_path / 'bad.onnx'
    bad.write_bytes(b'not an onnx model')
    try:
        stage_model(bad, tmp_path / 'runs')
    except Exception:
        pass
    else:
        raise AssertionError('Corrupt input must not be accepted')

def test_standard_runs_keep_their_own_report_and_remove_partial_results(tmp_path):
    """Повторные конвертации и ошибки не повреждают завершённый запуск."""
    path = create_tiny_model(tmp_path / 'quantized_standard.onnx')
    source_bytes = path.read_bytes()
    first, _ = quantize(path, 'standard')
    report = (first.parent / 'report.json').read_bytes()
    second, _ = quantize(path, 'standard')
    with pytest.raises(ModelError, match='JSON'):
        quantize(path, 'fast')
    assert first.parent != second.parent
    assert path.read_bytes() == source_bytes
    assert (first.parent / 'report.json').read_bytes() == report
    assert set(tmp_path.iterdir()) == {path, first.parent, second.parent}


def test_analysis_matches_shared_weights_and_older_opsets(tmp_path):
    """Пропускать общие веса и модели со старым opset."""
    path = create_tiny_model(tmp_path / 'shared.onnx')
    model = onnx.load(path)
    model.graph.node.append(helper.make_node('MatMul', ['input', 'W'], ['extra'], name='other'))
    onnx.save(model, path)
    report = analyze(path)
    assert report['candidates'] == []
    assert report['skipped'] == {'shared_or_public_weight': 2}
    model.opset_import[0].version = 12
    onnx.save(model, path)
    assert analyze(path)['skipped'] == {'weight_only_requires_opset_13': 1}


@pytest.mark.parametrize('mode', ['dynamic', 'static'])
def test_removed_modes_rejected(tmp_path, mode):
    """Не запускать удалённые конвертеры даже через Python API."""
    with pytest.raises(ModelError, match='Неизвестный режим'):
        quantize(tmp_path / 'unused.onnx', mode)
