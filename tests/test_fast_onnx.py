"""Интеграционные проверки Fast на реальном ONNX Runtime."""
import json
import numpy as np
import onnx
import pytest
from quantlab.engine import quantize
from quantlab.core import ModelError
from tests.test_core import create_tiny_model


def test_fast_roundtrip(tmp_path):
    """Проверить экспорт, раздельную калибровку и Global Guard."""
    path = create_tiny_model(tmp_path/'model.onnx')
    original = path.read_bytes()
    rng = np.random.default_rng(42)
    cal = tmp_path/'calibration.json'
    cal.write_text(json.dumps([{'input':rng.normal(size=(2,3)).tolist()} for _ in range(10)]))
    result,report = quantize(path,'fast',cal)
    assert result.exists() and path.read_bytes()==original
    onnx.checker.check_model(str(result))
    assert report['mode']=='fast' and report['selected_count']==1
    assert sum(report['split_batches'].values())==10
    assert report['output_fidelity']['status']=='measured'
    assert report['work_budget']['used'] <= report['work_budget']['limit']
    old = report['global_guard']['baseline']['mse_by_output']
    new = report['output_fidelity']['mse_by_output']
    assert all(new[k]<=old[k] for k in old)
    assert (result.parent/'report.json').is_file()
    assert not (result.parent/'minmax_reference.onnx').exists()
    with pytest.raises(ModelError,match='JSON'):
        quantize(path,'fast')
