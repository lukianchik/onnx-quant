"""Проверка полного цикла Balanced в ONNX Runtime."""
import json
import numpy as np
import onnx
import pytest
from quantlab.engine import quantize
from quantlab.core import ModelError
from tests.test_core import create_tiny_model


def test_balanced_roundtrip(tmp_path):
    """Экспортировать Progressive, сохранить бюджет и не ухудшить MinMax на Check."""
    path=create_tiny_model(tmp_path/'model.onnx')
    source=path.read_bytes()
    rng=np.random.default_rng(42)
    cal=tmp_path/'calibration.json'
    cal.write_text(json.dumps([{'input':rng.normal(size=(2,3)).tolist()} for _ in range(10)]))
    output,report=quantize(path,'balanced',cal,work_limit=250_000_000)
    assert output.exists() and path.read_bytes()==source
    onnx.checker.check_model(str(output))
    assert report['mode']=='balanced' and report['algorithm']=='Progressive'
    assert report['selected_count']==1 and 'steps' in report['layers'][0]
    assert report['work_budget']['limit']==250_000_000
    assert report['work_budget']['used']<=250_000_000
    old=report['global_guard']['baseline']['mse_by_output']
    assert all(report['output_fidelity']['mse_by_output'][k]<=v for k,v in old.items())
    assert (output.parent/'report.json').is_file()
    with pytest.raises(ModelError,match='JSON'):
        quantize(path,'balanced')
    with pytest.raises(ModelError,match='бюджет'):
        quantize(path,'balanced',cal,work_limit=1)
