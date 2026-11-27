"""Интеграционные проверки Quality в целевом ORT-контейнере."""
import json
import numpy as np
import onnx
import pytest
from quantlab.engine import quantize
from quantlab.core import ModelError
from tests.test_core import create_tiny_model


@pytest.mark.parametrize('task',['generic','classification'])
def test_quality_roundtrip(tmp_path,task):
    """Проверить экспорт, независимые пакеты, бюджет и защиту MinMax."""
    path=create_tiny_model(tmp_path/'model.onnx')
    original=path.read_bytes()
    rng=np.random.default_rng(42)
    cal=tmp_path/'calibration.json'
    cal.write_text(json.dumps([{'input':rng.normal(size=(2,3)).tolist()} for _ in range(16)]))
    output,report=quantize(path,'quality',cal,work_limit=250_000_000,task=task)
    assert output.exists() and path.read_bytes()==original
    onnx.checker.check_model(str(output))
    assert report['algorithm']=='Progressive-Global' and report['mode']=='quality'
    assert set(report['split_batches'])=={'fit','dev','search','check'}
    assert sum(report['split_batches'].values())==16
    assert all(report['split_batches'].values())
    search=report['global_search']
    assert search['calls']<=search['cap'] and len(search['active'])<=4
    assert report['work_budget']['used']<=250_000_000
    assert search['bank_bytes']<=search['bank_limit_bytes']
    old=report['global_guard']['baseline']['mse_by_output']
    assert all(report['output_fidelity']['mse_by_output'][k]<=v for k,v in old.items())
    assert (output.parent/'report.json').is_file()
    assert set(report['selection'])==set(report['local_selection'])
    if not report['global_guard']['accepted']:
        assert set(report['selection'].values())=={'minmax'}
    with pytest.raises(ModelError,match='JSON'):
        quantize(path,'quality')


@pytest.mark.parametrize('task,kind',[('generic','exact_multioutput_greedy'),('classification','original_V2')])
def test_quality_search_runs_real_ort(tmp_path,monkeypatch,task,kind):
    """Принудительно сохранить отличающийся кандидат, проверить реальные ORT-пробы."""
    from quantlab.weight_only import progressive as p
    from quantlab.weight_only.robust import Loss
    def select(matrices,fit,dev,remaining,retain=None):
        """Подготовить baseline и намеренно плохой альтернативный payload."""
        base=[p.minmax(w) for w in matrices]
        alt=[(np.zeros_like(q),s.copy()) for q,s in base]
        retain('minmax',base,Loss(1,1,1))
        retain('single',alt,Loss(2,2,1))
        return base,base,1,{'method':'MinMax','steps':[]}
    monkeypatch.setattr(p,'select_progressive',select)
    path=create_tiny_model(tmp_path/'model.onnx')
    rng=np.random.default_rng(77)
    cal=tmp_path/'cal.json'
    cal.write_text(json.dumps([{'input':rng.normal(size=(2,3)).tolist()} for _ in range(16)]))
    _,report=quantize(path,'quality',cal,task=task)
    search=report['global_search']
    assert search['kind']==kind and search['calls']>=2 and 'error' not in search
    assert report['selection']=={'W':'minmax'} and not report['global_guard']['accepted']
