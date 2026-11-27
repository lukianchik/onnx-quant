"""Тесты оценки датасета на небольшом детерминированном ONNX-классификаторе."""

import json
from pathlib import Path

import numpy as np
import pytest

from quantlab.core import ModelError
from quantlab.evaluation import _classification_scores, evaluate_dataset, load_dataset
from quantlab.engine import quantize
from tests.test_core import create_tiny_model


def test_dataset_scores():
    truth = np.asarray([0, 1, 1, 0])
    pred = np.asarray([0, 0, 1, 0])
    got = _classification_scores(truth, pred, 2)
    assert got['accuracy'] == 0.75
    assert got['confusion_matrix'] == [[2, 0], [1, 1]]
    assert 0.70 < got['f1_macro'] < 0.75


def test_load_demo_csv_and_unlabeled(tmp_path):
    file = tmp_path / 'examples.csv'
    file.write_text('text,label\nGood,Positive\nBad,Negative\n', encoding='utf-8')
    samples = load_dataset(file, is_demo=True)
    assert [s['label'] for s in samples] == [1, 0]
    with pytest.raises(ModelError, match='CSV'):
        load_dataset(file, is_demo=False)

    file = tmp_path / 'examples.json'
    file.write_text(json.dumps([{'input': [[0, 1, 2]]}]), encoding='utf-8')
    assert load_dataset(file, is_demo=False)[0]['label'] is None


def test_mixed_labels_rejected(tmp_path):
    file = tmp_path / 'examples.jsonl'
    file.write_text('{"text":"good","label":1}\n{"text":"bad"}\n', encoding='utf-8')
    with pytest.raises(ModelError, match='Либо у всех'):
        load_dataset(file, is_demo=True)


def test_labeled_and_unlabeled_generic_evaluation(tmp_path):
    # Test only generic tensors, independent of Hugging Face connectivity.
    original = create_tiny_model(tmp_path / 'test.onnx')
    quantized, _ = quantize(original, 'standard')
    samples = [
        {'inputs': {'input': [[1., 0., 0.]]}, 'label': 0},
        {'inputs': {'input': [[0., 1., 0.]]}, 'label': 0},
        {'inputs': {'input': [[0., 0., 1.]]}, 'label': 1},
    ]
    path = tmp_path / 'dataset.json'
    path.write_text(json.dumps(samples), encoding='utf-8')
    results = evaluate_dataset(original, quantized, path, is_demo=False)
    assert results['samples'] == 3 and results['labeled']
    assert results['fp32']['accuracy'] == 1.0
    assert results['int8']['accuracy'] >= 2 / 3
    assert results['changed_predictions'] >= 0
    assert results['latency_ms_per_sample']['fp32'] >= 0

    path.write_text(json.dumps([{'input': s['inputs']['input']} for s in samples]), encoding='utf-8')
    results = evaluate_dataset(original, quantized, path, is_demo=False)
    assert results['labeled'] is False
    assert 'fp32' not in results
    assert 'mean_absolute_output_difference' in results


def test_out_of_range_label_rejected(tmp_path):
    original = create_tiny_model(tmp_path / 'test.onnx')
    quantized, _ = quantize(original, 'standard')
    path = tmp_path / 'dataset.json'
    path.write_text(json.dumps([{'input': [[1, 0, 0]], 'label': 999}]), encoding='utf-8')
    with pytest.raises(ModelError, match='вне диапазона'):
        evaluate_dataset(original, quantized, path, is_demo=False)


def test_changed_predictions_report_is_json_serializable(tmp_path, monkeypatch):
    """Регрессия: индексы numpy.int64 от np.flatnonzero несовместимы с JSON."""
    from types import SimpleNamespace
    import quantlab.evaluation as evaluation

    dataset = tmp_path / 'mismatch.json'
    dataset.write_text(
        json.dumps([{'inputs': {'input': [[1.0]]}, 'label': 0}]),
        encoding='utf-8',
    )

    class FakeSession:
        def __init__(self, filename):
            self.is_fp32 = Path(filename).name == 'fp32.onnx'

        def get_outputs(self):
            return [SimpleNamespace(name='logits')]

        def run(self, _names, _feed):
            logits = [[3.0, 1.0]] if self.is_fp32 else [[1.0, 3.0]]
            return [np.array(logits, dtype=np.float32)]

    monkeypatch.setattr(evaluation, 'session', FakeSession)
    monkeypatch.setattr(
        evaluation, '_feeds',
        lambda samples, session, demo: iter([({'input': np.array([[1.0]])}, 1)]),
    )
    result = evaluation.evaluate_dataset(
        tmp_path / 'fp32.onnx', tmp_path / 'int8.onnx', dataset, False,
    )
    assert result['changed_predictions'] == 1
    assert result['changed_examples'] == [{'index': 0, 'fp32': 0, 'int8': 1, 'label': 0}]
    assert type(result['changed_examples'][0]['index']) is int
    assert json.loads(json.dumps(result, ensure_ascii=False)) == result


def test_large_confusion_matrix_keeps_macro_denominator():
    """Для большого словаря классов сохранять только наблюдаемые пары."""
    labels = np.array([0, 0, 12000])
    predictions = np.array([1, 1, 12000])
    scores = _classification_scores(labels, predictions, 12001)
    assert scores['f1_macro'] == pytest.approx(1 / 12001)
    assert scores['confusion_matrix_sparse'] == [
        {'true': 0, 'predicted': 1, 'count': 2},
        {'true': 12000, 'predicted': 12000, 'count': 1},
    ]


def test_tensor_names_can_match_dataset_fields(tmp_path):
    """Имена входов text и label допустимы для числовой модели."""
    path = tmp_path / 'samples.json'
    inputs = {'text': [[1.0]], 'label': [[2.0]]}
    path.write_text(json.dumps([{'inputs': inputs, 'label': 0}]))
    assert load_dataset(path, False) == [{'inputs': inputs, 'label': 0}]
    path.write_text(json.dumps([{'text': [[3.0]]}]))
    assert load_dataset(path, False) == [{'inputs': {'text': [[3.0]]}, 'label': None}]
