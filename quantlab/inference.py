"""Универсальный инференс по тензорам и адаптер демонстрационной BERT-модели."""

import json
from pathlib import Path

import numpy as np

from quantlab.core import ModelError
from quantlab.runtime import BERT_TOKENIZER, make_feed, session, text_batches

BERT_LABELS = ['Negative', 'Positive']


def model_inputs(path: Path):
    return [{'name': s.name, 'type': s.type, 'shape': s.shape} for s in session(path).get_inputs()]


def prepare_input(path: Path, kind: str, text: str, tensors_json: str):
    sess = session(path)
    return _prepare_feed(sess, kind, text, tensors_json)


def _prepare_feed(sess, kind, text, tensors_json):
    """Собрать входы для уже открытой сессии."""
    if kind == 'Текст (BERT-Tiny SST-2)':
        feed, _ = next(text_batches(sess, [text], BERT_TOKENIZER))
        return feed
    try:
        parsed = json.loads(tensors_json)
    except json.JSONDecodeError as exc:
        raise ModelError(f'Ошибка JSON: {exc}') from exc
    return make_feed(sess, parsed)


def summarize_outputs(sess, arrays, kind):
    summary = {}
    for spec, arr in zip(sess.get_outputs(), arrays):
        flat = np.asarray(arr).reshape(-1)
        vals = [float(x) for x in flat[:16]] if np.issubdtype(arr.dtype, np.number) else [str(x) for x in flat[:16]]
        summary[spec.name] = {'shape': list(arr.shape), 'dtype': str(arr.dtype), 'first_values': vals}
    if kind == 'Текст (BERT-Tiny SST-2)' and arrays:
        logits = np.asarray(arrays[0])
        if logits.shape == (1, 2):
            z = logits[0].astype(np.float64)
            exp = np.exp(z - np.max(z))
            probs = (exp / exp.sum()).tolist()
            cls = int(np.argmax(probs))
            summary['classification'] = {
                'label': BERT_LABELS[cls],
                'confidence': round(probs[cls], 6),
                'probabilities': probs,
            }
    return summary


def compare_inference(reference: Path, quantized: Path | None,
                      kind: str, text: str, tensors_json: str):
    ref_sess = session(reference)
    feed = _prepare_feed(ref_sess, kind, text, tensors_json)
    fp32 = ref_sess.run(None, feed)
    result = {'FP32': summarize_outputs(ref_sess, fp32, kind)}
    if quantized:
        q_sess = session(quantized)
        int8 = q_sess.run(None, feed)
        result['INT8'] = summarize_outputs(q_sess, int8, kind)
        if len(fp32) == len(int8) and all(a.shape == b.shape for a, b in zip(fp32, int8)):
            errors = [float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))
                      for a, b in zip(fp32, int8) if a.size and np.issubdtype(a.dtype, np.number)]
            result['comparison'] = {'max_absolute_output_difference': max(errors) if errors else None}
    return result
