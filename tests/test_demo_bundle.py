"""Проверка встроенных в Docker-образ данных BERT и SST-2 при их наличии."""

import json
import os
from pathlib import Path

import pytest


def test_bundled_demo_assets_if_available():
    root = Path(os.getenv('QUANTLAB_EXAMPLES', '/app/examples'))
    manifest_path = root / 'demo_manifest.json'
    if not manifest_path.exists():
        pytest.skip('Bundle is created during Docker build')

    import onnx
    import onnxruntime as ort
    from transformers import AutoTokenizer

    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    assert manifest['calibration_count'] == 256
    assert manifest['evaluation_count'] == 872

    model_path = root / 'bert_tiny_fp32.onnx'
    onnx.checker.check_model(str(model_path))
    tokenizer = AutoTokenizer.from_pretrained(root / 'tokenizer', local_files_only=True)
    encoded = tokenizer(['Good movie.', 'Bad movie.'], return_tensors='np', padding=True)
    sess = ort.InferenceSession(str(model_path), providers=['CPUExecutionProvider'])
    feed = {spec.name: encoded[spec.name] for spec in sess.get_inputs()}
    output = sess.run(None, feed)[0]
    assert output.shape == (2, 2)

    for filename, expected in [('calibration_samples.json', 256), ('evaluation_samples.json', 872)]:
        rows = json.loads((root / filename).read_text(encoding='utf-8'))
        assert len(rows) == expected
        assert all(isinstance(s['text'], str) and s['label'] in (0, 1) for s in rows)
