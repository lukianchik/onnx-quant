"""CPU-сессии ONNX Runtime и подготовка входных тензоров приложения."""

import os
from pathlib import Path

import numpy as np
import onnxruntime as ort

from quantlab.core import ModelError

BERT_TOKENIZER = str(Path(os.getenv('QUANTLAB_EXAMPLES', '/app/examples')) / 'tokenizer')
BERT_BATCH = 16


def session(path: Path | bytes):
    """Создать CPU-сессию с ограниченным числом потоков."""
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    source = path if isinstance(path, bytes) else str(path)
    return ort.InferenceSession(source, sess_options=options, providers=['CPUExecutionProvider'])


DTYPES = {
    'tensor(float)': np.float32, 'tensor(double)': np.float64,
    'tensor(float16)': np.float16, 'tensor(int64)': np.int64,
    'tensor(int32)': np.int32, 'tensor(int16)': np.int16,
    'tensor(int8)': np.int8, 'tensor(uint8)': np.uint8,
    'tensor(bool)': np.bool_,
}


def make_feed(session: ort.InferenceSession, raw: dict) -> dict:
    """Преобразовать массивы JSON к типам входов модели."""
    if not isinstance(raw, dict):
        raise ModelError('Входные данные должны быть JSON-объектом')
    result = {}
    for spec in session.get_inputs():
        if spec.name not in raw:
            raise ModelError(f'Отсутствует вход модели: {spec.name}')
        dtype = DTYPES.get(spec.type)
        if dtype is None:
            raise ModelError(f'Неподдерживаемый тип {spec.name}: {spec.type}')
        result[spec.name] = np.asarray(raw[spec.name], dtype=dtype)
    return result


def text_batches(sess, texts, tokenizer_id):
    """Преобразовать тексты BERT в пакеты входных тензоров модели."""
    from transformers import AutoTokenizer
    if 'input_ids' not in {spec.name for spec in sess.get_inputs()}:
        raise ModelError('В модели нет input_ids. Выберите JSON-тензоры')
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, trust_remote_code=False)
    for start in range(0, len(texts), BERT_BATCH):
        batch = texts[start:start + BERT_BATCH]
        encoded = tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors='np')
        yield make_feed(sess, dict(encoded)), len(batch)
