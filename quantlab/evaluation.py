"""Воспроизводимая оценка ONNX-классификаторов на размеченных и неразмеченных данных."""

import csv
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from quantlab.core import ModelError
from quantlab.runtime import BERT_TOKENIZER, make_feed, session, text_batches

MAX_DATASET_BYTES = 8 * 1024 * 1024
MAX_SAMPLES = 2000


def load_dataset(path: Path, is_demo: bool) -> list[dict]:
    """Загрузить примеры из CSV (только текст BERT), JSON или JSONL."""
    path = Path(path)
    if not path.is_file() or not (0 < path.stat().st_size <= MAX_DATASET_BYTES):
        raise ModelError('Датасет пустой, отсутствует или больше 8 МиБ')

    if path.suffix.lower() == '.csv':
        if not is_demo:
            raise ModelError('CSV с text,label поддерживается только для встроенной BERT-Tiny')
        with path.open(encoding='utf-8-sig', newline='') as stream:
            reader = csv.DictReader(stream)
            if not reader.fieldnames or 'text' not in reader.fieldnames:
                raise ModelError('CSV должен содержать колонку text (необязательно label)')
            data = list(reader)
    elif path.suffix.lower() in {'.json', '.jsonl'}:
        try:
            with path.open(encoding='utf-8-sig') as stream:
                if path.suffix.lower() == '.jsonl':
                    data = [json.loads(line) for line in stream if line.strip()]
                else:
                    data = json.load(stream)
        except (ValueError, UnicodeDecodeError) as exc:
            raise ModelError(f'Некорректный JSON-датасет: {exc}') from exc
        if isinstance(data, dict) and 'samples' in data:
            data = data['samples']
    else:
        raise ModelError('Поддерживаются только JSON, JSONL и CSV')

    if not isinstance(data, list) or not (1 <= len(data) <= MAX_SAMPLES):
        raise ModelError('Датасет должен содержать 1–2000 примеров')
    if not all(isinstance(item, dict) for item in data):
        raise ModelError('Каждый пример должен быть JSON-объектом')

    validated = []
    for i, item in enumerate(data):
        value = item.get('label')
        if value is not None and str(value).strip() != '':
            if is_demo and isinstance(value, str) and value.strip().lower() in {'positive', 'negative'}:
                label = {'negative': 0, 'positive': 1}[value.strip().lower()]
            else:
                try:
                    label = int(value)
                except (TypeError, ValueError) as exc:
                    raise ModelError(f'Пример {i + 1}: label должен быть индексом класса') from exc
                if isinstance(value, float) and label != value:
                    raise ModelError(f'Пример {i + 1}: label должен быть целым числом')
                if isinstance(value, str) and str(label) != value.strip():
                    raise ModelError(f'Пример {i + 1}: label должен быть целым числом')
            if label < 0:
                raise ModelError(f'Пример {i + 1}: индекс класса не может быть отрицательным')
        else:
            label = None

        if 'text' in item and is_demo:
            if not isinstance(item['text'], str) or not item['text'].strip():
                raise ModelError('Текстовые примеры поддерживаются только для BERT-Tiny')
            if len(item['text']) > 8000:
                raise ModelError(f'Пример {i + 1}: слишком длинный текст')
            validated.append({'text': item['text'], 'label': label})
        else:
            raw = item.get('inputs', item)
            if not isinstance(raw, dict) or not raw:
                raise ModelError(f'Пример {i + 1}: ожидаются входные тензоры в inputs')
            inputs = raw if 'inputs' in item else {k: v for k, v in raw.items() if k != 'label'}
            validated.append({'inputs': inputs, 'label': label})
    if any(s['label'] is None for s in validated) and any(s['label'] is not None for s in validated):
        raise ModelError('Либо у всех примеров есть label, либо ни у одного')
    if any('text' in s for s in validated) and any('inputs' in s for s in validated):
        raise ModelError('Не смешивайте текстовые и тензорные примеры в одном датасете')
    return validated


def _feeds(samples, reference_session, is_demo):
    """Поочерёдно выдавать входные тензоры и число примеров для одинаковой проверки обеих моделей."""
    if 'text' in samples[0]:
        if not is_demo:
            raise ModelError('Текстовый датасет можно использовать только с демомоделью BERT-Tiny')
        yield from text_batches(reference_session, [item['text'] for item in samples], BERT_TOKENIZER)
    else:
        for sample in samples:
            feed = make_feed(reference_session, sample['inputs'])
            batch_sizes = [value.shape[0] for value in feed.values() if value.ndim]
            if not batch_sizes or any(size != 1 for size in batch_sizes):
                raise ModelError('Для универсального JSON ожидается 1 пример в батче (первая размерность = 1)')
            yield feed, 1


def _classification_scores(labels: np.ndarray, predictions: np.ndarray, classes: int) -> dict:
    """Рассчитать macro-метрики по всем классам, включая отсутствующие."""
    actual = np.bincount(labels, minlength=classes)
    predicted = np.bincount(predictions, minlength=classes)
    correct = np.bincount(labels[labels == predictions], minlength=classes)
    precision = correct / np.maximum(predicted, 1)
    recall = correct / np.maximum(actual, 1)
    f1 = 2 * correct / np.maximum(actual + predicted, 1)
    scores = {
        'accuracy': float(np.mean(labels == predictions)),
        'precision_macro': float(precision.mean()),
        'recall_macro': float(recall.mean()),
        'f1_macro': float(f1.mean()),
    }
    if classes <= 256:
        matrix = np.zeros((classes, classes), dtype=np.int64)
        np.add.at(matrix, (labels, predictions), 1)
        scores['confusion_matrix'] = matrix.tolist()
    else:
        pairs, counts = np.unique(np.column_stack((labels, predictions)), axis=0, return_counts=True)
        scores['confusion_matrix_sparse'] = [
            {'true': int(true), 'predicted': int(pred), 'count': int(count)}
            for (true, pred), count in zip(pairs, counts, strict=True)
        ]
    return scores


def evaluate_dataset(reference: Path, quantized: Path, dataset_path: Path, is_demo: bool) -> dict:
    """Оценить FP32 и INT8 на одних пакетах без предположений о задаче вне классификации."""
    samples = load_dataset(dataset_path, is_demo)
    ref = session(reference)
    qnt = session(quantized)
    ref_names = [s.name for s in ref.get_outputs()]
    qnt_names = [s.name for s in qnt.get_outputs()]
    if ref_names != qnt_names:
        raise ModelError('Выходы исходной и INT8-модели различаются по именам')

    labels_present = samples[0]['label'] is not None
    labels = np.asarray([s['label'] for s in samples], dtype=np.int64) if labels_present else None
    total = len(samples)
    diff_sum = 0.0
    diff_count = 0
    diff_max = 0.0
    ref_time = 0.0
    qnt_time = 0.0
    ref_predictions = []
    qnt_predictions = []
    classification_ok = True
    class_count = None
    processed = 0

    for feed, batch_count in _feeds(samples, ref, is_demo):
        begin = perf_counter()
        a = ref.run(None, feed)
        ref_time += perf_counter() - begin
        begin = perf_counter()
        b = qnt.run(None, feed)
        qnt_time += perf_counter() - begin
        if len(a) != len(b):
            raise ModelError('Число выходных тензоров исходной и INT8-модели не совпадает')
        for ar, br in zip(a, b):
            if ar.shape != br.shape or ar.dtype != br.dtype:
                raise ModelError('Формы или типы выходных тензоров не совпадают')
            if not (np.issubdtype(ar.dtype, np.number) and np.issubdtype(br.dtype, np.number)):
                raise ModelError('Сравнение поддерживает только числовые выходы')
            delta = np.abs(ar.astype(np.float64) - br.astype(np.float64))
            if not np.isfinite(delta).all():
                raise ModelError('На выходах модели обнаружены NaN или Inf')
            diff_sum += float(delta.sum())
            diff_count += delta.size
            if delta.size:
                diff_max = max(diff_max, float(delta.max()))

        if not a or a[0].ndim != 2 or a[0].shape[0] != batch_count or a[0].shape[1] < 2:
            classification_ok = False
        elif class_count is not None and class_count != a[0].shape[1]:
            classification_ok = False
        else:
            class_count = a[0].shape[1]
            ref_predictions.extend(np.argmax(a[0], axis=1).tolist())
            qnt_predictions.extend(np.argmax(b[0], axis=1).tolist())
        processed += batch_count

    if processed != total:
        raise ModelError('Количество выполненных примеров не совпадает с датасетом')
    if labels_present and not classification_ok:
        raise ModelError('Accuracy/F1 возможны только для моделей с выходом [batch, num_classes]')
    if labels_present and (labels >= class_count).any():
        raise ModelError('В датасете есть label вне диапазона классов модели')

    report = {
        'dataset': Path(dataset_path).name,
        'samples': total,
        'labeled': labels_present,
        'task': 'classification' if classification_ok else 'numeric_outputs',
        'mean_absolute_output_difference': diff_sum / diff_count if diff_count else 0.0,
        'max_absolute_output_difference': diff_max,
        'latency_ms_per_sample': {
            'fp32': round(ref_time * 1000 / total, 4),
            'int8': round(qnt_time * 1000 / total, 4),
        },
        'speedup': round(ref_time / qnt_time, 3) if qnt_time else None,
        'note': 'Время учитывает только инференс, без подготовки входов; выполняется один прогон.',
    }
    if classification_ok:
        fp = np.asarray(ref_predictions, dtype=np.int64)
        qt = np.asarray(qnt_predictions, dtype=np.int64)
        report['prediction_agreement'] = float(np.mean(fp == qt))
        report['changed_predictions'] = int(np.sum(fp != qt))
        report['class_count'] = class_count
        if labels_present:
            report['fp32'] = _classification_scores(labels, fp, class_count)
            report['int8'] = _classification_scores(labels, qt, class_count)
        report['changed_examples'] = [
            {'index': int(i), 'fp32': int(fp[i]), 'int8': int(qt[i]),
             **({'label': int(labels[i])} if labels_present else {})}
            for i in np.flatnonzero(fp != qt)[:30]
        ]
    return report
