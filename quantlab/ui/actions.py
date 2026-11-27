"""Обработчики интерфейса и конфигурация рабочей области."""

import json
import os
from pathlib import Path

import gradio as gr

from quantlab.core import ModelError, analyze, run_directory, stage_model
from quantlab.engine import quantize
from quantlab.inference import compare_inference, model_inputs, BERT_TOKENIZER
from quantlab.evaluation import evaluate_dataset
from quantlab.ui.presentation import (
    dataset_summary, empty_state, inference_summary, model_summary, quant_summary, workspace_header,
)

DATA_DIR = Path(os.getenv('QUANTLAB_DATA', '/app/data')).resolve()
EXAMPLES_DIR = Path(os.getenv('QUANTLAB_EXAMPLES', '/app/examples')).resolve()
RUNS_DIR = DATA_DIR / 'runs'

EXAMPLE_MODEL = EXAMPLES_DIR / 'bert_tiny_fp32.onnx'
EXAMPLE_CALIB = EXAMPLES_DIR / 'calibration_samples.json'
EXAMPLE_EVAL = EXAMPLES_DIR / 'evaluation_samples.json'

DEMO_LABEL = 'BERT-Tiny · SST-2 (встроенная)'
TEXT_INPUT = 'Текст (BERT-Tiny SST-2)'
JSON_INPUT = 'JSON-тензоры'
MODE_HINTS = {
    'standard': 'Поканальный MinMax. Добавьте JSON, если хотите измерить ошибку выходов.',
    'fast': 'Один кандидат на слой. Нужно от 5 независимых пакетов калибровки.',
    'balanced': 'Последовательный поиск по слою в пределах бюджета. Нужно от 5 пакетов.',
    'quality': 'Дополнительная проверка сочетаний весов. Нужно от 6 независимых пакетов.',
}

EMPTY_MODEL = empty_state('Граф модели', 'Нажмите «Проанализировать», чтобы увидеть размер и подходящие слои.')
EMPTY_QUANT = empty_state('Результат появится здесь', 'Выберите режим и запустите квантование.')
EMPTY_INFER = empty_state('Нет предсказаний', 'Введите данные и запустите инференс.')
EMPTY_DATASET = empty_state('Нет оценки', 'После квантования можно сравнить модели на датасете.')


def source_path(upload, selected):
    """Определить путь к выбранной модели."""
    if upload:
        return Path(str(upload))
    if selected == DEMO_LABEL and EXAMPLE_MODEL.is_file():
        return EXAMPLE_MODEL
    raise ModelError('Загрузите .onnx или выберите доступную демомодель')


def candidates_table(report):
    """Подготовить строки таблицы кандидатов."""
    return [[
        c['operator'], c['node'], str(c['shape']), c['axis'], c['channels']
    ] for c in report['candidates']]


def analyze_ui(upload, selected):
    """Подготовить модель и сбросить результаты предыдущего запуска."""
    try:
        source = source_path(upload, selected)
        is_demo = not upload and selected == DEMO_LABEL
        prepared = stage_model(source, RUNS_DIR)
        report = analyze(prepared)
        display_name = 'BERT-Tiny · SST-2' if is_demo else source.name
        info = model_inputs(prepared)
        details = {
            'inputs': info,
            'outputs': report['outputs'],
            'operators': report['operators'],
            'skipped': report['skipped'],
            'opset': report['opset'],
        }
        return (
            str(prepared), None, is_demo,
            model_summary(report, display_name), candidates_table(report), details,
            EMPTY_QUANT, {}, None, None,
            EMPTY_INFER, {}, EMPTY_DATASET, {}, None,
            workspace_header(display_name),
            gr.update(choices=[TEXT_INPUT, JSON_INPUT] if is_demo else [JSON_INPUT],
                      value=TEXT_INPUT if is_demo else JSON_INPUT),
            gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=False),
            gr.update(visible=False), gr.update(visible=False),
            gr.update(visible=is_demo),
        )
    except Exception as exc:
        raise gr.Error(f'Не удалось проанализировать модель: {exc}') from exc


def quantize_ui(prepared, mode, uploaded_calib, is_demo, work_limit=1_000_000_000, quality_task='auto'):
    """Выполнить квантование и обновить результаты."""
    if not prepared:
        raise gr.Error('Сначала выберите модель и нажмите «Проанализировать»')
    try:
        calib = Path(str(uploaded_calib)) if uploaded_calib else None
        tokenizer_id = None
        if mode in ('fast', 'balanced', 'quality') and is_demo and calib is None and EXAMPLE_CALIB.is_file():
            calib = EXAMPLE_CALIB
        if calib is not None and is_demo:
            tokenizer_id = BERT_TOKENIZER
        task = ('classification' if is_demo else 'generic') if quality_task == 'auto' else quality_task
        budget = int(work_limit) if mode in ('balanced', 'quality') else None
        quantized_path, metrics = quantize(
            Path(prepared), mode, calib, tokenizer_id, work_limit=budget, task=task,
        )
        return (
            str(quantized_path), str(quantized_path),
            str(quantized_path.parent / 'report.json'), metrics,
            quant_summary(metrics), EMPTY_INFER, {}, EMPTY_DATASET, {}, None,
            gr.update(interactive=True), gr.update(visible=True), gr.update(visible=False),
        )
    except Exception as exc:
        raise gr.Error(f'Не удалось квантовать модель: {exc}') from exc


def infer_ui(reference, quantized, kind, text, tensors_json):
    """Сравнить выходы исходной и квантованной моделей."""
    if not reference:
        raise gr.Error('Сначала проанализируйте модель')
    try:
        result = compare_inference(
            Path(reference), Path(quantized) if quantized else None,
            kind, text, tensors_json,
        )
        return inference_summary(result), result
    except Exception as exc:
        raise gr.Error(f'Не удалось выполнить инференс: {exc}') from exc


def dataset_ui(reference, quantized, uploaded, is_demo):
    """Оценить датасет и сохранить отдельный отчёт."""
    if not reference or not quantized:
        raise gr.Error('Сначала проанализируйте и квантовайте модель')
    try:
        dataset_path = Path(str(uploaded)) if uploaded else None
        if dataset_path is None and is_demo and EXAMPLE_EVAL.is_file():
            dataset_path = EXAMPLE_EVAL
        if dataset_path is None:
            raise ModelError('Загрузите датасет JSON, JSONL или CSV')
        result = evaluate_dataset(
            Path(reference), Path(quantized), dataset_path, is_demo,
        )
        with run_directory(Path(quantized).parent, 'evaluation_') as run:
            report_file = run / 'dataset_report.json'
            report_file.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
        return dataset_summary(result), result, str(report_file), gr.update(visible=True)
    except Exception as exc:
        raise gr.Error(f'Не удалось оценить датасет: {exc}') from exc


def reset_source():
    """Снять результаты и входы предыдущей модели при смене источника."""
    return (
        None, None, False, EMPTY_MODEL, [], {}, EMPTY_QUANT, {}, None, None,
        EMPTY_INFER, {}, EMPTY_DATASET, {}, None, workspace_header(),
        gr.update(choices=[JSON_INPUT], value=JSON_INPUT),
        gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False),
        gr.update(visible=False), gr.update(visible=False), gr.update(visible=False),
        None, None, '',
    )


def change_source(selected):
    """Переключить демомодель и пользовательский файл."""
    return (gr.update(value=None, visible=selected != DEMO_LABEL),
            gr.update(visible=selected == DEMO_LABEL))


def change_mode(selected):
    """Обновить параметры режима и сбросить устаревший результат."""
    return (
        MODE_HINTS[selected], gr.update(visible=True),
        gr.update(visible=selected in ('balanced', 'quality')), gr.update(visible=selected == 'quality'),
        None, EMPTY_QUANT, {}, None, None, EMPTY_INFER, {}, EMPTY_DATASET, {}, None,
        gr.update(interactive=False), gr.update(visible=False), gr.update(visible=False),
    )


def change_input_kind(selected):
    """Переключить формат входа и очистить предсказания."""
    return (
        gr.update(visible=(selected == TEXT_INPUT)),
        gr.update(visible=(selected == JSON_INPUT)),
        EMPTY_INFER, {},
    )
