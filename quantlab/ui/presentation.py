"""Читаемые сводки с безопасным экранированием HTML для интерфейса квантования."""

from html import escape


MODE_LABELS = {
    'standard': 'Standard · MinMax',
    'fast': 'Fast · Single-Robust',
    'balanced': 'Balanced · Progressive',
    'quality': 'Quality · Progressive-Global',
}


def workspace_header(model_name=None):
    """Показать активную модель во всех вкладках."""
    source = escape(model_name) if model_name else 'Модель не выбрана'
    return (
        '<header class="ql-topbar"><div class="ql-brand">'
        '<span class="ql-mark" aria-hidden="true"><svg viewBox="0 0 24 24" fill="none" '
        'stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round">'
        '<path d="M3 5h7v7H3zM14 12h7v7h-7zM10 8.5h4m-2 0v7h2"/></svg></span>'
        '<div><h1>ONNX Quantization Lab</h1><span class="ql-source">'
        f'{source}</span></div></div><span class="ql-runtime">ONNX Runtime · CPU</span></header>'
    )


def empty_state(title: str, description: str) -> str:
    """Показать подсказку до появления результата."""
    return (
        '<div class="ql-empty">'
        f'<strong>{escape(title)}</strong>'
        f'<span>{escape(description)}</span>'
        '</div>'
    )


def _stat(label: str, value: str, note: str = '') -> str:
    note_html = f'<small>{escape(note)}</small>' if note else ''
    return (
        '<div class="ql-stat">'
        f'<span>{escape(label)}</span>'
        f'<strong>{escape(value)}</strong>'
        f'{note_html}'
        '</div>'
    )


def model_summary(report: dict, display_name: str) -> str:
    """Кратко представить граф, не перегружая интерфейс техническими деталями."""
    count = len(report['candidates'])
    state = 'Найдены кандидаты' if count else 'Нет кандидатов weight-only'
    tone = 'success' if count else 'warning'
    return (
        '<div class="ql-result-head">'
        '<div>'
        f'<h3>{escape(display_name)}</h3>'
        '</div>'
        f'<span class="ql-pill ql-pill-{tone}">{state}</span>'
        '</div>'
        '<div class="ql-stat-grid">'
        + _stat('Размер модели', f"{report['size_mib']:.2f} MiB")
        + _stat('Операторов', str(report['nodes']))
        + _stat('Слоёв для INT8-весов', str(count))
        + '</div>'
        '<div class="ql-meta-line">'
        f"{report['initializers']} тензоров весов · "
        f"{report['candidate_parameters']:,} параметров доступны для INT8"
        '</div>'
    )


def quant_summary(metrics: dict) -> str:
    """Показать основные метрики квантования и результат проверки."""
    reduction = float(metrics['size_reduction_percent'])
    reduction_label = f"{reduction:+.2f}%"
    direction = 'Меньше исходной' if reduction >= 0 else 'Больше исходной'
    method = MODE_LABELS[metrics['mode']]
    fidelity = metrics.get('output_fidelity', {})
    detail = (' · MSE измерен' if fidelity.get('status') == 'measured'
              else ' · Численная точность не проверена')
    detail += f" · Покрытие FP32/FP16-параметров: {metrics['coverage_percent']:.2f}%"
    if metrics['mode'] in ('fast', 'balanced', 'quality'):
        accepted_name = {'fast':'Single-Robust','balanced':'Progressive','quality': ('Progressive-Global' if metrics.get('effective_algorithm') == 'Progressive-Global' else 'Progressive')}[metrics['mode']]
        detail += (f' · Global Guard: принят {accepted_name}' if metrics['global_guard']['accepted']
                   else ' · Global Guard: MinMax')
    if metrics['mode'] in ('balanced','quality'):
        budget = metrics.get('work_budget', {})
        if budget.get('limit'):
            detail += f" · Бюджет поиска: {100*budget['used']/budget['limit']:.1f}%"
        layers = metrics.get('layers', [])
        if layers:
            stopped = sum(layer.get('stop_reason') == 'robust_target_reached' for layer in layers)
            detail += f" · Достигнут порог 25%: {stopped}/{len(layers)}"
    if metrics['mode'] == 'quality':
        search = metrics.get('global_search', {})
        detail += f" · Глобальные проверки: {int(search.get('calls',0))}/{int(search.get('cap',0))}"
    return (
        '<div class="ql-result-head">'
        '<div>'
        f'<h3>{method}</h3></div>'
        '<span class="ql-pill ql-pill-success">Проверка пройдена</span>'
        '</div>'
        '<div class="ql-stat-grid">'
        + _stat('Сокращение размера', reduction_label, direction)
        + _stat('Исходная модель', f"{metrics['fp32_size_mib']:.2f} MiB")
        + _stat('Новый INT8', f"{metrics['int8_size_mib']:.2f} MiB")
        + '</div>'
        '<div class="ql-meta-line">'
        f"Преобразовано слоёв: {metrics['selected_count']}"
        ' · ONNX checker: OK · ONNX Runtime: OK'
        + detail + '</div>'
    )


def _confidence_card(model: str, classification: dict | None, outputs: dict) -> str:
    """Показать вероятности классов текста или сведения о выходных тензорах."""
    if classification and 'confidence' in classification and 'label' in classification:
        labels = {'Positive': 'Позитивный', 'Negative': 'Негативный'}
        label = labels.get(classification['label'], classification['label'])
        value = max(0.0, min(100.0, float(classification['confidence']) * 100.0))
        return (
            '<div class="ql-prediction">'
            f'<span class="ql-prediction-kicker">{escape(model)}</span>'
            f'<strong>{escape(str(label))}</strong>'
            f'<div class="ql-confidence"><span>{value:.1f}% уверенность</span>'
            f'<div class="ql-meter"><div style="width:{value:.2f}%"></div></div>'
            '</div></div>'
        )
    output_details = list(outputs.items())
    if output_details:
        name, first = output_details[0]
        shape = str(first.get('shape', '?'))
        summary = f"{name} · shape {shape}"
    else:
        summary = 'Выходы не обнаружены'
    return (
        '<div class="ql-prediction">'
        f'<span class="ql-prediction-kicker">{escape(model)}</span>'
        '<strong>Вычислено</strong>'
        f'<p>{escape(summary)}</p>'
        '</div>'
    )


def inference_summary(result: dict) -> str:
    """Показать сравнение FP32 и INT8; полные массивы оставить в JSON."""
    fp32 = result.get('FP32') or {}
    int8 = result.get('INT8') or {}
    cards = _confidence_card('Исходная модель', fp32.get('classification'), fp32)
    if int8:
        cards += _confidence_card('INT8', int8.get('classification'), int8)
        comparison = result.get('comparison') or {}
        delta = comparison.get('max_absolute_output_difference')
        if delta is not None:
            detail = f'Максимальное расхождение выходов: {float(delta):.6f}'
        else:
            detail = 'Сравнение выходных тензоров недоступно'
    else:
        detail = 'Запустите квантование, чтобы сравнить исходную модель и INT8.'
    return (
        '<div class="ql-prediction-grid">' + cards + '</div>'
        f'<div class="ql-meta-line">{escape(detail)}</div>'
    )


def dataset_summary(report: dict) -> str:
    """Показать метрики датасета; не выводить Accuracy для данных без меток."""
    count = int(report['samples'])
    head = (
        '<div class="ql-result-head"><div>'
        f'<h3>{escape(str(report["dataset"]))}</h3></div>'
        f'<span class="ql-pill ql-pill-success">{count} примеров</span></div>'
    )
    if report.get('labeled') and 'fp32' in report and 'int8' in report:
        fp = report['fp32']
        qt = report['int8']
        content = (
            _stat('Accuracy · исходная', f"{fp['accuracy']:.2%}")
            + _stat('Accuracy INT8', f"{qt['accuracy']:.2%}")
            + _stat('Δ Accuracy', f"{(qt['accuracy'] - fp['accuracy']) * 100:+.2f} п.п.")
            + _stat('Macro F1 · исходная', f"{fp['f1_macro']:.3f}")
            + _stat('Macro F1 · INT8', f"{qt['f1_macro']:.3f}")
            + _stat('Изменено ответов', str(report.get('changed_predictions', 0)))
        )
    else:
        content = (
            _stat('Примеры', str(count))
            + _stat('Среднее |Δ|', f"{report['mean_absolute_output_difference']:.5f}")
            + _stat('Макс. |Δ|', f"{report['max_absolute_output_difference']:.5f}")
        )
    agreement = report.get('prediction_agreement')
    agreement_text = f"Совпадение ответов: {agreement:.2%} · " if agreement is not None else ''
    speedup = report.get('speedup')
    speed_text = f" · Скорость: {speedup:.2f}×" if speedup is not None else ''
    note = 'Есть метки классов' if report.get('labeled') else 'Без меток: Accuracy/F1 недоступны'
    return (
        head + '<div class="ql-stat-grid">' + content + '</div>'
        + '<div class="ql-meta-line">'
        + escape(f"{agreement_text}{note}{speed_text}") + '</div>'
    )
