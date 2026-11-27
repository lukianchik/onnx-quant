"""Компоновка четырёх вкладок, привязки событий и запуск Gradio."""

from pathlib import Path

import gradio as gr

from quantlab.ui.actions import (
    DATA_DIR, RUNS_DIR, EXAMPLE_MODEL, DEMO_LABEL, TEXT_INPUT, JSON_INPUT, MODE_HINTS,
    EMPTY_MODEL, EMPTY_QUANT, EMPTY_INFER, EMPTY_DATASET,
    analyze_ui, quantize_ui, infer_ui, dataset_ui, reset_source,
    change_source, change_mode, change_input_kind,
)
from quantlab.ui.presentation import MODE_LABELS, workspace_header

ASSETS = Path(__file__).resolve().parents[2] / 'assets'


def summary_html(value):
    """Создать HTML-сводку с отдельными стилями Gradio 6."""
    return gr.HTML(value, css_template=(ASSETS / 'components.css').read_text(encoding='utf-8'),
                   apply_default_css=False)


def create_app():
    """Собрать рабочую область без запуска HTTP-сервера."""
    with gr.Blocks(title='ONNX Quantization Lab') as app, gr.Column(elem_classes=['quantlab-app']):
        prepared_state = gr.State(None)
        quantized_state = gr.State(None)
        is_demo_state = gr.State(False)

        header = summary_html(workspace_header())

        with gr.Tabs(elem_classes=['ql-tabs']):
            with gr.Tab('01  Модель'):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=4, min_width=310, elem_classes=['ql-panel']):
                        gr.Markdown('### Источник модели', elem_classes=['ql-panel-heading'])
                        has_demo = EXAMPLE_MODEL.is_file()
                        example = gr.Radio(
                            choices=[('Демомодель', DEMO_LABEL), ('Файл ONNX', 'upload')] if has_demo else [('Файл ONNX', 'upload')],
                            value=DEMO_LABEL if has_demo else 'upload', label='Источник',
                            show_label=False, visible=has_demo,
                        )
                        demo_info = gr.Markdown('**BERT-Tiny · SST-2**  \nВстроенный классификатор тональности английских текстов.', visible=has_demo, elem_classes=['ql-demo-info'])
                        uploaded = gr.File(label='Модель ONNX · до 256 МиБ', file_types=['.onnx'], type='filepath',
                                           visible=not has_demo, elem_classes=['ql-upload'])
                        analyze_btn = gr.Button('Проанализировать', variant='primary', elem_classes=['ql-action'])
                    with gr.Column(scale=6, min_width=350, elem_classes=['ql-panel', 'ql-result-panel']):
                        gr.Markdown('### Обзор модели', elem_classes=['ql-panel-heading'])
                        model_overview = summary_html(EMPTY_MODEL)
                with gr.Accordion('Слои для квантования весов', open=False, elem_classes=['ql-details']):
                    candidates = gr.Dataframe(
                        headers=['Оператор', 'Узел', 'Форма весов', 'Ось', 'Каналы'],
                        datatype=['str', 'str', 'str', 'number', 'number'],
                        interactive=False, show_label=False,
                        elem_classes=['ql-table'],
                    )
                with gr.Accordion('Технические детали графа', open=False, elem_classes=['ql-details']):
                    structure = gr.JSON(value={}, show_label=False)

            with gr.Tab('02  Квантование'):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=4, min_width=310, elem_classes=['ql-panel']):
                        gr.Markdown('### Настройки', elem_classes=['ql-panel-heading'])
                        mode = gr.Dropdown(
                            choices=[(label, key) for key, label in MODE_LABELS.items()],
                            value='standard', label='Алгоритм', filterable=False,
                        )
                        mode_hint = gr.Markdown(MODE_HINTS['standard'], elem_classes=['ql-hint'])
                        with gr.Column() as calib_section:
                            calib_file = gr.File(label='Калибровка · JSON', file_types=['.json'], type='filepath', elem_classes=['ql-upload'])
                        work_limit = gr.Dropdown(choices=[('Малый · 250 млн', '250000000'), ('Обычный · 1 млрд', '1000000000'), ('Большой · 4 млрд', '4000000000')], value='1000000000', label='Бюджет поиска · условные единицы', visible=False)
                        quality_task = gr.Dropdown(choices=[('Автоматически', 'auto'), ('Общий / несколько выходов','generic'), ('Классификация · логиты [batch, classes]','classification')], value='auto', label='Тип задачи Quality', visible=False)
                        quant_btn = gr.Button('Квантовать модель', interactive=False, variant='primary', elem_classes=['ql-action'])
                    with gr.Column(scale=6, min_width=350, elem_classes=['ql-panel', 'ql-result-panel']):
                        gr.Markdown('### Результат', elem_classes=['ql-panel-heading'])
                        quant_overview = summary_html(EMPTY_QUANT)
                        with gr.Row(visible=False) as downloads:
                            quant_download = gr.File(label='INT8 ONNX', interactive=False, elem_classes=['ql-download'])
                            report_download = gr.File(label='JSON-отчёт', interactive=False, elem_classes=['ql-download'])
                with gr.Accordion('Подробные метрики', open=False, elem_classes=['ql-details']):
                    quant_metrics = gr.JSON(value={}, show_label=False)

            with gr.Tab('03  Проверка'):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=4, min_width=310, elem_classes=['ql-panel']):
                        gr.Markdown('### Входные данные', elem_classes=['ql-panel-heading'])
                        kind = gr.Radio(
                            choices=[JSON_INPUT], value=JSON_INPUT, label='Формат', elem_classes=['ql-radio'],
                        )
                        with gr.Column(visible=False) as text_section:
                            input_text = gr.Textbox(
                                label='Предложение',
                                value='This movie is absolutely terrible.', lines=3,
                            )
                        with gr.Column() as json_section:
                            input_json = gr.Textbox(
                                label='Входные тензоры · JSON', lines=6,
                                placeholder='{"input": [[1.0, 2.0, 3.0]]}',
                            )
                        infer_btn = gr.Button('Запустить инференс', interactive=False, variant='primary', elem_classes=['ql-action'])
                    with gr.Column(scale=6, min_width=350, elem_classes=['ql-panel', 'ql-result-panel']):
                        gr.Markdown('### Предсказания', elem_classes=['ql-panel-heading'])
                        inference_overview = summary_html(EMPTY_INFER)
                with gr.Accordion('Формы и первые значения выходов', open=False, elem_classes=['ql-details']):
                    inference_result = gr.JSON(value={}, show_label=False)

            with gr.Tab('04  Датасет'):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=4, min_width=310, elem_classes=['ql-panel']):
                        gr.Markdown('### Источник датасета', elem_classes=['ql-panel-heading'])
                        dataset_file = gr.File(
                            label='Датасет · CSV / JSON / JSONL',
                            file_types=['.csv', '.json', '.jsonl'], type='filepath', elem_classes=['ql-upload'],
                        )
                        dataset_hint = gr.Markdown(
                            'Для BERT-Tiny по умолчанию используются 872 размеченных текста SST-2. '
                            'Загрузка файла необязательна.', visible=False,
                            elem_classes=['ql-hint'],
                        )
                        dataset_btn = gr.Button(
                            'Оценить датасет', interactive=False, variant='primary', elem_classes=['ql-action'],
                        )
                    with gr.Column(scale=6, min_width=350, elem_classes=['ql-panel', 'ql-result-panel']):
                        gr.Markdown('### Метрики', elem_classes=['ql-panel-heading'])
                        dataset_overview = summary_html(EMPTY_DATASET)
                        with gr.Row(visible=False) as evaluation_download:
                            dataset_download = gr.File(label='Отчёт оценки · JSON', interactive=False, elem_classes=['ql-download'])
                with gr.Accordion('Детальные результаты и матрицы ошибок', open=False, elem_classes=['ql-details']):
                    dataset_result = gr.JSON(value={}, show_label=False)

        analysis_outputs = [
            prepared_state, quantized_state, is_demo_state, model_overview, candidates, structure,
            quant_overview, quant_metrics, quant_download, report_download,
            inference_overview, inference_result, dataset_overview, dataset_result, dataset_download,
            header, kind, quant_btn, infer_btn, dataset_btn, downloads, evaluation_download, dataset_hint,
        ]
        check_outputs = [inference_overview, inference_result, dataset_overview, dataset_result, dataset_download]

        example.change(
            change_source,
            inputs=[example], outputs=[uploaded, demo_info],
        )
        gr.on([uploaded.change, example.change], reset_source,
              outputs=[*analysis_outputs, calib_file, dataset_file, input_json], concurrency_id='workspace')
        analyze_btn.click(analyze_ui, inputs=[uploaded, example], outputs=analysis_outputs, concurrency_id='workspace')
        quant_btn.click(
            quantize_ui,
            inputs=[prepared_state, mode, calib_file, is_demo_state, work_limit, quality_task],
            outputs=[quantized_state, quant_download, report_download, quant_metrics, quant_overview,
                     *check_outputs, dataset_btn, downloads, evaluation_download], concurrency_id='workspace',
        )
        infer_btn.click(
            infer_ui, inputs=[prepared_state, quantized_state, kind, input_text, input_json],
            outputs=[inference_overview, inference_result], concurrency_id='workspace',
        )
        dataset_btn.click(
            dataset_ui, inputs=[prepared_state, quantized_state, dataset_file, is_demo_state],
            outputs=[dataset_overview, dataset_result, dataset_download, evaluation_download], concurrency_id='workspace',
        )
        mode.change(
            change_mode,
            inputs=[mode],
            outputs=[mode_hint, calib_section, work_limit, quality_task, quantized_state, quant_overview, quant_metrics,
                     quant_download, report_download, *check_outputs, dataset_btn, downloads, evaluation_download],
            concurrency_id='workspace',
        )
        kind.change(
            change_input_kind,
            inputs=[kind], outputs=[text_section, json_section, inference_overview, inference_result],
        )
    return app


def make_theme():
    """Согласовать встроенные компоненты Gradio с нейтральной палитрой."""
    return gr.themes.Base(
        primary_hue=gr.themes.colors.blue,
        neutral_hue=gr.themes.colors.gray,
        spacing_size=gr.themes.sizes.spacing_sm,
        radius_size=gr.themes.sizes.radius_sm,
        font=['Segoe UI', 'system-ui', 'sans-serif'],
        font_mono=['Consolas', 'ui-monospace', 'monospace'],
    ).set(
        body_background_fill='#f7f8fa', body_background_fill_dark='#101114',
        body_text_color='#20232a', body_text_color_dark='#e9eaee',
        body_text_color_subdued='#656d7b', body_text_color_subdued_dark='#a0a6b2',
        background_fill_primary='#ffffff', background_fill_primary_dark='#17191e',
        background_fill_secondary='#f0f2f5', background_fill_secondary_dark='#20232a',
        border_color_primary='#dce0e6', border_color_primary_dark='#323640',
        block_background_fill='transparent', block_background_fill_dark='transparent',
        block_border_color='#dce0e6', block_border_color_dark='#323640',
        block_label_background_fill='transparent', block_label_background_fill_dark='transparent',
        block_label_text_color='#525b6a', block_label_text_color_dark='#b7bdc9',
        block_label_border_width='0px', block_label_border_width_dark='0px',
        block_title_background_fill='transparent', block_title_background_fill_dark='transparent',
        block_title_text_color='#525b6a', block_title_text_color_dark='#b7bdc9',
        block_title_border_width='0px', block_title_border_width_dark='0px',
        block_shadow='none', block_shadow_dark='none',
        panel_background_fill='transparent', panel_background_fill_dark='transparent',
        panel_border_width='0px', panel_border_width_dark='0px',
        input_background_fill='#ffffff', input_background_fill_dark='#1e2128',
        input_border_color='#d4d9e1', input_border_color_dark='#363b45',
        input_background_fill_hover='#f7f8fa', input_background_fill_hover_dark='#242831',
        input_border_color_focus='#396ad7', input_border_color_focus_dark='#81a6ff',
        input_shadow='none', input_shadow_dark='none',
        button_primary_background_fill='#285ccc', button_primary_background_fill_dark='#376bdf',
        button_primary_background_fill_hover='#204cae', button_primary_background_fill_hover_dark='#4479eb',
        button_primary_text_color='#ffffff', button_primary_text_color_dark='#ffffff',
        button_primary_border_color='#285ccc', button_primary_border_color_dark='#376bdf',
        button_primary_shadow='none', button_primary_shadow_dark='none',
        button_secondary_background_fill='#f0f2f5', button_secondary_background_fill_dark='#20232a',
        button_secondary_text_color='#20232a', button_secondary_text_color_dark='#e9eaee',
        button_secondary_border_color='#dce0e6', button_secondary_border_color_dark='#363b45',
        checkbox_label_background_fill='#f0f2f5', checkbox_label_background_fill_dark='#20232a',
        checkbox_label_background_fill_selected='#e8efff', checkbox_label_background_fill_selected_dark='#243451',
        checkbox_label_text_color_selected='#244fab', checkbox_label_text_color_selected_dark='#a8c2ff',
        checkbox_label_border_color_selected='#a8bff2', checkbox_label_border_color_selected_dark='#4c6da6',
        checkbox_background_color_selected='#285ccc', checkbox_background_color_selected_dark='#376bdf',
        color_accent='#396ad7', color_accent_soft='#e8efff', color_accent_soft_dark='#243451',
        table_border_color='#dce0e6', table_border_color_dark='#323640',
        table_even_background_fill='#f7f8fa', table_even_background_fill_dark='#1b1e24',
        table_odd_background_fill='#ffffff', table_odd_background_fill_dark='#17191e',
    )


def launch(app):
    """Запустить локальный сервис с существующими настройками Docker."""
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    app.queue(default_concurrency_limit=1, max_size=8)
    app.launch(
        server_name='0.0.0.0', server_port=7860, theme=make_theme(),
        css=(ASSETS / 'theme.css').read_text(encoding='utf-8'),
        footer_links=[], allowed_paths=[str(DATA_DIR)], show_error=True,
    )
