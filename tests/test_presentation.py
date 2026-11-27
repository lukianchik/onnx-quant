"""Проверка краткости и безопасного экранирования сводок интерфейса."""

from quantlab.presentation import empty_state, inference_summary, model_summary, quant_summary


def test_model_summary_escapes_uploaded_name():
    report = {
        'candidates': [{'node': 'dense'}],
        'candidate_parameters': 120,
        'size_mib': 2.5,
        'nodes': 12,
        'initializers': 3,
    }
    output = model_summary(report, '<script>alert(1)</script>.onnx')
    assert '<script>' not in output
    assert '&lt;script&gt;' in output
    assert '2.50 MiB' in output
    assert 'Найдены кандидаты' in output
    assert 'Dynamic' not in output and 'Static QDQ' not in output
    assert '120 параметров' in output


def test_quantization_summary_is_human_readable():
    metrics = {
        'mode': 'standard', 'selected_count': 12, 'coverage_percent': 50,
        'output_fidelity': {'status': 'measured'},
        'size_reduction_percent': 6.39,
        'fp32_size_mib': 16.8, 'int8_size_mib': 15.73,
    }
    output = quant_summary(metrics)
    assert 'Standard · MinMax' in output
    assert '+6.39%' in output
    assert 'Проверка пройдена' in output
    assert '12' in output


def test_prediction_summary_shows_confidence_and_difference():
    results = {
        'FP32': {
            'classification': {'label': 'Negative', 'confidence': 0.993},
        },
        'INT8': {
            'classification': {'label': 'Negative', 'confidence': 0.991},
        },
        'comparison': {'max_absolute_output_difference': 0.0123},
    }
    summary = inference_summary(results)
    assert 'Негативный' in summary
    assert '99.3%' in summary
    assert '99.1%' in summary
    assert '0.012300' in summary


def test_empty_state_escapes_text():
    output = empty_state('<unsafe>', '& details')
    assert '<unsafe>' not in output
    assert '&lt;unsafe&gt;' in output
    assert '&amp; details' in output


def test_dataset_summary_does_not_fake_accuracy_without_labels():
    from quantlab.presentation import dataset_summary
    report = {
        'dataset': '<bad>.csv', 'samples': 4, 'labeled': False,
        'mean_absolute_output_difference': .01,
        'max_absolute_output_difference': .05,
        'speedup': 1.2,
    }
    output = dataset_summary(report)
    assert '<bad>' not in output
    assert 'Accuracy/F1 недоступны' in output
    assert '1.20×' in output


def test_html_component_css_is_scoped_and_complete():
    from pathlib import Path
    import gradio as gr

    path = Path(__file__).parents[1] / 'assets' / 'components.css'
    css = path.read_text(encoding='utf-8')
    for component in ('.ql-stat-grid', '.ql-prediction-grid', '.ql-topbar', '.ql-result-head'):
        assert component in css
    component = gr.HTML('<div class="ql-stat-grid"></div>', css_template=css, apply_default_css=False)
    assert component.css_template == css
    assert component.apply_default_css is False


def test_standard_summary_without_inputs():
    """Не выдавать структурную проверку за проверку численной точности."""
    metrics = {'mode': 'standard', 'selected_count': 1, 'size_reduction_percent': 20,
               'fp32_size_mib': 1, 'int8_size_mib': 0.8, 'coverage_percent': 25,
               'output_fidelity': {'status': 'not_run_no_inputs'}}
    html = quant_summary(metrics)
    assert 'Standard · MinMax' in html
    assert 'Численная точность не проверена' in html
    assert '25.00%' in html


def test_fast_summary_fallback():
    """Показать откат к MinMax вместо ложного сообщения об улучшении."""
    metrics = {'mode':'fast','selected_count':1,'size_reduction_percent':20,
               'fp32_size_mib':1,'int8_size_mib':.8,'coverage_percent':25,
               'output_fidelity':{'status':'measured'},'global_guard':{'accepted':False}}
    html = quant_summary(metrics)
    assert 'Fast · Single-Robust' in html and 'Global Guard: MinMax' in html


def test_balanced_summary():
    """Показать принятый Progressive либо явный откат к MinMax."""
    metrics = {'mode':'balanced','selected_count':1,'size_reduction_percent':20,
               'fp32_size_mib':1,'int8_size_mib':.8,'coverage_percent':25,
               'output_fidelity':{'status':'measured'},'global_guard':{'accepted':True}}
    assert 'Global Guard: принят Progressive' in quant_summary(metrics)
    metrics['global_guard']['accepted'] = False
    assert 'Global Guard: MinMax' in quant_summary(metrics)


def test_quality_summary():
    """Показать фактически принятую политику и расход глобальных проверок."""
    metrics = {'mode':'quality','selected_count':1,'size_reduction_percent':20,
               'fp32_size_mib':1,'int8_size_mib':.8,'coverage_percent':25,
               'output_fidelity':{'status':'measured'},'global_guard':{'accepted':True},
               'effective_algorithm':'Progressive-Global','global_search':{'calls':6,'cap':9}}
    html=quant_summary(metrics)
    assert 'Quality · Progressive-Global' in html
    assert 'Global Guard: принят Progressive-Global' in html and '6/9' in html
    metrics['effective_algorithm']='Progressive'
    assert 'Global Guard: принят Progressive ·' in quant_summary(metrics)
    metrics['global_guard']['accepted']=False
    assert 'Global Guard: MinMax' in quant_summary(metrics)
