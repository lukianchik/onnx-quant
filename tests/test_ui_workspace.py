"""Регрессии компоновки, темы и контрактов событий интерфейса."""

from quantlab.ui import actions
from quantlab.ui.workspace import create_app, make_theme


def test_workspace_tabs_and_modes():
    """Все вкладки и четыре режима доступны после рефакторинга."""
    config = create_app().get_config_file()
    components = config['components']
    assert [item['props']['label'] for item in components if item['type'] == 'tabitem'] == [
        '01  Модель', '02  Квантование', '03  Проверка', '04  Датасет',
    ]
    mode = next(item['props'] for item in components if item['type'] == 'dropdown'
                and item['props'].get('label') == 'Алгоритм')
    assert [key for _, key in mode['choices']] == ['standard', 'fast', 'balanced', 'quality']
    assert set(actions.MODE_HINTS) == {'standard', 'fast', 'balanced', 'quality'}
    assert mode['value'] == 'standard'


def test_workspace_event_contracts():
    """Количество выходов обработчиков соответствует компонентам Gradio."""
    deps = {d['api_name']: d for d in create_app().get_config_file()['dependencies']}
    assert len(actions.reset_source()) == len(deps['reset_source']['outputs']) == 26
    for mode in actions.MODE_HINTS:
        assert len(actions.change_mode(mode)) == len(deps['change_mode']['outputs']) == 17
    assert len(actions.change_input_kind(actions.JSON_INPUT)) == len(deps['change_input_kind']['outputs']) == 4
    assert len(actions.change_source(actions.DEMO_LABEL)) == len(deps['change_source']['outputs']) == 2
    for name, count in [('analyze_ui', 23), ('quantize_ui', 13), ('infer_ui', 2), ('dataset_ui', 4)]:
        assert len(deps[name]['outputs']) == count


def test_workspace_neutral_theme():
    """Основные цвета оболочки и кнопок заданы для обеих тем."""
    theme = make_theme()
    assert theme.body_background_fill == '#f7f8fa'
    assert theme.body_background_fill_dark == '#101114'
    assert theme.button_primary_background_fill_dark == '#376bdf'
