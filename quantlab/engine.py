"""Единая точка входа для четырёх режимов квантования весов."""

import json
from pathlib import Path

from quantlab.core import ModelError, load_model
from quantlab.runtime import make_feed, session, text_batches


class JSONCalibrationReader:
    """Читать числовые пакеты JSON или текстовые примеры для демомодели BERT."""

    def __init__(self, model_path: Path, json_path: Path, tokenizer_id: str | None = None):
        sess = session(model_path)
        loaded = json.loads(Path(json_path).read_text(encoding='utf-8'))
        if isinstance(loaded, dict) and 'samples' in loaded:
            loaded = loaded['samples']
        if not isinstance(loaded, list) or not loaded or len(loaded) > 5000:
            raise ModelError('Калибровка: ожидается JSON-массив из 1–5000 примеров')
        if all(isinstance(x, dict) and isinstance(x.get('text'), str) for x in loaded):
            if not tokenizer_id:
                raise ModelError('Для текстовой калибровки нужен токенизатор BERT-Tiny')
            texts = [sample['text'] for sample in loaded]
            self._feeds = [feed for feed, _ in text_batches(sess, texts, tokenizer_id)]
        else:
            self._feeds = [make_feed(sess, item) for item in loaded]
        del loaded
        for feed in self._feeds:
            try:
                sess.run(None, feed)
            except Exception as exc:
                raise ModelError(f'Некорректный пример калибровки: {exc}') from exc

    def __len__(self):
        """Вернуть число пакетов калибровки."""
        return len(self._feeds)

    def __iter__(self):
        """Перебирать подготовленные пакеты повторно для каждой стадии."""
        return iter(self._feeds)


def quantize(path: Path, mode: str, calibration_file: Path | None = None,
             tokenizer_id: str | None = None, work_limit: int | None = None, task: str = 'generic') -> tuple[Path, dict]:
    """Квантовать модель выбранным методом и вернуть путь к ONNX и метрики."""
    path = Path(path)
    if mode not in ('standard', 'fast', 'balanced', 'quality'):
        raise ModelError(f'Неизвестный режим: {mode}')
    from quantlab.weight_only.engine import quantize_standard, quantize_fast, quantize_balanced, quantize_quality

    feeds = ()
    if calibration_file is not None:
        # Запрет внешних файлов действует до первой сессии ORT.
        load_model(path)
        feeds = JSONCalibrationReader(path, Path(calibration_file), tokenizer_id)
    if mode == 'standard':
        return quantize_standard(path, feeds)
    if calibration_file is None:
        raise ModelError(f'Для {mode} нужен JSON калибровочных данных')
    if mode == 'fast':
        return quantize_fast(path, feeds)
    if mode == 'balanced':
        return quantize_balanced(path, feeds, work_limit)
    return quantize_quality(path, feeds, work_limit, task)
