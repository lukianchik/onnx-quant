# Research / Исследования

Эксперимент v4 (24.11.2026) сравнивает стратегии выбора параметров INT8-квантования на восьми моделях. Исследовательский стенд имеет отдельное окружение и не является частью приложения.

- [Математические основы](../docs/MATHEMATICS.md)
- [Финальные результаты RU](results/12_benchmark_v4_final/README.ru.md) / [EN](results/12_benchmark_v4_final/README.md)
- [Итоговый notebook](notebooks/07_quantization_benchmark_v4_final_kaggle.ipynb) (без сохранённых outputs)
- `benchmark_v4/` — оригинальные исходники и тесты стенда.

Опубликованы агрегированные и подробные метрики v4: 396 успешных конфигураций, 12 ожидаемых пропусков V2, 533/534 проверенных операторных вхождения. Веса, исходные датасеты и необработанные журналы не распространяются. Отчёт основан на сохранённых измерениях.

## Проверка исходников на CPU

У исследовательского стенда отдельное окружение. `requirements-kaggle.txt` предполагает готовый Kaggle PyTorch/CUDA и Transformers <5; не устанавливайте его поверх окружения GUI. Для локальных unit/integration tests достаточно следующего набора (это не полное окружение воспроизведения pretrained-экспериментов):

```powershell
python -m venv .venv-research
.\.venv-research\Scripts\python.exe -m pip install numpy onnx onnxruntime threadpoolctl pytest torch scikit-learn psutil
$env:PYTHONPATH = "$PWD\research\benchmark_v4;$PWD\research\benchmark_v4\algorithms"
.\.venv-research\Scripts\python.exe -m pytest -q research/benchmark_v4 --basetemp=.local/research-tests
.\.venv-research\Scripts\python.exe research/benchmark_v4/benchmark.py --help
# Необязательный короткий CPU smoke, результаты только локально:
.\.venv-research\Scripts\python.exe research/benchmark_v4/benchmark.py --provider cpu --math-backend numpy --profile smoke --models regression --seeds 42 --methods MinMax --threads 1 --output .local/benchmark-smoke
```

На Linux/macOS используйте `.venv-research/bin/python` и `PYTHONPATH="$PWD/research/benchmark_v4:$PWD/research/benchmark_v4/algorithms"`. CUDA-проверки теста `test_cuda_math.py` выполняются только при доступном GPU; без него проверяется PyTorch CPU backend. По умолчанию `pytest` в корне запускает только тесты приложения (`pytest.ini`); исследовательские тесты запускайте явно с указанным `PYTHONPATH`.

Для полного pretrained benchmark нужны дополнительные зависимости из исходного requirements, загрузки моделей/данных и подходящее CPU/CUDA-окружение. Исторический финальный GPU-прогон использовал две T4. Загрузка pretrained-моделей в CLI использует Linux-модуль `fcntl`; для этой ветки нужен Linux/Kaggle, Windows-проверки синтетических моделей её не покрывают. Воспроизведение предполагает неизменные алгоритмы, пороги и разделение fit/dev/search/check/test.

## English

The v4 benchmark compares INT8 candidate-selection policies on eight models. The final reports, selected numeric evidence, self-contained notebook (outputs cleared), and extracted benchmark sources are retained. Raw archives, datasets, weights and earlier experiment dumps are excluded. The original benchmark used two T4 GPUs; policy costs are replay estimates, not product timings. Use the separate research environment above; see the final report for evaluation limitations.

## Методология разработки / Development methodology

При исследовании и разработке использовались LLM как вспомогательные инструменты. / LLMs were used as research and development aids.
