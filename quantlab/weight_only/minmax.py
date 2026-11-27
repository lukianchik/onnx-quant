"""MinMax из исследовательского ядра v4, без калибровочных активаций."""

import numpy as np

def _scale32(values):
    """Ограничить масштаб конечным положительным FP32."""
    # Любой из 255 симметричных кодов должен деквантоваться в конечный FP32.
    smallest = np.nextafter(np.float32(0), np.float32(1))
    largest = float(np.finfo(np.float32).max)
    upper = np.float32(largest / 127)
    if float(upper) * 127 > largest:
        upper = np.nextafter(upper, np.float32(0))
    return np.clip(values, float(smallest), float(upper)).astype(np.float32)



def minmax(weights):
    """Квантовать матрицу входы × выходы симметрично по каналам."""
    w = np.asarray(weights)
    if w.ndim != 2 or 0 in w.shape or w.dtype != np.float32:
        raise ValueError("Ожидается непустая конечная FP32-матрица")
    peak = np.maximum(w.max(axis=0), -w.min(axis=0)).astype(np.float64)
    if not np.isfinite(peak).all():
        raise ValueError("Ожидается непустая конечная FP32-матрица")
    scales = _scale32(np.where(peak == 0, 1.0, peak / 127))
    codes = np.empty_like(w, dtype=np.int8)
    # Сохраняем деление и округление в FP64, обрабатывая матрицу по частям.
    block_rows = max(1, 131072 // w.shape[1])
    for start in range(0, len(w), block_rows):
        rows = slice(start, start + block_rows)
        values = np.divide(w[rows], scales, dtype=np.float64)
        np.rint(values, out=values)
        np.clip(values, -127, 127, out=values)
        codes[rows] = values
    return codes, scales
