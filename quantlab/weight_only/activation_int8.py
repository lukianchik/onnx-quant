"""Симметричное INT8-квантование весов с учётом калибровочных активаций."""

import numpy as np

from quantlab.weight_only.minmax import _scale32, minmax


def _matrix(values):
    """Проверить и привести матрицу к вычислительному типу."""
    values = np.asarray(values)
    if values.ndim != 2 or 0 in values.shape or values.dtype.kind not in "fiu":
        raise ValueError("Ожидается непустая двумерная вещественная матрица.")
    with np.errstate(over="ignore", invalid="ignore"):
        values = values.astype(np.float32, copy=False)
    if not np.isfinite(values).all():
        raise ValueError("Элементы должны быть конечными в формате FP32.")
    return values


def _pairs(x):
    """Построить коррелированные пары признаков."""
    pairs = []
    for start in range(0, x.shape[1], 32):
        block = x[:, start:start + 32]
        gram = block.T @ block
        energy = gram.diagonal()
        total = energy[:, None] + energy
        scores = np.divide(2 * abs(gram), total, out=np.zeros_like(gram),
                           where=total > 0)
        for i in range(len(gram) - 1):
            j = i + 1 + int(np.argmax(scores[i, i + 1:]))
            if scores[i, j] > 0:
                sign = -1 if gram[i, j] > 0 else 1
                pairs.append((start + i, start + j, sign))
                scores[:, j] = 0
                scores[j, :] = 0
    return pairs


def _move(residual, direction, codes, scales, i, j=None, sign=1):
    """Обновить коды и остаток координатным шагом."""
    energy = direction @ direction
    if energy == 0:
        return
    low, high = -127 - codes[i], 127 - codes[i]
    if j is not None:
        second = sign * codes[j]
        low = np.maximum(low, -127 - second)
        high = np.minimum(high, 127 - second)
    projection = np.einsum("n,nb->b", direction, residual)
    step = np.rint(np.clip(projection / (energy * scales), low, high)).astype(np.int16)
    if np.any(step):
        codes[i] += step
        if j is not None:
            codes[j] += sign * step
        residual -= direction[:, None] * (scales * step)


def _loss(x, weights, codes, scales):
    """Вычислить потери по выходным каналам."""
    decoded = np.multiply(codes, scales, dtype=np.float32)
    residual = x @ (weights - decoded)
    return np.einsum("nb,nb->b", residual, residual)


def quantize_int8(activations, weights, *, sweeps=3, paired=True):
    """Возвращает (коды INT8, масштабы FP32) для произведения X @ W.

    X имеет форму (наблюдения, входы), W — (входы, выходы). Вещественные
    входные данные приводятся к FP32. Коды имеют форму W, масштабы — (выходы,).
    Параметр sweeps ограничивает число проходов. Ноль возвращает MinMax.
    paired=False отключает парные шаги (Single-3).
    """
    x, w = _matrix(activations), _matrix(weights)
    if x.shape[1] != w.shape[0]:
        raise ValueError("Число входных признаков X и W должно совпадать.")
    integer = isinstance(sweeps, (int, np.integer)) and not isinstance(sweeps, (bool, np.bool_))
    if not integer or sweeps < 0:
        raise ValueError("Число проходов должно быть целым неотрицательным.")

    codes, scales = minmax(w)
    if sweeps == 0:
        return codes, scales
    x, w = x.astype(np.float64), w.astype(np.float64)
    pairs = _pairs(x) if paired else []

    for start in range(0, w.shape[1], 32):
        cols = slice(start, start + 32)
        weights_block, best, stored = w[:, cols], codes[:, cols], scales[cols]
        best_loss = _loss(x, weights_block, best, stored)
        for _ in range(sweeps):
            if not np.any(best_loss):
                break
            scale = stored.astype(np.float64)
            trial = best.astype(np.int16)
            residual = x @ (weights_block - trial * scale)
            for i in range(x.shape[1]):
                _move(residual, x[:, i], trial, scale, i)
            for i, j, sign in pairs:
                _move(residual, x[:, i] + sign * x[:, j], trial, scale, i, j, sign)

            # Пересчитываем остаток перед одномерной задачей по масштабу.
            residual = x @ (weights_block - trial * scale)
            response = x @ trial
            energy = np.einsum("nb,nb->b", response, response)
            cross = np.einsum("nb,nb->b", response, residual)
            delta = np.divide(cross, energy, out=np.zeros_like(cross), where=energy > 0)
            fitted = _scale32(scale + delta)

            improved = False
            for candidate in (stored.copy(), fitted):
                loss = _loss(x, weights_block, trial, candidate)
                keep = loss < best_loss
                if np.any(keep):
                    best[:, keep] = trial[:, keep]
                    stored[keep] = candidate[keep]
                    best_loss[keep] = loss[keep]
                    improved = True
            if not improved:
                break
    return codes, scales
