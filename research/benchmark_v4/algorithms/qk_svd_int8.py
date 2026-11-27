"""Квантование одной обычной головы внимания через факторизацию формы QK."""

import numpy as np


def _fp32(value, ndim):
    a = np.asarray(value)
    if a.dtype != np.float32 or a.ndim != ndim or not a.size or not np.isfinite(a).all():
        raise ValueError("Ожидается непустой конечный массив FP32 заданной размерности.")
    return a


def minmax_int8(weights):
    """Возвращает матрицу кодов и вектор масштабов по её выходным столбцам."""
    w = _fp32(weights, 2).astype(np.float64)
    peak = np.max(np.abs(w), axis=0)
    limit = float(np.finfo(np.float32).max)
    upper = np.float32(limit / 127)
    if 127 * float(upper) > limit:
        upper = np.nextafter(upper, np.float32(0))
    lower = np.nextafter(np.float32(0), np.float32(1))
    s = np.clip(np.where(peak > 0, peak / 127, 1), lower, upper).astype(np.float32)
    return np.rint(w / s).clip(-127, 127).astype(np.int8), s


def _augment(w, bias):
    if bias is None:
        return w
    b = _fp32(bias, 1)
    if b.shape != (w.shape[1],):
        raise ValueError("Длина смещения должна совпадать с шириной головы.")
    return np.vstack((w, b))


def _balanced(a, b):
    qa, ra = np.linalg.qr(a)
    qb, rb = np.linalg.qr(b)
    u, s, vt = np.linalg.svd(ra @ rb.T, full_matrices=False)
    root = np.sqrt(s)
    left, right = np.zeros_like(a), np.zeros_like(b)
    left[:, :s.size] = (qa @ u) * root
    right[:, :s.size] = (qb @ vt.T) * root
    return left, right


def _pack(factor, rows):
    q, s = minmax_int8(factor[:rows])
    bias = factor[-1].copy() if factor.shape[0] > rows else None
    return q, s, bias


def _decoded(packed):
    q, s, bias = packed
    return _augment(q * s, bias)


def _core_blocks(matrices):
    joined = np.concatenate(matrices, axis=1).astype(np.float64, copy=False)
    return np.hsplit(np.linalg.qr(joined, mode="r"), len(matrices))


def quantize_qk_head(query, key, query_bias=None, key_bias=None):
    """Возвращает две тройки (коды, масштабы, смещение) и признак выбора SVD.

    Входы — FP32-матрицы одной головы в ориентации (входы, ширина головы).
    Смещения необязательны. Допустим прямой скалярный продукт Q и K без
    дополнительных потребителей, позиционных вращений и нормализации Q/K.
    """
    wq, wk = _fp32(query, 2), _fp32(key, 2)
    if wq.shape[1] != wk.shape[1]:
        raise ValueError("Ширины проекций одной головы должны совпадать.")
    a, b = _augment(wq, query_bias), _augment(wk, key_bias)
    base_a, base_b = _pack(a, len(wq)), _pack(b, len(wk))
    fallback = base_a, base_b, False
    a, b = a.astype(np.float64), b.astype(np.float64)
    try:
        left, right = _balanced(a, b)
        limit = float(np.finfo(np.float32).max)
        if max(np.max(abs(left)), np.max(abs(right))) > limit:
            return fallback
        left, right = left.astype(np.float32), right.astype(np.float32)
        test_a, test_b = _pack(left, len(wq)), _pack(right, len(wk))
        ar = _core_blocks((a, left, _decoded(base_a), _decoded(test_a)))
        br = _core_blocks((b, right, _decoded(base_b), _decoded(test_b)))
    except np.linalg.LinAlgError:
        return fallback

    ref, folded, old, new = [x @ y.T for x, y in zip(ar, br)]
    norm = np.linalg.norm
    size = max(*a.shape, *b.shape, 4 * a.shape[1])
    magnitude = max(norm(x) * norm(y) for x, y in zip(ar, br))
    guard = 16 * np.finfo(np.float64).eps * size * magnitude
    if norm(folded - ref) + guard > 8 * float(np.finfo(np.float32).eps) * norm(ref):
        return fallback
    if norm(new - ref) + guard >= norm(old - ref):
        return fallback
    return test_a, test_b, True
