"""Проверки MinMax и расположения каналов без ONNX Runtime."""

import numpy as np
import pytest

from quantlab.weight_only.minmax import minmax
from quantlab.weight_only.graph import Target


def test_minmax_extreme_and_zero():
    """Нули, субнормальные и предельные FP32 остаются конечными."""
    w = np.array([[0, np.finfo(np.float32).max, 1e-45],
                  [0, -np.finfo(np.float32).max, 0]], dtype=np.float32)
    q, s = minmax(w)
    assert q.dtype == np.int8 and s.dtype == np.float32
    assert np.all(q >= -127) and np.all(q <= 127)
    assert np.all(q[:, 0] == 0) and s[0] == 1
    assert np.isfinite(q * s).all() and np.all(s > 0)


@pytest.mark.parametrize('op,shape,attrs', [
    ('MatMul', (3, 5), {}), ('Gemm', (5, 3), {'transB': 1}),
    ('Conv', (4, 2, 3), {'group': 2}),
    ('Conv', (4, 1, 3, 3), {'group': 4}),
    ('Conv', (4, 2, 2, 2, 2), {'group': 2}),
])
def test_storage_layout(op, shape, attrs):
    """Проверить коды и масштабы в исходном порядке каналов."""
    w = np.random.default_rng(24).normal(size=shape).astype(np.float32)
    t = Target('W', 'node', op, 'X', 'Y', w, attrs)
    q, s = t.pack([minmax(m) for m in t.matrices()])
    expected_q, expected_s = minmax(np.moveaxis(w, t.axis, -1).reshape(-1, w.shape[t.axis]))
    expected_q = np.moveaxis(expected_q.reshape(np.moveaxis(w, t.axis, -1).shape), -1, t.axis)
    np.testing.assert_array_equal(q, expected_q)
    np.testing.assert_array_equal(s, expected_s)
    stored_q, stored_s = t.stored(t.canonical(q, s))
    np.testing.assert_array_equal(stored_q, q)
    np.testing.assert_array_equal(stored_s, s)


def test_nonfinite_rejected():
    """Неконечные веса запрещены для MinMax."""
    with pytest.raises(ValueError):
        minmax(np.array([[np.nan]], dtype=np.float32))


def test_blocked_minmax_preserves_codes_at_rounding_boundaries():
    """Проверить несколько блоков, полуцелые значения и неконтинуальные веса."""
    values = np.array([0, 1.5, 2.5, -3.5, 126.5, -127], dtype=np.float32)
    weights = np.tile(values, (512, 128)).T
    codes, scales = minmax(weights)
    expected = np.rint(weights.astype(np.float64)).astype(np.int8)
    np.testing.assert_array_equal(codes, expected)
    np.testing.assert_array_equal(scales, np.ones(weights.shape[1], dtype=np.float32))
