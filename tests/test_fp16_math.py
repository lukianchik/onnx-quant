"""Проверки FP16-адаптера без внешних моделей и ONNX Runtime."""
import numpy as np
import pytest

from quantlab.weight_only.calibration import sample_rows
from quantlab.weight_only.graph import Target, fp16_safe_scales
from quantlab.weight_only.minmax import minmax


@pytest.mark.parametrize('op,shape,attrs', [
    ('MatMul', (3, 4), {}), ('Gemm', (4, 3), {'transB': 1}),
    ('Conv', (4, 2, 3, 3), {'group': 2}),
])
def test_fp16_layout(op, shape, attrs):
    """Сохранить исходный тип и расположение каналов при поиске в FP32."""
    w = np.random.default_rng(22).normal(size=shape).astype(np.float16)
    t = Target('W', 'node', op, 'X', 'Y', w, attrs)
    matrices = t.matrices()
    assert all(m.dtype == np.float32 for m in matrices)
    q, s = t.pack([minmax(m) for m in matrices])
    ref = np.moveaxis(w.astype(np.float32), t.axis, -1)
    rq, rs = minmax(ref.reshape(-1, ref.shape[-1]))
    np.testing.assert_array_equal(q, np.moveaxis(rq.reshape(ref.shape), -1, t.axis))
    np.testing.assert_array_equal(s, rs)
    assert t.original.dtype == np.float16


def test_fp16_scale_bounds():
    """Ограничить экстремальные масштабы и оставить нулевой канал неизменным."""
    q = np.array([[127, 0, 2], [-127, 0, -2]], np.int8)
    s = np.array([1000, 1, 30000], np.float32)
    safe = fp16_safe_scales(q, s, 1)
    assert safe.dtype == np.float32 and np.all(safe > 0)
    assert safe[0] < s[0] and safe[1] == s[1] and safe[2] == s[2]
    assert np.isfinite((q * safe).astype(np.float16)).all()
    np.testing.assert_array_equal(s, [1000, 1, 30000])


def test_fp16_activation_capture():
    """Принять округлённые FP16-выходы и передать поиску строки FP32."""
    rng = np.random.default_rng(29)
    w = rng.normal(size=(3, 4)).astype(np.float16)
    x = rng.normal(size=(8, 3)).astype(np.float16)
    y = (x.astype(np.float32) @ w.astype(np.float32)).astype(np.float16)
    t = Target('W', 'dense', 'MatMul', 'X', 'Y', w, {})
    rows, n = sample_rows(x, y, t, t.matrices(), 6, rng)
    assert n == 8 and rows[0].shape == (6, 3) and rows[0].dtype == np.float32
    with pytest.raises(ValueError, match='adapter mismatch'):
        sample_rows(x, y + np.float16(5), t, t.matrices(), 6, rng)


def test_fp16_grouped_conv_capture():
    """Проверить выбор патчей сгруппированной свёртки и bias."""
    rng = np.random.default_rng(33)
    w = rng.normal(size=(4, 2, 2)).astype(np.float16)
    x = rng.normal(size=(2, 4, 5)).astype(np.float16)
    b = rng.normal(size=4).astype(np.float16)
    y = np.empty((2, 4, 4), np.float32)
    for n in range(2):
        for c in range(4):
            for j in range(4):
                group = c // 2
                y[n, c, j] = np.sum(x[n, group*2:group*2+2, j:j+2].astype(np.float32) * w[c].astype(np.float32)) + b[c]
    t = Target('W', 'conv', 'Conv', 'X', 'Y', w, {'group': 2}, b)
    rows, n = sample_rows(x, y.astype(np.float16), t, t.matrices(), 6, rng)
    assert n == 8 and len(rows) == 2
    assert all(r.shape == (6, 4) and r.dtype == np.float32 for r in rows)
