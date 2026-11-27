"""Минимальные синтетические проверки квантования весов."""

import unittest

import numpy as np

from activation_int8 import quantize_int8


class QuantizationTests(unittest.TestCase):
    def test_paired_move_escapes_single_coordinate_minimum(self):
        w = np.array([[0.25], [0], [127]], dtype=np.float32)
        for sign in (-1, 1):
            with self.subTest(sign=sign):
                x = np.array([[4, sign * 5, 0], [0, 0, 1]], dtype=np.float32)
                q, s = quantize_int8(x, w, sweeps=1)
                np.testing.assert_array_equal(q[:, 0], [-1, sign, 127])
                np.testing.assert_array_equal(s, [1])
                np.testing.assert_array_equal(x @ (q * s), x @ w)

    def test_least_squares_scale(self):
        x = np.eye(2, dtype=np.float32)
        w = np.array([[127], [64.49]], dtype=np.float32)
        q, s = quantize_int8(x, w, sweeps=1)
        expected = (127**2 + 64 * float(w[1, 0])) / (127**2 + 64**2)
        np.testing.assert_array_equal(q[:, 0], [127, 64])
        np.testing.assert_array_equal(s, [np.float32(expected)])

    def test_minmax_bound_storage_and_repeatability(self):
        rng = np.random.default_rng(17)
        x = rng.normal(size=(7, 35)).astype(np.float32)
        x[:, 1] = x[:, 0]
        x[:, 2] = 0
        w = rng.normal(size=(35, 33)).astype(np.float32)
        w[:, -1] = 0
        peak = np.max(abs(w.astype(np.float64)), axis=0)
        base_s = np.where(peak == 0, 1, peak / 127).astype(np.float32)
        base_q = np.rint(w.astype(np.float64) / base_s).clip(-127, 127).astype(np.int8)
        q0, s0 = quantize_int8(x, w, sweeps=0)
        np.testing.assert_array_equal(q0, base_q)
        np.testing.assert_array_equal(s0, base_s)
        q, s = quantize_int8(x, w)
        self.assertEqual((q.shape, s.shape), (w.shape, (33,)))
        self.assertEqual((q.dtype, s.dtype), (np.int8, np.float32))
        self.assertTrue(np.all(q >= -127) and np.all(s > 0))
        self.assertEqual(q.nbytes + s.nbytes, w.size + 4 * w.shape[1])
        for actual, repeated in zip((q, s), quantize_int8(x, w)):
            np.testing.assert_array_equal(actual, repeated)

        def loss(codes, scales):
            diff = w.astype(np.float64) - (codes * scales)
            return np.sum((x.astype(np.float64) @ diff) ** 2, axis=0)

        self.assertTrue(np.all(loss(q, s) <= loss(base_q, base_s)))

    def test_degenerate_and_extreme_inputs(self):
        tiny = np.nextafter(np.float32(0), np.float32(1))
        w = np.array([[tiny, np.finfo(np.float32).max, 0]], dtype=np.float32)
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            for value in (0, 1):
                q, s = quantize_int8([[value]], w)
                self.assertTrue(np.isfinite(q * s).all() and np.all(s > 0))
                self.assertEqual((q * s)[0, 0], tiny)
                self.assertEqual((q[0, 2], s[2]), (0, 1))
                if value == 0:
                    q0, s0 = quantize_int8([[0]], w, sweeps=0)
                    np.testing.assert_array_equal(q, q0)
                    np.testing.assert_array_equal(s, s0)
        for x, weights in (([[np.nan]], [[0]]), ([[1, 2]], [[3]])):
            with self.assertRaises(ValueError):
                quantize_int8(x, weights)


if __name__ == "__main__":
    unittest.main()
