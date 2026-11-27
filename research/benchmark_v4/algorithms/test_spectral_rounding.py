"""Синтетические проверки геометрии, защитного механизма и формата INT8."""
import unittest
from unittest.mock import patch
import numpy as np
from spectral_rounding import Config, _round_block, quantize

class SpectralTests(unittest.TestCase):
    def test_potential_and_coordinated_rounding(self):
        gram = np.ones((6, 6))
        low, high = np.full(6, -.49), np.full(6, .51)
        end = _round_block(np.zeros(6), low, high, gram, np.zeros(6), 1, 1e-12)
        self.assertTrue(np.all((end == low) | (end == high)))
        self.assertLessEqual(end @ gram @ end, -low @ high + 1e-12)
        self.assertLess(end @ gram @ end, low @ gram @ low)
        self.assertGreater(np.count_nonzero(end == high), 1)
    def test_full_score_and_determinism(self):
        rng = np.random.default_rng(31)
        x = rng.normal(size=(5, 22)).astype(np.float32)
        x[:, 1:4] = x[:, :1]; x[:, -1] = 0
        w = rng.normal(size=(22, 5)).astype(np.float32); w[:, -1] = 0
        base = quantize(w, None); new = quantize(w, x)
        def loss(result):
            residual = x.astype(float) @ ((result.codes*result.scales).astype(float)-w)
            return np.sum(residual**2, axis=0)
        self.assertTrue(np.all(loss(new) <= loss(base)))
        np.testing.assert_array_equal(new.scales, base.scales)
        np.testing.assert_array_equal(new.codes[-1], base.codes[-1])
        self.assertTrue(new.applied)
        np.testing.assert_array_equal(new.codes, quantize(w, x).codes)
        self.assertEqual((new.codes.dtype, new.scales.dtype), (np.int8, np.float32))
        self.assertEqual(new.codes.nbytes+new.scales.nbytes, w.size+4*w.shape[1])
    def test_resource_and_degeneracy_fallback(self):
        w = np.eye(3, dtype=np.float32)
        for x, cfg, reason in ((w, Config(memory_limit=0), "budget_or_width"), (w, Config(work_limit=0), "budget_or_width"), (w, Config(), "flat_blocks"), (w[:0], Config(), "invalid_calibration"), (w*np.nan, Config(), "invalid_calibration")):
            result = quantize(w, x, cfg)
            self.assertFalse(result.applied); self.assertEqual(result.diagnostics["reason"], reason)
            np.testing.assert_array_equal(result.codes, quantize(w, None).codes)
        with patch("numpy.linalg.eigvalsh", side_effect=np.linalg.LinAlgError):
            self.assertEqual(quantize(w, w).diagnostics["reason"], "factorization_failed")
    def test_extreme_weights_and_shape(self):
        tiny = np.nextafter(np.float32(0), np.float32(1)); big = np.finfo(np.float32).max
        w = np.array([[big, tiny, 0], [-big/2, -tiny, 0], [1, tiny, 0]], np.float32)
        x = np.array([[big, big, 0], [tiny, tiny, 0]], np.float32)
        with np.errstate(invalid="raise", divide="raise", over="raise"):
            result = quantize(w, x)
        self.assertTrue(np.isfinite(result.codes*result.scales).all())
        self.assertTrue((result.scales > 0).all() and result.codes.min() >= -127)
        for bad in (w.astype(float), w.ravel(), w*np.nan):
            with self.assertRaises(ValueError): quantize(bad, x)
        with self.assertRaises(ValueError): quantize(w, x[:, :-1])

if __name__ == "__main__": unittest.main()
