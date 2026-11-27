"""Малые синтетические проверки квантования формы QK."""

import unittest

import numpy as np

from qk_svd_int8 import _balanced, minmax_int8, quantize_qk_head


class HeadTests(unittest.TestCase):
    def test_affine_scores_and_bound(self):
        a = np.array([[1., 3.], [0., 2.], [2., -1.]])
        b = np.array([[1., -1.], [2., 1.], [0., 3.], [1., 0.]])
        left, right = _balanced(a, b)
        x = np.array([[1., -2., 1.], [3., 1., 1.]])
        y = np.array([[0., 1., 2., 1.], [3., 0., -2., 1.]])
        np.testing.assert_allclose((x @ left) @ (y @ right).T,
                                   (x @ a) @ (y @ b).T, rtol=1e-13, atol=1e-13)
        bound = np.linalg.norm(left, axis=0) @ np.linalg.norm(right, axis=0)
        self.assertAlmostEqual(bound, np.linalg.svd(a @ b.T, compute_uv=False).sum())

    def test_shear_improves_operator_and_preserves_layout(self):
        a = np.array([[1, 1024], [0, 2]], np.float32)
        b = np.array([[1, 0], [-1024, 1]], np.float32)
        bias = np.zeros(2, np.float32)
        qa, qb, used = quantize_qk_head(a, b, bias)
        self.assertTrue(used)
        self.assertIsNone(qb[2])
        self.assertEqual(qa[2].dtype, np.float32)
        for weight, (codes, scales, _) in zip((a, b), (qa, qb)):
            self.assertEqual((codes.dtype, scales.dtype), (np.int8, np.float32))
            self.assertEqual((codes.shape, scales.shape), (weight.shape, (2,)))
            self.assertEqual(codes.nbytes + scales.nbytes, weight.size + 4 * weight.shape[1])
        qa0, sa0 = minmax_int8(a)
        qb0, sb0 = minmax_int8(b)
        target = a.astype(float) @ b.T
        old = (qa0 * sa0).astype(float) @ (qb0 * sb0).T
        new = (qa[0] * qa[1]).astype(float) @ (qb[0] * qb[1]).T
        self.assertGreater(np.linalg.norm(old - target), 2)
        self.assertLess(np.linalg.norm(new - target), 1e-5)
        again = quantize_qk_head(a, b, bias)
        for before, after in zip((qa, qb), again[:2]):
            for value, repeated in zip(before, after):
                np.testing.assert_array_equal(value, repeated)

    def test_zero_and_exact_minmax(self):
        for a in (np.zeros((2, 4), np.float32), np.eye(2, 4, dtype=np.float32)):
            b = np.eye(3, 4, dtype=np.float32)
            qa, qb, used = quantize_qk_head(a, b)
            self.assertFalse(used)
            for w, (q, s, bias) in zip((a, b), (qa, qb)):
                np.testing.assert_array_equal(q * s, w)
                self.assertIsNone(bias)
                self.assertTrue(np.all(s > 0))

    def test_bias_is_packed_in_original_slots(self):
        a = np.array([[1, 1024], [0, 2]], np.float32)
        b = np.array([[1, 0], [-1024, 1]], np.float32)
        qa, qb, used = quantize_qk_head(a, b, a[1], b[0])
        self.assertTrue(used)
        x = np.array([[1., 2.], [-2., 1.]])
        expected = (x @ a + a[1]) @ (x @ b + b[0]).T
        actual = (x @ (qa[0] * qa[1]) + qa[2]) @ (x @ (qb[0] * qb[1]) + qb[2]).T
        np.testing.assert_allclose(actual, expected, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
