"""Синтетические проверки выбора INT8-весов по конечным логитам."""

import unittest

import numpy as np

from logit_int8 import select_int8


def variant(codes, scale=1 / 64):
    q = np.array(codes, dtype=np.int8, ndmin=2)
    return q, np.full(q.shape[1], scale, dtype=np.float32)


def decoded(pair):
    return pair[0] * pair[1]


class LogitTests(unittest.TestCase):
    def test_additive_cancellation_and_probe_budget(self):
        bank = {"a": [variant(65), variant(68)], "b": [variant(64), variant(60)]}
        fit = np.array([[1], [2]], dtype=np.float32)
        dev = np.array([[-1], [3]], dtype=np.float32)
        calls = []

        def infer(weights, x):
            calls.append("fit" if x is fit else "dev")
            slope = 2 if weights is None else decoded(weights["a"]) + decoded(weights["b"])
            return np.column_stack((x * slope, np.zeros_like(x)))

        for _ in range(2):
            calls.clear()
            self.assertEqual(select_int8(bank, infer, fit, dev, width=2), {"a": 1, "b": 1})
            self.assertEqual(calls, ["fit"] * 4 + ["dev"] * 3)

    def test_nonlinear_interaction_rolls_back(self):
        bank = {"a": [variant(64), variant(80)], "b": [variant(64), variant(50)]}
        fit = np.array([[1]], dtype=np.float32)
        dev = np.array([[2]], dtype=np.float32)
        checked = []

        def infer(weights, x):
            gain = 66 / 64 if weights is None else decoded(weights["a"]) * decoded(weights["b"])
            if x is dev and weights is not None:
                checked.append(float(gain.item()))
            return np.column_stack((x * gain, np.zeros_like(x)))

        self.assertEqual(select_int8(bank, infer, fit, dev), {"a": 0, "b": 0})
        self.assertEqual(checked, [1.0, (80 / 64) * (50 / 64)])

    def test_development_rejects_new_flip_despite_lower_mse(self):
        bank = {"w": [variant([[8], [32]]), variant([[-1], [64]])]}
        reference = np.array([[0.125], [1]], dtype=np.float32)
        fit = np.array([[0, 1]], dtype=np.float32)
        dev = np.eye(2, dtype=np.float32)

        def infer(weights, x):
            value = x @ (reference if weights is None else decoded(weights["w"]))
            return np.column_stack((value, np.zeros_like(value)))

        z0, z1 = (infer({"w": pair}, dev) for pair in bank["w"])
        target = infer(None, dev)
        self.assertLess(np.mean((z1 - target)**2), np.mean((z0 - target)**2))
        self.assertEqual(select_int8(bank, infer, fit, dev), {"w": 0})

    def test_multiclass_margin_changes_choice(self):
        ref = np.array([[0.5, -0.5, 0]], dtype=np.float32)
        bank = {"w": [variant([64, 0, 32]), variant([16, -32, 12]), variant([44, -20, 12])]}
        fit = np.array([[1]], dtype=np.float32)
        dev = np.array([[2]], dtype=np.float32)

        def infer(weights, x):
            return x @ (ref if weights is None else decoded(weights["w"]))

        first, second = (decoded(pair) for pair in bank["w"][1:])
        self.assertLess(np.mean((first - ref)**2), np.mean((second - ref)**2))
        self.assertEqual(select_int8(bank, infer, fit, dev), {"w": 2})


if __name__ == "__main__":
    unittest.main()
