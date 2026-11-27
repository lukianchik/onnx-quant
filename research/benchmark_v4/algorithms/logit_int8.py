"""Послойный выбор вариантов INT8 по откликам конечных логитов."""

import numpy as np


def _validate(bank):
    if not bank:
        raise ValueError("Нужна хотя бы одна матрица.")
    for name, variants in bank.items():
        if not variants:
            raise ValueError(f"{name}: отсутствует базовый вариант.")
        shape = None
        for codes, scales in variants:
            if (codes.dtype != np.int8 or codes.ndim != 2 or 0 in codes.shape
                    or (shape is not None and codes.shape != shape)):
                raise ValueError(f"{name}: нужны матрицы INT8 одной формы.")
            shape = codes.shape
            if (scales.dtype != np.float32 or scales.shape != (shape[1],)
                    or not np.all(np.isfinite(scales) & (scales > 0))
                    or np.any(codes == -128)):
                raise ValueError(f"{name}: неверные коды или масштабы.")
            peak = np.maximum(codes.max(axis=0), -codes.min(axis=0))
            with np.errstate(over="ignore"):
                if not np.isfinite(peak * scales).all():
                    raise ValueError(f"{name}: переполнение при деквантизации.")


def _logits(infer, weights, data, shape=None):
    raw = np.asarray(infer(weights, data))
    if (raw.ndim != 2 or raw.shape[0] == 0 or raw.shape[1] < 2
            or raw.dtype.kind != "f" or (shape is not None and raw.shape != shape)):
        raise ValueError("Ожидаются логиты формы (примеры, классы) в едином порядке.")
    z = raw.astype(np.float64, copy=True)
    if not np.isfinite(z).all() or np.max(np.abs(z)) > float(np.finfo(np.float32).max):
        raise ValueError("Логиты должны быть конечными и в диапазоне FP32.")
    return z


class _Target:
    def __init__(self, reference):
        self.reference = reference
        self.labels = reference.argmax(axis=1)
        self.rows = np.arange(len(reference))
        self.rivals = np.arange(reference.shape[1]) != self.labels[:, None]
        self.floor = 0.5 * self.margin(reference)

    def margin(self, z):
        rival = np.max(z, axis=1, where=self.rivals, initial=-np.inf)
        return z[self.rows, self.labels] - rival

    def loss(self, z):
        mse = np.mean(np.square(z - self.reference))
        deficit = np.maximum(self.floor - self.margin(z), 0.0)
        return mse + np.mean(np.square(deficit)), mse


def select_int8(bank, infer, calibration, development, *, width=4):
    """Возвращает словарь индексов выбранных вариантов по именам матриц.

    bank задаёт порядок обхода. Для каждого имени нужна последовательность
    пар (коды INT8 [входы, выходы], масштабы FP32 [выходы]). Варианты
    строятся на калибровке. Нулевые индексы задают исходную INT8-модель.
    infer(None, data) запускает FP32, infer(полный_словарь_пар, data) —
    модель с указанными весами. Каждый вызов независим и возвращает логиты
    всей выборки в одном порядке. Выборки должны быть непересекающимися
    и пригодными для повторного прохода. При равных логитах побеждает
    первый класс. width ограничивает число частичных решений в поиске.
    """
    if (isinstance(width, (bool, np.bool_)) or not isinstance(width, (int, np.integer))
            or width < 1):
        raise ValueError("Ширина поиска должна быть целым положительным числом.")
    _validate(bank)
    base = {name: variants[0] for name, variants in bank.items()}
    zeros = dict.fromkeys(bank, 0)
    fit = _Target(_logits(infer, None, calibration))
    anchor = _logits(infer, base, calibration, fit.reference.shape)
    base_loss = fit.loss(anchor)[0]
    beam = [((), anchor)]

    for name, variants in bank.items():
        shifts = [None]
        for variant in variants[1:]:
            model = {**base, name: variant}
            shifts.append(_logits(infer, model, calibration, anchor.shape) - anchor)

        ranked = []
        for parent, (_, z) in enumerate(beam):
            for k, shift in enumerate(shifts):
                trial = z if k == 0 else z + shift
                ranked.append((fit.loss(trial)[0], parent, k))
        ranked.sort()
        survivors = []
        for _, parent, k in ranked[:width]:
            path, z = beam[parent]
            survivors.append((path + (k,), z if k == 0 else z + shifts[k]))
        beam = survivors

    path, estimate = beam[0]
    if not any(path) or fit.loss(estimate)[0] >= base_loss:
        return zeros
    indices = dict(zip(bank, path))
    proposal = {name: bank[name][k] for name, k in indices.items()}
    dev = _Target(_logits(infer, None, development))
    if dev.reference.shape[1] != anchor.shape[1]:
        raise ValueError("Наборы классов на калибровке и проверке должны совпадать.")
    old = _logits(infer, base, development, dev.reference.shape)
    new = _logits(infer, proposal, development, dev.reference.shape)
    old_loss, old_mse = dev.loss(old)
    new_loss, new_mse = dev.loss(new)
    old_ok = old.argmax(axis=1) == dev.labels
    new_ok = new.argmax(axis=1) == dev.labels
    if new_loss > old_loss or new_mse > old_mse or np.any(old_ok & ~new_ok):
        return zeros
    if new_loss < old_loss or new_mse < old_mse or np.any(new_ok & ~old_ok):
        return indices
    return zeros
