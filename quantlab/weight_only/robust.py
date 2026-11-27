"""Робастный критерий v4 без prior и экспериментального replay."""
from dataclasses import dataclass
import numpy as np

BASE = "minmax"

@dataclass(frozen=True)
class Loss:
    """Средние квадраты на fit/dev и энергия ошибки весов на выходной канал."""
    fit: float
    dev: float
    energy: float


def choose(losses, radius=None, confidence=None):
    """Выбирает по верхней границе приращения потерь относительно MinMax."""
    b = losses[BASE]
    if any(not np.isfinite(v) or v < 0 for l in losses.values() for v in vars(l).values()):
        raise ValueError("Потери и энергии должны быть конечными и неотрицательными.")
    if radius is not None and (not np.isfinite(radius) or radius < 0):
        raise ValueError("Недопустимый радиус.")
    rho = (abs(b.dev-b.fit)/b.energy if b.energy else 0.) if radius is None else radius
    bounds = {BASE: 0.}
    for name, l in losses.items():
        if name == BASE:
            continue
        fit, dev = l.fit-b.fit, l.dev-b.dev
        bounds[name] = dev + abs(dev-fit) + rho*(l.energy+b.energy)
        if confidence is not None:
            upper = confidence.get(name, np.inf)
            bounds[name] = max(bounds[name], upper) if np.isfinite(upper) else np.inf
    name = min(bounds, key=bounds.get)
    return name, dict(radius=rho, upper=bounds, reason="robust_gain" if name != BASE else "baseline")


def _payload(pair, shape):
    """Проверить представление INT8 и допустимость масштабов."""
    q, s = pair
    q, s = np.asarray(q), np.asarray(s)
    if (q.shape != shape or q.dtype != np.int8 or s.dtype != np.float32
            or s.shape != (shape[1],) or np.any(q == -128)
            or not np.all(np.isfinite(s) & (s>0))):
        raise ValueError("Некорректный формат INT8.")
    peak = np.maximum(q.max(axis=0), -q.min(axis=0))
    with np.errstate(over="ignore"):
        if not np.isfinite(peak*s).all():
            raise ValueError("Переполнение деквантизации.")
    return q, s


def _loss(w, pair, fit, dev):
    """Вычислить потери Fit, Dev и энергию ошибки весов."""
    q, s = _payload(pair, w.shape)
    energy = 0.
    sums = [0., 0.]
    for j in range(0, w.shape[1], 32):
        sl = slice(j, j+32)
        with np.errstate(over="raise", invalid="raise"):
            e = (q[:, sl]*s[sl]).astype(float)-w[:, sl]
        energy += np.sum(e*e)
        for k, x in enumerate((fit, dev)):
            for i in range(0, len(x), 128):
                r = x[i:i+128].astype(float) @ e
                sums[k] += np.sum(r*r)
    m = w.shape[1]
    return Loss(sums[0]/(len(fit)*m), sums[1]/(len(dev)*m), energy/m)
