"""Спектральное зависимое округление весовых матриц в симметричный INT8."""

from dataclasses import dataclass
import numpy as np

from quantlab.weight_only.minmax import minmax


@dataclass(frozen=True)
class Config:
    """Предел ширины блока, расчётной работы и дополнительной памяти в байтах."""
    block_cap: int = 12
    work_limit: int = 150_000_000
    memory_limit: int = 64 << 20


@dataclass
class Result:
    """Коды, масштабы, факт принятия хотя бы одного канала и диагностика."""
    codes: np.ndarray
    scales: np.ndarray
    applied: bool
    diagnostics: dict


def _matrix(a):
    """Проверить форму и тип матрицы."""
    if not isinstance(a, np.ndarray) or a.ndim != 2 or a.dtype != np.float32:
        raise ValueError("Нужна двумерная матрица NumPy в формате FP32.")
    return a


def _finite(a):
    """Проверить конечность элементов порциями."""
    chunks = np.nditer(a, flags=["external_loop", "buffered", "zerosize_ok"], buffersize=4096)
    return all(np.isfinite(v).all() for v in chunks)


def _plan(n, d, m, cfg):
    """Подобрать допустимую ширину блока в пределах бюджета."""
    for b in range(min(cfg.block_cap, d), 1, -1):
        work = 6*n*d*m + 2*n*d*b + d*m*(b**3 + 12*b)
        memory = 8*(4*n*b + 8*n + 32*d + 4*d*b + 24*b*b) + 131072
        if work <= cfg.work_limit and memory <= cfg.memory_limit:
            return b, work, memory
    return None


def _geometry(x, b):
    """Построить блочную геометрию активаций."""
    blocks = []
    norms = np.zeros(x.shape[1])
    for i in range(0, len(norms), b):
        sl = slice(i, min(i+b, len(norms)))
        part = x[:, sl].astype(float)
        length = np.linalg.norm(part, axis=0)
        norms[sl] = length
        part /= np.where(length > 0, length, 1)
        gram = part.T @ part
        np.fill_diagonal(gram, 1.)
        spectrum = np.linalg.eigvalsh(gram)
        tol = 64 * np.finfo(float).eps * max(x.shape[0], len(length))
        blocks.append((sl, gram, int(np.sum(spectrum > 1+tol)), tol))
    return blocks, norms


def _matvec(x, v, blocks):
    """Вычислить матрично-векторное произведение по блокам."""
    out = np.zeros(x.shape[0])
    for sl, _, _, _ in blocks:
        out += x[:, sl].astype(float) @ v[sl]
    return out


def _endpoint(v, lo, hi, u):
    """Найти ближайшую границу допустимого шага."""
    edge = np.where(u > 0, hi, lo)
    steps = np.full(v.size, np.inf)
    np.divide(edge-v, u, where=u != 0, out=steps)
    hit = int(np.argmin(steps))
    result = np.clip(v + steps[hit]*u, lo, hi)
    result[hit] = edge[hit]
    return result


def _round_block(start, lo, hi, gram, grad, coupled, tol):
    """Выполнить согласованное округление блока."""
    v = start.copy()
    active = (lo < v) & (v < hi)
    while active.any():
        f = np.flatnonzero(active)
        c = gram[np.ix_(f, f)]
        u = np.zeros(len(f))
        u[0] = 1.
        if coupled and len(f) > 1:
            vals, basis = np.linalg.eigh(c)
            if vals[0] < 1-tol:
                u = basis[:, 0].copy()
                u[np.abs(u) < 16*np.finfo(float).eps] = 0.
                if u[np.argmax(np.abs(u))] < 0:
                    u *= -1
                if u @ (c @ u-u) > 0:
                    u.fill(0.)
                    u[0] = 1.
        vf, low, high = v[f], lo[f], hi[f]
        lin = 2*grad[f] + (high-vf) + (low-vf)
        endpoints = [_endpoint(vf, low, high, direction) for direction in (u, -u)]
        costs = []
        for point in endpoints:
            step = point-vf
            costs.append(lin @ step + step @ (c @ step-step))
        new = endpoints[int(costs[1] < costs[0])]
        grad += gram[:, f] @ (new-vf)
        v[f] = new
        active[f] = (low < new) & (new < high)
    return v


def _column(x, w, base, s, blocks, norms):
    """Проверить спектральный кандидат для одного канала."""
    low_code = np.floor(w / float(s)).clip(-127, 127).astype(np.int16)
    high_code = np.minimum(low_code+1, 127)
    lo = np.multiply(low_code, s, dtype=np.float32).astype(float)
    hi = np.multiply(high_code, s, dtype=np.float32).astype(float)
    center = np.clip(w, lo, hi)
    residual = _matvec(x, center-w, blocks)
    candidate = base.copy()
    for sl, gram, coupled, tol in blocks:
        length = norms[sl]
        part = x[:, sl].astype(float)
        grad = (part.T @ residual) / np.where(length > 0, length, 1)
        low = (lo[sl]-w[sl])*length
        high = (hi[sl]-w[sl])*length
        start = np.clip(np.zeros_like(low), low, high)
        end = _round_block(start, low, high, gram, grad, coupled, tol)
        candidate[sl] = np.where(length == 0, base[sl],
                                 np.where(end == high, high_code[sl], low_code[sl]))
        decoded = np.multiply(candidate[sl], s, dtype=np.float32)
        residual += part @ (decoded.astype(float)-center[sl])
    def loss(codes):
        """Оценить ошибку выхода канала."""
        err = _matvec(x, (codes*s).astype(float)-w, blocks)
        return float(err @ err)
    return candidate, loss(base), loss(candidate)


def quantize(weights, activations, config=None):
    """Квантует W[вход, выход] по X[наблюдение, вход]. None включает MinMax.

    Требуются массивы FP32. Неконечные веса и несогласованные формы ошибочны.
    Пустая или неконечная калибровка даёт MinMax. Бюджеты относятся к ветви
    спектрального поиска. Входы, обязательный результат и память BLAS отделены.
    """
    cfg = Config() if config is None else config
    if not isinstance(cfg, Config) or any(type(v) is not int or v < 0 for v in vars(cfg).values()):
        raise ValueError("Пределы ресурсов должны быть целыми неотрицательными числами.")
    w = _matrix(weights)
    q, s = minmax(w)
    info = {"reason": "no_calibration", "accepted": 0}
    result = Result(q, s, False, info)
    if activations is None:
        return result
    x = _matrix(activations)
    if x.shape[1] != w.shape[0]:
        raise ValueError("Число признаков не совпадает.")
    if not x.size or not _finite(x):
        info["reason"] = "invalid_calibration"
        return result
    plan = _plan(len(x), *w.shape, cfg)
    if plan is None:
        info["reason"] = "budget_or_width"
        return result
    b, work, memory = plan
    info.update(block=b, work_estimate=work, workspace_estimate=memory)
    try:
        blocks, norms = _geometry(x, b)
        info["protected_ranks"] = [r for _, _, r, _ in blocks]
        if not any(info["protected_ranks"]):
            info["reason"] = "flat_blocks"
            return result
        for j in range(w.shape[1]):
            v, old, new = _column(x, w[:, j].astype(float), q[:, j], s[j], blocks, norms)
            if np.isfinite(new) and new < old*(1-64*np.finfo(float).eps):
                q[:, j] = v
                info["accepted"] += 1
    except np.linalg.LinAlgError:
        result.codes, result.scales = minmax(w)
        info.update(reason="factorization_failed", accepted=0)
        return result
    result.applied = info["accepted"] > 0
    info["reason"] = "accepted" if result.applied else "no_gain"
    return result
