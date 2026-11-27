"""Выбор алгоритма для одного формата симметричных весов INT8."""

from dataclasses import dataclass
from itertools import combinations
import numpy as np

METHODS = ("single", "paired", "sdr")
BASE = "minmax"


def features(d, n, p, stable, corr, peak, rms):
    """Общие для архива и исполнения безразмерные признаки до построения кандидатов."""
    if min(d, n, p) <= 0 or min(stable, corr, peak, rms) < 0:
        raise ValueError("Некорректные размеры или характеристики.")
    f = np.array([np.log1p(d/n), stable/p, corr, np.log1p(peak/rms if rms else 0.)])
    if not np.isfinite(f).all():
        raise ValueError("Неконечный профиль.")
    return f


def _matrix(a):
    a = np.asarray(a)
    if a.dtype != np.float32 or a.ndim != 2 or not a.size:
        raise ValueError("Нужна непустая двумерная матрица FP32.")
    for part in np.nditer(a, flags=["buffered", "external_loop"], buffersize=8192):
        if not np.isfinite(part).all():
            raise ValueError("Неконечный вход.")
    return a


def profile(w, x):
    """Профиль по калибровке; спектр относится только к выбранным координатам."""
    w, x = _matrix(w), _matrix(x)
    d = w.shape[0]
    if x.shape[1] != d:
        raise ValueError("Несогласованные размерности.")
    p = min(64, d)
    cols = np.linspace(0, d-1, p, dtype=int)
    gram = np.zeros((p, p))
    for i in range(0, len(x), 512):
        part = x[i:i+512, cols].astype(float)
        gram += part.T @ part
    values, vectors = np.linalg.eigh(gram)
    values = values.clip(0)
    total = values.sum()
    stable = total/values[-1] if total else 0.
    norm = np.sqrt(gram.diagonal())
    denom = norm[:, None]*norm
    corr = np.divide(gram, denom, out=np.zeros_like(gram), where=denom>0)
    np.fill_diagonal(corr, 0.)
    correlation = abs(corr).sum()/max(1, p*(p-1))
    energy = 0.
    for part in np.nditer(w, flags=["buffered", "external_loop"], buffersize=8192):
        a = part.astype(float)
        energy += a @ a
    rms = np.sqrt(energy/w.size)
    peak = max(abs(float(w.min())), abs(float(w.max())))
    probs = values[values>0]/total if total else np.array([])
    top = vectors[:, values >= values[-1]*(1-1e-10)] if total else vectors[:, :0]
    coherence = p*np.max(np.sum(top**2, axis=1))/top.shape[1] if total else 0.
    return features(d, len(x), p, stable, correlation, peak, rms), dict(
        sampled=p, stable=stable, entropy_rank=np.exp(-probs @ np.log(probs)) if total else 0.,
        coherence=coherence, gram=gram/len(x))


class Portfolio:
    """История по доменам: одна строка профиля и три выигрыша на уникальный оператор."""

    def __init__(self, domains):
        self.domains = []
        for x, g in domains:
            x, g = np.asarray(x, float), np.asarray(g, float)
            if (x.ndim != 2 or x.shape[1] != 4 or not len(x) or g.shape != (len(x), 3)
                    or not np.isfinite(x).all() or not np.isfinite(g).all()):
                raise ValueError("Неверная форма или значения обучающей истории.")
            self.domains.append((x.copy(), np.clip(g, 0, 1)))
        if self.domains:
            all_x = np.concatenate([x for x, _ in self.domains])
            self.low, self.high = all_x.min(0), all_x.max(0)
            spread = np.diff(np.quantile(all_x, [.25, .75], axis=0), axis=0)[0]
            self.scale = np.where(spread>1e-12, spread, np.maximum(self.high-self.low, 1e-12))

    def plan(self, f, quotes, budget):
        """Резервирует подмножество по сметам без вызовов квантователей."""
        f = np.asarray(f, float)
        if f.shape != (4,) or not np.isfinite(f).all() or not np.isfinite(budget) or budget < 0:
            raise ValueError("Неверный профиль или бюджет.")
        if any(k not in METHODS or not np.isfinite(v) or v < 0 for k,v in quotes.items()):
            raise ValueError("Недопустимая смета.")
        if not self.domains:
            return (), "no_history"
        if np.any((f < self.low-1e-10) | (f > self.high+1e-10)):
            return (), "outside_history"
        neighbors = []
        for x, g in self.domains:
            order = np.argsort(np.sum(((x-f)/self.scale)**2, axis=1), kind="stable")
            neighbors.append(g[order[:int(np.ceil(np.sqrt(len(x))))]])
        best, chosen = (0., 0., 0), ()
        for count in range(1, 4):
            for ids in combinations(range(3), count):
                names = tuple(METHODS[i] for i in ids)
                if any(k not in quotes for k in names):
                    continue
                cost = sum(quotes[k] for k in names)
                if cost > budget:
                    continue
                gain = min(np.max(g[:, ids], axis=1).mean() for g in neighbors)
                score = (float(gain), -cost, -count)
                if score > best:
                    best, chosen = score, names
        return chosen, "covered" if chosen else "budget_or_no_gain"


@dataclass(frozen=True)
class Loss:
    """Средние квадраты на fit/dev и энергия ошибки весов на выходной канал."""
    fit: float
    dev: float
    energy: float


def paired_upper(differences, span, candidates, alpha=.05):
    """Верхняя граница Хёффдинга для средних разностей независимых единиц.

    span — заранее известная ширина диапазона, а не размах данной выборки.
    candidates — число проверяемых альтернатив до просмотра этих данных.
    """
    d = np.asarray(differences, float)
    if (d.ndim != 1 or not d.size or not np.isfinite(d).all() or not 0 < alpha < 1
            or type(candidates) is not int or candidates < 1 or not np.isfinite(span)
            or span < 0 or np.ptp(d) > span):
        raise ValueError("Неверные предпосылки для оценки.")
    return d.mean() + span*np.sqrt(np.log(candidates/alpha)/(2*len(d)))


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


def select(w, fit, dev, baseline, builders, prior, quotes, budget=1., radius=None,
           confidence_fn=None):
    """Строит MinMax и только допущенные варианты. Функции принимают (W, Xfit).

    Возвращает имя, выбранную пару, банк допустимых пар и диагностику.
    Сметы в единицах вызывающей стороны; бюджет покрывает дополнительные построения.
    """
    w = _matrix(w)
    base = _payload(baseline(w, fit), w.shape)
    bank = {BASE: base}
    if fit is None or dev is None:
        return BASE, base, bank, {"reason": "missing_split", "attempted": ()}
    try:
        fit, dev = _matrix(fit), _matrix(dev)
    except ValueError:
        return BASE, base, bank, {"reason": "invalid_split", "attempted": ()}
    if dev.shape[1] != w.shape[0] or fit.shape[1] != w.shape[0]:
        raise ValueError("Несогласованные входы оператора.")
    try:
        vec, diag = profile(w, fit)
    except np.linalg.LinAlgError:
        return BASE, base, bank, {"reason": "profile_failure", "attempted": ()}
    allowed, pre = prior.plan(vec, {k:v for k,v in quotes.items() if k in builders}, budget)
    if not allowed:
        return BASE, base, bank, {"reason": pre, "attempted": (), "profile": diag}
    losses = {BASE: _loss(w, base, fit, dev)}
    failed = {}
    for name in allowed:
        try:
            pair = _payload(builders[name](w, fit), w.shape)
            losses[name] = _loss(w, pair, fit, dev)
            bank[name] = pair
        except (ArithmeticError, TypeError, ValueError, np.linalg.LinAlgError, MemoryError) as err:
            failed[name] = str(err)
    confidence = None if confidence_fn is None else {
        k: confidence_fn(w, base, pair, dev, len(allowed)) for k, pair in bank.items() if k != BASE}
    name, reason = choose(losses, radius, confidence)
    reason.update(profile=diag, pre=pre, attempted=allowed, failed=failed,
                  reserved=sum(quotes[k] for k in allowed), losses=losses)
    return name, bank[name], bank, reason


def search_priority(losses, radius=None):
    """Оценивает ширину интервала относительного выигрыша для очереди проб V2."""
    _, info = choose(losses, radius)
    b = losses[BASE]
    if not b.dev:
        return 0.
    widths = []
    for k, l in losses.items():
        if k != BASE:
            center = l.dev-b.dev
            half = info["upper"][k]-center
            if center-half < 0:
                widths.append(min(1., half/b.dev))
    return max(widths, default=0.)


def global_plan(options, priority, budget, limit=12, probe_price=0.):
    """Отбирает небольшую область для V2 и резервирует окончательные проверки."""
    if (type(budget) is not int or budget < 0 or type(limit) is not int or limit < 0
            or not np.isfinite(probe_price) or probe_price < 0):
        raise ValueError("Нужны целые неотрицательные лимиты.")
    remaining = budget-6
    active = []
    def density(k):
        return priority.get(k, 0.)/max(1, len(set(options[k]))-1)
    for name in sorted(options, key=lambda k: -density(k)):
        probes = len(set(options[name]))-1
        if 0 < probes <= remaining and density(name) > probe_price and len(active)<limit:
            active.append(name)
            remaining -= probes
    extra_cost = 3 + sum(len(set(options[k]))-1 for k in active)
    return tuple(active) if sum(priority[k] for k in active) > probe_price*extra_cost else ()


def confirm_outputs(reference, baseline, proposal):
    """Возвращает вектор MSE по именованным численным выходам фиксированного сравнения."""
    if not reference or set(reference) != set(baseline) or set(reference) != set(proposal):
        raise ValueError("Не совпадают выходы.")
    old, new = [], []
    for k in sorted(reference):
        arrays = [np.asarray(z[k]) for z in (reference, baseline, proposal)]
        if (any(a.dtype.kind not in "fiu" for a in arrays)
                or any(a.shape != arrays[0].shape for a in arrays) or not arrays[0].size):
            raise ValueError("Нужны согласованные численные выходы.")
        if not all(np.isfinite(a).all() for a in arrays):
            raise ValueError("Неконечный выход.")
        big_int = any(a.dtype.kind in "iu" and
                      (int(a.min()) < -(2**53) or int(a.max()) > 2**53) for a in arrays)
        if big_int and any(a.dtype.kind == "f" for a in arrays):
            raise ValueError("Нужен адаптер единиц для смешанных типов с большими целыми.")
        ref, b, p = (a.astype(object if big_int else float) for a in arrays)
        with np.errstate(over="raise", invalid="raise"):
            old.append(np.mean(np.asarray(b-ref, dtype=float)**2))
            new.append(np.mean(np.asarray(p-ref, dtype=float)**2))
    return np.array(old), np.array(new)


def finalize(local, options, priority, infer_fit, infer_check, budget, suggest=None, limit=12,
             probe_price=0.):
    """Проверяет локальную сборку и необязательное предложение внешнего V2.

    Конфигурации — словари имён вариантов. None обозначает FP32. suggest
    принимает локальную сборку, варианты активных операторов и лимитированный
    вызов infer_fit. infer_check использует отдельную фиксированную выборку.
    """
    base = dict.fromkeys(local, BASE)
    if set(options) != set(local) or any(BASE not in options[k] or v not in options[k]
                                         for k,v in local.items()):
        raise ValueError("Несогласованный банк.")
    planned = global_plan(options, priority, budget, limit, probe_price)
    active = planned if suggest else ()
    if budget < 3 or (local == base and not active):
        return base, {"calls": 0, "reason": "baseline"}
    calls = 0
    proposals = [dict(local)] if local != base else []
    if active:
        cap = 2 + sum(len(set(options[k]))-1 for k in active)
        def valid(config):
            return (isinstance(config, dict) and set(config) == set(local)
                    and all(v in options[k] for k,v in config.items())
                    and all(config[k] == local[k] for k in local if k not in active))
        def probe(config):
            nonlocal calls
            if calls >= cap:
                raise ValueError("Исчерпан бюджет V2.")
            if config is not None and not valid(config):
                raise ValueError("Запрошена конфигурация вне активного банка.")
            calls += 1
            return infer_fit(config)
        try:
            candidate = suggest(dict(local), {k: tuple(options[k]) for k in active}, probe)
            if valid(candidate) and candidate != base and candidate not in proposals:
                proposals.append(dict(candidate))
        except (ArithmeticError, TypeError, ValueError, np.linalg.LinAlgError):
            pass
    if not proposals:
        return base, {"calls": calls, "reason": "empty"}
    # Копии нужны, если адаптер переиспользует свои буферы вывода.
    def check(config):
        nonlocal calls
        calls += 1
        return {k: np.array(v, copy=True) for k,v in infer_check(config).items()}
    winner = base
    best = None
    try:
        ref, original = check(None), check(base)
        for candidate in proposals:
            try:
                old, new = confirm_outputs(ref, original, check(candidate))
            except (ArithmeticError, ValueError):
                continue
            if np.all(new <= old) and np.any(new < old):
                score = np.mean(np.divide(new, old, out=np.zeros_like(new), where=old>0))
                if best is None or score < best:
                    best, winner = score, candidate
    except (ArithmeticError, ValueError):
        return base, {"calls": calls, "reason": "invalid_reference"}
    return winner, {"calls": calls, "active": active, "reason": "checked"}
