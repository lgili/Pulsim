"""Design evaluation for optimisers (``add-design-optimization-workflow``).

An optimiser calls ``f(design)`` thousands of times and needs the same
answer for the same input, a finite answer or an explicit failure, and
a cost per call it can budget. :class:`Evaluator` wraps a user
``build(design)`` function into that contract::

    from pulsim.design import Evaluator

    ev = Evaluator(build, period=1e-5, loads=("RL",), output="vout",
                   constraints={"ripple_v": (None, 0.12)})
    rec = ev.evaluate({"L": 100e-6, "C": 47e-6})
    rec.status       # 'ok' | 'refused' | 'failed' | 'infeasible'
    rec.kpis         # efficiency, p_loss, ripple_v, ... (one steady period)

Every KPI is taken on ONE steady period with time-weighted integrals
on the run's own grid (right limits included at events), never a
plain ``.mean()`` of samples. A refusal, a non-convergence or a
violated bound is a typed record, never a default number; any other
exception propagates (it is a bug, not a design outcome).
"""

from __future__ import annotations

import functools
import hashlib
import json
import math
import time
import types
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np

__all__ = [
    "EvaluationRecord",
    "Evaluator",
    "Kpi",
    "SteadySpec",
    "ThermalSpec",
    "SteadyPeriod",
    "evaluate",
    "as_objectives",
    "as_constraints",
    "STATUSES",
]

STATUSES = ("ok", "refused", "failed", "infeasible")

class _Failed(Exception):
    """Internal: a design-specific non-convergence (-> ``failed``)."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


class _Infeasible(Exception):
    """Internal: a thermal runaway (-> ``infeasible``)."""

    def __init__(self, reason: str, violations: Mapping[str, float]):
        super().__init__(reason)
        self.reason = reason
        self.violations = dict(violations)


class FrozenDict(dict):
    """A read-only, picklable ``dict`` (records are immutable)."""

    def _ro(self, *a, **k):
        raise TypeError("EvaluationRecord mappings are read-only")

    __setitem__ = __delitem__ = clear = pop = popitem = _ro
    setdefault = update = __ior__ = _ro

    def __reduce__(self):
        return (FrozenDict, (dict(self),))

    def __hash__(self):  # noqa: D105 - frozen, so hashable
        return hash(tuple(sorted(self.items(), key=lambda kv: kv[0])))


def _fz(m) -> FrozenDict:
    return FrozenDict({k: _fz(v) if isinstance(v, Mapping)
                       else tuple(v) if isinstance(v, list) else v
                       for k, v in m.items()})


@dataclass(frozen=True)
class EvaluationRecord:
    """The outcome of one design (design D1). Equal records compare
    equal in every field except ``wall_s``."""

    design: Mapping[str, Any]
    design_id: str
    status: str
    #: Empty unless status is ``ok`` or ``infeasible``.
    kpis: Mapping[str, float]
    #: device -> {category: W}.
    losses: Mapping[str, Mapping[str, float]]
    #: device -> deg C (thermal loop only).
    t_j: Mapping[str, float]
    #: constraint -> amount >= 0 in the KPI's unit (0 = satisfied).
    violations: Mapping[str, float]
    reason: Optional[str]
    #: ``(engine, feature, device, kind)`` of the UnsupportedError.
    refusal: Optional[tuple]
    counters: Mapping[str, Any]
    provenance: Mapping[str, str]
    wall_s: float = field(default=0.0, compare=False)
    #: kpi -> {design variable: d kpi / d variable}, exact (the
    #: ``gradients=`` of the evaluator); empty otherwise.
    grads: Mapping[str, Mapping[str, float]] = field(
        default_factory=lambda: FrozenDict())

    @property
    def seed(self) -> int:
        """Deterministic seed for a user model that needs an RNG."""
        return int(self.design_id, 16) % 2**32


@dataclass(frozen=True)
class SteadySpec:
    """How the steady period is reached (design D5).

    * ``'pss'``: :func:`pulsim.periodic_steady_state` (Newton shooting
      on the ``dsed`` exact stepper); its refusal is ``refused`` and
      its non-convergence ``failed`` -- no transient fallback.
    * ``'transient'``: periods are simulated until the state at the
      period start settles (``tol``) or ``max_periods`` is reached
      (``failed``, ``'not_settled'``).
    * ``'auto'``: ``'pss'`` when the circuit is within its coverage
      (the capability check), otherwise ``'transient'``; the route is
      in ``counters['steady_method']``.
    """

    method: str = "auto"
    #: Settle tolerance of ``'transient'``: the extrapolated distance
    #: to the orbit, ``|dx_n| / (1 - rho)``, relative to each state's
    #: magnitude over the last period.
    tol: float = 1e-6
    max_periods: int = 4096
    #: Periods of the first transient run (doubled until settled).
    min_periods: int = 16
    max_period_runs: int = 60


@dataclass(frozen=True)
class ThermalSpec:
    """Junction temperature by a damped fixed point (design D4).

    ``r_th`` maps a device name to its junction-to-ambient thermal
    resistance (K/W). ``loss_at`` (optional) is a callable
    ``(device, P_sim, T_j) -> P`` giving a device's loss at a junction
    temperature from the loss the simulation produced; default: the
    simulated loss (no temperature dependence). ``temperature_dependent``
    tells the evaluator that ``build`` reads ``design['_T_j']`` (a
    mapping device -> deg C) so the circuit must be re-simulated at
    each iterate."""

    r_th: Mapping[str, float]
    t_amb: float = 25.0
    t_limit: float = 150.0
    tol: float = 0.1
    max_iter: int = 20
    loss_at: Optional[Callable[[str, float, float], float]] = None
    temperature_dependent: bool = False


@dataclass(frozen=True)
class Kpi:
    """A user KPI: ``fn(period: SteadyPeriod) -> float``."""

    name: str
    fn: Callable[["SteadyPeriod"], float]


@dataclass(frozen=True)
class SteadyPeriod:
    """What a user KPI sees: one steady period of the design.

    ``result`` is the ``simulate`` result holding the period
    ``[t_from, t_to]``; on the kernel KPI path it is simulated on first
    access."""

    _result: Any
    builder: Any
    design: Mapping[str, Any]
    t_from: float
    t_to: float
    seed: int
    #: The default KPIs and the per-device losses (read-only).
    kpis: Mapping[str, float] = field(default_factory=dict)
    losses: Mapping[str, Mapping[str, float]] = field(default_factory=dict)

    @property
    def result(self):
        if isinstance(self._result, _Deferred):
            object.__setattr__(self, "_result", self._result.make())
        return self._result

    def average(self, y) -> float:
        """Time-weighted mean of a per-sample trace over the period."""
        from .stats import time_average
        return time_average(self.result.times, y, self.t_from, self.t_to)

    def rms(self, y) -> float:
        from .stats import time_rms
        return time_rms(self.result.times, y, self.t_from, self.t_to)


# ----------------------------------------------------------------------
# hashing and provenance
# ----------------------------------------------------------------------

def _code_digest(code) -> str:
    """Bytecode, names and constants, nested code objects recursively
    (their repr carries a memory address) and sets sorted (their order
    follows the per-process string hash)."""
    def const(c):
        if isinstance(c, types.CodeType):
            return _code_digest(c)
        if isinstance(c, frozenset):
            return sorted(repr(x) for x in c)
        if isinstance(c, tuple):
            return [const(x) for x in c]
        return repr(c)
    h = hashlib.sha256(code.co_code)
    h.update(json.dumps([list(code.co_names), list(code.co_varnames),
                         list(code.co_freevars),
                         [const(c) for c in code.co_consts]]).encode())
    return h.hexdigest()


def _global_names(code, out):
    out.update(code.co_names)
    for c in code.co_consts:
        if isinstance(c, types.CodeType):
            _global_names(c, out)
    return out


def _opaque(v):
    # no canonical form: type + repr, so two such values are equal only
    # when their reprs are (an address in the repr makes it unique)
    return {"opaque": f"{type(v).__module__}.{type(v).__qualname__}",
            "repr": repr(v)}


def _canon_value(v, _seen=None):
    if isinstance(v, (bool, str)) or v is None:
        return v
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        return float(v).hex()      # exact, and NaN/inf are representable
    seen = set() if _seen is None else _seen
    if id(v) in seen:
        return {"recursive": f"{type(v).__qualname__}"}
    seen = seen | {id(v)}

    def canon(x):
        return _canon_value(x, seen)

    def canon_or_opaque(x):
        try:
            return canon(x)
        except (TypeError, ValueError):
            return _opaque(x)
    if isinstance(v, Mapping):
        return {str(k): canon(x) for k, x in sorted(v.items())}
    if isinstance(v, (tuple, list)):
        return [canon(x) for x in v]
    if isinstance(v, np.ndarray):
        return canon(v.tolist())
    if isinstance(v, types.ModuleType):
        return {"module": v.__name__}
    if isinstance(v, functools.partial):
        return {"partial": canon(v.func), "args": canon(list(v.args)),
                "keywords": canon(dict(v.keywords))}
    if isinstance(v, types.MethodType):
        return {"method": canon(v.__func__), "self": canon_or_opaque(v.__self__)}
    if isinstance(v, types.FunctionType):
        # name + code + defaults + captured values + the globals it reads:
        # two closures of one factory, or two lambdas reading different
        # globals, are different options
        code = v.__code__
        g = v.__globals__
        glob = {n: canon_or_opaque(g[n])
                for n in sorted(_global_names(code, set())) if n in g}
        return {"fn": f"{v.__module__}.{v.__qualname__}",
                "code": _code_digest(code),
                "defaults": canon(list(v.__defaults__ or ())),
                "kwdefaults": canon(dict(v.__kwdefaults__ or {})),
                "cells": [canon_or_opaque(c.cell_contents)
                          for c in v.__closure__ or ()],
                "globals": glob}
    if isinstance(v, type):
        return {"class": f"{v.__module__}.{v.__qualname__}"}
    if hasattr(v, "__dict__"):
        return {"type": f"{type(v).__module__}.{type(v).__qualname__}",
                "fields": canon(vars(v))}
    if callable(v):
        return f"{getattr(v, '__module__', '?')}.{getattr(v, '__qualname__', type(v).__qualname__)}"
    raise TypeError(
        f"values must be numbers, strings, booleans, arrays, functions or "
        f"mappings / sequences of them, got {type(v).__name__}: {v!r}")


def _digest(obj) -> str:
    s = json.dumps(_canon_value(obj), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(s.encode()).hexdigest()


@functools.lru_cache(maxsize=1)
def _build_hash() -> str:
    from . import _pulsim
    path = getattr(_pulsim, "__file__", None)
    if not path:
        return "unknown"
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


# ----------------------------------------------------------------------
# traces
# ----------------------------------------------------------------------

def _with_right_limits(res, y_of_states):
    """``(t, y)`` with each event's right limit inserted after its left
    limit (repeated instant), so a jump is integrated as a jump."""
    from ._result_views import states_as_array
    t = np.asarray(res.times, dtype=float)
    y = np.asarray(y_of_states(states_as_array(res)), dtype=float)
    idx = getattr(res, "right_limit_index", None)
    rl = getattr(res, "right_limit_states", None)
    if idx is None or rl is None or len(idx) == 0:
        return t, y
    idx = np.asarray(idx, dtype=np.int64)
    yr = np.asarray(y_of_states(np.asarray(rl, dtype=float)), dtype=float)
    return np.insert(t, idx + 1, t[idx]), np.insert(y, idx + 1, yr)


def _node_col(builder, name):
    k = int(builder.node_id_of(name))
    return (lambda st: st[:, k]) if k >= 0 else (lambda st: 0.0 * st[:, 0])


def _pair_col(nodes):
    a, b = (int(n) for n in nodes)

    def f(st):
        va = st[:, a] if a >= 0 else 0.0
        vb = st[:, b] if b >= 0 else 0.0
        return va - vb + 0.0 * st[:, 0]
    return f


def _window(t, y, a, b):
    from .stats import time_window
    return time_window(t, y, a, b)


def _mean(t, y, a, b):
    from .stats import time_average
    return time_average(t, y, a, b)


def _rms(t, y, a, b):
    from .stats import time_rms
    return time_rms(t, y, a, b)


class _Deferred:
    """A result made on first access (:attr:`SteadyPeriod.result`)."""

    def __init__(self, make):
        self.make = make


def _extreme(tw, yw, k, sign):
    """The extremum near sample ``k``: the vertex of the parabola
    through samples ``k-1, k, k+1`` when ``k`` is a smooth interior
    extremum (distinct instants: no event), else the sample itself.
    The sampled maximum is off by O(h^2) (5e-7 of the ripple at 2000
    samples per period); the vertex by O(h^3)."""
    y0 = yw[k]
    if not (0 < k < len(yw) - 1):
        return y0
    ts, ys = tw[k - 1:k + 2], yw[k - 1:k + 2]
    if not (ts[0] < ts[1] < ts[2]) or sign * (ys[1] - ys[0]) < 0 \
            or sign * (ys[1] - ys[2]) < 0:
        return y0
    d1 = (ys[1] - ys[0]) / (ts[1] - ts[0])
    d2 = (ys[2] - ys[1]) / (ts[2] - ts[1])
    c2 = (d2 - d1) / (ts[2] - ts[0])          # y = y1 + c1 (t-t1) + c2 (t-t1)^2
    if sign * c2 >= 0:
        return y0
    c1 = d1 + c2 * (ts[1] - ts[0])
    tv = -c1 / (2 * c2)
    if not (ts[0] - ts[1] <= tv <= ts[2] - ts[1]):
        return y0
    return y0 + c1 * tv + c2 * tv * tv


def _p2p(t, y, a, b):
    """Peak-to-peak over ``[a, b]``, extrema refined (:func:`_extreme`)."""
    tw, yw = _window(t, y, a, b)
    hi = _extreme(tw, yw, int(np.argmax(yw)), +1)
    lo = _extreme(tw, yw, int(np.argmin(yw)), -1)
    return float(hi - lo)


# ----------------------------------------------------------------------
# the evaluator
# ----------------------------------------------------------------------

_SEMIS = ("switch", "diode", "shockley_diode")


class Evaluator:
    """Evaluate designs into :class:`EvaluationRecord` (design D1).

    Parameters
    ----------
    build
        ``build(design) -> builder`` or ``(builder, switch_fn)``. Called
        once per evaluation (and per thermal iterate when
        ``thermal.temperature_dependent``); it must not share mutable
        state between calls.
    period
        The switching period ``T`` (s), or ``period(design) -> T``.
    loads
        Names of the devices whose dissipation is the output power.
    sources
        Names of the DC voltage sources that deliver the input power
        (default: every DC voltage source).
    output
        Node whose peak-to-peak (``ripple_v``) and mean (``v_out``) are
        KPIs.
    kpis
        Extra :class:`Kpi` objects.
    constraints
        ``{kpi: (lo, hi)}``, either bound ``None``.
    thermal
        A :class:`ThermalSpec`, or ``None``.
    steady
        A :class:`SteadySpec`.
    engine
        ``'auto'`` (default), ``'exact'`` (alias ``'dsed'``), ``'trbdf2'`` or ``'pwl'``:
        the engine of the transient route (``'pss'`` is the ``dsed``
        exact stepper).
    samples_per_period
        Stored samples per period of the KPI run (the ``dsed`` step
        cap is ``T / samples_per_period``).
    options
        Extra ``simulate`` keywords of the transient route (``dt`` for
        ``engine='pwl'``, tolerances).
    loss_options
        Keywords of :func:`pulsim.device_loss_summary`
        (``switch_specs``, ``diode_specs``, ``core_loss_specs``,
        ``conduction_specs``).
    gradients
        ``{design variable: '<component>.<field>'}`` (or a list of names
        that are both): exact gradients of ``v_out``, ``ripple_v`` and
        ``ripple_i.<L>`` on the ``pss`` route (the orbit sensitivities of
        :func:`pulsim.periodic_steady_state`), in ``record.grads``; those
        KPI values are then the closed forms the gradients belong to.
        The circuit value must equal the design value (a scaled mapping
        would give a scaled gradient: refused). The transient route and
        the thermal loop are refused (``UnsupportedError``, feature
        ``'gradients'``); no gradient is ever a finite difference.
    """

    def __init__(self, build: Callable, *, period,
                 loads: Sequence[str] = (),
                 sources: Optional[Sequence[str]] = None,
                 output: Optional[str] = None,
                 kpis: Sequence[Kpi] = (),
                 constraints: Optional[Mapping[str, tuple]] = None,
                 thermal: Optional[ThermalSpec] = None,
                 steady: SteadySpec = SteadySpec(),
                 engine: str = "auto",
                 samples_per_period: int = 2000,
                 options: Optional[Mapping[str, Any]] = None,
                 loss_options: Optional[Mapping[str, Any]] = None,
                 gradients: Any = None):
        if steady.method not in ("pss", "transient", "auto"):
            raise ValueError(f"SteadySpec.method must be 'pss', 'transient' "
                             f"or 'auto', got {steady.method!r}")
        if engine == "exact":    # the canonical name; 'dsed' its alias
            engine = "dsed"
        if engine not in ("auto", "dsed", "trbdf2", "pwl"):
            raise ValueError(f"engine must be 'auto', 'exact' (alias 'dsed'), "
                             f"'trbdf2' or 'pwl', got {engine!r}")
        if steady.method == "pss" and engine not in ("auto", "dsed"):
            raise ValueError(
                f"SteadySpec(method='pss') runs on the dsed exact stepper; "
                f"engine={engine!r} contradicts it (use method='transient').")
        if steady.min_periods < 8 or steady.max_periods < steady.min_periods:
            raise ValueError("SteadySpec needs 8 <= min_periods <= max_periods "
                             "(the settle estimate reads x(n/2) ... x(n))")
        if not (isinstance(samples_per_period, int) and samples_per_period >= 4):
            raise ValueError("samples_per_period must be an int >= 4")
        self.build = build
        self.period = period
        self.loads = tuple(loads)
        self.sources = None if sources is None else tuple(sources)
        self.output = output
        self.kpis = tuple(kpis)
        self.constraints = {k: tuple(v) for k, v in (constraints or {}).items()}
        for k, v in self.constraints.items():
            if len(v) != 2:
                raise ValueError(f"constraint {k!r}: give (lo, hi), got {v!r}")
        self.thermal = thermal
        self.steady = steady
        self.engine = engine
        self.samples_per_period = samples_per_period
        self.options = dict(options or {})
        self.loss_options = dict(loss_options or {})
        if gradients is not None and not isinstance(gradients, Mapping):
            gradients = {str(g): str(g) for g in gradients}
        self.gradients = dict(gradients) if gradients else None
        if self.gradients and thermal is not None:
            from ._refusal import unsupported
            raise unsupported(
                "design.evaluate", "gradients", "Evaluator: gradients= with "
                "a thermal loop: the junction temperature depends on the "
                "parameters through the loss model, which the exact path "
                "does not differentiate (refused, never approximated).")
        self._options_digest = _digest(self._config())

    # -- identity --------------------------------------------------------
    def _config(self):
        th = None
        if self.thermal is not None:
            th = asdict(self.thermal)
        return dict(
            build=self.build, period=self.period, loads=self.loads,
            sources=self.sources, output=self.output,
            kpis=[(k.name, k.fn) for k in self.kpis],
            constraints=self.constraints, thermal=th,
            steady=asdict(self.steady), engine=self.engine,
            samples=self.samples_per_period, options=self.options,
            loss_options=self.loss_options, gradients=self.gradients)

    def design_id(self, design: Mapping[str, Any]) -> str:
        return _digest({"design": dict(design),
                        "options": self._options_digest})

    def _provenance(self):
        from . import __version__
        return {"pulsim": str(__version__), "build": _build_hash(),
                "options": self._options_digest}

    # -- public ----------------------------------------------------------
    def evaluate(self, design: Mapping[str, Any]) -> EvaluationRecord:
        """One design -> one record (design D3 status mapping)."""
        from ._refusal import UnsupportedError
        t0 = time.perf_counter()
        design = dict(design)
        did = self.design_id(design)
        counters: dict = {"simulations": 0}
        base = dict(design=_fz(design), design_id=did,
                    provenance=_fz(self._provenance()))

        grads: dict = {}

        def rec(status, kpis=(), losses=(), t_j=(), violations=(),
                reason=None, refusal=None):
            return EvaluationRecord(
                status=status, kpis=_fz(dict(kpis)), losses=_fz(dict(losses)),
                t_j=_fz(dict(t_j)), violations=_fz(dict(violations)),
                reason=reason, refusal=refusal, counters=_fz(counters),
                wall_s=time.perf_counter() - t0,
                grads=_fz({k: _fz(v) for k, v in grads.items()}), **base)

        try:
            kpis, losses, t_j = self._evaluate(design, did, counters, grads)
        except UnsupportedError as e:
            return rec("refused", reason=str(e), refusal=(
                str(e.engine), str(e.feature), str(e.device), str(e.kind)))
        except _Failed as e:
            return rec("failed", reason=e.reason)
        except _Infeasible as e:
            return rec("infeasible", violations=e.violations, reason=e.reason)

        bad = sorted(k for k, v in kpis.items() if not math.isfinite(v))
        if bad:
            return rec("failed", reason=f"non-finite KPI: {', '.join(bad)}")
        violations = {}
        for name, (lo, hi) in self.constraints.items():
            if name not in kpis:
                raise KeyError(
                    f"constraint {name!r} names no KPI; KPIs: {sorted(kpis)}")
            v = kpis[name]
            violations[name] = max(0.0, (lo - v) if lo is not None else 0.0,
                                   (v - hi) if hi is not None else 0.0)
        status = "infeasible" if any(v > 0 for v in violations.values()) \
            else "ok"
        return rec(status, kpis, losses, t_j, violations)

    __call__ = evaluate

    def evaluate_many(self, designs: Sequence[Mapping[str, Any]], *,
                      workers: Optional[int] = None,
                      backend: str = "thread") -> list:
        """Records in input order (design D6), on the job runner of
        :func:`pulsim.sweep` (``'thread'``: the kernel releases the
        GIL; ``'process'`` for Python-heavy designs -- ``build`` and
        the KPIs must then pickle). Design-specific outcomes never stop
        the batch; it stops with the refusal when its first three
        designs are ``refused`` with identical fields (the
        configuration, not the design, is unsupported). Any other
        exception propagates."""
        from ._refusal import unsupported
        from .sweep import _run_jobs
        jobs = [(self, dict(d)) for d in designs]
        head = jobs[:3]
        out, _, _ = _run_jobs(head, parallel=False, backend="thread",
                              n_workers=1, verbose=False,
                              job=_evaluate_job, reraise=True)
        refs = {r.refusal for r in out}
        # 'consistent_mode' is found mid-run, at a state that depends on
        # the design: it never condemns the configuration
        if len(out) == 3 and all(r.status == "refused" for r in out) \
                and len(refs) == 1 and out[0].refusal[1] != "consistent_mode":
            e, f, d, k = out[0].refusal
            raise unsupported(
                e, f, "evaluate_many: the first three designs were refused "
                "with identical fields, so the configuration is "
                f"unsupported: {out[0].reason}", device=d, kind=k)
        rest = jobs[3:]
        if rest:
            more, _, _ = _run_jobs(
                rest, parallel=workers != 1, backend=backend,
                n_workers=workers, verbose=False, job=_evaluate_job,
                reraise=True)
            out += more
        return out

    # -- one evaluation --------------------------------------------------
    def _period_of(self, design):
        T = self.period(design) if callable(self.period) else self.period
        T = float(T)
        if not (T > 0 and math.isfinite(T)):
            raise ValueError(f"period must be positive and finite, got {T!r}")
        return T

    def _build(self, design):
        out = self.build(design)
        if isinstance(out, tuple):
            if len(out) != 2:
                raise ValueError("build(design) must return a builder or "
                                 "(builder, switch_fn)")
            return out
        return out, None

    def _evaluate(self, design, did, counters, grads=None):
        T = self._period_of(design)
        b, sf = self._build(design)
        run, a, z = self._steady(b, sf, T, counters)
        th = self.thermal
        if th is None:
            rows, losses = self._loss_rows(b, run, a, z)
            kpis = self._kpis(b, run, a, z, design, did, rows, losses,
                              counters)
            if self.gradients:
                self._gradients(b, run, design, kpis, grads, counters)
            return kpis, losses, {}
        names = {c["name"] for c in b.components()}
        for d in th.r_th:
            if d not in names:
                raise KeyError(f"ThermalSpec.r_th names {d!r}, which is not "
                               f"a device of the circuit")
        tj = {d: float(th.t_amb) for d in th.r_th}
        omega, prev, dist = 1.0, math.inf, math.inf
        for k in range(1, th.max_iter + 1):
            rows, losses = self._loss_rows(b, run, a, z, tj)
            losses = self._losses_at(losses, tj)
            new = {}
            for d, r in th.r_th.items():
                t_hat = th.t_amb + float(r) * sum(losses[d].values())
                new[d] = tj[d] + omega * (t_hat - tj[d])
            inc = max(abs(new[d] - tj[d]) for d in tj)
            if not math.isfinite(inc):
                raise _Failed(f"non-finite thermal loss at T_j = {tj}")
            q = inc / prev if prev > 0 else 0.0
            prev, tj = inc, new
            counters["thermal_iterations"] = k
            counters["thermal_increment"] = inc
            hot = max(tj[d] - th.t_limit for d in tj)
            if q >= 1.0:
                # the increments do not shrink: above the limit, or with
                # damping exhausted, there is no fixed point to reach
                if hot > 0 or omega < 1.0 / 64:
                    raise _Infeasible(
                        f"thermal_runaway: T_j = {max(tj.values()):.4g} C and "
                        f"rising (increment ratio {q:.3g}) at iteration {k}",
                        {"thermal_runaway": max(hot, inc)})
                omega *= 0.5        # the increment grew: damp
            elif k >= 2:
                # distance to the fixed point of a contraction with ratio q
                dist = inc * q / (1.0 - q)
                if dist < th.tol:
                    break
            if th.temperature_dependent:
                # the electrical solution depends on T_j: re-simulate
                b, sf = self._build({**design, "_T_j": _fz(tj)})
                run, a, z = self._steady(b, sf, T, counters)
        else:
            raise _Infeasible(
                f"thermal_runaway: no fixed point within {th.max_iter} "
                f"iterations (last increment {inc:.4g} K, estimated "
                f"distance {dist:.4g} K)", {"thermal_runaway": inc})
        counters["thermal_distance"] = dist
        hot = max(tj[d] - th.t_limit for d in tj)
        if hot > 0:
            raise _Infeasible(
                f"thermal_limit: the fixed point T_j = {max(tj.values()):.4g} C "
                f"exceeds t_limit = {th.t_limit:g} C", {"thermal_limit": hot})
        # the losses AT the converged temperatures (no re-simulation:
        # the run is within th.tol of them)
        rows, losses = self._loss_rows(b, run, a, z, tj)
        losses = self._losses_at(losses, tj)
        kpis = self._kpis(b, run, a, z, design, did, rows, losses, counters)
        for d in tj:
            kpis[f"t_j.{d}"] = tj[d]
        return kpis, losses, tj

    def _losses_at(self, losses, tj):
        """Apply ``ThermalSpec.loss_at`` (device, P, T_j) -> P, scaling
        the device's categories (``conduction`` when it had none)."""
        th = self.thermal
        out = {d: dict(c) for d, c in losses.items()}
        if th.loss_at is None:
            for d in th.r_th:
                out.setdefault(d, {"conduction": 0.0})
            return out
        for d in th.r_th:
            cat = out.get(d) or {"conduction": 0.0}
            p0 = sum(cat.values())
            p1 = float(th.loss_at(d, p0, tj[d]))
            out[d] = ({c: v * p1 / p0 for c, v in cat.items()} if p0 > 0
                      else {"conduction": p1})
        return out

    def _steady(self, b, sf, T, counters):
        """``(result, t_from, t_to)`` of one steady period."""
        from .periodic_steady_state import _pss_refusal
        method = self.steady.method
        if method == "auto":
            if self.engine not in ("auto", "dsed"):
                method = "transient"
                counters["steady_route_reason"] = (
                    f"engine={self.engine!r}: pss runs on dsed only")
            else:
                # the capability probe builds the exact stepper: a kernel
                # error there is a failed design, not a crash (_engine)
                e = _engine(lambda: _pss_refusal(b, T, sf))
                if e is None:
                    method = "pss"
                else:
                    # A documented route, decided BEFORE any run: the
                    # circuit is outside the exact stepper's coverage.
                    method = "transient"
                    counters["steady_route_reason"] = (
                        f"pss refused: {e.feature or e.kind or 'circuit'}"
                        f"{' (' + e.device + ')' if e.device else ''}")
        counters["steady_method"] = method
        if self.gradients and method != "pss":
            from ._refusal import unsupported
            raise unsupported(
                "design.evaluate", "gradients", "Evaluator: gradients= on "
                f"the {method} route ({counters.get('steady_route_reason', '')}"
                "): exact gradients exist on the pss route only (refused, "
                "never a finite difference).")
        if method == "pss":
            return self._pss(b, sf, T, counters)
        return self._transient(b, sf, T, counters)

    def _transient(self, b, sf, T, counters):
        """Simulate ``n`` periods (``n`` doubling from ``min_periods``)
        until the true state at the period starts has settled."""
        from . import simulate
        spec = self.steady
        engine = self.engine
        kw = dict(self.options)
        n = max(2, spec.min_periods)
        while True:
            run = _engine(lambda: simulate(
                b, t_end=n * T, engine=engine, switch_fn=sf, **kw))
            self._count(run, counters, b, sf)
            counters["route"] = f"transient/{getattr(run, 'engine_used', engine)}"
            counters["periods"] = n
            err = _settle_error(b, run, T, n)
            counters["settle_error"] = err
            if err <= spec.tol:
                break
            if n >= spec.max_periods:
                raise _Failed(
                    f"not_settled: after {n} periods the extrapolated "
                    f"distance to the periodic orbit is {err:.3e} of the "
                    f"state scale (tol {spec.tol:g}); raise "
                    "SteadySpec.max_periods or use method='pss'")
            n = min(2 * n, spec.max_periods)
        if getattr(run, "engine_used", engine) != "dsed":
            # a dt grid summed to n*T can stop an ulp short of it
            # (4e-7 x 200 = 7.999999999999999e-05): read the run's end
            t1 = float(np.asarray(run.times)[-1])
            t1 = t1 if abs(t1 - n * T) <= 1e-9 * T else n * T
            return run, t1 - T, t1
        # dsed: the settle run has no step cap; re-run the last period
        # from its true state with the KPI sampling.
        x0 = np.asarray(run.states_reduced, dtype=float)[-1]
        run2 = _engine(lambda: simulate(
            b, t_end=T, dt=T / self.samples_per_period, engine="dsed",
            switch_fn=sf, initial_state=x0,
            **{k: v for k, v in kw.items() if k != "dt"}))
        self._count(run2, counters, b, sf)
        return run2, 0.0, T

    def _pss(self, b, sf, T, counters):
        from . import simulate
        from ._pulsim import ConvergenceError  # type: ignore[import-not-found]
        from ._period_kpis import Period, Spec, spec_for
        from .periodic_steady_state import periodic_steady_state
        counters["route"] = "pss/dsed"
        spec = spec_for(b, self.output, self.loss_options)
        obs = spec if isinstance(spec, Spec) else None

        def solve():
            try:
                return periodic_steady_state(
                    b, period=T, switch_fn=sf,
                    max_period_runs=self.steady.max_period_runs,
                    sensitivities=(sorted(set(self.gradients.values()))
                                   if self.gradients else None),
                    _observables=obs)
            except (ConvergenceError, np.linalg.LinAlgError) as e:
                # typed at the source: a non-convergence of the design (a
                # singular I - M from a degenerate design included). An
                # unextractable state space is UnsupportedError(feature=
                # "state_space") from _Solver and passes as a refusal.
                raise _Failed(str(e)) from e

        r = _engine(solve)
        if not r.stable:
            # an orbit the circuit never reaches: its KPIs are no answer
            raise _Failed(
                "unstable_orbit: Floquet multipliers outside the unit "
                f"circle {[(round(abs(m), 6), k) for _, m, k in r.unstable_multipliers]}")
        counters["pss_period_runs"] = counters.get(
            "pss_period_runs", 0) + int(r.n_period_runs)
        counters["engine_steps"] = counters.get(
            "engine_steps", 0) + int(r.n_engine_steps)
        counters["simulations"] += int(r.n_period_runs)
        ctx = r._ctx
        counters["python_callbacks"] = int(ctx.n_python_callbacks)
        counters["nogil_period_runs"] = int(ctx.n_nogil_runs)
        d = r._period

        def sampled(n):
            run = _engine(lambda: simulate(
                b, t_end=n * T, dt=T / self.samples_per_period,
                engine="dsed", switch_fn=sf, initial_state=r.x0, rtol=1e-11,
                atol=1e-14))
            self._count(run, counters, b)
            return run

        if obs is not None and d is not None and d["integrals_ok"]:
            # the KPIs are the kernel's closed-form period integrals;
            # waveforms only when a user KPI reads them
            counters["kpi_path"] = "kernel"
            kp = Period(obs, d, T)
            _raise_if_suspect_kernel(kp, b)
            # two periods, the second read: as the sampled route below
            kp.lazy_run = _Deferred(lambda: sampled(2))
            kp.pss_orbit = r
            return kp, 0.0, T
        counters["kpi_path"] = "sampled: " + (
            spec if obs is None else
            "no kernel period map" if d is None else d["integrals_reason"])
        # Two periods, the second measured: a switch edge AT the section
        # is then an edge inside the window (a one-period run starts in
        # the new state and its turn-on is never seen: E_on was lost).
        run = sampled(2)
        if self.gradients:
            run.pss_orbit = r
        return run, T, 2 * T

    def _gradients(self, b, run, design, kpis, grads, counters):
        """``grads[kpi][variable]`` from the orbit sensitivities; the
        KPI values are replaced by the closed forms they belong to."""
        from .sensitivity import _prepare
        r = run.pss_orbit
        sr = r.sensitivity
        vals = {p.name: p.value for p in _prepare(
            b, list(sr.params), r._ctx.sf, r.period)}
        base = {c["name"]: dict(c["params"]) for c in b.components()}
        for var, name in self.gradients.items():
            if var not in design:
                raise KeyError(f"gradients: design variable {var!r} is not "
                               f"in the design {sorted(design)}")
            self._check_map(design, var, name, base, r.period)
            dv, pv = float(design[var]), vals[name]
            if not abs(dv - pv) <= 1e-12 * max(abs(dv), abs(pv)):
                raise ValueError(
                    f"gradients: design[{var!r}] = {dv!r} but {name} = "
                    f"{pv!r} in the circuit; the gradient is with respect "
                    "to the circuit value, so a scaled mapping would be "
                    "wrong. Map variables that ARE the circuit value.")
        want = {}
        if self.output is not None:
            want["v_out"] = (self.output, "avg")
            want["ripple_v"] = (self.output, "ripple")
        for c in b.components():
            if c["kind"] == "inductor":
                want[f"ripple_i.{c['name']}"] = (c["name"], "ripple")
        for k, (sig, kind) in want.items():
            kv = sr.kpi(sig, kind)
            kpis[k] = kv.value
            if kv.non_unique:
                # attained at two instants: not differentiable, no gradient
                counters["grads_refused"] = ", ".join(filter(None, (
                    counters.get("grads_refused", ""),
                    f"{k}: non_unique extremum")))
                continue
            g = {var: float(kv.grad[name])
                 for var, name in self.gradients.items()}
            if not all(math.isfinite(v) for v in g.values()):
                counters["grads_refused"] = ", ".join(filter(None, (
                    counters.get("grads_refused", ""),
                    f"{k}: non-finite gradient")))
                continue
            grads[k] = g
        # an event that nearly grazed: the gradient is exact for this
        # event pattern but a nearby design may switch pattern; simulate
        # warns once per call site, so the record carries it
        nm = len(getattr(sr, "flags", None) or ())
        if nm:
            counters["grad_near_misses"] = nm

    def _check_map(self, design, var, name, base, T):
        """``var`` must drive exactly the field ``name``: rebuild with it
        moved and compare every component (a variable feeding two
        inductors would give a partial derivative as the total)."""
        from .sensitivity import _prepare
        dv = float(design[var])
        b2, sf2 = self._build({**design, var: dv * (1 + 1e-6) + 1e-300})
        moved = sorted(
            f"{c['name']}.{k}" for c in b2.components()
            for k, v in c["params"].items()
            if base.get(c["name"], {}).get(k) != v)
        comp, fld = name.rsplit(".", 1)
        if fld == "duty":
            d2 = _prepare(b2, [name], sf2, T)[0].value
            ok = not moved and abs(d2 - dv * (1 + 1e-6)) <= 1e-9
        else:
            ok = len(moved) == 1 and moved[0].startswith(comp + ".")
        if not ok:
            raise ValueError(
                f"gradients: design[{var!r}] moves {moved or 'no element'} "
                f"in the circuit, not {name} alone; the gradient would be "
                "a partial derivative presented as the total. Map each "
                "variable to the one field it sets.")

    @staticmethod
    def _count(run, counters, b, sf=None):
        counters["simulations"] += 1
        _raise_if_suspect(run, b)
        n = getattr(run, "n_accept", None)
        if n is None:
            n = run.num_steps()
        counters["engine_steps"] = counters.get("engine_steps", 0) + int(n)
        # accepted-with-error notes simulate only warns about (not failures)
        nb = len(getattr(run, "event_iteration_breaches", None) or ())
        nc = int((getattr(run, "_trbdf2_stats", None) or {}).get("n_chatter_breaks", 0))
        if nb:
            counters["event_iteration_breaches"] = counters.get("event_iteration_breaches", 0) + nb
        if nc:
            counters["chatter_breaks"] = counters.get("chatter_breaks", 0) + nc
        # accepted above tolerance (h at its floor) / Newton accepted at
        # roundoff: counted, not failed (DECISIONS: design workflow 14)
        nf = int((getattr(run, "_trbdf2_stats", None) or {}).get("n_forced_accepts", 0))
        nr = int(getattr(run, "newton_roundoff_accepts", 0) or 0)
        if nf:
            counters["forced_accepts"] = counters.get("forced_accepts", 0) + nf
        # trbdf2 steps accepted within the error estimate's own roundoff
        # floor (a g_off-held node): honest, counted (DECISIONS 5)
        nfl = int((getattr(run, "_trbdf2_stats", None) or {}).get(
            "n_roundoff_floor_accepts", 0))
        if nfl:
            counters["roundoff_floor_accepts"] = counters.get(
                "roundoff_floor_accepts", 0) + nfl
        if nr:
            counters["newton_roundoff_accepts"] = counters.get(
                "newton_roundoff_accepts", 0) + nr
        # a step re-taken at dt/2^k: integrated MORE finely than asked,
        # so the result is not less accurate -- counted, ok (DECISIONS 5)
        nd = len(getattr(run, "dt_retries", None) or ())
        if nd:
            counters["dt_retries"] = counters.get("dt_retries", 0) + nd
        # dsed polling a schedule without next_edge_after: every edge it
        # sees is bisected to the instant; only a pulse narrower than
        # dt_max/10 can be missed -- counted, ok (DECISIONS 5)
        if sf is not None and getattr(run, "engine_used", None) == "dsed":
            from ._dsed_dispatch import _has_edge_oracle
            if not _has_edge_oracle(sf):
                counters["polled_switch_fn_runs"] = counters.get(
                    "polled_switch_fn_runs", 0) + 1
        from . import _resolution_flagged
        if _resolution_flagged(run):
            # the fixed-step ResolutionWarning: dt may not resolve the run
            counters["resolution_warnings"] = counters.get(
                "resolution_warnings", 0) + 1
        counters["python_callbacks"] = len(
            tuple(getattr(run, "step_loop_python_callbacks", ()) or ()))

    # -- KPIs ------------------------------------------------------------
    def _loss_rows(self, b, run, a, z, tj=None):
        """``(rows, losses)``: :func:`device_loss_summary` rows by name
        and the per-device categories of every non-load device. ``tj``
        (device -> deg C) is the temperature the switches' 3-D loss
        tables are read at (the thermal loop, design D4)."""
        from .losses import device_loss_summary
        opts = dict(self.loss_options)
        if tj:
            ss = {k: dict(v) for k, v in (opts.get("switch_specs") or {}
                                          ).items()}
            for dev, spec in ss.items():
                if dev in tj and ("E_on_table" in spec or "E_off_table" in spec):
                    spec["Tj"] = float(tj[dev])
            if ss:
                opts["switch_specs"] = ss
        from ._period_kpis import Period, loss_rows
        if isinstance(run, Period):
            # the period's edge log is kernel output: an inconsistent one
            # ("edges do not alternate") fails this design, not the batch
            rows = _engine(lambda: loss_rows(b, run, opts))
        else:
            rows = {r["name"]: r for r in device_loss_summary(
                b, run, t_from=a, t_to=z, **opts) if "name" in r}
        for n in self.loads:
            if n not in rows or "P_avg" not in rows[n]:
                raise ValueError(f"load {n!r} has no dissipated power in "
                                 f"device_loss_summary (a resistor?)")
        losses: dict = {}
        for n, r in rows.items():
            if n in self.loads:
                continue
            cat = {}
            if "P_avg" in r:
                # The datasheet conduction model, when given, replaces
                # the resistor-pair's v^2.g (DECISIONS 3).
                cat["resistive" if r["kind"] == "resistor"
                    else "conduction"] = float(
                        r.get("P_cond_model_avg", r["P_avg"]))
            if "P_sw_avg" in r:
                cat["switching" if r["kind"] == "switch"
                    else "recovery"] = float(r["P_sw_avg"])
            if "P_core_avg" in r:
                cat["core"] = float(r["P_core_avg"])
            if cat:
                losses[n] = cat
        return rows, losses

    def _kpis(self, b, run, a, z, design, did, rows, losses, counters):
        from ._period_kpis import Period
        from ._refusal import unsupported
        kp = run if isinstance(run, Period) else None
        comps = {c["name"]: c for c in b.components()}
        kpis: dict = {}
        src = self.sources
        if src is None:
            src = [n for n, c in comps.items() if c["kind"] in (
                "voltage_source", "pwm_voltage_source", "pulse_voltage_source",
                "sine_voltage_source")]
        # A device that can dissipate or deliver power and that neither
        # the loss summary nor the source measurement covers would make
        # the loss total and the efficiency silently wrong (a Lauritzen
        # diode's 1.36 W of 28.4 W was missing): refused, never summed.
        uncovered = [n for n, c in comps.items()
                     if c["kind"] not in ("inductor", "capacitor",
                                          "voltage_source")
                     and "P_avg" not in rows.get(n, {}) and n not in src]
        if uncovered and self.loads:
            c = comps[uncovered[0]]
            raise unsupported(
                "design.evaluate", "losses", "design.evaluate: the power of "
                f"{', '.join(uncovered)} ({c['kind']}...) is in neither "
                "device_loss_summary nor the measured sources, so the loss "
                "total and the efficiency would be incomplete. Evaluate "
                "without `loads=` for the other KPIs.",
                device=uncovered[0], kind=c["kind"])
        p_in = 0.0 if src else None
        for n in src:
            c = comps[n]
            if c["kind"] != "voltage_source":
                raise unsupported(
                    "design.evaluate", "sources", f"design.evaluate: source "
                    f"{n!r} is a {c['kind']}; the input power is measured "
                    "on DC voltage sources only. Name the DC sources in "
                    "`sources=`.", device=n, kind=c["kind"])
            # MNA branch current flows + -> - inside the source.
            if kp is not None:
                p_in -= float(c["params"]["V"]) * kp.mean(("i", n))
                continue
            col = int(b.pool.branch_var_id_for_source(c["branch_id"], b.graph))
            t, i = _with_right_limits(run, lambda st, col=col: st[:, col])
            p_in -= float(c["params"]["V"]) * _mean(t, i, a, z)
        if p_in is not None:
            kpis["p_in"] = p_in
        if uncovered:
            counters["losses_not_covered"] = ", ".join(uncovered)
        else:
            p_loss = sum(sum(c.values()) for c in losses.values())
            kpis["p_loss"] = p_loss
        if self.loads:
            p_out = sum(float(rows[n]["P_avg"]) for n in self.loads)
            kpis["p_out"] = p_out
            # P_out / (P_out + losses): with a datasheet model or a
            # loss table the losses include what the resistor-pair
            # circuit does not dissipate; without them it equals
            # P_out / P_in at the orbit (DECISIONS 4).
            kpis["efficiency"] = (p_out / (p_out + p_loss)
                                  if p_out + p_loss > 0 else math.nan)
        if self.output is not None:
            if kp is not None:
                kpis["v_out"] = kp.mean(("node", self.output))
                kpis["ripple_v"] = kp.p2p(("node", self.output))
            else:
                t, v = _with_right_limits(run, _node_col(b, self.output))
                kpis["v_out"] = _mean(t, v, a, z)
                kpis["ripple_v"] = _p2p(t, v, a, z)
        for n, c in comps.items():
            if c["kind"] == "inductor":
                if kp is not None:
                    key = ("i", n)
                    kpis[f"ripple_i.{n}"] = kp.p2p(key)
                    kpis[f"i_peak.{n}"] = kp.peak_abs(key)
                    kpis[f"i_rms.{n}"] = kp.rms(key)
                    continue
                t = np.asarray(run.times, float)
                i = np.asarray(run.i(n), float)
                kpis[f"ripple_i.{n}"] = _p2p(t, i, a, z)
                tw, iw = _window(t, i, a, z)
                kpis[f"i_peak.{n}"] = float(np.abs(iw).max())
                kpis[f"i_rms.{n}"] = _rms(t, i, a, z)
            elif c["kind"] in _SEMIS and n in rows:
                r = rows[n]
                kpis[f"i_peak.{n}"] = float(r["i_peak"])
                kpis[f"i_rms.{n}"] = float(r["i_rms"])
                if kp is not None:
                    kpis[f"v_peak.{n}"] = kp.peak_abs(("v", n))
                    continue
                t, v = _with_right_limits(run, _pair_col(c["nodes"]))
                _, vw = _window(t, v, a, z)
                kpis[f"v_peak.{n}"] = float(np.abs(vw).max())
        if self.kpis:
            if kp is not None:
                # the waveforms of the orbit, simulated on first access
                run, a, z = kp.lazy_run, kp.T, 2 * kp.T
            per = SteadyPeriod(run, builder=b, design=design,
                               t_from=a, t_to=z, seed=int(did, 16) % 2**32,
                               kpis=_fz(kpis), losses=_fz(losses))
            for k in self.kpis:
                if k.name in kpis:
                    raise ValueError(f"user KPI {k.name!r} has the name of a "
                                     "default KPI")
                kpis[k.name] = float(k.fn(per))
        return kpis


def _true_state(b, run):
    """``(t, X)``: inductor currents, capacitor voltages and every
    column of the engine's state vector (node voltages, branch
    currents, the internal states of stateful devices) per sample.
    A Python ``switch_fn``'s own memory is not in it (DECISIONS 9)."""
    from ._result_views import states_as_array
    st = np.asarray(states_as_array(run), dtype=float)
    cols = []
    for c in b.components():
        if c["kind"] == "inductor":
            cols.append(np.asarray(run.i(c["name"]), dtype=float))
        elif c["kind"] == "capacitor":
            cols.append(_pair_col(c["nodes"])(st))
    t = np.asarray(run.times, dtype=float)
    X = np.stack(cols, axis=1) if cols else np.zeros((t.size, 0))
    return t, np.hstack([X, st.reshape(t.size, -1)])


def _settle_error(b, run, T, n):
    """Estimated distance of the state at ``nT`` to the periodic orbit,
    in units of each state's magnitude over the last period, the
    largest over the states (DECISIONS 9).

    Per state, two geometric extrapolations, ``x(k) = x* + mu^k c``:
    the last period, ``d1 = |x(n) - x(n-1)|``, ``rho = d1 / d0``,
    distance ``d1 rho / (1 - rho)``; and the last half, ``d_a =
    |x(n) - x(3n/4)|``, ``r = d_a / d_b``, distance ``d_a r / (1 - r)``.
    The larger is taken: during a start-up overshoot the quarters
    contract fast (r = 0.19 on the DCM buck at 32 periods) while each
    period still moves 1.1 % of the way (rho = 0.989), and the quarter
    estimate alone read 6e-3 for a state 51 % off. A ratio >= 1 (the
    engine's noise, or a drift) gives the sum of the two differences.
    A small period-to-period change alone is not a small error with a
    slow mode."""
    from .stats import time_average
    t, X = _true_state(b, run)
    if X.shape[1] == 0:
        return 0.0
    ks = (n // 2, (3 * n) // 4, n - 2, n - 1, n)
    if all(np.min(np.abs(t - k * T)) <= 1e-9 * T for k in ks):
        # the engine landed every period start: the state there is exact
        xs = np.array([[np.interp(k * T, t, X[:, j])
                        for j in range(X.shape[1])] for k in ks])
    else:
        # an edge between two samples: interpolating across its kink
        # reads 1e-4 of the state (trbdf2, PWM source); the mean over
        # the period ending at kT converges like the state and is not
        # sensitive to where the grid falls
        xs = np.array([[time_average(t, X[:, j], (k - 1) * T, k * T)
                        for j in range(X.shape[1])] for k in ks])
    last = (t >= (n - 1) * T) & (t <= n * T)
    scale = np.abs(X[last]).max(axis=0)
    scale = np.maximum(scale, 1e-12 * max(scale.max(), 1e-300))
    d = np.abs(np.diff(xs, axis=0)) / scale   # rows: b, (skip), 0, 1

    def geo(new, old):
        if new <= 1e-13:     # rounding: a ratio would be noise
            return new
        q = new / old if old > 0 else math.inf
        return new * q / (1.0 - q) if q < 1.0 else new + old

    xa = np.abs(xs[4] - xs[1]) / scale       # x(n) - x(3n/4)
    return float(max(max(geo(xa[j], d[0, j]), geo(d[3, j], d[2, j]))
                     for j in range(X.shape[1])))


def _raise_if_suspect(run, b) -> None:
    """A run whose own checks found its trajectory wrong is ``failed``.

    ``simulate`` only WARNS for these two, and a warning is shown once
    per call site: in an optimiser loop the record would be ``ok`` with
    a wrong number (an implausible node voltage, e.g. an open inductor;
    a trbdf2 diode held in a state its commutation rule forbids)."""
    volt = getattr(run, "_implausible_voltage", None)
    nu = int((getattr(run, "counters", None) or {}).get("unlocated_gate_steps", 0) or 0)
    if nu:
        # each such edge was moved to the next grid point: up to a whole
        # step of duty error per edge (DECISIONS 5)
        raise _Failed(
            f"unlocated_gate_edges: {nu} gate change(s) from a switch_fn "
            "without next_edge_after(t) were taken at the next grid point "
            "under sub-step correction; give the schedule an oracle or put "
            "the edges on the dt grid")
    if volt is not None:
        from . import _pulsim as _k  # type: ignore[import-not-found]
        raise _Failed("implausible_voltage: " + _k.describe_implausible_voltage(
            b.graph, b.pool, volt))
    guards = list(getattr(run, "inductor_guard_actions", None) or ())
    if guards:
        raise _Failed(f"inductor_guard: the trace of {len(guards)} inductor(s) "
                      "is the configured limit, not a solution")
    if getattr(run, "engine_used", None) != "trbdf2":
        return
    stats = getattr(run, "_trbdf2_stats", None) or {}
    if stats.get("n_gate_undersampled", 0) > 0:
        raise _Failed(
            f"gate_undersampled: a {float(stats.get('gate_pulse_min', 0.0)):.3g} s "
            "gate pulse was narrower than the switch_fn sampling, so pulses "
            "may have been stepped over; pass dt or use NativePwm2Switch")
    from ._diode_state import diode_state_traces
    try:
        bad = [tr.name for tr in diode_state_traces(b, run).values()
               if tr.n_rule_violations]
    except (AttributeError, TypeError):  # no PWL diode record
        bad = []
    if bad:
        raise _Failed(f"missed_commutation: trbdf2 held diode(s) "
                      f"{', '.join(bad)} in a state their commutation "
                      "rule forbids; cross-check with engine='pwl' and dt")
    pf = float(stats.get("probe_floor_used", 0.0) or 0.0)
    if pf > 0.0:
        # a subnet reached only through switches/diodes: its common mode
        # is set by g_off; variants of the E3 bridge read 323.94 V vs
        # 324.51 V and a 76.7 kV sample (DECISIONS 5)
        raise _Failed(
            "floating_subnet: a subnet reaches the circuit only through "
            f"switches/diodes (the zero-time probe needed {pf:.3g} s); tie "
            "it to ground or use engine='pwl'")


def _raise_if_suspect_kernel(kp, b) -> None:
    """:func:`_raise_if_suspect` for a kernel period: the implausible-
    voltage detector on every node's extrema (the dsed run has no guard
    and no trbdf2 record, so that is the only check that applies)."""
    from . import _pulsim as _k  # type: ignore[import-not-found]
    t, X = kp.node_extrema_trace()
    volt = _k.find_implausible_voltage(b.graph, b.pool, t, X, 100.0)
    if volt is not None and volt.node >= 0:
        raise _Failed("implausible_voltage: " + _k.describe_implausible_voltage(
            b.graph, b.pool, volt))


def _engine(fn):
    """Run an engine call; a kernel non-convergence (a RuntimeError,
    ``SimulationAborted`` included) becomes ``failed``. Refusals
    (``UnsupportedError``, a ValueError) and cancellation pass."""
    from . import Cancelled
    try:
        return fn()
    except (Cancelled, NotImplementedError, RecursionError):
        raise      # a stub, a refusal (_UnsupportedNotImplemented), a bug
    except RuntimeError as e:
        if type(e).__name__ == "_CxxCancelled":
            raise
        raise _Failed(f"{type(e).__name__}: {e}") from e


def _evaluate_job(args):
    """Job of the sweep runner: ``(record, step-loop callbacks)``."""
    ev, design = args
    r = ev.evaluate(design)
    return r, ("python",) if r.counters.get("python_callbacks") else ()


def evaluate(build: Callable, design: Mapping[str, Any], **evaluator_kw
             ) -> EvaluationRecord:
    """``Evaluator(build, **evaluator_kw).evaluate(design)``."""
    return Evaluator(build, **evaluator_kw).evaluate(design)


# ----------------------------------------------------------------------
# optimiser adaptors (design D3)
# ----------------------------------------------------------------------

def as_objectives(record: EvaluationRecord, names: Sequence[str], *,
                  maximise: Sequence[str] = (), penalty: float = 1e30
                  ) -> np.ndarray:
    """The KPI vector to MINIMISE (``maximise`` names negated), or
    ``penalty`` in every component for ``refused`` / ``failed`` and for
    an ``infeasible`` record without KPIs (a thermal runaway). The
    status stays on the record: count it, do not read the penalty as
    a value."""
    if record.status in ("refused", "failed") or not record.kpis:
        return np.full(len(names), float(penalty))
    return np.array([-record.kpis[n] if n in maximise else record.kpis[n]
                     for n in names], dtype=float)


def as_constraints(record: EvaluationRecord, names: Sequence[str], *,
                   penalty: float = 1e30) -> np.ndarray:
    """The violation vector (``<= 0`` feasible, NSGA-II convention), or
    ``penalty`` in every component for ``refused`` / ``failed`` and for
    a record without KPIs (a thermal runaway)."""
    if record.status in ("refused", "failed") or not record.kpis:
        return np.full(len(names), float(penalty))
    return np.array([record.violations[n] for n in names], dtype=float)
