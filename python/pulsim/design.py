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
from collections import abc as _abc
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np

import sys

from . import Cancelled
from . import _period_kpis as _pk
from ._pulsim import ConvergenceError  # type: ignore[import-not-found]
from ._refusal import UnsupportedError, unsupported
from .periodic_steady_state import _closed_loop_source
from ._design_kernel import _psum

# modules read at call time (tests replace their attributes): a
# function-level import costs importlib calls in every evaluation
_pkg = sys.modules[__package__]
_pss = sys.modules[__package__ + ".periodic_steady_state"]

__all__ = [
    "EvaluationRecord",
    "Evaluator",
    "Kpi",
    "Constraint",
    "PeriodicityError",
    "SteadySpec",
    "ThermalSpec",
    "SteadyPeriod",
    "evaluate",
    "as_objectives",
    "as_constraints",
    "STATUSES",
]

STATUSES = ("ok", "refused", "failed", "infeasible")


class PeriodicityError(Exception):
    """The exact route's shooting map and a plain ``simulate`` period
    disagree at the orbit: a defect of the map, raised out of
    ``evaluate`` / ``evaluate_many`` (never a record; not a
    ``RuntimeError``, so no engine wrapper turns it into ``failed``)."""


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


_PLAIN = (float, int, str, bool, type(None))


def _fz(m) -> FrozenDict:
    out = {}
    for k, v in m.items():
        # scalars first: an abc isinstance per KPI value was the record's
        # largest cost (DECISIONS 19)
        if type(v) in _PLAIN:
            out[k] = v
        elif isinstance(v, _abc.Mapping):
            out[k] = _fz(v)
        else:
            out[k] = tuple(v) if isinstance(v, list) else v
    return FrozenDict(out)


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
      on the ``dsed`` exact stepper, or on the Evaluator's ``'pwl'`` /
      ``'trbdf2'`` engine); its refusal is ``refused`` and
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
    """Junction temperature: ``T_j = t_amb + r_th P(T_j)`` per device.

    ``r_th`` maps a device name to its junction-to-ambient thermal
    resistance (K/W). ``loss_at`` (optional) is a callable
    ``(device, P_sim, T_j) -> P`` giving a device's loss at a junction
    temperature from the loss the simulation produced; default: the
    simulated loss (no temperature dependence). ``temperature_dependent``
    tells the evaluator that ``build`` reads ``design['_T_j']`` (a
    mapping device -> deg C) so the circuit must be re-simulated at
    each iterate.

    ``mode``: ``'kernel'`` solves the fixed point in the kernel on the
    exact steady-state route (DECISIONS 28): the orbit is fixed, the
    loss tables are re-read at each commutation of the period (turn-on:
    ``v`` before, ``i`` after; turn-off the reverse) at the trial
    ``T_j``, and the root is exact (``thermal_residual``, K), with a
    runaway refused when ``r_th dP/dT_j >= 1``. ``'fixed_point'`` is the
    damped iteration, stopping within ``tol`` of the fixed point.
    ``'auto'`` (default): ``'kernel'`` on the exact route without
    ``loss_at`` or ``temperature_dependent``, else ``'fixed_point'``.
    ``conduction`` (kernel mode): device -> ``{"i": [...], "tj": [...],
    "v_on": [[...]]}``, an on-state voltage table ``V_on(i, T_j)`` linear
    in ``i`` (one or two current knots); the device's heat is then
    ``V_on(|i|, T_j) |i|`` over its on-state current, reported as
    ``p_cond_table.<device>``, while the record's ``conduction`` loss
    stays the circuit's own (the energy balance)."""

    r_th: Mapping[str, float]
    t_amb: float = 25.0
    t_limit: float = 150.0
    tol: float = 0.1
    max_iter: int = 20
    loss_at: Optional[Callable[[str, float, float], float]] = None
    temperature_dependent: bool = False
    mode: str = "auto"
    conduction: Optional[Mapping[str, Mapping[str, Any]]] = None


def _check_thermal(th):
    """Refuse a ThermalSpec that cannot run, by name, at construction."""
    if th.mode not in ("auto", "kernel", "fixed_point"):
        raise ValueError(f"ThermalSpec.mode: 'auto', 'kernel' or "
                         f"'fixed_point', got {th.mode!r}")
    if th.mode == "kernel" and (th.loss_at is not None
                                or th.temperature_dependent):
        raise ValueError(
            "ThermalSpec(mode='kernel'): the kernel reads the loss tables "
            "on a fixed orbit; loss_at= and temperature_dependent= need "
            "the Python loop (mode='fixed_point')")
    for d, c in (th.conduction or {}).items():
        if th.mode == "fixed_point" or th.loss_at is not None \
                or th.temperature_dependent:
            raise ValueError(
                f"ThermalSpec.conduction[{d!r}]: conduction tables are "
                "read by the kernel fixed point only (mode='kernel' or "
                "'auto', without loss_at / temperature_dependent)")
        if d not in th.r_th:
            raise KeyError(f"ThermalSpec.conduction names {d!r}, which has "
                           "no r_th")
        _conduction_table(c)


def _conduction_table(c):
    """``{"i", "tj", "v_on"}`` -> the kernel ``ConductionTable``."""
    from ._pulsim import ConductionTable
    extra = sorted(set(c) - {"i", "tj", "v_on"})
    if extra:
        raise ValueError(f"a conduction table takes i, tj, v_on; got {extra}")
    v = np.asarray(c["v_on"], dtype=float)
    return ConductionTable([float(x) for x in c["i"]],
                           [float(x) for x in c["tj"]],
                           v.ravel(order="C").tolist())


@dataclass(frozen=True)
class Kpi:
    """A user KPI: ``fn(period: SteadyPeriod) -> float``.

    The constructors are declarative -- a kind and a target, no code --
    so they hash by value, pickle (``evaluate_many(backend='process')``)
    and are computed by the evaluator itself: on the kernel route from
    the closed-form period integrals (the node joins the kernel's
    observables), on a sampled route from the period's samples with
    right limits, exactly as the default KPIs. One that names a default
    KPI is that default. ``Kpi.efficiency()``, ``Kpi.ripple(node)``
    (peak-to-peak, ``ripple_v.<node>``), ``Kpi.mean(node)``
    (time-weighted, ``v_avg.<node>``), ``Kpi.i_peak(inductor)``,
    ``Kpi.i_rms(inductor)``, ``Kpi.ripple_i(inductor)``,
    ``Kpi.formula(name, expr)``."""

    name: str
    fn: Callable[["SteadyPeriod"], float]

    @classmethod
    def efficiency(cls) -> "Kpi":
        return cls("efficiency", _KpiSpec("efficiency"))

    @classmethod
    def ripple(cls, node: str) -> "Kpi":
        return cls(f"ripple_v.{node}", _KpiSpec("ripple_v", str(node)))

    @classmethod
    def mean(cls, node: str) -> "Kpi":
        return cls(f"v_avg.{node}", _KpiSpec("v_avg", str(node)))

    @classmethod
    def i_peak(cls, inductor: str) -> "Kpi":
        return cls(f"i_peak.{inductor}", _KpiSpec("i_peak", str(inductor)))

    @classmethod
    def i_rms(cls, inductor: str) -> "Kpi":
        return cls(f"i_rms.{inductor}", _KpiSpec("i_rms", str(inductor)))

    @classmethod
    def ripple_i(cls, inductor: str) -> "Kpi":
        return cls(f"ripple_i.{inductor}", _KpiSpec("ripple_i",
                                                     str(inductor)))

    @classmethod
    def formula(cls, name: str, expr: str) -> "Kpi":
        """``name = expr``: arithmetic over KPI names computed before it
        (``i_peak.L``, ``v_out``, earlier user KPIs), design variables
        and ``loss.<device>`` (that device's total loss): numbers,
        ``+ - * / **``, ``max``, ``min``, ``abs``, ``sqrt``, ``exp``,
        ``log``, ``inf``. Declarative, so it pickles and the kernel batch
        computes it (DECISIONS 26)."""
        from ._design_kernel import Formula
        return cls(str(name), _KpiSpec("formula", Formula(expr)))


@dataclass(frozen=True)
class _KpiSpec:
    """The body of a declarative :class:`Kpi` (computed by the
    evaluator; calling it on a :class:`SteadyPeriod` is an error)."""

    kind: str
    target: Any = None      # a node / device name, or a Formula

    def __call__(self, per):
        raise TypeError("a declarative Kpi is computed by the Evaluator")


@dataclass(frozen=True)
class Constraint:
    """A bound on a KPI that hashes by value and pickles:
    ``Constraint.max('ripple_v', 0.05)``, ``Constraint.min('efficiency',
    0.95)``, ``Constraint.between(kpi, lo, hi)``.
    ``Evaluator(constraints=[...])`` takes a sequence of them as well as
    the ``{kpi: (lo, hi)}`` mapping."""

    kpi: str
    lo: Optional[float] = None
    hi: Optional[float] = None

    @classmethod
    def max(cls, kpi: str, hi: float) -> "Constraint":
        return cls(str(kpi), None, float(hi))

    @classmethod
    def min(cls, kpi: str, lo: float) -> "Constraint":
        return cls(str(kpi), float(lo), None)

    @classmethod
    def between(cls, kpi: str, lo: float, hi: float) -> "Constraint":
        return cls(str(kpi), float(lo), float(hi))


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
    if callable(v) and callable(getattr(v, "next_edge_after", None)):
        # an opaque native schedule (SwitchSchedule): by its behaviour,
        # the first edges and the mask after each (else two duties of one
        # helper would share every design id)
        return {"schedule": _schedule_fingerprint(v)}
    if callable(v):
        return f"{getattr(v, '__module__', '?')}.{getattr(v, '__qualname__', type(v).__qualname__)}"
    raise TypeError(
        f"values must be numbers, strings, booleans, arrays, functions or "
        f"mappings / sequences of them, got {type(v).__name__}: {v!r}")


def _schedule_fingerprint(sf, n_edges=64):
    """``[(t.hex(), mask)]``: the mask at 0 and after each of the first
    ``n_edges`` edges. shortcut: two schedules equal on these edges and
    different later share it; 64 edges cover several periods of every
    built-in helper."""
    out = [("0", str(sf(0.0)))]
    t = 0.0
    for _ in range(n_edges):
        t = float(sf.next_edge_after(t))
        if not math.isfinite(t):
            break
        out.append((t.hex(), str(sf(t))))
    return out


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

def _with_right_limits(res, y_of_states, edges=None):
    """``(t, y)`` with each event's right limit inserted after its left
    limit (repeated instant), so a jump is integrated as a jump.

    ``edges`` (for integrals; the schedule's gate-edge instants): a
    fixed-step ``pwl`` run records no right limit. Where the sample AT a
    gate edge holds the left limit, integrating across it charged half a
    step of the old value to the new mask's step (``p_in`` 1 % high on a
    DCM buck at T/200, O(h), DECISIONS 29). Each side is integrated
    separately: the right limit at the edge is extrapolated from the two
    samples after it (exact for a quantity linear there; O(h^2))."""
    from ._result_views import states_as_array
    t = np.asarray(res.times, dtype=float)
    y = np.asarray(y_of_states(states_as_array(res)), dtype=float)
    idx = getattr(res, "right_limit_index", None)
    rl = getattr(res, "right_limit_states", None)
    if edges is not None and (idx is None or len(idx) == 0):
        ss = getattr(res, "switch_states", None)
        ss = np.asarray([] if ss is None else ss, dtype=bool)
        if ss.ndim == 2 and ss.shape[0] == t.size and ss.shape[1] > 0:
            return _two_sided(t, y, ss, *edges)
    if idx is None or rl is None or len(idx) == 0:
        return t, y
    idx = np.asarray(idx, dtype=np.int64)
    yr = np.asarray(y_of_states(np.asarray(rl, dtype=float)), dtype=float)
    return np.insert(t, idx + 1, t[idx]), np.insert(y, idx + 1, yr)


def _two_sided(t, y, ss, sf, edges):
    """:func:`_with_right_limits` on a fixed-step run: where a gate edge
    of ``sf`` falls AT sample j and sample j still holds the old gate
    mask (the left limit) while j+1 holds the new one, the step after
    t_j ran on the new mask; the right limit is inserted at t_j, from
    samples j+1, j+2 (from j+1 alone when the mask lasts one step). Where the edge is at sample j+1 instead, the step
    ending there was taken with the new mask at its end (the engine's
    own first order at an edge, docs/known-limitations.md): the
    trapezoid of the samples is that step's energy, and the balance
    closes on it; left as it is."""
    n_bits = ss.shape[1]

    def bits(m):
        return np.array([bool(m.get(i)) for i in range(n_bits)])
    tol = 1e-12 * max(1.0, float(np.abs(t).max()))
    h = float(np.min(np.diff(t))) if t.size > 1 else 0.0
    ins_at, ins_t, ins_y = [], [], []
    for e in edges:
        j = int(np.searchsorted(t, e - tol))
        before, after = bits(sf(e - 1e-6 * h)), bits(sf(e))
        g = before != after
        if not g.any():
            continue                 # a schedule instant, no mask change
        if not (j < t.size and abs(t[j] - e) <= tol):
            # an edge between samples is first order in the engine and
            # pairs with its neighbours' trapezoids (a dead time): one
            # side corrected and not the other measured worse (VSI,
            # 4e-4 -> 1.5e-3); no correction for the run
            return t, y
        if not (j + 2 < t.size and t[j + 2] > t[j + 1] > t[j]):
            continue
        if (ss[j][g] != before[g]).any() \
                or (ss[j + 1][g] != after[g]).any():
            continue
        ins_at.append(j + 1)
        ins_t.append(t[j])
        # the new side's own samples: two give a line; one (a mask held
        # for a single step, a dead time) gives its value
        ins_y.append(y[j + 1] - (y[j + 2] - y[j + 1])
                     * ((t[j + 1] - t[j]) / (t[j + 2] - t[j + 1]))
                     if (ss[j + 1] == ss[j + 2]).all() else y[j + 1])
    if not ins_at:
        return t, y
    return np.insert(t, ins_at, ins_t), np.insert(y, ins_at, ins_y)


def _gate_edges(sf, a, z):
    """``(sf, edge instants in [a, z])``, or None (an opaque callable,
    or no edge)."""
    nxt = getattr(sf, "next_edge_after", None)
    if nxt is None or not callable(nxt):
        return None
    out, t = [], a - 1e-9 * (z - a)
    for _ in range(1_000_000):
        t_new = float(nxt(t))
        if not math.isfinite(t_new) or t_new > z + 1e-12 * abs(z):
            return (sf, out) if out else None
        if not t_new > t:
            return None          # a schedule that does not advance: no correction
        out.append(t := t_new)
    return None                  # over a million edges: no correction


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
        state between calls. Give ``build`` or ``circuit``, not both.
    circuit
        A :class:`pulsim.Circuit` (``builder.build()``) shared by every
        evaluation and every worker thread: a design is
        ``circuit.with_params`` / ``with_value`` of it through
        ``fields``, with no rebuild and one symbolic analysis per batch.
        Or a picklable zero-argument function returning it (a
        module-level function, a ``functools.partial``): the evaluator
        then reaches process workers, each building the base once and
        checking its ``value_hash`` against the parent's.
    fields
        With ``circuit``: ``{design variable: target}``, the target a
        named parameter (``builder.param``) or ``"<component>.<field>"``
        (the names of ``Circuit.devices()`` and of the sensitivities).
        Design keys not in it (a frequency, a part name) are read by
        ``period``, ``schedule`` and the KPIs only.
    switch_fn, schedule
        With ``circuit``: the gate schedule, fixed (``switch_fn``) or
        per design (``schedule(design) -> switch_fn``); neither for a
        circuit without controlled switches.
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
        the engine of the transient route and of ``method='pss'``
        (``'auto'``/``'exact'``: the ``dsed`` exact stepper; ``'pwl'``,
        ``'trbdf2'``: the orbit of that engine's own transient, route
        ``'pss/pwl'``). ``method='auto'`` takes ``'pss'`` on the exact
        stepper only.
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

    def __init__(self, build: Optional[Callable] = None, *, period,
                 circuit: Any = None,
                 fields: Optional[Mapping[str, str]] = None,
                 switch_fn: Any = None,
                 schedule: Optional[Callable] = None,
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
        if steady.min_periods < 8 or steady.max_periods < steady.min_periods:
            raise ValueError("SteadySpec needs 8 <= min_periods <= max_periods "
                             "(the settle estimate reads x(n/2) ... x(n))")
        if not (isinstance(samples_per_period, int) and samples_per_period >= 4):
            raise ValueError("samples_per_period must be an int >= 4")
        if (build is None) == (circuit is None):
            raise TypeError("Evaluator: give build= (a factory) or circuit= "
                            "(a pulsim.Circuit), exactly one")
        self._circuit_factory = None
        if circuit is not None and callable(circuit) and \
                not isinstance(circuit, _pkg.Circuit):
            self._circuit_factory, circuit = circuit, circuit()
        self.circuit = circuit
        self.fields = dict(fields or {})
        self.switch_fn = switch_fn
        self.schedule = schedule
        if circuit is None:
            if fields or switch_fn is not None or schedule is not None:
                raise TypeError("Evaluator: fields=, switch_fn= and "
                                "schedule= go with circuit=; a build= "
                                "factory returns its own schedule")
        else:
            self._targets, self._spec_static = _resolve_fields(
                circuit, self.fields)
            self._circuit_hash = f"{circuit.value_hash():016x}"
            lb = circuit.to_builder() if hasattr(circuit, "to_builder") \
                else circuit
            ports = getattr(lb, "_device_loss_names", lambda: [])()
            if ports:
                # value_hash (design ids, the process cache) does not see
                # loss ports, and the evaluator reads losses from loss_options
                raise unsupported(
                    "design", "device_losses",
                    f"Evaluator(circuit=...): {ports[0]!r} has a loss port "
                    "(set_device_losses); the evaluator's losses come from "
                    "loss_options= and its design ids do not cover loss "
                    "ports. Build the circuit without it.", device=ports[0])
            if switch_fn is not None and schedule is not None:
                raise TypeError("Evaluator: give switch_fn= (fixed) or "
                                "schedule= (per design), not both")
            src = _closed_loop_source(switch_fn)
            from ._circuit_ir import c_blocks_of, graph_driven_of
            if not src and graph_driven_of(circuit):
                src = ("a control graph's drive_source ("
                       f"{', '.join(graph_driven_of(circuit))})")
            if not src and c_blocks_of(circuit):
                src = "C block(s) of add_c_block ({})".format(", ".join(
                    str(getattr(h, "name", h)) for h in c_blocks_of(circuit)))
            if src:
                # here, not per design: the kernel batch never calls _build
                # (and before the options digest, which cannot read a gate)
                raise unsupported(
                    "design", "closed_loop",
                    f"Evaluator(circuit=...): driven by {src}; "
                    "the evaluator runs no controller, so it would hold its "
                    "initial value. Give an open-loop schedule "
                    "(make_pwm_switch_fn at the design's duty).")
            if thermal is not None and thermal.temperature_dependent:
                raise ValueError(
                    "Evaluator(circuit=...): ThermalSpec.temperature_"
                    "dependent needs a build= factory that reads "
                    "design['_T_j']; a fixed circuit has no such input")
        self._spec_cache: dict = {}
        #: (topology, event pattern) -> relative deviation of the independent
        #: periodicity check (review item c1), this process only
        self.periodicity_checks: dict = {}
        self.build = build
        self.period = period
        self.loads = tuple(loads)
        self.sources = None if sources is None else tuple(sources)
        self.output = output
        self.kpis = tuple(kpis)
        names = [k.name for k in self.kpis]
        dup = sorted({n for n in names if names.count(n) > 1})
        if dup:
            raise ValueError(f"kpis: {dup} given twice")
        self._nodes = tuple(sorted({
            k.fn.target for k in self.kpis if isinstance(k.fn, _KpiSpec)
            and k.fn.kind in ("ripple_v", "v_avg")} - {output}))
        if constraints is not None and not isinstance(constraints, Mapping):
            cs = list(constraints)
            bad = [c for c in cs if not isinstance(c, Constraint)]
            if bad:
                raise TypeError(f"constraints: give a {{kpi: (lo, hi)}} "
                                f"mapping or Constraint objects, got "
                                f"{bad[0]!r}")
            names = [c.kpi for c in cs]
            dup = sorted({n for n in names if names.count(n) > 1})
            if dup:
                raise ValueError(f"constraints: {dup} bounded twice; use "
                                 "Constraint.between")
            constraints = {c.kpi: (c.lo, c.hi) for c in cs}
        self.constraints = {k: tuple(v) for k, v in (constraints or {}).items()}
        for k, v in self.constraints.items():
            if len(v) != 2:
                raise ValueError(f"constraint {k!r}: give (lo, hi), got {v!r}")
        self.thermal = thermal
        if thermal is not None:
            _check_thermal(thermal)
        self.steady = steady
        self.engine = engine
        self.samples_per_period = samples_per_period
        self.options = dict(options or {})
        for k in ("closed_loops", "step_observer"):
            if self.options.get(k) is not None:
                raise unsupported(
                    "design", "closed_loop",
                    f"Evaluator: options[{k!r}] runs a controller; the "
                    "evaluator's steady-state runs restart it per run and per "
                    "design (its state carries over), so records would "
                    "depend on evaluation order. Evaluate the open-loop plant "
                    "at a fixed duty.")
        if _closed_loop_source(self.options.get("b_extra_fn")):
            raise unsupported(
                "design", "closed_loop",
                "Evaluator: options['b_extra_fn'] is a control graph's "
                "driven-source overlay; the evaluator runs no controller.")
        self.loss_options = dict(loss_options or {})
        if gradients is not None and not isinstance(gradients, Mapping):
            gradients = {str(g): str(g) for g in gradients}
        self.gradients = dict(gradients) if gradients else None
        if self.gradients and thermal is not None:
            raise unsupported(
                "design.evaluate", "gradients", "Evaluator: gradients= with "
                "a thermal loop: the junction temperature depends on the "
                "parameters through the loss model, which the exact path "
                "does not differentiate (refused, never approximated).")
        self._options_digest = _digest(self._config())
        self._frozen = (self.kpis, dict(self.constraints))

    # -- pickling (the process backend) ---------------------------------
    def __getstate__(self):
        st = dict(self.__dict__)
        st["_spec_cache"] = {}
        st["periodicity_checks"] = {}
        st.pop("_prov", None)
        st.pop("_kplan", None)      # kernel tables: rebuilt on demand
        st.pop("last_batch", None)
        if self._circuit_factory is not None:
            st["circuit"] = None     # rebuilt by the factory in the worker
        return st

    def __setstate__(self, st):
        self.__dict__.update(st)
        if self.circuit is None and self._circuit_factory is not None:
            want = f"{self._circuit_hash}"
            c = _BASES.get(want)
            if c is None:
                c = self._circuit_factory()
                got = f"{c.value_hash():016x}"
                if got != want:
                    raise RuntimeError(
                        "Evaluator(circuit=<factory>): the factory built a "
                        f"circuit with value_hash {got} in this process, "
                        f"not the parent's {want}; it must be "
                        "deterministic")
                _BASES[want] = c
            self.circuit = c

    # -- identity --------------------------------------------------------
    def _config(self):
        th = None
        if self.thermal is not None:
            th = asdict(self.thermal)
        form = dict(build=self.build)
        if self.circuit is not None:
            # the base by its value hash (structure and every value), the
            # design by its variables: no per-record circuit hashing
            form = dict(circuit=self._circuit_hash,
                        fields=self.fields, switch_fn=self.switch_fn,
                        schedule=self.schedule)
        return dict(
            **form, period=self.period, loads=self.loads,
            sources=self.sources, output=self.output,
            kpis=[(k.name, k.fn) for k in self.kpis],
            constraints=self.constraints, thermal=th,
            steady=asdict(self.steady), engine=self.engine,
            samples=self.samples_per_period, options=self.options,
            loss_options=self.loss_options, gradients=self.gradients)

    def design_id(self, design: Mapping[str, Any]) -> str:
        # the parameter vector: a flat design of numbers and strings is
        # canonicalised inline (what _canon_value gives it, faster)
        flat = {}
        for k, v in design.items():
            t = type(v)
            if t is float:
                flat[str(k)] = v.hex()
            elif t is str or t is bool or t is int:
                flat[str(k)] = v
            else:
                flat = None
                break
        if flat is None:
            return _digest({"design": dict(design),
                            "options": self._options_digest})
        s = json.dumps({"design": flat, "options": self._options_digest},
                       sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(s.encode()).hexdigest()

    def _provenance(self):
        # one frozen mapping per evaluator: it does not depend on the design
        prov = self.__dict__.get("_prov")
        if prov is None:
            from . import __version__
            prov = self._prov = _fz({"pulsim": str(__version__),
                                     "build": _build_hash(),
                                     "options": self._options_digest})
        return prov

    # -- public ----------------------------------------------------------
    def evaluate(self, design: Mapping[str, Any]) -> EvaluationRecord:
        """One design -> one record (design D3 status mapping)."""
        t0 = time.perf_counter()
        self._check_frozen()
        design = dict(design)
        did = self.design_id(design)
        counters: dict = {"simulations": 0}
        base = dict(design=_fz(design), design_id=did,
                    provenance=self._provenance())

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

    def _check_frozen(self):
        """kpis= and constraints= key every design id and the kernel plan;
        a change after construction is refused, never run with the old
        ones. (Not a config digest: a schedule closure that records its
        calls changes its digest and is legitimate.)"""
        k0, c0 = self._frozen
        if self.kpis is not k0 or self.constraints != c0:
            raise ValueError(
                "Evaluator: kpis= or constraints= changed "
                "after the evaluator was made; the design ids and the "
                "kernel plan were built from the old ones. Make a new "
                "Evaluator instead of mutating this one.")

    def evaluate_many(self, designs: Sequence[Mapping[str, Any]], *,
                      workers: Optional[int] = None,
                      backend: Optional[str] = None) -> list:
        """Records in input order (design D6). ``backend``:

        * ``None`` (default): the kernel batch when the evaluator's plan
          allows it (circuit form, exact steady-state route, declarative
          KPIs and constraints: DECISIONS 26) -- every design in one
          kernel call on ``workers`` native threads, no Python per
          design; otherwise, and for a design the kernel does not
          finish, the Python path, its record's ``counters['batch']``
          naming why. ``self.last_batch`` counts both.
        * ``'kernel'``: the same, refused (``ValueError`` naming the
          reason) when the plan does not allow it.
        * ``'thread'``: the Python path on the job runner of
          :func:`pulsim.sweep` (the kernel releases the GIL).
        * ``'process'``: Python-heavy designs -- ``build`` and the KPIs
          must then pickle.

        Design-specific outcomes never stop the batch; it stops with the
        refusal when its first three designs are ``refused`` with
        identical fields (the configuration, not the design, is
        unsupported). Any other exception propagates."""
        from .sweep import _run_jobs
        self._check_frozen()
        if backend not in (None, "kernel", "thread", "process"):
            raise ValueError(f"backend must be None, 'kernel', 'thread' or "
                             f"'process', got {backend!r}")
        if backend in (None, "kernel"):
            from . import _design_kernel as _dk
            plan = _dk.plan_of(self)
            if isinstance(plan, str):
                if backend == "kernel":
                    raise ValueError(f"evaluate_many(backend='kernel'): the "
                                     f"evaluator has no kernel plan: {plan}")
                recs = self.evaluate_many(designs, workers=workers,
                                          backend="thread")
                self.last_batch = _fz({"designs": len(recs), "kernel": 0,
                                       "python": len(recs),
                                       "python_reasons": {plan: len(recs)}})
                return [dataclasses_replace(r, counters=_fz(
                    {**r.counters, "batch": "python: " + plan}))
                        for r in recs]
            designs = [dict(d) for d in designs]
            out = _dk.run(self, plan, designs[:3], 1)
            self._refusal_rule(out)
            if len(designs) > 3:
                head = self.last_batch
                out += _dk.run(self, plan, designs[3:], workers)
                self.last_batch = _merge_batches(head, self.last_batch)
            return out
        jobs = [(self, dict(d)) for d in designs]
        if backend == "process" and workers != 1 and len(jobs) > 3:
            # refused by name before the head runs or a worker starts
            self._check_picklable([d for _, d in jobs[3:]])
        head = jobs[:3]
        out, _, _ = _run_jobs(head, parallel=False, backend="thread",
                              n_workers=1, verbose=False,
                              job=_evaluate_job, reraise=True)
        self._refusal_rule(out)
        rest = jobs[3:]
        if rest and backend == "process" and workers != 1:
            # chunks: the evaluator is pickled once per chunk, not per
            # design; four chunks per worker balance the load
            import os
            nw = workers or max(1, (os.cpu_count() or 2) - 1)
            size = -(-len(rest) // (4 * nw))
            chunks = [(self, [d for _, d in rest[i:i + size]])
                      for i in range(0, len(rest), size)]
            more, _, _ = _run_jobs(
                chunks, parallel=True, backend="process", n_workers=nw,
                verbose=False, job=_evaluate_chunk, reraise=True)
            out += [r for recs in more for r in recs]
        elif rest:
            more, _, _ = _run_jobs(
                rest, parallel=workers != 1, backend=backend,
                n_workers=workers, verbose=False, job=_evaluate_job,
                reraise=True)
            out += more
        return out

    @staticmethod
    def _refusal_rule(out):
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

    def _check_picklable(self, designs):
        """Refuse a process batch whose evaluator or designs do not
        pickle, naming the part, before a worker starts."""
        import pickle
        parts = []
        if self._circuit_factory is not None:
            parts.append(("circuit (factory)", self._circuit_factory))
        elif self.circuit is not None:
            parts.append(("circuit", self.circuit))
        parts += [("build", self.build), ("switch_fn", self.switch_fn),
                  ("schedule", self.schedule), ("period", self.period)]
        parts += [(f"kpis[{k.name!r}]", k.fn) for k in self.kpis]
        if self.thermal is not None:
            parts.append(("thermal.loss_at", self.thermal.loss_at))
        parts += [(f"options[{k!r}]", v) for k, v in self.options.items()]
        parts += [(f"loss_options[{k!r}]", v)
                  for k, v in self.loss_options.items()]
        parts += [(f"designs[{i + 3}]", d) for i, d in enumerate(designs)]
        parts.append(("the evaluator", self))   # anything not named above
        for name, obj in parts:
            try:
                pickle.dumps(obj)
            except Exception as e:  # noqa: BLE001 - any pickling failure
                hint = ("pass circuit= a module-level function returning the "
                        "circuit (it then rebuilds in each worker)"
                        if name == "circuit" else
                        "use a module-level function, a declarative Kpi "
                        "(Kpi.ripple(node), Kpi.i_peak(inductor), ...) or "
                        "Constraint objects")
                raise unsupported(
                    "design.evaluate_many", "backend", "evaluate_many("
                    f"backend='process'): {name} does not pickle ({e}), so "
                    f"a process worker cannot receive it; {hint}, or use "
                    "backend='thread'.", device=name) from e

    # -- one evaluation --------------------------------------------------
    def _period_of(self, design):
        T = self.period(design) if callable(self.period) else self.period
        T = float(T)
        if not (T > 0 and math.isfinite(T)):
            raise ValueError(f"period must be positive and finite, got {T!r}")
        return T

    def _circuit_of(self, design):
        """The design's circuit: ``with_params`` of its parameter
        targets, then ``with_value`` of its field targets."""
        c = self.circuit
        prm = {t: float(design[v]) for v, (kind, t) in self._targets.items()
               if kind == "param"}
        if prm:
            c = c.with_params(prm)
        for v, (kind, t) in self._targets.items():
            if kind == "field":
                c = c.with_value(t, float(design[v]))
        return c

    def _build(self, design):
        """``(src, switch_fn)``: ``src`` the builder of ``build`` or the
        design's :class:`pulsim.Circuit`."""
        if self.circuit is not None:
            missing = sorted(v for v in self._targets if v not in design)
            if missing:
                raise KeyError(f"design has no {missing} (fields= of the "
                               f"evaluator); design keys: {sorted(design)}")
            sf = (self.schedule(design) if self.schedule is not None
                  else self.switch_fn)
            out = (self._circuit_of(design), sf)
        else:
            out = self.build(design)
        if isinstance(out, tuple):
            if len(out) != 2:
                raise ValueError("build(design) must return a builder or "
                                 "(builder, switch_fn)")
        else:
            out = (out, None)
        src = _closed_loop_source(out[1])
        # a builder's side lists or a Circuit's ControlDecls
        from ._circuit_ir import c_blocks_of, graph_driven_of
        driven = graph_driven_of(out[0])
        blocks = [str(getattr(h, "name", h)) for h in c_blocks_of(out[0])]
        if driven and not src:
            src = f"a control graph's drive_source ({', '.join(driven)})"
        if blocks and not src:
            src = f"C block(s) of add_c_block ({', '.join(blocks)})"
        if src:
            # no step_observer reaches the evaluator's runs: the gate or
            # source would follow a controller that never updates
            raise unsupported(
                "design", "closed_loop",
                f"Evaluator: build(design) returned a circuit driven by "
                f"{src}; the evaluator runs no controller, so it would hold "
                "its initial value. Return an open-loop schedule "
                "(make_pwm_switch_fn at the design's duty).")
        return out

    def _evaluate(self, design, did, counters, grads=None):
        T = self._period_of(design)
        src, sf = self._build(design)
        run, a, z, b = self._steady(src, sf, T, counters)
        th = self.thermal
        if th is None:
            rows, losses = self._loss_rows(b, run, a, z)
            kpis = self._kpis(b, run, a, z, design, did, rows, losses,
                              counters, sf)
            if self.gradients:
                self._gradients(b, run, design, kpis, grads, counters)
            return kpis, losses, {}
        names = {c["name"] for c in b.components()}
        for d in th.r_th:
            if d not in names:
                raise KeyError(f"ThermalSpec.r_th names {d!r}, which is not "
                               f"a device of the circuit")
        mode = th.mode
        exact = isinstance(run, _pk.Period)
        if mode == "auto":
            mode = ("kernel" if exact and th.loss_at is None
                    and not th.temperature_dependent else "fixed_point")
        if mode == "kernel" and not exact:
            raise unsupported(
                "design", "thermal",
                f"ThermalSpec(mode='kernel'): the route is "
                f"{counters.get('route', '?')!r}, not the exact steady-state "
                "route (steady=SteadySpec(method='pss'), engine 'dsed') "
                "whose period map the kernel reads; use mode='fixed_point'")
        if mode == "fixed_point" and th.conduction:
            raise unsupported(
                "design", "thermal",
                "ThermalSpec.conduction: conduction tables are read by the "
                f"kernel fixed point, which needs the exact route (this is "
                f"{counters.get('route', '?')!r})")
        if mode == "kernel":
            return self._thermal_kernel(b, run, a, z, design, did, counters)
        counters["thermal_mode"] = "fixed_point"
        tj = {d: float(th.t_amb) for d in th.r_th}
        omega, prev, dist = 1.0, math.inf, math.inf
        for k in range(1, th.max_iter + 1):
            rows, losses = self._loss_rows(b, run, a, z, tj)
            losses = self._losses_at(losses, tj)
            new = {}
            for d, r in th.r_th.items():
                t_hat = th.t_amb + float(r) * _psum(losses[d].values())
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
                src, sf = self._build({**design, "_T_j": _fz(tj)})
                run, a, z, b = self._steady(src, sf, T, counters)
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
        kpis = self._kpis(b, run, a, z, design, did, rows, losses, counters,
                          sf)
        for d in tj:
            kpis[f"t_j.{d}"] = tj[d]
        return kpis, losses, tj

    def _thermal_kernel(self, b, run, a, z, design, did, counters):
        """``mode='kernel'``: each device's ``T_j = t_amb + r_th P(T_j)``
        solved by the kernel on the fixed orbit (DECISIONS 28)."""
        from ._pulsim import _junction_fixed_point
        th = self.thermal
        rows, losses = self._loss_rows(b, run, a, z)
        specs = self.loss_options.get("switch_specs") or {}
        semis = {n: bit for n, _, bit in run.spec.semis}
        kinds = {n: kind for n, kind, _ in run.spec.semis}
        tj, evals, resid, heat = {}, 0, 0.0, {}
        for d, r_th in th.r_th.items():
            sp = specs.get(d) or {}
            on_t, off_t = sp.get("E_on_table"), sp.get("E_off_table")
            cat = dict(losses.get(d) or {})
            on = off = []
            if on_t is not None or off_t is not None:
                for t, ref, curve in ((on_t, "E_on_ref", "E_on_curve"),
                                      (off_t, "E_off_ref", "E_off_curve")):
                    if t is None and (float(sp.get(ref, 0.0)) > 0
                                      or sp.get(curve) is not None):
                        raise unsupported(
                            "design", "thermal",
                            f"switch_specs[{d!r}] mixes a loss table with "
                            f"{ref} / {curve}; the kernel fixed point reads "
                            "tables only (mode='fixed_point')")
                on, off = (
                    [(v, i) for v, i, soft in pts if not soft]
                    for pts in _pk.commutations(run.edges(d, semis[d]),
                                                kinds[d] == "switch"))
                cat.pop("switching", None)
                cat.pop("recovery", None)
            cond, q1, q2, i_max = None, 0.0, 0.0, 0.0
            if d in (th.conduction or {}):
                cond = _conduction_table(th.conduction[d])
                st = run.st[run.spec.idx[("ion", d)]]
                if float(st[4]) >= 0.0:
                    q1, i_max = float(st[0]), float(st[2])
                elif float(st[2]) <= 0.0:
                    q1, i_max = -float(st[0]), -float(st[4])
                else:
                    raise unsupported(
                        "design", "thermal",
                        f"ThermalSpec.conduction[{d!r}]: the on-state "
                        "current changes sign over the period, so int |i| "
                        "is not a period integral")
                q2 = float(st[1])
                cat.pop("conduction", None)
            try:
                out = _junction_fixed_point(
                    on, off, None if on_t is None else on_t._kernel(),
                    None if off_t is None else off_t._kernel(), run.T,
                    _psum(cat.values()), float(r_th), float(th.t_amb), cond,
                    q1, q2, i_max)
            except ValueError as e:
                if cond is not None and "conduction table" in str(e):
                    raise unsupported(
                        "design", "thermal",
                        f"ThermalSpec.conduction[{d!r}]: {e}") from e
                raise _Failed(f"thermal: {d}: {e}") from e
            except RuntimeError as e:    # the root was not located
                raise _Failed(f"thermal: {d}: {e}") from e
            evals += int(out["evaluations"])
            if out["runaway"]:
                counters["thermal_mode"] = "kernel"
                counters["thermal_iterations"] = evals
                raise _Infeasible(
                    f"thermal_runaway: {d} has no junction temperature above "
                    f"t_amb = {th.t_amb:g} C (r_th dP/dT_j = "
                    f"{out['slope']:.4g} >= 1 at {out['tj']:.4g} C)",
                    {"thermal_runaway": max(out["slope"] - 1.0, 1e-12)})
            tj[d] = float(out["tj"])
            resid = max(resid, abs(float(out["residual"])))
            if cond is not None:
                heat[d] = float(out["p_cond_table"])
        counters["thermal_mode"] = "kernel"
        counters["thermal_iterations"] = evals
        counters["thermal_residual"] = resid
        hot = max(tj[d] - th.t_limit for d in tj)
        if hot > 0:
            raise _Infeasible(
                f"thermal_limit: the fixed point T_j = {max(tj.values()):.4g} C "
                f"exceeds t_limit = {th.t_limit:g} C", {"thermal_limit": hot})
        rows, losses = self._loss_rows(b, run, a, z, tj)
        losses = self._losses_at(losses, tj)
        kpis = self._kpis(b, run, a, z, design, did, rows, losses, counters)
        for d in tj:
            kpis[f"t_j.{d}"] = tj[d]
        for d, v in heat.items():
            kpis[f"p_cond_table.{d}"] = v
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
            p0 = _psum(cat.values())
            p1 = float(th.loss_at(d, p0, tj[d]))
            out[d] = ({c: v * p1 / p0 for c, v in cat.items()} if p0 > 0
                      else {"conduction": p1})
        return out

    def _steady(self, b, sf, T, counters):
        """``(result, t_from, t_to, builder)`` of one steady period;
        ``b`` is a builder or (``circuit=``) a :class:`pulsim.Circuit`,
        which the runs take as it is (one symbolic analysis per
        structure)."""
        method = self.steady.method
        if method == "auto":
            if self.engine not in ("auto", "dsed"):
                method = "transient"
                counters["steady_route_reason"] = (
                    f"engine={self.engine!r}: pss runs on dsed only")
            else:
                # the capability probe builds the exact stepper: a kernel
                # error there is a failed design, not a crash (_engine)
                e = _engine(lambda: _pss._pss_refusal(b, T, sf))
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
            raise unsupported(
                "design.evaluate", "gradients", "Evaluator: gradients= on "
                f"the {method} route ({counters.get('steady_route_reason', '')}"
                "): exact gradients exist on the pss route only (refused, "
                "never a finite difference).")
        if method == "pss":
            return self._pss(b, sf, T, counters)
        return self._transient(b, sf, T, counters)

    def _transient(self, src, sf, T, counters):
        """Simulate ``n`` periods (``n`` doubling from ``min_periods``)
        until the true state at the period starts has settled."""
        b = _as_builder(src)
        spec = self.steady
        engine = self.engine
        kw = dict(self.options)
        _refuse_ignored_options(kw, engine)
        n = max(2, spec.min_periods)
        while True:
            run = _engine(lambda: _pkg.simulate(
                src, t_end=n * T, engine=engine, switch_fn=sf, **kw))
            # 'auto' resolves per circuit: check against what ran
            _refuse_ignored_options(kw, getattr(run, "engine_used", engine))
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
            return run, t1 - T, t1, b
        # dsed: the settle run has no step cap; re-run the last period
        # from its true state with the KPI sampling.
        x0 = np.asarray(run.states_reduced, dtype=float)[-1]
        run2 = _engine(lambda: _pkg.simulate(
            src, t_end=T, dt=T / self.samples_per_period, engine="dsed",
            switch_fn=sf, initial_state=x0,
            **{k: v for k, v in kw.items() if k != "dt"}))
        self._count(run2, counters, b, sf)
        return run2, 0.0, T, b

    def _check_periodic(self, src, sf, T, r, b):
        """Review item c1: the kernel's verification period IS the
        shooting map, so it cannot catch a defect of that map. One plain
        ``simulate(engine='dsed')`` period from ``x*`` (another entry
        point: dispatch, adapter, event projection) must return to
        ``x*``. It costs one to two evaluations (DECISIONS 22), so it runs
        once per (topology, event pattern) per evaluator and process,
        logged in ``periodicity_checks``. A passing check is a sample,
        not a per-design guarantee; a mismatch -- or a check run that the
        engine itself fails -- raises :class:`PeriodicityError` out of the
        batch (a defect of the map, not an outcome of the design), so no
        record depends on which design was checked. A design with an
        event within 1e-9 T of the section (where the map's attach rule
        and the engine's can differ) is always checked: the pattern alone
        does not stand for it."""
        key = (tuple((c["kind"], tuple(c.get("nodes", ())))
                     for c in b.components()),
               tuple(name for name, _ in r.pattern))
        at_section = any(min(f, 1.0 - f) <= 1e-9 for _, f in r.pattern)
        done = None if at_section else self.periodicity_checks.get(key)
        if done is None:
            try:
                run = _pkg.simulate(
                    src, t_end=T, dt=T, engine="dsed", switch_fn=sf,
                    initial_state=r.x0, rtol=1e-11, atol=1e-14)
            except (RuntimeError, ValueError) as e:
                if type(e).__name__ == "_CxxCancelled":
                    raise
                raise PeriodicityError(
                    "periodicity: a plain simulate(engine='dsed') period "
                    f"from the shooting's x* failed ({type(e).__name__}: "
                    f"{e}) where the shooting map ran it: the map and the "
                    "engine disagree for the event pattern "
                    f"{[n for n, _ in r.pattern]}") from e
            xs = np.asarray(run.states_reduced, dtype=float)
            dev = np.abs(xs[-1] - r.x0)
            den = np.abs(xs).max(axis=0) + np.abs(r.x0)
            # a state that sits at zero is measured against the largest
            den = np.maximum(den, 1e-12 * float(den.max(initial=0.0)))
            with np.errstate(divide="ignore", invalid="ignore"):
                q = np.where(dev == 0.0, 0.0, dev / np.where(den > 0, den,
                                                             1e-300))
            done = float(q.max())
            if not at_section:
                self.periodicity_checks[key] = done
        if not done <= _PERIODIC_RTOL:
            raise PeriodicityError(
                "periodicity: a plain simulate(engine='dsed') period from "
                "the shooting's x* moved it by "
                f"{done:.3e} (relative, tolerance {_PERIODIC_RTOL:.0e}); the "
                "shooting map and the engine disagree for the event "
                f"pattern {[n for n, _ in r.pattern]}")

    def _spec(self, src):
        """The kernel observables: once per evaluator on the circuit
        form when no design variable moves a value they read."""
        static = self.circuit is not None and self._spec_static
        key = src.structure_hash() if static else None
        spec = self._spec_cache.get(key) if static else None
        if spec is None:
            spec = _pk.spec_for(
                _as_builder(src), self.output, self.loss_options,
                self._nodes, tuple(sorted(
                    (self.thermal.conduction or {}) if self.thermal else ())))
            if static:
                self._spec_cache[key] = spec   # read-only once stored
        return spec

    def _pss(self, src, sf, T, counters):
        if self.engine in ("pwl", "trbdf2"):
            return self._pss_transient(src, sf, T, counters)
        counters["route"] = "pss/dsed"
        if self.options:
            # the exact stepper takes no simulate keywords: dropping them
            # would answer a question the caller did not ask
            raise unsupported(
                "design.evaluate", "options",
                f"Evaluator: options {sorted(self.options)} on the pss/dsed "
                "route: the exact stepper has no step size, tolerance or "
                "other simulate keyword to honour them with (refused, never "
                "dropped). Remove options=, or pick the transient route "
                "(SteadySpec(method='transient')) or engine='pwl' / "
                "'trbdf2'.")
        spec = self._spec(src)
        obs = spec if isinstance(spec, _pk.Spec) else None

        def solve():
            try:
                return _pss.periodic_steady_state(
                    src, period=T, switch_fn=sf,
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
        b = ctx.builder
        if ctx._map is not None:
            self._check_periodic(src, sf, T, r, b)
        counters["python_callbacks"] = int(ctx.n_python_callbacks)
        counters["nogil_period_runs"] = int(ctx.n_nogil_runs)
        counters["pss_loop"] = ctx.loop
        counters["pss_iterations"] = len(r.residual_history)
        d = r._period

        def sampled(n):
            run = _engine(lambda: _pkg.simulate(
                src, t_end=n * T, dt=T / self.samples_per_period,
                engine="dsed", switch_fn=sf, initial_state=r.x0, rtol=1e-11,
                atol=1e-14))
            self._count(run, counters, b)
            return run

        if obs is not None and d is not None and d["integrals_ok"]:
            # the KPIs are the kernel's closed-form period integrals;
            # waveforms only when a user KPI reads them
            counters["kpi_path"] = "kernel"
            kp = _pk.Period(obs, d, T)
            _raise_if_suspect_kernel(kp, b)
            # two periods, the second read: as the sampled route below
            kp.lazy_run = _Deferred(lambda: sampled(2))
            kp.pss_orbit = r
            return kp, 0.0, T, b
        counters["kpi_path"] = "sampled: " + (
            spec if obs is None else
            "no kernel period map" if d is None else d["integrals_reason"])
        # Two periods, the second measured: a switch edge AT the section
        # is then an edge inside the window (a one-period run starts in
        # the new state and its turn-on is never seen: E_on was lost).
        run = sampled(2)
        if self.gradients:
            run.pss_orbit = r
        return run, T, 2 * T, b

    def _pss_transient(self, src, sf, T, counters):
        """``method='pss'`` on the fixed or variable engine: the orbit of
        that engine's own transient (add-periodic-steady-state-shooting
        5.1-5.3); the KPIs read the second of two periods run from it."""
        engine = self.engine
        counters["route"] = f"pss/{engine}"
        if self.gradients:
            raise unsupported(
                "design.evaluate", "gradients", "Evaluator: gradients= on "
                f"pss/{engine}: exact orbit sensitivities exist on the "
                "exact stepper only (refused, never a finite difference).")
        kw = dict(self.options)
        _refuse_ignored_options(kw, engine)
        dt = kw.pop("dt", None)

        def solve():
            try:
                return _pss.periodic_steady_state(
                    src, period=T, switch_fn=sf, engine=engine, dt=dt,
                    options=kw, max_period_runs=self.steady.max_period_runs)
            except (ConvergenceError, np.linalg.LinAlgError) as e:
                raise _Failed(str(e)) from e

        r = _engine(solve)
        if not r.stable:
            raise _Failed(
                "unstable_orbit: Floquet multipliers outside the unit "
                f"circle {[(round(abs(m), 6), k) for _, m, k in r.unstable_multipliers]}")
        m = r._ctx
        counters["pss_period_runs"] = counters.get(
            "pss_period_runs", 0) + int(r.n_period_runs)
        counters["engine_steps"] = counters.get(
            "engine_steps", 0) + int(r.n_engine_steps)
        counters["simulations"] += int(r.n_period_runs)
        counters["pss_loop"] = m.loop
        counters["pss_iterations"] = len(r.residual_history)
        counters["pss_jacobians"] = int(m.n_jacobians)
        counters["pss_broyden_updates"] = int(m.n_broyden)
        counters["kpi_path"] = "sampled: pss on " + engine
        # Two periods, the second measured (an edge AT the section is
        # then inside the window), the same restart rule as the map.
        run = _engine(lambda: m.simulate(r.x0, periods=2))
        b = m.builder
        self._count(run, counters, b)
        return run, T, 2 * T, b

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
        b2 = _as_builder(b2)
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
        if isinstance(run, _pk.Period):
            # the period's edge log is kernel output: an inconsistent one
            # ("edges do not alternate") fails this design, not the batch
            rows = _engine(lambda: _pk.loss_rows(b, run, opts))
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

    def _kpis(self, b, run, a, z, design, did, rows, losses, counters,
              sf=None):
        kp = run if isinstance(run, _pk.Period) else None
        # a fixed-step run's gate edges: its integrals take each side
        # separately (_with_right_limits)
        gates = (_gate_edges(sf, a, z) if kp is None and getattr(
            run, "engine_used", "") == "pwl" else None)
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
            t, i = _with_right_limits(run, lambda st, col=col: st[:, col],
                                      edges=gates)
            p_in -= float(c["params"]["V"]) * _mean(t, i, a, z)
        if p_in is not None:
            kpis["p_in"] = p_in
        if uncovered:
            counters["losses_not_covered"] = ", ".join(uncovered)
        else:
            p_loss = _psum(_psum(c.values()) for c in losses.values())
            kpis["p_loss"] = p_loss
        if self.loads:
            p_out = _psum(float(rows[n]["P_avg"]) for n in self.loads)
            kpis["p_out"] = p_out
            # P_out / (P_out + losses): with a datasheet model or a
            # loss table the losses include what the resistor-pair
            # circuit does not dissipate; without them it equals
            # P_out / P_in at the orbit (DECISIONS 4).
            kpis["efficiency"] = (p_out / (p_out + p_loss)
                                  if p_out + p_loss > 0 else math.nan)
            _energy_balance(
                comps, src, rows, losses, p_in, counters,
                _stored_power_kernel(kp, comps) if kp is not None
                else _stored_power(run, comps, a, z),
                "exact" if kp is not None else "sampled" if getattr(
                    run, "engine_used", "") == "dsed" else "stepped",
                self.samples_per_period)
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
        user = []
        for k in self.kpis:
            if isinstance(k.fn, _KpiSpec):
                kpis[k.name] = _declared(k, kpis, kp, run, a, z, b,
                                         design, losses)
            elif k.name in kpis:
                raise ValueError(f"user KPI {k.name!r} has the name of a "
                                 "default KPI")
            else:
                user.append(k)
        if user:
            if kp is not None:
                # the waveforms of the orbit, simulated on first access
                run, a, z = kp.lazy_run, kp.T, 2 * kp.T
            per = SteadyPeriod(run, builder=b, design=design,
                               t_from=a, t_to=z, seed=int(did, 16) % 2**32,
                               kpis=_fz(kpis), losses=_fz(losses))
            for k in user:
                kpis[k.name] = float(k.fn(per))
        return kpis


def _declared(k, kpis, kp, run, a, z, b, design, losses):
    """The value of a declarative :class:`Kpi`: the default KPI it names,
    a node's peak-to-peak / time-weighted mean on the route's own
    footing (closed form on the kernel route, samples otherwise), or a
    formula of those."""
    s = k.fn
    if s.kind == "formula":
        from ._design_kernel import formula_lookup
        return s.target.evaluate(formula_lookup(kpis, design, losses))
    if s.kind in ("efficiency", "i_peak", "i_rms", "ripple_i"):
        name = s.kind if s.target is None else f"{s.kind}.{s.target}"
        if name not in kpis:
            raise KeyError(
                f"Kpi.{s.kind}({s.target or ''}): {name!r} is not a KPI of "
                "this evaluation (efficiency needs loads=; the current "
                f"KPIs name inductors). KPIs: {sorted(kpis)}")
        return float(kpis[name])
    if s.kind in ("ripple_v", "v_avg"):
        try:
            col = int(b.node_id_of(s.target))
        except Exception as e:  # noqa: BLE001 - re-raised by name
            raise KeyError(f"Kpi.{'ripple' if s.kind == 'ripple_v' else 'mean'}"
                           f"({s.target!r}): no such node") from e
        if col < 0:
            return 0.0       # ground
        if kp is not None:
            key = ("node", s.target)
            return kp.p2p(key) if s.kind == "ripple_v" else kp.mean(key)
        t, v = _with_right_limits(run, lambda st, col=col: st[:, col])
        return _p2p(t, v, a, z) if s.kind == "ripple_v" else _mean(t, v, a, z)
    raise ValueError(f"unknown declarative KPI kind {s.kind!r}")


def dataclasses_replace(rec, **kw):
    from dataclasses import replace
    return replace(rec, **kw)


def _merge_batches(a, b):
    """Batch counters of two consecutive kernel runs, summed."""
    out = dict(b)
    for k in ("designs", "kernel", "python", "symbolic_analyses",
              "python_calls_in_kernel", "periodicity_checks"):
        out[k] = a.get(k, 0) + b.get(k, 0)
    reasons = dict(a.get("python_reasons", {}))
    for k, v in b.get("python_reasons", {}).items():
        reasons[k] = reasons.get(k, 0) + v
    out["python_reasons"] = reasons
    out["wall_s"] = a.get("wall_s", 0.0) + b.get("wall_s", 0.0)
    return _fz(out)


def _as_builder(src):
    """A builder for the post-processing (names, pool, graph) of a run
    whose input was a builder or a :class:`pulsim.Circuit`."""
    return src if hasattr(src, "pool") else src.to_builder()


# kinds whose values the kernel observables never read (spec_for reads
# source values, conductances, V_f): a whitelist, so a new kind is
# re-resolved per design until it is listed here
_SPEC_FREE_KINDS = ("inductor", "capacitor", "resistor")


def _resolve_fields(circuit, fields):
    """``({variable: ('param'|'field', target)}, spec_static)`` with every
    target checked against ``circuit``; ``spec_static`` is False when a
    variable moves a value the kernel observables read."""
    from . import Circuit
    if not isinstance(circuit, Circuit):
        raise TypeError(f"Evaluator(circuit=...) takes a pulsim.Circuit "
                        f"(builder.build()), got {type(circuit).__name__}")
    devs = {d["name"]: d for d in circuit.devices()}
    kinds = {c["name"]: c["kind"] for c in circuit.components()}
    params = {pr.name for pr in circuit.parameters()}
    out, moved = {}, set()
    for var, t in fields.items():
        t = str(t)
        if "." in t:
            circuit.get_value(t)   # ValueError naming device and field
            out[str(var)] = ("field", t)
            moved.add(t.rsplit(".", 1)[0])
        elif t in params:
            out[str(var)] = ("param", t)
            moved.update(n for n, d in devs.items() if t in d["params"].values())
        else:
            raise ValueError(
                f"Evaluator: fields[{var!r}] = {t!r} is neither a parameter "
                f"of the circuit ({sorted(params)}) nor '<component>.<field>'")
    return out, all(kinds.get(n) in _SPEC_FREE_KINDS for n in moved)


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


# simulate() keywords an engine only WARNS it ignores (_dsed_dispatch.
# _validate_engine_kwargs, simulate's integrator note): an Evaluator option
# among them is refused by name instead of dropped.
_IGNORED_BY = {
    "pwl": ("rtol", "atol", "dt_init", "integrator", "stiffness_threshold",
            "h_bdf2"),
    "trbdf2": ("integrator", "stiffness_threshold", "h_bdf2"),
}


def _refuse_ignored_options(options, engine):
    bad = sorted(k for k in _IGNORED_BY.get(engine, ())
                 if options.get(k) is not None)
    if bad:
        raise unsupported(
            "design.evaluate", "options",
            f"Evaluator options {bad}: engine {engine!r} ignores them "
            "(refused, never dropped). Remove them or pick an engine that "
            "honours them.")


def _engine(fn):
    """Run an engine call; a kernel non-convergence (a RuntimeError,
    ``SimulationAborted`` included) becomes ``failed``. Refusals
    (``UnsupportedError``, a ValueError) and cancellation pass."""
    try:
        return fn()
    except (Cancelled, NotImplementedError, RecursionError):
        raise      # a stub, a refusal (_UnsupportedNotImplemented), a bug
    except RuntimeError as e:
        if type(e).__name__ == "_CxxCancelled":
            raise
        raise _Failed(f"{type(e).__name__}: {e}") from e


# |P_in - sum P - dE/T| / P_in, an order above the worst record measured
# (DECISIONS 22): kernel period 3.9e-12 (DCM buck); a dsed run sampled
# n times per period 1.3e-9 (n = 2000), 1.3e-7 (200), 2.1e-6 (50) -- the
# rows integrate products of per-step lines, O(1/n^2) -- so 1e-8 (2000/n)^2;
# time-stepping engines 3.9e-2 (pwl at T/25 with sub-step correction)
_ENERGY_RTOL = {"exact": 1e-8, "sampled": 1e-8, "stepped": 1e-1}


def _stored_power(run, comps, a, z):
    """``dE/T`` of the inductors and capacitors from ``a`` to ``z`` of a
    sampled window (a transient route ends a settle tolerance short of
    the orbit; a kernel period is the orbit: 0)."""
    from ._result_views import states_as_array
    t = np.asarray(run.times, dtype=float)
    st = None
    e = 0.0
    for n, c in comps.items():
        if c["kind"] == "inductor":
            y, k = np.asarray(run.i(n), dtype=float), float(
                c["params"]["L_henries"])
        elif c["kind"] == "capacitor":
            st = states_as_array(run) if st is None else st
            y, k = _pair_col(c["nodes"])(st), float(c["params"]["C_farads"])
        else:
            continue
        e += 0.5 * k * (float(np.interp(z, t, y)) ** 2
                        - float(np.interp(a, t, y)) ** 2)
    return e / (z - a)


def _stored_power_kernel(kp, comps):
    """``dE/T`` over a kernel period from its end values: the shooting
    stops at a scaled residual, so ``x(T)`` is ``x*`` only to that."""
    def at(key, end):
        return float((kp.y_end if end else kp.y_start)[kp._j(key)])

    def vc(nodes, end):
        return _psum(sg * at(("node#", int(k)), end)
                     for k, sg in zip(nodes, (1.0, -1.0)) if int(k) >= 0)
    e = 0.0
    for n, c in comps.items():
        if c["kind"] == "inductor":
            k, y0, y1 = float(c["params"]["L_henries"]), at(("i", n), False), \
                at(("i", n), True)
        elif c["kind"] == "capacitor":
            k, y0, y1 = float(c["params"]["C_farads"]), vc(c["nodes"], False), \
                vc(c["nodes"], True)
        else:
            continue
        e += 0.5 * k * (y1 * y1 - y0 * y0)
    return e / kp.T


def _energy_balance(comps, src, rows, losses, p_in, counters, p_stored,
                    route, n_samples=2000):
    """``P_in = P_out + sum P_loss`` over the steady period, from the
    same integrals as the KPIs: the sources' delivered power against
    every device's dissipated ``P_avg`` (loads included) and the power
    stored in L and C over the window (``p_stored``; 0 on the orbit).
    Recorded as
    ``counters['energy_residual']`` (relative to ``P_in``) with
    ``counters['p_table']``, the datasheet / table losses (switching,
    recovery, core) that are outside the circuit's own balance. A
    residual above ``_ENERGY_RTOL[route]`` fails the design: the KPIs disagree
    with each other. Not formed (no counter) when a source of power is
    outside the measurement (an unnamed source, a current source)."""
    if p_in is None or not p_in > 0:
        return
    if any(c["kind"] not in ("resistor", "inductor", "capacitor", "switch",
                             "diode", "voltage_source") for c in comps.values()) \
            or any(c["kind"] == "voltage_source" and n not in src
                   for n, c in comps.items()):
        return
    p_dis = _psum(float(r["P_avg"]) for r in rows.values() if "P_avg" in r)
    res = (p_in - p_dis - p_stored) / p_in
    counters["energy_residual"] = res
    counters["p_table"] = _psum(v for c in losses.values()
                                for k, v in c.items()
                                if k in ("switching", "recovery", "core"))
    tol = _ENERGY_RTOL[route] * (max(1.0, (2000.0 / n_samples) ** 2)
                                 if route == "sampled" else 1.0)
    if not abs(res) <= tol:
        raise _Failed(
            f"energy_balance: P_in = {p_in:.9g} W but the devices dissipate "
            f"{p_dis:.9g} W and store {p_stored:.3g} W (relative residual "
            f"{res:.3e}, tolerance {tol:.0e} on the {route} route): the "
            "steady period's integrals disagree")


# a plain period from x* returns within rounding (measured: 2.2e-16 CCM
# buck, 6.9e-17 DCM buck, 2.0e-16 boost, 4.0e-16 buck Pareto); the
# shooting stops at a scaled residual of 1e-9, one order above that
_PERIODIC_RTOL = 1e-8

_BASES: dict = {}   # value hash -> base Circuit, per worker process


def _evaluate_chunk(args):
    """Process job: ``(records of a chunk of designs, callbacks)``."""
    ev, designs = args
    recs = [ev.evaluate(d) for d in designs]
    return recs, (("python",) if any(r.counters.get("python_callbacks")
                                     for r in recs) else ())


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
