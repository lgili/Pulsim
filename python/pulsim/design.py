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

# Messages of periodic_steady_state that are a non-convergence of the
# design, not a configuration error (those are re-raised).
_PSS_FAILURES = ("no periodic orbit found", "Newton made no progress",
                 "I - M is singular", "the orbit is not isolated",
                 "grazing event", "a period run failed",
                 "the verification period moved")


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
    return FrozenDict({k: _fz(v) if isinstance(v, Mapping) else v
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
    """What a user KPI sees: one steady period of the design."""

    result: Any
    builder: Any
    design: Mapping[str, Any]
    t_from: float
    t_to: float
    seed: int

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

def _canon_value(v):
    if isinstance(v, (bool, str)) or v is None:
        return v
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        return float(v).hex()      # exact, and NaN/inf are representable
    if isinstance(v, Mapping):
        return {str(k): _canon_value(x) for k, x in sorted(v.items())}
    if isinstance(v, (tuple, list)):
        return [_canon_value(x) for x in v]
    if callable(v):
        return f"{getattr(v, '__module__', '?')}.{getattr(v, '__qualname__', repr(v))}"
    raise TypeError(
        f"design values must be numbers, strings or booleans (or "
        f"mappings / sequences of them), got {type(v).__name__}: {v!r}")


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
        ``'auto'`` (default), ``'dsed'``, ``'trbdf2'`` or ``'pwl'``:
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
                 loss_options: Optional[Mapping[str, Any]] = None):
        if steady.method not in ("pss", "transient", "auto"):
            raise ValueError(f"SteadySpec.method must be 'pss', 'transient' "
                             f"or 'auto', got {steady.method!r}")
        if engine not in ("auto", "dsed", "trbdf2", "pwl"):
            raise ValueError(f"engine must be 'auto', 'dsed', 'trbdf2' or "
                             f"'pwl', got {engine!r}")
        if steady.method == "pss" and engine not in ("auto", "dsed"):
            raise ValueError(
                f"SteadySpec(method='pss') runs on the dsed exact stepper; "
                f"engine={engine!r} contradicts it (use method='transient').")
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
            loss_options=self.loss_options)

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

        def rec(status, kpis=(), losses=(), t_j=(), violations=(),
                reason=None, refusal=None):
            return EvaluationRecord(
                status=status, kpis=_fz(dict(kpis)), losses=_fz(dict(losses)),
                t_j=_fz(dict(t_j)), violations=_fz(dict(violations)),
                reason=reason, refusal=refusal, counters=_fz(counters),
                wall_s=time.perf_counter() - t0, **base)

        try:
            kpis, losses, t_j = self._evaluate(design, did, counters)
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
        if len(out) == 3 and all(r.status == "refused" for r in out) \
                and len(refs) == 1:
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

    def _evaluate(self, design, did, counters):
        T = self._period_of(design)
        b, sf = self._build(design)
        run, a, z = self._steady(b, sf, T, counters)
        rows, losses = self._loss_rows(b, run, a, z)
        kpis = self._kpis(b, run, a, z, design, did, rows, losses, counters)
        return kpis, losses, {}

    def _steady(self, b, sf, T, counters):
        """``(result, t_from, t_to)`` of one steady period."""
        from ._refusal import UnsupportedError
        method = self.steady.method
        if method == "auto":
            if self.engine not in ("auto", "dsed"):
                method = "transient"
                counters["steady_route_reason"] = (
                    f"engine={self.engine!r}: pss runs on dsed only")
            else:
                try:
                    _pss_capability(b, T, sf)
                    method = "pss"
                except UnsupportedError as e:
                    # A documented route, decided BEFORE any run: the
                    # circuit is outside the exact stepper's coverage.
                    method = "transient"
                    counters["steady_route_reason"] = (
                        f"pss refused: {e.feature or e.kind or 'circuit'}"
                        f"{' (' + e.device + ')' if e.device else ''}")
        counters["steady_method"] = method
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
            self._count(run, counters)
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
            return run, (n - 1) * T, n * T
        # dsed: the settle run has no step cap; re-run the last period
        # from its true state with the KPI sampling.
        x0 = np.asarray(run.states_reduced, dtype=float)[-1]
        run2 = _engine(lambda: simulate(
            b, t_end=T, dt=T / self.samples_per_period, engine="dsed",
            switch_fn=sf, initial_state=x0,
            **{k: v for k, v in kw.items() if k != "dt"}))
        self._count(run2, counters)
        return run2, 0.0, T

    def _pss(self, b, sf, T, counters):
        from . import simulate
        from .periodic_steady_state import periodic_steady_state
        counters["route"] = "pss/dsed"
        try:
            r = _engine(lambda: periodic_steady_state(
                b, period=T, switch_fn=sf,
                max_period_runs=self.steady.max_period_runs))
        except ValueError as e:
            if _is_pss_failure(e):
                raise _Failed(str(e)) from e
            raise
        counters["pss_period_runs"] = int(r.n_period_runs)
        counters["engine_steps"] = int(r.n_engine_steps)
        counters["simulations"] += int(r.n_period_runs)
        run = _engine(lambda: simulate(
            b, t_end=T, dt=T / self.samples_per_period, engine="dsed",
            switch_fn=sf, initial_state=r.x0, rtol=1e-11, atol=1e-14))
        self._count(run, counters)
        return run, 0.0, T

    @staticmethod
    def _count(run, counters):
        counters["simulations"] += 1
        n = getattr(run, "n_accept", None)
        if n is None:
            n = run.num_steps()
        counters["engine_steps"] = counters.get("engine_steps", 0) + int(n)
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
        from ._refusal import unsupported
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
            col = int(b.pool.branch_var_id_for_source(c["branch_id"], b.graph))
            t, i = _with_right_limits(run, lambda st, col=col: st[:, col])
            # MNA branch current flows + -> - inside the source.
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
            t, v = _with_right_limits(run, _node_col(b, self.output))
            kpis["v_out"] = _mean(t, v, a, z)
            kpis["ripple_v"] = _p2p(t, v, a, z)
        for n, c in comps.items():
            if c["kind"] == "inductor":
                kpis[f"ripple_i.{n}"] = _p2p(
                    np.asarray(run.times, float), np.asarray(run.i(n), float),
                    a, z)
            elif c["kind"] in _SEMIS and n in rows:
                r = rows[n]
                kpis[f"i_peak.{n}"] = float(r["i_peak"])
                kpis[f"i_rms.{n}"] = float(r["i_rms"])
                t, v = _with_right_limits(run, _pair_col(c["nodes"]))
                _, vw = _window(t, v, a, z)
                kpis[f"v_peak.{n}"] = float(np.abs(vw).max())
        if self.kpis:
            per = SteadyPeriod(result=run, builder=b, design=design,
                               t_from=a, t_to=z, seed=int(did, 16) % 2**32)
            for k in self.kpis:
                kpis[k.name] = float(k.fn(per))
        return kpis


def _pss_capability(b, T, sf):
    """Raise the UnsupportedError ``periodic_steady_state`` would raise
    for this circuit, before running anything (its applicability
    checks of the circuit and the schedule)."""
    from .periodic_steady_state import _Solver
    _Solver(b, T, sf)


def _true_state(b, run):
    """``(t, X)``: inductor currents and capacitor voltages per sample
    (continuous across events, so interpolation is exact at them)."""
    cols = []
    st = None
    for c in b.components():
        if c["kind"] == "inductor":
            cols.append(np.asarray(run.i(c["name"]), dtype=float))
        elif c["kind"] == "capacitor":
            if st is None:
                from ._result_views import states_as_array
                st = states_as_array(run)
            cols.append(_pair_col(c["nodes"])(st))
    t = np.asarray(run.times, dtype=float)
    return t, (np.stack(cols, axis=1) if cols else np.zeros((t.size, 0)))


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


def _is_pss_failure(e) -> bool:
    from ._refusal import UnsupportedError
    return (not isinstance(e, UnsupportedError)
            and str(e).startswith("periodic_steady_state: ")
            and any(m in str(e) for m in _PSS_FAILURES))


def _engine(fn):
    """Run an engine call; a kernel non-convergence (a RuntimeError,
    ``SimulationAborted`` included) becomes ``failed``. Refusals
    (``UnsupportedError``, a ValueError) and cancellation pass."""
    from . import Cancelled
    try:
        return fn()
    except Cancelled:
        raise
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
    ``penalty`` in every component for ``refused`` / ``failed``. The
    status stays on the record: count it, do not read the penalty as
    a value."""
    if record.status in ("refused", "failed"):
        return np.full(len(names), float(penalty))
    return np.array([-record.kpis[n] if n in maximise else record.kpis[n]
                     for n in names], dtype=float)


def as_constraints(record: EvaluationRecord, names: Sequence[str], *,
                   penalty: float = 1e30) -> np.ndarray:
    """The violation vector (``<= 0`` feasible, NSGA-II convention), or
    ``penalty`` in every component for ``refused`` / ``failed``."""
    if record.status in ("refused", "failed"):
        return np.full(len(names), float(penalty))
    return np.array([record.violations[n] for n in names], dtype=float)
