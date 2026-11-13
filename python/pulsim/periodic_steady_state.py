"""Periodic steady state by Newton shooting on the true state.

A converter's periodic steady state is the fixed point of its
one-period map ``P``. This module solves

    G(x0) = P(x0) - x0 = 0,       (M - I).delta = -G(x0)

by Newton iteration, where ``x0`` is the TRUE state at the section
``t = 0`` (capacitor voltages and inductor currents, the reduced
state of the ``dsed`` engine) and ``M = dP/dx0`` is the monodromy
matrix, formed as the ordered product of the flow Jacobian of each
segment and the saltation matrix of each event:

    M = Phi_K . S_K . ... . S_1 . Phi_0,     Phi_k = exp(A_k h_k)
    S = I + (f+ - f-) . grad(g)^T / (grad(g)^T . f- + g_t)

Every map evaluation is one period of the ``dsed`` exact stepper
with its events located in time, so the commutation pattern is free
to change between iterations (a DCM buck started from zero passes
through a CCM pattern on its way to the orbit). ``M`` is the
derivative of the map as computed, so Newton converges
quadratically: 5 map evaluations from the zero state on the audit's
DCM buck, against thousands of periods of transient.

Slice 1 of ``add-periodic-steady-state-shooting``: it runs on
today's ``dsed`` exact stepper and therefore covers what that
stepper covers — linear R, L, C, controlled switches, PWL diodes,
DC and sine sources, gate schedules that can name their edges
(``make_pwm_switch_fn`` & co., ``NativePwm2Switch``,
``NativeMultiMaskPwm``, or a callable with ``next_edge_after``).
Everything else is REFUSED by name, never approximated: the fixed
and TR-BDF2 engines, nonlinear and stateful devices,
comparator-driven gates and controller states wait for the
Phase-2/3 changes the proposal lists.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

__all__ = [
    "periodic_steady_state",
    "PeriodicSteadyStateResult",
    "SampledModel",
]

_RTOL = 1e-9            # D4 scaled-norm relative tolerance
_ARMIJO = 1e-4
_N_LAMBDA = 9           # lambda in {1, 1/2, ..., 2^-8}
_MARGINAL = 1e-10       # | |mu| - 1 | below this: not an isolated orbit
_BRANCH_RE = re.compile(r"\(branch #(\d+)\)")
_DIODE_ON, _DIODE_OFF = 1, 2


def _err(msg: str) -> ValueError:
    return ValueError("periodic_steady_state: " + msg)


def _unsup(feature: str, msg: str, *, device: str = "",
           kind: str = "") -> ValueError:
    """A capability refusal of this analysis (a ValueError subclass)."""
    from ._refusal import unsupported
    return unsupported("periodic_steady_state", feature,
                       "periodic_steady_state: " + msg,
                       device=device, kind=kind)


# ----------------------------------------------------------------------
# Result types
# ----------------------------------------------------------------------

@dataclass
class SampledModel:
    """``x[k+1] = A.x[k] + B.u[k]``, ``y[k] = C.x[k] + D.u[k]``
    around the orbit, sampled at the section once per period."""

    A: np.ndarray
    B: np.ndarray
    C: np.ndarray
    D: np.ndarray
    inputs: list
    outputs: list
    period: float

    def response(self, frequencies) -> np.ndarray:
        """``H(z) = C (zI - A)^-1 B + D`` at ``z = exp(j 2 pi f T)``,
        shape ``(len(frequencies), n_outputs, n_inputs)``. Valid
        below the sampling Nyquist ``1/(2T)``."""
        f = np.atleast_1d(np.asarray(frequencies, dtype=float))
        n = self.A.shape[0]
        out = np.empty((f.size, self.C.shape[0], self.B.shape[1]),
                       dtype=complex)
        for k, fk in enumerate(f):
            z = np.exp(2j * np.pi * fk * self.period)
            out[k] = self.C @ np.linalg.solve(z * np.eye(n) - self.A,
                                              self.B) + self.D
        return out


@dataclass
class PeriodicSteadyStateResult:
    """The periodic orbit, its stability and what it cost."""

    #: The orbit's true state at the section (t = 0, the start of a
    #: period), in the order of ``state_labels``. Resume with
    #: ``simulate(..., engine='dsed', initial_state=x0)``.
    x0: np.ndarray
    state_labels: list
    #: One period of waveforms from ``x0`` (a ``simulate`` result).
    orbit: Any
    #: ``M = dP/dx0`` on the true state.
    monodromy: np.ndarray
    #: Floquet multipliers (eigenvalues of ``M``).
    multipliers: np.ndarray
    #: ``max|mu| < 1``. An unstable orbit is returned, not refused.
    stable: bool
    #: ``(index, mu, kind)`` for every multiplier outside the unit
    #: circle; kind is ``flip (period-doubling)``, ``fold`` or
    #: ``complex pair (Neimark-Sacker)``.
    unstable_multipliers: list
    #: ``(event name, instant / T)`` at the orbit.
    pattern: list
    #: ``max|P(x0) - x0|`` of the verification period.
    residual: float
    #: ``max|P(x) - x|`` of every accepted Newton iterate.
    residual_history: list
    #: The event names of each of those iterates' periods (the
    #: pattern is free to change between iterations).
    pattern_history: list
    #: One-period runs, verification included (the cost meter).
    n_period_runs: int
    #: Engine steps over all those runs.
    n_engine_steps: int
    period: float
    _ctx: Any = field(default=None, repr=False)
    #: ``{parameter: dx*/dp}`` when ``sensitivities=`` was given
    #: (``(I - M)^-1 dP/dp``, in ``state_labels`` order).
    dx0: dict = field(default_factory=dict)
    #: ``{parameter: dP/dp}`` at ``x0`` (one sensitivity period).
    dPdp: dict = field(default_factory=dict)
    #: The orbit's :class:`pulsim.sensitivity.SensitivityResult`
    #: (``S(0) = dx*/dp``) when ``sensitivities=`` was given.
    sensitivity: Any = field(default=None, repr=False)

    def kpi(self, signal, kind="avg", *, with_grad=True):
        """Orbit KPI over one period with its exact gradient (the second
        sensitivity period of D5); needs ``sensitivities=``."""
        if self.sensitivity is None:
            raise ValueError(
                "kpi: solve with periodic_steady_state(..., "
                "sensitivities=[...]) to get KPI gradients.")
        return self.sensitivity.kpi(signal, kind, with_grad=with_grad)

    @property
    def snapshot(self) -> np.ndarray:
        """Alias of ``x0`` (the resumable state)."""
        return self.x0

    @property
    def exact(self):
        """``orbit.exact``: the exact stepper's per-mask accuracy report."""
        return getattr(self.orbit, "exact", None)

    @property
    def floquet_radius(self) -> float:
        return float(np.abs(self.multipliers).max()) \
            if self.multipliers.size else 0.0

    def sampled_model(self, inputs: Sequence = (),
                      outputs: Sequence = ()) -> SampledModel:
        """Sampled-data small-signal model around the orbit.

        ``inputs``: ``("duty", switch_name)`` — the switch's turn-off
        edges move by ``T`` per unit (trailing-edge modulation) — or
        ``("source", dc_source_name)`` — the source's value. ``B`` is
        ``dP/du`` from the same propagators and saltation matrices as
        ``M``. ``outputs``: a state label, an inductor name or a node
        name, sampled at the section. ``A = M``."""
        return self._ctx.sampled_model(self, list(inputs),
                                       list(outputs))


# ----------------------------------------------------------------------
# Propagators
# ----------------------------------------------------------------------

def _phi_gamma(A: np.ndarray, h: float):
    """``exp(A h)`` and ``int_0^h exp(A s) ds`` in the modal form
    the dsed stepper uses; a Taylor scaling-and-squaring of the
    augmented block when the eigenbasis is unusable."""
    n = A.shape[0]
    lam, V = np.linalg.eig(A)
    if np.linalg.cond(V) < 1e10:
        Vi = np.linalg.inv(V)
        s = lam * h
        e = np.exp(s)
        small = np.abs(s) < 1e-6
        safe = np.where(small, 1.0, s)
        phi1 = np.where(small, 1 + s * (0.5 + s * (1 / 6 + s / 24)),
                        (e - 1) / safe)
        E = ((V * e) @ Vi).real
        G = ((V * (h * phi1)) @ Vi).real
        return E, G
    aug = np.zeros((2 * n, 2 * n))
    aug[:n, :n] = A * h
    aug[:n, n:] = np.eye(n) * h
    nrm = np.abs(aug).sum(axis=1).max()
    sq = max(0, int(math.ceil(math.log2(nrm))) + 1) if nrm > 0.5 else 0
    X = aug / 2.0 ** sq
    term = np.eye(2 * n)
    out = np.eye(2 * n)
    for k in range(1, 20):
        term = term @ X / k
        out = out + term
    for _ in range(sq):
        out = out @ out
    return out[:n, :n], out[:n, n:]


# ----------------------------------------------------------------------
# Applicability
# ----------------------------------------------------------------------

class _ConstSchedule:
    """All controlled switches open; diode bits belong to the
    scheduler. For circuits with diodes only."""

    def __init__(self, n):
        from . import SwitchStateMask
        self._m = SwitchStateMask(n)

    def __call__(self, t):
        return self._m

    def next_edge_after(self, t):
        return math.inf


def _schedule_edges(sf, t_lo, t_hi):
    out, t = [], t_lo
    while True:
        e = float(sf.next_edge_after(t))
        if not (e <= t_hi) or not math.isfinite(e):
            return out
        if e <= t:
            raise _err(
                f"the gate schedule's next_edge_after({t!r}) returned "
                f"{e!r}, not a later instant — its edges cannot be "
                "located.")
        out.append(e)
        if len(out) > 100000:
            raise _err("the gate schedule has more than 1e5 edges in "
                       "one period.")
        t = e


def _check_schedule(sf, T):
    from . import SwitchSchedule
    if isinstance(sf, SwitchSchedule) and not sf.has_edges:
        raise _unsup(
            "switch_fn", "the gate schedule cannot name its edges (a "
            "make_combined_switch_fn with a plain Python callable in "
            "it). Its edges would be found by sampling, which is not "
            "reproducible to rounding between map evaluations, so the "
            "one-period map would not be differentiable. Build it "
            "from the C++ helpers only.")
    if not hasattr(sf, "next_edge_after"):
        raise _unsup(
            "switch_fn", "switch_fn is an opaque Python callable (no "
            "next_edge_after(t)). Its edges would be found by "
            "sampling, which is not reproducible to rounding between "
            "map evaluations. Use pulsim.make_pwm_switch_fn & co., "
            "NativePwm2Switch / NativeMultiMaskPwm, or give the "
            "callable a next_edge_after(t) method.")
    e1 = _schedule_edges(sf, 0.0, T)
    e2 = _schedule_edges(sf, T, 2 * T)
    tol = 1e-12 * T * 4
    same = (len(e1) == len(e2)
            and all(abs((b - T) - a) <= tol for a, b in zip(e1, e2)))
    if same:
        grid = [0.0] + e1 + [T]
        for a, b in zip(grid[:-1], grid[1:]):
            tm = 0.5 * (a + b)
            if not (sf(tm) == sf(tm + T)):
                same = False
                break
    if not same:
        raise _err(
            f"the gate schedule is not periodic with period={T!r}: "
            f"its edges in (0, T] are {e1[:6]}{'...' if len(e1) > 6 else ''} "
            f"and in (T, 2T] minus T are "
            f"{[b - T for b in e2[:6]]}{'...' if len(e2) > 6 else ''}. "
            "The period must be a multiple of every schedule's "
            "period.")


def _check_circuit(builder, T, comps):
    if builder.pool.has_nonlinear_devices():
        bad = [c for c in comps if c["kind"] in (
            "nonlinear_diode", "mosfet_level1", "igbt_level1",
            "saturable_inductor", "nonlinear_capacitor",
            "shockley_diode", "lauritzen_diode", "pmsm_mna", "power_mosfet",
            "unknown")]
        names = [c["name"] for c in bad]
        raise _unsup(
            "", device=names[0] if names else "",
            kind=(builder.pool.kind_of(bad[0]["branch_id"]).name
                  if bad else ""), msg=
            "the circuit has nonlinear or stateful devices "
            f"({', '.join(names) or 'see builder.components()'}), "
            "whose state the exact-path solver cannot read or set "
            "(slice 1 covers linear R, L, C, controlled sources, "
            "switches, PWL diodes, DC, sine and current-waveform "
            "sources). They need the unified device "
            "interface; use a long transient meanwhile.")
    for c in comps:
        kind = c["kind"]
        if kind in ("pwm_voltage_source", "pulse_voltage_source"):
            raise _unsup(
                "", device=c["name"],
                kind=builder.pool.kind_of(c["branch_id"]).name, msg=
                f"source {c['name']!r} is a {kind}: its edges are not "
                "events of the exact stepper, so the one-period map "
                "would not be computed exactly. Model the gate with a "
                "switch and a schedule, or use a transient.")
        why = _not_t_periodic(kind, c["params"], T)
        if why:
            raise _unsup(
                "period", device=c["name"],
                kind=builder.pool.kind_of(c["branch_id"]).name, msg=
                f"source {c['name']!r} ({kind}) is not periodic with "
                f"period={T!r}: {why}, so the circuit has no T-periodic "
                "steady state. Use the common period of all sources "
                "and schedules.")


def _scaled(v, x, scale, rtol):
    """``max_i |v_i| / (rtol (scale_i + |x_i|))``, the solver's norm."""
    den = rtol * (scale + np.abs(x))
    with np.errstate(divide="ignore", invalid="ignore"):
        q = np.where(v == 0.0, 0.0, np.abs(v) / np.where(
            den > 0, den, 1e-300))
    return float(q.max())


def _divides(P, T):
    """Is ``P`` a period of the section, i.e. ``T/P`` a positive integer?"""
    if not (P > 0 and math.isfinite(P)):
        return False
    k = round(T / P)
    return k >= 1 and abs(T / P - k) <= 1e-12 * max(1.0, T / P)


def _not_t_periodic(kind, prm, T):
    """Why a waveform source is not T-periodic on [0, inf), or ''."""
    if kind in ("sine_voltage_source", "sine_current_source"):
        f = float(prm.get("frequency", float("nan")))
        if f == 0.0:
            return ""      # a DC source
        if not _divides(1.0 / f if f > 0 else float("nan"), T):
            return f"at {f:g} Hz, f.T = {f * T:.12g} is not a positive integer"
    elif kind == "pulse_current_source":
        P, t0 = float(prm["period"]), float(prm["t_start"])
        d = float(prm["rise_time"]) + float(prm["pulse_width"]) + float(
            prm["fall_time"])
        if not (P > 0):
            return "a one-shot pulse (period 0)"
        if not _divides(P, T):
            return f"its period {P!r} does not divide T"
        if not (0.0 <= t0 and t0 + d <= P):
            # before t_start it sits at i_initial; a later period does not
            return (f"its first pulse (t_start {t0!r} + rise, width, fall "
                    f"{d!r}) does not end within its first period")
    elif kind == "pwl_current_source":
        t = prm["t"]
        if not prm["periodic"]:
            return "a table with periodic=False"
        if not _divides(t[-1] - t[0], T):
            return f"its table period {t[-1] - t[0]!r} does not divide T"
        if t[0] > 0.0:
            return f"its table starts at t = {t[0]!r} > 0 (constant before)"
    return ""


def _labels(builder, cache, mask, comps):
    _A, _b, rows, is_cap, _B = cache.compute_lti_state_space(mask)
    graph = builder.graph
    labels, devices = [], []
    for r, cap in zip(rows, is_cap):
        r = int(r)
        if cap:
            labels.append(f"v({graph.node_name(r)})")
            devices.append([c["name"] for c in comps
                            if c["kind"] == "capacitor"
                            and r in tuple(c["nodes"])])
        else:
            name = None
            for c in comps:
                if c["kind"] != "inductor":
                    continue
                try:
                    if int(builder.pool.branch_var_id_for_inductor(
                            c["branch_id"], graph)) == r:
                        name = c["name"]
                        break
                except Exception:  # noqa: BLE001 — not an inductor row
                    continue
            labels.append(f"i({name})" if name else f"x[{r}]")
            devices.append([name] if name else [])
    return labels, devices


# ----------------------------------------------------------------------
# One period
# ----------------------------------------------------------------------

@dataclass
class _Traj:
    x0: np.ndarray
    x_end: np.ndarray
    seg_masks: list           # mask of each segment
    bounds: list              # [0, t1, ..., T]
    events: list              # dicts: t, name, type, old, new, x
    scale: np.ndarray         # max |x_i(t)| over the period
    n_steps: int


class _Solver:
    def __init__(self, builder, period, switch_fn):
        from . import PwlStateSpaceCache, SwitchStateMask
        from . import _pulsim
        self.builder = builder
        self.T = float(period)
        self.comps = list(builder.components())
        _check_circuit(builder, self.T, self.comps)
        n_sw = builder.graph.num_switches
        _n, _nd, controlled = _pulsim._switch_census(builder.graph,
                                                      builder.pool)
        if switch_fn is None:
            if len(controlled):
                raise _err(
                    f"the circuit has {len(controlled)} controlled "
                    "switch(es) and no switch_fn: there is no gate "
                    "schedule to be periodic in.")
            switch_fn = _ConstSchedule(n_sw)
        _check_schedule(switch_fn, self.T)
        self.sf = switch_fn
        self._SSM = SwitchStateMask
        self.n_sw = n_sw
        self.cache = PwlStateSpaceCache(builder.graph, builder.pool)
        self.cache.build_lazy(self.T)
        self.ad = _pulsim._NativeCircuitBuilderAdapter(builder,
                                                       self.cache, None)
        self._run_native = _pulsim.run_ped_native_builder
        m0 = switch_fn(0.0)
        try:
            self.ad.set_mask(m0)
            n = self.ad.n_state
            self.labels, self.label_devices = _labels(
                builder, self.cache, m0, self.comps)
        except (RuntimeError, NotImplementedError) as e:
            raise _err(
                "the exact stepper cannot extract this circuit's "
                f"state space ({str(e).splitlines()[0]}).") from e
        if n == 0:
            raise _err(
                "the circuit has no capacitors or inductors, so it "
                "has no periodic state: its solution is algebraic at "
                "every instant.")
        self.n = n
        self.dynamic = bool(self.ad.has_dynamic_sources)
        self._A = {}
        self.n_runs = 0
        self.n_steps = 0

    # -- per-mask data ---------------------------------------------------
    def A(self, mask):
        key = str(mask)
        a = self._A.get(key)
        if a is None:
            self.ad.set_mask(mask)
            a = np.asarray(self.ad.A_matrix(), dtype=float)
            self._A[key] = a
        return a

    def f(self, mask, t, x):
        self.ad.set_mask(mask)
        return np.asarray(self.ad.rhs(t, x), dtype=float)

    # -- the map -----------------------------------------------------------
    def run(self, x, sf=None):
        x = np.asarray(x, dtype=float).copy()
        sf = self.sf if sf is None else sf
        self.n_runs += 1
        try:
            r = self._run_native(
                self.ad, sf, x, self.T, rtol=1e-11, atol=1e-14,
                dt_init=self.T * 1e-6, dt_max=self.T, store_every=1,
                enable_event_projection=False)
        except RuntimeError as e:
            raise _err(f"a period run failed: {e}") from e
        times = np.asarray(r["times"], dtype=float)
        states = np.asarray(r["states"], dtype=float)
        self.n_steps += int(r["n_accept"])
        events, bounds = [], [0.0]
        log = list(r["event_log"])
        if log:
            seg_masks = [log[0]["old_mask"]]
        else:
            row = np.asarray(r["switch_states"])[0]
            m = self._SSM(self.n_sw)
            for i, bit in enumerate(row):
                m.set(i, bool(bit))
            seg_masks = [m]
        for e in log:
            k = int(np.argmin(np.abs(times - e["t"])))
            events.append(dict(t=float(e["t"]), name=str(e["name"]),
                               type=int(e["predicate_type"]),
                               old=e["old_mask"], new=e["new_mask"],
                               x=states[k].copy()))
            bounds.append(float(e["t"]))
            seg_masks.append(e["new_mask"])
        bounds.append(self.T)
        return _Traj(x0=x, x_end=states[-1].copy(), seg_masks=seg_masks,
                     bounds=bounds, events=events,
                     scale=np.abs(states).max(axis=0), n_steps=int(
                         r["n_accept"]))

    # -- events --------------------------------------------------------
    def _diode(self, name):
        m = _BRANCH_RE.search(name)
        if m:
            bid = int(m.group(1))
            for c in self.comps:
                if c["branch_id"] == bid and c["kind"] == "diode":
                    return c
        raise _err(
            f"event {name!r} is a state event whose predicate the "
            "solver cannot read, so its saltation matrix cannot be "
            "formed.")

    def _g(self, ad, ev, t, x):
        c = self._diode(ev["name"])
        a, k = (int(v) for v in c["nodes"])
        p = c["params"]
        V_f, V_th = float(p.get("V_f", 0.0)), float(p["V_th"])
        thr = V_f if ev["type"] == _DIODE_OFF else (
            V_th if V_f == 0.0 else max(V_th, V_f))
        ad.set_mask(ev["old"])
        return float(ad.node_pair_voltage(a, k, t, x)) - thr

    def _grad_g(self, ev):
        t, x = ev["t"], ev["x"]
        g0 = self._g(self.ad, ev, t, x)
        grad = np.empty(self.n)
        for j in range(self.n):
            xj = x.copy()
            xj[j] += 1.0           # affine in x: exact up to rounding
            grad[j] = self._g(self.ad, ev, t, xj) - g0
        g_t = 0.0
        if self.dynamic:
            dt = self.T * 1e-6
            g_t = (self._g(self.ad, ev, t + dt, x)
                   - self._g(self.ad, ev, t - dt, x)) / (2 * dt)
        return g0, grad, g_t

    def _saltation_terms(self, ev):
        g0, grad, g_t = self._grad_g(ev)
        fm = self.f(ev["old"], ev["t"], ev["x"])
        fp = self.f(ev["new"], ev["t"], ev["x"])
        den = float(grad @ fm) + g_t
        if abs(den) <= 1e-12 * (np.linalg.norm(grad)
                                 * np.linalg.norm(fm) + abs(g_t)):
            raise _err(
                f"grazing event: {ev['name']} at t = "
                f"{ev['t'] / self.T:.12g} T is tangent to its switching "
                "surface (grad(g).f = 0), where the period map is not "
                "differentiable.")
        return g0, grad, den, fm, fp

    def pattern(self, traj):
        # The stepper brackets each event to max(4u|t|, 1e-12 h); a Newton
        # step on the instant moved the DCM turn-off by 3e-22 s (none on
        # the boost), so the located instant is reported as it is.
        return [(ev["name"], ev["t"] / self.T) for ev in traj.events]

    # -- monodromy (augmented with inputs) ----------------------------
    def jacobian(self, traj, inputs=()):
        """``[[M, B], [0, I]]`` for the augmented state ``(x, u)``."""
        n, m = self.n, len(inputs)
        J = np.eye(n + m)
        for k, mask in enumerate(traj.seg_masks):
            if k > 0:
                J = self._event_jump(traj.events[k - 1], inputs) @ J
            h = traj.bounds[k + 1] - traj.bounds[k]
            E, G = _phi_gamma(self.A(mask), h)
            P = np.eye(n + m)
            P[:n, :n] = E
            for i, inp in enumerate(inputs):
                beta = inp.dbdu(mask, traj.bounds[k])
                if beta is not None:
                    P[:n, n + i] = G @ beta
            J = P @ J
        return J

    def _event_jump(self, ev, inputs):
        n, m = self.n, len(inputs)
        S = np.eye(n + m)
        if ev["type"] in (_DIODE_ON, _DIODE_OFF):
            g0, grad, den, fm, fp = self._saltation_terms(ev)
            gtil = np.zeros(n + m)
            gtil[:n] = grad
            for i, inp in enumerate(inputs):
                gtil[n + i] = inp.dgdu(ev)
            S[:n, :] += np.outer(fp - fm, gtil) / den
        else:
            for i, inp in enumerate(inputs):
                dtau = inp.dtau(ev)
                if dtau:
                    fm = self.f(ev["old"], ev["t"], ev["x"])
                    fp = self.f(ev["new"], ev["t"], ev["x"])
                    S[:n, n + i] += (fm - fp) * dtau
        return S

    # -- diagnostics ---------------------------------------------------
    def name_states(self, v, scale=None):
        v = np.abs(np.asarray(v))
        if scale is not None and np.all(scale > 0):
            v = v / scale       # compare in units of each state's swing
        idx = [i for i in np.argsort(-v) if v[i] >= 0.1 * v.max()]
        parts = []
        for i in idx:
            dev = self.label_devices[i]
            parts.append(self.labels[i]
                         + (f" [{', '.join(dev)}]" if dev else ""))
        return ", ".join(parts)

    def output_row(self, name, mask0):
        """``(C row, node)`` for an output sampled at the section."""
        if name in self.labels:
            row = np.zeros(self.n)
            row[self.labels.index(name)] = 1.0
            return row, None
        if f"i({name})" in self.labels:
            row = np.zeros(self.n)
            row[self.labels.index(f"i({name})")] = 1.0
            return row, None
        try:
            node = int(self.builder.node_id_of(name))
        except Exception:  # noqa: BLE001 — not a node either
            node = None
        if node is None or node < 0:
            raise _err(
                f"output {name!r} is not a state label ({self.labels}), "
                "an inductor or a node name.")
        self.ad.set_mask(mask0)
        z = np.zeros(self.n)
        v0 = float(self.ad.node_pair_voltage(node, -1, 0.0, z))
        row = np.empty(self.n)
        for j in range(self.n):
            e = z.copy()
            e[j] = 1.0
            row[j] = float(self.ad.node_pair_voltage(node, -1, 0.0, e)) - v0
        return row, node

    def sampled_model(self, res, inputs, outputs):
        traj = self.orbit_traj
        objs = [self._input(spec, traj) for spec in inputs]
        try:
            J = self.jacobian(traj, objs)
        finally:
            for o in objs:
                o.close()
        n = self.n
        C = np.zeros((len(outputs), n))
        D = np.zeros((len(outputs), len(objs)))
        for r, name in enumerate(outputs):
            C[r], node = self.output_row(name, traj.seg_masks[0])
            if node is not None:
                for i, o in enumerate(objs):
                    D[r, i] = o.dydu(node, traj)
        return SampledModel(A=J[:n, :n].copy(), B=J[:n, n:].copy(), C=C,
                            D=D, inputs=list(inputs),
                            outputs=list(outputs), period=self.T)

    def _input(self, spec, traj):
        kind, name = spec
        if kind == "duty":
            return _DutyInput(self, name)
        if kind == "source":
            return _SourceInput(self, name, traj)
        raise _err(f"input kind {kind!r}: use 'duty' or 'source'.")


class _DutyInput:
    """Trailing-edge duty of one controlled switch."""

    def __init__(self, s, name):
        self.s = s
        self.bit = int(s.builder.switch_index_of(name))
        self.name = name

    def dbdu(self, mask, t):
        return None

    def dgdu(self, ev):
        return 0.0

    def dtau(self, ev):
        if ev["type"] == 0 and ev["old"].get(self.bit) \
                and not ev["new"].get(self.bit):
            return self.s.T
        return 0.0

    def dydu(self, node, traj):
        return 0.0

    def close(self):
        pass


class _SourceInput:
    """Value of one DC voltage source (b and the recovery are affine
    in it: differences against a second extraction are exact)."""

    def __init__(self, s, name, traj):
        from . import PwlStateSpaceCache
        from . import _pulsim
        self.s = s
        c = [c for c in s.comps if c["name"] == name]
        if not c or c[0]["kind"] != "voltage_source":
            raise _err(f"input source {name!r} is not a DC voltage "
                       "source of this circuit.")
        self.name = name
        self.V = float(c[0]["params"]["V"])
        self.dV = max(1.0, abs(self.V))
        self._b, self._g, self._y = {}, {}, {}
        b = s.builder
        b.update_voltage_source_V(name, self.V + self.dV)
        try:
            cache = PwlStateSpaceCache(b.graph, b.pool)
            cache.build_lazy(s.T)
            ad = _pulsim._NativeCircuitBuilderAdapter(b, cache, None)
            for k, mask in enumerate(traj.seg_masks):
                t = traj.bounds[k]
                ad.set_mask(mask)
                b2 = np.asarray(ad.b_vector(t), dtype=float)
                s.ad.set_mask(mask)
                b1 = np.asarray(s.ad.b_vector(t), dtype=float)
                self._b[(str(mask), t)] = (b2 - b1) / self.dV
            for ev in traj.events:
                if ev["type"] in (_DIODE_ON, _DIODE_OFF):
                    self._g[id(ev)] = (s._g(ad, ev, ev["t"], ev["x"])
                                       - s._g(s.ad, ev, ev["t"], ev["x"])
                                       ) / self.dV
            self._ad2 = ad
            self._mask0 = traj.seg_masks[0]
            self._x0 = traj.x0
            self._ymemo = {}
            for c2 in s.comps:
                for node in c2["nodes"]:
                    node = int(node)
                    if node >= 0 and node not in self._ymemo:
                        ad.set_mask(self._mask0)
                        s.ad.set_mask(self._mask0)
                        self._ymemo[node] = (
                            float(ad.node_pair_voltage(node, -1, 0.0,
                                                       self._x0))
                            - float(s.ad.node_pair_voltage(
                                node, -1, 0.0, self._x0))) / self.dV
        finally:
            b.update_voltage_source_V(name, self.V)

    def dbdu(self, mask, t):
        return self._b[(str(mask), t)]

    def dgdu(self, ev):
        return self._g.get(id(ev), 0.0)

    def dtau(self, ev):
        return 0.0

    def dydu(self, node, traj):
        return self._ymemo.get(node, 0.0)

    def close(self):
        self._ad2 = None


# ----------------------------------------------------------------------
# Newton
# ----------------------------------------------------------------------

def _classify(mu):
    if abs(mu.imag) > 1e-12 * max(1.0, abs(mu)):
        return "complex pair (Neimark-Sacker)"
    return "flip (period-doubling)" if mu.real < 0 else "fold"


def periodic_steady_state(builder, *, period: float, switch_fn=None,
                          start_from=None, max_period_runs: int = 60,
                          n_relax: int = 5,
                          rtol: float = _RTOL,
                          sensitivities=None) -> PeriodicSteadyStateResult:
    """Periodic steady state of a switching circuit by Newton shooting
    on the true state, on the ``dsed`` exact stepper.

    Parameters
    ----------
    builder
        The circuit (linear R, L, C, switches, PWL diodes, DC and sine
        sources). Anything else is refused by name.
    period
        ``T``. Every gate schedule and waveform source (sine, pulse
        and table current sources included) must be periodic in it
        (their own periods divide it); one that is not is refused.
    switch_fn
        A gate schedule that can name its edges: ``make_pwm_switch_fn``
        & co., ``NativePwm2Switch``, ``NativeMultiMaskPwm`` or a
        callable with ``next_edge_after(t)``. Omit only for a circuit
        without controlled switches.
    start_from
        Initial guess for the state at the section: a vector in
        ``state_labels`` order or a previous result. Default zero.
    max_period_runs
        Budget of one-period runs; exceeding it raises with the
        residual history, the last pattern and the last multipliers.
    n_relax
        Periods of plain transient taken when no backtracking step is
        accepted (at most three such rounds).
    rtol
        Scaled-norm tolerance: stop when every
        ``|G_i| <= rtol (scale_i + |x_i|)``, ``scale_i`` the largest
        ``|x_i(t)|`` seen over the period runs, AND the Newton step
        meets the same bound (or one step has been taken from such a
        residual): with a slow mode the state error is about
        ``|G| / (1 - |mu|)``, not ``|G|``.

    sensitivities
        Parameter names (``"<component>.<field>"``: ``R``, ``L``, ``C``,
        ``V`` of linear elements and DC sources, ``"<switch>.duty"`` of a
        trailing-edge PWM). Sets ``dx0`` = ``(I - M)^-1 dP/dp`` and
        ``sensitivity`` (orbit KPI gradients through ``kpi``) at the
        cost of one more period run; see ``pulsim.sensitivity``.

    The section is ``t = 0``: for the zero-phase built-in PWM helpers
    that is the rising gate edge.
    """
    if not (period > 0 and math.isfinite(period)):
        raise _err(f"period must be a positive finite number, got "
                   f"{period!r}")
    s = _Solver(builder, period, switch_fn)
    n = s.n
    if start_from is None:
        x = np.zeros(n)
    elif isinstance(start_from, PeriodicSteadyStateResult):
        x = np.asarray(start_from.x0, dtype=float).copy()
    else:
        x = np.asarray(start_from, dtype=float).reshape(-1).copy()
    if x.shape != (n,):
        raise _err(f"start_from has {x.size} entries; the state has {n} "
                   f"({s.labels}).")

    scale = np.zeros(n)

    def norm(traj):
        nonlocal scale
        scale = np.maximum(scale, traj.scale)
        G = traj.x_end - traj.x0
        den = rtol * (scale + np.abs(traj.x0))
        with np.errstate(divide="ignore", invalid="ignore"):
            q = np.where(G == 0.0, 0.0, np.abs(G) / np.where(
                den > 0, den, 1e-300))
        return float(q.max())

    history: list = []
    patterns: list = []
    last = {"pattern": None, "mu": None}

    def budget(extra=""):
        if s.n_runs >= max_period_runs:
            raise _err(
                f"no periodic orbit found within max_period_runs="
                f"{max_period_runs}{extra}. Residual history "
                f"max|P(x)-x|: {[f'{h:.3e}' for h in history]}; last "
                f"pattern: {last['pattern']}; multipliers of the last "
                f"monodromy: {last['mu']}.")

    traj = s.run(x)
    relax_rounds = 0
    polished = False
    while True:
        r = norm(traj)
        history.append(float(np.abs(traj.x_end - traj.x0).max()))
        patterns.append([ev["name"] for ev in traj.events])
        M = s.jacobian(traj)
        mu = np.linalg.eigvals(M)
        last["mu"] = mu
        last["pattern"] = [(ev["name"], ev["t"] / s.T)
                           for ev in traj.events]
        IM = np.eye(n) - M
        U, sig, Vt = np.linalg.svd(IM)
        singular = sig[-1] <= 1e-10 * max(1.0, sig[0])
        G = traj.x_end - traj.x0
        if r <= 1.0:
            # A small residual is not a small state error: with a slow
            # mode (|mu| -> 1) the error is about residual/(1 - |mu|).
            # Stop when the Newton step itself is within tolerance, or
            # once one step has been taken from a converged residual
            # (exact on a linear map; with |mu| -> 1 the step is
            # rounding noise times 1/(1 - |mu|) and never gets
            # smaller). A marginal orbit is left to the multiplier
            # check below the loop.
            if singular or polished or _scaled(
                    np.linalg.solve(IM, G), traj.x0, scale, rtol) <= 1.0:
                break
        if singular:
            raise _err(
                "I - M is singular: a Floquet multiplier equals 1, so "
                "the orbit is not isolated (a floating capacitor, an "
                "inductor loop with no resistance, an integrator with "
                "no path to its input). States of the marginal "
                f"direction: {s.name_states(Vt[-1], scale)}.")
        delta = np.linalg.solve(IM, G)    # (M - I) delta = -G
        accepted = False
        lam = 1.0
        for _ in range(_N_LAMBDA):
            budget()
            tt = s.run(traj.x0 + lam * delta)
            r_new = norm(tt)
            if r_new <= (1.0 - _ARMIJO * lam) * norm(traj) or (
                    r <= 1.0 and r_new <= 1.0):
                polished = r <= 1.0
                traj = tt
                accepted = True
                break
            lam *= 0.5
        if accepted:
            continue
        relax_rounds += 1
        if relax_rounds > 3:
            raise _err(
                "Newton made no progress after three rounds of "
                f"{n_relax} relaxation periods. Residual history "
                f"max|P(x)-x|: {[f'{h:.3e}' for h in history]}; last "
                f"pattern: {last['pattern']}; multipliers: {last['mu']}.")
        for _ in range(n_relax):
            budget()
            traj = s.run(traj.x_end)

    # ---- the orbit ----
    x0 = traj.x0.copy()
    M = s.jacobian(traj)
    mu = np.linalg.eigvals(M)
    lam_, vec = np.linalg.eig(M)
    for i, m_ in enumerate(lam_):
        if abs(abs(m_) - 1.0) < _MARGINAL:
            raise _err(
                f"the orbit is not isolated: Floquet multiplier {m_:.12g} "
                "lies on the unit circle, so neighbouring solutions "
                "never decay to it (a lossless loop or an undamped "
                "integrator). States of the marginal direction: "
                f"{s.name_states(vec[:, i], scale)}.")
    budget(" (no run left for the verification period)")
    from . import simulate as _simulate
    orbit = _simulate(builder, t_end=s.T, dt=s.T, engine="dsed",
                      switch_fn=s.sf, initial_state=x0, rtol=1e-11,
                      atol=1e-14)
    s.n_runs += 1
    s.n_steps += int(getattr(orbit, "n_accept", 0))
    x_back = np.asarray(orbit.states_reduced[-1], dtype=float)
    residual = float(np.abs(x_back - x0).max())
    if residual > 1e3 * max(history[-1], rtol * float(
            (scale + np.abs(x0)).max())):
        raise _err(
            f"the verification period moved the orbit by {residual:.3e} "
            f"(the last Newton residual was {history[-1]:.3e}): the "
            "map is not reproducible between runs.")
    unstable = [(i, complex(m_), _classify(m_)) for i, m_ in
                enumerate(mu) if abs(m_) >= 1.0]
    res = PeriodicSteadyStateResult(
        x0=x0, state_labels=list(s.labels), orbit=orbit, monodromy=M,
        multipliers=mu, stable=not unstable,
        unstable_multipliers=unstable, pattern=s.pattern(traj),
        residual=residual, residual_history=history,
        pattern_history=patterns, n_period_runs=s.n_runs, n_engine_steps=s.n_steps,
        period=s.T, _ctx=s)
    s.orbit_traj = traj
    if sensitivities is not None:
        from .sensitivity import orbit_sensitivities
        orbit_sensitivities(res, s, sensitivities)
    return res
