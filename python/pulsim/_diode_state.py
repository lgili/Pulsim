"""Per-sample conduction state of every PWL switched diode in a result.

One source of truth for :meth:`SimulationResult.i`,
:func:`pulsim.device_loss_summary` and :func:`pulsim.device_thermal_summary`.

Why the voltage cannot say
--------------------------
``SwitchedDiode`` (``core/include/pulsim/models/switched_diode.hpp``) is a
pure conductance in both states — the stamp (``pwl/assemble.hpp``) is
``g_on`` or ``g_off`` between anode and cathode with NO voltage offset, so
a conducting diode drops ``i / g_on`` whatever ``V_th`` is. ``V_th`` only
enters ``decide_next_state`` (OFF -> ON iff ``v_D >= V_th``; ON stays iff
``g_on * v_D > 0``). A diode carrying 48 A at ``g_on = 1e3`` sits at 48 mV,
inside ``0 < v_D < V_th``, where BOTH states are self-consistent. The old
rule ``closed = v_D > V_th`` reported such a diode idle (0 A, 0 W).

What this module reads instead
------------------------------
The kernel records, with every stored sample, the switch / diode mask it
STAMPED for the solve that produced that sample
(``SimulationResult.switch_states``, ``core/include/pulsim/solver/
result.hpp``). The branch current is then exact by construction::

    i_D[k] = (g_on if switch_states[k, bit] else g_off) * v_D[k]

on every engine that writes the record: ``engine='pwl'`` (including the
event-iterated, sub-step-corrected and dt-retried steps — the record is the
LAST solve of the step), ``engine='auto'`` / ``'trbdf2'`` (an event landing
is a real trap solve under the PRE-commutation mask, and is recorded as
such — no ``g_on * V_th`` phantom from the post-flip bits) and
``engine='dsed'`` (each full MNA sample is reconstructed through its own
mode mask). Nothing is inferred, so a snubber across the diode, an ideal
transformer, a VCVS probe or any other device on the diode's nodes changes
nothing.

``result.switch_states_unsolved[k]`` flags the rare sample whose state was
not a solve under its recorded mask (a sample 0 seeded from
``initial_state``); the recorded mask there is the bits the solver carried
in, which say nothing about the solve that produced the state, so the row
has no defined current: ``result.i()`` returns the first solved sample's
value there and the loss and thermal summaries start at that sample
(``_result_views.solved_part``). The trace below keeps the carried bits.

When a result carries no record (a hand-built result, a stitched
``AdaptiveResult`` whose segments lacked it, the DSED Python-adapter path),
the state is replayed from the kernel's commutation rule on the stored
samples (OFF -> ON at ``v_D >= V_th``, ON -> OFF at ``v_D <= 0``) and a
``RuntimeWarning`` names the reason. That replay is the legacy fallback,
not a second source of truth: it misses a turn-on the event iteration
resolved inside one step, which is exactly why the record exists.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from . import _result_views as _views


@dataclass
class DiodeStateTrace:
    """Per-sample conduction state of one switched diode.

    ``closed[k]`` is True when the ON stamp (``g_on``) produced sample
    ``k``; ``current = v * (g_on if closed else g_off)`` is then the
    branch current the solver actually carried. ``recorded`` is False
    when ``closed`` had to be replayed (no kernel record).

    ``v_start`` / ``closed_start`` are the same at the START of the step
    after each sample (``_result_views.step_start_states``): equal to
    ``v`` / ``closed`` except where ``engine='dsed'`` fired an event at
    the sample, whose stored row is the left limit. Time integrals run
    each step from ``current_start[k-1]`` to ``current[k]``."""

    branch_id: int
    name: str
    g_on: float
    g_off: float
    V_th: float
    v: np.ndarray
    closed: np.ndarray
    recorded: bool = True
    n_rule_violations: int = 0
    max_violation_V: float = 0.0
    violation_t_range: Tuple[float, float] = (float("nan"), float("nan"))
    v_start: Optional[np.ndarray] = field(default=None, repr=False)
    closed_start: Optional[np.ndarray] = field(default=None, repr=False)
    _messages: List[str] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        if self.v_start is None:
            self.v_start = self.v
        if self.closed_start is None:
            self.closed_start = self.closed

    @property
    def current(self) -> np.ndarray:
        return self.v * np.where(self.closed, self.g_on, self.g_off)

    @property
    def current_start(self) -> np.ndarray:
        """Branch current at the start of the step after each sample."""
        return self.v_start * np.where(self.closed_start, self.g_on,
                                       self.g_off)

    @property
    def conducting(self) -> np.ndarray:
        """Per-STEP conduction flag: element ``k`` describes the step
        ``(t[k-1], t[k]]`` that produced sample ``k`` (element 0 the
        initial point), for right-hold time weighting
        (``pulsim.losses._time_fraction``).

        A step conducts when its stamp is ON AND the diode carries a
        material forward current (``g_on·v_D > 1e-6 · i_peak``) at
        EITHER end of it:

        * either end, because the step that carries the current down
          to its zero ends at ``v_D ≈ 0`` (``engine='auto'`` lands the
          turn-off ON the current zero, at ``v_D ≈ -4e-9 V``); judging
          the step by that end alone dropped whole ON steps;
        * a material current, because a stamp left ON at the boundary
          is not conduction a duty cycle should count:
          ``engine='auto'`` rides a V_th = 0 diode ON at
          ``v_D ≈ -3e-10 V``, and the fixed engine chatters a V_th = 0
          diode ON/OFF at ``v_D ≈ 1e-10 V`` (≈ 1e-7 A).

        A step that ENDS where the ON and OFF stamps would carry the
        same, immaterial current (``g_on·|v_D|`` below the threshold)
        keeps the state of the step before it: on the fixed engine the
        step in which a current crosses zero is stored with the OFF
        stamp its event iteration settled on, although it conducted
        until the zero."""
        closed = np.asarray(self.closed, dtype=bool)
        if closed.size == 0:
            return closed.copy()
        i_pk = float(np.max(np.abs(self.current)))
        thr = max(1e-12, 1e-6 * i_pk)
        i_fwd = self.g_on * np.asarray(self.v, dtype=float)
        fwd = i_fwd > thr
        # A step's start is the right limit of the sample before it
        # (v_start; the sample itself except at a DSED event).
        fwd_start = self.g_on * np.asarray(self.v_start, dtype=float) > thr
        either = fwd.copy()
        either[1:] |= fwd_start[:-1]
        stamp = closed.copy()
        stamp[1:] |= closed[:-1] & (np.abs(i_fwd[1:]) <= thr)
        return stamp & either

    def warn(self, stacklevel: int = 3) -> None:
        """Emit one ``RuntimeWarning`` per problem found."""
        for msg in self._messages:
            warnings.warn(msg, RuntimeWarning, stacklevel=stacklevel + 1)


def _replay_kernel_rule(v: np.ndarray, V_th: float) -> np.ndarray:
    """Legacy fallback: the kernel's commutation rule replayed on the
    stored samples — ON where ``v >= V_th`` (and ``v > 0``), OFF where
    ``v <= 0``, the previous state inside the band (initially OFF)."""
    n = v.size
    st = np.full(n, -1, dtype=np.int8)
    st[(v > 0.0) & (v >= V_th)] = 1
    st[v <= 0.0] = 0
    idx = np.where(st >= 0, np.arange(n), -1)
    np.maximum.accumulate(idx, out=idx)
    return np.where(idx >= 0, st[np.clip(idx, 0, None)], 0).astype(bool)


def _cache_key(result, times) -> tuple:
    return (int(times.size),
            float(times[0]) if times.size else 0.0,
            float(times[-1]) if times.size else 0.0)


def diode_state_traces(builder, result, *,
                       switch_fn: Optional[Callable[[float], Any]] = None,
                       ) -> Dict[int, DiodeStateTrace]:
    """Conduction state of every switched diode, from the kernel's
    stamped-mask record (``switch_fn`` is accepted for signature
    compatibility and not used: a diode's state is solver-owned).

    Returns ``{branch_id: DiodeStateTrace}``. Cached on ``result``."""
    del switch_fn
    states = _views.states_as_array(result)
    times = _views.times_as_array(result)
    key = _cache_key(result, times)
    cached = getattr(result, "_diode_state_cache", None)
    if cached is not None and cached[0] == key:
        return cached[1]
    traces = _build(builder, result, states, times)
    try:
        result._diode_state_cache = (key, traces)
    except AttributeError:  # pragma: no cover — slotted result type
        pass
    return traces


def diode_state_trace(builder, result, branch_id: int, *,
                      switch_fn: Optional[Callable[[float], Any]] = None,
                      warn: bool = True,
                      stacklevel: int = 3) -> DiodeStateTrace:
    """The :class:`DiodeStateTrace` of one diode; warns if asked, ONCE
    per result and diode.

    Python's default filter shows a warning once per call site, so a
    script that read ``res.i("D1")`` on four lines printed the same
    missed-commutation warning four times, and a fifth time from
    ``device_loss_summary``. The diodes already warned about are kept
    on the result (on the full result, not on the
    :func:`_result_views.solved_part` view of it); a result that
    accepts no attributes warns on every call."""
    tr = diode_state_traces(builder, result, switch_fn=switch_fn)[
        int(branch_id)]
    if warn and tr._messages:
        owner = (result._result if isinstance(result, _views._SolvedPart)
                 else result)
        seen = getattr(owner, "_diode_warnings_emitted", None)
        if seen is None:
            seen = set()
            try:
                owner._diode_warnings_emitted = seen
            except AttributeError:  # pragma: no cover — slotted result
                seen = None
        if seen is None or int(branch_id) not in seen:
            if seen is not None:
                seen.add(int(branch_id))
            for msg in tr._messages:
                warnings.warn(msg, RuntimeWarning, stacklevel=stacklevel)
    return tr


def _event_sample_mask(result, times: np.ndarray) -> np.ndarray:
    """Samples stored at a commutation record's time (trbdf2 event
    landings, committed under the PRE-flip mask just past the
    crossing) or at an event-iteration breach (the fixed engine kept
    the last solve without converging) — where the kernel's own rule
    legitimately does not hold for the stored stamp, within the
    envelope ``_build`` bounds it by."""
    n = times.size
    out = np.zeros(n, dtype=bool)
    if not n:
        return out
    ts: List[float] = []
    if getattr(result, "engine_used", None) == "trbdf2":
        ts += [float(ev.t_estimated)
               for ev in (getattr(result, "commutation_events", []) or [])]
    ts += [float(br.t)
           for br in (getattr(result, "event_iteration_breaches", []) or [])]
    if ts:
        q = np.asarray(ts, dtype=float)
        k = np.clip(np.searchsorted(times, q), 0, n - 1)
        hit = times[k] == q
        out[k[hit]] = True
        # A zero-time cascade can store several samples at one t.
        for kk in k[hit].tolist():
            j = kk + 1
            while j < n and times[j] == times[kk]:
                out[j] = True
                j += 1
    return out


def _build(builder, result, states, times) -> Dict[int, DiodeStateTrace]:
    comps = list(builder.components())
    diodes = [c for c in comps if c.get("kind") == "diode"]
    if not diodes:
        return {}
    n = int(times.size)
    node_v = _views.node_voltage_trace
    rec, why = _views.recorded_switch_states(result, builder)
    bits = _views.switch_bits_by_branch(builder) if rec is not None else {}
    states_s, rec_s = _views.step_start_states(result, states, rec)
    unsolved = _views.recorded_unsolved(result, n) if rec is not None \
        else np.zeros(n, dtype=bool)
    landing = _event_sample_mask(result, times) & ~unsolved
    rest = ~landing & ~unsolved
    # A landing is the PRE-flip stamp solved just past its crossing, so
    # the rule fails there only by what the crossing allows. One that
    # leaves the run's own envelope (below) is not a landing artefact
    # but a state the circuit never reaches otherwise — a gate edge
    # 1 ulp after a resume's t_start stored a buck's D1 ON across M1
    # turning on (24 kA, i_peak 25 A elsewhere) — and is checked.
    # Measured over the suite's diode circuits: legitimate landings sit
    # at |i|/i_env <= 1.0015 (that one inside a run already warned for
    # 25 held samples) and v/v_env <= 0.17 (a source stepping 2.9 V
    # across V_th); that shoot-through sits at |i|/i_env ~ 950.
    n_nodes = int(getattr(getattr(builder, "graph", None), "num_nodes", 0))
    v_env = float(np.abs(states[rest][:, :n_nodes]).max()) \
        if n_nodes and rest.any() else 0.0
    engine = getattr(result, "engine_used", "?")

    out: Dict[int, DiodeStateTrace] = {}
    for d in diodes:
        bid = int(d["branch_id"])
        prm = d.get("params", {}) or {}
        g_on = float(prm.get("g_on", float("nan")))
        g_off = float(prm.get("g_off", float("nan")))
        V_th = float(prm.get("V_th", 0.0))
        f, to = d["nodes"]
        v = node_v(states, int(f)) - node_v(states, int(to))
        name = str(d.get("name", ""))
        bit = bits.get(bid)
        if rec is not None and bit is not None:
            closed = np.asarray(rec[:, bit], dtype=bool)
            v_s = c_s = None
            if states_s is not states:
                v_s = (node_v(states_s, int(f))
                       - node_v(states_s, int(to)))
                c_s = np.asarray(rec_s[:, bit], dtype=bool)
            tr = DiodeStateTrace(branch_id=bid, name=name, g_on=g_on,
                                 g_off=g_off, V_th=V_th, v=v,
                                 closed=closed, recorded=True,
                                 v_start=v_s, closed_start=c_s)
        else:
            reason = why if rec is None else (
                "its mask bit could not be resolved from the builder")
            closed = _replay_kernel_rule(v, V_th)
            tr = DiodeStateTrace(branch_id=bid, name=name, g_on=g_on,
                                 g_off=g_off, V_th=V_th, v=v,
                                 closed=closed, recorded=False)
            in_band = (v > 0.0) & (v < V_th)
            tr._messages.append(
                f"diode {name!r}: the result carries no record of the "
                f"switch / diode mask the solver stamped ({reason}), so "
                f"its conduction state was replayed from the kernel's "
                f"commutation rule on the stored samples (ON at "
                f"v_D >= V_th = {V_th:g} V, OFF at v_D <= 0). "
                f"{int(in_band.sum())} of {n} samples sit inside the "
                f"0 < v_D < V_th band, where that replay cannot see a "
                f"turn-on the solver resolved within one step; the "
                f"reported current, duty and loss there may be wrong. "
                f"To get the record, {_views.record_advice(result)}")
        # The kernel's own rule, checked against what it stamped. Not a
        # reporting problem — the current above is what the solver
        # carried — but a held-across-the-threshold state means the
        # engine missed a commutation, which the user must hear about.
        if tr.recorded and n:
            # Envelope of a landing: an ON stamp carries no more current
            # than the diode carries on any other solved sample; an OFF
            # stamp holds no more forward voltage than any node reaches.
            i_abs = np.abs(tr.current)
            i_env = float(i_abs[rest].max()) if rest.any() else 0.0
            exempt = unsolved | (landing & np.where(closed, i_abs <= i_env,
                                                    v <= v_env))
            off_fwd = ~exempt & ~closed & (v - V_th > 1e-3) & (v > 1e-3)
            on_rev = ~exempt & closed & (v < -1e-3)
            viol = off_fwd | on_rev
            if viol.any():
                tr.n_rule_violations = int(viol.sum())
                dv = np.where(closed, -v, v - V_th)
                tr.max_violation_V = float(dv[viol].max())
                tv = times[viol]
                tr.violation_t_range = (float(tv.min()), float(tv.max()))
                # Name what actually happened (a misleading example sent
                # a reader of the auto-engine bridge looking for a
                # missed turn-on when the diode was held ON reverse
                # biased).
                kinds = []
                if on_rev.any():
                    kinds.append(
                        f"ON while reverse-biased on {int(on_rev.sum())} "
                        f"samples, by up to "
                        f"{float((-v[on_rev]).max()):.3g} V")
                if off_fwd.any():
                    kinds.append(
                        f"OFF while forward-biased above V_th = "
                        f"{V_th:g} V on {int(off_fwd.sum())} samples, by "
                        f"up to {float((v - V_th)[off_fwd].max()):.3g} V")
                n_land = int((viol & landing).sum())
                if n_land:
                    kinds.append(
                        f"{n_land} of them at a commutation landing whose "
                        f"pre-flip stamp carried more current, or held "
                        f"more forward voltage, than any other solved "
                        f"sample (the reported peaks and energies "
                        f"include it)")
                tr._messages.append(
                    f"diode {name!r}: on {tr.n_rule_violations} samples "
                    f"(t = {tv.min():.6g} .. {tv.max():.6g} s) the "
                    f"simulation held the diode in a state its "
                    f"commutation rule forbids: {'; '.join(kinds)}. The "
                    f"reported current is what the solver actually "
                    f"stamped, but the engine ({engine!r}) missed a "
                    f"commutation from t = {tv.min():.6g} s on, so from "
                    f"there the waveforms of the whole circuit (every "
                    f"switched branch, not only this diode) are suspect; "
                    f"cross-check with engine='pwl' and a fixed dt.")
        out[bid] = tr
    return out


# The kernel's current ceiling above which the Shockley exponential is
# continued by its tangent (ShockleyDiode::Params::i_lim; the builder
# does not expose it).
_SHOCKLEY_I_LIM = 1e6


def _shockley_branch(u: np.ndarray, I_S: float, vte: float) -> np.ndarray:
    """``I_S·(exp(u/vte) − 1)``, continued by its tangent above the
    voltage where it reaches ``_SHOCKLEY_I_LIM``, as the kernel's
    ``limited_exp_branch_``."""
    vl = vte * np.log(_SHOCKLEY_I_LIM / I_S + 1.0)
    e_l = np.exp(vl / vte)
    i_l = I_S * (e_l - 1.0)
    g_l = I_S * e_l / vte
    exp_part = I_S * np.expm1(np.minimum(u, vl) / vte)
    return np.where(u > vl, i_l + g_l * (u - vl), exp_part)


def shockley_current(v: np.ndarray, params: Dict[str, float]) -> np.ndarray:
    """Branch current (anode to cathode) of an ``add_shockley_diode``
    at junction voltage ``v``: the kernel's law
    (``core/include/pulsim/models/shockley_diode.hpp``)::

        i = I_S·(exp(v/(n·V_T)) − 1) − [BV > 0] I_S·(exp((−BV − v)/(n·V_T)) − 1)
            + v·G_min

    with each exponential continued by its tangent above 1e6 A. The
    state holds the node voltages the solver converged to, so this is
    the current it carried, to its Newton tolerance."""
    v = np.asarray(v, dtype=float)
    I_S = float(params["I_S"])
    vte = float(params["n"]) * float(params["V_T"])
    i = _shockley_branch(v, I_S, vte)
    BV = float(params.get("BV", 0.0))
    if BV > 0.0:
        i = i - _shockley_branch(-BV - v, I_S, vte)
    return i + v * float(params.get("G_min", 0.0))
