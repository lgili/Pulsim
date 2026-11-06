"""Pulsim v2 — loss accumulator + efficiency calculator.

Tracks conduction loss + switching loss + total energy over a
simulation window, on a per-device or per-converter basis. Mirrors
the v1 ``pulsim::v1::LossAccumulator`` + ``EfficiencyCalculator``
API so users migrating from v1 see the same surface.

Two modes:

1. **Real-time** — feed power samples from a step_observer::

       loss = p.LossAccumulator()
       def observer(t, x):
           i_L = x[ind_idx]
           loss.add_sample(P_cond=R_on * i_L ** 2, dt=DT)
       res = p.simulate(b, ..., step_observer=observer)
       print(f"avg = {loss.average_power():.2f} W")

2. **Post-hoc** — analyse the full :class:`SimulationResult` after
   the run::

       res = p.simulate(b, t_end=..., dt=...)
       summary = p.device_loss_summary(b, res)
       print(summary)

   :func:`device_loss_summary` walks every resistor, inductor,
   ideal-switch and switched-diode branch and reports per-device
   conduction statistics. Switch and diode states come from the
   kernel's per-sample record of the mask it stamped
   (``result.switch_states``); ``switch_fn=`` is only needed for a
   result built without that record. Pass ``core_loss_specs=`` to add
   Steinmetz / iGSE core loss on selected inductors.

The reconstruction is post-hoc only — the kernel itself stays
fixed-dt and the result trace is processed in NumPy. No kernel
binding changes are required for v1.6 closure.
"""

from __future__ import annotations

import warnings
from typing import Any, Callable, Dict, List, Mapping, Optional, Union

import numpy as np


from .loss_tables import (
    as_loss_table as _as_loss_table,
    resolve_tj as _resolve_tj,
)
from . import _diode_state
from . import _result_views as _views
from ._result_views import time_weighted_rms as _time_weighted_rms
from . import magnetic as _magnetic


__all__ = [
    "LossAccumulator",
    "EfficiencyCalculator",
    "device_loss_summary",
    "average_power_at_node",
]


# =============================================================================
# Real-time accumulator
# =============================================================================

class LossAccumulator:
    """Accumulates conduction + switching loss over a simulation run.

    Mirrors the v1 ``LossAccumulator`` API: feed
    ``add_sample(P_cond, dt)`` at every step and
    ``add_switching_event(E_sw)`` at every commutation. Queries:
    :meth:`total_energy`, :meth:`conduction_energy`,
    :meth:`switching_energy`, :meth:`average_power`, etc.

    Parameters
    ----------
    label
        Optional name for reporting / debugging. Default empty.
    """

    __slots__ = ("label",
                  "_total_energy", "_conduction_energy",
                  "_switching_energy", "_n_samples",
                  "_duration", "_n_switch_events")

    def __init__(self, label: str = ""):
        self.label = label
        self._total_energy = 0.0
        self._conduction_energy = 0.0
        self._switching_energy = 0.0
        self._n_samples = 0
        self._duration = 0.0
        self._n_switch_events = 0

    # ----- mutators --------------------------------------------------

    def reset(self) -> None:
        """Zero all counters."""
        self._total_energy = 0.0
        self._conduction_energy = 0.0
        self._switching_energy = 0.0
        self._n_samples = 0
        self._duration = 0.0
        self._n_switch_events = 0

    def add_sample(self, P_cond: float, dt: float) -> None:
        """Add an instantaneous conduction-power sample. Integrates
        ``P_cond · dt`` into the conduction energy.
        """
        dE = float(P_cond) * float(dt)
        self._conduction_energy += dE
        self._total_energy += dE
        self._n_samples += 1
        self._duration += float(dt)

    def add_switching_event(self, E_sw: float) -> None:
        """Add a single switching-loss event (Joules). Used at every
        turn-on / turn-off transition with E_sw computed from the
        device data sheet (V_DS · I_D · t_sw / 2 + ...)."""
        E = float(E_sw)
        self._switching_energy += E
        self._total_energy += E
        self._n_switch_events += 1

    # ----- queries ---------------------------------------------------

    @property
    def total_energy(self) -> float:
        """Total accumulated energy (J)."""
        return self._total_energy

    @property
    def conduction_energy(self) -> float:
        """Energy dissipated in conduction (J) = ∫ P_cond dt."""
        return self._conduction_energy

    @property
    def switching_energy(self) -> float:
        """Energy dissipated in switching events (J) = Σ E_sw."""
        return self._switching_energy

    @property
    def duration(self) -> float:
        """Window over which samples were taken (s)."""
        return self._duration

    @property
    def n_samples(self) -> int:
        return self._n_samples

    @property
    def n_switch_events(self) -> int:
        return self._n_switch_events

    def average_power(self) -> float:
        """Mean dissipated power over the window (W) — conduction
        + switching combined."""
        return self._total_energy / self._duration \
                  if self._duration > 0 else 0.0

    def average_conduction_power(self) -> float:
        return self._conduction_energy / self._duration \
                  if self._duration > 0 else 0.0

    def average_switching_power(self) -> float:
        return self._switching_energy / self._duration \
                  if self._duration > 0 else 0.0

    def __repr__(self) -> str:
        return (f"LossAccumulator(label={self.label!r}, "
                  f"P_avg={self.average_power():.3g} W, "
                  f"E={self._total_energy:.3g} J, "
                  f"n={self._n_samples})")


# =============================================================================
# Efficiency calculator
# =============================================================================

class EfficiencyCalculator:
    """Static helpers to express converter efficiency η in the three
    common ways. All return η as a fraction in ``[0, 1]`` (multiply
    by 100 for percent)."""

    @staticmethod
    def from_power(P_in: float, P_out: float) -> float:
        """η = P_out / P_in. Returns 0 when P_in ≤ 0."""
        return float(P_out) / float(P_in) if P_in > 0 else 0.0

    @staticmethod
    def from_losses(P_out: float, P_loss: float) -> float:
        """η = P_out / (P_out + P_loss). Robust when P_in isn't
        directly measured."""
        denom = float(P_out) + float(P_loss)
        return float(P_out) / denom if denom > 0 else 0.0

    @staticmethod
    def from_voltages_currents(V_in: float, I_in: float,
                                  V_out: float, I_out: float) -> float:
        """η = (V_out · I_out) / (V_in · I_in)."""
        P_in = float(V_in) * float(I_in)
        P_out = float(V_out) * float(I_out)
        return EfficiencyCalculator.from_power(P_in, P_out)


# =============================================================================
# Post-hoc helpers — work on a SimulationResult
# =============================================================================
#
# The result-walk primitives (``states_as_array``,
# ``node_voltage_trace``, ``evaluate_switch_mask_trace``) live in
# :mod:`pulsim._result_views` and are shared with :mod:`pulsim.thermal`.
# Module-local aliases keep the call sites short.

_states_as_array = _views.states_as_array
_times_as_array = _views.times_as_array
_node_voltage_trace = _views.node_voltage_trace


def average_power_at_node(builder,
                              result,
                              *,
                              node_name: str,
                              current_branch_id: int) -> float:
    """Compute average power flowing into / through a node, given
    the node voltage and the branch carrying the current. Useful for
    P_in / P_out of a converter::

        P_in  = average_power_at_node(b, res, node_name="vin",
                                        current_branch_id=src_branch)
        P_out = average_power_at_node(b, res, node_name="vout",
                                        current_branch_id=load_branch)
        eta = p.EfficiencyCalculator.from_power(P_in, P_out)

    Parameters
    ----------
    node_name
        Name of the node whose voltage to use.
    current_branch_id
        Branch ID of the source / load whose current to multiply
        with the node voltage. The branch must be an inductor or
        a voltage source (so it has a current state-vector entry).
    """
    node_idx = builder.node_id_of(node_name)
    states = _states_as_array(result)
    times = _times_as_array(result)
    v = states[:, node_idx]
    try:
        i_idx = builder.pool.branch_var_id_for_inductor(
            current_branch_id, builder.graph)
    except Exception:
        i_idx = builder.pool.branch_var_id_for_source(
            current_branch_id, builder.graph)
    i = states[:, i_idx]
    P = v * i
    if times.size < 2:
        return 0.0
    return float(np.trapezoid(P, times) / (times[-1] - times[0]))


# ---------------------------------------------------------------
# Helpers for the diode / switch / magnetic paths.
# ---------------------------------------------------------------

def _conduction_stats(v: np.ndarray,
                      g: Union[float, np.ndarray],
                      times: np.ndarray,
                      v_start: "Optional[np.ndarray]" = None,
                      g_start: "Optional[Union[float, np.ndarray]]" = None,
                      ) -> Dict[str, float]:
    """Compute (i_avg, i_rms, i_peak, P_avg, E_total) for a branch
    whose current trace is ``i = v · g`` and whose instantaneous
    power is ``P = v · i = v² · g``.

    ``g`` may be a scalar (constant conductance, e.g. a resistor or
    on-state switch) or an array matching ``v`` (state-dependent
    conductance, e.g. a switch toggling between ``g_on`` / ``g_off``).

    Each step ``(t[k-1], t[k]]`` is the line from its START,
    ``(v_start, g_start)[k-1]``, to its end ``(v, g)[k]``, integrated
    exactly (``_result_views.pl_integral`` / ``pl_integral_product``).
    The start defaults to the sample before the step; it differs only
    where ``engine='dsed'`` fired an event, whose stored sample is the
    left limit (``_result_views.step_start_states``).
    """
    def _arr(x, like):
        if isinstance(x, np.ndarray):
            return x.astype(float, copy=False)
        return np.full_like(like, float(x))

    g_arr = _arr(g, v)
    vs = v if v_start is None else np.asarray(v_start, dtype=float)
    gs = g_arr if g_start is None else _arr(g_start, v)
    i = v * g_arr
    i_s = vs * gs
    T = (times[-1] - times[0]) if times.size > 1 else 1.0
    if T <= 0:
        T = 1.0
    if times.size > 1:
        Q = _views.pl_integral(times, i, i_s)
        E = _branch_energy(times, v, g_arr, vs, gs)
        i_avg, P_avg, E_total = Q / T, E / T, E
    else:
        p = v * i
        i_avg, P_avg, E_total = float(i.mean()), float(p.mean()), 0.0
    i_peak = float(max(np.abs(i).max(), np.abs(i_s).max())) if i.size \
        else 0.0
    return {
        "i_avg": float(i_avg),
        # Time-weighted, not sample-averaged: on a variable-step
        # grid the samples cluster at commutations (where ripple
        # peaks), and an unweighted mean read +4.6% high on a buck
        # diode at rtol=1e-6 — a bias that GREW as the tolerance
        # tightened.
        "i_rms": _time_weighted_rms(i, times, i_s),
        "i_peak": i_peak,
        "P_avg": float(P_avg),
        "E_total": float(E_total),
    }


def _branch_energy(times: np.ndarray, v: np.ndarray, g: np.ndarray,
                   v_start: np.ndarray, g_start: np.ndarray) -> float:
    """``∫ v·i dt`` of a conductance branch (``i = g·v``), per step.

    A step whose two ends carry the same conductance (one stamp over
    the whole step, the usual case and every ``engine='dsed'`` step
    started from its right limit) is the exact integral of the linear
    voltage squared, ``g·h·(a² + a·c + c²) / 3``. A step whose ends
    differ (the fixed and TR-BDF2 engines store no right limit, so the
    step that flips a switch starts from a sample solved under the
    OTHER stamp) has no line to integrate: the product of the two
    lines read a freewheel diode's loss NEGATIVE (−24 V at the g_off
    end times the g_on end's current), so that step keeps the
    trapezoid of the two end powers, as before."""
    t = np.asarray(times, dtype=float)
    if t.size < 2:
        return 0.0
    h = np.diff(t)
    a, c = v_start[:-1], v[1:]
    ga, gc = g_start[:-1], g[1:]
    same = ga == gc
    lin = gc * (a * a + a * c + c * c) / 3.0
    trap = 0.5 * (ga * a * a + gc * c * c)
    return float(np.dot(h, np.where(same, lin, trap)))


def _conduction_model_coeffs(spec: Mapping[str, Any]) -> "tuple[float, float]":
    """Pull ``(V_f0, r_on)`` from a conduction-model spec.

    Accepts datasheet-friendly aliases so the same helper covers diodes
    (``V_f0`` / ``r_d``), MOSFETs (``r_on`` only — V_f0 = 0) and IGBTs
    (``V_ce0`` / ``r_ce``)::

        V_f0 ← V_f0 | V_F0 | V_ce0 | V_CE0 | V_F   (default 0)
        r_on ← r_on | R_on | r_d | r_ce | R_CE     (default 0)
    """
    def _first(d, keys, default=0.0):
        for k in keys:
            if k in d:
                return float(d[k])
        return float(default)
    V_f0 = _first(spec, ("V_f0", "V_F0", "V_ce0", "V_CE0", "V_F"))
    r_on = _first(spec, ("r_on", "R_on", "r_d", "R_d", "r_ce", "R_CE"))
    return V_f0, r_on


def _conduction_power_offset_slope(i_arr: np.ndarray,
                                   spec: Mapping[str, Any]) -> np.ndarray:
    """Per-step conduction power for a PWL device characteristic.

    ``P_cond(t) = V_f0 · |i(t)| + r_on · i(t)²``

    This is the standard offset-plus-slope conduction model
    (Erickson & Maksimović Ch. 3): a fixed forward-voltage drop ``V_f0``
    plus an ohmic ``r_on``. It is applied to the device's *actual*
    current trace from the simulation, so the loss tracks the real
    waveform — strictly more accurate than the pure-resistive ``v²·g``
    reconstruction for IGBTs and diodes, whose V_f0 offset dominates at
    low current.
    """
    V_f0, r_on = _conduction_model_coeffs(spec)
    i = np.asarray(i_arr, dtype=float)
    return V_f0 * np.abs(i) + r_on * i * i


def _conduction_model_energy(times: np.ndarray, i_arr: np.ndarray,
                             i_start: np.ndarray,
                             spec: Mapping[str, Any]) -> float:
    """``∫ (V_f0·|i| + r_on·i²) dt`` over the run, each step the line
    from ``i_start[k-1]`` to ``i_arr[k]`` integrated exactly (the same
    waveform ``_conduction_stats`` integrates)."""
    V_f0, r_on = _conduction_model_coeffs(spec)
    if np.asarray(times).size < 2:
        return 0.0
    return float(V_f0 * _views.pl_integral_abs(times, i_arr, i_start)
                 + r_on * _views.pl_integral_product(
                     times, i_arr, i_arr, i_start, i_start))


def _interp_energy_curve(x_query, curve) -> np.ndarray:
    """Interpolate a datasheet switching-energy curve ``E(I)``.

    ``curve`` is a sequence of ``(I, E)`` points (current [A], energy
    [J], measured at a fixed reference bus voltage). Returns the
    piecewise-linearly interpolated energy at each ``x_query`` current,
    with **linear extrapolation** beyond the tabulated range — more
    honest than clamping when a higher-power operating point pushes the
    current past the datasheet curve. Output is clamped to ``>= 0``
    (switching energy is non-negative).
    """
    pts = sorted((float(i), float(e)) for i, e in curve)
    if not pts:
        raise ValueError("switching-energy curve needs >= 1 (I, E) point")
    xs = np.asarray([q[0] for q in pts], dtype=float)
    ys = np.asarray([q[1] for q in pts], dtype=float)
    xq = np.atleast_1d(np.asarray(x_query, dtype=float))
    if xs.size == 1:
        out = np.full_like(xq, ys[0])
    else:
        out = np.interp(xq, xs, ys)
        below = xq < xs[0]
        if below.any():
            s = (ys[1] - ys[0]) / (xs[1] - xs[0])
            out = np.where(below, ys[0] + s * (xq - xs[0]), out)
        above = xq > xs[-1]
        if above.any():
            s = (ys[-1] - ys[-2]) / (xs[-1] - xs[-2])
            out = np.where(above, ys[-1] + s * (xq - xs[-1]), out)
    return np.maximum(out, 0.0)


def _time_fraction(mask: np.ndarray, times: np.ndarray) -> float:
    """Fraction of the run's TIME during which ``mask`` is True.

    ``mask[k]`` is the state of the STEP that produced sample ``k``
    (the switch mask the engine solved ``(t[k-1], t[k]]`` with, or a
    diode's stamp — see ``_diode_state.DiodeStateTrace``), so each
    interval is weighted by the state at its RIGHT end:
    ``sum(mask[1:] · diff(t)) / (t[-1] − t[0])``.

    Time-weighted on purpose: a variable-step engine clusters samples
    at commutations, so a plain ``mask.mean()`` over-weights exactly
    the instants a device changes state. Right-hold, not trapezoid:
    the trapezoid credits a long ON step only half its length when
    the step's other end is an OFF sample — on ``engine='auto'`` an
    ON step of 1.4 µs followed by a femtosecond turn-off landing lost
    0.7 µs per cycle. On a uniform grid this differs from the sample
    mean by at most one sample."""
    m = np.asarray(mask, dtype=float)
    t = np.asarray(times, dtype=float)
    if m.size == 0:
        return 0.0
    if t.size != m.size or t.size < 2 or not (t[-1] > t[0]):
        return float(m.mean())
    return float(np.dot(m[1:], np.diff(t)) / (t[-1] - t[0]))


# A conduction stretch whose forward charge is below this fraction of
# the recovery charge the spec asks for is not a turn-off: its capped
# energy (Q_F · V_R) is below this fraction of one real event's.
_RR_MATERIAL_FRACTION = 1e-3
# Without a recovery charge scale (counts only, or an E_rr_curve), a
# stretch whose forward current never reaches this fraction of a
# reference current is dropped (see _diode_switching_loss).
_RR_MATERIAL_CURRENT_FRACTION = 1e-3


# Recovery time used to place the V_R reading when diode_specs gives no
# ``t_rr``: the order of a fast-recovery power diode's. Only WHERE V_R
# is read depends on it, never the recovered charge.
_DEFAULT_T_RR = 100e-9


# Two readings of v_D one recovery time after a turn-off that falls on a
# long step agree when they differ by less than this fraction (plus a
# floor of _RR_FLAT_FLOOR · max|v_D| of the run).
_RR_FLAT_RTOL = 0.05
_RR_FLAT_FLOOR = 1e-6
# A turn-off is FORCED (hard-commutated) when the diode still carries at
# least this fraction of its stretch's peak forward current on its last
# conducting sample.
_RR_FORCED_FRACTION = 0.1


def _reverse_voltage_at(t: np.ndarray, v: np.ndarray, k: int, end: int,
                        t_eval: float, flat_atol: float):
    """Signed ``v_D`` at ``t_eval`` read from the BLOCKING samples
    ``k .. end-1`` of one turn-off, and whether that reading depends on
    the grid.

    The samples are first paired into step midpoints,
    ``((t[j] + t[j+1]) / 2, (v[j] + v[j+1]) / 2)``. On a uniform grid
    that removes the fixed engine's trapezoidal alternation exactly
    (``V_s ± A`` on successive samples averages to ``V_s``) and leaves
    ramps and plateaus unchanged. ``t_eval`` past the first midpoint
    reads their linear interpolant. ``t_eval`` before it (the turn-off
    step itself is longer than ``t_rr``) is extrapolated back along
    the line through the first two midpoints: the post-turn-off
    trajectory continued to ``t_eval``, NOT a line from the last
    conducting sample. A left-limit sample (TR-BDF2 landing, DSED
    event) is the state before the voltage jump, and drawing a line
    from it inside a long step would read a fraction of a jump the
    circuit made at once. The extrapolation is clipped to the range
    between that left-limit value and the first midpoint, so the
    curvature of a slow rise (a line rectifier's ``v_D ∝ (t − t_0)²``
    after its current zero) cannot overshoot into a forward voltage.

    Returns ``(v_reading, flat)``. ``flat`` is ``None`` when no
    extrapolation was needed; otherwise whether the extrapolated
    reading agrees with the first-step plateau ``vm[0]``, i.e. the
    trajectory is flat across the long step, so where inside it the
    reading falls does not matter IF the voltage jumped at the
    turn-off (see the caller)."""
    tb = t[k:end]
    vb = v[k:end]
    if tb.size < 2:
        return float(vb[0]), None
    tm = 0.5 * (tb[:-1] + tb[1:])
    vm = 0.5 * (vb[:-1] + vb[1:])
    if t_eval >= tm[0]:
        return float(np.interp(t_eval, tm, vm)), None
    if tm.size < 2 or not (tm[1] > tm[0]):
        return float(vm[0]), False
    slope = (vm[1] - vm[0]) / (tm[1] - tm[0])
    e0 = float(vm[0])
    v_left = float(v[k - 1])
    e1 = float(np.clip(e0 + slope * (t_eval - tm[0]),
                       min(v_left, e0), max(v_left, e0)))
    flat = abs(e1 - e0) <= (_RR_FLAT_RTOL * max(abs(e1), abs(e0))
                            + flat_atol)
    return e1, bool(flat)


def _other_bits_flipped(masks: "Optional[np.ndarray]",
                        bit: int) -> "Optional[np.ndarray]":
    """Per sample, whether any switch-kind bit OTHER than ``bit``
    changed on the step ending there (row k vs row k-1 of the recorded
    stamped mask). ``None`` without a usable record."""
    if masks is None or masks.ndim != 2 or not (0 <= bit < masks.shape[1]):
        return None
    others = np.delete(masks, bit, axis=1)
    out = np.zeros(masks.shape[0], dtype=bool)
    if others.shape[1] and masks.shape[0] > 1:
        out[1:] = np.any(others[1:] != others[:-1], axis=1)
    return out


def _chatter_gaps(trace, other_flip: "Optional[np.ndarray]" = None,
                  ) -> "tuple[np.ndarray, int]":
    """Blocking gaps of one diode that are chatter inside a single
    conduction, not turn-offs.

    Returns ``(conducting, n_gaps)``: ``trace.conducting`` with those
    gaps filled, and how many there were. A gap (a maximal run of
    non-conducting steps between two conducting ones) is chatter when
    the diode never blocks across it: the step-averaged ``v_D`` (the
    midpoint of each step's two ends, from the last conducting sample
    to the next one) stays above ``−2·V_th``, AND no other bit of the
    recorded stamped mask flips on those steps (``other_flip``; a
    commutation forced by another switch is a real turn-off even when
    the voltage it leaves is small, e.g. the first cycles of a boost
    whose output is still below V_th). A turn-off is the start of
    blocking, so a gap the diode spends within a reflection of its own
    commutation band is part of the conduction around it. For
    ``V_th = 0`` any reverse step average is blocking, so nothing
    changes there.

    Why: a V_th > 0 diode with a capacitor across it (a snubber, a
    junction capacitance) on the fixed engine. The ON step (``g_on``
    across ``Cs``, τ = 0.1 ns ≪ dt) is reflected by the trapezoidal
    rule from ``v_D ≈ V_th`` to ``≈ −V_th``; the diode drops to OFF,
    the inductor current recharges ``Cs`` up to ``V_th`` over a few
    steps, and the cycle repeats through the whole conduction
    interval. On the round-2 snubbed boost (Cs = 100 nF across D1,
    V_th = 0.7, Q_rr 50 nC) every gap followed more than Q_rr of
    forward charge, so 4580 "turn-offs" were charged and f_sw read
    1.53 MHz on a 100 kHz converter (15887 at dt = T/2000, 299 at
    V_th = 0), with no warning. The gaps last ``≈ 2·V_th·Cs / i_L`` in
    time, so their sample count grows as dt shrinks and no length
    rule separates them. The reflected sample sits at ``2·i/g_on −
    V_th − δ`` (δ: how far past V_th the crossing solve went, up to one
    step's rise Δv), so the lowest step average is about ``−V_th −
    Δv/2``: measured −0.34 / −0.62 / −0.62 V at dt = T/100 / T/400 /
    T/2000, too close to −V_th = −0.7 V for a ±V_th rule, hence 2·V_th
    (good for Δv up to 2·V_th per step). A real turn-off reaches the
    circuit's reverse voltage (−24 to −90 V there; −43 V on an RL
    half-wave), and every turn-off of a switched converter that spans
    a gate edge is excluded by ``other_flip`` whatever its voltage.

    The stamped state itself is left alone: duty and conduction loss
    report what the solver carried, and :func:`device_loss_summary`
    warns (:func:`_warn_chatter`)."""
    cond = np.asarray(trace.conducting, dtype=bool).copy()
    n = cond.size
    if n < 3:
        return cond, 0
    V_th = max(float(getattr(trace, "V_th", 0.0)), 0.0)
    v = np.asarray(trace.v, dtype=float)
    v_s = np.asarray(getattr(trace, "v_start", v), dtype=float)
    ks = np.nonzero(cond[:-1] & ~cond[1:])[0] + 1
    on_idx = np.nonzero(cond)[0]
    ends = np.append(on_idx, n)[np.searchsorted(on_idx, ks)]
    # Step averages: step j spans (t[j-1], t[j]], from v_s[j-1] to v[j].
    mid = np.empty(n)
    mid[0] = v[0]
    mid[1:] = 0.5 * (v_s[:-1] + v[1:])
    n_gaps = 0
    for k, end in zip(ks.tolist(), ends.tolist()):
        # Steps k .. end: from the last conducting sample k-1 to the
        # next conducting sample `end` (none: the run ends blocking).
        if end >= n:
            continue
        if other_flip is not None and bool(other_flip[k:end + 1].any()):
            continue
        if float(mid[k:end + 1].min()) >= -2.0 * V_th:
            cond[k:end] = True
            n_gaps += 1
    return cond, n_gaps


def _diode_turn_offs(trace, times: np.ndarray,
                     t_rr: float = _DEFAULT_T_RR,
                     other_flip: "Optional[np.ndarray]" = None,
                     ) -> List[Dict[str, float]]:
    """Turn-off instants of one diode, read from its recorded
    conduction-state trace (``pulsim._diode_state``).

    One record per end of a conducting stretch (a maximal run of
    ``DiodeStateTrace.conducting`` steps): ``t`` is the first sample
    the diode blocks, ``t_off`` the end of its last conducting step
    (``t[k-1]``), ``I_F`` the forward current on the last conducting
    sample, ``I_F_peak`` the largest forward current of the stretch,
    ``Q_F`` the forward charge it conducted (``max(i_D, 0)`` over the
    stretch's steps, each from its start ``current_start`` to its end),
    ``material_current`` whether ``I_F_peak >= 1e-3 · i_peak`` of the
    whole run (the first cut of the count-only rule of
    :func:`_diode_switching_loss`),
    ``V_R`` the reverse
    voltage the diode recovers against, and ``unresolved`` whether the
    time grid can place that reading.

    ``V_R`` is the reverse voltage ``max(−v_D, 0)`` one recovery time
    after the turn-off, at ``t_off + t_rr`` (clipped to the blocking
    interval): a PHYSICAL
    instant, the same on every engine and step size. It is read from
    the blocking samples only (:func:`_reverse_voltage_at`), paired
    into step midpoints so the fixed engine's trapezoidal alternation
    cancels: where a turned-off diode leaves a node held only by
    ``g_off``, the stiff L·g_off mode (τ ≈ 0.2 ps) is not damped and
    the node alternates ``V_s ± A`` on successive samples.

    Why not a window counted in steps (the previous rule, two MEAN
    steps of the run after the first blocking sample): that window is
    a different physical interval on every grid. Where ``v_D`` rises
    slowly after a natural current zero (a line rectifier) it read
    1.66 mV on ``engine='pwl'`` at dt = 1 µs and 1.81 V on
    ``engine='auto'`` (200 µs steps there), E_rr 1087× apart; on a
    snubbed DCM boost the window reached into the L–Cs ring and E_rr
    grew 8× from dt = T/1000 to T/20. Nor the first blocking sample:
    ``engine='auto'`` lands the turn-off and then takes femtosecond
    steps through the g_off transient, where single samples read
    thousands of volts; those steps carry no time weight here.

    ``unresolved`` is set when the step that carries the turn-off,
    ``t[k] − t[k−1]``, is longer than ``t_rr``: the solver then says
    nothing about how ``v_D`` rose inside that step, and V_R (hence
    E_rr) depends on the step size. The one exception is a FORCED
    turn-off (another bit of the recorded mask flips on the same step,
    ``other_flip[k]``, and ``I_F >= 0.1 · I_F_peak``) whose blocking
    voltage is flat
    after the step: there the voltage jumped at the turn-off and any
    reading inside the step is the plateau. :func:`device_loss_summary`
    counts and warns about the unresolved ones.

    ``Q_F`` exists for the stretches that are not a
    forward conduction worth a recovery: on an RL half-wave rectifier
    at ``V_th = 0.7`` the fixed engine's trapezoidal g_off/L stiff mode
    turns the reverse-biased diode ON for single steps at ~36 µA
    (5.6e-6 · i_peak, ~7e-11 C each). Charging each of those the full
    ``Q_rr · V_R`` inflated ``E_sw_total`` 4× (48 "turn-offs" instead
    of 4). See :func:`_diode_switching_loss`.

    Gaps that :func:`_chatter_gaps` identifies as chatter inside one
    conduction are filled first, so the stretches on either side are
    one stretch with one turn-off (its ``Q_F`` the sum)."""
    # Transitions of the per-step CONDUCTING flag, not of the raw
    # stamp: a V_th = 0 diode the fixed engine chatters ON/OFF at
    # ~1e-7 A is not recovering from a forward current each time.
    closed, _ = _chatter_gaps(trace, other_flip)
    n = closed.size
    if n < 2:
        return []
    t = np.asarray(times, dtype=float)
    ks = np.nonzero(closed[:-1] & ~closed[1:])[0] + 1
    v = np.asarray(trace.v, dtype=float)
    i = np.asarray(trace.current, dtype=float)
    i_fwd = np.maximum(i, 0.0)
    # Where each step starts (the right limit at an engine='dsed'
    # event; the sample itself elsewhere).
    i_s = np.asarray(getattr(trace, "current_start", i), dtype=float)
    i_fwd_s = np.maximum(i_s, 0.0)
    i_pk = float(np.max(np.abs(i))) if i.size else 0.0
    flat_atol = _RR_FLAT_FLOOR * (float(np.max(np.abs(v))) if v.size
                                  else 0.0)
    # Forward charge per step (t[j-1], t[j]], cumulated, so a stretch's
    # charge is a difference of two entries.
    dq = np.zeros(n)
    dq[1:] = 0.5 * (i_fwd_s[:-1] + i_fwd[1:]) * np.diff(t)
    q_cum = np.cumsum(dq)
    # Index of the next ON sample after each turn-off (n if none), and
    # the first conducting step of each stretch.
    on_idx = np.nonzero(closed)[0]
    ends = np.append(on_idx, n)[np.searchsorted(on_idx, ks)]
    starts_all = np.nonzero(closed & ~np.concatenate(([False],
                                                       closed[:-1])))[0]
    out: List[Dict[str, float]] = []
    for k, end in zip(ks.tolist(), ends.tolist()):
        k1 = int(starts_all[np.searchsorted(starts_all, k) - 1])
        # Conducting steps k1 .. k-1 (step j spans (t[j-1], t[j]]).
        Q_F = float(q_cum[k - 1] - (q_cum[k1 - 1] if k1 > 0 else 0.0))
        lo_s = max(k1 - 1, 0)
        I_F_peak = float(max(i_fwd[lo_s:k].max(), i_fwd_s[lo_s:k].max()))
        # The blocking interval runs from t_off to the last blocking
        # sample before the diode conducts again (or the run ends).
        t_off = float(t[k - 1])
        t_blk = float(t[end - 1])
        t_eval = min(t_off + t_rr, t_blk)
        v_read, flat = _reverse_voltage_at(t, v, k, end, t_eval,
                                           flat_atol)
        # Reverse bias only: a diode that stopped conducting inside its
        # 0 < v_D < V_th band blocks no voltage yet.
        V_R = max(-v_read, 0.0)
        I_F = float(abs(i[k - 1]))
        # A turn-off step longer than t_rr hides how v_D rose. That is
        # harmless only for a FORCED commutation (another switch cut
        # the diode off while it still carried current, so the reverse
        # voltage appears at once in the ideal-switch model) whose
        # post-turn-off plateau is flat. A natural one (current decayed
        # to its zero) rises with the circuit's own dynamics, which a
        # step longer than t_rr cannot show: a snubbed DCM boost at
        # dt = T/20 aliased its 2 µs L-Cs ring into a flat 54 V
        # "plateau" against 1.9 V resolved. Forced = another bit of the
        # recorded stamped mask flipped on the same step, with forward
        # current still flowing; without a record, the current alone.
        forced = I_F >= _RR_FORCED_FRACTION * I_F_peak
        if other_flip is not None:
            forced = forced and bool(other_flip[k])
        # (1e-9 relative slack: dt = T/100 must not miss t_rr = T/100
        # by a rounding error of the stored times.)
        unresolved = (float(t[k] - t[k - 1]) > t_rr * (1.0 + 1e-9)
                      and not (forced and bool(flat)))
        out.append({"t": float(t[k]), "t_off": t_off, "V_R": float(V_R),
                    "I_F": I_F, "forced": bool(forced),
                    "I_F_peak": I_F_peak, "Q_F": Q_F,
                    "unresolved": bool(unresolved),
                    "material_current": bool(
                        I_F_peak >= _RR_MATERIAL_CURRENT_FRACTION * i_pk)})
    return out


def _recovery_time(spec: Mapping[str, Any], name: str) -> float:
    """``t_rr`` of a diode_specs entry (s), or the documented default."""
    raw = spec.get("t_rr")
    if raw is None:
        return _DEFAULT_T_RR
    t_rr = float(raw)
    if not (np.isfinite(t_rr) and t_rr > 0):
        raise ValueError(
            f"diode_specs[{name!r}]: t_rr must be a positive time in "
            f"seconds, got {raw!r}.")
    return t_rr


def _warn_recovery_quality(name: str, entry: Mapping[str, Any],
                           t_rr: float) -> None:
    """One RuntimeWarning per diode for each way its E_rr total is not
    a property of the circuit alone (see _diode_turn_offs and
    _diode_switching_loss)."""
    n = int(entry.get("n_turn_off_events", 0))
    n_unres = int(entry.get("n_turn_off_events_unresolved", 0))
    n_cap = int(entry.get("n_turn_off_events_capped", 0))
    n_drop = int(entry.get("n_turn_off_events_dropped", 0))
    E_drop = float(entry.get("E_sw_dropped", 0.0))
    E_tot = float(entry.get("E_sw_total", 0.0))
    if n_drop > 0 and E_drop > 0.01 * E_tot:
        warnings.warn(
            f"device_loss_summary: {n_drop} conduction stretches of "
            f"{name!r} conducted less than 1e-3 of the recovery charge "
            f"and were not counted as turn-offs, but together they "
            f"would add {E_drop:.3g} J (E_sw_total {E_tot:.3g} J). So "
            f"many tiny re-conductions usually mean a ringing or "
            f"chatter the time grid does not resolve; refine dt or use "
            f"engine='auto' to check.",
            RuntimeWarning, stacklevel=3)
    if n_unres > 0 and float(entry.get("E_sw_total", 0.0)) > 0:
        warnings.warn(
            f"device_loss_summary: {n_unres} of {n} reverse-recovery "
            f"turn-offs of {name!r} happen on a step longer than "
            f"t_rr = {t_rr:.3g} s. V_R is read at t_off + t_rr, inside "
            f"that step, where the solver has no information on how "
            f"v_D rose, so E_rr there depends on the step size. Use a "
            f"step no longer than t_rr around the turn-offs "
            f"(engine='pwl' with dt <= t_rr, or a smaller dt_max on "
            f"the adaptive engines), or give the diode's datasheet "
            f"t_rr in diode_specs if it is longer than {t_rr:.3g} s.",
            RuntimeWarning, stacklevel=3)
    if n_cap > 0:
        warnings.warn(
            f"device_loss_summary: {n_cap} of {n} turn-offs of {name!r} "
            f"conducted less forward charge than the recovery charge "
            f"and were capped at Q_F * V_R. Short re-conductions like "
            f"these usually come from a ringing the time grid does not "
            f"resolve (for example a snubber whose time constant is "
            f"shorter than dt); they are also counted in "
            f"n_turn_off_events and f_sw_estimate. Refine dt or use "
            f"engine='auto' to check.",
            RuntimeWarning, stacklevel=3)


def _warn_chatter(name: str, n_gaps: int, n_turn_offs: int,
                  t_gaps: float, t_cond: float, V_th: float) -> None:
    """RuntimeWarning when a diode's record chatters inside its
    conduction (:func:`_chatter_gaps`) for a material share of it: at
    least 3 gaps, at least one per real turn-off, covering at least 1 %
    of the conduction time. (The femtosecond landing gaps of
    ``engine='auto'`` cover ~1e-10 of it; the snubbed fixed-engine
    chatter 28-56 % in ~14 gaps per conduction. An RL half-wave at
    V_th = 2 merges 3 in-band waits before its 4 turn-ons, 1.4 % of the
    time: harmless for the count, and not chatter.)"""
    if (n_gaps < 3 or n_gaps < n_turn_offs
            or not (t_gaps >= 0.01 * t_cond)):
        return
    warnings.warn(
        f"device_loss_summary: the solver stamped {name!r} OFF for "
        f"{n_gaps} gaps ({100.0 * t_gaps / t_cond:.3g} % of its "
        f"conduction time) in which the diode never blocked more than "
        f"2·V_th = {2.0 * V_th:.3g} V of reverse voltage ({n_turn_offs} "
        f"real turn-offs). That is "
        f"numerical chatter, typically a "
        f"capacitor across a V_th > 0 diode whose fast mode the fixed "
        f"engine's trapezoidal rule leaves undamped. The gaps are not "
        f"counted as turn-offs, but the diode's current, duty and "
        f"conduction loss follow the chattering stamp (the current "
        f"detours through the capacitor in each gap). A smaller dt "
        f"does not help (the mode is ~g_on·C fast; the gaps covered 28 % "
        f"of the conduction at dt = T/400 and 56 % at T/2000 in one "
        f"boost): check with engine='auto', whose step control damps "
        f"it, or put the capacitor's series resistance in the model.",
        RuntimeWarning, stacklevel=3)


def _diode_switching_loss(turn_offs: List[Dict[str, float]],
                          times: np.ndarray,
                          v_D: np.ndarray,
                          spec: Mapping[str, Any],
                          i_D: "Optional[np.ndarray]" = None,
                          ) -> Dict[str, float]:
    """Post-hoc reverse-recovery switching loss at each diode turn-off.

    Mirrors PSIM's diode-loss panel — two equivalent entry points:

    * **Raw ``Q_rr``** — loss per event ``E_rr = Q_rr · |V_R|`` (the
      simplest pattern; assumes ``Q_rr`` is roughly constant over
      the operating range, valid for hard-recovery diodes).
    * **Datasheet pair ``(E_rr_ref, V_R_ref)``** — loss per event
      ``E_rr = E_rr_ref · (|V_R| / V_R_ref)``. Use this when the
      datasheet quotes ``E_rr`` (energy per pulse) at a specific
      reference voltage, which is more common for fast/soft recovery
      diodes.

    Either form yields the same number when
    ``E_rr_ref = Q_rr · V_R_ref``. ``turn_offs`` comes from
    :func:`_diode_turn_offs`: ``V_R`` is the settled reverse voltage
    the diode blocks after the turn-off — the voltage it recovers
    against (full ``V_in`` for a hard-switched freewheel diode, near
    zero for a naturally commutated line rectifier).

    ``V_R`` is read one recovery time ``t_rr`` after the turn-off
    (``spec["t_rr"]``, default 100 ns); turn-offs whose own step is
    longer than ``t_rr`` cannot place that reading and are counted in
    ``n_turn_off_events_unresolved``.

    Two guards keep immaterial conduction out of the total, and both
    are reported:

    * a stretch whose forward charge ``Q_F`` is below ``1e-3`` of the
      recovery charge the spec asks for (``Q_rr``, ``E_rr_ref /
      V_R_ref``) is not a turn-off
      at all (``n_turn_off_events_dropped``; excluded from
      ``n_turn_off_events``, ``f_sw_estimate`` and the total). With
      the cap below, such a stretch could add at most 1e-3 of one
      real event's energy, so dropping it cannot change the total
      materially; ``E_sw_dropped`` reports what the dropped stretches
      would have added, and a RuntimeWarning fires if that exceeds
      1 % of the total. A spec with no recovery at all (counts only)
      judges the peak forward current ``I_F_peak`` instead, against
      ``1e-3`` of the median ``I_F_peak`` of the stretches that reach
      ``1e-3`` of the run's peak current. An ``E_rr_curve`` spec judges
      ``I_F_peak`` against ``1e-3`` of the curve's smallest tabulated
      positive current: the curve's own axis, on which E_rr itself is
      read (at ``I_F``, the current the diode carries when it is cut
      off). A curve's charge would scale with the stretch's own
      current, so a curve through the origin would ask nothing of a
      chatter blip and count every one.
      The materiality is judged against the event's own charge, not
      against the run's peak current: a boost with a 4 kA start-up
      inrush dropped all 237 of its 3-4 A light-load recoveries
      (each above Q_rr) that way, E_sw 4.9× and f_sw 5× low, silently;
    * no event recovers more energy than ``Q_F · V_R``, the stretch's
      own forward charge recovered against ``V_R``
      (``n_turn_off_events_capped``).

    The ideal ``SwitchedDiode`` kernel model has no physical
    recovery charge; this is a datasheet annotation, matching
    PSIM / LTspice "ideal-diode-with-Qrr" post-processing.
    """
    n_all = len(turn_offs)
    T = (times[-1] - times[0]) if times.size > 1 else 0.0
    Q_rr = float(spec.get("Q_rr", 0.0))
    E_rr_ref = float(spec.get("E_rr_ref", 0.0))
    V_R_ref = float(spec.get("V_R_ref", 0.0))
    E_rr_curve = spec.get("E_rr_curve")
    if E_rr_curve is not None and V_R_ref <= 0:
        raise ValueError(
            "diode_specs: E_rr_curve requires a positive V_R_ref.")
    if E_rr_curve is None and E_rr_ref > 0 and V_R_ref <= 0:
        raise ValueError(
            "diode_specs: E_rr_ref requires a positive V_R_ref "
            "(the reference reverse voltage from the datasheet).")
    has_rr = Q_rr > 0 or E_rr_ref > 0 or E_rr_curve is not None
    # The current axis of an E_rr_curve: its smallest tabulated
    # positive current (None for a curve with no positive current).
    I_tab = None
    if E_rr_curve is not None:
        I_pos = [float(i) for i, _ in E_rr_curve if float(i) > 0]
        I_tab = min(I_pos) if I_pos else None

    def _q_req(evs) -> np.ndarray:
        """Recovery charge the spec asks of each event (the energy it
        charges, before the Q_F cap, divided by V_R)."""
        if E_rr_curve is not None:
            I_F = np.asarray([ev["I_F"] for ev in evs], dtype=float)
            return _interp_energy_curve(I_F, E_rr_curve) / V_R_ref
        if E_rr_ref > 0:
            return np.full(len(evs), E_rr_ref / V_R_ref)
        return np.full(len(evs), Q_rr)

    def _peak(evs) -> np.ndarray:
        return np.asarray([ev.get("I_F_peak", np.inf) for ev in evs],
                          dtype=float)

    if not turn_offs:
        keep = np.zeros(0, dtype=bool)
    elif E_rr_curve is not None and I_tab is not None:
        # A curve is indexed by current, so materiality is judged on
        # that axis: the stretch's peak forward current against the
        # smallest current the datasheet tabulates. Not by charge: the
        # charge a curve asks for scales with the current, and a curve
        # through the origin (PLECS tables start at (0 A, 0 J)) asks a
        # 36 µA chatter blip for 1e-12 C, so all 44 blips of an RL
        # half-wave were counted (48 turn-offs, f_sw 720 Hz on 60 Hz).
        keep = _peak(turn_offs) >= _RR_MATERIAL_CURRENT_FRACTION * I_tab
    elif has_rr:
        Q_F_all = np.asarray([ev.get("Q_F", np.inf) for ev in turn_offs],
                             dtype=float)
        if E_rr_curve is not None:
            # A curve with no positive current: a constant charge.
            q_ref = np.full(len(turn_offs), max(
                float(e) for _, e in E_rr_curve) / V_R_ref)
        else:
            q_ref = _q_req(turn_offs)
        keep = Q_F_all >= _RR_MATERIAL_FRACTION * q_ref
    else:
        # Counts only: no charge or current scale from the spec. The
        # run's peak current is the first cut; the reference is then
        # the MEDIAN peak of the stretches that pass it (the typical
        # conduction), which is never above the run's peak, so this
        # drops no stretch the run-peak rule kept. A 4 kA start-up
        # inrush set the run-peak threshold at 4 A and dropped all 237
        # of a boost's 3-4 A light-load recoveries (f_sw 20 kHz on
        # 100 kHz).
        pk = _peak(turn_offs)
        first = np.asarray([ev.get("material_current",
                                   ev.get("material", True))
                            for ev in turn_offs], dtype=bool)
        if first.any() and np.all(np.isfinite(pk)):
            ref = float(np.median(pk[first]))
            keep = pk >= _RR_MATERIAL_CURRENT_FRACTION * ref
        else:
            keep = first
    dropped = [ev for ev, k in zip(turn_offs, keep) if not k]
    turn_offs = [ev for ev, k in zip(turn_offs, keep) if k]
    n_off = len(turn_offs)
    E_dropped = 0.0
    if has_rr and dropped:
        V_R_d = np.asarray([ev["V_R"] for ev in dropped], dtype=float)
        Q_F_d = np.asarray([ev.get("Q_F", 0.0) for ev in dropped],
                           dtype=float)
        E_dropped = float(np.sum(np.minimum(_q_req(dropped), Q_F_d)
                                 * V_R_d))

    bookkeeping = {
        "n_turn_off_events": int(n_off),
        "n_turn_off_events_dropped": int(n_all - n_off),
        "n_turn_off_events_capped": 0,
        "n_turn_off_events_unresolved": int(sum(
            1 for ev in turn_offs if ev.get("unresolved", False))),
        "f_sw_estimate": (float(n_off) / T) if T > 0 else 0.0,
        "E_sw_dropped": E_dropped,
    }
    if n_off == 0 or not has_rr:
        return {"E_sw_total": 0.0, "P_sw_avg": 0.0, **bookkeeping}

    V_R = np.asarray([ev["V_R"] for ev in turn_offs], dtype=float)

    if E_rr_curve is not None:
        # Nonlinear E_rr(I_F): interpolate the datasheet curve at the
        # forward current the diode was carrying just before recovery.
        I_F = np.asarray([ev["I_F"] for ev in turn_offs], dtype=float)
        E_per_event = _interp_energy_curve(I_F, E_rr_curve) * (V_R / V_R_ref)
    elif E_rr_ref > 0:
        E_per_event = E_rr_ref * (V_R / V_R_ref)
    else:
        E_per_event = Q_rr * V_R
    # A diode cannot recover more charge than it conducted: cap each
    # event at Q_F · V_R, the energy of recovering ALL of the stretch's
    # forward charge (the datasheet Q_rr assumes a forward current that
    # stored at least that much).
    Q_F = np.asarray([ev.get("Q_F", np.inf) for ev in turn_offs],
                     dtype=float)
    cap = Q_F * V_R
    capped = E_per_event > cap
    E_per_event = np.where(capped, cap, E_per_event)
    bookkeeping["n_turn_off_events_capped"] = int(capped.sum())
    E_total = float(np.sum(E_per_event))
    return {
        "E_sw_total": E_total,
        "P_sw_avg": (E_total / T) if T > 0 else 0.0,
        **bookkeeping,
    }


def _switch_switching_loss(closed_trace: np.ndarray,
                            times: np.ndarray,
                            v_SW: np.ndarray,
                            i_SW: np.ndarray,
                            spec: Mapping[str, Any],
                            *,
                            closed_start: "Optional[np.ndarray]" = None,
                            v_start: "Optional[np.ndarray]" = None,
                            i_start: "Optional[np.ndarray]" = None,
                            ) -> Dict[str, float]:
    """Post-hoc turn-on / turn-off switching loss for an ideal-switch
    branch (MOSFET / IGBT equivalent).

    PSIM-style datasheet annotation: ``E_on`` and ``E_off`` are taken
    from the device datasheet at a reference operating point
    ``(V_ref, I_ref)`` and scaled linearly by both the blocking
    voltage and the conducting current at the transition::

        E_on_event  = E_on_ref  · (V_blocking / V_ref) · (I_load / I_ref)
        E_off_event = E_off_ref · (V_blocking / V_ref) · (I_load / I_ref)

    Transitions (turn-on False→True, turn-off True→False) are read
    from ``closed_trace``: the stamped mask the kernel recorded for
    every sample (``result.switch_states``) on a result from
    :func:`pulsim.simulate`, or, only for a result without that
    record, ``switch_fn`` re-evaluated after the run (with a
    RuntimeWarning).

    At each edge between samples k-1 and k, we take ``|v_SW[k-1]|``
    as the blocking voltage (the value the switch was holding off
    just before / just after the transition) and ``|i_SW[k]|`` as
    the load current the device sees while conducting.

    ``closed_start`` / ``v_start`` / ``i_start`` are the state at the
    start of the step after each sample
    (``_result_views.step_start_states``). Where the edge fired AT
    sample ``k-1`` (``engine='dsed'`` stores that event sample as the
    left limit and the switch already sits in its new state in
    ``closed_start[k-1]``), the post-edge quantity is read there, at
    the edge itself, instead of one whole step later: with 1 µs DSED
    steps the next sample is well into the on-time, and a DCM boost
    switch's E_on read 4.7 % high.
    """
    E_on_ref  = float(spec.get("E_on_ref",  0.0))
    E_off_ref = float(spec.get("E_off_ref", 0.0))
    E_on_curve = spec.get("E_on_curve")
    E_off_curve = spec.get("E_off_curve")
    # PLECS-style 3-D tables E(v, i, Tj). They take precedence over
    # the curve and the single-point form, because they carry
    # strictly more information — see `pulsim.loss_tables` for the
    # 40 % that the temperature-free model was leaving out.
    E_on_table = spec.get("E_on_table")
    E_off_table = spec.get("E_off_table")
    tj_spec = spec.get("Tj")
    if (E_on_table is not None or E_off_table is not None) \
            and tj_spec is None:
        raise ValueError(
            "switch_specs: a 3-D loss table is indexed by junction "
            "temperature, so `Tj` must be given alongside it — a "
            "scalar for a fixed junction temperature, an array on "
            "the result's time grid, or a callable t -> Tj. Without "
            "it there is no defensible temperature to read the "
            "table at, and picking one silently would hide exactly "
            "the error the table exists to fix.")
    V_ref = float(spec.get("V_ref", 0.0))
    I_ref = float(spec.get("I_ref", 0.0))
    T = (times[-1] - times[0]) if times.size > 1 else 0.0

    # Detect edges. Diff of the boolean trace gives ±1 at transitions.
    diff = np.diff(closed_trace.astype(np.int8))
    turn_on_k  = np.flatnonzero(diff == +1) + 1   # index post-edge
    turn_off_k = np.flatnonzero(diff == -1) + 1
    n_on  = int(turn_on_k.size)
    n_off = int(turn_off_k.size)
    bookkeeping = {
        "n_turn_on_events": n_on,
        "n_turn_off_events": n_off,
        "f_sw_estimate": (float(n_on + n_off) / (2.0 * T)) if T > 0
                                else 0.0,
    }
    has_on = ((E_on_ref > 0) or (E_on_curve is not None)
              or (E_on_table is not None))
    has_off = ((E_off_ref > 0) or (E_off_curve is not None)
               or (E_off_table is not None))
    if (n_on == 0 and n_off == 0) or (not has_on and not has_off):
        return {
            "E_sw_on_total": 0.0,
            "E_sw_off_total": 0.0,
            "E_sw_total": 0.0,
            "P_sw_avg": 0.0,
            **bookkeeping,
        }
    # V_ref anchors the CURVE and single-point forms only. A 3-D
    # table carries voltage as one of its own axes, so demanding a
    # reference voltage there would be asking for a number that has
    # nowhere to go.
    needs_v_ref = ((has_on and E_on_table is None)
                   or (has_off and E_off_table is None))
    if needs_v_ref and V_ref <= 0:
        raise ValueError(
            "switch_specs: V_ref must be positive — it anchors the "
            "datasheet E_on/E_off reference bus voltage. (A 3-D "
            "E_on_table / E_off_table needs no V_ref: voltage is one "
            "of the table's axes.)")

    # Blocking voltage = the OFF-state v_SW (across the open switch).
    # That's the sample BEFORE a turn-on edge and the sample AFTER a
    # turn-off edge — in both cases the switch is "open / blocking".
    # Load current = the ON-state |i_SW| in the conducting interval
    # adjacent to the edge. For turn-on that's the sample at k (just
    # after closing); for turn-off it's the sample at k-1 (just before
    # opening).
    cl = np.asarray(closed_trace, dtype=bool)
    cl_s = cl if closed_start is None else np.asarray(closed_start, bool)
    v_s = v_SW if v_start is None else np.asarray(v_start, dtype=float)
    i_s = i_SW if i_start is None else np.asarray(i_start, dtype=float)

    def _post_edge(arr: np.ndarray, arr_s: np.ndarray,
                   idxs: np.ndarray) -> np.ndarray:
        """``arr`` just after the edge between ``idxs - 1`` and
        ``idxs``: at the start of the step if the switch is already in
        its new state there, else at the end sample ``idxs``."""
        pre = np.clip(idxs - 1, 0, arr.size - 1)
        at_edge = cl_s[pre] == cl[idxs]
        return np.where(at_edge, arr_s[pre], arr[idxs])

    def _edge_loss(idxs: np.ndarray, E_ref: float, curve,
                    *, blocking_at_post_edge: bool,
                    load_at_post_edge: bool,
                    table=None) -> float:
        if idxs.size == 0 or (E_ref <= 0 and curve is None
                               and table is None):
            return 0.0
        pre_k = np.clip(idxs - 1, 0, v_SW.size - 1)
        load_k = idxs if load_at_post_edge else pre_k
        V_blk = np.abs(_post_edge(v_SW, v_s, idxs)
                       if blocking_at_post_edge else v_SW[pre_k])
        I_load = np.abs(_post_edge(i_SW, i_s, idxs)
                        if load_at_post_edge else i_SW[pre_k])
        if table is not None:
            # The 3-D table needs no V_ref scaling: the voltage IS
            # one of its axes.
            tab = _as_loss_table(table)
            tj_arr = _resolve_tj(tj_spec, times)
            return float(np.sum(tab(V_blk, I_load, tj_arr[load_k])))
        if curve is not None:
            # Nonlinear E(I) from the datasheet curve (at V_ref), scaled
            # linearly by the actual blocking voltage.
            E_base = _interp_energy_curve(I_load, curve)
            return float(np.sum(E_base * (V_blk / V_ref)))
        # Linear (single-point) datasheet scaling.
        if I_ref <= 0:
            raise ValueError(
                "switch_specs: I_ref must be positive when using the "
                "single-point E_on_ref/E_off_ref scaling.")
        return float(np.sum(E_ref * (V_blk / V_ref) * (I_load / I_ref)))

    # Turn-on:  V_blk = pre-edge (open), I_load = post-edge (just closed).
    # Turn-off: V_blk = post-edge (now open), I_load = pre-edge (was closed).
    E_on_total = _edge_loss(turn_on_k,  E_on_ref, E_on_curve,
                              blocking_at_post_edge=False,
                              load_at_post_edge=True,
                              table=E_on_table)
    E_off_total = _edge_loss(turn_off_k, E_off_ref, E_off_curve,
                              blocking_at_post_edge=True,
                              load_at_post_edge=False,
                              table=E_off_table)
    E_total = E_on_total + E_off_total
    return {
        "E_sw_on_total": E_on_total,
        "E_sw_off_total": E_off_total,
        "E_sw_total": E_total,
        "P_sw_avg": (E_total / T) if T > 0 else 0.0,
        **bookkeeping,
    }


def _resolve_core_material(spec: Mapping[str, Any]) -> Any:
    """Pull a ``CoreMaterial`` out of a core_loss_specs entry. Accepts
    either ``{"material": "<catalog-name>"}`` or
    ``{"material": <CoreMaterial>}`` or
    ``{"K": ..., "alpha": ..., "beta": ...}`` (inline triplet)."""
    mat = spec.get("material")
    if mat is None:
        if "K" in spec and "alpha" in spec and "beta" in spec:
            return _magnetic.CoreMaterial(
                name=str(spec.get("name", "inline")),
                K=float(spec["K"]),
                alpha=float(spec["alpha"]),
                beta=float(spec["beta"]),
            )
        raise ValueError(
            "core_loss_specs entry must include either 'material' "
            "(name or CoreMaterial) or an inline {K, alpha, beta} "
            "triplet.")
    if isinstance(mat, str):
        return _magnetic.core_material(mat)
    return mat


def _inductor_core_loss(i_arr: np.ndarray,
                         times: np.ndarray,
                         L: float,
                         spec: Mapping[str, Any]) -> Dict[str, float]:
    """Compute Steinmetz / iGSE core loss for one inductor given the
    current trace + a per-device spec.

    Spec required keys:
      * ``N_turns``  — winding turn count.
      * ``A_core``   — magnetic cross-section (m²).
      * ``V_core``   — core volume (m³).
      * either ``material`` (str or CoreMaterial) or
        inline ``{K, alpha, beta}``.
    Spec optional keys:
      * ``use_igse`` (default True) — iGSE on the trace; falls back
        to classic Steinmetz at the dominant frequency when False.
      * ``f_op`` — frequency override for Steinmetz mode (Hz). If
        omitted, estimated from the trace via the largest non-DC FFT
        bin.
    """
    N = float(spec["N_turns"])
    A = float(spec["A_core"])
    V = float(spec["V_core"])
    if N <= 0 or A <= 0 or V <= 0:
        raise ValueError(
            f"core_loss_specs requires N_turns, A_core and V_core "
            f"all strictly positive; got N={N}, A={A}, V={V}.")
    material = _resolve_core_material(spec)

    # Reconstruct flux density B(t) = L · i(t) / (N · A).
    B = (float(L) * i_arr) / (N * A)
    B_peak = float(np.max(np.abs(B))) if B.size else 0.0

    use_igse = bool(spec.get("use_igse", True)) and times.size >= 4
    P_v: float = 0.0
    if use_igse:
        try:
            P_v = float(_magnetic.igse_loss_density(B, times, material))
        except ValueError:
            # iGSE rejects traces shorter than one period or with
            # non-positive ΔB (DC-only signal). Fall through to the
            # Steinmetz path, which handles those degenerate cases
            # gracefully.
            use_igse = False
    if not use_igse:
        # Steinmetz needs a frequency. Estimate from FFT if not given.
        f_op = spec.get("f_op")
        if f_op is None:
            f_op = _estimate_dominant_frequency(B, times)
        P_v = float(_magnetic.steinmetz_loss_density(
            B_peak, float(f_op), material))
    P_core = float(P_v * V)
    T = (times[-1] - times[0]) if times.size > 1 else 0.0
    return {
        "P_core_avg": P_core,
        "E_core_total": P_core * T,
        "B_peak": B_peak,
    }


def _estimate_dominant_frequency(B: np.ndarray,
                                  times: np.ndarray) -> float:
    """Estimate the dominant frequency in B(t) via the FFT, ignoring
    the DC bin. Returns 0 if the trace is too short or flat."""
    if times.size < 4 or B.size < 4:
        return 0.0
    # A variable-step grid has no single sample rate; resample
    # before inferring one (no-op on a fixed grid).
    from ._result_views import grid_is_uniform, resample_uniform
    if not grid_is_uniform(times):
        times, B, dt = resample_uniform(times, B)
    else:
        dt = float(np.mean(np.diff(times)))
    if dt <= 0:
        return 0.0
    # Subtract mean to suppress DC.
    spec = np.abs(np.fft.rfft(B - B.mean()))
    freqs = np.fft.rfftfreq(B.size, d=dt)
    if spec.size <= 1:
        return 0.0
    # Find the dominant non-DC bin. argmax over spec[1:] then offset.
    idx = int(np.argmax(spec[1:])) + 1
    return float(freqs[idx])


# ---------------------------------------------------------------
# Top-level summary.
# ---------------------------------------------------------------

def device_loss_summary(
    builder,
    result,
    *,
    switch_fn: Optional[Callable[[float], Any]] = None,
    core_loss_specs: Optional[Mapping[Any, Mapping[str, Any]]] = None,
    diode_specs: Optional[Mapping[Any, Mapping[str, Any]]] = None,
    switch_specs: Optional[Mapping[Any, Mapping[str, Any]]] = None,
    conduction_specs: Optional[Mapping[Any, Mapping[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Walk every resistor, inductor, ideal-switch and switched-diode
    branch in ``builder`` and report each one's conduction
    statistics.

    Supported device kinds:

    * **resistor** — current is reconstructed from the node-voltage
      difference and ``R_ohms``. Reports ``i_avg``, ``i_rms``,
      ``i_peak`` plus ``P_avg`` and ``E_total``.
    * **inductor** — current is in the state vector directly.
      Reports the same current metrics. If a matching entry exists
      in ``core_loss_specs``, the entry also gets ``P_core_avg``,
      ``E_core_total`` and ``B_peak`` (Steinmetz / iGSE).
    * **diode** (ideal switched diode with ``g_on`` / ``g_off``) —
      the per-sample conduction state is the stamp the solver recorded
      (:mod:`pulsim._diode_state`; NOT ``v_D > V_th`` — a conducting
      PWL diode drops ``i / g_on``, far below ``V_th``). Reports the
      same current + power metrics as the resistor path, plus
      ``duty_conducting`` (time-weighted fraction of the run carrying
      forward current). A ``RuntimeWarning`` names the diode when the
      result carries no record (the state is then replayed from the
      kernel rule) or when the engine held it across its own
      commutation rule.
    * **switch** (ideal switch with ``g_on`` / ``g_off``) — the
      recorded stamped mask; without a record, ``switch_fn`` is
      re-evaluated (with a warning), and without either the switch
      entry is skipped (kept honest — no silent default).

    A run seeded with ``initial_state`` (or resumed from a snapshot
    without ``stamped_mask``) has an UNSOLVED row 0
    (``result.switch_states_unsolved``): its state came from a solve
    whose mask the record does not know, so no switched-branch current
    is defined there. Every statistic, average and integral then runs
    from the first solved sample to the end of the run (one step less
    than the run on a fixed grid); see
    ``pulsim._result_views.solved_part``.

    Parameters
    ----------
    builder, result
        A populated :class:`CircuitBuilder` and the
        :class:`SimulationResult` produced by it.
    switch_fn : callable, optional
        ``t -> SwitchStateMask`` used during the simulation. Not needed
        for a result of :func:`pulsim.simulate`: switch states are read
        from the recorded stamped mask (``result.switch_states``).
        Only a result without that record falls back to re-evaluating
        it (exact for a stateless switch_fn; a RuntimeWarning says
        so).
    core_loss_specs : mapping, optional
        Per-inductor core-loss configuration. Keys are inductor
        branch *names* (preferred, case-sensitive) or numeric
        branch ids. **Unknown names / ids raise ``KeyError``** —
        the helper refuses to silently ignore a typo. Values are
        mappings with::

            {"material": "N87" | CoreMaterial,
             "N_turns": int, "A_core": float (m²),
             "V_core": float (m³),
             "use_igse": bool = True,
             "f_op": float (Hz)  # optional, Steinmetz fallback}

        An inline triplet ``{"K": ..., "alpha": ..., "beta": ...}``
        is also accepted in place of ``material``.
    diode_specs : mapping, optional
        Per-diode datasheet annotations for reverse-recovery
        switching loss (PSIM-style). Keys are diode branch names or
        numeric branch ids. Two equivalent entry points::

            # Raw recovery charge — assumes Q_rr ~ constant.
            {"Q_rr": float (C)}

            # Datasheet energy + reference voltage — preferred for
            # fast / soft-recovery diodes that quote E_rr directly.
            {"E_rr_ref": float (J), "V_R_ref": float (V)}

            # Nonlinear datasheet E_rr(I_F) curve at V_R_ref — the most
            # faithful: E_rr is interpolated at the forward current the
            # diode carried just before each recovery.
            {"E_rr_curve": [(I_F, E_rr), …], "V_R_ref": float (V)}

        Any form also accepts ``"t_rr": float (s)``, the reverse
        recovery time (default 100 ns). For each end of a conducting
        stretch of the diode's recorded state the helper takes the
        reverse voltage ``V_R = max(−v_D, 0)`` at ``t_off + t_rr`` (one
        recovery time after
        the turn-off, a physical instant independent of the step size)
        and accumulates ``E_rr_event = Q_rr · V_R`` or
        ``E_rr_event = E_rr_ref · (V_R / V_R_ref)``, capped at the
        stretch's own forward charge times ``V_R``; stretches whose
        forward charge is below ``1e-3`` of the recovery charge asked
        for are dropped (by peak forward current when the spec is an
        ``E_rr_curve`` or gives no recovery, see
        ``_diode_switching_loss``). A blocking gap across which no step
        average of ``v_D`` falls below ``−2·V_th`` is chatter inside one
        conduction, not a turn-off (every diode entry reports
        ``n_chatter_gaps``; a RuntimeWarning fires when they cover
        1 % or more of the conduction time). Reports ``E_sw_total``, ``P_sw_avg``,
        ``n_turn_off_events``, ``n_turn_off_events_dropped``,
        ``E_sw_dropped``, ``n_turn_off_events_capped``,
        ``n_turn_off_events_unresolved``, ``f_sw_estimate`` and the
        ``t_rr`` used. A RuntimeWarning names the diode when a
        turn-off falls on a step longer than ``t_rr`` (E_rr then
        follows the step size) or when events were capped.
    switch_specs : mapping, optional
        Per-switch datasheet annotations for MOSFET / IGBT-style
        turn-on / turn-off switching loss (PSIM-style). Keys are
        switch branch names or branch ids. Values::

            {"E_on_ref":  float (J),
             "E_off_ref": float (J),
             "V_ref":     float (V),
             "I_ref":     float (A)}

        Or, for the nonlinear datasheet energy-vs-current curves
        (interpolated at the actual switched current per event, linearly
        extrapolated beyond the tabulated range), supply ``E_on_curve``
        / ``E_off_curve`` instead of the single-point ``E_on_ref`` /
        ``E_off_ref`` (``I_ref`` is then unnecessary)::

            {"E_on_curve":  [(I, E_on), …],
             "E_off_curve": [(I, E_off), …],
             "V_ref":       float (V)}

        Transitions are read from the stamped mask the kernel
        recorded for every sample (``result.switch_states``); no
        ``switch_fn`` is needed for a result of
        :func:`pulsim.simulate`. Only a result without that record
        falls back to re-evaluating ``switch_fn`` (and warns). Per
        event::

            E_on  = E_on_ref  · (|V_blk| / V_ref) · (|I_load| / I_ref)
            E_off = E_off_ref · (|V_blk| / V_ref) · (|I_load| / I_ref)

        ``V_blk`` is the sample just before the edge,
        ``I_load`` is the sample just after. Reports
        ``E_sw_on_total``, ``E_sw_off_total``, ``E_sw_total``,
        ``P_sw_avg``, plus ``n_turn_on_events`` /
        ``n_turn_off_events`` / ``f_sw_estimate``.
    conduction_specs : mapping, optional
        Per-diode / per-switch offset+slope conduction model. Keys are
        device names or branch ids. Values give the PWL device
        characteristic (datasheet aliases accepted)::

            {"V_f0": float (V),   # forward-voltage offset (or V_ce0)
             "r_on": float (Ω)}   # ohmic slope        (or r_ce / r_d)

        When present, the entry gains ``P_cond_model_avg`` /
        ``E_cond_model`` computed as ``V_f0·|i| + r_on·i²`` over the
        device's actual current trace — strictly more accurate than the
        pure-resistive ``v²·g`` ``P_avg`` for IGBTs and diodes (the
        ``V_f0`` offset dominates conduction loss at low current). MOSFETs
        set ``V_f0 = 0`` and use ``r_on`` alone.

    Returns
    -------
    list of dict
        One entry per known device in branch-id order. Unknown
        kinds (sources, capacitors, ...) are skipped silently.
    """
    # From the first solved sample on: a seeded, unsolved row 0 has no
    # defined switched-branch current (_result_views.solved_part).
    result = _views.solved_part(result)
    states = _states_as_array(result)
    times = _times_as_array(result)
    T = (times[-1] - times[0]) if times.size > 1 else 1.0

    try:
        comps_by_bid = {c["branch_id"]: c for c in builder.components()}
    except Exception:
        comps_by_bid = {}
    known_names = {c.get("name") for c in comps_by_bid.values()
                     if c.get("name")}
    known_bids = set(comps_by_bid)

    def _build_spec_lookup(
        specs: Optional[Mapping[Any, Mapping[str, Any]]],
        kw: str,
    ) -> tuple[Dict[int, Mapping[str, Any]],
                  Dict[str, Mapping[str, Any]]]:
        """Split a ``{name|bid: spec}`` mapping into name- and
        bid-keyed lookups, raising loudly on unknown references."""
        by_bid: Dict[int, Mapping[str, Any]] = {}
        by_name: Dict[str, Mapping[str, Any]] = {}
        if specs is None:
            return by_bid, by_name
        for key, spec in specs.items():
            if isinstance(key, int):
                if key not in known_bids:
                    raise KeyError(
                        f"{kw}: branch_id={key!r} not found in the "
                        f"builder (known ids: {sorted(known_bids)}).")
                by_bid[key] = spec
            else:
                k_str = str(key)
                if k_str not in known_names:
                    raise KeyError(
                        f"{kw}: no device named {k_str!r} in the "
                        f"builder (known names: {sorted(known_names)}).")
                by_name[k_str] = spec
        return by_bid, by_name

    # Build per-spec lookups (name + bid keyed) with strict validation
    # — unknown references are user errors, not silent skips.
    core_by_bid, core_by_name = _build_spec_lookup(
        core_loss_specs, "core_loss_specs")
    diode_by_bid, diode_by_name = _build_spec_lookup(
        diode_specs, "diode_specs")
    switch_by_bid, switch_by_name = _build_spec_lookup(
        switch_specs, "switch_specs")
    cond_by_bid, cond_by_name = _build_spec_lookup(
        conduction_specs, "conduction_specs")

    # The kernel's per-sample record of the stamped mask (None for a
    # result that carries none) and each switch-kind branch's bit.
    kernel_masks, _ = _views.recorded_switch_states(result, builder)
    try:
        switch_bits = _views.switch_bits_by_branch(builder)
    except Exception:  # noqa: BLE001 — a duck-typed builder
        switch_bits = {}
    # Where each step STARTS (the right limit of the sample before it;
    # the sample itself except at an engine='dsed' event). Every time
    # integral below runs each step from there to its end sample.
    states_s, masks_s = _views.step_start_states(result, states,
                                                 kernel_masks)

    summary: List[Dict[str, Any]] = []
    switch_seq_idx = 0  # increments once per Switch / Diode branch
                        # in branch-id order — matches the kernel
                        # convention in pulsim/pwl/assemble.hpp.

    for bid in range(builder.graph.num_branches):
        desc = comps_by_bid.get(bid, {})
        kind = desc.get("kind", "unknown")
        name = desc.get("name", "")

        if kind == "inductor":
            try:
                i_idx = builder.pool.branch_var_id_for_inductor(
                    bid, builder.graph)
            except Exception:
                continue
            i_arr = states[:, i_idx]
            i_arr_s = states_s[:, i_idx]
            entry: Dict[str, Any] = {
                "branch_id": int(bid),
                "kind": "inductor",
                "name": name,
                "i_avg": (_views.pl_integral(times, i_arr, i_arr_s) / T
                          if T > 0 and times.size > 1
                          else float(i_arr.mean())),
                "i_rms": _time_weighted_rms(i_arr, times, i_arr_s),
                "i_peak": float(np.abs(i_arr).max()),
            }
            # Optional Steinmetz / iGSE core loss.
            spec = core_by_bid.get(bid) or core_by_name.get(name)
            if spec is not None:
                L = float(desc.get("params", {}).get(
                    "L_henries", float("nan")))
                if np.isfinite(L) and L > 0:
                    entry.update(
                        _inductor_core_loss(i_arr, times, L, spec))
            summary.append(entry)
            continue

        if kind == "resistor":
            R = float(desc.get("params", {}).get("R_ohms", float("nan")))
            if not np.isfinite(R) or R <= 0:
                continue
            from_id, to_id = desc.get("nodes", (None, None))
            if from_id is None or to_id is None:
                continue
            v_from = _node_voltage_trace(states, int(from_id))
            v_to   = _node_voltage_trace(states, int(to_id))
            v_R = v_from - v_to
            v_R_s = (_node_voltage_trace(states_s, int(from_id))
                     - _node_voltage_trace(states_s, int(to_id)))
            stats = _conduction_stats(v_R, 1.0 / R, times, v_R_s)
            summary.append({
                "branch_id": int(bid),
                "kind": "resistor",
                "name": name,
                "R_ohms": R,
                **stats,
            })
            continue

        if kind == "diode":
            # PWL SwitchedDiode. Its ON state is a pure conductance —
            # the drop is i/g_on whatever V_th is; V_th only gates the
            # OFF→ON commutation — so `v_D > V_th` is NOT the
            # conduction state (it called a 48 A bridge diode idle).
            # The state comes from the one shared reconstruction in
            # pulsim._diode_state, also used by result.i() and the
            # thermal summary.
            params = desc.get("params", {})
            g_on  = float(params.get("g_on", float("nan")))
            g_off = float(params.get("g_off", float("nan")))
            V_th  = float(params.get("V_th", 0.0))
            if not (np.isfinite(g_on) and np.isfinite(g_off)):
                switch_seq_idx += 1
                continue
            from_id, to_id = desc.get("nodes", (None, None))
            if from_id is None or to_id is None:
                switch_seq_idx += 1
                continue
            d_trace = _diode_state.diode_state_trace(
                builder, result, bid, switch_fn=switch_fn, stacklevel=3)
            v_D = d_trace.v
            g_arr = np.where(d_trace.closed, g_on, g_off)
            g_arr_s = np.where(d_trace.closed_start, g_on, g_off)
            stats = _conduction_stats(v_D, g_arr, times,
                                      d_trace.v_start, g_arr_s)
            entry: Dict[str, Any] = {
                "branch_id": int(bid),
                "kind": "diode",
                "name": name,
                "V_th": V_th,
                "duty_conducting": _time_fraction(d_trace.conducting,
                                                  times),
                **stats,
            }
            d_flip = _other_bits_flipped(kernel_masks,
                                         switch_bits.get(int(bid), -1))
            cond_m, n_chat = _chatter_gaps(d_trace, d_flip)
            entry["n_chatter_gaps"] = int(n_chat)
            if n_chat and times.size > 1:
                h = np.diff(times)
                gap = cond_m & ~np.asarray(d_trace.conducting, dtype=bool)
                _warn_chatter(
                    name, n_chat,
                    int(np.count_nonzero(cond_m[:-1] & ~cond_m[1:])),
                    float(np.dot(h, gap[1:])), float(np.dot(h, cond_m[1:])),
                    V_th)
            # Optional datasheet offset+slope conduction model
            # (V_f0 + r_d·I) applied to the actual current trace.
            c_spec = (cond_by_bid.get(bid) or cond_by_name.get(name))
            if c_spec is not None:
                i_D = v_D * g_arr
                V_f0, r_on = _conduction_model_coeffs(c_spec)
                E_m = _conduction_model_energy(
                    times, i_D, d_trace.v_start * g_arr_s, c_spec)
                entry["P_cond_model_avg"] = (
                    E_m / T if T > 0 and times.size > 1
                    else float(_conduction_power_offset_slope(
                        i_D, c_spec).mean()))
                entry["E_cond_model"] = E_m
                entry["V_f0"] = V_f0
                entry["r_on"] = r_on
            # Optional datasheet reverse-recovery loss.
            d_spec = (diode_by_bid.get(bid)
                      or diode_by_name.get(name))
            if d_spec is not None:
                t_rr = _recovery_time(d_spec, name)
                entry.update(
                    _diode_switching_loss(
                        _diode_turn_offs(
                            d_trace, times, t_rr=t_rr,
                            other_flip=d_flip),
                        times, v_D, d_spec, i_D=v_D * g_arr))
                entry["t_rr"] = t_rr
                _warn_recovery_quality(name, entry, t_rr)
            summary.append(entry)
            switch_seq_idx += 1
            continue

        if kind == "switch":
            params = desc.get("params", {})
            g_on  = float(params.get("g_on", float("nan")))
            g_off = float(params.get("g_off", float("nan")))
            if not (np.isfinite(g_on) and np.isfinite(g_off)):
                switch_seq_idx += 1
                continue
            sw_bit = switch_bits.get(int(bid), switch_seq_idx)
            if switch_fn is None and not (
                    kernel_masks is not None
                    and 0 <= sw_bit < kernel_masks.shape[1]):
                # Honest skip — no recorded mask and no switch_fn: no
                # way to know the per-step state.
                switch_seq_idx += 1
                continue
            from_id, to_id = desc.get("nodes", (None, None))
            if from_id is None or to_id is None:
                switch_seq_idx += 1
                continue
            v_SW = (_node_voltage_trace(states, int(from_id))
                    - _node_voltage_trace(states, int(to_id)))
            # The kernel's record of the stamped mask; without one, a
            # SwitchMaskRecorder, else switch_fn re-evaluated under a
            # voltage-consistency guard (see _result_views).
            closed = _views.resolve_switch_closed_trace(
                switch_fn=switch_fn, result=result, times=times,
                switch_idx=sw_bit, v_branch=v_SW, name=name)
            closed_s = closed
            if (masks_s is not None and masks_s is not kernel_masks
                    and 0 <= sw_bit < masks_s.shape[1]):
                closed_s = np.asarray(masks_s[:, sw_bit], dtype=bool)
            v_SW_s = (_node_voltage_trace(states_s, int(from_id))
                      - _node_voltage_trace(states_s, int(to_id)))
            g_arr = np.where(closed, g_on, g_off)
            g_arr_s = np.where(closed_s, g_on, g_off)
            i_SW = v_SW * g_arr  # branch current at each sample
            i_SW_s = v_SW_s * g_arr_s  # ... at the start of the next step
            stats = _conduction_stats(v_SW, g_arr, times, v_SW_s, g_arr_s)
            entry: Dict[str, Any] = {
                "branch_id": int(bid),
                "kind": "switch",
                "name": name,
                "duty_closed": _time_fraction(closed, times),
                **stats,
            }
            # Optional offset+slope conduction model (V_ce0 + r_ce·I for
            # an IGBT, or r_on·I for a MOSFET) on the actual current.
            c_spec = (cond_by_bid.get(bid) or cond_by_name.get(name))
            if c_spec is not None:
                V_f0, r_on = _conduction_model_coeffs(c_spec)
                E_m = _conduction_model_energy(times, i_SW, i_SW_s, c_spec)
                entry["P_cond_model_avg"] = (
                    E_m / T if T > 0 and times.size > 1
                    else float(_conduction_power_offset_slope(
                        i_SW, c_spec).mean()))
                entry["E_cond_model"] = E_m
                entry["V_f0"] = V_f0
                entry["r_on"] = r_on
            s_spec = (switch_by_bid.get(bid)
                      or switch_by_name.get(name))
            if s_spec is not None:
                entry.update(
                    _switch_switching_loss(closed, times, v_SW,
                                            i_SW, s_spec,
                                            closed_start=closed_s,
                                            v_start=v_SW_s,
                                            i_start=i_SW_s))
            summary.append(entry)
            switch_seq_idx += 1
            continue

        # Other kinds (sources, capacitors, ...) — skip silently.

    return summary
