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
   conduction statistics. Pass ``switch_fn=`` to reconstruct
   switch conduction loss from the same deterministic mask the
   simulation used; pass ``core_loss_specs=`` to add Steinmetz /
   iGSE core loss on selected inductors.

The reconstruction is post-hoc only — the kernel itself stays
fixed-dt and the result trace is processed in NumPy. No kernel
binding changes are required for v1.6 closure.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Mapping, Optional, Union

import numpy as np

from . import _result_views as _views
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
_evaluate_switch_mask_trace = _views.evaluate_switch_mask_trace


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
                      times: np.ndarray) -> Dict[str, float]:
    """Compute (i_avg, i_rms, i_peak, P_avg, E_total) for a branch
    whose current trace is ``i = v · g`` and whose instantaneous
    power is ``P = v · i = v² · g``.

    ``g`` may be a scalar (constant conductance, e.g. a resistor or
    on-state switch) or an array matching ``v`` (state-dependent
    conductance, e.g. a switch toggling between ``g_on`` / ``g_off``).
    """
    if isinstance(g, np.ndarray):
        g_arr = g.astype(float, copy=False)
    else:
        g_arr = np.full_like(v, float(g))
    i = v * g_arr
    p = v * i
    T = (times[-1] - times[0]) if times.size > 1 else 1.0
    if T <= 0:
        T = 1.0
    return {
        "i_avg": float(np.trapezoid(i, times) / T) if times.size > 1
                   else float(i.mean()),
        "i_rms": float(np.sqrt(np.mean(i ** 2))),
        "i_peak": float(np.abs(i).max()) if i.size else 0.0,
        "P_avg": float(np.trapezoid(p, times) / T) if times.size > 1
                   else float(p.mean()),
        "E_total": float(np.trapezoid(p, times)) if times.size > 1
                     else 0.0,
    }


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


def _diode_switching_loss(events_for_branch,
                          times: np.ndarray,
                          v_D: np.ndarray,
                          spec: Mapping[str, Any]) -> Dict[str, float]:
    """Post-hoc reverse-recovery switching loss from
    ``SimulationResult.commutation_events``.

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
    ``E_rr_ref = Q_rr · V_R_ref``. ``V_R`` itself is linearly
    interpolated from the result trace at each event time, so the
    formula tracks the actual reverse voltage the diode sees —
    independent of the simulation dt.

    The ideal ``SwitchedDiode`` kernel model has no physical
    recovery charge; this is a datasheet annotation, matching
    PSIM / LTspice "ideal-diode-with-Qrr" post-processing.
    """
    turn_offs = [ev for ev in events_for_branch
                  if bool(ev.new_state) is False]
    n_off = len(turn_offs)
    T = (times[-1] - times[0]) if times.size > 1 else 0.0
    Q_rr = float(spec.get("Q_rr", 0.0))
    E_rr_ref = float(spec.get("E_rr_ref", 0.0))
    V_R_ref = float(spec.get("V_R_ref", 0.0))

    bookkeeping = {
        "n_turn_off_events": int(n_off),
        "f_sw_estimate": (float(n_off) / T) if T > 0 else 0.0,
    }
    if n_off == 0 or (Q_rr <= 0 and E_rr_ref <= 0):
        return {"E_sw_total": 0.0, "P_sw_avg": 0.0, **bookkeeping}

    # Interpolate |v_D| at every turn-off instant.
    t_off = np.asarray([float(ev.t_estimated) for ev in turn_offs])
    V_R = np.interp(t_off, times, np.abs(v_D))

    if E_rr_ref > 0:
        if V_R_ref <= 0:
            raise ValueError(
                "diode_specs: E_rr_ref requires a positive V_R_ref "
                "(the reference reverse voltage from the datasheet).")
        E_per_event = E_rr_ref * (V_R / V_R_ref)
    else:
        E_per_event = Q_rr * V_R
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
                            spec: Mapping[str, Any]) -> Dict[str, float]:
    """Post-hoc turn-on / turn-off switching loss for an ideal-switch
    branch (MOSFET / IGBT equivalent).

    PSIM-style datasheet annotation: ``E_on`` and ``E_off`` are taken
    from the device datasheet at a reference operating point
    ``(V_ref, I_ref)`` and scaled linearly by both the blocking
    voltage and the conducting current at the transition::

        E_on_event  = E_on_ref  · (V_blocking / V_ref) · (I_load / I_ref)
        E_off_event = E_off_ref · (V_blocking / V_ref) · (I_load / I_ref)

    Transitions are detected from the user-supplied ``switch_fn``
    mask trace — the kernel itself emits no commutation events for
    ideal switches, but the deterministic switch_fn lets us
    reconstruct turn-on (False→True) and turn-off (True→False)
    edges exactly.

    At each edge between samples k-1 and k, we take ``|v_SW[k-1]|``
    as the blocking voltage (the value the switch was holding off
    just before / just after the transition) and ``|i_SW[k]|`` as
    the load current the device sees while conducting.
    """
    E_on_ref  = float(spec.get("E_on_ref",  0.0))
    E_off_ref = float(spec.get("E_off_ref", 0.0))
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
    if (n_on == 0 and n_off == 0) or (E_on_ref <= 0 and E_off_ref <= 0):
        return {
            "E_sw_on_total": 0.0,
            "E_sw_off_total": 0.0,
            "E_sw_total": 0.0,
            "P_sw_avg": 0.0,
            **bookkeeping,
        }
    if V_ref <= 0 or I_ref <= 0:
        raise ValueError(
            "switch_specs: V_ref and I_ref must be positive — they "
            "anchor the datasheet E_on/E_off reference operating "
            "point.")

    # Blocking voltage = the OFF-state v_SW (across the open switch).
    # That's the sample BEFORE a turn-on edge and the sample AFTER a
    # turn-off edge — in both cases the switch is "open / blocking".
    # Load current = the ON-state |i_SW| in the conducting interval
    # adjacent to the edge. For turn-on that's the sample at k (just
    # after closing); for turn-off it's the sample at k-1 (just before
    # opening).
    def _edge_loss(idxs: np.ndarray, E_ref: float,
                    *, blocking_at_post_edge: bool,
                    load_at_post_edge: bool) -> float:
        if idxs.size == 0 or E_ref <= 0:
            return 0.0
        blk_k  = idxs if blocking_at_post_edge \
                       else np.clip(idxs - 1, 0, v_SW.size - 1)
        load_k = idxs if load_at_post_edge \
                       else np.clip(idxs - 1, 0, v_SW.size - 1)
        V_blk = np.abs(v_SW[blk_k])
        I_load = np.abs(i_SW[load_k])
        return float(np.sum(E_ref * (V_blk / V_ref) * (I_load / I_ref)))

    # Turn-on:  V_blk = pre-edge (open), I_load = post-edge (just closed).
    # Turn-off: V_blk = post-edge (now open), I_load = pre-edge (was closed).
    E_on_total = _edge_loss(turn_on_k,  E_on_ref,
                              blocking_at_post_edge=False,
                              load_at_post_edge=True)
    E_off_total = _edge_loss(turn_off_k, E_off_ref,
                              blocking_at_post_edge=True,
                              load_at_post_edge=False)
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
      conduction state is inferred from the node-voltage drop and
      the device's ``V_th``. Reports the same current + power metrics
      as the resistor path.
    * **switch** (ideal switch with ``g_on`` / ``g_off``) — requires
      ``switch_fn`` so the per-step closed/open mask can be sampled
      deterministically. Without ``switch_fn`` the switch entry is
      skipped (kept honest — no silent default).

    Parameters
    ----------
    builder, result
        A populated :class:`CircuitBuilder` and the
        :class:`SimulationResult` produced by it.
    switch_fn : callable, optional
        ``t -> SwitchStateMask`` used during the simulation. Pass
        the same callable here to enable switch-branch loss
        reconstruction.
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

        For each turn-off event in
        ``result.commutation_events`` matching the branch the
        helper interpolates ``|V_R(t_event)|`` and accumulates
        ``E_rr_event = Q_rr · V_R`` or
        ``E_rr_event = E_rr_ref · (V_R / V_R_ref)``. Reports
        ``E_sw_total``, ``P_sw_avg``, ``n_turn_off_events``,
        ``f_sw_estimate``.
    switch_specs : mapping, optional
        Per-switch datasheet annotations for MOSFET / IGBT-style
        turn-on / turn-off switching loss (PSIM-style). Keys are
        switch branch names or branch ids. Values::

            {"E_on_ref":  float (J),
             "E_off_ref": float (J),
             "V_ref":     float (V),
             "I_ref":     float (A)}

        Transitions are detected from the deterministic
        ``switch_fn`` mask (so ``switch_fn=`` is required when
        ``switch_specs=`` is). Per event::

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

    # Bucket commutation events by branch_id so the diode handler
    # can iterate only its own.
    events = list(getattr(result, "commutation_events", []))
    events_by_branch: Dict[int, list] = {}
    for ev in events:
        events_by_branch.setdefault(int(ev.branch_id), []).append(ev)

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
            entry: Dict[str, Any] = {
                "branch_id": int(bid),
                "kind": "inductor",
                "name": name,
                "i_avg": (float(np.trapezoid(i_arr, times) / T)
                          if T > 0 else float(i_arr.mean())),
                "i_rms": float(np.sqrt(np.mean(i_arr ** 2))),
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
            stats = _conduction_stats(v_R, 1.0 / R, times)
            summary.append({
                "branch_id": int(bid),
                "kind": "resistor",
                "name": name,
                "R_ohms": R,
                **stats,
            })
            continue

        if kind == "diode":
            # SwitchedDiode: closed when v_D > V_th. Use g_on while
            # forward-biased, g_off while reverse-biased.
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
            v_D = (_node_voltage_trace(states, int(from_id))
                   - _node_voltage_trace(states, int(to_id)))
            closed = v_D > V_th
            g_arr = np.where(closed, g_on, g_off)
            stats = _conduction_stats(v_D, g_arr, times)
            entry: Dict[str, Any] = {
                "branch_id": int(bid),
                "kind": "diode",
                "name": name,
                "V_th": V_th,
                "duty_conducting": float(closed.mean())
                                       if closed.size else 0.0,
                **stats,
            }
            # Optional datasheet offset+slope conduction model
            # (V_f0 + r_d·I) applied to the actual current trace.
            c_spec = (cond_by_bid.get(bid) or cond_by_name.get(name))
            if c_spec is not None:
                i_D = v_D * g_arr
                p_model = _conduction_power_offset_slope(i_D, c_spec)
                V_f0, r_on = _conduction_model_coeffs(c_spec)
                entry["P_cond_model_avg"] = (
                    float(np.trapezoid(p_model, times) / T)
                    if T > 0 else float(p_model.mean()))
                entry["E_cond_model"] = (float(np.trapezoid(p_model, times))
                                         if times.size > 1 else 0.0)
                entry["V_f0"] = V_f0
                entry["r_on"] = r_on
            # Optional datasheet reverse-recovery loss.
            d_spec = (diode_by_bid.get(bid)
                      or diode_by_name.get(name))
            if d_spec is not None:
                ev_branch = events_by_branch.get(int(bid), [])
                entry.update(
                    _diode_switching_loss(ev_branch, times, v_D, d_spec))
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
            if switch_fn is None:
                # Honest skip — without switch_fn we have no way to
                # know the per-step mask state.
                switch_seq_idx += 1
                continue
            from_id, to_id = desc.get("nodes", (None, None))
            if from_id is None or to_id is None:
                switch_seq_idx += 1
                continue
            closed = _evaluate_switch_mask_trace(
                switch_fn, times, switch_seq_idx)
            v_SW = (_node_voltage_trace(states, int(from_id))
                    - _node_voltage_trace(states, int(to_id)))
            g_arr = np.where(closed, g_on, g_off)
            i_SW = v_SW * g_arr  # branch current at each sample
            stats = _conduction_stats(v_SW, g_arr, times)
            entry: Dict[str, Any] = {
                "branch_id": int(bid),
                "kind": "switch",
                "name": name,
                "duty_closed": float(closed.mean())
                                   if closed.size else 0.0,
                **stats,
            }
            # Optional offset+slope conduction model (V_ce0 + r_ce·I for
            # an IGBT, or r_on·I for a MOSFET) on the actual current.
            c_spec = (cond_by_bid.get(bid) or cond_by_name.get(name))
            if c_spec is not None:
                p_model = _conduction_power_offset_slope(i_SW, c_spec)
                V_f0, r_on = _conduction_model_coeffs(c_spec)
                entry["P_cond_model_avg"] = (
                    float(np.trapezoid(p_model, times) / T)
                    if T > 0 else float(p_model.mean()))
                entry["E_cond_model"] = (float(np.trapezoid(p_model, times))
                                         if times.size > 1 else 0.0)
                entry["V_f0"] = V_f0
                entry["r_on"] = r_on
            s_spec = (switch_by_bid.get(bid)
                      or switch_by_name.get(name))
            if s_spec is not None:
                entry.update(
                    _switch_switching_loss(closed, times, v_SW,
                                            i_SW, s_spec))
            summary.append(entry)
            switch_seq_idx += 1
            continue

        # Other kinds (sources, capacitors, ...) — skip silently.

    return summary
