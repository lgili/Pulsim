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

   :func:`device_loss_summary` walks every resistor and inductor
   branch, integrates ``i² · R`` over time, and reports a table.

Switch and diode losses require per-step switch-state introspection
that is not yet plumbed through the pybind11 bindings (the kernel
emits :class:`CommutationEvent` records for diode events but does
not expose the deterministic switch-fn state per timestep). See
``KNOWN_LIMITATIONS.md`` § "Per-device loss reporting" — closing
that gap is tracked for v1.4.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np


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

def _states_as_array(result) -> np.ndarray:
    """Convert SimulationResult.states (list of vectors or numpy array)
    to a 2D numpy array shape (N, state_size)."""
    s = result.states
    if hasattr(s, "shape"):
        return np.asarray(s, dtype=float)
    return np.asarray([list(v) for v in s], dtype=float)


def _times_as_array(result) -> np.ndarray:
    t = result.times
    return np.asarray(t, dtype=float)


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


def _node_voltage_trace(states: np.ndarray, node_id: int) -> np.ndarray:
    """Return the voltage trace for ``node_id``. Ground (id < 0) is
    treated as the reference and reported as a zero trace."""
    if node_id < 0:
        return np.zeros(states.shape[0], dtype=float)
    return states[:, node_id]


def device_loss_summary(builder,
                            result) -> List[Dict[str, Any]]:
    """Walk every inductor and resistor branch in ``builder`` and
    report each one's conduction statistics.

    The summary covers two device kinds:

    * **inductors** — current is in the state vector directly; the
      entry includes ``i_avg``, ``i_rms``, ``i_peak``. Conduction
      loss is left to the caller (an ideal inductor has none).
    * **resistors** — current is reconstructed from the node-voltage
      difference and the device's stored ``R_ohms``, so the entry
      also reports ``P_avg`` (mean dissipated power, W) and
      ``E_total`` (energy integrated over the run, J).

    Switches and diodes are deliberately omitted: their per-step
    state isn't exposed by the bindings yet, so we cannot
    reconstruct an exact i(t). Closing that gap is tracked in
    ``KNOWN_LIMITATIONS.md`` for v1.4.

    Returns
    -------
    list of dict
        One entry per inductor / resistor in branch-id order. Each
        dict carries the keys ``branch_id``, ``kind``, ``name`` plus
        the kind-specific statistics above.
    """
    states = _states_as_array(result)
    times = _times_as_array(result)
    T = (times[-1] - times[0]) if times.size > 1 else 1.0

    # Build a {branch_id: descriptor} index from the builder so we
    # can look up resistor params + node ids without re-enumerating.
    try:
        comps_by_bid = {c["branch_id"]: c for c in builder.components()}
    except Exception:
        comps_by_bid = {}

    summary: List[Dict[str, Any]] = []

    for bid in range(builder.graph.num_branches):
        desc = comps_by_bid.get(bid, {})
        kind = desc.get("kind", "unknown")
        name = desc.get("name", "")

        if kind == "inductor":
            # Inductor current is in the state vector directly.
            try:
                i_idx = builder.pool.branch_var_id_for_inductor(
                    bid, builder.graph)
            except Exception:
                continue
            i_arr = states[:, i_idx]
            summary.append({
                "branch_id": int(bid),
                "kind": "inductor",
                "name": name,
                "i_avg": float(np.trapezoid(i_arr, times) / T) if T > 0
                           else float(i_arr.mean()),
                "i_rms": float(np.sqrt(np.mean(i_arr ** 2))),
                "i_peak": float(np.abs(i_arr).max()),
            })
            continue

        if kind == "resistor":
            # Reconstruct i_R = (v_from - v_to) / R from node
            # voltages. Ground (node_id < 0) is the reference.
            R = float(desc.get("params", {}).get("R_ohms", float("nan")))
            if not np.isfinite(R) or R <= 0:
                continue
            from_id, to_id = desc.get("nodes", (None, None))
            if from_id is None or to_id is None:
                continue
            v_from = _node_voltage_trace(states, int(from_id))
            v_to   = _node_voltage_trace(states, int(to_id))
            v_R = v_from - v_to
            i_R = v_R / R
            p_R = v_R * i_R  # = i² · R, but cheaper from v_R · i_R
            summary.append({
                "branch_id": int(bid),
                "kind": "resistor",
                "name": name,
                "R_ohms": R,
                "i_avg": float(np.trapezoid(i_R, times) / T) if T > 0
                           else float(i_R.mean()),
                "i_rms": float(np.sqrt(np.mean(i_R ** 2))),
                "i_peak": float(np.abs(i_R).max()),
                "P_avg": (float(np.trapezoid(p_R, times) / T)
                          if T > 0 else float(p_R.mean())),
                "E_total": (float(np.trapezoid(p_R, times))
                            if times.size > 1 else 0.0),
            })
            continue

        # Other kinds (switches, diodes, sources, ...) — skip for now.
        # See KNOWN_LIMITATIONS.md for the v1.4 roadmap.

    return summary
