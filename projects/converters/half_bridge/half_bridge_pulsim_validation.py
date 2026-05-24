"""Half-bridge converter cross-validation against Pulsim.

Builds a transformer-isolated half-bridge in the Pulsim ``CircuitBuilder``.
The half-bridge alternates two switches (high-side ``S1``, low-side
``S2``) around the rail midpoint, driving the transformer primary with
``±V_g/2``.

Topology (rail-splitter + full-wave bridge rectifier on the secondary)::

    Vin ──┬── C_split_top ── V_mid ── C_split_bot ──┬── gnd
          │                                         │
          ├── S1 (HS) ──┬── sw ──── T1.primary ─────┤  ← primary tied to V_mid
          │             │                          │
          └── S2 (LS) ──┴── sw                     │

    T1.secondary (sec_top → sec_bot):
        sec_top ─ D_rec_p ─┐
                          ├── vout_pre ─ L_f ─ vout ── Cout ── R_load ── sec_neg
        sec_bot ─ D_rec_n ─┘
        sec_neg ─ D_fw_p  → sec_top
        sec_neg ─ D_fw_n  → sec_bot

    sec_neg ── R_iso ── gnd  (1 µΩ tie to keep MNA single-component)

Because Pulsim's ``add_transformer`` exposes only a two-winding device,
the textbook center-tapped secondary is reshaped into a single
secondary winding feeding a 4-diode full-wave bridge. The peak
rectified voltage is ``n · V_g/2`` and conduction is ``2D`` per period,
so the average ``V_o = n D V_g`` still matches the analytical formula.

Switch ordering (insertion order = bit order in the switch mask):
  bit 0 → S1 (HS), bit 1 → S2 (LS).
"""

from __future__ import annotations

import math

import numpy as np


def _build_builder(bp, k_coupling: float = 0.999):
    import pulsim as p

    L_p_mag = max(float(bp.L) * 100.0, 1e-3)
    L_s_mag = float(bp.n) ** 2 * L_p_mag

    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", float(bp.V_g))
    b.add_capacitor("C_split_top", "vin", "V_mid", 100e-6)
    b.add_capacitor("C_split_bot", "V_mid", "gnd",  100e-6)

    b.add_mosfet_with_body_diode("S1", "vin", "sw",
                                   R_on=1e-2, R_off=1e9, V_F=0.7)
    b.add_mosfet_with_body_diode("S2", "sw",  "gnd",
                                   R_on=1e-2, R_off=1e9, V_F=0.7)

    b.add_transformer(
        "T1",
        p_from="sw", p_to="V_mid",
        s_from="sec_top", s_to="sec_bot",
        L_p=L_p_mag, L_s=L_s_mag, k=float(k_coupling),
    )

    b.add_diode("D_rec_p", "sec_top", "vout_pre", 1e3, 1e-9, V_th=0.7)
    b.add_diode("D_rec_n", "sec_bot", "vout_pre", 1e3, 1e-9, V_th=0.7)
    b.add_diode("D_fw_p",  "sec_neg", "sec_top",  1e3, 1e-9, V_th=0.7)
    b.add_diode("D_fw_n",  "sec_neg", "sec_bot",  1e3, 1e-9, V_th=0.7)

    b.add_inductor("L_f", "vout_pre", "vout", float(bp.L))
    b.add_capacitor("Cout", "vout", "sec_neg", float(bp.C))
    b.add_resistor("Rload", "vout", "sec_neg", float(bp.R))
    b.add_resistor("Rgnd", "sec_neg", "gnd", 1.0e-6)
    return b


def _alternating_switch_fn(num_switches: int,
                            f_sw: float,
                            duty: float,
                            dead_time: float):
    """Switch_fn that alternates S1 (bit 0) and S2 (bit 1) per period.

    Sequence within one period ``T = 1/f_sw``:

        [0, D·T)         S1 ON
        [D·T, T/2)       both OFF (dead-time)
        [T/2, T/2 + D·T) S2 ON
        [T/2 + D·T, T)   both OFF
    """
    import pulsim as p

    T = 1.0 / f_sw
    on_time = duty * T

    def switch_fn(t):
        phase = math.fmod(t, T)
        m = p.SwitchStateMask(num_switches)
        if 0.0 <= phase < on_time - dead_time * 0.5:
            m.set(0, True)                          # S1
        elif T * 0.5 <= phase < T * 0.5 + on_time - dead_time * 0.5:
            m.set(1, True)                          # S2
        return m

    return switch_fn


def simulate_half_bridge(bp,
                           duty: float | None = None,
                           t_end: float = 3e-3,
                           dt: float = 2e-8,
                           k_coupling: float = 0.999,
                           ) -> "tuple[np.ndarray, np.ndarray]":
    """Run a Pulsim half-bridge simulation with constant per-switch duty.

    Returns ``(times, v_out)``.
    """
    import pulsim as p  # noqa: F401

    b = _build_builder(bp, k_coupling=k_coupling)
    if duty is None:
        duty = float(bp.D)

    sw_fn = _alternating_switch_fn(
        num_switches=b.graph.num_switches,
        f_sw=float(bp.f_sw),
        duty=float(duty),
        dead_time=float(bp.t_dead),
    )

    res = p.simulate(
        b, t_end=float(t_end), dt=float(dt),
        switch_fn=sw_fn,
        max_event_iterations=8,
    )

    out_idx = b.node_id_of("vout")
    times = np.asarray(res.times, dtype=float)
    states = (np.asarray(res.states, dtype=float)
              if hasattr(res.states, "shape")
              else np.asarray([list(v) for v in res.states], dtype=float))
    v_out = states[:, out_idx]
    return times, v_out
