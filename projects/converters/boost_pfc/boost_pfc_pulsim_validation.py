"""Boost PFC cross-validation against Pulsim.

Builds a single-phase AC-input boost PFC in the Pulsim ``CircuitBuilder``:
``add_sine_voltage_source`` (mains) → ``pulsim.topology.add_bridge_rectifier``
(4-diode full-wave) → boost stage (L, switch, diode, C, R_load).

Topology::

    v_ac(t) ─┤  ┌─ bridge rectifier ─┐
             │  │   (D1..D4)         │
             │  │                    │
             └─┤ac_a            dc_pos├─┬── L ── sw ──┬── D ── vout
               │                      │ │             │
               │ac_b            dc_neg├─┘ C_in        Q1 (LS MOSFET)
               │                      │               │
                                                     gnd
                                       vout ── Cout ── gnd
                                                       │
                                                      R_load → gnd

The control loop is **not** implemented in this validation module — we
just drive the boost switch at a fixed duty cycle. The cross-check
verifies that the bridge produces the textbook ``|V_pk sin(ω_line t)|``
rectified envelope and that the bulk DC bus rises toward an open-loop
steady state.
"""

from __future__ import annotations

import numpy as np


def _build_builder(bp):
    import pulsim as p
    from pulsim.topology import add_bridge_rectifier

    b = p.CircuitBuilder()

    V_g_pk = float(np.sqrt(2.0) * bp.V_ac_nom)
    b.add_sine_voltage_source(
        "Vac", "ac_a", "ac_b",
        v_dc=0.0, v_amplitude=V_g_pk,
        frequency=float(bp.f_line), phase=0.0,
    )

    # No AC-side ground tie: dc_neg is already gnd. A milliohm tie from ac_b
    # to gnd shorts the lower bridge diode and turns the bridge into a
    # half-wave rectifier (AI_GUIDE trap 7).

    add_bridge_rectifier(
        b, "BR",
        ac_a="ac_a", ac_b="ac_b",
        dc_pos="rect_pos", dc_neg="gnd",
        g_on=1e3, g_off=1e-9, V_th=0.7,
    )

    # The HF input capacitor every PFC has after the bridge. Without it
    # rect_pos floats whenever the DCM inductor current is zero.
    b.add_capacitor("C_in", "rect_pos", "gnd", 1e-6)
    b.add_inductor("L_pfc", "rect_pos", "sw", float(bp.L))
    b.add_mosfet_with_body_diode("Q1", "sw", "gnd",
                                   R_on=1e-2, R_off=1e9, V_F=0.7)
    b.add_diode("D_boost", "sw", "vout", 1e3, 1e-9, V_th=0.7)
    b.add_capacitor("Cout", "vout", "gnd", float(bp.C))
    b.add_resistor("Rload", "vout", "gnd", float(bp.R_load))
    return b


def simulate_pfc(bp,
                  duty: float = 0.45,
                  t_end: float = 0.04,
                  dt: float = 1e-7,
                  ) -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
    """Run an open-loop boost-PFC simulation at constant duty cycle.

    Default ``t_end = 40 ms`` covers two full 50 Hz line cycles — long
    enough to verify the rectified envelope and observe the bulk DC
    bus climbing toward an open-loop steady state.

    Returns ``(times, v_rect, v_out)``: ``v_rect`` is the rectified
    line voltage at ``rect_pos``; ``v_out`` is the bulk DC bus.
    """
    import pulsim as p

    b = _build_builder(bp)

    sw_fn = p.make_pwm_switch_fn(
        frequency=float(bp.f_sw), duty=float(duty),
        # The four bridge diodes come first: Q1 is bit 4, not 0.
        switch_idx=b.switch_index_of("Q1"), num_switches=b.graph.num_switches,
    )
    res = p.simulate(
        b, t_end=float(t_end), dt=float(dt),
        switch_fn=sw_fn,
        max_event_iterations=8,
    )

    rect_idx = b.node_id_of("rect_pos")
    out_idx = b.node_id_of("vout")
    times = np.asarray(res.times, dtype=float)
    states = (np.asarray(res.states, dtype=float)
              if hasattr(res.states, "shape")
              else np.asarray([list(v) for v in res.states], dtype=float))
    v_rect = states[:, rect_idx]
    v_out = states[:, out_idx]
    return times, v_rect, v_out
