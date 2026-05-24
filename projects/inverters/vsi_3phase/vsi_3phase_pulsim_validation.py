"""Three-phase VSI cross-validation against Pulsim.

Builds a 2-level voltage source inverter feeding a Y-connected RL load
in the Pulsim ``CircuitBuilder``. Uses the topology + control helpers
shipped with current Pulsim:

* ``pulsim.topology.add_three_phase_vsi`` — registers the 6 ideal
  switches (HSa, LSa, HSb, LSb, HSc, LSc) — matches
  ``ThreePhaseLegIndices`` exactly.
* ``pulsim.topology.add_three_phase_rl_load`` — Y-connected per-phase
  R+L (currents readable via
  ``builder.pool.branch_var_id_for_inductor``).
* ``pulsim.make_three_phase_spwm_fn`` — SPWM driver with built-in
  shoot-through prevention (HS and LS in the same leg never close
  simultaneously by construction).

The body diodes are added explicitly to clamp the phase midpoint
during turn-off (the VSI helper itself doesn't add them).
"""

from __future__ import annotations

import numpy as np


def _build_builder(vp, R_load: float | None = None, L_load: float = 5e-3):
    import pulsim as p
    from pulsim.topology import add_three_phase_vsi, add_three_phase_rl_load

    R = float(R_load) if R_load is not None else float(vp.R_load)

    b = p.CircuitBuilder()
    b.add_voltage_source("Vdc", "vdc", "gnd", float(vp.V_dc))

    add_three_phase_vsi(
        b, "INV",
        vdc_pos="vdc", vdc_neg="gnd",
        out_a="mid_a", out_b="mid_b", out_c="mid_c",
        R_on=1e-3, R_off=1e9,
    )

    # 6 anti-parallel body diodes (the VSI helper does not add them).
    # They let the inductive load current free-wheel during turn-off
    # without unbounded dV/dt on the phase midpoint.
    for tag, anode, cathode in [
        ("D_HS_A", "mid_a", "vdc"), ("D_LS_A", "gnd", "mid_a"),
        ("D_HS_B", "mid_b", "vdc"), ("D_LS_B", "gnd", "mid_b"),
        ("D_HS_C", "mid_c", "vdc"), ("D_LS_C", "gnd", "mid_c"),
    ]:
        b.add_diode(tag, anode, cathode, 1e3, 1e-9, V_th=0.7)

    add_three_phase_rl_load(
        b, "LD",
        node_a="mid_a", node_b="mid_b", node_c="mid_c",
        node_neutral="n",
        R=R, L=float(L_load),
        topology="star",
    )
    b.add_resistor("R_n_to_gnd", "n", "gnd", 1.0e-3)
    return b


def simulate_vsi(vp,
                  m_a: float | None = None,
                  t_end: float = 0.05,
                  dt: float = 5e-7,
                  dead_time: float = 0.0,
                  ) -> "tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]":
    """Run an open-loop 3-phase VSI simulation with SPWM.

    Returns ``(times, v_mid_a, v_mid_b, v_mid_c)``: the three phase-leg
    midpoint voltages w.r.t. the inverter ground rail.
    """
    import pulsim as p

    if m_a is None:
        m_a = float(vp.m_a)

    b = _build_builder(vp)

    legs = p.ThreePhaseLegIndices(
        hs_a=0, ls_a=1,
        hs_b=2, ls_b=3,
        hs_c=4, ls_c=5,
    )
    spwm = p.make_three_phase_spwm_fn(
        carrier_frequency=float(vp.f_sw),
        modulation_frequency=float(vp.f_o),
        modulation_index=float(m_a),
        legs=legs,
        num_switches=b.graph.num_switches,
        dead_time=float(dead_time),
        modulation_phase=0.0,
        carrier_phase=0.0,
    )

    res = p.simulate(
        b, t_end=float(t_end), dt=float(dt),
        switch_fn=spwm,
        max_event_iterations=8,
    )

    a_idx = b.node_id_of("mid_a")
    b_idx = b.node_id_of("mid_b")
    c_idx = b.node_id_of("mid_c")
    times = np.asarray(res.times, dtype=float)
    states = (np.asarray(res.states, dtype=float)
              if hasattr(res.states, "shape")
              else np.asarray([list(v) for v in res.states], dtype=float))
    v_a = states[:, a_idx]
    v_b = states[:, b_idx]
    v_c = states[:, c_idx]
    return times, v_a, v_b, v_c
