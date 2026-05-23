"""Buck-boost converter cross-validation against Pulsim.

Builds an inverting buck-boost in the Pulsim ``CircuitBuilder``::

   Vin ── Q1 ── sw ──┬── D ── vout  (vout settles negative w.r.t. gnd)
                     │              vout ── Cout ── gnd
                    L1                       │
                     │                      R_load → gnd
                    gnd

The simulator returns ``vout`` as a negative voltage in steady state
(this is an inverting topology). We model ``|V_o|`` as positive in the
analytical model and flip the sign here so the two overlay directly.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def _build_builder(bp):
    import pulsim as p

    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", float(bp.V_g))
    b.add_mosfet_with_body_diode("Q1", "vin", "sw",
                                   R_on=1e-3, R_off=1e9, V_F=0.7)
    b.add_inductor("L1", "sw", "gnd", float(bp.L))
    b.add_diode("D1", "vout", "sw", 1e3, 1e-9, V_th=0.7)
    b.add_capacitor("Cout", "vout", "gnd", float(bp.C))
    b.add_resistor("Rload", "vout", "gnd", float(bp.R))
    return b


def _warm_start(b, bp):
    n_state = b.pool.state_size(b.graph)
    x0 = [0.0] * n_state
    out_idx = b.node_id_of("vout")
    x0[out_idx] = -float(bp.V_o)
    I_L0 = float(bp.V_o / (bp.R * (1.0 - bp.D)))
    for bid in range(b.graph.num_branches):
        try:
            i_idx = b.pool.branch_var_id_for_inductor(bid, b.graph)
            x0[i_idx] = I_L0
            break
        except Exception:
            continue
    return x0


def simulate_buck_boost(bp,
                          duty: float,
                          t_end: float = 5e-3,
                          dt: float = 1e-7,
                          warm_start: bool = True,
                          ) -> "tuple[np.ndarray, np.ndarray]":
    """Run a Pulsim buck-boost simulation with constant duty.

    Returns ``(times, v_out_magnitude)``. ``v_out_magnitude`` is
    ``-v_out`` so it matches the analytical model's positive
    convention.
    """
    import pulsim as p

    b = _build_builder(bp)
    initial_state: Optional[list] = _warm_start(b, bp) if warm_start else None

    sw_fn = p.make_pwm_switch_fn(
        frequency=float(bp.f_sw), duty=float(duty),
        switch_idx=0, num_switches=b.graph.num_switches,
    )
    res = p.simulate(
        b, t_end=float(t_end), dt=float(dt),
        switch_fn=sw_fn,
        initial_state=initial_state,
        max_event_iterations=8,
    )

    out_idx = b.node_id_of("vout")
    times = np.asarray(res.times, dtype=float)
    states = (np.asarray(res.states, dtype=float)
              if hasattr(res.states, "shape")
              else np.asarray([list(v) for v in res.states], dtype=float))
    v_out_mag = -states[:, out_idx]
    return times, v_out_mag
