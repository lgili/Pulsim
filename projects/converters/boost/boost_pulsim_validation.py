"""Boost converter cross-validation against Pulsim.

Builds the topology the analytical model fits — input-side inductor,
low-side MOSFET shorting ``sw`` to ground during ON, boost diode
passing current ``sw → vout`` during OFF — and runs a fixed-dt
Pulsim transient::

   Vin ── L ── sw ──┬── D ── vout
                    │
                    Q1 (LS MOSFET + body diode)
                    │
                   gnd                   vout ── Cout ── gnd
                                          │
                                         R_load → gnd

API notes mirror :mod:`buck_pulsim_validation`: string node names,
``add_mosfet_with_body_diode`` driven by ``make_pwm_switch_fn``,
``initial_state=`` warm-start instead of inductor/capacitor IC args.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def _build_boost_builder(bp):
    import pulsim as p

    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", float(bp.V_g))
    b.add_inductor("L1", "vin", "sw", float(bp.L))
    b.add_mosfet_with_body_diode("Q1", "sw", "gnd",
                                   R_on=1e-3, R_off=1e9, V_F=0.7)
    b.add_diode("D1", "sw", "vout", 1e3, 1e-9, V_th=0.7)
    b.add_capacitor("Cout", "vout", "gnd", float(bp.C))
    b.add_resistor("Rload", "vout", "gnd", float(bp.R))
    return b


def _warm_start(b, bp):
    """Steady-state seed: ``I_L = V_o / (R · (1 - D))``, ``V_C = V_o``."""
    n_state = b.pool.state_size(b.graph)
    x0 = [0.0] * n_state
    out_idx = b.node_id_of("vout")
    x0[out_idx] = float(bp.V_o)
    I_L0 = float(bp.V_o / (bp.R * (1.0 - bp.D)))
    for bid in range(b.graph.num_branches):
        try:
            i_idx = b.pool.branch_var_id_for_inductor(bid, b.graph)
            x0[i_idx] = I_L0
            break
        except Exception:
            continue
    return x0


def simulate_boost(bp,
                    duty: float,
                    t_end: float = 5e-3,
                    dt: float = 1e-7,
                    warm_start: bool = True,
                    ) -> "tuple[np.ndarray, np.ndarray]":
    """Run a Pulsim boost simulation with constant duty.

    Returns ``(times, v_out)`` parallel arrays.
    """
    import pulsim as p

    b = _build_boost_builder(bp)
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
    v_out = states[:, out_idx]
    return times, v_out


def simulate_boost_with_step(bp,
                              duty_before: float,
                              duty_after: float,
                              t_step: float,
                              t_end: float = 5e-3,
                              dt: float = 1e-7,
                              ) -> "tuple[np.ndarray, np.ndarray]":
    """Run a boost sim with a duty step at ``t_step``.

    Useful for visualising the RHP-zero dip-and-recover that follows a
    positive duty step in any boost-derived converter.
    """
    import pulsim as p

    b = _build_boost_builder(bp)
    x0 = _warm_start(b, bp)

    sw_before = p.make_pwm_switch_fn(
        frequency=float(bp.f_sw), duty=float(duty_before),
        switch_idx=0, num_switches=b.graph.num_switches,
    )
    sw_after = p.make_pwm_switch_fn(
        frequency=float(bp.f_sw), duty=float(duty_after),
        switch_idx=0, num_switches=b.graph.num_switches,
    )

    def sw_fn(t):
        return sw_before(t) if t < t_step else sw_after(t)

    res = p.simulate(
        b, t_end=float(t_end), dt=float(dt),
        switch_fn=sw_fn,
        initial_state=x0,
        max_event_iterations=8,
    )

    out_idx = b.node_id_of("vout")
    times = np.asarray(res.times, dtype=float)
    states = (np.asarray(res.states, dtype=float)
              if hasattr(res.states, "shape")
              else np.asarray([list(v) for v in res.states], dtype=float))
    v_out = states[:, out_idx]
    return times, v_out
