"""Buck converter cross-validation against Pulsim v2.

This module is the v2-native replacement for the legacy v1
``build_pulsim_buck`` function that used to live inline in
``01_buck_modeling.ipynb``. The notebook now just imports this
module and calls :func:`simulate_buck_v2`.

The function below builds the same physical buck topology the
analytical state-space model is fitting (V_g → MOSFET + body diode
→ freewheeling diode → L → C ∥ R_load → gnd) and runs a fixed-dt
v2 simulation with a constant PWM duty cycle. Use the returned
``(times, v_out)`` arrays as the ground-truth waveform to compare
against the analytical ``signal.step(Gvd, …)`` prediction.

Pulsim API changes worth noting (vs the legacy v1 notebook):

* No explicit ``add_node`` — nodes are inferred from the names
  passed to ``add_*``.
* No ``vcswitch`` — use ``add_mosfet_with_body_diode`` + a
  ``switch_fn`` that drives the gate.
* No initial-condition kwarg on ``add_inductor`` / ``add_capacitor``
  — pass the warm-start state via ``simulate(..., initial_state=…)``.
* ``PulseParams`` is gone — PWM is handled by ``make_pwm_switch_fn``
  which takes ``(frequency, duty, switch_idx, num_switches)``.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def simulate_buck_v2(bp,
                       duty: float,
                       t_end: float = 5e-3,
                       dt: float = 1e-7,
                       warm_start: bool = True,
                       ) -> "tuple[np.ndarray, np.ndarray]":
    """Run a Pulsim v2 buck simulation with constant duty.

    Parameters
    ----------
    bp
        A :class:`BuckParams` instance with ``V_g``, ``V_o``, ``L``,
        ``C``, ``R``, ``f_sw`` attributes.
    duty
        Steady-state duty cycle (0–1).
    t_end
        Simulation length in seconds. Default 5 ms covers ~8 LC
        natural periods of a 12 V / 100 µH / 100 µF design.
    dt
        Fixed simulation step. Default 100 ns — well below the PWM
        period for any switching converter.
    warm_start
        If True, seed the inductor current and capacitor voltage
        to their steady-state operating-point values so the
        captured waveform doesn't include the multi-period
        startup transient.

    Returns
    -------
    (times, v_out)
        Two numpy arrays, parallel. ``v_out`` is the voltage at the
        ``"out"`` node sampled at the simulation dt.
    """
    import pulsim.v2 as pv2  # noqa: F401 (local import, skip-friendly)

    b = pv2.CircuitBuilder()
    b.add_voltage_source("Vdc", "vin", "gnd", float(bp.V_g))
    b.add_mosfet_with_body_diode("Q1", "vin", "sw",
                                      R_on=1e-3, R_off=1e9, V_F=0.7)
    b.add_diode("D_FW", "gnd", "sw",
                  g_on=1e3, g_off=1e-9, V_th=0.7)
    b.add_inductor("L1", "sw", "out", float(bp.L))
    b.add_capacitor("Cout", "out", "gnd", float(bp.C))
    b.add_resistor("Rload", "out", "gnd", float(bp.R))

    # Steady-state warm-start: inductor at I_L = V_o/R, cap at V_o.
    initial_state: Optional[list] = None
    if warm_start:
        n_state = b.pool.state_size(b.graph)
        x0 = [0.0] * n_state
        out_idx = b.node_id_of("out")
        x0[out_idx] = float(bp.V_o)
        # The inductor's branch current is at branch_var_id_for_inductor.
        # The branch ID was just incremented inside add_inductor; the
        # safe way to find it is to walk branches.
        for bid in range(b.graph.num_branches):
            try:
                i_idx = b.pool.branch_var_id_for_inductor(
                    bid, b.graph)
                x0[i_idx] = float(bp.V_o / bp.R)
                break
            except Exception:
                continue
        initial_state = x0

    sw_fn = pv2.make_pwm_switch_fn(
        frequency=float(bp.f_sw),
        duty=float(duty),
        switch_idx=0,
        num_switches=b.graph.num_switches)

    res = pv2.simulate(
        b, t_end=float(t_end), dt=float(dt),
        switch_fn=sw_fn,
        initial_state=initial_state,
        max_event_iterations=8)

    # Extract node voltage at "out" from the state-vector history.
    out_idx = b.node_id_of("out")
    times = np.asarray(res.times, dtype=float)
    if hasattr(res.states, "shape"):
        states = np.asarray(res.states, dtype=float)
    else:
        states = np.asarray(
            [list(v) for v in res.states], dtype=float)
    v_out = states[:, out_idx]
    return times, v_out


def simulate_buck_v2_with_step(bp,
                                   duty_before: float,
                                   duty_after: float,
                                   t_step: float,
                                   t_end: float = 5e-3,
                                   dt: float = 1e-7,
                                   ) -> "tuple[np.ndarray, np.ndarray]":
    """Run a buck sim with a duty step at ``t_step`` for small-signal
    response cross-check.

    Builds the same circuit as :func:`simulate_buck_v2` but uses a
    time-varying switch function that flips duty at ``t_step``.
    Returns ``(times, v_out)`` for comparison with the analytical
    ``signal.step(Gvd, …)`` prediction.
    """
    import pulsim.v2 as pv2

    b = pv2.CircuitBuilder()
    b.add_voltage_source("Vdc", "vin", "gnd", float(bp.V_g))
    b.add_mosfet_with_body_diode("Q1", "vin", "sw",
                                      R_on=1e-3, R_off=1e9, V_F=0.7)
    b.add_diode("D_FW", "gnd", "sw",
                  g_on=1e3, g_off=1e-9, V_th=0.7)
    b.add_inductor("L1", "sw", "out", float(bp.L))
    b.add_capacitor("Cout", "out", "gnd", float(bp.C))
    b.add_resistor("Rload", "out", "gnd", float(bp.R))

    n_state = b.pool.state_size(b.graph)
    x0 = [0.0] * n_state
    out_idx = b.node_id_of("out")
    x0[out_idx] = float(bp.V_o)
    for bid in range(b.graph.num_branches):
        try:
            i_idx = b.pool.branch_var_id_for_inductor(bid, b.graph)
            x0[i_idx] = float(bp.V_o / bp.R)
            break
        except Exception:
            continue

    # Two PWM generators, switched at t_step.
    sw_before = pv2.make_pwm_switch_fn(
        frequency=float(bp.f_sw), duty=float(duty_before),
        switch_idx=0, num_switches=b.graph.num_switches)
    sw_after = pv2.make_pwm_switch_fn(
        frequency=float(bp.f_sw), duty=float(duty_after),
        switch_idx=0, num_switches=b.graph.num_switches)

    def sw_fn(t):
        return sw_before(t) if t < t_step else sw_after(t)

    res = pv2.simulate(
        b, t_end=float(t_end), dt=float(dt),
        switch_fn=sw_fn,
        initial_state=x0,
        max_event_iterations=8)

    times = np.asarray(res.times, dtype=float)
    if hasattr(res.states, "shape"):
        states = np.asarray(res.states, dtype=float)
    else:
        states = np.asarray(
            [list(v) for v in res.states], dtype=float)
    v_out = states[:, out_idx]
    return times, v_out
