"""Buck converter cross-validation against Pulsim.

Builds the same physical buck topology the analytical state-space model
fits — ``V_g → MOSFET + body diode → freewheeling diode → L → C ∥
R_load → gnd`` — and runs a fixed-dt Pulsim transient at a constant
PWM duty. Returns the simulated ``(times, v_out)`` arrays for comparison
against the analytical step response.

Notes on the Pulsim API used here:

* Nodes are inferred from the string names passed to ``add_*`` — no
  explicit ``add_node`` step is needed.
* Switching is decoupled from the topology. ``add_mosfet_with_body_diode``
  registers the device; an external ``switch_fn`` (see
  ``pulsim.make_pwm_switch_fn``) drives the gate.
* Initial conditions are passed via the ``initial_state`` kwarg of
  ``pulsim.simulate``; ``add_inductor`` / ``add_capacitor`` themselves
  take no IC argument.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def _build_buck_builder(bp):
    """Construct the Pulsim ``CircuitBuilder`` for an ideal buck.

    Private helper — both :func:`simulate_buck` and
    :func:`simulate_buck_with_step` share exactly this topology.
    """
    import pulsim as p

    b = p.CircuitBuilder()
    b.add_voltage_source("Vdc", "vin", "gnd", float(bp.V_g))
    b.add_mosfet_with_body_diode("Q1", "vin", "sw",
                                   R_on=1e-3, R_off=1e9, V_F=0.7)
    b.add_diode("D_FW", "gnd", "sw", 1e3, 1e-9, V_th=0.7)
    b.add_inductor("L1", "sw", "out", float(bp.L))
    b.add_capacitor("Cout", "out", "gnd", float(bp.C))
    b.add_resistor("Rload", "out", "gnd", float(bp.R))
    return b


def _warm_start(b, bp):
    """Seed the state vector at the ideal CCM operating point
    ``I_L = V_o/R``, ``V_C = V_o``."""
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
    return x0


def simulate_buck(bp,
                   duty: float,
                   t_end: float = 5e-3,
                   dt: float = 1e-7,
                   warm_start: bool = True,
                   ) -> "tuple[np.ndarray, np.ndarray]":
    """Run a Pulsim buck simulation with constant duty.

    Parameters
    ----------
    bp
        A :class:`BuckParams` instance with ``V_g``, ``V_o``, ``L``,
        ``C``, ``R``, ``f_sw`` attributes.
    duty
        Steady-state duty cycle (0–1).
    t_end
        Simulation length in seconds. Default 5 ms covers ~8 LC
        natural periods of the design defaults.
    dt
        Fixed simulation step. Default 100 ns — well below the PWM
        period for any switching converter.
    warm_start
        If True, seed the inductor at ``I_L = V_o/R`` and the cap at
        ``V_o`` so the captured waveform is the steady-state ripple,
        not the inrush transient.

    Returns
    -------
    (times, v_out)
        Two parallel numpy arrays. ``v_out`` is the voltage at the
        ``"out"`` node sampled at ``dt``.
    """
    import pulsim as p

    b = _build_buck_builder(bp)

    initial_state: Optional[list] = _warm_start(b, bp) if warm_start else None

    sw_fn = p.make_pwm_switch_fn(
        frequency=float(bp.f_sw),
        duty=float(duty),
        switch_idx=0,
        num_switches=b.graph.num_switches,
    )

    res = p.simulate(
        b, t_end=float(t_end), dt=float(dt),
        switch_fn=sw_fn,
        initial_state=initial_state,
        max_event_iterations=8,
    )

    out_idx = b.node_id_of("out")
    times = np.asarray(res.times, dtype=float)
    states = (np.asarray(res.states, dtype=float)
              if hasattr(res.states, "shape")
              else np.asarray([list(v) for v in res.states], dtype=float))
    v_out = states[:, out_idx]
    return times, v_out


def simulate_buck_with_step(bp,
                              duty_before: float,
                              duty_after: float,
                              t_step: float,
                              t_end: float = 5e-3,
                              dt: float = 1e-7,
                              ) -> "tuple[np.ndarray, np.ndarray]":
    """Run a buck sim with a duty step at ``t_step``.

    Builds the same circuit as :func:`simulate_buck` and uses a
    time-varying switch function that flips duty at ``t_step``.
    Returns ``(times, v_out)`` so the transient can be overlaid on
    ``scipy.signal.step(Gvd, …)``.
    """
    import pulsim as p

    b = _build_buck_builder(bp)
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

    out_idx = b.node_id_of("out")
    times = np.asarray(res.times, dtype=float)
    states = (np.asarray(res.states, dtype=float)
              if hasattr(res.states, "shape")
              else np.asarray([list(v) for v in res.states], dtype=float))
    v_out = states[:, out_idx]
    return times, v_out
