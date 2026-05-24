"""Flyback converter cross-validation against Pulsim.

Builds an isolated flyback in the Pulsim ``CircuitBuilder``. The
transformer is a two-winding linear coupled inductor with magnetising
inductance ``L_p = L_m`` and ``L_s = n² · L_m`` so the turns ratio is
``n = sqrt(L_s / L_p)``.

Topology::

   Vin ── T1.primary ── Q1 ── gnd     T1.secondary ── D1 ── vout
                                                                │
                                       vout ── Cout ── R_load ── sec_neg
                                                                │
                                       sec_neg ── R_iso ── gnd

The secondary ground is tied to primary ground through a tiny
``R_iso = 1 µΩ`` so the MNA sees a single connected component without
breaking the isolation in any meaningful electrical sense.
"""

from __future__ import annotations

import numpy as np


def _build_builder(bp, k_coupling: float = 0.99):
    import pulsim as p

    L_p = float(bp.L_m)
    L_s = float(bp.n) ** 2 * float(bp.L_m)

    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", float(bp.V_g))
    b.add_mosfet_with_body_diode("Q1", "sw", "gnd",
                                   R_on=1e-2, R_off=1e9, V_F=0.7)
    b.add_transformer(
        "T1",
        p_from="vin", p_to="sw",
        s_from="sec_anode", s_to="sec_neg",
        L_p=L_p, L_s=L_s, k=float(k_coupling),
    )
    b.add_diode("D1", "sec_anode", "vout", 1e3, 1e-9, V_th=0.7)
    b.add_capacitor("Cout", "vout", "sec_neg", float(bp.C))
    b.add_resistor("Rload", "vout", "sec_neg", float(bp.R))
    b.add_resistor("Rgnd", "sec_neg", "gnd", 1.0e-6)
    return b


def simulate_flyback(bp,
                      duty: float,
                      t_end: float = 1e-3,
                      dt: float = 2e-8,
                      k_coupling: float = 0.99,
                      ) -> "tuple[np.ndarray, np.ndarray]":
    """Run a Pulsim flyback simulation with constant duty.

    Cold-started (no warm-state) because the transformer coupling
    makes the steady-state IC vector non-trivial. With ``t_end = 1 ms``
    the design defaults settle to within ~10 % of the analytical
    operating point.

    Returns ``(times, v_out)``.
    """
    import pulsim as p

    b = _build_builder(bp, k_coupling=k_coupling)

    sw_fn = p.make_pwm_switch_fn(
        frequency=float(bp.f_sw), duty=float(duty),
        switch_idx=0, num_switches=b.graph.num_switches,
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
