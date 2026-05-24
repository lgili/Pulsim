"""Forward converter cross-validation against Pulsim.

Builds a transformer-isolated forward converter in the Pulsim
``CircuitBuilder``. The forward is "an isolated buck": energy is
transferred primary → secondary **during the ON interval**, an explicit
output LC filter stores energy, and a freewheel diode maintains
inductor current during OFF — exactly like a buck.

Topology::

   Vin ── T1.primary ── Q1 ── gnd
                                                 D_rec       L_f       vout
   T1.secondary (sec_anode → sec_neg)            ────►───────╲╲╲╲──────┬──
                                                                       │
                                                  D_fw                Cout
                                                  ▲                    │
                                                 sec_neg ────────────  R_load
                                                                       │
                                                  sec_neg ── R_iso ── gnd

The textbook reset winding is omitted because Pulsim's
``add_transformer`` exposes only a two-winding device. On a linear
(non-saturating) magnetic, steady-state operation is indistinguishable
from the reset-winding-present case — the DC ratio still matches
``V_o = n · D · V_g``.
"""

from __future__ import annotations

import numpy as np


def _build_builder(bp, k_coupling: float = 0.999):
    import pulsim as p

    # Make the magnetising inductance much larger than the filter
    # inductor so the transformer is "stiff" (energy passes through,
    # not stored).
    L_p_mag = max(float(bp.L) * 100.0, 1e-3)
    L_s_mag = float(bp.n) ** 2 * L_p_mag

    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", float(bp.V_g))
    b.add_mosfet_with_body_diode("Q1", "sw", "gnd",
                                   R_on=1e-2, R_off=1e9, V_F=0.7)
    b.add_transformer(
        "T1",
        p_from="vin", p_to="sw",
        s_from="sec_anode", s_to="sec_neg",
        L_p=L_p_mag, L_s=L_s_mag, k=float(k_coupling),
    )
    b.add_diode("D_rec", "sec_anode", "l_in", 1e3, 1e-9, V_th=0.7)
    b.add_diode("D_fw", "sec_neg", "l_in", 1e3, 1e-9, V_th=0.7)
    b.add_inductor("L_f", "l_in", "vout", float(bp.L))
    b.add_capacitor("Cout", "vout", "sec_neg", float(bp.C))
    b.add_resistor("Rload", "vout", "sec_neg", float(bp.R))
    b.add_resistor("Rgnd", "sec_neg", "gnd", 1.0e-6)
    return b


def simulate_forward(bp,
                      duty: float,
                      t_end: float = 2e-3,
                      dt: float = 2e-8,
                      k_coupling: float = 0.999,
                      ) -> "tuple[np.ndarray, np.ndarray]":
    """Run a Pulsim forward simulation with constant duty.

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
