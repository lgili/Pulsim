"""Three-phase NPC 3-level inverter cross-validation against Pulsim.

Builds the NPC 3-level topology in the Pulsim ``CircuitBuilder`` —
12 controlled switches (4 per leg × 3 legs), 6 clamping diodes
(2 per leg × 3 legs), a stiff split DC bus (two voltage sources
clamping the neutral point), and a Y-connected RL load. Pulsim ships
helpers for 2-level VSI topologies but not for NPC, so we wire
everything by hand.

Switch ordering (insertion order = bit index in the ``SwitchStateMask``)::

    leg A: S1a=0, S2a=1, S3a=2, S4a=3
    leg B: S1b=4, S2b=5, S3b=6, S4b=7
    leg C: S1c=8, S2c=9, S3c=10, S4c=11
    clamping diodes: indices 12..17 (event-driven, ignored by modulator)

PWM driver
----------

The standard NPC modulation is **multicarrier Phase-Disposition (PD)
PWM** — two triangular carriers stacked vertically over
``[-1, +1]``, both at the same frequency and phase. The reference is
compared against each:

* ``v_ref > tri_upper`` → state ``P`` → close ``S1, S2``
* ``v_ref < tri_lower`` → state ``N`` → close ``S3, S4``
* otherwise → state ``O`` → close ``S2, S3``

``make_npc_pd_pwm_fn`` builds the ``switch_fn`` closure that does this
for all 3 legs at once.

Design notes (Pulsim cache constraints)
---------------------------------------

* Switches use ``g_on = 10 S`` (R_on = 100 mΩ) rather than the textbook
  1 mΩ. The PWL cache eagerly enumerates state-space matrices for
  every 2^N switch combination, including invalid ones
  (shoot-through, leg open-circuit). With R_on too small those
  combinations become numerically singular; at 100 mΩ everything is
  well-conditioned and the modulator never commands invalid states
  anyway.
* The DC bus is modelled as two stiff voltage sources (each V_dc/2)
  clamping the neutral point at 0 V. This sacrifices the open-loop
  NP drift dynamics, but unlocks the basic 3-level waveform
  validation. The NP balancing study in ``02_npc_balancing.ipynb``
  uses a pure-Python forward-Euler simulator where the NP is a real
  state variable.
* Phase-output nodes are anchored to NP through 1 MΩ pull resistors —
  prevents any phase from becoming truly floating in the cache's
  enumeration, with negligible effect on real operation.
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np


def _build_npc_builder(p, R_load: float | None = None, L_load: float = 5e-3,
                         C_dc: float | None = None):
    """Construct the NPC 3-level builder.

    Node names (you can probe these via ``b.node_id_of("name")``):

    * ``vdc_pos`` (= +V_dc/2 above NP), ``vdc_neg`` (= -V_dc/2).
    * ``np`` — the neutral point, the midpoint of the DC bus split.
    * ``mid_a`` / ``mid_b`` / ``mid_c`` — the three phase outputs.
    * ``n1_a`` / ``n2_a`` (and same for b, c) — the internal NPC leg
      nodes between S1–S2 and S3–S4. Each is tied to the NP through
      one clamping diode.
    * ``n`` — the load Y-neutral, tied to ``np`` through ``R_neutral``.
    """
    import pulsim as pp

    R = float(R_load) if R_load is not None else float(p.R_load)
    C_split = float(C_dc) if C_dc is not None else float(p.C_dc)

    b = pp.CircuitBuilder()

    # DC bus: a single source split into two halves by C_dc + C_dc.
    # We put the source between vdc_pos and vdc_neg and let the cap
    # midpoint settle at half. Initially we tie the NP to "np" through
    # the two split caps; the load neutral connects to np too so the
    # circuit has a single connected component.
    # DC bus model: two stiff voltage sources clamping the neutral
    # point. This sacrifices the NP balancing dynamics (the NP cannot
    # drift in this model) but makes the basic 3-level waveform
    # validation tractable in the Pulsim PWL cache.
    #
    # The split-cap topology (V_dc / 2 caps in series) introduces
    # cache-build singularities because Pulsim's PWL cache eagerly
    # enumerates all 2^N switch combinations including invalid ones
    # (shoot-through, isolated phase outputs), some of which become
    # numerically singular when the only DC reference for NP is a
    # capacitor-only path.
    #
    # The NP balancing controller study lives in
    # ``02_npc_balancing.ipynb`` — that notebook uses a pure-Python
    # forward-Euler simulator where the NP is a real state variable
    # and we can directly observe its drift.
    b.add_voltage_source("Vdc_top", "vdc_pos", "np",      0.5 * float(p.V_dc))
    b.add_voltage_source("Vdc_bot", "np",      "vdc_neg", 0.5 * float(p.V_dc))
    # Track the (would-be) split caps as observers — they don't affect
    # the NP voltage but let us probe the cap currents for the
    # balancing-analysis notebook.
    b.add_capacitor("C_dc_top", "vdc_pos", "np", C_split)
    b.add_capacitor("C_dc_bot", "np", "vdc_neg", C_split)

    # ---- Three NPC legs ----
    # Add ALL 12 controllable switches FIRST so they get contiguous
    # bit indices [0..11] = [S1a, S2a, S3a, S4a, S1b, S2b, S3b, S4b,
    # S1c, S2c, S3c, S4c]. Clamping diodes (next loop) use
    # ``add_nonlinear_diode`` and do NOT enter the switch mask.
    #
    # NOTE: switch R_on chosen at 100 mΩ (g_on=10 S) — not the textbook
    # 1 mΩ. Pulsim's PWL cache eagerly enumerates all 2^N switch
    # combinations including invalid ones like shoot-through
    # (S1+S4 ON simultaneously). With R_on=1 mΩ those become
    # numerically singular (4 mΩ across V_dc → 100 kA). At 100 mΩ the
    # matrices stay well-conditioned and the modulator never commands
    # the invalid states anyway, so real-world operation is identical.
    g_on, g_off = 10.0, 1e-9
    for suffix in ("a", "b", "c"):
        n1 = f"n1_{suffix}"
        n2 = f"n2_{suffix}"
        mid = f"mid_{suffix}"
        b.add_switch(f"S1{suffix}", "vdc_pos", n1, g_on=g_on, g_off=g_off)
        b.add_switch(f"S2{suffix}", n1, mid,        g_on=g_on, g_off=g_off)
        b.add_switch(f"S3{suffix}", mid, n2,        g_on=g_on, g_off=g_off)
        b.add_switch(f"S4{suffix}", n2, "vdc_neg", g_on=g_on, g_off=g_off)

    # Now add the 6 clamping diodes (NPC's defining feature) — switched
    # diodes (event-driven, occupy switch mask). They land at indices
    # 12..17. The PWL cache still enumerates 2^18 combinations but the
    # generous switch / diode resistances (100 mΩ ON, 1 GΩ OFF) keep
    # all matrices well-conditioned.
    #
    # Clamping diode polarity:
    #   D_clamp_top: anode at NP, cathode at n1 — clamps n1 to NP
    #     when S1 OFF, S2 ON (state O upper half).
    #   D_clamp_bot: anode at n2, cathode at NP — clamps n2 to NP
    #     when S4 OFF, S3 ON (state O lower half).
    g_on_dio, g_off_dio = 10.0, 1e-9
    for suffix in ("a", "b", "c"):
        n1 = f"n1_{suffix}"
        n2 = f"n2_{suffix}"
        b.add_diode(f"D_c_top_{suffix}", "np", n1, g_on_dio, g_off_dio, V_th=0.7)
        b.add_diode(f"D_c_bot_{suffix}", n2, "np", g_on_dio, g_off_dio, V_th=0.7)

    # ---- Y-connected RL load ----
    for suffix in ("a", "b", "c"):
        mid = f"mid_{suffix}"
        node_load_mid = f"load_mid_{suffix}"
        b.add_resistor(f"R_{suffix}", mid, node_load_mid, R)
        b.add_inductor(f"L_{suffix}", node_load_mid, "n", float(L_load))

    # Tie the load neutral to the bus NP through a tiny resistor for MNA
    # connectivity (so the floating Y neutral is anchored).
    b.add_resistor("R_neutral", "n", "np", 1.0e-3)

    # ---- Numerical anchors on internal nodes ----
    # The PWL cache enumerates state-space matrices for ALL 2^N switch
    # combinations including invalid ones (e.g. S1+S4 ON with
    # S2+S3 OFF), some of which would leave internal nodes
    # electrically floating. Add very large pull-up/down resistors so
    # every node has a DC reference path regardless of switch state.
    # 1 MΩ is ~10000× the load impedance, so the impact on normal
    # operation is negligible (load currents in mA range, leakage
    # currents in µA).
    R_anchor = 1e6
    for suffix in ("a", "b", "c"):
        b.add_resistor(f"R_anc_n1_{suffix}", f"n1_{suffix}", "vdc_pos", R_anchor)
        b.add_resistor(f"R_anc_n2_{suffix}", f"n2_{suffix}", "vdc_neg", R_anchor)
        b.add_resistor(f"R_anc_mid_{suffix}", f"mid_{suffix}", "np", R_anchor)
    return b


def make_npc_pd_pwm_fn(p, m_a: float | None = None,
                          modulation_phase: float = 0.0,
                          carrier_phase: float = 0.0,
                          num_switches: int = 12) -> Callable:
    """Build a ``switch_fn(t)`` that drives the 12 NPC switches with
    multicarrier PD-PWM at the modulation index ``m_a``.

    Each leg's reference signal is a sinusoid at ``p.f_o`` with the
    standard ABC phase shifts (0°, -120°, -240°). The two PD carriers
    occupy ``[-1, 0]`` and ``[0, +1]`` at frequency ``p.f_sw``.

    The 12 switches are laid out in groups of 4 per leg:
    ``[S1a, S2a, S3a, S4a, S1b, S2b, S3b, S4b, S1c, S2c, S3c, S4c]``.
    Per-leg state-to-switch mapping (see
    :data:`npc_3phase_model.SWITCH_TABLE`):

    * P: S1=ON, S2=ON, S3=OFF, S4=OFF
    * O: S1=OFF, S2=ON, S3=ON, S4=OFF
    * N: S1=OFF, S2=OFF, S3=ON, S4=ON
    """
    import pulsim as pp

    if m_a is None:
        m_a = float(p.m_a)
    omega_o = 2.0 * math.pi * float(p.f_o)
    f_sw = float(p.f_sw)
    leg_phases = (0.0, -2.0 * math.pi / 3.0, -4.0 * math.pi / 3.0)

    def switch_fn(t):
        mask = pp.SwitchStateMask(num_switches)
        # One common pair of triangular carriers across all three legs.
        carrier_phi = (2.0 * math.pi * f_sw * t + carrier_phase) % (2.0 * math.pi)
        tri = (2.0 / math.pi) * math.asin(math.sin(carrier_phi))
        tri_upper = 0.5 + 0.5 * tri
        tri_lower = -0.5 + 0.5 * tri

        for leg_idx, theta_phi in enumerate(leg_phases):
            v_ref = m_a * math.sin(omega_o * t + modulation_phase - theta_phi)
            base = leg_idx * 4

            if v_ref > tri_upper:
                # State P: close S1, S2.
                mask.set(base + 0, True)
                mask.set(base + 1, True)
            elif v_ref < tri_lower:
                # State N: close S3, S4.
                mask.set(base + 2, True)
                mask.set(base + 3, True)
            else:
                # State O: close S2, S3.
                mask.set(base + 1, True)
                mask.set(base + 2, True)

        return mask

    return switch_fn


def simulate_npc(p,
                  m_a: float | None = None,
                  t_end: float = 0.05,
                  dt: float = 5e-7,
                  ) -> "tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]":
    """Run an open-loop NPC 3-level simulation with PD-PWM.

    Returns ``(times, v_mid_a, v_mid_b, v_mid_c, v_np)`` — the three
    phase-leg midpoint voltages (w.r.t. ``vdc_neg``, so they range
    over ``[0, V_dc]`` not ``[-V_dc/2, V_dc/2]``), plus the neutral
    point voltage (so you can watch it drift open-loop).
    """
    import pulsim as pp

    if m_a is None:
        m_a = float(p.m_a)

    b = _build_npc_builder(p)
    sw_fn = make_npc_pd_pwm_fn(p, m_a=m_a, num_switches=b.graph.num_switches)

    res = pp.simulate(
        b, t_end=float(t_end), dt=float(dt),
        switch_fn=sw_fn,
        max_event_iterations=8,
    )

    a_idx = b.node_id_of("mid_a")
    b_idx = b.node_id_of("mid_b")
    c_idx = b.node_id_of("mid_c")
    np_idx = b.node_id_of("np")
    times = np.asarray(res.times, dtype=float)
    states = (np.asarray(res.states, dtype=float)
              if hasattr(res.states, "shape")
              else np.asarray([list(v) for v in res.states], dtype=float))
    v_a = states[:, a_idx]
    v_b = states[:, b_idx]
    v_c = states[:, c_idx]
    v_np = states[:, np_idx]
    return times, v_a, v_b, v_c, v_np
