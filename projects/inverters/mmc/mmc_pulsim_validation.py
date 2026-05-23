"""Single-phase MMC cross-validation against Pulsim.

Builds a single-phase MMC with **N = 3** sub-modules per arm in the
Pulsim ``CircuitBuilder``. Each sub-module is a half-bridge with two
controllable switches (insertion + bypass) and a floating capacitor.
A stiff DC source supplies $\\pm V_{dc}/2$ on the bus rails.

Switch ordering (insertion order = bit index in the switch mask)::

    upper arm:
      SM_u0: S1_u0=0, S2_u0=1
      SM_u1: S1_u1=2, S2_u1=3
      SM_u2: S1_u2=4, S2_u2=5
    lower arm:
      SM_l0: S1_l0=6, S2_l0=7
      SM_l1: S1_l1=8, S2_l1=9
      SM_l2: S1_l2=10, S2_l2=11

Per-SM switch convention (half-bridge):

* ``S1`` (top) between SM's internal cap-top and the SM midpoint
  (= terminal A).
* ``S2`` (bottom) between the SM midpoint and the cap-bottom
  (= terminal B).
* INSERT state: S1 ON, S2 OFF → v_SM = +v_cap.
* BYPASS state: S1 OFF, S2 ON → v_SM = 0.
* Forbidden: both ON (cap short) or both OFF (arm current has no path).

Design notes (same Pulsim cache constraints as ``npc_3phase``)
--------------------------------------------------------------

* Switches use ``g_on = 10 S`` (R_on = 100 mΩ) — keeps every
  2^N enumerated mask numerically well-conditioned.
* DC bus modelled as two stiff voltage sources — sacrifices the
  open-loop bus-voltage dynamics, but unlocks the basic multilevel
  waveform validation.
* Each SM cap and arm node anchored with 1 MΩ pull resistors to
  prevent floating nodes during the cache's exhaustive enumeration.
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np


def _build_mmc_builder(p, R_load: float | None = None,
                          L_load: float | None = None):
    """Construct the single-phase MMC builder.

    Node-name conventions:

    * ``vdc_pos`` / ``vdc_neg`` — DC rails at ±V_dc/2 relative to
      ``gnd_dc`` (the bus midpoint).
    * ``arm_up_top`` — top of upper-arm inductor (= ``vdc_pos`` side).
    * ``arm_lo_bot`` — bottom of lower-arm inductor (= ``vdc_neg`` side).
    * ``arm_up_in[i]`` / ``arm_up_out[i]`` — series junctions inside
      the upper arm chain. ``arm_up_in[0]`` = bottom of L_arm_up,
      ``arm_up_out[N-1]`` = ``ac_out``.
    * ``cap_top_up_<i>`` / ``cap_top_lo_<i>`` — internal cap-positive
      node of each SM (the cap-negative is the SM's terminal B = arm
      junction).
    * ``ac_out`` — single AC output (the junction between upper and
      lower arms).
    * ``ac_load_mid`` — internal node between R_load and L_load.
    * ``gnd_ac`` — return side of the AC load.
    """
    import pulsim as pp

    R = float(R_load) if R_load is not None else float(p.R_load)
    L = float(L_load) if L_load is not None else float(p.L_load)
    N = int(p.N_sm)

    b = pp.CircuitBuilder()

    # ---- Stiff DC bus: ±V_dc/2 around the bus midpoint ----
    b.add_voltage_source("Vdc_top", "vdc_pos", "gnd_dc", 0.5 * float(p.V_dc))
    b.add_voltage_source("Vdc_bot", "gnd_dc", "vdc_neg", 0.5 * float(p.V_dc))

    # ---- Arm inductors ----
    b.add_inductor("L_arm_up", "vdc_pos", "arm_up_0_in", float(p.L_arm))
    b.add_inductor("L_arm_lo", "arm_lo_2_out", "vdc_neg", float(p.L_arm))

    # ---- Upper arm: N SMs in series, switches FIRST so mask indices are contiguous ----
    # Junction nodes: arm_up_0_in → arm_up_1_in → arm_up_2_in → ac_out.
    # (We name them sequentially so each SM i has in=arm_up_i_in, out=arm_up_{i+1}_in,
    #  with arm_up_N_in renamed to ac_out for the last SM.)
    def junction_up(i: int) -> str:
        if i == N:
            return "ac_out"
        return f"arm_up_{i}_in"

    g_on, g_off = 10.0, 1e-9
    # Add ALL controllable switches first (12 total) so mask indices
    # [0..11] are the controllable ones.
    for i in range(N):
        # SM_u<i>:  midpoint (= terminal A) = junction_up(i)
        #           terminal B (= cap_bot) = junction_up(i+1)
        #           internal cap_top = "cap_top_up_<i>"
        midpoint = junction_up(i)
        cap_bot = junction_up(i + 1)
        cap_top = f"cap_top_up_{i}"
        # S1 between cap_top and midpoint  (INSERT → cap in series)
        b.add_switch(f"S1_up_{i}", cap_top, midpoint, g_on=g_on, g_off=g_off)
        # S2 between midpoint and cap_bot  (BYPASS → cap shorted out)
        b.add_switch(f"S2_up_{i}", midpoint, cap_bot, g_on=g_on, g_off=g_off)

    def junction_lo(i: int) -> str:
        if i == 0:
            return "ac_out"
        if i == N:
            return "arm_lo_2_out"          # bottom of arm = top of L_arm_lo
        return f"arm_lo_{i}_in"

    for i in range(N):
        midpoint = junction_lo(i)
        cap_bot = junction_lo(i + 1)
        cap_top = f"cap_top_lo_{i}"
        b.add_switch(f"S1_lo_{i}", cap_top, midpoint, g_on=g_on, g_off=g_off)
        b.add_switch(f"S2_lo_{i}", midpoint, cap_bot, g_on=g_on, g_off=g_off)

    # ---- Now the 6 SM caps (do NOT enter the switch mask) ----
    for i in range(N):
        cap_top = f"cap_top_up_{i}"
        cap_bot = junction_up(i + 1)
        b.add_capacitor(f"C_up_{i}", cap_top, cap_bot, float(p.C_sm))
    for i in range(N):
        cap_top = f"cap_top_lo_{i}"
        cap_bot = junction_lo(i + 1)
        b.add_capacitor(f"C_lo_{i}", cap_top, cap_bot, float(p.C_sm))

    # ---- AC load (RL, single-phase) ----
    b.add_resistor("R_load", "ac_out", "ac_load_mid", R)
    b.add_inductor("L_load", "ac_load_mid", "gnd_ac", L)

    # ---- Tie AC return to DC midpoint with a tiny resistor ----
    b.add_resistor("R_ac_to_gnd", "gnd_ac", "gnd_dc", 1.0e-3)

    # ---- Numerical anchors on every potentially-floating node ----
    # Same defensive pattern as npc_3phase: prevents the cache's
    # exhaustive enumeration from hitting singular configurations.
    R_anchor = 1e6
    for i in range(N):
        b.add_resistor(f"R_anc_cap_up_{i}", f"cap_top_up_{i}", "gnd_dc", R_anchor)
        b.add_resistor(f"R_anc_cap_lo_{i}", f"cap_top_lo_{i}", "gnd_dc", R_anchor)
    for i in range(1, N):
        b.add_resistor(f"R_anc_up_{i}", f"arm_up_{i}_in", "gnd_dc", R_anchor)
        if i < N:
            b.add_resistor(f"R_anc_lo_{i}", f"arm_lo_{i}_in", "gnd_dc", R_anchor)
    b.add_resistor("R_anc_ac_out", "ac_out", "gnd_dc", R_anchor)

    return b


def make_mmc_psc_pwm_fn(p, m_a: float | None = None,
                            modulation_phase: float = 0.0,
                            num_switches: int = 12) -> Callable:
    """Build a ``switch_fn(t)`` that drives the 12 MMC switches with
    Phase-Shifted Carrier (PSC) PWM.

    Reference signals (open-loop, no cap balancing):

        d_up(t) = (1 - m_a · sin(omega_o·t)) / 2
        d_lo(t) = (1 + m_a · sin(omega_o·t)) / 2

    For each arm, PSC-PWM compares the duty against N carriers
    phase-shifted by 2π/N. The number-of-carriers-below-duty equals
    the number of SMs to insert. Without sort-and-select, we simply
    insert the FIRST n SMs of the arm (i.e. SM_0, SM_1, ..., SM_{n-1}).
    This is the open-loop scheme — cap voltages will drift unless
    balanced.

    The 12 switches are laid out in groups of 2 per SM:
    ``[S1_u0, S2_u0, S1_u1, S2_u1, S1_u2, S2_u2,
       S1_l0, S2_l0, S1_l1, S2_l1, S1_l2, S2_l2]``.
    Per-SM mapping:

    * INSERT (n_inserted > i): S1 ON, S2 OFF.
    * BYPASS (n_inserted ≤ i): S1 OFF, S2 ON.
    """
    import pulsim as pp

    if m_a is None:
        m_a = float(p.m_a)
    omega_o = 2.0 * math.pi * float(p.f_o)
    f_carrier = float(p.f_carrier)
    N = int(p.N_sm)

    def switch_fn(t):
        mask = pp.SwitchStateMask(num_switches)
        # Arm references.
        s = math.sin(omega_o * t + modulation_phase)
        d_up = max(0.0, min(1.0, 0.5 * (1.0 - m_a * s)))
        d_lo = max(0.0, min(1.0, 0.5 * (1.0 + m_a * s)))

        # PSC-PWM: count carriers below the reference.
        def n_insert_arm(duty: float) -> int:
            n = 0
            for k in range(N):
                phi = (2.0 * math.pi * f_carrier * t
                        + k * 2.0 * math.pi / N) % (2.0 * math.pi)
                tri = 0.5 + 0.5 * (2.0 / math.pi) * math.asin(math.sin(phi))
                if duty > tri:
                    n += 1
            return n

        n_up = n_insert_arm(d_up)
        n_lo = n_insert_arm(d_lo)

        # Apply to upper arm: insert the first n_up SMs.
        for i in range(N):
            base = i * 2
            if i < n_up:
                mask.set(base + 0, True)   # S1 = ON
            else:
                mask.set(base + 1, True)   # S2 = ON
        # Apply to lower arm: insert the first n_lo SMs.
        for i in range(N):
            base = N * 2 + i * 2
            if i < n_lo:
                mask.set(base + 0, True)
            else:
                mask.set(base + 1, True)

        return mask

    return switch_fn


def make_mmc_sort_and_select_fn(p, builder,
                                    m_a: float | None = None,
                                    modulation_phase: float = 0.0):
    """Build (switch_fn, step_observer) for PSC-PWM with sort-and-select
    capacitor balancing.

    The step observer reads the SM cap voltages and arm-current signs
    out of the state vector at every dt and updates a closure shared
    with the switch_fn. The switch_fn uses PSC-PWM to decide HOW MANY
    SMs to insert per arm, then sort-and-select to decide WHICH SMs
    (inserting the lowest-voltage caps when the arm is charging, the
    highest when discharging).

    This is the canonical MMC balancing technique — robust, simple
    and scales to any N.
    """
    import pulsim as pp

    if m_a is None:
        m_a = float(p.m_a)
    omega_o = 2.0 * math.pi * float(p.f_o)
    f_carrier = float(p.f_carrier)
    N = int(p.N_sm)
    num_switches = builder.graph.num_switches

    # Cap top/bot node indices (closure-captured).
    cap_up_idx_top = [builder.node_id_of(f"cap_top_up_{i}") for i in range(N)]
    cap_lo_idx_top = [builder.node_id_of(f"cap_top_lo_{i}") for i in range(N)]
    cap_up_idx_bot = []
    cap_lo_idx_bot = []
    for i in range(N):
        if i + 1 == N:
            cap_up_idx_bot.append(builder.node_id_of("ac_out"))
            cap_lo_idx_bot.append(builder.node_id_of("arm_lo_2_out"))
        else:
            cap_up_idx_bot.append(builder.node_id_of(f"arm_up_{i+1}_in"))
            cap_lo_idx_bot.append(builder.node_id_of(f"arm_lo_{i+1}_in"))

    # Inductor branch indices for arm currents.
    # The first two inductors added are L_arm_up and L_arm_lo (in that order).
    arm_current_indices = []
    for bid in range(builder.graph.num_branches):
        try:
            i_idx = builder.pool.branch_var_id_for_inductor(bid, builder.graph)
            arm_current_indices.append(i_idx)
            if len(arm_current_indices) >= 2:
                break
        except Exception:
            continue
    if len(arm_current_indices) < 2:
        raise RuntimeError("Could not locate arm-inductor branch currents.")

    state = {
        "v_caps_up": np.full(N, float(p.V_C_nominal)),
        "v_caps_lo": np.full(N, float(p.V_C_nominal)),
        "i_arm_up": 0.0,
        "i_arm_lo": 0.0,
    }

    def step_observer(_t, x):               # observer is time-independent
        x_arr = np.asarray(x)
        for i in range(N):
            state["v_caps_up"][i] = x_arr[cap_up_idx_top[i]] - x_arr[cap_up_idx_bot[i]]
            state["v_caps_lo"][i] = x_arr[cap_lo_idx_top[i]] - x_arr[cap_lo_idx_bot[i]]
        state["i_arm_up"] = float(x_arr[arm_current_indices[0]])
        state["i_arm_lo"] = float(x_arr[arm_current_indices[1]])

    def switch_fn(t):
        mask = pp.SwitchStateMask(num_switches)
        s = math.sin(omega_o * t + modulation_phase)
        d_up = max(0.0, min(1.0, 0.5 * (1.0 - m_a * s)))
        d_lo = max(0.0, min(1.0, 0.5 * (1.0 + m_a * s)))

        def n_insert_arm(duty: float) -> int:
            n = 0
            for k in range(N):
                phi = (2.0 * math.pi * f_carrier * t
                        + k * 2.0 * math.pi / N) % (2.0 * math.pi)
                tri = 0.5 + 0.5 * (2.0 / math.pi) * math.asin(math.sin(phi))
                if duty > tri:
                    n += 1
            return n

        n_up = n_insert_arm(d_up)
        n_lo = n_insert_arm(d_lo)

        # Sort-and-select: select WHICH SMs to insert based on cap
        # voltage + arm current sign.
        from mmc_model import sort_and_select
        sel_up = sort_and_select(n_up, state["v_caps_up"],
                                    +1 if state["i_arm_up"] > 0 else -1)
        sel_lo = sort_and_select(n_lo, state["v_caps_lo"],
                                    +1 if state["i_arm_lo"] > 0 else -1)

        for i in range(N):
            base = i * 2
            if sel_up[i]:
                mask.set(base + 0, True)
            else:
                mask.set(base + 1, True)
        for i in range(N):
            base = N * 2 + i * 2
            if sel_lo[i]:
                mask.set(base + 0, True)
            else:
                mask.set(base + 1, True)
        return mask

    return switch_fn, step_observer


def simulate_mmc(p,
                  m_a: float | None = None,
                  t_end: float = 0.05,
                  dt: float = 5e-6,
                  balance_caps: bool = True,
                  ) -> dict:
    """Run a single-phase MMC simulation with PSC-PWM.

    Parameters
    ----------
    p
        :class:`MMCParams` instance.
    m_a
        Modulation index (defaults to ``p.m_a``).
    t_end, dt
        Simulation window and timestep.
    balance_caps
        If ``True`` (default), use **sort-and-select** capacitor
        balancing — the canonical MMC technique. If ``False``, use
        the naive "insert the first n SMs always" rule that
        dramatically illustrates the open-loop drift problem.

    Returns a dict with arrays:

    * ``times``
    * ``v_ac``         — phase output voltage (= V(ac_out) - V(gnd_dc))
    * ``v_arm_up``     — upper-arm voltage across the SM stack
    * ``v_arm_lo``     — lower-arm voltage across the SM stack
    * ``v_caps_up``    — (N, T) array of upper-arm cap voltages
    * ``v_caps_lo``    — (N, T) array of lower-arm cap voltages
    """
    import pulsim as pp

    if m_a is None:
        m_a = float(p.m_a)

    b = _build_mmc_builder(p)
    if balance_caps:
        sw_fn, step_obs = make_mmc_sort_and_select_fn(p, b, m_a=m_a)
        res = pp.simulate(
            b, t_end=float(t_end), dt=float(dt),
            switch_fn=sw_fn,
            step_observer=step_obs,
            max_event_iterations=8,
        )
    else:
        sw_fn = make_mmc_psc_pwm_fn(p, m_a=m_a,
                                       num_switches=b.graph.num_switches)
        res = pp.simulate(
            b, t_end=float(t_end), dt=float(dt),
            switch_fn=sw_fn,
            max_event_iterations=8,
        )

    ac_idx = b.node_id_of("ac_out")
    gnd_dc_idx = b.node_id_of("gnd_dc")
    arm_up_in_idx = b.node_id_of("arm_up_0_in")
    arm_lo_out_idx = b.node_id_of("arm_lo_2_out")
    cap_up_idx = [b.node_id_of(f"cap_top_up_{i}") for i in range(p.N_sm)]
    cap_lo_idx = [b.node_id_of(f"cap_top_lo_{i}") for i in range(p.N_sm)]
    # cap-bottom node names (= arm junction below the SM)
    cap_up_bot_idx = []
    for i in range(p.N_sm):
        if i + 1 == p.N_sm:
            cap_up_bot_idx.append(ac_idx)
        else:
            cap_up_bot_idx.append(b.node_id_of(f"arm_up_{i+1}_in"))
    cap_lo_bot_idx = []
    for i in range(p.N_sm):
        if i + 1 == p.N_sm:
            cap_lo_bot_idx.append(arm_lo_out_idx)
        else:
            cap_lo_bot_idx.append(b.node_id_of(f"arm_lo_{i+1}_in"))

    times = np.asarray(res.times, dtype=float)
    states = (np.asarray(res.states, dtype=float)
              if hasattr(res.states, "shape")
              else np.asarray([list(v) for v in res.states], dtype=float))

    v_gnd_dc = states[:, gnd_dc_idx]
    v_ac = states[:, ac_idx] - v_gnd_dc
    v_arm_up = states[:, arm_up_in_idx] - states[:, ac_idx]
    v_arm_lo = states[:, ac_idx] - states[:, arm_lo_out_idx]
    v_caps_up = np.array([
        states[:, t] - states[:, b_] for t, b_ in zip(cap_up_idx, cap_up_bot_idx)
    ])
    v_caps_lo = np.array([
        states[:, t] - states[:, b_] for t, b_ in zip(cap_lo_idx, cap_lo_bot_idx)
    ])

    return {
        "times":     times,
        "v_ac":      v_ac,
        "v_arm_up":  v_arm_up,
        "v_arm_lo":  v_arm_lo,
        "v_caps_up": v_caps_up,
        "v_caps_lo": v_caps_lo,
    }
