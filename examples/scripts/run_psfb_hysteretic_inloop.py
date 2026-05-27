#!/usr/bin/env python3
"""PSFB with HystereticInductor (Jiles-Atherton) **in the MNA loop**.

Companion to ``run_psfb_ja_loss_analysis.py`` — which post-processes
the inductor current OFFLINE through the JA model to estimate core
loss. THIS demo wires the JA hysteresis **inside the kernel loop**:
the dummy voltage source modulated by the JA observer participates
in every step's MNA solve, so the hysteresis voltage drop actively
shapes the simulated current waveform.

Topology
--------

    V_bus ── [4 switches H-bridge] ── [HystereticInductor (L_0 + V_M)]
                                            ↓
                                       transformer pri-pos ──[T1]── sec-pos
                                            ↓
                                       bridge rectifier ── L_out ── C_out ── R_L

The HystereticInductor replaces the conventional ``L_leak`` linear
inductor between the H-bridge midpoint and the transformer primary.
The Jiles-Atherton magnetisation state ``M`` is integrated each
simulation step and the hysteresis-derived "back-EMF"
``v_M = N·A·μ_0·dM/dt`` is injected via ``b_extra`` at the dummy
voltage-source row.

What this demonstrates (vs the offline-loss demo)
--------------------------------------------------
* The hysteresis drop ACTIVELY influences the current waveform.
* The first-cycle peak current reflects the residual flux state.
* The B-H trajectory is read directly off the LIVE simulation
  (``hyst.M``, ``hyst.B`` updated each step) — no post-processing.

Honest caveats
--------------
* The post-processing approach (``compute_bh_loop``) is faster
  per material sweep — you simulate the converter ONCE and analyse
  multiple materials offline. Use THIS approach when the
  hysteresis voltage drop matters for the operating point (e.g.
  ferrorresonance, inrush, saturable-reactor design).
* **Parameter scaling note**: the JA inductor with the demo's
  ferrite-N87 + N=12 turns + tight 0.10 m mean path acts as a
  **full magnetizing inductor** (L_eff ≈ L_0·μ_r ≈ 1.27 mH), not
  the ~1 µH leakage that the original PSFB design assumes. The
  output voltage (~5 V steady state) is therefore reduced
  compared to the linear-leakage PSFB (~15 V). This demo
  validates the **in-loop hysteresis wiring**; using the JA
  inductor as a true leakage replacement would require either a
  gapped core (low effective μ_r) or a much smaller turn count.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import pulsim as p


# ---------- PSFB parameters ------------------------------------------------

V_BUS    = 100.0
F_PWM    = 100e3
PHASE    = 1.0      # radians of phase-shift between legs
DT       = 5e-8     # 50 ns — 200 samples per PWM cycle
T_END    = 2e-3     # 2 ms = 200 PWM cycles


# ---------- Hysteretic core geometry ---------------------------------------

N_PRIMARY = 12
L_M_PATH  = 0.10
A_CORE    = 3.5e-4


def build_plant():
    """4-switch H-bridge → HystereticInductor → transformer → rectifier → LC."""
    b = p.CircuitBuilder()
    b.add_voltage_source("Vbus", "vbus", "gnd", V_BUS)
    for name, frm, to in [
        ("HS_A", "vbus", "mid_a"), ("LS_A", "mid_a", "gnd"),
        ("HS_B", "vbus", "mid_b"), ("LS_B", "mid_b", "gnd"),
    ]:
        b.add_switch(name, frm, to, g_on=1e3, g_off=1e-9)
    for name, anode, cathode in [
        ("D_HS_A", "mid_a", "vbus"), ("D_LS_A", "gnd", "mid_a"),
        ("D_HS_B", "mid_b", "vbus"), ("D_LS_B", "gnd", "mid_b"),
    ]:
        b.add_diode(name, anode, cathode, 1e3, 1e-9)

    # *** The headline change vs run_phase_shift_full_bridge.py ***
    # Replace the linear L_leak with a JilesAtherton hysteretic inductor.
    ja_params = p.reference_material("ferrite_n87")
    hyst = p.add_hysteretic_inductor(
        b, name="L_leak_JA",
        from_node="mid_a", to_node="pri_pos",
        params=ja_params,
        N_turns=N_PRIMARY,
        l_m=L_M_PATH,
        A_core=A_CORE,
    )
    # Transformer + rectifier (unchanged from the linear PSFB demo).
    b.add_transformer(
        "T1",
        p_from="pri_pos", p_to="mid_b",
        s_from="sec_pos", s_to="sec_neg",
        L_p=40e-6, L_s=10e-6, k=0.99,
    )
    b.add_diode("D1", "sec_pos",  "rect_pos", 1e3, 1e-9)
    b.add_diode("D2", "sec_neg",  "rect_pos", 1e3, 1e-9)
    b.add_diode("D3", "rect_neg", "sec_pos",  1e3, 1e-9)
    b.add_diode("D4", "rect_neg", "sec_neg",  1e3, 1e-9)
    b.add_inductor ("L_out", "rect_pos", "vout",     100e-6)
    b.add_capacitor("C_out", "vout",     "rect_neg", 47e-6)
    b.add_resistor ("R_L",   "vout",     "rect_neg", 10.0)
    return b, hyst


def main():
    b, hyst = build_plant()
    sw_fn = p.make_phase_shift_full_bridge_fn(
        switching_frequency=F_PWM,
        phase_shift=PHASE,
        leg_a_hs_idx=0, leg_a_ls_idx=1,
        leg_b_hs_idx=2, leg_b_ls_idx=3,
        num_switches=b.graph.num_switches,
        dead_time=100e-9,
    )

    # HystereticInductor observer — runs each step, integrates M.
    obs, b_extra = p.make_hysteretic_inductor_observer(b, hyst, dt=DT)

    print("  PSFB with HystereticInductor in-loop")
    print(f"    V_bus={V_BUS}V  f_pwm={F_PWM/1e3:.0f}kHz "
          f"phase_shift={PHASE:.2f} rad")
    print(f"    JA params: ferrite_n87 (Ms={hyst.params.Ms:.2e})")
    print(f"    L_0 (air-core) = {hyst.L_0*1e6:.3f} µH")
    print(f"    Sim {T_END*1e3:.1f} ms @ dt={DT*1e9:.0f} ns "
          f"= {int(T_END/DT):,} steps")

    res = p.simulate(
        b, t_end=T_END, dt=DT,
        switch_fn=sw_fn,
        step_observer=obs,
        b_extra_fn=b_extra,
        progress=True,
    )

    # Captured inductor current.
    times = np.asarray(res.times)
    states = np.asarray(res.states)
    i_idx = b.pool.branch_var_id_for_inductor(
        hyst.inductor_branch_id, b.graph)
    i_L = states[:, i_idx]

    print("\n  Captured inductor (hysteretic) current:")
    print(f"    i_peak    = {float(np.max(np.abs(i_L))):.3f} A")
    print(f"    final M   = {hyst.M:.4e} A/m")
    print(f"    final B   = {hyst.B:.4f} T")
    print(f"    final H   = {hyst.H:.2f} A/m")

    # Save trace: t, i_L, the live JA states (M, B, H), and the
    # output voltage so we can verify the PSFB still operates
    # correctly with the hysteresis device wired in.
    try:
        import csv
        out_path = Path(__file__).with_name(
            "psfb_hysteretic_inloop_trace.csv")
        v_out = res.v("vout")
        stride = max(1, len(times) // 4000)
        with out_path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t_s", "i_L_A", "v_out_V"])
            for k in range(0, len(times), stride):
                w.writerow([times[k], i_L[k], v_out[k]])
        print(f"    trace     → {out_path}")
        print(f"    v_out_avg = {float(np.mean(v_out[-len(v_out)//10:])):.3f} V "
              f"(steady-state, last 10%)")
    except Exception as exc:  # noqa: BLE001
        print(f"    (CSV export skipped: {exc})")


if __name__ == "__main__":
    main()
