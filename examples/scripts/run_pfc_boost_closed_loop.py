#!/usr/bin/env python3
"""Single-phase PFC boost — cascaded current + voltage loops.

   V_ac (155V peak / 60 Hz) ── 4-diode bridge ── |v_in| ── L_pfc ──┬── D_boost ── V_bus ──┬── R_load
                                                                    │                      │
                                                                    Q1 (MOSFET) ── gnd     C_bus

PFC objective:
  * Force input current `i_L` to be a sinusoid IN PHASE with the
    AC mains (power factor = 1).
  * Hold DC bus voltage at a constant 200 V despite the 120-Hz
    ripple inherent to single-phase rectification.

Two cascaded PI loops:
  * **Outer voltage loop** — slow (~10 Hz BW). Reads V_bus, compares
    to V_BUS_REF. Output = K (the current-amplitude command).
  * **Inner current loop** — fast (~5 kHz BW). Reads i_L. Reference =
    K · |sin(ω_grid · t)|. Output = duty cycle for Q1.

Stress for the kernel:
  - 4-diode bridge rectifier with HARD commutation twice per AC cycle.
  - Active boost switch with software-defined PWM.
  - Two PI controllers running at different rates.
  - Reference current is TIME-VARYING (chases the mains envelope).
  - 120 Hz ripple on V_bus → outer loop has to integrate that out.
"""

from __future__ import annotations

import math
from math import fabs
from pathlib import Path

import numpy as np

import pulsim as p


V_AC_PEAK   = 155.0           # 110 Vrms × √2 ≈ 155 V
F_AC        = 60.0
V_BUS_REF   = 200.0           # boost target
F_PWM       = 100e3
T_PWM       = 1.0 / F_PWM
DT          = 5.0e-7          # 0.5 µs → 200 samples/PWM
T_END       = 200.0e-3        # 12 AC cycles → outer loop has time to settle
T_LOAD_STEP = 50.0e-3         # unused

# Inner current loop — tracks |sin| at 120 Hz. Bandwidth must
# be much higher than that → BW ~ 5 kHz, well below F_PWM/10 = 10 kHz.
KP_I = 0.01
KI_I = 500.0
DUTY_MIN = 0.05
DUTY_MAX = 0.95

# Outer voltage loop — slow enough to NOT respond to 120 Hz bus ripple
# (BW ~ 10 Hz, two decades below the ripple frequency). The 120 Hz
# ripple is inherent — only a giant C_bus can shrink it.
KP_V = 0.05
KI_V = 2.0
K_MIN  = 0.1                  # min current-amplitude command [A]
K_MAX  = 8.0                  # raised to allow more current at heavy load
TAU_LPF_V = 5.0e-3            # 30 Hz LPF on V_bus measurement


def build_plant() -> p.CircuitBuilder:
    b = p.CircuitBuilder()
    # AC mains: a single sine source ac_l → ac_n (line vs neutral).
    b.add_sine_voltage_source(
        "V_ac", "ac_l", "ac_n",
        v_dc=0.0, v_amplitude=V_AC_PEAK,
        frequency=F_AC, phase=0.0,
    )
    # Reference the mains loosely. A 1 mΩ tie here plus the 1 mΩ
    # rect_n tie below shorted the source through D3 on every negative
    # half-cycle (51.7 kA peak) — AI_GUIDE trap 7.
    b.add_resistor("R_n", "ac_n", "gnd", 1.0e6)

    # 4-diode full-bridge rectifier: ac_l, ac_n → rect_p, rect_n
    b.add_diode("D1", "ac_l",   "rect_p", 1e3, 1e-9, V_th=0.5)  # idx 1
    b.add_diode("D2", "ac_n",   "rect_p", 1e3, 1e-9, V_th=0.5)  # idx 2
    b.add_diode("D3", "rect_n", "ac_l",   1e3, 1e-9, V_th=0.5)  # idx 3
    b.add_diode("D4", "rect_n", "ac_n",   1e3, 1e-9, V_th=0.5)  # idx 4

    # Pin rect_n to ground via small R (otherwise it floats).
    b.add_resistor("R_rect_n", "rect_n", "gnd", 1.0e-3)

    # Boost inductor (from rectified bus to switch node).
    b.add_inductor("L_pfc", "rect_p", "sw", 1.0e-3)
    # Active switch.
    b.add_mosfet  ("Q1", "sw", "rect_n", R_on=1e-3, R_off=1e9)
    # Boost diode.
    b.add_diode   ("D_boost", "sw", "vbus", 1e3, 1e-9, V_th=0.5)
    # Output cap + load.
    b.add_capacitor("C_bus", "vbus", "rect_n", 470e-6)
    b.add_resistor ("R_load", "vbus", "rect_n", 100.0)
    return b


def main() -> None:
    builder = build_plant()
    vbus_idx = builder.node_id_of("vbus")
    ac_l_idx = builder.node_id_of("ac_l")
    ac_n_idx = builder.node_id_of("ac_n")
    # L_pfc is branch index 10 (1 Vsrc + 1 Rn + 4 diodes + 1 Rrect_n + 1 L_pfc).
    # Actually: 0=Vac, 1=Rn, 2=D1, 3=D2, 4=D3, 5=D4, 6=Rrn, 7=L_pfc.
    iL_idx = builder.pool.branch_var_id_for_inductor(7, builder.graph)
    print(f"  num_branches:   {builder.num_branches}")
    print(f"  num_switches:   {builder.graph.num_switches}")
    print(f"  iL_pfc state idx = {iL_idx}")

    # --- Controllers --------------------------------------------------------
    pi_v = p.PIController(Kp=KP_V, Ki=KI_V,
                            output_min=K_MIN, output_max=K_MAX,
                            integrator_state=2.0)   # warm-start at typ. load
    pi_i = p.PIController(Kp=KP_I, Ki=KI_I,
                            output_min=DUTY_MIN, output_max=DUTY_MAX)
    lpf_v = p.FirstOrderLowPass(tau=TAU_LPF_V)
    V_LOOP_DT = 2.0e-3            # outer loop runs at 500 Hz

    # Outer voltage loop — throttled to 500 Hz so it doesn't fight the
    # 120 Hz bus ripple. `bind_pi_to_duty_callable` returns a getter for
    # the latest output (here = current-amplitude command, NOT duty).
    K_amp_get, v_loop_observer, _v_hist = p.bind_pi_to_duty_callable(
        builder,
        pi=pi_v,
        measured=lambda x: lpf_v.state,
        setpoint=V_BUS_REF,
        freq=1.0 / V_LOOP_DT,
    )

    # Inner current loop fires every dt (not throttled — must track the
    # |sin| envelope), so we keep it as a tight inline update.
    duty = [0.5]

    def observe(t: float, x) -> None:
        # LPF on bus voltage every dt → outer loop reads its state.
        lpf_v.update(input_value=float(x[vbus_idx]), dt=DT)
        v_loop_observer(t, x)

        # Inner current loop, every dt: i_ref tracks |v_ac| × K_amp.
        # |v_ac| line-to-neutral: v(ac_l) alone is ~0 on the negative
        # half-cycle (D3 clamps ac_l to rect_n = gnd).
        i_ref = K_amp_get() * fabs(float(x[ac_l_idx] - x[ac_n_idx])) / V_AC_PEAK
        duty[0] = pi_i.update(setpoint=i_ref,
                                 measured=float(x[iL_idx]), dt=DT)

    num_switches = builder.graph.num_switches
    q1_idx = builder.switch_index_of("Q1")

    def switch_fn(t: float):
        phase = math.fmod(t, T_PWM) / T_PWM
        m = p.SwitchStateMask(num_switches)
        if phase < duty[0]:
            m.set(q1_idx, True)
        return m

    print("\n  PFC boost CL:")
    print(f"    inner: PI(Kp={KP_I}, Ki={KI_I}), duty ∈ [{DUTY_MIN}, {DUTY_MAX}]")
    print(f"    outer: PI(Kp={KP_V}, Ki={KI_V}), K_amp ∈ [{K_MIN}, {K_MAX}]")
    print(f"    V_bus target: {V_BUS_REF:.0f} V, mains: {V_AC_PEAK:.0f} V peak @ {F_AC:.0f} Hz")

    res = p.simulate(
        builder, t_end=T_END, dt=DT,
        switch_fn=switch_fn,
        step_observer=observe,
        max_event_iterations=12,
    )
    print(f"  samples: {res.num_steps()}")

    times = np.asarray(res.times) * 1e3   # ms
    v_bus_arr = res.v("vbus")
    v_ac_arr  = res.v("ac_l") - res.v("ac_n")
    i_L_arr   = np.array([s[iL_idx]   for s in res.states])

    # KPI
    # Last 3 AC cycles (the outer loop settles by ~150 ms) → settled
    # V_bus and i_L tracking. Time-weighted (p.steady_state_stats):
    # correct on any grid, unlike a plain sample mean.
    ss_bus = p.steady_state_stats(res, "vbus", last=3.0 / F_AC)
    v_bus_mean = ss_bus.mean
    v_bus_ripple = ss_bus.ripple_pp
    print("\n  KPI:")
    print(f"    V_bus mean (late):  {v_bus_mean:.2f} V (target {V_BUS_REF:.0f})")
    print(f"    V_bus ripple (pp):  {v_bus_ripple:.2f} V")
    print(f"    i_L peak:           {i_L_arr.max():.3f} A (cold-start inrush)")
    # True power factor P / (V_rms · I_rms) over the same 3 cycles.
    t0 = T_END - 3.0 / F_AC
    i_ac = -res.i("V_ac")
    p_ac = p.time_average(res.times, v_ac_arr * i_ac, t_from=t0)
    v_rms = np.sqrt(p.time_average(res.times, v_ac_arr ** 2, t_from=t0))
    i_rms = np.sqrt(p.time_average(res.times, i_ac ** 2, t_from=t0))
    print(f"    PF = P/(V_rms·I_rms): {p_ac / (v_rms * i_rms):.3f}  "
          f"(P_ac = {p_ac:.0f} W)")
    # Power-factor proxy: correlation between i_L and |v_ac|, sample-
    # weighted (this script runs on a fixed dt, where that is fine).
    k_late = int(0.6 * res.num_steps())
    abs_vac = np.abs(v_ac_arr[k_late:])
    iL_late = i_L_arr[k_late:]
    if iL_late.std() > 0 and abs_vac.std() > 0:
        pf_proxy = np.corrcoef(iL_late, abs_vac)[0, 1]
        print(f"    PF proxy (corr i_L, |v_ac|): {pf_proxy:.3f} (target → 1.0)")

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, (ax_vac, ax_iL, ax_vbus) = plt.subplots(
        3, 1, figsize=(11, 8), sharex=True)
    ax_vac.plot(times, v_ac_arr, color="tab:blue", lw=0.5)
    ax_vac.set_ylabel("V_ac (mains) [V]"); ax_vac.grid(alpha=0.3)
    ax_vac.set_title("PFC boost — cascaded voltage + current loops")

    ax_iL.plot(times, i_L_arr, color="tab:green", lw=0.4, label="i_L")
    ax_iL.plot(times, K_MAX * np.abs(v_ac_arr) / V_AC_PEAK,
                color="r", ls="--", lw=0.5, alpha=0.5,
                label="ref envelope (K_MAX·|sin|)")
    ax_iL.set_ylabel("i_L [A]"); ax_iL.grid(alpha=0.3)
    ax_iL.legend(loc="upper right")

    ax_vbus.plot(times, v_bus_arr, color="tab:red", lw=0.5)
    ax_vbus.axhline(V_BUS_REF, color="k", ls="--", lw=0.6,
                     label=f"target ({V_BUS_REF:.0f} V)")
    ax_vbus.set_xlabel("time [ms]"); ax_vbus.set_ylabel("V_bus [V]")
    ax_vbus.legend(loc="lower right"); ax_vbus.grid(alpha=0.3)

    plt.tight_layout()
    out = Path(__file__).resolve().parent / "output" / "pfc_boost_closed_loop.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=120)
    print(f"\n  plot → {out}")


if __name__ == "__main__":
    main()
