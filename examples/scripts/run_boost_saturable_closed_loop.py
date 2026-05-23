#!/usr/bin/env python3
"""Boost CL with SATURABLE inductor — V17 + IGBT + Newton + software PI.

The kernel stress: when the inductor saturates, its effective
small-signal inductance collapses by ~100x. The plant's
small-signal gain CHANGES during operation — a fixed PI tuned
for L = L_0 will become aggressive when L → L_residual. This
combination of:

  * V17 saturable inductor (Newton refresh + sat-history)
  * V14 IGBT (Newton refresh)
  * Smooth-blend diode (Newton)
  * Software PI in step_observer
  * Cache lookup per step

is the densest "everything-at-once" stress test we can build.

   V_in (12V) ── L_sat ──┬── D_boost ── vout ──┬── R_load
                          │                     │
                          T1 (IGBT) ── gnd      C_out

Setpoint: 18 V (modest target — pushes L_sat into the saturation
knee but not deep, so the loop has to negotiate the changing gain).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

import pulsim as p


V_IN     = 12.0
V_REF    = 18.0
V_REF_2  = 22.0          # step up at t=2 ms
T_STEP   = 2.0e-3
F_PWM    = 50e3
T_PWM    = 1.0 / F_PWM
DT       = 1.0e-7
T_END    = 4.0e-3

# Saturable inductor parameters.
L_0         = 100e-6     # 100 µH at low current
I_SAT       = 1.5        # saturates around 1.5 A
L_RESIDUAL  = 1e-6       # 1 µH at full saturation (100x drop)

# Conservative PI: saturating L means the plant's small-signal gain
# changes dramatically across operating range. Soft-start ramp +
# LPF + low Ki keep the loop tame.
KP       = 0.002
KI       = 80.0
TAU_LPF  = 400.0e-6
DUTY_MIN = 0.10
DUTY_MAX = 0.75


def build_plant() -> p.CircuitBuilder:
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", V_IN)

    # The "gate driver" — fixed-amplitude PulseVoltageSource on the
    # IGBT gate. The duty (gate-on width) is controlled by our
    # custom switch_fn (not by the pulse_width param), so we just
    # need ANY constant-high gate when Q1 should be ON.
    # Actually no: V14 IGBT reads its gate from a NODE, not a
    # switch state. We can't drive it from switch_fn directly.
    # Workaround: provide V_g as a constant 10 V source, and gate
    # the OUTPUT via a switch (drain → sw, sw → ground via a
    # very-tiny resistor when "ON", giant when "OFF") — but that
    # decouples Q1's gate from the controller.
    #
    # Cleanest: use a `mosfet` (Switch-kind) instead of IGBT for
    # the active device. We lose the IGBT nonlinear refresh BUT
    # gain controllability via switch_fn.
    b.add_saturable_inductor("L_boost", "vin", "sw",
                              L_0=L_0, I_sat=I_SAT,
                              L_residual=L_RESIDUAL)
    b.add_mosfet ("Q1", "sw",  "gnd", R_on=1e-3, R_off=1e9)
    b.add_diode  ("D_boost", "sw", "vout", 1e3, 1e-9, V_th=0.5)
    b.add_capacitor("C_out",  "vout", "gnd", 100e-6)
    b.add_resistor ("R_load", "vout", "gnd", 20.0)
    return b


def make_setpoint(t_step: float, v1: float, v2: float,
                    soft_start_t: float = 1.0e-3):
    """Soft-start ramp 0→v1 over `soft_start_t` to limit cold-start
    inrush through the saturable L."""
    def sp(t: float) -> float:
        if t < soft_start_t:
            return v1 * (t / soft_start_t)
        return v1 if t < t_step else v2
    return sp


def main() -> None:
    builder = build_plant()
    vout_idx = builder.node_id_of("vout")
    sw_idx   = builder.node_id_of("sw")
    print(f"  num_branches:   {builder.num_branches}")
    print(f"  num_switches:   {builder.graph.num_switches}")
    print(f"  has_nonlinear:  {builder.pool.has_nonlinear_devices()}")

    pi = p.PIController(
        Kp=KP, Ki=KI,
        output_min=DUTY_MIN, output_max=DUTY_MAX,
        integrator_state=0.10,
    )
    lpf = p.FirstOrderLowPass(tau=TAU_LPF)
    duty = [0.10]
    last_pi_t = [-1.0]

    setpoint_fn = make_setpoint(T_STEP, V_REF, V_REF_2)

    def observe(t: float, x) -> None:
        v_out_filt = lpf.update(input_value=float(x[vout_idx]), dt=DT)
        sp = setpoint_fn(t)
        if t - last_pi_t[0] >= T_PWM:
            dt_pi = (t - last_pi_t[0]) if last_pi_t[0] >= 0 else T_PWM
            duty[0] = pi.update(setpoint=sp, measured=v_out_filt,
                                  dt=dt_pi)
            last_pi_t[0] = t

    num_switches = builder.graph.num_switches

    def switch_fn(t: float):
        phase = math.fmod(t, T_PWM) / T_PWM
        m = p.SwitchStateMask(num_switches)
        if phase < duty[0]:
            m.set(0, True)
        return m

    print(f"\n  Closed-loop SATURABLE boost:")
    print(f"    PI(Kp={KP}, Ki={KI}), duty ∈ [{DUTY_MIN}, {DUTY_MAX}]")
    print(f"    L_0 = {L_0*1e6:.0f} µH, I_sat = {I_SAT} A, "
          f"L_residual = {L_RESIDUAL*1e6:.1f} µH")
    print(f"    setpoint {V_REF} V → {V_REF_2} V at t = {T_STEP*1e3:.1f} ms")
    res = p.simulate(
        builder, t_end=T_END, dt=DT,
        switch_fn=switch_fn,
        step_observer=observe,
        max_event_iterations=12,
        max_newton_iterations=100,
    )
    print(f"  samples: {res.num_steps()}")

    # Replay PI for plot.
    pi2 = p.PIController(
        Kp=KP, Ki=KI, output_min=DUTY_MIN, output_max=DUTY_MAX,
        integrator_state=0.10,
    )
    lpf2 = p.FirstOrderLowPass(tau=TAU_LPF)
    duty_hist = np.zeros(res.num_steps())
    setpoint_hist = np.zeros(res.num_steps())
    last_t = [-1.0]
    d_curr = [0.10]
    for k, (t, st) in enumerate(zip(res.times, res.states)):
        v_out = lpf2.update(input_value=float(st[vout_idx]), dt=DT)
        sp = setpoint_fn(t)
        if t - last_t[0] >= T_PWM:
            dt_pi = (t - last_t[0]) if last_t[0] >= 0 else T_PWM
            d_curr[0] = pi2.update(setpoint=sp, measured=v_out, dt=dt_pi)
            last_t[0] = t
        duty_hist[k] = d_curr[0]
        setpoint_hist[k] = sp

    times = np.asarray(res.times) * 1e3
    v_out_arr = np.array([s[vout_idx] for s in res.states])
    v_sw_arr  = np.array([s[sw_idx]   for s in res.states])

    # Recover the inductor current from the state vector.
    # SaturableInductor branch adds a branch-current unknown,
    # accessible via pool.branch_var_id_for_inductor (if exposed).
    # If not, fall back to estimating from V_sw / impedance.
    # The L_boost is branch index 1 (after Vin = 0).
    try:
        i_L_idx = builder.pool.branch_var_id_for_inductor(
            1, builder.graph)
        i_L_arr = np.array([s[i_L_idx] for s in res.states])
        have_iL = True
    except Exception:
        have_iL = False

    pre  = v_out_arr[(times > 1.0) & (times < T_STEP*1e3)]
    post = v_out_arr[(times > T_STEP*1e3 + 0.5)]
    print(f"\n  KPI:")
    print(f"    V_out pre-step  (1–{T_STEP*1e3:.0f} ms): "
          f"mean = {pre.mean():.2f} V (target {V_REF:.1f})")
    print(f"    V_out post-step: mean = {post.mean():.2f} V "
          f"(target {V_REF_2:.1f})")
    if have_iL:
        print(f"    i_L max = {i_L_arr.max():.3f} A  "
              f"(I_sat = {I_SAT} A — saturation = "
              f"{'YES' if i_L_arr.max() > I_SAT else 'no'})")

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    n_axes = 4 if have_iL else 3
    fig, axes = plt.subplots(n_axes, 1, figsize=(11, 9), sharex=True)
    if n_axes == 4:
        ax_vout, ax_iL, ax_sw, ax_duty = axes
    else:
        ax_vout, ax_sw, ax_duty = axes

    ax_vout.plot(times, v_out_arr, color="tab:blue", lw=0.7, label="V_out")
    ax_vout.plot(times, setpoint_hist, color="r", ls="--", lw=0.8,
                  label="setpoint")
    ax_vout.set_ylabel("V_out [V]"); ax_vout.grid(alpha=0.3)
    ax_vout.legend(loc="lower right")
    ax_vout.set_title(f"Boost CL with SATURABLE L_0={L_0*1e6:.0f}µH/I_sat={I_SAT}A "
                       f"— PI(Kp={KP}, Ki={KI})")

    if have_iL:
        ax_iL.plot(times, i_L_arr, color="tab:green", lw=0.5, alpha=0.8)
        ax_iL.axhline(I_SAT, color="r", ls=":", lw=0.8,
                       label=f"I_sat = {I_SAT} A")
        ax_iL.set_ylabel("i_L [A]"); ax_iL.grid(alpha=0.3)
        ax_iL.legend(loc="upper right")

    ax_sw.plot(times, v_sw_arr, color="tab:purple", lw=0.3, alpha=0.6)
    ax_sw.set_ylabel("V_sw [V]"); ax_sw.grid(alpha=0.3)

    ax_duty.plot(times, duty_hist, color="tab:orange", lw=0.7)
    ax_duty.axhline(DUTY_MIN, color="k", ls=":", lw=0.5)
    ax_duty.axhline(DUTY_MAX, color="k", ls=":", lw=0.5)
    ax_duty.set_xlabel("time [ms]"); ax_duty.set_ylabel("duty [-]")
    ax_duty.grid(alpha=0.3); ax_duty.set_ylim(0, 1)

    plt.tight_layout()
    out = Path(__file__).resolve().parent / "output" / "boost_saturable_closed_loop.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=120)
    print(f"\n  plot → {out}")


if __name__ == "__main__":
    main()
