#!/usr/bin/env python3
"""Auto-tune a buck PI from its measured plant Bode.

Three-step workflow that real control engineers do by hand:

  1. Sweep the open-loop plant — measure G_vd(jω).
  2. Pick a target crossover f_c and phase margin (e.g. 500 Hz, 60°).
  3. Use `tune_pi_from_bode(...)` to compute Kp + Ki analytically.
  4. Verify by simulating the closed loop with the tuned PI and a
     setpoint step. Plot V_out + L(jω) Bode with PM/GM markers.

This makes "PI tuning" no-mystery-allowed — every step is shown
and the formulas are visible. Compare with the hand-tuned
`run_buck_closed_loop.py` which took multiple iterations.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

import pulsim.v2 as p


# Plant
V_IN     = 24.0
V_REF    = 12.0
L_VAL    = 100e-6
C_VAL    = 47e-6
R_LOAD   = 5.0
F_PWM    = 100e3
T_PWM    = 1.0 / F_PWM

# Bode sweep settings (matches what run_ac_sweep_buck.py uses).
EPS_DUTY    = 0.05
N_BODE_PTS  = 18
F_MIN_BODE  = 100.0
F_MAX_BODE  = 8000.0
DT_BODE     = T_PWM * EPS_DUTY / 4.0     # 125 ns

# Auto-tuning targets.
# For PM=60° on a PI loop, ∠C ∈ (-90°, 0°) requires plant phase
# at crossover to lie in (-120°, -30°). For our buck, that's right
# around the LC resonance ≈ 2 kHz. Choose f_c = 2 kHz where plant
# phase ≈ -50° — that gives ∠C ≈ -70° (a slightly under-damped
# integrator, ω_z slightly below f_c).
F_CROSSOVER = 2000.0
PM_TARGET   = 60.0

# Closed-loop sim settings.
F_PWM_CL    = 100e3
DT_CL       = 5e-7
T_END_CL    = 3e-3
T_SETPOINT_STEP = 1.5e-3


def build_buck() -> p.CircuitBuilder:
    b = p.CircuitBuilder()
    b.add_voltage_source        ("Vin", "vin", "gnd", V_IN)
    b.add_mosfet_with_body_diode("Q1",  "vin", "sw",
                                   R_on=1e-3, R_off=1e9, V_F=0.7)
    b.add_diode                 ("D_FW","gnd", "sw", 1e3, 1e-9, V_th=0.7)
    b.add_inductor              ("L1",  "sw", "vout", L_VAL)
    b.add_capacitor             ("Cout","vout","gnd", C_VAL)
    b.add_resistor              ("R_L", "vout","gnd", R_LOAD)
    return b


def sweep_plant_bode():
    """Step 1: sweep the open-loop plant at constant D=0.5."""
    builder = build_buck()
    vout_idx = builder.node_id_of("vout")
    n_sw = builder.graph.num_switches
    state_n = builder.pool.state_size(builder.graph)

    D_OP = V_REF / V_IN   # 0.5 for 12V out from 24V in

    def excite(t, eps, freq):
        return [0.0] * state_n

    def make_switch_fn(eps, freq):
        omega = 2 * math.pi * freq
        def sw(t):
            duty = D_OP + eps * math.sin(omega * t)
            phase = math.fmod(t, T_PWM) / T_PWM
            m = p.SwitchStateMask(n_sw)
            if phase < duty:
                m.set(0, True)
            return m
        return sw

    freqs = np.logspace(math.log10(F_MIN_BODE),
                          math.log10(F_MAX_BODE), N_BODE_PTS)
    print(f"  Sweeping buck open-loop plant (D={D_OP}) at "
          f"{len(freqs)} freqs from {F_MIN_BODE:.0f} Hz to "
          f"{F_MAX_BODE/1e3:.1f} kHz...")

    def dt_fn(f):
        return min(DT_BODE, (1.0 / f) / 200.0)

    result = p.run_ac_sweep(
        builder,
        freqs=freqs,
        excite_fn=excite,
        switch_fn_factory=make_switch_fn,
        output_idx=vout_idx,
        dt_fn=dt_fn,
        perturbation_amplitude=EPS_DUTY,
        cycles_per_point=10,
        settling_cycles=6,
        start_from_dc_op=False,
        verbose=False,
    )
    return result, builder, vout_idx


def closed_loop_sim(builder, vout_idx, Kp, Ki):
    """Step 4: simulate the closed loop with the auto-tuned PI."""
    n_sw = builder.graph.num_switches
    duty = [0.5]
    last_pi_t = [-1.0]

    # Use the same LPF + sample-at-PWM-rate pattern as our manual
    # tunes — keeps the apples-to-apples comparison fair.
    pi = p.PIController(Kp=Kp, Ki=Ki, output_min=0.05, output_max=0.95)
    lpf = p.FirstOrderLowPass(tau=320e-6)

    setpoint_fn = lambda t: V_REF if t < T_SETPOINT_STEP else V_REF + 2.0

    def observe(t, x):
        v_out_filt = lpf.update(input_value=float(x[vout_idx]),
                                 dt=DT_CL)
        if t - last_pi_t[0] >= 1.0 / F_PWM_CL:
            dt_pi = ((t - last_pi_t[0]) if last_pi_t[0] >= 0
                       else 1.0 / F_PWM_CL)
            duty[0] = pi.update(setpoint=setpoint_fn(t),
                                  measured=v_out_filt, dt=dt_pi)
            last_pi_t[0] = t

    def switch_fn(t):
        phase = math.fmod(t, T_PWM) / T_PWM
        m = p.SwitchStateMask(n_sw)
        if phase < duty[0]:
            m.set(0, True)
        return m

    return p.simulate(builder, t_end=T_END_CL, dt=DT_CL,
                        switch_fn=switch_fn, step_observer=observe,
                        max_event_iterations=8), setpoint_fn


def main() -> None:
    # ---- Step 1: Bode sweep ------------------------------------------------
    print("=" * 60)
    print("STEP 1 — sweep open-loop plant Bode")
    print("=" * 60)
    plant_result, builder, vout_idx = sweep_plant_bode()

    # ---- Step 2: Auto-tune PI ---------------------------------------------
    print(f"\n{'=' * 60}")
    print(f"STEP 2 — auto-tune PI for f_c={F_CROSSOVER:.0f} Hz, "
          f"PM={PM_TARGET}°")
    print(f"{'=' * 60}")
    tune = p.tune_pi_from_bode(
        plant_result.freqs, plant_result.H,
        f_crossover=F_CROSSOVER,
        phase_margin_deg=PM_TARGET,
        output_min=0.05, output_max=0.95,
    )
    print(f"  plant @ {F_CROSSOVER:.0f} Hz: "
          f"|G| = {tune['plant_mag_at_crossover']:.3f}  "
          f"(={20*math.log10(tune['plant_mag_at_crossover']):+.2f} dB)")
    print(f"                    ∠G = "
          f"{tune['plant_phase_at_crossover_deg']:+.2f}°")
    print(f"\n  Auto-tuned PI:")
    print(f"    Kp = {tune['Kp']:.5f}")
    print(f"    Ki = {tune['Ki']:.2f}")
    print(f"    achieved PM = {tune['achieved_pm_deg']:.2f}°")
    if tune["warnings"]:
        for w in tune["warnings"]:
            print(f"  ⚠ {w}")

    # ---- Step 3: compute loop gain L = C·G + verify GM/PM ------------------
    print(f"\n{'=' * 60}")
    print("STEP 3 — loop gain L(jω) = C(jω) · G(jω), verify GM/PM")
    print(f"{'=' * 60}")
    L = p.loop_gain(plant_result.freqs, plant_result.H,
                     tune["Kp"], tune["Ki"])
    pm = p.phase_margin_from_loop(plant_result.freqs, L)
    gm = p.gain_margin_from_loop(plant_result.freqs, L)
    print(f"  PM = {pm:.2f}°  (target {PM_TARGET}°)")
    print(f"  GM = {gm:.2f} dB"
          if math.isfinite(gm)
          else f"  GM = ∞  (loop phase never reaches −180° "
               f"within sweep range)")

    # ---- Step 4: simulate closed-loop step response -----------------------
    print(f"\n{'=' * 60}")
    print("STEP 4 — closed-loop step response with auto-tuned PI")
    print(f"{'=' * 60}")
    res, setpoint_fn = closed_loop_sim(builder, vout_idx,
                                          tune["Kp"], tune["Ki"])
    times = np.asarray(res.times) * 1e3
    v_out = np.array([s[vout_idx] for s in res.states])
    k_pre = (times > 1.0) & (times < T_SETPOINT_STEP * 1e3)
    k_post = times > 2.5
    print(f"  V_out pre-step  (1-{T_SETPOINT_STEP*1e3:.1f} ms): "
          f"mean={v_out[k_pre].mean():.3f} V (target {V_REF:.1f})")
    print(f"  V_out post-step (>2.5 ms): "
          f"mean={v_out[k_post].mean():.3f} V (target {V_REF + 2.0:.1f})")

    # ---- Plot ------------------------------------------------------------
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig = plt.figure(figsize=(13, 9))
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.25)
    ax_pmag = fig.add_subplot(gs[0, 0])
    ax_lmag = fig.add_subplot(gs[0, 1])
    ax_pph  = fig.add_subplot(gs[1, 0], sharex=ax_pmag)
    ax_lph  = fig.add_subplot(gs[1, 1], sharex=ax_lmag)
    ax_t    = fig.add_subplot(gs[2, 0])
    ax_t2   = fig.add_subplot(gs[2, 1])

    # LEFT column: plant Bode
    ax_pmag.semilogx(plant_result.freqs, plant_result.mag_dB, lw=1.4,
                       color="tab:blue")
    ax_pmag.axvline(F_CROSSOVER, color="r", ls=":", lw=0.8,
                      label=f"f_c target ({F_CROSSOVER:.0f} Hz)")
    ax_pmag.set_ylabel("plant |G| [dB]"); ax_pmag.grid(True, which="both", alpha=0.3)
    ax_pmag.legend(loc="best"); ax_pmag.set_title("Open-loop plant G(jω)")

    ax_pph.semilogx(plant_result.freqs, plant_result.phase_deg, lw=1.4,
                       color="tab:blue")
    ax_pph.set_xlabel("freq [Hz]"); ax_pph.set_ylabel("∠G [deg]")
    ax_pph.grid(True, which="both", alpha=0.3)

    # RIGHT column: loop gain L(jω)
    mag_L = 20 * np.log10(np.abs(L))
    ph_L = np.degrees(np.angle(L))
    ph_L_unwrap = np.degrees(np.unwrap(np.angle(L)))
    ax_lmag.semilogx(plant_result.freqs, mag_L, lw=1.4,
                       color="tab:orange")
    ax_lmag.axhline(0, color="k", lw=0.5)
    ax_lmag.set_ylabel("loop |L| [dB]")
    ax_lmag.grid(True, which="both", alpha=0.3)
    ax_lmag.set_title(f"Loop gain L = C·G — PM={pm:.1f}°, "
                       f"GM={'∞' if not math.isfinite(gm) else f'{gm:.1f} dB'}")
    ax_lph.semilogx(plant_result.freqs, ph_L_unwrap, lw=1.4,
                      color="tab:orange")
    ax_lph.axhline(-180, color="k", ls="--", lw=0.5)
    ax_lph.set_xlabel("freq [Hz]"); ax_lph.set_ylabel("∠L [deg]")
    ax_lph.grid(True, which="both", alpha=0.3)

    # Bottom row: closed-loop time response (spans both columns visually)
    ax_t.plot(times, v_out, color="tab:blue", lw=0.6, label="V_out")
    ax_t.plot(times, [setpoint_fn(t) for t in res.times],
                color="r", ls="--", lw=0.8, label="setpoint")
    ax_t.set_xlabel("time [ms]"); ax_t.set_ylabel("V_out [V]")
    ax_t.legend(loc="best"); ax_t.grid(alpha=0.3)
    ax_t.set_title(f"Step response — Kp={tune['Kp']:.4f}, "
                     f"Ki={tune['Ki']:.1f}")
    ax_t2.axis("off")
    info = (
        f"AUTO-TUNED PI from plant Bode\n\n"
        f"Target:\n"
        f"  f_crossover = {F_CROSSOVER:.0f} Hz\n"
        f"  PM_target   = {PM_TARGET}°\n\n"
        f"Plant @ f_c:\n"
        f"  |G| = {tune['plant_mag_at_crossover']:.3f} "
        f"({20*math.log10(tune['plant_mag_at_crossover']):+.2f} dB)\n"
        f"  ∠G  = {tune['plant_phase_at_crossover_deg']:+.2f}°\n\n"
        f"Computed gains:\n"
        f"  Kp = {tune['Kp']:.5f}\n"
        f"  Ki = {tune['Ki']:.2f}\n\n"
        f"Verified loop gain:\n"
        f"  PM achieved = {pm:.2f}°\n"
        f"  GM achieved = "
        f"{'∞' if not math.isfinite(gm) else f'{gm:.1f} dB'}"
    )
    ax_t2.text(0.05, 0.95, info, transform=ax_t2.transAxes,
                  fontfamily="monospace", fontsize=10,
                  verticalalignment="top",
                  bbox=dict(boxstyle="round,pad=0.6",
                             facecolor="lightyellow"))

    plt.tight_layout()
    out = Path(__file__).resolve().parent / "output" / "auto_tune_buck.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=120)
    print(f"\n  plot → {out}")


if __name__ == "__main__":
    main()
