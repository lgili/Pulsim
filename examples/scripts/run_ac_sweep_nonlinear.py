#!/usr/bin/env python3
"""AC sweep on a NONLINEAR plant — saturable boost at two operating points.

This is the "swept-sine works on anything" demo. Same circuit, two
duty cycles → two operating points → two different small-signal
plants because of the saturable inductor:

   V_in (12 V) ── L_sat (L_0=100µH, I_sat=1.5A, L_res=1µH) ──┬── D ── vout
                                                              │
                                                              Q1 ── gnd

  Operating point A: D = 0.30
    V_out_avg = V_in / (1−D) = 17.1 V
    I_L_avg   ≈ V_out / R_load = 17.1 / 30 = 0.57 A
    Since 0.57 A < I_sat = 1.5 A, the small-signal L ≈ L_0 = 100 µH
    → ω_LC = (1−D)/√(LC) = 0.7/√(1e-8) ≈ 7000 rad/s → f_LC ≈ 1.1 kHz

  Operating point B: D = 0.50
    V_out_avg = 24 V, I_L_avg = 1.6 A (above I_sat!)
    Saturable inductor SOFTENS toward L_residual
    → effective L drops, ω_LC SHIFTS UP

The swept-sine technique should automatically pick up the
small-signal response at each operating point. Plotting both
Bodes on the same axes shows the resonance shift — a real,
measurable consequence of the saturating core that an LTI
analysis would miss entirely.

This is the kind of measurement a power engineer makes with a
network analyzer on a real prototype to find where their
loop's plant changes versus load.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

import pulsim as p


V_IN     = 12.0
L_0      = 100e-6
I_SAT    = 1.5
L_RES    = 1e-6
C_OUT    = 100e-6
R_LOAD   = 30.0
F_PWM    = 50e3
T_PWM    = 1.0 / F_PWM

# Two operating points to compare.
D_OP_A   = 0.30   # below I_sat → L_eff ≈ L_0
D_OP_B   = 0.50   # above I_sat → L_eff softens

EPS_DUTY    = 0.04
F_MIN_BODE  = 200.0          # lower bound raised so sims are cheaper
F_MAX_BODE  = 8000.0
N_PTS       = 12             # 12 points (vs 18) — Newton+sat is expensive


def build_saturable_boost() -> p.CircuitBuilder:
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", V_IN)
    b.add_saturable_inductor(
        "L_boost", "vin", "sw",
        L_0=L_0, I_sat=I_SAT, L_residual=L_RES,
    )
    b.add_mosfet ("Q1",      "sw", "gnd", R_on=1e-3, R_off=1e9)
    b.add_diode  ("D_boost", "sw", "vout", 1e3, 1e-9, V_th=0.5)
    b.add_capacitor("C_out",  "vout", "gnd", C_OUT)
    b.add_resistor ("R_load", "vout", "gnd", R_LOAD)
    return b


def sweep_at_operating_point(D_op: float):
    builder = build_saturable_boost()
    vout_idx = builder.node_id_of("vout")
    n_sw = builder.graph.num_switches
    state_n = builder.pool.state_size(builder.graph)

    def excite(t, eps, freq):
        return [0.0] * state_n

    def make_switch_fn(eps, freq):
        omega = 2 * math.pi * freq
        def sw(t):
            duty = D_op + eps * math.sin(omega * t)
            phase = math.fmod(t, T_PWM) / T_PWM
            m = p.SwitchStateMask(n_sw)
            if phase < duty:
                m.set(0, True)
            return m
        return sw

    freqs = np.logspace(math.log10(F_MIN_BODE),
                          math.log10(F_MAX_BODE), N_PTS)
    DT_BODE = T_PWM * EPS_DUTY / 4.0

    def dt_fn(f):
        return min(DT_BODE, (1.0 / f) / 200.0)

    print(f"  Sweeping D={D_op}…")
    result = p.run_ac_sweep(
        builder,
        freqs=freqs,
        excite_fn=excite,
        switch_fn_factory=make_switch_fn,
        output_idx=vout_idx,
        dt_fn=dt_fn,
        perturbation_amplitude=EPS_DUTY,
        cycles_per_point=5,           # fewer cycles per freq
        settling_cycles=4,            # tighter settling
        start_from_dc_op=False,
        verbose=True,
    )
    return result


def main() -> None:
    print("Nonlinear AC sweep — saturable boost at two operating points")
    print("=" * 60)
    print(f"Plant: V_in={V_IN}V, L_0={L_0*1e6}µH, I_sat={I_SAT}A, "
          f"L_residual={L_RES*1e6:.1f}µH")
    print(f"       C={C_OUT*1e6}µF, R_load={R_LOAD}Ω, F_PWM={F_PWM/1e3}kHz")
    print()

    print(f"OP A — D = {D_OP_A}  (V_out ≈ {V_IN/(1-D_OP_A):.1f} V, "
          f"I_L_avg ≈ {V_IN/((1-D_OP_A)**2 * R_LOAD):.2f} A — "
          f"below I_sat → L_eff ≈ L_0)")
    res_A = sweep_at_operating_point(D_OP_A)
    print(f"OP B — D = {D_OP_B}  (V_out ≈ {V_IN/(1-D_OP_B):.1f} V, "
          f"I_L_avg ≈ {V_IN/((1-D_OP_B)**2 * R_LOAD):.2f} A — "
          f"saturated → L_eff softens)")
    res_B = sweep_at_operating_point(D_OP_B)

    # Find resonance peaks.
    f_peak_A = res_A.freqs[int(np.argmax(res_A.mag_dB))]
    f_peak_B = res_B.freqs[int(np.argmax(res_B.mag_dB))]
    print(f"\nResonance peaks:")
    print(f"  D={D_OP_A}: f_peak ≈ {f_peak_A:.0f} Hz  "
          f"(peak {res_A.mag_dB.max():.1f} dB)")
    print(f"  D={D_OP_B}: f_peak ≈ {f_peak_B:.0f} Hz  "
          f"(peak {res_B.mag_dB.max():.1f} dB)")
    print(f"  shift: {f_peak_B/f_peak_A:.2f}x (saturable L SHIFTS the LC pole)")

    # Analytical (using L_0 for both, for reference — shows
    # the small-signal model would predict the SAME resonance,
    # but reality moves with saturation).
    def analytical(D, L_used):
        s = 1j * 2 * np.pi * res_A.freqs
        omega_LC2 = (1 - D)**2 / (L_used * C_OUT)
        # G_vd ~ V_in / ((1-D)² · (LC s² + L/R s + (1-D)²))
        return V_IN / (L_used * C_OUT * s**2
                       + (L_used / R_LOAD) * s
                       + (1 - D)**2)

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, (ax_mag, ax_phase) = plt.subplots(2, 1, figsize=(11, 7),
                                              sharex=True)

    ax_mag.semilogx(res_A.freqs, res_A.mag_dB,
                      color="tab:blue", lw=1.4,
                      label=f"D={D_OP_A} (below I_sat — L=L_0)")
    ax_mag.semilogx(res_B.freqs, res_B.mag_dB,
                      color="tab:red", lw=1.4,
                      label=f"D={D_OP_B} (saturated — L softens)")
    # Overlay the analytical LTI prediction with L=L_0 for D=0.3
    H_anal_A = analytical(D_OP_A, L_0)
    ax_mag.semilogx(res_A.freqs, 20*np.log10(np.abs(H_anal_A)),
                      color="tab:blue", ls=":", lw=0.8, alpha=0.7,
                      label=f"analytical (L=L_0, D={D_OP_A})")
    ax_mag.axvline(f_peak_A, color="tab:blue", ls=":", lw=0.5, alpha=0.5)
    ax_mag.axvline(f_peak_B, color="tab:red",  ls=":", lw=0.5, alpha=0.5)
    ax_mag.set_ylabel("|H| [dB]"); ax_mag.grid(True, which="both", alpha=0.3)
    ax_mag.legend(loc="best", fontsize=9)
    ax_mag.set_title("Saturable boost control-to-output — same plant, "
                       "two operating points")

    ax_phase.semilogx(res_A.freqs, res_A.phase_deg,
                        color="tab:blue", lw=1.4)
    ax_phase.semilogx(res_B.freqs, res_B.phase_deg,
                        color="tab:red",  lw=1.4)
    ax_phase.set_xlabel("freq [Hz]"); ax_phase.set_ylabel("∠H [deg]")
    ax_phase.grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    out = Path(__file__).resolve().parent / "output" / "ac_sweep_nonlinear.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=120)
    print(f"\n  plot → {out}")


if __name__ == "__main__":
    main()
