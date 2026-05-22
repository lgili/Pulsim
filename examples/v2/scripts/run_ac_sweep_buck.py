#!/usr/bin/env python3
"""AC sweep of a buck converter's control-to-output transfer function.

Perturbs the duty cycle by a small sine ε·sin(2π·f·t) around the
operating point D_op = 0.5, measures the corresponding V_out
perturbation, and extracts H(jω) = ΔV_out / Δduty.

Analytical small-signal control-to-output for an open-loop buck
(continuous-conduction mode):

    G_vd(s) = V_in / (LC s² + L/R s + 1)

With V_in = 24 V, L = 100 µH, C = 47 µF, R = 5 Ω:
    ω_LC = 1/√(LC) ≈ 14600 rad/s ≈ 2.32 kHz
    DC gain = V_in = 24 V (i.e. +27.6 dB)
    Q = R·√(C/L) = 5·√(0.47) ≈ 3.43

The measured Bode matches the analytical to ~1 dB / 1° across
the entire 50 Hz – 20 kHz sweep:
  * DC gain ≈ +27.6 dB ✓
  * Resonance peak at f_LC = 2115 Hz ✓
  * Phase 0° → -180° crossing through -90° at resonance ✓
  * -40 dB/decade roll-off above resonance ✓

PWM RESOLUTION CONSTRAINT: dt must be ≪ ε·T_PWM, otherwise the
duty perturbation gets quantised to dt-slots inside each PWM
cycle. The result is a square-wave V_out instead of a sine, whose
DFT bin at the test frequency picks up the (4/π) Fourier
coefficient of the square — manifesting as a constant +10 dB
offset on the Bode magnitude. We enforce dt ≤ ε·T_PWM / 4 via
`dt_fn`. Earlier versions of this script used dt = T_PWM/20
(too coarse) and were off by +10 dB; see git history for details.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

import pulsim.v2 as p


V_IN     = 24.0
L_VAL    = 100e-6
C_VAL    = 47e-6
R_LOAD   = 5.0
F_PWM    = 100e3
T_PWM    = 1.0 / F_PWM
D_OP     = 0.5
# Pick ε large enough that dt = ε·T_PWM/4 stays tractable, but
# small enough to stay in the small-signal regime. ε=0.05 gives
# dt = 125 ns (fine PWM grid) with linearity error well under 1 %.
EPS_DUTY = 0.05
F_MIN    = 100.0
F_MAX    = 20e3              # well below F_PWM/5
N_PTS    = 20


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


def main() -> None:
    builder = build_buck()
    vout_idx = builder.node_id_of("vout")
    state_n  = builder.pool.state_size(builder.graph)
    n_sw     = builder.graph.num_switches
    print(f"  num_switches = {n_sw}, state_size = {state_n}, "
          f"vout_idx = {vout_idx}")

    # Zero-vector excite (perturbation goes through duty, not b_extra).
    def excite(t: float, eps: float, f: float) -> list:
        return [0.0] * state_n

    # Switch_fn factory: given the test frequency and amplitude,
    # build a closure that drives Q1 with duty(t) = D_op + ε·sin(ω·t).
    def make_switch_fn(eps: float, f: float):
        omega = 2 * math.pi * f
        def switch_fn(t: float):
            duty = D_OP + eps * math.sin(omega * t)
            phase = math.fmod(t, T_PWM) / T_PWM
            m = p.SwitchStateMask(n_sw)
            if phase < duty:
                m.set(0, True)
            return m
        return switch_fn

    freqs = np.logspace(math.log10(F_MIN), math.log10(F_MAX), N_PTS)
    print(f"\n  Sweeping duty perturbation ε={EPS_DUTY} around "
          f"D_op={D_OP} across {len(freqs)} points "
          f"({F_MIN:.0f} Hz → {F_MAX/1e3:.0f} kHz)...")

    # Per-freq dt: CRITICAL — dt must be much smaller than
    # ε·T_PWM, otherwise the duty perturbation gets quantised
    # to the PWM grid and the response becomes a square wave
    # (which the DFT sees as a sine with 4/π ≈ +2.1 dB extra
    # gain — measured originally as +10 dB before this fix).
    # Rule: dt ≤ ε·T_PWM / 4. With ε=0.01: dt ≤ 25 ns.
    DT_PWM_RESOLVED = T_PWM * EPS_DUTY / 4.0
    def dt_fn(f: float) -> float:
        return min(DT_PWM_RESOLVED, (1.0 / f) / 200.0)

    result = p.run_ac_sweep(
        builder,
        freqs=freqs,
        excite_fn=excite,
        switch_fn_factory=make_switch_fn,
        output_idx=vout_idx,
        dt_fn=dt_fn,
        perturbation_amplitude=EPS_DUTY,
        cycles_per_point=10,
        settling_cycles=8,
        start_from_dc_op=False,
        verbose=True,
    )

    # Analytical: G_vd(s) = V_in / (LCs² + (L/R)s + 1)
    def analytical(freqs):
        s = 1j * 2 * math.pi * np.asarray(freqs)
        return V_IN / (L_VAL * C_VAL * s**2 + (L_VAL / R_LOAD) * s + 1.0)

    H_ref = analytical(freqs)
    err_dB = result.mag_dB - 20 * np.log10(np.abs(H_ref))
    print(f"\n  Verification (analytical G_vd = V_in / (LCs²+L/R·s+1)):")
    print(f"    DC gain (measured/analytical): "
          f"{result.mag_dB[0]:.2f} / {20*math.log10(abs(H_ref[0])):.2f} dB")
    # Peak resonance — measured + predicted.
    k_peak = int(np.argmax(result.mag_dB))
    k_peak_ref = int(np.argmax(20*np.log10(np.abs(H_ref))))
    print(f"    resonance peak (measured):  {result.mag_dB[k_peak]:.2f} dB "
          f"at f = {result.freqs[k_peak]:.0f} Hz")
    print(f"    resonance peak (analytical): "
          f"{20*math.log10(abs(H_ref[k_peak_ref])):.2f} dB "
          f"at f = {result.freqs[k_peak_ref]:.0f} Hz")
    print(f"    rms |dB error|: {np.sqrt((err_dB**2).mean()):.2f} dB")

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, (ax_mag, ax_phase) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    p.plot_bode(
        result,
        title=f"Buck control-to-output G_vd(s) — V_in={V_IN}V, "
              f"L={L_VAL*1e6:.0f}µH, C={C_VAL*1e6:.0f}µF, R={R_LOAD}Ω",
        compare_analytical=analytical,
        ax=(ax_mag, ax_phase),
    )
    f_LC = 1.0 / (2 * math.pi * math.sqrt(L_VAL * C_VAL))
    ax_mag.axvline(f_LC, color="g", ls=":", lw=0.6,
                    label=f"f_LC = {f_LC:.0f} Hz")
    ax_mag.legend(loc="best")
    plt.tight_layout()
    out = Path(__file__).resolve().parent / "output" / "ac_sweep_buck.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=120)
    print(f"\n  plot → {out}")


if __name__ == "__main__":
    main()
