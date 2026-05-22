#!/usr/bin/env python3
"""AC sweep of a 1st-order RC low-pass filter.

   V_in (= V_dc + ε·sin(ωt)) ── R ── V_out ── C ── gnd

Analytical: H(jω) = 1 / (1 + jω·RC)

With R = 1 kΩ, C = 1 µF → τ = 1 ms, corner at 159.15 Hz.

The swept-sine technique should match the analytical formula
to within ~0.5 dB and ~2° over the entire 4-decade sweep.
This is the "ground truth" test that validates the v2 AC
analysis machinery.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

import pulsim.v2 as p


R       = 1000.0
C       = 1.0e-6
V_DC    = 5.0
F_MIN   = 10.0
F_MAX   = 100e3
N_PTS   = 30                 # log-spaced
EPS     = 0.1                # 100 mV perturbation
SAMPLES_PER_CYCLE = 100      # auto-pick dt per freq


def build_rc() -> p.CircuitBuilder:
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "n0", "gnd", V_DC)
    b.add_resistor      ("R",   "n0", "vc",   R)
    b.add_capacitor     ("C",   "vc", "gnd", C)
    return b


def main() -> None:
    builder = build_rc()
    vc_idx  = builder.node_id_of("vc")
    state_n = builder.pool.state_size(builder.graph)
    vin_branch_var = builder.pool.branch_var_id_for_source(
        0, builder.graph)
    print(f"  state_size = {state_n}, "
          f"Vin branch_var_id = {vin_branch_var}, "
          f"vc node_idx = {vc_idx}")

    def excite(t: float, eps: float, freq: float) -> list:
        # Inject ε·sin(2π·f·t) on Vin's constraint row.
        # In v2's MNA convention, the constraint reads
        # (V_from − V_to) − V_set = 0; adding +ε·sin to the
        # constraint residual shifts V_set by −ε·sin (so V_from
        # actually moves by +ε·sin — which is what we want).
        out = [0.0] * state_n
        out[vin_branch_var] = -eps * math.sin(
            2 * math.pi * freq * t)
        return out

    freqs = np.logspace(math.log10(F_MIN), math.log10(F_MAX), N_PTS)
    print(f"\n  Sweeping {len(freqs)} points from {F_MIN:.0f} Hz "
          f"to {F_MAX/1e3:.0f} kHz...")
    result = p.run_ac_sweep(
        builder,
        freqs=freqs,
        excite_fn=excite,
        output_idx=vc_idx,
        samples_per_cycle=SAMPLES_PER_CYCLE,
        perturbation_amplitude=EPS,
        cycles_per_point=10,
        settling_cycles=5,
        verbose=False,
    )

    # Analytical reference: H(jω) = 1 / (1 + jωRC)
    def analytical(freqs):
        omega = 2 * math.pi * freqs
        return 1.0 / (1.0 + 1j * omega * R * C)

    # Compare at a few key frequencies.
    H_ref = analytical(freqs)
    err_dB = result.mag_dB - 20 * np.log10(np.abs(H_ref))
    err_deg = result.phase_deg - np.degrees(np.angle(H_ref))
    err_deg = np.mod(err_deg + 180, 360) - 180
    print(f"\n  Verification (analytical |H|=1/(1+jωRC)):")
    print(f"    max |dB error|: {np.abs(err_dB).max():.3f} dB")
    print(f"    max |deg error|: {np.abs(err_deg).max():.3f} deg")
    print(f"    corner freq: f_c = {1/(2*math.pi*R*C):.2f} Hz")

    # Plot.
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, (ax_mag, ax_phase) = plt.subplots(
        2, 1, figsize=(10, 6), sharex=True)
    p.plot_bode(result,
                  title=f"RC low-pass — R={R:.0f}Ω, C={C*1e6:.1f}µF, "
                        f"f_c = {1/(2*math.pi*R*C):.1f} Hz",
                  compare_analytical=analytical,
                  ax=(ax_mag, ax_phase))
    # Mark the corner.
    f_c = 1 / (2 * math.pi * R * C)
    ax_mag.axvline(f_c, color="g", ls=":", lw=0.6,
                    label=f"f_c = {f_c:.1f} Hz")
    ax_mag.legend(loc="best")
    plt.tight_layout()
    out = Path(__file__).resolve().parent / "output" / "ac_sweep_rc.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=120)
    print(f"  plot → {out}")


if __name__ == "__main__":
    main()
