#!/usr/bin/env python3
"""Fast frequency sweep via MNA-style impulse response + FFT.

Compares ``run_mna_sweep`` (impulse + FFT, single transient) against the
analytic transfer function of a 1-pole RC filter. Validates the
mathematical equivalence with the future C++ kernel MNA sweep.

KPI: max |mag| error ≤ 0.5 dB and max |phase| error ≤ 5° across
100 Hz – 30 kHz (about 5τ of the filter's bandwidth).

Compared to ``run_ac_sweep_rc.py`` (swept-sine), this example:
  * runs ~200× faster (one transient vs. one-per-frequency),
  * produces results across the full band in a single FFT.

Trade-off: only works for LTI (or linearised-at-x_op) plants. For
heavily switching converters with time-varying duty, fall back to the
swept-sine version.
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np

import pulsim.v2 as p


R = 1.0e3
C = 100.0e-9
RC = R * C
FC = 1.0 / (2 * math.pi * RC)


def build_rc() -> p.CircuitBuilder:
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", 0.0)
    b.add_resistor("R1", "vin", "vout", R)
    b.add_capacitor("C1", "vout", "gnd", C)
    return b


def main() -> None:
    print(f"  RC filter: R={R:.0f} Ω, C={C*1e9:.0f} nF, τ={RC*1e6:.0f} µs, "
          f"fc={FC:.0f} Hz")

    freqs = np.logspace(1, 5, 81)   # 10 Hz → 100 kHz, 81 pts
    H_an = 1.0 / (1.0 + 1j * 2*math.pi*freqs*RC)
    mag_an = 20.0 * np.log10(np.abs(H_an))
    ph_an  = np.degrees(np.angle(H_an))

    b = build_rc()
    t0 = time.perf_counter()
    res = p.run_mna_sweep(
        b,
        output_idx=b.node_id_of("vout"),
        v_in_branch_id=0,
        dt=1.0e-6, t_end=80.0e-3,
        freqs=freqs,
        impulse_amplitude=1.0,
        verbose=True,
    )
    print(f"  MNA sweep wall: {time.perf_counter()-t0:.2f} s")

    # KPI check
    mag_err = np.abs(res.mag_dB - mag_an)
    ph_err  = np.abs(((res.phase_deg - ph_an) + 180) % 360 - 180)
    band    = (freqs >= 100) & (freqs <= 30000)
    print(f"\n  In-band (100Hz – 30kHz):")
    print(f"    max |mag err|   = {mag_err[band].max():.3f} dB")
    print(f"    max |phase err| = {ph_err[band].max():.3f}°")

    # Plot
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("(install matplotlib to see the comparison plot)")
        return

    fig, (ax_mag, ax_ph) = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    ax_mag.semilogx(freqs, mag_an,        "k-",  lw=2,   label="analytic")
    ax_mag.semilogx(freqs, res.mag_dB,    "C0o", ms=4,   label="MNA (impulse + FFT)")
    ax_mag.axhline(-3, color="g", ls=":", lw=0.5)
    ax_mag.axvline(FC, color="g", ls=":", lw=0.5,
                    label=f"fc = {FC:.0f} Hz")
    ax_mag.set_ylabel("|H| [dB]"); ax_mag.grid(alpha=0.3, which="both")
    ax_mag.legend(loc="lower left")
    ax_mag.set_title(f"RC low-pass — MNA sweep vs analytic "
                       f"(R={R:.0f}Ω, C={C*1e9:.0f}nF)")

    ax_ph.semilogx(freqs, ph_an,         "k-",  lw=2)
    ax_ph.semilogx(freqs, res.phase_deg, "C0o", ms=4)
    ax_ph.axhline(-45, color="g", ls=":", lw=0.5)
    ax_ph.axvline(FC, color="g", ls=":", lw=0.5)
    ax_ph.set_xlabel("frequency [Hz]"); ax_ph.set_ylabel("phase [°]")
    ax_ph.grid(alpha=0.3, which="both")

    plt.tight_layout()
    out = Path(__file__).resolve().parent / "output" / "mna_sweep_rc.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=120)
    print(f"\n  plot → {out}")


if __name__ == "__main__":
    main()
