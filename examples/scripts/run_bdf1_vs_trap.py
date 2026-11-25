#!/usr/bin/env python3
"""BDF1 vs trapezoidal on a stiff RLC step response.

Plant: 1V step → R(1mΩ) → L(1µH) → C(1µF) → R_load(1kΩ) → gnd.
LC resonance: ω_n = 1/√(LC) = 1e6 rad/s, ζ = R/(2·√(L/C)) ≈ 0.0005
(VERY underdamped — natural ring).

Trapezoidal is A-stable but not L-stable: on a stiff step with
dt ≫ 1/ω_n it can leave a zipper (alternating-sign ringing at the
step rate). BDF1 (implicit Euler) is L-stable but first order: it
damps any ringing, physical or not.

At the dt used here (100 ns, ω_n·dt = 0.1) the ring is resolved and
trapezoidal is the ACCURATE one: its peak (1.996 V) matches the
analytic 1 + exp(−πζ/√(1−ζ²)) = 1.997 V (ζ = 0.001 with the load),
while BDF1's numerical damping (ζ_eff ≈ ω_n·dt/2 = 0.05, 50× the
circuit's) cuts the peak to 1.853 V.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import pulsim as p
from pulsim import _pulsim as _k


R     = 1e-3       # 1 mΩ series
L     = 1e-6       # 1 µH
C     = 1e-6       # 1 µF
R_LOAD = 1e3       # 1 kΩ — barely damps
T_END = 30e-6      # 30 µs ≈ 5 resonant periods
DT    = 1.0e-7     # 100 ns


def build_plant() -> p.CircuitBuilder:
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", 1.0)
    b.add_resistor("R_s", "vin", "n1", R)
    b.add_inductor("L1", "n1", "vout", L)
    b.add_capacitor("C1", "vout", "gnd", C)
    b.add_resistor("R_load", "vout", "gnd", R_LOAD)
    return b


def run_trap(b):
    return p.simulate(b, t_end=T_END, dt=DT)


def run_bdf1(b):
    opts = _k.SimulationOptions(t_start=0.0, t_end=T_END, dt=DT)
    num_sw = b.graph.num_switches
    default_mask = _k.SwitchStateMask(num_sw)
    for i in range(num_sw):
        default_mask.set(i, True)
    sw_fn = lambda t: default_mask  # noqa: E731
    return _k.run_transient_bdf1(b, opts, sw_fn)


def main() -> None:
    b1 = build_plant()
    b2 = build_plant()

    print(f"  Plant: R={R*1e3} mΩ, L={L*1e6} µH, C={C*1e6} µF, "
          f"R_load={R_LOAD/1e3:.1f} kΩ")
    omega_n = 1.0 / np.sqrt(L*C)
    print(f"  ω_n = {omega_n:.2e} rad/s "
          f"({omega_n/(2*np.pi)/1e3:.0f} kHz)")
    zeta = R / (2 * np.sqrt(L/C))
    print(f"  ζ = {zeta:.4f}  (very underdamped → ringing)")
    print(f"  dt = {DT*1e9:.0f} ns, T_end = {T_END*1e6:.0f} µs")

    print("\n  Running trap…")
    res_trap = run_trap(b1)
    print(f"    {res_trap.num_steps()} samples")

    print("  Running BDF1…")
    res_bdf1 = run_bdf1(b2)
    print(f"    {res_bdf1.num_steps()} samples")

    t_trap = np.asarray(res_trap.times)
    t_bdf1 = np.asarray(res_bdf1.times)
    v_trap = np.asarray(res_trap.states)[:,
                                              b1.node_id_of("vout")]
    v_bdf1 = np.asarray(res_bdf1.states)[:,
                                              b2.node_id_of("vout")]

    print("\n  Peak overshoot:")
    print(f"    trap = {v_trap.max():.4f} V")
    print(f"    BDF1 = {v_bdf1.max():.4f} V "
          f"(numerical damping of the first-order method)")
    zeta_tot = zeta + np.sqrt(L / C) / (2 * R_LOAD)
    print(f"    analytic = {1 + np.exp(-np.pi * zeta_tot / np.sqrt(1 - zeta_tot**2)):.4f} V")
    print("  Final V_out:")
    print(f"    trap = {v_trap[-1]:.4f} V")
    print(f"    BDF1 = {v_bdf1[-1]:.4f} V")

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(1, 1, figsize=(11, 5))
    ax.plot(t_trap*1e6, v_trap, "C0-", lw=0.9,
              label="trapezoidal (2nd-order, A-stable)")
    ax.plot(t_bdf1*1e6, v_bdf1, "C3-", lw=0.9,
              label="BDF1 / implicit Euler (1st-order, L-stable)")
    ax.set_xlabel("time [µs]"); ax.set_ylabel("V_out [V]")
    ax.set_title(f"Stiff RLC step response — trap vs BDF1  "
                   f"(dt = {DT*1e9:.0f} ns, ω_n ≈ 1 MHz)")
    ax.grid(alpha=0.3); ax.legend(loc="upper right")
    out = Path(__file__).resolve().parent / "output" / "bdf1_vs_trap.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=120)
    print(f"\n  plot → {out}")


if __name__ == "__main__":
    main()
