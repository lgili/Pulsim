#!/usr/bin/env python3
"""Post-hoc loss → T_j pipeline using `device_thermal_summary`.

End-to-end electro-thermal characterisation without any kernel
plumbing. The script:

1. Builds a half-bridge-style switching converter (V_DC → R →
   ideal MOSFET → gnd) and drives it with a 50 % PWM.
2. Runs `pulsim.simulate(...)` once.
3. Calls `pulsim.device_loss_summary(...)` to extract per-device
   conduction + switching loss (PSIM-style: `switch_specs` with
   E_on / E_off referenced at a datasheet operating point).
4. Calls `pulsim.device_thermal_summary(...)` to layer those
   losses on top of a 3-stage Foster network (`Z_th(t)`) and
   compute `T_j(t)` for every device that has a thermal spec.
5. Plots and prints the per-device loss + temperature table.

Usage::

    python examples/scripts/run_device_loss_to_thermal.py [--plot]
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

import pulsim as p


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="device_loss_summary → device_thermal_summary demo.")
    parser.add_argument("--plot", action="store_true",
                          help="Show the T_j(t) plot (needs matplotlib).")
    args = parser.parse_args(argv)

    # ------------------------------------------------------------
    # 1. Plant — DC source, line resistor, ideal MOSFET to ground.
    # ------------------------------------------------------------
    V_DC = 48.0
    R_LINE = 1.0
    G_ON, G_OFF = 100.0, 1e-9

    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", V_DC)
    b.add_resistor("R_line", "vin", "ds", R_LINE)
    b.add_switch("M1", "ds", "gnd", g_on=G_ON, g_off=G_OFF)

    # ------------------------------------------------------------
    # 2. 50 % PWM at 20 kHz.
    # ------------------------------------------------------------
    F_SW = 20_000.0
    T_SW = 1.0 / F_SW

    def pwm(t: float):
        m = p.SwitchStateMask(b.graph.num_switches)
        m.set(0, bool((t % T_SW) < 0.5 * T_SW))
        return m

    # Long enough to settle the slow Foster pole.
    N_PERIODS = 1_000
    res = p.simulate(b, t_end=N_PERIODS * T_SW, dt=T_SW / 50,
                       switch_fn=pwm)

    # ------------------------------------------------------------
    # 3. Loss summary — conduction (R + switch) + datasheet
    #    switching loss (PSIM-style E_on / E_off at a reference
    #    operating point).
    # ------------------------------------------------------------
    switch_specs = {
        "M1": {
            "E_on_ref":  120e-6,    # 120 µJ datasheet @ (V_ref, I_ref)
            "E_off_ref": 180e-6,
            "V_ref":     V_DC,      # 48 V test condition
            "I_ref":     V_DC / R_LINE,
        }
    }
    loss = p.device_loss_summary(
        b, res, switch_fn=pwm, switch_specs=switch_specs)

    print("\n=== device_loss_summary ===")
    print(f"{'name':<8} {'kind':<10} {'P_cond':>10}  {'P_sw':>10}  "
          f"{'duty':>7}")
    for e in loss:
        P_sw = e.get("P_sw_avg", 0.0)
        duty = e.get("duty_closed", e.get("duty_conducting", float("nan")))
        print(f"{e['name']:<8} {e['kind']:<10} "
              f"{e['P_avg']:>8.3f} W  {P_sw:>8.3f} W  "
              f"{duty:>6.3f}")

    # ------------------------------------------------------------
    # 4. Thermal pipeline — Foster Z_th(t) → T_j(t).
    #    Two-stage representative of a TO-220 silicon MOSFET on
    #    a small heatsink.
    # ------------------------------------------------------------
    foster = [
        p.FosterStage(R_th_K_per_W=0.30, tau_s=2e-3),   # die-to-case
        p.FosterStage(R_th_K_per_W=1.20, tau_s=80e-3),  # case-to-heatsink
    ]
    therm = p.device_thermal_summary(
        b, res,
        thermal_specs={"M1": {"stages": foster, "T_ambient_C": 40.0}},
        switch_fn=pwm,
        switch_specs=switch_specs,
        T_ambient_C=40.0,
    )

    print("\n=== device_thermal_summary ===")
    print(f"{'name':<8} {'P_total':>10}  {'R_th':>8}  "
          f"{'T_amb':>8}  {'T_j avg':>9}  {'T_j peak':>10}")
    for e in therm:
        print(f"{e['name']:<8} "
              f"{e['P_total_avg']:>8.3f} W  "
              f"{e['R_th_total']:>6.2f} K/W  "
              f"{e['T_ambient_C']:>6.1f} °C  "
              f"{e['T_j_avg']:>7.2f} °C  "
              f"{e['T_j_peak']:>8.2f} °C")

    # Sanity: T_j_steady_state asymptote = T_amb + P_total · R_th.
    e = therm[0]
    T_inf = e["T_ambient_C"] + e["P_total_avg"] * e["R_th_total"]
    print(f"\nExpected steady-state asymptote: {T_inf:.2f} °C "
          f"(actual T_j_peak: {e['T_j_peak']:.2f} °C)")

    # ------------------------------------------------------------
    # 5. Optional plot.
    # ------------------------------------------------------------
    if args.plot:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("\n(matplotlib not installed — skipping plot)")
            return 0

        times = np.asarray(res.times)
        fig, (ax_p, ax_t) = plt.subplots(2, 1, sharex=True,
                                              figsize=(8, 5))
        # P_cond reconstructed: switch v² · g_arr — pull from the
        # summary's P_avg as a constant overlay just to anchor the
        # eye. The detailed P_cond(t) lives inside
        # device_thermal_summary; expose it via T_j(t) instead.
        ax_p.axhline(e["P_total_avg"], color="C0",
                       label="P_total_avg")
        ax_p.axhline(e["P_sw_avg"], color="C1", linestyle="--",
                       label="P_sw_avg (datasheet)")
        ax_p.set_ylabel("Power [W]")
        ax_p.legend(loc="lower right")
        ax_p.set_title(f"M1: P_total_avg = {e['P_total_avg']:.2f} W "
                          f"(η ≈ {(1 - e['P_total_avg'] / (V_DC**2 / R_LINE)) * 100:.1f} %)")

        ax_t.plot(times * 1e3, e["T_j_trace"], color="C3",
                    label="T_j(t)")
        ax_t.axhline(e["T_ambient_C"], color="0.5", linestyle=":",
                       label=f"T_amb = {e['T_ambient_C']:.0f} °C")
        ax_t.axhline(T_inf, color="C2", linestyle="--",
                       label=f"T_j_∞ = {T_inf:.1f} °C")
        ax_t.set_xlabel("t [ms]")
        ax_t.set_ylabel("Junction T [°C]")
        ax_t.legend(loc="lower right")

        out = "examples/scripts/out/device_loss_to_thermal.png"
        fig.tight_layout()
        try:
            fig.savefig(out, dpi=140)
            print(f"\nPlot saved to {out}")
        except Exception as exc:  # noqa: BLE001
            print(f"\n(could not save plot: {exc})")
        plt.show()

    return 0


if __name__ == "__main__":
    sys.exit(main())
