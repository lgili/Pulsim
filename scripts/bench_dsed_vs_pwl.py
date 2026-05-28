"""Bridge.11 DSED-vs-PWL wall-clock benchmark sweep.

Runs the same converter end-to-end through both engines and reports
wall-clock + steps + per-step cost + final-state agreement. Run after
Bridge.11 lands so the native C++ scheduler path is exercised.

Usage:
    python scripts/bench_dsed_vs_pwl.py
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
import pulsim as p


# ---------------------------------------------------------------------------
# PWM helpers — class with `next_edge_after` so DSED gate-edge fast path fires
# ---------------------------------------------------------------------------

class _PWM2Switch:
    """2-switch complementary PWM (HS on for D·T, LS on for (1-D)·T)."""
    def __init__(self, T_sw: float, D: float, hs_first: bool = True):
        self.T_sw, self.D = T_sw, D
        self.m_a = p.SwitchStateMask(2)
        self.m_a.set(0, True);  self.m_a.set(1, False)
        self.m_b = p.SwitchStateMask(2)
        self.m_b.set(0, False); self.m_b.set(1, True)
        if not hs_first:
            self.m_a, self.m_b = self.m_b, self.m_a

    def __call__(self, t):
        phase = (t / self.T_sw) % 1.0
        return self.m_a if phase < self.D else self.m_b

    def next_edge_after(self, t):
        k = int(math.floor(t / self.T_sw))
        eps = 1e-15
        for c in (k * self.T_sw + self.D * self.T_sw,
                  (k + 1) * self.T_sw,
                  (k + 1) * self.T_sw + self.D * self.T_sw):
            if c > t + eps:
                return c
        return (k + 2) * self.T_sw


# ---------------------------------------------------------------------------
# Converters
# ---------------------------------------------------------------------------

def build_buck_ccm(V_in=24.0, L=100e-6, C=100e-6, R=2.4):
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "in", "gnd", V_in)
    G_ON, G_OFF = 1e6, 1e-9
    b.add_switch("SW_HS", "in", "sw", G_ON, G_OFF)
    b.add_switch("SW_LS", "sw", "gnd", G_ON, G_OFF)
    b.add_inductor("L", "sw", "out", L)
    b.add_capacitor("C", "out", "gnd", C)
    b.add_resistor("R", "out", "gnd", R)
    return b


def build_boost_ccm(V_in=12.0, L=100e-6, C=100e-6, R=10.0):
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "in", "gnd", V_in)
    G_ON, G_OFF = 1e6, 1e-9
    b.add_inductor("L", "in", "sw", L)
    # LS closed during D·T (inductor charges from V_in to gnd)
    b.add_switch("SW_LS", "sw", "gnd", G_ON, G_OFF)
    # HS closed during (1-D)·T (sync rectifier — current pumps to out)
    b.add_switch("SW_HS", "sw", "out", G_ON, G_OFF)
    b.add_capacitor("C", "out", "gnd", C)
    b.add_resistor("R", "out", "gnd", R)
    return b


def build_buck_boost(V_in=24.0, L=100e-6, C=100e-6, R=5.0):
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "in", "gnd", V_in)
    G_ON, G_OFF = 1e6, 1e-9
    # Input switch from V_in to "sw"
    b.add_switch("SW_HS", "in", "sw", G_ON, G_OFF)
    # Inductor from "sw" to gnd (charges to V_in when HS closed)
    b.add_inductor("L", "sw", "gnd", L)
    # Output diode/switch from "sw" to "out_neg" (cap inverts polarity)
    b.add_switch("SW_LS", "sw", "out_neg", G_ON, G_OFF)
    # Output cap & load between gnd and out_neg (cap voltage is NEGATIVE)
    b.add_capacitor("C", "gnd", "out_neg", C)
    b.add_resistor("R", "gnd", "out_neg", R)
    return b


def build_halfbridge_sine_input(V_dc=24.0, V_amp=4.0, f_in=1e3,
                                  L=100e-6, C=100e-6, R=5.0):
    """Half-bridge driven by sine-modulated DC input."""
    b = p.CircuitBuilder()
    b.add_sine_voltage_source("Vin", "in", "gnd",
                                v_dc=V_dc, v_amplitude=V_amp,
                                frequency=f_in, phase=0.0)
    G_ON, G_OFF = 1e6, 1e-9
    b.add_switch("SW_HS", "in", "sw", G_ON, G_OFF)
    b.add_switch("SW_LS", "sw", "gnd", G_ON, G_OFF)
    b.add_inductor("L", "sw", "out", L)
    b.add_capacitor("C", "out", "gnd", C)
    b.add_resistor("R", "out", "gnd", R)
    return b


def build_floating_cap_rlc(V=12.0, R1=2.0, R2=8.0, L=10e-6, C=10e-6):
    """RLC with floating cap — exercises Bridge.5.1b on a non-switching
    circuit (no PWM events → DSED only gets the t_end event)."""
    b = p.CircuitBuilder()
    b.add_voltage_source("V1", "vin", "gnd", V)
    b.add_resistor("R1",   "vin", "n1",  R1)
    b.add_inductor("L1",   "n1",  "n2",  L)
    b.add_capacitor("C1",  "n2",  "n3",  C)   # FLOATING
    b.add_resistor("R2",   "n3",  "gnd", R2)
    return b


def build_npc_split_bus_discharge(V_dc=100.0, R_src=1.0, C=10e-6,
                                    R_load=100.0):
    """NPC-style 2-cap floating split bus (no switching). Exercises
    Bridge.5.1b's NPC test under a longer simulation window."""
    b = p.CircuitBuilder()
    b.add_voltage_source("Vdc", "vin", "gnd", V_dc)
    b.add_resistor("R_src",       "vin",   "n_pos", R_src)
    b.add_capacitor("C_top",      "n_pos", "n_mid", C)
    b.add_capacitor("C_bot",      "n_mid", "n_neg", C)
    b.add_resistor("R_load",      "n_neg", "gnd",   R_load)
    return b


# ---------------------------------------------------------------------------
# Bench harness
# ---------------------------------------------------------------------------

@dataclass
class BenchResult:
    name: str
    dsed_ms: float
    pwl_ms: float
    dsed_steps: int
    pwl_steps: int
    speedup: float
    dsed_final: float
    pwl_final: float
    n_events: int


def _wall_clock(fn, n_runs=5):
    """Run `fn()` n_runs times; return min wall-clock in ms + the last result."""
    res = None
    times_ms = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        res = fn()
        times_ms.append((time.perf_counter() - t0) * 1000.0)
    return min(times_ms), res


def bench_one(name: str, build_fn, sf, t_end: float, pwl_dt: float,
               state_index: int = 0, has_switches: bool = True,
               n_runs: int = 5):
    """Bench DSED vs PWL on one converter."""
    # Warm-up — Pulsim caches per-mask, so first run pays the build cost
    b = build_fn()
    if has_switches:
        _ = p.simulate(b, t_end=t_end, engine='dsed', integrator='rk45',
                        rtol=1e-6, switch_fn=sf)
        _ = p.simulate(b, t_end=t_end, dt=pwl_dt, switch_fn=sf)
    else:
        _ = p.simulate(b, t_end=t_end, engine='dsed', integrator='rk45',
                        rtol=1e-6)
        _ = p.simulate(b, t_end=t_end, dt=pwl_dt)

    # Measure DSED
    b = build_fn()
    if has_switches:
        dsed_ms, r_dsed = _wall_clock(lambda: p.simulate(
            b, t_end=t_end, engine='dsed', integrator='rk45',
            rtol=1e-6, switch_fn=sf), n_runs=n_runs)
    else:
        dsed_ms, r_dsed = _wall_clock(lambda: p.simulate(
            b, t_end=t_end, engine='dsed', integrator='rk45',
            rtol=1e-6), n_runs=n_runs)

    # Measure PWL
    b = build_fn()
    if has_switches:
        pwl_ms, r_pwl = _wall_clock(lambda: p.simulate(
            b, t_end=t_end, dt=pwl_dt, switch_fn=sf), n_runs=n_runs)
    else:
        pwl_ms, r_pwl = _wall_clock(lambda: p.simulate(
            b, t_end=t_end, dt=pwl_dt), n_runs=n_runs)

    return BenchResult(
        name=name,
        dsed_ms=dsed_ms,
        pwl_ms=pwl_ms,
        dsed_steps=r_dsed.num_steps(),
        pwl_steps=r_pwl.num_steps(),
        speedup=pwl_ms / dsed_ms,
        dsed_final=float(r_dsed.states[-1][state_index]),
        pwl_final=float(r_pwl.states[-1][state_index]),
        n_events=getattr(r_dsed, "n_events", 0),
    )


def main():
    print("=" * 86)
    print(f"{'Converter':<28} {'DSED ms':>8} {'PWL ms':>8} "
          f"{'Speedup':>8} {'DSED stp':>8} {'PWL stp':>8} "
          f"{'Events':>7}")
    print("=" * 86)

    results = []

    # ----- Switched converters (5 ms window) -----
    sf_100k = _PWM2Switch(T_sw=1.0 / 100e3, D=0.5)
    sf_boost = _PWM2Switch(T_sw=1.0 / 100e3, D=0.5, hs_first=False)

    cases = [
        ("Buck CCM 24V→12V 100k",
         lambda: build_buck_ccm(), sf_100k, 5e-3, 100e-9, 0, True),
        ("Boost 12V→24V 100k",
         lambda: build_boost_ccm(), sf_boost, 10e-3, 100e-9, 0, True),
        ("Buck-boost 24V→-? 100k",
         lambda: build_buck_boost(), sf_100k, 5e-3, 100e-9, 0, True),
        ("Half-bridge + sine V_in",
         lambda: build_halfbridge_sine_input(), sf_100k, 5e-3, 100e-9, 0,
         True),
    ]

    for name, bf, sf, t_end, dt, idx, has_sw in cases:
        try:
            r = bench_one(name, bf, sf, t_end, dt, state_index=idx,
                           has_switches=has_sw)
            results.append(r)
            print(f"{name:<28} {r.dsed_ms:>8.2f} {r.pwl_ms:>8.2f} "
                  f"{r.speedup:>7.1f}× {r.dsed_steps:>8d} {r.pwl_steps:>8d} "
                  f"{r.n_events:>7d}")
        except Exception as e:
            print(f"{name:<28}  FAILED: {type(e).__name__}: "
                  f"{str(e).split(chr(10))[0][:50]}")

    # ----- Non-switched (LTI decay; tests floating-cap + native adapter) -----
    print("-" * 86)
    decay_cases = [
        ("Floating-cap RLC discharge",
         lambda: build_floating_cap_rlc(), 1e-3, 1e-7, 0),
        ("NPC split-bus 100V (2 caps)",
         lambda: build_npc_split_bus_discharge(), 5e-3, 1e-6, 0),
    ]

    for name, bf, t_end, dt, idx in decay_cases:
        try:
            r = bench_one(name, bf, sf=None, t_end=t_end, pwl_dt=dt,
                           state_index=idx, has_switches=False)
            results.append(r)
            print(f"{name:<28} {r.dsed_ms:>8.2f} {r.pwl_ms:>8.2f} "
                  f"{r.speedup:>7.1f}× {r.dsed_steps:>8d} {r.pwl_steps:>8d} "
                  f"{r.n_events:>7d}")
        except Exception as e:
            print(f"{name:<28}  FAILED: {type(e).__name__}: "
                  f"{str(e).split(chr(10))[0][:50]}")

    print("=" * 86)
    print()

    # Honest summary
    if results:
        speedups = [r.speedup for r in results]
        geo = math.exp(sum(math.log(s) for s in speedups) / len(speedups))
        print(f"Speedups: min={min(speedups):.2f}×, "
              f"geo-mean={geo:.2f}×, max={max(speedups):.2f}×")
        print()
        print("Per-step costs (µs/step):")
        for r in results:
            dsed_us = r.dsed_ms * 1000 / max(r.dsed_steps, 1)
            pwl_us = r.pwl_ms * 1000 / max(r.pwl_steps, 1)
            print(f"  {r.name:<28} DSED={dsed_us:>7.2f}  PWL={pwl_us:>6.2f}")
        print()
        print("Final-state agreement (DSED vs PWL on state[0]):")
        for r in results:
            if abs(r.pwl_final) > 1e-9:
                err_pct = abs(r.dsed_final - r.pwl_final) / abs(r.pwl_final) * 100
                print(f"  {r.name:<28} DSED={r.dsed_final:>10.4f}  "
                      f"PWL={r.pwl_final:>10.4f}  err={err_pct:.2f}%")
            else:
                print(f"  {r.name:<28} DSED={r.dsed_final:>10.4f}  "
                      f"PWL={r.pwl_final:>10.4f}  (PWL ≈ 0)")


if __name__ == "__main__":
    main()
