#!/usr/bin/env python3
"""Adversarial parameter sweep — find kernel/numerical edge cases.

Runs the same closed-loop buck topology with deliberately-bad
parameter choices and reports what survived:

  1. dt = 1 ns (very small)            — float-point precision stress
  2. dt = 50 µs (above Nyquist)        — should alias, may diverge
  3. Ki = 1e6 (saturating integrator)  — extreme windup
  4. Kp = 0 (pure I, no proportional)  — slow + oscillatory
  5. Kp = 10, Ki = 0                   — pure P, large steady-state err
  6. negative Kp                        — POSITIVE feedback → diverges
  7. enable_substep_state_correction=True with aggressive duty
  8. cold start with x_0 = 0 vs DC OP

Each case captures: (steady_state_v_out, crashed?, sim_time, sample_count).
Print a results table.
"""

from __future__ import annotations

import math
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import pulsim as p


V_IN  = 24.0
V_REF = 12.0
F_PWM = 100e3
T_PWM = 1.0 / F_PWM


@dataclass
class CaseSpec:
    name: str
    dt: float
    t_end: float
    Kp: float
    Ki: float
    start_from_dc_op: bool = False
    enable_substep: bool = False


def build_plant() -> p.CircuitBuilder:
    """Standard buck, used for every sweep case."""
    b = p.CircuitBuilder()
    b.add_voltage_source        ("Vin", "vin", "gnd", V_IN)
    b.add_mosfet_with_body_diode("Q1",  "vin", "sw",
                                   R_on=1e-3, R_off=1e9, V_F=0.7)
    b.add_diode                 ("D_FW","gnd", "sw", 1e3, 1e-9, V_th=0.7)
    b.add_inductor              ("L1",  "sw",  "vout", 100e-6)
    b.add_capacitor             ("Cout","vout","gnd",  47e-6)
    b.add_resistor              ("R_L", "vout", "gnd", 5.0)
    return b


def run_case(spec: CaseSpec) -> dict:
    """Run one adversarial case, return summary dict.
    Catches ALL exceptions so the sweep continues."""
    result = {
        "name": spec.name,
        "dt_ns": int(spec.dt * 1e9),
        "Kp": spec.Kp, "Ki": spec.Ki,
        "samples": 0,
        "v_out_late": float("nan"),
        "crashed": False,
        "error": "",
        "wall_time_s": 0.0,
    }
    t0 = time.perf_counter()
    try:
        builder = build_plant()
        vout_idx = builder.node_id_of("vout")

        pi = p.PIController(
            Kp=spec.Kp, Ki=spec.Ki,
            output_min=0.05, output_max=0.95,
        )
        duty = [0.5]

        def observe(t: float, x) -> None:
            v_out = float(x[vout_idx])
            duty[0] = pi.update(setpoint=V_REF, measured=v_out, dt=spec.dt)

        num_switches = builder.graph.num_switches

        def switch_fn(t: float):
            phase = math.fmod(t, T_PWM) / T_PWM
            m = p.SwitchStateMask(num_switches)
            if phase < duty[0]: m.set(0, True)
            return m

        res = p.simulate(
            builder, t_end=spec.t_end, dt=spec.dt,
            switch_fn=switch_fn, step_observer=observe,
            start_from_dc_op=spec.start_from_dc_op,
            max_event_iterations=8,
            enable_substep_state_correction=spec.enable_substep,
        )
        result["samples"] = res.num_steps()

        # Steady-state V_out (last 10 % of samples).
        k_late = int(0.9 * res.num_steps())
        v_out_arr = np.array([s[vout_idx] for s in res.states[k_late:]])
        result["v_out_late"] = float(v_out_arr.mean())
    except Exception as e:
        result["crashed"] = True
        result["error"] = str(e)[:120]
        traceback.print_exc()
    result["wall_time_s"] = time.perf_counter() - t0
    return result


def main() -> None:
    cases = [
        CaseSpec("baseline (sane)",          dt=1e-7,  t_end=2e-3,  Kp=0.05, Ki=800.0),
        CaseSpec("dt = 1 ns (tiny)",         dt=1e-9,  t_end=10e-6, Kp=0.05, Ki=800.0),
        CaseSpec("dt = 50 µs (>Nyquist)",    dt=5e-5,  t_end=10e-3, Kp=0.05, Ki=800.0),
        CaseSpec("Ki = 1e6 (huge)",          dt=1e-7,  t_end=2e-3,  Kp=0.05, Ki=1e6),
        CaseSpec("Kp=0 (pure I)",            dt=1e-7,  t_end=4e-3,  Kp=0.0,  Ki=1000.0),
        CaseSpec("Ki=0 (pure P)",            dt=1e-7,  t_end=2e-3,  Kp=10.0, Ki=0.0),
        CaseSpec("Kp<0 (positive feedback)", dt=1e-7,  t_end=2e-3,  Kp=-0.1, Ki=-200.0),
        CaseSpec("start_from_dc_op=True",    dt=1e-7,  t_end=2e-3,  Kp=0.05, Ki=800.0,
                  start_from_dc_op=True),
        CaseSpec("substep correction ON",    dt=1e-7,  t_end=2e-3,  Kp=0.05, Ki=800.0,
                  enable_substep=True),
    ]

    print("Adversarial closed-loop buck sweep — V_in=24V, V_ref=12V")
    print("-" * 86)
    print(f"{'case':36} {'dt':>8} {'Kp':>7} {'Ki':>9} "
          f"{'v_out':>8} {'samples':>9} {'time':>8}")
    print("-" * 86)
    for spec in cases:
        r = run_case(spec)
        status = "CRASH" if r["crashed"] else f"{r['v_out_late']:8.3f}"
        print(f"{r['name']:36} {spec.dt*1e9:>5.0f}ns "
              f"{r['Kp']:>7.3f} {r['Ki']:>9.1f} "
              f"{status:>8} {r['samples']:>9d} "
              f"{r['wall_time_s']:>7.2f}s")
        if r["crashed"]:
            print(f"  → {r['error']}")
    print("-" * 86)


if __name__ == "__main__":
    main()
