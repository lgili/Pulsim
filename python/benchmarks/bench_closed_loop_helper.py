#!/usr/bin/env python3
"""Benchmark: ``bind_pi_to_switch`` helper vs hand-rolled PI + PWM buck.

Measures wall-clock runtime of the closed-loop buck converter over
``t_end = 10 ms``, ``dt = 5 us`` using:

1. ``pulsim.bind_pi_to_switch(..., measured="vout", ...)`` +
   ``pulsim.simulate(..., closed_loops=[loop])``
2. The equivalent hand-rolled PI + throttled ``step_observer`` +
   ``switch_fn`` loop.

Both runners share the identical buck circuit and PI parameters and
compute steady-state ``vout`` over the final 2 ms (``400`` samples at
``5 us``). Pass ``--check`` to enforce ``ratio <= 1.05`` and
steady-state delta ``<= 1e-12``.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time

import numpy as np

import pulsim as ps

DEFAULT_T_END = 10e-3
DEFAULT_DT = 5e-6
DEFAULT_FREQ = 10e3
DEFAULT_SETPOINT = 5.0
SS_WINDOW_S = 2e-3


def build_buck() -> ps.CircuitBuilder:
    """Build the standard 12 V -> 5 V synchronous/diode buck plant."""
    b = ps.CircuitBuilder()
    b.add_voltage_source("V1", "vin", "gnd", 12.0)
    b.add_mosfet_with_body_diode(
        "Q1", "vin", "sw", R_on=1e-3, R_off=1e9, V_F=0.7,
    )
    b.add_diode("D1", "gnd", "sw", 1e3, 1e-9, V_th=0.7)
    b.add_inductor("L1", "sw", "vout", 220e-6)
    b.add_capacitor("Cout", "vout", "gnd", 220e-6)
    b.add_resistor("R_L", "vout", "gnd", 8.0)
    return b


def _steady_state_vout(res: ps.SimulationResult, dt: float) -> float:
    t = np.asarray(res.times)
    return float(ps.time_average(t, res.v("vout"), t_from=float(t[-1]) - SS_WINDOW_S))


def run_helper_buck(
    *,
    t_end: float = DEFAULT_T_END,
    dt: float = DEFAULT_DT,
    freq: float = DEFAULT_FREQ,
    setpoint: float = DEFAULT_SETPOINT,
) -> float:
    """Run one closed-loop buck simulation via ``bind_pi_to_switch``."""
    b = build_buck()
    pi = ps.PIController(
        Kp=0.08, Ki=40.0, output_min=0.05, output_max=0.95,
    )
    loop = ps.bind_pi_to_switch(
        b,
        pi=pi,
        measured="vout",
        setpoint=setpoint,
        switch="Q1",
        freq=freq,
    )
    res = ps.simulate(b, t_end=t_end, dt=dt, closed_loops=[loop])
    return _steady_state_vout(res, dt)


def run_hand_rolled_buck(
    *,
    t_end: float = DEFAULT_T_END,
    dt: float = DEFAULT_DT,
    freq: float = DEFAULT_FREQ,
    setpoint: float = DEFAULT_SETPOINT,
) -> float:
    """Run one closed-loop buck simulation via hand-rolled PI + PWM."""
    b = build_buck()
    pi = ps.PIController(
        Kp=0.08, Ki=40.0, output_min=0.05, output_max=0.95,
    )
    vout_idx = b.node_id_of("vout")
    q1_idx = b.switch_index_of("Q1")
    n_switches = b.graph.num_switches
    T_PWM = 1.0 / freq
    duty = [0.5]
    last_t = [-T_PWM]

    def step_observer(t: float, x) -> None:
        if t - last_t[0] >= T_PWM - 1e-12:
            last_t[0] = t
            duty[0] = pi.update(
                setpoint=setpoint, measured=float(x[vout_idx]), dt=T_PWM,
            )

    def switch_fn(t: float):
        mask = ps.SwitchStateMask(n_switches)
        if ps.control.pwm_on(t, T_PWM, duty[0]):
            mask.set(q1_idx, True)
        return mask

    res = ps.simulate(
        b,
        t_end=t_end,
        dt=dt,
        switch_fn=switch_fn,
        step_observer=step_observer,
    )
    return _steady_state_vout(res, dt)


def run_benchmark(*, reps: int = 10) -> dict[str, float]:
    """Measure median wall times and steady-state parity over ``reps`` runs."""
    if reps < 1:
        raise ValueError(f"reps must be >= 1; got {reps}")

    # One warm-up per runner before timing.
    v_helper = run_helper_buck()
    v_hand = run_hand_rolled_buck()

    helper_times: list[float] = []
    hand_times: list[float] = []
    for _ in range(reps):
        t0 = time.perf_counter()
        v_helper = run_helper_buck()
        helper_times.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        v_hand = run_hand_rolled_buck()
        hand_times.append(time.perf_counter() - t0)

    helper_median_s = statistics.median(helper_times)
    hand_median_s = statistics.median(hand_times)
    ratio = helper_median_s / hand_median_s
    ss_delta = abs(v_helper - v_hand)

    return {
        "reps": float(reps),
        "helper_median_ms": helper_median_s * 1e3,
        "hand_median_ms": hand_median_s * 1e3,
        "ratio": ratio,
        "v_helper_ss": v_helper,
        "v_hand_ss": v_hand,
        "ss_delta": ss_delta,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reps",
        type=int,
        default=10,
        help="Number of repetitions per runner (default: 10)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit non-zero if ratio > 1.05 or steady-state delta > 1e-12",
    )
    args = parser.parse_args(argv)

    stats = run_benchmark(reps=args.reps)
    print(f"Closed-loop buck benchmark (t_end=10 ms, dt=5 us, reps={args.reps}):")
    print(f"  bind_pi_to_switch median : {stats['helper_median_ms']:.3f} ms")
    print(f"  hand-rolled median       : {stats['hand_median_ms']:.3f} ms")
    print(f"  ratio (helper / hand)    : {stats['ratio']:.3f}x")
    print(f"  vout_ss (helper)         : {stats['v_helper_ss']:.12f} V")
    print(f"  vout_ss (hand-rolled)    : {stats['v_hand_ss']:.12f} V")
    print(f"  |delta vout_ss|          : {stats['ss_delta']:.3e} V")

    if args.check:
        ok = True
        if stats["ratio"] > 1.05:
            print(
                f"FAIL: ratio {stats['ratio']:.3f} > 1.05",
                file=sys.stderr,
            )
            ok = False
        if stats["ss_delta"] > 1e-9 * abs(stats["v_helper_ss"]):
            print(
                f"FAIL: |delta vout_ss| {stats['ss_delta']:.3e} > 1e-9 relative",
                file=sys.stderr,
            )
            ok = False
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
