#!/usr/bin/env python3
"""Benchmark: per-step cost of the MMC L0/L1/L2/L3 layers.

Since Phase 20.12 every MMC step function in ``pulsim.mmc`` is a
thin dispatch into the C++ kernel
(``core/include/pulsim/mmc/arm.hpp``). This script measures the
wall-clock cost of each layer's step function on a workload sized
to a realistic per-arm simulation.

Numbers reported are µs/call; multiply by (n_arms × n_steps) to
estimate the per-simulation contribution.
"""

from __future__ import annotations

import math
import time

import numpy as np

import pulsim as p
import pulsim.mmc as mmc_mod


# Iteration counts large enough that timer noise is negligible.
N_PS_PWM    = 200_000
N_AVG_STEP  = 200_000
N_ML_STEP   = 200_000
N_EQ_STEP   = 200_000
N_DT_STEP   =  50_000     # L3 sorts per call, so 50k is plenty.


def bench(name: str, n: int, fn) -> None:
    # Warm-up to remove first-call effects (e.g., JIT-warmed numpy
    # routines or cold-cache pages on the array buffers).
    fn(min(2000, n))
    t0 = time.perf_counter()
    fn(n)
    elapsed = time.perf_counter() - t0
    print(f"  {name:38s} {elapsed*1e3:7.1f} ms  "
          f"({elapsed/n*1e6:5.2f} µs/call)")


def bench_ps_pwm(n: int) -> None:
    cpp = mmc_mod._cpp_ps_pwm  # type: ignore[attr-defined]
    for k in range(n):
        cpp(0.5 + 0.4 * math.sin(k * 1e-3),
            k * 1e-5, 8, 1000.0, "half_bridge")


def bench_avg_step(n: int) -> None:
    params = p.MmcArmAverageParams(n_sm=10, c_sm=1e-3, v_c0=500.0)
    v_C = params.v_c0
    for _ in range(n):
        v_C, _ = mmc_mod.mmc_arm_average_step(v_C, 0.5, 4.0, 1e-6, params)


def bench_ml_step(n: int) -> None:
    params = p.MmcArmMultilevelParams(
        n_sm=8, c_sm=1e-3, v_c0=500.0, f_carrier=1000.0,
    )
    v_C = params.v_c0
    for k in range(n):
        v_C, _, _ = mmc_mod.mmc_arm_multilevel_step(
            v_C, 0.5, 4.0, 1e-6, k * 1e-6, params,
        )


def bench_eq_step(n: int) -> None:
    params = p.MmcArmEquivalentParams(
        n_sm=8, c_sm=1e-3, v_c0=500.0,
        f_carrier=1000.0, t_dead=20e-6, t_min=0.0,
    )
    state = p.make_l2_state(params)
    for k in range(n):
        mmc_mod.mmc_arm_equivalent_step(
            state, 0.5, 4.0, 1e-6, k * 1e-6, params,
        )


def bench_dt_step(n: int) -> None:
    params = p.MmcArmDetailedParams(
        n_sm=8, c_sm=1e-3, v_c0=500.0,
        f_carrier=1000.0, balancing="sort_and_select",
    )
    state = p.make_l3_state(params)
    for k in range(n):
        mmc_mod.mmc_arm_detailed_step(
            state, 0.5, 4.0, 1e-6, k * 1e-6, params,
        )


def main() -> None:
    print("=" * 64)
    print("MMC per-step benchmark — all layers run in C++")
    print("=" * 64)
    print()
    print("Layer                                 Wall time   Per call")
    print("-" * 64)
    bench("L0:  ps_pwm_switching_function (N=8)", N_PS_PWM, bench_ps_pwm)
    bench("L0:  mmc_arm_average_step",            N_AVG_STEP, bench_avg_step)
    bench("L1:  mmc_arm_multilevel_step (N=8)",   N_ML_STEP,  bench_ml_step)
    bench("L2:  mmc_arm_equivalent_step (N=8)",   N_EQ_STEP,  bench_eq_step)
    bench("L3:  mmc_arm_detailed_step (N=8)",     N_DT_STEP,  bench_dt_step)
    print()
    print("Note: per-arm step rate scales linearly with N_SM for L0/L1, "
          "and with N·log(N) for L3 (sort-and-select).")


if __name__ == "__main__":
    main()
