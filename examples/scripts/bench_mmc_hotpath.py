#!/usr/bin/env python3
"""Benchmark: MMC L0/L1 hot-path Python vs C++ implementations.

The C++ helpers in ``core/include/pulsim/mmc/arm.hpp`` are header-
only and exposed via three private bindings on ``pulsim._pulsim``:

  * ``_cpp_ps_pwm_switching_function``  ← inner N-SM carrier loop
  * ``_cpp_mmc_arm_average_step``       ← L0 single forward-Euler step
  * ``_cpp_mmc_arm_multilevel_step``    ← L1 step (PS-PWM + L0 dynamics)

``pulsim/mmc.py`` dispatches to those at runtime when the kernel
extension is loaded (``_HAS_CPP_MMC = True``). This script measures
the wall-clock speedup on a workload representative of the per-
observer call in a real simulation: thousands of repeated calls per
arm per simulation, per-step dt.
"""

from __future__ import annotations

import math
import time

import pulsim as p
import pulsim.mmc as mmc_mod

# Iteration counts large enough that timer noise is negligible.
N_PS_PWM    = 200_000   # PS-PWM-only loop
N_AVG_STEP  = 200_000   # L0 step
N_ML_STEP   = 200_000   # L1 step


def bench_ps_pwm() -> None:
    print(f"\nps_pwm_switching_function (N={N_PS_PWM} calls, n_sm=8):")
    n_sm = 8
    f_carrier = 1000.0

    # C++
    cpp = mmc_mod._cpp_ps_pwm  # type: ignore[attr-defined]
    t0 = time.perf_counter()
    for k in range(N_PS_PWM):
        cpp(0.5 + 0.4 * math.sin(k * 1e-3),
            k * 1e-5, n_sm, f_carrier, "half_bridge")
    t_cpp = time.perf_counter() - t0

    # Python (force-disable C++ dispatch)
    save = mmc_mod._HAS_CPP_MMC
    mmc_mod._HAS_CPP_MMC = False
    try:
        t0 = time.perf_counter()
        for k in range(N_PS_PWM):
            mmc_mod.ps_pwm_switching_function(
                0.5 + 0.4 * math.sin(k * 1e-3),
                k * 1e-5, n_sm, f_carrier,
                sm_type="half_bridge",
            )
        t_py = time.perf_counter() - t0
    finally:
        mmc_mod._HAS_CPP_MMC = save

    print(f"  Python : {t_py*1e3:7.1f} ms   ({t_py/N_PS_PWM*1e6:5.2f} µs/call)")
    print(f"  C++    : {t_cpp*1e3:7.1f} ms   ({t_cpp/N_PS_PWM*1e6:5.2f} µs/call)")
    print(f"  speedup: {t_py/t_cpp:5.1f}×")


def bench_avg_step() -> None:
    print(f"\nmmc_arm_average_step (N={N_AVG_STEP} calls):")
    params = p.MmcArmAverageParams(n_sm=10, c_sm=1e-3, v_c0=500.0)

    t0 = time.perf_counter()
    v_C = params.v_c0
    for k in range(N_AVG_STEP):
        v_C, _ = mmc_mod.mmc_arm_average_step(v_C, 0.5, 4.0, 1e-6, params)
    t_cpp = time.perf_counter() - t0

    save = mmc_mod._HAS_CPP_MMC
    mmc_mod._HAS_CPP_MMC = False
    try:
        t0 = time.perf_counter()
        v_C = params.v_c0
        for k in range(N_AVG_STEP):
            v_C, _ = mmc_mod.mmc_arm_average_step(v_C, 0.5, 4.0, 1e-6, params)
        t_py = time.perf_counter() - t0
    finally:
        mmc_mod._HAS_CPP_MMC = save

    print(f"  Python : {t_py*1e3:7.1f} ms   ({t_py/N_AVG_STEP*1e6:5.2f} µs/call)")
    print(f"  C++    : {t_cpp*1e3:7.1f} ms   ({t_cpp/N_AVG_STEP*1e6:5.2f} µs/call)")
    print(f"  speedup: {t_py/t_cpp:5.1f}×")


def bench_ml_step() -> None:
    print(f"\nmmc_arm_multilevel_step (N={N_ML_STEP} calls, n_sm=8):")
    params = p.MmcArmMultilevelParams(
        n_sm=8, c_sm=1e-3, v_c0=500.0, f_carrier=1000.0,
    )

    t0 = time.perf_counter()
    v_C = params.v_c0
    for k in range(N_ML_STEP):
        v_C, _, _ = mmc_mod.mmc_arm_multilevel_step(
            v_C, 0.5, 4.0, 1e-6, k * 1e-6, params,
        )
    t_cpp = time.perf_counter() - t0

    save = mmc_mod._HAS_CPP_MMC
    mmc_mod._HAS_CPP_MMC = False
    try:
        t0 = time.perf_counter()
        v_C = params.v_c0
        for k in range(N_ML_STEP):
            v_C, _, _ = mmc_mod.mmc_arm_multilevel_step(
                v_C, 0.5, 4.0, 1e-6, k * 1e-6, params,
            )
        t_py = time.perf_counter() - t0
    finally:
        mmc_mod._HAS_CPP_MMC = save

    print(f"  Python : {t_py*1e3:7.1f} ms   ({t_py/N_ML_STEP*1e6:5.2f} µs/call)")
    print(f"  C++    : {t_cpp*1e3:7.1f} ms   ({t_cpp/N_ML_STEP*1e6:5.2f} µs/call)")
    print(f"  speedup: {t_py/t_cpp:5.1f}×")


def main() -> None:
    print("=" * 64)
    print("MMC hot-path benchmark — pulsim Python vs C++")
    print("=" * 64)
    print(f"_HAS_CPP_MMC = {mmc_mod._HAS_CPP_MMC}")
    bench_ps_pwm()
    bench_avg_step()
    bench_ml_step()


if __name__ == "__main__":
    main()
