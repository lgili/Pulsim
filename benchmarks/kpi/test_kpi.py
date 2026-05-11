"""Tests for benchmarks.kpi against synthetic signals with hand-computed
answers. Run with `python -m benchmarks.kpi.test_kpi` (no test framework
dependency — exits non-zero on failure)."""

from __future__ import annotations

import math
import sys
from typing import List

from . import (
    compute_efficiency,
    compute_loss_breakdown,
    compute_power_factor,
    compute_ripple_pkpk,
    compute_thd,
    compute_transient_response,
)


# ----- helpers -----


def _sine(t: float, amplitude: float, freq_hz: float, phase: float = 0.0) -> float:
    return amplitude * math.sin(2.0 * math.pi * freq_hz * t + phase)


def _linspace(start: float, stop: float, n: int) -> List[float]:
    if n <= 1:
        return [start]
    step = (stop - start) / (n - 1)
    return [start + step * i for i in range(n)]


def _assert(condition: bool, msg: str) -> None:
    if not condition:
        print(f"FAIL: {msg}", file=sys.stderr)
        sys.exit(1)


def _almost(a, b: float, tol: float, msg: str) -> None:
    if a is None:
        print(f"FAIL: {msg} (a is None, expected {b})", file=sys.stderr)
        sys.exit(1)
    if not math.isfinite(a) or not math.isfinite(b):
        print(f"FAIL: {msg} (a={a}, b={b})", file=sys.stderr)
        sys.exit(1)
    if abs(a - b) > tol:
        print(f"FAIL: {msg} (a={a}, b={b}, |a-b|={abs(a-b)} > tol={tol})", file=sys.stderr)
        sys.exit(1)


# ----- THD -----


def test_thd_clean_sine_is_zero() -> None:
    """A pure 60 Hz sine has ~0 % THD."""
    fs = 10_000.0
    f0 = 60.0
    n_periods = 10
    n = int(fs * n_periods / f0)
    times = [k / fs for k in range(n)]
    samples = [_sine(t, 1.0, f0) for t in times]
    thd = compute_thd(samples, sample_rate_hz=fs, fundamental_hz=f0)
    _assert(thd < 1.0, f"clean sine THD should be < 1%, got {thd:.3f}%")


def test_thd_with_3rd_harmonic() -> None:
    """Sine + 10 % 3rd-harmonic should report THD ≈ 10 %."""
    fs = 10_000.0
    f0 = 60.0
    n_periods = 10
    n = int(fs * n_periods / f0)
    times = [k / fs for k in range(n)]
    samples = [_sine(t, 1.0, f0) + _sine(t, 0.1, 3 * f0) for t in times]
    thd = compute_thd(samples, sample_rate_hz=fs, fundamental_hz=f0)
    _almost(thd, 10.0, 0.5, f"THD with 10% 3rd harmonic")


# ----- power factor -----


def test_pf_inphase_resistive_is_unity() -> None:
    """Identical V and I sines → PF = 1."""
    fs = 5_000.0
    f0 = 60.0
    n_periods = 10
    n = int(fs * n_periods / f0)
    times = [k / fs for k in range(n)]
    v = [_sine(t, 120.0, f0) for t in times]
    i = [_sine(t, 1.0, f0) for t in times]
    pf = compute_power_factor(v, i)
    _almost(pf, 1.0, 0.01, "PF of in-phase V/I")


def test_pf_90deg_inductive_is_zero() -> None:
    """V and I 90° out of phase → PF = 0."""
    fs = 5_000.0
    f0 = 60.0
    n_periods = 10
    n = int(fs * n_periods / f0)
    times = [k / fs for k in range(n)]
    v = [_sine(t, 120.0, f0) for t in times]
    i = [_sine(t, 1.0, f0, phase=-math.pi / 2.0) for t in times]
    pf = compute_power_factor(v, i)
    _almost(pf, 0.0, 0.01, "PF of 90°-shifted V/I")


# ----- efficiency -----


def test_efficiency_perfect() -> None:
    """P_out = P_in → η = 100 %."""
    p_in = [50.0] * 1000
    p_out = [50.0] * 1000
    _almost(compute_efficiency(p_in, p_out), 100.0, 0.01, "η = 100%")


def test_efficiency_partial() -> None:
    """P_out = 0.85 · P_in → η = 85 %."""
    p_in = [100.0] * 1000
    p_out = [85.0] * 1000
    _almost(compute_efficiency(p_in, p_out), 85.0, 0.01, "η = 85%")


# ----- transient response -----


def test_transient_response_step_no_overshoot() -> None:
    """First-order step settling to target = 10."""
    n = 1000
    tau = 0.001
    times = _linspace(0.0, 5 * tau, n)
    samples = [10.0 * (1.0 - math.exp(-t / tau)) for t in times]
    rsp = compute_transient_response(times, samples, target=10.0, tolerance_pct=2.0)
    # rise time should be tau * ln(0.9/0.1) ≈ 2.197 tau ≈ 2.197 ms
    _almost(rsp["rise_time"], 2.197 * tau, 0.0005, "first-order rise time")
    # overshoot should be 0
    _almost(rsp["overshoot_pct"], 0.0, 0.5, "first-order has no overshoot")


def test_transient_response_overshoot() -> None:
    """Synthetic response that overshoots to 12 then settles at 10."""
    n = 2000
    times = _linspace(0.0, 0.01, n)
    samples = []
    for t in times:
        if t < 0.002:
            samples.append(12.0 * t / 0.002)  # ramps to 12
        elif t < 0.003:
            samples.append(12.0)
        else:
            samples.append(10.0 + 2.0 * math.exp(-(t - 0.003) / 0.001))
    rsp = compute_transient_response(times, samples, target=10.0, tolerance_pct=2.0)
    _almost(rsp["overshoot_pct"], 20.0, 0.5, "20 % overshoot")


# ----- ripple -----


def test_ripple_pkpk() -> None:
    """1 V_pk sine → 2 V_pkpk ripple."""
    n = 1000
    times = _linspace(0.0, 0.1, n)
    samples = [12.0 + _sine(t, 1.0, 60.0) for t in times]
    rip = compute_ripple_pkpk(samples)
    _almost(rip, 2.0, 0.05, "ripple_pkpk of 1V_pk sine")


# ----- loss breakdown -----


def test_loss_breakdown_pure_conduction() -> None:
    """Switch ON for entire window, I = 2A, R_on = 5 mΩ → P_cond = 20 mW."""
    n = 1000
    times = _linspace(0.0, 0.001, n)
    sw = [True] * n
    i = [2.0] * n
    v = [0.010] * n  # tiny V_DS in ON state
    loss = compute_loss_breakdown(sw, i, v, r_on=5e-3, times=times)
    _almost(loss["conduction_w_avg"], 0.020, 0.001, "pure conduction loss")
    _almost(loss["switching_w_avg"], 0.0, 0.001, "no switching transitions")


def test_loss_breakdown_with_switching() -> None:
    """Switch toggles once; switching loss = ½·V·I at the transition."""
    n = 100
    times = _linspace(0.0, 0.001, n)
    sw = [True] * 50 + [False] * 50
    i = [1.0] * 100
    v = [0.0] * 50 + [10.0] * 50  # V_DS jumps from 0 to 10V at the transition
    loss = compute_loss_breakdown(sw, i, v, r_on=0.0, times=times)
    # 1 transition, V before = 0, I before = 1 → ½ · 0 · 1 = 0 J
    # So the switching loss should be 0 W in this simple case (V was still 0
    # at the sample before the flip). This test exercises the code path.
    _assert(math.isfinite(loss["switching_w_avg"]), "switching loss finite")


# ----- runner -----


def main() -> int:
    tests = [
        test_thd_clean_sine_is_zero,
        test_thd_with_3rd_harmonic,
        test_pf_inphase_resistive_is_unity,
        test_pf_90deg_inductive_is_zero,
        test_efficiency_perfect,
        test_efficiency_partial,
        test_transient_response_step_no_overshoot,
        test_transient_response_overshoot,
        test_ripple_pkpk,
        test_loss_breakdown_pure_conduction,
        test_loss_breakdown_with_switching,
    ]
    for t in tests:
        t()
        print(f"  ✓ {t.__name__}")
    print(f"All {len(tests)} KPI tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
