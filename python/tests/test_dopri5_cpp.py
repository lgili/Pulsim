"""Verify the C++ Dormand-Prince 5(4) solver
(`pulsim._pulsim.dopri5_solve`) matches the Python `DormandPrince5`
on canonical ODE benchmarks.

The C++ implementation at
`core/include/pulsim/integrators/dormand_prince5.hpp` is a header-only
template that mirrors the Python class step-for-step. Both should
produce identical trajectories (to the same tolerance) on the same RHS.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

pulsim = pytest.importorskip("pulsim")


_HAS_CPP_DOPRI5 = (
    hasattr(pulsim, "_pulsim") and
    hasattr(pulsim._pulsim, "dopri5_solve")
)


@pytest.mark.skipif(not _HAS_CPP_DOPRI5,
                       reason="C++ dopri5_solve not in this build")
def test_dopri5_cpp_matches_python_on_linear_decay():
    """Solve dx/dt = −x, x(0)=1 from t=0 to 2. Analytical: x(t)=e^(−t).
    Both the C++ and Python solvers must hit e^(−2) ≈ 0.1353 to 1e-6."""
    def f(t, x):
        return np.array([-x[0]], dtype=np.float64)

    x0 = np.array([1.0])

    cpp = pulsim._pulsim.dopri5_solve(
        f, 0.0, 2.0, x0, rtol=1e-8, atol=1e-10)
    py = pulsim.DormandPrince5(f=f, rtol=1e-8, atol=1e-10).solve(
        (0.0, 2.0), x0)

    analytical = math.exp(-2.0)

    assert abs(cpp["x"][-1, 0] - analytical) < 1e-6, (
        f"C++ DOPRI5 missed exp(-2): {cpp['x'][-1, 0]:.10f}")
    assert abs(py.x[-1, 0] - analytical) < 1e-6, (
        f"Python DOPRI5 missed exp(-2): {py.x[-1, 0]:.10f}")

    # The two solvers should also agree with each other.
    assert abs(cpp["x"][-1, 0] - py.x[-1, 0]) < 1e-8, (
        f"C++ vs Python disagree: cpp={cpp['x'][-1, 0]:.10f}, "
        f"py={py.x[-1, 0]:.10f}")


@pytest.mark.skipif(not _HAS_CPP_DOPRI5,
                       reason="C++ dopri5_solve not in this build")
def test_dopri5_cpp_matches_python_on_van_der_pol():
    """Van der Pol oscillator (μ=10) — stiff-ish nonlinear classic.
    Compare C++ vs Python final states after one limit-cycle interval."""
    mu = 10.0

    def f(t, x):
        return np.array(
            [x[1], mu * (1.0 - x[0] ** 2) * x[1] - x[0]],
            dtype=np.float64)

    x0 = np.array([2.0, 0.0])
    t_end = 5.0

    cpp = pulsim._pulsim.dopri5_solve(
        f, 0.0, t_end, x0, rtol=1e-7, atol=1e-9)
    py = pulsim.DormandPrince5(f=f, rtol=1e-7, atol=1e-9).solve(
        (0.0, t_end), x0)

    # Final state should agree within RMS error proportional to the tol.
    err = np.linalg.norm(cpp["x"][-1] - py.x[-1])
    norm = np.linalg.norm(py.x[-1])
    rel = err / max(norm, 1e-12)

    # Different step paths may diverge slightly on a stiff problem; 1%
    # is well within the achievable accuracy for matched tolerances.
    assert rel < 0.01, (
        f"vdP final-state mismatch: cpp={cpp['x'][-1]}, py={py.x[-1]}, "
        f"rel err={rel:.4e}")

    # Step accounting should also be in the same ballpark.
    assert cpp["n_accepted"] > 50, (
        f"C++ DOPRI5 took only {cpp['n_accepted']} accepted steps "
        f"on van der Pol — likely degenerate.")
    assert cpp["n_f_evals"] >= cpp["n_accepted"] * 6


@pytest.mark.skipif(not _HAS_CPP_DOPRI5,
                       reason="C++ dopri5_solve not in this build")
def test_dopri5_cpp_stats_fields():
    """Smoke test on the returned dict shape."""
    def f(t, x):
        return -x

    x0 = np.array([1.0, 2.0, 3.0])
    res = pulsim._pulsim.dopri5_solve(f, 0.0, 1.0, x0)
    for key in ("t", "x", "n_accepted", "n_rejected", "n_f_evals"):
        assert key in res, f"missing key {key!r} in result dict"
    assert res["t"].ndim == 1
    assert res["x"].ndim == 2
    assert res["x"].shape[1] == 3
    assert res["x"].shape[0] == res["t"].shape[0]
    assert res["n_accepted"] >= 1
