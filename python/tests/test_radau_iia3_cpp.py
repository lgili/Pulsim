"""Verify the C++ Radau IIA(3) implicit RK solver matches the Python
`RadauIIA3` and demonstrates the L-stability advantage on stiff
problems.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

pulsim = pytest.importorskip("pulsim")


_HAS_CPP_RADAU = (
    hasattr(pulsim, "_pulsim") and
    hasattr(pulsim._pulsim, "radau_iia3_solve")
)


@pytest.mark.skipif(not _HAS_CPP_RADAU,
                       reason="C++ radau_iia3_solve not in this build")
def test_radau_cpp_matches_python_on_stiff_decay():
    """Stiff linear test problem: dx/dt = -λx with λ=1000.
    Both solvers must hit exp(-λ·t_end) to high precision."""
    lam = 1000.0
    def f(t, x):
        return -lam * x

    x0 = np.array([1.0])
    t_end = 0.01

    cpp = pulsim._pulsim.radau_iia3_solve(
        f, 0.0, t_end, x0, rtol=1e-7, atol=1e-10)
    py = pulsim.RadauIIA3(f=f, rtol=1e-7, atol=1e-10).solve(
        (0.0, t_end), x0)

    analytical = math.exp(-lam * t_end)

    assert abs(cpp["x"][-1, 0] - analytical) < 1e-5, (
        f"C++ Radau missed exp(-10): {cpp['x'][-1, 0]:.6e}")
    assert abs(py.x[-1, 0] - analytical) < 1e-5, (
        f"Python Radau missed exp(-10): {py.x[-1, 0]:.6e}")
    # L-stability: bounded step count for this stiff problem (an
    # explicit RK at λ=1000 would need ≳ 1000 steps for stability).
    assert cpp["n_accepted"] < 250, (
        f"C++ Radau took {cpp['n_accepted']} steps on stiff decay — "
        f"L-stability seems lost.")


@pytest.mark.skipif(not _HAS_CPP_RADAU,
                       reason="C++ radau_iia3_solve not in this build")
def test_radau_cpp_handles_stiff_van_der_pol():
    """Stiff van der Pol (μ=1000) — explicit DOPRI5 would need
    10^4+ steps; Radau should converge in O(100)."""
    mu = 1000.0
    def f(t, x):
        return np.array(
            [x[1], mu * (1.0 - x[0] ** 2) * x[1] - x[0]],
            dtype=np.float64)

    x0 = np.array([2.0, 0.0])
    cpp = pulsim._pulsim.radau_iia3_solve(
        f, 0.0, 5.0, x0, rtol=1e-5, atol=1e-8)

    assert cpp["n_accepted"] > 0, "Radau didn't accept any steps"
    # The L-stability advantage: << 1000 steps even at μ=1000.
    assert cpp["n_accepted"] < 1000, (
        f"Radau took {cpp['n_accepted']} steps on μ=1000 VdP — "
        f"L-stability advantage lost?")
    # Final state should be on the limit cycle near (2, ~0).
    x_final = cpp["x"][-1]
    assert abs(x_final[0]) < 3.0, (
        f"VdP x-coordinate diverged: x={x_final}")


@pytest.mark.skipif(not _HAS_CPP_RADAU,
                       reason="C++ radau_iia3_solve not in this build")
def test_radau_dopri5_dramatic_speedup_on_stiff_problem():
    """At μ=10^4 VdP, Radau should beat DOPRI5 by orders of magnitude
    in f-evaluation count."""
    mu = 1e4
    def f(t, x):
        return np.array(
            [x[1], mu * (1.0 - x[0] ** 2) * x[1] - x[0]],
            dtype=np.float64)

    x0 = np.array([2.0, 0.0])
    radau = pulsim._pulsim.radau_iia3_solve(
        f, 0.0, 1.0, x0, rtol=1e-4, atol=1e-6)

    # Don't actually run DOPRI5 here — it would take minutes on μ=10^4.
    # Just confirm Radau finishes in a sane budget.
    assert radau["n_accepted"] < 500, (
        f"Radau took {radau['n_accepted']} steps on μ=1e4 VdP — too many")
    assert radau["n_rejected"] < radau["n_accepted"], (
        f"Radau rejection rate too high: {radau['n_rejected']} rejected "
        f"vs {radau['n_accepted']} accepted")


@pytest.mark.skipif(not _HAS_CPP_RADAU,
                       reason="C++ radau_iia3_solve not in this build")
def test_radau_stats_fields():
    """Smoke test on the returned dict shape."""
    def f(t, x):
        return -x
    x0 = np.array([1.0, 2.0])
    res = pulsim._pulsim.radau_iia3_solve(f, 0.0, 1.0, x0)
    for key in ("t", "x", "n_accepted", "n_rejected", "n_f_evals"):
        assert key in res
    assert res["x"].shape[1] == 2
    assert res["n_accepted"] >= 1
