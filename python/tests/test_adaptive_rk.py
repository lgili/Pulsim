"""Tests for the adaptive Runge-Kutta integrators (Phase 2.4).

Validates :class:`DormandPrince5` and :class:`RadauIIA3` against
analytical solutions and against each other on a stiff problem.
Both run in pure Python — no C++ extension required.
"""
from __future__ import annotations

import math

import numpy as np
import pytest


pulsim = pytest.importorskip("pulsim")
_HAS_RK = hasattr(pulsim, "DormandPrince5")


# -----------------------------------------------------------------------
# Construction
# -----------------------------------------------------------------------

def test_dopri5_constructor_defaults():
    if not _HAS_RK:
        pytest.skip()
    solver = pulsim.DormandPrince5(f=lambda t, x: -x)
    assert solver.rtol == pytest.approx(1e-5)
    assert solver.atol == pytest.approx(1e-8)
    assert solver.safety == pytest.approx(0.9)


def test_radau_constructor_defaults():
    if not _HAS_RK:
        pytest.skip()
    solver = pulsim.RadauIIA3(f=lambda t, x: -x)
    assert solver.rtol == pytest.approx(1e-5)
    assert solver.newton_max_iter == 8


# -----------------------------------------------------------------------
# Analytical accuracy
# -----------------------------------------------------------------------

@pytest.mark.skipif(not _HAS_RK, reason="adaptive RK not available")
def test_dopri5_exponential_decay():
    """dx/dt = -x, x(0)=1 should give exp(-t) within 1e-7."""
    solver = pulsim.DormandPrince5(
        f=lambda t, x: -x, rtol=1e-8, atol=1e-10)
    res = solver.solve(t_span=(0.0, 1.0), x0=np.array([1.0]))
    err = abs(res.x[-1, 0] - math.exp(-1.0))
    assert err < 1e-7, f"DOPRI5 exp(-1) error {err:.2e}"
    assert res.n_accepted > 0
    assert res.n_rejected < res.n_accepted   # mostly accepts


@pytest.mark.skipif(not _HAS_RK, reason="adaptive RK not available")
def test_radau_exponential_decay():
    """Same test for Radau IIA(3) — wider tol because step-doubling
    error estimate is coarser than DOPRI5's embedded pair."""
    solver = pulsim.RadauIIA3(
        f=lambda t, x: -x, rtol=1e-6, atol=1e-9)
    res = solver.solve(t_span=(0.0, 1.0), x0=np.array([1.0]))
    err = abs(res.x[-1, 0] - math.exp(-1.0))
    assert err < 1e-4, f"Radau IIA(3) exp(-1) error {err:.2e}"


@pytest.mark.skipif(not _HAS_RK, reason="adaptive RK not available")
def test_dopri5_simple_harmonic_oscillator():
    """dx/dt = [x[1], -x[0]] — period 2π. After one period the state
    should return to the initial condition within tolerance."""
    def sho(t, x):
        return np.array([x[1], -x[0]])

    solver = pulsim.DormandPrince5(f=sho, rtol=1e-9, atol=1e-12)
    res = solver.solve(t_span=(0.0, 2 * math.pi), x0=np.array([1.0, 0.0]))
    err = np.linalg.norm(res.x[-1] - np.array([1.0, 0.0]))
    assert err < 1e-6, (
        f"SHO after 1 period: state error {err:.2e} (final {res.x[-1]})")


@pytest.mark.skipif(not _HAS_RK, reason="adaptive RK not available")
def test_dopri5_order_of_convergence():
    """Refining rtol by 1e3 should drop the error by ~1e3·(5/4) (DOPRI is
    5th-order; controller is 4th-order in step size, so error ~ rtol)."""
    def sho(t, x):
        return np.array([x[1], -x[0]])

    errs = []
    for rt in (1e-4, 1e-6, 1e-8):
        s = pulsim.DormandPrince5(f=sho, rtol=rt, atol=rt * 1e-3)
        r = s.solve(t_span=(0.0, 2 * math.pi), x0=np.array([1.0, 0.0]))
        errs.append(np.linalg.norm(r.x[-1] - np.array([1.0, 0.0])))
    # Each refinement should reduce error by at least 10× (loose
    # check; controller targets ~rtol, so the ratio is 100× nominally).
    assert errs[1] < errs[0] * 0.1, (
        f"DOPRI5 didn't improve: {errs}")
    assert errs[2] < errs[1] * 0.1


# -----------------------------------------------------------------------
# Stiff systems — where Radau shines
# -----------------------------------------------------------------------

@pytest.mark.skipif(not _HAS_RK, reason="adaptive RK not available")
def test_radau_beats_dopri5_on_stiff_problem():
    """Van der Pol oscillator (mu=100): Radau should use SIGNIFICANTLY
    fewer steps than DOPRI5 because of L-stability."""
    def vdp(t, x):
        return np.array([x[1], 100.0 * (1.0 - x[0] ** 2) * x[1] - x[0]])

    dopri = pulsim.DormandPrince5(f=vdp, rtol=1e-4, atol=1e-7)
    r_d = dopri.solve(t_span=(0.0, 0.5), x0=np.array([2.0, 0.0]))

    radau = pulsim.RadauIIA3(f=vdp, rtol=1e-4, atol=1e-7)
    r_r = radau.solve(t_span=(0.0, 0.5), x0=np.array([2.0, 0.0]))

    # Final states must agree (both are correct).
    err = np.linalg.norm(r_d.x[-1] - r_r.x[-1])
    assert err < 0.05, (
        f"DOPRI5 final={r_d.x[-1]} vs Radau final={r_r.x[-1]} "
        f"(err={err:.3e}) — solutions disagree, integration bug.")
    # Radau uses fewer steps (the headline win — 2x or more on this
    # problem).
    assert r_r.n_accepted < 0.5 * r_d.n_accepted, (
        f"Radau supposed to use far fewer steps on stiff systems: "
        f"DOPRI5 used {r_d.n_accepted}, Radau used {r_r.n_accepted}.")


# -----------------------------------------------------------------------
# AdaptiveSolution shape
# -----------------------------------------------------------------------

@pytest.mark.skipif(not _HAS_RK, reason="adaptive RK not available")
def test_solution_shape():
    solver = pulsim.DormandPrince5(f=lambda t, x: -x)
    res = solver.solve(t_span=(0.0, 1.0), x0=np.array([1.0, 2.0]))
    assert isinstance(res, pulsim.AdaptiveSolution)
    assert res.t.ndim == 1
    assert res.x.ndim == 2
    assert res.x.shape[0] == res.t.size
    assert res.x.shape[1] == 2
    assert res.num_steps() == res.t.size
    assert res.t[0] == 0.0
    assert res.t[-1] == pytest.approx(1.0)
    assert res.n_accepted + res.n_rejected > 0
    assert len(res.dt_history) == res.n_accepted


@pytest.mark.skipif(not _HAS_RK, reason="adaptive RK not available")
def test_invalid_tspan_raises():
    solver = pulsim.DormandPrince5(f=lambda t, x: -x)
    with pytest.raises(ValueError):
        solver.solve(t_span=(1.0, 0.5), x0=np.array([1.0]))
