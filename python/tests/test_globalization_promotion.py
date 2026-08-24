"""v2.0 Phase 2: Newton promotes line search when it diverges.

The kernel already auto-promotes Levenberg-Marquardt, on two
triggers: a singular factorize, and a near-miss stall (residual
already tiny, ||dx|| plateaued). Neither sees the one condition
backtracking exists for — a full Newton step that makes the residual
WORSE — so a plainly diverging Newton fell through both and the run
died.

Promoting rather than defaulting line search on is deliberate.
Measured on this rectifier, backtracking from the first iteration
costs ~30% on a run that never needed it; the comparison that
triggers the promotion costs nothing.

WHAT IT IS WORTH, AND WHERE IT SITS. Over a 27-point sweep of
(peak voltage, sharpness, dt), with the step ladder disabled:

    logistic fix alone .................. 11 / 27 fail
    + line-search promotion .............  4 / 27 fail
    + dt-halving retry ..................  0 / 27 fail

Each layer is cheaper than the one above it and is measured on what
the one below leaves behind.
"""

import warnings

import numpy as np
import pytest

import pulsim as p


def rectifier(vpk, kappa):
    b = p.CircuitBuilder()
    b.add_sine_voltage_source("Vac", "ac", "gnd", 0.0, vpk, 60.0)
    q = p.IdealDiodeParams()
    q.kappa = kappa
    b.add_nonlinear_diode("D", "ac", "vout", q)
    b.add_resistor("R", "vout", "gnd", 50.0)
    b.add_capacitor("C", "vout", "gnd", 100e-6)
    return b


def _run(vpk, kappa, dt, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return p.simulate(rectifier(vpk, kappa), t_end=1.7e-2, dt=dt,
                           enable_nonlinear_refresh=True,
                           max_dt_halvings=0, **kw)


def test_a_diverging_newton_recovers_without_being_asked():
    """170 V at dt = 1e-4 diverges under plain Newton. Nothing is
    configured; the loop notices the residual grew and starts
    backtracking."""
    res = _run(170.0, 20.0, 1e-4)
    v = np.asarray(res.v("vout"))
    assert np.isfinite(v).all()
    assert v.max() == pytest.approx(169.3, abs=1.0)


def test_the_promotion_does_not_change_a_converged_answer():
    """Line search changes the PATH to the root, not the root. Asking
    for it explicitly must land in the same place as letting the loop
    decide."""
    auto = np.asarray(_run(170.0, 20.0, 1e-5).states)
    forced = np.asarray(
        _run(170.0, 20.0, 1e-5, enable_newton_line_search=True).states)
    assert np.max(np.abs(auto - forced)) < 1e-9


def test_the_ladder_composes_over_the_whole_sweep():
    """The Phase-2 gate: every one of these converges with nothing
    asked of the user. With the step ladder disabled a few still
    fail, and those are exactly what the ladder is for."""
    grid = [(vpk, k, dt)
            for vpk in (24.0, 170.0, 400.0)
            for k in (20.0, 60.0, 100.0)
            for dt in (2e-4, 1e-4, 5e-5)]
    assert len(grid) == 27

    without_ladder = []
    for vpk, k, dt in grid:
        try:
            _run(vpk, k, dt)
        except RuntimeError:
            without_ladder.append((vpk, k, dt))
    # A handful survive the promotion — the ladder's residual band.
    assert 0 < len(without_ladder) <= 6, without_ladder

    for vpk, k, dt in grid:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = p.simulate(rectifier(vpk, k), t_end=1.7e-2, dt=dt,
                              enable_nonlinear_refresh=True)
        v = np.asarray(res.v("vout"))
        assert np.isfinite(v).all(), (vpk, k, dt)
        assert v.max() == pytest.approx(vpk - 0.7, rel=0.03), (vpk, k, dt)
