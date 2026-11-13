"""Newton acceptance at roundoff level (add-newton-roundoff-acceptance).

Two series inductors and a Shockley diode on the fixed engine (core
review numerics/p7b_roundoff.py). At small dt the companion
conductance 2L/dt makes one ulp of the inductor current a residual far
above tol_res, so ||dx|| and the residual sit on powers of two and the
solve aborted although the iterate was exact to roundoff. A solve that
ends at the budget is now accepted when its row-wise backward error is
at most 8 eps, ||dx|| stagnated, and ||dx|| <= 1e-6 |x|.
"""

import warnings

import numpy as np
import pytest

import pulsim as p


def _run(dt, L, n=200):
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "in", "gnd", 10.0)
    b.add_resistor("R", "in", "a", 10.0)
    b.add_inductor("L1", "a", "m", L, 0.9)
    b.add_inductor("L2", "m", "d", L, 0.9)
    b.add_shockley_diode("D", "d", "gnd")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        r = p.simulate(b, t_end=n * dt, dt=dt, engine="pwl",
                       max_dt_halvings=0)
    return r, [str(w.message) for w in caught]


# (L, dt, needs the acceptance today). Measured before this change: 100 uH
# aborts at 1e-11, 1e-12, 1e-13 s (||dx|| 1.863e-9, 1.490e-8, 1.192e-7);
# 10 mH aborts at 1e-9 ... 1e-11 s. On the in-house LU of the Newton
# workspace (update-newton-persistent-solver A2, integration batch 5)
# 100 uH at 1e-11 s converges without a stall (418 iterations, no
# acceptance); with Eigen::SparseLU it aborted. The stall is roundoff
# luck; the other cases still need the acceptance.
CASES = [(100e-6, 1e-10, False), (100e-6, 1e-11, False),
         (100e-6, 1e-12, True), (100e-6, 1e-13, True),
         (10e-3, 1e-8, False), (10e-3, 1e-9, True), (10e-3, 1e-10, True),
         (10e-3, 1e-11, True), (10e-3, 1e-12, False)]


@pytest.mark.parametrize("L, dt, needs_acceptance", CASES)
def test_roundoff_stall_completes_and_is_reported(L, dt, needs_acceptance):
    r, msgs = _run(dt, L)
    n_acc = r.newton_roundoff_accepts
    assert (n_acc > 0) == needs_acceptance
    if needs_acceptance:
        assert r.newton_roundoff_omega_max <= 8 * np.finfo(float).eps
        assert any("accepted" in m and "8 eps" in m for m in msgs), msgs
    # The answer is the physics, not noise: over 200 steps i(t) - i0 is
    # linear in t_end/L, so (i_end - 0.9) * L / dt is one constant.
    # Measured: 2.8754e-5 at L = 10 mH, dt = 1e-8 (no acceptance).
    k = (np.asarray(r.i("L1"))[-1] - 0.9) * L / dt
    assert k == pytest.approx(2.8754e-5 * 1e-2 / 1e-8, rel=1e-3)


def test_a_run_that_converged_is_untouched():
    """Acceptance is evaluated only where the solve used to throw: a
    run with no stall records no acceptance (and so takes the path it
    took before, bit for bit)."""
    r, _ = _run(1e-8, 10e-3)
    assert r.newton_roundoff_accepts == 0
    assert r.newton_roundoff_omega_max == 0.0
