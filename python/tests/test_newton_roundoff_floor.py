"""Newton acceptance at roundoff level (add-newton-roundoff-acceptance).

Two series inductors and a Shockley diode on the fixed engine (core
review numerics/p7b_roundoff.py). At small dt the companion
conductance 2L/dt makes one ulp of the inductor current a residual far
above tol_res, so ||dx|| and the residual sit on powers of two and the
solve aborted although the iterate was exact to roundoff. A solve that
ends at the budget is now accepted when its row-wise backward error is
at most 8 eps, ||dx|| stagnated, and ||dx|| <= 1e-6 |x|.
"""

import math
import warnings

import numpy as np
import pytest
from scipy.optimize import brentq

import pulsim as p


VT = 0.025852                       # add_shockley_diode's default V_T (I_S 1e-12, n 1, G_min 1e-12)


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
# luck; the other cases still need the acceptance. Release defaults (the
# restart rule's SDIRK start, device_path='interface'): 100 uH at 1e-11 s
# stalls again (7 acceptances) — the same roundoff luck, the other way.
CASES = [(100e-6, 1e-10, False), (100e-6, 1e-11, True),
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
    # The answer is the physics, not noise. Truth: the ODE linearised at
    # i0 = 0.9 A (the change over 200 steps is <= 3e-5 relative), so
    # (i_end - 0.9) L / dt = 100 (1 - v_D) (1 - e^-x) / x, x = t_end / tau.
    # The restart-less fixed step was 0.25 % low (28.754 vs 28.826: its
    # trapezoidal start loses half a step); the restart rule's start is
    # exact. Measured with the rule: |rel| <= 4.5e-6 (roundoff at 1e-12 s).
    v_d = brentq(lambda v: 1e-12 * math.expm1(v / VT) + 1e-12 * v - 0.9, 0.0, 2.0,
                 xtol=1e-15)
    r_d = 1 / (1e-12 * math.exp(v_d / VT) / VT + 1e-12)
    x = 100 * (dt / L) * (10.0 + r_d)
    k = (np.asarray(r.i("L1"))[-1] - 0.9) * L / dt
    assert k == pytest.approx(100 * (1 - v_d) * -math.expm1(-x) / x, rel=1e-5)


def test_a_run_that_converged_is_untouched():
    """Acceptance is evaluated only where the solve used to throw: a
    run with no stall records no acceptance (and so takes the path it
    took before, bit for bit)."""
    r, _ = _run(1e-8, 10e-3)
    assert r.newton_roundoff_accepts == 0
    assert r.newton_roundoff_omega_max == 0.0


def test_the_run_budget_is_an_option_and_its_abort_is_typed():
    """100 uH at dt = 1e-13 s accepts every step at roundoff. With the
    budget at 2 the third consecutive acceptance raises the typed
    NewtonRoundoffBudgetExceeded, a SimulationAborted carrying the
    partial result; the default (130) is not reached in 50 steps."""
    assert issubclass(p.NewtonRoundoffBudgetExceeded, p.SimulationAborted)
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "in", "gnd", 10.0)
    b.add_resistor("R", "in", "a", 10.0)
    b.add_inductor("L1", "a", "m", 100e-6, 0.9)
    b.add_inductor("L2", "m", "d", 100e-6, 0.9)
    b.add_shockley_diode("D", "d", "gnd")
    kw = dict(t_end=50e-13, dt=1e-13, engine="pwl", max_dt_halvings=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert p.simulate(b, **kw).newton_roundoff_accepts > 2
        with pytest.raises(p.NewtonRoundoffBudgetExceeded,
                           match="3 consecutive") as ei:
            p.simulate(b, newton_roundoff_budget=2, **kw)
    assert ei.value.partial is not None
    with pytest.raises(ValueError, match="newton_roundoff_budget"):
        p.simulate(b, newton_roundoff_budget=0, **kw)
