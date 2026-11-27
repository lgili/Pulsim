"""The floating-point fixed point acceptance is trbdf2/esdirk43 only.

Newton's own-scale acceptance (DECISIONS 12 of add-newton-roundoff-
acceptance, added for ll13 on trbdf2) is gated by
``NewtonRoundoffAcceptance::judge_own_scale``: the variable-step engine
sets it, the fixed-step engine leaves it off and keeps 2.1.0's Newton
path (C++: test_newton_roundoff_acceptance.cpp, the fixed-point case).

The circuit reaches the stall on pwl: 40 A in a 1 H inductor forced
through a Shockley diode into 10 MOhm. The first step starts cold
(v = 0) and lands at 4e8 V, where ||dx|| sits below one ulp (6e-8 V) but
above tol_dx (1e-9 V) and the diode's KCL row is round-off of its own
magnitudes, inf eps of the cold start's. Measured on PyPI 2.1.0: dt = 1 ns
retries the first step at dt/2 and completes; dt = 1 ps aborts with
"Roundoff acceptance declined". Without the gate both completed with no
retry (the stalled iterate accepted at its own magnitudes).
"""
import warnings

import pytest

import pulsim as p


def _circuit():
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "in", "gnd", 10.0)
    b.add_inductor("L1", "in", "a", 1.0, 40.0)
    b.add_shockley_diode("D", "a", "b")
    b.add_resistor("Rb", "b", "gnd", 1e7)
    return b


@pytest.mark.parametrize("dt", [1e-9, 1e-12])
def test_pwl_does_not_accept_a_fixed_point_at_its_own_scale(dt):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            r = p.simulate(_circuit(), t_end=30 * dt, dt=dt, engine="pwl")
        except p.SimulationAborted as e:
            # 2.1.0's outcome at 1 ps: declined at the start's scale
            assert "Roundoff acceptance declined" in str(e)
            return
    # 2.1.0's outcome at 1 ns: the stalled first step is retried at dt/2,
    # never accepted as it stands
    assert len(r.dt_retries) >= 1
    assert r.dt_retries[0].t <= dt   # the first step
    assert max(abs(v) for v in r.v("a")) > 3e8
