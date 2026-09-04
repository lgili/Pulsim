"""Builder initial conditions must reach whichever engine runs.

`c0=` / `i0=` / `set_initial` are recorded on the builder, and
`simulate()` synthesises them into an `initial_state` vector when the
caller did not pass one. That synthesis used to sit ~200 lines into
the PWL tail — PAST both early-return branches. So:

    b.add_inductor("L", "a", "gnd", 1e-3, i0=5.0)
    simulate(b, ..., engine="trbdf2")   # started at 0 A

Nothing raised. The solve converged, and the trace was a perfectly
ordinary energisation exponential from the WRONG state — the same
circuit answering differently depending on which engine the router
happened to pick, which is the worst shape a defect can take. It was
found while verifying the TR-BDF2 second stage: a convergence study
against a fixed-step reference showed ~100% error that did not
shrink with h, because the two engines were not solving the same
problem.

The synthesis now runs before the dispatch. It still has to run
AFTER the preflight, which may insert regularizing ties and change
the state vector's length — `test_preflight_ties_do_not_desync...`
below pins that ordering, since getting it backwards sizes the
vector for a different circuit.
"""

import numpy as np
import pytest

import pulsim as p

TAU = 1e-3  # R = 1 Ω, L = 1 mH


def _rl(engine, **kw):
    b = p.CircuitBuilder()
    b.add_resistor("R", "a", "gnd", 1.0)
    b.add_inductor("L", "a", "gnd", 1e-3, i0=5.0)
    res = p.simulate(b, t_end=5 * TAU, engine=engine, **kw)
    i = np.asarray(res.i("L"))
    return float(i[0]), float(i[-1])


@pytest.mark.parametrize("engine,kw", [
    ("pwl", dict(dt=1e-6)),
    ("trbdf2", dict(dt=1e-5, rtol=1e-10, atol=1e-12)),
])
def test_an_inductor_starts_at_its_declared_current(engine, kw):
    i0, i_end = _rl(engine, **kw)
    assert i0 == pytest.approx(5.0, rel=1e-6)
    # Free decay through R: 5·e^-5 after five time constants.
    assert i_end == pytest.approx(5.0 * np.exp(-5.0), rel=1e-3)


def test_the_two_engines_agree_on_the_same_circuit():
    """The regression in one line: whatever the answer is, it must
    not depend on which engine the router picked. Before the fix the
    two differed by 100 % — one decayed from 5 A, the other from 0.

    The 1e-3 tolerance is not slack, it is the FIXED engine's own
    error: measured against the analytic 5·e^-5 on this circuit it
    is off by exactly dt/(2·tau) and falls only first order with dt
    (2.0e-3 / 1.0e-3 / 5.0e-4 / 2.5e-4 at dt = 4e-6 … 5e-7), the
    signature of a backward-Euler first step. TR-BDF2 hits the same
    analytic value to 5e-7 here. That first-order floor on the fixed
    engine is a separate, pre-existing finding; this test is pinned
    loose enough to pass over it rather than silently encoding it as
    correct."""
    _, pwl_end = _rl("pwl", dt=1e-6)
    _, var_end = _rl("trbdf2", dt=1e-5, rtol=1e-10, atol=1e-12)
    assert var_end == pytest.approx(pwl_end, rel=1e-3)
    # And the variable engine is the one on the analytic value.
    assert var_end == pytest.approx(5.0 * np.exp(-5.0), rel=1e-5)


def test_a_capacitor_starts_at_its_declared_voltage():
    def run(engine, **kw):
        b = p.CircuitBuilder()
        b.add_resistor("R", "a", "gnd", 1e3)
        b.add_capacitor("C", "a", "gnd", 1e-6, c0=7.0)
        res = p.simulate(b, t_end=5e-3, engine=engine, **kw)
        v = np.asarray(res.v("a"))
        return float(v[0]), float(v[-1])

    for engine, kw in (("pwl", dict(dt=1e-6)),
                       ("trbdf2", dict(dt=1e-5, rtol=1e-10, atol=1e-12))):
        v0, v_end = run(engine, **kw)
        assert v0 == pytest.approx(7.0, rel=1e-6), engine
        assert v_end == pytest.approx(7.0 * np.exp(-5.0), rel=1e-3), engine


def test_an_explicit_initial_state_still_wins():
    """The synthesis only fills in when the caller passed nothing."""
    b = p.CircuitBuilder()
    b.add_resistor("R", "a", "gnd", 1.0)
    b.add_inductor("L", "a", "gnd", 1e-3, i0=5.0)
    n = len(np.asarray(b.initial_state()))
    res = p.simulate(b, t_end=TAU, dt=1e-6, engine="pwl",
                     initial_state=np.zeros(n))
    assert float(np.asarray(res.i("L"))[0]) == pytest.approx(0.0, abs=1e-12)


def test_preflight_ties_do_not_desync_the_state_vector():
    """A circuit whose preflight inserts a bleeder: the synthesised
    vector has to be sized for the REGULARIZED circuit. Ask the
    builder too early and the length is wrong — which is why the
    synthesis sits after the preflight and not at the top of
    simulate()."""
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "a", "gnd", 10.0)
    b.add_capacitor("Cin", "a", "m", 1e-6)
    b.add_capacitor("Cout", "m", "gnd", 1e-6, c0=2.0)
    with pytest.warns(UserWarning, match="regulariz"):
        res = p.simulate(b, t_end=1e-4, dt=1e-7, engine="pwl")
    # It ran, and the declared 2 V is there — not silently dropped
    # because the vector was sized for the pre-tie circuit.
    assert float(np.asarray(res.v("m"))[0]) == pytest.approx(2.0, rel=1e-3)


def test_dsed_refuses_builder_ics_by_name():
    """DSED solves a REDUCED state (one entry per reactive element),
    not the full-MNA vector the builder synthesises, and for a
    floating capacitor that reduced state is the congruence-
    transformed difference rather than any single MNA row. So the
    vector cannot simply be forwarded. Until that projection exists,
    the run is refused BY NAME — it used to start every state at
    zero and return a plausible transient instead."""
    b = p.CircuitBuilder()
    b.add_resistor("R", "a", "gnd", 1.0)
    b.add_inductor("L", "a", "gnd", 1e-3, i0=5.0)
    with pytest.raises(ValueError, match="initial conditions"):
        p.simulate(b, t_end=TAU, dt=1e-6, engine="dsed")


def test_dsed_still_runs_when_no_ics_were_declared():
    """The refusal must be scoped to circuits that actually declare
    ICs — not a blanket block on the engine."""
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "a", "gnd", 1.0)
    b.add_resistor("R", "a", "m", 1.0)
    b.add_inductor("L", "m", "gnd", 1e-3)
    res = p.simulate(b, t_end=5 * TAU, dt=1e-6, engine="dsed")
    assert float(np.asarray(res.i("L"))[-1]) == pytest.approx(1.0, rel=1e-2)
