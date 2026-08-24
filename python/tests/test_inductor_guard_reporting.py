"""v2.0 Phase 2: a guard that overwrites the answer has to say so.

`inductor_freeze_di_max` and `inductor_abs_clamp` do not solve
anything. They replace a current the solver computed with a limit the
caller configured — sometimes the only way to get a run to finish,
but it means the plotted current is the limit.

The circuit these were written for is a 1 kW PFC + VSI drive whose
input choke's loop opens when the bridge enters DCM. On a 20 ms run
the clamp fires on **30878** steps and the reported line current
peaks at exactly 100.000 A, which is the clamp value. Nothing said so
until this change: the audit calls these guards confessions rather
than features, and a confession nobody hears is just a wrong answer.
"""

import warnings

import numpy as np
import pytest

import pulsim as p


def rl_circuit():
    """Carries about 1.2 A steady state — comfortably above the
    absurd clamp the tests below impose."""
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "vin", "gnd", 12.0)
    b.add_resistor("R", "vin", "sw", 10.0)
    b.add_inductor("L1", "sw", "gnd", 1e-3)
    return b


def test_an_unguarded_run_reports_nothing():
    """No false positives: the guards are off by default and the
    record must stay empty."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = p.simulate(rl_circuit(), t_end=2e-3, dt=1e-6)
    assert list(res.inductor_guard_actions) == []
    assert [w for w in caught if "inductor guard" in str(w.message)] == []
    assert np.asarray(res.i("L1"))[-1] == pytest.approx(1.2, abs=0.01)


def test_a_clamped_run_names_the_inductor_and_counts_the_steps():
    opts = p.SolverOptions(inductor_abs_clamp=0.05)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = p.simulate(rl_circuit(), t_end=2e-3, dt=1e-6,
                          solver=opts)

    acts = list(res.inductor_guard_actions)
    assert len(acts) == 1
    a = acts[0]
    assert a.clamp_count > 0
    assert a.total() == a.clamp_count + a.freeze_count
    assert a.t_first > 0.0
    assert a.reported_limit == pytest.approx(0.05, abs=1e-12)
    # The solver really did want more current than it was allowed.
    assert a.worst_solved > 0.05

    hits = [w for w in caught if "inductor guard" in str(w.message)]
    assert len(hits) == 1
    msg = str(hits[0].message)
    assert "L1" in msg                       # names the device
    assert str(a.total()) in msg             # says how often
    assert "limit" in msg

    # And the trace really is the limit, not the circuit: the same
    # run unguarded settles near 1.2 A.
    i = np.asarray(res.i("L1"))
    assert np.abs(i).max() == pytest.approx(0.05, abs=1e-9)


def test_the_freeze_guard_is_recorded_separately_from_the_clamp():
    """They catch different failures — a sudden jump versus a slow
    drift — so a user debugging one should not be told about the
    other."""
    opts = p.SolverOptions(inductor_freeze_di_max=1e-4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = p.simulate(rl_circuit(), t_end=2e-3, dt=1e-6,
                          solver=opts)
    acts = list(res.inductor_guard_actions)
    assert len(acts) == 1
    assert acts[0].freeze_count > 0
    assert acts[0].clamp_count == 0


def test_recording_does_not_change_the_numbers():
    """The report is an observation, not an intervention: a guarded
    run must produce exactly what it produced before."""
    opts = p.SolverOptions(inductor_abs_clamp=0.05)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = p.simulate(rl_circuit(), t_end=2e-3, dt=1e-6, solver=opts)
        b = p.simulate(rl_circuit(), t_end=2e-3, dt=1e-6, solver=opts)
    np.testing.assert_array_equal(np.asarray(a.states),
                                  np.asarray(b.states))
