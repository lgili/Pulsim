"""v2.0 Phase 2: a run that dies part-way brings the run with it.

A simulation that fails at 90% used to return nothing at all — the
exception carried a message and every sample computed before the
failure was destroyed with the stack. On a run that takes minutes
that is the difference between "here is where it broke, and here is
the waveform leading into it" and "start again".

The default is still an exception. Returning a truncated result as if
it were whole would be exactly the silent wrong answer this project
keeps removing, so the partial trace has to be asked for by catching
the type.
"""

import warnings

import numpy as np
import pytest

import pulsim as p


def failing_rectifier():
    """A 170 V mains rectifier at a step it cannot take. With
    `max_dt_halvings=0` the local step reduction is disabled, so this
    is the pre-v2.0 hard failure."""
    b = p.CircuitBuilder()
    b.add_sine_voltage_source("Vac", "ac", "gnd", 0.0, 170.0, 60.0)
    q = p.IdealDiodeParams()
    q.kappa = 20.0
    b.add_nonlinear_diode("D", "ac", "vout", q)
    b.add_resistor("R", "vout", "gnd", 50.0)
    b.add_capacitor("C", "vout", "gnd", 100e-6)
    return b


def _fail():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return p.simulate(failing_rectifier(), t_end=1.7e-2, dt=1e-4,
                           enable_nonlinear_refresh=True,
                           max_dt_halvings=0)


def test_the_failure_is_still_a_failure():
    """No silent truncation: the caller who does not ask still gets
    an exception, and it is still a RuntimeError so existing
    `except RuntimeError` code keeps working."""
    with pytest.raises(RuntimeError):
        _fail()
    with pytest.raises(p.SimulationAborted):
        _fail()


def test_it_carries_what_it_computed():
    with pytest.raises(p.SimulationAborted) as exc:
        _fail()
    e = exc.value

    assert e.t_failed > 0.0
    partial = e.partial
    assert partial.num_steps() > 10          # a real slice of the run

    t = np.asarray(partial.times)
    s = np.asarray(partial.states)
    assert np.isfinite(s).all()              # only the GOOD samples
    assert t[0] == pytest.approx(0.0)
    # It stops before the step it could not take.
    assert t[-1] < e.t_failed
    assert len(t) == s.shape[0]


def test_the_partial_trace_is_on_the_same_grid():
    """It is a prefix of the run that would have happened, not a
    resampling of it."""
    with pytest.raises(p.SimulationAborted) as exc:
        _fail()
    t = np.asarray(exc.value.partial.times)
    np.testing.assert_allclose(t, np.arange(len(t)) * 1e-4,
                               rtol=0, atol=1e-12)


def test_the_message_still_names_the_cause():
    """Carrying data must not cost the diagnostic Phase 1 built."""
    with pytest.raises(p.SimulationAborted) as exc:
        _fail()
    msg = str(exc.value)
    assert "converge" in msg
    # ...and it still localises the failure to a row.
    assert "node" in msg or "current through" in msg


def test_a_successful_run_raises_nothing():
    """No false positives on the happy path."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = p.simulate(failing_rectifier(), t_end=1.7e-2, dt=1e-5,
                          enable_nonlinear_refresh=True)
    assert res.num_steps() > 0
    assert np.isfinite(np.asarray(res.states)).all()


def test_a_cancelled_run_is_not_an_abort():
    """`should_continue` returning False is a deliberate stop, not a
    failure — it returns the partial trace normally rather than
    raising."""
    calls = {"n": 0}

    def stop_soon():
        calls["n"] += 1
        return calls["n"] < 25

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = p.simulate(failing_rectifier(), t_end=1.7e-2, dt=1e-5,
                          enable_nonlinear_refresh=True,
                          should_continue=stop_soon)
    assert 0 < res.num_steps() < 100
