"""F-Z-6: the ll13 active-clamp forward reference call (trbdf2, rtol 1e-8) must not
depend on the run length.

Root cause 1 (the t = 0 abort): a zero start without a stateful kind kept the init's
derivative slots (v_L = 0 for the 48 V the leakage inductor carries at t = 0+), the
first step's estimate was O(h) at every h, the step sat on h_min = span * 1e-11 and
the floor decided the run (forced accept at 5e-16 s for 50 us, Newton numerically
singular at 3e-17 s for 3 us).
"""
import dataclasses
import pathlib
import sys
import warnings

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmarks" / "tools"))
import v1_reader  # noqa: E402

LL13 = ROOT / "benchmarks/circuits/switching/isolated/ll13_active_clamp_forward.yaml"


def _run(t_end, rtol, **kw):
    case = dataclasses.replace(v1_reader.load(LL13), t_end=t_end)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return case.run("trbdf2", rtol=rtol, atol=rtol * 1e-4, **kw)


@pytest.mark.parametrize("t_end,rtol", [(3e-6, 1e-8), (3e-6, 1e-10), (100e-6, 1e-10)])
def test_start_does_not_sit_on_the_floor(t_end, rtol):
    # these aborted at t = 0 ("Newton failed ... with h already at h_min")
    # before the first turn-off, with the floors of a run of t_end (h_max = t_end / 1000)
    r = _run(1.4e-6, rtol, dt=t_end / 1000)
    st = r._trbdf2_stats
    assert st["n_forced_accepts"] == 0
    assert st["n_derivative_refreshes"] >= 1   # the start took the probe's slots
    h_min = t_end * 1e-11   # the run's floor (span_ref = 1000 h_max)
    # times[1]: the start probe (its states move over it), committed as a step;
    # the first TR-BDF2 step after it was the floor: forced, or Newton singular
    assert r.times[2] - r.times[1] > 10 * h_min



def _on_grid(r, g):
    t, x = np.asarray(r.times), np.asarray(r.states)
    return np.array([np.interp(g, t, x[:, j]) for j in range(x.shape[1])]).T


def test_run_length_does_not_change_the_common_span():
    # runs of 5, 20 and 50 us agree on [0, 3 us] (measured: rms 1.2e-6 of each
    # state's peak, max 4.6e-6, edges included; the reference class is 1e-3)
    g = np.linspace(0.0, 3e-6, 3001)
    runs = [_on_grid(_run(te, 1e-8), g) for te in (5e-6, 20e-6, 50e-6)]
    scale = np.abs(runs[0]).max(axis=0) + 1e-30
    for x in runs[1:]:
        d = np.abs(x - runs[0]) / scale
        assert np.sqrt(np.mean(d ** 2, axis=0)).max() < 1e-4
        assert d.max() < 1e-3


@pytest.mark.xfail(strict=True, reason="root cause 2: the 1.5 us turn-off edge probe "
                   "stalls and the step falls to h_min (F-Z-6 part 2, open)")
def test_three_microsecond_run_survives_the_first_turn_off():
    _run(3e-6, 1e-8)


def _rl():
    import pulsim as p
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "in", "gnd", 10.0)
    b.add_resistor("R", "in", "n", 10.0)
    b.add_inductor("L", "n", "gnd", 1e-3)
    return p, b


def _sim(p, b, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return p.simulate(b, engine="trbdf2", dt=1e-6, **kw)


def test_start_slots_resume_like_the_uninterrupted_run():
    # zero start: v_L(0+) = 10 V against the init slot's 0 V, so the start takes
    # the probe's slots; a short run, its snapshot and a resume reach the
    # uninterrupted run's end state, and the resume takes no second start probe
    p, b = _rl()
    full = _sim(p, b, t_end=1e-3)
    head = _sim(p, b, t_end=2e-6)
    assert head._trbdf2_stats["n_derivative_refreshes"] == 1
    tail = _sim(p, b, t_end=1e-3, resume_from=head.final_snapshot)
    assert tail._trbdf2_stats["n_derivative_refreshes"] == 0
    a, c = np.asarray(full.states[-1]), np.asarray(tail.states[-1])
    assert np.allclose(c, a, rtol=1e-6, atol=1e-9)
    # the exact end current (V/R)(1 - e^{-t R/L}) = 1 - e^{-10}, at the run's tolerance
    k = b.state_var_names().index("I(L)")
    assert c[k] == pytest.approx(1.0 - np.exp(-10.0), rel=1e-5)


@pytest.mark.parametrize("rtol", [None, 1e-8])
def test_a_ramp_from_zero_is_a_consistent_start(rtol):
    # rc_step: v_in ramps from 0 (5e9 V/s), v_in(0) = 0 = v_C(0) is consistent;
    # the probe's slot at t + dtp carries the ramp (slot' * dtp), which the
    # two-length extrapolation to dtp -> 0 removes: no start refresh, only the
    # ramp corner's (1 ns) edge refresh the run had before the start probe
    case = v1_reader.load(ROOT / "benchmarks/circuits/linear/rc_step.yaml")
    kw = {} if rtol is None else dict(rtol=rtol, atol=1e-12)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = case.run("trbdf2", **kw)
    assert r._trbdf2_stats["n_derivative_refreshes"] == 1
    # the inconsistent start (10 V across an inductor at t = 0+) is still seen
    p, b = _rl()
    assert _sim(p, b, t_end=2e-6, rtol=1e-8, atol=1e-12)._trbdf2_stats[
        "n_derivative_refreshes"] >= 1


def test_a_given_initial_state_takes_no_start_probe():
    # initial_state is seeded as a DC operating point (zero slots by contract):
    # PSS period runs start there and must keep the map they had
    p, b = _rl()
    r = _sim(p, b, t_end=1e-4, initial_state=np.zeros(len(b.state_var_names())))
    assert r._trbdf2_stats["n_derivative_refreshes"] == 0
