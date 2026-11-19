"""Restart rule without the split: a scheduled gate edge strictly inside a step.

Design D4 reads the gate of step (t_prev, t] at t_prev, which places an edge ON the
grid at the start of the first step that carries it (order 2). An edge strictly
inside the step was then applied at t: the sample at t was solved in the pre-edge
mode, a whole span off on an algebraic node at every off-grid edge (exact-truth
scoreboard: vcswitch_resistor 0.059, the only nonzero configuration; ll06 V(sw)
0.117 against plain pwl's 0.055). Now such a step reads the gate's left limit at t
(the oracle says an edge is inside): the sample at t is in the mode of t. On-grid
edges keep D4.

Truth: the 50-digit exact engine (benchmarks/tools/exact_truth.py on the
exact-truth branch), conductance form, sampled with mpmath.
"""
import warnings
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
RULE = dict(discontinuity_restart=True)


def _case(monkeypatch, rel):
    monkeypatch.syspath_prepend(str(ROOT / "benchmarks" / "tools"))
    pytest.importorskip("yaml")
    import v1_reader

    return v1_reader.load(ROOT / "benchmarks" / "circuits" / rel)


def test_a_resistive_load_reads_the_mode_of_every_sample(monkeypatch):
    """No reactive element (the static path). Exact: the control pulse is high at
    t = 0 (periodic) and falls at 1.5 ns; rises at 20.0005 us, falls at 50.0015 us."""
    case = _case(monkeypatch, "switching/basic/vcswitch_resistor.yaml")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plain = case.run("pwl")
        r = case.run("pwl", **RULE)
    t, v = np.asarray(r.times), np.asarray(r.v("out"))
    for tq, exact in ((1e-7, 9.999999e-7), (2.01e-5, 9.999900001), (5.01e-5, 9.999999e-7)):
        k = int(np.argmin(np.abs(t - tq)))
        # before: 9.9999 V, 1e-6 V, 9.9999 V (the pre-edge mode); now exact to ~1e-15
        assert v[k] == pytest.approx(exact, abs=1e-6)
    # every sample of a static circuit is exact on plain pwl; the rule now agrees
    np.testing.assert_allclose(v, np.asarray(plain.v("out")), rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("div", [1, 2])
def test_the_sample_after_an_off_grid_turn_off_is_freewheeling(monkeypatch, div):
    """ll06 buck: S1 turns off at 8.28 us (off the 0.2 us grid and the 0.1 us one),
    D1 takes the inductor current. Exact V(sw): -0.0578 V at 8.4 us, -0.0584 V at
    8.3 us (before: +48.0 V off, S1 still on)."""
    case = _case(monkeypatch, "switching/buck/ll06_buck_vcswitch.yaml")
    case.t_end = 1e-5
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = case.run("pwl", dt=case.dt / div, **RULE)
    t, v = np.asarray(r.times), np.asarray(r.v("sw"))
    tq, exact = (8.4e-6, -0.0577906272207) if div == 1 else (8.3e-6, -0.0583788407376)
    k = int(np.argmin(np.abs(t - tq)))
    # measured +4.7e-4 V off at dt and dt/2 (the edge is placed at t_prev: the inductor
    # current is first order off); before +48.0 V
    assert v[k] == pytest.approx(exact, abs=1e-2)



def _pulse_circuit(dead_time=None):
    import pulsim as p

    b = p.CircuitBuilder()
    b.add_voltage_source("V", "in", "gnd", 10.0)
    b.add_switch("S", "in", "x", 1e3, 1e-9)
    b.add_inductor("L", "x", "out", 1e-3)
    b.add_resistor("R", "out", "gnd", 1.0)
    b.add_resistor("Rx", "x", "gnd", 10.0)
    n, k = b.graph.num_switches, b.switch_index_of("S")
    if dead_time is None:
        return b, p.make_pwm_switch_fn(1e5, 0.04, k, n)
    b.add_switch("SL", "x", "gnd", 1e-3, 1e-9)   # a weak low side: L stays fed by S
    n = b.graph.num_switches
    return b, p.make_dead_time_pwm_pair_fn(1e5, 0.06, k, b.switch_index_of("SL"), n, dead_time)


def _on_times_us(r):
    return [round(t * 1e6) for t, m in zip(np.asarray(r.times), r.switch_states)
            if t > 0 and m[0]]


def test_a_pulse_inside_one_step_is_not_dropped():
    """A gate that leaves and re-enters within one step (a 0.4 us pulse at every
    10 us carrier start, dt = 1 us; ll15's closed loop at its 5 % minimum duty is
    the same shape) keeps design D4 under the rule without the split: the left
    limit at t would solve every step OFF and drop the pulse's charge (ll15 V(sw)
    0.066 -> 0.216 against trbdf2); D4 widens it to the step. The split places
    it: the inductor current after the first pulse is V * 0.4 us / L, then decays
    with L / (R + Rx) for 0.6 us."""
    import pulsim as p

    b, fn = _pulse_circuit()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = p.simulate(b, 1e-4, engine="pwl", dt=1e-6, switch_fn=fn, **RULE)
        s = p.simulate(b, 1e-4, engine="pwl", dt=1e-6, switch_fn=fn,
                       enable_substep_state_correction=True, **RULE)
    # D4: the sample one step after each carrier start is solved ON, nothing else
    assert _on_times_us(r) == list(range(1, 100, 10))
    t = np.asarray(s.times)
    i_split = float(np.asarray(s.i("L"))[np.argmin(np.abs(t - 1e-6))])
    assert i_split == pytest.approx(10.0 * 0.4e-6 / 1e-3 * np.exp(-0.6e-6 * 11.0 / 1e-3), rel=5e-3)


def test_a_dead_time_leg_keeps_its_narrow_pulse():
    """The same per bit: the high side is ON for 0.5 us from each carrier start and
    the low side turns on 0.1 us later, inside the same step. The mask at t (low
    side ON) differs from the one before the step (both OFF), but the high-side bit
    is back where it was: the step keeps D4 (high side ON)."""
    import pulsim as p

    b, fn = _pulse_circuit(dead_time=1e-7)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = p.simulate(b, 1e-4, engine="pwl", dt=1e-6, switch_fn=fn, **RULE)
    assert _on_times_us(r) == list(range(1, 100, 10))
