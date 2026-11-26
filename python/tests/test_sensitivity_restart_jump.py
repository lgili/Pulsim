"""add-parameter-sensitivities group 6 on the restart rule's consistent
start: a restart whose event MOVES the state (an impulsive event: the jump
of the capacitor voltages / inductor currents is read from the restart's
probe stages and the histories restart from x0 + J) is refused by name with
sensitivities — dJ/dp is not carried by the stage recurrence. The same
circuit before that event returns a number that matches central differences
of the same scheme.

V -> R -> c1 ; C1: c1-gnd ; S: c1-c2 (1 mOhm) ; C2, RL: c2-gnd. S conducts
for the first 0.3 ms of each 1 ms period: at t = 1 ms it closes on two
capacitors at different voltages, a loop with tau = 0.5 ns << dt.
"""

import warnings

import numpy as np
import pytest

import pulsim as p

TP, D, DT, R0 = 1e-3, 0.3, 1e-6, 1e3
MOVED = "restarted from a state its event moved"


def _run(t_end, split, R=R0, **kw):
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "in", "gnd", 10.0)
    b.add_resistor("R", "in", "c1", R)
    b.add_capacitor("C1", "c1", "gnd", 1e-6)
    b.add_switch("S", "c1", "c2", 1e3, 1e-9)
    b.add_capacitor("C2", "c2", "gnd", 1e-6)
    b.add_resistor("RL", "c2", "gnd", 1e3)
    sf = p.make_pwm_switch_fn(1 / TP, D, b.switch_index_of("S"), b.graph.num_switches)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return p.simulate(b, t_end, dt=DT, engine="pwl", switch_fn=sf,
                          discontinuity_restart=True,
                          enable_substep_state_correction=split, **kw)


@pytest.mark.parametrize("split", [False, True])
def test_a_restart_that_moves_the_state_is_refused_by_name(split):
    # the control: the closing at 1 ms IS impulsive on this engine
    recs = [x for x in _run(1.2 * TP, split).reinit_records if x.t == pytest.approx(TP)]
    assert len(recs) == 1 and sorted(recs[0].states) == ["C1", "C2"], recs
    with pytest.raises(p.UnsupportedError) as ei:
        _run(1.2 * TP, split, sensitivities=["R.R"])
    assert MOVED in str(ei.value) and "impulsive" in str(ei.value)


@pytest.mark.parametrize("split", [False, True])
def test_the_same_circuit_before_the_jump_returns_a_true_number(split):
    t_end = 0.9 * TP          # past the opening edge (a fixed instant), before the closing
    r = _run(t_end, split, sensitivities=["R.R"])
    assert not [x for x in r.reinit_records if x.t > 0]
    s = float(r.sens("c1", "R.R")[-1])
    h = 1e-4 * R0
    fd = (float(np.asarray(_run(t_end, split, R=R0 + h).v("c1"))[-1])
          - float(np.asarray(_run(t_end, split, R=R0 - h).v("c1"))[-1])) / (2 * h)
    assert s == pytest.approx(fd, rel=1e-6) and abs(s) > 1e-4, (s, fd)
