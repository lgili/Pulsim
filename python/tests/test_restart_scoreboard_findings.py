"""Findings of the restart-rule scoreboard investigation
(openspec/changes/update-number-changing-release/fixes/restart-scoreboard.md).

The storage convention (release group B): a stored sample at an event is the
LEFT limit and ``result.right_limit_states`` holds the RIGHT limit, the
state value one instant after the event; ``pulsim.with_event_limits``
inserts it one ulp after the event as a value of the trajectory. F-SB-1 (the
quadrature line's start stored as the right limit) is fixed:
test_restart_right_limit.py.
"""

from __future__ import annotations

import warnings

import numpy as np

import pulsim as p

def _rc_closing():
    # 10 V -> ideal switch -> 1 ohm -> 1 uF from 0 V; the gate closes at 2.25 us, dt = tau/2
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "vin", "gnd", 10.0)
    b.add_ideal_switch("S", "vin", "a", 0.0, 0.0)
    b.add_resistor("R", "a", "c", 1.0)
    b.add_capacitor("C", "c", "gnd", 1e-6, 0.0)
    k = b.switch_index_of("S")

    def fn(t):
        m = p.SwitchStateMask(1)
        m.set(k, t >= 2.25e-6)
        return m
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return p.simulate(b, t_end=6e-6, engine="pwl", dt=0.5e-6, switch_fn=fn,
                          discontinuity_restart=True)


def test_a_continuous_state_has_equal_left_and_right_limits():
    r = _rc_closing()
    assert r.right_limit_index is not None and len(r.right_limit_index) > 0
    v = p.with_event_limits(r)
    t, vc = np.asarray(v.times), np.asarray(v.v("c"))
    # each right-limit row sits one ulp after its left limit
    k = np.flatnonzero(t[1:] == np.nextafter(t[:-1], np.inf))
    assert k.size == len(r.right_limit_index)
    np.testing.assert_allclose(vc[k + 1], vc[k], rtol=0.0, atol=1e-9)
