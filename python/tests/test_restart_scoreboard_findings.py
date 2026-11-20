"""Findings of the restart-rule scoreboard investigation
(openspec/changes/update-number-changing-release/fixes/restart-scoreboard.md).

The storage convention (release group B): a stored sample at an event is the
LEFT limit and ``result.right_limit_states`` holds the RIGHT limit, the
state where the next step starts; ``pulsim.with_event_limits`` inserts it one
ulp after the event as a value of the trajectory.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import pulsim as p

F_SB_1 = pytest.mark.xfail(strict=True, reason=(
    "finding F-SB-1: under the restart rule the fixed engine records as the right limit "
    "the start of the restart step's quadrature line, 2·w1·X1 + (1 − 2·w1)·X2, at every "
    "restart step (the event's and the window's next two): a capacitor voltage, continuous, "
    "reads 0 V left / 0.163 V right at the closing and jumps at two instants with no event"))


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


@F_SB_1
def test_a_continuous_state_has_equal_left_and_right_limits():
    r = _rc_closing()
    assert r.right_limit_index is not None and len(r.right_limit_index) > 0
    v = p.with_event_limits(r)
    t, vc = np.asarray(v.times), np.asarray(v.v("c"))
    # each right-limit row sits one ulp after its left limit
    k = np.flatnonzero(t[1:] == np.nextafter(t[:-1], np.inf))
    assert k.size == len(r.right_limit_index)
    np.testing.assert_allclose(vc[k + 1], vc[k], rtol=0.0, atol=1e-9)
