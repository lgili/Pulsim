"""``res.v(<ground>)`` is zeros on every engine.

Ground has node id -1 (it is the reference, not a state), and
``states[:, -1]`` is the LAST state column -- a branch current. Before
this fix ``res.v("gnd")`` returned it with no error: 1 V into 1 ohm read
``res.v("gnd") == -1``.
"""
from __future__ import annotations

import numpy as np
import pytest

import pulsim as p


def _rc():
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "a", "gnd", 1.0)
    b.add_resistor("R", "a", "b", 1.0)
    b.add_capacitor("C", "b", "gnd", 1e-6)
    return b


@pytest.mark.parametrize("engine,kw", [
    ("pwl", {"dt": 1e-7}),
    ("trbdf2", {}),
    ("dsed", {}),
])
@pytest.mark.parametrize("gnd", ["gnd", "GND", "0"])
def test_ground_voltage_is_zero(engine, kw, gnd):
    b = _rc()
    res = p.simulate(b, t_end=5e-6, engine=engine, **kw)
    v = res.v(gnd)
    assert isinstance(v, np.ndarray)
    assert v.shape == np.asarray(res.times).shape
    assert np.all(v == 0.0)
    # The control: the last state column is NOT zero on this circuit
    # (it is the source current), so the test can tell the bug apart.
    assert np.any(np.asarray(res.states)[:, -1] != 0.0)
    assert res.v(gnd, 0) == 0.0
    assert np.all(res.v(gnd, slice(0, 3)) == 0.0)
    # A non-ground node still reads its column.
    assert res.v("a")[-1] == pytest.approx(1.0)


def test_average_power_at_ground_node_is_zero():
    b = _rc()
    res = p.simulate(b, t_end=5e-6, dt=1e-7)
    bid = 0                                 # "V", the first branch
    P = p.average_power_at_node(b, res, node_name="gnd",
                                current_branch_id=bid)
    assert P == 0.0
