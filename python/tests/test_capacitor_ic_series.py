"""A capacitor IC is a voltage ACROSS the capacitor, wherever it sits.

``CircuitBuilder.initial_state()`` wrote ``c0`` into the node voltage of
the positive terminal only. Two series capacitors given 10 V each then
started at 0 V (upper) and 10 V (lower), in any build order: the upper
node was overwritten by the lower cap's write. Found auditing notebook
31 (split DC link precharged to V_dc/2 each): its C1 started empty.
"""
from __future__ import annotations

import numpy as np
import pytest

import pulsim as p


def _series_pair(order):
    b = p.CircuitBuilder()
    caps = {"C1": ("top", "mid"), "C2": ("mid", "gnd")}
    for name in order:
        b.add_capacitor(name, *caps[name], 1e-3, c0=10.0)
    b.add_resistor("RL", "top", "gnd", 1e3)
    return b


@pytest.mark.parametrize("order", [("C1", "C2"), ("C2", "C1")])
def test_series_capacitor_ics_are_differences(order):
    b = _series_pair(order)
    x0 = np.asarray(b.initial_state())
    assert x0[b.node_id_of("top")] == pytest.approx(20.0)
    assert x0[b.node_id_of("mid")] == pytest.approx(10.0)
    r = p.simulate(b, t_end=1e-6, dt=1e-7)
    v = lambda n: r.v(n)[0]  # noqa: E731
    assert v("top") - v("mid") == pytest.approx(10.0, abs=1e-6)
    assert v("mid") == pytest.approx(10.0, abs=1e-6)


def test_floating_capacitor_ic_keeps_its_difference():
    b = p.CircuitBuilder()
    b.add_capacitor("C", "a", "b", 1e-6, c0=5.0)
    b.add_resistor("Ra", "a", "gnd", 1e3)
    b.add_resistor("Rb", "b", "gnd", 1e3)
    x0 = np.asarray(b.initial_state())
    # Unchanged placement for a lone floating cap: v(from) = c0, v(to) = 0.
    assert x0[b.node_id_of("a")] == pytest.approx(5.0)
    assert x0[b.node_id_of("b")] == pytest.approx(0.0)


def test_floating_series_pair_is_walked_from_its_bottom_node():
    b = p.CircuitBuilder()
    b.add_capacitor("C1", "a", "m", 1e-6, c0=4.0)
    b.add_capacitor("C2", "m", "z", 1e-6, c0=6.0)
    for n in ("a", "m", "z"):
        b.add_resistor(f"R{n}", n, "gnd", 1e3)
    x0 = np.asarray(b.initial_state())
    v = {n: x0[b.node_id_of(n)] for n in ("a", "m", "z")}
    assert (v["a"], v["m"], v["z"]) == pytest.approx((10.0, 6.0, 0.0))


def test_capacitor_ics_that_break_kvl_are_refused():
    b = p.CircuitBuilder()
    b.add_capacitor("C1", "a", "gnd", 1e-6, c0=5.0)
    b.add_capacitor("C2", "a", "gnd", 1e-6, c0=3.0)
    with pytest.raises(ValueError, match="break KVL"):
        b.initial_state()
