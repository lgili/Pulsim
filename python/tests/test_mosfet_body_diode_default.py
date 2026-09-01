"""The body diode is part of the device, not an accessory (C.1).

Audit C.1, "diodo de corpo intrínseco por padrão". Every vertical
power MOSFET has a body diode — it is formed by the same p-n
junction that makes the transistor, and there is no such thing as
one without it. Pulsim made it opt-in, which was wrong twice:

* PHYSICALLY. A gate-off MOSFET in an inductive path is
  R_off = 1 GOhm, so the freewheeling current has nowhere to go
  and the node runs away. That is not a corner case — it is the
  normal state of the low-side device in every synchronous
  converter during dead time.

* BY REVEALED PREFERENCE. This repository's own call sites voted
  54 to 17 for `add_mosfet_with_body_diode` over bare
  `add_mosfet`. A default overridden three times out of four is
  the wrong default.

The cost is nil: the PWL cache factors only the switch states a
run actually visits, so a branch that never changes state is one
more stamp, not another power of two. Measured on chains of 2-8
MOSFETs, with and without: 1.00x.

`body_diode=False` stays available for a device that genuinely
has none — an eGaN HEMT conducts in reverse through the channel
rather than a p-n junction.
"""

import numpy as np
import pytest

import pulsim as p


def _freewheel(**mosfet_kwargs):
    """Gate-off MOSFET with an inductor trying to freewheel
    through it. Without a body diode there is no path."""
    b = p.CircuitBuilder()
    b.add_inductor("L", "sw", "gnd", 1e-3, i0=50.0)
    b.add_mosfet("M1", "sw", "gnd", **mosfet_kwargs)
    b.add_resistor("Rload", "sw", "gnd", 1e5)
    return b


def _all_off(b):
    n = b.graph.num_switches
    return lambda t: p.SwitchStateMask(n)


def test_the_body_diode_is_there_by_default():
    b = _freewheel()
    assert b.node_id_of("sw") >= 0
    # It is a real, named device — reachable like any other.
    res = p.simulate(b, t_end=1e-6, dt=1e-9, engine="pwl")
    i_body = np.asarray(res.i("M1_body"))
    assert np.all(np.isfinite(i_body))


def test_the_default_gives_the_freewheeling_current_a_path():
    """The whole point. The node stays at a forward drop below
    ground instead of running away."""
    b = _freewheel()
    res = p.simulate(b, t_end=1e-6, dt=1e-9, engine="pwl",
                     switch_fn=_all_off(b))
    v = float(np.asarray(res.v("sw"))[-1])
    assert -2.0 < v < 0.0, v


def test_without_it_the_node_runs_away_and_the_current_dies():
    """`body_diode=False` is still available, and this is what it
    costs on an inductive load: the node hits megavolts on the
    first step and the inductor's 50 A is gone within a
    microsecond. That is correct for an idealized open circuit —
    and it is the state the old default put every gate-off
    MOSFET in.

    (The voltage-sanity guard stays quiet here because it
    calibrates against the largest source in the circuit, and
    this one has no source at all. The assertion is on the
    physics rather than on the warning for that reason.)
    """
    b = _freewheel(body_diode=False)
    res = p.simulate(b, t_end=1e-6, dt=1e-9, engine="pwl",
                     switch_fn=_all_off(b))
    v = np.abs(np.asarray(res.v("sw")))
    i = np.asarray(res.i("L"))
    assert v.max() > 1e6, v.max()
    assert abs(i[-1]) < 1e-6, i[-1]

    # With the diode, the same circuit simply freewheels.
    b2 = _freewheel()
    res2 = p.simulate(b2, t_end=1e-6, dt=1e-9, engine="pwl",
                      switch_fn=_all_off(b2))
    assert np.abs(np.asarray(res2.v("sw"))).max() < 1.0
    assert float(np.asarray(res2.i("L"))[-1]) == pytest.approx(
        50.0, rel=1e-3)


def test_opting_out_leaves_no_body_device():
    b = _freewheel(body_diode=False)
    res = p.simulate(b, t_end=1e-8, dt=1e-8, engine="pwl")
    with pytest.raises(Exception):
        res.i("M1_body")


def test_level1_mosfet_defaults_the_same_way():
    b = p.CircuitBuilder()
    b.add_voltage_source("Vg", "g", "gnd", 0.0)
    b.add_inductor("L", "sw", "gnd", 1e-3, i0=50.0)
    b.add_mosfet_level1("M1", "sw", "gnd", "g", 50.0, 3.0)
    res = p.simulate(b, t_end=1e-6, dt=1e-9, engine="pwl")
    assert np.all(np.isfinite(np.asarray(res.i("M1_body"))))
    v = float(np.asarray(res.v("sw"))[-1])
    assert -2.0 < v < 0.0, v


def test_forward_conduction_is_unaffected():
    """The body diode must not shunt the on-state. Gate-on
    device carrying current sees only R_on."""
    b = p.CircuitBuilder()
    b.add_voltage_source("V1", "a", "gnd", 10.0)
    b.add_resistor("R1", "a", "d", 1.0)
    b.add_mosfet("M1", "d", "gnd", R_on=1e-3)

    n = b.graph.num_switches

    def sf(t):
        m = p.SwitchStateMask(n)
        # Masks are set BY NAME, so the extra body-diode branch
        # does not shift anything a caller wrote by hand.
        m.set(b.switch_index_of("M1"), True)
        return m

    res = p.simulate(b, t_end=1e-8, dt=1e-8, engine="pwl",
                     switch_fn=sf)
    v_d = float(np.asarray(res.v("d"))[-1])
    assert v_d == pytest.approx(10.0 * 1e-3 / (1.0 + 1e-3),
                                 rel=0.02)
