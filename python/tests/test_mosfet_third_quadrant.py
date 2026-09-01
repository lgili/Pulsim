"""MOSFET third-quadrant symmetry — synchronous rectification (C.1).

A MOSFET has no built-in drain and source: with V_DS < 0 the
terminals swap roles and the channel conducts just as well the
other way. That is what synchronous rectification IS, and it is
the most common thing a MOSFET does in a modern converter.

Evaluating the FORWARD polynomial at negative V_DS instead is not
merely inaccurate, it is NON-MONOTONE. Measured on the level-1
model (K = 50, V_T = 3, V_GS = 10) before the fix:

    V_DS      i (A)
    -0.01        7      <- correct, resistive
    -20      20400
    -40      21600      <- turns around
    -50          0      <- crosses zero
    -63     -63063

So an inductor freewheeling through a gated-on device had SEVERAL
solutions, and Newton settled on a far one: v(sw) = -63 V where
the channel's own 1.43 mΩ gives -14 mV — reported as 544 W of
loss with no warning at all. The anti-parallel body diode the
model's header used to prescribe as the fix did not prevent it
(that case landed at -63 V; without the diode, -50 V).
"""

import numpy as np
import pytest

import pulsim as p

K, VT, VG = 50.0, 3.0, 10.0
R_ON = 1.0 / (2 * K * (VG - VT))      # 1.4286 mΩ


def _i_at(vds):
    b = p.CircuitBuilder()
    b.add_voltage_source("Vd", "d", "gnd", vds)
    b.add_voltage_source("Vg", "g", "gnd", VG)
    # with_body_diode=False on purpose: these tests characterise
    # the CHANNEL. Since v2.0 the body diode is on by default
    # (audit C.1), and in the third quadrant it would carry the
    # current the channel is being measured for.
    b.add_mosfet_level1("M1", "d", "gnd", "g", K, VT,
                         with_body_diode=False)
    res = p.simulate(b, t_end=1e-8, dt=1e-8, engine="pwl")
    return float(np.asarray(res.i("Vd"))[-1])


def test_the_channel_is_monotone_in_both_quadrants():
    """The property that removes the spurious operating points:
    one V_DS per current, everywhere."""
    vds = [-100, -63, -50, -30, -20, -10, -5, -1, -0.1, -0.01,
           0.01, 0.1, 1, 5, 10, 20]
    i = [_i_at(v) for v in vds]
    d = np.diff(i)
    assert np.all(d < 0) or np.all(d > 0), list(zip(vds, i))


def test_it_mirrors_when_the_terminals_swap():
    """Not exactly odd, and it should not be. Swapping the
    terminals also moves the reference the gate is measured
    against: forward sees V_OV = V_GS − V_T, reverse sees
    V_OV − V_DS, because the terminal now acting as the source
    sits |V_DS| lower. So the magnitudes differ by roughly
    |V_DS|/V_OV — 0.14% at 10 mV here — and a real device is
    asymmetric in exactly that way. (An earlier version of this
    test demanded exact oddness and failed at 0.14%: the
    expectation was wrong, not the model.)"""
    V_OV = VG - VT
    for v in (0.01, 0.1, 1.0):
        fwd, rev = _i_at(v), _i_at(-v)
        expected_skew = v / V_OV
        got = abs(rev / -fwd - 1.0)
        assert got < 3.0 * expected_skew, (v, fwd, rev)
        assert got > 0.1 * expected_skew, (v, fwd, rev)


def test_small_signal_resistance_is_r_on_both_ways():
    for v in (0.01, -0.01):
        r = abs(v / _i_at(v))
        assert r == pytest.approx(R_ON, rel=0.01)


def test_synchronous_rectification_shows_millivolts():
    """The circuit that produced -63 V and 544 W of phantom loss:
    an inductor freewheeling through a gated-ON low-side MOSFET."""
    def run(gate_on, with_body_diode):
        b = p.CircuitBuilder()
        b.add_voltage_source("Vg", "g", "gnd", VG if gate_on else 0.0)
        b.add_inductor("L", "sw", "out", 100e-6, i0=10.0)
        b.add_capacitor("Co", "out", "gnd", 1e-3, c0=5.0)
        b.add_resistor("Ro", "out", "gnd", 0.5)
        b.add_mosfet_level1("MLS", "sw", "gnd", "g", K, VT,
                             with_body_diode=with_body_diode)
        res = p.simulate(b, t_end=2e-6, dt=1e-9, engine="pwl")
        return (float(np.asarray(res.v("sw"))[-1]),
                float(np.asarray(res.i("L"))[-1]))

    for wbd in (True, False):
        v_sw, i_l = run(gate_on=True, with_body_diode=wbd)
        # The channel carries it: V = -I·R_on, tens of millivolts.
        assert v_sw == pytest.approx(-i_l * R_ON, rel=0.05), v_sw
        assert abs(v_sw * i_l) < 1.0          # was 544 W

    # Gate OFF with a body diode is the diode's job, unchanged.
    v_off, i_off = run(gate_on=False, with_body_diode=True)
    assert -0.05 < v_off < 0.0


def _i_gate(vds, vg):
    b = p.CircuitBuilder()
    b.add_voltage_source("Vd", "d", "gnd", vds)
    b.add_voltage_source("Vg", "g", "gnd", vg)
    b.add_mosfet_level1("M1", "d", "gnd", "g", K, VT,
                         with_body_diode=False)
    res = p.simulate(b, t_end=1e-8, dt=1e-8, engine="pwl")
    return float(np.asarray(res.i("Vd"))[-1])


def test_cutoff_is_measured_against_the_lower_terminal():
    """Also physics the symmetrized law now gets right, and worth
    stating because it looks like a bug at first glance.

    With V_G = 0 and the drain pulled to −5 V, the gate is +5 V
    above the terminal that is now acting as the source, so the
    channel forms and the device conducts — the false-turn-on
    mechanism that makes negative gate drive necessary on some
    parts. The old forward-only law could not see it.

    OFF means below threshold against BOTH terminals.
    """
    assert abs(_i_gate(-5.0, 0.0)) > 1.0        # V_GD = +5 > V_T
    assert abs(_i_gate(-5.0, -10.0)) < 1e-3     # below both
    assert abs(_i_gate(+5.0, 0.0)) < 1e-3       # below both
