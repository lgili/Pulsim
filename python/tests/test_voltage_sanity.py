"""v2.0 Phase 2: 2.9 megavolts on a 48 V circuit, reported in silence.

An inductor whose conduction path opens produces, in an idealized
model, an unbounded voltage — `v = L·di/dt` with `di/dt` forced to
`-i/dt` in one step. Pulsim reported it finitely and without comment:

    Vin(48 V) — L(1 mH) — S ——| gnd,  S opening at 10 kHz
    max |v(sw)| = 2.9e+06 V,  isfinite everywhere

Nothing caught it. The inductor freeze and clamp guards watch the
CURRENT, and the current here stays at a believable 14 A. It is the
voltage that leaves physics, and no real circuit does this — the
switch avalanches, or its parasitic capacitance rings, or the
designer fitted a snubber.

The check names the node and stops there. Inserting a snubber would
mean choosing its value, which is a modelling decision belonging to
whoever knows the design's stand-off voltage — substituting one
silently is the failure this whole phase has been removing.
"""

import warnings

import numpy as np
import pytest

import pulsim as p


def opening_inductor(g_off=1e-12, L=1e-3):
    """Vin — L — S — gnd, and nothing else. When S opens, the
    inductor current has no path except the switch's own leakage."""
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", 48.0)
    b.add_inductor("L", "vin", "sw", L)
    b.add_switch("S", "sw", "gnd", 1e3, g_off)
    return b


def _sw(t, T=1e-4):
    m = p.SwitchStateMask(1)
    m.set(0, (t % T) < 0.5 * T)
    return m


def test_the_megavolt_is_named():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = p.simulate(opening_inductor(), t_end=1e-3, dt=1e-8,
                          switch_fn=_sw)
    hits = [w for w in caught if "largest voltage any source" in
            str(w.message)]
    assert len(hits) == 1
    msg = str(hits[0].message)
    assert "sw" in msg                    # the node
    assert "48" in msg                    # the scale it is judged against
    assert "snubber" in msg               # and what to do about it

    v = np.asarray(res.v("sw"))
    assert np.abs(v).max() > 1e5          # the premise really holds
    assert res._implausible_voltage.node >= 0


def test_an_ordinary_converter_says_nothing():
    """No false positives. A boost legitimately exceeds its input,
    and the 100x default has to accommodate that."""
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", 12.0)
    b.add_inductor("L", "vin", "sw", 100e-6)
    b.add_switch("S", "sw", "gnd", 1e3, 1e-9)
    b.add_diode("D", "sw", "vout", 1e3, 1e-9)
    b.add_capacitor("C", "vout", "gnd", 47e-6)
    b.add_resistor("Rl", "vout", "gnd", 20.0)

    def sw2(t, T=2e-5):
        m = p.SwitchStateMask(2)
        m.set(0, (t % T) < 0.5 * T)
        return m

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = p.simulate(b, t_end=2e-3, dt=1e-8, switch_fn=sw2)
    assert [w for w in caught
            if "largest voltage any source" in str(w.message)] == []
    vout = np.asarray(res.v("vout"))
    assert vout.max() > 12.0              # it really is boosting
    assert getattr(res, "_implausible_voltage", None) is None


def test_the_scan_is_read_only_and_reports_the_worst_node():
    from pulsim import _pulsim as k

    b = opening_inductor()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = p.simulate(b, t_end=1e-3, dt=1e-8, switch_fn=_sw)
    before = np.array(res.states, copy=True)

    f = k.find_implausible_voltage(b.graph, b.pool, res)
    assert f.node >= 0
    assert f.source_scale == pytest.approx(48.0)
    assert f.peak > 100 * f.source_scale
    assert 0.0 < f.t_peak <= 1e-3
    # It observes; it does not touch the trace.
    np.testing.assert_array_equal(np.asarray(res.states), before)


def test_a_generous_factor_can_be_asked_for():
    """The bound is a judgement, so it is a parameter. A factor above
    the observed ratio must report nothing."""
    from pulsim import _pulsim as k

    b = opening_inductor()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = p.simulate(b, t_end=1e-3, dt=1e-8, switch_fn=_sw)
    assert k.find_implausible_voltage(b.graph, b.pool, res,
                                       factor=1e9).node == -1


def test_a_restart_rule_cut_is_named_not_called_a_node_voltage():
    """F-Z-8: under the restart rule an inductor current cut by an ideal
    switch is an impulse record, and the right-limit row at the cut holds the
    conductance-form model's own limit (i·R_off for the L/R_off mode, 4.8e10 V
    here). The warning names the cut — switch, inductor, current, the energy
    ½·L·i² booked once — instead of a 'largest voltage' node finding. Closed
    form: Vin — R — L — S(R_on) charging, i(t_e) = V/(R+R_on)·(1 − e^(−t_e/τ))."""
    V, R, L, R_on, t_e = 48.0, 5.0, 1e-3, 1e-3, 1e-3
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", V)
    b.add_resistor("R", "vin", "a", R)
    b.add_inductor("L", "a", "sw", L)
    b.add_switch("S", "sw", "gnd", 1.0 / R_on, 1e-12)

    def sf(t):
        m = p.SwitchStateMask(1)
        m.set(0, t < t_e)
        return m

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = p.simulate(b, t_end=2e-3, dt=1e-6, switch_fn=sf, engine="pwl",
                         discontinuity_restart=True)
    i_e = V / (R + R_on) * (1.0 - np.exp(-t_e * (R + R_on) / L))
    (rec,) = res.reinit_records
    # trapezoid at h/τ = 5e-3: (h/τ)²/12 = 2e-6
    assert rec.dW == pytest.approx(0.5 * L * i_e ** 2, rel=1e-5)
    msgs = [str(w.message) for w in caught]
    assert not [m for m in msgs if "largest voltage any source" in m]
    (hit,) = [m for m in msgs if "cut the" in m]
    assert "'S'" in hit and "'L'" in hit and f"{i_e:.4g} A" in hit
    assert f"{rec.dW:.4g} J" in hit
    # the row itself is unchanged: the model's right limit (i continuous)
    v = p.with_event_limits(res)
    t = np.asarray(v.times)
    (j,) = [q for q in np.flatnonzero(t[1:] == np.nextafter(t[:-1], np.inf))
            if t[q] > 0]   # the cut's row (not the start's, F-Z-10)
    il = np.asarray(v.i("L"))
    assert il[j + 1] == pytest.approx(il[j], rel=1e-9)
    assert abs(np.asarray(v.v("sw"))[j + 1]) > 1e9
    assert res._implausible_voltage.node >= 0
