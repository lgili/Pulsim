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


# release: the stored megavolt is the rule-off engine's; under the default
# restart rule the cut is an impulse record and the trace stays at 48 V.
STORED = dict(discontinuity_restart=False)


@pytest.mark.parametrize("kw", [STORED, pytest.param({}, marks=pytest.mark.xfail(strict=True, reason=(
    "release finding F-Z-15: a restart-rule cut is named (F-Z-8 fixed) but judged at its right-limit row's i*R_off (2.4e12 V at sw), not at |lambda|/dt (2.4e5 V = 5000 x 48 V): voltage_sanity_factor cannot silence it and the kernel twin's peak is not L*i/dt; same with discontinuity_restart=True explicit")))], ids=["stored", "impulse"])
def test_the_megavolt_is_named(kw):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = p.simulate(opening_inductor(), t_end=1e-3, dt=1e-8,
                          switch_fn=_sw, **kw)
    # rule off: the stored megavolt; rule on: the cut is named instead (F-Z-8, 3e61b05d / 1abf156f)
    hits = [w for w in caught if ("largest voltage any source" if kw else "cut the") in
            str(w.message)]
    assert len(hits) == 1
    msg = str(hits[0].message)
    assert "sw" in msg                    # the node
    if kw:
        assert "48" in msg                # the scale it is judged against
        assert "snubber" in msg           # and what to do about it

    v = np.asarray(res.v("sw"))
    if kw:
        assert np.abs(v).max() > 1e5      # the premise really holds
    else:
        # the premise under the restart rule: no sample holds the spike, the
        # cut is ten impulse records of L*i = 2.4e-3 V*s; the warning names
        # who cut whom and reads the impulse over one step (2.4e5 V)
        assert np.abs(v).max() <= 48.0 * (1 + 1e-9)
        cuts = [r for r in res.reinit_records if r.kind == "cut_set"]
        assert len(cuts) == 10 and cuts[0].states == ["L"] and cuts[0].devices == ["S"]
        assert abs(cuts[0].lambda_) == pytest.approx(2.4e-3, rel=1e-3)
        assert "'S' cut the 2.4 A current of 'L'" in msg and "reinit_records" in msg
        assert res._implausible_voltage.peak == pytest.approx(2.4e5, rel=1e-3)
        assert res._implausible_voltage.source_scale == pytest.approx(48.0)
    assert res._implausible_voltage.node >= 0


@pytest.mark.xfail(strict=True, reason=(
    "release finding F-Z-15: a restart-rule cut is named (F-Z-8 fixed) but judged at its right-limit row's i*R_off (2.4e12 V at sw), not at |lambda|/dt (2.4e5 V = 5000 x 48 V): voltage_sanity_factor cannot silence it and the kernel twin's peak is not L*i/dt; same with discontinuity_restart=True explicit"))
def test_an_impulse_below_the_factor_says_nothing():
    """The impulse is judged as the stored spike was (|lambda| / dt against
    factor x the source scale): 2.4e5 V is 5000 x 48 V; a factor above that
    reports nothing, and the records stay."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = p.simulate(opening_inductor(), t_end=1e-3, dt=1e-8,
                          switch_fn=_sw, voltage_sanity_factor=6000.0)
    assert [w for w in caught
            if "largest voltage any source" in str(w.message)] == []
    assert getattr(res, "_implausible_voltage", None) is None
    assert len(res.reinit_records) == 10


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
        res = p.simulate(b, t_end=1e-3, dt=1e-8, switch_fn=_sw, **STORED)
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
        res = p.simulate(b, t_end=1e-3, dt=1e-8, switch_fn=_sw, **STORED)
    assert k.find_implausible_voltage(b.graph, b.pool, res).node >= 0
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
    assert "'S'" in hit and f"the {i_e:.4g} A current of 'L'" in hit
    assert f"½·L·i² = {rec.dW:.4g} J" in hit   # one state: dW is its own
    # the row itself is unchanged: the model's right limit (i continuous)
    v = p.with_event_limits(res)
    t = np.asarray(v.times)
    (j,) = [q for q in np.flatnonzero(t[1:] == np.nextafter(t[:-1], np.inf))
            if t[q] > 0]   # the cut's row (not the start's, F-Z-10)
    il = np.asarray(v.i("L"))
    assert il[j + 1] == pytest.approx(il[j], rel=1e-9)
    assert abs(np.asarray(v.v("sw"))[j + 1]) > 1e9
    assert res._implausible_voltage.node >= 0


def test_a_cut_set_names_the_inductive_state_not_a_capacitor_it_also_moved(monkeypatch):
    """F-Z-8 follow-up on half_bridge_transformer_dc: the cut-set record's states
    list every state the jump moved in entry order (Cdiv1, Cdiv2, T1.p, T1.s);
    the warning said "'S_h' cut the current of 'Cdiv1'" — a capacitor named as a
    cut current. It names the record's own kind (an inductive winding) now."""
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    monkeypatch.syspath_prepend(str(root / "benchmarks" / "tools"))
    pytest.importorskip("yaml")
    import v1_reader
    case = v1_reader.load(root / "benchmarks/circuits/magnetics/half_bridge_transformer_dc.yaml")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        r = case.run("pwl", discontinuity_restart=True)
    hits = [str(w.message) for w in caught if "cut the" in str(w.message)]
    assert hits
    assert "of 'Cdiv" not in hits[0] and "current of 'T1." in hits[0]
    rec = next(q for q in r.reinit_records if f"t = {q.t:.6g} s" in hits[0])
    assert rec.kind == "cut_set" and rec.states[0] == "Cdiv1"   # the listing that misled
    assert f"{rec.dW:.4g} J" in hits[0] and "½·L·i²" not in hits[0]   # several states
