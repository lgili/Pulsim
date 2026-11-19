"""add-complementarity-mode-selection task 5.1: the passivity classes the mode selection knows
(D3), on real circuits. Reference: pwl at dt = 1e-7 (the same conductance-form devices)."""

import warnings

import numpy as np
import pytest

import pulsim as p


def _run(b, engine, t_end=0.04, **kw):
    if engine == "pwl":
        kw.setdefault("dt", 1e-7)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return p.simulate(b, t_end=t_end, engine=engine, **kw)


def _bridge_behind_ideal_transformer():
    # 100 V / 50 Hz mains, ideal transformer n = 0.5 (secondary 50 V peak), a four-diode bridge
    # into 100 uF || 100 ohm. The DC side floats; 1 Mohm ties it to the secondary's return.
    b = p.CircuitBuilder()
    b.add_sine_voltage_source("V", "a", "gnd", 0.0, 100.0, 50.0)
    b.add_ideal_transformer("T1", "a", "gnd", "s", "r", 0.5)
    b.add_resistor("Rr", "r", "gnd", 1e6)
    for name, an, ca in (("D1", "s", "p"), ("D2", "n", "s"), ("D3", "r", "p"), ("D4", "n", "r")):
        b.add_diode(name, an, ca, 1e3, 1e-9)
    b.add_capacitor("C", "p", "n", 100e-6)
    b.add_resistor("R", "p", "n", 100.0)
    b.add_resistor("Rn", "n", "r", 1e6)
    return b


def _vpn(r):
    return np.asarray(r.v("p")) - np.asarray(r.v("n"))


def test_bridge_behind_an_ideal_transformer_is_passive_nonreciprocal():
    b = _bridge_behind_ideal_transformer()
    r = _run(b, "dsed")
    cls = {"class": "passive_nonreciprocal", "device": "T1", "kind": "IdealTransformer"}
    assert r.diagnostics["mode_guard"]["passivity"] == cls
    ref = _run(_bridge_behind_ideal_transformer(), "pwl")
    # pwl at 1e-7: 31.876241231725 V; dsed default (guard, 2 repairs) 31.876241227679 V
    assert _vpn(ref)[-1] == pytest.approx(31.8762412, abs=1e-6)
    assert _vpn(r)[-1] == pytest.approx(_vpn(ref)[-1], abs=1e-7)
    # The guard judges against the decision's significance bound (exact engine session 16).
    # Truth (pwl+split, dt 1e-8): at 5.981 / 25.981 ms D1 and D4 tie (they chatter for 0.27 us,
    # then both block), so either mode is admissible there and the repairs are ~1e-7 V; at
    # 21.7896 ms D4 turns on, and the settle's D4-OFF is a real error (repaired, 0.32 V).
    g = r.diagnostics["mode_guard"]
    assert 1 <= g["repairs"] <= 4
    real = [v for tr, v in zip(g["repair_times"], g["repair_violations"]) if abs(tr - 21.7896e-3) < 1e-6]
    assert real and min(real) > 0.1
    for tr, v in zip(g["repair_times"], g["repair_violations"]):
        assert abs(tr - 21.7896e-3) < 1e-6 or (
            min(abs(tr - 5.98113e-3), abs(tr - 25.98113e-3)) < 1e-7 and v < 1e-6)
    # The engine's zero band at 100 V (1e-12 of the largest node voltage): inside it a diode is
    # a tie. Since the guard repair of session 6 (complementarity DECISIONS #34), D2 is ON from
    # the 20 ms repair carrying down to -6.4e-8 A (-6.4e-11 V, inside the band), while the bus
    # agrees with pwl to 4e-12 V. The absolute -1e-9 A bound was below the engine's rounding.
    band = 1e-12 * 100.0
    for name, an, ca in (("D1", "s", "p"), ("D2", "n", "s"), ("D3", "r", "p"), ("D4", "n", "r")):
        on = np.asarray(r.switch_states)[:, b.switch_index_of(name)].astype(bool)
        vd = np.asarray(r.v(an)) - np.asarray(r.v(ca))
        assert on.any() and vd[on].min() > -band              # each diode conducts, never in reverse
        assert vd[~on].max() < 1e-9                             # and blocks only reverse voltage
    # the class is a build-time property: 'lcp' reports the same. Re-pinned at the merge of the
    # increment-form port model (was: right to 9 ms, then DiodeModeInconsistentError naming D2 at
    # 10.008 ms): the full 'lcp' run now lands on pwl, 31.876241228 V (more numbers:
    # test_mode_lcp.py::test_lcp_bridge_on_a_floating_source_is_not_decided_by_rounding)
    r_lcp = _run(_bridge_behind_ideal_transformer(), "dsed", mode_selection="lcp")
    assert r_lcp.diagnostics["mode_selection"]["passivity"] == cls
    assert _vpn(r_lcp)[-1] == pytest.approx(_vpn(ref)[-1], abs=1e-7)


def test_plain_rectifier_is_passive_reciprocal():
    b = p.CircuitBuilder()
    b.add_sine_voltage_source("V", "a", "gnd", 0.0, 10.0, 50.0)
    b.add_diode("D", "a", "o", 1e3, 1e-9)
    b.add_capacitor("C", "o", "gnd", 100e-6)
    b.add_resistor("R", "o", "gnd", 100.0)
    r = _run(b, "dsed")
    assert r.diagnostics["mode_guard"]["passivity"] == {
        "class": "passive_reciprocal", "device": "", "kind": ""}


def test_rectifier_fed_by_coupled_inductors_is_passive_reciprocal():
    # The coupled-inductor matrix is classified (PSD here, k = 0.99) and the dsed extraction takes
    # it (add-exact-propagator-strategy, coupled inductors): the run is passive-reciprocal and
    # follows pwl at a 1 us step (0.0007 V of a 50 V output measured).
    b = p.CircuitBuilder()
    b.add_sine_voltage_source("V", "a", "gnd", 0.0, 100.0, 50.0)
    b.add_transformer("T", "a", "gnd", "s", "r", 10e-3, 2.5e-3, 0.99)
    b.add_resistor("Rr", "r", "gnd", 1e6)
    b.add_diode("D1", "s", "o", 1e3, 1e-9)
    b.add_capacitor("C", "o", "r", 100e-6)
    b.add_resistor("R", "o", "r", 100.0)
    r = _run(b, "dsed")
    assert r.diagnostics["mode_guard"]["passivity"]["class"] == "passive_reciprocal"
    ref = _run(b, "pwl", dt=1e-6)
    t = np.asarray(r.times)
    v = np.asarray(r.v("o")) - np.asarray(r.v("r"))
    v_ref = np.interp(t, np.asarray(ref.times), np.asarray(ref.v("o")) - np.asarray(ref.v("r")))
    assert np.abs(v - v_ref).max() < 1e-3 * np.abs(v_ref).max()
