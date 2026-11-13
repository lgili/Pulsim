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
    assert r.diagnostics["mode_guard"]["repairs"] == 2
    for name, an, ca in (("D1", "s", "p"), ("D2", "n", "s"), ("D3", "r", "p"), ("D4", "n", "r")):
        on = np.asarray(r.switch_states)[:, b.switch_index_of(name)].astype(bool)
        vd = np.asarray(r.v(an)) - np.asarray(r.v(ca))
        assert on.any() and (1e3 * vd[on]).min() > -1e-9      # each diode conducts, never in reverse
        assert vd[~on].max() < 1e-9                             # and blocks only reverse voltage
    # the class is a build-time property: 'lcp' reports the same. Up to 9 ms 'lcp' is right
    # (35.2286954424 V, pwl 35.2286954469 V); after that it is DECISIONS #21's open defect (11.73 V
    # on the slice's tree). Integration batch 6b: this tree's stored-sample guard names it --
    # D2 OFF while forward-biased from t = 10.008 ms -- so the wrong number is no longer silent.
    r_lcp = _run(_bridge_behind_ideal_transformer(), "dsed", t_end=0.009, mode_selection="lcp")
    assert r_lcp.diagnostics["mode_selection"]["passivity"] == cls
    ref9 = _run(_bridge_behind_ideal_transformer(), "pwl", t_end=0.009)
    assert _vpn(r_lcp)[-1] == pytest.approx(_vpn(ref9)[-1], abs=1e-7)
    from pulsim._dsed_dispatch import DiodeModeInconsistentError
    with pytest.raises(DiodeModeInconsistentError, match="'D2' is OFF while forward-biased"):
        _run(_bridge_behind_ideal_transformer(), "dsed", mode_selection="lcp")


def test_plain_rectifier_is_passive_reciprocal():
    b = p.CircuitBuilder()
    b.add_sine_voltage_source("V", "a", "gnd", 0.0, 10.0, 50.0)
    b.add_diode("D", "a", "o", 1e3, 1e-9)
    b.add_capacitor("C", "o", "gnd", 100e-6)
    b.add_resistor("R", "o", "gnd", 100.0)
    r = _run(b, "dsed")
    assert r.diagnostics["mode_guard"]["passivity"] == {
        "class": "passive_reciprocal", "device": "", "kind": ""}


def test_rectifier_fed_by_coupled_inductors_is_refused_by_the_extractor():
    # The coupled-inductor matrix is classified (PSD here, k = 0.99), but the dsed extraction has
    # no inductance-matrix block yet (add-exact-propagator-strategy task 7): refused by name,
    # before any mode is selected. pwl runs it.
    b = p.CircuitBuilder()
    b.add_sine_voltage_source("V", "a", "gnd", 0.0, 100.0, 50.0)
    b.add_transformer("T", "a", "gnd", "s", "r", 10e-3, 2.5e-3, 0.99)
    b.add_resistor("Rr", "r", "gnd", 1e6)
    b.add_diode("D1", "s", "o", 1e3, 1e-9)
    b.add_capacitor("C", "o", "r", 100e-6)
    b.add_resistor("R", "o", "r", 100.0)
    with pytest.raises(p.UnsupportedError) as ei:
        _run(b, "dsed")
    assert (ei.value.engine, ei.value.feature) == ("dsed", "extraction")
    r = _run(b, "pwl", dt=1e-6)
    assert np.isfinite(np.asarray(r.v("o"))).all()
