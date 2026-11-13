"""add-complementarity-mode-selection task 1.2: today's diode decision, characterised.

Ported from openspec/changes/add-complementarity-mode-selection/probes/today_dsed.py and re-measured
on this branch (exact propagator slice 1). The two failures the probe recorded on the older kernel
("did not settle after 64 rounds" at rest, "is chattering" at 0.996 ms on the DCM buck) are gone
here: the zero-band look-ahead decides them. What remains is SILENT: on the boost with V_th <= 1e-6
dsed's settle picks a different diode mode than the physics and lands 1.01 V low at R = 500 and
0.13 V low at R = 50 after 400 periods, while pwl and trbdf2 agree with the reference.

Reference: the prototype's LCP-driven exact simulation (tests/oracles/lcp_reference.py) of the same
circuit with R_on = 1 mOhm switch and diode (g_on = 1e3), 400 periods from rest:
R = 500 -> v_out = 44.72165676 V, R = 50 -> 23.80641613 V.

Slice 2: the default run's guard and mode_selection='lcp' both land on the reference; the wrong
numbers remain pinned for the settle alone (pulsim._dsed_dispatch.MODE_GUARD = False).
"""
import warnings

import numpy as np
import pytest

import pulsim as p

T = 1e-5
GON, GOFF = 1e3, 1e-9
REF_BOOST = {500.0: 44.72165676412844, 50.0: 23.80641612978717}


def _boost(vth, R):
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", 12.0)
    b.add_inductor("L", "vin", "sw", 100e-6)
    b.add_switch("S", "sw", "gnd", GON, GOFF)
    b.add_diode("D1", "sw", "out", GON, GOFF, vth)
    b.add_capacitor("C", "out", "gnd", 47e-6)
    b.add_resistor("R", "out", "gnd", R)
    fn = p.make_pwm_switch_fn(1 / T, 0.5, b.switch_index_of("S"), b.graph.num_switches, 0.0)
    return b, fn


def _buck(vth):
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", 24.0)
    b.add_switch("S", "vin", "sw", GON, GOFF)
    b.add_diode("D1", "gnd", "sw", GON, GOFF, vth)
    b.add_inductor("L", "sw", "vout", 100e-6)
    b.add_capacitor("C", "vout", "gnd", 47e-6)
    b.add_resistor("R", "vout", "gnd", 50.0)
    fn = p.make_pwm_switch_fn(1 / T, 0.3, b.switch_index_of("S"), b.graph.num_switches, 0.0)
    return b, fn


def _end(r, node):
    return float(np.asarray(r.v(node))[-1])


@pytest.fixture(autouse=True)
def _quiet():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


def test_buck_at_rest_with_zero_threshold_runs_on_dsed():
    b, _ = _buck(0.0)
    r = p.simulate(b, t_end=1e-6, engine="dsed",
                   switch_fn=lambda t: p.SwitchStateMask(b.graph.num_switches))
    assert abs(_end(r, "sw")) < 1e-9


def test_dcm_buck_4000_periods_with_zero_threshold_matches_reference_on_dsed():
    b, fn = _buck(0.0)
    r = p.simulate(b, t_end=4000 * T, engine="dsed", switch_fn=fn)
    # prototype with R_on = 1 mOhm: 8.996246971827961 V
    assert _end(r, "vout") == pytest.approx(8.996246971827961, abs=2e-6)


@pytest.mark.parametrize("R", [500.0, 50.0])
def test_boost_zero_threshold_pwl_and_trbdf2_match_reference(R):
    for eng, kw in (("pwl", {"dt": 1e-8}), ("trbdf2", {})):
        b, fn = _boost(0.0, R)
        r = p.simulate(b, t_end=400 * T, engine=eng, switch_fn=fn, **kw)
        assert _end(r, "out") == pytest.approx(REF_BOOST[R], abs=5e-3), eng


@pytest.mark.parametrize("R", [500.0, 50.0])
def test_boost_zero_threshold_dsed_settle_is_wrong_and_the_guard_repairs_it(R):
    """The settle alone (the guard switched off) is still silently wrong, 1.0 / 0.13 V low.
    Slice 2's default-mode guard (task 4) judges each settled mode and lets the LCP decide the
    events it fails, and 'lcp' decides every event: both land on the reference. How wrong the
    settle is depends on rounding, so it is bounded, not pinned (was pinned to 1e-6 at the
    macOS-arm64 value; R = 50: 23.673043 V there, 23.668123 V on Linux x86-64, 23.660933 V on
    Linux arm64)."""
    import pulsim._dsed_dispatch as DD
    b, fn = _boost(0.0, R)
    check = DD.check_diode_modes                 # the stored-sample guard would name it
    DD.MODE_GUARD, DD.check_diode_modes = False, (lambda *a, **k: None)
    try:
        v_settle = _end(p.simulate(b, t_end=400 * T, engine="dsed", switch_fn=fn), "out")
    finally:
        DD.MODE_GUARD, DD.check_diode_modes = True, check
    assert REF_BOOST[R] - v_settle > 0.1                # measured 1.01 / 0.133 ... 0.145 V
    r = p.simulate(b, t_end=400 * T, engine="dsed", switch_fn=fn)
    assert _end(r, "out") == pytest.approx(REF_BOOST[R], abs=1e-5)
    assert r.diagnostics["mode_guard"]["repairs"] > 0
    r_lcp = p.simulate(b, t_end=400 * T, engine="dsed", switch_fn=fn, mode_selection="lcp")
    assert _end(r_lcp, "out") == pytest.approx(REF_BOOST[R], abs=1e-5)


def test_boost_with_a_band_is_right_on_dsed():
    """Control: the same dsed run with V_th = 1e-3 lands on the reference (R = 500)."""
    b, fn = _boost(1e-3, 500.0)
    r = p.simulate(b, t_end=400 * T, engine="dsed", switch_fn=fn)
    assert _end(r, "out") == pytest.approx(REF_BOOST[500.0], abs=1e-3)
