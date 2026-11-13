"""add-branch-form-ideal-switch: fixes from the group-5 review (H1, H3, H4, M2, M4, M5, M6, L2)."""

import warnings

import numpy as np
import pytest

import pulsim as p

RULE = dict(discontinuity_restart=True)


def _quiet(f):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return f()


def _mask(*bits):
    m = p.SwitchStateMask(len(bits))
    for i, v in enumerate(bits):
        m.set(i, bool(v))
    return m


def test_h1_forward_biased_ideal_diode_across_a_source_is_refused():
    # Before: released OFF, re-solved, cycle-detected as a breach on 100 of
    # 101 steps; the run went on with the diode stuck OFF.
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "a", "gnd", 10.0)
    b.add_diode("D", "a", "gnd", form="branch")
    b.add_resistor("R", "a", "c", 1.0)
    b.add_capacitor("C", "c", "gnd", 1e-6)
    with pytest.raises(p.UnsupportedError) as ei:
        _quiet(lambda: p.simulate(b, 1e-4, dt=1e-6, engine="pwl"))
    assert (ei.value.feature, ei.value.device, ei.value.kind) == (
        "ideal_voltage_loop", "D", "Diode")


def test_h3_nonlinear_state_with_an_ideal_element_is_refused_by_name():
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "in", "gnd", 10.0)
    b.add_ideal_switch("S", "in", "a")
    b.add_resistor("R", "a", "m", 1.0)
    b.add_saturable_inductor("Ls", "m", "gnd", 1e-3, 20.0)
    with pytest.raises(p.UnsupportedError) as ei:
        _quiet(lambda: p.simulate(b, 1e-4, dt=1e-6, engine="pwl",
                                  switch_fn=lambda t: _mask(t < 5e-5), **RULE))
    assert (ei.value.feature, ei.value.device) == ("dependent_states", "Ls")


def _rl_ideal():
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "a", "gnd", 10.0)
    b.add_ideal_switch("S", "a", "m")
    b.add_resistor("R", "m", "o", 1.0)
    b.add_inductor("L", "o", "gnd", 1e-3)
    return b


def test_h4_auto_routes_an_ideal_switch_away_from_trbdf2():
    fn = lambda t: _mask(True)  # noqa: E731
    with pytest.raises(ValueError, match="pass dt"):
        p.simulate(_rl_ideal(), 1e-4, engine="auto", switch_fn=fn)
    r = _quiet(lambda: p.simulate(_rl_ideal(), 1e-4, dt=1e-6, engine="auto", switch_fn=fn))
    assert r.engine_used == "pwl"


def test_m2_loop_closed_by_a_path_switch_is_recorded():
    # C1 (8 V) -- S1 (closes at t1) -- S2 (always closed) -- C2 (0 V).
    b = p.CircuitBuilder()
    b.add_capacitor("C1", "a", "gnd", 1e-6, 8.0)
    b.add_capacitor("C2", "b", "gnd", 1e-6, 0.0)
    b.add_resistor("R1", "a", "gnd", 1e9)
    b.add_resistor("R2", "b", "gnd", 1e9)
    b.add_ideal_switch("S2", "c", "b")
    b.add_ideal_switch("S1", "a", "c")
    k1, k2 = b.switch_index_of("S1"), b.switch_index_of("S2")

    def fn(t):
        m = p.SwitchStateMask(2)
        m.set(k2, True)
        m.set(k1, t > 5e-6)
        return m
    r = _quiet(lambda: p.simulate(b, 2e-5, dt=1e-6, engine="pwl", switch_fn=fn,
                                  auto_regularize=False, on_impulse="record", **RULE))
    recs = list(r.reinit_records)
    assert len(recs) == 1 and recs[0].kind == "capacitor_loop"
    assert set(recs[0].devices) == {"S1", "S2"}
    assert recs[0].dW == pytest.approx(16e-6, rel=1e-9)   # ½·(C/2)·8²


def test_m4_primary_behind_an_open_switch_is_not_floating():
    # Forward-converter shape: the primary node is held by v_p = v_s / n.
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "in", "gnd", 10.0)
    b.add_ideal_switch("S", "in", "p")
    b.add_ideal_transformer("T", "p", "gnd", "s", "gnd", 1.0)
    b.add_resistor("Rl", "s", "o", 10.0)
    b.add_capacitor("Co", "o", "gnd", 1e-6)
    r = _quiet(lambda: p.simulate(b, 2e-5, dt=1e-6, engine="pwl",
                                  switch_fn=lambda t: _mask(t < 1e-5)))
    assert np.all(np.isfinite(np.asarray(r.v("o"))))


def test_m5_loss_summary_says_it_skips_an_ideal_element():
    b = _rl_ideal()
    r = _quiet(lambda: p.simulate(b, 1e-5, dt=1e-6, engine="pwl",
                                  switch_fn=lambda t: _mask(True)))
    with pytest.warns(UserWarning, match="'S' is an ideal branch-form element"):
        p.device_loss_summary(b, r, switch_fn=lambda t: _mask(True))


def test_m6_ccm_buck_ideal_pair_runs_without_the_restart_rule():
    # The turn-off step keeps a solve whose mask is the cut-set (S off,
    # D off) while the diode already carries i_L: no impulse, no refusal.
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "in", "gnd", 24.0)
    b.add_ideal_switch("S", "in", "sw")
    b.add_diode("D", "gnd", "sw", form="branch")
    b.add_inductor("L", "sw", "o", 100e-6)
    b.add_capacitor("C", "o", "gnd", 100e-6)
    b.add_resistor("R", "o", "gnd", 1.0)
    T = 1e-5
    n, k = b.graph.num_switches, b.switch_index_of("S")   # D has a bit too

    def fn(t):
        m = p.SwitchStateMask(n)
        m.set(k, (t % T) < 0.5 * T)
        return m
    r = _quiet(lambda: p.simulate(b, 20 * T, dt=T / 50, engine="pwl", switch_fn=fn))
    assert len(r.reinit_records) == 0
    assert r.counters["ideal_loop_releases"] > 0  # each turn-on releases D
    assert np.asarray(r.i("L"))[-1] > 0.5        # CCM: never reaches zero


def test_l2_dc_start_checks_the_converged_mode():
    # Vsrc - D - out - C with no load: all-diodes-OFF leaves `out` floating,
    # the DC diode iteration turns D on. D on makes a source-capacitor loop
    # (restart rule); started consistent at DC it makes no record.
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "in", "gnd", 10.0)
    b.add_diode("D", "in", "out", form="branch")
    b.add_capacitor("C", "out", "gnd", 1e-6)
    r = _quiet(lambda: p.simulate(b, 1e-5, dt=1e-6, engine="pwl", start_from_dc_op=True,
                                  **RULE))
    assert np.asarray(r.v("out"))[0] == pytest.approx(10.0, abs=1e-9)
    assert len(r.reinit_records) == 0
