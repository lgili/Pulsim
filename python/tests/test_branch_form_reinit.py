"""add-branch-form-ideal-switch group 5 on `pwl`: dependent modes re-initialised
by the SDIRK x3 restart (design D4), recorded, and refused without the rule.

Oracle: `oracles/ideal_switch_reference.py` (the prototype, 40-digit probe
`probes/sdirk_restart_ideal.out`): the inductor cut-set at h = tau/20 records
49.3120 mJ from the run's own i_L(t-) = 9.93096483 A; the capacitor loop
lands both capacitors on 4.000000 V moving 6.0 uC and removing 24.0 uJ.
"""
import math
import warnings

import numpy as np
import pytest

import pulsim as p
from tests.oracles import ideal_switch_reference as ref

RULE = dict(discontinuity_restart=True)
TAU = 1e-3


def _cutset_builder():
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "vin", "gnd", 10.0)
    b.add_ideal_switch("S", "vin", "a", 0.0, 0.0)
    b.add_resistor("R", "a", "b", 1.0)
    b.add_inductor("L", "b", "gnd", 1e-3)
    return b


def _cutset(open_at=100, steps=106, **kw):
    b = _cutset_builder()
    h = TAU / 20
    k = b.switch_index_of("S")

    def fn(t):
        m = p.SwitchStateMask(1)
        m.set(k, t < open_at * h - 1e-3 * h)
        return m
    with warnings.catch_warnings():
        if kw.get("on_impulse") != "warn":
            warnings.simplefilter("ignore")
        return p.simulate(b, t_end=steps * h + 1e-3 * h, dt=h, engine="pwl",
                          switch_fn=fn, **kw), h


def test_cutset_without_the_rule_is_refused_by_name():
    with pytest.raises(p.UnsupportedError) as ei:
        _cutset()
    e = ei.value
    assert (e.engine, e.feature, e.device, e.kind) == ("pwl", "dependent_states", "S", "Switch")
    assert "discontinuity_restart=True" in str(e)


def test_cutset_is_reinitialised_and_recorded():
    r, h = _cutset(**RULE)
    t = np.asarray(r.times)
    iL, va = np.asarray(r.i("L")), np.asarray(r.v("a"))
    k0 = int(np.argmin(abs(t - 100 * h)))          # the sample at the opening
    o = ref.cutset("sdirk3", N=20, steps=106, open_at=100)
    # The engine starts with its own restart (the prototype with the
    # trapezoid: 9.93096483 A); it lands nearer the exact 9.93262053 A. The
    # record is checked against the run's own i_L(t-).
    assert iL[k0] == pytest.approx(10 * (1 - math.exp(-5)), rel=1e-5)
    assert abs(iL[k0] - 9.93262053) < abs(o["iL_minus"] - 9.93262053)
    assert np.all(iL[k0 + 1:] == 0.0)                              # exactly, from step 1
    assert np.max(np.abs(va[k0 + 2:])) <= 1e-9                     # no +-397 V
    assert len(r.reinit_records) == 1
    rec = r.reinit_records[0]
    assert rec.kind == "cut_set" and list(rec.devices) == ["S"] and list(rec.states) == ["L"]
    assert rec.t == pytest.approx(100 * h, rel=1e-12)
    assert abs(rec.lambda_) == pytest.approx(1e-3 * iL[k0], rel=1e-12)
    assert rec.dW == pytest.approx(0.5 * 1e-3 * iL[k0] ** 2, rel=1e-9)
    assert rec.dW == pytest.approx(49.3291e-3, rel=1e-5)   # prototype's run: 49.3120 mJ
    assert rec.projection_gap == 0.0
    assert rec.sample == k0 + 1                                    # the pulse sample
    # the pulse sample carries the impulse: about -L*i/(h*(1-gamma)) scale
    # scaled to this run's i_L(t-): the prototype's first-step sample
    # 959.018799 V (test_ideal_switch_reference) is linear in i_L(t-)
    assert va[k0 + 1] == pytest.approx(959.018799 * iL[k0] / o["iL_minus"], rel=1e-6)


def test_on_impulse_error_names_the_switch_and_the_inductor():
    with pytest.raises(p.UnsupportedError) as ei:
        _cutset(on_impulse="error", **RULE)
    assert (ei.value.feature, ei.value.device) == ("impulse", "S")
    assert "'S' makes L dependent" in str(ei.value)


def test_on_impulse_warn():
    b = _cutset_builder()
    with pytest.warns(UserWarning, match="impulsive re-initialisation"):
        r, _ = _cutset(on_impulse="warn", **RULE)
    assert len(r.reinit_records) == 1
    with pytest.raises(ValueError, match="on_impulse"):
        p.simulate(b, t_end=2e-4, dt=TAU / 20, engine="pwl", on_impulse="ignore",
                   switch_fn=lambda t: p.SwitchStateMask(1), **RULE)


def _caploop(h=1e-7, steps=5):
    b = p.CircuitBuilder()
    b.add_capacitor("C1", "x", "gnd", 1e-6, 10.0)
    b.add_capacitor("C2", "y", "gnd", 3e-6, 2.0)
    b.add_ideal_switch("S", "x", "y", 0.0, 1e-9)
    m = p.SwitchStateMask(1)
    m.set(0, True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # no preflight leak to ground (it would drain the pair at 3e-9 A)
        return p.simulate(b, t_end=steps * h + 1e-3 * h, dt=h, engine="pwl",
                          switch_fn=lambda t: m, auto_regularize=False, **RULE)


def test_capacitor_loop_is_reinitialised_and_recorded():
    r = _caploop()
    v1, v2 = np.asarray(r.v("x")), np.asarray(r.v("y"))
    assert v1[0] == 10.0 and v2[0] == 2.0
    assert np.max(np.abs(v1[1:] - 4.0)) <= 1e-12
    assert np.max(np.abs(v2[1:] - 4.0)) <= 1e-12
    iS = np.asarray(r.i("S"))
    assert np.max(np.abs(iS[2:])) < 1e-9
    o = ref.capacitor_loop("sdirk3", h=1e-7, steps=5)
    np.testing.assert_allclose(v1[1:], o[:, 0], rtol=0, atol=1e-12)
    (rec,) = r.reinit_records
    assert rec.kind == "capacitor_loop" and list(rec.devices) == ["S"]
    assert sorted(rec.states) == ["C1", "C2"]
    assert abs(rec.lambda_) == pytest.approx(6.0e-6, rel=1e-9)
    assert rec.dW == pytest.approx(24.0e-6, rel=1e-9)
    assert rec.t == 0.0 and rec.sample == 1


def test_a_consistent_dcm_turn_off_records_nothing():
    """An ideal diode turning off at i = 0 in a DCM buck: the cut-set it makes
    (switch and diode open around the inductor) is consistent."""
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", 24.0)
    b.add_ideal_switch("S", "vin", "sw", 0.0, 0.0)
    b.add_diode("D", "gnd", "sw", form="branch", R_on=0.0, g_off=0.0)
    b.add_inductor("L", "sw", "vout", 100e-6)
    b.add_capacitor("C", "vout", "gnd", 47e-6, ref.VC_REF)   # on the DCM orbit
    b.add_resistor("RL", "vout", "gnd", 50.0)
    T = 1e-5
    k = b.switch_index_of("S")

    def fn(t):
        m = p.SwitchStateMask(b.graph.num_switches)
        m.set(k, (t / T) % 1.0 < 0.3 - 1e-9)
        return m
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = p.simulate(b, t_end=20 * T, dt=T / 100, engine="pwl", switch_fn=fn,
                       **RULE)
    iL = np.asarray(r.i("L"))
    assert np.min(iL) >= -1e-12                     # the diode blocks
    sw = np.asarray(r.switch_states)
    idle = ~sw.any(axis=1)                           # switch and diode off
    assert idle.sum() > 300                          # DCM: idle intervals exist
    assert np.all(iL[idle] == 0.0)
    assert len(r.reinit_records) == 0


def test_trbdf2_refuses_the_cutset_by_name():
    b = _cutset_builder()
    with pytest.raises(p.UnsupportedError) as ei:
        p.simulate(b, t_end=1e-3, engine="trbdf2",
                   switch_fn=lambda t: p.SwitchStateMask(1))
    assert (ei.value.engine, ei.value.feature, ei.value.device) == (
        "trbdf2", "dependent_states", "S")


@pytest.mark.parametrize("engine", ["pwl", "trbdf2", "dsed", "exact", "auto"])
def test_on_impulse_value_checked_on_every_engine(engine):
    with pytest.raises(ValueError, match="on_impulse"):
        p.simulate(_cutset_builder(), t_end=1e-4, engine=engine, on_impulse="bogus")
