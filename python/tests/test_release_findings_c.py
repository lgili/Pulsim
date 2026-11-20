"""Release findings, group C: the new mode selection (``mode_selection='auto'``,
the complementarity / LCP selector). Each finding reproduced with the option set
explicitly on the PR line; notes and traces in
openspec/changes/update-number-changing-release/fixes/group-c.md.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

import pulsim as p


def _quiet(f):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return f()


# ---------------------------------------------------------------------------
# F-2-3 / F-5-9: a bridge diode held OFF while forward-biased (fixed engine, 'auto')
# ---------------------------------------------------------------------------

def _bridge(floating):
    """325 V / 50 Hz bridge, 1 mOhm source, 470 uF || 1 kOhm; the AC side floats
    (F-2-3, V_th = 0) or the DC side floats (F-5-9, V_th = 0.7)."""
    b = p.CircuitBuilder()
    if floating:
        b.add_sine_voltage_source("Vac", "s", "gnd", v_dc=0.0, v_amplitude=325.0,
                                  frequency=50.0, phase=0.0)
        b.add_resistor("Rs", "s", "a", 1e-3)
        p.topology.add_bridge_rectifier(b, "BR", ac_a="a", ac_b="gnd", dc_pos="p",
                                        dc_neg="n", V_th=0.7)
        b.add_capacitor("C", "p", "n", 470e-6)
        b.add_resistor("RL", "p", "n", 1000.0)
    else:
        b.add_sine_voltage_source("Vac", "s", "b", v_dc=0.0, v_amplitude=325.0,
                                  frequency=50.0, phase=0.0)
        b.add_resistor("Rs", "s", "a", 1e-3)
        p.topology.add_bridge_rectifier(b, "BR", ac_a="a", ac_b="b", dc_pos="p",
                                        dc_neg="gnd", V_th=0.0)
        b.add_capacitor("C", "p", "gnd", 470e-6)
        b.add_resistor("RL", "p", "gnd", 1000.0)
    return b


def _forward_biased_off(b, r):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        p.device_loss_summary(b, r)
    return [str(x.message) for x in w if "OFF while forward-biased" in str(x.message)]


@pytest.mark.parametrize("floating", [False, True], ids=["F-2-3", "F-5-9"])
def test_the_lcp_step_stores_the_state_it_decided_on(floating):
    """The LCP decides on the islands' KCL-corrected solve (a floating AC side /
    DC bus held by 1e-9 S leaks); the step stored the raw solve, whose common mode
    is the factor's rounding (1.6 mV, 13 mV): a decided-OFF diode read forward-biased.
    Truth for the all-OFF samples of the AC-floating bridge: KCL of the island
    {s, a, b} over the four equal leaks, v_a + v_b = v_p (a closed form)."""
    b = _bridge(floating)
    r = _quiet(lambda: p.simulate(b, 0.06, engine="pwl", dt=1e-6, mode_selection="auto"))
    assert _forward_biased_off(b, r) == []
    if not floating:
        bits = np.asarray([list(s) for s in r.switch_states])
        off = ~bits.any(axis=1)
        va, vb, vp = (np.asarray(r.v(n))[off] for n in ("a", "b", "p"))
        assert off.sum() > 1000
        assert np.max(np.abs(va + vb - vp)) <= 1e-9 * 325.0


# ---------------------------------------------------------------------------
# F-4-1 / F-3-2: one classification of not-posed circuits, by the user's name
# ---------------------------------------------------------------------------

def _ideal_loop(clamp):
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "a", "gnd", 10.0)
    if clamp:
        b.add_voltage_source("Vc", "k", "gnd", 3.0)
        b.add_diode("D", "a", "k", form="branch", R_on=0.0, g_off=0.0)
    else:
        b.add_diode("D", "a", "gnd", form="branch", R_on=0.0, g_off=0.0)
    b.add_resistor("R", "a", "gnd", 1.0)
    return b


@pytest.mark.parametrize("mode", ["legacy", "auto"])
@pytest.mark.parametrize("engine", ["dsed", "pwl"])
@pytest.mark.parametrize("clamp", [False, True])
def test_an_ideal_voltage_loop_is_named_by_every_selector(engine, mode, clamp):
    kw = {"dt": 1e-7} if engine == "pwl" else {}
    with pytest.raises(p.UnsupportedError) as ei:
        _quiet(lambda: p.simulate(_ideal_loop(clamp), 1e-6, engine=engine,
                                  mode_selection=mode, **kw))
    assert (ei.value.feature, ei.value.device, ei.value.kind) == (
        "ideal_voltage_loop", "D", "Diode")
    assert "add the missing path" not in str(ei.value)


@pytest.mark.parametrize("mode", ["legacy", "auto"])
def test_a_refusal_device_is_the_user_name(mode):
    """F-3-2 (the parked test of group G): an ideal diode across a source."""
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "a", "gnd", 10.0)
    b.add_diode("D", "a", "gnd", form="branch")
    b.add_resistor("R", "a", "c", 1.0)
    b.add_capacitor("C", "c", "gnd", 1e-6)
    with pytest.raises(p.UnsupportedError) as ei:
        _quiet(lambda: p.simulate(b, 1e-4, dt=1e-6, engine="pwl", mode_selection=mode))
    assert (ei.value.feature, ei.value.device, ei.value.kind) == (
        "ideal_voltage_loop", "D", "Diode")


def test_a_consistent_mode_refusal_names_the_devices():
    """F-3-2 on the other class that carries diodes: a current source into an
    ideal blocking diode (no path at all) is `consistent_mode`, its .device the
    user's names, the internal label only in the message."""
    b = p.CircuitBuilder()
    b.add_current_source("I", "gnd", "a", 1.0)
    b.add_diode("D", "gnd", "a", form="branch", R_on=0.0, g_off=0.0)
    b.add_resistor("R", "a", "c", 1.0)
    b.add_capacitor("C", "c", "gnd", 1e-6)
    b.add_diode("D2", "c", "gnd", form="branch", R_on=0.0, g_off=0.0)
    with pytest.raises(p.UnsupportedError) as ei:
        _quiet(lambda: p.simulate(b, 1e-5, dt=1e-6, engine="pwl", mode_selection="auto"))
    assert "branch #" not in ei.value.device
    assert ei.value.kind == "Diode"
