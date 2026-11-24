"""Task 4.2 — online electrothermal electrical feedback (V_on -> V_f, g_on).

Verifies that when ``electrical_feedback=True`` on ``set_device_losses``:
- two paralleled devices with unequal R_th share current equally with feedback
  off and unequally with feedback on (positive TC: hotter device takes less
  current; negative TC: hotter device takes more current);
- ``electrical_refreshes`` is reported on ``DeviceLossRecord`` and in
  ``res.device_energy()``;
- ``dT_refresh`` governs when parametric refactoring triggers;
- the caller's ``CircuitBuilder`` is never mutated across repeated runs;
- snapshot / resume preserves the feedback state;
- invalid configurations and engines without online feedback (``trbdf2``,
  ``dsed``, ``bdf1``) are refused cleanly by name.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import pulsim as p

T_AMB = 25.0


def _const_mask(bits: list[bool]):
    m = p.SwitchStateMask(len(bits))
    for k, val in enumerate(bits):
        m.set(k, val)
    return p.make_constant_switch_fn(m)


def _parallel_switches(*, feedback: bool, tc: float = 0.01,
                       dT_refresh: float = 5.0,
                       conduction: str = "electrical",
                       with_sense_resistors: bool = False):
    """Two identical paralleled switches (10 S at 25 C) sharing 20 A with
    unequal thermal resistances: S1 on 1 K/W, S2 on 5 K/W."""
    b = p.CircuitBuilder()
    b.add_current_source("I", "a", "gnd", 20.0)
    if with_sense_resistors:
        b.add_resistor("Rs1", "a", "n1", 1e-3)
        b.add_switch("S1", "n1", "gnd", 10.0, 1e-9)
        b.add_resistor("Rs2", "a", "n2", 1e-3)
        b.add_switch("S2", "n2", "gnd", 10.0, 1e-9)
    else:
        b.add_switch("S1", "a", "gnd", 10.0, 1e-9)
        b.add_switch("S2", "a", "gnd", 10.0, 1e-9)
    b.add_voltage_source("Vamb", "amb", "gnd", T_AMB)
    b.add_resistor("Rth1", "tj1", "amb", 1.0)
    b.add_capacitor("Cth1", "tj1", "amb", 5e-3)
    b.add_resistor("Rth2", "tj2", "amb", 5.0)
    b.add_capacitor("Cth2", "tj2", "amb", 1e-3)

    # V_on(i, Tj) = 0.1 * (1 + tc * (Tj - 25)) * i
    r25 = 0.1
    r125 = 0.1 * (1.0 + tc * 100.0)
    v_on = p.ConductionTable(
        [0.0, 10.0, 20.0], [25.0, 125.0],
        [0.0, 0.0, 10.0 * r25, 10.0 * r125, 20.0 * r25, 20.0 * r125],
    )
    for dev, junc in (("S1", "tj1"), ("S2", "tj2")):
        b.set_device_losses(
            dev,
            junction=junc,
            V_on=v_on,
            conduction=conduction,
            electrical_feedback=feedback,
            dT_refresh=dT_refresh,
        )
    all_on = _const_mask([True, True])
    return b, all_on


def _parallel_diodes(*, feedback: bool, dT_refresh: float = 2.0):
    """Two identical paralleled PWL diodes with negative TC sharing 20 A
    through unequal thermal paths (1 K/W vs 5 K/W)."""
    b = p.CircuitBuilder()
    b.add_current_source("I", "a", "gnd", 20.0)
    b.add_resistor("Rs1", "a", "n1", 1e-3)
    b.add_diode("D1", "n1", "gnd", 20.0, 1e-9, 0.8, 0.8)
    b.add_resistor("Rs2", "a", "n2", 1e-3)
    b.add_diode("D2", "n2", "gnd", 20.0, 1e-9, 0.8, 0.8)
    b.add_voltage_source("Vamb", "amb", "gnd", T_AMB)
    b.add_resistor("Rth1", "tj1", "amb", 1.0)
    b.add_capacitor("Cth1", "tj1", "amb", 5e-3)
    b.add_resistor("Rth2", "tj2", "amb", 5.0)
    b.add_capacitor("Cth2", "tj2", "amb", 1e-3)

    # Negative TC: V_f drops 0.8 V -> 0.6 V, R_on drops 0.05 -> 0.03 Ohm at 125 C
    # PLECS-style V(0)=0, V(10)=V_f + 10*R, V(20)=V_f + 20*R
    v_on = p.ConductionTable(
        [0.0, 10.0, 20.0], [25.0, 125.0],
        [0.0, 0.0, 0.8 + 0.5, 0.6 + 0.3, 0.8 + 1.0, 0.6 + 0.6],
    )
    for dev, junc in (("D1", "tj1"), ("D2", "tj2")):
        b.set_device_losses(
            dev,
            junction=junc,
            V_on=v_on,
            conduction="electrical",
            electrical_feedback=feedback,
            dT_refresh=dT_refresh,
        )
    return b


@pytest.mark.parametrize("with_sense", [False, True])
def test_paralleled_switches_positive_tc_unequal_rth(with_sense):
    """Task 4.2: identical paralleled switches with unequal R_th share current
    equally when electrical_feedback=False, and unequally (hotter device takes
    LESS current for positive TC) when electrical_feedback=True."""
    b_off, fn = _parallel_switches(feedback=False, with_sense_resistors=with_sense)
    r_off = p.simulate(b_off, 0.03, dt=1e-4, switch_fn=fn)
    i1_off = float(r_off.i("S1")[-1])
    i2_off = float(r_off.i("S2")[-1])
    assert abs(i1_off - i2_off) < 1e-12
    assert i1_off == pytest.approx(10.0, rel=1e-9)
    assert r_off.device_energy("S1")["electrical_refreshes"] == 0
    assert r_off.device_energy("S2")["electrical_refreshes"] == 0
    assert r_off._device_losses[0].electrical_refreshes == 0

    b_on, fn_on = _parallel_switches(feedback=True, with_sense_resistors=with_sense)
    r_on = p.simulate(b_on, 0.03, dt=1e-4, switch_fn=fn_on)
    i1_on = float(r_on.i("S1")[-1])
    i2_on = float(r_on.i("S2")[-1])
    # KCL holds even for conductance-form switches reconstructed via res.i()
    assert i1_on + i2_on == pytest.approx(20.0, rel=1e-9)
    if with_sense:
        assert i1_on == pytest.approx(float(r_on.i("Rs1")[-1]), rel=1e-12)
        assert i2_on == pytest.approx(float(r_on.i("Rs2")[-1]), rel=1e-12)
    # Device 2 has 5x higher R_th -> hotter -> higher R_on -> less current
    assert float(r_on.v("tj2")[-1]) > float(r_on.v("tj1")[-1]) + 10.0
    assert i1_on > i2_on + 1.0
    e1 = r_on.device_energy("S1")
    e2 = r_on.device_energy("S2")
    assert e1["electrical_refreshes"] > 0
    assert e2["electrical_refreshes"] > 0
    assert e2["electrical_refreshes"] >= e1["electrical_refreshes"]
    # Energy balance closes within 1e-12
    assert r_on.heat_injected + r_on.heat_pending == pytest.approx(
        e1["E_cond_electrical"] + e2["E_cond_electrical"], rel=1e-12
    )
    # Re-running the same CircuitBuilder is reproducible (builder not mutated)
    r_on2 = p.simulate(b_on, 0.03, dt=1e-4, switch_fn=fn_on)
    assert np.array_equal(r_on2.v("tj1"), r_on.v("tj1"))
    assert np.array_equal(r_on2.v("tj2"), r_on.v("tj2"))
    assert r_on2.device_energy("S2") == e2


def test_paralleled_diodes_negative_tc_crowds_current_into_hotter_device():
    """Negative-TC paralleled diodes: feedback shifts V_f and g_on so the
    hotter diode carries MORE current (thermal current crowding)."""
    b_off = _parallel_diodes(feedback=False)
    r_off = p.simulate(b_off, 0.03, dt=1e-4)
    i1_off = float(r_off.i("D1")[-1])
    i2_off = float(r_off.i("D2")[-1])
    assert abs(i1_off - i2_off) < 1e-12
    assert r_off.device_energy("D1")["electrical_refreshes"] == 0
    assert r_off.device_energy("D2")["electrical_refreshes"] == 0

    b_on = _parallel_diodes(feedback=True, dT_refresh=2.0)
    r_on = p.simulate(b_on, 0.03, dt=1e-4)
    i1_on = float(r_on.i("D1")[-1])
    i2_on = float(r_on.i("D2")[-1])
    assert i1_on == pytest.approx(float(r_on.i("Rs1")[-1]), rel=1e-12)
    assert i2_on == pytest.approx(float(r_on.i("Rs2")[-1]), rel=1e-12)
    assert i1_on + i2_on == pytest.approx(20.0, rel=1e-9)
    # Hotter D2 has lower V_f and lower R_on -> crowds MORE current
    assert float(r_on.v("tj2")[-1]) > float(r_on.v("tj1")[-1]) + 20.0
    assert i2_on > i1_on + 1.0
    assert r_on.device_energy("D1")["electrical_refreshes"] > 0
    assert r_on.device_energy("D2")["electrical_refreshes"] > 0


def test_dt_refresh_threshold_governs_refresh_count():
    """A threshold larger than ΔTj performs zero refreshes; tighter thresholds
    trigger more refreshes."""
    b_big, fn = _parallel_switches(feedback=True, dT_refresh=500.0)
    r_big = p.simulate(b_big, 0.03, dt=1e-4, switch_fn=fn)
    assert r_big.device_energy("S1")["electrical_refreshes"] == 0
    assert r_big.device_energy("S2")["electrical_refreshes"] == 0
    assert abs(float(r_big.i("S1")[-1]) - float(r_big.i("S2")[-1])) < 1e-12

    b_coarse, fn_c = _parallel_switches(feedback=True, dT_refresh=15.0)
    b_fine, fn_f = _parallel_switches(feedback=True, dT_refresh=2.0)
    r_coarse = p.simulate(b_coarse, 0.03, dt=1e-4, switch_fn=fn_c)
    r_fine = p.simulate(b_fine, 0.03, dt=1e-4, switch_fn=fn_f)
    assert (r_fine.device_energy("S2")["electrical_refreshes"]
            > r_coarse.device_energy("S2")["electrical_refreshes"] > 0)


def test_electrical_feedback_resume_matches_uninterrupted():
    """Splitting and resuming a run with electrical_feedback=True preserves
    refreshed parameters, T_ref, and refresh counts."""
    b, fn = _parallel_switches(feedback=True, dT_refresh=5.0,
                               with_sense_resistors=True)
    full = p.simulate(b, 0.03, dt=1e-4, switch_fn=fn)
    seg1 = p.simulate(b, 0.015, dt=1e-4, switch_fn=fn)
    seg2 = p.simulate(b, 0.03, t_start=0.015, dt=1e-4, switch_fn=fn,
                      resume_from=seg1.final_snapshot)
    assert float(seg2.v("tj1")[-1]) == pytest.approx(float(full.v("tj1")[-1]), rel=1e-11)
    assert float(seg2.v("tj2")[-1]) == pytest.approx(float(full.v("tj2")[-1]), rel=1e-11)
    assert float(seg2.i("Rs1")[-1]) == pytest.approx(float(full.i("Rs1")[-1]), rel=1e-11)
    assert float(seg2.i("Rs2")[-1]) == pytest.approx(float(full.i("Rs2")[-1]), rel=1e-11)
    assert seg2.device_energy("S1")["electrical_refreshes"] == full.device_energy("S1")["electrical_refreshes"]
    assert seg2.device_energy("S2")["electrical_refreshes"] == full.device_energy("S2")["electrical_refreshes"]


def test_electrical_feedback_validation_refusals():
    """Invalid electrical_feedback specs are refused at set_device_losses by name."""
    b = p.CircuitBuilder()
    b.add_current_source("I", "a", "gnd", 10.0)
    b.add_switch("S1", "a", "gnd", 10.0, 1e-9)
    b.add_voltage_source("Vamb", "amb", "gnd", T_AMB)
    b.add_resistor("Rth", "tj", "amb", 1.0)
    b.add_capacitor("Cth", "tj", "amb", 1e-3)
    v_on = p.ConductionTable([0.0, 10.0], [25.0, 125.0], [0.0, 0.0, 1.0, 2.0])

    with pytest.raises(ValueError, match="S1.*V_on"):
        b.set_device_losses("S1", junction="tj", electrical_feedback=True)
    with pytest.raises(ValueError, match="S1.*junction"):
        b.set_device_losses("S1", Tj=25.0, V_on=v_on, electrical_feedback=True)
    for bad_dt in (0.0, -5.0, math.nan, math.inf):
        with pytest.raises(ValueError, match="dT_refresh"):
            b.set_device_losses("S1", junction="tj", V_on=v_on,
                                electrical_feedback=True, dT_refresh=bad_dt)


@pytest.mark.parametrize("engine", ["trbdf2", "dsed", "bdf1"])
def test_non_pwl_engines_refuse_electrical_feedback_by_name(engine):
    """Engines without online electrical feedback refuse it with UnsupportedError
    naming the engine, feature='electrical_feedback', and device."""
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "in", "gnd", 10.0)
    b.add_switch("S1", "in", "o", 10.0, 1e-9)
    b.add_resistor("RL", "o", "gnd", 1.0)
    b.add_voltage_source("Vamb", "amb", "gnd", T_AMB)
    b.add_resistor("Rth", "tj", "amb", 1.0)
    b.add_capacitor("Cth", "tj", "gnd", 1e-3, T_AMB)
    v_on = p.ConductionTable([0.0, 10.0], [25.0, 125.0], [0.0, 0.0, 1.0, 2.0])
    b.set_device_losses("S1", junction="tj", V_on=v_on, electrical_feedback=True)
    fn = _const_mask([True])

    with pytest.raises(p.UnsupportedError) as ei:
        if engine == "bdf1":
            p._pulsim.run_transient_bdf1(
                b,
                p._pulsim.SimulationOptions(t_start=0.0, t_end=1e-4, dt=1e-6),
                fn,
            )
        else:
            p.simulate(b, 1e-4, engine=engine, switch_fn=fn)
    assert (ei.value.engine, ei.value.feature, ei.value.device) == (
        engine, "electrical_feedback", "S1",
    )
