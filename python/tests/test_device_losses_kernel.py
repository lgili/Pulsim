"""add-in-kernel-event-losses group 2: event and conduction losses read and
deposited as heat inside the fixed-step engine's time loop.

Thermal network: temperature in °C is a node voltage, heat in W a current
(add_cauer_thermal_network's convention). One Cauer stage: the junction
node ``tj`` with ``R_th`` and ``C_th`` to an ambient node held at 25 V.
Every mean is time-weighted over whole periods on an exactly clipped
window (``pulsim.time_average``); tolerances are the change's own.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import pulsim as p

T_AMB = 25.0
FSW = 10e3


def _table(e25, e125=None):
    """E(v, i, Tj) independent of v and i: e25 J at 25 °C, e125 at 125 °C."""
    if e125 is None:
        return p.LossTable(v_axis=[100.0], i_axis=[1.0], tj_axis=[25.0],
                           energy=np.full((1, 1, 1), e25))
    return p.LossTable(v_axis=[100.0], i_axis=[1.0], tj_axis=[25.0, 125.0],
                       energy=np.array([[[e25, e125]]]))


def _thermal(b, r_th, c_th, node="tj"):
    b.add_voltage_source("Vamb", "amb", "gnd", T_AMB)
    b.add_resistor("Rth", node, "amb", r_th)
    b.add_capacitor("Cth", node, "amb", c_th)


def _chopper(r_th, c_th):
    """S1 chops 10 V into 10 Ω at 10 kHz, duty 0.5. g_on = 1e6 S keeps its
    conduction loss at 1e-6 W, invisible next to 20 W of switching."""
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "in", "gnd", 10.0)
    b.add_switch("S1", "in", "o", 1e6, 1e-12)
    b.add_resistor("RL", "o", "gnd", 10.0)
    _thermal(b, r_th, c_th)
    fn = p.make_pwm_switch_fn(FSW, 0.5, b.switch_index_of("S1"),
                              b.graph.num_switches)
    return b, fn


def _period_mean(res, node="tj", periods=1):
    t = np.asarray(res.times)
    t_to = t[-1]
    return p.time_average(t, res.v(node), t_from=t_to - periods / FSW,
                          t_to=t_to)


def test_adiabatic_junction_rises_by_the_table_sum():
    """2.5: C_th = 0.01 J/K, R_th = 1e9 K/W, E_on = E_off = 1 mJ, ten
    events -> ΔT_j = 1.000 K within 1e-6; heat injected = table sum."""
    b, fn = _chopper(1e9, 0.01)
    with pytest.warns(RuntimeWarning, match="one temperature knot"):
        b.set_device_losses("S1", junction="tj", E_on=_table(1e-3),
                            E_off=_table(1e-3))
    # edges at k·50 µs: ten of them by 0.5 ms; stop 20 µs after the last
    res = p.simulate(b, 0.52e-3, dt=1e-6, switch_fn=fn)
    ev = res.loss_events("S1")
    assert ev.size == 10
    e = res.device_energy("S1")
    table_sum = math.fsum(ev["E"])
    assert table_sum == pytest.approx(10e-3, rel=1e-12)
    assert e["E_on"] + e["E_off"] == pytest.approx(table_sum, rel=1e-12)
    # every recorded joule is injected or pending (the ledger)
    assert res.heat_injected + res.heat_pending == pytest.approx(
        table_sum + e["E_cond_electrical"], rel=1e-12)
    tj = res.v("tj")
    rise = tj[-1] - tj[1]          # sample 1: the ambient source is on
    assert rise == pytest.approx(1.000, rel=1e-6)


@pytest.mark.parametrize("tj_fixed", [True, False])
def test_open_and_closed_loop_junction_means(tj_fixed):
    """2.6 / 2.7: one stage R_th = 1 K/W, C_th = 0.01 J/K, 10 kHz, duty
    0.5, E(T) = 1 mJ (1 + 0.01 (T - 25)) on both edges. Read at a fixed
    Tj = 25 the mean is 25 + 20 W = 45.0 °C ± 0.05 K; with junction= the
    loop closes at 25 + 20 / (1 - 0.2) = 50.0 °C ± 0.1 K."""
    b, fn = _chopper(1.0, 0.01)
    tab = _table(1e-3, 2e-3)
    if tj_fixed:
        b.set_device_losses("S1", Tj=25.0, heat_node="tj", E_on=tab,
                            E_off=tab)
    else:
        b.set_device_losses("S1", junction="tj", E_on=tab, E_off=tab)
    res = p.simulate(b, 0.1, dt=1e-6, switch_fn=fn)
    mean = _period_mean(res)
    if tj_fixed:
        assert mean == pytest.approx(45.0, abs=0.05)
    else:
        assert mean == pytest.approx(50.0, abs=0.1)
        # each event reads T_j(x⁻): the sample that starts the step whose
        # new mask commutes (an x⁺ read would differ by E/C_th = 1.2 K)
        ev = res.loss_events("S1")
        t = np.asarray(res.times)
        k = np.searchsorted(t, ev["t"])
        assert np.array_equal(t[k], ev["t"])
        assert np.array_equal(ev["Tj"], np.asarray(res.v("tj"))[k])
        np.testing.assert_allclose(
            ev["E"], 1e-3 * (1 + 0.01 * (ev["Tj"] - 25.0)), rtol=1e-12)


def _diode_rig(r_th=1.0, c_th=0.01):
    """10 A forced through a PWL diode V_f = 0.7 V, g_on = 20 S."""
    b = p.CircuitBuilder()
    b.add_current_source("I", "a", "gnd", 10.0)
    b.add_diode("D1", "a", "gnd", 20.0, 1e-9, 0.0, 0.7)
    _thermal(b, r_th, c_th)
    return b


def test_conduction_heat_electrical_and_table():
    """2.8: electrical -> 12.0 W, T_j = 37.0 °C ± 0.01 K; table V_on = 1.0
    + 0.002 (T - 25) V at 10 A, closed loop -> 25 + 10 / (1 - 0.02) =
    35.20 °C ± 0.01 K, both conduction energies reported."""
    t_end = 0.15                     # 15 τ: e^-15 · 12 K = 4e-6 K
    b = _diode_rig()
    b.set_device_losses("D1", junction="tj")
    res = p.simulate(b, t_end, dt=1e-5)
    t = np.asarray(res.times)
    w = (t_end - 1e-3, t_end)
    assert p.time_average(t, res.v("tj"), *w) == pytest.approx(37.0, abs=0.01)
    e = res.device_energy("D1")
    # 12 W over the run, from the first solved step
    assert e["E_cond_electrical"] / (t[-1] - t[1]) == pytest.approx(
        12.0, rel=1e-3)
    assert e["E_cond_table"] == 0.0

    b = _diode_rig()
    von = p.ConductionTable([10.0], [25.0, 125.0], [1.0, 1.2])
    b.set_device_losses("D1", junction="tj", V_on=von, conduction="table")
    res = p.simulate(b, t_end, dt=1e-5)
    tj = p.time_average(np.asarray(res.times), res.v("tj"), *w)
    assert tj == pytest.approx(25.0 + 10.0 / (1.0 - 0.02), abs=0.01)
    e = res.device_energy("D1")
    assert e["E_cond_table"] > 0.0 and e["E_cond_electrical"] > 0.0
    # the electrical side still dissipates 12 W; the heat follows the table
    assert e["E_cond_electrical"] / e["E_cond_table"] == pytest.approx(
        12.0 / 10.204, rel=2e-3)


def _vsi():
    b = p.CircuitBuilder()
    b.add_voltage_source("Vdc", "dcp", "gnd", 600.0)
    v = p.add_three_phase_vsi(b, "INV", vdc_pos="dcp", vdc_neg="gnd",
                              out_a="a", out_b="b", out_c="c")
    for ph in "abc":
        b.add_resistor(f"R{ph}", ph, f"m{ph}", 10.0)
        b.add_inductor(f"L{ph}", f"m{ph}", "n", 5e-3)
    b.add_resistor("Rn", "n", "gnd", 1e3)
    b.add_capacitor("Cn", "n", "gnd", 1e-6)
    legs = p.ThreePhaseLegIndices(*v.switch_indices)
    fn = p.make_three_phase_spwm_fn(FSW, 50.0, 0.8, legs,
                                    b.graph.num_switches, 1e-6)
    return b, fn


def _switch_names(_b):
    return [f"INV__{s}{ph}" for ph in "abc" for s in ("HS", "LS")]


def test_vsi_parity_with_device_loss_summary_at_fixed_tj():
    """2.9: a three-phase SPWM VSI with fixed Tj = 125; per switch the
    kernel E_on / E_off equal device_loss_summary(switch_specs=...)
    within 1e-9 (same samples, same rule, same table)."""
    b, fn = _vsi()
    rng = np.random.default_rng(7)
    e_on = p.LossTable(v_axis=[0.0, 400.0, 800.0], i_axis=[0.0, 30.0, 60.0],
                       tj_axis=[25.0, 150.0],
                       energy=rng.uniform(1e-3, 5e-3, (3, 3, 2)))
    e_off = p.LossTable(v_axis=[0.0, 400.0, 800.0], i_axis=[0.0, 30.0, 60.0],
                        tj_axis=[25.0, 150.0],
                        energy=rng.uniform(1e-3, 5e-3, (3, 3, 2)))
    names = _switch_names(b)
    assert len(names) == 6
    for n in names:
        b.set_device_losses(n, Tj=125.0, E_on=e_on, E_off=e_off)
    res = p.simulate(b, 20e-3, dt=1e-6, switch_fn=fn)
    specs = {n: {"E_on_table": e_on, "E_off_table": e_off, "Tj": 125.0}
             for n in names}
    rows = {r["name"]: r for r in p.device_loss_summary(
        b, res, switch_specs=specs)}
    for n in names:
        k = res.device_energy(n)
        assert k["n_on"] == rows[n]["n_turn_on_events"] > 50
        assert k["n_off"] == rows[n]["n_turn_off_events"] > 50
        assert k["E_on"] == pytest.approx(rows[n]["E_sw_on_total"], rel=1e-9)
        assert k["E_off"] == pytest.approx(rows[n]["E_sw_off_total"],
                                           rel=1e-9)


def test_diode_recovery_counted_at_each_natural_turn_off():
    """2.10: a PWL diode with an E_off (recovery) table deposits at each
    natural turn-off; the count equals its recorded turn-offs."""
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", 24.0)
    b.add_switch("Q1", "vin", "sw", 1e3, 1e-9)
    b.add_diode("D1", "gnd", "sw", 1e3, 1e-9)
    b.add_inductor("L", "sw", "vout", 100e-6)
    b.add_capacitor("C", "vout", "gnd", 47e-6)
    b.add_resistor("RL", "vout", "gnd", 10.0)
    _thermal(b, 1.0, 1e-3)
    fn = p.make_pwm_switch_fn(100e3, 0.5, b.switch_index_of("Q1"),
                              b.graph.num_switches)
    with pytest.warns(RuntimeWarning, match="one temperature knot"):
        b.set_device_losses("D1", junction="tj", E_off=_table(5e-6))
    res = p.simulate(b, 2e-3, dt=1e-7, switch_fn=fn)
    bit = b.switch_index_of("D1")
    sw = np.asarray(res.switch_states, dtype=bool)[:, bit]
    n_off = int(np.count_nonzero(sw[:-1] & ~sw[1:]))
    n_on = int(np.count_nonzero(~sw[:-1] & sw[1:]))
    e = res.device_energy("D1")
    assert n_off > 100
    assert e["n_off"] == n_off and e["n_on"] == n_on
    ev = res.loss_events("D1")
    assert np.count_nonzero(ev["kind"] == "off") == n_off
    assert e["E_off"] == pytest.approx(5e-6 * n_off, rel=1e-12)


def test_a_record_only_port_leaves_the_run_bit_identical():
    """2.11 (in-suite half): a port with a fixed Tj and no heat node only
    records; the electrical run is byte for byte the run without it. The
    other half (no port at all) is the golden comparison."""
    b0, fn = _vsi()
    r0 = p.simulate(b0, 2e-3, dt=1e-6, switch_fn=fn)
    b1, fn1 = _vsi()
    for n in _switch_names(b1):
        b1.set_device_losses(n, Tj=125.0, E_on=_table(1e-3))
    r1 = p.simulate(b1, 2e-3, dt=1e-6, switch_fn=fn1)
    assert np.asarray(r0.states).tobytes() == np.asarray(r1.states).tobytes()
    assert np.asarray(r0.times).tobytes() == np.asarray(r1.times).tobytes()
    assert r0.heat_injected == r1.heat_injected == 0.0


def test_energy_balance_electrical_heat_record():
    """Electrical energy removed from the circuit (device_loss_summary) =
    the record's E_cond_electrical (1e-12); heat the network received
    (C·ΔT + ∫(T - T_amb)/R dt) = heat_injected + heat_pending to the
    trapezoid's last half step (h/T_run)."""
    t_end = 0.02
    b = _diode_rig(r_th=1.0, c_th=0.01)
    b.set_device_losses("D1", junction="tj")
    res = p.simulate(b, t_end, dt=1e-5)
    rows = {r["name"]: r for r in p.device_loss_summary(b, res)}
    e = res.device_energy("D1")
    assert e["E_cond_electrical"] == pytest.approx(rows["D1"]["E_total"],
                                                   rel=1e-12)
    t = np.asarray(res.times)
    tj = np.asarray(res.v("tj"))
    k0 = 1                                # the first solved sample
    stored = 0.01 * (tj[-1] - tj[k0])
    leaked = np.trapezoid(tj[k0:] - T_AMB, t[k0:]) / 1.0
    network = stored + leaked
    ledger = res.heat_injected + res.heat_pending
    assert ledger == pytest.approx(e["E_cond_electrical"], rel=1e-12)
    assert network == pytest.approx(ledger, rel=2 * 1e-5 / t_end)


# trbdf2 runs it (3.1), dsed too (3.3: the junction loop, test_exact_junction_loop)
@pytest.mark.parametrize("engine", ["bdf1"])
def test_other_engines_refuse_a_loss_port_by_name(engine):
    b = _diode_rig()
    b.set_device_losses("D1", junction="tj")
    with pytest.raises(p.UnsupportedError) as ei:
        if engine == "bdf1":             # no simulate() name: the raw entry
            p._pulsim.run_transient_bdf1(
                b, p._pulsim.SimulationOptions(t_start=0.0, t_end=1e-3,
                                               dt=1e-5),
                lambda _t: p.SwitchStateMask(b.graph.num_switches))
        else:
            p.simulate(b, 1e-3, engine=engine, dt=1e-5)
    assert (ei.value.engine, ei.value.feature, ei.value.device) == (
        engine, "device_losses", "D1")


def test_set_device_losses_refusals_name_the_device():
    b = _diode_rig()
    with pytest.raises(ValueError, match="'Q9'|\"Q9\""):
        b.set_device_losses("Q9", Tj=25.0)
    with pytest.raises(ValueError, match="Rth.*Resistor|Resistor"):
        b.set_device_losses("Rth", Tj=25.0)
    with pytest.raises(ValueError, match="D1.*junction.*Tj"):
        b.set_device_losses("D1", junction="tj", Tj=125.0)
    with pytest.raises(ValueError, match="D1.*neither"):
        b.set_device_losses("D1")
    with pytest.raises(ValueError, match="D1.*V_on"):
        b.set_device_losses("D1", Tj=25.0, conduction="table")
    # a junction with no capacitor: refused when the run starts
    b2 = p.CircuitBuilder()
    b2.add_current_source("I", "a", "gnd", 1.0)
    b2.add_diode("D1", "a", "gnd", 20.0, 1e-9)
    b2.add_voltage_source("Vamb", "amb", "gnd", 25.0)
    b2.add_resistor("Rth", "tj", "amb", 1.0)
    b2.add_capacitor("Cx", "a", "gnd", 1e-6)
    b2.set_device_losses("D1", junction="tj")
    with pytest.raises(ValueError, match="D1.*'tj'.*capacit"):
        p.simulate(b2, 1e-4, dt=1e-6)
    # the ambient node: a capacitor touches it, but the source holds it
    b3 = _diode_rig()
    b3.set_device_losses("D1", junction="amb")
    with pytest.raises(ValueError, match="D1.*'amb'.*voltage source 'Vamb'"):
        p.simulate(b3, 1e-4, dt=1e-6)
    # run_transient_adaptive resumes in segments: refused by name
    from pulsim.adaptive import run_transient_adaptive
    b4 = _diode_rig()
    b4.set_device_losses("D1", junction="tj")
    with pytest.raises(p.UnsupportedError) as ei:
        run_transient_adaptive(b4, t_end=1e-4, dt_init=1e-6)
    assert (ei.value.feature, ei.value.device) == ("device_losses", "D1")


_YAML = """circuit:
  devices:
    - {type: voltage_source, name: V, from: in, to: gnd, V: 10.0}
    - type: igbt
      name: Q1
      collector: in
      emitter: o
      losses:
        junction: tj
        e_on:  {v: [100.0], i: [1.0], tj: [25.0, 125.0], energy: [[[1.0e-3, 2.0e-3]]]}
        e_off: {v: [100.0], i: [1.0], tj: [25.0, 125.0], energy: [[[1.0e-3, 2.0e-3]]]}
    - {type: resistor, name: RL, from: o, to: gnd, R: 10.0}
    - {type: voltage_source, name: Vamb, from: amb, to: gnd, V: 25.0}
    - {type: resistor, name: Rth, from: tj, to: amb, R: 1.0}
    - {type: capacitor, name: Cth, from: tj, to: amb, C: 0.01}
"""


def test_yaml_losses_block():
    """3.5: a YAML `losses:` block holds the tables for the device (same
    run as set_device_losses); an unknown key and junction + Tj are
    refused by name."""
    b = getattr(p.load_yaml_string(_YAML), "builder", None) or \
        p.load_yaml_string(_YAML)
    assert b._device_loss_names() == ["Q1"]
    fn = p.make_pwm_switch_fn(FSW, 0.5, b.switch_index_of("Q1"),
                              b.graph.num_switches)
    r = p.simulate(b, 1e-3, dt=1e-6, switch_fn=fn)
    ev = r.loss_events("Q1")
    assert ev.size >= 19 and np.all(ev["E"] >= 1e-3)
    with pytest.raises(Exception, match="unknown key 'e_onn'"):
        p.load_yaml_string(_YAML.replace("e_on:", "e_onn:"))
    with pytest.raises(Exception, match="Q1.*both junction and Tj"):
        p.load_yaml_string(_YAML.replace("junction: tj",
                                         "junction: tj\n        Tj: 125"))
    with pytest.raises(Exception, match="unknown key 'enrgy' in losses.e_on"):
        p.load_yaml_string(_YAML.replace("energy: [[[1.0e-3, 2.0e-3]]]}",
                                         "enrgy: [[[1.0e-3, 2.0e-3]]]}", 1))


def _bilinear(i_ax, tj_ax, v, i, tj):
    """V_on(i, T_j) of a (n_i, n_tj) grid, inside the grid."""
    along_i = np.array([np.interp(i, i_ax, v[:, k]) for k in range(len(tj_ax))])
    return np.array([np.interp(t, tj_ax, along_i[:, n])
                     for n, t in enumerate(tj)])


@pytest.mark.parametrize("current", [10.0, 30.0])
@pytest.mark.parametrize("t_amb", [30.0, 90.0])
def test_a_plecs_part_reaches_the_kernel_uncollapsed(tmp_path, current, t_amb):
    """3.6 / 3.7: import_plecs_xml -> attach_losses builds the part's
    Foster chain and hands its V_on(i, T_j) grid to the kernel unreduced:
    the table heat equals the bilinear grid read at the run's own
    (|i|, T_j) samples within 1e-9, at two currents and two temperatures
    (the grid is not linear in i, so a collapse to V_0 + r·i would miss)."""
    from pulsim import lib

    from .test_parts_library import _IGBT
    f = tmp_path / "AK40N120.xml"
    f.write_text(_IGBT)
    part = lib.import_plecs_xml(f)
    b = p.CircuitBuilder()
    b.add_current_source("I", "a", "gnd", current)
    b.add_diode("D1", "a", "gnd", 20.0, 1e-9, 0.0, 0.7)
    junction = part.attach_losses(b, "D1", T_amb=t_amb)
    assert junction == "Tj_D1"
    res = p.simulate(b, 0.02, dt=1e-5)
    e = res.device_energy("D1")
    t = np.asarray(res.times)
    on = np.asarray(res.switch_states)[:, b.switch_index_of("D1")]
    i = np.abs(np.where(on, 20.0 * (res.v("a") - 0.7), 1e-9 * res.v("a")))
    i_ax, tj_ax, v = part.conduction["v_on"]
    p_tab = np.where(on, _bilinear(i_ax, tj_ax, v, i, res.v(junction)) * i, 0.0)
    want = math.fsum(np.diff(t) * 0.5 * (p_tab[:-1] + p_tab[1:]))
    assert e["E_cond_table"] == pytest.approx(want, rel=1e-9)
    # the junction really heats: the chain carries the table power
    assert res.v(junction)[-1] > t_amb + 1.0


def test_attach_losses_refuses_a_synthetic_part():
    from pulsim import lib
    part = next(q for q in lib.list_parts() if q.is_synthetic)
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "in", "gnd", 10.0)
    b.add_switch("S1", "in", "gnd", 1e3, 1e-9)
    with pytest.raises(lib.ProvenanceError, match="allow_synthetic"):
        part.attach_losses(b, "S1", T_amb=25.0)


def test_a_long_transient_settles_on_the_periodic_orbit_fixed_point():
    """One reading rule on both routes: the slice-6 buck (10 µs period,
    linear-in-T_j tables) with its switch's loss port closed on R_th =
    20 K/W // C_th settles to the junction fixed point the periodic-orbit
    evaluator solves (ThermalSpec(mode='kernel')). Measured -2.8e-3 K at
    C_th = 3e-4 J/K and -8.5e-4 K at 1e-3: the residual is the junction's
    own sawtooth (each ~11 µJ event lifts a finite C_th, the orbit holds
    T_j constant), not the reading. P_sw agrees to 7e-5."""
    from .test_design_thermal_kernel import T, T_AMB, TABLE, _ev, buck
    rec = _ev(20.0).evaluate({})
    assert rec.status == "ok", rec.reason
    b, fn = buck(0.5)
    b.add_voltage_source("Vamb", "amb", "gnd", T_AMB)
    b.add_resistor("Rth", "tj", "amb", 20.0)
    b.add_capacitor("Cth", "tj", "amb", 3e-4)
    b.set_device_losses("S", junction="tj", E_on=TABLE, E_off=TABLE)
    # 15 closed-loop time constants (7 ms) at T/200
    res = p.simulate(b, 11000 * T, dt=T / 200, switch_fn=fn)
    t = np.asarray(res.times)
    tj = p.time_average(t, res.v("tj"), t_from=t[-1] - 10 * T, t_to=t[-1])
    assert tj == pytest.approx(rec.t_j["S"], abs=0.01)
    ev = res.loss_events("S")
    p_sw = ev[ev["t"] >= t[-1] - 10 * T]["E"].sum() / (10 * T)
    assert p_sw == pytest.approx(rec.losses["S"]["switching"], rel=1e-3)


@pytest.mark.parametrize("restart,rtol", [(False, 1e-11), (True, 1e-8)])
def test_energy_balance_of_switching_heat_with_and_without_restart(
        restart, rtol):
    """The heat the network received (C·ΔT + ∫(T - T_amb)/R dt) equals
    heat_injected on a switching run; with the restart rule the BE/SDIRK
    steps keep no capacitor current, so the two-step trapezoid deposit is
    no longer exact at each switch of step kind. Measured: 2.7e-13
    trapezoid only, 4.1e-10 with 1200 restart steps."""
    b, fn = _chopper(1.0, 0.01)
    tab = _table(1e-3, 2e-3)
    b.set_device_losses("S1", junction="tj", E_on=tab, E_off=tab)
    res = p.simulate(b, 0.02, dt=1e-6, switch_fn=fn,
                     discontinuity_restart=restart)
    assert (res.n_restart_steps > 0) == restart
    t = np.asarray(res.times)
    tj = np.asarray(res.v("tj"))
    network = 0.01 * (tj[-1] - tj[1]) + np.trapezoid(tj[1:] - T_AMB, t[1:])
    assert network == pytest.approx(res.heat_injected, rel=rtol)


# ------------------------------------------------ the edge under the restart rule

class _OneEdge:
    """Open until t_e, closed after; names its edge."""

    def __init__(self, n, t_e):
        self.n, self.t_e = n, t_e

    def __call__(self, t):
        m = p.SwitchStateMask(self.n)
        m.set(0, t >= self.t_e)
        return m

    def next_edge_after(self, t):
        return self.t_e if t < self.t_e else math.inf


@pytest.mark.parametrize("split", [False, True])
def test_a_restart_edge_is_read_at_its_right_limit(split):
    """A freewheeling inductor (10 A at t = 0, tau = L / (R + 1/g_on)) is
    picked up by the switch at t_e. The load current of that turn-on is the
    inductor's at the edge, 10 exp(-t_e / tau) (closed form). After it the
    current rises at (V - R i) / L = 2.9e4 A/s, so the end of the step reads
    1.2e-3 high after the 0.3 us that is left of it (a located edge) and
    2.9e-3 high after a whole one. The kernel reads the event's right limit,
    the row device_loss_summary reads (they agreed at 1e-9 with the rule
    off only). E_on is linear in i, so the booked energy is the current.
    split=False: the gate of a step is read 0.5001 dt before its end (F-SB-2),
    so the edge acts at the nearer grid point: 33.7 us is 0.7 dt into
    (33, 34] us, taken at 34 us."""
    g_on, R, L, V, t_e = 1e4, 10.0, 1e-3, 100.0, 33.7e-6
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "vdc", "gnd", V)
    b.add_switch("S", "vdc", "sw", g_on, 1e-9)
    b.add_diode("D", "gnd", "sw", g_on, 1e-9)
    b.add_inductor("L", "sw", "out", L, i0=10.0)
    b.add_resistor("R", "out", "gnd", R)
    tab = p.LossTable(v_axis=[0.0, 200.0], i_axis=[0.0, 20.0], tj_axis=[25.0, 150.0],
                      energy=np.array([[[0.0, 0.0], [20.0, 20.0]], [[0.0, 0.0], [20.0, 20.0]]]))
    b.set_device_losses("S", Tj=100.0, E_on=tab)
    res = p.simulate(b, 60e-6, dt=1e-6, engine="pwl", switch_fn=_OneEdge(b.graph.num_switches, t_e),
                     discontinuity_restart=True, enable_substep_state_correction=split)
    k = res.device_energy("S")
    assert k["n_on"] == 1
    t_read = t_e if split else 34e-6   # the nearer grid point
    truth = 10.0 * math.exp(-t_read * (R + 1.0 / g_on) / L)
    assert k["E_on"] == pytest.approx(truth, rel=2e-5)        # the end of the step: 1.2e-3 / 2.9e-3 high
    row = {r["name"]: r for r in p.device_loss_summary(
        b, res, switch_specs={"S": {"E_on_table": tab, "Tj": 100.0}})}["S"]
    assert k["E_on"] == pytest.approx(row["E_sw_on_total"], rel=1e-9)
