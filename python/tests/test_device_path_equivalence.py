"""The two device paths agree (add-unified-device-interface DECISIONS #23).

``device_path='legacy'`` stamps the Newton kinds by hand (the default);
``'interface'`` runs every kind's law on the device interface. Gate:
fixed-step trajectories within 1e-12 of each quantity's scale; TR-BDF2
within an rtol-proportional tolerance (its accepted step sequence may
differ by rounding); DC operating points within 1e-12; an unported kind
refuses ``'interface'`` by name.
"""
import numpy as np
import pytest

import pulsim as p


def _scale(states):
    return np.maximum(np.max(np.abs(states), axis=0), 1e-9)


def _gate(n_sw):
    return dict(switch_fn=lambda _t: p.SwitchStateMask(n_sw)) if n_sw else {}


def boost_mosfet_coss_saturable():
    """MOSFET L1 + Coss + saturable inductor + Shockley rectifier."""
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", 12.0)
    b.add_saturable_inductor("L", "vin", "sw", 47e-6, 4.0, 2e-6)
    b.add_mosfet_level1("M", "sw", "gnd", "g", K=0.5, V_T=3.0, kappa=4.0,
                        with_body_diode=False)
    b.add_nonlinear_capacitor("Coss", "sw", "gnd", 200e-12, 0.7, 0.45)
    b.add_resistor("Rs", "sw", "gnd", 2e3)
    b.add_shockley_diode("D", "sw", "out", 1e-12, 1.2, 0.025852)
    b.add_capacitor("C", "out", "gnd", 10e-6)
    b.add_resistor("R", "out", "gnd", 12.0)
    b.add_pulse_voltage_source("Vg", "g", "gnd", 0.0, 12.0, 0.0, 6e-6,
                               10e-6, 3e-7, 3e-7)
    return b


def chopper_igbt_vdmos():
    """IGBT L1 (no tail) + VDMOS C_gd law + saturable freewheel path."""
    b = p.CircuitBuilder()
    b.add_voltage_source("Vdc", "dc", "gnd", 60.0)
    b.add_igbt_level1("Q", "dc", "e", "g", V_T=5.0, kappa=4.0,
                      v_knee=0.01)
    b.add_shockley_diode("Dfw", "gnd", "e", 1e-11, 1.5, 0.025852)
    b.add_saturable_inductor("L", "e", "o", 200e-6, 8.0, 5e-6)
    b.add_resistor("R", "o", "gnd", 4.0)
    b.add_nonlinear_capacitor("Cgd", "g", "e",
                              law=p.ChargeLaw.vdmos_cgd(2e-9, 1e-10, 0.3))
    b.add_resistor("Rg", "gd", "g", 10.0)
    b.add_pulse_voltage_source("Vg", "gd", "gnd", 0.0, 15.0, 0.0, 12e-6,
                               25e-6, 1e-7, 1e-7)
    return b


CIRCUITS = {
    "boost_mosfet_coss_saturable": boost_mosfet_coss_saturable,
    "chopper_igbt_vdmos": chopper_igbt_vdmos,
}


def _run(make, path, **kw):
    b = make()
    return p.simulate(b, device_path=path, **_gate(b.graph.num_switches),
                      **kw)


@pytest.mark.parametrize("name", sorted(CIRCUITS))
def test_fixed_step_trajectories_agree_to_1e12(name):
    make = CIRCUITS[name]
    kw = dict(t_end=200e-6, dt=20e-9, engine="pwl")
    a = _run(make, "legacy", **kw)
    i = _run(make, "interface", **kw)
    xa, xi = np.asarray(a.states), np.asarray(i.states)
    assert xa.shape == xi.shape and xa.shape[0] > 9000
    assert np.array_equal(np.asarray(a.times), np.asarray(i.times))
    err = np.max(np.abs(xa - xi) / _scale(xa))
    assert err <= 1e-12, f"{name}: legacy vs interface {err:.2e} of scale"


@pytest.mark.parametrize("name", sorted(CIRCUITS))
def test_trbdf2_trajectories_agree_to_rtol(name):
    make = CIRCUITS[name]
    rtol = 1e-6
    kw = dict(t_end=200e-6, engine="trbdf2", rtol=rtol, atol=1e-9)
    a = _run(make, "legacy", **kw)
    i = _run(make, "interface", **kw)
    ta, xa = np.asarray(a.times), np.asarray(a.states)
    ti, xi = np.asarray(i.times), np.asarray(i.states)
    # The accepted step sequences differ by rounding (the controller's
    # error estimate moves in its last bits, and the grids part within a
    # few steps). Compare at the times both runs accepted: every gate
    # landing and the end are among them. Interpolating across an edge,
    # or comparing index by index, would measure the grids, not the paths.
    _, ia, ii = np.intersect1d(ta, ti, return_indices=True)
    assert len(ia) >= 10 and ta[-1] == ti[-1], (len(ia), len(ta))
    err = np.max(np.abs(xa[ia] - xi[ii]) / _scale(xa))
    assert err <= 10 * rtol, f"{name}: legacy vs interface {err:.2e}"


@pytest.mark.parametrize("name", sorted(CIRCUITS))
def test_dc_operating_points_agree(name):
    b = CIRCUITS[name]()
    xa = p.compute_dc_op(b, device_path="legacy")
    xi = p.compute_dc_op(b, device_path="interface")
    assert np.max(np.abs(xa - xi) / np.maximum(np.abs(xa), 1.0)) <= 1e-12


def test_dc_operating_point_with_every_static_kind_conducting():
    """The two circuits above start with their gates low (devices off, the
    stateful kinds open at DC); here every static kind conducts."""
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "a", "gnd", 24.0)
    b.add_voltage_source("Vg", "g", "gnd", 10.0)
    b.add_resistor("R1", "a", "d", 6.0)
    b.add_mosfet_level1("M", "d", "s", "g", K=0.5, V_T=3.0, kappa=4.0,
                        with_body_diode=False)
    b.add_shockley_diode("D1", "s", "k", 1e-12, 1.2, 0.025852)
    b.add_resistor("R2", "k", "gnd", 3.0)
    b.add_resistor("R3", "a", "c", 8.0)
    b.add_igbt_level1("Q", "c", "e", "g", V_T=5.0, kappa=4.0, v_knee=0.01)
    b.add_nonlinear_diode("D2", "e", "gnd", p.IdealDiodeParams(),
                          allow_non_passive=True)
    xa = np.asarray(p.compute_dc_op(b, device_path="legacy"))
    xi = np.asarray(p.compute_dc_op(b, device_path="interface"))
    assert np.max(np.abs(xa)) > 1.0
    assert np.max(np.abs(xa - xi) / np.maximum(np.abs(xa), 1.0)) <= 1e-12


def test_an_unported_kind_refuses_interface_by_name():
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "a", "gnd", 1.0)
    b.add_resistor("R", "a", "k", 10.0)
    b.add_lauritzen_diode("DL", "k", "gnd")
    with pytest.raises(ValueError, match="LauritzenDiode"):
        p.simulate(b, t_end=1e-6, dt=1e-8, engine="pwl",
                   device_path="interface")
    with pytest.raises(ValueError, match="LauritzenDiode"):
        p.simulate(b, t_end=1e-6, engine="trbdf2", device_path="interface")


def test_interface_is_refused_by_the_exact_engine_and_bad_names():
    b = boost_mosfet_coss_saturable()
    with pytest.raises(ValueError, match="device_path"):
        p.simulate(b, t_end=1e-6, dt=1e-8, device_path="hand")
    with pytest.raises(Exception, match="device_path"):
        p.simulate(b, t_end=1e-6, engine="dsed", device_path="interface")


def test_default_path_is_legacy():
    assert p.SimulationOptions().device_path == p.DevicePath.Legacy
