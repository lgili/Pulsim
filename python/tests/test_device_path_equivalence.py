"""The two device paths agree (add-unified-device-interface DECISIONS #23).

``device_path='legacy'`` stamps the Newton kinds by hand (the default);
``'interface'`` runs every kind's law on the device interface. Gate:
fixed-step trajectories within 1e-12 of each quantity's scale (measured; asserted
at 1e-9 for other compilers); TR-BDF2
within an rtol-proportional tolerance (its accepted step sequence may
differ by rounding); DC operating points within 1e-12; an unported kind
refuses ``'interface'`` by name.

The fixed-step gate is a ROUNDING claim only while both paths take the same
Newton iterations: a path that stops one iteration apart (another
platform's last bits deciding a convergence test) differs by up to the
Newton tolerance. ``_assert_paths_agree`` therefore asserts 1e-12 of scale
when the per-solve iteration counts are equal, and the converged tolerance
(``tol_newton_dx``, absolute) otherwise (DECISIONS #26, #29).
"""
import numpy as np
import pytest

import pulsim as p


def _scale(states):
    return np.maximum(np.max(np.abs(states), axis=0), 1e-9)


def _gate(n_sw):
    return dict(switch_fn=lambda _t: p.SwitchStateMask(n_sw)) if n_sw else {}


TOL_NEWTON_DX = 1e-9   # the engines' default absolute Newton step tolerance


def _assert_paths_agree(name, a, i):
    xa, xi = np.asarray(a.states), np.asarray(i.states)
    assert xa.shape == xi.shape
    assert np.array_equal(np.asarray(a.times), np.asarray(i.times))
    same_iterations = np.array_equal(
        np.asarray(a.newton.iterations_per_solve),
        np.asarray(i.newton.iterations_per_solve))
    if same_iterations:
        err = np.max(np.abs(xa - xi) / _scale(xa))
        # <= 1e-12 measured on macOS; through libm another compiler (FMA)
        # differs in the last bits: bound at 1e-9 (the CI lesson)
        assert err <= 1e-9, f"{name}: legacy vs interface {err:.2e} of scale"
    else:
        # A stopping difference touches a few solves; a wrong Jacobian
        # (still converging, more slowly) moves the counts everywhere.
        na = np.asarray(a.newton.iterations_per_solve)
        ni = np.asarray(i.newton.iterations_per_solve)
        assert na.shape == ni.shape, f"{name}: solve counts differ"
        assert np.count_nonzero(na != ni) <= max(3, na.size // 100), (
            f"{name}: Newton iteration counts differ on "
            f"{np.count_nonzero(na != ni)} of {na.size} solves")
        err = np.max(np.abs(xa - xi))
        assert err <= 10 * TOL_NEWTON_DX, (
            f"{name}: Newton iteration counts differ and the paths are "
            f"{err:.2e} apart (converged tolerance {TOL_NEWTON_DX:g})")


def _gate_bit(index, period):
    """Gate `index` on for the first half of every `period`."""
    def fn(t):
        m = p.SwitchStateMask(index + 1)
        m.set(index, (t % period) < 0.5 * period)
        return m
    return fn


def boost_mosfet_coss_saturable():
    """MOSFET L1 + Coss + saturable inductor + Shockley rectifier."""
    return _boost(3e-7, 4.0)


def _boost(edge, kappa):
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", 12.0)
    b.add_saturable_inductor("L", "vin", "sw", 47e-6, 4.0, 2e-6)
    b.add_mosfet_level1("M", "sw", "gnd", "g", K=0.5, V_T=3.0, kappa=kappa,
                        with_body_diode=False)
    b.add_nonlinear_capacitor("Coss", "sw", "gnd", 200e-12, 0.7, 0.45)
    b.add_resistor("Rs", "sw", "gnd", 2e3)
    b.add_shockley_diode("D", "sw", "out", 1e-12, 1.2, 0.025852)
    b.add_capacitor("C", "out", "gnd", 10e-6)
    b.add_resistor("R", "out", "gnd", 12.0)
    b.add_pulse_voltage_source("Vg", "g", "gnd", 0.0, 12.0, 0.0, 6e-6,
                               10e-6, edge, edge)
    return b


def mosfet_reverse_conduction_ac():
    """MOSFET L1 held on (V_GS = 12 V, far above threshold) carrying an AC
    current: V_DS crosses zero both ways — the reverse branch and the C¹
    point of DECISIONS #39 — where the two laws agree to rounding (the
    leak the interface law drops is K·a²·e^{-κa} ≈ 1e-14 A here)."""
    b = p.CircuitBuilder()
    b.add_sine_voltage_source("V", "in", "gnd", 0.0, 10.0, 50e3, 0.0)
    b.add_resistor("R", "in", "a", 2.0)
    b.add_inductor("L", "a", "d", 20e-6)
    b.add_mosfet_level1("M", "d", "gnd", "g", K=0.5, V_T=3.0, kappa=4.0,
                        with_body_diode=False)
    b.add_voltage_source("Vg", "g", "gnd", 12.0)
    return b


def boost_mosfet_fast_gate():
    """The boost with 1 ns gate edges and κ = 15: no fixed-step sample sits
    on a ramp, and off (a = −3 V) the model's leak K·a²·σ(κa)σ(−κa) is
    1e-19 A — the control of the recorded boost difference (DECISIONS #39;
    at κ = 4 the off-state leak at V_DS ≈ 0 is 28 µA, and the paths part by
    5e-9 of scale after every turn-off even with fast edges)."""
    return _boost(1e-9, 15.0)


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


def chopper_igbt_tail_lauritzen():
    """IGBT L1 WITH its turn-off tail (stored charge, a local row) and a
    Lauritzen freewheel diode (stored charge q_M, a local row) that
    recovers when the IGBT turns back on."""
    b = p.CircuitBuilder()
    b.add_voltage_source("Vdc", "dc", "gnd", 100.0)
    b.add_igbt_level1("Q", "dc", "e", "g", V_T=5.0, kappa=4.0, v_knee=0.01,
                      tau_tail=2e-6, k_tail=0.4)
    b.add_lauritzen_diode("Dfw", "gnd", "e", tau=2e-7, T_M=2e-8)
    b.add_inductor("L", "e", "o", 100e-6)
    b.add_resistor("R", "o", "gnd", 5.0)
    b.add_resistor("Rsn", "e", "gnd", 1e4)
    b.add_pulse_voltage_source("Vg", "g", "gnd", 0.0, 15.0, 0.0, 10e-6,
                               25e-6, 1e-7, 1e-7)
    return b


def saturable_permeance_core():
    """A winding on a saturable permeance (the nonlinear capacitor's tanh
    law in the magnetic domain) and an air gap, driven past the knee by a
    unipolar square wave: the core's Q(F) = Φ law on its Newton row."""
    b = p.CircuitBuilder()
    b.add_pulse_voltage_source("V", "e", "gnd", 0.0, 24.0, 0.0, 50e-6,
                               100e-6, 1e-7, 1e-7)
    b.add_resistor("R", "e", "w", 0.5)
    b.add_winding("W", "w", "gnd", "m1", "gnd", 25)
    b.add_saturable_permeance("Core", "m1", "gnd", 76e-6, 72e-3, mu_r=2000.0,
                              B_sat=0.35)
    b.add_air_gap("G", "m1", "gnd", 76e-6, 0.1e-3)   # in parallel: DC-solvable
    return b


CIRCUITS = {
    "boost_mosfet_coss_saturable": boost_mosfet_coss_saturable,
    "boost_mosfet_fast_gate": boost_mosfet_fast_gate,
    "chopper_igbt_vdmos": chopper_igbt_vdmos,
    "mosfet_reverse_conduction_ac": mosfet_reverse_conduction_ac,
    "chopper_igbt_tail_lauritzen": chopper_igbt_tail_lauritzen,
    "saturable_permeance_core": saturable_permeance_core,
}
# The boost passes its soft (κ = 4) MOSFET through threshold with V_DS near
# zero (0.3 µs gate ramps; the off state right after each turn-off): there
# the interface law drops the model's leak at V_DS = 0 (DECISIONS #39), so
# its two paths part BY DESIGN. Recorded below, not gated at 1e-12;
# `boost_mosfet_fast_gate` is its control. Every other circuit is gated.
RECORDED = {"boost_mosfet_coss_saturable"}
GATED = sorted(set(CIRCUITS) - RECORDED)
# TR-BDF2 compares at the times both runs accepted, which a gate edge
# provides; the AC circuit has none (4 common times of 2757), so its
# TR-BDF2 cell is the conformance MosfetLevel1 cell (AC, interpolated).
GATED_TRBDF2 = [n for n in GATED if n != "mosfet_reverse_conduction_ac"]


def _run(make, path, **kw):
    b = make()
    return p.simulate(b, device_path=path, **_gate(b.graph.num_switches),
                      **kw)


@pytest.mark.parametrize("name", GATED)
def test_fixed_step_trajectories_agree_to_1e12(name):
    make = CIRCUITS[name]
    kw = dict(t_end=200e-6, dt=20e-9, engine="pwl")
    a = _run(make, "legacy", **kw)
    i = _run(make, "interface", **kw)
    assert np.asarray(a.states).shape[0] > 9000
    _assert_paths_agree(name, a, i)


@pytest.mark.parametrize("name", GATED_TRBDF2)
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
    # every circuit, the recorded boost too: its DC point (gate low, V_DS =
    # 12 V) is e^{-48} of the leak away from V_DS = 0
    b = CIRCUITS[name]()
    xa = p.compute_dc_op(b, device_path="legacy")
    xi = p.compute_dc_op(b, device_path="interface")
    assert np.max(np.abs(xa - xi) / np.maximum(np.abs(xa), 1.0)) <= 1e-9   # 1e-12 measured


@pytest.mark.parametrize("engine", ["pwl", "trbdf2"])
def test_the_soft_boost_parts_by_the_mosfet_leak_only(engine):
    """Measured 2026-11-18: 2.5e-3 of scale (pwl), 2.1e-4 (trbdf2), largest
    on the first turn-on ramp (V_DS ≈ 0, a through threshold; the leak
    peaks at 0.439·K/κ² = 13.7 mA). Visible — the fix is live — and of the
    leak's order, not of a wrong law's; the control agrees to 1e-12."""
    kw = (dict(t_end=200e-6, dt=20e-9, engine="pwl") if engine == "pwl"
          else dict(t_end=200e-6, engine="trbdf2", rtol=1e-6, atol=1e-9))
    a = _run(boost_mosfet_coss_saturable, "legacy", **kw)
    i = _run(boost_mosfet_coss_saturable, "interface", **kw)
    ta, ti = np.asarray(a.times), np.asarray(i.times)
    xa, xi = np.asarray(a.states), np.asarray(i.states)
    _, ia, ii = np.intersect1d(ta, ti, return_indices=True)
    err = np.max(np.abs(xa[ia] - xi[ii]) / _scale(xa))
    assert 1e-6 < err < 1e-2, f"legacy vs interface {err:.2e} of scale"


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
    assert np.max(np.abs(xa - xi) / np.maximum(np.abs(xa), 1.0)) <= 1e-9   # 1e-12 measured


def test_a_gated_diode_runs_on_interface_with_its_gate_bit():
    """The gated diode (PWL IGBT, gate held) is a MODE kind: its gate bit and
    latch are discrete state in the switch mask and the mode selector, the
    same code on both paths; the Newton kinds beside it go through the
    interface. The gate must switch the device (its control: gate off)."""
    def build():
        b = p.CircuitBuilder()
        b.add_sine_voltage_source("V", "in", "gnd", 0.0, 20.0, 20e3, 0.0)
        b.add_resistor("R", "in", "o", 5.0)
        b.add_igbt("Qd", "o", "k", 1e-3, 1e9, V_ce0=0.7, reverse_blocking=True,
                   V_th=0.701)
        b.add_lauritzen_diode("DL", "k", "gnd")
        b.add_nonlinear_capacitor("Cj", "o", "gnd", 1e-9, 0.7, 0.45)
        return b
    kw = dict(t_end=200e-6, dt=20e-9, engine="pwl",
              switch_fn=_gate_bit(0, 37e-6))
    a = p.simulate(build(), device_path="legacy", **kw)
    i = p.simulate(build(), device_path="interface", **kw)
    _assert_paths_agree("gated", a, i)
    on = np.asarray(a.switch_states)[:, 0]
    assert on.any() and not on.all(), "the gate never switched the device"
    off = p.simulate(build(), device_path="interface", t_end=200e-6, dt=20e-9,
                     engine="pwl", switch_fn=lambda _t: p.SwitchStateMask(1))
    assert np.max(np.abs(np.asarray(off.i("Qd")))) < 1e-6 < np.max(
        np.abs(np.asarray(i.i("Qd"))))


def test_an_unported_kind_refuses_interface_by_name():
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "a", "gnd", 1.0)
    b.add_resistor("Ra", "a", "pa", 10.0)
    b.add_resistor("Rb", "a", "pb", 10.0)
    b.add_resistor("Rc", "a", "pc", 10.0)
    b.add_pmsm_mna("M", "pa", "pb", "pc", "n", "w", "th", 0.5, 1e-3, 1e-3,
                   0.05, 2, 1e-3, 1e-4)
    b.add_resistor("Rn", "n", "gnd", 1e6)
    with pytest.raises(ValueError, match="PmsmMna"):
        p.simulate(b, t_end=1e-6, dt=1e-8, engine="pwl",
                   device_path="interface")
    with pytest.raises(ValueError, match="PmsmMna"):
        p.simulate(b, t_end=1e-6, engine="trbdf2", device_path="interface")


def test_interface_is_refused_by_the_exact_engine_and_bad_names():
    b = boost_mosfet_coss_saturable()
    with pytest.raises(ValueError, match="device_path"):
        p.simulate(b, t_end=1e-6, dt=1e-8, device_path="hand")
    with pytest.raises(Exception, match="device_path"):
        p.simulate(b, t_end=1e-6, engine="dsed", device_path="interface")


def test_default_path_is_legacy():
    assert p.SimulationOptions().device_path == p.DevicePath.Legacy
