"""The two device paths agree (add-unified-device-interface DECISIONS #23).

``device_path='legacy'`` stamps the Newton kinds by hand (the default);
``'interface'`` runs every kind's law on the device interface. Gate:
fixed-step trajectories within 1e-12 of each quantity's scale; TR-BDF2
within an rtol-proportional tolerance (its accepted step sequence may
differ by rounding); DC operating points within 1e-12. (``set_path``'s
refusal of an unported kind has no test: every shipped Newton kind is
ported, and a dummy kind cannot be added to the compile-time kind list from
a test; DECISIONS #49.)

The fixed-step gate is a ROUNDING claim only while both paths take the same
Newton iterations: a path that stops one iteration apart (another
platform's last bits deciding a convergence test) differs by up to the
Newton tolerance. ``_assert_paths_agree`` therefore asserts 1e-12 of scale
when the per-solve iteration counts are equal, and the converged tolerance
(``tol_newton_dx``, absolute) otherwise (DECISIONS #26, #29).
"""
import math
from types import SimpleNamespace

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
        assert err <= 1e-12, f"{name}: legacy vs interface {err:.2e} of scale"
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
    point of DECISIONS #44."""
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
    on a ramp (the soft boost's 0.3 µs ramps put V_DS ≈ 0 through threshold,
    where the raw mirror jumped; both paths now run the C¹ law, #44)."""
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
# Every circuit is gated, the soft boost too: its MOSFET passes through
# threshold with V_DS near zero, and both paths run ONE law there (#44).
GATED = sorted(CIRCUITS)
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


def test_interface_is_refused_by_the_exact_engine_and_bad_names():
    b = boost_mosfet_coss_saturable()
    with pytest.raises(ValueError, match="device_path"):
        p.simulate(b, t_end=1e-6, dt=1e-8, device_path="hand")
    with pytest.raises(Exception, match="device_path"):
        p.simulate(b, t_end=1e-6, engine="dsed", device_path="interface")


def test_default_path_is_legacy():
    assert p.SimulationOptions().device_path == p.DevicePath.Legacy


# ---- the MNA machines (15.6, DECISIONS #42) ---------------------------------
# PMSM (constant L with saliency, flux map) and induction machine through
# their `Device` laws: q = −λ on the stator current rows (the flux, never
# L·i), increment form, conductance-form rows. Parameters are the machine
# tests' synthetic sets (test_pmsm_mna, test_induction_machine_mna,
# test_pmsm_flux_map), not a real part.

def _sine3(b, v_peak, f_e):
    for k, node in enumerate(("ua", "ub", "uc")):
        b.add_sine_voltage_source(
            f"Vs_{'abc'[k]}", node, "gnd", v_dc=0.0, v_amplitude=v_peak,
            frequency=f_e, phase=-2.0 * math.pi / 3.0 * k)


def pmsm_sine_start():
    """Salient PMSM (L_d = 1 mH, L_q = 3 mH) started from rest by a 30 Hz
    sine: θ, ω and the reluctance torque all move."""
    b = p.CircuitBuilder()
    _sine3(b, 12.0, 30.0)
    b.add_pmsm_mna("M1", "ua", "ub", "uc", "nn", "w", "th", R_s=0.5,
                   L_d=1e-3, L_q=3e-3, psi_pm=0.05, pole_pairs=4, J=1e-3,
                   B=1e-4)
    return b


def pmsm_locked_rotor():
    """The salient PMSM with ω held at 0 by a source (rotor locked) and a
    stepped dq voltage: the stator rows alone, θ fixed at θ0 = 0.3 rad."""
    b = p.CircuitBuilder()
    for k, (node, v) in enumerate(zip(("ua", "ub", "uc"), (6.0, -1.0, -5.0))):
        b.add_pulse_voltage_source(f"V{'abc'[k]}", node, "gnd", 0.0, v, 0.0,
                                   1.0, rise_time=50e-6)
    b.add_pmsm_mna("M1", "ua", "ub", "uc", "nn", "w", "th", R_s=0.5,
                   L_d=1e-3, L_q=3e-3, psi_pm=0.05, pole_pairs=4, J=1.0,
                   theta0=0.3)
    b.add_voltage_source("Vw", "w", "gnd", 0.0)
    return b


def _sat_map():
    g = np.arange(-30.0, 30.0 + 0.75, 1.5)
    D, Q = np.meshgrid(g, g, indexing="ij")
    r = np.hypot(D, Q) / 10.0
    f = np.where(r < 1e-3, 1e-3 * (1 - r * r / 3),
                 1e-3 * np.tanh(r) / np.where(r < 1e-3, 1.0, r))
    return p.FluxMapDQ.from_tables(g, g, 0.05 + f * D, 2e-3 * Q + f * Q,
                                   source="synthetic: log-cosh test map")


def flux_map_saturated_start():
    """Cross-coupled, saturating flux-map PMSM (|i| > 2·I_sat) started from
    rest — test_pmsm_flux_map's energy-balance circuit."""
    b = p.CircuitBuilder()
    _sine3(b, 12.0, 20.0)
    b.add_pmsm_flux_map("M1", "ua", "ub", "uc", "nn", "w", "th", R_s=0.5,
                        flux_map=_sat_map(), pole_pairs=4, J=2e-4, B=1e-4)
    return b


_IM = dict(R_s=0.5, L_ls=2e-3, R_r=0.4, L_lr=2e-3, L_m=60e-3, pole_pairs=2)
_W_SYNC = 2 * math.pi * 60.0 / 2


def _im(b, J, omega0=0.0):
    b.add_induction_machine_mna("M", "ua", "ub", "uc", "nn", "w", J=J,
                                omega0=omega0, **_IM)
    return b


def im_sine_start():
    """Induction machine, direct-on-line 100 V rms 60 Hz start (J = 0.01)."""
    b = p.CircuitBuilder()
    _sine3(b, 100.0 * math.sqrt(2), 60.0)
    return _im(b, 0.01)


def im_generator_held_speed():
    """Induction machine held 5 % ABOVE synchronous speed by a source on ω:
    generating (negative slip), the source sinks the shaft torque. J is
    small (its companion 2J/h·ω would round at 1e-9 A in the source current,
    and a held shaft does not read J)."""
    b = p.CircuitBuilder()
    _sine3(b, 100.0 * math.sqrt(2), 60.0)
    w = 1.05 * _W_SYNC
    _im(b, 1e-3, omega0=w)
    b.add_voltage_source("Vw", "w", "gnd", w)
    return b


class _SixStep:
    """Six-step (180°) gating of the bridge at 60 Hz, switch order (a+, a-,
    b+, b-, c+, c-), with the edge oracle TR-BDF2 lands its grid on."""

    T = 1.0 / 60.0

    def __call__(self, t):
        s = int(math.floor(t / (self.T / 6) + 1e-9)) % 6
        m = p.SwitchStateMask(6)
        for leg, shift in enumerate((0, 4, 2)):
            up = ((s + shift) % 6) < 3
            m.set(2 * leg, up)
            m.set(2 * leg + 1, not up)
        return m

    def next_edge_after(self, t):
        return (math.floor(t / (self.T / 6) + 1e-9) + 1) * self.T / 6


def im_inverter_start():
    """Induction machine started by a six-step 300 V bridge (ideal switches
    1 mΩ / 1 µS): every 1/360 s the stator voltages jump."""
    b = p.CircuitBuilder()
    b.add_voltage_source("Vdc", "dc", "gnd", 300.0)
    for x in ("a", "b", "c"):
        b.add_switch(f"S{x}p", "dc", f"u{x}", 1e3, 1e-6)
        b.add_switch(f"S{x}n", f"u{x}", "gnd", 1e3, 1e-6)
    return _im(b, 0.01)


# name: (builder, t_end, dt, switch_fn, the quantity that must move)
MACHINES = {
    "pmsm_sine_start": (pmsm_sine_start, 0.03, 5e-6, None, "w"),
    "pmsm_locked_rotor": (pmsm_locked_rotor, 5e-3, 1e-6, None, "M1_a"),
    "flux_map_saturated_start": (flux_map_saturated_start, 0.02, 2e-6, None,
                                 "w"),
    "im_sine_start": (im_sine_start, 0.03, 2e-5, None, "w"),
    "im_generator_held_speed": (im_generator_held_speed, 0.05, 5e-5, None,
                                "M_a"),
    "im_inverter_start": (im_inverter_start, 0.03, 1e-5, _SixStep(), "w"),
}


def _run_machine(name, path, engine="pwl"):
    make, t_end, dt, sw, _ = MACHINES[name]
    kw = dict(dt=dt) if engine == "pwl" else dict(rtol=1e-6, atol=1e-9)
    if sw is not None:
        kw["switch_fn"] = sw
    return p.simulate(make(), t_end=t_end, engine=engine, device_path=path,
                      **kw)


def _trace(res, what):
    return np.asarray(res.v(what) if what == "w" else res.i(what))


def _neutral_column(res):
    x, vn = np.asarray(res.states), np.asarray(res.v("nn"))
    return next(c for c in range(x.shape[1]) if np.array_equal(x[:, c], vn))


def _without(res, col):
    x = np.asarray(res.states)
    return SimpleNamespace(states=np.delete(x, col, axis=1), times=res.times,
                           newton=res.newton)


@pytest.mark.parametrize("name", sorted(MACHINES))
def test_machine_paths_agree_to_1e12_on_pwl(name):
    """Every state but a floating neutral to 1e-12 of its scale (measured
    2026-11-18: ≤ 2.5e-14). A floating neutral is physically 0 V (no
    zero-sequence flux): what it reads (1e-11 V) is linear-solve round-off
    on the a·L entries of the stator rows, different in the last bits on the
    two paths; it is checked to be zero on both, to 1e-9 of the phase
    voltages."""
    a = _run_machine(name, "legacy")
    i = _run_machine(name, "interface")
    what = MACHINES[name][4]
    assert np.ptp(_trace(a, what)) > 1.0, f"{name}: control: {what} never moved"
    v_phase = max(np.max(np.abs(np.asarray(a.v(x)))) for x in ("ua", "ub", "uc"))
    if np.max(np.abs(np.asarray(a.v("nn")))) <= 1e-9 * v_phase:   # floating
        assert np.max(np.abs(np.asarray(i.v("nn")))) <= 1e-9 * v_phase
        n = _neutral_column(a)
        a, i = _without(a, n), _without(i, n)
    _assert_paths_agree(name, a, i)


@pytest.mark.parametrize("name", ["pmsm_sine_start", "flux_map_saturated_start",
                                  "im_inverter_start"])
def test_machine_paths_agree_to_rtol_on_trbdf2(name):
    """At the end (both grids land on t_end; they part within a few steps,
    so earlier common times are rare), to 10·rtol of scale; a floating
    neutral is left out as on pwl."""
    a = _run_machine(name, "legacy", "trbdf2")
    i = _run_machine(name, "interface", "trbdf2")
    assert a.times[-1] == i.times[-1]
    xa, xi = np.asarray(a.states), np.asarray(i.states)
    scale = _scale(xa)
    if name != "im_inverter_start":   # floating neutral
        n = _neutral_column(a)
        xa, xi, scale = (np.delete(v, n, axis=-1) for v in (xa, xi, scale))
    err = np.max(np.abs(xa[-1] - xi[-1]) / scale)
    assert err <= 10 * 1e-6, f"{name}: legacy vs interface {err:.2e} of scale"


def _flux_map_energy_residual(res, fm, J=2e-4, B=1e-4, R_s=0.5):
    t = np.asarray(res.times)
    vn = np.asarray(res.v("nn"))
    cur = [np.asarray(res.i(f"M1_{x}")) for x in "abc"]
    v = [np.asarray(res.v(n)) - vn for n in ("ua", "ub", "uc")]
    w = np.asarray(res.v("w"))
    i_d = np.asarray(res.signal("M1.i_d"))
    i_q = np.asarray(res.signal("M1.i_q"))
    ev = np.array([fm.evaluate(a, c) for a, c in zip(i_d, i_q)])
    w_field = 1.5 * (ev[:, 1] * i_d + ev[:, 2] * i_q - ev[:, 0])
    e_in = np.trapezoid(sum(vk * ik for vk, ik in zip(v, cur)), t)
    e_out = (np.trapezoid(R_s * sum(ik * ik for ik in cur), t)
             + (w_field[-1] - w_field[0]) + 0.5 * J * (w[-1] ** 2 - w[0] ** 2)
             + B * np.trapezoid(w * w, t))
    return e_in, e_in - e_out


def test_machine_energy_balance_is_the_same_on_both_paths():
    """The saturated flux-map start conserves energy on 'interface' as on
    'legacy' (the residual is the trapezoidal rule's, ~1e-9 of the input):
    the law integrates the flux, so no ratchet adds or removes energy."""
    fm = _sat_map()
    ea, ra = _flux_map_energy_residual(_run_machine(
        "flux_map_saturated_start", "legacy"), fm)
    ei, ri = _flux_map_energy_residual(_run_machine(
        "flux_map_saturated_start", "interface"), fm)
    assert abs(ri) <= 2e-8 * abs(ei), (ei, ri)
    assert abs(ri - ra) <= 1e-3 * abs(ra) + 1e-12 * abs(ea), (ra, ri)
