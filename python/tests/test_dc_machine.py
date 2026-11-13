"""MNA-native DC machine (openspec add-mna-native-machines, step 1).

`CircuitBuilder.add_dc_machine` is a linear composite: armature R-L, an
ideal transformer of ratio K between the speed node and the armature,
the inertia as a capacitor J on the speed node. Every reference below is
an analytic property of the model (exact solution of the 2x2 linear
system, its DC limit, its power balance) — verification, not validation
against a real machine. The parameters are declared test values.
"""
import warnings

import numpy as np
import pytest

import pulsim as p

R, L, K, J = 0.5, 1e-3, 0.1, 1e-5          # declared test machine
V, F = 10.0, 200.0                          # sine drive
T_END = 20e-3


def _build(source="sine", T_load=0.0, B=0.0, V_dc=10.0):
    b = p.CircuitBuilder()
    if source == "sine":
        b.add_sine_voltage_source("V", "a", "gnd", 0.0, V, F)
    else:
        b.add_voltage_source("V", "a", "gnd", V_dc)
    b.add_dc_machine("M", "a", "gnd", "w", R_a=R, L_a=L, K=K, J=J,
                     B=B, T_load=T_load)
    return b


# Exact solution of L di/dt = V sin(Wt) - R i - K w, J dw/dt = K i from
# rest, through the augmented linear system z = [i, w, sin, cos].
_W = 2 * np.pi * F
_A = np.array([[-R / L, -K / L, V / L, 0.0], [K / J, 0, 0, 0],
               [0, 0, 0, _W], [0, 0, -_W, 0]])
_lam, _S = np.linalg.eig(_A)
_c0 = np.linalg.solve(_S, np.array([0, 0, 0, 1.0], dtype=complex))


def _exact(t):
    z = (_S * _c0) @ np.exp(np.outer(_lam, np.asarray(t)))
    return z[0].real, z[1].real


def _speed_error(r):
    t = np.asarray(r.times)
    _, w_ex = _exact(t)
    return np.max(np.abs(np.asarray(r.v("w")) - w_ex)) / np.max(np.abs(w_ex))


def _run(dt=None, engine="pwl"):
    kw = {"dt": dt} if engine == "pwl" else {}
    return p.simulate(_build(), t_end=T_END, engine=engine, **kw)


def test_sine_drive_matches_exact_solution_at_12_5_us():
    # Review probe p5b_dcmotor.py measured 7.45e-5 (observer: 1.59 %).
    assert _speed_error(_run(12.5e-6)) <= 1e-4


def test_second_order_in_the_step():
    # Consistent start (sine source at 0 V, machine at rest), so the
    # fixed engine is not first order from its start-up.
    errs = [_speed_error(_run(dt)) for dt in (100e-6, 50e-6, 25e-6, 12.5e-6)]
    ratios = [a / b for a, b in zip(errs, errs[1:])]
    assert all(r >= 3.5 for r in ratios), (errs, ratios)


def test_trbdf2_matches_exact_and_pwl():
    r = _run(engine="trbdf2")
    assert r.engine_used == "trbdf2"
    assert _speed_error(r) <= 2e-4
    fixed = _run(12.5e-6)
    w_ref = np.interp(np.asarray(r.times), np.asarray(fixed.times),
                      np.asarray(fixed.v("w")))
    peak = np.max(np.abs(w_ref))
    assert np.max(np.abs(np.asarray(r.v("w")) - w_ref)) <= 1e-3 * peak


def test_dc_operating_point_is_the_analytic_steady_state():
    b = _build("dc", T_load=0.2)
    # auto_regularize=False: the preflight does not count the ideal
    # transformer's primary as a DC path, so with B = 0 it would tie the
    # speed node to ground through 1 GOhm (a 9e-7 relative torque load).
    # The matrix is regular without it.
    x = p.compute_dc_op(b, gmin=0.0, auto_regularize=False)
    i_idx = b.pool.branch_var_id_for_inductor(b.branch_id_of("M.la"),
                                              b.graph)
    assert x[i_idx] == pytest.approx(2.0, rel=1e-9)            # T_load/K
    assert x[b.node_id_of("w")] == pytest.approx(90.0, rel=1e-9)  # (V-R i)/K


def test_start_up_settles_to_the_analytic_speed_and_current():
    B = 1e-4
    r = p.simulate(_build("dc", T_load=0.2, B=B), t_end=0.1, dt=10e-6,
                   engine="pwl")
    # Steady state: i = (T_load + B w)/K, V = R i + K w.
    w_ss = (10.0 - R * 0.2 / K) / (K + R * B / K)
    i_ss = (0.2 + B * w_ss) / K
    assert r.v("w")[-1] == pytest.approx(w_ss, rel=1e-6)
    assert r.i("M.la")[-1] == pytest.approx(i_ss, rel=1e-6)


def _energy_residual(r, B, T_load):
    """Electrical energy in minus (copper + friction + load + stored),
    relative to the energy in. Products are taken of interval averages:
    for the trapezoidal rule that is the discrete energy identity."""
    t = np.asarray(r.times)
    h = np.diff(t)
    avg = lambda y: 0.5 * (np.asarray(y)[1:] + np.asarray(y)[:-1])  # noqa: E731
    i, w, v = r.i("M.la"), r.v("w"), r.v("a")
    ia, wa = avg(i), avg(w)
    e_in = np.sum(h * avg(v) * ia)
    e_out = (np.sum(h * R * ia ** 2) + np.sum(h * B * wa ** 2)
             + np.sum(h * T_load * wa)
             + 0.5 * L * (i[-1] ** 2 - i[0] ** 2)
             + 0.5 * J * (w[-1] ** 2 - w[0] ** 2))
    return abs(e_in - e_out) / abs(e_in)


def test_power_is_conserved_through_the_coupling():
    B, T_load = 1e-4, 0.05
    b = p.CircuitBuilder()
    b.add_sine_voltage_source("V", "a", "gnd", 2.0, V, F)
    b.add_dc_machine("M", "a", "gnd", "w", R_a=R, L_a=L, K=K, J=J,
                     B=B, T_load=T_load)
    r = p.simulate(b, t_end=T_END, dt=12.5e-6, engine="pwl")
    # Measured 4.2e-9 (roundoff over 1600 steps): the interval-average
    # products are the trapezoidal rule's own discrete energy identity.
    assert _energy_residual(r, B, T_load) <= 1e-8
    # TR-BDF2's BDF2 stage does not satisfy that identity exactly;
    # measured 5.5e-6 (the design asked 1e-6; see DECISIONS.md D8).
    var = p.simulate(b, t_end=T_END, engine="trbdf2")
    assert _energy_residual(var, B, T_load) <= 2e-5


def test_legacy_kt_different_from_ke_is_refused():
    b = p.CircuitBuilder()
    with pytest.raises(ValueError, match="same number"):
        b.add_dc_machine("M", "a", "gnd", "w", R_a=R, L_a=L, Ke=0.1,
                         Kt=0.12, J=J)
    # Equal legacy constants are just K.
    b.add_dc_machine("M", "a", "gnd", "w", R_a=R, L_a=L, Ke=0.1, Kt=0.1,
                     J=J)


@pytest.mark.parametrize("bad", [dict(L_a=0.0), dict(J=-1.0),
                                 dict(K=0.0), dict(R_a=-0.1),
                                 dict(B=float("nan"))])
def test_invalid_parameters_are_refused_by_name(bad):
    kw = dict(R_a=R, L_a=L, K=K, J=J)
    kw.update(bad)
    with pytest.raises(ValueError, match="add_dc_machine\\('M7'\\)"):
        p.CircuitBuilder().add_dc_machine("M7", "a", "gnd", "w", **kw)


def test_theta_node_is_refused_until_a_vccs_exists():
    with pytest.raises(NotImplementedError, match="controlled"):
        p.CircuitBuilder().add_dc_machine("M", "a", "gnd", "w", R_a=R,
                                          L_a=L, K=K, J=J,
                                          theta_node="th")


def test_observer_is_deprecated_in_favour_of_add_dc_machine():
    b = p.CircuitBuilder()
    with pytest.warns(DeprecationWarning, match="add_dc_machine"):
        m = p.add_dc_motor(b, armature_pos="a", armature_neg="gnd",
                           R_a=R, L_a=L, Ke=K, J=J)
    with pytest.warns(DeprecationWarning, match="add_dc_machine"):
        p.make_dc_motor_observer(b, m, dt=1e-5)


def test_machine_needs_no_observer_and_no_warning():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _build()


def test_armature_impedance_in_the_ac_sweep():
    # Exact small-signal sweep (kernel complex MNA): 1 A injected into the
    # armature node, its voltage read back.
    from pulsim._pulsim import run_mna_sweep_kernel

    B = 1e-4
    b = p.CircuitBuilder()
    b.add_dc_machine("M", "a", "gnd", "w", R_a=R, L_a=L, K=K, J=J, B=B)
    f = np.logspace(0, 5, 10)
    k = b.node_id_of("a")
    z = np.asarray(run_mna_sweep_kernel(b.graph, b.pool,
                                        p.SwitchStateMask(0), list(f),
                                        k, k).H)
    s = 2j * np.pi * f
    np.testing.assert_allclose(z, R + s * L + K ** 2 / (s * J + B),
                               rtol=1e-9)


def test_dsed_runs_the_composite():
    # The design expected dsed to refuse the ideal transformer; it runs
    # it (DECISIONS.md D10). Pinned against the exact solution.
    r = _run(engine="dsed")
    assert r.engine_used == "dsed"
    assert _speed_error(r) <= 1e-5           # measured 1.8e-7
