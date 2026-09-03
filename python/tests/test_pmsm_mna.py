"""MNA-native PMSM — audit C.3 ("Máquinas nativas no MNA").

The Python PMSM is an OBSERVER: back-EMF from the previous step's
(theta, omega) injected through b_extra_fn, forward-Euler
mechanics, and three stator inductors carrying ONE average L.
Three consequences, each measured before this model existed:

* one-step-lag coupling is first order in dt. Open-loop PMSM,
  settled speed vs a dt = 2e-6 reference:

      dt       observer      native (this)
      1e-5      -0.058 %      -0.0001 %
      5e-5      -0.372 %      -0.0005 %      <- the tests' own dt
      2e-4      -1.983 %      -0.0010 %
      5e-4      -7.877 %      +0.0030 %

* the average L erases saliency from the electrical dynamics.
  Locked rotor, Rs = 0.5, Ld = 1 mH, Lq = 3 mH, step response:

      axis   physical   observer        native
      d       2.00 ms   4.00 (+100 %)   2.001 ms
      q       6.00 ms   4.00 (-33 %)    5.93 ms

  so an IPM's phase currents were simply wrong, with no L(theta)
  harmonics and no HFI possible at all;

* a machine living in a Python closure is invisible to anything
  that needs the map to be linear in the MNA state.

This model puts the machine IN the matrix: flux linkage as the
state, the full L(theta) = T^-1 diag(Ld, Lq, L0) T stamped per
Newton iteration, and omega / theta as NODES with capacitors J and
1 F — so the mechanics get the trapezoidal companion in the same
solve. Torque is the co-energy of the same L(theta); the
reluctance term appears because L(theta) is in the matrix.

Cost, honestly: ~33 us/step against ~3 us for the observer — a
Newton solve per step instead of a closure — but with ~100x the
step for the same accuracy.
"""

import math

import numpy as np
import pytest

import pulsim as p

RS, PSI, PP = 0.5, 0.05, 4


def _sources(b, f_e=30.0, v_peak=12.0):
    for k, node in enumerate(("ua", "ub", "uc")):
        b.add_sine_voltage_source(
            f"Vs_{'abc'[k]}", node, "gnd", v_dc=0.0, v_amplitude=v_peak,
            frequency=f_e, phase=-2.0 * math.pi / 3.0 * k)


def _native(dt, *, L_d=2e-3, L_q=2e-3, t_end=0.25, J=1e-3, B=1e-4):
    b = p.CircuitBuilder()
    _sources(b)
    b.add_pmsm_mna("M1", "ua", "ub", "uc", "nn", "w", "th",
                   R_s=RS, L_d=L_d, L_q=L_q, psi_pm=PSI, pole_pairs=PP,
                   J=J, B=B)
    return b, p.simulate(b, t_end=t_end, dt=dt, engine="pwl")


def _observer(dt, *, t_end=0.25):
    b = p.CircuitBuilder()
    _sources(b)
    m = p.add_pmsm(b, name="M1", phase_nodes=("ua", "ub", "uc"),
                   neutral_node="nn", R_s=RS, L_s=2e-3, psi_pm=PSI,
                   pole_pairs=PP, J=1e-3, B=1e-4)
    obs, bx = p.make_pmsm_observer(b, m, dt=dt)
    p.simulate(b, t_end=t_end, dt=dt, engine="pwl",
               step_observer=obs, b_extra_fn=bx)
    return m.mech.omega_rad_s


# ---------------------------------------------------------------
# Saliency is in the electrical dynamics.
# ---------------------------------------------------------------

def _locked_rotor_tau(axis, L_d=1e-3, L_q=3e-3):
    """Step a d- or q-axis voltage into a rotor held by J = 1e6 and
    read the current's 1 - 1/e time. Must be L_axis / R_s."""
    b = p.CircuitBuilder()
    v = 6.0
    va, vb, vc = ((v, -v / 2, -v / 2) if axis == "d"
                  else (0.0, v * math.sqrt(3) / 2, -v * math.sqrt(3) / 2))
    for k, (node, vk) in enumerate(zip(("ua", "ub", "uc"), (va, vb, vc))):
        b.add_voltage_source(f"Vs_{'abc'[k]}", node, "gnd", vk)
    b.add_pmsm_mna("M1", "ua", "ub", "uc", "nn", "w", "th",
                   R_s=RS, L_d=L_d, L_q=L_q, psi_pm=PSI, pole_pairs=PP,
                   J=1e6, B=0.0)
    res = p.simulate(b, t_end=0.03, dt=1e-6, engine="pwl")
    t = np.asarray(res.times)
    ia, ib, ic = (np.asarray(res.i(f"M1_{x}")) for x in "abc")
    # theta_e = 0: i_d = i_a, i_q = (i_b - i_c)/sqrt3
    i = ia if axis == "d" else (ib - ic) / math.sqrt(3)
    i_inf = i[-1]
    k = int(np.argmax(i >= (1.0 - math.exp(-1.0)) * i_inf))
    return float(t[k])


def test_d_axis_time_constant_is_Ld_over_Rs():
    assert _locked_rotor_tau("d") == pytest.approx(1e-3 / RS, rel=0.02)


def test_q_axis_time_constant_is_Lq_over_Rs():
    """The observer reads 4 ms here (-33 %); the physics is 6 ms."""
    assert _locked_rotor_tau("q") == pytest.approx(3e-3 / RS, rel=0.02)


def test_a_round_rotor_has_equal_axes():
    td = _locked_rotor_tau("d", L_d=2e-3, L_q=2e-3)
    tq = _locked_rotor_tau("q", L_d=2e-3, L_q=2e-3)
    assert td == pytest.approx(tq, rel=0.02)


# ---------------------------------------------------------------
# No lag: the two models agree where the observer is accurate,
# and the native one stays put where the observer drifts.
# ---------------------------------------------------------------

def test_agrees_with_the_observer_at_small_dt():
    """Same physics, same conventions: at dt = 2e-6 the lagged
    coupling's error is ~0.01 % and the two must coincide there.
    This is also the check that the torque sign and the phase
    offsets match the FOC chain."""
    _, res = _native(2e-6)
    w_native = float(np.asarray(res.v("w"))[-1])
    w_obs = _observer(2e-6)
    assert w_native == pytest.approx(w_obs, rel=2e-3), (w_native, w_obs)


def test_second_order_in_dt_where_the_observer_is_first_order():
    """Measured: observer -7.9 % at dt = 5e-4, native +0.003 %."""
    _, ref = _native(2e-6)
    w_ref = float(np.asarray(ref.v("w"))[-1])
    _, coarse = _native(5e-4)
    w_coarse = float(np.asarray(coarse.v("w"))[-1])
    native_err = abs(w_coarse / w_ref - 1.0)
    assert native_err < 5e-4, native_err          # < 0.05 %
    obs_err = abs(_observer(5e-4) / _observer(2e-6) - 1.0)
    assert obs_err > 0.05, obs_err                # > 5 %
    assert native_err < obs_err / 50


# ---------------------------------------------------------------
# Torque comes from the same L(theta) that is in the matrix.
# ---------------------------------------------------------------

def test_reluctance_torque_appears_on_a_salient_rotor():
    """With Ld != Lq the co-energy of L(theta) carries the term
    (3/2) pp (Ld - Lq) i_d i_q. The attached T_em trace is the dq
    form of the same thing; the two must agree — and differ from
    the magnet-only torque by that term."""
    _, res = _native(2e-5, L_d=1e-3, L_q=3e-3, t_end=0.15)
    t_em = np.asarray(res.signal("M1.T_em"))
    i_d = np.asarray(res.signal("M1.i_d"))
    i_q = np.asarray(res.signal("M1.i_q"))
    magnet = 1.5 * PP * PSI * i_q
    rel = 1.5 * PP * (1e-3 - 3e-3) * i_d * i_q
    assert np.allclose(t_em, magnet + rel, atol=1e-9)
    # And the reluctance term is not negligible on this machine.
    assert np.max(np.abs(rel)) > 0.05 * np.max(np.abs(magnet))


# ---------------------------------------------------------------
# API surface: everything is a node, a branch, or a named trace.
# ---------------------------------------------------------------

def test_speed_and_angle_are_nodes_and_traces():
    b, res = _native(5e-5, t_end=0.05)
    w = np.asarray(res.v("w"))
    th = np.asarray(res.v("th"))
    assert np.allclose(np.asarray(res.signal("M1.omega")), w)
    assert np.allclose(np.asarray(res.signal("M1.theta")), th)
    # theta is the integral of omega (1 F capacitor fed by omega).
    t = np.asarray(res.times)
    th_int = np.concatenate([[0.0], np.cumsum(0.5 * (w[1:] + w[:-1])
                                              * np.diff(t))])
    assert np.allclose(th, th_int, atol=2e-3 * max(1.0, abs(th[-1])))


def test_phase_currents_read_like_any_inductor():
    b, res = _native(5e-5, t_end=0.02)
    for ph in "abc":
        i = np.asarray(res.i(f"M1_{ph}"))
        assert i.shape == np.asarray(res.times).shape
        assert np.all(np.isfinite(i))
    # Star point: the three currents sum to zero by KCL.
    tot = sum(np.asarray(res.i(f"M1_{ph}")) for ph in "abc")
    assert np.max(np.abs(tot)) < 1e-9


def test_the_device_reports_its_kind():
    b = p.CircuitBuilder()
    _sources(b)
    b.add_pmsm_mna("M1", "ua", "ub", "uc", "nn", "w", "th",
                   R_s=RS, L_d=2e-3, L_q=2e-3, psi_pm=PSI, pole_pairs=PP,
                   J=1e-3)
    for ph in "abc":
        k = str(b.pool.kind_of(b.branch_id_of(f"M1_{ph}")))
        assert k == "StoredKind.PmsmMna", k
    m = b.pool.pmsm_mna_machines()
    assert len(m) == 1 and m[0]["params"]["pole_pairs"] == PP


def test_the_mechanical_balance_holds_over_any_window():
    """The point of mechanics-as-nodes: J*domega/dt = T_em - B*omega
    is stamped, not stepped, so integrating it over ANY window must
    close exactly (to trapezoidal accuracy):

        integral(T_em) = B * integral(omega) + J * (omega_end - omega_start)

    That is the honest form. A first version of this test asserted
    mean(T_em) = B*omega_sync over the tail of the run and failed by
    17 % at B = 1e-4 — the hunting around synchronous speed had not
    damped and the inertial term J*domega/dt does not average to
    zero over a window that does not close whole cycles. The
    balance below carries that term, so it holds regardless.

    And it IS a synchronous machine on a stiff supply: the mean
    speed sits on 2*pi*30/pp for any load it can carry, and a
    heavier viscous load shows up as more torque, not less speed."""
    w_sync = 2.0 * math.pi * 30.0 / PP
    J = 1e-3
    mean_torque = {}
    for B in (1e-4, 1e-2):
        _, res = _native(5e-5, B=B, J=J, t_end=0.4)
        t = np.asarray(res.times)
        m = t > 0.25
        tw, w = t[m], np.asarray(res.v("w"))[m]
        t_em = np.asarray(res.signal("M1.T_em"))[m]
        lhs = np.trapezoid(t_em, tw)
        rhs = B * np.trapezoid(w, tw) + J * (w[-1] - w[0])
        assert lhs == pytest.approx(rhs, rel=0.02, abs=1e-6), (B, lhs, rhs)
        assert w.mean() == pytest.approx(w_sync, rel=0.02), (B, w.mean())
        mean_torque[B] = t_em.mean()
    assert mean_torque[1e-2] > 5 * mean_torque[1e-4], mean_torque


# ---------------------------------------------------------------
# Refusals.
# ---------------------------------------------------------------

@pytest.mark.parametrize("kw,frag", [
    ({"J": 0.0}, "J"),
    ({"J": -1.0}, "J"),
    ({"B": -1.0}, "B"),
    ({"L_d": 0.0}, "L_d"),
    ({"L_q": -1e-3}, "L_q"),
    ({"pole_pairs": 2.5}, "pole_pairs"),
])
def test_bad_parameters_are_refused_by_name(kw, frag):
    base = dict(R_s=RS, L_d=2e-3, L_q=2e-3, psi_pm=PSI, pole_pairs=PP,
                J=1e-3)
    base.update(kw)
    b = p.CircuitBuilder()
    with pytest.raises(Exception, match=frag):
        b.add_pmsm_mna("M1", "ua", "ub", "uc", "nn", "w", "th", **base)


def test_it_blocks_the_trbdf2_router():
    """The flux history is wired for the trapezoidal companion only;
    routing onto a BDF2 second stage would converge and be wrong."""
    b = p.CircuitBuilder()
    _sources(b)
    b.add_pmsm_mna("M1", "ua", "ub", "uc", "nn", "w", "th",
                   R_s=RS, L_d=2e-3, L_q=2e-3, psi_pm=PSI, pole_pairs=PP,
                   J=1e-3)
    why = p._trbdf2_blockers(
        b, dt=None, step_observer=None, closed_loops=[], on_cache=None,
        live_stream=None, progress=False, start_from_dc_op=False,
        strict_event_iterations=False, switch_fn=None,
        controller_period=None, max_dt_halvings=None, store_every=None,
        mmc_arms=None, enable_substep_state_correction=None,
        inductor_freeze_di_max=None, inductor_abs_clamp=None)
    assert any("PMSM" in w for w in why), why
