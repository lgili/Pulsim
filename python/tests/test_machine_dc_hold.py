"""DC operating point of the MNA-native machines with held mechanics
(add-mna-native-machines 2.7, 3.5; DECISIONS D19).

The DC point holds the speed node at `omega0` and the angle node at
`theta0` and solves the stator with di/dt = 0, so the back-EMF is the one
of the prescribed speed. References are closed forms:

* PMSM, round rotor: i_k = (v_k − v_n − e_k)/R_s,
  e_k = ω_e·dλ_pm,k/dθ_e = −ω_e·ψ·sin(θ_e − 2πk/3), Σ i_k = 0;
* induction machine: i = (v − v_n)/R_s, and the rotor flux solves
  dλ_r/dt = 0: [[−1/τ_r, −ω_e], [ω_e, −1/τ_r]]·λ_r + (L_m/τ_r)·i_αβ = 0.

The first transient step continues the DC point: di/dt(0) = 0, so
|i(h) − i(0)| = O(h²) (the zero-state start of the same circuit jumps).
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

import pulsim as p

RS, LS, PSI, PP = 0.5, 2e-3, 0.05, 4
V = (10.0, -4.0, -6.0)


def _pmsm(*, omega0, theta0, L_q=LS, J=1.0, drive_speed=False, v=V):
    b = p.CircuitBuilder()
    for k, node in enumerate(("ua", "ub", "uc")):
        b.add_voltage_source(f"V{'abc'[k]}", node, "gnd", v[k])
    b.add_pmsm_mna("M1", "ua", "ub", "uc", "nn", "w", "th", R_s=RS, L_d=LS,
                   L_q=L_q, psi_pm=PSI, pole_pairs=PP, J=J, B=0.0,
                   omega0=omega0, theta0=theta0)
    if drive_speed:
        b.add_voltage_source("Vw", "w", "gnd", omega0)
    return b


def _run(b, dt=1e-6, n=20, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return p.simulate(b, t_end=n * dt, dt=dt, engine="pwl", **kw)


def _pmsm_closed_form(omega0, theta0):
    th_e, w_e = PP * theta0, PP * omega0
    e = np.array([-w_e * PSI * math.sin(th_e - 2 * math.pi / 3 * k)
                  for k in range(3)])
    u = np.array(V) - e
    return (u - u.mean()) / RS


def _phase_currents(res, k=0):
    return np.array([float(np.asarray(res.i(f"M1_{x}"))[k]) for x in "abc"])


def test_pmsm_dc_point_at_rest_is_v_over_r():
    # salient rotor: at ω0 = 0 the inductance plays no part at DC
    res = _run(_pmsm(omega0=0.0, theta0=0.3, L_q=3 * LS), start_from_dc_op=True)
    i0 = _phase_currents(res)
    v = np.array(V)
    assert i0 == pytest.approx((v - v.mean()) / RS, abs=1e-9)
    assert float(np.asarray(res.v("th"))[0]) == pytest.approx(0.3, abs=1e-12)
    assert float(np.asarray(res.v("w"))[0]) == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("omega0,theta0", [(50.0, 0.2), (-30.0, 1.1)])
def test_pmsm_dc_point_carries_the_back_emf_of_the_held_speed(omega0, theta0):
    res = _run(_pmsm(omega0=omega0, theta0=theta0), start_from_dc_op=True)
    ref = _pmsm_closed_form(omega0, theta0)
    assert np.abs(ref).max() > 1.0   # control: the back-EMF moves the point
    assert np.abs(ref - (np.array(V) - np.mean(V)) / RS).max() > 1.0
    assert _phase_currents(res) == pytest.approx(ref, abs=1e-9)
    assert float(np.asarray(res.v("w"))[0]) == pytest.approx(omega0, abs=1e-12)


@pytest.mark.parametrize("L_q", [LS, 3 * LS])
def test_first_step_continues_the_dc_point(L_q):
    # di/dt(0) = 0 at the DC point: the first step moves i by O(h²).
    # Bound: |d²i/dt²|·h²/2 with |d²i/dt²| <= (ω_e²·ψ + ω_e²·|ΔL|·|i|)/L_min
    # ~ 2.6e6 A/s² here (|i| <= 34 A, salient ΔL = 4 mH); h = 1e-6 gives
    # 1.3e-6 A. Measured 4.8e-7 (round), 1.3e-6 (salient), ratio 4.000
    # per halving; the zero-state start jumps by > 1e-3 A (control). The
    # angle capacitor must start at i_C = C·ω0 for this: at i_C = 0 the
    # first step advanced θ by h·ω0/2 and moved i by 2.4e-3 A.
    omega0, theta0 = 50.0, 0.2

    def first_step(dt):
        b = _pmsm(omega0=omega0, theta0=theta0, L_q=L_q, J=10.0)
        res = _run(b, dt=dt, start_from_dc_op=True)
        return np.abs(_phase_currents(res, 1) - _phase_currents(res, 0)).max()

    di, di_half = first_step(1e-6), first_step(5e-7)
    assert di < 4e-6
    assert di / di_half == pytest.approx(4.0, rel=0.05)
    cold = _run(_pmsm(omega0=omega0, theta0=theta0, L_q=L_q, J=10.0))
    assert np.abs(_phase_currents(cold, 1) - _phase_currents(cold, 0)).max() > 1e-3


def test_compute_dc_op_holds_the_mechanics_exactly_under_gmin():
    # the gmin floor must not land on the held rows: (1 + gmin)·ω = ω0
    # would put ω 5e-11 off at ω0 = 50
    b = _pmsm(omega0=50.0, theta0=0.2)
    x = np.asarray(p.compute_dc_op(b))
    assert x[b.node_id_of("w")] == pytest.approx(50.0, abs=1e-12)
    assert x[b.node_id_of("th")] == pytest.approx(0.2, abs=1e-12)
    i = np.array([x[b.pool.branch_var_id_for_inductor(b.branch_id_of(f"M1_{c}"),
                                                     b.graph)] for c in "abc"])
    assert i == pytest.approx(_pmsm_closed_form(50.0, 0.2), abs=1e-8)


@pytest.mark.parametrize("theta0", [0.0, 0.7])
def test_cold_start_at_an_initial_angle_does_not_jump(theta0):
    # zero sources, at rest: the PM flux linked at θ0 is the start
    # flux, so nothing flows (it seeded λ_pm(0) before: a jump at θ0≠0)
    res = _run(_pmsm(omega0=0.0, theta0=theta0, v=(0.0, 0.0, 0.0)))
    i = np.array([np.asarray(res.i(f"M1_{x}"))[:5] for x in "abc"])
    assert np.abs(i).max() < 1e-9
    assert float(np.asarray(res.v("th"))[0]) == pytest.approx(theta0, abs=1e-12)


def test_a_speed_node_driven_by_a_source_is_refused_by_name():
    b = _pmsm(omega0=50.0, theta0=0.0, drive_speed=True)
    with pytest.raises(Exception, match="Vw"):
        _run(b, start_from_dc_op=True)


def test_no_reference_tie_on_the_machine_nodes():
    # the stator conducts at DC and the mechanical nodes are held: the
    # preflight neither ties nor warns (it tied the neutral, the speed
    # and the angle nodes through 1 GOhm before)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res = p.simulate(_pmsm(omega0=0.0, theta0=0.0), t_end=1e-4, dt=1e-6,
                         engine="pwl")
    assert len(res.plan.ties) == 0


# ---- induction machine ------------------------------------------------
RR, LLS, LLR, LM, PPI = 0.4, 2e-3, 2e-3, 60e-3, 2


def _im(omega0):
    b = p.CircuitBuilder()
    for k, node in enumerate(("ua", "ub", "uc")):
        b.add_voltage_source(f"V{'abc'[k]}", node, "gnd", V[k])
    b.add_induction_machine_mna("M", "ua", "ub", "uc", "nn", "w", R_s=RS,
                                L_ls=LLS, R_r=RR, L_lr=LLR, L_m=LM,
                                pole_pairs=PPI, J=1.0, omega0=omega0)
    return b


@pytest.mark.parametrize("omega0", [0.0, 80.0])
def test_im_dc_point_is_v_over_r_and_the_steady_rotor_flux(omega0):
    res = _run(_im(omega0), start_from_dc_op=True)
    i = np.array([float(np.asarray(res.i(f"M_{x}"))[0]) for x in "abc"])
    v = np.array(V)
    assert i == pytest.approx((v - v.mean()) / RS, abs=1e-9)
    is_ab = np.array([(2 * i[0] - i[1] - i[2]) / 3, (i[1] - i[2]) / math.sqrt(3)])
    tau = (LLR + LM) / RR
    w_e = PPI * omega0
    A = np.array([[-1 / tau, -w_e], [w_e, -1 / tau]])
    lr_ref = np.linalg.solve(A, -(LM / tau) * is_ab)
    lr = np.array([float(np.asarray(res.v(f"M_psi_r{x}"))[0]) for x in "ab"])
    assert lr == pytest.approx(lr_ref, abs=1e-9 * max(1.0, np.abs(lr_ref).max()))
    # first step: no jump (the rotor flux is steady, di/dt = 0)
    i1 = np.array([float(np.asarray(res.i(f"M_{x}"))[1]) for x in "abc"])
    assert np.abs(i1 - i).max() < 1e-6 * np.abs(i).max()
