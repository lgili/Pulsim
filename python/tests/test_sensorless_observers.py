"""Tests for the sensorless rotor observers (Phase 2.3).

Validates that ``SlidingModeObserver`` locks onto the rotor electrical
angle of a PMSM running at a known speed within a sane tolerance,
without relying on the true mechanical state.

The MRAS test is exercised by a smoke run only — the validation
test that checks slip estimation accuracy needs the IM closed-loop
chain to land first.
"""
from __future__ import annotations

import math

import numpy as np
import pytest


pulsim = pytest.importorskip("pulsim")


_HAS_SMO = hasattr(pulsim, "SlidingModeObserver")


# -----------------------------------------------------------------------
# Pure-Python construction smoke (works even without C++ extension)
# -----------------------------------------------------------------------

def test_smo_constructor_defaults():
    if not hasattr(pulsim, "SlidingModeObserver"):
        pytest.skip("SlidingModeObserver not in this build")
    smo = pulsim.SlidingModeObserver(Rs=0.5, Ls=200e-6)
    assert smo.Rs == 0.5
    assert smo.Ls == 200e-6
    assert smo.theta_hat == 0.0
    assert smo.omega_hat == pytest.approx(2 * math.pi * 50.0)


def test_smo_update_returns_5tuple():
    if not hasattr(pulsim, "SlidingModeObserver"):
        pytest.skip("SlidingModeObserver not in this build")
    smo = pulsim.SlidingModeObserver(Rs=0.5, Ls=200e-6)
    out = smo.update(v_alpha=10.0, v_beta=0.0,
                          i_alpha=0.5, i_beta=0.0, dt=1e-5)
    assert len(out) == 5
    theta_hat, omega_hat, e_a, e_b, flag = out
    assert all(math.isfinite(v) for v in (theta_hat, omega_hat, e_a, e_b))
    assert flag in (0.0, 1.0)


def test_mras_constructor_from_motor_handle():
    if not hasattr(pulsim, "FluxMRASObserver"):
        pytest.skip("FluxMRASObserver not in this build")
    if not hasattr(pulsim, "add_induction_motor"):
        pytest.skip("add_induction_motor not available — IM dependency")
    # Build a stub IM only to extract parameters — no simulation needed.
    b = pulsim.CircuitBuilder()
    kw = pulsim.im_parameters_from_nameplate(
        P_rated_W=4_000.0, V_LL_V=400.0, f_Hz=50.0)
    kw.pop("_diagnostics", None)
    motor = pulsim.add_induction_motor(
        b, name="IM_stub",
        phase_nodes=("a", "b", "c"),
        neutral_node="n",
        T_load=0.0,
        **kw)
    obs = pulsim.FluxMRASObserver.from_motor(motor)
    assert obs.Rs == motor.R_s_ohm
    assert obs.Lm == motor.L_m_H


# -----------------------------------------------------------------------
# SMO functional test against a synthetic back-EMF signal
# -----------------------------------------------------------------------

@pytest.mark.skipif(not _HAS_SMO, reason="pulsim build lacks SlidingModeObserver")
def test_smo_locks_onto_synthetic_back_emf():
    """Feed the SMO synthetic PMSM-shape voltages and currents from a
    rotor spinning at a known speed. Within 200 ms the angle estimate
    SHOULD track the true angle within ~15°.

    Pure-Python integration (no C++ kernel involved) — keeps the test
    fast and isolates the observer from any other simulator behaviour.
    """
    Rs = 0.5
    Ls = 200e-6
    psi_pm = 0.01
    pp = 4
    omega_m = 200.0                    # 200 rad/s mechanical
    omega_e = pp * omega_m             # 800 rad/s electrical
    E_peak = pp * psi_pm * omega_m     # back-EMF peak

    smo = pulsim.SlidingModeObserver(
        Rs=Rs, Ls=Ls,
        K_sl=2.0 * max(E_peak, 1.0),   # set above peak EMF
        f_init_hz=omega_e / (2.0 * math.pi),
        omega_lpf=2.0 * math.pi * 2000.0,
    )

    dt = 1e-5                          # 10 µs
    t_end = 0.2                        # 200 ms
    n_steps = int(t_end / dt)

    # Closed-form steady-state stator currents (assume v_dq align such
    # that i_q = const, i_d = 0 — pure synchronous operation). For
    # simplicity drive a unit i_q.
    i_q_ref = 1.0
    theta_e = 0.0
    angle_errors = []
    for k in range(n_steps):
        # True rotor angle.
        theta_e += omega_e * dt
        if theta_e >= 2.0 * math.pi:
            theta_e -= 2.0 * math.pi
        c, s = math.cos(theta_e), math.sin(theta_e)
        # Stator currents (Park-inverse from i_d = 0, i_q = i_q_ref).
        i_alpha = -i_q_ref * s
        i_beta  = +i_q_ref * c
        # Back-EMF in stationary frame: e_α = −E·sin θ_e, e_β = +E·cos θ_e.
        e_alpha = -E_peak * s
        e_beta  = +E_peak * c
        # Voltage commands that would produce the assumed currents in
        # steady state (no derivative term needed for constant i_q):
        v_alpha = Rs * i_alpha + e_alpha
        v_beta  = Rs * i_beta  + e_beta

        out = smo.update(v_alpha=v_alpha, v_beta=v_beta,
                              i_alpha=i_alpha, i_beta=i_beta, dt=dt)
        theta_hat = out[0]
        # Track error in the last 50 ms (after the PLL has settled).
        if k > int(0.150 / dt):
            # Wrap the error to [-π, π].
            err = theta_hat - theta_e
            err = math.atan2(math.sin(err), math.cos(err))
            angle_errors.append(err)

    err_arr = np.asarray(angle_errors)
    # The PLL projection drives the d-component of the back-EMF
    # vector to zero — d̂ = 0 ⟺ θ̂ = θ_e. So in steady state we
    # expect ``theta_hat ≈ theta_e`` directly (no constant phase
    # offset to subtract).  The only residual error comes from
    # (a) the LPF group delay and (b) the saturating-sign
    # boundary-layer "softness".
    err_max_abs = float(np.max(np.abs(err_arr)))
    err_rms = float(np.sqrt(np.mean(err_arr ** 2)))
    # 20° peak / 10° RMS is a comfortable bound for this tuning
    # (K_sl = 2·E_peak, LPF = 2 kHz, PLL Kp=200, Ki=1e4) at the
    # synthetic fundamental of 800 rad/s ≈ 127 Hz.
    assert err_max_abs < math.radians(20.0), (
        f"SMO peak angle error {math.degrees(err_max_abs):.2f}°, "
        f"RMS {math.degrees(err_rms):.2f}°.")
    assert err_rms < math.radians(10.0), (
        f"SMO RMS angle error {math.degrees(err_rms):.2f}° too high.")


@pytest.mark.skipif(not hasattr(pulsim, "FluxMRASObserver"),
                       reason="FluxMRASObserver not available")
def test_mras_voltage_model_leaky_integrator_no_drift():
    """With ``voltage_model_hpf_omega > 0`` the reference flux must
    NOT drift indefinitely when fed a clean balanced sinusoidal
    excitation — even though ω̂ starts at zero and the adaptive
    loop hasn't yet locked, the reference-flux magnitude must
    settle to a bounded value (≤ a few Wb)."""
    Rs, Ls, Lr, Lm, Rr = 2.46, 0.0948, 0.0948, 0.0874, 1.23
    sigma = 1.0 - (Lm ** 2) / (Ls * Lr)
    L_sigma_s = sigma * Ls
    omega_e = 314.0
    psi_mag = 0.5
    i_mag = 5.0
    mras = pulsim.FluxMRASObserver(
        Rs=Rs, Ls=Ls, Lr=Lr, Lm=Lm, Rr=Rr,
        voltage_model_hpf_omega=5.0,
    )
    dt = 100e-6
    theta = 0.0
    for _ in range(int(0.5 / dt)):
        theta += omega_e * dt
        if theta >= 2 * math.pi:
            theta -= 2 * math.pi
        i_a = i_mag * math.cos(theta)
        i_b = i_mag * math.sin(theta)
        dpsi_a_dt = -psi_mag * omega_e * math.sin(theta)
        dpsi_b_dt = +psi_mag * omega_e * math.cos(theta)
        di_a_dt = -i_mag * omega_e * math.sin(theta)
        di_b_dt = +i_mag * omega_e * math.cos(theta)
        v_a = Rs * i_a + L_sigma_s * di_a_dt + (Lm / Lr) * dpsi_a_dt
        v_b = Rs * i_b + L_sigma_s * di_b_dt + (Lm / Lr) * dpsi_b_dt
        mras.update(v_alpha=v_a, v_beta=v_b,
                       i_alpha=i_a, i_beta=i_b, dt=dt)
    # Bounded — without the leaky integrator, ψ_r_ref would drift
    # away unboundedly under any DC bias in the input.
    psi_ref_mag = math.hypot(mras.psi_r_alpha_ref, mras.psi_r_beta_ref)
    assert 0.1 < psi_ref_mag < 5.0, (
        f"ψ_r_ref magnitude {psi_ref_mag:.3f} out of bounded range "
        f"— leaky integrator may have failed.")
    assert math.isfinite(mras.omega_hat)
