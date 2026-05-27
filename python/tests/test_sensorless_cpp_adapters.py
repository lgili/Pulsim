"""Verify the C++ sensorless-observer BlockChain adapters match the
Python observers on synthetic excitation.

Phase 2.3 C++ port — `pulsim._pulsim.CxxBlockChain.add_sliding_mode_observer`
and `add_flux_mras_observer` mirror their Python counterparts step-for-step.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

pulsim = pytest.importorskip("pulsim")

_HAS = (
    hasattr(pulsim, "_pulsim") and
    hasattr(pulsim._pulsim.CxxBlockChain(), "add_sliding_mode_observer") and
    hasattr(pulsim._pulsim.CxxBlockChain(), "add_flux_mras_observer")
)


@pytest.mark.skipif(not _HAS, reason="C++ sensorless adapters not in this build")
def test_smo_cpp_matches_python_on_synthetic_back_emf():
    """C++ SMO must produce the same θ̂ as the Python observer on
    identical synthetic PMSM excitation."""
    Rs, Ls, psi_pm, pp = 0.5, 200e-6, 0.01, 4
    omega_m = 200.0
    omega_e = pp * omega_m
    E_peak = pp * psi_pm * omega_m
    K_sl = 2.0 * max(E_peak, 1.0)
    omega_lpf = 2.0 * math.pi * 2000.0

    # Python reference
    py_smo = pulsim.SlidingModeObserver(
        Rs=Rs, Ls=Ls, K_sl=K_sl,
        f_init_hz=omega_e / (2 * math.pi),
        omega_lpf=omega_lpf)

    # C++ chain
    cxx = pulsim._pulsim.CxxBlockChain()
    cxx.add_sliding_mode_observer(
        Rs=Rs, Ls=Ls, K_sl=K_sl, omega_lpf=omega_lpf,
        f_init_hz=omega_e / (2 * math.pi),
        v_alpha_channel="v_alpha", v_beta_channel="v_beta",
        i_alpha_channel="i_alpha", i_beta_channel="i_beta",
        theta_hat_channel="theta_hat",
        omega_hat_channel="omega_hat")

    dt = 1e-5
    t_end = 0.15
    n_steps = int(t_end / dt)
    theta_e = 0.0
    py_theta_log = []
    cxx_theta_log = []

    for k in range(n_steps):
        theta_e += omega_e * dt
        if theta_e >= 2 * math.pi:
            theta_e -= 2 * math.pi
        c, s = math.cos(theta_e), math.sin(theta_e)
        i_a = -s
        i_b = +c
        e_a = -E_peak * s
        e_b = +E_peak * c
        v_a = Rs * i_a + e_a
        v_b = Rs * i_b + e_b

        # Python
        py_out = py_smo.update(v_alpha=v_a, v_beta=v_b,
                                  i_alpha=i_a, i_beta=i_b, dt=dt)
        py_theta_log.append(py_out[0])

        # C++ — feed channels then step
        cxx.set_channel("v_alpha", v_a)
        cxx.set_channel("v_beta", v_b)
        cxx.set_channel("i_alpha", i_a)
        cxx.set_channel("i_beta", i_b)
        # Trigger the chain step by calling its step_observer with
        # a dummy state vector.
        obs = cxx.make_step_observer(dt)
        obs(k * dt, np.zeros(1))
        cxx_theta_log.append(cxx.get_channel("theta_hat"))

    # Compare in steady state (last 30 ms).
    n_keep = int(0.03 / dt)
    py_arr = np.asarray(py_theta_log[-n_keep:])
    cxx_arr = np.asarray(cxx_theta_log[-n_keep:])
    diff = cxx_arr - py_arr
    diff = np.arctan2(np.sin(diff), np.cos(diff))  # wrap
    rms_diff = float(np.sqrt(np.mean(diff ** 2)))
    assert rms_diff < math.radians(2.0), (
        f"C++ SMO θ̂ differs from Python by RMS "
        f"{math.degrees(rms_diff):.3f}° (> 2°)")


@pytest.mark.skipif(not _HAS, reason="C++ sensorless adapters not in this build")
def test_mras_cpp_matches_python_on_synthetic_excitation():
    """C++ MRAS with normalised cross-product must converge to the
    true ω_e, matching the Python observer."""
    Rs, Ls, Lr, Lm, Rr = 2.46, 0.0948, 0.0948, 0.0874, 1.23
    sigma = 1.0 - (Lm ** 2) / (Ls * Lr)
    L_sigma_s = sigma * Ls
    omega_e_true = 200.0
    psi_mag = 0.5
    i_mag = 7.0

    py_mras = pulsim.FluxMRASObserver(
        Rs=Rs, Ls=Ls, Lr=Lr, Lm=Lm, Rr=Rr,
        Kp_mras=200.0, Ki_mras=5000.0,
        normalise_eps=True)

    cxx = pulsim._pulsim.CxxBlockChain()
    cxx.add_flux_mras_observer(
        Rs=Rs, Ls=Ls, Lr=Lr, Lm=Lm, Rr=Rr,
        Kp_mras=200.0, Ki_mras=5000.0,
        normalise_eps=True,
        v_alpha_channel="v_alpha", v_beta_channel="v_beta",
        i_alpha_channel="i_alpha", i_beta_channel="i_beta",
        omega_hat_channel="omega_hat")

    dt = 100e-6
    theta = 0.0
    obs = cxx.make_step_observer(dt)
    for k in range(int(0.5 / dt)):
        theta += omega_e_true * dt
        if theta >= 2 * math.pi:
            theta -= 2 * math.pi
        i_a = i_mag * math.cos(theta)
        i_b = i_mag * math.sin(theta)
        dpsi_a = -psi_mag * omega_e_true * math.sin(theta)
        dpsi_b = +psi_mag * omega_e_true * math.cos(theta)
        di_a = -i_mag * omega_e_true * math.sin(theta)
        di_b = +i_mag * omega_e_true * math.cos(theta)
        v_a = Rs * i_a + L_sigma_s * di_a + (Lm / Lr) * dpsi_a
        v_b = Rs * i_b + L_sigma_s * di_b + (Lm / Lr) * dpsi_b

        py_mras.update(v_alpha=v_a, v_beta=v_b,
                          i_alpha=i_a, i_beta=i_b, dt=dt)
        cxx.set_channel("v_alpha", v_a)
        cxx.set_channel("v_beta", v_b)
        cxx.set_channel("i_alpha", i_a)
        cxx.set_channel("i_beta", i_b)
        obs(k * dt, np.zeros(1))

    py_err = abs(py_mras.omega_hat - omega_e_true) / omega_e_true * 100
    cxx_err = (abs(cxx.get_channel("omega_hat") - omega_e_true)
               / omega_e_true * 100)
    assert py_err < 5.0, f"Python MRAS failed convergence: {py_err:.2f}%"
    assert cxx_err < 5.0, f"C++ MRAS failed convergence: {cxx_err:.2f}%"
    # Both should be in close agreement.
    delta = abs(cxx.get_channel("omega_hat") - py_mras.omega_hat)
    assert delta < 5.0, (
        f"C++ vs Python MRAS disagree: cxx={cxx.get_channel('omega_hat'):.2f}, "
        f"py={py_mras.omega_hat:.2f}, |Δ|={delta:.2f}")
