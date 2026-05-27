"""YAML `chain:` wiring for the C++ sensorless observers (Phase 2.3).

Verifies the two new ``_BLOCK_HANDLERS`` registrations in
:mod:`pulsim.yaml_chain` — ``sliding_mode_observer`` and
``flux_mras_observer`` — by declaring an observer purely via YAML,
poking synthetic excitation into the chain channels, and checking the
estimate against the analytical truth.

These blocks are pure channel-based (no topology / branch-var indices
needed), so the tests use a trivial dummy RC circuit just to satisfy
the YAML loader.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

pulsim = pytest.importorskip("pulsim")


_DUMMY_RC_YAML = """
circuit:
  devices:
    - {type: voltage_source, name: V1, from: a, to: gnd, V: 0.0}
    - {type: resistor,       name: R1, from: a, to: b,   R: 1.0}
    - {type: capacitor,      name: C1, from: b, to: gnd, C: 1.0e-6}
"""


def _has_cxx_observer(name: str) -> bool:
    return (hasattr(pulsim, "_pulsim")
            and hasattr(pulsim, "wire_chain_from_yaml")
            and hasattr(pulsim._pulsim.CxxBlockChain(), name))


@pytest.mark.skipif(not _has_cxx_observer("add_sliding_mode_observer"),
                    reason="C++ SMO adapter / YAML chain support not in this build")
def test_yaml_chain_wires_sliding_mode_observer():
    """YAML `sliding_mode_observer` block — feed synthetic PMSM
    back-EMF excitation into channels and check θ̂ locks within 20°
    RMS in the last 30 ms (matching the standalone-adapter tolerance
    after PLL lock)."""
    Rs, Ls, psi_pm, pp = 0.5, 200e-6, 0.01, 4
    omega_m = 200.0
    omega_e = pp * omega_m
    E_peak = pp * psi_pm * omega_m
    chain_spec = f"""
- type: sliding_mode_observer
  Rs: {Rs}
  Ls: {Ls}
  K_sl: {2.0 * max(E_peak, 1.0)}
  omega_lpf: {2.0 * math.pi * 2000.0}
  f_init_hz: {omega_e / (2 * math.pi)}
  v_alpha_channel: v_alpha
  v_beta_channel:  v_beta
  i_alpha_channel: i_alpha
  i_beta_channel:  i_beta
  theta_hat_channel: theta_hat
  omega_hat_channel: omega_hat
"""
    loaded = pulsim.load_yaml_string(_DUMMY_RC_YAML)
    chain = pulsim.wire_chain_from_yaml(loaded, chain_spec)

    dt = 1e-5
    t_end = 0.15
    n_steps = int(t_end / dt)
    theta_e = 0.0
    last_err = []
    obs = chain.make_step_observer(dt)
    for k in range(n_steps):
        theta_e += omega_e * dt
        if theta_e >= 2 * math.pi:
            theta_e -= 2 * math.pi
        c, s = math.cos(theta_e), math.sin(theta_e)
        chain.set_channel("v_alpha", Rs * (-s) + (-E_peak * s))
        chain.set_channel("v_beta",  Rs * (+c) + (+E_peak * c))
        chain.set_channel("i_alpha", -s)
        chain.set_channel("i_beta",  +c)
        obs(k * dt, np.zeros(1))
        if k > int(0.12 / dt):
            err = chain.get_channel("theta_hat") - theta_e
            err = math.atan2(math.sin(err), math.cos(err))
            last_err.append(err)

    rms = math.sqrt(sum(e * e for e in last_err) / len(last_err))
    assert rms < math.radians(20.0), (
        f"YAML-wired SMO didn't lock: RMS error {math.degrees(rms):.2f}°")


@pytest.mark.skipif(not _has_cxx_observer("add_flux_mras_observer"),
                    reason="C++ MRAS adapter / YAML chain support not in this build")
def test_yaml_chain_wires_flux_mras_observer():
    """YAML `flux_mras_observer` block — feed synthetic IM excitation
    (Schauder voltage-model equivalent) and verify ω̂ converges to the
    true ω_e within 5% after 500 ms."""
    Rs, Ls, Lr, Lm, Rr = 2.46, 0.0948, 0.0948, 0.0874, 1.23
    sigma = 1.0 - (Lm ** 2) / (Ls * Lr)
    L_sigma_s = sigma * Ls
    omega_e_true = 200.0
    psi_mag = 0.5
    i_mag = 7.0

    chain_spec = f"""
- type: flux_mras_observer
  Rs: {Rs}
  Ls: {Ls}
  Lr: {Lr}
  Lm: {Lm}
  Rr: {Rr}
  Kp_mras: 200.0
  Ki_mras: 5000.0
  normalise_eps: true
  v_alpha_channel: v_alpha
  v_beta_channel:  v_beta
  i_alpha_channel: i_alpha
  i_beta_channel:  i_beta
  omega_hat_channel: omega_hat
"""
    loaded = pulsim.load_yaml_string(_DUMMY_RC_YAML)
    chain = pulsim.wire_chain_from_yaml(loaded, chain_spec)

    dt = 100e-6
    theta = 0.0
    obs = chain.make_step_observer(dt)
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
        chain.set_channel("v_alpha",
                          Rs * i_a + L_sigma_s * di_a + (Lm / Lr) * dpsi_a)
        chain.set_channel("v_beta",
                          Rs * i_b + L_sigma_s * di_b + (Lm / Lr) * dpsi_b)
        chain.set_channel("i_alpha", i_a)
        chain.set_channel("i_beta",  i_b)
        obs(k * dt, np.zeros(1))

    omega_hat = chain.get_channel("omega_hat")
    err_pct = abs(omega_hat - omega_e_true) / omega_e_true * 100
    assert err_pct < 5.0, (
        f"YAML-wired MRAS didn't converge: ω̂={omega_hat:.2f}, "
        f"true={omega_e_true:.2f}, err={err_pct:.2f}%")
