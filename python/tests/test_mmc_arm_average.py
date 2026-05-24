"""Validation tests for the L0 MMC average-arm model.

The reference equations come from Sousa (2022), chapter 2:

    v_b  = m_b · v_C                          (2.13)
    dv_C / dt = (m_b · i_b) / C               (2.14)
    C = C_SM / N                              (def.)

We exercise the model against four closed-form cases that span the
DC-charging, sinusoidal-ripple, lossy-decay, and full-bridge-sign-swing
regimes, plus a handful of guard-rail / API-contract tests.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from pulsim.mmc import (
    MmcArmAverageParams,
    MmcArmAverageResult,
    mmc_arm_average_step,
    simulate_mmc_arm_average,
)


# ---------------------------------------------------------------------------
# Parameter contract
# ---------------------------------------------------------------------------

class TestParams:
    """Construction-time validation of MmcArmAverageParams."""

    def test_defaults(self):
        p = MmcArmAverageParams(n_sm=10, c_sm=1e-3)
        assert p.n_sm == 10
        assert p.c_sm == 1e-3
        assert p.sm_type == "half_bridge"
        assert p.v_c0 == 0.0
        assert p.r_p is None
        assert p.c_arm == pytest.approx(1e-4)  # 1e-3 / 10
        assert p.m_min == 0.0
        assert p.m_max == 1.0

    def test_full_bridge_range(self):
        p = MmcArmAverageParams(n_sm=4, c_sm=2e-3, sm_type="full_bridge")
        assert p.m_min == -1.0
        assert p.m_max == 1.0

    def test_reject_zero_or_negative_n_sm(self):
        with pytest.raises(ValueError, match="n_sm must be"):
            MmcArmAverageParams(n_sm=0, c_sm=1e-3)
        with pytest.raises(ValueError, match="n_sm must be"):
            MmcArmAverageParams(n_sm=-2, c_sm=1e-3)

    def test_reject_non_positive_c_sm(self):
        with pytest.raises(ValueError, match="c_sm must be"):
            MmcArmAverageParams(n_sm=4, c_sm=0.0)
        with pytest.raises(ValueError, match="c_sm must be"):
            MmcArmAverageParams(n_sm=4, c_sm=-1e-6)

    def test_reject_unknown_sm_type(self):
        with pytest.raises(ValueError, match="sm_type must be"):
            MmcArmAverageParams(  # type: ignore[arg-type]
                n_sm=4, c_sm=1e-3, sm_type="quarter_bridge",
            )

    def test_reject_non_positive_r_p(self):
        with pytest.raises(ValueError, match="r_p must be"):
            MmcArmAverageParams(n_sm=4, c_sm=1e-3, r_p=0.0)
        with pytest.raises(ValueError, match="r_p must be"):
            MmcArmAverageParams(n_sm=4, c_sm=1e-3, r_p=-100.0)


# ---------------------------------------------------------------------------
# Step function — minimal-unit closed-form checks
# ---------------------------------------------------------------------------

class TestStep:
    """Direct exercises of mmc_arm_average_step()."""

    def test_zero_current_keeps_v_C(self):
        params = MmcArmAverageParams(n_sm=10, c_sm=1e-3, v_c0=500.0)
        v_C_next, v_b = mmc_arm_average_step(
            v_C=500.0, m_b=0.5, i_b=0.0, dt=1e-5, params=params,
        )
        assert v_C_next == pytest.approx(500.0)
        assert v_b == pytest.approx(250.0)  # m_b · v_C = 0.5 · 500

    def test_v_b_equation(self):
        """v_b should equal m_b · v_C, regardless of current."""
        params = MmcArmAverageParams(n_sm=4, c_sm=1e-3)
        for m_b in (0.0, 0.3, 0.7, 1.0):
            for v_C in (100.0, 1000.0):
                _, v_b = mmc_arm_average_step(
                    v_C=v_C, m_b=m_b, i_b=42.0, dt=1e-6, params=params,
                )
                assert v_b == pytest.approx(m_b * v_C)

    def test_charging_matches_closed_form(self):
        """One-step forward Euler: Δv_C = (m·i/C) · dt."""
        params = MmcArmAverageParams(n_sm=10, c_sm=2e-3)  # C_arm = 2e-4 F
        m_b = 0.5
        i_b = 10.0
        dt = 1e-4
        # dv = (m·i / C_arm) · dt = (0.5 · 10 / 2e-4) · 1e-4 = 2.5 V
        v_C_next, _ = mmc_arm_average_step(
            v_C=0.0, m_b=m_b, i_b=i_b, dt=dt, params=params,
        )
        assert v_C_next == pytest.approx(2.5)

    def test_reject_non_positive_dt(self):
        params = MmcArmAverageParams(n_sm=4, c_sm=1e-3)
        with pytest.raises(ValueError, match="dt must be"):
            mmc_arm_average_step(0.0, 0.5, 1.0, dt=0.0, params=params)
        with pytest.raises(ValueError, match="dt must be"):
            mmc_arm_average_step(0.0, 0.5, 1.0, dt=-1e-6, params=params)

    def test_loss_resistance_decays_v_C(self):
        """With i_b=0 and r_p set, v_C should decay toward zero."""
        params = MmcArmAverageParams(
            n_sm=10, c_sm=1e-3, r_p=100.0, v_c0=1000.0,
        )
        # τ = R_p · C_arm = 100 · 1e-4 = 0.01 s; one step at dt=1e-5
        # gives Δv ≈ -v_C / τ · dt = -1.0 V.
        v_C_next, _ = mmc_arm_average_step(
            v_C=1000.0, m_b=0.0, i_b=0.0, dt=1e-5, params=params,
        )
        assert v_C_next == pytest.approx(999.0, rel=1e-6)


# ---------------------------------------------------------------------------
# Time-series driver — DC, sinusoidal, exponential decay
# ---------------------------------------------------------------------------

class TestSimulate:
    """End-to-end checks of simulate_mmc_arm_average()."""

    def test_returns_result_dataclass(self):
        params = MmcArmAverageParams(n_sm=4, c_sm=1e-3, v_c0=100.0)
        res = simulate_mmc_arm_average(
            duration=1e-3, dt=1e-5, m_b=0.5, i_b=0.0, params=params,
        )
        assert isinstance(res, MmcArmAverageResult)
        # duration / dt = 100 ⇒ 101 samples
        assert res.t.shape == (101,)
        assert res.v_C.shape == (101,)
        assert res.v_b.shape == (101,)
        assert res.m_b.shape == (101,)
        assert res.i_b.shape == (101,)
        # Echo back the params for traceability.
        assert res.params is params

    def test_constant_modulation_zero_current(self):
        """v_C stays at its initial value when i_b = 0 and no losses."""
        params = MmcArmAverageParams(n_sm=10, c_sm=1e-3, v_c0=750.0)
        res = simulate_mmc_arm_average(
            duration=10e-3, dt=1e-5,
            m_b=0.5, i_b=0.0,
            params=params,
        )
        # v_C never changes.
        assert np.allclose(res.v_C, 750.0)
        # v_b = m_b · v_C = 0.5 · 750 = 375 throughout.
        assert np.allclose(res.v_b, 375.0)

    def test_dc_charging_linear_ramp(self):
        """With constant m_b and i_b, v_C ramps linearly.

        Closed form: v_C(t) = v_C0 + (m_b · i_b / C_arm) · t.
        """
        params = MmcArmAverageParams(n_sm=10, c_sm=1e-3, v_c0=0.0)
        m_b = 0.5
        i_b = 4.0
        dvdt = (m_b * i_b) / params.c_arm  # = (0.5 · 4) / 1e-4 = 2e4 V/s
        T = 1e-3
        res = simulate_mmc_arm_average(
            duration=T, dt=1e-6,
            m_b=m_b, i_b=i_b,
            params=params,
        )
        # End-point should be ≈ dvdt · T = 20 V.
        assert res.v_C[-1] == pytest.approx(dvdt * T, rel=1e-4)
        # Trajectory should be a straight line (R² ≈ 1).
        slope, intercept = np.polyfit(res.t, res.v_C, deg=1)
        assert slope == pytest.approx(dvdt, rel=1e-4)
        assert intercept == pytest.approx(0.0, abs=1e-9)

    def test_sinusoidal_ripple_amplitude(self):
        """For m_b = M (constant) and i_b(t) = Î cos(ωt), the steady-state
        ripple has amplitude  M · Î / (C · ω) — the integral of the
        sinusoidal charging current divided by the equivalent capacitance.

        Mathematical derivation:
            dv_C/dt = (M · Î cos(ωt)) / C
            ⇒ v_C(t) = v_C0 + (M · Î / (C ω)) · sin(ωt)
            ⇒ ripple peak-to-peak = 2 · M · Î / (C · ω)
            ⇒ no DC drift (the integral of cos over a period is 0).
        """
        N = 10
        C_SM = 1e-3
        C_arm = C_SM / N           # = 1e-4 F
        params = MmcArmAverageParams(n_sm=N, c_sm=C_SM, v_c0=1000.0)
        M = 0.5
        I_hat = 50.0               # A
        f = 50.0                   # Hz
        omega = 2.0 * math.pi * f
        # Analytical peak ripple amplitude (zero-to-peak).
        v_ripple_hat = M * I_hat / (C_arm * omega)
        # ≈ 0.5 · 50 / (1e-4 · 2π·50) ≈ 795 V

        # Simulate 5 periods at dt small enough to resolve them. With
        # f=50 Hz, T=20 ms → 5T = 100 ms. dt = 10 µs → 10001 samples.
        T_period = 1.0 / f
        n_periods = 5
        duration = n_periods * T_period
        dt = T_period / 2000.0
        res = simulate_mmc_arm_average(
            duration=duration, dt=dt,
            m_b=M,
            i_b=lambda t: I_hat * math.cos(omega * t),
            params=params,
        )

        # Look at the LAST full period only — after the integrator has
        # finished its initial transient (here there isn't one because
        # the input mean is zero, but it's good hygiene).
        last_period_idx = res.t >= (duration - T_period)
        v_C_last = res.v_C[last_period_idx]

        ripple_zero_to_peak = 0.5 * (v_C_last.max() - v_C_last.min())
        assert ripple_zero_to_peak == pytest.approx(
            v_ripple_hat, rel=0.02,
        ), (
            f"Sinusoidal ripple amplitude does not match thesis eq (2.14): "
            f"got {ripple_zero_to_peak:.4f} V, expected {v_ripple_hat:.4f} V"
        )

        # Steady-state mean of v_C should be near the initial value: the
        # continuous solution is v_C(t) = v_C0 + (M·Î/(C·ω))·sin(ωt), which
        # has zero mean. Forward Euler leaks O(ω·dt) per sample, so we allow
        # ~0.25 % of the peak ripple as drift. (BDF1 or trapezoidal would
        # tighten this; not worth the complication at L0.)
        assert v_C_last.mean() == pytest.approx(
            1000.0, abs=0.005 * v_ripple_hat,
        )

    def test_lossy_arm_exponential_decay(self):
        """With r_p set and i_b = 0, v_C decays as exp(-t / (R_p · C_arm))."""
        params = MmcArmAverageParams(
            n_sm=10, c_sm=1e-3, r_p=1000.0, v_c0=1000.0,
        )
        tau = params.r_p * params.c_arm  # 1000 · 1e-4 = 0.1 s
        duration = 0.5  # 5τ → decayed to < 1 %
        dt = tau / 1000.0
        res = simulate_mmc_arm_average(
            duration=duration, dt=dt,
            m_b=0.0, i_b=0.0,
            params=params,
        )
        # End point: expected ≈ v_C0 · exp(-duration/τ) = 1000 · e⁻⁵ ≈ 6.74 V.
        expected = 1000.0 * math.exp(-duration / tau)
        # Forward Euler underestimates exponential decay slightly;
        # tolerance reflects the first-order truncation error.
        assert res.v_C[-1] == pytest.approx(expected, rel=0.01)
        # Mid-point at t = τ should equal v_C0 · e⁻¹ ≈ 367.88 V.
        idx = int(tau / dt)
        expected_mid = 1000.0 * math.exp(-1.0)
        assert res.v_C[idx] == pytest.approx(expected_mid, rel=0.01)

    def test_full_bridge_negative_modulation(self):
        """FB topology allows m_b ∈ [-1, 1]; v_b should follow the sign."""
        params = MmcArmAverageParams(
            n_sm=4, c_sm=1e-3, sm_type="full_bridge", v_c0=400.0,
        )
        res = simulate_mmc_arm_average(
            duration=1e-3, dt=1e-5,
            m_b=lambda t: -0.5 + 0.0 * t,  # constant negative
            i_b=0.0,
            params=params,
        )
        # v_C stays at 400 V (no current), v_b stays at m_b · v_C = -200 V.
        assert np.allclose(res.v_C, 400.0)
        assert np.allclose(res.v_b, -200.0)

    def test_callable_inputs_match_scalar(self):
        """Scalar and callable inputs should yield identical trajectories."""
        params = MmcArmAverageParams(n_sm=4, c_sm=1e-3, v_c0=100.0)
        res_scalar = simulate_mmc_arm_average(
            duration=1e-3, dt=1e-5, m_b=0.6, i_b=2.0, params=params,
        )
        res_callable = simulate_mmc_arm_average(
            duration=1e-3, dt=1e-5,
            m_b=lambda _t: 0.6,
            i_b=lambda _t: 2.0,
            params=params,
        )
        assert np.allclose(res_scalar.v_C, res_callable.v_C)
        assert np.allclose(res_scalar.v_b, res_callable.v_b)
        assert np.allclose(res_scalar.m_b, res_callable.m_b)
        assert np.allclose(res_scalar.i_b, res_callable.i_b)

    def test_reject_modulation_out_of_range(self):
        """HB rejects m_b < 0 or m_b > 1; FB rejects |m_b| > 1."""
        params_hb = MmcArmAverageParams(n_sm=4, c_sm=1e-3)
        # m_b = -0.1 at t=0 → out of [0, 1].
        with pytest.raises(ValueError, match="outside valid range"):
            simulate_mmc_arm_average(
                duration=1e-3, dt=1e-5,
                m_b=-0.1, i_b=0.0,
                params=params_hb,
            )
        # m_b = 1.5 → out of [0, 1].
        with pytest.raises(ValueError, match="outside valid range"):
            simulate_mmc_arm_average(
                duration=1e-3, dt=1e-5,
                m_b=1.5, i_b=0.0,
                params=params_hb,
            )

        params_fb = MmcArmAverageParams(
            n_sm=4, c_sm=1e-3, sm_type="full_bridge",
        )
        # m_b = -1.1 → out of [-1, 1].
        with pytest.raises(ValueError, match="outside valid range"):
            simulate_mmc_arm_average(
                duration=1e-3, dt=1e-5,
                m_b=-1.1, i_b=0.0,
                params=params_fb,
            )

    def test_reject_invalid_durations(self):
        params = MmcArmAverageParams(n_sm=4, c_sm=1e-3)
        with pytest.raises(ValueError, match="duration must be"):
            simulate_mmc_arm_average(
                duration=0.0, dt=1e-5, m_b=0.0, i_b=0.0, params=params,
            )
        with pytest.raises(ValueError, match="duration must be"):
            simulate_mmc_arm_average(
                duration=-1e-3, dt=1e-5, m_b=0.0, i_b=0.0, params=params,
            )
        with pytest.raises(ValueError, match="dt must be"):
            simulate_mmc_arm_average(
                duration=1e-3, dt=0.0, m_b=0.0, i_b=0.0, params=params,
            )
        with pytest.raises(ValueError, match="dt must be"):
            simulate_mmc_arm_average(
                duration=1e-3, dt=2e-3,  # dt > duration
                m_b=0.0, i_b=0.0, params=params,
            )


# ---------------------------------------------------------------------------
# Cross-consistency: step() and simulate() should agree.
# ---------------------------------------------------------------------------

def test_step_and_simulate_agree():
    """Hand-rolling a loop with mmc_arm_average_step() should reproduce
    the trajectory of simulate_mmc_arm_average() bit-for-bit."""
    params = MmcArmAverageParams(n_sm=8, c_sm=2e-3, v_c0=500.0)
    duration = 2e-3
    dt = 1e-5

    def m_b_of(t):
        return 0.4 + 0.2 * math.sin(2 * math.pi * 1e3 * t)

    def i_b_of(t):
        return 5.0 * math.cos(2 * math.pi * 1e3 * t)

    res = simulate_mmc_arm_average(
        duration=duration, dt=dt,
        m_b=m_b_of, i_b=i_b_of,
        params=params,
    )

    # Hand-rolled loop.
    n_points = len(res.t)
    v_C = params.v_c0
    v_C_traj = np.zeros(n_points)
    v_b_traj = np.zeros(n_points)
    for k in range(n_points):
        t_k = k * dt
        m_k = m_b_of(t_k)
        i_k = i_b_of(t_k)
        v_C_traj[k] = v_C
        v_b_traj[k] = m_k * v_C
        if k < n_points - 1:
            v_C, _ = mmc_arm_average_step(v_C, m_k, i_k, dt, params)

    np.testing.assert_allclose(res.v_C, v_C_traj, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(res.v_b, v_b_traj, rtol=1e-14, atol=1e-14)
