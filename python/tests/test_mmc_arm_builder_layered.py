"""Integration tests for the L1/L2/L3 MMC arm builder helpers.

Same pattern as ``test_mmc_arm_builder.py`` (L0): force a known
arm current via a current source and verify the per-layer dynamics
match the standalone simulator within the documented +1·dt
boundary bias.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import pulsim as p


# ---------------------------------------------------------------------------
# Shared topology helper (single arm + DC current source)
# ---------------------------------------------------------------------------

def _force_constant_i_b(b, i_const: float) -> None:
    """Wire a current source forcing ``i_const`` ``from→to=top→gnd``
    *through* the arm (charging convention)."""
    b.add_current_source("Id", "top", "gnd", float(i_const))


# ===========================================================================
# L1 — Multilevel arm in a circuit
# ===========================================================================

class TestL1:
    def test_dc_charging_through_circuit(self):
        """Force a constant current and verify v_C ramps at the
        time-averaged ``m·i/C_arm`` rate within one PS-PWM bit."""
        b = p.CircuitBuilder()
        params = p.MmcArmMultilevelParams(
            n_sm=8, c_sm=2e-3, v_c0=0.0, f_carrier=2000.0,
        )
        arm = p.add_mmc_arm_multilevel(
            b, name="A1", node_a="top", node_b="gnd",
            params=params, m_ref=0.5,
        )
        _force_constant_i_b(b, 4.0)

        dt = 1e-6
        duration = 1e-3
        obs, bex = p.make_mmc_arm_multilevel_observer(b, arm, dt=dt)
        p.simulate(
            b, t_end=duration, dt=dt,
            step_observer=obs, b_extra_fn=bex,
            start_from_dc_op=True,
        )
        # Closed form (L0 envelope): (0.5·4/1e-4)·1.001e-3 = 20.02 V.
        # L1 PS-PWM quantizes per step, so the rate fluctuates around
        # the L0 rate. Accept ±2 % (matches the standalone L1↔L0
        # test in test_mmc_arm_multilevel.py).
        expected = (0.5 * 4.0 / params.c_arm) * (duration + dt)
        assert arm.v_C == pytest.approx(expected, rel=0.02), (
            f"L1 in circuit: v_C={arm.v_C}, expected {expected}"
        )

    def test_two_arms_share_observer(self):
        b = p.CircuitBuilder()
        params_a = p.MmcArmMultilevelParams(
            n_sm=8, c_sm=2e-3, v_c0=0.0, f_carrier=2000.0,
        )
        arm_a = p.add_mmc_arm_multilevel(
            b, name="A", node_a="top_a", node_b="gnd",
            params=params_a, m_ref=0.5,
        )
        arm_b = p.add_mmc_arm_multilevel(
            b, name="B", node_a="top_b", node_b="gnd",
            params=params_a, m_ref=0.25,
        )
        b.add_current_source("IdA", "top_a", "gnd", 4.0)
        b.add_current_source("IdB", "top_b", "gnd", 8.0)
        dt = 1e-6
        duration = 1e-3
        obs, bex = p.make_mmc_arm_multilevel_observers(
            b, [arm_a, arm_b], dt=dt,
        )
        p.simulate(
            b, t_end=duration, dt=dt,
            step_observer=obs, b_extra_fn=bex,
            start_from_dc_op=True,
        )
        # Both arms target the same product 0.5·4 = 0.25·8 = 2 V·A.
        target = (2.0 / params_a.c_arm) * (duration + dt)
        assert arm_a.v_C == pytest.approx(target, rel=0.02)
        assert arm_b.v_C == pytest.approx(target, rel=0.02)

    def test_rejects_invalid_initial_modulation(self):
        b = p.CircuitBuilder()
        params = p.MmcArmMultilevelParams(n_sm=4, c_sm=1e-3)
        with pytest.raises(ValueError, match="outside valid range"):
            p.add_mmc_arm_multilevel(
                b, name="A", node_a="top", node_b="gnd",
                params=params, m_ref=-0.1,
            )

    def test_make_observer_guards(self):
        b = p.CircuitBuilder()
        with pytest.raises(ValueError, match="at least one"):
            p.make_mmc_arm_multilevel_observers(b, [], dt=1e-6)
        params = p.MmcArmMultilevelParams(n_sm=4, c_sm=1e-3)
        arm = p.add_mmc_arm_multilevel(
            b, name="A", node_a="top", node_b="gnd",
            params=params, m_ref=0.5,
        )
        with pytest.raises(ValueError, match="dt"):
            p.make_mmc_arm_multilevel_observer(b, arm, dt=0.0)


# ===========================================================================
# L2 — SM-equivalent arm in a circuit
# ===========================================================================

class TestL2:
    def test_dc_charging_through_circuit_zero_dead_time(self):
        """With t_dead = 0, L2 should track L0/L1 closed form."""
        b = p.CircuitBuilder()
        params = p.MmcArmEquivalentParams(
            n_sm=8, c_sm=2e-3, v_c0=0.0,
            f_carrier=2000.0, t_dead=0.0,
        )
        arm = p.add_mmc_arm_equivalent(
            b, name="A", node_a="top", node_b="gnd",
            params=params, m_ref=0.5,
        )
        _force_constant_i_b(b, 4.0)
        dt = 1e-6
        duration = 1e-3
        obs, bex = p.make_mmc_arm_equivalent_observer(b, arm, dt=dt)
        p.simulate(
            b, t_end=duration, dt=dt,
            step_observer=obs, b_extra_fn=bex,
            start_from_dc_op=True,
        )
        expected = (0.5 * 4.0 / params.c_arm) * (duration + dt)
        assert arm.v_C == pytest.approx(expected, rel=0.02)

    def test_dead_time_reduces_charging_rate_for_positive_current(self):
        """With t_dead > 0 and i_b > 0, free-wheel SMs are bypassed,
        which *reduces* the effective m_b at every transition. The
        L2 v_C should rise slightly less than the L1 (no-dead-time)
        equivalent over the same window."""
        # L2 with dead-time
        b1 = p.CircuitBuilder()
        params_l2 = p.MmcArmEquivalentParams(
            n_sm=8, c_sm=2e-3, v_c0=0.0,
            f_carrier=2000.0, t_dead=20e-6,
        )
        arm_l2 = p.add_mmc_arm_equivalent(
            b1, name="A", node_a="top", node_b="gnd",
            params=params_l2, m_ref=0.5,
        )
        _force_constant_i_b(b1, 4.0)
        # L1 reference (no dead-time)
        b2 = p.CircuitBuilder()
        params_l1 = p.MmcArmMultilevelParams(
            n_sm=8, c_sm=2e-3, v_c0=0.0, f_carrier=2000.0,
        )
        arm_l1 = p.add_mmc_arm_multilevel(
            b2, name="A", node_a="top", node_b="gnd",
            params=params_l1, m_ref=0.5,
        )
        _force_constant_i_b(b2, 4.0)

        dt = 1e-6
        duration = 1e-3
        obs2, bex2 = p.make_mmc_arm_equivalent_observer(b1, arm_l2, dt=dt)
        obs1, bex1 = p.make_mmc_arm_multilevel_observer(b2, arm_l1, dt=dt)
        p.simulate(b1, t_end=duration, dt=dt,
                       step_observer=obs2, b_extra_fn=bex2,
                       start_from_dc_op=True)
        p.simulate(b2, t_end=duration, dt=dt,
                       step_observer=obs1, b_extra_fn=bex1,
                       start_from_dc_op=True)

        # L2 should charge less due to dead-time bypass losses.
        assert arm_l2.v_C < arm_l1.v_C, (
            f"L2 v_C ({arm_l2.v_C}) should be ≤ L1 ({arm_l1.v_C}) "
            "due to dead-time"
        )
        # But not vastly less — dead-time duty for these params is
        # 2·N·f·t_dead = 2·8·2000·20e-6 = 64 % → the L2 rate is roughly
        # 36 % of L1. Be lenient (10 %–95 %).
        ratio = arm_l2.v_C / arm_l1.v_C
        assert 0.10 < ratio < 0.95, f"unexpected L2/L1 ratio = {ratio:.3f}"

    def test_state_carries_through_observer(self):
        """The MmcArmEquivalent.state object should accumulate
        per-SM bits and last-toggle times as the simulation runs."""
        b = p.CircuitBuilder()
        params = p.MmcArmEquivalentParams(
            n_sm=4, c_sm=1e-3, v_c0=400.0,
            f_carrier=1000.0, t_dead=20e-6,
        )
        arm = p.add_mmc_arm_equivalent(
            b, name="A", node_a="top", node_b="gnd",
            params=params, m_ref=0.5,
        )
        _force_constant_i_b(b, 2.0)
        obs, bex = p.make_mmc_arm_equivalent_observer(b, arm, dt=1e-6)
        p.simulate(
            b, t_end=2e-3, dt=1e-6,
            step_observer=obs, b_extra_fn=bex,
            start_from_dc_op=True,
        )
        # After 2 ms with f_switch = 8 kHz, every SM should have
        # toggled multiple times → last_toggle_time finite for all.
        assert (arm.state.last_toggle_time > 0).all()


# ===========================================================================
# L3 — Detailed per-SM arm in a circuit
# ===========================================================================

class TestL3:
    def test_dc_charging_through_circuit(self):
        """Sum-of-caps should follow the L0/L1 closed form."""
        b = p.CircuitBuilder()
        params = p.MmcArmDetailedParams(
            n_sm=8, c_sm=2e-3, v_c0=0.0,
            f_carrier=2000.0, balancing="sort_and_select",
        )
        arm = p.add_mmc_arm_detailed(
            b, name="A", node_a="top", node_b="gnd",
            params=params, m_ref=0.5,
        )
        _force_constant_i_b(b, 4.0)
        dt = 1e-6
        duration = 1e-3
        obs, bex = p.make_mmc_arm_detailed_observer(b, arm, dt=dt)
        p.simulate(
            b, t_end=duration, dt=dt,
            step_observer=obs, b_extra_fn=bex,
            start_from_dc_op=True,
        )
        # Closed form on the sum.
        expected_sum = (0.5 * 4.0 / params.c_arm) * (duration + dt)
        assert arm.v_C == pytest.approx(expected_sum, rel=0.05)
        # And the per-SM spread should stay tight under balancing.
        assert arm.v_C_spread < 1.0, (
            f"balanced spread = {arm.v_C_spread:.4f} V, "
            f"expected near zero"
        )

    def test_skewed_initial_state_balances_under_kernel(self):
        """Start with a 40 V per-SM spread; sort-and-select via the
        kernel-driven observer must shrink it just like the standalone
        driver does."""
        b = p.CircuitBuilder()
        params = p.MmcArmDetailedParams(
            n_sm=6, c_sm=1e-3, v_c0=600.0,
            f_carrier=1000.0, balancing="sort_and_select",
        )
        init = np.linspace(80.0, 120.0, 6)
        arm = p.add_mmc_arm_detailed(
            b, name="A", node_a="top", node_b="gnd",
            params=params, m_ref=lambda t: 0.5 + 0.3 * math.sin(2*math.pi*50*t),
            initial_v_C_per_sm=init,
        )
        # Drive with a sinusoidal current to alternate charging/discharging.
        # Pulsim doesn't have add_sine_current_source in 1.0.0 — fake it
        # via a current source that flips at quarter-period boundaries.
        # Simpler still: just inject a constant + use a long duration so
        # balancing kicks in via the natural carrier switching.
        b.add_current_source("Id", "top", "gnd", 5.0)
        dt = 5e-6
        duration = 30e-3
        obs, bex = p.make_mmc_arm_detailed_observer(b, arm, dt=dt)
        p.simulate(
            b, t_end=duration, dt=dt,
            step_observer=obs, b_extra_fn=bex,
            start_from_dc_op=True,
        )
        # Initial spread was 40 V; balancing must shrink it by ≥ 3×.
        assert arm.v_C_spread < 40.0 / 3.0, (
            f"final spread {arm.v_C_spread:.2f} V — "
            f"balancing should reduce by ≥ 3× from 40 V"
        )

    def test_rejects_initial_array_wrong_shape(self):
        b = p.CircuitBuilder()
        params = p.MmcArmDetailedParams(n_sm=4, c_sm=1e-3, v_c0=400.0)
        with pytest.raises(ValueError, match="shape"):
            p.add_mmc_arm_detailed(
                b, name="A", node_a="top", node_b="gnd",
                params=params, m_ref=0.5,
                initial_v_C_per_sm=np.array([1.0, 2.0]),
            )
