"""Integration tests for the L0 MMC arm wired into pulsim's kernel.

These tests stress the co-simulation path (step_observer + b_extra_fn).
The arm-capacitor state lives in Python (``arm.v_C``); the kernel only
sees a voltage source whose value is updated each step. We exercise:

  * Constant current driving constant modulation → linear v_C ramp
    (the kernel-side equivalent of the pure-Python ``dc_charging``
    test).
  * Full-bridge sign-swing on m_b.
  * Multi-arm observer (one closure, several arms).
  * Build-time guard rails (m_b(0) out of range, empty arm list,
    non-positive dt).

All tests pre-warm the kernel state with ``start_from_dc_op=True``
because the L0 model's correctness depends on having the steady-state
arm current at t=0. With DC-OP seeding the observer-driven v_C carries
a +1·dt step offset relative to the pure-Python driver — documented in
``make_mmc_arms_observer``'s docstring, well below L0 fidelity targets.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import pulsim as p


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_arm_with_current_source(
    *,
    n_sm: int,
    c_sm: float,
    v_c0: float,
    m_b,
    i_const: float,
    sm_type: str = "half_bridge",
):
    """Single arm + constant current source between gnd and top."""
    b = p.CircuitBuilder()
    params = p.MmcArmAverageParams(
        n_sm=n_sm, c_sm=c_sm, v_c0=v_c0,
        sm_type=sm_type,  # type: ignore[arg-type]
    )
    arm = p.add_mmc_arm_average(
        b, name="A1", node_a="top", node_b="gnd",
        params=params, m_b=m_b,
    )
    b.add_current_source("I_drive", "top", "gnd", i_const)
    return b, arm


# ---------------------------------------------------------------------------
# 1. Constant m_b, constant i_b → linear v_C ramp
# ---------------------------------------------------------------------------

def test_dc_charging_through_circuit():
    """Force i_b = 4 A through a constant-m_b arm; ``arm.v_C`` ramps at
    ``(m·i)/C_arm`` V/s. With DC-OP seeding the final value matches
    the closed form ``(m·i/C_arm)·(duration + dt)`` (one extra step
    from the boundary-fire observer convention — see docstring)."""
    n_sm = 10
    c_sm = 1e-3        # ⇒ C_arm = 1e-4 F
    m = 0.5
    i_const = 4.0
    v_c0 = 0.0
    duration = 1e-3
    dt = 1e-6

    b, arm = _build_arm_with_current_source(
        n_sm=n_sm, c_sm=c_sm, v_c0=v_c0, m_b=m, i_const=i_const,
    )

    obs, bex = p.make_mmc_arm_observer(b, arm, dt=dt)
    p.simulate(
        b, t_end=duration, dt=dt,
        step_observer=obs, b_extra_fn=bex,
        start_from_dc_op=True,
    )

    # Closed form: with DC OP the observer fires (duration/dt + 1)
    # times, each contributing dt·(m·i)/C_arm.
    expected = (m * i_const / arm.params.c_arm) * (duration + dt)
    assert arm.v_C == pytest.approx(expected, rel=1e-6), (
        f"got {arm.v_C} expected {expected}"
    )
    # ⇒ (0.5 · 4 / 1e-4) · (1e-3 + 1e-6) = 20.02 V


# ---------------------------------------------------------------------------
# 2. Full-bridge negative m_b
# ---------------------------------------------------------------------------

def test_full_bridge_negative_modulation():
    """FB arm with m_b = -0.5 and i_b = 4 A: v_C charges negatively
    at the same rate."""
    n_sm = 10
    c_sm = 1e-3
    m = -0.5
    i_const = 4.0
    v_c0 = 100.0
    duration = 1e-3
    dt = 1e-6

    b, arm = _build_arm_with_current_source(
        n_sm=n_sm, c_sm=c_sm, v_c0=v_c0,
        m_b=m, i_const=i_const, sm_type="full_bridge",
    )
    obs, bex = p.make_mmc_arm_observer(b, arm, dt=dt)
    p.simulate(
        b, t_end=duration, dt=dt,
        step_observer=obs, b_extra_fn=bex,
        start_from_dc_op=True,
    )

    expected = v_c0 + (m * i_const / arm.params.c_arm) * (duration + dt)
    assert arm.v_C == pytest.approx(expected, rel=1e-6)
    # ⇒ 100 + (-0.5 · 4 / 1e-4) · 1.001e-3 = 100 - 20.02 = 79.98 V


# ---------------------------------------------------------------------------
# 3. Kernel-coupled ≡ standalone (up to the +1 step boundary offset)
# ---------------------------------------------------------------------------

def test_kernel_path_matches_standalone_simulate():
    """Drive both the standalone model and the kernel-coupled model
    with the same i_b(t) and confirm v_C trajectories agree within
    one forward-Euler step."""
    n_sm = 8
    c_sm = 2e-3
    m = 0.4
    i_const = 3.0
    v_c0 = 50.0
    duration = 5e-4
    dt = 1e-6
    params = p.MmcArmAverageParams(n_sm=n_sm, c_sm=c_sm, v_c0=v_c0)

    # Standalone trajectory.
    res = p.simulate_mmc_arm_average(
        duration=duration, dt=dt,
        m_b=m, i_b=i_const,
        params=params,
    )
    # Standalone v_C[-1] represents v_C(t=duration). The kernel-coupled
    # path produces v_C(t=duration+dt) due to the boundary-fire
    # observer convention — exactly one extra forward-Euler step.
    expected = res.v_C[-1] + dt * (m * i_const / params.c_arm)

    # Kernel-coupled trajectory.
    b = p.CircuitBuilder()
    arm = p.add_mmc_arm_average(
        b, name="A", node_a="top", node_b="gnd",
        params=params, m_b=m,
    )
    b.add_current_source("Id", "top", "gnd", i_const)
    obs, bex = p.make_mmc_arm_observer(b, arm, dt=dt)
    p.simulate(b, t_end=duration, dt=dt,
                  step_observer=obs, b_extra_fn=bex,
                  start_from_dc_op=True)

    assert arm.v_C == pytest.approx(expected, rel=1e-6)


# ---------------------------------------------------------------------------
# 4. Two arms share one observer
# ---------------------------------------------------------------------------

def test_two_arms_share_observer():
    """A list of arms produces a single combined observer / b_extra
    pair that drives each independently."""
    b = p.CircuitBuilder()
    params_a = p.MmcArmAverageParams(n_sm=10, c_sm=1e-3, v_c0=0.0)
    params_b_ = p.MmcArmAverageParams(n_sm=10, c_sm=1e-3, v_c0=0.0)
    arm_a = p.add_mmc_arm_average(
        b, name="A1", node_a="top1", node_b="gnd",
        params=params_a, m_b=0.5,
    )
    arm_b = p.add_mmc_arm_average(
        b, name="A2", node_a="top2", node_b="gnd",
        params=params_b_, m_b=0.25,
    )
    b.add_current_source("Id1", "top1", "gnd", 4.0)
    b.add_current_source("Id2", "top2", "gnd", 8.0)

    dt = 1e-6
    duration = 1e-3
    obs, bex = p.make_mmc_arms_observer(b, [arm_a, arm_b], dt=dt)
    p.simulate(b, t_end=duration, dt=dt,
                  step_observer=obs, b_extra_fn=bex,
                  start_from_dc_op=True)

    # Both arms see independent dynamics. Account for the +1·dt offset.
    expected_a = (0.5 * 4 / params_a.c_arm) * (duration + dt)   # 20.02 V
    expected_b = (0.25 * 8 / params_b_.c_arm) * (duration + dt) # 20.02 V
    assert arm_a.v_C == pytest.approx(expected_a, rel=1e-6)
    assert arm_b.v_C == pytest.approx(expected_b, rel=1e-6)


# ---------------------------------------------------------------------------
# 5. Sinusoidal current driving constant m_b → ripple matches M·Î/(C·ω)
# ---------------------------------------------------------------------------

def test_sinusoidal_current_produces_expected_ripple():
    """Use ``add_sine_current_source`` (if available) to drive the arm
    with a sinusoidal i_b; v_C ripple zero-to-peak should equal
    ``M·Î / (C_arm · ω)`` within 5 % (forward-Euler tolerance)."""
    n_sm = 10
    c_sm = 1e-3
    m = 0.5
    I_hat = 50.0
    f = 50.0
    omega = 2.0 * math.pi * f
    v_c0 = 1000.0
    n_periods = 5
    T_period = 1.0 / f
    duration = n_periods * T_period
    dt = T_period / 2000.0

    b = p.CircuitBuilder()
    params = p.MmcArmAverageParams(n_sm=n_sm, c_sm=c_sm, v_c0=v_c0)
    arm = p.add_mmc_arm_average(
        b, name="A1", node_a="top", node_b="gnd",
        params=params, m_b=m,
    )
    if hasattr(b, "add_sine_current_source"):
        b.add_sine_current_source(
            "I_sine", "top", "gnd",
            i_dc=0.0, i_amplitude=I_hat, frequency=f, phase=0.0,
        )
    else:
        pytest.skip(
            "pulsim build lacks add_sine_current_source; "
            "the Python-only equivalent in test_mmc_arm_average.py "
            "already validates this physics.",
        )

    obs, bex = p.make_mmc_arm_observer(b, arm, dt=dt)
    v_C_log: list[tuple[float, float]] = []

    def logging_obs(t, x):
        obs(t, x)
        v_C_log.append((t, arm.v_C))

    p.simulate(b, t_end=duration, dt=dt,
                  step_observer=logging_obs, b_extra_fn=bex,
                  start_from_dc_op=True)

    v_ripple_hat = m * I_hat / (arm.params.c_arm * omega)
    log_arr = np.array(v_C_log)
    last_period_mask = log_arr[:, 0] >= (duration - T_period)
    v_C_last = log_arr[last_period_mask, 1]
    ripple_zero_to_peak = 0.5 * (v_C_last.max() - v_C_last.min())

    assert ripple_zero_to_peak == pytest.approx(v_ripple_hat, rel=0.05), (
        f"kernel-coupled sine ripple {ripple_zero_to_peak:.2f} V "
        f"does not match analytical {v_ripple_hat:.2f} V"
    )


# ---------------------------------------------------------------------------
# 6. Build-time guard rails
# ---------------------------------------------------------------------------

def test_invalid_initial_modulation_rejected():
    """``add_mmc_arm_average`` validates ``m_b(0)`` at build time."""
    b = p.CircuitBuilder()
    params = p.MmcArmAverageParams(n_sm=4, c_sm=1e-3)  # HB ⇒ [0, 1]
    with pytest.raises(ValueError, match="outside valid range"):
        p.add_mmc_arm_average(
            b, name="A", node_a="top", node_b="gnd",
            params=params, m_b=-0.5,
        )


def test_invalid_initial_modulation_callable_rejected():
    """Callable ``m_b`` is also checked at t=0."""
    b = p.CircuitBuilder()
    params = p.MmcArmAverageParams(n_sm=4, c_sm=1e-3)
    with pytest.raises(ValueError, match="outside valid range"):
        p.add_mmc_arm_average(
            b, name="A", node_a="top", node_b="gnd",
            params=params, m_b=lambda t: -0.5 - t,  # starts at -0.5
        )


def test_make_observer_rejects_empty_list():
    b = p.CircuitBuilder()
    with pytest.raises(ValueError, match="at least one"):
        p.make_mmc_arms_observer(b, [], dt=1e-6)


def test_make_observer_rejects_non_positive_dt():
    b = p.CircuitBuilder()
    params = p.MmcArmAverageParams(n_sm=4, c_sm=1e-3)
    arm = p.add_mmc_arm_average(
        b, name="A", node_a="top", node_b="gnd",
        params=params, m_b=0.5,
    )
    with pytest.raises(ValueError, match="dt must be"):
        p.make_mmc_arm_observer(b, arm, dt=0.0)
    with pytest.raises(ValueError, match="dt must be"):
        p.make_mmc_arm_observer(b, arm, dt=-1e-6)


# ===========================================================================
# Three-phase DC/AC MMC topology helper
# ===========================================================================

class TestThreePhaseDcAc:
    """Tests for ``add_mmc_three_phase_dc_ac`` — the full 6-arm
    topology helper."""

    def test_structure_correct_number_of_branches(self):
        """A 3-φ MMC adds 6 arm sources + 6 arm inductors = 12 branches."""
        b = p.CircuitBuilder()
        n_before = b.graph.num_branches
        mmc = p.add_mmc_three_phase_dc_ac(
            b,
            dc_pos="dc_p", dc_neg="gnd",
            ac_nodes=("ac_a", "ac_b", "ac_c"),
            n_sm=10, c_sm=1e-3, l_b=1e-3,
            v_c0=100.0,
            m_signals=(0.5, 0.5, 0.5),  # auto-complement for lower arms
        )
        n_after = b.graph.num_branches
        # 6 voltage sources (arms) + 6 inductors = 12 new branches.
        assert n_after - n_before == 12

        # Six arms accessible.
        assert isinstance(mmc, p.MmcThreePhaseDcAc)
        assert len(mmc.all_arms) == 6
        assert len(mmc.upper_arms) == 3
        assert len(mmc.lower_arms) == 3
        # Each carries its initial v_C.
        for arm in mmc.all_arms:
            assert arm.v_C == 100.0

    def test_three_signal_tuple_auto_complements_lower_arms(self):
        """For half-bridge, providing only 3 m_signals fills in the
        complementary 3 for the lower arms so v_arm_p + v_arm_n = v_C
        (matching the DC bus)."""
        b = p.CircuitBuilder()
        mmc = p.add_mmc_three_phase_dc_ac(
            b,
            dc_pos="dc_p", dc_neg="gnd",
            ac_nodes=("ac_a", "ac_b", "ac_c"),
            n_sm=4, c_sm=1e-3, l_b=1e-3,
            v_c0=200.0,
            m_signals=(0.3, 0.6, 0.9),
        )
        # Upper-arm m_b(0) is what the user passed.
        assert mmc.arm_a_p.m_b_fn(0.0) == pytest.approx(0.3)
        assert mmc.arm_b_p.m_b_fn(0.0) == pytest.approx(0.6)
        assert mmc.arm_c_p.m_b_fn(0.0) == pytest.approx(0.9)
        # Lower-arm m_b(0) is the complement (1 - m_p) for HB.
        assert mmc.arm_a_n.m_b_fn(0.0) == pytest.approx(0.7)
        assert mmc.arm_b_n.m_b_fn(0.0) == pytest.approx(0.4)
        assert mmc.arm_c_n.m_b_fn(0.0) == pytest.approx(0.1)

    def test_three_signal_tuple_full_bridge_negates(self):
        """For full-bridge, the 3-tuple shorthand negates rather than
        complements (since FB m_b ∈ [-1, 1])."""
        b = p.CircuitBuilder()
        mmc = p.add_mmc_three_phase_dc_ac(
            b,
            dc_pos="dc_p", dc_neg="gnd",
            ac_nodes=("ac_a", "ac_b", "ac_c"),
            n_sm=4, c_sm=1e-3, l_b=1e-3,
            sm_type="full_bridge",
            v_c0=200.0,
            m_signals=(0.3, -0.4, 0.5),
        )
        assert mmc.arm_a_n.m_b_fn(0.0) == pytest.approx(-0.3)
        assert mmc.arm_b_n.m_b_fn(0.0) == pytest.approx(0.4)
        assert mmc.arm_c_n.m_b_fn(0.0) == pytest.approx(-0.5)

    def test_six_signal_tuple_passes_through_independently(self):
        """A 6-tuple gives independent per-arm control."""
        b = p.CircuitBuilder()
        mmc = p.add_mmc_three_phase_dc_ac(
            b,
            dc_pos="dc_p", dc_neg="gnd",
            ac_nodes=("ac_a", "ac_b", "ac_c"),
            n_sm=4, c_sm=1e-3, l_b=1e-3,
            v_c0=200.0,
            m_signals=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6),
        )
        observed = [arm.m_b_fn(0.0) for arm in mmc.all_arms]
        assert observed == pytest.approx([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])

    def test_dc_equilibrium_no_circulating_current(self):
        """Matched DC bus and v_C: zero current flows through arms,
        v_C never drifts."""
        V_dc = 200.0           # DC bus
        n_sm = 4
        c_sm = 1e-3            # C_arm = 2.5e-4 F
        l_b = 1e-3
        v_c0 = V_dc            # matched: v_C = V_dc so m=0.5 gives V_arm = 100 V

        b = p.CircuitBuilder()
        b.add_voltage_source("Vdc", "dc_p", "gnd", V_dc)
        mmc = p.add_mmc_three_phase_dc_ac(
            b,
            dc_pos="dc_p", dc_neg="gnd",
            ac_nodes=("ac_a", "ac_b", "ac_c"),
            n_sm=n_sm, c_sm=c_sm, l_b=l_b,
            v_c0=v_c0,
            m_signals=(0.5, 0.5, 0.5),
        )
        # No AC load — ac_X nodes float; current must be 0 by KCL.
        # We need *some* path to ground for the DC bus; the v-source
        # already provides that internally. But pulsim's MNA may need
        # a path from each phase midpoint to gnd. Add huge bleed
        # resistors so MNA is well-posed but they draw negligible
        # current.
        b.add_resistor("R_bleed_a", "ac_a", "gnd", 1e9)
        b.add_resistor("R_bleed_b", "ac_b", "gnd", 1e9)
        b.add_resistor("R_bleed_c", "ac_c", "gnd", 1e9)

        dt = 1e-5
        obs, bex = p.make_mmc_arms_observer(b, mmc.all_arms, dt=dt)
        p.simulate(b, t_end=1e-3, dt=dt,
                       step_observer=obs, b_extra_fn=bex,
                       start_from_dc_op=True)

        # At equilibrium, all six v_C should be ~v_c0 (no charging).
        # Allow 1 % drift to absorb bleed-resistor leakage + numerical
        # forward-Euler jitter.
        for arm in mmc.all_arms:
            assert arm.v_C == pytest.approx(v_c0, rel=0.01), (
                f"arm {arm.name} drifted to v_C = {arm.v_C} "
                f"(expected ~{v_c0})"
            )

    # -------- Guard rails --------

    def test_rejects_ac_nodes_wrong_length(self):
        b = p.CircuitBuilder()
        with pytest.raises(ValueError, match="ac_nodes"):
            p.add_mmc_three_phase_dc_ac(
                b,
                dc_pos="dc_p", dc_neg="gnd",
                ac_nodes=("a", "b"),  # type: ignore[arg-type]
                n_sm=4, c_sm=1e-3, l_b=1e-3,
                m_signals=(0.5, 0.5, 0.5),
            )

    def test_rejects_invalid_l_b(self):
        b = p.CircuitBuilder()
        with pytest.raises(ValueError, match="l_b"):
            p.add_mmc_three_phase_dc_ac(
                b,
                dc_pos="dc_p", dc_neg="gnd",
                ac_nodes=("a", "b", "c"),
                n_sm=4, c_sm=1e-3, l_b=0.0,
                m_signals=(0.5, 0.5, 0.5),
            )

    def test_rejects_m_signals_wrong_length(self):
        b = p.CircuitBuilder()
        with pytest.raises(ValueError, match="m_signals"):
            p.add_mmc_three_phase_dc_ac(
                b,
                dc_pos="dc_p", dc_neg="gnd",
                ac_nodes=("a", "b", "c"),
                n_sm=4, c_sm=1e-3, l_b=1e-3,
                m_signals=(0.5, 0.5),  # neither 3 nor 6
            )
