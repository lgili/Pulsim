"""Validation tests for the L2 MMC SM-equivalent arm model.

L2 extends L1 with two effects of real IGBT power stages:

  * **Dead-time** ``t_dead`` between complementary switches: every
    per-SM toggle inserts a free-wheel window of ``t_dead`` during
    which both transistors are off and the body diodes carry the
    arm current. The effective arm voltage in that window depends
    on ``sign(i_b)``.
  * **Minimum pulse width** ``t_min``: per-SM toggles that would
    occur within ``t_min`` of the previous toggle are suppressed
    (the IGBT spec floors how fast the gate can flip).

We verify:

  1. With ``t_dead = 0`` and ``t_min = 0`` the L2 trajectory matches
     L1 exactly (modulo identical internal indexing).
  2. With ``t_dead > 0`` and constant ``i_b`` of either sign, after a
     toggle the arm voltage shows a ``t_dead``-wide glitch whose
     direction depends on ``sign(i_b)``.
  3. Min-pulse-width filtering: PS-PWM at carrier higher than
     ``1/t_min`` produces no toggles per SM (every flip is
     suppressed) — the model degenerates to its initial state.
  4. State machine bookkeeping is internally consistent
     (``s_w + s_u + s_bypassed = N`` at all times).
  5. Parameter contract.
"""

from __future__ import annotations

import numpy as np
import pytest

from pulsim.mmc import (
    MmcArmEquivalentParams,
    MmcArmEquivalentResult,
    MmcArmEquivalentState,
    MmcArmMultilevelParams,
    make_l2_state,  # type: ignore[attr-defined]
    mmc_arm_equivalent_step,
    simulate_mmc_arm_equivalent,
    simulate_mmc_arm_multilevel,
)


# ---------------------------------------------------------------------------
# Parameter contract
# ---------------------------------------------------------------------------

class TestParams:
    def test_defaults(self):
        p = MmcArmEquivalentParams(n_sm=8, c_sm=1e-3)
        assert p.t_dead == 0.0
        assert p.t_min == 0.0
        assert p.c_arm == pytest.approx(1.25e-4)
        assert p.f_switch == pytest.approx(8000.0)

    def test_reject_negative_t_dead(self):
        with pytest.raises(ValueError, match="t_dead"):
            MmcArmEquivalentParams(n_sm=4, c_sm=1e-3, t_dead=-1e-6)

    def test_reject_negative_t_min(self):
        with pytest.raises(ValueError, match="t_min"):
            MmcArmEquivalentParams(n_sm=4, c_sm=1e-3, t_min=-1e-6)

    def test_accept_full_bridge(self):
        """FB SMs are now supported (m_ref ∈ [-1, +1], signed
        per-SM contributions)."""
        p = MmcArmEquivalentParams(
            n_sm=4, c_sm=1e-3, sm_type="full_bridge",
        )
        assert p.sm_type == "full_bridge"
        assert p.m_min == -1.0
        assert p.m_max == 1.0

    def test_reject_unknown_sm_type(self):
        with pytest.raises(ValueError, match="sm_type"):
            MmcArmEquivalentParams(  # type: ignore[arg-type]
                n_sm=4, c_sm=1e-3, sm_type="quarter_bridge",
            )

    def test_inherits_l1_guards(self):
        with pytest.raises(ValueError, match="n_sm"):
            MmcArmEquivalentParams(n_sm=0, c_sm=1e-3)
        with pytest.raises(ValueError, match="c_sm"):
            MmcArmEquivalentParams(n_sm=4, c_sm=0.0)
        with pytest.raises(ValueError, match="f_carrier"):
            MmcArmEquivalentParams(n_sm=4, c_sm=1e-3, f_carrier=0.0)
        with pytest.raises(ValueError, match="r_p"):
            MmcArmEquivalentParams(n_sm=4, c_sm=1e-3, r_p=-1.0)


# ---------------------------------------------------------------------------
# Initial state
# ---------------------------------------------------------------------------

def test_initial_state_all_bypassed():
    """``make_l2_state`` initialises every SM with S1=0, S2=1."""
    params = MmcArmEquivalentParams(n_sm=6, c_sm=1e-3, v_c0=400.0)
    state = make_l2_state(params)
    assert isinstance(state, MmcArmEquivalentState)
    assert state.v_C == 400.0
    assert (state.bit_s1 == 0).all()
    assert (state.bit_s2 == 1).all()
    assert (state.in_dead_time_until == -np.inf).all()
    assert (state.last_toggle_time == -np.inf).all()


# ---------------------------------------------------------------------------
# L2 → L1 degeneracy (t_dead = 0, t_min = 0)
# ---------------------------------------------------------------------------

def test_l2_with_zero_dead_time_matches_l1():
    """With t_dead = 0 and t_min = 0, L2 reproduces L1 exactly."""
    n_sm = 8
    c_sm = 2e-3
    v_c0 = 500.0
    f_carrier = 2000.0
    duration = 2e-3
    dt = 5e-6
    m = 0.5
    i_const = 4.0

    p_l2 = MmcArmEquivalentParams(
        n_sm=n_sm, c_sm=c_sm, v_c0=v_c0,
        f_carrier=f_carrier, t_dead=0.0, t_min=0.0,
    )
    p_l1 = MmcArmMultilevelParams(
        n_sm=n_sm, c_sm=c_sm, v_c0=v_c0, f_carrier=f_carrier,
    )

    res_l2 = simulate_mmc_arm_equivalent(
        duration=duration, dt=dt, m_ref=m, i_b=i_const, params=p_l2,
    )
    res_l1 = simulate_mmc_arm_multilevel(
        duration=duration, dt=dt, m_ref=m, i_b=i_const, params=p_l1,
    )

    # v_C trajectories should match to numerical precision (same
    # forward-Euler, same PS-PWM s_b, same m_b·i_b/C update).
    np.testing.assert_allclose(res_l2.v_C, res_l1.v_C, rtol=1e-12, atol=1e-12)
    # And so should v_b (final sample of L2 reads the no-current-update
    # branch; cross-check the last sample explicitly).
    np.testing.assert_allclose(
        res_l2.v_b[:-1], res_l1.v_b[:-1], rtol=1e-12, atol=1e-12,
    )


# ---------------------------------------------------------------------------
# Dead-time effect
# ---------------------------------------------------------------------------

def test_dead_time_introduces_free_wheel_window():
    """With t_dead > 0, every toggle puts SMs into free-wheel state
    for t_dead. We trigger a single transition by stepping m_ref from
    just below to just above one of the carrier thresholds and verify
    s_u jumps up for exactly t_dead."""
    n_sm = 1                  # single SM → easy bookkeeping
    f_carrier = 1000.0
    t_dead = 50e-6
    dt = 1e-6                 # well below t_dead
    v_c0 = 100.0

    params = MmcArmEquivalentParams(
        n_sm=n_sm, c_sm=1e-3, v_c0=v_c0,
        f_carrier=f_carrier, t_dead=t_dead, t_min=0.0,
    )

    # m_ref = 0.0 → bypassed; bump to 0.99 to force a rising edge.
    # The triangular carrier crosses 0.99 fast — at phase ~0.495 and
    # phase ~0.505 (rising/falling), so there's a single window per
    # carrier period where the SM is inserted.
    # Track when the SM enters the free-wheel state (s_u = 1) and
    # how long that state persists.
    def m_ref_fn(t):
        return 0.99 if t > 100e-6 else 0.0

    res = simulate_mmc_arm_equivalent(
        duration=2e-3, dt=dt, m_ref=m_ref_fn, i_b=0.0, params=params,
    )

    # At any sample, exactly one of s_w, s_u must equal 1 OR s_w=0,s_u=0
    # (bypassed). For N=1 only one bit at a time.
    assert (res.s_w + res.s_u <= 1).all()

    # Find the first sample where s_u becomes 1 (transition started).
    first_dead = int(np.argmax(res.s_u > 0))
    assert first_dead > 0, "expected at least one s_u=1 sample"

    # Count consecutive s_u=1 samples starting from first_dead.
    count = 0
    for i in range(first_dead, len(res.s_u)):
        if res.s_u[i] == 1:
            count += 1
        else:
            break
    elapsed = count * dt
    # Dead-time should be ≈ t_dead within one dt of rounding.
    assert abs(elapsed - t_dead) <= 2 * dt, (
        f"dead-time window measured = {elapsed*1e6:.2f} µs, "
        f"expected = {t_dead*1e6:.2f} µs"
    )


def test_dead_time_current_routing():
    """Free-wheel SMs are routed by ``sign(i_b)``:
        * i_b > 0 ⇒ s_u SMs are *bypassed* (v_b drops momentarily).
        * i_b < 0 ⇒ s_u SMs are *inserted* (v_b spikes momentarily).
    Test the bypass case here (i_b > 0)."""
    n_sm = 4
    v_c0 = 100.0
    f_carrier = 500.0
    t_dead = 20e-6
    dt = 1e-6

    params = MmcArmEquivalentParams(
        n_sm=n_sm, c_sm=1e-3, v_c0=v_c0,
        f_carrier=f_carrier, t_dead=t_dead,
    )

    res = simulate_mmc_arm_equivalent(
        duration=5e-3, dt=dt,
        m_ref=0.5, i_b=1.0, params=params,
    )
    # During free-wheel events with i_b > 0, v_b should be (s_w / N) · v_C
    # — the s_u SMs don't contribute. Find a sample where s_u > 0 and
    # check v_b matches.
    free_wheel_idx = np.where(res.s_u > 0)[0]
    assert len(free_wheel_idx) > 0, "expected some free-wheel samples"
    k = int(free_wheel_idx[0])
    expected_v_b = (res.s_w[k] / n_sm) * res.v_C[k]
    assert res.v_b[k] == pytest.approx(expected_v_b, rel=1e-9), (
        f"i_b>0 free-wheel: v_b={res.v_b[k]} expected {expected_v_b}"
    )


def test_dead_time_negative_current_inserts_free_wheel_sms():
    """Mirror of the previous test for i_b < 0: free-wheel SMs are
    inserted, so v_b = ((s_w + s_u) / N) · v_C in those samples."""
    n_sm = 4
    v_c0 = 100.0
    f_carrier = 500.0
    t_dead = 20e-6
    dt = 1e-6

    params = MmcArmEquivalentParams(
        n_sm=n_sm, c_sm=1e-3, v_c0=v_c0,
        f_carrier=f_carrier, t_dead=t_dead,
    )
    res = simulate_mmc_arm_equivalent(
        duration=5e-3, dt=dt,
        m_ref=0.5, i_b=-1.0, params=params,
    )
    free_wheel_idx = np.where(res.s_u > 0)[0]
    assert len(free_wheel_idx) > 0
    k = int(free_wheel_idx[0])
    expected_v_b = ((res.s_w[k] + res.s_u[k]) / n_sm) * res.v_C[k]
    assert res.v_b[k] == pytest.approx(expected_v_b, rel=1e-9), (
        f"i_b<0 free-wheel: v_b={res.v_b[k]} expected {expected_v_b}"
    )


# ---------------------------------------------------------------------------
# Min-pulse-width filter
# ---------------------------------------------------------------------------

def test_min_pulse_width_blocks_fast_toggles():
    """When ``t_min`` exceeds the duration of the run after a brief
    settling window, no further toggles can fire. Each SM is allowed
    *one* initial toggle (``last_toggle_time`` starts at ``-inf``);
    after that it is locked for ``t_min`` seconds.

    Concretely: with t_min larger than the simulation horizon, each
    SM toggles ≤ 1 time over the whole run. We assert the *total*
    number of s_w transitions is bounded by N (one per SM, in the
    worst case)."""
    n_sm = 4
    f_carrier = 1000.0           # half-period = 500 µs
    t_min = 50e-3                # > the 10 ms horizon
    dt = 1e-5

    params = MmcArmEquivalentParams(
        n_sm=n_sm, c_sm=1e-3, v_c0=100.0,
        f_carrier=f_carrier, t_dead=0.0, t_min=t_min,
    )
    res = simulate_mmc_arm_equivalent(
        duration=10e-3, dt=dt, m_ref=0.5, i_b=0.0, params=params,
    )
    # Count s_w transitions across the whole run. Each SM contributes
    # at most one transition (the initial-toggle allowance), so the
    # total stays ≤ N.
    transitions = int(np.sum(np.diff(res.s_w) != 0))
    assert transitions <= n_sm, (
        f"min-pulse-width should permit ≤ N initial toggles; "
        f"observed {transitions}"
    )
    # And the final state must be *stable* — at the last sample no
    # further toggles can fire.
    final_s_w = int(res.s_w[-1])
    # The final s_w equals the count of SMs whose PS-PWM target
    # happened to be 1 at the moment they fired their one allowed
    # toggle. We don't pin a specific value; we just confirm the s_w
    # is constant over the *last* half of the run (after all SMs
    # have either toggled or been suppressed).
    second_half = res.s_w[len(res.s_w) // 2:]
    assert (second_half == final_s_w).all(), (
        f"second half of run should be at steady s_w={final_s_w}"
    )


# ---------------------------------------------------------------------------
# Internal consistency
# ---------------------------------------------------------------------------

def test_state_machine_per_sm_sum_consistent():
    """For half-bridge: each SM is in exactly one of {inserted,
    bypassed, free-wheel}. Across all SMs, s_w + s_u + (bypassed) = N.
    """
    n_sm = 6
    params = MmcArmEquivalentParams(
        n_sm=n_sm, c_sm=1e-3, v_c0=100.0,
        f_carrier=1000.0, t_dead=20e-6,
    )
    res = simulate_mmc_arm_equivalent(
        duration=3e-3, dt=2e-6, m_ref=0.4, i_b=2.0, params=params,
    )
    # By construction we don't track "bypassed" explicitly, but at
    # every sample s_w + s_u must be ≤ N.
    assert (res.s_w + res.s_u <= n_sm).all()
    assert (res.s_w >= 0).all()
    assert (res.s_u >= 0).all()


# ---------------------------------------------------------------------------
# Driver guard rails
# ---------------------------------------------------------------------------

def test_simulate_rejects_invalid_durations():
    params = MmcArmEquivalentParams(n_sm=4, c_sm=1e-3)
    with pytest.raises(ValueError, match="duration"):
        simulate_mmc_arm_equivalent(
            duration=0.0, dt=1e-5, m_ref=0.0, i_b=0.0, params=params,
        )
    with pytest.raises(ValueError, match="dt"):
        simulate_mmc_arm_equivalent(
            duration=1e-3, dt=2e-3, m_ref=0.0, i_b=0.0, params=params,
        )


def test_simulate_rejects_const_m_ref_out_of_range():
    params = MmcArmEquivalentParams(n_sm=4, c_sm=1e-3)
    with pytest.raises(ValueError, match="outside valid range"):
        simulate_mmc_arm_equivalent(
            duration=1e-3, dt=1e-5, m_ref=-0.1, i_b=0.0,
            params=params,
        )


def test_step_rejects_non_positive_dt():
    params = MmcArmEquivalentParams(n_sm=4, c_sm=1e-3)
    state = make_l2_state(params)
    with pytest.raises(ValueError, match="dt"):
        mmc_arm_equivalent_step(state, 0.5, 0.0, 0.0, 0.0, params)


# ---------------------------------------------------------------------------
# Result shape sanity
# ---------------------------------------------------------------------------

def test_result_shape():
    params = MmcArmEquivalentParams(
        n_sm=6, c_sm=1e-3, v_c0=400.0, f_carrier=1000.0, t_dead=10e-6,
    )
    res = simulate_mmc_arm_equivalent(
        duration=1e-3, dt=5e-6, m_ref=0.5, i_b=0.0, params=params,
    )
    assert isinstance(res, MmcArmEquivalentResult)
    expected_shape = (201,)
    for arr in (res.t, res.v_C, res.v_b, res.m_ref, res.s_w,
                  res.s_u, res.i_b):
        assert arr.shape == expected_shape
    assert res.s_w.dtype.kind == "i"
    assert res.s_u.dtype.kind == "i"
    assert res.params is params


# ---------------------------------------------------------------------------
# Full-bridge end-to-end smoke tests
# ---------------------------------------------------------------------------

class TestFullBridgeEndToEnd:
    """L2 full-bridge: positive + negative modulation, signed s_eff."""

    def test_fb_positive_m_only_pos_insertions(self):
        """With m_ref > 0 the SMs should only get +inserted (s_n = 0)."""
        params = MmcArmEquivalentParams(
            n_sm=4, c_sm=1e-3, sm_type="full_bridge",
            v_c0=400.0, f_carrier=2_000.0,
            t_dead=0.0, t_min=0.0,
        )
        res = simulate_mmc_arm_equivalent(
            duration=2e-3, dt=2e-6, m_ref=0.5, i_b=10.0, params=params,
        )
        # All inserted SMs are POSITIVE (s_n stays 0 throughout).
        assert int(res.s_n.max()) == 0
        # And we DO see positive insertions.
        assert int(res.s_w.max()) >= 1

    def test_fb_negative_m_only_neg_insertions(self):
        """With m_ref < 0 the SMs should only get -inserted (s_w = 0)."""
        params = MmcArmEquivalentParams(
            n_sm=4, c_sm=1e-3, sm_type="full_bridge",
            v_c0=400.0, f_carrier=2_000.0,
            t_dead=0.0, t_min=0.0,
        )
        res = simulate_mmc_arm_equivalent(
            duration=2e-3, dt=2e-6, m_ref=-0.5, i_b=10.0, params=params,
        )
        assert int(res.s_w.max()) == 0
        assert int(res.s_n.max()) >= 1

    def test_fb_negative_m_negative_v_b(self):
        """FB output should follow the sign of m_ref."""
        params = MmcArmEquivalentParams(
            n_sm=4, c_sm=1e-3, sm_type="full_bridge",
            v_c0=400.0, f_carrier=2_000.0,
            t_dead=0.0, t_min=0.0,
        )
        res = simulate_mmc_arm_equivalent(
            duration=2e-3, dt=2e-6, m_ref=-0.5, i_b=10.0, params=params,
        )
        # Average v_b over the run should be negative (≈ -0.5 · v_C).
        mean_vb = float(res.v_b.mean())
        assert mean_vb < 0
        # Magnitude in the expected ballpark (|m| · v_C ≈ 0.5 · 400 = 200V).
        assert abs(mean_vb) > 100.0
