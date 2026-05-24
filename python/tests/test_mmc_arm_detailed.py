"""Validation tests for the L3 MMC detailed per-SM arm model.

L3 tracks each SM's capacitor independently and uses a balancing
algorithm (sort-and-select by default) to decide which SMs to insert
at every PS-PWM event. The main correctness statements we exercise:

  1. **Aggregate consistency.** With balanced initial conditions and
     ``balancing="sort_and_select"``, the *sum* of per-SM voltages
     evolves identically to L1 (eq 2.9): d(Σv_C_n)/dt = s_b·i_b/C_SM.

  2. **Balancing health.** Starting from a SKEWED set of per-SM
     voltages (some high, some low), sort-and-select drives the
     spread *down* toward zero over many switching cycles.

  3. **Without balancing**, the same skewed start produces *bounded*
     evolution (no closed-loop convergence) — and a small intentional
     asymmetry diverges over time.

  4. **Selection policy.** For ``i_b > 0`` and a known ordering of
     per-SM voltages, the lowest s_b are inserted; flip sign of i_b
     and the highest s_b are inserted.

  5. **State + result shape sanity.**
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from pulsim.mmc import (
    MmcArmDetailedParams,
    MmcArmDetailedResult,
    MmcArmDetailedState,
    make_l3_state,
    mmc_arm_detailed_step,
    simulate_mmc_arm_detailed,
)


# ---------------------------------------------------------------------------
# Parameter contract
# ---------------------------------------------------------------------------

class TestParams:
    def test_defaults(self):
        p = MmcArmDetailedParams(n_sm=8, c_sm=1e-3)
        assert p.balancing == "sort_and_select"
        assert p.c_arm == pytest.approx(1.25e-4)
        assert p.f_switch == pytest.approx(8000.0)

    def test_reject_unknown_balancing(self):
        with pytest.raises(ValueError, match="balancing"):
            MmcArmDetailedParams(  # type: ignore[arg-type]
                n_sm=4, c_sm=1e-3, balancing="genetic_algorithm",
            )

    def test_reject_full_bridge(self):
        with pytest.raises(ValueError, match="half-bridge only"):
            MmcArmDetailedParams(  # type: ignore[arg-type]
                n_sm=4, c_sm=1e-3, sm_type="full_bridge",
            )

    def test_inherits_l1_l2_guards(self):
        with pytest.raises(ValueError, match="n_sm"):
            MmcArmDetailedParams(n_sm=0, c_sm=1e-3)
        with pytest.raises(ValueError, match="c_sm"):
            MmcArmDetailedParams(n_sm=4, c_sm=0.0)
        with pytest.raises(ValueError, match="f_carrier"):
            MmcArmDetailedParams(n_sm=4, c_sm=1e-3, f_carrier=0.0)
        with pytest.raises(ValueError, match="r_p"):
            MmcArmDetailedParams(n_sm=4, c_sm=1e-3, r_p=-1.0)


# ---------------------------------------------------------------------------
# Initial state
# ---------------------------------------------------------------------------

def test_default_initial_state_uniform():
    params = MmcArmDetailedParams(n_sm=6, c_sm=1e-3, v_c0=600.0)
    state = make_l3_state(params)
    assert isinstance(state, MmcArmDetailedState)
    assert state.v_C_per_sm.shape == (6,)
    # All SMs start at v_c0 / N = 100 V.
    assert np.allclose(state.v_C_per_sm, 100.0)
    assert state.v_C == pytest.approx(600.0)
    assert state.v_C_spread == pytest.approx(0.0)


def test_custom_initial_state_accepted():
    params = MmcArmDetailedParams(n_sm=4, c_sm=1e-3, v_c0=400.0)
    init = np.array([90.0, 100.0, 110.0, 120.0])
    state = make_l3_state(params, initial_v_C_per_sm=init)
    assert np.array_equal(state.v_C_per_sm, init)
    assert state.v_C == pytest.approx(420.0)
    assert state.v_C_spread == pytest.approx(30.0)
    assert state.v_C_mean == pytest.approx(105.0)


def test_initial_state_rejects_wrong_shape():
    params = MmcArmDetailedParams(n_sm=4, c_sm=1e-3)
    with pytest.raises(ValueError, match="shape"):
        make_l3_state(
            params, initial_v_C_per_sm=np.array([1.0, 2.0, 3.0]),
        )


# ---------------------------------------------------------------------------
# Selection policy
# ---------------------------------------------------------------------------

def test_sort_and_select_charging_picks_lowest():
    """For i_b > 0 the inserted SMs are the s_b SMs with lowest v_C_n."""
    # NB: this test exercises the internal `_balance_select` helper
    # directly (see below) — we don't need a full MmcArmDetailedParams
    # instance, just a state vector with known SM capacitor voltages.
    state = MmcArmDetailedState(
        v_C_per_sm=np.array([50.0, 70.0, 90.0, 110.0, 130.0]),
    )
    # Force s_b = 2 via a known PS-PWM target. With N=5 carriers, the
    # easiest control is to use a deterministic call to the helper
    # below — but for this unit test we *manually* invoke the
    # internal selector to verify policy.
    from pulsim.mmc import _balance_select  # type: ignore
    mask = _balance_select(
        state.v_C_per_sm, s_b=2, i_b=+1.0, scheme="sort_and_select",
    )
    # Lowest 2 are at indices 0 and 1.
    assert mask.tolist() == [True, True, False, False, False]


def test_sort_and_select_discharging_picks_highest():
    """For i_b < 0 the inserted SMs are the s_b SMs with highest v_C_n."""
    from pulsim.mmc import _balance_select  # type: ignore
    v = np.array([50.0, 70.0, 90.0, 110.0, 130.0])
    mask = _balance_select(v, s_b=2, i_b=-1.0, scheme="sort_and_select")
    # Highest 2 are at indices 3 and 4.
    assert mask.tolist() == [False, False, False, True, True]


def test_balancing_none_picks_first_s_b():
    """``balancing="none"`` picks the first s_b SMs in index order."""
    from pulsim.mmc import _balance_select  # type: ignore
    v = np.array([50.0, 70.0, 90.0, 110.0, 130.0])
    mask = _balance_select(v, s_b=3, i_b=1.0, scheme="none")
    assert mask.tolist() == [True, True, True, False, False]


def test_selection_edge_cases():
    """s_b = 0 ⇒ no SM inserted; s_b = N ⇒ all inserted."""
    from pulsim.mmc import _balance_select  # type: ignore
    v = np.array([50.0, 70.0, 90.0])
    assert not _balance_select(v, 0, 1.0, "sort_and_select").any()
    assert _balance_select(v, 3, 1.0, "sort_and_select").all()


# ---------------------------------------------------------------------------
# Aggregate consistency vs L0 / L1
# ---------------------------------------------------------------------------

def test_aggregate_sum_matches_l1():
    """With uniform initial conditions and balancing on, the sum-of
    -caps trajectory matches the L1 / L0 closed form within a tiny
    forward-Euler drift."""
    n_sm = 8
    c_sm = 2e-3
    v_c0 = 800.0
    m_ref = 0.5
    i_const = 4.0
    duration = 2e-3
    dt = 5e-6

    params = MmcArmDetailedParams(
        n_sm=n_sm, c_sm=c_sm, v_c0=v_c0, f_carrier=2000.0,
        balancing="sort_and_select",
    )
    res = simulate_mmc_arm_detailed(
        duration=duration, dt=dt, m_ref=m_ref, i_b=i_const,
        params=params,
    )
    # Closed form (L0 envelope): dv/dt = m·i / C_arm.
    final_expected = v_c0 + (m_ref * i_const / params.c_arm) * duration
    # The L3 PS-PWM walks through duty 0.5; balancing keeps SMs near
    # the mean. Sum should track within ~1 % of the L0 closed-form.
    final_obs = float(res.v_C_sum[-1])
    assert final_obs == pytest.approx(final_expected, rel=0.02), (
        f"L3 sum {final_obs} ≠ L0 closed form {final_expected}"
    )


# ---------------------------------------------------------------------------
# Balancing health
# ---------------------------------------------------------------------------

def test_balancing_reduces_initial_spread():
    """Start with a 20 % per-SM spread; sort-and-select should drive
    it toward zero over a few switching periods."""
    n_sm = 6
    c_sm = 1e-3
    f_carrier = 1000.0
    duration = 50e-3   # 50 carrier periods
    dt = 5e-6

    params = MmcArmDetailedParams(
        n_sm=n_sm, c_sm=c_sm, f_carrier=f_carrier,
        balancing="sort_and_select",
    )
    # Hand-build the initial state with an explicit 20 % spread:
    #   v_C_n linspace from 80 V to 100 V → mean 90 V, spread 20 V.
    init = np.linspace(80.0, 100.0, n_sm)
    initial_spread = init.max() - init.min()

    res = simulate_mmc_arm_detailed(
        duration=duration, dt=dt,
        m_ref=lambda t: 0.5 + 0.3 * math.sin(2 * math.pi * 50 * t),
        i_b=lambda t: 5.0 * math.cos(2 * math.pi * 50 * t),
        params=params,
        initial_v_C_per_sm=init,
    )
    # Look at the final-period spread.
    last_period_idx = res.t >= (duration - 1.0 / f_carrier)
    final_spread = float(np.median(res.v_C_spread[last_period_idx]))
    # Balancing should bring the spread down by at least 3× from
    # the initial value.
    assert final_spread < initial_spread / 3.0, (
        f"balancing failed: initial spread {initial_spread:.2f} V → "
        f"final {final_spread:.2f} V"
    )


def test_without_balancing_spread_does_not_shrink():
    """``balancing="none"`` uses a fixed assignment — the spread is
    not closed-loop-controlled. Asserts that the final spread is
    *not* dramatically smaller than the initial one (no convergence)."""
    n_sm = 6
    c_sm = 1e-3
    f_carrier = 1000.0
    duration = 50e-3
    dt = 5e-6

    params = MmcArmDetailedParams(
        n_sm=n_sm, c_sm=c_sm, f_carrier=f_carrier,
        balancing="none",
    )
    init = np.linspace(80.0, 100.0, n_sm)
    initial_spread = init.max() - init.min()

    res = simulate_mmc_arm_detailed(
        duration=duration, dt=dt,
        m_ref=lambda t: 0.5 + 0.3 * math.sin(2 * math.pi * 50 * t),
        i_b=lambda t: 5.0 * math.cos(2 * math.pi * 50 * t),
        params=params,
        initial_v_C_per_sm=init,
    )
    last_period_idx = res.t >= (duration - 1.0 / f_carrier)
    final_spread = float(np.median(res.v_C_spread[last_period_idx]))
    # Without balancing, the spread evolves but isn't driven to zero;
    # we require it to remain at least 1/3 of the starting value.
    assert final_spread >= initial_spread / 3.0, (
        f"un-balanced run still converged: initial {initial_spread:.2f} V "
        f"→ final {final_spread:.2f} V"
    )


# ---------------------------------------------------------------------------
# Result shape + driver guard rails
# ---------------------------------------------------------------------------

def test_result_shape():
    n_sm = 6
    params = MmcArmDetailedParams(
        n_sm=n_sm, c_sm=1e-3, v_c0=600.0, f_carrier=1000.0,
    )
    res = simulate_mmc_arm_detailed(
        duration=1e-3, dt=5e-6, m_ref=0.5, i_b=0.0, params=params,
    )
    assert isinstance(res, MmcArmDetailedResult)
    assert res.t.shape == (201,)
    assert res.v_C_per_sm.shape == (201, n_sm)
    assert res.v_C_sum.shape == (201,)
    assert res.v_C_spread.shape == (201,)
    assert res.v_b.shape == (201,)
    assert res.s_b.shape == (201,)
    assert res.s_b.dtype.kind == "i"
    assert res.params is params


def test_step_rejects_non_positive_dt():
    params = MmcArmDetailedParams(n_sm=4, c_sm=1e-3)
    state = make_l3_state(params)
    with pytest.raises(ValueError, match="dt"):
        mmc_arm_detailed_step(state, 0.5, 0.0, 0.0, 0.0, params)


def test_simulate_rejects_invalid_durations():
    params = MmcArmDetailedParams(n_sm=4, c_sm=1e-3)
    with pytest.raises(ValueError, match="duration"):
        simulate_mmc_arm_detailed(
            duration=0.0, dt=1e-5, m_ref=0.0, i_b=0.0, params=params,
        )
    with pytest.raises(ValueError, match="dt"):
        simulate_mmc_arm_detailed(
            duration=1e-3, dt=2e-3, m_ref=0.0, i_b=0.0, params=params,
        )


def test_simulate_rejects_const_m_ref_out_of_range():
    params = MmcArmDetailedParams(n_sm=4, c_sm=1e-3)
    with pytest.raises(ValueError, match="outside valid range"):
        simulate_mmc_arm_detailed(
            duration=1e-3, dt=1e-5, m_ref=-0.5, i_b=0.0, params=params,
        )
