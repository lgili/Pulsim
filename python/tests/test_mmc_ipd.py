"""Tests for the In-Phase Disposition (IPD) modulation strategy.

IPD ships alongside PS-PWM as an alternative ``modulation_scheme``
on ``MmcArmMultilevelParams`` / ``MmcArmEquivalentParams`` /
``MmcArmDetailedParams``. Its signature behavior (vs PS-PWM):

  1. All ``N`` carriers share the same phase but sit at stacked DC
     bands of width ``1/N``.
  2. At any instant, ``s_b`` ∈ ``{floor(m·N), floor(m·N) + 1}`` —
     only two adjacent levels per step.
  3. For ``m = k/N`` (k ∈ ℤ ∩ [0, N]) exactly, ``s_b = k`` constant
     (no switching transitions needed).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import pulsim as p


# ---------------------------------------------------------------------------
# IPD switching function — direct unit tests.
# ---------------------------------------------------------------------------

class TestIpdSwitchingFunction:
    @pytest.mark.parametrize("n_sm", [1, 4, 5, 10, 30])
    def test_hb_range(self, n_sm):
        """HB s_b ∈ {0..N} for every (m, t)."""
        for m in [0.0, 0.13, 0.5, 0.79, 1.0]:
            for t in np.linspace(0, 5e-3, 51):
                s = p.ipd_switching_function(m, float(t), n_sm, 1000.0)
                assert 0 <= s <= n_sm, (
                    f"IPD s_b={s} out of [0, {n_sm}] at m={m}, t={t}"
                )

    def test_two_adjacent_levels_only(self):
        """The headline IPD property: at any time, ``s_b`` only takes
        two adjacent values around ``m · N``."""
        n_sm = 8
        m = 0.42  # m·N = 3.36 → expect levels {3, 4}
        seen: set[int] = set()
        for t in np.linspace(0, 10e-3, 1001):
            s = p.ipd_switching_function(m, float(t), n_sm, 1000.0)
            seen.add(s)
        # IPD should hit exactly {3, 4}; allow {3} alone if the
        # triangle never crosses 0.36.
        assert seen <= {3, 4}, (
            f"IPD visited {sorted(seen)} for m=0.42 N=8; expected {{3, 4}}"
        )

    def test_exact_integer_level_no_switching(self):
        """When m = k/N exactly, s_b should stay at k (no transitions)."""
        n_sm = 5
        for k in (0, 1, 2, 3, 4, 5):
            m = k / n_sm
            samples = [
                p.ipd_switching_function(m, float(t), n_sm, 1000.0)
                for t in np.linspace(0, 5e-3, 201)
            ]
            assert all(s == k for s in samples), (
                f"IPD m={m} (k={k}/N) varied: saw {set(samples)}"
            )

    def test_time_average_matches_m_ref(self):
        """Time-averaged s_b/N tracks m_ref to within 1/(2N) for a
        broad sweep — the same accuracy IPD's two-level switching can
        deliver."""
        n_sm = 10
        f = 1000.0
        T = 1.0 / f
        for m in (0.1, 0.27, 0.5, 0.83, 0.95):
            ts = np.linspace(0.0, T, 5001, endpoint=False)
            s_vals = [
                p.ipd_switching_function(float(m), float(t), n_sm, f)
                for t in ts
            ]
            mean_norm = float(np.mean(s_vals)) / n_sm
            # IPD's ω ripple is bounded; mean over an integer number
            # of carrier periods matches m_ref to within 1/(2N).
            assert abs(mean_norm - m) <= 0.5 / n_sm + 1e-3, (
                f"IPD mean(s_b)/N = {mean_norm:.4f} vs m_ref = {m}"
            )

    def test_clamps_out_of_range_silently(self):
        # Below 0 → 0 (no SMs fire).
        assert p.ipd_switching_function(-0.5, 0.0, 4, 1000.0) == 0
        # Above 1 (off the peak) → all N fire.
        assert p.ipd_switching_function(2.0, 1e-7, 4, 1000.0) == 4

    def test_fb_sign_follows_m_ref(self):
        for m in (-0.4, -0.1, 0.1, 0.5):
            for t in np.linspace(0, 1e-3, 21):
                s = p.ipd_switching_function(
                    m, float(t), 4, 1000.0, sm_type="full_bridge",
                )
                if m > 0:
                    assert s >= 0
                elif m < 0:
                    assert s <= 0


# ---------------------------------------------------------------------------
# Integration into the layered step functions.
# ---------------------------------------------------------------------------

class TestIpdInLayers:
    def test_l1_with_ipd(self):
        """L1 multilevel-arm sim with IPD runs end-to-end + cap stays
        balanced."""
        params = p.MmcArmMultilevelParams(
            n_sm=8, c_sm=1e-3, v_c0=400.0, f_carrier=1000.0,
            modulation_scheme="ipd",
        )
        res = p.simulate_mmc_arm_multilevel(
            duration=2e-3, dt=5e-6, m_ref=0.5, i_b=0.0,
            params=params,
        )
        # With i_b=0 v_C stays put.
        assert np.allclose(res.v_C, 400.0)
        # IPD with constant m=0.5 visits {3, 4} at most (since 0.5·8 = 4 exact
        # — actually s_base=4, m_frac=0 → tri > 0 mostly, so s=4).
        unique = set(int(x) for x in np.unique(res.s_b))
        assert unique <= {3, 4}, (
            f"L1+IPD s_b unique levels {sorted(unique)} should be ⊆ {{3, 4}}"
        )

    def test_l1_ipd_vs_ps_pwm_disagree_at_runtime(self):
        """At an off-grid m_ref, IPD and PS-PWM produce different
        instantaneous s_b traces — this is the whole point of having
        two strategies."""
        m_ref = 0.37
        n_sm = 10
        # Sample over one carrier period.
        ts = np.linspace(0.0, 1e-3, 201)
        s_ipd = [p.ipd_switching_function(m_ref, float(t), n_sm, 1000.0) for t in ts]
        s_ps  = [p.ps_pwm_switching_function(m_ref, float(t), n_sm, 1000.0) for t in ts]
        # They CAN'T match identically (different mathematical machinery).
        assert s_ipd != s_ps

    def test_l2_with_ipd(self):
        """L2 with dead-time + IPD scheme runs and produces sensible
        state-machine bookkeeping."""
        params = p.MmcArmEquivalentParams(
            n_sm=6, c_sm=1e-3, v_c0=300.0,
            f_carrier=1000.0, t_dead=10e-6,
            modulation_scheme="ipd",
        )
        res = p.simulate_mmc_arm_equivalent(
            duration=3e-3, dt=2e-6,
            m_ref=lambda t: 0.5 + 0.3 * math.sin(2 * math.pi * 100 * t),
            i_b=2.0, params=params,
        )
        # s_w + s_u ≤ N at every step.
        assert (res.s_w + res.s_u <= params.n_sm).all()

    def test_l3_with_ipd(self):
        """L3 detailed + IPD: sum-of-caps stays in the right ballpark."""
        params = p.MmcArmDetailedParams(
            n_sm=6, c_sm=1e-3, v_c0=600.0,
            f_carrier=1000.0, balancing="sort_and_select",
            modulation_scheme="ipd",
        )
        res = p.simulate_mmc_arm_detailed(
            duration=2e-3, dt=5e-6,
            m_ref=0.5, i_b=0.0, params=params,
        )
        assert np.allclose(res.v_C_sum, 600.0)


# ---------------------------------------------------------------------------
# Parameter contract — IPD as a valid choice.
# ---------------------------------------------------------------------------

class TestModulationSchemeContract:
    def test_multilevel_accepts_ipd(self):
        p.MmcArmMultilevelParams(
            n_sm=4, c_sm=1e-3, modulation_scheme="ipd",
        )

    def test_multilevel_rejects_unknown_scheme(self):
        with pytest.raises(ValueError, match="modulation_scheme"):
            p.MmcArmMultilevelParams(  # type: ignore[arg-type]
                n_sm=4, c_sm=1e-3, modulation_scheme="ipd_x",
            )

    def test_equivalent_accepts_ipd(self):
        p.MmcArmEquivalentParams(
            n_sm=4, c_sm=1e-3, modulation_scheme="ipd",
        )

    def test_detailed_accepts_ipd(self):
        p.MmcArmDetailedParams(
            n_sm=4, c_sm=1e-3, modulation_scheme="ipd",
        )
