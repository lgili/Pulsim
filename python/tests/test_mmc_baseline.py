"""Regression tests for MMC analytical baselines (Phase 20.19).

These tests run on every commit (pytest) and catch regressions in
the MMC L0/L1/L2 stack via independent closed-form / cross-layer
checks. They are independent of the Sousa (2022) thesis reference,
which has known parameter inconsistencies (see notebook 01
discussion in README.md).

Architecture: the tests delegate to ``mmc_baseline_tests.py`` (sits
under ``projects/inverters/mmc_3phase/``) and add a sys.path hook
to import from there. Each pytest function is a *thin wrapper*
around one of the ``BaselineResult``-returning helpers — keeping
the heavy lifting in the project module and the test file slim
and easy to maintain.

To run::

    pytest python/tests/test_mmc_baseline.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make the project helpers importable.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_MMC_DIR = _PROJECT_ROOT / "projects" / "inverters" / "mmc_3phase"
sys.path.insert(0, str(_MMC_DIR))

from mmc_3phase_model import GeanThesisParams  # noqa: E402
from mmc_baseline_tests import (  # noqa: E402
    BaselineResult,
    test_open_circuit as _test_open_circuit,
    test_dc_zero_input as _test_dc_zero_input,
    test_l0_ac_amplitude as _test_l0_ac_amplitude,
    test_l0_v_c_ripple as _test_l0_v_c_ripple,
    test_energy_conservation as _test_energy_conservation,
    test_cap_balance as _test_cap_balance,
    test_layer_avg_v_c_consistency as _test_layer_avg_v_c,
    test_layer_fundamental_i_a as _test_layer_fund_i_a,
    test_thd_ordering as _test_thd_ordering,
)


@pytest.fixture(scope="module")
def thesis_params() -> GeanThesisParams:
    """Default Sousa (2022) operating point — V_dc=640V, M=0.85,
    N=5, C_SM=470µF, balanced 3-φ R+L load.

    Used as the *operating point* for all baseline tests. The
    point itself isn't being validated against Sousa — we're
    validating the simulator's *behaviour* at this point via
    closed-form checks.
    """
    return GeanThesisParams()


# ============================================================================
# Tier 1 — Analytical limit cases
# ============================================================================


class TestTier1Analytical:
    """Closed-form / limit-case checks for the L0 average MMC."""

    def test_open_circuit(self, thesis_params: GeanThesisParams) -> None:
        """R_load → ∞ ⇒ no AC current flows."""
        r: BaselineResult = _test_open_circuit(thesis_params)
        assert r.passed, r.msg

    def test_dc_zero_input(self, thesis_params: GeanThesisParams) -> None:
        """m_p = m_n = 0.5 constant ⇒ zero current, v_C unchanged."""
        r: BaselineResult = _test_dc_zero_input(thesis_params)
        assert r.passed, r.msg

    def test_l0_ac_amplitude(self, thesis_params: GeanThesisParams) -> None:
        """|i_a|_peak matches ``M·V_dc / (2·|Z_AC|)`` within 5 %."""
        r: BaselineResult = _test_l0_ac_amplitude(thesis_params)
        assert r.passed, r.msg

    def test_l0_v_c_ripple_order_of_magnitude(
        self, thesis_params: GeanThesisParams,
    ) -> None:
        """v_C pkpk is within 2× of the first-order analytical
        ``ΔV = |m·i|_{1ω} / (ω·C_arm)`` × 2."""
        r: BaselineResult = _test_l0_v_c_ripple(thesis_params)
        assert r.passed, r.msg

    def test_energy_conservation(self, thesis_params: GeanThesisParams) -> None:
        """P_load + P_loss is positive, finite, and below the
        plausible upper bound."""
        r: BaselineResult = _test_energy_conservation(thesis_params)
        assert r.passed, r.msg

    def test_cap_balance(self, thesis_params: GeanThesisParams) -> None:
        """v_C drift across the steady-state window < 10 %."""
        r: BaselineResult = _test_cap_balance(thesis_params)
        assert r.passed, r.msg


# ============================================================================
# Tier 2 — Layer-to-layer consistency
# ============================================================================


class TestTier2LayerConsistency:
    """L0 ↔ L1 ↔ L2 averages must agree (different ripple is fine).

    Catches: discretization bias in L1 (would show up as DC offset
    in v_C average); modulation-scheme bug; topology mismatch
    between layers.
    """

    def test_avg_v_c_l0_l1(self, thesis_params: GeanThesisParams) -> None:
        """⟨v_C⟩_L0 and ⟨v_C⟩_L1 agree within 2 %."""
        r: BaselineResult = _test_layer_avg_v_c(thesis_params)
        assert r.passed, r.msg

    def test_fundamental_i_a_l0_l1(self, thesis_params: GeanThesisParams) -> None:
        """1ω-component of i_a in L0 and L1 agree within 10 %."""
        r: BaselineResult = _test_layer_fund_i_a(thesis_params)
        assert r.passed, r.msg

    def test_thd_ordering_l0_l1_l2(self, thesis_params: GeanThesisParams) -> None:
        """THD(L0) < THD(L1) ≤ THD(L2) — physics ordering."""
        r: BaselineResult = _test_thd_ordering(thesis_params)
        assert r.passed, r.msg
