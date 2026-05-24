"""Regression tests for CMC SVM analytical helpers (Phase 21).

Validates the modulation theory directly from Gili (2024) Sec 2.2.1
(Eqs 7a-7d, Tab. 1-4) — independent of any pulsim simulation. Catches
regressions in the SVM module before they propagate to the L0/L1
plant builders.

To run::

    pytest python/tests/test_cmc_svm.py -v
"""

from __future__ import annotations

import sys
from math import cos, pi, sqrt
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_CMC_DIR = _PROJECT_ROOT / "projects" / "inverters" / "cmc_3phase"
sys.path.insert(0, str(_CMC_DIR))

from cmc_3phase_model import (  # noqa: E402
    CMC_ACTIVE_VECTORS,
    CMC_ROTATIONAL_VECTORS,
    CMC_ZERO_VECTORS,
    CmcParams,
    build_l0_plant,
    make_cmc_gate_signals,
    predict_i_out_peak,
    predict_load_impedance,
    predict_load_power_factor,
    rms,
    run_l0_open_loop,
    svm_active_vectors_for_sectors,
    svm_duty_cycles,
    svm_max_modulation,
    svm_sector_pair,
    svm_step,
    switch_mask_for_config,
    switch_mask_for_state,
    thd,
)
import numpy as np  # noqa: E402


# ============================================================================
# Tier 1 — Analytical SVM theory (closed-form vs thesis equations)
# ============================================================================


class TestModulationLimit:
    """Eq 11 of the thesis — theoretical maximum modulation index."""

    def test_unity_pf(self) -> None:
        """m_max = √3/2 at unity power factor."""
        assert svm_max_modulation(0.0) == pytest.approx(sqrt(3) / 2.0)

    def test_30_degree_pf(self) -> None:
        """m_max scales with |cos(phi_i)|."""
        assert svm_max_modulation(pi / 6) == pytest.approx(
            sqrt(3) / 2.0 * cos(pi / 6),
        )

    def test_90_degree_pf(self) -> None:
        """At phi_i = 90° (purely reactive), m_max = 0."""
        assert svm_max_modulation(pi / 2) == pytest.approx(0.0, abs=1e-10)


class TestSectorIdentification:
    """Sector 1 should contain angle 0° (centred on α axis)."""

    def test_sector_1_at_zero(self) -> None:
        K_v, K_i, a_til, b_til = svm_sector_pair(0.0, 0.0)
        assert K_v == 1
        assert K_i == 1
        assert a_til == pytest.approx(0.0, abs=1e-10)
        assert b_til == pytest.approx(0.0, abs=1e-10)

    @pytest.mark.parametrize(
        "alpha_deg, expected_K",
        [
            (0.0, 1),
            (45.0, 2),
            (75.0, 2),     # well inside sector 2 (covers [30°, 90°])
            (120.0, 3),
            (-30.0 + 1.0, 1),    # just inside sector 1
            (-31.0, 6),          # just outside (wraps to sector 6)
        ],
    )
    def test_sector_assignment(self, alpha_deg: float, expected_K: int) -> None:
        K_v, _, _, _ = svm_sector_pair(alpha_deg * pi / 180.0, 0.0)
        assert K_v == expected_K, f"α={alpha_deg}° → got K_v={K_v}, expected {expected_K}"

    def test_sector_boundary_unique(self) -> None:
        """At a sector boundary (α=90° lies between sectors 2 and 3),
        we accept either of the two adjacent sectors — but never an
        unrelated one."""
        K_v, _, _, _ = svm_sector_pair(pi / 2, 0.0)  # 90°
        assert K_v in {2, 3}, f"α=90° boundary → K_v={K_v} (expected 2 or 3)"

    @pytest.mark.parametrize("alpha_deg", [0.0, 15.0, 45.0, 75.0, 135.0, 195.0])
    def test_sectorial_angle_range(self, alpha_deg: float) -> None:
        """Sectorial angle α̃ always in [-π/6, π/6]."""
        _, _, a_til, _ = svm_sector_pair(alpha_deg * pi / 180.0, 0.0)
        assert -pi / 6 - 1e-9 <= a_til <= pi / 6 + 1e-9, \
            f"α̃ = {a_til*180/pi:.2f}° out of [-30°, 30°]"


class TestDutyCycleConstraints:
    """Eq 8 of the thesis: |δ^I| + |δ^II| + |δ^III| + |δ^IV| ≤ 1."""

    @pytest.mark.parametrize("m", [0.1, 0.3, 0.5, 0.7, 0.86])
    @pytest.mark.parametrize("alpha_deg", [0, 15, 30, 45, 90, 135, 200, 300])
    @pytest.mark.parametrize("beta_deg", [0, 30, 60, 120, 240])
    def test_duty_sum_within_bound(
        self, m: float, alpha_deg: float, beta_deg: float,
    ) -> None:
        d = svm_duty_cycles(m, alpha_deg * pi / 180.0, beta_deg * pi / 180.0)
        total = sum(abs(x) for x in d)
        assert total <= 1.0 + 1e-9, (
            f"Σ|δ| = {total:.4f} > 1 at m={m}, α={alpha_deg}°, β={beta_deg}°"
        )

    def test_zero_modulation_yields_zero_duties(self) -> None:
        d = svm_duty_cycles(0.0, pi / 4, pi / 3)
        assert all(abs(x) < 1e-12 for x in d)

    def test_phi_i_at_90_degrees_raises(self) -> None:
        """cos(phi_i) = 0 → divisão por zero detectada."""
        with pytest.raises(ValueError, match="cos"):
            svm_duty_cycles(0.5, 0.0, 0.0, phi_i=pi / 2)


# ============================================================================
# Tier 2 — Switching state correctness (Tab. 1-4 of the thesis)
# ============================================================================


class TestSwitchStateTables:
    """Validate the 27 switching states match Sec 2.2 of the thesis."""

    def test_27_states_total(self) -> None:
        n = len(CMC_ZERO_VECTORS) + len(CMC_ACTIVE_VECTORS) + len(CMC_ROTATIONAL_VECTORS)
        assert n == 27, f"got {n} states, expected 27"

    def test_active_count_18(self) -> None:
        assert len(CMC_ACTIVE_VECTORS) == 18

    def test_zero_count_3(self) -> None:
        assert len(CMC_ZERO_VECTORS) == 3

    def test_rotational_count_6(self) -> None:
        assert len(CMC_ROTATIONAL_VECTORS) == 6

    def test_active_pairs_complementary(self) -> None:
        """+k and -k should differ (Tab. 2 — they're opposite directions)."""
        for k in range(1, 10):
            assert CMC_ACTIVE_VECTORS[+k] != CMC_ACTIVE_VECTORS[-k], (
                f"+{k} and -{k} have identical state"
            )

    def test_zero_states_all_same_input(self) -> None:
        """A zero state ties all 3 outputs to ONE input phase."""
        for label, state in CMC_ZERO_VECTORS.items():
            assert state[0] == state[1] == state[2], (
                f"{label} = {state} is not a zero state"
            )

    def test_rotational_states_are_permutations(self) -> None:
        """A rotational state is a permutation of (A, B, C)."""
        for label, state in CMC_ROTATIONAL_VECTORS.items():
            assert sorted(state) == [0, 1, 2], (
                f"{label} = {state} is not a permutation"
            )


class TestSwitchMaskMapping:
    """The 9-bit mask must always have exactly 3 switches ON
    (the topology constraint — one input per output)."""

    @pytest.mark.parametrize("config_id", list(CMC_ACTIVE_VECTORS.keys()))
    def test_active_mask_has_three_on(self, config_id: int) -> None:
        mask = switch_mask_for_config(config_id)
        assert len(mask) == 9
        assert sum(mask) == 3, f"config {config_id} → mask {mask} has {sum(mask)} ON"

    @pytest.mark.parametrize("config_id", list(CMC_ZERO_VECTORS.keys()))
    def test_zero_mask_has_three_on(self, config_id: str) -> None:
        mask = switch_mask_for_config(config_id)
        assert sum(mask) == 3

    @pytest.mark.parametrize("config_id", list(CMC_ROTATIONAL_VECTORS.keys()))
    def test_rotational_mask_has_three_on(self, config_id: str) -> None:
        mask = switch_mask_for_config(config_id)
        assert sum(mask) == 3

    def test_unique_masks_for_active(self) -> None:
        """Each of the 18 active configs maps to a unique mask."""
        masks = {switch_mask_for_config(k) for k in CMC_ACTIVE_VECTORS}
        assert len(masks) == 18

    def test_zero_state_mask_pattern(self) -> None:
        """0_1 (all outputs → A) means S_1, S_2, S_3 ON."""
        assert switch_mask_for_state((0, 0, 0)) == (1, 1, 1, 0, 0, 0, 0, 0, 0)

    def test_specific_active_pattern_plus_1(self) -> None:
        """+1 in the thesis: S_1, S_5, S_6 ON. With our (col-major)
        numbering: out_a→A (S_1), out_b→B (S_5), out_c→B (S_6) ⇒
        mask = (1, 0, 0, 0, 1, 1, 0, 0, 0)."""
        assert switch_mask_for_config(+1) == (1, 0, 0, 0, 1, 1, 0, 0, 0)

    def test_rotational_R1_pattern(self) -> None:
        """R_1 (identity permutation a→A, b→B, c→C): S_1, S_5, S_9 ON."""
        assert switch_mask_for_config("R_1") == (1, 0, 0, 0, 1, 0, 0, 0, 1)


# ============================================================================
# Tier 3 — Tab. 4 vector selection per sector pair
# ============================================================================


class TestTab4VectorSelection:
    """The 4 active-vector positions for each (K_v, K_i) — Tab. 4 row 1."""

    def test_kv1_ki1(self) -> None:
        assert svm_active_vectors_for_sectors(1, 1) == (+9, +7, +3, +1)

    def test_kv1_ki4_same_as_ki1(self) -> None:
        """K_i = 4 is the +180° image of K_i = 1; Tab. 4 lists same pattern."""
        assert (
            svm_active_vectors_for_sectors(1, 4)
            == svm_active_vectors_for_sectors(1, 1)
        )

    def test_kv2_ki1(self) -> None:
        assert svm_active_vectors_for_sectors(2, 1) == (+6, +4, +9, +7)

    def test_kv3_ki1(self) -> None:
        assert svm_active_vectors_for_sectors(3, 1) == (+3, +1, +6, +4)


# ============================================================================
# Tier 4 — Full SVM step output
# ============================================================================


class TestSvmStep:
    """Validate the symmetric switching sequence (Fig 5) end-to-end."""

    def test_returns_9_tuple(self) -> None:
        mask = svm_step(0.0, 0.5, 2 * pi * 30, 2 * pi * 60)
        assert isinstance(mask, tuple) and len(mask) == 9

    def test_always_three_switches_on(self) -> None:
        """Critical invariant: regardless of t and m, exactly 3 ON."""
        params = CmcParams(m_depth=0.7)
        gate = make_cmc_gate_signals(params)
        # Sample 1000 random-ish times over 5 ms
        import random
        random.seed(42)
        for _ in range(1000):
            t = random.uniform(0, 5e-3)
            mask = gate(t)
            assert sum(mask) == 3, f"At t={t*1e6:.1f}µs: mask={mask}, sum={sum(mask)}"

    def test_overmodulation_clamps_safely(self) -> None:
        """At m > m_max, sum(|δ|) > 1; sequence should clamp without crash."""
        params = CmcParams(m_depth=0.95)  # > 0.866
        gate = make_cmc_gate_signals(params)
        # Should still produce valid 3-switch masks
        for t in [0.0, 1e-5, 5e-5, 1e-4]:
            mask = gate(t)
            assert sum(mask) == 3


# ============================================================================
# Tier 5 — L0 averaged plant validation (output-side)
# ============================================================================


class TestL0Plant:
    """L0 (averaged Venturini-style) plant: synthesised output sinusoids
    driving a Y-connected RL load. Validates pulsim end-to-end on the
    output side, against closed-form analytical predictions."""

    @pytest.fixture(scope="class")
    def params(self) -> CmcParams:
        return CmcParams(
            V_in_peak=311.13, f_in=60.0, f_out=30.0, m_depth=0.5,
            R_load=5.0, L_load=10e-3,
        )

    @pytest.fixture(scope="class")
    def result(self, params: CmcParams):
        plant = build_l0_plant(params)
        return run_l0_open_loop(plant, t_end=200e-3, dt=10e-6)

    def test_i_out_peak_matches_analytical(
        self, params: CmcParams, result,
    ) -> None:
        """|I_o| = V_o_peak / |Z_load| should match within 1 %."""
        mask = result.t >= 150e-3
        i_a_pk = float(np.max(np.abs(result.i_a_out[mask])))
        i_a_pk_pred = predict_i_out_peak(params)
        rel_err = abs(i_a_pk - i_a_pk_pred) / i_a_pk_pred
        assert rel_err < 0.01, (
            f"i_a peak = {i_a_pk:.4f} A vs {i_a_pk_pred:.4f} A predicted, "
            f"rel-err = {rel_err*100:.2f}%"
        )

    def test_i_out_rms_matches_analytical(
        self, params: CmcParams, result,
    ) -> None:
        """RMS = peak / √2 for pure sinusoid."""
        mask = result.t >= 150e-3
        i_a_rms = rms(result.i_a_out[mask])
        i_a_rms_pred = predict_i_out_peak(params) / np.sqrt(2.0)
        rel_err = abs(i_a_rms - i_a_rms_pred) / i_a_rms_pred
        assert rel_err < 0.01, (
            f"i_a RMS = {i_a_rms:.4f} A vs {i_a_rms_pred:.4f} A, "
            f"rel-err = {rel_err*100:.2f}%"
        )

    def test_balanced_three_phase_output(self, result) -> None:
        """All three load currents have the same peak (balanced 3-φ)."""
        mask = result.t >= 150e-3
        peaks = [
            float(np.max(np.abs(x[mask])))
            for x in (result.i_a_out, result.i_b_out, result.i_c_out)
        ]
        max_dev = max(peaks) - min(peaks)
        max_dev_rel = max_dev / max(peaks)
        assert max_dev_rel < 1e-3, (
            f"3-φ peaks unbalanced: {peaks}, max rel deviation {max_dev_rel*100:.4f}%"
        )

    def test_load_phase_shift_120_degrees(
        self, params: CmcParams, result,
    ) -> None:
        """The three load currents are 120° apart in phase. Measured
        via DFT of the fundamental — robust against window-edge
        artefacts of the zero-crossing method."""
        mask = result.t >= 100e-3
        # Extract an integer number of fundamental periods for clean FFT.
        fs = 1.0 / 10e-6
        T_out = 1.0 / params.f_out
        n_periods = 3
        n_win = int(round(n_periods * T_out * fs))
        ia = result.i_a_out[mask][:n_win]
        ib = result.i_b_out[mask][:n_win]
        ic = result.i_c_out[mask][:n_win]
        # Coherent DFT bin at f_out (no windowing needed for integer cycles).
        k = int(round(params.f_out * n_win / fs))
        spec_a = np.fft.rfft(ia)[k]
        spec_b = np.fft.rfft(ib)[k]
        spec_c = np.fft.rfft(ic)[k]
        # Phase shift in radians, then degrees
        phase_b_minus_a = np.angle(spec_b) - np.angle(spec_a)
        phase_c_minus_a = np.angle(spec_c) - np.angle(spec_a)
        # Wrap to (-π, +π]
        phase_b_minus_a = ((phase_b_minus_a + pi) % (2 * pi)) - pi
        phase_c_minus_a = ((phase_c_minus_a + pi) % (2 * pi)) - pi
        # Expected: b lags by 2π/3 (-120°), c leads by 2π/3 (+120°)
        assert abs(phase_b_minus_a - (-2 * pi / 3)) < 0.05, (
            f"Phase b−a = {np.degrees(phase_b_minus_a):.2f}°, "
            f"expected −120° ± 3°"
        )
        assert abs(phase_c_minus_a - (+2 * pi / 3)) < 0.05, (
            f"Phase c−a = {np.degrees(phase_c_minus_a):.2f}°, "
            f"expected +120° ± 3°"
        )

    def test_pure_sinusoid_low_thd(self, params: CmcParams, result) -> None:
        """L0 has no switching ripple ⇒ THD should be near zero
        (limited by FFT windowing artefacts to ~3 %)."""
        mask = result.t >= 100e-3
        fs = 1.0 / 10e-6
        n_win = int(round(3 * (1.0 / params.f_out) * fs))
        thd_pct = thd(result.i_a_out[mask][:n_win], fs, params.f_out)
        # Allow some windowing leakage but flag any actual harmonic content
        assert thd_pct < 5.0, (
            f"THD = {thd_pct:.2f}%; expected < 5% for ideal L0"
        )

    def test_load_power_factor_close_to_R_L_theory(
        self, params: CmcParams,
    ) -> None:
        """The theoretical load PF should match cos(arctan(ωL/R)) exactly."""
        pf_theory = predict_load_power_factor(params)
        # Construct analytically: arctan(ω·L / R) for R=5, ω=2π·30, L=10mH
        arg = params.omega_out * params.L_load / params.R_load
        pf_expected = 1.0 / np.sqrt(1.0 + arg ** 2)
        assert abs(pf_theory - pf_expected) < 1e-12
