"""Regression tests for M3C Fast SVM (Phase 22.1).

Validates the modulation theory directly from Gili (2024) Sec 3.2
(Eqs 25, 26a-d, 28, 29, 30) — independent of any pulsim simulation.

To run::

    pytest python/tests/test_m3c_fast_svm.py -v
"""

from __future__ import annotations

import sys
from math import pi
from pathlib import Path

import numpy as np
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_M3C_DIR = _PROJECT_ROOT / "projects" / "inverters" / "m3c_3phase"
sys.path.insert(0, str(_M3C_DIR))

from m3c_3phase_model import (  # noqa: E402
    LG_TRANSFORM_MATRIX,
    M3cParams,
    abc_to_lg,
    fast_svm_4_vectors,
    fast_svm_duty_cycles,
    fast_svm_pick_triangle,
    fast_svm_step,
    lg_to_abc,
    make_fast_svm_fn,
)


# ============================================================================
# Tier 1 — LG coordinate transform (Eq 25)
# ============================================================================


class TestLgTransform:
    """The lgγ transform should map integer abc → integer lgγ (Sec 3.2)."""

    @pytest.mark.parametrize("v_abc", [
        (1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 1), (2, -1, 1),
        (-3, 2, 0), (5, -3, 2), (10, -5, -5),
    ])
    def test_integer_input_integer_output(self, v_abc) -> None:
        v_lg = abc_to_lg(v_abc)
        for x in v_lg:
            assert abs(x - round(x)) < 1e-12, (
                f"v_lg[{list(v_lg).index(x)}] = {x} not integer"
            )

    @pytest.mark.parametrize("v_abc", [
        (1.5, -2.3, 0.7), (10, 20, -30), (0, 0, 0),
        (np.pi, -np.e, 1.0),
    ])
    def test_inverse_round_trip(self, v_abc) -> None:
        v_lg = abc_to_lg(v_abc)
        v_back = lg_to_abc(v_lg)
        err = np.max(np.abs(np.array(v_abc, dtype=float) - v_back))
        assert err < 1e-10, f"round-trip err {err:.2e} > 1e-10"

    def test_transform_matrix_determinant(self) -> None:
        """LG matrix should have determinant 3 (non-singular, scales
        the common-mode axis by 3×)."""
        assert abs(np.linalg.det(LG_TRANSFORM_MATRIX) - 3.0) < 1e-12

    def test_known_example_from_thesis(self) -> None:
        """V_a=1, V_b=-1, V_c=0 → V_l=2, V_g=-1, V_γ=0 (per Fig 30)."""
        v_lg = abc_to_lg((1.0, -1.0, 0.0))
        assert tuple(v_lg.tolist()) == (2.0, -1.0, 0.0)


# ============================================================================
# Tier 2 — Fast SVM 4-vector lookup (Eqs 26a-d)
# ============================================================================


class TestFastSvmFourVectors:
    """The 4 adjacent integer vectors define the cell containing V_ref."""

    @pytest.mark.parametrize("v_l,v_g,V_ul,V_lu,V_ll,V_uu", [
        # Thesis Eq 27 worked example
        (-1.8, 1.2, (-1, 1), (-2, 2), (-2, 1), (-1, 2)),
        # First-quadrant fraction
        (0.3, 0.4, (1, 0), (0, 1), (0, 0), (1, 1)),
        # Origin (exact integer point)
        (0.0, 0.0, (0, 0), (0, 0), (0, 0), (0, 0)),
        # Negative quadrant
        (-2.7, -1.4, (-2, -2), (-3, -1), (-3, -2), (-2, -1)),
    ])
    def test_four_vectors_correct(self, v_l, v_g, V_ul, V_lu, V_ll, V_uu) -> None:
        actual = fast_svm_4_vectors(v_l, v_g)
        expected = (V_ul, V_lu, V_ll, V_uu)
        assert actual == expected, (
            f"V_ref=({v_l},{v_g}): expected {expected}, got {actual}"
        )

    @pytest.mark.parametrize("v_l,v_g", [
        (-1.8, 1.2), (0.3, 0.4), (2.5, -1.5), (5.0, 5.0),
    ])
    def test_v_ref_inside_quadrilateral(self, v_l, v_g) -> None:
        """V_ref must lie inside (or on boundary of) the quadrilateral
        formed by the 4 corners."""
        V_ul, V_lu, V_ll, V_uu = fast_svm_4_vectors(v_l, v_g)
        all_l = [V_ul[0], V_lu[0], V_ll[0], V_uu[0]]
        all_g = [V_ul[1], V_lu[1], V_ll[1], V_uu[1]]
        assert min(all_l) - 1e-12 <= v_l <= max(all_l) + 1e-12
        assert min(all_g) - 1e-12 <= v_g <= max(all_g) + 1e-12


# ============================================================================
# Tier 3 — Triangle selection (Eq 28, geometrically corrected)
# ============================================================================


class TestFastSvmTriangleSelection:
    """The triangle selection must pick the *closer* of (V_ll, V_uu)
    so all duty cycles remain non-negative."""

    @pytest.mark.parametrize("v_l,v_g,expected", [
        (-1.8, 1.2, "ll"),     # thesis example, V_ll closer
        (-1.2, 1.8, "uu"),     # mirror, V_uu closer
        (0.3, 0.4, "ll"),
        (0.7, 0.8, "uu"),
        (0.5, 0.5, "ll"),      # exact midpoint goes to "ll" by convention
    ])
    def test_picks_closer_corner(self, v_l, v_g, expected) -> None:
        tri = fast_svm_pick_triangle(v_l, v_g)
        assert tri == expected, (
            f"V_ref=({v_l},{v_g}): expected {expected}, got {tri}"
        )


# ============================================================================
# Tier 4 — Duty-cycle equations (Eqs 29, 30)
# ============================================================================


class TestFastSvmDutyCycles:
    """All three duty cycles must be ≥ 0 and sum to 1."""

    @pytest.mark.parametrize("v_l", [-3, -1.8, -0.5, 0, 0.3, 1.5, 2.7])
    @pytest.mark.parametrize("v_g", [-3, -1.8, -0.5, 0, 0.3, 1.5, 2.7])
    def test_duty_sum_to_one(self, v_l, v_g) -> None:
        d_ul, d_lu, d_third, _ = fast_svm_duty_cycles(v_l, v_g)
        assert abs(d_ul + d_lu + d_third - 1.0) < 1e-9

    @pytest.mark.parametrize("v_l", [-3, -1.8, -0.5, 0.3, 1.5, 2.7])
    @pytest.mark.parametrize("v_g", [-3, -1.8, -0.5, 0.3, 1.5, 2.7])
    def test_duty_cycles_nonnegative(self, v_l, v_g) -> None:
        d_ul, d_lu, d_third, _ = fast_svm_duty_cycles(v_l, v_g)
        assert d_ul >= -1e-12, f"d_ul = {d_ul} < 0"
        assert d_lu >= -1e-12, f"d_lu = {d_lu} < 0"
        assert d_third >= -1e-12, f"d_third = {d_third} < 0"

    def test_origin_gives_unit_duty_on_third(self) -> None:
        """At V_ref = (0, 0), all 4 corners coincide at origin so
        d_third = 1 (only one vector active)."""
        d_ul, d_lu, d_third, _ = fast_svm_duty_cycles(0.0, 0.0)
        assert abs(d_ul) < 1e-12
        assert abs(d_lu) < 1e-12
        assert abs(d_third - 1.0) < 1e-12

    def test_thesis_eq27_example_full_duty(self) -> None:
        """The Eq 27 thesis example V_ref=(-1.8, 1.2) should produce
        clean duty cycles after our triangle correction."""
        d_ul, d_lu, d_third, tri = fast_svm_duty_cycles(-1.8, 1.2)
        assert tri == "ll"
        assert abs(d_ul - 0.2) < 1e-9
        assert abs(d_lu - 0.2) < 1e-9
        assert abs(d_third - 0.6) < 1e-9


# ============================================================================
# Tier 5 — Full SVM step (time-domain sweep)
# ============================================================================


class TestFastSvmStep:
    """End-to-end Fast SVM should produce valid duty cycles at all
    times throughout the AC reference cycle, for any modulation index
    within the theoretical limit."""

    @pytest.mark.parametrize("m_v", [0.1, 0.3, 0.5, 0.7, 0.9])
    def test_step_valid_throughout_cycle(self, m_v: float) -> None:
        params = M3cParams(m_v=m_v)
        fn = make_fast_svm_fn(params, side="output")
        T_out = 1.0 / params.f_out
        for t in np.linspace(0, T_out, 100, endpoint=False):
            _, _, _, d_a, d_b, d_c = fn(float(t))
            assert abs(d_a + d_b + d_c - 1.0) < 1e-9, (
                f"t={t*1e3:.2f}ms m={m_v}: Σd = {d_a+d_b+d_c}"
            )
            assert min(d_a, d_b, d_c) >= -1e-12, (
                f"t={t*1e3:.2f}ms m={m_v}: negative duty {(d_a,d_b,d_c)}"
            )

    def test_step_input_side_works(self) -> None:
        """Input-side SVM should run the same algorithm with f_in."""
        params = M3cParams(m_c=0.8)
        fn = make_fast_svm_fn(params, side="input")
        for t in [0.0, 5e-3, 10e-3]:
            _, _, _, d_a, d_b, d_c = fn(t)
            assert abs(d_a + d_b + d_c - 1.0) < 1e-9

    def test_invalid_side_raises(self) -> None:
        params = M3cParams()
        with pytest.raises(ValueError, match="side"):
            fast_svm_step(0.0, params, side="invalid_side")  # type: ignore[arg-type]


# ============================================================================
# Tier 6 — Params + topology defaults
# ============================================================================


class TestM3cParams:
    """Sanity checks on M3cParams defaults — should match Tab. 15
    of the thesis."""

    def test_default_topology(self) -> None:
        p = M3cParams()
        assert p.n_modules == 9
        assert p.n_sm_per_module == 6
        assert p.n_total_sm == 54

    def test_thirteen_line_levels(self) -> None:
        """With N=6 SMs/module, the M3C produces 13 line-line levels
        (Sec 4.1 of thesis)."""
        p = M3cParams(n_sm_per_module=6)
        assert p.n_levels_LL == 13

    def test_capacitor_total_per_module(self) -> None:
        p = M3cParams(n_sm_per_module=6, v_cap_nominal=4000)
        assert p.v_cap_total_per_module == 24000.0  # 24 kV per module

    def test_T_s_at_2kHz(self) -> None:
        p = M3cParams(f_switching=2_000.0)
        assert abs(p.T_s - 500e-6) < 1e-12
