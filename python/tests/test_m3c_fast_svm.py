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
    ALL_VALID_CONFIGURATIONS,
    LG_TRANSFORM_MATRIX,
    M3cParams,
    ModuleConfiguration,
    abc_to_lg,
    configurations_by_distribution,
    configurations_containing_module,
    connection_cost,
    enumerate_valid_configurations,
    fast_svm_4_vectors,
    fast_svm_duty_cycles,
    fast_svm_pick_triangle,
    fast_svm_step,
    lg_to_abc,
    make_fast_svm_fn,
    select_best_connection,
    solve_module_currents,
    solve_module_voltages,
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


# ============================================================================
# Tier 7 — Module connection configurations (Sec 4.3 of thesis)
# ============================================================================


class TestModuleConfigurations:
    """The 81 valid M3C configurations per Sec 4.3 of the thesis."""

    def test_total_count_is_81(self) -> None:
        """Sec 4.3 states 81 valid configurations exist."""
        assert len(ALL_VALID_CONFIGURATIONS) == 81

    def test_enumerate_returns_same_set(self) -> None:
        """Calling the enumerator multiple times returns the same
        set of configurations."""
        configs_1 = enumerate_valid_configurations()
        configs_2 = enumerate_valid_configurations()
        # Same length and same set of grids
        assert len(configs_1) == len(configs_2)
        grids_1 = {cfg.grid for cfg in configs_1}
        grids_2 = {cfg.grid for cfg in configs_2}
        assert grids_1 == grids_2

    def test_all_have_exactly_5_active_modules(self) -> None:
        """The M3C "5 modules conducting" constraint."""
        for cfg in ALL_VALID_CONFIGURATIONS:
            assert cfg.n_active() == 5, (
                f"{cfg.to_string()} has {cfg.n_active()} active, "
                f"expected 5"
            )

    def test_all_pass_is_valid(self) -> None:
        for cfg in ALL_VALID_CONFIGURATIONS:
            assert cfg.is_valid()

    def test_all_have_no_zero_rows_or_cols(self) -> None:
        """Every input phase and every output phase must have at
        least one active connection (rule 1, Sec 4.3)."""
        for cfg in ALL_VALID_CONFIGURATIONS:
            for i in range(3):
                assert cfg.row_sum(i) >= 1, (
                    f"row {i} has no connections in {cfg.to_string()}"
                )
            for j in range(3):
                assert cfg.col_sum(j) >= 1, (
                    f"col {j} has no connections in {cfg.to_string()}"
                )

    def test_all_have_valid_distributions(self) -> None:
        """Every config's row-sum AND col-sum distribution must be
        (1,1,3) or (1,2,2) — never (1,2,3) or (0,1,4) etc."""
        valid_dists = ({1, 1, 3}, {1, 2, 2})
        for cfg in ALL_VALID_CONFIGURATIONS:
            row_dist = sorted(cfg.row_sum(i) for i in range(3))
            col_dist = sorted(cfg.col_sum(j) for j in range(3))
            assert set(row_dist) in [{1, 3}, {1, 2}], row_dist
            assert sorted(row_dist) in [[1, 1, 3], [1, 2, 2]]
            assert sorted(col_dist) in [[1, 1, 3], [1, 2, 2]]

    def test_distributions_balance(self) -> None:
        """Sum of row-sums = sum of col-sums = 5 for every config."""
        for cfg in ALL_VALID_CONFIGURATIONS:
            assert sum(cfg.row_sum(i) for i in range(3)) == 5
            assert sum(cfg.col_sum(j) for j in range(3)) == 5

    def test_configurations_by_distribution_partition(self) -> None:
        """The 4 distribution patterns must partition the 81 configs."""
        groups = configurations_by_distribution()
        total = sum(len(cfgs) for cfgs in groups.values())
        assert total == 81
        # 4 distinct (row, col) distribution combinations
        assert len(groups) == 4

    def test_configurations_have_unique_grids(self) -> None:
        """No duplicate grids in the enumeration."""
        grids = {cfg.grid for cfg in ALL_VALID_CONFIGURATIONS}
        assert len(grids) == 81

    def test_active_modules_count_matches_n_active(self) -> None:
        """``active_modules()`` returns 5-element list."""
        for cfg in ALL_VALID_CONFIGURATIONS:
            assert len(cfg.active_modules()) == 5

    def test_to_string_format(self) -> None:
        """The string representation is non-empty and has the
        expected structure."""
        cfg = ALL_VALID_CONFIGURATIONS[0]
        s = cfg.to_string()
        assert "a b c" in s
        assert "A " in s
        assert "B " in s
        assert "C " in s
        n_ticks = s.count("✓")
        assert n_ticks == 5


# ============================================================================
# Tier 8 — Module voltage + current solvers + cost function (Phase 22.3)
# ============================================================================


# Thesis Sec 4.3 worked example (Figures 42-43, Eqs 31-34):
#   * Active modules: M_Ab, M_Ac, M_Ba, M_Ca, M_Cb (a (2,2,1)×(2,2,1)).
#   * Short: M_Ba.
#   * V_input = (-1, 0, 0); V_output = (1, 0, 0).
#   * Expected: M_Ba=0, M_Ca=0, M_Cb=-1, M_Ab=-2, M_Ac=-2.
_THESIS_EXAMPLE_CFG = ModuleConfiguration(grid=(
    (False, True,  True),
    (True,  False, False),
    (True,  True,  False),
))


class TestModuleVoltageSolver:
    """Validate ``solve_module_voltages`` against the worked example of
    Sec 4.3 of the Gili thesis (pages 83-85, Eqs 31-34) and against
    cross-check properties (linearity, line-voltage reconstruction)."""

    def test_thesis_sec4_3_example_exact(self) -> None:
        """Reproduces the thesis Figure 43 result exactly."""
        V = solve_module_voltages(
            _THESIS_EXAMPLE_CFG,
            (1, 0),                        # short = M_Ba
            V_input=[-1.0, 0.0, 0.0],
            V_output=[1.0, 0.0, 0.0],
        )
        assert V[(1, 0)] == pytest.approx(0.0)   # M_Ba (short)
        assert V[(2, 0)] == pytest.approx(0.0)   # M_Ca (Eq 31b)
        assert V[(2, 1)] == pytest.approx(-1.0)  # M_Cb (Eq 32b)
        assert V[(0, 1)] == pytest.approx(-2.0)  # M_Ab (Eq 33b)
        assert V[(0, 2)] == pytest.approx(-2.0)  # M_Ac (Eq 34)

    def test_short_is_always_zero(self) -> None:
        """For any cfg / any active module chosen as short, that
        module's computed voltage must be exactly 0."""
        rng = np.random.default_rng(seed=42)
        for cfg in ALL_VALID_CONFIGURATIONS[:20]:
            for short in cfg.active_modules():
                V = solve_module_voltages(
                    cfg, short,
                    V_input=rng.uniform(-3, 3, 3),
                    V_output=rng.uniform(-3, 3, 3),
                )
                assert V[short] == pytest.approx(0.0, abs=1e-12), (
                    f"cfg={cfg}, short={short}: M_short={V[short]}"
                )

    def test_invalid_short_raises(self) -> None:
        """If the short module is not in the active set, raise."""
        # M_Aa is NOT active in _THESIS_EXAMPLE_CFG (grid[0][0]=False).
        with pytest.raises(ValueError, match="not active"):
            solve_module_voltages(
                _THESIS_EXAMPLE_CFG, (0, 0),
                V_input=[-1, 0, 0], V_output=[1, 0, 0],
            )

    def test_wrong_input_shape_raises(self) -> None:
        with pytest.raises(ValueError, match="length-3"):
            solve_module_voltages(
                _THESIS_EXAMPLE_CFG, (1, 0),
                V_input=[0.0, 0.0],  # only 2 elements
                V_output=[0.0, 0.0, 0.0],
            )

    def test_linearity_in_input(self) -> None:
        """Module voltages are linear in V_input + V_output references."""
        cfg = _THESIS_EXAMPLE_CFG
        short = (1, 0)
        V_in_1 = np.array([-1.0, 0.5, 0.5])
        V_out_1 = np.array([0.8, -0.4, -0.4])
        V_in_2 = np.array([2.0, -1.0, -1.0])
        V_out_2 = np.array([-0.6, 0.3, 0.3])
        alpha, beta = 1.7, -0.4

        V1 = solve_module_voltages(cfg, short, V_in_1, V_out_1)
        V2 = solve_module_voltages(cfg, short, V_in_2, V_out_2)
        Vc = solve_module_voltages(
            cfg, short,
            alpha * V_in_1 + beta * V_in_2,
            alpha * V_out_1 + beta * V_out_2,
        )
        for key in V1:
            expected = alpha * V1[key] + beta * V2[key]
            assert Vc[key] == pytest.approx(expected, abs=1e-12)

    @pytest.mark.parametrize("seed", range(5))
    def test_returns_5_module_voltages(self, seed: int) -> None:
        """Output dict must have exactly 5 entries (one per active mod)."""
        rng = np.random.default_rng(seed=seed)
        cfg = ALL_VALID_CONFIGURATIONS[seed * 7 % 81]
        short = cfg.active_modules()[0]
        V = solve_module_voltages(
            cfg, short,
            V_input=rng.uniform(-2, 2, 3),
            V_output=rng.uniform(-2, 2, 3),
        )
        assert len(V) == 5
        assert set(V.keys()) == set(cfg.active_modules())


class TestModuleCurrentSolver:
    """Validate ``solve_module_currents`` — KCL at every node, leaf-
    stripping algorithm. Properties: KCL balance, conservation,
    linearity, leaf-current pass-through."""

    def test_kcl_at_input_nodes(self) -> None:
        """For each active configuration, the computed currents must
        satisfy KCL at every input node:
            sum_y I_xy = I_input[x] for each x.
        """
        rng = np.random.default_rng(seed=7)
        for cfg in ALL_VALID_CONFIGURATIONS[:30]:
            I_in = rng.uniform(-10, 10, 3)
            # Force conservation by reshaping I_out.
            I_out_base = rng.uniform(-10, 10, 3)
            I_out = I_out_base - (I_out_base.sum() - I_in.sum()) / 3
            assert np.isclose(I_in.sum(), I_out.sum(), atol=1e-12)

            I = solve_module_currents(cfg, I_in, I_out)

            for i in range(3):
                lhs = sum(
                    I[(ii, jj)] for (ii, jj) in cfg.active_modules()
                    if ii == i
                )
                assert lhs == pytest.approx(I_in[i], abs=1e-9), (
                    f"KCL@in[{i}] fails for {cfg.active_modules()}: "
                    f"{lhs} != {I_in[i]}"
                )

    def test_kcl_at_output_nodes(self) -> None:
        """Similarly, sum_x I_xy = I_output[y] for each y."""
        rng = np.random.default_rng(seed=11)
        for cfg in ALL_VALID_CONFIGURATIONS[:30]:
            I_in = rng.uniform(-10, 10, 3)
            I_out_base = rng.uniform(-10, 10, 3)
            I_out = I_out_base - (I_out_base.sum() - I_in.sum()) / 3
            I = solve_module_currents(cfg, I_in, I_out)

            for j in range(3):
                lhs = sum(
                    I[(ii, jj)] for (ii, jj) in cfg.active_modules()
                    if jj == j
                )
                assert lhs == pytest.approx(I_out[j], abs=1e-9)

    def test_zero_terminal_currents_gives_zero_module_currents(self) -> None:
        for cfg in ALL_VALID_CONFIGURATIONS[:10]:
            I = solve_module_currents(cfg, [0.0, 0.0, 0.0],
                                      [0.0, 0.0, 0.0])
            for v in I.values():
                assert v == pytest.approx(0.0)

    def test_conservation_violation_raises(self) -> None:
        with pytest.raises(ValueError, match="conservation"):
            solve_module_currents(
                _THESIS_EXAMPLE_CFG,
                I_input=[1.0, 0.0, 0.0],
                I_output=[1.0, 1.0, 0.0],  # sum=2, not 1
            )

    def test_returns_5_module_currents(self) -> None:
        cfg = _THESIS_EXAMPLE_CFG
        I = solve_module_currents(cfg, [10.0, -5.0, -5.0],
                                  [3.0, -3.0, 0.0])
        assert len(I) == 5
        assert set(I.keys()) == set(cfg.active_modules())

    def test_linearity_in_currents(self) -> None:
        """Module currents are linear in terminal currents."""
        cfg = _THESIS_EXAMPLE_CFG
        I_in_1 = np.array([5.0, -2.0, -3.0])
        I_out_1 = np.array([1.5, -1.5, 0.0])
        I_in_2 = np.array([-1.0, 0.5, 0.5])
        I_out_2 = np.array([0.2, -0.1, -0.1])
        alpha, beta = 0.7, 0.3

        I1 = solve_module_currents(cfg, I_in_1, I_out_1)
        I2 = solve_module_currents(cfg, I_in_2, I_out_2)
        Ic = solve_module_currents(
            cfg,
            alpha * I_in_1 + beta * I_in_2,
            alpha * I_out_1 + beta * I_out_2,
        )
        for key in I1:
            expected = alpha * I1[key] + beta * I2[key]
            assert Ic[key] == pytest.approx(expected, abs=1e-12)


class TestCostFunction:
    """Validate the cost function (Sec 5.5.3 Eqs 161-163)."""

    @pytest.fixture
    def std_params(self) -> dict:
        """Sec 5.5.3 / Tab 15: T_s=2 kHz, C=680 µF, N=6."""
        return {
            "T_s": 1.0 / 2000.0,
            "C_sm": 680e-6,
            "n_sm_per_module": 6,
        }

    def test_balanced_caps_zero_currents_gives_zero_cost(
        self, std_params: dict,
    ) -> None:
        """Perfectly balanced caps with zero terminal currents give 0."""
        cfg = _THESIS_EXAMPLE_CFG
        V_caps = np.full(9, 1000.0)
        cost = connection_cost(
            cfg, V_caps,
            I_input=[0.0, 0.0, 0.0], I_output=[0.0, 0.0, 0.0],
            **std_params,
        )
        assert cost == pytest.approx(0.0, abs=1e-12)

    def test_cost_is_nonnegative(self, std_params: dict) -> None:
        """Sum of squares: always ≥ 0."""
        rng = np.random.default_rng(seed=23)
        for cfg in ALL_VALID_CONFIGURATIONS[:15]:
            V_caps = 1000.0 + rng.uniform(-100, 100, 9)
            I_in = rng.uniform(-10, 10, 3)
            I_out_base = rng.uniform(-10, 10, 3)
            I_out = I_out_base - (I_out_base.sum() - I_in.sum()) / 3
            cost = connection_cost(
                cfg, V_caps, I_in, I_out, **std_params,
            )
            assert cost >= 0.0

    def test_imbalanced_caps_no_current_gives_sum_of_squared_eps(
        self, std_params: dict,
    ) -> None:
        """With zero terminal currents, ΔV = 0 for every module so
        cost = sum (V_xy - mean)²."""
        cfg = _THESIS_EXAMPLE_CFG
        V_caps = np.array([
            1100.0, 900.0, 1000.0,
            1050.0, 950.0, 1000.0,
            1000.0, 1000.0, 1000.0,
        ])
        cost = connection_cost(
            cfg, V_caps,
            I_input=[0.0, 0.0, 0.0], I_output=[0.0, 0.0, 0.0],
            **std_params,
        )
        expected = float(np.sum((V_caps - V_caps.mean()) ** 2))
        assert cost == pytest.approx(expected, rel=1e-12)

    def test_wrong_vcaps_shape_raises(self, std_params: dict) -> None:
        with pytest.raises(ValueError, match="length 9"):
            connection_cost(
                _THESIS_EXAMPLE_CFG,
                V_caps=np.zeros(5),
                I_input=[0.0, 0.0, 0.0], I_output=[0.0, 0.0, 0.0],
                **std_params,
            )


class TestConfigurationsContainingModule:
    """Each of the 9 modules appears in exactly 45 of 81 valid
    configurations (the 81 → 45 reduction of Sec 5.5.3 §4)."""

    @pytest.mark.parametrize("i", range(3))
    @pytest.mark.parametrize("j", range(3))
    def test_each_module_in_exactly_45(self, i: int, j: int) -> None:
        cfgs = configurations_containing_module(i, j)
        assert len(cfgs) == 45

    @pytest.mark.parametrize("i", range(3))
    @pytest.mark.parametrize("j", range(3))
    def test_returned_configs_all_active(self, i: int, j: int) -> None:
        for cfg in configurations_containing_module(i, j):
            assert cfg.is_active(i, j)


class TestSelectBestConnection:
    """Validate the Sec 5.5.3 best-connection selector."""

    @pytest.fixture
    def std_params(self) -> dict:
        return {
            "T_s": 1.0 / 2000.0,
            "C_sm": 680e-6,
            "n_sm_per_module": 6,
        }

    def test_returns_valid_config(self, std_params: dict) -> None:
        """The returned best config must contain the short module
        and be in the 81 valid set."""
        rng = np.random.default_rng(seed=99)
        V_caps = 1000.0 + rng.uniform(-50, 50, 9)
        I_in = rng.uniform(-5, 5, 3)
        I_out_base = rng.uniform(-5, 5, 3)
        I_out = I_out_base - (I_out_base.sum() - I_in.sum()) / 3

        best_cfg, best_cost = select_best_connection(
            short_module=(1, 0),
            V_caps=V_caps, I_input=I_in, I_output=I_out, **std_params,
        )
        assert best_cfg.is_active(1, 0)
        assert best_cfg in ALL_VALID_CONFIGURATIONS
        assert best_cost >= 0.0

    def test_picks_minimum_among_45(self, std_params: dict) -> None:
        """Brute-force the 45 candidates and verify best_cost is the
        global minimum."""
        rng = np.random.default_rng(seed=123)
        V_caps = 1000.0 + rng.uniform(-50, 50, 9)
        I_in = rng.uniform(-5, 5, 3)
        I_out_base = rng.uniform(-5, 5, 3)
        I_out = I_out_base - (I_out_base.sum() - I_in.sum()) / 3
        short = (0, 2)

        best_cfg, best_cost = select_best_connection(
            short_module=short, V_caps=V_caps,
            I_input=I_in, I_output=I_out, **std_params,
        )
        # Manual brute force.
        all_costs = [
            connection_cost(
                cfg, V_caps, I_in, I_out, **std_params,
            )
            for cfg in configurations_containing_module(*short)
        ]
        assert best_cost == pytest.approx(min(all_costs))
        assert best_cfg.is_active(*short)
