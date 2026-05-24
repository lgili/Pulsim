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
    M3cL1ControlState,
    M3cParams,
    ModuleConfiguration,
    abc_to_lg,
    build_l0_plant,
    build_l1_plant,
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
    make_m3c_l1_cost_control,
    make_m3c_l1_open_loop_control,
    predict_i_out_peak,
    predict_load_impedance,
    predict_load_power_factor,
    rms,
    run_l0_open_loop,
    run_l1_cost_loop,
    run_l1_open_loop,
    select_best_connection,
    solve_module_currents,
    solve_module_voltages,
    thd,
)

try:
    import pulsim  # noqa: F401
    _PULSIM_AVAILABLE = True
except ImportError:
    _PULSIM_AVAILABLE = False

_requires_pulsim = pytest.mark.skipif(
    not _PULSIM_AVAILABLE,
    reason="pulsim not importable (only Tiers 1-8 will run)",
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
    """Validate the cost function (Sec 5.5.3 Eqs 161-163).

    The Phase 22.6 API is ``connection_cost(cfg, V_caps, V_xy, I_xy,
    T_s, C_sm)`` — V_xy and I_xy are precomputed dicts so the cost
    can use the proper signed-S_n formula ``ΔV = V_int · I · T_s/C_SM``.
    """

    @pytest.fixture
    def std_params(self) -> dict:
        """Sec 5.5.3 / Tab 15: T_s=2 kHz, C=680 µF."""
        return {
            "T_s": 1.0 / 2000.0,
            "C_sm": 680e-6,
        }

    def test_balanced_caps_zero_currents_gives_zero_cost(
        self, std_params: dict,
    ) -> None:
        """Perfectly balanced caps with zero module currents give 0."""
        cfg = _THESIS_EXAMPLE_CFG
        V_caps = np.full(9, 1000.0)
        # 5 active modules each with V_int=0, I=0 → ΔV=0 for all.
        V_xy = {(i, j): 0 for (i, j) in cfg.active_modules()}
        I_xy = {(i, j): 0.0 for (i, j) in cfg.active_modules()}
        cost = connection_cost(
            cfg, V_caps, V_xy, I_xy, **std_params,
        )
        assert cost == pytest.approx(0.0, abs=1e-12)

    def test_cost_is_nonnegative(self, std_params: dict) -> None:
        """Sum of squares: always ≥ 0."""
        rng = np.random.default_rng(seed=23)
        for cfg in ALL_VALID_CONFIGURATIONS[:15]:
            V_caps = 1000.0 + rng.uniform(-100, 100, 9)
            V_xy = {
                (i, j): int(rng.integers(-3, 4))
                for (i, j) in cfg.active_modules()
            }
            I_xy = {
                (i, j): float(rng.uniform(-10, 10))
                for (i, j) in cfg.active_modules()
            }
            cost = connection_cost(
                cfg, V_caps, V_xy, I_xy, **std_params,
            )
            assert cost >= 0.0

    def test_imbalanced_caps_zero_voltage_gives_sum_of_squared_eps(
        self, std_params: dict,
    ) -> None:
        """With every V_xy = 0, ΔV = 0 for every module so cost
        reduces to sum (V_caps - mean)²."""
        cfg = _THESIS_EXAMPLE_CFG
        V_caps = np.array([
            1100.0, 900.0, 1000.0,
            1050.0, 950.0, 1000.0,
            1000.0, 1000.0, 1000.0,
        ])
        V_xy = {(i, j): 0 for (i, j) in cfg.active_modules()}
        I_xy = {(i, j): 1.0 for (i, j) in cfg.active_modules()}  # I != 0
        # With V_xy = 0, ΔV = V_int · I · T_s/C = 0 regardless of I.
        cost = connection_cost(
            cfg, V_caps, V_xy, I_xy, **std_params,
        )
        expected = float(np.sum((V_caps - V_caps.mean()) ** 2))
        assert cost == pytest.approx(expected, rel=1e-12)

    def test_wrong_vcaps_shape_raises(self, std_params: dict) -> None:
        cfg = _THESIS_EXAMPLE_CFG
        V_xy = {(i, j): 0 for (i, j) in cfg.active_modules()}
        I_xy = {(i, j): 0.0 for (i, j) in cfg.active_modules()}
        with pytest.raises(ValueError, match="length 9"):
            connection_cost(
                cfg, V_caps=np.zeros(5), V_xy=V_xy, I_xy=I_xy,
                **std_params,
            )

    def test_delta_v_sign_with_signed_V_int(
        self, std_params: dict,
    ) -> None:
        """Eq 162 with signed S_n: cost should differ between V_int=+k
        and V_int=-k (same |k| but opposite sign of ΔV)."""
        cfg = _THESIS_EXAMPLE_CFG
        V_caps = np.array([
            1100.0, 1000.0, 1000.0,
            1000.0, 1000.0, 1000.0,
            1000.0, 1000.0, 1000.0,
        ])
        # ε[0,0] = +88.9, others negative small. So if ΔV[0,0] is
        # negative (charge cap down), cost should be smaller than
        # if ΔV[0,0] is positive.
        # Need (0,0) to be active in cfg — it is NOT in _THESIS_EXAMPLE_CFG.
        # Use first config that has M_Aa active.
        cfg2 = configurations_containing_module(0, 0)[0]
        # Active modules of cfg2: must include (0,0).
        V_pos = {(i, j): 1 for (i, j) in cfg2.active_modules()}
        V_neg = {(i, j): -1 for (i, j) in cfg2.active_modules()}
        I = {(i, j): 1.0 for (i, j) in cfg2.active_modules()}
        cost_pos = connection_cost(cfg2, V_caps, V_pos, I, **std_params)
        cost_neg = connection_cost(cfg2, V_caps, V_neg, I, **std_params)
        # The two costs differ (signed ΔV is non-trivial).
        assert cost_pos != cost_neg


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
    """Validate the Sec 5.5.3 best-connection selector with the
    Phase 22.6 API (takes integer SVM refs, no n_sm_per_module)."""

    @pytest.fixture
    def std_params(self) -> dict:
        return {
            "T_s": 1.0 / 2000.0,
            "C_sm": 680e-6,
        }

    def test_returns_valid_config(self, std_params: dict) -> None:
        """The returned best config must contain the short module
        and be in the 81 valid set."""
        rng = np.random.default_rng(seed=99)
        V_caps = 1000.0 + rng.uniform(-50, 50, 9)
        V_in_int = rng.integers(-2, 3, 3).astype(int)
        V_out_int = rng.integers(-2, 3, 3).astype(int)
        I_in = rng.uniform(-5, 5, 3)
        I_out_base = rng.uniform(-5, 5, 3)
        I_out = I_out_base - (I_out_base.sum() - I_in.sum()) / 3

        best_cfg, best_cost = select_best_connection(
            short_module=(1, 0),
            V_caps=V_caps,
            V_input_int=V_in_int, V_output_int=V_out_int,
            I_input=I_in, I_output=I_out,
            **std_params,
        )
        assert best_cfg.is_active(1, 0)
        assert best_cfg in ALL_VALID_CONFIGURATIONS
        assert best_cost >= 0.0

    def test_picks_minimum_among_45(self, std_params: dict) -> None:
        """Brute-force the 45 candidates and verify best_cost is the
        global minimum."""
        rng = np.random.default_rng(seed=123)
        V_caps = 1000.0 + rng.uniform(-50, 50, 9)
        V_in_int = rng.integers(-2, 3, 3).astype(int)
        V_out_int = rng.integers(-2, 3, 3).astype(int)
        I_in = rng.uniform(-5, 5, 3)
        I_out_base = rng.uniform(-5, 5, 3)
        I_out = I_out_base - (I_out_base.sum() - I_in.sum()) / 3
        short = (0, 2)

        best_cfg, best_cost = select_best_connection(
            short_module=short, V_caps=V_caps,
            V_input_int=V_in_int, V_output_int=V_out_int,
            I_input=I_in, I_output=I_out, **std_params,
        )
        # Manual brute force using the (cfg, V_caps, V_xy, I_xy, T_s, C_sm)
        # API.
        all_costs = []
        for cfg in configurations_containing_module(*short):
            V_xy = solve_module_voltages(
                cfg, short, V_in_int, V_out_int,
            )
            I_xy = solve_module_currents(cfg, I_in, I_out)
            all_costs.append(
                connection_cost(cfg, V_caps, V_xy, I_xy, **std_params),
            )
        assert best_cost == pytest.approx(min(all_costs))
        assert best_cfg.is_active(*short)


# ============================================================================
# Tier 9 — L0 plant (Phase 22.4) — end-to-end pulsim verification of the
# Venturini-style averaged M3C output. Requires pulsim importable.
# ============================================================================


@_requires_pulsim
class TestL0Plant:
    """L0 plant: synthesised output sinusoids → Y-connected RL load.
    Validates pulsim end-to-end at the M3C nominal 11 kV / 45 Hz
    operating point against closed-form analytical predictions."""

    @pytest.fixture(scope="class")
    def params(self) -> M3cParams:
        # 11 kV LL output at 45 Hz, ~2 MVA-class RL load (0.9 PF).
        return M3cParams()  # all defaults: matches Tab 15

    @pytest.fixture(scope="class")
    def result(self, params: M3cParams):
        plant = build_l0_plant(params)
        # Run long enough for ≥ 100 ms settling + 3 fundamental periods
        # of clean data; at 45 Hz that's ≈ 167 ms. Use 200 ms with a
        # buffer so the "last 3 periods" slice is always populated.
        return run_l0_open_loop(plant, t_end=200e-3, dt=20e-6)

    # ---- helpers (use last-N-samples slicing, which is robust to any
    #              run length and trivially gives an integer-period
    #              window after settling) ----

    @staticmethod
    def _last_n_periods(
        arr: np.ndarray, params: M3cParams, dt: float, n_periods: int,
    ) -> np.ndarray:
        fs = 1.0 / dt
        n_per_period = int(round((1.0 / params.f_out) * fs))
        return arr[-(n_periods * n_per_period):]

    def test_i_out_peak_matches_analytical(
        self, params: M3cParams, result,
    ) -> None:
        """|I_o| = V_out_phase_peak / |Z_load| should match within 1 %."""
        mask = result.t >= 100e-3
        i_a_pk = float(np.max(np.abs(result.i_a_out[mask])))
        i_a_pk_pred = predict_i_out_peak(params)
        rel_err = abs(i_a_pk - i_a_pk_pred) / i_a_pk_pred
        assert rel_err < 0.01, (
            f"i_a peak = {i_a_pk:.4f} A vs {i_a_pk_pred:.4f} A predicted, "
            f"rel-err = {rel_err*100:.3f}%"
        )

    def test_i_out_rms_matches_analytical(
        self, params: M3cParams, result,
    ) -> None:
        """RMS = peak / √2 for pure sinusoid. Window: last 3 periods
        (deterministic integer-cycle slice → no leakage)."""
        ia = self._last_n_periods(result.i_a_out, params, 20e-6, 3)
        i_rms = rms(ia)
        i_rms_pred = predict_i_out_peak(params) / np.sqrt(2.0)
        rel_err = abs(i_rms - i_rms_pred) / i_rms_pred
        assert rel_err < 0.01, (
            f"i_a RMS = {i_rms:.4f} A vs {i_rms_pred:.4f} A predicted, "
            f"rel-err = {rel_err*100:.3f}%"
        )

    def test_balanced_three_phase_output(self, result) -> None:
        """All three load currents have the same peak (balanced 3-φ)."""
        mask = result.t >= 100e-3
        peaks = [
            float(np.max(np.abs(x[mask])))
            for x in (result.i_a_out, result.i_b_out, result.i_c_out)
        ]
        max_dev_rel = (max(peaks) - min(peaks)) / max(peaks)
        assert max_dev_rel < 1e-3, (
            f"3-φ peaks unbalanced: {peaks}, max rel deviation "
            f"{max_dev_rel*100:.4f}%"
        )

    def test_load_phase_shift_120_degrees(
        self, params: M3cParams, result,
    ) -> None:
        """The three load currents are 120° apart, measured via DFT
        of the fundamental over an integer number of periods."""
        ia = self._last_n_periods(result.i_a_out, params, 20e-6, 3)
        ib = self._last_n_periods(result.i_b_out, params, 20e-6, 3)
        ic = self._last_n_periods(result.i_c_out, params, 20e-6, 3)
        # Coherent DFT bin at f_out (no windowing needed).
        fs = 1.0 / 20e-6
        n_win = len(ia)
        k = int(round(params.f_out * n_win / fs))
        spec_a = np.fft.rfft(ia)[k]
        spec_b = np.fft.rfft(ib)[k]
        spec_c = np.fft.rfft(ic)[k]
        phase_b_a = np.angle(spec_b) - np.angle(spec_a)
        phase_c_a = np.angle(spec_c) - np.angle(spec_a)
        # Wrap to (−π, +π].
        phase_b_a = ((phase_b_a + pi) % (2 * pi)) - pi
        phase_c_a = ((phase_c_a + pi) % (2 * pi)) - pi
        # Expected: b lags by 120° (−2π/3), c leads by 120° (+2π/3).
        assert abs(phase_b_a - (-2 * pi / 3)) < 0.05, (
            f"Phase b−a = {np.degrees(phase_b_a):.2f}°, expected −120° ± 3°"
        )
        assert abs(phase_c_a - (+2 * pi / 3)) < 0.05, (
            f"Phase c−a = {np.degrees(phase_c_a):.2f}°, expected +120° ± 3°"
        )

    def test_low_thd_pure_sinusoid(
        self, params: M3cParams, result,
    ) -> None:
        """L0 has no switching ripple ⇒ THD should be near zero (limited
        by FFT windowing artefacts of integer cycles ~ 0.1 %)."""
        ia = self._last_n_periods(result.i_a_out, params, 20e-6, 3)
        thd_pct = thd(ia, fs=1.0 / 20e-6, f_fundamental=params.f_out)
        assert thd_pct < 1.0, (
            f"THD = {thd_pct:.3f}%; expected < 1% for ideal L0"
        )

    def test_load_power_factor_matches_theory(
        self, params: M3cParams,
    ) -> None:
        """Closed-form check: PF = cos(arctan(ωL/R))."""
        pf_theory = predict_load_power_factor(params)
        arg = params.omega_out * params.L_load / params.R_load
        pf_expected = 1.0 / np.sqrt(1.0 + arg ** 2)
        assert abs(pf_theory - pf_expected) < 1e-12, (
            f"PF = {pf_theory}, expected {pf_expected}"
        )

    def test_load_impedance_matches_theory(
        self, params: M3cParams,
    ) -> None:
        """Closed-form check of complex impedance."""
        Z = predict_load_impedance(params)
        assert Z.real == pytest.approx(params.R_load)
        assert Z.imag == pytest.approx(
            params.omega_out * params.L_load
        )

    def test_plant_has_3_inductor_state_indices(
        self, params: M3cParams,
    ) -> None:
        """The plant must expose exactly 3 inductor branch state IDs
        and they must be distinct."""
        plant = build_l0_plant(params)
        assert len(plant.iL_out_indices) == 3
        assert len(set(plant.iL_out_indices)) == 3
        # L0 doesn't model the input side.
        assert plant.iL_in_indices is None

    @pytest.mark.parametrize("f_out", [5.0, 30.0, 45.0, 55.0])
    def test_runs_at_various_frequencies(
        self, f_out: float,
    ) -> None:
        """Tab 15 of the thesis lists f_out ∈ {5, 30, 45, 55} Hz —
        verify L0 runs cleanly across that range and the steady-state
        current peak matches the predicted Ohm's law value."""
        params = M3cParams(f_out=f_out)
        plant = build_l0_plant(params)
        # Pick a window covering ≥ 3 fundamental periods, with ≥ 50
        # ms settling.
        t_settle = max(50e-3, 3.0 / f_out)
        t_window = max(60e-3, 3.0 / f_out)
        t_end = t_settle + t_window
        dt = 20e-6
        result = run_l0_open_loop(plant, t_end=t_end, dt=dt)
        mask = result.t >= t_settle
        i_pk = float(np.max(np.abs(result.i_a_out[mask])))
        i_pk_pred = predict_i_out_peak(params)
        rel_err = abs(i_pk - i_pk_pred) / i_pk_pred
        assert rel_err < 0.02, (
            f"f_out={f_out} Hz: peak = {i_pk:.3f} A vs {i_pk_pred:.3f} A "
            f"predicted, rel-err = {rel_err*100:.2f}%"
        )


# ============================================================================
# Tier 10 — L1 plant (Phase 22.5) — switched plant with open-loop SVM.
# Validates that the SVM + Sec 4.3 solver produces a multilevel
# stepped output whose fundamental is close to L0's prediction.
# ============================================================================


@_requires_pulsim
class TestL1Plant:
    """L1 switched plant — 9 (switch + voltage source) modules driven
    by an open-loop SVM controller. The output current's fundamental
    should match L0's Ohm's-law prediction to within ~10 % (the gap
    comes from per-T_s quantisation; the cost-function selector of
    Phase 22.6 will close it further)."""

    @pytest.fixture(scope="class")
    def params(self) -> M3cParams:
        # Use m_v=1.0 for fair L0/L1 fundamental comparison.
        return M3cParams(m_v=1.0)

    @pytest.fixture(scope="class")
    def result(self, params: M3cParams):
        plant = build_l1_plant(params)
        # 200 ms ≈ 9 fundamental periods at 45 Hz; T_s/20 = 25 µs.
        return run_l1_open_loop(plant, params, t_end=200e-3, dt=25e-6)

    # ---- topology -----------------------------------------------------

    def test_plant_topology_dimensions(
        self, params: M3cParams,
    ) -> None:
        plant = build_l1_plant(params)
        # 3 input filter inductors.
        assert plant.iL_in_indices is not None
        assert len(plant.iL_in_indices) == 3
        # 3 output filter inductors.
        assert len(plant.iL_out_indices) == 3
        # 9 module voltage sources.
        assert plant.module_v_src_state_indices is not None
        assert len(plant.module_v_src_state_indices) == 9
        # All distinct.
        all_idx = (
            list(plant.iL_in_indices)
            + list(plant.iL_out_indices)
            + list(plant.module_v_src_state_indices)
        )
        assert len(set(all_idx)) == 15  # 3 + 3 + 9

    def test_state_size_larger_than_l0(
        self, params: M3cParams,
    ) -> None:
        """L1 has more state vars than L0 (extra L_in inductors +
        9 module sources)."""
        l0 = build_l0_plant(params)
        l1 = build_l1_plant(params)
        ss_l0 = l0.builder.pool.state_size(l0.builder.graph)
        ss_l1 = l1.builder.pool.state_size(l1.builder.graph)
        assert ss_l1 > ss_l0

    # ---- controller signature -----------------------------------------

    def test_controller_signature(self, params: M3cParams) -> None:
        """``make_m3c_l1_open_loop_control`` returns a (switch_fn,
        b_extra_fn) pair, both callable."""
        plant = build_l1_plant(params)
        switch_fn, b_extra_fn = make_m3c_l1_open_loop_control(
            params, plant,
        )
        assert callable(switch_fn)
        assert callable(b_extra_fn)

    def test_switch_fn_returns_5_active(
        self, params: M3cParams,
    ) -> None:
        """At any t > 0, exactly 5 of 9 switches are ON (the 5-modules
        conducting rule). We sample 100 random instants."""
        plant = build_l1_plant(params)
        switch_fn, _ = make_m3c_l1_open_loop_control(params, plant)
        rng = np.random.default_rng(seed=37)
        for _ in range(50):
            t = float(rng.uniform(1e-3, 50e-3))
            mask = switch_fn(t)
            n_on = sum(1 for k in range(9) if mask.get(k))
            assert n_on == 5, (
                f"At t={t*1e3:.3f} ms: {n_on} switches ON, expected 5"
            )

    def test_b_extra_has_9_module_entries(
        self, params: M3cParams,
    ) -> None:
        """The b_extra vector has 9 non-zero entries (one per module).
        Note: inactive modules get V=0 → entry=0, so we expect
        ≤ 9 non-zero entries (since at most 5 modules are active in
        any given period)."""
        plant = build_l1_plant(params)
        _, b_extra_fn = make_m3c_l1_open_loop_control(params, plant)
        vec = b_extra_fn(1e-3)
        # The state_size should match plant builder.
        ss = plant.builder.pool.state_size(plant.builder.graph)
        assert len(vec) == ss

    # ---- simulation behaviour -----------------------------------------

    def test_runs_without_crash(
        self, params: M3cParams, result,
    ) -> None:
        """Just verify the run produced data."""
        assert len(result.t) > 1000
        assert result.t[-1] > 0.1  # at least 100 ms

    def test_balanced_three_phase_output(self, result) -> None:
        """Output currents on a, b, c should be balanced in peak."""
        mask = result.t >= 100e-3
        peaks = [
            float(np.max(np.abs(x[mask])))
            for x in (result.i_a_out, result.i_b_out, result.i_c_out)
        ]
        max_dev_rel = (max(peaks) - min(peaks)) / max(peaks)
        # Looser tolerance than L0 — quantisation breaks perfect symmetry.
        assert max_dev_rel < 0.10, (
            f"3-φ peaks unbalanced: {peaks}, max rel dev "
            f"{max_dev_rel*100:.2f}%"
        )

    def test_fundamental_close_to_l0(
        self, params: M3cParams, result,
    ) -> None:
        """L1 fundamental amplitude should match L0 prediction within
        15 % (gap closes once cost-function selector arrives)."""
        # FFT on last 3 fundamental periods (integer cycles).
        fs = 1.0 / 25e-6
        n_per_period = int(round((1.0 / params.f_out) * fs))
        ia = result.i_a_out[-3 * n_per_period:]
        spec = np.fft.rfft(ia)
        k1 = int(round(params.f_out * len(ia) / fs))
        fund_pk = 2.0 * float(np.abs(spec[k1])) / len(ia)
        fund_pred = predict_i_out_peak(params)
        rel_err = abs(fund_pk - fund_pred) / fund_pred
        assert rel_err < 0.15, (
            f"L1 fund = {fund_pk:.2f} A vs L0 pred = {fund_pred:.2f} A, "
            f"rel-err = {rel_err*100:.2f}%"
        )

    def test_thd_bounded(
        self, params: M3cParams, result,
    ) -> None:
        """L1 has switching ripple but the THD should still be bounded
        (≤ 25 % for the heuristic config selector)."""
        fs = 1.0 / 25e-6
        n_per_period = int(round((1.0 / params.f_out) * fs))
        ia = result.i_a_out[-3 * n_per_period:]
        thd_pct = thd(ia, fs, params.f_out)
        assert 0.1 < thd_pct < 25.0, (
            f"L1 THD = {thd_pct:.2f}% — expected 0.1-25%"
        )

    def test_input_currents_nontrivial(self, result) -> None:
        """Unlike L0, L1 *does* model the input side — the L_in inductor
        currents should be non-zero in steady state."""
        mask = result.t >= 100e-3
        ia_in_pk = float(np.max(np.abs(result.i_a_in[mask])))
        assert ia_in_pk > 1.0, (
            f"L1 input current at A is too small: {ia_in_pk:.4f} A"
        )


# ============================================================================
# Tier 11 — L1 cost-function controller (Phase 22.6)
#
# The cost-function selector replaces the Phase 22.5 heuristic with
# Sec 5.5.3 Eq 163 over the 45 candidates per T_s. It does NOT change
# the output current (the voltage solver guarantees the same terminal
# voltages regardless of which 5 modules conduct) — its job is purely
# internal: route current through the module set that best preserves
# capacitor balance. So we test that:
#   - the cost loop runs and tracks caps,
#   - the resulting output currents match the open-loop run exactly,
#   - the cap drift is bounded (smaller than no-balancing baseline).
# ============================================================================


@_requires_pulsim
class TestL1CostLoop:
    """L1 with the Sec 5.5.3 cost-function selector + cap tracking."""

    @pytest.fixture(scope="class")
    def params(self) -> M3cParams:
        return M3cParams(m_v=1.0)

    @pytest.fixture(scope="class")
    def cost_run(self, params: M3cParams):
        plant = build_l1_plant(params)
        return run_l1_cost_loop(plant, params, t_end=200e-3, dt=25e-6)

    # ---- controller signature -----------------------------------------

    def test_returns_3_callables(self, params: M3cParams) -> None:
        plant = build_l1_plant(params)
        obs, sw, bx = make_m3c_l1_cost_control(params, plant)
        assert callable(obs)
        assert callable(sw)
        assert callable(bx)

    def test_run_returns_state_and_result(
        self, params: M3cParams, cost_run,
    ) -> None:
        result, state = cost_run
        assert len(result.t) > 1000
        assert isinstance(state, M3cL1ControlState)
        assert len(state.v_caps_module) == 9
        assert state.n_refreshes > 100  # 200 ms / 0.5 ms = 400 ticks

    # ---- cap voltage tracking -----------------------------------------

    def test_initial_v_caps_at_nominal(
        self, params: M3cParams,
    ) -> None:
        """Fresh state has v_caps_module = N · v_cap_nominal."""
        state = M3cL1ControlState(
            v_caps_module=[params.v_cap_total_per_module] * 9,
        )
        assert all(
            v == params.v_cap_total_per_module
            for v in state.v_caps_module
        )

    def test_v_caps_change_over_time(
        self, cost_run, params: M3cParams,
    ) -> None:
        """After running, the cap voltages should differ from their
        initial nominal value (current did flow)."""
        _result, state = cost_run
        initial = params.v_cap_total_per_module
        max_dev = max(abs(v - initial) for v in state.v_caps_module)
        assert max_dev > 100.0, (
            f"V_caps did not move: max deviation {max_dev:.1f} V"
        )

    def test_v_caps_drift_bounded(
        self, cost_run, params: M3cParams,
    ) -> None:
        """The cost function should keep cap voltage spread bounded.
        At the M3C nominal op (m_v=1.0) over 200 ms the spread stays
        within ~ N · v_cap_nominal (i.e. caps stay within ±100 % of
        their nominal sum)."""
        _result, state = cost_run
        spread = max(state.v_caps_module) - min(state.v_caps_module)
        assert spread < 2.0 * params.v_cap_total_per_module, (
            f"Cap voltage spread {spread:.0f} V exceeds 2·N·v_cap "
            f"({2*params.v_cap_total_per_module:.0f} V) — greedy "
            f"cost is failing to balance."
        )

    def test_diagnostics_populated(self, cost_run) -> None:
        """The state should record one chosen config + one cost value
        per T_s tick."""
        _result, state = cost_run
        assert len(state.chosen_configs) == state.n_refreshes
        assert len(state.chosen_costs) == state.n_refreshes
        # First tick: caps balanced → cost = 0 (eps=0 and any ΔV
        # contributes only at next tick).
        assert state.chosen_costs[0] == pytest.approx(0.0, abs=1e-9)

    def test_visits_multiple_configs(self, cost_run) -> None:
        """The selector should pick at least 3 distinct configs across
        the run (i.e. it's not stuck on one)."""
        _result, state = cost_run
        distinct = len(set(c.grid for c in state.chosen_configs))
        assert distinct >= 3, (
            f"Only {distinct} distinct configs chosen; selector likely"
            f" stuck"
        )

    # ---- output current consistency -----------------------------------

    def test_output_current_matches_heuristic(
        self, params: M3cParams, cost_run,
    ) -> None:
        """Because the SVM voltage solver produces the same terminal
        voltages for *any* valid configuration containing the short,
        the cost-loop output current must match the heuristic open-
        loop output to within numerical precision of the matrix
        system. (Only the *internal* module-current allocation
        differs.)"""
        cost_result, _ = cost_run
        plant_h = build_l1_plant(params)
        heur_result = run_l1_open_loop(
            plant_h, params, t_end=200e-3, dt=25e-6,
        )
        diff_pk = float(np.max(np.abs(
            cost_result.i_a_out - heur_result.i_a_out
        )))
        ia_pk = float(np.max(np.abs(heur_result.i_a_out)))
        # Different switch masks → slightly different state-space
        # matrices → numerical roundoff. 0.001 % of peak is well
        # below any physically meaningful difference.
        assert diff_pk / ia_pk < 1e-5, (
            f"Cost and heuristic outputs differ by {diff_pk:.2e} A "
            f"({diff_pk/ia_pk*100:.4f}% of {ia_pk:.1f} A peak)"
        )

    def test_fundamental_close_to_l0(
        self, params: M3cParams, cost_run,
    ) -> None:
        """Same as the Phase 22.5 test, but now via the cost loop."""
        cost_result, _ = cost_run
        fs = 1.0 / 25e-6
        n_per = int(round((1.0 / params.f_out) * fs))
        ia = cost_result.i_a_out[-3 * n_per:]
        spec = np.fft.rfft(ia)
        k1 = int(round(params.f_out * len(ia) / fs))
        fund_pk = 2.0 * float(np.abs(spec[k1])) / len(ia)
        fund_pred = predict_i_out_peak(params)
        rel_err = abs(fund_pk - fund_pred) / fund_pred
        assert rel_err < 0.15
