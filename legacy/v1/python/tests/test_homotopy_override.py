"""simplify-and-harden-numerical-surface — Phase 7 + 8 Python tests.

Verifies the Python-side exposure of the new DC-strategy machinery:
  - DCStrategy.Auto, Override, Homotopy, + the 5 concrete values
  - HomotopyConfig defaults + tunable fields
  - DCConvergenceConfig.strategy_override round-trip
  - DCConvergenceConfig.homotopy_config round-trip
  - LinearSolverKind.Auto/Direct/Iterative are reachable
"""

import pulsim as ps


def test_dc_strategy_enum_has_all_documented_values():
    members = ps.DCStrategy.__members__
    expected = {"Auto", "Override", "Direct",
                "GminStepping", "SourceStepping",
                "PseudoTransient", "Homotopy"}
    assert set(members.keys()) == expected


def test_linear_solver_kind_has_abstract_and_concrete_values():
    members = ps.LinearSolverKind.__members__
    # Phase 8 abstract values
    assert "Auto" in members
    assert "Direct" in members
    assert "Iterative" in members
    # 6 concrete engines stay available
    for name in ("SparseLU", "EnhancedSparseLU", "KLU",
                 "GMRES", "BiCGSTAB", "CG"):
        assert name in members, f"missing concrete engine {name}"


def test_homotopy_config_defaults():
    h = ps.HomotopyConfig()
    assert h.enable is True
    assert h.ladder_steps == 5
    assert h.max_newton_per_step == 10


def test_homotopy_config_tunable():
    h = ps.HomotopyConfig()
    h.enable = False
    h.ladder_steps = 20
    h.max_newton_per_step = 50
    assert h.enable is False
    assert h.ladder_steps == 20
    assert h.max_newton_per_step == 50


def test_dc_convergence_config_has_homotopy_subblock():
    cfg = ps.DCConvergenceConfig()
    # Defaults: strategy = Auto, homotopy_config enabled with 5 steps.
    assert cfg.strategy == ps.DCStrategy.Auto
    assert cfg.homotopy_config.enable is True
    assert cfg.homotopy_config.ladder_steps == 5


def test_dc_strategy_override_round_trip():
    cfg = ps.DCConvergenceConfig()
    cfg.strategy = ps.DCStrategy.Override
    cfg.strategy_override = ps.DCStrategy.PseudoTransient
    assert cfg.strategy == ps.DCStrategy.Override
    assert cfg.strategy_override == ps.DCStrategy.PseudoTransient


def test_dc_strategy_override_default_is_direct():
    cfg = ps.DCConvergenceConfig()
    # When the user picks Override but doesn't pick a concrete strategy,
    # the default override is Direct (the conservative starting point).
    assert cfg.strategy_override == ps.DCStrategy.Direct
