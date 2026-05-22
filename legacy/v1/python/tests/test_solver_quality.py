"""simplify-and-harden-numerical-surface — Phase 8.3 Python tests.

Verifies the friendly SolverQuality knob via Python bindings:
  - Enum has Fast / Default / Best
  - LinearSolverStackConfig.solver_quality defaults to Default
  - apply_solver_quality() maps to iterative_config.preconditioner:
        Fast    → PreconditionerKind.None_
        Default → PreconditionerKind.ILU0
        Best    → PreconditionerKind.ILUT
"""

import pulsim as ps


def test_solver_quality_enum_has_three_values():
    members = ps.SolverQuality.__members__
    assert set(members.keys()) == {"Fast", "Default", "Best"}


def test_default_solver_quality_is_default():
    cfg = ps.LinearSolverStackConfig()
    assert cfg.solver_quality == ps.SolverQuality.Default


def test_apply_solver_quality_fast_disables_preconditioner():
    cfg = ps.LinearSolverStackConfig()
    cfg.solver_quality = ps.SolverQuality.Fast
    cfg.apply_solver_quality()
    assert cfg.iterative_config.preconditioner == ps.PreconditionerKind.None_


def test_apply_solver_quality_default_picks_ilu0():
    cfg = ps.LinearSolverStackConfig()
    cfg.solver_quality = ps.SolverQuality.Default
    cfg.apply_solver_quality()
    assert cfg.iterative_config.preconditioner == ps.PreconditionerKind.ILU0


def test_apply_solver_quality_best_picks_ilut():
    cfg = ps.LinearSolverStackConfig()
    cfg.solver_quality = ps.SolverQuality.Best
    cfg.apply_solver_quality()
    assert cfg.iterative_config.preconditioner == ps.PreconditionerKind.ILUT


def test_solver_quality_does_not_clobber_other_fields():
    cfg = ps.LinearSolverStackConfig()
    cfg.size_threshold = 9999
    cfg.allow_fallback = False
    cfg.solver_quality = ps.SolverQuality.Best
    cfg.apply_solver_quality()
    # Verify the preconditioner update didn't touch unrelated fields.
    assert cfg.size_threshold == 9999
    assert cfg.allow_fallback is False
