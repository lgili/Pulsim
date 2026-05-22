"""simplify-and-harden-numerical-surface — Phase 3 MVP tests.

Verifies the `opts.advanced()` reference view:
  - All 9 sub-fields are reachable (newton, timestep, lte, bdf_order,
    dc, stiffness, fallback, formulation, linear_solver)
  - Mutation through the view propagates to the flat field
  - Mutation through the flat field propagates to the view
  - The view doesn't introduce data duplication
"""

import pulsim as ps


def test_advanced_view_exposes_all_nine_subfields():
    opts = ps.SimulationOptions.from_preset(ps.Preset.Robust, 1e-6, 1e-3)
    adv = opts.advanced()
    # Just probe each one is reachable.
    assert adv.newton is not None
    assert adv.timestep is not None
    assert adv.lte is not None
    assert adv.bdf_order is not None
    assert adv.dc is not None
    assert adv.stiffness is not None
    assert adv.fallback is not None
    assert adv.formulation is not None
    assert adv.linear_solver is not None


def test_mutation_through_view_reaches_flat_field():
    opts = ps.SimulationOptions.from_preset(ps.Preset.Robust, 1e-6, 1e-3)
    opts.advanced().newton.max_iterations = 777
    assert opts.newton_options.max_iterations == 777


def test_mutation_through_flat_field_reaches_view():
    opts = ps.SimulationOptions.from_preset(ps.Preset.Robust, 1e-6, 1e-3)
    opts.newton_options.max_iterations = 333
    # Re-fetch view — should see the change.
    assert opts.advanced().newton.max_iterations == 333


def test_advanced_view_for_stiffness():
    opts = ps.SimulationOptions.from_preset(ps.Preset.Robust, 1e-6, 1e-3)
    adv = opts.advanced()
    assert adv.stiffness.enable is True
    adv.stiffness.enable = False
    assert opts.stiffness_config.enable is False


def test_advanced_view_for_dc_strategy():
    opts = ps.SimulationOptions()
    adv = opts.advanced()
    assert adv.dc.strategy == ps.DCStrategy.Auto
    adv.dc.strategy = ps.DCStrategy.Override
    adv.dc.strategy_override = ps.DCStrategy.PseudoTransient
    assert opts.dc_config.strategy == ps.DCStrategy.Override
    assert opts.dc_config.strategy_override == ps.DCStrategy.PseudoTransient


def test_advanced_view_for_linear_solver_quality():
    """SolverQuality (Phase 8.3) reachable via the view."""
    opts = ps.SimulationOptions()
    adv = opts.advanced()
    adv.linear_solver.solver_quality = ps.SolverQuality.Best
    adv.linear_solver.apply_solver_quality()
    assert opts.linear_solver.solver_quality == ps.SolverQuality.Best
    assert opts.linear_solver.iterative_config.preconditioner == ps.PreconditionerKind.ILUT
