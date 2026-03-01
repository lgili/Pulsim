"""Architecture contract tests for fixed/variable shared solver services."""

from __future__ import annotations

import pulsim as ps


def _build_rc_circuit() -> ps.Circuit:
    circuit = ps.Circuit()
    n_in = circuit.add_node("in")
    n_out = circuit.add_node("out")
    gnd = circuit.ground()

    circuit.add_voltage_source("V1", n_in, gnd, 5.0)
    circuit.add_resistor("R1", n_in, n_out, 1_000.0)
    circuit.add_capacitor("C1", n_out, gnd, 1e-6, 0.0)
    return circuit


def _run_mode(mode: ps.StepMode) -> ps.SimulationResult:
    opts = ps.SimulationOptions()
    opts.tstart = 0.0
    opts.tstop = 5e-6
    opts.dt = 1e-6
    opts.dt_min = 1e-9
    opts.dt_max = 2e-6
    opts.enable_bdf_order_control = False
    opts.linear_solver.order = [ps.LinearSolverKind.KLU]
    opts.linear_solver.fallback_order = [ps.LinearSolverKind.SparseLU]

    opts.step_mode = mode
    if mode == ps.StepMode.Fixed:
        opts.adaptive_timestep = False
    else:
        opts.adaptive_timestep = True

    result = ps.Simulator(_build_rc_circuit(), opts).run_transient()
    assert result.success
    assert result.final_status == ps.SolverStatus.Success
    return result


def test_fixed_and_variable_modes_share_solver_service_contracts() -> None:
    fixed = _run_mode(ps.StepMode.Fixed)
    variable = _run_mode(ps.StepMode.Variable)

    for result in (fixed, variable):
        linear = result.linear_solver_telemetry
        backend = result.backend_telemetry
        assert linear.total_solve_calls >= 1
        assert linear.total_factorize_calls >= 1
        assert linear.total_iterations >= 0
        assert linear.total_fallbacks >= 0
        assert backend.selected_backend == "native"
        assert backend.solver_family == "native"
        assert backend.equation_assemble_system_calls >= 1
        assert backend.equation_assemble_residual_calls >= 1
        assert backend.linear_factor_cache_hits >= 0
        assert backend.linear_factor_cache_misses >= 0
        assert backend.linear_factor_cache_invalidations >= 0
        assert backend.reserved_output_samples >= 0
        assert backend.time_series_reallocations >= 0
        assert backend.state_series_reallocations >= 0
        assert backend.virtual_channel_reallocations >= 0

    linear_contract_fields = (
        "total_solve_calls",
        "total_analyze_calls",
        "total_factorize_calls",
        "total_iterations",
        "total_fallbacks",
    )
    backend_contract_fields = (
        "selected_backend",
        "solver_family",
        "equation_assemble_system_calls",
        "equation_assemble_residual_calls",
        "linear_factor_cache_hits",
        "linear_factor_cache_misses",
        "linear_factor_cache_invalidations",
        "linear_factor_cache_last_invalidation_reason",
        "reserved_output_samples",
        "time_series_reallocations",
        "state_series_reallocations",
        "virtual_channel_reallocations",
    )

    for field in linear_contract_fields:
        assert hasattr(fixed.linear_solver_telemetry, field)
        assert hasattr(variable.linear_solver_telemetry, field)
    for field in backend_contract_fields:
        assert hasattr(fixed.backend_telemetry, field)
        assert hasattr(variable.backend_telemetry, field)
