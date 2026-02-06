"""Tests for runtime-complete simulation bindings (SimulationOptions/Simulator/YamlParser)."""

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


def test_simulator_with_simulation_options_runs_transient() -> None:
    circuit = _build_rc_circuit()

    opts = ps.SimulationOptions()
    opts.tstart = 0.0
    opts.tstop = 5e-4
    opts.dt = 1e-6
    opts.integrator = ps.Integrator.Trapezoidal
    opts.linear_solver.order = [ps.LinearSolverKind.KLU]
    opts.linear_solver.fallback_order = [ps.LinearSolverKind.SparseLU]

    sim = ps.Simulator(circuit, opts)
    result = sim.run_transient()

    assert result.success
    assert result.final_status == ps.SolverStatus.Success
    assert len(result.time) > 2
    assert result.total_steps > 0
    assert result.newton_iterations_total >= 0


def test_yaml_parser_load_string_returns_circuit_and_options() -> None:
    content = """
schema: pulsim-v1
version: 1
simulation:
  tstart: 0
  tstop: 5e-4
  dt: 1e-6
  integrator: trapezoidal
  solver:
    order: [klu]
components:
  - type: voltage_source
    name: V1
    nodes: [in, 0]
    waveform: {type: dc, value: 5}
  - type: resistor
    name: R1
    nodes: [in, out]
    value: 1k
  - type: capacitor
    name: C1
    nodes: [out, 0]
    value: 1u
    ic: 0
"""
    parser = ps.YamlParser()
    circuit, options = parser.load_string(content)

    assert parser.errors == []
    assert circuit.num_devices() == 3
    assert options.tstop > options.tstart

    sim = ps.Simulator(circuit, options)
    result = sim.run_transient()
    assert result.success


def test_yaml_parser_strict_mode_reports_unknown_field() -> None:
    content = """
schema: pulsim-v1
version: 1
simulation:
  unknown_field: 123
components:
  - type: resistor
    name: R1
    nodes: [in, 0]
    value: 1k
"""
    options = ps.YamlParserOptions()
    options.strict = True
    parser = ps.YamlParser(options)
    parser.load_string(content)

    assert any("unknown_field" in msg for msg in parser.errors)
