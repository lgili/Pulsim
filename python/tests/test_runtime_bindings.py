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


def test_yaml_parser_supports_legacy_si_suffix_words() -> None:
    content = """
schema: pulsim-v1
version: 1
simulation:
  tstart: 0
  tstop: 2milli
  dt: 10micro
components:
  - type: voltage_source
    name: V1
    nodes: [in, 0]
    waveform: {type: dc, value: 12}
  - type: resistor
    name: R1
    nodes: [in, out]
    value: 2kilo
  - type: resistor
    name: R2
    nodes: [out, 0]
    value: 2K
  - type: capacitor
    name: C1
    nodes: [out, 0]
    value: 1uF
"""

    parser = ps.YamlParser()
    circuit, options = parser.load_string(content)

    assert parser.errors == []
    assert abs(options.tstop - 2e-3) < 1e-12
    assert abs(options.dt - 10e-6) < 1e-12

    dc = ps.dc_operating_point(circuit)
    assert dc.success
    # Divider 2k/2k with Vin=12V -> V(out)=6V
    assert abs(dc.newton_result.solution[1] - 6.0) < 1e-6


def test_yaml_parser_maps_thermal_fields_and_rejects_json() -> None:
    content = """
schema: pulsim-v1
version: 1
simulation:
  tstart: 0
  tstop: 1e-4
  dt: 1e-6
  thermal:
    enabled: true
    ambient: 30
    policy: loss_with_temperature_scaling
    default_rth: 1.2
    default_cth: 0.4
components:
  - type: voltage_source
    name: Vg
    nodes: [gate, 0]
    waveform: {type: dc, value: 10}
  - type: voltage_source
    name: Vd
    nodes: [drain, 0]
    waveform: {type: dc, value: 12}
  - type: mosfet
    name: M1
    nodes: [gate, drain, source]
    thermal:
      rth: 0.8
      cth: 0.1
      temp_init: 35
      temp_ref: 25
      alpha: 0.006
  - type: resistor
    name: Rload
    nodes: [source, 0]
    value: 5
"""
    parser = ps.YamlParser()
    _, options = parser.load_string(content)
    assert parser.errors == []
    assert options.thermal.enable
    assert abs(options.thermal.ambient - 30.0) < 1e-12
    assert options.thermal.policy == ps.ThermalCouplingPolicy.LossWithTemperatureScaling
    assert "M1" in options.thermal_devices
    assert abs(options.thermal_devices["M1"].rth - 0.8) < 1e-12

    parser_json = ps.YamlParser()
    parser_json.load_string('{"schema":"pulsim-v1","version":1,"components":[]}')
    assert any("JSON netlists are unsupported" in msg for msg in parser_json.errors)


def test_electro_thermal_coupling_emits_thermal_telemetry() -> None:
    circuit = ps.Circuit()
    n_gate = circuit.add_node("gate")
    n_drain = circuit.add_node("drain")
    n_source = circuit.add_node("source")
    gnd = circuit.ground()

    circuit.add_voltage_source("Vg", n_gate, gnd, 10.0)
    circuit.add_voltage_source("Vd", n_drain, gnd, 20.0)
    mparams = ps.MOSFETParams()
    mparams.vth = 2.5
    mparams.kp = 0.01
    mparams.lambda_ = 0.0
    mparams.g_off = 1e-8
    circuit.add_mosfet("M1", n_gate, n_drain, n_source, mparams)
    circuit.add_resistor("Rload", n_source, gnd, 100.0)

    opts = ps.SimulationOptions()
    opts.tstart = 0.0
    opts.tstop = 1e-4
    opts.dt = 1e-6
    opts.dt_min = opts.dt
    opts.dt_max = opts.dt
    opts.adaptive_timestep = False
    opts.enable_bdf_order_control = False
    opts.enable_losses = True
    opts.thermal.enable = True
    opts.thermal.ambient = 25.0
    opts.thermal.policy = ps.ThermalCouplingPolicy.LossWithTemperatureScaling
    tcfg = ps.ThermalDeviceConfig()
    tcfg.rth = 0.5
    tcfg.cth = 1e-4
    tcfg.temp_init = 25.0
    tcfg.temp_ref = 25.0
    tcfg.alpha = 0.004
    opts.thermal_devices = {"M1": tcfg}

    sim = ps.Simulator(circuit, opts)
    result = sim.run_transient()
    assert result.success
    assert result.thermal_summary.enabled
    assert result.thermal_summary.max_temperature >= result.thermal_summary.ambient
    assert any(item.device_name == "M1" for item in result.thermal_summary.device_temperatures)
