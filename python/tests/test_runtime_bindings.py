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


def test_yaml_parser_maps_fallback_controls() -> None:
    content = """
schema: pulsim-v1
version: 1
simulation:
  tstop: 1e-4
  dt: 1e-6
  max_step_retries: 4
  fallback:
    trace_retries: true
    enable_transient_gmin: true
    gmin_retry_threshold: 2
    gmin_initial: 1e-8
    gmin_max: 1e-4
    gmin_growth: 5
components:
  - type: resistor
    name: R1
    nodes: [in, 0]
    value: 1k
"""
    parser = ps.YamlParser()
    _, options = parser.load_string(content)
    assert parser.errors == []
    assert options.max_step_retries == 4
    assert options.fallback_policy.trace_retries
    assert options.fallback_policy.enable_transient_gmin
    assert options.fallback_policy.gmin_retry_threshold == 2
    assert abs(options.fallback_policy.gmin_initial - 1e-8) < 1e-16
    assert abs(options.fallback_policy.gmin_max - 1e-4) < 1e-12
    assert abs(options.fallback_policy.gmin_growth - 5.0) < 1e-12


def test_yaml_parser_gui_parity_slice_registers_virtual_components() -> None:
    content = """
schema: pulsim-v1
version: 1
components:
  - type: voltage_source
    name: V1
    nodes: [in, 0]
    waveform: {type: dc, value: 12}
  - type: BJT_PNP
    name: QP1
    nodes: [ctrl, in, out]
    beta: 80
  - type: THYRISTOR
    name: SCR1
    nodes: [gate, out, 0]
  - type: FUSE
    name: F1
    nodes: [out, load]
    initial_state: true
  - type: RELAY
    name: K1
    nodes: [coil_p, coil_n, com, no, nc]
  - type: OP_AMP
    name: A1
    nodes: [ref, fb, ctrl]
  - type: VOLTAGE_PROBE
    name: VP1
    nodes: [out, 0]
"""
    parser = ps.YamlParser()
    circuit, _ = parser.load_string(content)
    assert parser.errors == []
    assert circuit.num_devices() >= 4
    assert circuit.num_virtual_components() >= 3
    assert any("PULSIM_YAML_W_COMPONENT_SURROGATE" in msg for msg in parser.warnings)
    assert any("PULSIM_YAML_W_COMPONENT_VIRTUAL" in msg for msg in parser.warnings)


def test_virtual_probe_evaluation_from_state_vector() -> None:
    circuit = ps.Circuit()
    n_in = circuit.add_node("in")
    gnd = circuit.ground()
    circuit.add_voltage_source("V1", n_in, gnd, 5.0)
    circuit.add_virtual_component("voltage_probe", "VP", [n_in, gnd], {}, {})
    circuit.add_virtual_component("current_probe", "IP", [n_in, gnd], {}, {"target_component": "V1"})
    circuit.add_virtual_component("power_probe", "PP", [n_in, gnd], {}, {"target_component": "V1"})

    x = [0.0] * circuit.system_size()
    x[n_in] = 8.0
    x[circuit.num_nodes()] = 0.5

    values = circuit.evaluate_virtual_signals(x)
    assert values["VP"] == 8.0
    assert values["IP"] == 0.5
    assert values["PP"] == 4.0


def test_mixed_domain_scheduler_is_deterministic() -> None:
    circuit = ps.Circuit()
    n_ref = circuit.add_node("ref")
    n_fb = circuit.add_node("fb")
    n_out = circuit.add_node("out")
    gnd = circuit.ground()
    circuit.add_voltage_source("V1", n_ref, gnd, 3.0)
    circuit.add_resistor("Rfb", n_ref, n_fb, 10.0)
    circuit.add_virtual_component("pi_controller", "PI1", [n_ref, n_fb, n_out], {"gain": 2.0}, {})

    x = [0.0] * circuit.system_size()
    x[n_ref] = 3.0
    x[n_fb] = 1.0
    step = circuit.execute_mixed_domain_step(x, 1e-6)

    assert step.phase_order == ["electrical", "control", "events", "instrumentation"]
    assert "PI1" in step.channel_values


def test_simulation_result_includes_virtual_channels() -> None:
    circuit = ps.Circuit()
    n_in = circuit.add_node("in")
    gnd = circuit.ground()
    circuit.add_voltage_source("V1", n_in, gnd, 5.0)
    circuit.add_resistor("R1", n_in, gnd, 100.0)
    circuit.add_virtual_component("voltage_probe", "VP", [n_in, gnd], {}, {})

    opts = ps.SimulationOptions()
    opts.tstart = 0.0
    opts.tstop = 5e-6
    opts.dt = 1e-6
    opts.dt_min = opts.dt
    opts.dt_max = opts.dt
    opts.adaptive_timestep = False
    opts.enable_bdf_order_control = False

    sim = ps.Simulator(circuit, opts)
    result = sim.run_transient()
    assert result.success
    assert result.mixed_domain_phase_order == ["electrical", "control", "events", "instrumentation"]
    assert "VP" in result.virtual_channels
    assert len(result.virtual_channels["VP"]) == len(result.time)
    assert "VP" in result.virtual_channel_metadata
    assert result.virtual_channel_metadata["VP"].component_type == "voltage_probe"
    assert result.virtual_channel_metadata["VP"].domain == "instrumentation"


def test_virtual_channel_metadata_includes_relay_state_channel() -> None:
    circuit = ps.Circuit()
    n_coil_p = circuit.add_node("coil_p")
    n_coil_n = circuit.add_node("coil_n")
    n_com = circuit.add_node("com")
    n_no = circuit.add_node("no")
    n_nc = circuit.add_node("nc")
    gnd = circuit.ground()
    circuit.add_voltage_source("V1", n_coil_p, gnd, 12.0)
    circuit.add_virtual_component(
        "relay",
        "K1",
        [n_coil_p, n_coil_n, n_com, n_no, n_nc],
        {"pickup_voltage": 6.0, "dropout_voltage": 3.0},
        {},
    )

    metadata = circuit.virtual_channel_metadata()
    assert "K1" in metadata
    assert metadata["K1"].component_type == "relay"
    assert metadata["K1.state"].domain == "events"


def test_scope_channels_emit_waveforms_with_metadata() -> None:
    circuit = ps.Circuit()
    n_in = circuit.add_node("in")
    gnd = circuit.ground()
    circuit.add_voltage_source("V1", n_in, gnd, 10.0)
    circuit.add_resistor("R1", n_in, gnd, 1_000.0)
    circuit.add_virtual_component("electrical_scope", "SCOPE_E", [n_in, gnd], {}, {})
    circuit.add_virtual_component("thermal_scope", "SCOPE_T", [n_in], {}, {})

    opts = ps.SimulationOptions()
    opts.tstart = 0.0
    opts.tstop = 3e-6
    opts.dt = 1e-6
    opts.dt_min = opts.dt
    opts.dt_max = opts.dt
    opts.adaptive_timestep = False
    opts.enable_bdf_order_control = False

    result = ps.Simulator(circuit, opts).run_transient()
    assert result.success
    assert "SCOPE_E" in result.virtual_channels
    assert "SCOPE_T" in result.virtual_channels
    assert len(result.virtual_channels["SCOPE_E"]) == len(result.time)
    assert result.virtual_channel_metadata["SCOPE_E"].domain == "instrumentation"
    assert result.virtual_channel_metadata["SCOPE_T"].domain == "thermal"


def test_fallback_trace_records_retry_reasons() -> None:
    circuit = ps.Circuit()
    n_in = circuit.add_node("in")
    n_out = circuit.add_node("out")
    gnd = circuit.ground()
    circuit.add_voltage_source("V1", n_in, gnd, 5.0)
    circuit.add_resistor("R1", n_in, n_out, 1_000.0)
    circuit.add_capacitor("C1", n_out, gnd, 1e-6, 0.0)

    opts = ps.SimulationOptions()
    opts.tstart = 0.0
    opts.tstop = 1e-4
    opts.dt = 1e-6
    opts.dt_min = opts.dt
    opts.dt_max = opts.dt
    opts.adaptive_timestep = False
    opts.enable_bdf_order_control = False
    opts.max_step_retries = 3
    opts.linear_solver.order = [ps.LinearSolverKind.CG]
    opts.linear_solver.allow_fallback = False
    opts.linear_solver.auto_select = False
    opts.fallback_policy.trace_retries = True
    opts.fallback_policy.enable_transient_gmin = True
    opts.fallback_policy.gmin_retry_threshold = 1
    opts.fallback_policy.gmin_initial = 1e-8
    opts.fallback_policy.gmin_max = 1e-4

    sim = ps.Simulator(circuit, opts)
    x0 = [0.0] * (circuit.num_nodes() + circuit.num_branches())
    result = sim.run_transient(x0)
    assert not result.success
    assert len(result.fallback_trace) > 0
    reasons = {entry.reason for entry in result.fallback_trace}
    assert ps.FallbackReasonCode.NewtonFailure in reasons
    assert ps.FallbackReasonCode.TransientGminEscalation in reasons
    assert ps.FallbackReasonCode.MaxRetriesExceeded in reasons
