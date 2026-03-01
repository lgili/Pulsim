"""Tests for runtime-complete simulation bindings (SimulationOptions/Simulator/YamlParser)."""

from __future__ import annotations

from pathlib import Path

import pulsim as ps
import pytest


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


def test_procedural_run_transient_remains_compatible_with_canonical_simulator_surface() -> None:
    circuit = _build_rc_circuit()

    opts = ps.SimulationOptions()
    opts.tstart = 0.0
    opts.tstop = 2e-4
    opts.dt = 1e-6
    opts.dt_min = opts.dt
    opts.dt_max = opts.dt
    opts.adaptive_timestep = False
    opts.newton_options.num_nodes = int(circuit.num_nodes())
    opts.newton_options.num_branches = int(circuit.num_branches())

    x0 = circuit.initial_state()
    canonical = ps.Simulator(circuit, opts).run_transient(x0)
    proc_time, proc_states, proc_ok, _ = ps.run_transient(
        circuit,
        opts.tstart,
        opts.tstop,
        opts.dt,
        x0,
        opts.newton_options,
        opts.linear_solver,
        robust=False,
        auto_regularize=False,
    )

    assert canonical.success is True
    assert proc_ok is True
    assert len(proc_time) > 2
    assert len(canonical.time) > 2
    assert abs(float(proc_time[-1]) - opts.tstop) < 1e-12
    assert abs(float(canonical.time[-1]) - opts.tstop) < 1e-12
    assert abs(float(proc_states[-1][1]) - float(canonical.states[-1][1])) < 5e-3


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

    assert any("PULSIM_YAML_E_UNKNOWN_FIELD" in msg for msg in parser.errors)
    assert any("simulation.unknown_field" in msg for msg in parser.errors)


def test_yaml_parser_reports_type_mismatch_with_expected_and_received_classes() -> None:
    content = """
schema: pulsim-v1
version: 1
simulation:
  dt: {invalid: true}
components:
  - type: resistor
    name: R1
    nodes: [in, 0]
    value: 1k
"""
    parser = ps.YamlParser()
    parser.load_string(content)

    assert any("PULSIM_YAML_E_TYPE_MISMATCH" in msg for msg in parser.errors)
    assert any("simulation.dt" in msg for msg in parser.errors)
    assert any("expected number" in msg and "got map" in msg for msg in parser.errors)


def test_yaml_parser_warns_deprecated_adaptive_timestep_alias_in_non_strict_mode() -> None:
    content = """
schema: pulsim-v1
version: 1
simulation:
  adaptive_timestep: true
  tstop: 1e-4
  dt: 1e-6
components:
  - type: resistor
    name: R1
    nodes: [in, 0]
    value: 1k
"""
    options = ps.YamlParserOptions()
    options.strict = False
    parser = ps.YamlParser(options)
    _, parsed_options = parser.load_string(content)

    assert parser.errors == []
    assert parsed_options.adaptive_timestep is True
    assert any("PULSIM_YAML_W_DEPRECATED_FIELD" in msg for msg in parser.warnings)
    assert any("simulation.adaptive_timestep" in msg for msg in parser.warnings)
    assert any("simulation.step_mode" in msg for msg in parser.warnings)


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


def test_yaml_parser_emits_migration_warnings_for_legacy_backend_controls() -> None:
    content = """
schema: pulsim-v1
version: 1
simulation:
  tstop: 1e-4
  dt: 1e-6
  backend: auto
  sundials:
    enabled: true
  fallback:
    enable_backend_escalation: true
    backend_escalation_threshold: 3
    enable_native_reentry: true
    sundials_recovery_window: 5e-6
components:
  - type: resistor
    name: R1
    nodes: [in, 0]
    value: 1k
"""
    parser_opts = ps.YamlParserOptions()
    parser_opts.strict = False
    parser = ps.YamlParser(parser_opts)
    _, options = parser.load_string(content)
    assert parser.errors == []
    assert options.step_mode == ps.StepMode.Variable
    assert parser.warnings
    joined = "\n".join(parser.warnings)
    assert "simulation.backend" in joined
    assert "simulation.sundials" in joined
    assert "simulation.fallback.enable_backend_escalation" in joined
    assert "simulation.fallback.backend_escalation_threshold" in joined
    assert "simulation.fallback.enable_native_reentry" in joined
    assert "simulation.fallback.sundials_recovery_window" in joined


def test_yaml_parser_maps_canonical_step_mode_and_advanced_overrides() -> None:
    content = """
schema: pulsim-v1
version: 1
simulation:
  tstop: 2e-5
  dt: 1e-6
  step_mode: fixed
  advanced:
    solver:
      order: [klu]
components:
  - type: resistor
    name: R1
    nodes: [in, 0]
    value: 1k
"""
    parser = ps.YamlParser()
    _, options = parser.load_string(content)
    assert parser.errors == []
    assert options.step_mode == ps.StepMode.Fixed
    assert options.adaptive_timestep is False
    assert options.linear_solver.order == [ps.LinearSolverKind.KLU]
    assert options.integrator == ps.Integrator.TRBDF2

    options.step_mode = ps.StepMode.Variable
    assert options.step_mode == ps.StepMode.Variable
    assert options.adaptive_timestep is True

    options.adaptive_timestep = False
    assert options.step_mode == ps.StepMode.Fixed
    assert options.adaptive_timestep is False


def test_yaml_parser_reports_migration_error_for_legacy_backend_keys_in_strict_mode() -> None:
    content = """
schema: pulsim-v1
version: 1
simulation:
  tstop: 1e-4
  dt: 1e-6
  backend: auto
  advanced:
    backend: sundials
    sundials:
      enabled: true
    fallback:
      enable_backend_escalation: true
components:
  - type: resistor
    name: R1
    nodes: [in, 0]
    value: 1k
"""
    parser = ps.YamlParser()
    parser.load_string(content)
    assert parser.errors
    assert any("simulation.backend" in msg for msg in parser.errors)
    assert any("simulation.advanced.backend" in msg for msg in parser.errors)
    assert any("simulation.advanced.sundials" in msg for msg in parser.errors)
    assert any("simulation.advanced.fallback.enable_backend_escalation" in msg for msg in parser.errors)
    assert any("simulation.step_mode" in msg for msg in parser.errors)


def test_backend_telemetry_binding_fields() -> None:
    telemetry = ps.BackendTelemetry()
    telemetry.formulation_mode = "native"
    telemetry.function_evaluations = 10
    assert telemetry.formulation_mode == "native"
    assert telemetry.function_evaluations == 10


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


def test_yaml_parser_registers_fuse_event_controller() -> None:
    content = """
schema: pulsim-v1
version: 1
simulation:
  tstart: 0
  tstop: 1e-5
  dt: 1e-6
components:
  - type: voltage_source
    name: V1
    nodes: [in, 0]
    waveform: {type: dc, value: 10}
  - type: fuse
    name: F1
    nodes: [in, out]
    rating: 5
    blow_i2t: 2
  - type: resistor
    name: R1
    nodes: [out, 0]
    value: 10
"""
    parser = ps.YamlParser()
    circuit, _ = parser.load_string(content)
    assert parser.errors == []
    assert "F1" in circuit.virtual_component_names()
    metadata = circuit.virtual_channel_metadata()
    assert "F1.state" in metadata
    assert metadata["F1.state"].domain == "events"


def test_yaml_parser_registers_relay_dual_contact_controller() -> None:
    content = """
schema: pulsim-v1
version: 1
components:
  - type: voltage_source
    name: Vcoil
    nodes: [coil_p, coil_n]
    waveform: {type: dc, value: 12}
  - type: relay
    name: K1
    nodes: [coil_p, coil_n, com, no, nc]
    pickup_voltage: 8
    dropout_voltage: 3
    contact_resistance: 0.05
    off_resistance: 1e9
"""
    parser = ps.YamlParser()
    circuit, _ = parser.load_string(content)
    assert parser.errors == []
    assert "K1" in circuit.virtual_component_names()
    # Relay is implemented as two electrical switches + one event controller.
    assert circuit.num_devices() >= 3
    metadata = circuit.virtual_channel_metadata()
    assert "K1.state" in metadata
    assert "K1.no_state" in metadata
    assert "K1.nc_state" in metadata
    assert metadata["K1.no_state"].domain == "events"


def test_yaml_parser_registers_thyristor_event_controller() -> None:
    content = """
schema: pulsim-v1
version: 1
components:
  - type: voltage_source
    name: Vg
    nodes: [gate, 0]
    waveform: {type: dc, value: 2}
  - type: thyristor
    name: SCR1
    nodes: [gate, anode, 0]
    gate_threshold: 1.0
    holding_current: 0.1
    latch_current: 0.2
"""
    parser = ps.YamlParser()
    circuit, _ = parser.load_string(content)
    assert parser.errors == []
    assert "SCR1" in circuit.virtual_component_names()
    metadata = circuit.virtual_channel_metadata()
    assert "SCR1.state" in metadata
    assert metadata["SCR1.state"].domain == "events"
    assert any("PULSIM_YAML_W_COMPONENT_SURROGATE" in msg for msg in parser.warnings)


def test_yaml_parser_registers_saturable_inductor_controller() -> None:
    content = """
schema: pulsim-v1
version: 1
components:
  - type: saturable_inductor
    name: Lsat
    nodes: [in, 0]
    inductance: 1m
    saturation_current: 2
    saturation_inductance: 100u
"""
    parser = ps.YamlParser()
    circuit, _ = parser.load_string(content)
    assert parser.errors == []
    assert circuit.num_devices() == 1
    assert "Lsat" in circuit.virtual_component_names()
    metadata = circuit.virtual_channel_metadata()
    assert "Lsat.l_eff" in metadata
    assert metadata["Lsat.l_eff"].domain == "electrical"


def test_yaml_parser_registers_coupled_inductor_pair_controller() -> None:
    content = """
schema: pulsim-v1
version: 1
components:
  - type: coupled_inductor
    name: K1
    nodes: [p1, p2, s1, s2]
    l1: 1m
    l2: 4m
    coupling: 0.95
"""
    parser = ps.YamlParser()
    circuit, _ = parser.load_string(content)
    assert parser.errors == []
    # Coupled inductor expands to two inductor branches plus one virtual controller.
    assert circuit.num_devices() == 2
    assert "K1" in circuit.virtual_component_names()
    metadata = circuit.virtual_channel_metadata()
    assert "K1.mutual" in metadata
    assert metadata["K1.mutual"].domain == "electrical"


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
    assert metadata["K1.no_state"].domain == "events"
    assert metadata["K1.nc_state"].domain == "events"


def test_op_amp_saturates_to_output_rails() -> None:
    circuit = ps.Circuit()
    n_pos = circuit.add_node("v_plus")
    n_neg = circuit.add_node("v_minus")
    n_out = circuit.add_node("v_out")
    circuit.add_virtual_component(
        "op_amp",
        "A1",
        [n_pos, n_neg, n_out],
        {"open_loop_gain": 1e5, "rail_low": -2.0, "rail_high": 2.0},
        {},
    )

    x = [0.0] * circuit.system_size()
    x[n_pos] = 1.0
    high = circuit.execute_mixed_domain_step(x, 1e-6)
    assert high.channel_values["A1"] == 2.0

    x[n_pos] = -1.0
    low = circuit.execute_mixed_domain_step(x, 2e-6)
    assert low.channel_values["A1"] == -2.0


def test_comparator_hysteresis_is_stateful() -> None:
    circuit = ps.Circuit()
    n_in = circuit.add_node("in")
    n_ref = circuit.add_node("ref")
    circuit.add_virtual_component(
        "comparator",
        "CMP1",
        [n_in, n_ref],
        {"threshold": 0.0, "hysteresis": 2.0, "high": 5.0, "low": 0.0},
        {},
    )

    x = [0.0] * circuit.system_size()
    x[n_in] = 0.8
    below_upper = circuit.execute_mixed_domain_step(x, 1e-6)
    assert below_upper.channel_values["CMP1"] == 0.0

    x[n_in] = 1.2
    above_upper = circuit.execute_mixed_domain_step(x, 2e-6)
    assert above_upper.channel_values["CMP1"] == 5.0

    x[n_in] = 0.4
    hold_high = circuit.execute_mixed_domain_step(x, 3e-6)
    assert hold_high.channel_values["CMP1"] == 5.0

    x[n_in] = -1.2
    below_lower = circuit.execute_mixed_domain_step(x, 4e-6)
    assert below_lower.channel_values["CMP1"] == 0.0


def test_pi_controller_anti_windup_recovers_faster_than_unclamped_integrator() -> None:
    circuit = ps.Circuit()
    n_err = circuit.add_node("err")
    n_ref = circuit.add_node("ref")
    circuit.add_virtual_component(
        "pi_controller",
        "PI_AW",
        [n_err, n_ref],
        {
            "kp": 0.0,
            "ki": 1.0,
            "output_min": -1.0,
            "output_max": 1.0,
            "anti_windup": 1.0,
        },
        {},
    )
    circuit.add_virtual_component(
        "pi_controller",
        "PI_NO_AW",
        [n_err, n_ref],
        {
            "kp": 0.0,
            "ki": 1.0,
            "output_min": -1.0,
            "output_max": 1.0,
            "anti_windup": 0.0,
        },
        {},
    )

    x = [0.0] * circuit.system_size()
    x[n_err] = 2.0
    circuit.execute_mixed_domain_step(x, 0.0)
    circuit.execute_mixed_domain_step(x, 1.0)
    circuit.execute_mixed_domain_step(x, 2.0)

    x[n_err] = -2.0
    recovery = circuit.execute_mixed_domain_step(x, 3.0)
    assert recovery.channel_values["PI_AW"] <= 0.0
    assert recovery.channel_values["PI_NO_AW"] > 0.0


def test_pid_controller_uses_error_derivative() -> None:
    circuit = ps.Circuit()
    n_err = circuit.add_node("err")
    n_ref = circuit.add_node("ref")
    circuit.add_virtual_component(
        "pid_controller",
        "PID1",
        [n_err, n_ref],
        {"kp": 0.0, "ki": 0.0, "kd": 1.0},
        {},
    )

    x = [0.0] * circuit.system_size()
    x[n_err] = 0.0
    first = circuit.execute_mixed_domain_step(x, 0.0)
    assert first.channel_values["PID1"] == 0.0

    x[n_err] = 1.0
    second = circuit.execute_mixed_domain_step(x, 1.0)
    assert abs(second.channel_values["PID1"] - 1.0) < 1e-12

    x[n_err] = 3.0
    third = circuit.execute_mixed_domain_step(x, 2.0)
    assert abs(third.channel_values["PID1"] - 2.0) < 1e-12


def test_math_block_operations_and_divide_by_zero_behavior() -> None:
    circuit = ps.Circuit()
    n_a = circuit.add_node("a")
    n_b = circuit.add_node("b")
    circuit.add_virtual_component("math_block", "M_ADD", [n_a, n_b], {}, {"operation": "add"})
    circuit.add_virtual_component("math_block", "M_SUB", [n_a, n_b], {}, {"operation": "sub"})
    circuit.add_virtual_component("math_block", "M_MUL", [n_a, n_b], {}, {"operation": "mul"})
    circuit.add_virtual_component("math_block", "M_DIV", [n_a, n_b], {}, {"operation": "div"})

    x = [0.0] * circuit.system_size()
    x[n_a] = 6.0
    x[n_b] = 2.0
    step = circuit.execute_mixed_domain_step(x, 1e-6)
    assert step.channel_values["M_ADD"] == 8.0
    assert step.channel_values["M_SUB"] == 4.0
    assert step.channel_values["M_MUL"] == 12.0
    assert step.channel_values["M_DIV"] == 3.0

    x[n_b] = 0.0
    div_zero = circuit.execute_mixed_domain_step(x, 2e-6)
    assert div_zero.channel_values["M_DIV"] == 0.0


def test_pwm_generator_supports_input_driven_duty_with_limits() -> None:
    circuit = ps.Circuit()
    n_ctrl = circuit.add_node("ctrl")
    circuit.add_virtual_component(
        "pwm_generator",
        "PWM1",
        [n_ctrl],
        {
            "frequency": 1.0,
            "duty_from_input": 1.0,
            "duty_gain": 0.5,
            "duty_offset": 0.1,
            "duty_min": 0.2,
            "duty_max": 0.8,
        },
        {},
    )

    metadata = circuit.virtual_channel_metadata()
    assert "PWM1.duty" in metadata
    assert metadata["PWM1.duty"].domain == "control"

    x = [0.0] * circuit.system_size()
    x[n_ctrl] = 2.0
    high_duty = circuit.execute_mixed_domain_step(x, 0.25)
    assert abs(high_duty.channel_values["PWM1.duty"] - 0.8) < 1e-12
    assert high_duty.channel_values["PWM1"] == 1.0

    x[n_ctrl] = -2.0
    low_duty = circuit.execute_mixed_domain_step(x, 0.75)
    assert abs(low_duty.channel_values["PWM1.duty"] - 0.2) < 1e-12
    assert low_duty.channel_values["PWM1"] == 0.0


def test_yaml_pulse_voltage_source_can_drive_target_switch_component() -> None:
    content = """
schema: pulsim-v1
version: 1
components:
  - type: voltage_source
    name: VDRV
    nodes: [drv, 0]
    target_component: Q1
    waveform:
      type: pulse
      v_initial: 0
      v_pulse: 10
      t_delay: 0
      t_rise: 1e-9
      t_fall: 1e-9
      t_width: 2e-6
      period: 4e-6
  - type: igbt
    name: Q1
    nodes: [gate, main, 0]
    vth: 5
    g_on: 2000
    g_off: 1e-9
"""
    parser = ps.YamlParser()
    circuit, _ = parser.load_string(content)
    assert parser.errors == []

    n_main = circuit.get_node("main")
    x = [0.0] * circuit.system_size()
    x[n_main] = 1.0

    circuit.execute_mixed_domain_step(x, 1e-6)  # high interval
    _, f_on = circuit.assemble_jacobian(x)
    g_on = abs(float(f_on[n_main])) / max(abs(float(x[n_main])), 1e-9)

    circuit.execute_mixed_domain_step(x, 3e-6)  # low interval
    _, f_off = circuit.assemble_jacobian(x)
    g_off = abs(float(f_off[n_main])) / max(abs(float(x[n_main])), 1e-9)

    assert g_on > g_off * 10.0


def test_integrator_respects_output_limits() -> None:
    circuit = ps.Circuit()
    n_in = circuit.add_node("in")
    n_ref = circuit.add_node("ref")
    circuit.add_virtual_component(
        "integrator",
        "INT1",
        [n_in, n_ref],
        {"output_min": -1.0, "output_max": 1.0},
        {},
    )

    x = [0.0] * circuit.system_size()
    x[n_in] = 2.0
    circuit.execute_mixed_domain_step(x, 0.0)
    step1 = circuit.execute_mixed_domain_step(x, 1.0)
    step2 = circuit.execute_mixed_domain_step(x, 2.0)
    assert step1.channel_values["INT1"] == 1.0
    assert step2.channel_values["INT1"] == 1.0


def test_differentiator_alpha_filter_smooths_output() -> None:
    circuit = ps.Circuit()
    n_in = circuit.add_node("in")
    n_ref = circuit.add_node("ref")
    circuit.add_virtual_component(
        "differentiator",
        "D1",
        [n_in, n_ref],
        {"alpha": 0.5},
        {},
    )

    x = [0.0] * circuit.system_size()
    x[n_in] = 0.0
    circuit.execute_mixed_domain_step(x, 0.0)

    x[n_in] = 10.0
    step_up = circuit.execute_mixed_domain_step(x, 1.0)
    assert abs(step_up.channel_values["D1"] - 5.0) < 1e-12

    hold = circuit.execute_mixed_domain_step(x, 2.0)
    assert abs(hold.channel_values["D1"] - 2.5) < 1e-12


def test_rate_limiter_caps_rising_and_falling_edges() -> None:
    circuit = ps.Circuit()
    n_in = circuit.add_node("in")
    n_ref = circuit.add_node("ref")
    circuit.add_virtual_component(
        "rate_limiter",
        "RL1",
        [n_in, n_ref],
        {"rising_rate": 1.0, "falling_rate": 2.0},
        {},
    )

    x = [0.0] * circuit.system_size()
    circuit.execute_mixed_domain_step(x, 0.0)

    x[n_in] = 10.0
    rise = circuit.execute_mixed_domain_step(x, 1.0)
    assert rise.channel_values["RL1"] == 1.0

    x[n_in] = -10.0
    fall = circuit.execute_mixed_domain_step(x, 2.0)
    assert fall.channel_values["RL1"] == -1.0


def test_hysteresis_block_uses_high_low_states() -> None:
    circuit = ps.Circuit()
    n_in = circuit.add_node("in")
    n_ref = circuit.add_node("ref")
    circuit.add_virtual_component(
        "hysteresis",
        "HYS1",
        [n_in, n_ref],
        {"threshold": 0.0, "hysteresis": 1.0, "high": 3.0, "low": -1.0},
        {},
    )

    x = [0.0] * circuit.system_size()
    x[n_in] = 0.3
    below_upper = circuit.execute_mixed_domain_step(x, 1e-6)
    assert below_upper.channel_values["HYS1"] == -1.0

    x[n_in] = 0.6
    above_upper = circuit.execute_mixed_domain_step(x, 2e-6)
    assert above_upper.channel_values["HYS1"] == 3.0

    x[n_in] = -0.6
    below_lower = circuit.execute_mixed_domain_step(x, 3e-6)
    assert below_lower.channel_values["HYS1"] == -1.0


def test_lookup_table_performs_linear_interpolation() -> None:
    circuit = ps.Circuit()
    n_in = circuit.add_node("in")
    n_ref = circuit.add_node("ref")
    circuit.add_virtual_component(
        "lookup_table",
        "LUT1",
        [n_in, n_ref],
        {},
        {"x": "[0, 1, 2]", "y": "[0, 10, 20]"},
    )

    x = [0.0] * circuit.system_size()
    x[n_in] = 0.5
    mid = circuit.execute_mixed_domain_step(x, 1e-6)
    assert abs(mid.channel_values["LUT1"] - 5.0) < 1e-12

    x[n_in] = 1.5
    high = circuit.execute_mixed_domain_step(x, 2e-6)
    assert abs(high.channel_values["LUT1"] - 15.0) < 1e-12


def test_transfer_function_supports_num_den_coefficients() -> None:
    circuit = ps.Circuit()
    n_in = circuit.add_node("in")
    n_ref = circuit.add_node("ref")
    circuit.add_virtual_component(
        "transfer_function",
        "TF1",
        [n_in, n_ref],
        {},
        {"num": "[0.5, 0.5]", "den": "[1.0, -0.5]"},
    )

    x = [0.0] * circuit.system_size()
    x[n_in] = 1.0
    first = circuit.execute_mixed_domain_step(x, 0.0)
    assert abs(first.channel_values["TF1"] - 0.5) < 1e-12

    second = circuit.execute_mixed_domain_step(x, 1.0)
    assert abs(second.channel_values["TF1"] - 1.25) < 1e-12


def test_delay_block_outputs_time_shifted_signal() -> None:
    circuit = ps.Circuit()
    n_in = circuit.add_node("in")
    n_ref = circuit.add_node("ref")
    circuit.add_virtual_component(
        "delay_block",
        "DL1",
        [n_in, n_ref],
        {"delay": 1.0},
        {},
    )

    x = [0.0] * circuit.system_size()
    x[n_in] = 1.0
    t0 = circuit.execute_mixed_domain_step(x, 0.0)
    assert t0.channel_values["DL1"] == 1.0

    x[n_in] = 3.0
    t05 = circuit.execute_mixed_domain_step(x, 0.5)
    assert t05.channel_values["DL1"] == 1.0

    x[n_in] = 5.0
    t15 = circuit.execute_mixed_domain_step(x, 1.5)
    assert abs(t15.channel_values["DL1"] - 3.0) < 1e-12


def test_sample_hold_updates_only_at_sample_period() -> None:
    circuit = ps.Circuit()
    n_in = circuit.add_node("in")
    n_ref = circuit.add_node("ref")
    circuit.add_virtual_component(
        "sample_hold",
        "SH1",
        [n_in, n_ref],
        {"sample_period": 1.0},
        {},
    )

    x = [0.0] * circuit.system_size()
    x[n_in] = 0.0
    t0 = circuit.execute_mixed_domain_step(x, 0.0)
    assert t0.channel_values["SH1"] == 0.0

    x[n_in] = 5.0
    t04 = circuit.execute_mixed_domain_step(x, 0.4)
    assert t04.channel_values["SH1"] == 0.0

    t11 = circuit.execute_mixed_domain_step(x, 1.1)
    assert t11.channel_values["SH1"] == 5.0

    x[n_in] = -2.0
    t16 = circuit.execute_mixed_domain_step(x, 1.6)
    assert t16.channel_values["SH1"] == 5.0


def test_state_machine_supports_set_reset_mode() -> None:
    circuit = ps.Circuit()
    n_set = circuit.add_node("set")
    n_reset = circuit.add_node("reset")
    circuit.add_virtual_component(
        "state_machine",
        "SM1",
        [n_set, n_reset],
        {"threshold": 0.5, "high": 10.0, "low": -10.0},
        {"mode": "set_reset"},
    )

    x = [0.0] * circuit.system_size()
    base = circuit.execute_mixed_domain_step(x, 0.0)
    assert base.channel_values["SM1"] == -10.0

    x[n_set] = 1.0
    set_state = circuit.execute_mixed_domain_step(x, 1.0)
    assert set_state.channel_values["SM1"] == 10.0

    x[n_set] = 0.0
    x[n_reset] = 1.0
    reset_state = circuit.execute_mixed_domain_step(x, 2.0)
    assert reset_state.channel_values["SM1"] == -10.0


def test_relay_event_controller_drives_no_and_nc_switches() -> None:
    circuit = ps.Circuit()
    n_coil_p = circuit.add_node("coil_p")
    n_coil_n = circuit.add_node("coil_n")
    n_com = circuit.add_node("com")
    n_no = circuit.add_node("no")
    n_nc = circuit.add_node("nc")

    circuit.add_switch("K1__no", n_com, n_no, False, 10.0, 1e-6)
    circuit.add_switch("K1__nc", n_com, n_nc, True, 10.0, 1e-6)
    circuit.add_virtual_component(
        "relay",
        "K1",
        [n_coil_p, n_coil_n, n_com, n_no, n_nc],
        {"pickup_voltage": 6.0, "dropout_voltage": 3.0, "initial_closed": 0.0},
        {"target_component_no": "K1__no", "target_component_nc": "K1__nc"},
    )

    x = [0.0] * circuit.system_size()
    x[n_coil_p] = 12.0
    x[n_coil_n] = 0.0
    on_step = circuit.execute_mixed_domain_step(x, 1e-6)
    assert on_step.channel_values["K1.state"] == 1.0
    assert on_step.channel_values["K1.no_state"] == 1.0
    assert on_step.channel_values["K1.nc_state"] == 0.0
    g_on, _ = circuit.assemble_dc()
    assert g_on[n_com][n_no] < -1.0
    assert g_on[n_com][n_nc] > -1e-3

    x[n_coil_p] = 0.0
    off_step = circuit.execute_mixed_domain_step(x, 2e-6)
    assert off_step.channel_values["K1.state"] == 0.0
    assert off_step.channel_values["K1.no_state"] == 0.0
    assert off_step.channel_values["K1.nc_state"] == 1.0
    g_off, _ = circuit.assemble_dc()
    assert g_off[n_com][n_nc] < -1.0
    assert g_off[n_com][n_no] > -1e-3


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


def test_signal_mux_demux_mapping_is_deterministic() -> None:
    circuit = ps.Circuit()
    n_a = circuit.add_node("a")
    n_b = circuit.add_node("b")
    n_c = circuit.add_node("c")
    circuit.add_virtual_component(
        "signal_mux",
        "MUX1",
        [n_a, n_b, n_c],
        {"select_index": 1.0},
        {},
    )
    circuit.add_virtual_component(
        "signal_demux",
        "DMX1",
        [n_c, n_a, n_b],
        {},
        {},
    )

    x = [0.0] * circuit.system_size()
    x[n_a] = 2.0
    x[n_b] = 5.0
    x[n_c] = -1.0

    step = circuit.execute_mixed_domain_step(x, 1e-6)
    assert step.channel_values["MUX1"] == 5.0
    assert step.channel_values["DMX1"] == -1.0


def test_fuse_event_controller_trips_by_i2t() -> None:
    circuit = ps.Circuit()
    n_in = circuit.add_node("in")
    gnd = circuit.ground()
    circuit.add_voltage_source("V1", n_in, gnd, 5.0)
    circuit.add_switch("F_SW", n_in, gnd, True, 1e3, 1e-9)
    circuit.add_virtual_component(
        "fuse",
        "F_EVT",
        [n_in, gnd],
        {"g_on": 1e3, "blow_i2t": 1.0, "initial_closed": 1.0},
        {"target_component": "F_SW"},
    )

    x = [0.0] * circuit.system_size()
    x[n_in] = 5.0
    circuit.execute_mixed_domain_step(x, 0.0)
    step = circuit.execute_mixed_domain_step(x, 1e-2)
    assert step.channel_values["F_EVT.i2t"] > 1.0
    assert step.channel_values["F_EVT.state"] == 0.0


def test_breaker_event_controller_trips_by_time_overcurrent() -> None:
    circuit = ps.Circuit()
    n_in = circuit.add_node("in")
    gnd = circuit.ground()
    circuit.add_voltage_source("V1", n_in, gnd, 5.0)
    circuit.add_switch("B_SW", n_in, gnd, True, 1e3, 1e-9)
    circuit.add_virtual_component(
        "circuit_breaker",
        "B_EVT",
        [n_in, gnd],
        {"g_on": 1e3, "trip_current": 1000.0, "trip_time": 5e-3, "initial_closed": 1.0},
        {"target_component": "B_SW"},
    )

    x = [0.0] * circuit.system_size()
    x[n_in] = 5.0
    circuit.execute_mixed_domain_step(x, 0.0)
    mid = circuit.execute_mixed_domain_step(x, 2e-3)
    end = circuit.execute_mixed_domain_step(x, 6e-3)
    assert mid.channel_values["B_EVT.state"] == 1.0
    assert end.channel_values["B_EVT.trip_timer"] >= 5e-3
    assert end.channel_values["B_EVT.state"] == 0.0


def test_thyristor_latches_and_releases_on_holding_current() -> None:
    circuit = ps.Circuit()
    n_gate = circuit.add_node("gate")
    n_anode = circuit.add_node("anode")
    n_cathode = circuit.ground()
    circuit.add_switch("SCR_SW", n_anode, n_cathode, False, 1e3, 1e-9)
    circuit.add_virtual_component(
        "thyristor",
        "SCR_EVT",
        [n_gate, n_anode, n_cathode],
        {
            "gate_threshold": 1.0,
            "holding_current": 100.0,
            "latch_current": 500.0,
            "g_on": 1e3,
        },
        {"target_component": "SCR_SW"},
    )

    x = [0.0] * circuit.system_size()
    x[n_gate] = 2.0
    x[n_anode] = 1.0  # 1V * 1e3S = 1000A estimated => latches
    trig = circuit.execute_mixed_domain_step(x, 1e-6)
    assert trig.channel_values["SCR_EVT.trigger"] == 1.0
    assert trig.channel_values["SCR_EVT.state"] == 1.0

    x[n_gate] = 0.0
    keep = circuit.execute_mixed_domain_step(x, 2e-6)
    assert keep.channel_values["SCR_EVT.state"] == 1.0

    x[n_anode] = 1e-4  # 0.1A estimated => below holding current
    release = circuit.execute_mixed_domain_step(x, 3e-6)
    assert release.channel_values["SCR_EVT.state"] == 0.0


def test_triac_triggers_both_polarities() -> None:
    circuit = ps.Circuit()
    n_gate = circuit.add_node("gate")
    n_t1 = circuit.add_node("t1")
    n_t2 = circuit.ground()
    circuit.add_switch("TRI_SW", n_t1, n_t2, False, 1e3, 1e-9)
    circuit.add_virtual_component(
        "triac",
        "TRI_EVT",
        [n_gate, n_t1, n_t2],
        {
            "gate_threshold": 1.0,
            "holding_current": 10.0,
            "latch_current": 100.0,
            "g_on": 1e3,
        },
        {"target_component": "TRI_SW"},
    )

    x = [0.0] * circuit.system_size()
    x[n_gate] = -2.0
    x[n_t1] = -1.0  # reverse polarity but should still trigger for triac
    trig = circuit.execute_mixed_domain_step(x, 1e-6)
    assert trig.channel_values["TRI_EVT.trigger"] == 1.0
    assert trig.channel_values["TRI_EVT.state"] == 1.0

    x[n_gate] = 0.0
    hold = circuit.execute_mixed_domain_step(x, 2e-6)
    assert hold.channel_values["TRI_EVT.state"] == 1.0

    x[n_t1] = 0.0
    off = circuit.execute_mixed_domain_step(x, 3e-6)
    assert off.channel_values["TRI_EVT.state"] == 0.0


def test_latching_regularization_clamps_virtual_parameters() -> None:
    circuit = ps.Circuit()
    n_gate = circuit.add_node("gate")
    n_anode = circuit.add_node("anode")
    n_cathode = circuit.ground()
    circuit.add_switch("SCR_SW", n_anode, n_cathode, False, 1e9, 1e-15)
    circuit.add_virtual_component(
        "thyristor",
        "SCR_EVT",
        [n_gate, n_anode, n_cathode],
        {
            "gate_threshold": 0.0,
            "holding_current": 0.0,
            "latch_current": 0.0,
            "g_on": 1e9,
            "g_off": 1e-15,
        },
        {"target_component": "SCR_SW"},
    )

    changed = circuit.apply_numerical_regularization()
    assert changed > 0
    params = {vc.name: vc for vc in circuit.virtual_components()}["SCR_EVT"].numeric_params
    assert params["gate_threshold"] >= 1e-3
    assert params["holding_current"] >= 1e-6
    assert params["latch_current"] >= params["holding_current"]
    assert params["g_on"] <= 5e5
    assert params["g_off"] >= 1e-9


def test_saturable_inductor_effective_inductance_decreases_with_current() -> None:
    circuit = ps.Circuit()
    n_in = circuit.add_node("in")
    gnd = circuit.ground()
    circuit.add_inductor("Lsat", n_in, gnd, 1e-3, 0.0)
    circuit.add_virtual_component(
        "saturable_inductor",
        "Lsat",
        [n_in, gnd],
        {
            "inductance": 1e-3,
            "saturation_current": 1.0,
            "saturation_inductance": 1e-4,
            "saturation_exponent": 2.0,
        },
        {"target_component": "Lsat"},
    )

    br = circuit.num_nodes()
    x_low = [0.0] * circuit.system_size()
    x_low[n_in] = 1.0
    x_low[br] = 0.1
    j_low, _ = circuit.assemble_jacobian(x_low)

    x_high = [0.0] * circuit.system_size()
    x_high[n_in] = 1.0
    x_high[br] = 10.0
    j_high, _ = circuit.assemble_jacobian(x_high)

    assert abs(j_high[br][br]) < abs(j_low[br][br])


def test_coupled_inductor_adds_mutual_terms_to_jacobian() -> None:
    circuit = ps.Circuit()
    n_p = circuit.add_node("p")
    n_s = circuit.add_node("s")
    gnd = circuit.ground()
    circuit.add_inductor("K1__L1", n_p, gnd, 1e-3, 0.0)
    circuit.add_inductor("K1__L2", n_s, gnd, 4e-3, 0.0)
    circuit.add_virtual_component(
        "coupled_inductor",
        "K1",
        [n_p, gnd, n_s, gnd],
        {"l1": 1e-3, "l2": 4e-3, "coupling": 0.9},
        {"target_component_1": "K1__L1", "target_component_2": "K1__L2"},
    )

    br1 = circuit.num_nodes()
    br2 = circuit.num_nodes() + 1
    x = [0.0] * circuit.system_size()
    x[n_p] = 2.0
    x[n_s] = -1.0
    x[br1] = 1.5
    x[br2] = -0.5

    j, f = circuit.assemble_jacobian(x)
    assert abs(j[br1][br2]) > 1e-6
    assert abs(j[br2][br1]) > 1e-6
    assert abs(f[br1]) > 1e-9
    assert abs(f[br2]) > 1e-9


def test_yaml_parser_rejects_invalid_control_block_parameters() -> None:
    invalid_op_amp = """
schema: pulsim-v1
version: 1
components:
  - type: op_amp
    name: A1
    nodes: [vp, vn, out]
    rail_low: 5
    rail_high: 2
"""
    parser = ps.YamlParser()
    parser.load_string(invalid_op_amp)
    assert any("rail_low must be < rail_high" in msg for msg in parser.errors)

    invalid_pwm = """
schema: pulsim-v1
version: 1
components:
  - type: pwm_generator
    name: PWM1
    nodes: [ctrl]
    frequency: 10k
    duty_min: 0.9
    duty_max: 0.2
"""
    parser = ps.YamlParser()
    parser.load_string(invalid_pwm)
    assert any("duty_min/duty_max must satisfy" in msg for msg in parser.errors)

    invalid_math = """
schema: pulsim-v1
version: 1
components:
  - type: math_block
    name: M1
    nodes: [a, b]
    operation: xor
"""
    parser = ps.YamlParser()
    parser.load_string(invalid_math)
    assert any("operation must be add/sub/mul/div" in msg for msg in parser.errors)


def test_yaml_parser_rejects_invalid_signal_processing_block_parameters() -> None:
    invalid_limiter = """
schema: pulsim-v1
version: 1
components:
  - type: limiter
    name: LIM1
    nodes: [a, b]
    min: 5
    max: 1
"""
    parser = ps.YamlParser()
    parser.load_string(invalid_limiter)
    assert any("min must be <= max" in msg for msg in parser.errors)

    invalid_differentiator = """
schema: pulsim-v1
version: 1
components:
  - type: differentiator
    name: D1
    nodes: [a, b]
    alpha: 1.5
"""
    parser = ps.YamlParser()
    parser.load_string(invalid_differentiator)
    assert any("alpha must be in [0, 1]" in msg for msg in parser.errors)

    invalid_rate_limiter = """
schema: pulsim-v1
version: 1
components:
  - type: rate_limiter
    name: RL1
    nodes: [a, b]
    rising_rate: -10
"""
    parser = ps.YamlParser()
    parser.load_string(invalid_rate_limiter)
    assert any("rising_rate must be >= 0" in msg for msg in parser.errors)


def test_yaml_parser_rejects_invalid_advanced_control_block_parameters() -> None:
    invalid_lookup = """
schema: pulsim-v1
version: 1
components:
  - type: lookup_table
    name: LUT1
    nodes: [in, ref]
    x: [0, 1]
    y: [0]
"""
    parser = ps.YamlParser()
    parser.load_string(invalid_lookup)
    assert any("lookup_table requires x and y arrays" in msg for msg in parser.errors)

    invalid_transfer = """
schema: pulsim-v1
version: 1
components:
  - type: transfer_function
    name: TF1
    nodes: [in, ref]
    num: [1, 1]
    den: [0, 1]
"""
    parser = ps.YamlParser()
    parser.load_string(invalid_transfer)
    assert any("den[0] must be non-zero" in msg for msg in parser.errors)

    invalid_state_machine = """
schema: pulsim-v1
version: 1
components:
  - type: state_machine
    name: SM1
    nodes: [in]
    mode: random
"""
    parser = ps.YamlParser()
    parser.load_string(invalid_state_machine)
    assert any("state_machine mode must be" in msg for msg in parser.errors)


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
    assert (
        ps.FallbackReasonCode.TransientGminEscalation in reasons
        or ps.FallbackReasonCode.StiffnessBackoff in reasons
    )
    assert ps.FallbackReasonCode.MaxRetriesExceeded in reasons


def test_recovery_ladder_stages_are_deterministic() -> None:
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
    opts.max_step_retries = 5
    opts.linear_solver.order = [ps.LinearSolverKind.CG]
    opts.linear_solver.allow_fallback = False
    opts.linear_solver.auto_select = False
    opts.fallback_policy.trace_retries = True
    opts.fallback_policy.enable_transient_gmin = True
    opts.fallback_policy.gmin_retry_threshold = 1
    opts.fallback_policy.gmin_initial = 1e-8
    opts.fallback_policy.gmin_max = 1e-4

    x0 = [0.0] * (circuit.num_nodes() + circuit.num_branches())
    result_a = ps.Simulator(circuit, opts).run_transient(x0)
    result_b = ps.Simulator(circuit, opts).run_transient(x0)

    assert not result_a.success
    assert not result_b.success

    stage_actions_a = [entry.action for entry in result_a.fallback_trace if entry.action.startswith("recovery_stage_")]
    stage_actions_b = [entry.action for entry in result_b.fallback_trace if entry.action.startswith("recovery_stage_")]
    assert stage_actions_a == stage_actions_b

    expected_prefixes = [
        "recovery_stage_dt_backoff",
        "recovery_stage_globalization",
        "recovery_stage_stiff_profile",
        "recovery_stage_regularization",
    ]

    first_index: dict[str, int] = {}
    for prefix in expected_prefixes:
        matches = [idx for idx, action in enumerate(stage_actions_a) if action.startswith(prefix)]
        assert matches, f"missing stage action prefix: {prefix}"
        first_index[prefix] = matches[0]

    assert first_index["recovery_stage_dt_backoff"] < first_index["recovery_stage_globalization"]
    assert first_index["recovery_stage_globalization"] < first_index["recovery_stage_stiff_profile"]
    assert first_index["recovery_stage_stiff_profile"] < first_index["recovery_stage_regularization"]


def test_linear_factor_cache_telemetry_is_exposed() -> None:
    circuit = ps.Circuit()
    n_in = circuit.add_node("in")
    n_out = circuit.add_node("out")
    gnd = circuit.ground()
    circuit.add_voltage_source("V1", n_in, gnd, 5.0)
    circuit.add_resistor("R1", n_in, n_out, 1_000.0)
    circuit.add_capacitor("C1", n_out, gnd, 1e-6, 0.0)

    opts = ps.SimulationOptions()
    opts.tstart = 0.0
    opts.tstop = 5e-6
    opts.dt = 1e-6
    opts.dt_min = 1e-12
    opts.dt_max = 1e-6
    opts.adaptive_timestep = False
    opts.enable_bdf_order_control = False

    x0 = [0.0] * (circuit.num_nodes() + circuit.num_branches())
    result = ps.Simulator(circuit, opts).run_transient(x0)
    assert result.success
    assert result.backend_telemetry.state_space_primary_steps >= 1
    assert result.backend_telemetry.linear_factor_cache_misses >= 1
    assert result.backend_telemetry.linear_factor_cache_hits >= 1
    assert (
        result.backend_telemetry.linear_factor_cache_hits
        + result.backend_telemetry.linear_factor_cache_misses
    ) >= result.backend_telemetry.state_space_primary_steps


def test_invalid_time_window_sets_typed_diagnostic() -> None:
    circuit = ps.Circuit()
    n_in = circuit.add_node("in")
    n_out = circuit.add_node("out")
    gnd = circuit.ground()
    circuit.add_voltage_source("V1", n_in, gnd, 5.0)
    circuit.add_resistor("R1", n_in, n_out, 1_000.0)
    circuit.add_capacitor("C1", n_out, gnd, 1e-6, 0.0)

    opts = ps.SimulationOptions()
    opts.tstart = 0.0
    opts.tstop = -1.0
    opts.dt = 1e-6

    result = ps.Simulator(circuit, opts).run_transient()
    assert not result.success
    assert result.backend_telemetry.failure_reason == "invalid_time_window"


def test_variable_step_accept_reject_pattern_is_deterministic() -> None:
    root = Path(__file__).resolve().parents[2]
    case_path = root / "benchmarks" / "circuits" / "diode_rectifier.yaml"
    parser_opts = ps.YamlParserOptions()
    parser_opts.strict = False

    def run_once() -> ps.SimulationResult:
        parser = ps.YamlParser(parser_opts)
        circuit, opts = parser.load(str(case_path))
        assert parser.errors == []

        opts.adaptive_timestep = True
        opts.enable_bdf_order_control = False
        opts.fallback_policy.trace_retries = True
        opts.max_step_retries = max(opts.max_step_retries, 6)
        opts.dt_min = min(opts.dt_min, max(opts.dt * 0.02, 1e-9))
        opts.dt_max = max(opts.dt_max, opts.dt * 8.0)

        return ps.Simulator(circuit, opts).run_transient()

    result_a = run_once()
    result_b = run_once()

    assert result_a.success == result_b.success
    assert result_a.final_status == result_b.final_status
    assert result_a.total_steps == result_b.total_steps
    assert result_a.timestep_rejections == result_b.timestep_rejections
    assert result_a.timestep_rejections > 0

    trace_a = [(entry.reason, entry.action) for entry in result_a.fallback_trace]
    trace_b = [(entry.reason, entry.action) for entry in result_b.fallback_trace]
    assert trace_a == trace_b

    assert len(result_a.time) == len(result_b.time)
    for ta, tb in zip(result_a.time, result_b.time):
        assert ta == pytest.approx(tb, abs=1e-15)
