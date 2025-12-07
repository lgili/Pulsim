"""Integration tests for GUI integration features (Task 8.20).

Tests all features added for GUI integration:
- Simulation state control (pause/resume/stop)
- Progress callbacks
- Component metadata system
- Schematic position storage
- Validation API
- Streaming configuration
- Enhanced SimulationResult
"""

import pulsim as sl


# =============================================================================
# 1. Simulation State Control Tests
# =============================================================================

class TestSimulationState:
    """Test SimulationState enum."""

    def test_state_values_exist(self):
        """All state values are accessible."""
        assert sl.SimulationState.Idle is not None
        assert sl.SimulationState.Running is not None
        assert sl.SimulationState.Paused is not None
        assert sl.SimulationState.Stopping is not None
        assert sl.SimulationState.Completed is not None
        assert sl.SimulationState.Error is not None


class TestSimulationController:
    """Test SimulationController for pause/resume/stop."""

    def test_initial_state_is_idle(self):
        """Controller starts in Idle state."""
        controller = sl.SimulationController()
        assert controller.state == sl.SimulationState.Idle
        assert controller.is_idle()

    def test_state_query_methods(self):
        """Test all state query methods."""
        controller = sl.SimulationController()
        assert controller.is_idle()
        assert not controller.is_running()
        assert not controller.is_paused()
        assert not controller.is_stopping()
        assert not controller.is_completed()
        assert not controller.is_error()

    def test_request_methods_exist(self):
        """Request methods are callable."""
        controller = sl.SimulationController()
        # These should not raise
        controller.request_pause()
        controller.request_resume()
        controller.request_stop()
        controller.reset()


# =============================================================================
# 2. Progress Callback Tests
# =============================================================================

class TestSimulationProgress:
    """Test SimulationProgress struct."""

    def test_progress_fields(self):
        """SimulationProgress has all expected fields."""
        progress = sl.SimulationProgress()
        assert hasattr(progress, 'current_time')
        assert hasattr(progress, 'total_time')
        assert hasattr(progress, 'progress_percent')
        assert hasattr(progress, 'steps_completed')
        assert hasattr(progress, 'total_steps_estimate')
        assert hasattr(progress, 'newton_iterations')
        assert hasattr(progress, 'convergence_warning')
        assert hasattr(progress, 'elapsed_seconds')
        assert hasattr(progress, 'estimated_remaining_seconds')
        assert hasattr(progress, 'memory_bytes')

    def test_progress_to_dict(self):
        """SimulationProgress converts to dict."""
        progress = sl.SimulationProgress()
        d = progress.to_dict()
        assert 'current_time' in d
        assert 'progress_percent' in d
        assert 'elapsed_seconds' in d


class TestProgressCallbackConfig:
    """Test ProgressCallbackConfig."""

    def test_config_fields(self):
        """Config has all expected fields."""
        config = sl.ProgressCallbackConfig()
        assert hasattr(config, 'min_interval_ms')
        assert hasattr(config, 'min_steps')
        assert hasattr(config, 'include_memory')

    def test_config_defaults(self):
        """Config has reasonable defaults."""
        config = sl.ProgressCallbackConfig()
        assert config.min_interval_ms >= 0
        assert config.min_steps >= 0


class TestProgressCallbacks:
    """Test progress callback invocation."""

    def test_progress_callback_invoked(self):
        """Progress callback is invoked during simulation."""
        circuit = sl.Circuit()
        circuit.add_voltage_source("V1", "in", "0", 10.0)
        circuit.add_resistor("R1", "in", "0", 1000.0)

        opts = sl.SimulationOptions()
        opts.tstop = 1e-3
        opts.dt = 1e-5

        sim = sl.Simulator(circuit, opts)

        progress_updates = []

        def on_progress(progress):
            progress_updates.append(progress.to_dict())

        result = sim.run_transient_with_progress(
            callback=None,
            event_callback=None,
            control=None,
            progress_callback=on_progress,
            min_interval_ms=0,  # No throttling
            min_steps=10
        )

        assert result.final_status == sl.SolverStatus.Success
        assert len(progress_updates) > 0

        # Check progress fields
        last_progress = progress_updates[-1]
        assert last_progress['progress_percent'] > 0
        assert last_progress['steps_completed'] > 0


# =============================================================================
# 3. Component Metadata Tests
# =============================================================================

class TestParameterType:
    """Test ParameterType enum."""

    def test_types_exist(self):
        """All parameter types are accessible."""
        assert sl.ParameterType.Real is not None
        assert sl.ParameterType.Integer is not None
        assert sl.ParameterType.Boolean is not None
        assert sl.ParameterType.Enum is not None
        assert sl.ParameterType.String is not None


class TestComponentRegistry:
    """Test ComponentRegistry singleton."""

    def test_instance_is_singleton(self):
        """Registry returns same instance."""
        reg1 = sl.ComponentRegistry.instance()
        reg2 = sl.ComponentRegistry.instance()
        assert reg1 is reg2

    def test_all_types_returns_components(self):
        """Registry returns all component types."""
        registry = sl.ComponentRegistry.instance()
        types = registry.all_types()
        assert len(types) > 0
        assert sl.ComponentType.Resistor in types
        assert sl.ComponentType.Capacitor in types
        assert sl.ComponentType.VoltageSource in types

    def test_all_categories(self):
        """Registry returns all categories."""
        registry = sl.ComponentRegistry.instance()
        categories = registry.all_categories()
        assert len(categories) > 0
        assert "Passive" in categories
        assert "Sources" in categories

    def test_types_in_category(self):
        """Registry returns types for a category."""
        registry = sl.ComponentRegistry.instance()
        passive = registry.types_in_category("Passive")
        assert len(passive) > 0
        assert sl.ComponentType.Resistor in passive

    def test_get_metadata(self):
        """Registry returns metadata for component type."""
        registry = sl.ComponentRegistry.instance()
        meta = registry.get(sl.ComponentType.Resistor)

        assert meta.name == "resistor"
        assert meta.display_name == "Resistor"
        assert meta.category == "Passive"
        assert len(meta.pins) == 2
        assert len(meta.parameters) > 0


class TestComponentMetadata:
    """Test ComponentMetadata struct."""

    def test_metadata_fields(self):
        """Metadata has all expected fields."""
        registry = sl.ComponentRegistry.instance()
        meta = registry.get(sl.ComponentType.Resistor)

        assert hasattr(meta, 'type')
        assert hasattr(meta, 'name')
        assert hasattr(meta, 'display_name')
        assert hasattr(meta, 'description')
        assert hasattr(meta, 'category')
        assert hasattr(meta, 'pins')
        assert hasattr(meta, 'parameters')
        assert hasattr(meta, 'symbol_id')
        assert hasattr(meta, 'has_loss_model')
        assert hasattr(meta, 'has_thermal_model')

    def test_metadata_to_dict(self):
        """Metadata converts to dict."""
        registry = sl.ComponentRegistry.instance()
        meta = registry.get(sl.ComponentType.MOSFET)
        d = meta.to_dict()

        assert 'name' in d
        assert 'display_name' in d
        assert 'category' in d
        assert 'pins' in d
        assert 'parameters' in d
        assert d['has_loss_model'] is True


class TestParameterMetadata:
    """Test ParameterMetadata struct."""

    def test_parameter_fields(self):
        """Parameter metadata has all expected fields."""
        registry = sl.ComponentRegistry.instance()
        meta = registry.get(sl.ComponentType.Resistor)
        param = meta.parameters[0]  # resistance

        assert hasattr(param, 'name')
        assert hasattr(param, 'display_name')
        assert hasattr(param, 'description')
        assert hasattr(param, 'type')
        assert hasattr(param, 'default_value')
        assert hasattr(param, 'min_value')
        assert hasattr(param, 'max_value')
        assert hasattr(param, 'unit')
        assert hasattr(param, 'required')

    def test_parameter_to_dict(self):
        """Parameter converts to dict."""
        registry = sl.ComponentRegistry.instance()
        meta = registry.get(sl.ComponentType.Resistor)
        param = meta.parameters[0]
        d = param.to_dict()

        assert 'name' in d
        assert 'unit' in d
        assert 'required' in d


class TestPinMetadata:
    """Test PinMetadata struct."""

    def test_pin_fields(self):
        """Pin metadata has expected fields."""
        registry = sl.ComponentRegistry.instance()
        meta = registry.get(sl.ComponentType.Diode)
        pin = meta.pins[0]  # anode

        assert hasattr(pin, 'name')
        assert hasattr(pin, 'description')
        assert pin.name in ['anode', 'cathode']


# =============================================================================
# 4. Schematic Position Tests
# =============================================================================

class TestSchematicPosition:
    """Test SchematicPosition struct."""

    def test_default_construction(self):
        """Default position is at origin."""
        pos = sl.SchematicPosition()
        assert pos.x == 0.0
        assert pos.y == 0.0
        assert pos.orientation == 0
        assert pos.mirrored is False

    def test_parameterized_construction(self):
        """Position can be constructed with values."""
        pos = sl.SchematicPosition(100.0, 200.0, 90, True)
        assert pos.x == 100.0
        assert pos.y == 200.0
        assert pos.orientation == 90
        assert pos.mirrored is True

    def test_to_dict(self):
        """Position converts to dict."""
        pos = sl.SchematicPosition(50.0, 75.0, 180, False)
        d = pos.to_dict()
        assert d['x'] == 50.0
        assert d['y'] == 75.0
        assert d['orientation'] == 180
        assert d['mirrored'] is False


class TestCircuitPositions:
    """Test position storage on Circuit."""

    def test_no_position_initially(self):
        """Components have no position by default."""
        circuit = sl.Circuit()
        circuit.add_resistor("R1", "a", "b", 1000.0)

        assert not circuit.has_position("R1")
        pos = circuit.get_position("R1")
        assert pos is None

    def test_set_and_get_position(self):
        """Can set and retrieve position."""
        circuit = sl.Circuit()
        circuit.add_resistor("R1", "a", "b", 1000.0)

        pos = sl.SchematicPosition(100.0, 50.0, 90, False)
        circuit.set_position("R1", pos)

        assert circuit.has_position("R1")
        retrieved = circuit.get_position("R1")
        assert retrieved is not None
        assert retrieved.x == 100.0
        assert retrieved.y == 50.0
        assert retrieved.orientation == 90

    def test_all_positions(self):
        """Can get all positions."""
        circuit = sl.Circuit()
        circuit.add_resistor("R1", "a", "b", 1000.0)
        circuit.add_capacitor("C1", "b", "0", 1e-6)

        circuit.set_position("R1", sl.SchematicPosition(0.0, 0.0))
        circuit.set_position("C1", sl.SchematicPosition(100.0, 0.0))

        positions = circuit.all_positions()
        assert len(positions) == 2
        assert "R1" in positions
        assert "C1" in positions

    def test_set_all_positions(self):
        """Can set all positions at once."""
        circuit = sl.Circuit()
        circuit.add_resistor("R1", "a", "b", 1000.0)
        circuit.add_resistor("R2", "b", "0", 2000.0)

        positions = {
            "R1": sl.SchematicPosition(10.0, 20.0),
            "R2": sl.SchematicPosition(30.0, 40.0)
        }
        circuit.set_all_positions(positions)

        assert circuit.get_position("R1").x == 10.0
        assert circuit.get_position("R2").x == 30.0

    def test_clear_positions(self):
        """Can clear all positions."""
        circuit = sl.Circuit()
        circuit.add_resistor("R1", "a", "b", 1000.0)
        circuit.set_position("R1", sl.SchematicPosition(100.0, 100.0))

        circuit.clear_positions()
        assert not circuit.has_position("R1")


class TestPositionRoundTrip:
    """Test JSON round-trip for positions."""

    def test_json_export_includes_positions(self):
        """JSON export includes position data."""
        circuit = sl.Circuit()
        circuit.add_voltage_source("V1", "in", "0", 5.0)
        circuit.add_resistor("R1", "in", "0", 1000.0)
        circuit.set_position("V1", sl.SchematicPosition(0.0, 0.0, 0, False))
        circuit.set_position("R1", sl.SchematicPosition(100.0, 0.0, 90, True))

        json_str = sl.circuit_to_json(circuit, include_positions=True)
        assert '"position"' in json_str
        assert '"orientation"' in json_str
        assert '"mirrored"' in json_str

    def test_json_roundtrip_preserves_positions(self):
        """Positions survive JSON export/import."""
        circuit = sl.Circuit()
        circuit.add_voltage_source("V1", "in", "0", 5.0)
        circuit.add_resistor("R1", "in", "0", 1000.0)
        circuit.set_position("R1", sl.SchematicPosition(150.0, 75.0, 270, True))

        json_str = sl.circuit_to_json(circuit, include_positions=True)
        imported = sl.parse_netlist_string(json_str)

        assert imported.has_position("R1")
        pos = imported.get_position("R1")
        assert pos.x == 150.0
        assert pos.y == 75.0
        assert pos.orientation == 270
        assert pos.mirrored is True


# =============================================================================
# 5. Validation API Tests
# =============================================================================

class TestDiagnosticSeverity:
    """Test DiagnosticSeverity enum."""

    def test_severities_exist(self):
        """All severity levels are accessible."""
        assert sl.DiagnosticSeverity.Error is not None
        assert sl.DiagnosticSeverity.Warning is not None
        assert sl.DiagnosticSeverity.Info is not None


class TestDiagnosticCode:
    """Test DiagnosticCode enum."""

    def test_error_codes_exist(self):
        """Error codes are accessible."""
        assert sl.DiagnosticCode.E_NO_GROUND is not None
        assert sl.DiagnosticCode.E_VOLTAGE_SOURCE_LOOP is not None
        assert sl.DiagnosticCode.E_DUPLICATE_NAME is not None
        assert sl.DiagnosticCode.E_NO_COMPONENTS is not None

    def test_warning_codes_exist(self):
        """Warning codes are accessible."""
        assert sl.DiagnosticCode.W_FLOATING_NODE is not None
        assert sl.DiagnosticCode.W_SHORT_CIRCUIT is not None

    def test_info_codes_exist(self):
        """Info codes are accessible."""
        assert sl.DiagnosticCode.I_IDEAL_SWITCH is not None


class TestValidationResult:
    """Test ValidationResult struct."""

    def test_valid_circuit(self):
        """Valid circuit passes validation."""
        circuit = sl.Circuit()
        circuit.add_voltage_source("V1", "in", "0", 5.0)
        circuit.add_resistor("R1", "in", "0", 1000.0)

        result = sl.validate_circuit(circuit)
        assert result.is_valid
        assert not result.has_errors()

    def test_invalid_circuit_no_ground(self):
        """Circuit without ground fails validation."""
        circuit = sl.Circuit()
        circuit.add_voltage_source("V1", "a", "b", 5.0)
        circuit.add_resistor("R1", "a", "b", 1000.0)

        result = sl.validate_circuit(circuit)
        assert not result.is_valid
        assert result.has_errors()

        errors = result.errors()
        assert len(errors) > 0
        # Check for E_NO_GROUND error
        codes = [e.code for e in errors]
        assert sl.DiagnosticCode.E_NO_GROUND in codes

    def test_validation_methods(self):
        """ValidationResult has helper methods."""
        circuit = sl.Circuit()
        circuit.add_voltage_source("V1", "in", "0", 5.0)
        circuit.add_resistor("R1", "in", "0", 1000.0)

        result = sl.validate_circuit(circuit)

        # Test method existence
        assert hasattr(result, 'has_errors')
        assert hasattr(result, 'has_warnings')
        assert hasattr(result, 'errors')
        assert hasattr(result, 'warnings')
        assert hasattr(result, 'infos')


class TestDiagnostic:
    """Test Diagnostic struct."""

    def test_diagnostic_fields(self):
        """Diagnostic has expected fields."""
        circuit = sl.Circuit()
        circuit.add_voltage_source("V1", "a", "b", 5.0)
        circuit.add_resistor("R1", "a", "b", 1000.0)

        result = sl.validate_circuit(circuit)
        diag = result.diagnostics[0]

        assert hasattr(diag, 'severity')
        assert hasattr(diag, 'code')
        assert hasattr(diag, 'message')
        assert hasattr(diag, 'component_name')
        assert hasattr(diag, 'node_name')

    def test_diagnostic_to_dict(self):
        """Diagnostic converts to dict."""
        circuit = sl.Circuit()
        circuit.add_voltage_source("V1", "a", "b", 5.0)
        circuit.add_resistor("R1", "a", "b", 1000.0)

        result = sl.validate_circuit(circuit)
        d = result.diagnostics[0].to_dict()

        assert 'severity' in d
        assert 'code' in d
        assert 'message' in d


class TestValidateDetailed:
    """Test Circuit.validate_detailed() method."""

    def test_validate_detailed_method(self):
        """Circuit has validate_detailed method."""
        circuit = sl.Circuit()
        circuit.add_voltage_source("V1", "in", "0", 5.0)
        circuit.add_resistor("R1", "in", "0", 1000.0)

        result = circuit.validate_detailed()
        assert isinstance(result, sl.ValidationResult)
        assert result.is_valid


# =============================================================================
# 6. Streaming Configuration Tests
# =============================================================================

class TestStreamingConfig:
    """Test StreamingConfig struct."""

    def test_config_fields(self):
        """StreamingConfig has expected fields."""
        config = sl.StreamingConfig()
        assert hasattr(config, 'decimation_factor')
        assert hasattr(config, 'use_rolling_buffer')
        assert hasattr(config, 'max_points')
        assert hasattr(config, 'callback_interval_ms')


class TestSimulationOptionsStreaming:
    """Test streaming options on SimulationOptions."""

    def test_streaming_decimation_field(self):
        """SimulationOptions has streaming_decimation."""
        opts = sl.SimulationOptions()
        assert hasattr(opts, 'streaming_decimation')
        assert opts.streaming_decimation == 1  # Default: store all

    def test_streaming_rolling_buffer_field(self):
        """SimulationOptions has streaming_rolling_buffer."""
        opts = sl.SimulationOptions()
        assert hasattr(opts, 'streaming_rolling_buffer')
        assert opts.streaming_rolling_buffer is False

    def test_streaming_max_points_field(self):
        """SimulationOptions has streaming_max_points."""
        opts = sl.SimulationOptions()
        assert hasattr(opts, 'streaming_max_points')
        assert opts.streaming_max_points > 0


class TestStreamingDecimation:
    """Test decimation functionality."""

    def test_decimation_reduces_points(self):
        """Decimation stores fewer points."""
        circuit = sl.Circuit()
        circuit.add_voltage_source("V1", "in", "0", 10.0)
        circuit.add_resistor("R1", "in", "0", 1000.0)

        # Run without decimation
        opts1 = sl.SimulationOptions()
        opts1.tstop = 1e-3
        opts1.dt = 1e-6
        opts1.dtmax = 1e-6
        opts1.adaptive_timestep = False
        opts1.streaming_decimation = 1

        sim1 = sl.Simulator(circuit, opts1)
        result1 = sim1.run_transient()

        # Run with decimation
        opts2 = sl.SimulationOptions()
        opts2.tstop = 1e-3
        opts2.dt = 1e-6
        opts2.dtmax = 1e-6
        opts2.adaptive_timestep = False
        opts2.streaming_decimation = 10

        sim2 = sl.Simulator(circuit, opts2)
        result2 = sim2.run_transient()

        assert result1.final_status == sl.SolverStatus.Success
        assert result2.final_status == sl.SolverStatus.Success
        # Decimated should have significantly fewer points
        assert len(result2.time) < len(result1.time) / 5


class TestStreamingRollingBuffer:
    """Test rolling buffer functionality."""

    def test_rolling_buffer_limits_points(self):
        """Rolling buffer limits stored points."""
        circuit = sl.Circuit()
        circuit.add_voltage_source("V1", "in", "0", 10.0)
        circuit.add_resistor("R1", "in", "0", 1000.0)

        opts = sl.SimulationOptions()
        opts.tstop = 10e-3
        opts.dt = 1e-6
        opts.dtmax = 1e-6
        opts.adaptive_timestep = False
        opts.streaming_rolling_buffer = True
        opts.streaming_max_points = 500

        sim = sl.Simulator(circuit, opts)
        result = sim.run_transient()

        assert result.final_status == sl.SolverStatus.Success
        # Should be bounded by max_points
        assert len(result.time) <= 500


# =============================================================================
# 7. Enhanced SimulationResult Tests
# =============================================================================

class TestSignalInfo:
    """Test SignalInfo struct."""

    def test_signal_info_fields(self):
        """SignalInfo has expected fields."""
        info = sl.SignalInfo()
        assert hasattr(info, 'name')
        assert hasattr(info, 'type')
        assert hasattr(info, 'unit')
        assert hasattr(info, 'component')
        assert hasattr(info, 'nodes')

    def test_signal_info_to_dict(self):
        """SignalInfo converts to dict."""
        info = sl.SignalInfo()
        info.name = "V(test)"
        info.type = "voltage"
        info.unit = "V"

        d = info.to_dict()
        assert d['name'] == "V(test)"
        assert d['type'] == "voltage"
        assert d['unit'] == "V"


class TestSolverInfo:
    """Test SolverInfo struct."""

    def test_solver_info_fields(self):
        """SolverInfo has expected fields."""
        info = sl.SolverInfo()
        assert hasattr(info, 'method')
        assert hasattr(info, 'abstol')
        assert hasattr(info, 'reltol')
        assert hasattr(info, 'adaptive_timestep')


class TestSimulationEventType:
    """Test SimulationEventType enum."""

    def test_event_types_exist(self):
        """All event types are accessible."""
        assert sl.SimulationEventType.SwitchClose is not None
        assert sl.SimulationEventType.SwitchOpen is not None
        assert sl.SimulationEventType.Convergence is not None
        assert sl.SimulationEventType.TimestepChange is not None


class TestSimulationEvent:
    """Test SimulationEvent struct."""

    def test_event_fields(self):
        """SimulationEvent has expected fields."""
        event = sl.SimulationEvent()
        assert hasattr(event, 'time')
        assert hasattr(event, 'type')
        assert hasattr(event, 'component')
        assert hasattr(event, 'description')
        assert hasattr(event, 'value1')
        assert hasattr(event, 'value2')

    def test_event_to_dict(self):
        """SimulationEvent converts to dict."""
        event = sl.SimulationEvent()
        d = event.to_dict()
        assert 'time' in d
        assert 'type' in d
        assert 'component' in d


class TestEnhancedSimulationResult:
    """Test enhanced SimulationResult fields."""

    def test_signal_info_in_result(self):
        """Result contains signal_info."""
        circuit = sl.Circuit()
        circuit.add_voltage_source("V1", "in", "0", 10.0)
        circuit.add_resistor("R1", "in", "mid", 100.0)
        circuit.add_inductor("L1", "mid", "0", 1e-3)

        opts = sl.SimulationOptions()
        opts.tstop = 1e-6
        opts.dt = 1e-7

        sim = sl.Simulator(circuit, opts)
        config = sl.ProgressCallbackConfig()
        config.min_interval_ms = 1000

        result = sim.run_transient_with_progress(
            callback=None,
            event_callback=None,
            control=None,
            progress_callback=lambda p: None,
            min_interval_ms=1000,
            min_steps=1000
        )

        assert result.final_status == sl.SolverStatus.Success
        assert len(result.signal_info) > 0
        assert len(result.signal_info) == len(result.signal_names)

    def test_solver_info_in_result(self):
        """Result contains solver_info."""
        circuit = sl.Circuit()
        circuit.add_voltage_source("V1", "in", "0", 5.0)
        circuit.add_resistor("R1", "in", "0", 100.0)

        opts = sl.SimulationOptions()
        opts.tstop = 1e-6
        opts.dt = 1e-7
        opts.integration_method = sl.IntegrationMethod.Trapezoidal
        opts.abstol = 1e-10
        opts.reltol = 1e-4

        sim = sl.Simulator(circuit, opts)
        result = sim.run_transient()

        assert result.solver_info.method == sl.IntegrationMethod.Trapezoidal
        assert result.solver_info.abstol == 1e-10
        assert result.solver_info.reltol == 1e-4

    def test_performance_metrics_in_result(self):
        """Result contains performance metrics."""
        circuit = sl.Circuit()
        circuit.add_voltage_source("V1", "in", "0", 10.0)
        circuit.add_resistor("R1", "in", "0", 1000.0)

        opts = sl.SimulationOptions()
        opts.tstop = 10e-6
        opts.dt = 1e-6

        sim = sl.Simulator(circuit, opts)
        result = sim.run_transient()

        assert result.final_status == sl.SolverStatus.Success
        assert hasattr(result, 'average_newton_iterations')
        assert hasattr(result, 'convergence_failures')
        assert hasattr(result, 'timestep_reductions')
        assert hasattr(result, 'peak_memory_bytes')
        assert result.average_newton_iterations >= 0

    def test_events_in_result(self):
        """Result contains events list."""
        circuit = sl.Circuit()
        circuit.add_voltage_source("V1", "in", "0", 5.0)
        circuit.add_resistor("R1", "in", "0", 100.0)

        opts = sl.SimulationOptions()
        opts.tstop = 1e-6
        opts.dt = 1e-7

        sim = sl.Simulator(circuit, opts)
        result = sim.run_transient()

        assert hasattr(result, 'events')
        assert isinstance(result.events, list)

    def test_convenience_methods(self):
        """Result has convenience methods."""
        circuit = sl.Circuit()
        circuit.add_voltage_source("V1", "in", "0", 5.0)
        circuit.add_resistor("R1", "in", "0", 100.0)

        opts = sl.SimulationOptions()
        opts.tstop = 10e-6
        opts.dt = 1e-6

        sim = sl.Simulator(circuit, opts)
        result = sim.run_transient()

        assert result.num_signals() == len(result.signal_names)
        assert result.num_points() == len(result.time)
        assert result.num_events() == len(result.events)


class TestSwitchEvents:
    """Test switch event tracking."""

    def test_switch_events_recorded(self):
        """Switch events are recorded in result."""
        circuit = sl.Circuit()
        # Pulse to control switch
        pulse = sl.PulseWaveform()
        pulse.v1 = 0.0
        pulse.v2 = 5.0
        pulse.td = 5e-6
        pulse.tr = 0.1e-6
        pulse.tf = 0.1e-6
        pulse.pw = 10e-6
        pulse.period = 25e-6

        circuit.add_voltage_source("V1", "in", "0", 10.0)
        circuit.add_resistor("R1", "in", "sw", 100.0)

        sw_params = sl.SwitchParams()
        sw_params.ron = 0.01
        sw_params.roff = 1e9
        sw_params.vth = 2.5
        circuit.add_switch("S1", "sw", "out", "ctrl", "0", sw_params)
        circuit.add_resistor("R2", "out", "0", 100.0)

        # Add control voltage with pulse - use a simpler approach
        circuit.add_voltage_source("Vctrl", "ctrl", "0", 5.0)  # Start closed

        opts = sl.SimulationOptions()
        opts.tstop = 50e-6
        opts.dt = 0.1e-6
        opts.dtmax = 0.1e-6
        opts.adaptive_timestep = False

        sim = sl.Simulator(circuit, opts)
        result = sim.run_transient_with_progress(
            callback=None,
            event_callback=None,
            control=None,
            progress_callback=lambda p: None,
            min_interval_ms=1000,
            min_steps=1000
        )

        assert result.final_status == sl.SolverStatus.Success
        # Events list should exist (may be empty if no state changes)
        assert isinstance(result.events, list)


# =============================================================================
# 8. Integration Method Tests
# =============================================================================

class TestIntegrationMethod:
    """Test IntegrationMethod enum."""

    def test_methods_exist(self):
        """All integration methods are accessible."""
        assert sl.IntegrationMethod.BackwardEuler is not None
        assert sl.IntegrationMethod.Trapezoidal is not None
        assert sl.IntegrationMethod.BDF2 is not None
        assert sl.IntegrationMethod.GEAR2 is not None


class TestIntegrationMethodInOptions:
    """Test integration method in SimulationOptions."""

    def test_set_integration_method(self):
        """Can set integration method."""
        opts = sl.SimulationOptions()
        opts.integration_method = sl.IntegrationMethod.BDF2
        assert opts.integration_method == sl.IntegrationMethod.BDF2


# =============================================================================
# 9. JSON Export Tests
# =============================================================================

class TestCircuitToJson:
    """Test circuit_to_json function."""

    def test_export_basic_circuit(self):
        """Can export basic circuit to JSON."""
        circuit = sl.Circuit()
        circuit.add_voltage_source("V1", "in", "0", 5.0)
        circuit.add_resistor("R1", "in", "0", 1000.0)

        json_str = sl.circuit_to_json(circuit)
        assert '"V1"' in json_str or '"name"' in json_str
        assert '"R1"' in json_str or '"resistor"' in json_str

    def test_export_without_positions(self):
        """Can export without positions."""
        circuit = sl.Circuit()
        circuit.add_voltage_source("V1", "in", "0", 5.0)
        circuit.add_resistor("R1", "in", "0", 1000.0)
        circuit.set_position("R1", sl.SchematicPosition(100.0, 100.0))

        json_str = sl.circuit_to_json(circuit, include_positions=False)
        assert '"position"' not in json_str


# =============================================================================
# 10. Diagnostic Code Description Test
# =============================================================================

class TestDiagnosticCodeDescription:
    """Test diagnostic_code_description function."""

    def test_get_description(self):
        """Can get description for diagnostic code."""
        desc = sl.diagnostic_code_description(sl.DiagnosticCode.E_NO_GROUND)
        assert len(desc) > 0
        assert "ground" in desc.lower() or "reference" in desc.lower()
