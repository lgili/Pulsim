## ADDED Requirements

### Requirement: Simulation State Control

The simulation engine SHALL support interactive control with pause and resume capability.

#### Scenario: Pause running simulation
- **GIVEN** a simulation is running
- **WHEN** request_pause() is called on the SimulationController
- **THEN** the simulation SHALL:
  - Complete the current timestep
  - Enter Paused state
  - Allow state query and result access
  - Not consume CPU while paused

#### Scenario: Resume paused simulation
- **GIVEN** a simulation is in Paused state
- **WHEN** request_resume() is called
- **THEN** the simulation SHALL continue from where it paused with state Running

#### Scenario: Query simulation state
- **GIVEN** a SimulationController instance exists
- **WHEN** state() is called
- **THEN** it SHALL return one of: Idle, Running, Paused, Stopping, Completed, Error

#### Scenario: Stop from any state
- **GIVEN** a simulation is running or paused
- **WHEN** request_stop() is called
- **THEN** the simulation SHALL:
  - Transition to Stopping state
  - Complete current timestep gracefully
  - Transition to Completed state
  - Make partial results available

#### Scenario: Thread-safe state access
- **GIVEN** a simulation is running in a background thread
- **WHEN** state() is called from the main thread
- **THEN** the call SHALL return immediately without blocking

#### Scenario: Wait for state change
- **GIVEN** a GUI needs to synchronize with simulation
- **WHEN** wait_for_state(target, timeout_ms) is called
- **THEN** the call SHALL block until:
  - The target state is reached, OR
  - The timeout expires (returns false)

### Requirement: Progress Callback with Metrics

The simulation engine SHALL provide detailed progress information during execution.

#### Scenario: Progress callback invocation
- **GIVEN** a progress callback is registered via ProgressCallbackConfig
- **WHEN** the simulation runs
- **THEN** the callback SHALL be invoked with SimulationProgress containing:
  - current_time (simulation time in seconds)
  - total_time (target end time)
  - progress_percent (0.0 to 100.0)
  - steps_completed (total timesteps computed)
  - newton_iterations (iterations for current step)
  - elapsed_seconds (wall-clock time since start)
  - estimated_remaining_seconds (-1 if unknown)

#### Scenario: Configurable callback frequency
- **GIVEN** a callback is registered with min_interval_ms = 100
- **THEN** the callback SHALL be invoked at most every 100ms wall-clock time

#### Scenario: Minimum step interval
- **GIVEN** a callback is registered with min_steps = 100
- **THEN** the callback SHALL be invoked at most every 100 timesteps

#### Scenario: Convergence warning in progress
- **GIVEN** Newton solver takes more than 10 iterations for a step
- **THEN** the progress SHALL include convergence_warning = true

#### Scenario: Progress callback returns control
- **GIVEN** a progress callback is invoked
- **THEN** returning false from the callback SHALL request simulation stop

### Requirement: Component Metadata API

The library SHALL provide metadata about components for GUI display.

#### Scenario: Get component metadata
- **GIVEN** a ComponentType enum value
- **WHEN** ComponentRegistry::instance().get(type) is called
- **THEN** ComponentMetadata SHALL be returned containing:
  - type (the ComponentType)
  - name (internal identifier, e.g., "resistor")
  - display_name (human-readable, e.g., "Resistor")
  - description (help text)
  - category (e.g., "Passive", "Semiconductor", "Source")
  - pins (list of PinMetadata)
  - parameters (list of ParameterMetadata)
  - symbol_id (for GUI symbol rendering)

#### Scenario: Get parameter metadata
- **GIVEN** a ComponentMetadata is retrieved
- **THEN** each ParameterMetadata SHALL contain:
  - name (internal name, e.g., "resistance")
  - display_name (e.g., "Resistance")
  - description (help text)
  - type (Real, Integer, Boolean, Enum, String)
  - default_value (optional)
  - min_value, max_value (for numeric types)
  - unit (e.g., "ohm", "F", "H")
  - enum_values (for Enum type)
  - required (bool)

#### Scenario: Get pin metadata
- **GIVEN** a ComponentMetadata is retrieved
- **THEN** each PinMetadata SHALL contain:
  - name (e.g., "anode", "cathode", "drain", "gate", "source")
  - description (e.g., "Positive terminal")

#### Scenario: List all component types
- **WHEN** ComponentRegistry::instance().all_types() is called
- **THEN** all registered ComponentType values SHALL be returned

#### Scenario: Filter by category
- **WHEN** ComponentRegistry::instance().types_in_category("Semiconductor") is called
- **THEN** only semiconductor component types SHALL be returned (Diode, MOSFET, IGBT)

#### Scenario: Validate parameter value
- **GIVEN** a parameter name and proposed value
- **WHEN** validate_parameter(ComponentType, param_name, value) is called
- **THEN** ValidationResult SHALL indicate:
  - is_valid (bool)
  - A Diagnostic with error_message if invalid
  - A Diagnostic with warning_message if suboptimal (e.g., very high resistance)

### Requirement: Schematic Position Storage

The circuit representation SHALL support storing schematic positions for GUI use.

#### Scenario: Set component position
- **GIVEN** a component named "R1" exists in a circuit
- **WHEN** circuit.set_position("R1", {x: 100, y: 200, orientation: 90, mirrored: false}) is called
- **THEN** the position SHALL be stored with the circuit

#### Scenario: Get component position
- **GIVEN** a component has a position set
- **WHEN** circuit.get_position("R1") is called
- **THEN** the position (x, y, orientation, mirrored) SHALL be returned

#### Scenario: Position is optional
- **GIVEN** a component without position set
- **WHEN** circuit.get_position("R1") is called
- **THEN** std::nullopt SHALL be returned

#### Scenario: Position in JSON netlist
- **GIVEN** a circuit with positioned components
- **WHEN** exported to JSON via NetlistParser::to_json()
- **THEN** positions SHALL be included in the output:
```json
{
  "name": "R1",
  "type": "resistor",
  "nodes": ["in", "out"],
  "params": {"resistance": 1000},
  "position": {"x": 100, "y": 200, "orientation": 90, "mirrored": false}
}
```

#### Scenario: Position preservation on parse
- **GIVEN** a JSON netlist with positions
- **WHEN** parsed via NetlistParser::parse_file()
- **THEN** positions SHALL be preserved in the Circuit object

#### Scenario: Bulk position operations
- **GIVEN** a circuit with many components
- **WHEN** circuit.all_positions() is called
- **THEN** a map of all component positions SHALL be returned

#### Scenario: Position orientation values
- **GIVEN** a SchematicPosition
- **THEN** orientation SHALL be one of: 0, 90, 180, 270 (degrees clockwise)

### Requirement: Validation API with Detailed Errors

The library SHALL provide comprehensive circuit validation with actionable error messages.

#### Scenario: Validate circuit
- **GIVEN** a Circuit object
- **WHEN** circuit.validate_detailed() is called
- **THEN** ValidationResult SHALL contain:
  - is_valid (true if no errors)
  - diagnostics (list of Diagnostic)

#### Scenario: Diagnostic structure
- **GIVEN** a validation produces diagnostics
- **THEN** each Diagnostic SHALL contain:
  - severity (Error, Warning, Info)
  - code (DiagnosticCode enum)
  - message (human-readable description)
  - component_name (if component-level)
  - node_name (if node-level)
  - related_components (list of affected component names)

#### Scenario: Floating node detection
- **GIVEN** node "X" has only one connection
- **WHEN** validation runs
- **THEN** a Diagnostic SHALL be generated:
  - severity: Warning
  - code: W_FLOATING_NODE
  - message: "Node 'X' has only one connection (floating)"
  - node_name: "X"

#### Scenario: Missing ground detection
- **GIVEN** no node is named "0" or connected to ground
- **WHEN** validation runs
- **THEN** a Diagnostic SHALL be generated:
  - severity: Error
  - code: E_NO_GROUND
  - message: "Circuit has no ground reference node"

#### Scenario: Voltage source loop detection
- **GIVEN** voltage sources V1 and V2 form a loop
- **WHEN** validation runs
- **THEN** a Diagnostic SHALL be generated:
  - severity: Error
  - code: E_VOLTAGE_SOURCE_LOOP
  - message: "Voltage source loop detected"
  - related_components: ["V1", "V2"]

#### Scenario: Short circuit detection
- **GIVEN** a very low-impedance path between voltage sources
- **WHEN** validation runs
- **THEN** a Diagnostic SHALL be generated:
  - severity: Warning
  - code: W_SHORT_CIRCUIT
  - message describing the path

#### Scenario: Invalid parameter detection
- **GIVEN** a resistor has negative resistance
- **WHEN** validation runs
- **THEN** a Diagnostic SHALL be generated:
  - severity: Error
  - code: E_INVALID_PARAMETER
  - component_name: "R1"
  - message: "Resistance must be positive"

#### Scenario: has_errors convenience method
- **GIVEN** a ValidationResult
- **WHEN** result.has_errors() is called
- **THEN** true SHALL be returned if any diagnostic has severity Error

### Requirement: Result Streaming Configuration

The simulation SHALL support efficient result streaming for long simulations.

#### Scenario: Configure result decimation
- **GIVEN** SimulationOptions with streaming.decimation_factor = 10
- **WHEN** simulation runs
- **THEN** only every 10th timestep SHALL be stored in results

#### Scenario: Rolling buffer mode
- **GIVEN** SimulationOptions with streaming.use_rolling_buffer = true and max_points = 10000
- **WHEN** simulation generates more than 10000 points
- **THEN** older points SHALL be discarded, keeping only the most recent 10000

#### Scenario: Stream results during simulation
- **GIVEN** a SimulationCallback is registered
- **WHEN** simulation runs
- **THEN** the callback SHALL receive each stored timestep as it completes

#### Scenario: Decimation with callback
- **GIVEN** decimation_factor = 10 and a callback is registered
- **THEN** the callback SHALL be invoked for every timestep, but only decimated points stored

### Requirement: Enhanced Simulation Result

The SimulationResult SHALL include additional metadata for GUI display.

#### Scenario: Signal metadata
- **GIVEN** a simulation completes
- **THEN** SimulationResult.signal_info SHALL contain for each signal:
  - name (e.g., "V(out)")
  - type ("voltage", "current", "power")
  - unit ("V", "A", "W")
  - component (associated component name, empty if node voltage)
  - nodes (list of related node names)

#### Scenario: Solver metadata
- **GIVEN** a simulation completes
- **THEN** SimulationResult.solver_info SHALL contain:
  - method (IntegrationMethod used)
  - abstol, reltol (tolerances used)
  - adaptive_timestep (whether adaptive stepping was enabled)

#### Scenario: Performance statistics
- **GIVEN** a simulation completes
- **THEN** SimulationResult SHALL include:
  - average_newton_iterations (mean iterations per step)
  - convergence_failures (steps that required damping)
  - timestep_reductions (adaptive timestep reductions)
  - peak_memory_bytes (maximum memory used, -1 if not tracked)

#### Scenario: Event log in results
- **GIVEN** switches changed state during simulation
- **THEN** SimulationResult.events SHALL contain SwitchEvent entries with:
  - switch_name
  - time
  - new_state (true = closed)
  - voltage (across switch at event)
  - current (through switch at event)

### Requirement: Circuit Serialization

The library SHALL support complete circuit serialization including GUI data.

#### Scenario: Export circuit to JSON
- **GIVEN** a Circuit object with components and positions
- **WHEN** NetlistParser::to_json(circuit) is called
- **THEN** a JSON string SHALL be returned with:
  - Circuit name
  - All components with types, nodes, and parameters
  - All positions (if set)
  - Format version number

#### Scenario: Round-trip preservation
- **GIVEN** a Circuit is exported to JSON and re-imported
- **THEN** all data SHALL be preserved exactly, including:
  - Component names and parameters
  - Node names
  - Positions and orientations

#### Scenario: Version compatibility
- **GIVEN** a JSON netlist from an older version
- **WHEN** parsed
- **THEN** missing fields SHALL use sensible defaults

### Requirement: Python Bindings

All GUI integration APIs SHALL be accessible from Python.

#### Scenario: SimulationController in Python
- **GIVEN** the pulsim Python module
- **THEN** SimulationController SHALL be available with:
  - state property returning SimulationState enum
  - request_pause(), request_resume(), request_stop() methods
  - wait_for_state(state, timeout_ms) method

#### Scenario: Progress callback in Python
- **GIVEN** Python code defines a callback function
- **WHEN** passed to Simulator with ProgressCallbackConfig
- **THEN** the callback SHALL be invoked with SimulationProgress as dict

#### Scenario: ComponentRegistry in Python
- **GIVEN** the pulsim Python module
- **THEN** ComponentRegistry.get(type) SHALL return ComponentMetadata as dict

#### Scenario: ValidationResult in Python
- **GIVEN** circuit.validate_detailed() is called in Python
- **THEN** ValidationResult SHALL be returned with:
  - is_valid property
  - diagnostics list of Diagnostic dicts
  - has_errors(), has_warnings() methods

#### Scenario: Position handling in Python
- **GIVEN** a Circuit in Python
- **THEN** set_position(name, dict) and get_position(name) SHALL work with:
  - dict format: {"x": float, "y": float, "orientation": int, "mirrored": bool}
