## 1. Simulation State Control

- [ ] 1.1 Define SimulationState enum in `types.hpp`
  - Idle, Running, Paused, Stopping, Completed, Error
- [ ] 1.2 Create `SimulationController` class in new `simulation_control.hpp`
  - Atomic state storage
  - Mutex and condition variable for synchronization
- [ ] 1.3 Implement `state()` method (thread-safe read)
- [ ] 1.4 Implement `request_pause()` method
- [ ] 1.5 Implement `request_resume()` method
- [ ] 1.6 Implement `request_stop()` method
- [ ] 1.7 Implement `wait_for_state(target, timeout_ms)` method
- [ ] 1.8 Integrate SimulationController into Simulator::run_transient()
- [ ] 1.9 Replace old SimulationControl with new controller
- [ ] 1.10 Unit tests for all state transitions
- [ ] 1.11 Thread safety tests with TSAN

## 2. Progress Callback System

- [ ] 2.1 Define `SimulationProgress` struct in `types.hpp`
  - current_time, total_time, progress_percent
  - steps_completed, newton_iterations
  - elapsed_seconds, estimated_remaining_seconds
  - convergence_warning
- [ ] 2.2 Define `ProgressCallback` function type
- [ ] 2.3 Define `ProgressCallbackConfig` struct
  - callback, min_interval_ms, min_steps, include_memory
- [ ] 2.4 Add ProgressCallbackConfig to SimulationOptions
- [ ] 2.5 Implement progress tracking in simulation loop
- [ ] 2.6 Implement estimated time remaining calculation
- [ ] 2.7 Implement callback throttling based on interval/steps
- [ ] 2.8 Add convergence warning detection (>10 iterations)
- [ ] 2.9 Unit tests for progress callbacks
- [ ] 2.10 Performance benchmark (callback overhead <5%)

## 3. Component Metadata System

- [ ] 3.1 Define `ParameterType` enum in new `metadata.hpp`
  - Real, Integer, Boolean, Enum, String
- [ ] 3.2 Define `ParameterMetadata` struct
  - name, display_name, description, type
  - default_value, min_value, max_value, unit
  - enum_values, required
- [ ] 3.3 Define `PinMetadata` struct
  - name, description
- [ ] 3.4 Define `ComponentMetadata` struct
  - type, name, display_name, description, category
  - pins, parameters, symbol_id
  - has_loss_model, has_thermal_model
- [ ] 3.5 Create `ComponentRegistry` singleton class
- [ ] 3.6 Implement `get(ComponentType)` method
- [ ] 3.7 Implement `all_types()` method
- [ ] 3.8 Implement `types_in_category(category)` method
- [ ] 3.9 Register metadata for Resistor
- [ ] 3.10 Register metadata for Capacitor
- [ ] 3.11 Register metadata for Inductor
- [ ] 3.12 Register metadata for VoltageSource (with waveform variants)
- [ ] 3.13 Register metadata for CurrentSource
- [ ] 3.14 Register metadata for Diode
- [ ] 3.15 Register metadata for Switch
- [ ] 3.16 Register metadata for MOSFET
- [ ] 3.17 Register metadata for IGBT
- [ ] 3.18 Register metadata for Transformer
- [ ] 3.19 Implement parameter validation function
- [ ] 3.20 Unit tests for all metadata

## 4. Schematic Position Storage

- [ ] 4.1 Define `SchematicPosition` struct in `circuit.hpp`
  - x, y (double)
  - orientation (int: 0, 90, 180, 270)
  - mirrored (bool)
- [ ] 4.2 Add positions_ map to Circuit class
- [ ] 4.3 Implement `set_position(name, position)` method
- [ ] 4.4 Implement `get_position(name)` method (returns optional)
- [ ] 4.5 Implement `has_position(name)` method
- [ ] 4.6 Implement `all_positions()` method
- [ ] 4.7 Implement `set_all_positions(map)` method
- [ ] 4.8 Update JSON parser to read position field
- [ ] 4.9 Implement `NetlistParser::to_json(circuit)` for export
- [ ] 4.10 Add position field to JSON output
- [ ] 4.11 Unit tests for position round-trip

## 5. Validation API

- [ ] 5.1 Define `DiagnosticSeverity` enum in new `validation.hpp`
  - Error, Warning, Info
- [ ] 5.2 Define `DiagnosticCode` enum with all error/warning codes
- [ ] 5.3 Define `Diagnostic` struct
  - severity, code, message
  - component_name, node_name, parameter_name
  - related_components
- [ ] 5.4 Define `ValidationResult` struct
  - is_valid, diagnostics
  - has_errors(), has_warnings(), errors(), warnings()
- [ ] 5.5 Implement `Circuit::validate_detailed()` method
- [ ] 5.6 Implement floating node detection
- [ ] 5.7 Implement missing ground detection
- [ ] 5.8 Implement voltage source loop detection
- [ ] 5.9 Implement inductor/voltage source loop detection
- [ ] 5.10 Implement short circuit detection
- [ ] 5.11 Implement parameter validation for each component type
- [ ] 5.12 Implement duplicate component name detection
- [ ] 5.13 Unit tests for each diagnostic type

## 6. Result Streaming Configuration

- [ ] 6.1 Define `StreamingConfig` struct in `types.hpp`
  - decimation_factor, use_rolling_buffer, max_points
  - callback_interval_ms
- [ ] 6.2 Add StreamingConfig to SimulationOptions
- [ ] 6.3 Implement decimation in simulation loop
- [ ] 6.4 Implement rolling buffer storage
- [ ] 6.5 Separate callback invocation from storage
- [ ] 6.6 Unit tests for decimation
- [ ] 6.7 Unit tests for rolling buffer
- [ ] 6.8 Memory usage tests for long simulations

## 7. Enhanced SimulationResult

- [ ] 7.1 Define `SignalInfo` struct
  - name, type, unit, component, nodes
- [ ] 7.2 Define `SolverInfo` struct
  - method, abstol, reltol, adaptive_timestep
- [ ] 7.3 Add signal_info vector to SimulationResult
- [ ] 7.4 Add solver_info to SimulationResult
- [ ] 7.5 Add average_newton_iterations to SimulationResult
- [ ] 7.6 Add convergence_failures to SimulationResult
- [ ] 7.7 Add timestep_reductions to SimulationResult
- [ ] 7.8 Add peak_memory_bytes to SimulationResult
- [ ] 7.9 Add events vector to SimulationResult
- [ ] 7.10 Populate signal_info during simulation setup
- [ ] 7.11 Track performance statistics during simulation
- [ ] 7.12 Collect switch events during simulation
- [ ] 7.13 Unit tests for enhanced result

## 8. Python Bindings

- [ ] 8.1 Expose SimulationState enum
- [ ] 8.2 Expose SimulationController class
  - state property
  - request_pause, request_resume, request_stop methods
  - wait_for_state method
- [ ] 8.3 Expose SimulationProgress as dict-convertible
- [ ] 8.4 Expose ProgressCallbackConfig
- [ ] 8.5 Add Python callback wrapper for progress
- [ ] 8.6 Expose ComponentMetadata as dict
- [ ] 8.7 Expose ParameterMetadata as dict
- [ ] 8.8 Expose PinMetadata as dict
- [ ] 8.9 Expose ComponentRegistry singleton
- [ ] 8.10 Expose DiagnosticSeverity and DiagnosticCode enums
- [ ] 8.11 Expose Diagnostic as dict
- [ ] 8.12 Expose ValidationResult with methods
- [ ] 8.13 Add validate_detailed() to Circuit binding
- [ ] 8.14 Expose SchematicPosition as dict
- [ ] 8.15 Add position methods to Circuit binding
- [ ] 8.16 Expose to_json() function
- [ ] 8.17 Expose StreamingConfig
- [ ] 8.18 Expose SignalInfo and SolverInfo
- [ ] 8.19 Update SimulationResult binding with new fields
- [ ] 8.20 Python integration tests for all features

## 9. Documentation

- [ ] 9.1 Document SimulationController in header
- [ ] 9.2 Document progress callback system
- [ ] 9.3 Document ComponentRegistry API
- [ ] 9.4 Document validation API
- [ ] 9.5 Document position storage
- [ ] 9.6 Document streaming configuration
- [ ] 9.7 Update Python docstrings
- [ ] 9.8 Create GUI integration example (Python)
- [ ] 9.9 Update README with new features
