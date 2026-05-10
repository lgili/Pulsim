## ADDED Requirements

### Requirement: Custom Device Registration from Python
Python bindings SHALL expose `pulsim.register_device(name, residual_fn, num_pins, params_schema, jacobian_fn=None)` allowing users to define custom devices in pure Python.

#### Scenario: Register a JFET-style custom device
- **GIVEN** a Python callable `jfet_residual(x_terminals, t, dt, params) -> residual_vec`
- **WHEN** `pulsim.register_device("MyJFET", jfet_residual, num_pins=3, params_schema={...})` is called
- **THEN** the device type becomes available for use in subsequent `Circuit` construction
- **AND** the device participates in topology signature, telemetry, and parity workflows

#### Scenario: Custom device with user-supplied Jacobian
- **GIVEN** a registration that includes `jacobian_fn`
- **WHEN** the kernel stamps the device during a Newton iteration
- **THEN** the user's `jacobian_fn` is invoked instead of AD
- **AND** validation can run AD vs supplied-Jacobian agreement at startup

#### Scenario: Custom device without Jacobian uses AD propagation
- **GIVEN** a registration without `jacobian_fn`
- **WHEN** the kernel needs a Jacobian
- **THEN** the binding evaluates the residual at perturbed inputs to derive a numerical Jacobian
- **AND** the user is informed in documentation that this path is intended for prototyping

### Requirement: Jacobian Validation Surface in Python
Python bindings SHALL expose `Simulator.validate_jacobians(operating_points)` returning a structured result identifying any device whose AD or manual Jacobian disagrees with FD reference.

#### Scenario: Programmatic validation
- **WHEN** Python code calls `sim.validate_jacobians([{"x": ..., "t": 0.0}])`
- **THEN** the call returns a list of `(device_name, terminal_pair, max_delta)` for any failures
- **AND** an empty list indicates all devices pass

### Requirement: AD Mode Toggle from Python
Python bindings SHALL expose `SimulationOptions.use_manual_jacobian: bool` (default `False`) allowing per-run override of the AD vs manual Jacobian path.

#### Scenario: Force manual path from Python
- **GIVEN** `options.use_manual_jacobian = True`
- **WHEN** the simulation runs
- **THEN** devices invoke their legacy manual stamps
- **AND** a deprecation warning is logged once
