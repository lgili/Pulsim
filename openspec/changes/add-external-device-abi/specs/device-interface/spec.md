## ADDED Requirements

### Requirement: External Device C ABI

The kernel SHALL load a Newton device from a shared library through a
versioned C descriptor that declares the device name, its pin and
internal-unknown counts, a parameter field table (name, unit, default,
bounds), fixed sparsity patterns for the Jacobians, and an `eval`
function returning the residual contributions `f` and `q` and their
Jacobians `∂f/∂x` and `∂q/∂x`. The row convention, the meaning of `f`
and `q`, and the time formulas SHALL be those of the device interface
for built-in Newton devices; an external device SHALL NOT supply its
own time discretisation. Optional descriptor entries SHALL provide
parameter validation, state initialisation and acceptance, event
functions and named outputs. No rebuild of Pulsim SHALL be needed to
add a device.

#### Scenario: External Shockley diode equals the built-in
- **GIVEN** a shared library implementing the Shockley law of the
  built-in diode (including its linear extension above `i_lim` and
  `G_min`) and a 20 V half-wave rectifier
- **WHEN** the circuit runs on `pwl` with the external device and with
  `add_shockley_diode`, and the DC operating point is computed for both
- **THEN** every node voltage agrees within 1e-12 relative

#### Scenario: Charge-only device
- **GIVEN** an external device with `q = C0·V0/(1−m)·[(1+v/V0)^(1−m) − 1]`,
  `C0 = 2 nF`, `V0 = 25 V`, `m = 0.5`, and no `f`
- **WHEN** it replaces `add_nonlinear_capacitor` in a 0 → 400 V charge
- **THEN** the node voltages agree within 1e-12 relative and the charge
  delivered is 312.31 nC

#### Scenario: Internal unknown
- **GIVEN** an external diode with one internal charge unknown
  implementing the Lauritzen equations
- **WHEN** it runs a reverse-recovery transient beside the built-in
- **THEN** the currents agree within 1e-10 relative

### Requirement: External Device in Every Analysis

An external device SHALL be solved by the same Newton engines, DC
operating point and small-signal analysis as a built-in Newton device,
SHALL keep its state in the kernel's device-state vector so that
snapshots, rejected steps and steady-state solvers restore it, and
SHALL be refused by the conformance error on an engine that does not
solve Newton devices.

#### Scenario: Small-signal conductance
- **GIVEN** the external Shockley diode biased at 1 A
- **WHEN** the AC analysis is run
- **THEN** its small-signal resistance is `n·V_T/(I + I_S)` within 1e-9
  relative

#### Scenario: Variable-step engine
- **WHEN** the half-wave rectifier runs on `trbdf2` with the external
  and with the built-in diode
- **THEN** node voltages agree within 1e-9 relative

#### Scenario: Unsupported engine
- **WHEN** a circuit with an external device is run on `dsed`
- **THEN** it raises the conformance error naming the device and the
  engine

### Requirement: External Device Registration Checks

Registration SHALL refuse, with an error naming the device and the
reason: an ABI version mismatch, a descriptor without `eval`, a
sparsity-pattern index outside the device's rows, a duplicate pattern
entry, a parameter outside its declared bounds, a non-finite value
returned at the test points, and a supplied Jacobian that disagrees with
central differences of `f` and `q`. A non-finite value returned during a
run SHALL be reported as a solver failure naming the device.

#### Scenario: Wrong Jacobian refused
- **GIVEN** a library whose `∂f/∂x` entry is 1 % too large
- **WHEN** the device is added to a circuit
- **THEN** registration fails naming the device and the pattern entry

#### Scenario: ABI version mismatch
- **WHEN** a library built against another ABI version is loaded
- **THEN** loading fails naming the library and both versions

### Requirement: Native Code Trust Boundary

Native code SHALL be loaded only at the caller's explicit request. A
YAML netlist that declares an external device SHALL be refused unless
the load call passes `allow_native_code=True`. Compiled libraries SHALL
be cached in a directory private to the current user, and a cached file
not owned by that user SHALL NOT be loaded.

#### Scenario: Netlist with native code
- **WHEN** a netlist containing `type: external_device` is loaded
  without `allow_native_code=True`
- **THEN** loading fails with an error that names the device and the
  option

#### Scenario: Foreign cache entry
- **GIVEN** a file with the expected cache name that is owned by
  another user
- **WHEN** the same source is compiled
- **THEN** the file is rebuilt and the foreign file is not loaded

### Requirement: Device from Declared Equations

Python SHALL build an external device from symbolic expressions for `f`
and `q` over declared pins, internal unknowns and parameters, deriving
the Jacobians and their sparsity patterns symbolically and generating
and compiling C for the external-device ABI. Expressions that are not
continuously differentiable SHALL be refused with the offending
sub-expression unless the caller explicitly allows them.

#### Scenario: Declared Shockley equals the built-in
- **GIVEN** `f = [i, −i]` with `i = I_S·(exp((v_a − v_k)/(n·V_T)) − 1)`
  declared symbolically
- **WHEN** the compiled device replaces the built-in in the half-wave
  rectifier, below the built-in's current limit
- **THEN** node voltages agree within 1e-12 relative

#### Scenario: Non-smooth expression refused
- **WHEN** an expression containing `Abs` or `Piecewise` is compiled
  without `allow_nonsmooth=True`
- **THEN** compilation fails naming the sub-expression

### Requirement: Compiled Verilog-A Host

The kernel SHALL load a device compiled from Verilog-A through the OSDI
interface, mapping its resistive residual and Jacobian to `f` and its
reactive residual and Jacobian to `q`. A descriptor feature the host
does not support SHALL be refused by name.

#### Scenario: Diode model against ngspice
- **GIVEN** one Verilog-A diode compiled to an OSDI library and used in
  the same circuit in ngspice and in Pulsim
- **WHEN** both run
- **THEN** node voltages agree within 1e-6 relative

#### Scenario: Unsupported feature
- **WHEN** a library requires a feature the host does not implement
- **THEN** loading fails naming the feature
