## ADDED Requirements

### Requirement: Run Plan Resolved Once Before Any Engine Code

A run SHALL be described by one immutable plan that is computed once,
before any engine code executes, and that holds every resolved choice:
engine and route reasons, time window, step and tolerance values,
Newton and event options, initial condition, composed gate drives,
tick handlers, source terms, requested features, record options,
preflight ties and diagnostics. Every option SHALL be resolved exactly
once (explicit value, options bundle, netlist, or default), and the
engines SHALL read the plan and nothing else.

#### Scenario: Every simulate() kwarg is resolved exactly once before any engine code runs
- **WHEN** `simulate()` is called with any combination of keywords
- **THEN** the plan's `resolved_from` mapping has exactly one entry for
  every parameter of `simulate()` other than the circuit
- **AND** no engine function has been entered when the plan is returned

#### Scenario: Engine code has no other input
- **WHEN** the engine entry functions are inspected
- **THEN** each takes the plan as its only argument and references no
  `simulate()` keyword

#### Scenario: Planning does not run a solver
- **WHEN** a plan is built for a circuit
- **THEN** no factorisation and no solve has been performed and the
  circuit object passed in is unchanged

### Requirement: Cross-Cutting Behaviour Applies To Every Engine

Cross-cutting behaviour SHALL be performed by engine-independent code:
initial-condition synthesis, snapshot validation, preflight
regularisation, the undriven-switch policy, composition of gate drives
and observers, the voltage-sanity check and result decoration. The same
circuit and options
SHALL either produce the same cross-cutting behaviour on every engine
or be refused by name on the engines that cannot honour it.

#### Scenario: Builder initial conditions on every engine
- **GIVEN** an inductor declared with `i0 = 5.0`
- **WHEN** the circuit runs on the fixed, the variable-step and the
  event-driven engine
- **THEN** the first sample of the inductor current is 5.0 A on each,
  or the engine refuses the run with an error naming the initial
  condition

#### Scenario: Invalid snapshot refused on every engine
- **WHEN** a run is resumed from an invalid snapshot on any engine
- **THEN** the same error type is raised before the run starts

#### Scenario: Voltage sanity on every engine
- **GIVEN** a circuit that produces an implausible node voltage
- **WHEN** it runs on any engine
- **THEN** the result carries the implausible-voltage diagnostic

### Requirement: Kernel Simulation Object

The kernel SHALL provide a `Simulation` object constructed from a
circuit and run options that plans the run and exposes `plan()`,
`run()`, `run_until(t)`, `snapshot()`, `restore()` and `diagnostics()`.
It SHALL be usable from C++ without Python and SHALL build the device
stamps itself: no caller-supplied callback is required for any
supported device to be simulated.

#### Scenario: C++ run with a Newton device and no callback
- **GIVEN** a C++ program that builds a buck with a Shockley diode
- **WHEN** it constructs `Simulation{circuit, options}` and calls
  `run()`
- **THEN** the result equals the result of the Python `simulate()` call
  with the same options, bit for bit

#### Scenario: Split run equals continuous run
- **GIVEN** a fixed-step simulation
- **WHEN** it is run with `run_until(t1)`, snapshotted, restored into a
  new `Simulation` and run to `t_end`
- **THEN** the concatenated trace is bit-identical to one `run()`

#### Scenario: Attachments are composed by the object
- **WHEN** two gate sources owning different switches and two tick
  handlers with different rates are attached
- **THEN** the run honours all four on every engine that supports
  ticks, with no composition code in the caller

### Requirement: Single Options Structure With One Defaults Table

Run options SHALL be one nested structure in which "not set" is an
empty optional, never a sentinel value, and every default SHALL be
defined in one place that the kernel, the Python signature and the
netlist loader all read. An option that the selected engine cannot
honour SHALL cause a refusal; it MUST NOT be accepted and ignored.

#### Scenario: One default per option
- **WHEN** the default relative and absolute tolerances are read from
  the kernel options, from `simulate()` and from a loaded netlist
- **THEN** the three agree for the same step strategy

#### Scenario: Zero is a value
- **WHEN** an option is set to `0` where zero is a meaningful value
- **THEN** it is used as zero, not replaced by the default

#### Scenario: Ignored option is refused
- **WHEN** a fixed-step-only option is set and the plan selects the
  variable-step engine explicitly
- **THEN** an `UnsupportedError` naming the option and the engine is
  raised

### Requirement: Typed Error Hierarchy

Every failure the kernel reports SHALL be an instance of a class
derived from `PulsimError`: `NetlistError` for a circuit or parameter
fault (naming the device, node or field), `OptionsError` for a run
option, `UnsupportedError` carrying `engine`, `feature`, `device` and
`kind`,
`ConvergenceError` and `SingularError` for a run that stopped (both
carrying the partial result and the time), and `Cancelled`. A refusal
MUST NOT be reported as a bare builtin exception.

#### Scenario: Unsupported combination is typed and complete
- **GIVEN** a circuit with a saturable inductor
- **WHEN** it is run on the event-driven engine
- **THEN** `UnsupportedError` is raised with `engine`, `device` and
  `feature` set, listing every refusal, not only the first

#### Scenario: Aborted run keeps its partial result
- **WHEN** Newton fails at time `t`
- **THEN** `ConvergenceError` is raised with `t`, the iteration count,
  the worst row's name and the partial trace up to the last accepted
  step

#### Scenario: A caller can let refusals through
- **WHEN** application code wraps a run in
  `except PulsimError: raise`
- **THEN** every kernel refusal passes that handler and no refusal is
  an untyped `ValueError`, `RuntimeError` or `NotImplementedError`

### Requirement: Structured Diagnostics

Every non-error message of the kernel SHALL be a `Diagnostic` with
severity, code, message, device and time, collected in order on the
plan and on the result. No diagnostic SHALL
exist only as a printed warning or as an ad-hoc attribute.

#### Scenario: Every warning is in the list
- **WHEN** a run emits warnings (regularising ties, forced accepts,
  gate under-sampling, implausible voltage)
- **THEN** each appears once in `result.diagnostics` with a stable code
- **AND** the Python layer has emitted exactly one `warnings.warn` per
  warning-severity diagnostic

#### Scenario: Diagnostics survive without a warning filter
- **WHEN** the caller runs with warnings silenced
- **THEN** `result.diagnostics` is unchanged

### Requirement: One Undriven-Switch Policy

A controlled switch that no gate source drives SHALL be handled by one
named policy (`Refuse`, `Open` or `Closed`) that is part of the run
options and behaves identically on every engine. The default policy
SHALL NOT differ between engines.

#### Scenario: Same behaviour on every engine
- **GIVEN** a half bridge with no gate drive
- **WHEN** it runs on each engine with the default policy
- **THEN** every engine gives the same outcome (the same refusal naming
  both switches, or the same mask)

#### Scenario: Explicit policy is honoured
- **WHEN** the policy is set to `Open`
- **THEN** both switches are open for the whole run on every engine and
  no diagnostic of severity error is produced

### Requirement: Routing Is Reported And Never Silent

The plan SHALL record the engine asked for, the engine selected and the
reasons for any difference, and the result SHALL expose them. With
`engine='auto'`, an explicit `dt` SHALL select the fixed step; without
it the variable-step engine SHALL be selected when no capability
refusal applies, and otherwise the run SHALL be refused with the
reasons.

#### Scenario: Auto with dt
- **WHEN** `simulate(circuit, t_end, dt=1e-6)` runs
- **THEN** `engine_used` is the fixed engine and the route reason says
  an explicit `dt` requests a fixed step

#### Scenario: Auto without dt and a fixed-only feature
- **WHEN** a fixed-only feature is requested with no `dt`
- **THEN** the run is refused with an error listing the feature and
  advising to pass `dt`
