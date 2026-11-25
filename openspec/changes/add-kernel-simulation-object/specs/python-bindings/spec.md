## ADDED Requirements

### Requirement: simulate() Is A Thin Wrapper Over The Kernel Plan

`pulsim.simulate()` SHALL keep its signature and SHALL do no more than
convert its arguments to run options, construct the kernel
`Simulation`, attach Python callbacks through the attachment
interface, run, and return the result. It MUST NOT contain
engine-specific branches. Results and warnings for every call that was
valid before this change SHALL be unchanged, except where this change
lists a breaking item.

#### Scenario: Results unchanged through the wrapper
- **WHEN** the reference harness and the test suite run through the
  wrapper
- **THEN** fixed-engine results are bit-identical to the previous
  build and the set of emitted warnings is equal

#### Scenario: No engine branch in the wrapper
- **WHEN** the body of `simulate()` is inspected
- **THEN** it contains no comparison against an engine name and is at
  most 150 lines excluding the docstring

#### Scenario: Fixed overhead does not rise
- **WHEN** a 100-step run of a five-node circuit is timed before and
  after
- **THEN** the wall time of `simulate()` is not above the previous
  value (best of ten, same machine)

### Requirement: Python Exception Classes Mirror The Kernel Hierarchy

The package SHALL export `PulsimError`, `NetlistError`, `OptionsError`,
`UnsupportedError`, `ConvergenceError`, `SingularError`,
`SimulationAborted` and `Cancelled`. For one major version each class
SHALL also derive from the builtin exception that the same failure
raised before (`ValueError`, `NotImplementedError`, `RuntimeError`,
`KeyError`), so that existing handlers keep working.

#### Scenario: Old handlers still catch
- **WHEN** code written for 2.0 catches `ValueError` around a refused
  engine / feature combination
- **THEN** it still catches the error, which is also an
  `UnsupportedError`

#### Scenario: New handlers are precise
- **WHEN** code catches `UnsupportedError`
- **THEN** it does not catch a `NetlistError` raised for a bad
  parameter value

### Requirement: Simulation Object And Run Options Binding

The package SHALL expose `pulsim.Simulation` and `pulsim.RunOptions`
with the kernel's methods and fields, including `plan`, `run`,
`run_until`, `snapshot`, `restore`, `diagnostics`, `add_gate`,
`add_tick` and `add_source_term`. A run with no Python callback
attached SHALL release the GIL for its whole duration.

#### Scenario: Object API equals function API
- **WHEN** the same circuit and options are run through
  `pulsim.Simulation(...).run()` and through `simulate()`
- **THEN** the two results are bit-identical

#### Scenario: Thread sweep scales
- **GIVEN** eight runs with native gate sources
- **WHEN** they run on four threads
- **THEN** the wall time is at most 0.4 of the sequential time and the
  results equal the sequential ones

### Requirement: Deprecated Raw Transient Entry Points

The raw transient bindings SHALL remain callable for one release:
`run_transient` and `run_transient_trbdf2`, with the option classes
`SimulationOptions` and `TrBdf2Options`. They SHALL emit a
`DeprecationWarning` naming `Simulation`, and SHALL build their device
stamps in the kernel so that no call simulates a Newton device as an
open circuit.

#### Scenario: Raw call with a Newton device
- **WHEN** the raw `run_transient` binding is called on a Shockley
  clamp with default arguments during the deprecation release
- **THEN** it emits a `DeprecationWarning` and the clamp voltage is
  0.594 V
