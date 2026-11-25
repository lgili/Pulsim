## ADDED Requirements

### Requirement: Device-Kind Capability Table

The kernel SHALL hold one compile-time table with one row per
`DevicePool::StoredKind`, stating the kind's name, its solve class
(linear, source, switch, PWL diode, static Newton, stateful Newton) and
the set of engines (fixed-step trapezoidal, TR-BDF2, DSED, BDF1) that
simulate it. The build SHALL fail when the table and the device variant
differ in length. Every decision of the form "does this circuit need
Newton" or "can this engine run this circuit" SHALL be derived from the
table; no other per-kind list SHALL exist for that purpose. The table
SHALL be readable from Python as `pulsim.engine_capabilities()`.

#### Scenario: Every kind has a row
- **GIVEN** a builder holding one instance of each of the 20 device
  kinds
- **WHEN** `pulsim.engine_capabilities()` is read
- **THEN** it has an entry for the kind of every branch, and each entry
  names its solve class and a boolean for each of `pwl`, `trbdf2`,
  `dsed`, `bdf1`

#### Scenario: The junction capacitor is a Newton kind
- **WHEN** a pool holding only a resistor, a source and a
  `NonlinearCapacitor` is queried
- **THEN** `requires_newton()` is true, and the table marks the kind as
  unsupported on `dsed` and `bdf1`

#### Scenario: A new kind without a row does not build
- **WHEN** a device alternative is added to the pool variant and no row
  is added to the table
- **THEN** compilation fails on a `static_assert`

### Requirement: Typed Unsupported Refusal

A refusal SHALL be typed: an engine that cannot simulate a device kind,
and an engine or analysis that cannot honour a requested feature, SHALL
raise `pulsim::UnsupportedError` (C++, derived from `std::invalid_argument`),
surfaced in Python as `pulsim.UnsupportedError`, a subclass of
`ValueError`. The exception SHALL carry `engine`, `feature`, `device`
(the branch name, empty when the refusal is not about one device) and
`kind`. The refusal SHALL be raised by the kernel driver itself, so that
the raw bindings and C++ callers receive it as well as `simulate()`.

#### Scenario: Junction capacitor refused on the event-driven engine
- **GIVEN** a 10 V source, 1 kΩ, a 100 nF capacitor and a junction
  capacitor `Cn` (`C0 = 1 µF`, `V0 = 1 V`, `m = 0.5`) on the same node
- **WHEN** `simulate(b, 1e-4, engine='dsed')` is called
- **THEN** it raises `pulsim.UnsupportedError` with `engine == 'dsed'`,
  `device == 'Cn'` and `kind == 'NonlinearCapacitor'`, and the same
  circuit run through `run_transient_bdf1` raises it with
  `engine == 'bdf1'`

#### Scenario: Existing handlers keep working
- **WHEN** the refusal above is raised inside
  `try: ... except ValueError:`
- **THEN** the handler catches it

#### Scenario: Extractor refusals keep their legacy base
- **WHEN** `simulate(engine='dsed')` is called on a circuit with coupled
  inductors
- **THEN** the exception is an `UnsupportedError` and is also an
  instance of `NotImplementedError`

### Requirement: No Silent Device Cell

For every device kind and every engine, a transient run SHALL either
agree with the reference engine or raise `UnsupportedError` naming the
engine and the device. Agreement SHALL mean: at 20 fixed instants, the
observable interpolated in time differs from the reference by at most
1 % of the reference observable's span. The reference SHALL be the
fixed-step engine at a step for which halving the step moves the
observable by less than 0.25 % of the span. Each test circuit SHALL be
shown to depend on its device: removing the device SHALL move the
observable by at least 10 % of the span on the reference engine.

#### Scenario: A supported cell agrees
- **GIVEN** the junction-capacitor circuit of the previous requirement
- **WHEN** it runs on `pwl` and on `trbdf2`
- **THEN** `v(100 µs)` is 1.02 V within 0.1 V on both, and 6.30 V
  within 0.1 V when `Cn` is removed

#### Scenario: An unsupported cell refuses
- **WHEN** the same circuit is run on `dsed` or `bdf1`
- **THEN** no result is returned and `UnsupportedError` is raised

#### Scenario: Any other outcome fails the suite
- **WHEN** a cell returns a result that disagrees with the reference, or
  raises an exception that is not `UnsupportedError`, or refuses a kind
  the table marks as supported
- **THEN** the conformance test fails and names the kind and the engine

### Requirement: No Silent Feature Cell

Each feature keyword SHALL be honoured or refused on every engine: for
every `simulate()` keyword that selects a feature and every engine, the
run SHALL either honour the feature with the same observable effect
as the reference engine or raise `UnsupportedError` whose `feature` is
the keyword name. A warning SHALL NOT stand in for a refusal when the
feature changes the simulated system.

#### Scenario: C blocks on the event-driven engine
- **GIVEN** a builder with a registered C block
- **WHEN** `simulate(b, t_end, engine='dsed')` is called
- **THEN** it raises `UnsupportedError` with `feature == 'c_blocks'`

#### Scenario: Fixed-step-only keywords
- **WHEN** `store_every=4` or `start_from_dc_op=True` is passed with
  `engine='trbdf2'` or `engine='dsed'`
- **THEN** each call raises `UnsupportedError` naming the keyword and
  the engine

#### Scenario: A feature both engines honour gives the same count
- **GIVEN** an observer with `controller_period = 100 µs` on a 2 ms run
- **WHEN** it runs on `trbdf2`
- **THEN** the observer is called 20 times
