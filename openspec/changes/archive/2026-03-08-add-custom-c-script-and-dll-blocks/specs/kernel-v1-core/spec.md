## ADDED Requirements

### Requirement: C_BLOCK Type in SignalEvaluator
The `SignalEvaluator` SHALL recognise `"C_BLOCK"` as a valid signal-domain
component type, include it in `SIGNAL_TYPES`, and evaluate it in topological order
each timestep.  During `build()`, the evaluator MUST initialise a `CBlockLibrary`
or `PythonCBlock` controller for each `C_BLOCK` component and store it in the
internal controller registry.  During `step(t)`, the evaluator MUST call
`ctl.step(t, dt, inputs)` where `dt` is the elapsed time since the previous
accepted step.  The evaluator MUST call `ctl.reset()` for all C-Block controllers
in its `reset()` method.

#### Scenario: C_BLOCK correctly transforms signal in a simple chain
- **GIVEN** a `SignalEvaluator` built from a circuit dict with:
  `CONSTANT(value=4.0) → C_BLOCK(gain×2) → PWM_GENERATOR`
- **WHEN** `evaluator.step(t=1e-3)` is called
- **THEN** the PWM component receives duty `8.0` (clamped to `1.0` by the evaluator)
- **AND** no exceptions are raised

#### Scenario: `dt` passed to C-Block reflects elapsed accepted timestep
- **GIVEN** a `PythonCBlock` that records the `dt` argument it receives
- **AND** an evaluator whose `step` is called at `t=0`, `t=1e-6`, `t=3e-6`
- **WHEN** the three calls are made
- **THEN** `dt` values recorded are `0.0`, `1e-6`, `2e-6` respectively

#### Scenario: `build()` fails with AlgebraicLoopError for self-referencing C_BLOCK
- **GIVEN** a circuit dict where the single output of a `C_BLOCK` feeds back into
  its only input with no delay
- **WHEN** `evaluator.build()` is called
- **THEN** `AlgebraicLoopError` is raised
- **AND** the C_BLOCK's id appears in `exc.cycle_ids`

#### Scenario: `reset()` reinitialises all C-Block controllers
- **GIVEN** a `SignalEvaluator` with two `C_BLOCK` components, both stateful
- **WHEN** `evaluator.reset()` is called after 50 steps
- **THEN** both C-Block controllers have their state reset
- **AND** the next `step()` call produces the same output as the very first step

### Requirement: C_BLOCK Compile-at-Build Support in SignalEvaluator
The `SignalEvaluator` MUST support automatic compilation of C-Block source files.
When a `C_BLOCK` component specifies `source` (a path to a `.c` file) instead of
`lib_path`, the evaluator MUST call `compile_cblock(source, ...)` during `build()`
and load the resulting shared library.  If compilation fails, `build()` SHALL raise
`CBlockCompileError` immediately.  The compiled library path MUST be cached and
SHALL NOT be recompiled on subsequent `build()` calls unless the source modification
time has changed.

#### Scenario: Source-based C_BLOCK compiles and runs during build
- **GIVEN** a circuit dict with `C_BLOCK` specifying `source: "gain.c"` and the
  file exists with valid C code; and a C compiler is available
- **WHEN** `evaluator.build()` is called
- **THEN** `gain.c` is compiled automatically
- **AND** `evaluator.step(t)` executes the compiled block without errors

#### Scenario: Source-based C_BLOCK compilation failure surfaces in build
- **GIVEN** a circuit dict with `C_BLOCK` specifying `source: "broken.c"` that
  has a syntax error
- **WHEN** `evaluator.build()` is called
- **THEN** `CBlockCompileError` is raised with the compiler stderr

#### Scenario: Source compilation is skipped if cached library is fresh
- **GIVEN** a `C_BLOCK` with a previously compiled library that is newer than the
  source file
- **WHEN** `evaluator.build()` is called a second time
- **THEN** the compiler subprocess is NOT invoked (cache hit)

### Requirement: C_BLOCK Multi-Output Vector and SIGNAL_DEMUX Integration
When a `C_BLOCK` has `n_outputs > 1`, the evaluator MUST store the full output
`list[float]` as the component state.  A downstream `SIGNAL_DEMUX` wired from
the C_BLOCK's output MUST receive the vector and distribute individual scalar
values by pin index.  This MUST happen in a single evaluation round (no extra DAG
passes).

#### Scenario: 3-output C_BLOCK distributes values via SIGNAL_DEMUX
- **GIVEN** a `C_BLOCK` with `n_outputs=3` returning `[1.1, 2.2, 3.3]`
  wired to a `SIGNAL_DEMUX` with pins `OUT1`, `OUT2`, `OUT3`
- **WHEN** the evaluator runs one step
- **THEN** the demux output pins carry values `1.1`, `2.2`, `3.3` respectively
- **AND** downstream consumers of `OUT2` see exactly `2.2`
