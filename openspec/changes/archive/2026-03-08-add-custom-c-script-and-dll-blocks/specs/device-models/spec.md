## ADDED Requirements

### Requirement: C-Block ABI Header
The system SHALL provide a versioned, stable C ABI header
`core/include/pulsim/v1/cblock_abi.h` that defines the interface contract for
user-defined signal-domain blocks compiled as shared libraries.  The header MUST
define: `PULSIM_CBLOCK_ABI_VERSION` (integer macro), `PulsimCBlockInfo` (struct),
`PulsimCBlockCtx` (opaque forward declaration), `pulsim_cblock_init_fn`,
`pulsim_cblock_step_fn`, and `pulsim_cblock_destroy_fn` (function pointer
typedefs).  `pulsim_cblock_step_fn` MUST be the only required export; `init` and
`destroy` are optional.

#### Scenario: Minimal C-Block compiles and loads
- **GIVEN** a C source file exporting only `pulsim_cblock_step` with signature matching `pulsim_cblock_step_fn`
- **WHEN** compiled with `compile_cblock()` and loaded with `CBlockLibrary`
- **THEN** the library loads without error
- **AND** `step()` can be called with `n_inputs` doubles and returns `n_outputs` doubles

#### Scenario: ABI version mismatch is rejected
- **GIVEN** a shared library that exports `pulsim_cblock_abi_version` returning a value different from `PULSIM_CBLOCK_ABI_VERSION`
- **WHEN** `CBlockLibrary` initialises
- **THEN** a `CBlockABIError` is raised
- **AND** the error message SHALL state the expected and found version numbers

#### Scenario: Optional lifecycle hooks called correctly
- **GIVEN** a C-Block that exports `pulsim_cblock_init` and `pulsim_cblock_destroy`
- **WHEN** `CBlockLibrary` is constructed, `step` is called multiple times, and then the library is released
- **THEN** `init` is called exactly once before the first `step`
- **AND** `destroy` is called exactly once at release

### Requirement: C-Block Signal-Domain Component
The simulation system SHALL support a `C_BLOCK` signal-domain component type that
accepts `n_inputs` floating-point signals as inputs and produces `n_outputs`
floating-point signals as outputs, evaluated once per accepted simulation timestep
in topological order within the `SignalEvaluator` DAG.  The block MUST support
persistent state between steps via a context pointer managed by the user's `init`
and `destroy` hooks.  The C-Block SHALL participate in algebraic-loop detection
identically to other signal blocks.

#### Scenario: Single-input single-output C-Block in a signal chain
- **GIVEN** a simulation with a `VOLTAGE_PROBE → C_BLOCK(gain) → PWM_GENERATOR` signal chain
- **WHEN** the simulation runs for 10 timesteps
- **THEN** the C-Block is evaluated exactly once per accepted timestep
- **AND** the PWM duty cycle reflects the C-Block output at each step

#### Scenario: Multi-output C-Block with SIGNAL_DEMUX
- **GIVEN** a `C_BLOCK` with `n_outputs=3` wired to a `SIGNAL_DEMUX` with 3 outputs
- **WHEN** the evaluator runs one step
- **THEN** each of the 3 demux outputs carries the correct scalar value from the C-Block output vector

#### Scenario: C-Block with persistent state (IIR filter)
- **GIVEN** a C-Block that implements a first-order IIR filter using a state variable allocated in `pulsim_cblock_init`
- **WHEN** the simulation runs 100 steps at dt=1e-6
- **THEN** the C-Block output converges toward the expected steady-state value
- **AND** calling `evaluator.reset()` reinitialises the filter state to zero

#### Scenario: C-Block algebraic loop detected
- **GIVEN** a circuit dict where a `C_BLOCK` output is wired back directly to its own input (no delay)
- **WHEN** `evaluator.build()` is called
- **THEN** `AlgebraicLoopError` is raised listing the C-Block in `cycle_ids`

#### Scenario: C-Block non-zero return code aborts simulation
- **GIVEN** a C-Block whose `step` function returns a non-zero error code under a specific input condition
- **WHEN** the simulation reaches that condition
- **THEN** the simulation stops
- **AND** a `CBlockRuntimeError` is raised with the return code and current simulation time

### Requirement: Python-Callable C-Block Fallback
The system SHALL support a `PythonCBlock` class that conforms to the same
interface as `CBlockLibrary` (`step()`, `reset()`, `n_inputs`, `n_outputs`) but
uses a Python callable instead of a compiled shared library.  `PythonCBlock` SHALL
be usable anywhere `CBlockLibrary` is accepted, enabling prototyping without a C
compiler.  The callable's persistent state SHALL be provided as a plain `dict`
context passed as the first argument.

#### Scenario: Python callable produces correct output
- **GIVEN** a `PythonCBlock` with `fn=lambda ctx, t, dt, inp: [inp[0] * 2.5]`
- **WHEN** `step(t=0.0, dt=1e-6, inputs=[4.0])` is called
- **THEN** the result is `[10.0]`

#### Scenario: Python callable stateful accumulation
- **GIVEN** a `PythonCBlock` with a function that accumulates input into `ctx["sum"]`
- **WHEN** `step` is called three times with inputs `[1.0]`, `[2.0]`, `[3.0]`
- **THEN** on the third call the output is `[6.0]`

#### Scenario: Python callable reset clears state
- **GIVEN** a stateful `PythonCBlock` after 5 steps
- **WHEN** `reset()` is called
- **THEN** `ctx` is an empty dict on the next `step` call
