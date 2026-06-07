## ADDED Requirements

### Requirement: Block Definition and Circuit Wiring
The system SHALL provide a custom-code block ("C block") with a
user-chosen number of inputs and outputs. Each input SHALL be a wire that
reads a circuit signal (a node voltage or a branch current); each output
SHALL be a wire that drives a controlled source (voltage or current)
inserted into the circuit between two user-specified nodes. Input wires
SHALL be resolved to state-vector locations when the block is added.

#### Scenario: Add a block with two inputs and one output
- **WHEN** a user adds a c-block with inputs `[("v","vout"), ("i","L1")]` and output `[("v","ctrl","gnd")]`
- **THEN** the block reads `V(vout)` and `I(L1)` each block step and drives a controlled voltage source between `ctrl` and `gnd` with the block's output value

#### Scenario: Input reads a branch current
- **WHEN** an input wire is `("i", "L1")`
- **THEN** the value presented to the block for that input equals the simulated current through `L1` at the block step

#### Scenario: Output drives the circuit
- **WHEN** the block sets output 0 to a value `u`
- **THEN** the controlled source bound to output 0 imposes `u` on the circuit until the next block step

### Requirement: Multi-Language Step Function
The block's step logic SHALL be expressible in Python, C, or C++ behind a
single logical interface `step(t, dt, inputs, outputs, state)`. Python
blocks SHALL accept a Python callable. C and C++ blocks SHALL use a fixed
`extern "C"` ABI so the same source/library works for both.

#### Scenario: Python callable block
- **WHEN** the user passes a Python function `fn(t, dt, inp, out, state)` that sets `out[0] = 0.5 * inp[0]`
- **THEN** each block step computes `out[0]` from the current `inp[0]` and drives the bound output

#### Scenario: Identical result across languages
- **WHEN** the same gain law is implemented as a Python callable, a compiled C library, and inline C++ source
- **THEN** the simulated circuit response is numerically identical (to solver tolerance) for all three

### Requirement: Configurable Sample Time with Zero-Order Hold
The block SHALL run at a user-chosen timestep `dt_block`. The step
function SHALL be invoked at `t = 0` and at each subsequent multiple of
`dt_block`; outputs SHALL be held constant (zero-order hold) between
invocations. `dt_block` SHALL default to the simulation `dt`, and the
system SHALL warn and clamp if `dt_block < dt`.

#### Scenario: Output is piecewise-constant at the block rate
- **WHEN** `dt_block = 10 · dt` and the block output changes each block step
- **THEN** the injected signal is piecewise-constant, updating only every 10 simulation steps

#### Scenario: Sub-dt sample time is clamped
- **WHEN** the user sets `dt_block` smaller than the simulation `dt`
- **THEN** the system emits a warning and runs the block once per simulation step

### Requirement: Compiled Shared-Library Blocks
The system SHALL load a user-compiled shared library (`.so`/`.dll`)
exporting the C-block ABI and call it each block step via `ctypes`,
without rebuilding the Pulsim kernel. The library MAY export optional
`init`/`term` entry points for per-block state allocation and teardown.

#### Scenario: Load and run a shared library
- **WHEN** the user passes `lib="block.so"` exporting `pulsim_cblock_step`
- **THEN** the block calls that symbol each block step, marshalling inputs/outputs as `double` buffers

#### Scenario: State lifecycle
- **WHEN** the library exports `pulsim_cblock_init`/`pulsim_cblock_term`
- **THEN** `init` is called once when the block is added and `term` once when the simulation result is released

### Requirement: Inline-Source Compilation
The system SHALL accept inline C or C++ source for the block, wrap it in
the ABI, compile it to a temporary shared library with the system
compiler, cache the artifact by source/flags hash, and load it via the
shared-library path. When no compiler is available the system SHALL raise
a clear error directing the user to the precompiled-library option.

#### Scenario: Inline C source compiles and runs
- **WHEN** the user passes `code="out[0] = 0.5*in[0];", lang="c"`
- **THEN** the source is compiled once, cached, and executed each block step

#### Scenario: Cached rebuild is skipped
- **WHEN** the same inline source and flags are used again
- **THEN** the cached shared library is reused without recompilation

#### Scenario: Missing compiler
- **WHEN** inline source is requested but no C/C++ compiler is found
- **THEN** the system raises an error explaining how to supply a precompiled `lib=` instead

### Requirement: Engine Support and Output Injection
Block outputs SHALL be injected into the circuit on the fixed-step (PWL)
engine via residual injection into the bound controlled sources. A block
with inputs only (no outputs) SHALL additionally be usable under the
variable-step (DSED) engine.

#### Scenario: Outputs require the PWL engine
- **WHEN** a block with at least one output is simulated with the PWL engine
- **THEN** the outputs are injected each step and affect the solution

#### Scenario: Inputs-only block under DSED
- **WHEN** a block has no outputs (e.g. logging or an observer) and the simulation uses the DSED engine
- **THEN** the block still reads its inputs each step without error

### Requirement: Persistent Block State
The block SHALL retain state across steps so stateful logic (integrators,
filters, counters, state machines) works. Python blocks SHALL receive a
mutable `state` object; C/C++ blocks SHALL receive their opaque `state`
pointer.

#### Scenario: Integrator accumulates
- **WHEN** a block integrates its input (`state += inp[0]*dt; out[0] = state`)
- **THEN** the output reflects the accumulated value across block steps
