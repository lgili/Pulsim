# python-bindings Specification

## Purpose
TBD - created by archiving change remove-cli-benchmark-dependency. Update Purpose after archive.
## Requirements
### Requirement: Runtime-Complete Simulation Objects
Python bindings SHALL expose runtime-complete simulation objects equivalent to v1 kernel execution controls, including `SimulationOptions` and `Simulator`.

#### Scenario: Configure and run simulation through class APIs
- **WHEN** Python code creates `SimulationOptions`, configures solver/integrator fields, and instantiates `Simulator`
- **THEN** the simulation executes with the configured options
- **AND** behavior matches kernel runtime semantics for those options

### Requirement: Periodic Steady-State API Exposure
Python bindings SHALL expose periodic steady-state methods supported by the v1 kernel.

#### Scenario: Run shooting method from Python
- **WHEN** Python code calls `Simulator.run_periodic_shooting(...)`
- **THEN** the method executes and returns structured periodic result fields (status, iterations, residuals)

#### Scenario: Run harmonic balance from Python
- **WHEN** Python code calls `Simulator.run_harmonic_balance(...)`
- **THEN** the method executes and returns structured harmonic-balance result fields

### Requirement: YAML Parser Exposure
Python bindings SHALL expose the v1 YAML parser interfaces needed to load netlists and simulation options with strict diagnostics.

#### Scenario: Parse YAML netlist through bound parser
- **WHEN** Python code loads a YAML netlist via exposed parser bindings
- **THEN** it receives parsed circuit and simulation options objects
- **AND** parser errors/warnings are accessible via Python

### Requirement: Structured Runtime Telemetry Exposure
Python bindings SHALL expose simulation result telemetry fields required by benchmark and validation workflows.

#### Scenario: Access telemetry after transient run
- **WHEN** Python code runs a transient simulation
- **THEN** result telemetry (iterations, rejections, runtime, solver telemetry) is available as structured fields
- **AND** can be consumed without parsing console text

### Requirement: Backward-Compatible Procedural API
Existing procedural entrypoints (`run_transient`, `dc_operating_point`, etc.) SHALL remain available during migration to class-based runtime APIs.

#### Scenario: Existing script uses procedural run API
- **WHEN** an existing Python script calls the procedural API
- **THEN** the script continues to run without mandatory migration in the same release window

### Requirement: Compatibility-Preserving Migration Surface
Python bindings SHALL preserve procedural API compatibility during the migration window while exposing canonical runtime surfaces for new development.

#### Scenario: Existing procedural script
- **WHEN** an existing script calls procedural entrypoints supported in the migration window
- **THEN** execution remains functional without mandatory rewrites
- **AND** deprecation guidance maps the call to canonical runtime APIs

#### Scenario: Canonical runtime usage
- **WHEN** new code uses canonical class-based runtime APIs
- **THEN** behavior matches the same underlying v1 kernel semantics as procedural compatibility paths
- **AND** telemetry parity is preserved

### Requirement: Structured Error and Failure Surface
Python bindings SHALL surface structured kernel diagnostics with stable reason codes and context fields.

#### Scenario: Invalid configuration from Python
- **WHEN** Python provides invalid simulation or solver configuration
- **THEN** bindings raise a structured error with deterministic reason code
- **AND** error context includes relevant field/path metadata

#### Scenario: Runtime failure propagation
- **WHEN** kernel execution terminates with a typed failure reason
- **THEN** Python receives the same reason code and terminal diagnostics
- **AND** no console-text parsing is required

### Requirement: Extension Introspection Surface
Python bindings SHALL expose read-only introspection for registered device, solver, and integrator capabilities.

#### Scenario: Query available runtime capabilities
- **WHEN** Python requests registered extension capabilities
- **THEN** bindings return structured metadata for available devices/solvers/integrators
- **AND** reported capabilities match active kernel registry state

### Requirement: KPI Telemetry Contract for Tooling
Python simulation results SHALL expose KPI-critical telemetry fields required by benchmark and regression tooling.

#### Scenario: Collect KPI telemetry from Python run
- **WHEN** benchmark tooling runs scenarios through Python bindings
- **THEN** result objects include fields needed for convergence, accuracy, runtime, event, and fallback KPIs
- **AND** the field schema is stable for automated consumers

### Requirement: Typed Electrothermal Characterization Bindings
Python bindings SHALL expose typed structures for datasheet-grade loss characterization and thermal-network configuration.

#### Scenario: Configure datasheet characterization from Python
- **WHEN** Python code configures semiconductor loss and thermal-network structures via typed bindings
- **THEN** runtime receives equivalent backend configuration without requiring YAML-only pathways
- **AND** invalid assignments fail with deterministic typed errors

### Requirement: Canonical Electrothermal Channel Metadata in Python
Python simulation results SHALL expose canonical loss and thermal channels with structured metadata sufficient for frontend routing.

#### Scenario: Frontend adapter reads channels via Python
- **WHEN** Python tooling enumerates `result.virtual_channels` and metadata
- **THEN** it can identify electrothermal channels by metadata fields (domain, quantity, source component, unit)
- **AND** no name-regex heuristic is required for channel classification

### Requirement: Backward-Compatible Summary and Telemetry Surface
Python bindings SHALL preserve existing summary payloads while adding richer per-sample electrothermal channels.

#### Scenario: Existing script consumes summaries only
- **WHEN** a script reads legacy `loss_summary`, `thermal_summary`, and `component_electrothermal`
- **THEN** behavior remains backward compatible
- **AND** summary values are consistent with reductions over canonical electrothermal channels

### Requirement: Python API Coverage for Missing GUI Components

Python bindings SHALL expose component-construction APIs or descriptor-based APIs sufficient to instantiate all currently missing GUI component types in runtime circuits.

#### Scenario: Build parity circuit from Python
- **GIVEN** Python code defining a circuit with each previously missing GUI component family
- **WHEN** the circuit is constructed via bindings
- **THEN** all components are accepted and mapped to backend runtime representations
- **AND** no missing-binding error is raised for those component families

### Requirement: Parameter Structs and Validation Exposure

Python bindings SHALL expose parameter structures and validation diagnostics for newly supported models.

#### Scenario: Invalid parameter validation from Python
- **WHEN** Python configures an invalid parameter set for a new component type
- **THEN** a structured exception/diagnostic is returned with component type and parameter context

### Requirement: Instrumentation Result Access

Python bindings SHALL expose probe/scope/routing outputs as structured result channels.

#### Scenario: Read probe/scope channels
- **GIVEN** a simulation containing probes and scopes
- **WHEN** Python reads simulation results
- **THEN** per-channel metadata and waveform values are available without post-hoc GUI-only reconstruction

### Requirement: Backward-Compatible Existing APIs

Existing Python runtime APIs SHALL remain functional while new component APIs are introduced.

#### Scenario: Existing script compatibility
- **WHEN** an existing script using current `Circuit.add_*` methods runs
- **THEN** behavior remains compatible
- **AND** introduction of new component APIs does not break prior workflows

### Requirement: Python-Only Supported Runtime Surface
Python bindings SHALL be the only supported user-facing runtime interface for simulation workflows.

#### Scenario: User follows supported workflow
- **WHEN** a user executes simulation workflows documented as supported
- **THEN** all workflows are available through the Python package interface
- **AND** documentation does not require direct C++ or legacy CLI usage

### Requirement: Full v1 Configuration Exposure
Python bindings SHALL expose all v1 solver, integrator, periodic, and thermal configuration required by declared converter workflows.

#### Scenario: Configure advanced converter run
- **WHEN** Python code configures declared v1 runtime options for a converter case
- **THEN** equivalent options are available through bindings without undocumented C++-only fallback

### Requirement: Converter Component and Thermal API Coverage
Python bindings SHALL expose APIs to build and run declared converter component sets with associated thermal and loss models.

#### Scenario: Build electro-thermal converter in Python
- **WHEN** a benchmark converter case uses declared electrical and thermal components
- **THEN** the case can be constructed and executed from Python without legacy adapters

### Requirement: Deprecated Surface Retirement Policy
Deprecated Python entrypoints SHALL include a migration path and versioned removal policy.

#### Scenario: Deprecated entrypoint present
- **WHEN** an entrypoint is marked for removal
- **THEN** bindings and docs provide a supported replacement and removal version
- **AND** CI includes migration coverage during the deprecation window

### Requirement: Canonical Mode-Based Transient Configuration
Python bindings SHALL expose a canonical transient mode selection equivalent to YAML `step_mode` semantics (`fixed` or `variable`).

#### Scenario: Configure fixed mode from Python
- **WHEN** Python code selects fixed mode through the canonical runtime API
- **THEN** the transient run executes with fixed-step macro-grid semantics
- **AND** output sampling follows deterministic fixed-grid behavior

#### Scenario: Configure variable mode from Python
- **WHEN** Python code selects variable mode through the canonical runtime API
- **THEN** the transient run executes with adaptive-step semantics
- **AND** telemetry includes adaptive acceptance/rejection metrics

### Requirement: Hybrid Segment Runtime Semantics in Python
Python bindings SHALL map canonical mode selection to hybrid segment-first runtime behavior without requiring legacy backend selectors.

#### Scenario: Canonical mode uses segment-first runtime
- **WHEN** Python config selects either `fixed` or `variable` canonical mode
- **THEN** runtime attempts state-space segment solve as primary path
- **AND** uses nonlinear DAE fallback deterministically when segment admissibility fails

### Requirement: Expert Override Exposure
Python bindings SHALL provide explicit expert override controls without requiring them for standard transient use.

#### Scenario: Standard use without expert overrides
- **WHEN** Python code configures only canonical mode and timing fields
- **THEN** the simulation runs with deterministic mode-derived default profiles

#### Scenario: Expert override application
- **WHEN** Python code supplies expert override controls
- **THEN** overrides are applied on top of canonical mode defaults
- **AND** invalid expert keys or values produce structured errors

### Requirement: Legacy Transient Configuration Migration Diagnostics
Python bindings SHALL provide deterministic migration diagnostics for removed legacy transient-backend controls.

#### Scenario: Deprecated legacy backend field usage
- **WHEN** Python code attempts to use deprecated backend-specific transient controls removed from supported runtime
- **THEN** bindings raise a structured configuration error
- **AND** include migration guidance to canonical mode-based controls

### Requirement: Electrothermal and Loss Surface Support
Python bindings SHALL expose loss and thermal configuration/results in canonical mode workflows.

#### Scenario: Configure electrothermal options from Python
- **WHEN** Python config enables losses and thermal coupling on a canonical mode run
- **THEN** runtime accepts the configuration without requiring legacy backend controls
- **AND** simulation results include `loss_summary` and `thermal_summary`

#### Scenario: Invalid electrothermal override values
- **WHEN** Python config provides invalid thermal/loss override values
- **THEN** bindings raise structured configuration errors
- **AND** preserve deterministic error messaging

### Requirement: Unified Per-Component Electrothermal Telemetry in Python
Python bindings SHALL expose a unified per-component electrothermal telemetry surface in `SimulationResult`.

#### Scenario: Access per-component losses and temperatures
- **WHEN** Python runs a transient simulation with electrothermal options
- **THEN** `SimulationResult` exposes per-component entries keyed by component identity
- **AND** each entry includes both loss and temperature fields with deterministic schema and ordering

#### Scenario: Thermal-disabled entry shape remains stable
- **WHEN** a component has no enabled thermal port
- **THEN** the component entry still includes thermal fields
- **AND** thermal status is explicit (`thermal_enabled=false`) with deterministic default values

### Requirement: Backward-Compatible Summary Surfaces
Python bindings SHALL keep existing `loss_summary` and `thermal_summary` surfaces while introducing unified per-component telemetry.

#### Scenario: Existing tooling reads legacy summaries
- **WHEN** Python tooling continues to consume `loss_summary` and `thermal_summary`
- **THEN** behavior remains backward compatible
- **AND** aggregate values remain consistent with reductions of the unified per-component telemetry

### Requirement: Python AC Sweep API Exposure
Python bindings SHALL expose frequency-domain analysis through canonical class-based and procedural APIs.

#### Scenario: Run AC sweep from class API
- **WHEN** Python code invokes AC sweep through canonical `Simulator` workflow
- **THEN** the run executes using the v1 kernel frequency-analysis path
- **AND** returned structures include typed sweep results and diagnostics

#### Scenario: Procedural compatibility entrypoint
- **WHEN** Python code uses procedural AC sweep entrypoints provided by bindings
- **THEN** behavior maps to the same kernel execution semantics as class-based API
- **AND** no console-text parsing is required to consume results

### Requirement: Structured Frequency-Domain Result Objects
Python bindings SHALL expose structured AC sweep result fields for frequency vector, complex response, magnitude/phase arrays, and derived metrics.

#### Scenario: Consume response data for plotting/reporting
- **WHEN** Python tooling reads AC sweep results
- **THEN** frequency and response arrays are available in structured fields
- **AND** metadata includes response quantity/unit context for frontend routing

#### Scenario: Undefined metrics are explicit
- **WHEN** crossover or stability margins are not mathematically defined for a response
- **THEN** result fields expose explicit undefined status/reason
- **AND** consumers can branch without heuristic checks

### Requirement: Structured Error Surface for AC Sweep
Python bindings SHALL propagate deterministic typed diagnostics for AC sweep parsing/preflight/runtime failures.

#### Scenario: Invalid frequency-analysis configuration
- **WHEN** Python submits invalid AC sweep options
- **THEN** bindings raise structured exceptions with deterministic reason codes and field context

#### Scenario: Kernel-side AC sweep failure
- **WHEN** kernel reports typed failure during AC sweep execution
- **THEN** Python receives equivalent failure reason and contextual details
- **AND** benchmark tooling can classify failures without regex parsing

### Requirement: Typed Waveform Post-Processing API in Python
Python bindings SHALL expose typed configuration and execution surfaces for waveform post-processing jobs.

#### Scenario: Configure and run post-processing jobs from Python
- **WHEN** Python code defines post-processing jobs (FFT/THD/time metrics/power-efficiency/loop metrics) using canonical typed fields
- **THEN** jobs execute through backend-owned logic
- **AND** result objects are returned in structured form without requiring ad-hoc script parsing.

#### Scenario: Invalid enum/value assignment
- **WHEN** Python code sets unsupported job type, window mode, or metric identifier
- **THEN** bindings fail deterministically with structured configuration diagnostics
- **AND** diagnostics include field context and accepted values.

### Requirement: Structured Python Result Surface
Python results SHALL expose post-processing outputs as structured, deterministic objects.

#### Scenario: Successful post-processing output
- **WHEN** post-processing executes successfully
- **THEN** Python receives structured scalar metrics, spectrum/harmonic payloads, and per-job metadata
- **AND** output ordering is deterministic for equivalent inputs.

#### Scenario: Metadata-driven channel interpretation
- **WHEN** Python consumers inspect post-processing outputs
- **THEN** units, domains, and source-signal bindings are available as structured metadata
- **AND** consumers do not need regex-based inference from label strings.

### Requirement: Structured Diagnostics and Undefined-Metric Reasons
Python bindings SHALL expose typed diagnostics for invalid jobs and undefined metrics.

#### Scenario: Invalid sampling/window prerequisites
- **WHEN** post-processing requests violate sampling/window constraints
- **THEN** Python receives deterministic typed diagnostics
- **AND** partial ambiguous output is not emitted for failed jobs.

#### Scenario: Metric undefined by physical/numerical conditions
- **WHEN** a metric cannot be defined (for example THD with zero fundamental component)
- **THEN** Python result includes deterministic undefined-reason code
- **AND** behavior is stable across repeated executions.

### Requirement: Backward-Compatible Runtime Workflows
Python transient/frequency workflows SHALL remain backward compatible when post-processing is not configured.

#### Scenario: Existing script without post-processing config
- **WHEN** an existing Python simulation script executes without post-processing jobs
- **THEN** behavior remains backward compatible
- **AND** no new mandatory fields are required.

### Requirement: `compile_cblock` Function
The Python package SHALL expose a top-level
`compile_cblock(source, *, output_dir, name, extra_cflags, compiler)` function
that compiles a C source file or string to a shared library suitable for use with
`CBlockLibrary`.  The function MUST:
- Accept `source` as either a `str` (inline C code) or a `pathlib.Path` (path to
  a `.c` file).
- Accept an optional `compiler: str | None = None` keyword argument.  When
  provided, the given executable path SHALL be used directly and `detect_compiler()`
  MUST NOT be called.  When `None`, the function MUST call `detect_compiler()` and
  raise `CBlockCompileError` if no compiler is found.
- Apply default flags: `-O2 -shared -fPIC -std=c11 -Wall -Wextra` on POSIX;
  `/LD /O2 /std:c11` on Windows MSVC.
- Return a `pathlib.Path` pointing to the compiled shared library.
- Raise `CBlockCompileError` with compiler stderr accessible via `.stderr_output`
  if compilation fails.

#### Scenario: Successful inline C compilation
- **GIVEN** a valid C source string containing `pulsim_cblock_step` and a detected system compiler
- **WHEN** `compile_cblock(source_str, name="my_block")` is called
- **THEN** a `.so` / `.dylib` / `.dll` file is created in a temporary directory
- **AND** the returned `Path` points to a file that exists and is loadable by `ctypes.CDLL`

#### Scenario: Successful file-based compilation
- **GIVEN** a path to a valid `.c` file
- **WHEN** `compile_cblock(Path("my_block.c"))` is called
- **THEN** compilation succeeds and the library path is returned

#### Scenario: Compilation error surfaces stderr
- **GIVEN** a C source string with a syntax error
- **WHEN** `compile_cblock()` is called
- **THEN** `CBlockCompileError` is raised
- **AND** `exc.stderr_output` contains the compiler error message
- **AND** `exc.source` reproduces the failing source for debugging

#### Scenario: No compiler available raises clear error
- **GIVEN** no C compiler is available and `PULSIM_CC` is not set
- **WHEN** `compile_cblock()` is called without `compiler=`
- **THEN** `CBlockCompileError` is raised with an actionable message explaining
  how to install a compiler, set `PULSIM_CC`, or pass `compiler=` explicitly

#### Scenario: Explicit compiler path bypasses auto-detection
- **GIVEN** a valid C source and the user passes `compiler="/opt/homebrew/bin/gcc-14"`
- **WHEN** `compile_cblock(source, compiler="/opt/homebrew/bin/gcc-14")` is called
- **THEN** the subprocess is invoked with that exact compiler executable
- **AND** `detect_compiler()` is NOT called
- **AND** compilation succeeds

#### Scenario: Invalid explicit compiler path raises clear error
- **GIVEN** the user passes `compiler="/nonexistent/gcc"`
- **WHEN** `compile_cblock()` is called
- **THEN** `CBlockCompileError` is raised with a message naming the path that was not found

#### Scenario: Extra compiler flags forwarded correctly
- **GIVEN** valid C source and `extra_cflags=["-DUSE_FEATURE=1", "-march=native"]`
- **WHEN** `compile_cblock()` is called
- **THEN** the compiler subprocess receives the extra flags appended after defaults
- **AND** compilation succeeds (flags are valid)

### Requirement: `detect_compiler` Function
The Python package SHALL expose `detect_compiler() -> str | None` that returns the
path to the best available C compiler on the current system, or `None` if none is
found.  Detection priority MUST be: (1) `PULSIM_CC` environment variable,
(2) `cc`, `gcc`, `clang` on POSIX, (3) `cl.exe` / `gcc.exe` on Windows.

#### Scenario: PULSIM_CC override respected
- **GIVEN** `PULSIM_CC=/usr/local/bin/cc` is set in the environment
- **WHEN** `detect_compiler()` is called
- **THEN** it returns `/usr/local/bin/cc` without probing the system

#### Scenario: Returns None gracefully when no compiler found
- **GIVEN** no compiler is on PATH and `PULSIM_CC` is not set (mocked in test)
- **WHEN** `detect_compiler()` is called
- **THEN** it returns `None` without raising an exception

### Requirement: `CBlockLibrary` Class
The Python package SHALL expose a `CBlockLibrary` class that loads a compiled
C-Block shared library via `ctypes`, resolves the required and optional symbols,
calls `init` if present, and provides a `step(t, dt, inputs) -> list[float]`
method for evaluating the block each timestep.  The class MUST:
- Raise `CBlockABIError` if `pulsim_cblock_step` symbol is not found.
- Raise `CBlockABIError` if the exported ABI version does not match
  `PULSIM_CBLOCK_ABI_VERSION`.
- Support use as a context manager (`with CBlockLibrary(path) as blk:`).
- Expose `n_inputs: int`, `n_outputs: int`, `name: str` properties.
- Provide `reset()` that reinitialises C state via `destroy` + `init`.

#### Scenario: Load and step a simple compiled library
- **GIVEN** a compiled `.so` for a gain block (`out[0] = in[0] * 3.0`)
- **WHEN** `CBlockLibrary(path)` is constructed and `step(t=0.0, dt=1e-6, inputs=[2.0])` is called
- **THEN** the result is `[6.0]`

#### Scenario: Context manager releases library
- **GIVEN** a `CBlockLibrary` used as a context manager
- **WHEN** the `with` block exits normally
- **THEN** `destroy` is called (if present) and the library is unloaded
- **AND** calling `step` after context exit raises `RuntimeError`

#### Scenario: `reset()` reinitialises stateful block
- **GIVEN** an IIR filter C-Block that has processed 50 steps (non-zero state)
- **WHEN** `reset()` is called
- **THEN** the next `step` call produces the same output as the very first step
  with the same input (state is back to initial conditions)

#### Scenario: Wrong input length raises ValueError
- **GIVEN** a `CBlockLibrary` with `n_inputs=2`
- **WHEN** `step(inputs=[1.0])` is called (only 1 input instead of 2)
- **THEN** `ValueError` is raised before calling the C function

### Requirement: `PythonCBlock` Class
The Python package SHALL expose a `PythonCBlock` class that accepts any Python
callable as a signal block, providing the same interface as `CBlockLibrary`.
The callable SHALL receive `(ctx: dict, t: float, dt: float, inputs: list[float])`
and return `list[float]`.  The `ctx` dict MUST persist across calls and be reset
by `reset()`.  `PythonCBlock` MUST be usable anywhere `CBlockLibrary` is accepted.

#### Scenario: Callable receives correct arguments on each step
- **GIVEN** a `PythonCBlock` wrapping a function that records all calls to a list
- **WHEN** `step` is called with `t=1e-3`, `dt=1e-6`, `inputs=[0.5]`
- **THEN** the recorded call has the exact values `t=1e-3`, `dt=1e-6`, `inputs=[0.5]`

#### Scenario: `PythonCBlock` interchangeable with `CBlockLibrary` in evaluator
- **GIVEN** a `SignalEvaluator` built with a `C_BLOCK` component carrying a `PythonCBlock` controller
- **WHEN** `evaluator.step(t)` is called
- **THEN** the block output propagates downstream identically to a compiled `CBlockLibrary`

### Requirement: `CBlockCompileError`, `CBlockABIError`, `CBlockRuntimeError`
The Python package SHALL expose three exception types:
- `CBlockCompileError(RuntimeError)` with attributes `.source: str`,
  `.stderr_output: str`, and `.compiler_path: str | None` (the compiler executable
  that was used or attempted, `None` if no compiler was found at all) — for
  compile-time failures.
- `CBlockABIError(RuntimeError)` with attributes `.expected_version: int` and
  `.found_version: int | None` — for ABI contract violations at load time.
- `CBlockRuntimeError(RuntimeError)` with attributes `.return_code: int`,
  `.t: float`, `.step_index: int` — for non-zero return codes from `step`.
All three SHALL be importable directly from `pulsim`.

#### Scenario: All three exception classes are importable from top-level package
- **GIVEN** `import pulsim as ps`
- **WHEN** `ps.CBlockCompileError`, `ps.CBlockABIError`, `ps.CBlockRuntimeError` are accessed
- **THEN** they are the correct exception classes (not `AttributeError`)

### Requirement: Convergence Policy Configuration in Python
Python bindings SHALL expose convergence-policy profiles and bounded tuning options for transient execution.

#### Scenario: Select robust convergence profile
- **WHEN** Python code selects profile `robust` for a challenging circuit
- **THEN** runtime receives corresponding policy configuration
- **AND** strict-mode contracts remain explicit and deterministic

#### Scenario: Override bounded policy knobs
- **WHEN** Python code overrides approved policy fields (for example event-burst guard or regularization bounds)
- **THEN** overrides are validated with deterministic errors for invalid ranges
- **AND** accepted values are reflected in runtime options

### Requirement: Structured Convergence Telemetry Exposure
Python result objects SHALL expose structured convergence telemetry produced by the policy engine.

#### Scenario: Inspect convergence diagnostics from Python
- **WHEN** a transient run completes (success or failure)
- **THEN** Python code can access failure classes, recovery stages, and policy actions as typed fields
- **AND** no diagnostic workflow requires string parsing
