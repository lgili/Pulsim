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

### Requirement: AC and FRA Methods on Simulator
Python bindings SHALL expose `Simulator.run_ac_sweep(options)` and `Simulator.run_fra(options)` returning structured results.

#### Scenario: Run AC sweep from Python
- **WHEN** Python code instantiates `AcSweepOptions(f_start=1, f_stop=1e6, points_per_decade=20, ...)` and calls `sim.run_ac_sweep(options)`
- **THEN** the call returns an `AcSweepResult` with frequencies, magnitudes, phases, real/imag arrays
- **AND** the arrays are contiguous numpy arrays interoperable with matplotlib

#### Scenario: Run FRA from Python
- **WHEN** Python code calls `sim.run_fra(FraOptions(...))`
- **THEN** the returned `FraResult` exposes the same shape as `AcSweepResult` plus `thd_at_frequency`
- **AND** convergence diagnostics are accessible

### Requirement: Linearization Method on Simulator
Python bindings SHALL expose `Simulator.linearize_around(x_op, t_op=0.0)` returning sparse `(E, A, B, C, D)` matrices.

#### Scenario: Linearize from Python
- **WHEN** Python calls `sim.linearize_around(dc_result.solution)`
- **THEN** the returned object exposes `E`, `A`, `B`, `C`, `D` as scipy sparse matrices
- **AND** the matrices can be passed to scipy `signal.dlti` or `control.ss` for downstream analysis

### Requirement: Bode/Nyquist Plotting Helpers
Python bindings SHALL provide `pulsim.bode_plot(result, ax=None, ...)` and `pulsim.nyquist_plot(result, ...)` for fast visualization.

#### Scenario: Bode plot end-to-end
- **WHEN** Python code runs `pulsim.bode_plot(ac_result)` with default arguments
- **THEN** matplotlib axes are returned showing magnitude (dB) and phase (deg) vs frequency (log-x)
- **AND** the plot is publication-quality without further user formatting

#### Scenario: Overlay AC and FRA
- **WHEN** Python code runs `pulsim.fra_overlay(ac_result, fra_result)`
- **THEN** both curves are plotted with distinct labels
- **AND** delta annotations highlight any region exceeding 1 dB / 5° divergence

### Requirement: Export Helpers
Python bindings SHALL provide CSV and JSON export for AC and FRA results.

#### Scenario: CSV export
- **WHEN** Python calls `pulsim.export_ac_csv(ac_result, "out.csv", format="magphase")`
- **THEN** a CSV with columns `f, mag_db, phase_deg` is written
- **AND** an alternative `format="complex"` writes `f, re, im`

#### Scenario: JSON round-trip
- **WHEN** Python calls `pulsim.export_ac_json(ac_result, "out.json")` and later `pulsim.load_ac_result("out.json")`
- **THEN** the loaded result is bit-identical to the original

### Requirement: Python Template Builder API
Python bindings SHALL expose `pulsim.templates.<name>(...)` builder functions that return ready-to-simulate `Circuit` objects.

#### Scenario: Buck builder from Python
- **WHEN** Python code calls `pulsim.templates.buck(vin=48, vout=12, pout=240, fsw=100e3)`
- **THEN** a `Circuit` is returned with all template wiring expanded
- **AND** the circuit can be used with `Simulator(circuit, options)` directly

#### Scenario: Builder reports parameters used
- **GIVEN** a builder call with partial parameters relying on auto-design
- **WHEN** `circuit.template_metadata()` is called
- **THEN** the returned dict includes the resolved parameter set including auto-designed values

### Requirement: Template Listing and Introspection
Python bindings SHALL expose `pulsim.list_templates()` and `pulsim.describe_template(name)` for runtime discovery.

#### Scenario: List templates
- **WHEN** Python calls `pulsim.list_templates()`
- **THEN** a list of `(name, version, description)` tuples is returned
- **AND** the list is stable across runs

#### Scenario: Describe template
- **WHEN** Python calls `pulsim.describe_template("buck_template")`
- **THEN** a structured object is returned with parameter schema, defaults, and links to docs

### Requirement: Template Use with Parameter Sweep
Templates SHALL be composable with the parameter-sweep API (post-`add-monte-carlo-parameter-sweep` landing).

#### Scenario: Buck parameter sweep
- **GIVEN** Python code that creates a template via builder and varies `Lout`
- **WHEN** the parameter sweep runs
- **THEN** each instance is a fresh template expansion
- **AND** results aggregate per parameter setting

### Requirement: Python Codegen API
Python bindings SHALL expose `pulsim.codegen.generate(circuit, target, step, out_dir, **opts)` and `pulsim.codegen.run_pil_bench(out_dir)`.

#### Scenario: Programmatic codegen
- **GIVEN** a `Circuit` object built via templates or YAML
- **WHEN** Python calls `pulsim.codegen.generate(circuit, "c99", 1e-6, "gen/")`
- **THEN** the output directory contains `pulsim_model.{c,h}`, `pulsim_topologies.c`, `Makefile`
- **AND** the function returns a `CodegenSummary` with budgets and warnings

#### Scenario: PIL bench programmatic
- **WHEN** Python calls `pulsim.codegen.run_pil_bench("gen/")`
- **THEN** the bench compiles the generated C, runs against native Pulsim, and returns `PilBenchResult` with parity verdict
- **AND** divergence above tolerance raises `PilParityError`

### Requirement: Sweep API in Python
Python bindings SHALL expose `pulsim.sweep(circuit_factory, parameters, metrics, executor, n_workers, seed)` returning a `SweepResult` object.

#### Scenario: Sweep call
- **WHEN** Python code calls `pulsim.sweep(circuit_factory, params_dict, metrics_list, executor="joblib", n_workers=8, seed=42)`
- **THEN** the sweep executes per spec
- **AND** returns a `SweepResult` with `to_pandas()`, `percentile()`, `failed` accessors

### Requirement: Distribution Helper Classes
Python bindings SHALL expose `pulsim.Distribution.{normal, uniform, loguniform, triangular, beta, custom}` for parameter declaration.

#### Scenario: Custom distribution
- **GIVEN** a Python callable `f(rng) -> sample`
- **WHEN** `Distribution.custom(f, n=500)` is supplied as a parameter spec
- **THEN** the sweep draws 500 samples via `f` with the seeded RNG

### Requirement: Sensitivity and Optimization Wrappers
Python bindings SHALL expose `pulsim.sensitivity(sweep_result, target_metric)` and (stretch) `pulsim.optimize(circuit_factory, objective, bounds, ...)`.

#### Scenario: Sensitivity from sweep
- **GIVEN** a Sobol sweep on 5 parameters and a target metric
- **WHEN** `pulsim.sensitivity(result, "efficiency")` is called
- **THEN** first-order and total-order Sobol indices per parameter are returned
- **AND** the values are deterministic given the seed

### Requirement: Python FMU Export and Import API
Python bindings SHALL expose `pulsim.fmu.export(circuit_or_path, fmu_path, **opts)` and `pulsim.fmu.load(fmu_path)`.

#### Scenario: Programmatic FMU export
- **GIVEN** a `Circuit` and an export configuration dict
- **WHEN** Python calls `pulsim.fmu.export(circuit, "buck.fmu", version="2.0", type="cs", inputs=[...], outputs=[...])`
- **THEN** the file `buck.fmu` is created
- **AND** the function returns an `FmuExportSummary` with the model description and validation result

#### Scenario: Programmatic FMU import as block
- **WHEN** Python calls `pulsim.fmu.load("foreign.fmu")` and uses the returned object as a `Circuit` block
- **THEN** the FMU is instantiated and its inputs/outputs are accessible by name
- **AND** the block participates in the simulation as a signal-domain component

### Requirement: Robustness Tier Surface in Python
Python bindings SHALL expose `pulsim.RobustnessTier` enum and `SimulationOptions.robustness` field.

#### Scenario: Set tier from Python
- **WHEN** Python code does `options.robustness = pulsim.RobustnessTier.Aggressive`
- **THEN** the simulation resolves the same profile as YAML-driven `simulation.robustness: aggressive`
- **AND** both invocations produce identical `BackendTelemetry.robustness_profile`

### Requirement: Deprecation of Python Wrapper Retry Layer
The Python wrapper retry layer in `pulsim/__init__.py:run_transient` (auto-bleeders, dt-halving retries) SHALL log a deprecation warning when triggered, with guidance pointing to the kernel-level robustness profile.

#### Scenario: Retry deprecation warning
- **GIVEN** legacy retry path is reached (e.g., behind `PULSIM_LEGACY_RETRY_FALLBACK=1`)
- **WHEN** the retry executes
- **THEN** a one-time deprecation warning is emitted
- **AND** the warning message names the replacement mechanism (`SimulationOptions.robustness`)

#### Scenario: Default path bypasses wrapper retry
- **GIVEN** PWL engine is resolved as the default for the circuit (post `refactor-pwl-switching-engine`)
- **WHEN** `run_transient` is called
- **THEN** the wrapper does not invoke retry/auto-bleeder logic
- **AND** the kernel handles recovery via the resolved robustness profile

### Requirement: Profile Inspection from Python
Python bindings SHALL expose `Simulator.options.robustness_profile` as a read-only structured object for debugging.

#### Scenario: Inspect resolved profile
- **WHEN** Python reads `sim.options.robustness_profile.newton_max_iter`
- **THEN** the value reflects the kernel-resolved profile
- **AND** Python users can compare profiles between runs deterministically

### Requirement: Modular pybind11 Binding Layout
The Python bindings SHALL be split across multiple translation units under `python/bindings/`, each ≤500 lines, with a single `main.cpp` orchestrator file containing the `PYBIND11_MODULE` entry.

#### Scenario: Modular layout
- **WHEN** the project is built
- **THEN** `python/bindings/` contains per-domain files (`devices.cpp`, `control.cpp`, `simulation.cpp`, `parser.cpp`, `solver.cpp`, `thermal.cpp`, `loss.cpp`, `analysis.cpp`, `main.cpp`)
- **AND** each non-main file ≤500 lines
- **AND** `main.cpp` calls `register_<domain>(m)` from each module

#### Scenario: Editing one domain
- **GIVEN** a one-line edit in `bindings/devices.cpp`
- **WHEN** an incremental build runs
- **THEN** only `bindings/devices.cpp` recompiles
- **AND** the link step pulls cached objects for unchanged domains

### Requirement: Build-Time Performance Targets
The build SHALL meet target metrics for clean and incremental builds:

#### Scenario: Clean build target
- **WHEN** a clean build runs on a CI baseline machine
- **THEN** total wallclock is at most 75% of the pre-refactor baseline
- **AND** the metric is recorded in CI artifacts

#### Scenario: Incremental build target
- **GIVEN** a one-line edit in any single source file
- **WHEN** an incremental build runs
- **THEN** wallclock is ≤10% of clean-build wallclock
- **AND** only the touched object plus the link step are recompiled

### Requirement: Public API Stability During Refactor
The bindings refactor SHALL preserve the existing Python public API and ABI bit-for-bit.

#### Scenario: Existing user notebooks
- **GIVEN** any existing example notebook in `examples/notebooks/`
- **WHEN** executed against the refactored bindings
- **THEN** results are bit-identical to pre-refactor
- **AND** no user-facing import or call requires modification

