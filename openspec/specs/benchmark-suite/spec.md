# benchmark-suite Specification

## Purpose
TBD - created by archiving change remove-cli-benchmark-dependency. Update Purpose after archive.
## Requirements
### Requirement: Python-First Benchmark Execution
The benchmark suite SHALL execute circuit scenarios through Python runtime APIs backed by the v1 kernel and SHALL NOT require an external `pulsim` executable.

#### Scenario: Run benchmark suite without CLI binary
- **WHEN** `benchmark_runner.py` is executed in an environment without a `pulsim` executable
- **THEN** scenarios are executed through Python runtime bindings
- **AND** the run produces standard benchmark artifacts

### Requirement: Backend-Independent Validation Outcomes
The benchmark suite SHALL produce validation outcomes independent of backend type and SHALL NOT mark scenarios as skipped due to missing CLI-only paths.

#### Scenario: Matrix run without CLI binary
- **WHEN** `validation_matrix.py` executes scenarios without an external `pulsim` executable
- **THEN** each scenario returns explicit `passed`, `failed`, or `baseline` status based on validation rules
- **AND** no scenario is skipped solely because execution is via Python runtime

### Requirement: Periodic Scenario Coverage in Matrix
The benchmark suite SHALL execute periodic steady-state scenarios (shooting and harmonic balance) through the Python runtime path.

#### Scenario: Periodic benchmark scenario
- **WHEN** a scenario declares shooting or harmonic balance options
- **THEN** the benchmark runner invokes the corresponding periodic runtime method
- **AND** emits result artifacts and validation status for that scenario

### Requirement: Structured Telemetry Source
Benchmark telemetry SHALL be collected from structured simulation result fields rather than command stdout parsing.

#### Scenario: Collect solver telemetry
- **WHEN** a scenario completes
- **THEN** nonlinear iterations, linear iterations, step counts, rejections, and runtime are read from simulation results
- **AND** telemetry is written to `results.json` in a stable schema

### Requirement: Runtime Parity for ngspice Comparator
The benchmark comparator SHALL run Pulsim through the same Python runtime path used by the core benchmark runner and SHALL support external SPICE parity backends with LTspice as a primary target.

#### Scenario: Pulsim vs LTspice comparison
- **WHEN** the comparator executes a mapped benchmark pair with LTspice backend configured
- **THEN** Pulsim waveforms are generated through the Python runtime path used by the benchmark suite
- **AND** comparison metrics are computed from mapped Pulsim and LTspice observables

#### Scenario: Pulsim vs ngspice comparison
- **WHEN** the comparator executes a mapped benchmark pair with ngspice backend configured
- **THEN** Pulsim runtime execution path remains identical to the core benchmark runner

### Requirement: YAML-first benchmark circuits
The system SHALL define benchmark circuits as YAML netlists compliant with the `netlist-yaml` capability and the `pulsim-v1` schema.

#### Scenario: Load a benchmark circuit
- **WHEN** a benchmark circuit is loaded
- **THEN** the YAML netlist is parsed using the `pulsim-v1` schema
- **AND** the circuit is accepted only if it conforms to `netlist-yaml`

### Requirement: Benchmark metadata
The system SHALL allow benchmark metadata to be provided via a `benchmark` block embedded in the YAML netlist or via a sidecar YAML file.

#### Scenario: Resolve benchmark metadata
- **WHEN** a benchmark circuit is loaded
- **THEN** metadata is resolved from the embedded `benchmark` block
- **OR** a sidecar YAML file if the embedded block is absent

### Requirement: Scenario matrix
The system SHALL allow a benchmark run to define multiple scenarios per circuit, each with solver and integrator settings.

#### Scenario: Run multiple scenarios
- **WHEN** a circuit defines multiple scenarios
- **THEN** the benchmark runner executes each scenario with its specified solver and integrator configuration

### Requirement: Benchmark artifacts
The system SHALL emit standardized benchmark artifacts for each run: `results.csv`, `results.json`, and `summary.json`.

#### Scenario: Produce standardized outputs
- **WHEN** a benchmark run completes
- **THEN** `results.csv` contains per-scenario numeric outputs
- **AND** `results.json` contains structured metadata and telemetry
- **AND** `summary.json` contains pass/fail validation results

### Requirement: Telemetry fields
The system SHALL record solver telemetry including nonlinear iterations, linear iterations, step count, residual norms, and wall-clock runtime.

#### Scenario: Record telemetry
- **WHEN** a benchmark scenario completes
- **THEN** telemetry fields are included in `results.json`

### Requirement: Validation types
The system SHALL support the validation types `analytical`, `reference`, `ltspice`, and `ngspice` for benchmark comparisons.

#### Scenario: Validate against LTspice reference
- **WHEN** a benchmark uses validation type `ltspice`
- **THEN** outputs are compared against LTspice-generated reference waveforms using configured observables and tolerances

### Requirement: Validation matrix
The system SHALL provide a validation matrix runner that executes solver and integrator combinations across the benchmark corpus.

#### Scenario: Execute validation matrix
- **WHEN** a validation matrix is invoked
- **THEN** the runner executes each solver/integrator combination for the selected circuits

### Requirement: Frozen Baseline Artifact Governance
The benchmark suite SHALL maintain versioned frozen baselines for KPI comparison, including environment fingerprint metadata.

#### Scenario: Freeze baseline snapshot
- **WHEN** a baseline freeze is created for a benchmark corpus version
- **THEN** artifacts store KPI values and environment fingerprint fields (compiler, flags, machine class)
- **AND** the baseline snapshot is immutable for regression-gate comparisons

#### Scenario: Baseline provenance check
- **WHEN** KPI gate evaluation runs in CI
- **THEN** the evaluator verifies baseline provenance metadata before comparing thresholds
- **AND** fails deterministically if provenance metadata is missing or inconsistent

### Requirement: KPI Non-Regression Gates
Benchmark and validation pipelines SHALL block merge when required KPI thresholds regress beyond configured limits.

#### Scenario: Regression threshold violation
- **WHEN** any required KPI exceeds allowed regression threshold against frozen baseline
- **THEN** CI marks the gate as failed
- **AND** merge is blocked until KPI compliance is restored or thresholds are explicitly revised

#### Scenario: KPI gate pass
- **WHEN** all required KPIs are within approved thresholds for the selected matrix
- **THEN** CI marks the non-regression gate as passed
- **AND** gate status is published in machine-readable artifacts

### Requirement: Canonical KPI Matrix Coverage
The benchmark suite SHALL enforce KPI coverage for converter, linear, stress, and electrothermal scenario classes across canonical runtime modes.

#### Scenario: Matrix execution with class coverage
- **WHEN** the benchmark matrix is executed for release gating
- **THEN** it includes scenario classes `converter`, `linear`, `stress`, and `electrothermal`
- **AND** reports KPI outputs per class and per runtime mode where applicable

### Requirement: Machine-Readable KPI Delta Reports
Each gated run SHALL emit machine-readable KPI delta reports suitable for automated trend and release decisions.

#### Scenario: Publish KPI delta report
- **WHEN** a gated benchmark run completes
- **THEN** artifacts include baseline, current values, absolute/relative deltas, and pass/fail status per KPI
- **AND** report schema remains stable across minor tool updates

### Requirement: Stress Testing Scenarios
The benchmark suite SHALL include complex topologies that stress the non-linear, highly oscillatory, and switching behavior of the runtime solvers.

#### Scenario: LLC Resonant Converter
- **WHEN** simulating `ll11_llc_resonant_converter.yaml`
- **THEN** the solver reliably resolves resonant tank oscillations and soft-switching transitions

#### Scenario: PFC Boost Converter
- **WHEN** simulating `ll12_pfc_boost_continuous.yaml`
- **THEN** the solver efficiently handles continuous conduction mode with active power factor correction switching patterns

#### Scenario: Active Clamp Forward Converter
- **WHEN** simulating `ll13_active_clamp_forward.yaml`
- **THEN** the solver correctly models transformer magnetizing reset and secondary side synchronous rectification

### Requirement: Closed-Loop Regulation Scenarios
The benchmark suite SHALL include complex closed-loop topologies that stress the mixed-domain solving capabilities, specifically the interactions between non-linear power stages and discrete/continuous control blocks (PI controllers, PWM generators).

#### Scenario: Buck Converter Closed-Loop
- **WHEN** simulating `ll14_buck_closed_loop.yaml`
- **THEN** the solver reliably resolves the timestep constraints imposed by the PWM generator and limits the PI controller windup, achieving steady-state regulation

#### Scenario: Boost Converter Closed-Loop
- **WHEN** simulating `ll15_boost_closed_loop.yaml`
- **THEN** the solver maintains numerical stability during the Right-Half-Plane (RHP) zero transient effects inherent to closed-loop boost topologies

#### Scenario: Flyback Converter Closed-Loop
- **WHEN** simulating `ll16_flyback_closed_loop.yaml`
- **THEN** the solver accurately resolves the discontinuous/continuous conduction modes across the isolation barrier while the control loop maintains the setpoint

### Requirement: Closed-Loop Converter Electrothermal Parity Gate
The benchmark suite SHALL include closed-loop converter scenarios that validate control behavior together with non-trivial electrothermal behavior.

#### Scenario: Closed-loop buck with PWM and PI
- **WHEN** the closed-loop buck electrothermal benchmark runs
- **THEN** control channels demonstrate bounded PI/PWM behavior
- **AND** semiconductor switching-loss components are non-zero when switching-loss models are configured
- **AND** thermal traces show physically consistent time evolution

### Requirement: Component-Minimum Electrothermal Theory Matrix
The validation suite SHALL include per-component minimum circuits with expected electrothermal behavior checks.

#### Scenario: Thermal-enabled component minimum circuit
- **WHEN** each supported thermal-capable component is simulated in a minimum deterministic circuit
- **THEN** simulated temperatures and losses are compared against theoretical or reference expectations
- **AND** errors must remain within configured tolerances

### Requirement: Electrothermal Channel/Summary Consistency Regression
Benchmark regression SHALL verify deterministic consistency between electrothermal time-series channels and summary payloads.

#### Scenario: Channel-to-summary reduction check
- **WHEN** an electrothermal benchmark completes
- **THEN** reductions over `P*` and `T*` channels match summary fields within configured tolerance
- **AND** mismatch fails gate deterministically

### Requirement: Electrothermal Performance Non-Regression Gate
Electrothermal benchmark gating SHALL include runtime and memory/allocation stability thresholds for rich datasheet-mode scenarios.

#### Scenario: Rich electrothermal benchmark run
- **WHEN** benchmark scenarios use datasheet-grade tables and multi-stage thermal networks
- **THEN** runtime and allocation telemetry are compared against approved thresholds
- **AND** regressions beyond thresholds fail the gate

### Requirement: Missing-Component Parity Matrix

The benchmark/validation suite SHALL include parity coverage for all component types that were previously missing from backend support.

#### Scenario: Per-component smoke matrix
- **WHEN** parity matrix tests are executed
- **THEN** each newly covered component type has at least one executable smoke scenario
- **AND** failures identify the component type and family explicitly

### Requirement: Family-Level Behavioral Validation

The suite SHALL include behavioral validation scenarios for each family: power semiconductors, protection, magnetic/networks, control/analog, and instrumentation/routing.

#### Scenario: Family regression gate
- **WHEN** CI runs benchmark validation
- **THEN** each family-level suite passes configured behavior checks
- **AND** regressions fail the gate with stable diagnostics

### Requirement: Unsupported-Component Regression Guard

Validation SHALL enforce that supported mode does not regress to unsupported-component errors for the declared GUI parity set.

#### Scenario: Unsupported error regression
- **WHEN** a parity fixture containing declared component types is built and simulated
- **THEN** no `Unsupported component type` error is emitted for declared types

### Requirement: External Simulator Path Configuration
Benchmark tooling SHALL accept explicit external simulator executable paths and fail with actionable diagnostics when misconfigured.

#### Scenario: LTspice path missing
- **WHEN** an LTspice parity run is requested without a valid executable path
- **THEN** the run fails with a clear configuration error
- **AND** the failure reason is recorded in benchmark artifacts

### Requirement: Converter Stress Catalog
The benchmark suite SHALL include a declared converter stress catalog covering light, medium, and heavy simulation tiers.

#### Scenario: Run stress catalog
- **WHEN** the stress benchmark command is executed
- **THEN** each catalog case reports convergence status, fallback path, and runtime metrics
- **AND** no case is silently skipped without explicit reason

### Requirement: Deterministic Benchmark Fields
Benchmark artifacts SHALL include deterministic fields suitable for reproducibility checks.

#### Scenario: Reproducibility check
- **WHEN** the same benchmark matrix is run twice on the same hardware class with fixed settings
- **THEN** deterministic fields in artifacts match within configured tolerances

### Requirement: Phase-Gated Solver Refactor Validation
Benchmark and validation tooling SHALL enforce phase-gated KPI checks for solver-core refactor milestones.

#### Scenario: Phase gate blocks regression
- **WHEN** a phase run exceeds configured regression thresholds for required KPIs
- **THEN** the phase is marked failed
- **AND** progression to subsequent implementation phase is blocked in CI

### Requirement: Canonical KPI Reporting
Benchmark artifacts SHALL include canonical KPIs for convergence, accuracy, event fidelity, and runtime efficiency.

#### Scenario: KPI artifact generation
- **WHEN** benchmark/parity/stress suites complete
- **THEN** artifacts include at least convergence success rate, parity RMS error, event-time error, runtime p50, and runtime p95
- **AND** values are emitted in machine-readable JSON summaries

### Requirement: Dual-Mode Coverage Matrix
The benchmark matrix SHALL include both canonical timestep modes (`fixed` and `variable`) across converter-focused scenarios.

#### Scenario: Converter matrix run
- **WHEN** matrix execution is triggered for converter suites
- **THEN** each selected converter case runs in both fixed and variable modes
- **AND** mode-specific KPI results are reported separately

### Requirement: Baseline Freeze and Comparison
Benchmark tooling SHALL support baseline freeze snapshots and automated comparison against the frozen baseline for each phase.

#### Scenario: Baseline comparison report
- **WHEN** a phase benchmark run completes
- **THEN** the report compares current KPIs against the frozen baseline snapshot
- **AND** flags pass/fail status per configured regression threshold

### Requirement: Hybrid Path and Electrothermal KPI Gates
Benchmark and stress tooling SHALL track and gate hybrid-path usage plus electrothermal regression metrics for converter-focused phases.

#### Scenario: Hybrid-path KPI emission
- **WHEN** converter-focused benchmark suites complete
- **THEN** reports include at least `state_space_primary_ratio` and `dae_fallback_ratio`
- **AND** required-threshold regressions in these KPIs fail the phase gate

#### Scenario: Electrothermal KPI emission and gating
- **WHEN** electrothermal-enabled benchmark suites complete
- **THEN** reports include at least `loss_energy_balance_error` and `thermal_peak_temperature_delta`
- **AND** required-threshold regressions fail the phase gate

### Requirement: Component-Level Electrothermal Validation Matrix
The benchmark and validation suite SHALL assert component-level electrothermal outputs, not only aggregate KPIs.

#### Scenario: Electrothermal reference circuit emits component telemetry
- **WHEN** an electrothermal benchmark circuit completes
- **THEN** reports include per-component losses and temperatures for declared components
- **AND** component metrics are compared against baseline tolerances

#### Scenario: Aggregate-to-component consistency checks
- **WHEN** benchmark post-processing computes aggregate electrothermal KPIs
- **THEN** aggregate totals are consistent with reductions over per-component telemetry
- **AND** inconsistencies fail the gate with deterministic diagnostics

#### Scenario: Thermal-port parser contract regression check
- **WHEN** strict-mode benchmark/parser validation runs include invalid thermal-port configurations
- **THEN** deterministic parser diagnostics are emitted and the run fails as expected

### Requirement: AC Sweep Benchmark Corpus Coverage
The benchmark suite SHALL include AC sweep scenarios covering at least one analytical linear circuit and one converter/control-use-case workflow.

#### Scenario: Analytical reference AC sweep case
- **WHEN** AC benchmark suite runs the analytical reference case
- **THEN** magnitude and phase responses are compared against analytical/reference values
- **AND** errors are reported in machine-readable KPI artifacts

#### Scenario: Converter/control AC sweep case
- **WHEN** AC benchmark suite runs a converter/control frequency-response case
- **THEN** benchmark verifies deterministic response generation and structured metric emission
- **AND** unsupported configurations fail with typed diagnostics

### Requirement: AC Sweep KPI Regression Gates
Benchmark gating SHALL enforce non-regression thresholds for AC sweep accuracy and runtime metrics.

#### Scenario: AC sweep regression threshold violation
- **WHEN** required AC sweep KPI thresholds are exceeded versus approved baseline
- **THEN** CI fails the gate deterministically
- **AND** artifacts include baseline/current values and delta metrics

#### Scenario: AC sweep gate pass
- **WHEN** required AC sweep KPIs remain within thresholds
- **THEN** CI reports gate pass
- **AND** machine-readable KPI report is published

### Requirement: AC Sweep Determinism Regression Gate
Benchmark validation SHALL include repeat-run determinism checks for AC sweep outputs on the same machine class.

#### Scenario: Repeat-run deterministic comparison
- **WHEN** the same AC sweep benchmark case executes multiple times with identical configuration
- **THEN** frequency grids are identical and response drift stays within configured tolerance
- **AND** determinism regressions fail the gate

### Requirement: Post-Processing Benchmark Fixture Coverage
The benchmark suite SHALL include deterministic fixture scenarios that validate waveform post-processing correctness across metric families.

#### Scenario: Spectral fixture with known harmonics
- **WHEN** FFT/THD fixtures with known harmonic content are executed
- **THEN** benchmark artifacts include spectral/THD errors against analytical or frozen-reference expectations
- **AND** errors remain within configured thresholds.

#### Scenario: Power-efficiency fixture
- **WHEN** efficiency/power-factor fixtures are executed
- **THEN** benchmark artifacts include derived metric errors against declared references
- **AND** deterministic pass/fail status is emitted.

### Requirement: Deterministic Expected-Failure Fixtures
Benchmark coverage SHALL include expected-failure fixtures for invalid post-processing contracts.

#### Scenario: Invalid window/sampling expected failure
- **WHEN** a fixture intentionally violates post-processing window/sampling prerequisites
- **THEN** benchmark run validates typed expected-failure diagnostics
- **AND** the case is marked passed only if expected diagnostics match deterministically.

### Requirement: Post-Processing KPI Non-Regression Gates
Benchmark KPI gating SHALL enforce post-processing metric correctness, determinism, and runtime-overhead constraints.

#### Scenario: Correctness gate
- **WHEN** post-processing KPI reports are evaluated
- **THEN** required metrics (for example FFT bin error, THD error, scalar metric error) stay within approved thresholds
- **AND** threshold violations fail the required gate.

#### Scenario: Determinism and overhead gate
- **WHEN** repeat-run and runtime-overhead KPIs are evaluated
- **THEN** determinism drift and overhead regressions stay within approved thresholds
- **AND** regressions beyond thresholds fail CI.

### Requirement: Machine-Readable Post-Processing KPI Reporting
Benchmark artifacts SHALL emit machine-readable post-processing KPI deltas and gate status.

#### Scenario: KPI report publication
- **WHEN** post-processing benchmark gate completes
- **THEN** artifacts include baseline/current values, deltas, and pass/fail status per KPI
- **AND** schema remains stable for automated tooling.

### Requirement: Convergence Class Stress Matrix
Benchmark and validation tooling SHALL provide a stress matrix indexed by convergence challenge class.

#### Scenario: Execute challenge-class matrix
- **WHEN** the stress matrix is executed
- **THEN** it includes at least classes `diode_heavy`, `switch_heavy`, `zero_cross`, `magnetic_nonlinear`, and `closed_loop_control`
- **AND** reports pass/fail and KPI outputs by class and runtime mode

### Requirement: Phase-Gated Progression for Robustness Program
CI SHALL enforce phase gates (A..F, ADV) for convergence-platform rollout.

#### Scenario: Phase gate regression
- **WHEN** required KPI thresholds fail for active phase
- **THEN** phase gate fails deterministically
- **AND** progression to next phase is blocked until restored

#### Scenario: Phase gate pass with artifact completeness
- **WHEN** a phase run meets KPI criteria
- **THEN** gate passes only if required telemetry and documentation artifacts are present
- **AND** artifacts are published in machine-readable form

### Requirement: Deterministic Reproducibility for Hard Cases
Hard-case benchmark runs SHALL provide deterministic reproducibility metadata and thresholds.

#### Scenario: Repeat hard-case run on same runner class
- **WHEN** the same hard-case matrix is re-executed under equivalent environment fingerprint
- **THEN** convergence-class KPIs remain within configured reproducibility tolerances
- **AND** deviations beyond tolerance fail reproducibility checks

### Requirement: Cross-Class Non-Regression Budget
Benchmark gates SHALL enforce explicit cross-class non-regression budgets to prevent overfitting convergence heuristics to a single circuit.

#### Scenario: Local gain causes cross-class regression
- **WHEN** a change improves one challenge class but exceeds regression budget in another stable class
- **THEN** the gate fails deterministically
- **AND** the report identifies improved and regressed classes side-by-side

