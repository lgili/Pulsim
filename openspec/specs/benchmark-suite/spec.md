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
The ngspice comparator SHALL use the same Python runtime simulation path as the core benchmark runner.

#### Scenario: Pulsim vs ngspice comparison
- **WHEN** `benchmark_ngspice.py` executes a mapped benchmark pair
- **THEN** Pulsim waveforms are generated through the Python runtime path used by the benchmark suite
- **AND** comparison metrics are computed from those outputs

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
The system SHALL support the validation types `analytical`, `reference`, and `ngspice` for benchmark comparisons.

#### Scenario: Validate against a reference
- **WHEN** a benchmark uses validation type `reference`
- **THEN** outputs are compared against the stored baseline data

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

### Requirement: Property-Based Testing Harness
The benchmark/test infrastructure SHALL include a property-based testing harness checking physical invariants across randomly-generated circuits.

#### Scenario: Property suite runs in CI
- **WHEN** the standard CI pipeline runs
- **THEN** the property-based suite executes within a ≤30 s budget
- **AND** failures block the merge

#### Scenario: Failure produces minimal repro
- **GIVEN** a property test that fails on a generated circuit
- **WHEN** Hypothesis shrinks the failure
- **THEN** the minimal repro circuit is emitted as YAML
- **AND** added to the regression corpus

### Requirement: KCL/KVL Per-Step Invariant
Property tests SHALL assert KCL (current balance at every node) and KVL (voltage balance around every loop) at every accepted simulation step within numerical tolerance.

#### Scenario: KCL on accepted step
- **WHEN** any simulation step is accepted
- **THEN** for every non-ground node, the absolute sum of incident currents is below 1e-6 relative to total source magnitude

#### Scenario: KVL on independent loop
- **WHEN** any simulation step is accepted
- **THEN** for every independent loop in the circuit, the sum of branch voltages is below 1e-6 relative tolerance

### Requirement: Tellegen and Energy Invariants
Property tests SHALL assert Tellegen's theorem and lossless energy conservation in compatible test circuits.

#### Scenario: Tellegen invariant
- **WHEN** any simulation step is accepted
- **THEN** `Σ_branches (v_k · i_k)` for compatible v, i is below 1e-6 absolute (with appropriate sign convention)

#### Scenario: Lossless energy conservation
- **GIVEN** a lossless RLC test circuit (R = 0)
- **WHEN** simulation runs across a full transient
- **THEN** `stored_energy(t) - integral(P_src dt)` is below 1e-6 relative tolerance over the transient

### Requirement: Passivity Invariant
Property tests SHALL assert per-element passivity: resistors dissipate (`v·i ≥ 0`), capacitors and inductors balance (cycle-averaged `v·i = 0`).

#### Scenario: Resistor passivity
- **WHEN** a resistor is part of any random circuit
- **THEN** at every step, `v_R · i_R` ≥ 0 within numerical noise

#### Scenario: Capacitor cycle balance
- **GIVEN** a capacitor in a periodic steady-state circuit
- **WHEN** integration over one period completes
- **THEN** `integral(v_C · i_C dt)` over the period is below 1e-6 relative

### Requirement: Periodicity Invariant
Property tests SHALL assert periodicity in periodic-steady-state results: `x(t+T) ≈ x(t)`.

#### Scenario: PWM steady-state periodicity
- **GIVEN** a PWM-driven circuit operating in periodic steady state
- **WHEN** simulation captures two consecutive periods
- **THEN** the state vectors at corresponding phase points match within 1e-4 relative

### Requirement: PWL Mode No-Newton Property
Property tests SHALL assert that in PWL mode within a stable topology window, Newton iteration count is zero.

#### Scenario: PWL stable window
- **GIVEN** a converter operating in PWL mode with no event in the current step
- **WHEN** the step is accepted
- **THEN** `BackendTelemetry.nonlinear_iterations` for that step equals 0

### Requirement: C++ Property Tests
The C++ test suite SHALL include RapidCheck-based property tests for invariants verifiable at MNA stamp level (Tellegen, KCL, KVL).

#### Scenario: C++ property test
- **WHEN** `ctest` runs
- **THEN** the RapidCheck-based property tests execute
- **AND** failures produce shrunken minimal-repro inputs

