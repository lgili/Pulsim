## MODIFIED Requirements

### Requirement: Python-First Benchmark Execution
The benchmark suite SHALL execute circuit scenarios through `pulsim.simulate` on the 2.0 kernel and SHALL NOT require an external `pulsim` executable or any module that is not in the repository.

#### Scenario: Run benchmark suite without CLI binary
- **WHEN** `benchmark_runner.py` is executed in an environment without a `pulsim` executable
- **THEN** scenarios are executed through Python runtime bindings
- **AND** the run produces standard benchmark artifacts

#### Scenario: Runner imports only the repository
- **WHEN** `benchmarks/tools/benchmark_runner.py` is imported with only the built `pulsim` package and the repository on the path
- **THEN** the import succeeds and no `pulsim_python_backend` module is referenced

### Requirement: YAML-first benchmark circuits
The system SHALL define benchmark circuits as YAML files in the `pulsim-v1` benchmark schema and SHALL read them with a reader that belongs to the benchmark harness and builds a `CircuitBuilder`, a gate schedule and run options for the 2.0 kernel. The kernel YAML loader SHALL NOT be required to accept that schema. Every benchmark circuit SHALL be listed in a triage manifest as `runs` or as `skipped` with a reason; no circuit SHALL be skipped without an entry.

#### Scenario: Load a benchmark circuit
- **WHEN** `benchmarks/circuits/linear/bridge_resistive_pure.yaml` is read by the harness reader and simulated on `engine='pwl'`
- **THEN** `V(a)` is 6.6667 V and `V(b)` is 4.0000 V within 1e-6 V

#### Scenario: Every circuit is accounted for
- **GIVEN** the 126 files under `benchmarks/circuits/`
- **WHEN** the triage test runs
- **THEN** the manifest has 126 entries, every `runs` entry completes on `engine='pwl'`, every `skipped` entry fails to load for the reason it states, and a circuit file without an entry fails the test

#### Scenario: A gated switch follows its v1 control source
- **GIVEN** a v1 `vcswitch` whose control node is driven by a `pwm` voltage source at 5 kHz and 50 % duty
- **WHEN** the reader builds the case
- **THEN** the 2.0 run uses a native schedule with period 200 µs and duty 0.5 for that switch, and the report lists the mapping

## ADDED Requirements

### Requirement: Time-Weighted Comparison Metrics
The benchmark harness SHALL compute every waveform error, mean and RMS as a time-weighted quantity over the stated window and SHALL NOT use an unweighted average over samples. The waveform error SHALL be `sqrt(∫(y − ref)² dt / window) / span`, integrated on the trace's own time grid against an analytic reference, and on the union of both time grids against another trace. For an observable that is not a state variable the window SHALL exclude the initial instant.

#### Scenario: Clustered grid does not bias the mean
- **GIVEN** the ramp `y = t` on [0, 1] sampled 1000 times in [0, 0.01] and 10 times in (0.01, 1]
- **WHEN** the harness computes its mean
- **THEN** the result is 0.5000 within 1e-12 (the unweighted mean of the samples is 0.0104)

#### Scenario: Known offset
- **WHEN** a trace equal to the reference plus one tenth of its span is compared with the reference
- **THEN** the waveform error is 0.1000 within 1e-12

### Requirement: Differential Engine Runner
The benchmark harness SHALL run any benchmark circuit on each of the engines `pwl`, `trbdf2` and `dsed` and SHALL classify each run as `agree`, `refused`, `disagree` or `error` against a stated reference. The reference SHALL be, in order of preference, the analytic solution, an ngspice waveform, the variable-step engine at `rtol = 1e-8`, or the fixed-step engine at a step shown to be converged. `refused` SHALL be reported only for the engine's typed refusal. `disagree` and `error` SHALL fail the run. The report SHALL name the reference and both error metrics for every cell.

#### Scenario: RC charge agrees on three engines
- **GIVEN** 10 V, 1 kΩ, 1 µF, 5 ms
- **WHEN** the runner compares each engine with `10·(1 − e^{−t/1 ms})`
- **THEN** the three cells are `agree` with waveform error at most 3.2e-3 on `pwl` at `dt = 10 µs`, 1.2e-6 on `trbdf2` at `rtol = 1e-6` and 1e-6 on `dsed`

#### Scenario: A silently different circuit is not `agree`
- **GIVEN** 10 V, 1 kΩ, 100 nF in parallel with a 1 µF junction capacitor
- **WHEN** the runner compares `dsed` with the fixed-step reference at 100 µs
- **THEN** the cell is `refused`, or `disagree` (6.32 V against 1.02 V) on a kernel that does not yet refuse it; it is never `agree`

### Requirement: Analytic Reference Set
The benchmark suite SHALL contain at least 20 circuits whose observable has a closed-form solution, or a solution exact up to a scalar root or a matrix exponential, covering linear RC, RL and RLC responses, sine and pulse sources, coupled and ideal transformers, a controlled source, the PWL diode with forward drop, a Shockley diode operating point, and CCM and DCM switching converters. Each circuit SHALL carry a tolerance per engine, and the tolerances SHALL be stored in the repository and SHALL only be loosened with a recorded reason.

#### Scenario: Underdamped RLC step
- **GIVEN** a 10 V step into 1 Ω, 1 mH and 10 µF in series
- **WHEN** the capacitor voltage is compared with `10·[1 − e^{−αt}(cos ω_d t + (α/ω_d) sin ω_d t)]`, `α = 500 s⁻¹`, `ω_d = 9987.49 rad/s`, over 10 ms
- **THEN** the first peak of the exact solution is 18.545 V, and the waveform error is at most 1.3e-3 on `pwl` at `dt = 1 µs` and 1.9e-4 on `trbdf2` at `rtol = 1e-6`

#### Scenario: A wrong component value is detected
- **WHEN** one component of an analytic circuit is changed by 1 %
- **THEN** the circuit fails on every engine that runs it

### Requirement: Fixed-Engine Golden Checkpoints
The benchmark suite SHALL pin the fixed-step engine on at least 40 circuits with golden checkpoints. Discrete data — the number of steps, the number of commutations and a digest of the stored switch-state record — SHALL be compared exactly. State values at 8 instants per circuit SHALL be compared with `|x − g| ≤ atol + rtol·|g|`, where `rtol` is recorded per circuit, passes on every CI platform, and does not exceed 1e-9 for circuits without Newton devices and 1e-6 otherwise. A bit-exact hash of floating-point output SHALL NOT be the regression key.

#### Scenario: A refactor that changes nothing passes on every platform
- **WHEN** the golden test runs on Linux x86-64, macOS arm64 and Windows with an unchanged kernel
- **THEN** it passes on all three

#### Scenario: A change of one part in a million is caught
- **WHEN** one resistor of a linear golden circuit is changed by 1e-6 relative
- **THEN** the golden test fails and reports the checkpoint, the state and the relative difference

#### Scenario: A changed switching sequence is caught
- **WHEN** a diode commutates one step earlier than in the golden run
- **THEN** the test fails on the switch-state digest before any value is compared

### Requirement: Counter-Based Performance Gate
The benchmark suite SHALL gate performance on deterministic counters reported by the kernel — steps, rejected steps, linear solves, numeric factorisations, symbolic analyses, Newton iterations and cache builds — and SHALL NOT assert on wall-clock time. On the fixed-step engine a counter SHALL NOT exceed its frozen value; on the variable-step and event-driven engines it SHALL NOT exceed 1.02 times its frozen value. Adding the counters SHALL NOT change any simulation result.

#### Scenario: An extra factorisation per step is caught
- **GIVEN** a build that performs one additional numeric factorisation per step
- **WHEN** the counter gate runs
- **THEN** it fails and names the circuit, the engine and the counter

#### Scenario: Wall time is not a criterion
- **WHEN** the same build is measured on a machine that is ten times slower
- **THEN** the gate result is unchanged

#### Scenario: Counters are increments only
- **WHEN** the existing test suite runs with the counters compiled in
- **THEN** every stored state trajectory is bit-identical to the run without them

### Requirement: Scheduled ngspice Comparison
The project SHALL run, on a schedule and on demand, a job that simulates every benchmark circuit that has an ngspice netlist with ngspice and with each Pulsim engine that accepts it, compares them with the time-weighted waveform error, records the ngspice version, publishes the per-circuit results as an artifact, and reports any cell whose outcome changed since the previous run. The job SHALL NOT gate pull requests.

#### Scenario: Nightly run
- **WHEN** the scheduled job runs
- **THEN** each of the benchmark circuits with a netlist under `benchmarks/spice/ngspice/` has one result per accepted engine in the artifact, with the reference marked as ngspice and its version

#### Scenario: A regression against ngspice is reported
- **WHEN** a cell that was `agree` in the previous run is `disagree`
- **THEN** the job reports the circuit, the engine and both error values
