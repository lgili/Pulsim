## 1. Contract and Schema
- [x] 1.1 Define canonical YAML contract (`simulation.post_processing`) for analysis jobs (FFT/THD/time metrics/power-efficiency/loop metrics).
- [x] 1.2 Define runtime option/result structures and typed diagnostic taxonomy for post-processing workflows.
- [x] 1.3 Add strict parser validation with deterministic field-path diagnostics for invalid jobs, windows, and signal references.

## 2. Kernel Runtime Implementation
- [x] 2.1 Implement backend post-processing pipeline with modular stages (signal resolver, window planner, metric engines, result assembler).
- [x] 2.2 Implement deterministic FFT/harmonic/THD computation path with explicit window function and fundamental-selection behavior.
- [x] 2.3 Implement deterministic time-domain and power metrics (RMS/mean/min/max/p2p/crest/ripple/power factor/efficiency).
- [x] 2.4 Implement typed undefined-metric behavior (reason codes) and fail-fast policy for invalid prerequisites.
- [x] 2.5 Ensure allocation-aware execution in repeated post-processing runs.

## 3. Result Surface and Telemetry
- [x] 3.1 Expose canonical structured post-processing outputs (scalar metrics, spectra, harmonics, job diagnostics).
- [x] 3.2 Provide metadata fields (unit/domain/source/signal binding) sufficient for frontend routing without heuristics.
- [x] 3.3 Emit telemetry required for KPI gating (runtime cost, determinism checks, undefined-metric counts).

## 4. Python Bindings and API
- [x] 4.1 Expose typed Python enums/classes/functions to configure and run post-processing jobs.
- [x] 4.2 Ensure class-based and procedural compatibility workflows can consume post-processing results.
- [x] 4.3 Provide structured Python error/diagnostic context for invalid post-processing configurations.

## 5. Benchmark and Regression Coverage
- [x] 5.1 Add deterministic fixture circuits for FFT/THD correctness (known harmonic content).
- [x] 5.2 Add deterministic fixture circuits for power/efficiency metrics with expected reference values.
- [x] 5.3 Add expected-failure benchmark fixtures for invalid windowing/sampling/signal contracts.
- [x] 5.4 Add determinism regression checks for repeated job execution.
- [x] 5.5 Add KPI gates for metric correctness and runtime-overhead non-regression.

## 6. Documentation and UX Boundary
- [x] 6.1 Publish backend contract docs with formulas/definitions for each supported metric family.
- [x] 6.2 Document frontend responsibilities and non-responsibilities for consuming post-processing outputs.
- [x] 6.3 Publish YAML + Python examples and a tutorial notebook for common workflows.
- [x] 6.4 Add migration notes from script-based post-processing to canonical backend jobs.

## 7. Quality Gates (Mandatory Before Merge)
- [x] 7.1 `openspec validate add-waveform-post-processing-tools --strict` passes.
- [x] 7.2 Targeted parser/kernel/python tests for post-processing pass locally and in CI.
- [x] 7.3 Benchmark KPI gate for post-processing correctness/determinism/overhead passes in CI.
- [x] 7.4 Acceptance evidence (artifacts, thresholds, formulas, known limitations) is recorded in change notes.
