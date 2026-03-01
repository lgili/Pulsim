## 0. Baseline freeze and guardrails
- [x] 0.1 Freeze current benchmark/parity/stress artifacts as baseline for regression comparison.
- [x] 0.2 Define KPI thresholds in benchmark configs (`success_rate`, `parity_rms_error`, `event_time_error`, `runtime_p95`).
- [x] 0.3 Add CI gate that blocks progression when KPI regression exceeds thresholds.
- [x] 0.4 Document baseline hardware/environment fingerprint for fair runtime comparison.

## 1. Unified service interfaces
- [x] 1.1 Introduce shared interfaces for `EquationAssembler`, `NonlinearSolveEngine`, `LinearSolveService`, `EventScheduler`, `RecoveryManager`, and `TelemetryCollector`.
- [x] 1.2 Refactor transient orchestrator to depend only on these interfaces.
- [x] 1.3 Add architecture tests ensuring fixed/variable modes call the same nonlinear and linear services.
- [x] 1.4 Gate: no KPI regression vs baseline after interface extraction.

## 2. Hybrid mathematical pipeline (state-space + fallback)
- [x] 2.1 Implement `SegmentModelService` to build/reuse `E/A/B/c` segment models keyed by topology signature.
- [x] 2.2 Implement `SegmentStepperService` as primary path with deterministic handoff to shared nonlinear DAE fallback.
- [x] 2.3 Remove duplicate step solve logic that only differs by wrapper/backend branch.
- [x] 2.4 Add unit tests for residual/Jacobian parity and segment-vs-fallback consistency across mode contexts.
- [x] 2.5 Gate: parity error, convergence success, and state-space-primary ratio are non-regressive.

## 3. Fixed-step engine
- [x] 3.1 Implement `FixedStepPolicy` with macro-grid determinism.
- [x] 3.2 Implement internal substep handling for event alignment and bounded convergence rescue.
- [x] 3.3 Ensure outputs remain aligned to user macro timestep.
- [x] 3.4 Add converter regression tests for fixed-step event accuracy.
- [x] 3.5 Gate: event-time KPI and runtime KPI pass fixed-mode matrix.

## 4. Variable-step engine
- [x] 4.1 Implement `VariableStepPolicy` with LTE + Newton-feedback coupled control.
- [x] 4.2 Implement growth/shrink guards and event clipping in adaptive controller.
- [x] 4.3 Integrate stiff-profile switching policy near discontinuities.
- [x] 4.4 Add regression tests for adaptive acceptance/rejection determinism.
- [x] 4.5 Gate: accuracy/runtime KPI pass variable-mode matrix.

## 5. Deterministic recovery ladder
- [x] 5.1 Implement ordered recovery stages (dt backoff, globalization escalation, stiffness profile downgrade, transient regularization).
- [x] 5.2 Enforce bounded retry budgets and rollback-to-last-accepted semantics.
- [x] 5.3 Emit typed fallback trace entries for each stage.
- [x] 5.4 Add failure-path tests that validate deterministic reason codes.
- [x] 5.5 Gate: convergence success KPI improves or remains non-regressive on hard converter set.

## 6. Linear solver consolidation and caching
- [x] 6.1 Consolidate direct/iterative policy into one runtime linear service.
- [x] 6.2 Implement topology-signature keyed symbolic/factorization reuse.
- [x] 6.3 Implement deterministic preconditioner lifecycle and invalidation rules.
- [x] 6.4 Add telemetry for cache hit/miss and fallback transitions.
- [x] 6.5 Gate: runtime p95 improves or remains within threshold on large/stiff suites.

## 7. Event scheduler and segmented integration
- [x] 7.1 Implement unified event calendar for PWM boundaries, dead-time, and threshold crossings.
- [x] 7.2 Implement earliest-event segmentation and local timestamp refinement.
- [x] 7.3 Minimize reinitialization scope after event commits.
- [x] 7.4 Add switching converter tests for multiple events in a single macro interval.
- [x] 7.5 Gate: event-time KPI and stability KPI pass converter stress tiers.

## 8. User-facing surface simplification
- [x] 8.1 Add canonical mode field (`fixed`/`variable`) in YAML and Python runtime surfaces.
- [x] 8.2 Map canonical mode to internal solver profiles with deterministic defaults.
- [x] 8.3 Keep expert overrides under explicit advanced sections.
- [x] 8.4 Add migration diagnostics for deprecated legacy transient-backend keys.
- [x] 8.5 Add parser and binding tests for canonical mode selection and migration errors.

## 9. Legacy path decommissioning
- [x] 9.1 Remove supported runtime routing through legacy alternate transient-backend branches.
- [x] 9.2 Remove duplicate projected-wrapper/transient wrapper code from supported execution path.
- [x] 9.3 Delete dead configuration branches and stale tests tied only to removed paths.
- [x] 9.4 Update docs to reflect single native core and dual-mode user choice.
- [x] 9.5 Gate: full benchmark/parity/stress suite passes with no unsupported legacy-path dependency.

## 10. Final validation and rollout
- [x] 10.1 Run full benchmark matrix with frozen baseline comparison report.
- [x] 10.2 Run parity suite against ngspice/LTspice and publish KPI deltas.
- [x] 10.3 Run converter-focused stress tiers and verify convergence and runtime gates.
- [x] 10.4 Publish migration guide with before/after configuration examples.
- [x] 10.5 Mark all tasks complete only after KPI gates pass and reports are archived.

## 11. Loss and electrothermal integration
- [x] 11.1 Implement `LossService` commit hooks for switching losses on events and conduction losses on accepted segments.
- [x] 11.2 Implement `ThermalService` RC-network stepping with deterministic coupling modes.
- [x] 11.3 Integrate bounded temperature-to-electrical parameter refresh on commit boundaries.
- [x] 11.4 Add converter regression tests for per-device losses and peak/final junction temperatures.
- [x] 11.5 Gate: loss-energy-balance and thermal-peak KPIs pass without required-metric regression.
