## Context
Pulsim v1 already contains robust native convergence features, event logic, and solver telemetry. However, transient execution currently includes overlapping solver orchestration branches, compatibility layers, and backend-specific recovery paths that duplicate mathematical work and complicate optimization.

This design defines a from-scratch solver architecture for the supported runtime path, focused on switched-converter simulation with two user-facing timestep modes:
- `fixed`
- `variable`

The internal mathematical core is hybrid by design:
- primary path: event-driven state-space segment solving
- fallback path: shared nonlinear DAE solve with deterministic escalation

All advanced behavior remains internal to a single native core.

## Goals
- Maximize switched-converter robustness, precision, and performance in one native architecture.
- Keep user-facing control minimal (mode selection only in common workflows).
- Eliminate duplicated mathematical pipelines.
- Make event-driven state-space the default execution path for converter segments.
- Keep loss and thermal evolution inside the same accepted-step/event commit model.
- Enforce phase gates so no implementation phase degrades objective metrics.
- Provide deterministic behavior and structured diagnostics.

## Non-Goals
- Preserving all legacy transient-backend configuration fields as supported behavior.
- Maintaining parallel solver architectures with independent residual/Jacobian pipelines.
- Introducing new device-physics models in this change.

## Design Principles
- **Single source of truth for equations**: one residual/Jacobian assembly pipeline.
- **Hybrid-first execution**: use segmented state-space solve when valid, with deterministic DAE fallback.
- **One nonlinear engine**: same Newton/globalization/recovery services for all transient modes.
- **One linear service stack**: common direct/iterative path with deterministic fallback.
- **Segmented time integration**: events define segment boundaries; integrators solve segments.
- **Integrated electrothermal accounting**: switching/conduction losses and thermal updates commit on accepted segments/events.
- **No hidden divergence in code paths**: fixed and variable modes differ by timestep policy, not by equation pipeline.
- **Performance by design**: cache pattern reuse, minimize allocations, avoid repeated symbolic work.
- **Validation-first rollout**: each phase blocked by benchmark/parity/stress gates.

## High-Level Architecture
`TransientOrchestrator` coordinates shared services:
- `EquationAssembler`
- `SegmentModelService` (builds `E`, `A`, `B`, `c` for current topology segment)
- `SegmentStepperService` (state-space segment advance and acceptance hints)
- `NonlinearSolveEngine`
- `LinearSolveService`
- `EventScheduler`
- `StepPolicy` (`FixedStepPolicy` or `VariableStepPolicy`)
- `RecoveryManager`
- `LossService`
- `ThermalService`
- `TelemetryCollector`

`TransientOrchestrator` executes this loop:
1. Query next segment target from `EventScheduler`.
2. Ask `StepPolicy` for candidate step(s).
3. Build or reuse current segment model (`E/A/B/c`, topology signature).
4. Attempt segment solve via `SegmentStepperService` (primary path).
5. If segment solve is not admissible, execute shared nonlinear/linear fallback.
6. Validate acceptance (error/event/convergence criteria).
7. Commit state/history/cache updates.
8. Commit losses and thermal updates.
9. Emit deterministic telemetry and output samples.

## Mathematical Core
### DAE/MNA Form
The core solves a unified implicit form:
- `F(x_{n+1}, x_n, dt, t_{n+1}, mode_state) = 0`

`EquationAssembler` provides:
- residual evaluation
- Jacobian assembly
- optional Jacobian-vector products for Krylov modes
- event-sensitive stamping updates without branching into separate backends

### Hybrid Segment Form
For each accepted topology segment `sigma`, the primary path uses:
- `E_sigma x_dot = A_sigma x + B_sigma u + c_sigma`

Execution policy:
- **state-space primary** when the segment is piecewise-linear/affine under current model policy
- **DAE fallback** when nonlinearity, conditioning, or model constraints require implicit nonlinear solve

Fallback transitions are deterministic and telemetry-tagged (reason code + segment signature).

### Shared State Objects
- `SystemState`: current unknown vector and derived quantities.
- `HistoryState`: integration history and accepted-step metadata.
- `TopologySignature`: hash/fingerprint for switch topology and matrix pattern reuse.
- `SolveContext`: tolerances, mode, event window, and retry index.

## Timestep Modes
### Fixed Mode
Purpose: deterministic output grid and controller-coupled simulation.

Behavior:
- User dt defines macro grid.
- Internal substeps are allowed only for event alignment and convergence rescue.
- Output is committed on macro-grid boundaries.
- Segment path preference: state-space segment solve first, nonlinear fallback only when required.
- Integrator policy:
  - event-adjacent substeps: stiff-safe low-order profile (e.g., BDF1)
  - smooth segments: higher-accuracy stiff-safe profile (e.g., TRBDF2)

Guarantees:
- deterministic sample times on user grid
- bounded internal work per macro step via retry limits

### Variable Mode
Purpose: best accuracy/runtime tradeoff.

Behavior:
- Adaptive step control with LTE and Newton-feedback coupling.
- Earliest-event clipping before each candidate step.
- PI-like controller with stability guards for growth/shrink limits.
- Segment path preference: state-space segment solve first, nonlinear fallback only when required.
- Integrator policy defaults to stiff-stable methods and can switch order internally.

Guarantees:
- LTE-constrained acceptance
- deterministic event-time targeting inside tolerance

## Nonlinear Solve Engine
The nonlinear engine is shared by both timestep modes and DC-derived assists.

Components:
- Newton core
- line-search damping
- trust-region bounds
- voltage/current step limiting
- optional Anderson/Broyden/JFNK acceleration hooks (same interfaces)

Acceptance criteria:
- weighted variable error
- residual norm
- finite-value checks
- stall/divergence detection

## Recovery Manager
Recovery ladder is deterministic and ordered:
1. retry with reduced dt
2. stronger globalization (damping/trust shrink)
3. temporary stiffness profile downgrade
4. transient regularization escalation (gmin/diagonal safeguards)
5. bounded repeat with telemetry trace
6. fail with explicit reason code when budget exhausted

Rules:
- no infinite retries
- each escalation increments a typed telemetry counter
- state rollback uses last accepted state only

## Loss and Electrothermal Core
`LossService` and `ThermalService` are part of the transient commit path, not post-processing.

### Loss accumulation policy
- **Switching losses** are committed at event transitions (`Eon`, `Eoff`, `Err` paths).
- **Conduction losses** are integrated over accepted segments only.
- Rejected attempts do not double-count energy.

### Thermal coupling policy
- Thermal states are updated through deterministic RC networks per device.
- Coupling modes:
  - one-way (`electrical -> thermal`)
  - staged two-way (`electrical -> thermal -> parameter refresh on commit windows`)
- Temperature-dependent electrical parameter refresh is bounded and deterministic.

### Electrothermal consistency
- Every accepted segment stores electrical energy, dissipated energy, and thermal state deltas.
- The runtime emits per-device and aggregate electrothermal summaries.

## Linear Solve Service
`LinearSolveService` is shared and policy-driven.

Features:
- deterministic primary/fallback order
- direct and iterative options behind one interface
- topology-signature-based symbolic reuse
- preconditioner lifecycle management and safe invalidation
- scaling and conditioning safeguards

Performance controls:
- pattern reuse on unchanged topology
- factorization reuse when numerically safe
- iterative early-stop and fallback thresholds

## Event Scheduler
`EventScheduler` provides deterministic segment boundaries for:
- PWM carrier boundaries
- dead-time transitions
- comparator/threshold crossings
- externally declared breakpoints

Algorithm:
- compute earliest pending boundary
- clip candidate step to boundary
- run local refinement for crossing timestamp
- emit event and update topology signature
- trigger controlled relinearization/reinitialization only when necessary

## Telemetry and Diagnostics
Telemetry is first-class and mode-independent.

Required fields:
- total steps and accepted/rejected counts
- nonlinear and linear iteration counts
- fallback/recovery trace with reason codes
- event count and event-time refinement statistics
- runtime wall-clock and phase timings
- cache hit/miss stats for symbolic/factorization/preconditioner reuse
- segment-path counters (state-space-primary vs DAE-fallback)
- per-device and aggregate loss metrics (switching/conduction/total)
- thermal telemetry (peak temperature, final temperature, coupling mode)

Failure diagnostics must include:
- terminal reason code
- last residual metrics
- last escalation stage
- topology signature at failure
- last segment-path classification and electrothermal commit status

## API and Configuration Surface
### Canonical user-facing controls
- timestep mode: `fixed` or `variable`
- simulation interval and base dt

### Expert controls
Advanced controls remain available in explicit expert sections, but are not required for normal usage.

### Legacy key handling
- legacy transient backend keys return deterministic migration diagnostics
- parser emits actionable mapping hints to new mode/profile fields

## Code De-duplication Strategy
- All residual/Jacobian assembly logic moved into shared services.
- Remove duplicate transient step implementations that differ only by backend wrapper.
- Ensure fixed/variable engines call the same nonlinear and linear services.
- Enforce interface-level reuse via architecture tests (component-level tests that fail on divergent duplicate pathways).

## Performance Strategy
- Zero-allocation hot loop target for steady-state stepping.
- Preallocated workspaces per solver context.
- Sparse pattern reuse keyed by topology signature.
- Minimize full reanalysis after event; use targeted invalidation.
- Deterministic ordering to stabilize cache behavior and regression comparability.

## Accuracy Strategy
- Event segmentation before integration acceptance.
- Consistent local-error accounting near discontinuities.
- Integrator profile selection based on segment type (event-adjacent vs smooth).
- Strict acceptance rules that combine LTE and nonlinear health metrics.

## Validation and Phase Gates
Each implementation phase must pass objective gates before proceeding.

### Core KPI set
- `convergence_success_rate`
- `parity_rms_error`
- `parity_final_value_error`
- `event_time_error`
- `runtime_p50` and `runtime_p95`
- `recovery_rate` and `mean_retries`
- `state_space_primary_ratio`
- `dae_fallback_ratio`
- `loss_energy_balance_error`
- `thermal_peak_temperature_delta`

### Gate policy
- Phase N cannot start until Phase N-1 KPIs are non-regressive against frozen baseline.
- Any regression beyond threshold blocks merge.
- Thresholds are stored in benchmark configs and versioned with artifacts.

### Initial default thresholds
- success-rate regression: max `-0.5` percentage points
- parity RMS regression: max `+5%`
- runtime p95 regression: max `+5%`
- event-time error regression: max `+10%`
- loss-energy-balance regression: max `+5%` on converter matrix
- thermal peak-temperature regression: max `+5%` on electrothermal matrix

## Migration Plan
1. Freeze baseline metrics and artifacts.
2. Introduce unified services and adapt both modes to shared pipeline.
3. Add segment-model and segment-step services (state-space primary path).
4. Integrate losses and thermal services into accepted-step/event commits.
5. Migrate YAML/Python surfaces to canonical mode selection.
6. Remove supported legacy transient backend routing from runtime path.
7. Run full benchmark/parity/stress/electrothermal gates.
8. Archive superseded change streams and update docs.

## Risks and Mitigations
- Risk: Early refactor may temporarily reduce convergence on niche topologies.
  - Mitigation: phase gates + corpus expansion before enabling defaults.
- Risk: Over-aggressive de-duplication may hide mode-specific optimizations.
  - Mitigation: keep mode-specific `StepPolicy` modules while sharing math services.
- Risk: Event segmentation cost may reduce throughput.
  - Mitigation: breakpoint coalescing, tolerance windows, and cache-preserving reinit.
- Risk: Incorrect segment classification may overuse fallback and reduce performance.
  - Mitigation: deterministic classification rules + KPI guard on state-space primary ratio.
- Risk: Electrothermal coupling may destabilize difficult nonlinear cases.
  - Mitigation: staged coupling policy, bounded parameter refresh, and conservative fallback triggers.
- Risk: Large API simplification may disrupt advanced users.
  - Mitigation: compatibility adapter with clear deprecation diagnostics for one migration window.

## Open Questions
- Final naming for canonical YAML/Python mode fields (`step_mode` vs `time_mode`).
- Exact deprecation window for legacy transient-backend keys.
- Whether expert-only controls remain fully public or move behind feature flags.
