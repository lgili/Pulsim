## Context
Pulsim v1 already contains robust native convergence features, event logic, and solver telemetry. However, transient execution currently includes overlapping solver orchestration branches, compatibility layers, and backend-specific recovery paths that duplicate mathematical work and complicate optimization.

This design defines a from-scratch solver architecture for the supported runtime path, focused on switched-converter simulation with two user-facing timestep modes:
- `fixed`
- `variable`

All advanced behavior remains internal to a single native core.

## Goals
- Maximize switched-converter robustness, precision, and performance in one native architecture.
- Keep user-facing control minimal (mode selection only in common workflows).
- Eliminate duplicated mathematical pipelines.
- Enforce phase gates so no implementation phase degrades objective metrics.
- Provide deterministic behavior and structured diagnostics.

## Non-Goals
- Preserving all legacy transient-backend configuration fields as supported behavior.
- Maintaining parallel solver architectures with independent residual/Jacobian pipelines.
- Introducing new device-physics models in this change.

## Design Principles
- **Single source of truth for equations**: one residual/Jacobian assembly pipeline.
- **One nonlinear engine**: same Newton/globalization/recovery services for all transient modes.
- **One linear service stack**: common direct/iterative path with deterministic fallback.
- **Segmented time integration**: events define segment boundaries; integrators solve segments.
- **No hidden divergence in code paths**: fixed and variable modes differ by timestep policy, not by equation pipeline.
- **Performance by design**: cache pattern reuse, minimize allocations, avoid repeated symbolic work.
- **Validation-first rollout**: each phase blocked by benchmark/parity/stress gates.

## High-Level Architecture
`TransientOrchestrator` coordinates shared services:
- `EquationAssembler`
- `NonlinearSolveEngine`
- `LinearSolveService`
- `EventScheduler`
- `StepPolicy` (`FixedStepPolicy` or `VariableStepPolicy`)
- `RecoveryManager`
- `TelemetryCollector`

`TransientOrchestrator` executes this loop:
1. Query next segment target from `EventScheduler`.
2. Ask `StepPolicy` for candidate step(s).
3. Solve implicit step via shared nonlinear/linear services.
4. Validate acceptance (error/event/convergence criteria).
5. Commit state/history/cache updates.
6. Emit deterministic telemetry and output samples.

## Mathematical Core
### DAE/MNA Form
The core solves a unified implicit form:
- `F(x_{n+1}, x_n, dt, t_{n+1}, mode_state) = 0`

`EquationAssembler` provides:
- residual evaluation
- Jacobian assembly
- optional Jacobian-vector products for Krylov modes
- event-sensitive stamping updates without branching into separate backends

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

Failure diagnostics must include:
- terminal reason code
- last residual metrics
- last escalation stage
- topology signature at failure

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

### Gate policy
- Phase N cannot start until Phase N-1 KPIs are non-regressive against frozen baseline.
- Any regression beyond threshold blocks merge.
- Thresholds are stored in benchmark configs and versioned with artifacts.

### Initial default thresholds
- success-rate regression: max `-0.5` percentage points
- parity RMS regression: max `+5%`
- runtime p95 regression: max `+5%`
- event-time error regression: max `+10%`

## Migration Plan
1. Freeze baseline metrics and artifacts.
2. Introduce unified services and adapt both modes to shared pipeline.
3. Migrate YAML/Python surfaces to canonical mode selection.
4. Remove supported legacy transient backend routing from runtime path.
5. Run full benchmark/parity/stress gates.
6. Archive superseded change streams and update docs.

## Risks and Mitigations
- Risk: Early refactor may temporarily reduce convergence on niche topologies.
  - Mitigation: phase gates + corpus expansion before enabling defaults.
- Risk: Over-aggressive de-duplication may hide mode-specific optimizations.
  - Mitigation: keep mode-specific `StepPolicy` modules while sharing math services.
- Risk: Event segmentation cost may reduce throughput.
  - Mitigation: breakpoint coalescing, tolerance windows, and cache-preserving reinit.
- Risk: Large API simplification may disrupt advanced users.
  - Mitigation: compatibility adapter with clear deprecation diagnostics for one migration window.

## Open Questions
- Final naming for canonical YAML/Python mode fields (`step_mode` vs `time_mode`).
- Exact deprecation window for legacy transient-backend keys.
- Whether expert-only controls remain fully public or move behind feature flags.
