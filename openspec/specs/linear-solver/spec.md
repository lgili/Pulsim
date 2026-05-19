# linear-solver Specification

## Purpose
TBD - created by archiving change improve-convergence-algorithms. Update Purpose after archive.
## Requirements
### Requirement: AdvancedLinearSolver Enhancement

The existing AdvancedLinearSolver SHALL be enhanced with the new optimization features.

#### Scenario: AdvancedLinearSolver with KLU

- **GIVEN** AdvancedLinearSolver configured with Backend::Auto
- **WHEN** KLU is available
- **THEN** KLU is used for all solves
- **AND** symbolic caching is enabled by default

#### Scenario: Backward compatibility

- **GIVEN** existing code using AdvancedLinearSolver
- **WHEN** no options are specified
- **THEN** behavior is compatible with previous version
- **AND** new optimizations are applied transparently

### Requirement: Signature-Keyed Linear Solver Reuse Contract
The linear solver service SHALL reuse symbolic, numeric-factorization, and preconditioner assets keyed by deterministic topology signature and solver policy identity.

#### Scenario: Reuse on unchanged signature
- **WHEN** consecutive solves execute with identical topology signature and compatible solver policy
- **THEN** reusable symbolic/factorization/preconditioner assets are reused
- **AND** telemetry records reuse hits for each cache class

#### Scenario: Incompatible policy prevents unsafe reuse
- **WHEN** solver policy or conditioning class changes in a way that invalidates reuse safety
- **THEN** incompatible assets are not reused
- **AND** a deterministic rebuild path is executed

### Requirement: Deterministic Cache Invalidation Reasons
The linear solver service SHALL expose deterministic invalidation reasons for cache rebuilds and fallback transitions. Reasons SHALL be drawn from a typed `CacheInvalidationReason` enumeration; a string mirror SHALL be retained for backward compatibility with telemetry consumers that parse text labels.

#### Scenario: Topology-driven invalidation
- **WHEN** switching events produce a new topology signature
- **THEN** incompatible caches are invalidated with reason `TopologyChanged` (string: `"topology_changed"`)
- **AND** rebuild telemetry includes per-reason counters in `BackendTelemetry`

#### Scenario: Stability-driven invalidation
- **WHEN** numeric health checks detect conditioning degradation beyond configured thresholds, OR when an accepted step's matrix hash differs from the previous step's within an unchanged topology
- **THEN** cache reuse is disabled for that solve with reason `NumericInstability`
- **AND** recovery follows the configured deterministic solver fallback policy

### Requirement: Allocation-Bounded Solve Loop
Linear solve hot paths SHALL avoid unbounded dynamic allocation during steady-state reuse windows. The segment-primary path's `build_model` workspaces SHALL be hoisted to `mutable` members so successive accepted steps reuse storage. Newton-DAE workspace migration to `Simulator`-level pre-allocation is tracked as a follow-up; the present requirement covers the hot loop measured by the buck benchmark.

#### Scenario: Iterative steady-state solve sequence
- **WHEN** iterative solves run across a stable segment sequence with cache-compatible signatures
- **THEN** dynamic allocations remain within configured bounded setup/rebuild points
- **AND** the wall-clock signature is consistent with no per-iteration heap growth (351× speedup on the 1000-step buck benchmark vs Newton-DAE baseline; literal heap-counter assertion deferred)

### Requirement: Structured Linear Failure Reasons
Linear solver failures SHALL be reported with typed reason codes suitable for nonlinear recovery and KPI tracking.

#### Scenario: Iteration budget exhaustion
- **WHEN** iterative linear solve exceeds configured iteration budget
- **THEN** the solver reports a structured reason such as `iteration_limit`
- **AND** nonlinear recovery receives the reason code without text parsing

#### Scenario: Numerical breakdown
- **WHEN** solver encounters numerical breakdown or singularity
- **THEN** the solver reports a structured reason such as `numerical_breakdown` or `singular_matrix`
- **AND** telemetry captures the terminal solver and fallback chain position

### Requirement: Per-Step Numeric Factor LRU Cache
The linear solver hot path of the segment-primary stepper SHALL maintain a per-key LRU cache of analyzed-and-factorized linear solvers, keyed on a value-aware hash of the discretized system matrix `E = M + (dt/2)·N`. Each cache entry holds its own `RuntimeLinearSolver` instance with persistent `analyzePattern + factorize` state, plus a `shared_ptr` to the underlying `SegmentLinearStateSpace` to keep the matrix alive for the entry's lifetime.

#### Scenario: Steady-state PWM cycling reuses cached factors
- **GIVEN** a PWL converter cycling between a small set of (topology, dt) tuples (e.g. buck Q-on / Q-off)
- **WHEN** an accepted step's matrix hash matches an existing cache entry
- **THEN** only `solve(rhs)` runs — no `analyze`, no `factorize`
- **AND** `linear_factor_cache_hits` increments

#### Scenario: First encounter of a (topology, dt) tuple misses
- **WHEN** a step's matrix hash is not in the cache
- **THEN** a fresh entry is allocated (evicting LRU as needed) and full `analyze + factorize + solve` runs once
- **AND** `linear_factor_cache_misses` increments

#### Scenario: Newton iterations on the segment-primary path
- **GIVEN** PWL admissibility holds (segment-primary serves the step)
- **WHEN** the per-key cache hits or misses are recorded
- **THEN** every cached entry pre-pays its analyze cost exactly once and reuses it on every subsequent visit (the within-stepper symbolic-only reuse from the original Phase-2 single-slot design is subsumed by per-key entry persistence)

### Requirement: Typed Cache Invalidation Reasons
Every cache invalidation event recorded against `SegmentStepOutcome` and `BackendTelemetry` SHALL carry a reason drawn from a closed enumeration `CacheInvalidationReason { None, TopologyChanged, StampParamChanged, GminEscalated, SourceSteppingActive, NumericInstability, ManualInvalidate }`. The legacy free-form `cache_invalidation_reason` string field SHALL remain in the outcome struct as a backward-compat mirror; the typed and string fields SHALL be written together via `SegmentStepOutcome::set_invalidation_reason()`.

#### Scenario: Topology change reason
- **WHEN** a switch event produces a topology bitmask different from the previous accepted step's
- **THEN** the segment stepper's invalidation reason is set to `TopologyChanged`
- **AND** `BackendTelemetry::linear_factor_cache_invalidations_topology_changed` increments
- **AND** `linear_factor_cache_last_invalidation_reason_typed` carries the typed value
- **AND** `linear_factor_cache_last_invalidation_reason` carries the canonical wire-compat string `"topology_changed"`

#### Scenario: Numeric instability reason
- **WHEN** the previous step's matrix hash differs from this step's within an unchanged topology (e.g. a fractional `dt` produced by VCSwitch bisection-to-event), or the active solver's `solve` fails on a cached entry
- **THEN** the invalidation reason is `NumericInstability`
- **AND** `BackendTelemetry::linear_factor_cache_invalidations_numeric_instability` increments

#### Scenario: Cycling-back hit suppresses the invalidation tag
- **WHEN** the new step's matrix hash differs from the previous step's, but the LRU already holds a cached factor for the new hash (cycling-back case)
- **THEN** `linear_factor_cache_hit = true` is set on the outcome
- **AND** the invalidation reason is overwritten to `None` to express "no factor was discarded — the LRU had it ready"

### Requirement: Bounded Cache With LRU Eviction
The numeric factor LRU cache SHALL bound its occupancy at a compile-time default of **64 entries** (chosen for the typical power-electronics workload of ≤ 16 distinct topologies × a few `dt` values, with one cache hit ~ 100 µs of analyze + factorize amortized across thousands of steps). When capacity is reached, the least-recently-used entry SHALL be evicted before the new entry is inserted. Cache occupancy SHALL be observable via `SegmentStepperService::linear_factor_cache_occupancy()`.

#### Scenario: Cache fits the workload
- **GIVEN** a converter with ≤ 64 distinct (topology, dt) tuples across the run
- **THEN** every distinct tuple is held in the cache for the run's duration
- **AND** every revisit hits

#### Scenario: Pathological eviction
- **GIVEN** a run that produces > 64 distinct matrix hashes (e.g. continuously varying `dt`)
- **WHEN** the cache is full and a new hash arrives
- **THEN** the LRU entry is evicted to make room
- **AND** the cache size never exceeds 64

### Requirement: Hot-Path Workspace Hoisting
The segment-primary stepper SHALL avoid per-step heap allocation of `SparseMatrix` and `Vector` workspaces. The `DefaultSegmentModelService::build_model` workspaces (`M`, `N`, `b_now`, `b_next`, plus discard buffers used for the second `assemble_state_space` call at `t_target`) SHALL live as `mutable` members of the service so the steady-state hot loop reuses storage. Eigen's `resize/setZero` retains the underlying allocations when shrinking back to the same dimensions, so a stable-topology window pays no incremental heap cost after warmup.

#### Scenario: Steady-state stepping retains storage
- **GIVEN** a 1000-step steady-state window with stable topology and fixed `dt`
- **WHEN** the simulation runs
- **THEN** the segment model workspaces are reused without re-allocation
- **AND** the buck wall-clock benchmark reports a 351× speedup over the Newton-DAE baseline (98.6 % cache hit rate); a literal heap-allocation-zero assertion is tracked as a follow-up requiring a custom allocator harness

### Requirement: Aggregate Linear-Solver Telemetry
`SimulationResult::linear_solver_telemetry` SHALL aggregate analyze / factorize / solve counters across the shared linear-solve service (Newton-DAE workload) and the segment stepper's per-key LRU cache (segment-primary workload). The segment stepper SHALL expose its aggregate via `SegmentStepperService::linear_solver_telemetry()` and the simulation SHALL sum the two sources in `Simulator::finalize_transient_telemetry`.

#### Scenario: Mixed-mode run reports unified counters
- **GIVEN** a simulation that alternates between segment-primary and Newton-DAE steps
- **WHEN** the user reads `SimulationResult::linear_solver_telemetry`
- **THEN** `total_analyze_calls`, `total_factorize_calls`, and `total_solve_calls` reflect the union of both paths' workload
- **AND** the `last_*` fields prefer the segment-primary path's most-recent values when any segment-primary work happened during the run

### Requirement: Collapsed Public Linear Solver Enum

The system SHALL expose exactly three values in its public-facing
`LinearSolverKind` enumeration: `Auto`, `Direct`, and `Iterative`.

The internal 6-value implementation enum (`SparseLU`,
`EnhancedSparseLU`, `KLU`, `GMRES`, `BiCGSTAB`, `CG`) SHALL be
renamed `internal::LinearSolverImpl` and SHALL NOT appear in the
documented user API.

The auto-selector behind these three values SHALL behave as:

- `Auto` (default): picks `Direct` when the MNA matrix has fewer than
  5000 rows, otherwise `Iterative`.
- `Direct`: picks the best-of `KLU`, `EnhancedSparseLU`, `SparseLU`
  available on the current build.
- `Iterative`: picks the best-of `GMRES`, `BiCGSTAB` for the system
  shape.

#### Scenario: Small circuit auto-selects direct solver

- **GIVEN** an MNA matrix with 200 rows
- **WHEN** `LinearSolverKind::Auto` is in effect
- **THEN** the active solver is a direct solver (`KLU` if available,
  else `EnhancedSparseLU` or `SparseLU`)

#### Scenario: Large circuit auto-selects iterative solver

- **GIVEN** an MNA matrix with 50000 rows
- **WHEN** `LinearSolverKind::Auto` is in effect
- **THEN** the active solver is `GMRES` (or `BiCGSTAB` for symmetric
  systems)

#### Scenario: User forces direct on large circuit

- **GIVEN** a user explicitly sets
  `opts.advanced.linear_solver.kind = LinearSolverKind::Direct` on a
  50000-row MNA
- **WHEN** the linear stage runs
- **THEN** the direct solver is used regardless of system size

### Requirement: Solver-Quality Knob Replaces Preconditioner Enum

The system SHALL expose a single `solver_quality:
Fast|Default|Best` knob under `opts.advanced.linear_solver`
replacing the previous 5-value preconditioner enumeration
(`None_`, `Jacobi`, `ILU0`, `ILUT`, `AMG`).

The auto-selector SHALL map `solver_quality` to a preconditioner
internally:

- `Fast` → no preconditioner (linear solver runs without
  preconditioning when applicable)
- `Default` → ILU0 (or Jacobi when the active solver does not
  support ILU0)
- `Best` → ILUT or AMG depending on system characteristics

#### Scenario: Default solver quality picks ILU0

- **GIVEN** a user accepts defaults on a converter simulation
- **WHEN** the iterative solver runs
- **THEN** ILU0 preconditioning is applied internally

#### Scenario: Best quality unlocks AMG on large systems

- **GIVEN** `opts.advanced.linear_solver.solver_quality = SolverQuality::Best`
  on a 100000-row MNA
- **WHEN** the iterative solver runs
- **THEN** AMG preconditioning is selected

### Requirement: Iterative Refinement on Direct Solve

The system SHALL automatically apply one round of iterative
refinement after the direct solver's back-substitution when the
relative residual norm exceeds `10 · ε_machine`.

The refinement step SHALL:
- Compute `r = b - A·x` using the most-recent factorization's
  numeric workspace
- Solve `A·δ = r` using the same factorization (no re-factorization)
- Update `x ← x + δ`
- Increment the `linear_refinement_steps` telemetry counter

Refinement SHALL be skipped when the active solver is iterative
(GMRES, BiCGSTAB, CG) because those algorithms apply equivalent
refinement internally.

#### Scenario: Ill-conditioned matrix triggers refinement

- **GIVEN** an MNA matrix with condition number `> 1e10` arising
  from a floating-cap multilevel topology
- **WHEN** the direct solver runs
- **THEN** the post-solve residual exceeds the threshold
- **AND** one refinement step fires
- **AND** the final residual is `< 10 · ε_machine · ||b||`

#### Scenario: Well-conditioned matrix incurs no refinement

- **GIVEN** an RC circuit MNA with condition number `< 1e4`
- **WHEN** the direct solver runs
- **THEN** the post-solve residual is below the threshold
- **AND** `linear_refinement_steps == 0`

#### Scenario: Iterative solvers skip refinement

- **GIVEN** the active linear solver is `GMRES`
- **WHEN** a solve completes
- **THEN** the refinement check is skipped entirely
- **AND** `linear_refinement_steps == 0`

### Requirement: Auto-Select Always-On Migration

The system SHALL deprecate the user-facing
`LinearSolverStackConfig::auto_select` (boolean), `size_threshold`,
and `nnz_threshold` fields in v0.11 and SHALL remove them in v0.12.
The auto-selector SHALL become unconditionally active. Users who
need to override the auto-selector's choice SHALL set
`opts.advanced.linear_solver.kind = LinearSolverKind::Direct` or
`Iterative` explicitly.

**Rationale**: In practice the auto-selector should always be active
and the thresholds should be internal implementation choices that
the kernel maintainers tune from benchmark data — not user-visible
configuration.

#### Scenario: Legacy auto_select field warns in v0.11

- **GIVEN** a user sets `opts.linear_solver.auto_select = false` in
  v0.11 along with a hand-picked solver kind
- **WHEN** the simulator constructs
- **THEN** the user's explicit kind choice is honoured
- **AND** a deprecation warning surfaces pointing the user at the
  `LinearSolverKind::Direct | Iterative` override path

#### Scenario: Auto-select threshold fields removed in v0.12

- **GIVEN** the same code in v0.12 (after the removal cycle)
- **WHEN** the simulator constructs
- **THEN** the `auto_select`, `size_threshold`, and `nnz_threshold`
  fields no longer compile
- **AND** the migration error points the user at the explicit
  `LinearSolverKind` override

### Requirement: Preconditioner Enum Migration to SolverQuality

The system SHALL deprecate the 5-value `PreconditionerKind` enum
(`None_`, `Jacobi`, `ILU0`, `ILUT`, `AMG`) on the public API in
v0.11 and SHALL remove the public surface in v0.12. The internal
implementation enum SHALL move under `internal::PreconditionerImpl`
and SHALL remain reachable via
`opts.advanced.linear_solver.iterative_config.preconditioner` for
power users.

The canonical user-facing replacement SHALL be the
`solver_quality: Fast|Default|Best` knob (see the
"Solver-Quality Knob Replaces Preconditioner Enum" requirement above).

**Rationale**: The 5-value preconditioner enum leaks sparse-linear-
algebra implementation choices into the user-facing API. A
power-electronics user simulating a buck converter does not need to
know what ILU0 vs ILUT is to get a working simulation.

#### Scenario: Legacy PreconditionerKind selection warns in v0.11

- **GIVEN** a user sets the preconditioner enum directly in v0.11
- **WHEN** the iterative solver runs
- **THEN** the choice is honoured at runtime
- **AND** a deprecation warning surfaces pointing the user at
  `SolverQuality`

#### Scenario: Public PreconditionerKind removed in v0.12

- **GIVEN** the same code in v0.12
- **WHEN** the code compiles
- **THEN** the public enum no longer exists; only the internal one
  remains, accessible under `opts.advanced.linear_solver.iterative_config`

