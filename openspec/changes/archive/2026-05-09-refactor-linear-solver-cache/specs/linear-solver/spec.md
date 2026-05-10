## ADDED Requirements

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

## MODIFIED Requirements

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
