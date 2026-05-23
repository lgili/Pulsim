## ADDED Requirements

### Requirement: Piecewise-Linear Switching Engine Activation
The v1 kernel SHALL provide a piecewise-linear (PWL) state-space integration path for transient simulation of switching circuits, selectable per simulation and per device.

#### Scenario: Auto mode with all switching devices supporting Ideal
- **GIVEN** a circuit whose switching devices all declare PWL support
- **WHEN** `simulation.switching_mode = auto`
- **THEN** the kernel resolves the run to the PWL path
- **AND** telemetry reports `state_space_primary_steps > 0` and `dae_fallback_steps == 0` for stable topology windows

#### Scenario: Behavioral fallback when device opts out
- **GIVEN** at least one switching device with `switching_mode = behavioral`
- **WHEN** `simulation.switching_mode = auto`
- **THEN** the kernel resolves to the DAE path for the entire simulation
- **AND** telemetry reports the resolved mode in `BackendTelemetry.formulation_mode`

### Requirement: No-Newton Stable Topology Stepping
In PWL mode, the kernel SHALL execute time steps within a stable topology using only a single linear solve, without Newton iteration.

#### Scenario: Stable topology window
- **GIVEN** a converter operating in steady-state with no event in the current step
- **WHEN** the kernel advances by one accepted step
- **THEN** Newton iteration count for the step is zero
- **AND** the linear solver is invoked exactly once with the cached topology factorization

#### Scenario: Topology transition on event
- **GIVEN** a switch event detected within the candidate step
- **WHEN** the step is processed
- **THEN** the kernel bisects to the event time within `event_tolerance`
- **AND** snaps the accepted step at the event boundary
- **AND** rebuilds the segment model for the new topology before continuing

### Requirement: Topology Signature as Bitmask
The kernel SHALL identify topology equivalence classes via a deterministic bitmask over switching-device states, independent of numeric Jacobian values.

#### Scenario: Identical bitmask reuses cached factorization
- **WHEN** consecutive accepted steps share the same topology bitmask
- **THEN** the cached symbolic and numeric factorization is reused without recomputation
- **AND** telemetry records a topology cache hit

#### Scenario: Bitmask change invalidates only that topology entry
- **WHEN** a single switch transitions while others remain
- **THEN** the new bitmask is computed in O(1)
- **AND** the previous cache entry remains valid for future reuse if the prior topology recurs

### Requirement: PWL Topology Cache Bound
The kernel SHALL bound the topology cache and emit a deterministic diagnostic when the bound is exceeded.

#### Scenario: Cache within bound
- **GIVEN** `simulation.pwl_topology_cache_max = 4096`
- **WHEN** the simulation visits ≤4096 distinct topology bitmasks
- **THEN** all entries remain cached and the run continues without diagnostic

#### Scenario: Cache pressure with eviction
- **GIVEN** the cache is full and a new topology occurs
- **WHEN** the kernel records the new entry
- **THEN** the least recently used entry is evicted
- **AND** telemetry increments `pwl_topology_cache_evictions`

#### Scenario: Topology explosion diagnostic
- **WHEN** topology evictions exceed `pwl_topology_explosion_threshold` (default 100)
- **THEN** the kernel returns `SimulationDiagnosticCode::PwlTopologyExplosion`
- **AND** the diagnostic message includes the evicted-vs-cached counts

### Requirement: PWL State-Space Decomposition
The kernel SHALL decompose the MNA system into reactive (M) and resistive (N) parts so that PWL stepping advances `M ẋ + N x = b(t)` directly without residual evaluation.

#### Scenario: Capacitor and inductor contributions go to M
- **GIVEN** a circuit with capacitors and inductors
- **WHEN** the segment model is built
- **THEN** capacitive `(C / dt)` and inductive `(L / dt)` contributions appear in M
- **AND** resistive contributions appear in N
- **AND** independent sources contribute to b(t) only

#### Scenario: Switch state determines N entries
- **WHEN** a PWL switch is in `on` state
- **THEN** N includes its `Ron` conductance
- **WHEN** the switch is in `off` state
- **THEN** N includes its `Roff` conductance only

## MODIFIED Requirements

### Requirement: Solver Telemetry
The v1 kernel SHALL expose solver telemetry for debugging and regression tracking.

#### Scenario: Telemetry capture
- **WHEN** a simulation completes
- **THEN** the result SHALL include counts of nonlinear iterations, linear iterations, and fallback events
- **AND** the selected solver policies SHALL be reported in a structured form

#### Scenario: PWL telemetry capture
- **WHEN** a PWL-mode simulation completes
- **THEN** the result SHALL include `state_space_primary_steps`, `dae_fallback_steps`, `pwl_topology_cache_size`, `pwl_topology_transitions`, `pwl_event_bisections`, and `pwl_topology_cache_evictions`
- **AND** these counters SHALL be deterministic across reruns on identical hardware

### Requirement: Hot-Path Allocation Discipline
The v1 kernel SHALL enforce allocation-bounded steady-state stepping in hot loops, with deterministic cache reuse/invalidation across topology transitions.

#### Scenario: Stable topology steady-state stepping
- **WHEN** repeated accepted steps run under unchanged topology signature
- **THEN** the hot stepping path performs no unplanned dynamic allocations
- **AND** reusable solver/integration caches are reused

#### Scenario: Topology transition cache invalidation
- **WHEN** a switch/event changes topology signature
- **THEN** incompatible cache entries are invalidated deterministically before next solve
- **AND** new cache state is rebuilt under the active signature

#### Scenario: PWL hot-path allocation bound
- **WHEN** a PWL simulation runs steady-state stepping under unchanged topology
- **THEN** segment-model allocation count remains zero across the steady-state window
- **AND** linear-factor reuse is observed via `linear_factor_cache_hits`
