## ADDED Requirements

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
