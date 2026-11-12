## ADDED Requirements

### Requirement: Newton Path Persistent Symbolic Analysis

A transient run with Newton devices SHALL perform the symbolic analysis
of the combined Jacobian once per distinct sparsity pattern, and SHALL
reuse it for every later Newton iteration of the run that has the same
pattern. The pattern SHALL be checked after each nonlinear refresh; a
changed pattern SHALL trigger a new analysis, never a factorisation on
a stale one. Callers that pass no workspace (the DC paths) SHALL keep
the previous behaviour.

#### Scenario: One analysis per run
- **GIVEN** a Shockley-diode buck run for 40 000 steps at a fixed `dt`
- **WHEN** the run completes
- **THEN** `counters["newton_analyses"]` is 1
- **AND** `counters["newton_factorizations"]` is at most
  `counters["newton_iterations"] + counters["newton_lm_attempts"]`

#### Scenario: Static and dynamic assembly are two patterns
- **GIVEN** a run that solves both a `dt = 0` and a `dt > 0` system
- **WHEN** the run completes
- **THEN** `counters["newton_analyses"]` is at most 2

#### Scenario: A refresh that changes its stamp positions
- **GIVEN** a user refresh that stamps an extra entry from iteration 3
- **WHEN** the solve continues
- **THEN** the analysis is redone once, the solve converges to the same
  root as without reuse, and `newton_analyses` increased by 1

### Requirement: Newton Path In-Place Jacobian

The combined Jacobian of a Newton iteration SHALL be written in place
into storage with the union pattern of the linear segment matrix and
the nonlinear stamps. Its pattern and every value SHALL equal those of
the sum `J_lin + J_nl` formed as a new matrix.

#### Scenario: Entry-by-entry identity
- **GIVEN** any iterate of a circuit in the Newton reference set
- **WHEN** the combined Jacobian is formed in place and as a new sum
- **THEN** the two matrices have the same index arrays and byte-equal
  values

### Requirement: Newton Linear-Solve Counters

A run SHALL report `newton_solves`, `newton_iterations`,
`newton_analyses`, `newton_factorizations`,
`newton_partial_refactors` and `newton_lm_attempts`. The counters SHALL
be deterministic for a given circuit and options and SHALL be zero for
a circuit without Newton devices.

#### Scenario: Linear circuit reports zeros
- **WHEN** a circuit without Newton devices is run
- **THEN** all six counters are 0

#### Scenario: Reproducible counts
- **WHEN** the same Newton circuit is run twice with the same options
- **THEN** the six counters are equal between the runs

### Requirement: Newton Low-Rank Refactorisation

The Newton path SHALL update its factor only along the elimination
paths of the columns the nonlinear devices touch, for an iteration
whose mask and step size are unchanged, when the in-house LU backs the
Newton path and the low-rank update is enabled. A pivot fault SHALL
fall back to a full factorisation of the same matrix and SHALL be
counted.

#### Scenario: Device columns only
- **GIVEN** a rectifier with one Shockley diode and a 500-section
  ladder, in-house backend, low-rank update enabled
- **WHEN** the run completes
- **THEN** `counters["newton_partial_refactors"]` is at least 80 % of
  `counters["newton_iterations"]`
- **AND** the iteration counts equal those of the run with full
  factorisations, with states within 1e-9 relative

#### Scenario: Pivot fault falls back
- **GIVEN** an iteration where the updated pivot fails the pivot test
- **WHEN** the low-rank update returns failure
- **THEN** the matrix is fully factorised and the solve continues
  without an error
