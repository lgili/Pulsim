## ADDED Requirements

### Requirement: Linear-Time Segment Assembly

The assembly of a switch state's matrix SHALL cost time proportional to
the number of stamps. The sparsity structure of `G`, `C` and `J` SHALL
be built once per cache and per assembly regime (dynamic `dt > 0`,
static `dt = 0`); every later assembly SHALL reuse that structure,
zero the values and add the stamps in the same order. The assembled
`J` and `b` SHALL be equal, entry by entry and bit for bit, to the
matrices produced by stamping into empty matrices. `CacheMetrics` SHALL
report `structure_builds`.

#### Scenario: Structure built once
- **GIVEN** a lazy cache for a three-phase VSI
- **WHEN** a run visits 87 masks
- **THEN** `structure_builds` is at most 2

#### Scenario: A first visit costs O(nnz)
- **GIVEN** switched ladders with 310 and 3008 unknowns
- **WHEN** a mask is visited for the first time after the structure
  exists
- **THEN** no sparse-structure insertion is performed
  (`structure_builds` unchanged)
- **AND** the measured build time grows by at most 12× for 10× the
  unknowns (local measurement, not a CI assertion)

#### Scenario: Same matrix as before
- **WHEN** every mask of every reference circuit is assembled in both
  regimes
- **THEN** `J` has the same index arrays and byte-equal values, and `b`
  is byte-equal, compared with the previous assembly

### Requirement: Shared Symbolic Analysis Across Switch Masks

A cache SHALL compute the symbolic analysis of its matrix (fill-reducing
column ordering and elimination tree) once per assembly regime and
SHALL share it, read-only, among the solvers of all its segments, event
entries and the rank-1 solver. A solver SHALL verify that a matrix has
the sparsity structure the shared analysis was computed for before
using it, and SHALL analyse privately otherwise. `CacheMetrics` SHALL
report `symbolic_analyses` and `symbolic_fallbacks`. Factors computed
on the shared analysis SHALL be bit-equal to factors computed on a
private analysis of the same matrix.

#### Scenario: One analysis for all masks
- **GIVEN** a lazy cache for a three-phase VSI on the fixed engine
- **WHEN** a run visits 87 masks
- **THEN** `symbolic_analyses` is at most 2 and `symbolic_fallbacks`
  is 0

#### Scenario: Both regimes
- **GIVEN** a cache that solves static and dynamic systems
- **WHEN** each has been solved for several masks
- **THEN** `symbolic_analyses` is exactly 2

#### Scenario: Structure mismatch is detected
- **WHEN** a solver is asked to adopt the shared analysis for a matrix
  with a different sparsity structure
- **THEN** it analyses that matrix privately, `symbolic_fallbacks`
  increases by 1 and the solution is correct

#### Scenario: Results unchanged
- **WHEN** the reference golden circuits are run
- **THEN** states are `np.array_equal` to the goldens on `pwl` and
  `trbdf2`

### Requirement: Fused Right-Hand-Side And Permutation Passes

The cached solve SHALL form the right-hand side and apply the row
permutation in one pass, using a constant term stored already permuted
per segment, and SHALL accept a caller-owned work vector. The solution
SHALL be bit-equal to the one produced by forming the right-hand side
first and permuting it afterwards. Callers that pass no work vector
SHALL get a thread-local one.

#### Scenario: Identical solution
- **WHEN** the reference golden circuits are run with the fused solve
- **THEN** states are `np.array_equal` to the goldens

#### Scenario: Concurrent solves on one cache
- **GIVEN** two threads solving on the same warm cache without passing
  a work vector
- **WHEN** both complete
- **THEN** each solution equals the single-threaded one

### Requirement: Reciprocal Pivot Storage

The in-house LU SHALL store the reciprocal of each pivot and multiply by
it in back substitution only when the option is enabled; it SHALL be
enabled by default only after the solve microbenchmark shows at least
1.15× at n = 1502 and the owner has accepted the changed rounding.
Solutions with the option enabled SHALL agree with the divide form
within 1e-9 relative.

#### Scenario: Option off is the divide form
- **WHEN** the reference golden circuits are run with reciprocal pivots
  disabled
- **THEN** states are `np.array_equal` to the goldens

#### Scenario: Option on stays within tolerance
- **WHEN** the same circuits are run with reciprocal pivots enabled
- **THEN** `max|x − x_ref| ≤ 1e-9 · max(1, max|x_ref|)` over each run
- **AND** the LU residual tests pass at their existing tolerances
