## ADDED Requirements

### Requirement: Newton Iteration Workspace

The fixed-step and TR-BDF2 engines SHALL own one Newton workspace per
run and SHALL pass it to every Newton solve of that run. After the first
step, a Newton iteration SHALL perform no heap allocation outside the
linear solver's own factorisation; when the in-house LU backs the
Newton path, a Newton iteration SHALL perform no heap allocation at
all. The workspace SHALL NOT be shared between concurrent runs.

#### Scenario: No allocation per iteration, reference backend
- **GIVEN** a Shockley-diode buck on the fixed engine with the Eigen
  backend
- **WHEN** steps 2 to 1000 run under the allocation counters of
  `test_zero_alloc`
- **THEN** the only allocations observed are inside
  `Eigen::SparseLU::factorize`

#### Scenario: No allocation per iteration, in-house backend
- **GIVEN** the same circuit with the in-house LU backend
- **WHEN** steps 2 to 1000 run under the allocation counters
- **THEN** 0 heap allocations are observed

#### Scenario: Concurrent runs do not share a workspace
- **GIVEN** two threads running the same Newton circuit on one eagerly
  built cache
- **WHEN** both runs complete
- **THEN** each result equals the single-threaded result

### Requirement: Newton Persistent Solver Result Identity

A run that uses the persistent Newton workspace SHALL, with the
reference (Eigen) backend, produce the same states, the same
switch-state record and the same number of Newton iterations per solve
as a run that builds a new solver for every iteration. With the in-house backend, the number
of Newton iterations per solve SHALL be unchanged and states SHALL agree
with the reference backend within 1e-9 relative.

#### Scenario: Bit-identical with the reference backend
- **WHEN** the Newton reference circuits are run on `pwl` and `trbdf2`
- **THEN** states are `np.array_equal` to the goldens and
  `counters["newton_iterations"]` equals the baseline value

#### Scenario: Tolerance with the in-house backend
- **WHEN** the Newton reference circuits are run with the in-house
  backend
- **THEN** `counters["newton_iterations"]` equals the reference-backend
  value
- **AND** `max|x − x_ref| ≤ 1e-9 · max(1, max|x_ref|)` over the run
