## ADDED Requirements

### Requirement: Mask-Invariant Pattern With Branch-Form Switches

The matrix assembled for a circuit with branch-form switches SHALL
have the same sparsity pattern for every switch mask, the five
structural entries of each device being present in both states even
when their value is zero, so that one symbolic analysis serves all
masks. Numeric factorization SHALL use partial pivoting per mask, a
single-switch flip SHALL be applied as a rank-one row update, and a
flip that changes the pivot order SHALL fall back to a full numeric
refactorization that is counted in the cache metrics.

#### Scenario: Pattern over all masks
- **GIVEN** a bridge of four branch-form switches
- **WHEN** the matrix is assembled for each of the 16 masks
- **THEN** the column pointers and row indices are identical for all of
  them

#### Scenario: Zero diagonal
- **GIVEN** the mask with every switch ON and `R_on = 0`
- **WHEN** it is factorized and solved
- **THEN** the relative residual `‖J·x − b‖/‖b‖` is at most 1e-12 and no
  "numerically singular" error is raised

#### Scenario: Single flip
- **GIVEN** a factor for one mask and a mask that differs by one switch
- **WHEN** the flip is applied as a rank-one update
- **THEN** the solution equals that of a full refactorization within
  1e-12, and a flip that needs a new pivot order is counted as a
  fallback

### Requirement: Structural Singularity Report Per Mask

The kernel SHALL check the connectivity of a mask before factorizing
its matrix whenever an ideal open switch (`g_off = 0`) or an ideal
closed switch (`R_on = 0`) is present, and SHALL raise an error naming
the mask, the nodes and the switches when a node has no path to ground
or when closed ideal switches and voltage sources form a loop. The check SHALL run once per mask and be cached with its
segment.

#### Scenario: Floating node
- **GIVEN** two ideal switches in series with nothing else connected to
  the node between them
- **WHEN** the run reaches the mask with both open
- **THEN** it raises an error naming that node and both switches, with
  the two remedies, before any factorization

#### Scenario: Loop of ideal closed switches
- **GIVEN** two ideal switches in parallel
- **WHEN** the run reaches the mask with both closed
- **THEN** it raises an error naming both switches

#### Scenario: Checked once
- **GIVEN** a converter that visits 5 masks over one million steps
- **WHEN** the run completes
- **THEN** the connectivity check ran 5 times
