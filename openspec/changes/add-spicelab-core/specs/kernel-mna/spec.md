## ADDED Requirements

### Requirement: MNA Matrix Assembly
The kernel SHALL assemble the Modified Nodal Analysis (MNA) matrix equation Gx = b for circuit analysis.

#### Scenario: Linear circuit assembly
- **WHEN** a circuit with only linear components is provided
- **THEN** the system assembles G matrix (conductances) and b vector (sources)
- **AND** the matrix is sparse (CSR/CSC format) for efficiency

#### Scenario: Nonlinear circuit linearization
- **WHEN** a circuit contains nonlinear components
- **THEN** the system assembles the Jacobian matrix at each Newton iteration
- **AND** companion models are used for dynamic elements (L, C)

#### Scenario: Ground node handling
- **WHEN** the matrix is assembled
- **THEN** the ground node (node 0) is eliminated from the system
- **AND** all node voltages are relative to ground

### Requirement: Sparse Matrix Storage
The MNA matrix SHALL use efficient sparse storage formats.

#### Scenario: Matrix creation
- **WHEN** the circuit topology is known
- **THEN** a symbolic factorization determines the sparsity pattern
- **AND** memory is pre-allocated for the pattern

#### Scenario: CSR/CSC storage
- **WHEN** the matrix is stored
- **THEN** Compressed Sparse Row or Column format is used
- **AND** zero entries are not stored

#### Scenario: Matrix reuse
- **WHEN** only matrix values change (not structure)
- **THEN** the sparsity pattern is reused
- **AND** symbolic factorization is not repeated

### Requirement: Component Stamps
Each component type SHALL have a well-defined MNA stamp.

#### Scenario: Resistor stamp
- **WHEN** a resistor R connects nodes i and j
- **THEN** G[i,i] += 1/R, G[j,j] += 1/R, G[i,j] -= 1/R, G[j,i] -= 1/R

#### Scenario: Capacitor companion model
- **WHEN** a capacitor C connects nodes i and j using Backward Euler
- **THEN** it is modeled as conductance Geq = C/dt with history current Ieq
- **AND** the stamp updates at each timestep

#### Scenario: Inductor companion model
- **WHEN** an inductor L connects nodes i and j using Backward Euler
- **THEN** it adds a branch current variable to the system
- **AND** is modeled as Req = L/dt with history voltage Veq

#### Scenario: Voltage source stamp
- **WHEN** a voltage source V connects nodes i and j
- **THEN** it adds a branch current variable
- **AND** enforces v_i - v_j = V via extra row/column

#### Scenario: Current source stamp
- **WHEN** a current source I connects nodes i and j
- **THEN** b[i] -= I and b[j] += I (current flows from i to j)

#### Scenario: Controlled source stamps
- **WHEN** a controlled source (VCVS, VCCS, CCVS, CCCS) is defined
- **THEN** appropriate off-diagonal entries are added
- **AND** additional variables are added for current-controlled sources

### Requirement: Variable Ordering
The MNA system SHALL order variables for efficient factorization.

#### Scenario: Default ordering
- **WHEN** no ordering is specified
- **THEN** node voltages come first, then branch currents
- **AND** nodes are numbered in netlist order

#### Scenario: Minimum degree ordering
- **WHEN** optimization is enabled
- **THEN** AMD or similar ordering minimizes fill-in
- **AND** the permutation is applied symmetrically

### Requirement: Node Mapping
The system SHALL maintain a mapping between netlist node names and matrix indices.

#### Scenario: Node lookup
- **WHEN** a node name is queried
- **THEN** the corresponding matrix index is returned
- **AND** unknown names produce an error

#### Scenario: Reverse lookup
- **WHEN** a matrix index is queried
- **THEN** the corresponding node name is returned
- **AND** voltage or current type is indicated

### Requirement: Matrix Updates for Switching
The MNA matrix SHALL support efficient updates when switches change state.

#### Scenario: Switch state change
- **WHEN** a switch changes from open to closed (or vice versa)
- **THEN** the matrix is updated to reflect the new topology
- **AND** minimal recomputation is performed

#### Scenario: Topology change detection
- **WHEN** an event causes a topology change
- **THEN** the system determines if refactorization is needed
- **AND** reuses factorization when possible
