## ADDED Requirements

### Requirement: MNA Convention

`pulsim::v2::stamping` SHALL adopt the following MNA convention,
documented in `stamping/mna_convention.hpp` and enforced by every
stamping function:

1. **State vector layout**: `[ v_0, v_1, …, v_{N-1}, i_branch_0,
   …, i_branch_{M-1} ]`. Node voltages first, then voltage-source
   branch currents.
2. **Ground**: `kGround = -1`. Not a row in J. Not an index in f.
   Stamping skips operations touching ground.
3. **Branch direction**: every branch has a "from" terminal and a
   "to" terminal. The branch current is conventionally
   `i = current flowing from from-terminal to to-terminal`.
4. **KCL sign**: at node N, the residual `f[N]` accumulates `+i`
   for each branch whose `from == N` and `-i` for each branch
   whose `to == N`. At convergence `f = 0` means KCL holds at
   every active node.
5. **Newton form**: callers solve `J · Δx = -f`. Layer 3
   provides J and f; Layer 5's Newton iteration consumes them.

The `BranchCoord` struct (`stamping/branch_coord.hpp`) carries the
minimal coordinate any 2-terminal stamper needs:
`{ Index from; Index to; Index branch_id; }`. Helpers
`read_node_voltage(x, node)` and `node_is_active(node)` handle the
ground sentinel transparently.

#### Scenario: read_node_voltage returns 0 for ground

- **GIVEN** a `Vector x = [5, 7, 3]` and `node = kGround`
- **WHEN** the user calls `read_node_voltage(x, node)`
- **THEN** the result SHALL be `0.0`.

#### Scenario: read_node_voltage returns x[i] for an active node

- **GIVEN** a `Vector x = [5, 7, 3]` and `node = 1`
- **WHEN** the user calls `read_node_voltage(x, node)`
- **THEN** the result SHALL equal `7.0`.

### Requirement: Generic 2-Terminal Device Stamper

`pulsim::v2::stamping::stamp_device<T>` SHALL be a templated free
function that stamps any `models::DeviceModel T` with
`T::num_terminals == 2` into a sparse Jacobian + residual.

Signature:
```cpp
template <models::DeviceModel T>
void stamp_device(sparse::Matrix& J, Vector& f, const Vector& x,
                  const BranchCoord& coord,
                  const typename T::Params& p) noexcept;
```

The function MUST:
1. Read terminal voltages: `v[0] = read_node_voltage(x, coord.from)`,
   `v[1] = read_node_voltage(x, coord.to)`.
2. Call `models::evaluate_current_and_jacobian<T>({v[0], v[1]}, p)`
   to obtain `(i, ∂i/∂v[k])`.
3. Stamp the residual contributions:
   `f[from] += i` (if from is active), `f[to] -= i` (if to is
   active).
4. Stamp the Jacobian contributions (4 entries):
   `J[from, from] += ∂i/∂v[0]`, `J[from, to] += ∂i/∂v[1]`,
   `J[to,   from] -= ∂i/∂v[0]`, `J[to,   to]   -= ∂i/∂v[1]`,
   gating each on whether its row and column are active nodes.
5. Reject 3-terminal+ device models at compile time via
   `static_assert(T::num_terminals == 2)`.

#### Scenario: Stamping a Resistor between active nodes

- **GIVEN** `Resistor::Params{ G = 2.0 }`, `coord = { from=1, to=2,
  branch_id=0 }`, and `Vector x = [0, 3, 1]`
- **WHEN** the user calls `stamp_device<Resistor>(J, f, x, coord, p)`
  on freshly-zeroed J and f
- **THEN** `f[1]` SHALL equal `+4.0` (= G·(x[1] - x[2]) = 2·2)
- **AND** `f[2]` SHALL equal `-4.0`
- **AND** `J(1,1)` SHALL equal `+2.0`, `J(1,2)` SHALL equal `-2.0`
- **AND** `J(2,1)` SHALL equal `-2.0`, `J(2,2)` SHALL equal `+2.0`.

#### Scenario: Stamping a Resistor to ground skips ground entries

- **GIVEN** `Resistor::Params{ G = 1.0 }`, `coord = { from=0, to=
  kGround, branch_id=0 }`, and `x = [5]`
- **WHEN** the user stamps
- **THEN** `f[0]` SHALL equal `+5.0` (current leaves node 0 toward
  ground)
- **AND** `J(0,0)` SHALL equal `+1.0`
- **AND** no other entries SHALL be touched (no row or column
  for ground).

#### Scenario: Stamping is additive across parallel devices

- **GIVEN** two Resistors with `G = 1.0` and `G = 2.0` both between
  nodes (0, 1)
- **WHEN** the user calls `stamp_device<Resistor>` twice with the
  same J/f
- **THEN** `J(0,0)` SHALL equal `+3.0` (sum of the two
  conductances).

#### Scenario: Stamping an IdealDiode wires AD partials correctly

- **GIVEN** an `IdealDiode` between nodes (0, 1) forward-biased
  with `x[0] = 1.0, x[1] = 0.0`
- **WHEN** the user calls `stamp_device<IdealDiode>`
- **THEN** the residual SHALL satisfy `f[0] + f[1] == 0` (KCL sanity
  to within numerical noise)
- **AND** the Jacobian entries `J(0,0)`, `J(0,1)`, `J(1,0)`, `J(1,1)`
  SHALL satisfy `J(0,0) + J(0,1) == 0` AND `J(1,0) + J(1,1) == 0`
  (the diode current depends only on `v_diode = v[0] - v[1]`).

### Requirement: Voltage-Source Constraint Stamper

`pulsim::v2::stamping::stamp_voltage_source` SHALL stamp a DC
voltage source as the standard MNA constraint + KCL contribution:

```cpp
void stamp_voltage_source(sparse::Matrix& J, Vector& f,
                          const Vector& x, const BranchCoord& coord,
                          Index branch_var_id, Real V) noexcept;
```

The function MUST:
1. KCL at the terminal nodes: add `+1` and `-1` to the columns
   indexed by `branch_var_id`, and the corresponding residual
   contributions `f[from] += x[branch_var_id]`, `f[to] -=
   x[branch_var_id]` (gated on active nodes).
2. Constraint row at index `branch_var_id`:
   `f[branch_var_id] = read_node_voltage(x, from) -
                       read_node_voltage(x, to) - V`,
   with Jacobian `J[branch_var_id, from] = +1` and
   `J[branch_var_id, to] = -1` (gated on active nodes).

At convergence (`x[from] - x[to] = V` and `i_branch =
device_load_current`), every residual entry the source touches
SHALL be zero.

#### Scenario: Source between two active nodes — residual + Jacobian

- **GIVEN** `coord = { from=0, to=1, branch_id=... }`, `V = 5.0`,
  `branch_var_id = 2`, and `x = [3.0, 0.0, 2.0]`
- **WHEN** the user calls `stamp_voltage_source` on freshly-zeroed
  J and f
- **THEN** `f[2]` SHALL equal `3 - 0 - 5 = -2.0` (constraint
  residual)
- **AND** `f[0]` SHALL equal `+2.0` (= x[branch_var_id])
- **AND** `f[1]` SHALL equal `-2.0`
- **AND** the Jacobian SHALL have `J(0,2) = +1`, `J(1,2) = -1`,
  `J(2,0) = +1`, `J(2,1) = -1`.

#### Scenario: Source between active node and ground

- **GIVEN** `coord = { from=0, to=kGround, ... }`, `V = 12.0`,
  `branch_var_id = 1`, and `x = [0.0, 0.0]`
- **WHEN** the user stamps
- **THEN** `f[1]` SHALL equal `0 - 0 - 12 = -12.0`
- **AND** `f[0]` SHALL equal `0.0` (= x[branch_var_id])
- **AND** `J(0,1) = +1` and `J(1,0) = +1` SHALL be the only
  non-ground entries (ground entries skipped).

### Requirement: Fixed-State Switch Stamper

`pulsim::v2::stamping::stamp_switch_fixed` SHALL stamp a switch
in a fixed (open or closed) state as a 2-terminal conductance.
Layer 4's PWL state-space cache calls this per Switch-kind branch
per segment, picking `closed` from the segment's `SwitchStateMask`.

```cpp
void stamp_switch_fixed(sparse::Matrix& J, Vector& f,
                        const Vector& x, const BranchCoord& coord,
                        bool closed, Real g_on, Real g_off) noexcept;
```

The function MUST stamp a 2x2 conductance block with `G = closed ?
g_on : g_off`, identical to a Resistor stamp with that
conductance. Ground handling matches `stamp_device`.

#### Scenario: Closed switch stamps g_on between its terminals

- **GIVEN** `coord = { from=0, to=1, ... }`, `closed = true`,
  `g_on = 1e3`, `g_off = 1e-9`, `x = [1.0, 0.0]`
- **WHEN** the user stamps on a freshly-zeroed J
- **THEN** `J(0,0)` SHALL equal `+1e3` and `J(0,1)` SHALL equal
  `-1e3`
- **AND** `J(1,0)` SHALL equal `-1e3` and `J(1,1)` SHALL equal
  `+1e3`
- **AND** `f[0]` SHALL equal `+1e3` (= G · (1.0 - 0.0))
- **AND** `f[1]` SHALL equal `-1e3`.

#### Scenario: Open switch stamps g_off (small but non-singular)

- **GIVEN** `coord = { from=0, to=1, ... }`, `closed = false`,
  `g_on = 1e3`, `g_off = 1e-9`, `x = [10.0, 0.0]`
- **WHEN** the user stamps
- **THEN** `J(0,0)` SHALL equal `+1e-9` (not zero — the matrix
  stays non-singular even with the switch open)
- **AND** `f[0]` SHALL equal `+1e-8` (= 1e-9 · 10).
