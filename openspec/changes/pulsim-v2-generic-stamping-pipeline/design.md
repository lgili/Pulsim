# Design — `pulsim-v2-generic-stamping-pipeline` (Layer 3)

## Goal

Take Layer 2's `(current, ∂current/∂v[k])` output and stamp it
into a sparse Jacobian + residual vector ready for Newton iteration.
ONE generic stamper per "category" (device, voltage-source,
switch), generic over the concrete device-model type. No
per-device-class hand-rolled stamping.

## MNA convention (locked in by Layer 3)

### State vector layout

```
x = [ v_0  v_1  ...  v_{N-1}  i_b0  i_b1  ...  i_b{M-1} ]
     |---- node voltages ----| |--- branch currents ---|
```

- N = `graph.num_nodes()` (the non-ground nodes; ground sits at
  the implicit index `kGround = -1` and is NOT in the state).
- M = number of branches with `kind == Source` (voltage sources
  need a branch-current unknown to close the MNA system).
- State vector total size = N + M.

### Sign convention

For every branch:
- Terminal 0 = "from" node
- Terminal 1 = "to" node
- "Current" = current flowing from terminal 0 to terminal 1
  (positive when v_from > v_to for a resistor)

KCL at node N: sum of (currents leaving N via each branch) = 0.
A branch contributes `+i` to `f[from]` and `-i` to `f[to]` (the
current that "leaves N" via that branch).

Newton form: solve `J · Δx = -f`. At convergence `f = 0` and the
KCL holds at every node.

### Ground handling

`kGround = -1`. NOT in the state vector. NOT a row in J. NOT an
index in f. Stamping helpers (`node_is_active`,
`read_node_voltage`) gate access — when stamping touches the
ground "row" or "column", the operation is silently skipped.

This avoids allocating a row for ground (which would be all zeros
by convention and just add ill-conditioning to the solve).

## Why three separate stampers, not one

Layer 3 ships three free functions:
1. `stamp_device<T>` — for `kind ∈ {PassiveLinear, Nonlinear}`
2. `stamp_voltage_source` — for `kind == Source`
3. `stamp_switch_fixed` — for `kind == Switch` (Layer 4 picks
   open/closed per segment)

They have fundamentally different structure:
- Generic device: 4 Jacobian entries + 2 residual rows.
- Voltage source: 4 Jacobian entries on the KCL rows + 2 entries
  on the constraint row + ALL of these involve a `branch_var_id`
  index that's not part of any device's terminal voltages.
- Switch: same shape as a Resistor BUT the conductance value
  depends on Layer 4's `SwitchStateMask` bit.

Trying to force them into one function would require knee-deep
`if constexpr` chains that obscure the math. Three small focused
functions read better, generate the same code (compiler inlines
everything), and let Layer 4 dispatch by `BranchKind` cleanly.

## Why the `stamp_device<T>` template is the v1-killer

v1's `runtime_circuit.hpp::stamp_mosfet_jacobian` (~250 LOC) does
exactly what `stamp_device<MOSFET>` does in ~30 LOC — but written
by hand, and duplicated from `MOSFET::stamp_jacobian_behavioral`
(another ~80 LOC), `MOSFET::stamp_jacobian_via_ad` (~60 LOC), and
`MOSFET::stamp_jacobian_ideal` (~70 LOC). FOUR places implementing
the same matrix-coordinate logic.

v2's `stamp_device<T>`:
```cpp
template <models::DeviceModel T>
void stamp_device(sparse::Matrix& J, Vector& f,
                  const Vector& x, const BranchCoord& coord,
                  const typename T::Params& p) noexcept {
    static_assert(T::num_terminals == 2);
    Real v0 = read_node_voltage(x, coord.from);
    Real v1 = read_node_voltage(x, coord.to);
    auto [i, J_partials] = models::evaluate_current_and_jacobian<T>(
        {v0, v1}, p);
    if (node_is_active(coord.from)) {
        f[coord.from] += i;
        J.coeffRef(coord.from, coord.from) += J_partials[0];
        if (node_is_active(coord.to))
            J.coeffRef(coord.from, coord.to) += J_partials[1];
    }
    if (node_is_active(coord.to)) {
        f[coord.to] -= i;
        J.coeffRef(coord.to, coord.to) -= J_partials[1];
        if (node_is_active(coord.from))
            J.coeffRef(coord.to, coord.from) -= J_partials[0];
    }
}
```

ONE function. Works for Resistor, IdealDiode, future Capacitor,
Inductor, IdealSwitch, every 2-terminal device the concept admits.
Adding a new 2-terminal device requires ZERO changes to Layer 3.

## Voltage source: why a separate constraint row

In standard MNA, a voltage source between nodes `(a, b)` imposes
`v_a - v_b = V`. This is a CONSTRAINT, not a current contribution.
The MNA way to handle it: add a new unknown `i_branch` (the
current through the voltage source) and a new equation
`v_a - v_b - V = 0`. The branch current then participates in the
KCL at nodes a and b.

The augmented system stays square (one new unknown, one new
equation). The Jacobian gets a constraint block:
```
[ ... +1 ... ]   ← KCL at a: contains +1 in the branch_var_id column
[ ... -1 ... ]   ← KCL at b: -1 in branch_var_id column
[ +1 -1 0 ... ] ← Constraint row: v_a - v_b = V
```

This is the textbook MNA approach. Layer 3 implements it directly.

## Switch handling: deferred to Layer 4

A switch is electrically just a resistor whose conductance flips
between two values based on a binary state. Layer 3 provides
`stamp_switch_fixed(closed, g_on, g_off)` that stamps the chosen
conductance — Layer 4 picks `closed` per segment from the
`SwitchStateMask`.

When Layer 4 materialises the matrices for a given switch state,
it iterates the graph's Switch-kind branches and calls
`stamp_switch_fixed(coord, mask.get(switch_idx), g_on, g_off)`.

## What this layer explicitly does NOT do

- **No 3-terminal+ devices.** Their stamper extends `BranchCoord`
  to a multi-pin coordinate (`std::span<const Index> nodes`) and
  parallels `stamp_device<T>`. Lands in its own OpenSpec along
  with the MOSFET/IGBT/transformer/motor device models.
- **No history-term stamping for caps/inductors.** Trapezoidal
  companion needs `dt` and the previous step's state — that's
  the integrator's responsibility (Layer 4 / 5). Layer 3 stamps
  the instantaneous device contribution at `x`.
- **No Newton iteration loop.** Layer 5 calls the stamping
  functions every iteration; Layer 3 just provides the
  primitives.
- **No matrix capacity / sparsity pre-analysis.** Layer 3 uses
  `J.coeffRef(...)` which inserts on first touch. Layer 4 will
  pre-analyze the sparsity pattern PER SEGMENT and call
  `J.makeCompressed()` before passing to the solver.

## Validation

`pulsim_v2_layer3_tests` covers:
- `BranchCoord` helpers (ground sentinel handling).
- `stamp_device<Resistor>` — analytical Jacobian.
- `stamp_device<IdealDiode>` — AD-derived Jacobian end-to-end.
- `stamp_device` ground-skipping.
- `stamp_device` parallel accumulation (two resistors between same
  nodes).
- `stamp_device` compile-time rejection of 3-terminal models.
- `stamp_voltage_source` — KCL + constraint row, ground handling,
  convergence-state residual check.
- `stamp_switch_fixed` — closed and open states.
- Integration: V-R-GND assembled end-to-end, solved via Layer 0's
  `SparseLuSolver`, verified against the analytical answer to
  1e-12.

Target: ≥ 25 assertions / ≥ 10 test cases. Layer 0/1/2 tests
stay green.
