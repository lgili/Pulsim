# Layer 3 — Generic stamping pipeline

The pipeline that turns Layer 2 device-model output into a stamped
sparse Jacobian + residual ready for Newton iteration. ONE generic
stamper for every 2-terminal `DeviceModel`. The v1 four-stamp
duplication is structurally impossible above this layer.

## Public surface

```cpp
// pulsim/v2/stamping/branch_coord.hpp
namespace pulsim::v2::stamping {
    struct BranchCoord { Index from, to, branch_id; };
    Real read_node_voltage(const Vector& x, Index node);  // ground-aware
    constexpr bool node_is_active(Index node);             // node >= 0
}

// pulsim/v2/stamping/stamp_device.hpp
template <models::DeviceModel T>
void stamp_device(sparse::Matrix& J, Vector& f, const Vector& x,
                  const BranchCoord& coord, const T::Params& p);

// pulsim/v2/stamping/stamp_voltage_source.hpp
void stamp_voltage_source(sparse::Matrix& J, Vector& f, const Vector& x,
                          const BranchCoord& coord,
                          Index branch_var_id, Real V);

// pulsim/v2/stamping/stamp_switch.hpp
void stamp_switch_fixed(sparse::Matrix& J, Vector& f, const Vector& x,
                        const BranchCoord& coord,
                        bool closed, Real g_on, Real g_off);
```

That's it. ~250 LOC of headers covers every 2-terminal device.

## MNA convention (the lock-in)

```
x = [ v_0  v_1  ...  v_{N-1}   i_b0  i_b1  ...  i_b{M-1} ]
     |---- node voltages ----| |--- branch currents ---|
```

- Ground (`kGround = -1`) is NOT in the state vector. Stamping
  routines skip ground rows/cols.
- Branch convention: terminal 0 = "from", terminal 1 = "to".
  Current flows from 0 to 1.
- KCL at node N: `f[N]` accumulates `+i` from every branch whose
  from-terminal is N, `-i` from every branch whose to-terminal is N.
- Newton form: solve `J · Δx = -f`. At convergence `f = 0`.

## The v1-killer pattern

v1's stamping logic for MOSFET, for example:

| File                       | Function                       | LOC |
|----------------------------|--------------------------------|-----|
| `mosfet.hpp`               | `stamp_jacobian_behavioral`    | ~80 |
| `mosfet.hpp`               | `stamp_jacobian_via_ad`        | ~60 |
| `mosfet.hpp`               | `stamp_jacobian_ideal`         | ~70 |
| `runtime_circuit.hpp`      | `stamp_mosfet_jacobian`        | ~250 |

FOUR places implementing the same math, with subtle drift between
them that caused the May 2026 diode-fails-after-reverse-bias bug.

v2's `stamp_device<T>` IS the equivalent — ~30 LOC, templated, ONE
place:

```cpp
template <models::DeviceModel T>
void stamp_device(sparse::Matrix& J, Vector& f, const Vector& x,
                  const BranchCoord& coord, const T::Params& p) {
    static_assert(T::num_terminals == 2);
    const std::array<Real, 2> v = {
        read_node_voltage(x, coord.from),
        read_node_voltage(x, coord.to),
    };
    const auto [i, partials] = evaluate_current_and_jacobian<T>(v, p);
    // Stamp residual + 4 Jacobian entries, gated on each
    // node being active (not ground).
}
```

Adding a new 2-terminal device touches the device's header ONLY —
write `current<S>`, declare the static constants, satisfy the
concept. Layer 3 doesn't change.

## Voltage source: why a separate constraint row

In standard MNA, a voltage source is a constraint
`v_from - v_to = V`, not a current contribution. The augmented
system adds:
1. A new unknown (the branch current `i_branch`).
2. A new equation (the constraint).
3. The branch current participates in the terminal-node KCLs.

`stamp_voltage_source` implements this directly. Caller passes
`branch_var_id` = the absolute state-vector index of this source's
branch-current unknown (= `N + source_index`).

## Switch: deferred-to-Layer-4 conductance switching

A switch is just a Resistor with binary conductance. Layer 3
provides `stamp_switch_fixed(closed, g_on, g_off)`; Layer 4 picks
`closed` per segment from the `SwitchStateMask`. The function is
mathematically identical to a Resistor stamp with the chosen `G`.

## Worked example: V-R-GND

The integration test `test_integration_vrgnd.cpp` builds the
classic single-loop circuit:

```
V_dc ─[Vsrc]─ n0 ─[R]─ gnd
```

State vector: `[v_n0, i_branch]`. Size 2.

```cpp
// Stamp both branches
sparse::Matrix J(2, 2); Vector f = Vector::Zero(2); Vector x = ...;
stamp_voltage_source(J, f, x, {0, kGround, 0}, 1, V_dc);
stamp_device<Resistor>(J, f, x, {0, kGround, 1}, {G});

// Solve one Newton step (linear system → exact)
sparse::SparseLuSolver solver;
solver.analyze(J); solver.factorize(J);
Vector delta; Vector neg_f = -f;
solver.solve(neg_f, delta);
x += delta;

// x[0] = V_dc, x[1] = -V_dc · G  ✅
```

ONE call to each stamper, ONE call to the solver, EXACT solution.
The 4-place v1 stamping logic is gone.

## What this layer does NOT do

- No 3-terminal+ devices. `stamp_device<T>` static-asserts
  `T::num_terminals == 2`. The multi-pin extension is a follow-up
  OpenSpec that ships alongside the MOSFET / IGBT / motor models.
- No state-space cache. Layer 4 calls these stampers per segment
  to build pre-factorized matrices.
- No Newton iteration loop. Layer 5 calls the stampers every
  Newton iteration.
- No integrator / history terms. Layer 4 / 5 owns the trapezoidal
  companion for capacitors and inductors.

## Validation

`pulsim_v2_layer3_tests` covers:
- **branch_coord** (3 cases): ground sentinel handling.
- **stamp_device** (5 cases): Resistor 2x2 block, ground skip,
  additivity (parallel resistors), IdealDiode AD-derived
  Jacobian + KCL sanity, 4-entry sparsity.
- **stamp_voltage_source** (3 cases): two-node + constraint row,
  ground side, residual-at-convergence.
- **stamp_switch** (3 cases): closed, open, ground-touching.
- **integration** (2 cases): V-R-GND assembled end-to-end, solved
  via `SparseLuSolver`, verified against analytical solution to
  1e-12 with both `x = 0` start AND directly-planted convergent
  state.

Current: **61 assertions / 16 test cases, all green.**
Total v2 surface to date: **360 assertions / 107 test cases**.
