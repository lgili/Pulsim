## Why

Layer 2 (`pulsim-v2-device-models-ad-driven`, archived) landed the
AD-only device-model pattern: every device exposes ONE templated
`current<S>(...)` function and Layer 2's `evaluate_current_and_
jacobian` helper returns `(current, ∂current/∂v[k])` for any device
satisfying the `DeviceModel` concept.

Layer 3 is the next bridge: take the (current, partials) output and
stamp it into a `sparse::Matrix J` + `Vector f` ready for Newton
iteration. ONE generic stamper that works for every device type
the concept admits. No per-device-class hand-rolled stamps. No
repetition of the matrix-coordinate logic across 22 device variants
the way v1 has it.

This OpenSpec lands the **2-terminal generic stamper**, plus the
voltage-source constraint stamper and the fixed-state switch
stamper. Three free functions, all in `pulsim::v2::stamping`.
Together they cover every 2-terminal device in the v2 catalogue
(Resistor, VoltageSource, IdealDiode, future Capacitor, Inductor,
IdealSwitch). 3-terminal+ devices (MOSFET, IGBT, motors) extend
the same pattern in a follow-up OpenSpec — they need a multi-pin
coordinate type that 2-terminal Layer 3 doesn't.

## What Changes

**New directory `core/include/pulsim/v2/stamping/`** with four
headers:

```
pulsim/v2/stamping/
├── branch_coord.hpp        # BranchCoord: nodes + branch id
├── mna_convention.hpp      # Documents the sign/index convention
├── stamp_device.hpp        # Generic 2-terminal device stamper
├── stamp_voltage_source.hpp# Voltage-source constraint row
└── stamp_switch.hpp        # Fixed-state switch (for Layer 4 segments)
```

**MNA convention (locked in by Layer 3)**:
- State vector layout: `[v_node_0, ..., v_node_{N-1}, i_branch_0,
  ..., i_branch_{M-1}]`. Node voltages first, then voltage-source
  branch currents.
- Ground (`kGround = -1`): not present in the state vector. Stamping
  routines skip rows/cols touching ground.
- Sign convention: a branch's "current" flows from terminal 0 to
  terminal 1. KCL at node N sums `+current` for every branch whose
  terminal 0 is N, `-current` for every branch whose terminal 1 is
  N. Residual `f[N]` = sum of currents leaving N.
- Newton form: solve `J · Δx = -f`. At convergence `f == 0`.

**Three free functions**:

1. `stamp_device<DeviceModel T>(J, f, x, coord, p)` — generic
   2-terminal device stamper. Reads terminal voltages from `x`,
   calls `evaluate_current_and_jacobian<T>`, stamps:
   - Residual: `f[from] += i`, `f[to] -= i`.
   - Jacobian: `J[from, from] += di_dv0`, `J[from, to] += di_dv1`,
     `J[to, from] -= di_dv0`, `J[to, to] -= di_dv1`.
   - Skips rows/cols touching `kGround` (ground node not in state).

2. `stamp_voltage_source(J, f, x, coord, branch_var_id, V)` —
   constraint row + KCL contribution:
   - KCL at `from` gets `+i_branch`; at `to` gets `-i_branch`.
   - Constraint row at `branch_var_id`: `v[from] - v[to] = V`,
     residual `f[branch_var_id] = x[from] - x[to] - V`.
   - Jacobian: `J[from, branch_var_id] = +1`, `J[to,
     branch_var_id] = -1`, `J[branch_var_id, from] = +1`,
     `J[branch_var_id, to] = -1`.

3. `stamp_switch_fixed(J, f, x, coord, closed, g_on, g_off)` —
   2-terminal binary-conductance stamper used by Layer 4 to
   materialise a switch-state segment. Behaves like a Resistor
   with `G = closed ? g_on : g_off`.

Plus its own test binary `pulsim_v2_layer3_tests` with one test
file per stamping function plus an integration test that assembles
a V-R-GND circuit and verifies the assembled (J, f) match the
analytical solution.

## Impact

- **Affected specs**:
  - NEW capability `kernel-v2-stamping` (MNA convention + the three
    stamping functions).
- **Affected code** (this proposal — estimated 800-1200 LOC added,
  0 LOC modified):
  - NEW `core/include/pulsim/v2/stamping/` (4 headers).
  - NEW `core/tests/v2/layer3/` (5 test files + main).
  - NEW CMake test target `pulsim_v2_layer3_tests`.
  - NEW `docs/pulsim-v2/layer3-stamping-pipeline.md` design note.
- **Migration**: none. Layer 3 is pure new code in `pulsim::v2`.
  v1 is not touched.
- **Risk**: low. Pure additive change with isolated tests.
- **What this proposal explicitly does NOT do**:
  - No 3-terminal+ device stamping (MOSFET, IGBT, transformer,
    motors). A follow-up OpenSpec extends `BranchCoord` to
    multi-pin and adds the matching stamper. The 2-terminal stamper
    is conceptually identical and lands first as the reference.
  - No state-space cache (Layer 4 — the PLECS-killer — consumes
    Layer 3 to build per-segment matrices, but is its own OpenSpec).
  - No Newton solver loop (Layer 5).
  - No integrator (trapezoidal companion, history terms for
    capacitors / inductors are Layer 4's responsibility — Layer 3
    stamps the instantaneous device contribution at the current `x`).
  - No frontend / Python bindings (Layer 6).
