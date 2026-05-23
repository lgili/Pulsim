## Why

Layer 0 (`bootstrap-pulsim-v2-kernel`, archived) gave the kernel its
numeric foundation. Layer 1 (`pulsim-v2-topology-and-switch-
enumeration`, archived) gave it the graph + switch combinatorics.
Layer 2 now needs to land the device-model layer — the math.

The v1 architectural review (`docs/architecture-review-v1.md`)
identified **quadruple-duplicated device stamps** as the single
biggest maintenance liability:

> MOSFET math lives in 4 places: `mosfet.hpp::stamp_jacobian_behavioral`,
> `mosfet.hpp::stamp_jacobian_via_ad`, `mosfet.hpp::stamp_jacobian_ideal`,
> AND `runtime_circuit.hpp::stamp_mosfet_jacobian`. Same pattern for
> IGBT, IdealDiode, switches. ~6 device families × 4 paths ≈ 24
> sites where the math lives. Every change has ~30 % chance of
> introducing drift between paths. The May 2026 diode-fails-after-
> reverse-bias bug came from exactly this drift.

Layer 2 fixes this PERMANENTLY by enforcing the **AD-only stamping**
pattern: each device exposes ONE templated function. The same
template instantiates for `double` (forward eval — tests, Layer 5
hot path) AND for the AD scalar (Jacobian extraction — Layer 3
stamping). Manual derivatives become structurally impossible:
the type system rejects them.

This OpenSpec lands:
1. The forward-mode AD scalar (`pulsim::v2::ad::ADReal<N>`).
2. The device-model concept (every model exposes `current<S>` and
   `kind`).
3. Three reference device models: `Resistor`, `VoltageSource`,
   `IdealDiode`. Enough to prove the pattern works on a linear
   passive, a source, and a nonlinear semi.

Future OpenSpecs add more device models (`Capacitor`, `Inductor`,
`MOSFET`, `IGBT`, motors, etc.) WITHOUT touching the AD machinery
or the concept. Each new device is one new header + tests; the
stamping pipeline (Layer 3) consumes the concept generically.

## What Changes

**New directory `core/include/pulsim/v2/`** with four sub-namespaces:

```
pulsim/v2/ad/
└── ad_scalar.hpp           # ADRealN<N> + math ops

pulsim/v2/models/
├── device_model.hpp        # The DeviceModel concept + helpers
├── resistor.hpp            # Resistor (linear, 2 terminals)
├── voltage_source.hpp      # Voltage source (source, 2 terminals)
└── ideal_diode.hpp         # Diode (nonlinear smooth-blend, 2 terminals)
```

**Strict layer discipline**:
- Layer 2 includes Layer 0 (`numeric/types.hpp`, `numeric/concepts.hpp`)
  and Layer 1 (`topology/graph.hpp` for `BranchKind`).
- Layer 2 does NOT include Eigen sparse, the solver, any matrix
  stamping — those are Layer 3+.

**The DeviceModel concept**:

```cpp
template <typename T>
concept DeviceModel = requires(const typename T::Params& p,
                                const std::array<Real, T::num_terminals>& v) {
    typename T::Params;
    { T::kind } -> std::convertible_to<topology::BranchKind>;
    { T::num_terminals } -> std::convertible_to<Size>;
    { T::is_linear } -> std::convertible_to<bool>;
    // The pivot: ONE current function, instantiable on any
    // FloatingPoint type S (so the same function works for
    // double AND for ADRealN<N>).
    { T::template current<Real>(v[0], v[0], p) } -> std::convertible_to<Real>;
};
```

Every device model satisfying the concept is automatically
consumable by Layer 3's generic stamper.

**ADRealN<N> forward-mode AD scalar**:

- Carries `Real value_` + `std::array<Real, N> derivs_`.
- Operator overloads: `+`, `-`, `*`, `/`, unary `-`, all comparisons
  with `Real` and other `ADRealN`.
- Math functions: `exp`, `log`, `sqrt`, `tanh`, `sinh`, `cosh`, `abs`.
- Seed pattern: `auto [v0, v1] = ad::seed<2>(v_anode, v_cathode)`
  creates two seeded inputs where each has a 1.0 in its slot and
  0.0 elsewhere.
- Stack-allocated, no heap. `sizeof(ADRealN<8>)` = 72 bytes (1
  Real value + 8 Real derivs). Cheap to construct, pass by value.

**Reference device models** (each ~50 LOC):

- `Resistor` — `i = G · (v_pos - v_neg)`. Linear. N=2. Tests forward
  eval + AD partial derivatives equal `G` analytically.
- `VoltageSource` — pure source contribution; current is the
  branch-current unknown. N=2. Tests the source-kind branch
  pattern.
- `IdealDiode` — `i = α·(v - V_F0)/R_d + (1-α)·v·G_off` with
  `α = sigmoid(κ·(v - V_F0))`. Nonlinear. Tests that AD derivatives
  match a finite-difference baseline within `1e-6`.

Plus its own test binary `pulsim_v2_layer2_tests` with five test
files (one per header).

## Impact

- **Affected specs**:
  - NEW capability `kernel-v2-ad-scalar` (forward-mode AD).
  - NEW capability `kernel-v2-device-models` (model concept + first
    three devices).
- **Affected code** (this proposal — estimated 1000-1500 LOC added,
  0 LOC modified):
  - NEW `core/include/pulsim/v2/ad/` (1 header, ad_scalar.hpp).
  - NEW `core/include/pulsim/v2/models/` (4 headers).
  - NEW `core/tests/v2/layer2/` (5 test files + main).
  - NEW CMake test target `pulsim_v2_layer2_tests`.
  - NEW `docs/pulsim-v2/layer2-ad-and-device-models.md` design note.
- **Migration**: none. Layer 2 is pure new code in `pulsim::v2`.
  Nothing in v1 is touched.
- **Risk**: low. Pure additive change.
- **What this proposal explicitly does NOT do**:
  - No matrix stamping. Layer 3 takes Layer 2's AD-derived
    Jacobian and stamps it into a sparse matrix.
  - No state-space cache. Layer 4 consumes both Layer 2 and Layer
    3 to build the cache.
  - No MOSFET / IGBT / Capacitor / Inductor / motors. Each is a
    one-header follow-up that drops into the model concept.
  - No reverse-mode AD. Layer 2 forward-mode covers Jacobian
    extraction for devices with ≤ ~8 terminals; reverse mode is
    a future optimisation when ≫ 8.
  - No symbolic / equation-DSL layer. That's a v3 conversation if
    Layer 2 hits its limits.
