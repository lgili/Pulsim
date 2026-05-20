# Pulsim v1 Architecture Review (May 2026)

This document distills the high-level architectural review that motivated
the `pulsim::v2` clean-slate rebuild. It diagnoses seven structural
problems in v1, identifies the four winning patterns that v2 must adopt,
explains the language-choice analysis, and lays out the phased path
to feature parity.

## TL;DR

- v1 is **incrementally unfixable**: seven structural problems compound
  on each other. Each one was introduced as a reasonable trade-off; the
  combination has crossed the threshold where in-place refactor costs
  exceed clean-slate rebuild costs.
- The single biggest performance gap vs PSIM/PLECS is **lack of PWL
  state-space topology caching**. This alone accounts for the 10-50×
  speed gap on power-electronics workloads.
- **Language stays C++23.** Julia + ModelingToolkit was a viable
  alternative but the team's C++ investment and Eigen/KLU ecosystem
  outweigh the equation-DSL benefits at this stage.
- v2 builds bottom-up in parallel with v1. v1 stays production until
  v2 reaches feature parity at Layer 6 (~12 weeks single-developer
  pace). No flag-day cutover; users opt in via `use_v2=True`.

## The seven structural problems in v1

### 1. Quadruple-duplicated device stamps

Concrete example: the MOSFET model lives in 4 places, each with subtle
sign-convention drift:

| File                       | Function                       | Convention            |
|----------------------------|--------------------------------|-----------------------|
| `mosfet.hpp`               | `stamp_jacobian_behavioral`    | Norton-offset         |
| `mosfet.hpp`               | `stamp_jacobian_via_ad`        | Norton-offset         |
| `mosfet.hpp`               | `stamp_jacobian_ideal`         | Mixed (caused diode bug) |
| `runtime_circuit.hpp`      | `stamp_mosfet_jacobian`        | Physical current      |

Same math, 4 places. Same pattern for IGBT, IdealDiode, switches: ~6
device families × 4 paths ≈ 24 sites where the math lives. Every change
has ~30 % chance of introducing drift. The May 2026 diode-fails-after-
reverse-bias bug came from exactly this drift.

### 2. `std::variant<22 devices>` AoS storage

```cpp
using DeviceVariant = std::variant<
    Resistor, Capacitor, Inductor, VoltageSource, ..., 22 types total>;
```

Every element occupies the size of the largest member (~512 B). A 16 B
Resistor wastes 32× cache. Every kernel operation is an `std::visit` →
22 lambda instantiations → switch + indirect call.

Bus errors in `analyze_circuit_robustness` came from variant alignment
operations. SIMD vectorization of per-device-type stamping is impossible.

### 3. **No PWL state-space topology caching (the architectural reason
PLECS dominates)**

For N switches, there are up to `2^N` topology combinations. PLECS
computes `(A, B, C, D)` and the pre-factorized Tustin LU for each
combination ONCE at initialization. Per simulation step:
1. Detect which combination is active.
2. Table lookup of the cached factorization.
3. Triangular solve.

Cost per step: ~`O(N)` triangular substitution. Pulsim v1 runs full
Newton-Raphson on the global system every step: 5-30 factorizations +
5-30 triangular solves until convergence — even on circuits that are
piecewise-linear by construction.

**Expected v2 speedup with PWL caching: 10-50× on PE workloads.**
This is THE architectural pattern that makes PLECS dominant.

### 4. Mixed-order integrators

| State                       | Integrator        |
|-----------------------------|-------------------|
| MNA caps / inductors (linear) | Trapezoidal companion |
| MOSFET smooth-blend region   | Implicit Newton   |
| Motor `ψ_r`, `ω`             | Forward Euler (post-A5 mostly Heun) |
| PSC `V_cap`                  | Forward Euler (post-A5 trapezoidal) |
| Mechanical `θ`               | Semi-implicit     |

Trapezoidal + forward Euler do not compose to order 2. The coupled
system is effectively order 1. PLECS uses Tustin (trapezoidal)
uniformly across all states.

### 5. No equation / DSL layer

Adding a new device today requires editing 6 files:
1. `core/include/pulsim/v1/components/<dev>.hpp` (math + 4 stamps)
2. `runtime_circuit.hpp` (variant entry, dispatcher, hand-rolled stamp,
   accessors)
3. `device_traits<X>` specialization
4. `yaml_parser.cpp` (parser branch)
5. `python/bindings.cpp` (Python wrapper)
6. Tests

Modelica describes the same device in **ONE `.mo` file** and generates
everything else. v2 plans an equation-style description layer that
captures this benefit.

### 6. Single-threaded everywhere

Newton solver is serial. Sparse factorize is serial. No subcircuit
partitioning, no parallel sparse solve, no parallel stamping. PLECS
uses Intel MKL multi-threaded direct solver. For a 3φ inverter the
three legs are nearly-independent during most of the PWM period, but
v1 treats them as one coupled 30+ equation system.

### 7. Convergence aids piled on as default-on options

```cpp
struct FallbackPolicyOptions { ... };
struct ModelRegularizationOptions { ... };
struct StiffnessConfig { ... };
struct DCConvergenceConfig { gmin, source, pseudo, init };
```

Each was added for a specific bug. **PLECS does NOT have any of these**
— PWL state-space is linear per segment, so Newton always converges
on the first try when needed. Pulsim's convergence aids are a
consequence of running Newton where state-space would converge
without iteration.

## The four patterns v2 MUST adopt

### 4.1 PWL state-space topology caching (Layer 4)

The PLECS-killer. Pre-enumerate switch combinations, pre-factorize
each, per-step lookup + triangular solve. See `docs/pulsim-v2/` for
the Layer 4 design.

### 4.2 Strict layer separation

Seven layers, each in one folder, each with own test binary. Layer N
depends only on layers < N. Cross-layer dependency is a compile error.

### 4.3 Automatic differentiation as the ONLY stamping path

Device math is written ONCE as a templated function:
```cpp
template <numeric::FloatingPoint S>
S drain_current(S vg, S vd, S vs, const MosfetParams& p) noexcept;
```

The same template instantiates for `double` (forward evaluation),
`ADReal` (Jacobian via AD), and AD-with-Hessian (sensitivity analysis).
The 4-place stamp duplication disappears.

### 4.4 SoA storage with per-device-type vectorization

```cpp
struct DevicePool {
    std::vector<Mosfet> mosfets;
    std::vector<IGBT> igbts;
    std::vector<Diode> diodes;
    ...
};
```

Stamping is a loop over each pool — compiler can auto-vectorize. Cache
density restored. `std::visit` overhead gone.

## Why C++23 and not Julia / Rust / Modelica

### C++23 wins (and was chosen)

- Existing investment: ~30 000 LOC v1 code, contributor familiarity
- Ecosystem: Eigen, KLU, UMFPACK, MKL — best-in-class sparse LA
  available natively
- AD: opt-in via templates; no runtime overhead
- C++20/23 concepts + ranges make the layered architecture
  expressible cleanly
- Cutover is the smallest possible (v1 and v2 in the same repo)

### Julia + ModelingToolkit was a strong contender

- Equation-DSL solves problem 5 elegantly (devices as `.jl` files)
- DifferentialEquations.jl is the gold-standard ODE/DAE solver suite
- Speed competitive with C++ when written well (`PowerSimulations.jl`,
  `ElectricGrid.jl` already exist)
- Lost on: smaller PE talent pool, "time-to-first-plot" JIT latency,
  weaker GUI/deployment story, ecosystem maturity vs C++ + Eigen.

### Rust was a non-starter

- Strong language but power-electronics ecosystem is essentially
  zero — `faer-rs` exists for sparse LA but you'd write everything
  else from scratch
- 6-month learning curve for a C++ team buys you what exactly?

### Modelica + OpenModelica embedded

- Solves problem 5 (equation-DSL) elegantly
- Performance varies; OpenModelica (free) slower than Dymola (paid)
- Locks v2 to the Modelica ecosystem permanently
- Out of scope for v2 — could be a v3 conversation if v2 hits its
  limits

## The phased plan

| Layer | Subfolder                  | Scope                                | Effort  |
|-------|----------------------------|--------------------------------------|---------|
| 0     | `numeric/` + `sparse/`     | Types, vector, sparse LA, solver     | ~3 days |
| 1     | `topology/`                | Graph, switch combinatorics          | ~1 week |
| 2     | `models/`                  | AD-only device models                | ~2 weeks |
| 3     | `stamping/`                | Generic stamping pipeline            | ~1 week |
| 4     | `pwl_state_space/`         | **The PLECS-killer cache**           | ~3-4 weeks |
| 5     | `solver/`                  | Per-step dispatch + Newton fallback  | ~2 weeks |
| 6a    | `runtime/`                 | Circuit builder API                  | ~1 week |
| 6b    | `frontend/`                | Python bindings + YAML loader        | ~2 weeks |

Total: ~12-13 weeks single-developer pace. Each layer is independently
mergeable; v1 stays production throughout.

Cutover criteria: Layer 6b ships with feature parity on all `[v1]`
tests; Python users opt in via `pulsim.use_v2 = True`; once ≥ 90 % of
real workloads run on v2, v1 enters maintenance, then archive.
