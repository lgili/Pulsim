# Design — `pulsim-v2-pwl-state-space-cache` (Layer 4)

## The performance pivot

PLECS dominates power-electronics simulation because of ONE
architectural choice: it pre-factorises the MNA matrix per switch
combination and reuses the factor across many simulation steps.

For a circuit with N switches:
- v1 per step: `assemble_jacobian()` (~ms) + `solver.factorize()`
  (~ms) + Newton iterations (5-30 ×). All of that, every single
  step.
- PLECS-style per step: `cache.lookup(mask)` (~ns) +
  `solver.solve()` (~µs). That's it.

The asymptotic speedup is the ratio of "factorize cost" to
"triangular solve cost", which is the matrix's condition-number ×
its sparsity pattern. For sparse PE circuits, it's 10-50×.

Layer 4 V0 lands the cache architecture for **static-only**
circuits (Resistor + VoltageSource + Switch). V1 follow-ups add
caps/inductors (with the trapezoidal companion) and nonlinear
devices (with per-segment Newton on top of the cached factor).

## How the four lower layers compose

```
┌─────────────────────────────────────────────────────────┐
│  Layer 4: PwlStateSpaceCache                            │
│                                                          │
│   build():                                               │
│     for mask in Layer 1's enumerate_switch_states(N):   │
│       (J, b) = assemble_segment(graph, pool, mask)      │
│                  ↓                                       │
│       Layer 3 stamps each branch into J via dispatch    │
│       (stamp_device<Resistor> / stamp_voltage_source /   │
│        stamp_switch_fixed)                              │
│                  ↓                                       │
│       Layer 2's evaluate_current_and_jacobian (called   │
│       by stamp_device) returns (i, ∂i/∂v) for each      │
│       device                                             │
│                  ↓                                       │
│       Layer 0's DirectSolver.analyze + factorize once   │
│                  ↓                                       │
│       store {J, b, solver} keyed by mask                │
│                                                          │
│   solve(mask, b_extra, x):                              │
│     auto& seg = segments_[mask];                        │
│     seg.solver->solve(seg.b_constant + b_extra, x);     │
└─────────────────────────────────────────────────────────┘
```

Each layer below contributes exactly the abstraction Layer 4 needs.
No layer is over- or under-built relative to this consumer.

## V0 scope: static-only circuits

The big design decision: **start with resistor / voltage source /
switch only**. This:

1. Keeps V0 buildable in ~1-2 days of focused work.
2. Proves the cache architecture works end-to-end.
3. Exercises every layer below (Layer 0 solver, Layer 1
   enumeration, Layer 2 device models, Layer 3 stamping).
4. Provides the API Layer 5 will eventually consume — the hot-loop
   `solve(mask, b_extra, x)` signature is the same for V0 and V1+.

What V0 explicitly excludes:
- **Capacitors / inductors**: need the trapezoidal companion
  `g_eq = 2C/dt`, history term `-i_hist` added to b. The MATRIX
  becomes dt-dependent; need to rebuild the cache when dt changes
  (Layer 5 adaptive integrator's invalidation hook).
- **Nonlinear devices**: per-segment Newton iteration on top of
  the cached factor. The cache stores the LINEAR part; Layer 5
  iterates the residual contributions from `kind == Nonlinear`
  branches.
- **Node-equivalence dimension reduction**: closed switches short
  their endpoints, so 3 shorted nodes could share 1 matrix row.
  V0 doesn't bother — it stamps `g_on` (large but finite) and
  lets the LU factor handle the large-but-not-infinite
  conductance. Saves complexity for V1.

## DevicePool — heterogeneous registry

The `Graph` from Layer 1 stores `BranchKind` per branch but NOT
the parameters (resistor's `G`, source's `V`, switch's
`g_on/g_off`). Those live in `DevicePool`.

```cpp
class DevicePool {
public:
    void add_resistor(Index branch_id, Resistor::Params);
    void add_voltage_source(Index branch_id, VoltageSource::Params);
    void add_switch(Index branch_id, Real g_on, Real g_off);

    enum class StoredKind { Resistor, VoltageSource, Switch };
    StoredKind kind_of(Index branch_id) const;

    // Per-kind lookups (throw if branch_id has the wrong kind).
    const Resistor::Params&      resistor_params(Index)      const;
    const VoltageSource::Params& voltage_source_params(Index) const;
    Real                          switch_g_on(Index)          const;
    Real                          switch_g_off(Index)         const;

    // State-vector layout helpers
    Index branch_var_id_for_source(Index branch_id) const;
    Size  num_voltage_sources() const;
    Size  state_size(const Graph& graph) const;  // = N + M
};
```

Internal storage: `std::unordered_map<Index, StoredEntry>` where
`StoredEntry` is a discriminated union (kind + the relevant
params). V0 supports three kinds; V1 will add more without
changing the API.

Future device types (Capacitor, Inductor, MOSFET, ...) add a new
`add_*` method and a new `StoredKind` variant. Layer 3 grows
matching stamper dispatch. Layer 4's `assemble_segment` grows the
dispatch arm.

## PwlSegment — the cached per-state record

```cpp
struct PwlSegment {
    sparse::Matrix J;                                  // MNA matrix
    Vector b_constant;                                  // -V from sources
    std::unique_ptr<sparse::DirectSolver> solver;      // factorized
    Size state_size = 0;
};
```

Move-only (the `unique_ptr` makes copy nonsensical). The cache
owns a `std::unordered_map<SwitchStateMask, PwlSegment>` mapping
state → factorized record.

## Why constant matrix dimension across segments

V0 doesn't use Layer 1's `NodeEquivalence` to reduce matrix
dimension when switches close. Instead, closed switches are
stamped as `g_on` (typically `1e3`, large but finite). Two
consequences:

1. **All segments have the same matrix dimension** — easier to
   compose with Layer 5 (no state-vector translation per segment).
2. **Slightly larger matrices than the theoretical minimum** —
   negligible for sparse-LU which is dominated by nnz, not n.

V1's optimization (reduce dimension via NodeEquivalence) lands as
an opt-in `cache.build_with_node_merging()` after V0 proves the
architecture. Keep V0 simple.

## Hot-loop API: `solve(mask, b_extra, x)`

```cpp
void PwlStateSpaceCache::solve(const SwitchStateMask& mask,
                                const Vector& b_extra,
                                Vector& x) const {
    const auto& seg = segments_.at(mask);
    Vector b = seg.b_constant + b_extra;
    seg.solver->solve(b, x);
}
```

- `b_extra` is for time-varying source modulation in V1 (PWM,
  sine, pulse). In V0 it's typically zero.
- `solve` doesn't compute J. The factor is cached.
- O(nnz(L) + nnz(U)) per call — typically a few µs for circuits
  with ≤ 100 nodes.

That's the entire hot loop. NO `analyze`, NO `factorize`, NO
Newton iteration. Just lookup + triangular solve. **The
architectural reason v2 will compete with PLECS.**

## Risks

1. **Memory for large N**: 2^N segments × matrix + factor. At
   N=20 there are 1M segments. Each ≈ 1-10 KB → 1-10 GB. We
   document the limit and let V1 add lazy materialisation (only
   build the segments actually visited during a simulation).
   V0 is fine for N ≤ 16 (≤ 65k segments, ≤ 100 MB).

2. **Branch-iteration order vs SwitchStateMask bit order**:
   Layer 1's `SwitchStateMask` uses bit `i` for the `i`-th
   `Switch`-kind branch in branch-iteration order. Layer 4's
   `assemble_segment` must use the SAME counter. This is locked
   in by the test `test_assemble` covering ordering edge cases.

3. **Switch + ground sentinel interactions**: a switch with one
   terminal at ground stamps only the active row, but
   `stamp_switch_fixed` already handles this (Layer 3's `node_is_active`
   gate). Tested in Layer 3 — Layer 4 inherits.

## What V0 hands to Layer 5

```cpp
// Layer 5 hot loop (sketch)
void simulator_step(Real t, Real dt) {
    // 1. Update switch state (PWM, event detection, etc.)
    update_switch_state(switch_state_, t);
    // 2. Build dynamic RHS for time-varying sources, history terms
    Vector b_extra = compute_b_extra(t, dt);
    // 3. Cached solve
    cache_.solve(switch_state_, b_extra, x_);
    // 4. Done. NO Newton, NO factorize, NO assemble.
}
```

For static circuits (V0 scope) `b_extra` is constant and the
above becomes just step 3 — pure cached solve. Already
competitive with PLECS on this restricted scope.

## Validation

`pulsim_v2_layer4_tests` covers:
- **DevicePool** (6 cases): add methods, state_size,
  branch_var_id_for_source, kind_of, wrong-kind lookup throws.
- **Segment** (2 cases): move-only contract.
- **assemble_segment** (6 cases): empty graph, single resistor,
  single source, switch open/closed, chopper full assembly.
- **PwlStateSpaceCache** (5 cases): N=0/1/4 segment counts,
  missing-mask throws, solve correctness on V-R-GND.
- **Integration: chopper** (4 cases): chopper assembly + cache
  build, ON state ≈ V_dc, OFF state ≈ 0, 10k-lookup performance
  smoke.

Target: ≥ 35 assertions / ≥ 15 test cases. Layers 0/1/2/3 stay
green.
