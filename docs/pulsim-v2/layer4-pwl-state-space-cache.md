# Layer 4 — PWL state-space cache (the PLECS-killer)

The architectural pivot. For each switch combination Layer 1
enumerates, pre-build the per-segment MNA matrix (Layer 3 stampers
+ Layer 2 device models) and pre-factorise via Layer 0's
`DirectSolver`. Per-step hot loop: hash-map lookup + triangular
solve. NO assemble, NO factorize, NO Newton iteration per step.

**Expected speedup vs v1: 10-50× on PE workloads.**

## Why PWL caching beats Newton-per-step

For a circuit with N switches and a matrix of dimension n with
nnz non-zeros:

| Cost (per simulation step)   | v1 Newton-per-step | v2 PWL-cached |
|------------------------------|---------------------|---------------|
| Assemble Jacobian            | ~O(branches)        | O(1) lookup   |
| Sparse factorization (LU)    | ~O(n^1.5) typical   | NONE (pre-done) |
| Triangular solve             | ~O(nnz)             | ~O(nnz)       |
| Newton iterations (per step) | 5-30                | 1 (linear)    |
| **Total**                    | **~10-50× larger** | **baseline**  |

The cache pays its build cost ONCE upfront. Per-step cost is just
the triangular solve. For circuits running 100k+ simulation steps
(typical PE convergence to steady state) the amortisation is
overwhelming.

## How Layer 4 composes Layers 0-3

```
┌─────────────────────────────────────────────────────────┐
│  PwlStateSpaceCache.build():                            │
│                                                          │
│    for mask in enumerate_switch_states(N):  ← Layer 1   │
│                                                          │
│      assemble_segment(graph, pool, mask, J, b):         │
│        for branch in graph.branches():       ← Layer 1  │
│          dispatch by branch.kind:                       │
│            PassiveLinear → stamp_device<Resistor> ← L3  │
│            Source       → stamp_voltage_source    ← L3  │
│            Switch       → stamp_switch_fixed      ← L3  │
│                                                          │
│      (stamp_device internally calls                     │
│       evaluate_current_and_jacobian → AD)    ← Layer 2  │
│                                                          │
│      solver = make_default_solver()          ← Layer 0  │
│      solver.analyze(J)                       ← Layer 0  │
│      solver.factorize(J)                     ← Layer 0  │
│      segments_[mask] = {J, b, solver}                   │
│                                                          │
│  PwlStateSpaceCache.solve(mask, b_extra, x):            │
│    seg = segments_.at(mask)                             │
│    rhs = -(seg.b_constant + b_extra)                    │
│    seg.solver.solve(rhs, x)                  ← Layer 0  │
└─────────────────────────────────────────────────────────┘
```

Layer 0's three-phase `DirectSolver` lifecycle (`analyze /
factorize / solve`) is the bedrock — separation of "structural
factorisation" (per topology) from "triangular solve" (per step)
is what enables the cache.

## DevicePool — branch_id → params

Layer 1's `Graph` stores `BranchKind` per branch but NOT the
parameters. `DevicePool` bridges that gap with three V0 add methods
(Resistor, VoltageSource, Switch). Internal storage:
`unordered_map<Index, variant>`. Layer 4 V1 will add Capacitor,
Inductor; V2 will add 3-terminal devices.

State-vector layout helpers:
- `state_size(graph)` = `graph.num_nodes() + num_voltage_sources()`
- `branch_var_id_for_source(branch_id, graph)` returns absolute
  state-vector index for a Source-kind branch's branch-current
  unknown (= `num_nodes() + insertion_index`).

## PwlSegment — the cached per-state record

```cpp
struct PwlSegment {
    sparse::Matrix J;                                  // MNA matrix
    Vector b_constant;                                  // -V from sources
    std::unique_ptr<sparse::DirectSolver> solver;      // factorized
    Size state_size;
};
```

Move-only. The cache stores one segment per switch state in an
`unordered_map<SwitchStateMask, PwlSegment>`.

## V0 scope: static-only circuits

Layer 4 V0 supports Resistor + VoltageSource + Switch. That covers:
- Chopper circuits (V_dc + Switch + R)
- Resistive dividers
- Half-bridges driving R loads
- Any circuit that combines those primitives

NOT in V0:
- **Capacitors / Inductors** — need the trapezoidal companion
  `g_eq = 2C/dt` + history term `-i_hist` added to b. The MATRIX
  becomes dt-dependent; cache must invalidate when dt changes.
  V1 add.
- **Nonlinear devices** — per-segment Newton iteration on top of
  the cached factor. Layer 5 will use Sherman-Morrison or just
  refactor when needed. V1+ add.
- **Node-equivalence dimension reduction** — closed switches short
  their endpoints, so 3 shorted nodes could share 1 matrix row.
  V0 doesn't bother — it stamps `g_on` (large but finite). V1
  optional optimization.

## Worked example: chopper circuit

```
V_dc(12V) ─[Source b0]─ vin ─[Switch b1: 1e3/1e-9]─ vout ─[R b2: G=0.1]─ GND
```

```cpp
// Build the graph + device pool
Graph g;
Index vin  = g.add_node("vin");
Index vout = g.add_node("vout");
g.add_branch(vin, g.ground(), BranchKind::Source);
g.add_branch(vin, vout,       BranchKind::Switch);
g.add_branch(vout, g.ground(),BranchKind::PassiveLinear);

DevicePool pool;
pool.add_voltage_source(0, {12.0});
pool.add_switch(1, /*g_on=*/1e3, /*g_off=*/1e-9);
pool.add_resistor(2, {/*G=*/0.1});

// Build the cache once, run the hot loop forever.
PwlStateSpaceCache cache(g, pool);
cache.build();   // 2 segments (1 switch)

// Switch ON: vout ≈ V_dc · g_on / (g_on + G) = 11.9988 V
SwitchStateMask on(1); on.set(0, true);
Vector b_extra = Vector::Zero(3);
Vector x;
cache.solve(on, b_extra, x);
// x = [12, 11.9988, -1.2]   ← O(nnz) triangular solve

// Switch OFF: vout ≈ 0 (g_off + R divider)
SwitchStateMask off(1);
cache.solve(off, b_extra, x);
// x = [12, 1.2e-7, -1.2e-8]
```

Both states resolve to their analytical answers in O(nnz) — the
LU factor was computed ONCE per state at `build()` and reused for
every `solve()` call.

## What's deferred to follow-ups

- **Capacitor + Inductor** with trapezoidal companion + history
  terms (Layer 4 V1 OpenSpec).
- **Nonlinear devices** with per-segment Newton on top of the
  cached factor (Layer 4 V2 OpenSpec).
- **NodeEquivalence dimension reduction** for closed-switch
  shorts (Layer 4 optimisation OpenSpec).
- **Lazy build-on-first-lookup** for `N > 20` circuits where 2^N
  segments don't all fit in memory (Layer 4 scaling OpenSpec).
- **Sherman-Morrison rank-1 update** between Gray-code adjacent
  segments — would halve build time at large N.
- **Newton solver + integrator + event detector**: Layer 5.
- **Python bindings, YAML loader, schematic frontend**: Layer 6.

## Validation

`pulsim_v2_layer4_tests` covers:
- **DevicePool** (7 cases): add methods, state_size,
  branch_var_id_for_source, kind_of, wrong-kind throw, missing
  branch throw, switch param round-trip.
- **PwlSegment** (2 cases): move-only contract, default state.
- **assemble_segment** (6 cases): empty graph, resistor stamp,
  source constraint row, switch open/closed, full chopper
  assembly.
- **PwlStateSpaceCache** (5 cases): N=0/1/4 segment counts,
  missing-mask throw, V-R-GND solve correctness.
- **Integration chopper** (4 cases): cache build, ON-state
  divider exact, OFF-state ≈ 0, 10k-lookup performance smoke
  (< 1 s).

Current: **58 assertions / 24 test cases, all green.**
Total v2 surface: **418 assertions / 131 test cases**.
