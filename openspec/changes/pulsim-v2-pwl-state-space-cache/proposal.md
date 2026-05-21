## Why

The May 2026 architecture review identified **PWL state-space topology
caching** as the SINGLE biggest performance gap between v1 and PLECS:

> For N switches there are up to 2^N topology combinations. PLECS
> computes `(A, B, C, D)` and the pre-factorized Tustin LU for each
> combination ONCE at initialization. Per-step: lookup + triangular
> solve. Pulsim v1 runs full Newton-Raphson on the global system
> EVERY step (5-30 factorizations per step) on circuits that are
> topologically piecewise-linear by construction. Expected v2 speedup
> with PWL caching: **10-50× on PE workloads**. This is THE
> architectural reason PLECS dominates.

Layer 4 lands the cache. Layers 0-3 gave us the bricks:
- Layer 0: `DirectSolver` with the `analyze / factorize / solve`
  separation. **Pre-factorize once, solve many.** That's the
  entire performance pivot.
- Layer 1: `enumerate_switch_states` + `TopologyKey` for the
  cache map key.
- Layer 2: `DeviceModel` concept so the cache consumes any device
  uniformly.
- Layer 3: generic `stamp_device<T>` / `stamp_voltage_source` /
  `stamp_switch_fixed` to build the per-segment matrix.

Layer 4 composes them: enumerate the switch states from Layer 1,
build the per-state matrix using Layer 3 stampers + Layer 2 device
models, factorize via Layer 0, store the result keyed by Layer 1's
`TopologyKey`. Per-step lookup is a hash-map probe plus a
triangular solve.

## What Changes

**Scope decision — Layer 4 V0 = static-only circuits**:

Layer 4 V0 supports `Resistor`, `VoltageSource`, `IdealSwitch`.
No capacitors / inductors (need trapezoidal companion + history),
no nonlinear devices (need Newton iteration on the segment). Those
are V1 / V2 extensions.

This gives us a working PWL cache for the canonical "chopper"
circuit class (V_dc → switch → R), the resistive divider class,
and any circuit that combines those primitives. It proves the cache
architecture and provides the hot-loop API Layer 5 will consume.

**New directory `core/include/pulsim/v2/pwl/`** with four headers:

```
pulsim/v2/pwl/
├── device_pool.hpp         # branch_id → (kind, params) registry
├── segment.hpp             # PwlSegment: matrix + RHS + factorized solver
├── assemble.hpp            # assemble_segment(graph, pool, mask) → matrix + b
└── cache.hpp               # PwlStateSpaceCache: build, lookup, hot loop
```

**`DevicePool`** — Heterogeneous registry mapping `branch_id` to the
(kind, params) tuple. Layer 1's `Graph` knows only `BranchKind`;
the pool adds the params. Three add methods (`add_resistor`,
`add_voltage_source`, `add_switch`); per-branch lookup returns the
right params. State-vector layout: each `Source`-kind branch gets a
branch-current unknown after the node voltages.

**`PwlSegment`** — Per-switch-state record. Holds:
- The assembled `sparse::Matrix J` (kept for diagnostics + reuse).
- The constant RHS `Vector b_constant` (voltage-source `-V` terms).
- The pre-factorized `unique_ptr<DirectSolver>` ready for `solve`.

**`assemble_segment`** — Builds J + b for a single
`SwitchStateMask`. Loops every branch in the graph, dispatches by
its `BranchKind` to the right Layer 3 stamper:
- `PassiveLinear` → `stamp_device<Resistor>` (V0 only handles
  Resistor; future Capacitor/Inductor extend the dispatch table)
- `Source` → `stamp_voltage_source`
- `Switch` → `stamp_switch_fixed` with `closed = mask.get(switch_idx)`
- `Nonlinear` → SKIPPED in V0 (matches the static-only scope)

**`PwlStateSpaceCache`** — Main class. Constructor takes a `Graph`
+ `DevicePool`. `build()` enumerates switch states via Layer 1's
Gray-code iterator, calls `assemble_segment` for each, factorizes
via Layer 0's `make_default_solver()`, stores keyed by
`SwitchStateMask`. `lookup(mask)` returns the cached segment in
`O(1)`. `solve(mask, b_extra, x)` is the hot-loop entry point:
combines `b_constant` with optional dynamic RHS contribution and
returns the solution via cached factorized solver.

**Per-step performance**: lookup = one `std::unordered_map` probe
(O(1)); solve = one triangular substitution from the cached factor
(O(nnz)). NO `analyze`, NO `factorize`, NO Newton iteration per
step — that's all amortised across the build phase.

Plus its own test binary `pulsim_v2_layer4_tests` with one file
per header plus an integration test on a chopper circuit
(V_dc + Switch + R + GND) that verifies all 2 switch states
produce the analytical node-voltage in `O(1)` lookup time.

## Impact

- **Affected specs**:
  - NEW capability `kernel-v2-pwl-cache` (DevicePool + Segment +
    assemble + cache).
- **Affected code** (this proposal — estimated 1000-1400 LOC added,
  0 LOC modified):
  - NEW `core/include/pulsim/v2/pwl/` (4 headers).
  - NEW `core/tests/v2/layer4/` (5 test files + main).
  - NEW CMake test target `pulsim_v2_layer4_tests`.
  - NEW `docs/pulsim-v2/layer4-pwl-state-space-cache.md` design
    note explaining the PLECS-style architecture.
- **Migration**: none. Pure new code in `pulsim::v2`. v1 untouched.
- **Risk**: medium. The cache architecture is the most complex
  piece of v2 so far — it spans every lower layer and is the
  blueprint for everything Layer 5+ will do. The risk is mitigated
  by limiting V0 scope to static-only circuits (no caps/inductors,
  no nonlinear devices), giving us a clean working baseline
  before extending.
- **What this proposal explicitly does NOT do**:
  - No capacitors / inductors. They need the trapezoidal
    companion (dt-dependent matrix entries + history terms). V1
    follow-up.
  - No nonlinear devices in the cache. `Nonlinear`-kind branches
    are SKIPPED in V0 assembly. They'll re-enter as a Newton
    iteration on top of the cached factor in a V1 follow-up.
  - No 3-terminal devices. Inherits Layer 3's 2-terminal-only
    scope.
  - No node-equivalence dimension reduction. Closed switches are
    stamped as `g_on` (large but finite); the matrix dimension
    stays constant across segments. Cleaner V0, lets V1 add the
    `NodeEquivalence`-based row merging as an optimisation.
  - No Layer 5 (Newton solver, integrator, event detection). The
    cache only exposes `solve(mask, b_extra, x)`; Layer 5 calls it
    per timestep.
  - No frontend, no Python bindings (Layer 6).
