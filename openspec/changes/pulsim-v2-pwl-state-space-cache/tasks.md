## Phase 1 — DevicePool (~0.5 days)

### 1.1 `pwl/device_pool.hpp`
- [x] 1.1.1 `class DevicePool` with three add methods:
      - `void add_resistor(Index branch_id, Resistor::Params)`
      - `void add_voltage_source(Index branch_id,
        VoltageSource::Params)`
      - `void add_switch(Index branch_id, Real g_on, Real g_off)`
- [x] 1.1.2 Internal storage: `std::unordered_map<Index,
      StoredEntry>` where `StoredEntry` is a small variant carrying
      the kind + params for that branch.
- [x] 1.1.3 Enum `StoredKind { Resistor, VoltageSource, Switch }`
      + per-kind getters: `kind_of(Index)`, `resistor_params(Index)`,
      `voltage_source_params(Index)`, `switch_params(Index)`.
- [x] 1.1.4 `branch_var_id_for_source(Index branch_id)` — returns
      the absolute state-vector index of the branch-current
      unknown for a Source-kind branch. Computed at insertion
      time: `num_nodes + (count of sources added so far)`.
- [x] 1.1.5 `state_size(const Graph& graph)` — total state-vector
      size for the cache's matrices: `graph.num_nodes() +
      num_voltage_sources()`.

### 1.2 Tests `tests/v2/layer4/test_device_pool.cpp`
- [x] 1.2.1 Empty pool: `state_size(empty_graph) == 0`.
- [x] 1.2.2 Add 1 resistor → `state_size == graph.num_nodes()` (no
      branch unknowns added).
- [x] 1.2.3 Add 2 voltage sources + 1 resistor →
      `state_size == graph.num_nodes() + 2`.
- [x] 1.2.4 `branch_var_id_for_source` returns sequential indices
      starting at `graph.num_nodes()`.
- [x] 1.2.5 `kind_of(branch_id)` returns the right enum for each
      add type.
- [x] 1.2.6 Looking up parameters for the wrong branch_id throws
      `std::out_of_range` (the lookup is unambiguous; mismatched
      kind queries are programmer errors).

## Phase 2 — PwlSegment (~0.25 days)

### 2.1 `pwl/segment.hpp`
- [x] 2.1.1 `struct PwlSegment` holding:
      - `sparse::Matrix J` — the assembled MNA matrix
      - `Vector b_constant` — voltage-source `-V` contributions
      - `std::unique_ptr<sparse::DirectSolver> solver` — already
        `analyze + factorize`'d, ready for solve
      - `Size state_size` — N + M (for sanity-checking callers)
- [x] 2.1.2 Move-only (the unique_ptr makes copy nonsensical).
- [x] 2.1.3 No methods beyond aggregate initialisation. The cache
      consumes it as data.

### 2.2 Tests `tests/v2/layer4/test_segment.cpp`
- [x] 2.2.1 Construct an empty segment + check defaults.
- [x] 2.2.2 Compile-time: `static_assert(!std::is_copy_constructible_v
      <PwlSegment>)` — guards the move-only contract.

## Phase 3 — Segment assembly (~0.75 days)

### 3.1 `pwl/assemble.hpp` — `assemble_segment`
- [x] 3.1.1 `void assemble_segment(const Graph& graph, const
      DevicePool& pool, const SwitchStateMask& mask, sparse::Matrix&
      J, Vector& b)` — zeroes J + b, then loops every branch in
      the graph.
- [x] 3.1.2 For each branch, dispatch by `BranchKind`:
      - `PassiveLinear` → must be a Resistor (V0 scope). Look up
        params via `pool.resistor_params(branch_id)`, call
        `stamping::stamp_device<Resistor>`.
      - `Source` → look up params + branch_var_id via the pool,
        call `stamping::stamp_voltage_source`.
      - `Switch` → look up params via `pool.switch_params(...)`,
        pick `closed = mask.get(switch_idx)` (counter advances per
        Switch-kind branch), call
        `stamping::stamp_switch_fixed`.
      - `Nonlinear` → SKIPPED in V0 (logged via a comment;
        no Layer-4 V0 build error — Layer 5 will handle them later).
- [x] 3.1.3 Uses `Vector x = Vector::Zero(state_size)` during
      stamping (the static stamping in V0 doesn't depend on the
      operating point — Resistors/Switches are linear and the
      voltage source contributes its constant `-V` to b regardless).

### 3.2 Tests `tests/v2/layer4/test_assemble.cpp`
- [x] 3.2.1 Empty graph → J is empty (0 x 0).
- [x] 3.2.2 Single Resistor between (0, gnd) with G=2 →
      J(0,0) == +2, b == zero.
- [x] 3.2.3 Single VoltageSource V=12 between (0, gnd) → J has 2
      entries on the constraint cross, b[branch_var_id] == -12.
- [x] 3.2.4 Single switch between (0, gnd), mask all-zero (open) →
      J(0,0) == g_off.
- [x] 3.2.5 Same switch, mask bit set (closed) → J(0,0) == g_on.
- [x] 3.2.6 Chopper assembly: V_dc(0, gnd) + Switch(0, 1) +
      R(1, gnd). With switch closed, J should have the expected
      conductance pattern; with open, J(1,1) reflects the open
      switch (R only — but wait, R is between 1 and gnd so
      J(1,1) = G_R + g_off; that's the right answer).

## Phase 4 — PwlStateSpaceCache (~1 day)

### 4.1 `pwl/cache.hpp`
- [x] 4.1.1 `class PwlStateSpaceCache` — constructor takes
      `const Graph&` and `const DevicePool&` (the cache borrows
      both; lifetime is the caller's responsibility).
- [x] 4.1.2 `void build()` — main build loop:
      - For each `SwitchStateMask m` from
        `enumerate_switch_states(graph.num_switches())`:
        - `assemble_segment(graph, pool, m, J, b)`
        - `compress_in_place(J)`
        - `make_default_solver()` → `analyze(J)` → `factorize(J)`
        - Construct `PwlSegment{J, b, solver, state_size}`
        - Insert into `segments_[m]` (unordered_map)
- [x] 4.1.3 `[[nodiscard]] const PwlSegment& lookup(
      const SwitchStateMask&) const` — throws `std::out_of_range`
      if the mask was not built (programmer error; the cache is
      complete after build()).
- [x] 4.1.4 `[[nodiscard]] Size num_segments() const` — for
      diagnostic / test use.
- [x] 4.1.5 `void solve(const SwitchStateMask&, const Vector&
      b_extra, Vector& x) const` — hot-loop entry. Combines
      `b_constant + b_extra`, calls the segment's solver. Layer 5
      uses `b_extra` for time-varying source values (PWM duty
      modulation in V1; in V0 it can be zero).
- [x] 4.1.6 Memory hint: a small `num_switches > 16` warning in
      `build()` (printed to nothing in production, but documented
      in design.md — at 17 switches there are 131k segments, each
      with a sparse matrix + factor, ~MB-scale memory). Layer 4
      V0 doesn't enforce the limit; the warning sets expectations.

### 4.2 Tests `tests/v2/layer4/test_cache.cpp`
- [x] 4.2.1 Build a cache for a graph with 0 switches → exactly 1
      segment (the empty mask).
- [x] 4.2.2 Build for 1 switch → exactly 2 segments.
- [x] 4.2.3 Build for 4 switches → exactly 16 segments.
- [x] 4.2.4 `lookup` for a never-built mask throws.
- [x] 4.2.5 `solve` on the V-R-GND-with-switch circuit at switch
      ON state returns the exact analytical answer (`v_n0 ≈ V_dc`).

## Phase 5 — Integration: chopper circuit end-to-end (~0.5 days)

### 5.1 `tests/v2/layer4/test_integration_chopper.cpp`
- [x] 5.1.1 Build the chopper: V_dc → Switch → R → GND.
      Nodes: `v_out` (between switch and resistor).
      Switches: 1. State-vector size: 2 (v_out + i_V).
- [x] 5.1.2 Build the cache (2 segments — switch open/closed).
- [x] 5.1.3 Verify segment ON: v_out = V_dc · g_on / (g_on + G_R)
      ≈ V_dc (for g_on=1e3, G_R=0.1 → v_out ≈ V_dc within < 0.01 %).
- [x] 5.1.4 Verify segment OFF: v_out = V_dc · g_off / (g_off +
      G_R) ≈ 0 (for g_off=1e-9, G_R=0.1 → v_out ≈ 1e-8).
- [x] 5.1.5 Performance smoke: `lookup` 10,000 times in a tight
      loop must take less than 100 ms (we're not benchmarking, but
      this catches accidental O(n²) regressions in the map lookup).

## Phase 6 — Documentation (~0.25 days)

### 6.1 `docs/pulsim-v2/layer4-pwl-state-space-cache.md`
- [x] 6.1.1 Section "Why PWL caching beats Newton-per-step" —
      complexity analysis. 5-30× Newton iterations per step vs ONE
      triangular solve.
- [x] 6.1.2 Section "How the layers compose" — the Layer 0 / 1 /
      2 / 3 dependencies, with a code walkthrough of the build
      loop.
- [x] 6.1.3 Section "What V0 does NOT do" — caps / inductors,
      nonlinear devices, node-equivalence dimension reduction.
      Pointers to follow-up OpenSpecs.

## Phase 7 — Validation

- [x] 7.1 `pulsim_v2_layer4_tests` MUST pass with zero failures.
      Initial target: ≥ 35 assertions / ≥ 15 test cases.
- [x] 7.2 Layers 0 + 1 + 2 + 3 tests MUST stay green.
- [x] 7.3 v1 suites MUST stay green.
- [x] 7.4 `openspec validate pulsim-v2-pwl-state-space-cache
      --strict` MUST pass.
