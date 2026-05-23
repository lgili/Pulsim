## Phase 1 — MNA convention + BranchCoord (~0.5 days)

### 1.1 `stamping/branch_coord.hpp`
- [x] 1.1.1 `struct BranchCoord { Index from; Index to; Index
      branch_id; };`. The minimal coordinate every 2-terminal
      stamper takes. `from` / `to` are node indices (>= 0) or
      `kGround` (-1). `branch_id` is for diagnostics / cache
      invalidation (Layer 4 keys on it).
- [x] 1.1.2 Helper `Real read_node_voltage(const Vector& x, Index
      node) noexcept` — returns `x[node]` if `node >= 0`, else
      `Real{0}` (ground convention).
- [x] 1.1.3 Helper `bool node_is_active(Index node) noexcept` —
      returns `node >= 0`. Cuts out rows/cols touching ground in
      the stamping loops.

### 1.2 `stamping/mna_convention.hpp`
- [x] 1.2.1 Documents the convention in a header-comment block (no
      executable code — pure documentation embedded as a
      compilable `.hpp` so the convention is searchable from any
      stamping file).
- [x] 1.2.2 Includes a static-assert that `kGround == -1` (locks
      the assumption Layer 3 relies on).

### 1.3 Tests `tests/v2/layer3/test_branch_coord.cpp`
- [x] 1.3.1 `read_node_voltage` returns `x[i]` for `i >= 0`.
- [x] 1.3.2 `read_node_voltage` returns 0 for `kGround`.
- [x] 1.3.3 `node_is_active` returns true for non-negative indices,
      false for `kGround`.

## Phase 2 — Generic 2-terminal device stamper (~1 day)

### 2.1 `stamping/stamp_device.hpp`
- [x] 2.1.1 `template <models::DeviceModel T> void stamp_device(
      sparse::Matrix& J, Vector& f, const Vector& x, const
      BranchCoord& coord, const typename T::Params& p) noexcept`.
- [x] 2.1.2 Reads `v[0] = read_node_voltage(x, coord.from)` and
      `v[1] = read_node_voltage(x, coord.to)`.
- [x] 2.1.3 Calls `models::evaluate_current_and_jacobian<T>(v, p)`
      → gets `(i, std::array<Real, 2> J_partials)`.
- [x] 2.1.4 Stamps residual:
      - `if (node_is_active(coord.from)) f[coord.from] += i;`
      - `if (node_is_active(coord.to))   f[coord.to]   -= i;`
- [x] 2.1.5 Stamps Jacobian (4 entries, gated on each node being
      active):
      - `J[from, from] += J_partials[0]`
      - `J[from, to]   += J_partials[1]`
      - `J[to,   from] -= J_partials[0]`
      - `J[to,   to]   -= J_partials[1]`
- [x] 2.1.6 Static-assert `T::num_terminals == 2` — the 2-terminal
      stamper rejects 3-terminal devices at compile time. The
      multi-pin extension is a follow-up OpenSpec; Layer 3 V0
      is 2-terminal only.

### 2.2 Tests `tests/v2/layer3/test_stamp_device.cpp`
- [x] 2.2.1 Resistor between nodes (1, 2), G = 2, x = [0, 3, 1] →
      i = 4; `f[1] += 4`, `f[2] -= 4`; J(1,1)+=2, J(1,2)-=2,
      J(2,1)-=2, J(2,2)+=2.
- [x] 2.2.2 Resistor between node 1 and ground (kGround) →
      only `f[1] += i`, only J(1,1) entry. No stamps for the
      ground row/column.
- [x] 2.2.3 IdealDiode forward-biased between nodes (0, 1) →
      Jacobian off-diagonal entries match the per-terminal AD
      partials from Layer 2; residual entries sum to zero (KCL
      sanity).
- [x] 2.2.4 Static-assert: trying to stamp a hypothetical
      3-terminal model (mock with `num_terminals = 3`) fails at
      compile time (verified via `if constexpr (requires { ... })`).
- [x] 2.2.5 Stamping is additive: stamping two resistors in
      parallel between the same nodes accumulates their
      conductances on the diagonal.

## Phase 3 — Voltage-source constraint stamper (~0.5 days)

### 3.1 `stamping/stamp_voltage_source.hpp`
- [x] 3.1.1 `void stamp_voltage_source(sparse::Matrix& J, Vector& f,
      const Vector& x, const BranchCoord& coord, Index
      branch_var_id, Real V) noexcept`.
- [x] 3.1.2 KCL contribution at the terminal nodes:
      - `J[from, branch_var_id] += 1`
      - `J[to,   branch_var_id] -= 1`
      - `f[from] += x[branch_var_id]`
      - `f[to]   -= x[branch_var_id]`
- [x] 3.1.3 Constraint row at `branch_var_id`:
      - `J[branch_var_id, from] += 1`
      - `J[branch_var_id, to]   -= 1`
      - `f[branch_var_id] = read_node_voltage(x, from) -
        read_node_voltage(x, to) - V`
- [x] 3.1.4 Ground handling: when `from == kGround` or
      `to == kGround`, the corresponding entries are skipped
      (no row/col exists for ground) but the constraint still
      reads the ground voltage as 0.

### 3.2 Tests `tests/v2/layer3/test_stamp_voltage_source.cpp`
- [x] 3.2.1 Source between nodes (0, 1) with V = 5, x = [3, 0,
      2_branch], branch_var_id = 2:
      - Residual `f[2] = 3 - 0 - 5 = -2`
      - KCL `f[0] += 2`, `f[1] -= 2`
      - Jacobian: J(0,2)=+1, J(1,2)=-1, J(2,0)=+1, J(2,1)=-1
- [x] 3.2.2 Source between node 0 and ground, V = 12 → constraint
      `f[branch_var_id] = x[0] - 0 - 12 = x[0] - 12`. Stamps only
      the active-row entries (J(0, branch_var_id), J(branch_var_id,
      0)).
- [x] 3.2.3 At convergence (`x[0] = V`, `i_branch = 0`), the
      residual is zero on all touched rows.

## Phase 4 — Fixed-state switch stamper (~0.25 days)

### 4.1 `stamping/stamp_switch.hpp`
- [x] 4.1.1 `void stamp_switch_fixed(sparse::Matrix& J, Vector& f,
      const Vector& x, const BranchCoord& coord, bool closed,
      Real g_on, Real g_off) noexcept`. Used by Layer 4 to
      materialise per-segment matrices: the SwitchStateMask bit
      picks `g_on` or `g_off`.
- [x] 4.1.2 Internal implementation reuses the Resistor stamping
      pattern with `G = closed ? g_on : g_off`. No new logic —
      pure composition of the existing stamper.

### 4.2 Tests `tests/v2/layer3/test_stamp_switch.cpp`
- [x] 4.2.1 Closed switch (g_on = 1e3) between nodes (0, 1)
      stamps a 1e3-conductance block at (0,0), (0,1), (1,0),
      (1,1).
- [x] 4.2.2 Open switch (g_off = 1e-9) stamps a 1e-9 conductance
      block — circuit can still solve (matrix non-singular) but
      effectively isolates the two nodes.
- [x] 4.2.3 Switch touching ground stamps only the active-row
      entry.

## Phase 5 — Integration test: V-R-GND circuit (~0.5 days)

### 5.1 `tests/v2/layer3/test_integration_vrgnd.cpp`
- [x] 5.1.1 Assemble a 1-node circuit: V_dc → node 0 → R → GND.
      State vector: `[v_node_0, i_branch]`. Stamp the voltage
      source and the resistor manually.
- [x] 5.1.2 At convergence the state must satisfy `v_node_0 = V`
      and `i_branch = V·G` (Ohm's law). Use the assembled (J, f)
      to do ONE Newton step from `x = [0, 0]` and verify the
      result.
- [x] 5.1.3 Use Layer 0's `SparseLuSolver` to solve `J · Δx = -f`
      and check the Newton step lands at the analytical answer
      within 1e-12 (the system is linear, one Newton step is
      exact).

## Phase 6 — Documentation (~0.25 days)

### 6.1 `docs/pulsim-v2/layer3-stamping-pipeline.md`
- [x] 6.1.1 Section "MNA convention" — state vector layout, sign
      convention, ground handling.
- [x] 6.1.2 Section "How Layer 3 kills the v1 stamping duplication"
      — one generic `stamp_device<T>` for every device satisfying
      the concept. Worked example using IdealDiode.
- [x] 6.1.3 Section "What's deferred to follow-ups" — 3-terminal
      devices, history-term stamping for caps/inductors (Layer 4
      trapezoidal companion).

## Phase 7 — Validation

- [x] 7.1 `pulsim_v2_layer3_tests` MUST pass with zero failures.
      Initial target: ≥ 25 assertions / ≥ 10 test cases.
- [x] 7.2 Layer 0 + 1 + 2 tests MUST stay green.
- [x] 7.3 v1 suites MUST stay green.
- [x] 7.4 `openspec validate pulsim-v2-generic-stamping-pipeline
      --strict` MUST pass.
