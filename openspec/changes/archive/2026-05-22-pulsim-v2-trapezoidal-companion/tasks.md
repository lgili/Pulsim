## Phase 1 — Capacitor + Inductor device models (~0.5 days)

### 1.1 `models/capacitor.hpp`
- [ ] 1.1.1 `struct Capacitor` with `Params{Real C}`,
      `kind = PassiveLinear`, `num_terminals = 2`,
      `is_dynamic = true` (new constexpr bool flag — Layer 4
      assemble dispatches on it).
- [ ] 1.1.2 `static Real g_eq(Real dt, Params)` → `2 * C / dt`.
- [ ] 1.1.3 `static Real history_term(Real v_prev, Real i_prev,
      Real dt, Params)` → `(2 * C / dt) * v_prev + i_prev`.
- [ ] 1.1.4 `template<FloatingPoint S> static S current(...)` —
      returns 0 (dynamic devices don't fit the static-current
      contract; companion stamping handles it separately).

### 1.2 `models/inductor.hpp`
- [ ] 1.2.1 `struct Inductor` with `Params{Real L}`,
      `kind = PassiveLinear`, `num_terminals = 2`,
      `is_dynamic = true`.
- [ ] 1.2.2 `static Real g_eq_inv(Real dt, Params)` →
      `dt / (2 * L)` — used in the constraint row.
- [ ] 1.2.3 `static Real history_term(Real v_prev, Real i_prev,
      Real dt, Params)` → `i_prev + (dt / (2*L)) * v_prev`.
- [ ] 1.2.4 `template<FloatingPoint S> static S current(...)` —
      returns 0.

### 1.3 Tests — `tests/v2/layer4_v1/test_capacitor.cpp`
- [ ] 1.3.1 `g_eq(C=1µF, dt=1µs) == 2.0`.
- [ ] 1.3.2 `g_eq(C=1µF, dt=10µs) == 0.2`.
- [ ] 1.3.3 `history_term(v=10, i=0.5, dt=1µs, C=1µF) ==
      g_eq · 10 + 0.5 == 20.5`.
- [ ] 1.3.4 Reflexive check: i_{n+1} = g_eq · v_{n+1} − I_hist
      reproduces back-substitution sanity.

### 1.4 Tests — `tests/v2/layer4_v1/test_inductor.cpp`
- [ ] 1.4.1 `g_eq_inv(L=1mH, dt=1µs) == 0.0005`.
- [ ] 1.4.2 `history_term(v=12, i=2, dt=1µs, L=1mH) ==
      2 + 0.0005·12 == 2.006`.

## Phase 2 — DevicePool extension (~0.25 days)

### 2.1 `pwl/device_pool.hpp` — additions
- [ ] 2.1.1 Add `add_capacitor(Index, Capacitor::Params)`. The
      capacitor does NOT add a branch-current unknown — its
      stamp is purely on the node-voltage rows.
- [ ] 2.1.2 Add `add_inductor(Index, Inductor::Params)`. The
      inductor DOES add a branch-current unknown (like a voltage
      source) — `state_size` grows by 1 per inductor. Internal
      counter `num_inductor_unknowns_` tracks the offset.
- [ ] 2.1.3 Extend `StoredKind` enum to include `Capacitor` and
      `Inductor`.
- [ ] 2.1.4 New getters: `capacitor_params`, `inductor_params`,
      `branch_var_id_for_inductor`.
- [ ] 2.1.5 `state_size(graph)` now returns
      `num_nodes + num_voltage_sources + num_inductors`.
- [ ] 2.1.6 New helper `num_dynamic_branches()` returns the
      count of Capacitor + Inductor entries — used by
      HistoryState to size itself.

### 2.2 Tests — `tests/v2/layer4_v1/test_device_pool_dynamic.cpp`
- [ ] 2.2.1 Pool with 1 cap → `state_size == num_nodes`
      (no extra unknown).
- [ ] 2.2.2 Pool with 1 inductor → `state_size == num_nodes + 1`.
- [ ] 2.2.3 Pool with 1 source + 1 inductor →
      `state_size == num_nodes + 2`.
- [ ] 2.2.4 `branch_var_id_for_inductor` returns the right
      offset (after sources, ordered by insertion).

## Phase 3 — Companion stamping helper (~0.5 days)

### 3.1 `stamping/stamp_companion.hpp`
- [ ] 3.1.1 `stamp_capacitor_companion(J, b, coord, g_eq,
      history)` — stamps the 4-entry conductance block (like a
      resistor with G = g_eq) AND adds `±history` to the b
      vector on the from/to rows.
- [ ] 3.1.2 `stamp_inductor_companion(J, b, coord, branch_var_id,
      g_eq_inv, history)` — stamps:
      - KCL on from-node: `+i_L` (column branch_var_id).
      - KCL on to-node:   `−i_L`.
      - Constraint row:   `v_from − v_to − g_eq_inv_recip · i_L
        = −g_eq_inv_recip · history`.
      Wait — the constraint shape is `v_from − v_to − (2L/dt) ·
      (i_n+1 − history) = 0`. Stamping:
        - J(row=branch_var_id, col=from) = +1
        - J(row=branch_var_id, col=to)   = −1
        - J(row=branch_var_id, col=branch_var_id) = −(2L/dt)
          = −1 / g_eq_inv
        - b(row=branch_var_id) = −(2L/dt) · history
          = −history / g_eq_inv
- [ ] 3.1.3 Both functions handle `kGround` endpoints
      gracefully (skip row/col stamps where endpoint is ground).

### 3.2 Tests — `tests/v2/layer4_v1/test_stamp_companion.cpp`
- [ ] 3.2.1 Capacitor between (node0, GND) with C=1µF, dt=1µs,
      no history → J(0,0) = 2.0, b(0) = 0.
- [ ] 3.2.2 Capacitor with history=10 → J(0,0) = 2.0,
      b(0) = +10.0 (current source pushes INTO node 0).
- [ ] 3.2.3 Inductor between (node0, GND) with L=1mH, dt=1µs,
      branch_var_id=1, no history → J pattern matches the
      voltage-source pattern with the `−2L/dt` on the diagonal.
- [ ] 3.2.4 Edge case: capacitor with both endpoints active
      (no ground touch) → all 4 entries stamped.

## Phase 4 — Layer 4 cache dt-aware build (~0.5 days)

### 4.1 `pwl/assemble.hpp` — extended dispatch
- [ ] 4.1.1 Add `Real dt` parameter to `assemble_segment` (after
      mask). When `dt == 0`, dynamic devices are SKIPPED (V0
      backwards compat).
- [ ] 4.1.2 New dispatch arm under `PassiveLinear`:
      - Use `pool.kind_of(branch.id)` to distinguish R / C / L
        (V0 only had Resistor under PassiveLinear).
      - Resistor → existing path.
      - Capacitor → `stamp_capacitor_companion(...)` with
        `history = 0` (history term is added by HistoryState via
        b_extra, NOT during assembly).
      - Inductor → `stamp_inductor_companion(...)` with
        `history = 0`.
- [ ] 4.1.3 Document the convention: the assembled `b_constant`
      holds ONLY the voltage-source `−V` terms. History
      contributions come in via `b_extra` at solve time.

### 4.2 `pwl/cache.hpp` — dt-aware build
- [ ] 4.2.1 Add private `Real dt_` member, default 0.
- [ ] 4.2.2 New `void build(Real dt)` method — stores dt,
      calls assemble_segment with that dt for every segment.
- [ ] 4.2.3 Keep existing `void build()` as a shim that calls
      `build(Real{0})` — V0 backwards compat.
- [ ] 4.2.4 Add `[[nodiscard]] Real dt() const noexcept`.
- [ ] 4.2.5 If `build(dt)` is called twice with different dt,
      clear and rebuild. (Document: not a hot path; rebuilding
      is expensive.)

### 4.3 Tests — `tests/v2/layer4_v1/test_dt_aware_cache.cpp`
- [ ] 4.3.1 Cache built with `build()` on a graph with no caps/L
      behaves identically to V0 cache (regression).
- [ ] 4.3.2 Cache built with `build(dt=1µs)` on a graph with one
      cap stamps `g_eq = 2C/dt` correctly.
- [ ] 4.3.3 Cache built with `build(dt=1µs)` then `build(dt=2µs)`
      rebuilds — `cache.dt() == 2µs`.
- [ ] 4.3.4 Cache built with `build()` (no dt) on a graph with a
      capacitor SILENTLY skips the cap (the stamp doesn't fire).
      Document this behaviour.

## Phase 5 — HistoryState (~0.75 days)

### 5.1 `pwl/history_state.hpp`
- [ ] 5.1.1 `struct HistoryEntry { Index branch_id; StoredKind
      kind; Real v_prev = 0; Real i_prev = 0; }`.
- [ ] 5.1.2 `class HistoryState`:
      - Constructor: `(const Graph&, const DevicePool&)` —
        builds entries vector from pool's caps + inductors.
      - `void reset()` — zeros all entries.
      - `[[nodiscard]] Vector compute_b_extra(const Vector& x_prev,
        Real dt) const` — returns a `Vector` of size
        `pool.state_size(graph)`, populated with history-term
        contributions on the right rows.
      - `void update_from_state(const Vector& x_prev)` — reads
        v_n and i_n for each entry from x_prev, stores them.
- [ ] 5.1.3 Inductor branch-current i_n is read directly from
      `x_prev[branch_var_id_for_inductor(...)]`. Capacitor's
      i_n is computed from the trap relation:
      `i_n = g_eq · (v_n − v_n_minus_1)` ... actually no — the
      cap's i_prev is part of the trap state, not computed from
      v. We need to STORE it explicitly. So `update_from_state`
      must compute i_n at step n from: i_n = g_eq · v_n − history_n
      where history_n is the value used at step n. That's a
      one-step recurrence; entry maintains both v and i.

      Cleaner: store `i_prev` and `v_prev`. At step n+1:
      - compute history from (v_n, i_n) ← previous step's values.
      - solve → x_{n+1}
      - extract v_{n+1} from x_{n+1}
      - compute i_{n+1} = g_eq · v_{n+1} − history (from the
        device's companion equation)
      - store (v_{n+1}, i_{n+1}) for next step.

      So the API needs the dt as an arg to `update_from_state` to
      compute i_{n+1} via g_eq. Or we precompute it during
      `compute_b_extra`. Choose: precompute at the START of
      compute_b_extra (which is called AFTER the previous step
      finalised x). Actually, the cleanest layout is:
      - After cache.solve at step n, the loop calls
        `history.update_from_state(x_n, dt)`. This computes
        i_n = g_eq · v_n − history_n_minus_1 for each cap, stores
        (v_n, i_n).
      - At step n+1, history.compute_b_extra(dt) uses the stored
        values.

### 5.2 Tests — `tests/v2/layer5_v1/test_history_state.cpp`
- [ ] 5.2.1 HistoryState for a graph with no caps/L → empty
      entries, compute_b_extra returns zero vector.
- [ ] 5.2.2 HistoryState for one cap → one entry, initialised
      to zeros. compute_b_extra at the first step returns zeros
      (no previous state).
- [ ] 5.2.3 After update_from_state with known x_prev, the
      entry's v_prev matches the cap's terminal voltage and
      i_prev matches `g_eq · v_prev` (the current at step n
      assuming all-zero IC).
- [ ] 5.2.4 reset() zeros all entries.

## Phase 6 — Layer 5 run_transient with history (~0.5 days)

### 6.1 `solver/run_transient.hpp` — extended loop
- [ ] 6.1.1 `run_transient` now needs access to the Graph and
      DevicePool (to build HistoryState). Two API options:
      - **A.** Add `const Graph&` and `const DevicePool&` as
        extra args.
      - **B.** Add a `History` parameter and let the caller
        construct it.

      Choose **A** (simpler — the user already has those
      objects to build the cache).

- [ ] 6.1.2 New signature:
      ```cpp
      SimulationResult run_transient(
          const PwlStateSpaceCache& cache,
          const Graph& graph,
          const DevicePool& pool,
          const SimulationOptions& opts,
          const SwitchScheduleFn& switch_fn,
          const BExtraFn& b_extra_fn = {});
      ```
      Drops `Size state_size` — derived from `pool.state_size(graph)`.

- [ ] 6.1.3 Keep the V0 5-arg signature as a backwards-compat
      overload (forwards to the new one with a synthesised
      empty Graph + Pool? No — that breaks the API. Better:
      mark V0 signature deprecated and ship a one-line shim
      that requires a graph/pool).

      Decision: **break the API for V0 callers** — they need
      to pass graph + pool now. The existing chopper test gets
      updated. Migration burden is low (1 line per call site).

- [ ] 6.1.4 Loop body becomes:
      ```
      history.reset();
      Vector x = Vector::Zero(state_size);
      for k = 0 .. N - 1:
        t = t_start + k * dt
        b_extra = history.compute_b_extra(x, dt);
        if (b_extra_fn) b_extra += b_extra_fn(t);
        auto mask = switch_fn(t);
        cache.solve(mask, b_extra, x);
        history.update_from_state(x, dt);
        record(t, x);
      ```

- [ ] 6.1.5 Validation: if `cache.dt() != opts.dt`, throw
      `std::invalid_argument` — the cache was built for a
      different dt.

### 6.2 Tests — `tests/v2/layer5_v1/test_run_transient_history.cpp`
- [ ] 6.2.1 Static-only circuit (no caps/L) gives bit-identical
      result to the V0 run (regression).
- [ ] 6.2.2 cache.dt() ≠ opts.dt throws.

## Phase 7 — Integration: RC, RL, RLC (~1 day)

### 7.1 `tests/v2/layer5_v1/test_integration_rc.cpp`
- [ ] 7.1.1 Build RC: V_dc(5V) → R(1Ω) → C(1µF) → GND.
      τ = RC = 1µs. dt = 10ns (τ/100). t_end = 5τ = 5µs (500
      steps).
- [ ] 7.1.2 Simulate; verify V_C(t) matches
      `V_dc · (1 − e^{−t/τ})` within < 0.5 % at every sample.
- [ ] 7.1.3 V_C(5τ) ≈ V_dc · 0.9933 (within 1 %).

### 7.2 `tests/v2/layer5_v1/test_integration_rl.cpp`
- [ ] 7.2.1 Build RL: V_dc(12V) → R(1Ω) → L(10µH) → GND.
      τ = L/R = 10µs.
- [ ] 7.2.2 Simulate; verify I_L(t) matches
      `(V/R) · (1 − e^{−t/τ})` within < 0.5 %.

### 7.3 `tests/v2/layer5_v1/test_integration_rlc.cpp`
- [ ] 7.3.1 Build RLC series: V_dc(10V) → R(0.5Ω) → L(1µH) → C(1µF) → GND.
      ω_n = 1/√(LC) = 10^6 rad/s. ζ = (R/2)·√(C/L) = 0.5/2 ·
      √(1µF/1µH) = 0.25 · 1 = 0.25 (underdamped).
      ω_d = ω_n √(1−ζ²) ≈ 0.968 · 10^6 rad/s.
- [ ] 7.3.2 Simulate 5 periods at dt = T/100. Verify the first
      zero-crossing of V_C(t) occurs at the analytical
      `t_1 = (π/2 − φ) / ω_d` within < 2 %.
- [ ] 7.3.3 Verify the envelope decays as `e^{−ζω_n t}` within
      < 5 % (trapezoidal preserves energy fairly but not
      perfectly).

## Phase 8 — Documentation (~0.25 days)

### 8.1 `docs/pulsim-v2/layer4-v1-trapezoidal-companion.md`
- [ ] 8.1.1 Section "The companion-model math" — derivation
      for C and L from the trap rule.
- [ ] 8.1.2 Section "Why trap, not BE / BDF" — second-order +
      energy-preserving justification.
- [ ] 8.1.3 Section "Cache dt-dependency" — why we rebuild on
      dt change.
- [ ] 8.1.4 Section "History plumbing" — Layer 5's role.
- [ ] 8.1.5 Worked example: RC charging.
- [ ] 8.1.6 Worked example: RLC ringdown.
- [ ] 8.1.7 Section "What V2 will need (DC OP, adaptive dt,
      events, nonlinear)".

## Phase 9 — Validation gates

- [ ] 9.1 `pulsim_v2_layer4_v1_tests` + `pulsim_v2_layer5_v1_tests`
      MUST pass with zero failures. Initial target: ≥ 30 cases
      across both binaries.
- [ ] 9.2 All existing Layer 0/1/2/3/4/5 tests MUST stay green
      (the V0 build() shim has no behaviour change).
- [ ] 9.3 v1 `pulsim_tests` MUST stay green.
- [ ] 9.4 `openspec validate pulsim-v2-trapezoidal-companion
      --strict` MUST pass.
