## Why

V2 ships R / L / C linear elements, switches, diodes, and
MOSFETs/IGBTs (V1 of Layer 2). What's still missing for
practical SMPS work is **magnetic coupling**: transformers
and coupled inductors. Without it, isolated topologies
(flyback, forward, push-pull, full-bridge, half-bridge) can't
be modelled.

V2 of Layer 2 ships a **two-winding transformer** as a pair
of coupled inductors with the standard trap-companion
discretization. The model:

- v_p = L_p · di_p/dt + M · di_s/dt
- v_s = M · di_p/dt + L_s · di_s/dt
- M = k · √(L_p · L_s),  0 ≤ k ≤ 1

`k = 1` is perfect coupling (idealized turns ratio); k < 1
adds leakage inductance.

The trap-companion discretization is identical to the
existing single-inductor model PLUS a cross-term `(2M/dt)`
between the two winding branch currents. Implementation
reuses the existing Inductor stamping + a small
"transformer coupling" registry.

## What Changes

**Scope decision — Layer 2 V2** (two-winding linear
transformer):

- New `pulsim/v2/models/transformer.hpp` with
  `TwoWindingTransformer::Params { L_p, L_s, k }` and the
  mutual-inductance helper `M(p, dt) = k·√(L_p·L_s)`.

- Extend `DevicePool`:
  - `add_transformer_coupling(p_branch_id, s_branch_id,
      params)` registers the coupling between two
    already-added inductor branches.
  - `transformer_couplings()` returns the list of `(p_id,
      s_id, params)` triples for `assemble.hpp` /
    `HistoryState` to iterate.

- Extend `assemble.hpp`: after the per-branch stamping
  loop, iterate `transformer_couplings()` and stamp the
  cross-terms:
  - `J[p_constraint_row, s_branch_var_col] += -(2M/dt)`
  - `J[s_constraint_row, p_branch_var_col] += -(2M/dt)`

- Extend `HistoryState`:
  - Track previous-step currents for all inductors (already
    done).
  - In `compute_b_extra`, for each transformer coupling
    add `(2M/dt) · i_s_prev` to `b_extra(p_row)` and
    `(2M/dt) · i_p_prev` to `b_extra(s_row)`.

- Add `CircuitBuilder::add_transformer(name, p_from, p_to,
  s_from, s_to, L_p, L_s, k=1.0)` that creates the two
  inductor branches + registers the coupling.

- Python binding: `pulsim.v2.CircuitBuilder.add_transformer(...)`.

- Tests:
  - **Coupled-energy test**: 1:1 transformer (L_p=L_s=1mH,
    k=1) with V_dc step on primary; secondary should
    show the magnetizing-current ramp.
  - **Turns-ratio test**: 2:1 transformer (L_p=4·L_s,
    k=1) at sinusoidal steady state; v_secondary / v_primary
    ≈ 1/2.
  - **k=0 isolation test**: with k=0, primary and
    secondary should be electrically isolated (no current
    flows between them).
  - **Builder ergonomics**: `add_transformer` returns the
    builder for chaining; creates exactly 2 branches
    (one per winding).
  - **Python smoke test**: build a transformer from Python
    and verify branch count.

## Impact

- **Affected specs**: ADDED requirement on
  `kernel-v2-solver` for `TwoWindingTransformer` +
  `add_transformer` helpers.
- **Affected code** (~400 LOC):
  - NEW `core/include/pulsim/v2/models/transformer.hpp`
  - MODIFIED `core/include/pulsim/v2/pwl/device_pool.hpp`
    (+ coupling registry)
  - MODIFIED `core/include/pulsim/v2/pwl/assemble.hpp`
    (+ cross-term pass)
  - MODIFIED `core/include/pulsim/v2/pwl/history_state.hpp`
    (+ cross-term history)
  - MODIFIED `core/include/pulsim/v2/builder/circuit_builder.hpp`
    (+ add_transformer)
  - MODIFIED `python/bindings_v2_kernel.cpp` (+ Python binding)
  - NEW tests in `core/tests/v2/layer5_v1/test_transformer.cpp`
  - NEW Python test cases
- **Migration**: zero. Pure additive — existing tests
  unchanged.
- **Risk**: medium. Cross-coupling stamping changes the
  MNA matrix's sparsity pattern; tests guard the
  arithmetic.
