## Context

Each device file in `core/include/pulsim/v1/components/` contains a hand-derived `stamp_jacobian_impl` method. For example, [mosfet.hpp:88-102](core/include/pulsim/v1/components/mosfet.hpp:88) hand-stamps `gm` and `gds` contributions into nine matrix entries with sign-sensitive arithmetic. This pattern duplicates the chain-rule across components and is the most likely source of silent convergence-rate degradations.

AD is a mature solution. Eigen ships `AutoDiffScalar` (forward-mode, header-only); Enzyme offers reverse-mode at LLVM IR level; CppAD/CasADi provide stand-alone toolkits. We choose Eigen `AutoDiffScalar` because it is already a dependency, requires no new build system, and forward-mode is appropriate for ≤32-terminal stamps (the dominant device size in power electronics).

## Goals / Non-Goals

**Goals:**
- Eliminate hand-derived Jacobians from new devices.
- Allow Python users to register custom devices without C++.
- Catch existing manual-stamp bugs via FD vs AD cross-validation.
- Keep PWL hot path (Phase-0 work) unaffected.

**Non-Goals:**
- Reverse-mode AD (we don't need scalar→vector pullbacks; forward-mode is correct for stamp Jacobian shape).
- AD for time-stepping (the integrator coefficients are independent of state; no AD needed).
- AD for parameter sensitivity (covered by future `parameter-sweep` change).
- Replacing companion-model coefficients for L/C; trapezoidal Tustin is closed-form and AD-free.

## Decisions

### Decision 1: Forward-mode AD via `Eigen::AutoDiffScalar<Eigen::VectorXd>`
- **What**: `ADReal = Eigen::AutoDiffScalar<Eigen::VectorXd>` carries the value plus a derivative vector sized to terminal count. For a 3-terminal MOSFET, derivatives are a 3-vector.
- **Why**: Forward-mode is O(n) per Jacobian column, but for stamps (small n, small fan-out) this is optimal. Eigen integrates seamlessly with our existing `SparseMatrix`/`VectorXd` types.
- **Alternatives considered**:
  - *Reverse-mode (Enzyme)* — overkill, requires LLVM passes, brittle across compilers.
  - *CppAD / CasADi* — adds heavy dependency; CasADi requires symbolic graph build, not ergonomic for procedural device code.
  - *Finite differences* — already supported, but accuracy degrades near operating-point switches; slower and less robust than AD.

### Decision 2: Hybrid — AD only for nonlinear devices in Behavioral mode
- **What**: Linear devices keep direct stamps. Nonlinear devices (`Behavioral`-mode MOSFET/IGBT/diode) use AD. PWL-mode devices stamp constants from topology.
- **Why**: AD has measurable per-stamp cost (~10–30%). For linear devices the Jacobian is exactly known and constant per topology; AD adds cost for zero benefit. PWL hot path is allocation-bounded and AD-free by construction.
- **Trade-off**: Slightly more code paths, but each is simpler than monolithic AD-everywhere.

### Decision 3: Validation layer that runs FD vs AD at startup
- **What**: New `Simulator::validate_jacobians(operating_points)` evaluates each device at user-specified or curated operating points using both finite differences and AD, asserts agreement within 1e-8 relative.
- **Why**: This is how we discover and retire any incorrect manual stamps that exist today, and how we keep AD honest after refactors. Also doubles as a regression gate in CI.
- **Cost**: Sub-millisecond per device; runs only on `--validate-jacobians` or once per CI job.

### Decision 4: Python custom devices via pybind11 + capture into AD
- **What**: User passes a Python callable `residual_fn(x, t, dt, params)`. The C++ side wraps it; for AD we build a thin trampoline that calls the function with `ADReal` inputs (Eigen handles the value+derivative arithmetic transparently for elementary ops, but Python returns plain doubles).
- **Implementation note**: Because Python evaluates with `double`, we run the user's callable separately for each derivative direction (perturbed input). For typical power-electronics devices (3–4 terminals), this is 3–4 Python calls per stamp — acceptable for prototyping. Performance-sensitive users supply `jacobian_fn` directly.
- **Why**: This is the lowest-friction extension story. Anything more advanced (XLA, JAX) we can layer on later.

### Decision 5: Deprecate manual stamps over one release window
- **What**: Manual `stamp_jacobian_impl` retained behind `PULSIM_LEGACY_MANUAL_JACOBIAN`. Deprecation warning when used. Removed after one release; separate change `remove-legacy-manual-jacobians` archives this.
- **Why**: Allows incremental migration, gives users time to validate convergence equivalence on their own circuits.

## Risks / Trade-offs

| Risk | Mitigation |
|------|------------|
| AD overhead degrades convergence-bound performance | Hybrid strategy; PWL path AD-free; benchmark gate ≤15% total overhead |
| Python custom-device performance is poor | Provide `jacobian_fn` opt-in; document expected use case (prototyping, not production hot loop) |
| Eigen `AutoDiffScalar` quirks (e.g. with `std::complex`) | Restrict to real-valued scalars in v1; complex-AD deferred |
| Compile-time impact from heavy Eigen template instantiation | Explicitly instantiate AD device residuals in `.cpp` files where possible |
| FD vs AD cross-validation may flag known-good devices as failures near regime transitions | Curate operating points away from kinks; use `1e-8` relative tolerance not absolute |
| Custom Python devices interact poorly with multithreading | Document that GIL is held during stamp; recommend C++ for heavy-perf devices |

## Migration Plan

1. **Phase 0 (this PR)**: Land AD bridge + hybrid strategy + per-device cross-validation. AD is the default for nonlinear devices; manual path behind flag.
2. **Phase 1**: Burn-in period (one release). Collect convergence data from users; address any AD-introduced regressions.
3. **Phase 2** (separate change): Remove manual stamps. Update docs.

Rollback: `PULSIM_LEGACY_MANUAL_JACOBIAN=1` env var or `SimulationOptions.use_manual_jacobian = true` restores prior path bit-for-bit.

## Open Questions

- Should AD seed-vector size be inferred from device terminal count or globally `n_state`? *Lean: per-device for sparsity-aware perf.*
- How to expose AD-derived Jacobian sparsity pattern to KLU symbolic factorization? *Lean: existing `jacobian_pattern_impl` static metadata; AD just fills values.*
- Is `Eigen::AutoDiffScalar` thread-safe in our usage? *Need to verify; likely yes since each stamp is local.*
