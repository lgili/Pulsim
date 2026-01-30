## Why

With `unify-v1-core` completed, PulsimCore now has a single runtime v1 kernel and YAML-only netlists. The next bottleneck is solver power and robustness: stiff switching circuits still need stronger nonlinear/linear strategies, better conditioning, and clearer telemetry. We also want predictable performance while keeping the Python/CLI surface stable.

This change focuses on upgrading the **solver stack** (linear + nonlinear) and the **runtime execution pipeline** to make simulations both faster and more reliable without introducing a separate v2 core.

## What Changes

### Solver Stack Upgrade (Runtime v1)
- Add **iterative linear solvers** (GMRES/BiCGSTAB/CG) with preconditioners (ILU0/Jacobi) and runtime auto-selection.
- Add **nonlinear accelerators** (Anderson/Broyden) and optional **Newton–Krylov** path.
- Expand fallback order and deterministic solver selection.
- Expose solver telemetry (iteration counts, fallbacks, conditioning hints).

### Robustness for Stiff Switching Circuits
- Stiffness-aware transient control (order caps, dt adaptation when repeated rejections occur).
- Improved Jacobian reuse and solver warm-starts.
- Event-aligned step splitting tuned for hard switching edges.

### YAML + Python/CLI Configuration
- Extend YAML schema to configure linear solver, nonlinear strategy, preconditioner, and fallback order.
- Keep Python/CLI API stable; new options are additive and optional.

## Impact

- Affected specs: `kernel-v1-core`, `netlist-yaml`.
- Affected code: `core/include/pulsim/v1`, `core/src/v1`, YAML parser, Python bindings, docs.
- JSON is **not** part of the migration path (YAML-only remains mandatory).

## Success Criteria

1. **Accuracy**: Analytical and SPICE comparisons within 0.1% where applicable.
2. **Robustness**: Difficult switching circuits converge with fewer failures and clear fallback traces.
3. **Performance**: 2x speedup on benchmark suite or reduced iteration counts for same accuracy.
4. **Determinism**: Fixed-order solver selection with reproducible results per hardware class.
5. **Compatibility**: Existing v1 Python/CLI flows continue to work without code changes.
