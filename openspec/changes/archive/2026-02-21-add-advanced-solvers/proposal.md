## Why

Even with the unified v1 runtime core, stiff switching circuits and large sparse systems still hit limits in convergence speed and robustness. We need stronger nonlinear/linear solvers and better transient integrators to improve accuracy and reduce runtime, while keeping the Python/CLI workflow stable.

Key gaps to address:
- No Jacobian‑free Newton–Krylov (JFNK) path for large systems.
- Limited preconditioning choices for iterative solvers.
- Missing stiff‑aware integrators (TR‑BDF2 / Rosenbrock‑W / SDIRK).
- No periodic steady‑state methods (shooting / harmonic balance) for switching converters.
- YAML solver config does not allow separate primary vs fallback solver order.
- CG can be selected even when the matrix is not SPD.

## What Changes

### Solver Stack Upgrades
- Add JFNK with Krylov linear solver and Jacobian‑vector products.
- Add preconditioners beyond ILU0/Jacobi (ILUT and AMG) where possible.
- Enforce SPD checks for CG; disallow CG when the matrix is not SPD.
- Allow explicit primary solver order and fallback order.

### Stiff‑Aware Integrators
- Implement TR‑BDF2 and Rosenbrock‑W/SDIRK integrators.
- Add stiffness detection to adapt order/timestep and avoid instability.

### Periodic Steady‑State
- Add periodic steady‑state solvers (shooting and harmonic balance) for switching converters.

### YAML + API Configuration
- Extend YAML schema to configure new solver options and integrators.
- Keep Python/CLI behavior unchanged unless the new options are used.

## Impact

- Affected specs: `kernel-v1-core`, `netlist-yaml`.
- Affected code: `core/include/pulsim/v1`, `core/src/v1`, YAML parser, Python bindings.

## Success Criteria

1. **Accuracy**: Analytical and SPICE comparisons remain within 0.1% where applicable.
2. **Robustness**: Stiff switching circuits converge with fewer failures and clear solver telemetry.
3. **Performance**: 2x speedup or equivalent iteration reduction on benchmark suite.
4. **Determinism**: Fixed solver order and reproducible results per hardware class.
5. **Compatibility**: Existing v1 Python/CLI flows continue to work without changes.
