## Context

The v1 runtime kernel is now the single simulation engine (YAML-only). The next step is improving solver power and robustness to handle stiff switching converters reliably while keeping the Python/CLI API stable.

## Goals / Non-Goals

### Goals
- Stronger linear + nonlinear solver stack (direct + iterative + acceleration)
- Deterministic solver selection and clear telemetry
- Robust transient behavior for stiff switching circuits
- Maintain v1 runtime API and YAML schema (additive options only)

### Non-Goals
- Separate v2 core or parallel API
- GPU acceleration
- Full SPICE feature parity

## Decisions

### 1) Runtime-First v1 Core
**Decision**: Keep `pulsim::v1` runtime circuit as the sole IR; avoid a second core.

**Rationale**:
- Already used by Python and CLI
- Runtime construction is required for GUI/Notebook workflows
- Compile-time optimizations can still be applied locally (device stamping, SIMD kernels)

### 2) Pluggable Solver Stack
**Decision**: Introduce a runtime-configurable solver stack with deterministic fallback order.

**Linear Solvers**:
- Direct: KLU (primary), Eigen SparseLU (fallback)
- Iterative: GMRES/BiCGSTAB/CG with preconditioners

**Nonlinear Solvers**:
- Newton with line search and trust-region damping
- Anderson/Broyden acceleration as optional strategies
- Newton–Krylov path using iterative linear solvers

### 3) Preconditioning + Scaling
**Decision**: Support ILU0/Jacobi preconditioners and optional scaling to stabilize iterative solvers.

### 4) Stiffness-Aware Transient Control
**Decision**: Detect stiffness and adjust dt/order dynamically (BDF order caps, rejection backoff).

### 5) Telemetry + Determinism
**Decision**: Emit solver telemetry (iterations, fallbacks, conditioning hints) and enforce deterministic solver order.

### 6) YAML Solver Configuration
**Decision**: Extend the YAML `simulation` section with optional `solver` settings (linear, nonlinear, preconditioner, fallback order).

## Risks / Trade-offs
- Iterative solvers may require tuning; provide conservative defaults and robust fallback to direct solvers.
- Additional configuration options increase surface area; strict validation is required.

## Migration Plan
1. Add solver configuration model (C++ + YAML).
2. Implement iterative linear solvers + preconditioners.
3. Implement nonlinear accelerators and Newton–Krylov.
4. Add telemetry and determinism tests.
5. Update docs and examples.
