## Context

The unified v1 runtime core now handles DC + transient simulation and YAML netlists. To reach production‑grade performance for stiff switching power electronics, we need stronger solver strategies and stiff‑aware integrators without changing the surface API.

## Goals / Non‑Goals

### Goals
- Add JFNK and advanced preconditioning for large sparse systems.
- Add stiff‑stable integrators (TR‑BDF2, Rosenbrock‑W/SDIRK).
- Enable periodic steady‑state methods (shooting / harmonic balance).
- Keep deterministic solver selection and clear telemetry.
- Extend YAML configuration in a backward‑compatible way.

### Non‑Goals
- GPU acceleration.
- Full SPICE model parity.
- New device models beyond current v1 set.

## Decisions

### 1) Solver Order Separation
Keep **primary** and **fallback** solver orders distinct. The primary order is attempted first, and fallback is used only after failure. Both orders are deterministic.

### 2) JFNK Path
Add Jacobian‑vector products (finite‑difference J·v) to allow Newton–Krylov without assembling the full Jacobian for large systems.

### 3) Preconditioners
Support ILUT and AMG (when available) as optional preconditioners, with conservative defaults (Jacobi / ILU0) for small systems.

### 4) SPD‑Safe CG
CG is only valid for SPD systems. Enforce SPD checks (symmetry + Cholesky test) and disable CG if the matrix fails.

### 5) Stiff Integrators
Add TR‑BDF2 and Rosenbrock‑W/SDIRK for stiff switching transients, and integrate stiffness detection with adaptive timestep + order control.

### 6) Periodic Steady‑State
Add shooting and harmonic balance (HB) as optional post‑transient solvers for periodic switching converters.

## Risks / Trade‑offs
- JFNK requires careful finite‑difference step sizing and good preconditioners.
- AMG introduces an additional dependency (must be optional).
- HB/shooting adds complexity; keep it opt‑in.

## Migration Plan
1. Add primary/fallback order separation + SPD checks for CG.
2. Add JFNK path and telemetry.
3. Add TR‑BDF2 integrator, then Rosenbrock‑W.
4. Add periodic steady‑state solvers.
5. Extend YAML config + Python/CLI exposure.
