# 9. Architecture Walkthrough

!!! info "Status: outline / next iteration"
    Source material: `docs/internals/README.md` plus the
    per-layer docs `docs/internals/layer{0..9}-*.md`. This
    chapter is the bridge — it summarises the layer stack
    at a level a reader who's been through chapters 1-8 can
    consume in 10 minutes.

The 10-layer codebase, top-to-bottom. Where each algorithm
from chapters 2-7 actually lives, what depends on what, and
where to start reading if you want to extend Pulsim.

## Planned sections

1. **The layer model**: why we organise the kernel into 10
   layers (Layer 0 = numeric primitives → Layer 9 = Python
   `simulate()` facade). The strict acyclic dependency rule
   that lets each layer be tested independently.

2. **Layer-by-layer summary** (one paragraph each):
   - Layer 0 — numeric types, `Real`, `Index`, dense vector,
     sparse matrix, `DirectSolver` interface
   - Layer 1 — topology graph, switch-state-mask enumeration
   - Layer 2 — device models (resistor, capacitor, MOSFET,
     IGBT, diode) + automatic differentiation
   - Layer 3 — stamping pipeline (device → MNA matrix
     contribution)
   - Layer 4 — PWL state-space cache (chapter 4) + trapezoidal
     companion + DC operating point + Newton refinement
   - Layer 5 — solver + event detection + run_transient
   - Layer 6 — `CircuitBuilder` (user-facing C++ API)
   - Layer 7 — Python bindings (pybind11)
   - Layer 8 — YAML loader
   - Layer 9 — Python `simulate()` ergonomic facade

3. **Dependency graph**: visualised as a DAG. Each layer
   depends on all layers below; no cycles.

4. **The "where do I add X?" cheat sheet**:
   - new device kind (e.g. JFET) → Layer 2 + Layer 3
   - new ODE discretisation (e.g. BDF2) → Layer 4
   - new event-detection algorithm → Layer 5
   - new sparse-LU algorithm (e.g. BTF, COLAMD) → Layer 0
     + chapter 6's `PulsimSparseLuSolver`

5. **Test-suite mapping**: each layer has its own test binary
   (`pulsim_core_layer{0..9}_tests`) under `core/tests/layer*/`.
   17,275 assertions total at v1.3.0. The test-coverage rule:
   any new feature lands with its layer's test binary updated
   in the same PR.

## Planned figures

- **Fig 9.1** — Layer stack diagram (Layer 0 at the bottom,
  Layer 9 at the top). Each layer shaded by complexity (# of
  source files, # of assertions). The chapters 1-7 algorithms
  annotated next to the layers they live in.
- **Fig 9.2** — Cross-layer dependency DAG. Mermaid graph
  showing the strict acyclic structure.
- **Fig 9.3** — Test-binary execution time matrix (rows =
  layer, columns = configuration: Debug / Release / Sanitizers).
  Heatmap.

## Cross-references

- [Layer-by-Layer Internals (the README)](../internals/README.md)
  is the canonical entry point. This chapter is the executive
  summary; the README is the detail.
- [Build System](../build-system.md) explains how the layers
  compose at link time (one shared `pulsim::core` interface
  library, header-only kernel).
