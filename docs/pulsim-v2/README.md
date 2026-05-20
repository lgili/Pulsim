# Pulsim v2 — clean-slate kernel

`pulsim::v2` is a parallel-namespace kernel rebuild started in May 2026
to fix seven structural problems in `pulsim::v1` that incremental
refactor could not address economically. See
[../architecture-review-v1.md](../architecture-review-v1.md) for the
detailed diagnosis.

## Layered architecture

```
┌──────────────────────────────────────────────────────────────────┐
│ Layer 6: Frontend  (Python bindings, YAML loader, schematic in)  │
├──────────────────────────────────────────────────────────────────┤
│ Layer 5: Solver  (per-step segment dispatch, Newton fallback)    │
├──────────────────────────────────────────────────────────────────┤
│ Layer 4: PWL State-Space Cache  (the PLECS-killer layer)         │
├──────────────────────────────────────────────────────────────────┤
│ Layer 3: Stamping  (one generic stamper, AD-driven)              │
├──────────────────────────────────────────────────────────────────┤
│ Layer 2: Device Models  (math only, single source of truth, AD)  │
├──────────────────────────────────────────────────────────────────┤
│ Layer 1: Topology  (graph + switch combinatorics + enumeration)  │
├──────────────────────────────────────────────────────────────────┤
│ Layer 0: Numeric Primitives + Sparse Linear Algebra              │
└──────────────────────────────────────────────────────────────────┘
```

Each layer:
- Lives in one subfolder under `core/include/pulsim/v2/`.
- Has its own test binary `pulsim_v2_layerN_tests`.
- Depends ONLY on layers strictly below (compile-time enforced).
- Has a small, clean public-API surface.
- Can be replaced wholesale without touching anything else.

## Per-layer status and design docs

| Layer | Subfolder              | OpenSpec change-id                          | Status  | Design doc                                       |
|-------|------------------------|---------------------------------------------|---------|--------------------------------------------------|
| 0     | `numeric/` + `sparse/` | `bootstrap-pulsim-v2-kernel`                | ✅      | [layer0-numeric-and-sparse.md](layer0-numeric-and-sparse.md) |
| 1     | `topology/`            | `pulsim-v2-topology-and-switch-enumeration` | pending | tbd                                              |
| 2     | `models/`              | `pulsim-v2-device-models-ad-driven`         | pending | tbd                                              |
| 3     | `stamping/`            | `pulsim-v2-generic-stamping-pipeline`       | pending | tbd                                              |
| 4     | `pwl_state_space/`     | `pulsim-v2-pwl-state-space-cache`           | pending | tbd                                              |
| 5     | `solver/`              | `pulsim-v2-solver-and-events`               | pending | tbd                                              |
| 6     | `runtime/` + `frontend/` | `pulsim-v2-circuit-builder-api` + 2 more  | pending | tbd                                              |

## Why parallel-namespace

v1 stays in `pulsim::v1`. v2 lives in `pulsim::v2`. Zero coupling. The
two namespaces coexist in the same repository and the same shared
library binary. Users opt in:

```python
import pulsim
ckt = pulsim.Circuit()        # v1 default
ckt2 = pulsim.v2.Circuit()    # explicit v2 (when Layer 6 lands)
```

When Layer 6b ships feature parity on all `[v1]` tests, Python adds a
global toggle `pulsim.use_v2 = True` so existing scripts can flip
without code changes. v1 enters maintenance mode once ≥ 90 % of real
workloads run on v2.

No flag-day cutover. No "v1 broken for a week while we migrate." v1 is
NEVER touched during the v2 build-out.

## Build + test

```bash
# Build only Layer 0 (today)
cmake --build build --target pulsim_v2_layer0_tests

# Run Layer 0 tests
build/core/pulsim_v2_layer0_tests
```

Each future layer adds its own test binary. The full v2 test surface
is the union of `pulsim_v2_layerN_tests` for N = 0..6.

## See also

- [`../architecture-review-v1.md`](../architecture-review-v1.md) — why
  v2 exists (the seven structural problems in v1).
- [`layer0-numeric-and-sparse.md`](layer0-numeric-and-sparse.md) —
  Layer 0 design decisions.
- `openspec/changes/archive/2026-05-20-bootstrap-pulsim-v2-kernel/` —
  the OpenSpec proposal that landed Layer 0 (after archive).
