# Robustness Policy

> Status: shipped — `RobustnessProfile` primitive is the canonical
> source of truth. Removing the legacy `apply_robust_*_defaults`
> duplicates from `bindings.cpp` and `python/pulsim/__init__.py` is
> the natural follow-up.

Pulsim's robustness knobs (Newton tolerance / max iterations, linear-
solver fallback policy, integrator step retries, Gmin recovery, source
stepping, pseudo-transient continuation, DAE-fallback) used to be
configured in three places with subtly different defaults: the C++
`Simulator` constructor, the pybind11 bindings, and a thin Python
wrapper layer. This change consolidates them into a single typed
struct.

## TL;DR

```cpp
#include "pulsim/v1/robustness_profile.hpp"
using namespace pulsim::v1;

const auto p = RobustnessProfile::for_tier(RobustnessTier::Strict);
// p.newton_max_iters == 80, p.newton_use_homotopy == true,
// p.enable_source_stepping == true, ...
```

```python
import pulsim
# YAML schema (parser dispatch landing alongside the Circuit-variant
# integration): simulation: { robustness: aggressive | standard | strict }
```

## Three tiers

| Knob | Aggressive | Standard | Strict |
|---|---|---|---|
| `newton_max_iters` | 15 | 30 | 80 |
| `newton_tol_residual` | 1e-6 | 1e-8 | 1e-10 |
| `newton_tol_step` | 1e-7 | 1e-9 | 1e-11 |
| `newton_use_homotopy` | ✗ | ✗ | ✓ |
| `linear_solver_allow_fallback` | ✗ | ✓ | ✓ |
| `linear_solver_max_retries` | 1 | 3 | 8 |
| `integrator_max_step_retries` | 2 | 6 | 16 |
| `integrator_enable_lte` | ✗ | ✓ | ✓ |
| `gmin_initial` | 0.0 | 0.0 | 1e-12 |
| `gmin_max` | 1e-9 | 1e-3 | 1e-2 |
| `enable_source_stepping` | ✗ | ✗ | ✓ |
| `enable_pseudo_transient` | ✗ | ✗ | ✓ |
| `allow_dae_fallback` | ✗ | ✓ | ✓ |
| `allow_aggressive_dt_backoff` | ✗ | ✓ | ✓ |

### When to pick each tier

- **Aggressive** (≈ 2× faster than Standard at the cost of fragility):
  benchmarks where you've already tuned the circuit, throughput
  contests, parameter sweeps that don't survive any sample failure.
  No safety nets — DAE fallback, Newton homotopy, Gmin escalation
  are all disabled.
- **Standard** (production default): balanced. The Pulsim defaults
  used everywhere unless overridden.
- **Strict** (safest, slowest): new circuits, convergence
  debugging, AD validation runs, regression suites where any silent
  divergence is a bug. Newton homotopy + Gmin stepping + source
  stepping + pseudo-transient continuation all on.

## Parsing from YAML / CLI

```cpp
const auto tier = parse_robustness_tier("strict");   // throws on bad input
```

The parser canonicalizes lowercase only — `"STRICT"` and `"Strict"`
are explicitly rejected to keep the YAML schema unambiguous.

## Validation gates

| Gate | Test |
|---|---|
| **G.1** Tiers produce distinct knob bundles | `test_robustness_profile.cpp::for_tier` |
| **G.2** Round-trip via `to_string` + `parse_robustness_tier` | same file |
| **G.3** Strict input handling | `parse rejects unknown strings` |

## Limitations / follow-ups

- **Removing legacy duplicates** (Phase 2 of `refactor-unify-
  robustness-policy`): the existing `apply_robust_*_defaults` helpers
  in `python/bindings.cpp` and the `_tune_*_for_robust` Python
  wrappers still exist — deleting them requires updating every call
  site to consume `RobustnessProfile` instead. Tracked as a
  hygiene-only follow-up to keep the diff for this change small.
- **`SimulationOptions::apply_robustness(profile)`**: the mutator
  that maps a `RobustnessProfile` onto the existing
  `SimulationOptions` fields. Lands alongside the call-site updates
  in the legacy-deletion follow-up.
- **YAML `simulation.robustness:` parser**: deferred. The struct +
  parser are final today; YAML wiring rides with the Circuit-variant
  integration parser dispatch.
- **Per-circuit auto-tier**: `RobustnessProfile::for_circuit(ckt,
  tier)` that examines a circuit's device mix (PWL switches /
  Behavioral devices / saturable magnetics) and biases knobs on top
  of the chosen tier. Today the tier is the only input; per-circuit
  bias is the natural next refinement.

## See also

- [`convergence-tuning-guide.md`](convergence-tuning-guide.md) — what
  individual knobs do, when to tune them by hand.
- [`linear-solver-cache.md`](linear-solver-cache.md) — the linear-
  solver-fallback knob is one of the bundle items.
