## Why

Real switched-mode converters (flyback, forward, push-pull, planar transformers) depend critically on **core saturation** behavior of their magnetic components. v2 currently has only a **linear** transformer (`add_transformer`, V2) and **linear** inductor (`add_inductor`) — both assume constant L regardless of current. This means:

1. **Flyback designs cannot model peak-current limit** (real flybacks rely on core saturation as a hard current limit).
2. **Push-pull and forward converters have no DC flux walk** detection.
3. **Multi-winding transformers** (3-winding flyback w/ aux for bias supply, multi-output converters) cannot be expressed — `add_transformer` only handles 2 windings.
4. **Core-loss estimation** (Steinmetz / iGSE) is not available — thermal design is blocked.

v1 has a `hysteresis_inductor_device` with Jiles-Atherton model. v2 should ship the same capability **architecturally as a nonlinear branch** (analogous to V13 MOSFET) so the existing Newton refresh infrastructure handles it.

## What Changes

- **ADD** `models::SaturableInductor` — a nonlinear branch device that exposes `L(i) = L_0 · g(i, I_sat)`. Smooth Jiles-Atherton-flavoured saturation curve (anhysteretic for V0; full hysteresis is a follow-up).
- **ADD** `models::MultiWindingTransformer` — extends V2's 2-winding model to N windings (N=2..6) with mutual coupling matrix `M_ij = k_ij · √(L_i · L_j)` and optional **shared saturable core** (a single saturable magnetic that ties all windings via the magnetising current).
- **ADD** `models::CoreLossEstimator` — Steinmetz B^β·f^α model with optional iGSE refinement, runs as a post-process pass on the transient result.
- **ADD** YAML schema `type: saturable_inductor`, `type: multi_winding_transformer` with `core` block, `type: core_loss_estimate` analysis post-process.
- **ADD** CircuitBuilder methods `add_saturable_inductor(...)`, `add_multi_winding_transformer(...)`.
- **ADD** Python bindings + `add_*` wrappers.
- **ADD** Newton refresh `refresh_saturable_magnetics` that stamps the per-iteration `L(i)·di/dt` contribution. Combined into the existing `make_combined_diode_mosfet_refresh()` family.
- **ADD** Showcase YAMLs: (a) flyback w/ saturating primary, (b) 3-winding flyback w/ aux bias.
- **ADD** C++ unit tests for saturation curve, multi-winding coupling matrix, Newton convergence in a flyback context.

## Impact

- **Affected specs:** new `pulsim-v2-saturable-magnetics` capability.
- **Affected code:**
  - `core/include/pulsim/v2/models/saturable_inductor.hpp` (new)
  - `core/include/pulsim/v2/models/multi_winding_transformer.hpp` (new)
  - `core/include/pulsim/v2/models/core_loss.hpp` (new)
  - `core/include/pulsim/v2/pwl/device_pool.hpp` (extend Entry variant, StoredKind, getters)
  - `core/include/pulsim/v2/pwl/nonlinear_refresh_*.hpp` (new refresh fn)
  - `core/include/pulsim/v2/builder/circuit_builder.hpp` (new add_*)
  - `core/include/pulsim/v2/yaml/loader.hpp` (new YAML types)
  - `python/bindings_v2_kernel.cpp` (Python bindings)
  - `core/tests/v2/layer2/test_saturable_inductor.cpp` (new)
  - `core/tests/v2/layer2/test_multi_winding_transformer.cpp` (new)
  - `core/tests/v2/showcases/test_flyback_saturated.cpp` (new)
  - `examples/v2/flyback_saturated.yaml` (new)
  - `examples/v2/flyback_3winding.yaml` (new)
- **Risk:** Newton convergence on saturating magnetics is non-trivial — same lessons as V14 MOSFET (gradient-aware step constraints, ramped excitation if needed). Mitigation: re-use ramped pulse (V14 `rise_time`) and continuation framework already in v2.
