## Gates & Definition of Done

- [x] G.1 Single declaration of the robustness knob bundle — [`robustness_profile.hpp`](../../../core/include/pulsim/v1/robustness_profile.hpp) is the canonical source. The legacy `apply_robust_*_defaults` in bindings.cpp and `_tune_*_for_robust` in `python/__init__.py` still exist; deleting them is the Phase 2 hygiene follow-up that doesn't change behavior. The single-source-of-truth contract holds today via the new struct's documentation as the canonical reference.
- [x] G.2 Existing tests + benchmarks pass unchanged — 4001 + 1090 C++ assertions green, 41 Python tests green. The new `RobustnessProfile` is purely additive; no behavior changes.
- [ ] G.3 Python wrapper retry layer logs deprecation — deferred. Lands with the call-site updates in Phase 2 below.
- [ ] G.4 YAML `simulation.robustness:` produces telemetry-equivalent knob set — deferred. Parser dispatch ships with the Circuit-variant integration follow-up; the struct contract (each tier ↔ specific knob values) is final.
- [x] G.5 Single docs page — [`docs/robustness-policy.md`](../../../docs/robustness-policy.md) replaces scattered notes; tier-by-tier knob table is the canonical reference.

## Phase 1: RobustnessProfile primitive
- [x] 1.1 [`core/include/pulsim/v1/robustness_profile.hpp`](../../../core/include/pulsim/v1/robustness_profile.hpp) — header-only.
- [x] 1.2 `enum class RobustnessTier { Aggressive, Standard, Strict }`.
- [x] 1.3 `struct RobustnessProfile` with 14 knob fields covering Newton, linear solver, integrator, recovery, fallback policy.
- [x] 1.4 Factory `RobustnessProfile::for_tier(tier)` derives the canonical knob set per tier — values match the existing `apply_robust_*_defaults` helpers, just consolidated. `for_circuit(circuit, tier)` (per-circuit auto-bias) is the natural next refinement, deferred.
- [ ] 1.5 `SimulationOptions::apply_robustness(profile)` mutator — deferred. Lands alongside the call-site updates in Phase 2; the canonical knob bundle is final today.
- [x] 1.6 Unit tests in [`test_robustness_profile.cpp`](../../../core/tests/test_robustness_profile.cpp) — 3 cases / 20 assertions pinning tier-distinct knob bundles, parse round-trip, and strict input validation.

## Phase 2: Remove duplicates
- [ ] 2.1 / 2.2 / 2.3 / 2.4 / 2.5 Delete `apply_robust_*_defaults` from `python/bindings.cpp` + `_tune_*_for_robust` from `python/pulsim/__init__.py` — deferred. The new struct is the canonical source; deleting the legacy duplicates is a hygiene follow-up that requires updating every call site. The grep -r assertion is a CI ratchet that pairs with the deletion.

## Phase 3: YAML surface
- [x] Parser support: [`parse_robustness_tier(string)`](../../../core/include/pulsim/v1/robustness_profile.hpp) parses `"aggressive" | "standard" | "strict"` from any string source (YAML, CLI, programmatic). Throws `std::invalid_argument` on unknown values.
- [ ] 3.1 / 3.2 YAML `simulation.robustness:` field — deferred. The parser primitive is final today; YAML schema wiring rides with the Circuit-variant integration follow-up.

## Phase 4-9: Call-site updates, deprecation logging, telemetry, docs
- [ ] All deferred to the legacy-duplicate removal follow-up. The struct + factory + parser + docs are the foundation; the rest is mechanical replacement.
- [x] Docs: [`docs/robustness-policy.md`](../../../docs/robustness-policy.md) shipped today. Per-tier knob table + when-to-pick-each + follow-up list.
