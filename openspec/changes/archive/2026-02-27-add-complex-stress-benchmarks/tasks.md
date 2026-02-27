## 1. Specification & Benchmarks
- [x] 1.1 Create `ll11_llc_resonant_converter.yaml` to stress highly resonant non-linear switching topologies.
- [x] 1.2 Create `ll12_pfc_boost_continuous.yaml` to stress continuous conduction mode with high PWM resolution bridging.
- [x] 1.3 Create `ll13_active_clamp_forward.yaml` to stress secondary-side switching and transformer magnetizing behavior.
- [x] 1.4 Register the new circuits in `benchmarks/local_limit/benchmarks_local_limit.yaml` under difficulties 7-8.

## 2. Validation
- [x] 2.1 Run `make benchmark-local-limit MAX_RUNTIME=60` and verify the backend can solve the circuits without crashing.
- [x] 2.2 If the backend chokes on the new circuits, debug solver profiles or initialization steps in `core/src/v1/simulation.cpp` to enhance automatic fallback convergence.
- [x] 2.3 Rerun `make test` to ensure no basic sanity checks are broken by backend tweaks.
