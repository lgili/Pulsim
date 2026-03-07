## 0. Baseline and guardrails
- [ ] 0.1 Freeze runtime benchmark and telemetry baselines before extraction.
- [ ] 0.2 Define module-level regression KPIs (`determinism`, `allocation`, `runtime_p95`).
- [ ] 0.3 Add CI gate for module-level KPI regression.

## 1. Module contract scaffolding
- [x] 1.1 Introduce runtime module interface and lifecycle hook contracts.
- [x] 1.2 Introduce module capability/dependency manifest types.
- [x] 1.3 Implement deterministic module dependency resolver with typed diagnostics.
- [x] 1.4 Add architecture tests validating module ordering and failure diagnostics.

## 2. Non-breaking extraction adapters
- [x] 2.1 Extract event/topology logic into `events_topology_module` adapters.
- [x] 2.2 Extract mixed-domain control scheduler logic into `control_mixed_domain_module`.
- [x] 2.3 Extract loss tracking path into `loss_module`.
- [x] 2.4 Extract thermal tracking path into `thermal_module`.
- [x] 2.5 Extract channel/metadata emission into `telemetry_channels_module`.
- [x] 2.6 Keep external YAML/Python/channel behavior unchanged.

## 3. Orchestrator simplification
- [x] 3.1 Refactor runtime orchestrator to policy-only flow over module hooks.
- [ ] 3.2 Remove duplicated cross-cutting code from orchestrator files after extraction.
- [x] 3.3 Add boundary checks preventing direct module-internal coupling.

## 4. Quality and performance gates
- [x] 4.1 Add deterministic replay tests for module output order.
- [ ] 4.2 Add hot-path allocation tests for stable-topology stepping.
- [ ] 4.3 Add channel/summary consistency tests under modular execution.
- [ ] 4.4 Run benchmark/parity/stress and compare against frozen baseline.

## 5. Documentation and contributor onboarding
- [x] 5.1 Document module architecture and extension points for contributors.
- [x] 5.2 Document module dependency rules and anti-patterns.
- [x] 5.3 Update contributor guide with module-scoped testing workflow.
