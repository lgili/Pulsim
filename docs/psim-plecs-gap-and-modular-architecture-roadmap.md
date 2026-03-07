# PSIM/PLECS Gap Analysis and Modular Architecture Roadmap

Snapshot date: 2026-03-07

## 1. Executive Summary

Pulsim is already strong in:
- switched-converter transient simulation (fixed/variable step),
- mixed-domain control + PWM integration,
- electrothermal observability (canonical `T(...)` and `P*` channels with metadata),
- scriptable backend and CI-centric validation.

Compared with PSIM and PLECS, the largest gaps are not only numerical models, but productized workflows:
- built-in advanced analyses (small-signal, Bode, periodic operating point as first-class UX flows),
- integrated code-generation/HIL ecosystems,
- broader multi-domain package depth (especially mechanical + real-time deployment),
- stronger module boundaries for independent evolution by community contributors.

## 2. External Benchmark References (Official Sources)

PSIM:
- [Altair PSIM product page](https://altair.com/psim)
- [Altair PSIM applications page](https://altair.com/psim-applications/)

PLECS:
- [PLECS product overview](https://www.plexim.com/products/plecs)
- [PLECS Blockset](https://www.plexim.com/products/plecs/plecs_blockset)
- [PLECS Standalone](https://www.plexim.com/products/plecs/plecs_standalone)
- [PLECS Analysis Tools](https://www.plexim.com/products/plecs/analysis_tools)
- [PLECS Simulation Scripts](https://www.plexim.com/products/plecs/simulation_scripts)
- [PLECS Coder](https://www.plexim.com/products/plecs_coder)
- [RT Box](https://www.plexim.com/products/rt_box)

Pulsim baseline:
- [Supported Components Catalog](supported-components-catalog.md)
- [Backend Architecture](backend-architecture.md)
- [Electrothermal Workflow](electrothermal-workflow.md)
- [Frontend Control and Signals Contract](frontend-control-signals.md)

## 3. Capability Matrix (PSIM vs PLECS vs Pulsim)

Legend:
- `Strong`: production-grade and first-class in product workflow
- `Partial`: available in backend/runtime, but limited breadth or packaging
- `Gap`: missing or not yet productized

| Capability | PSIM | PLECS | Pulsim (2026-03-07) | Gap vs Leaders |
| --- | --- | --- | --- | --- |
| Power converter transient simulation | Strong | Strong | Strong | Low |
| Switched-system robustness focus | Strong | Strong | Strong | Low |
| Built-in control blocks + PWM | Strong | Strong | Strong (`pi/pid/pwm/state_machine/...`) | Low |
| Component catalog breadth for power electronics | Strong | Strong | Strong (power + control + instrumentation + surrogates) | Low |
| Thermal + loss coupling workflow | Strong | Strong | Strong (scalar + datasheet surfaces, `single_rc/foster/cauer`, shared sink) | Low |
| Manufacturer-ready loss model workflow in UX | Strong | Strong (ready-to-use thermal/loss models) | Partial (backend-ready, UX tooling still basic) | Medium |
| Small-signal / frequency-response workflow | Present in suites | Strong (open-loop/closed-loop tools, Bode) | Partial (`shooting`/`harmonic_balance`, no first-class small-signal UX flow) | High |
| Steady-state operating point workflow for switching converters | Present in suites | Strong | Partial (periodic methods exist, not yet productized as dedicated analysis module) | Medium/High |
| Simulink-native integration | Co-simulation links | Strong (Blockset) | Gap (no direct Simulink integration contract) | High |
| Script automation ecosystem | Strong | Strong (MATLAB/Octave/XML-RPC/Python paths) | Strong (Python-first) | Low |
| Embedded code generation from control diagram | Strong (SimCoder) | Strong (PLECS Coder) | Gap | High |
| HIL/real-time product line | Via broader ecosystem | Strong (RT Box) | Gap | High |
| Multi-domain depth (electrical + magnetic + thermal + mechanical) | Strong ecosystem | Strong | Partial (electrical + thermal + some magnetics; no full mechanics domain) | High |
| Design verification tools (Monte Carlo, sensitivity, fault) | Strong (explicitly productized) | Present via scripts/tools | Partial (possible via scripting/tests, no dedicated analysis module UX/API) | Medium/High |

## 4. What Pulsim Already Has (Important)

Backend capabilities already aligned with professional workflows:
- Deterministic event-driven transient core with robust fallback telemetry.
- Canonical electrothermal contract:
  - `T(<component>)`
  - `Pcond/Psw_on/Psw_off/Prr/Ploss(<component>)`
  - strict consistency against summaries.
- Mixed-domain control scheduling modes (`auto/continuous/discrete`).
- High component coverage for converter/control use-cases.
- Extensibility primitives:
  - extension registry metadata/contracts,
  - transient service registry,
  - virtual component architecture.

Main blocker now is architectural packaging: too much policy and integration logic still concentrated in large orchestrator files, which increases coupling and slows isolated evolution.

## 5. Architectural Target (Modular Core)

Design goal:
- each functional concern evolves independently,
- new modules plug in without editing central orchestrator logic,
- module-level tests and benchmarks prevent cross-module regressions.

### 5.1 Proposed Module Families

1. `electrical-solve`:
- equation assembly, nonlinear/linear solve orchestration adapters.

2. `events-topology`:
- switch boundary calendar, event refinement, topology signature updates.

3. `control-mixed-domain`:
- virtual control block execution, scheduler policy, sampled-data handling.

4. `losses`:
- conduction/switching/recovery accounting, datasheet surface evaluation.

5. `thermal`:
- thermal network integration, coupling, temperature scaling policies.

6. `analysis`:
- periodic, small-signal, frequency-response, future verification analyses.

7. `telemetry-channels`:
- channel registration, metadata ownership, reduction contracts, KPI hooks.

8. `adapters`:
- YAML parser binding, Python binding, future gRPC/GUI-facing contracts.

### 5.2 Runtime Hook Contract

Every module should implement deterministic hooks:
- `on_run_initialize`
- `on_step_attempt`
- `on_step_accepted`
- `on_sample_emit`
- `on_finalize`

And explicitly declare:
- required capabilities/dependencies,
- produced channels/telemetry,
- failure diagnostics namespace.

### 5.3 Ownership Rules

- Core orchestrator owns time integration flow only.
- Modules own their state and outputs.
- Cross-module interaction occurs only through typed contracts and immutable views.
- No module reads another module’s internals directly.

## 6. Recommended Refactor Phasing

### Phase A: Contract Extraction (no behavior change)
- Introduce runtime module interface + dependency resolver.
- Wrap existing loss/thermal/control/event logic into adapter modules.
- Keep existing API and outputs exactly unchanged.

### Phase B: Pipeline Segmentation
- Move sampling/output emission to dedicated telemetry-channels module.
- Move periodic analysis paths into `analysis` module package.
- Add module-level telemetry for CPU time, allocations, and misses.

### Phase C: Productization Gaps
- Add first-class small-signal/frequency analysis module.
- Add verification-analysis module (Monte Carlo/sensitivity/fault orchestration).
- Add optional codegen/HIL bridge contracts (backend-ready interfaces first).

### Phase D: Ecosystem Expansion
- Optional Simulink/FMI bridge adapter contract.
- Optional mechanics-domain package.
- Optional device-model package manager/import tooling.

## 7. Backend vs GUI Responsibility (for this roadmap)

Backend should own:
- all physics and numerical integration,
- analysis kernels and deterministic contracts,
- validation diagnostics and KPI metrics.

GUI should own:
- user experience for model input and analysis setup,
- visualization templates and workflow orchestration,
- import/export assistants.

GUI should not own:
- synthetic reconstruction of thermal/loss physics,
- heuristic derivation of backend analysis outputs.

## 8. Immediate Next Step

Implement `refactor-modular-runtime-kernel` (OpenSpec change) as a non-breaking internal refactor:
- extract module contracts,
- modularize execution pipeline,
- preserve all existing user-facing YAML/Python contracts while enabling independent module evolution.
