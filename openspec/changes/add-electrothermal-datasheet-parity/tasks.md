## 1. Schema and Contracts
- [x] 1.1 Extend YAML schema for datasheet loss characterization (`scalar` and `datasheet` modes) with strict validation.
- [ ] 1.2 Extend YAML schema for thermal network kinds (`single_rc`, `foster`, `cauer`) and optional shared coupling descriptors.
- [x] 1.3 Add deterministic diagnostics for dimensional mismatch, invalid ranges, and unsupported combinations.
- [x] 1.4 Preserve backward compatibility mappings for existing `loss` and `thermal` blocks.

## 2. Kernel Loss Engine
- [x] 2.1 Implement multidimensional interpolation path for `Eon/Eoff/Err` and temperature-dependent conduction models.
- [x] 2.2 Generalize switching-event detection to include forced `MOSFET`/`IGBT` transitions and native switching components.
- [x] 2.3 Ensure diode reverse-recovery loss accounting is consistent with event transitions and configured data.
- [ ] 2.4 Keep deterministic ordering and no unplanned hot-loop allocations after warm-up.

## 3. Kernel Thermal Engine
- [ ] 3.1 Implement thermal solver support for `single_rc`, `foster`, and `cauer` networks.
- [ ] 3.2 Implement optional shared thermal coupling for multiple components on common sink context.
- [ ] 3.3 Guarantee stable, deterministic thermal integration and bounded temperature-scaling behavior.

## 4. Runtime Outputs and Metadata
- [x] 4.1 Export canonical loss channels per component: `Pcond`, `Psw_on`, `Psw_off`, `Prr`, `Ploss`.
- [x] 4.2 Keep canonical thermal channels (`T(component)`) and align all channels to `result.time`.
- [x] 4.3 Extend virtual channel metadata with domain, source component, physical quantity, and unit for all new channels.
- [x] 4.4 Enforce exact consistency checks between channel reductions and summary telemetry surfaces.

## 5. Python Bindings
- [x] 5.1 Expose typed Python API for new loss characterization and thermal network structures.
- [x] 5.2 Expose runtime result channels/metadata without requiring name-based heuristics.
- [x] 5.3 Preserve backward-compatible summaries and existing procedural/class APIs.

## 6. Validation and Benchmarking
- [x] 6.1 Add closed-loop buck electrothermal regression validating PI+PWM control with non-zero semiconductor switching losses.
- [ ] 6.2 Add component-minimum thermal tests comparing simulated traces against theoretical/expected behavior.
- [x] 6.3 Add parser contract tests for strict and non-strict datasheet/thermal definitions.
- [ ] 6.4 Add electrothermal performance gates (runtime/memory/allocation counters) for rich-loss scenarios.

## 7. Documentation and GUI Boundary
- [ ] 7.1 Document full backend electrothermal contract (input schema, outputs, metadata, guarantees).
- [ ] 7.2 Document explicit GUI-only responsibilities (input UX, curve import UX, visualization layout).
- [ ] 7.3 Document forbidden GUI behaviors (no synthetic physics, no heuristic thermal reconstruction).
- [ ] 7.4 Publish migration guide from scalar-only loss setup to datasheet-grade modeling.

## 8. Quality Gate
- [x] 8.1 Run `openspec validate add-electrothermal-datasheet-parity --strict`.
- [x] 8.2 Execute targeted test suites (`python/tests`, benchmark electrothermal matrix, runtime regression).
- [ ] 8.3 Record acceptance evidence and unresolved issues in change notes before implementation PR.
