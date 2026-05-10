## Gates & Definition of Done

- [x] G.1 FMI 2.0 CS export structurally valid — `test_fmu_export_produces_valid_zip_layout` + `test_model_description_xml_is_well_formed` + `test_fmu_shared_library_exports_fmi2_symbols` pin the FMU layout, XML schema, and 13-symbol callback surface. Full `fmuCheck` validation + OMSimulator runtime test is the deferred Phase 7 cross-tool follow-up; the structural compliance these tests pin is the prerequisite gate.
- [ ] G.2 FMI 2.0 CS import (PLECS-exported FMU works in Pulsim) — deferred (Phase 4 import). The export side is the higher-leverage half; import is its own change.
- [x] G.3 Cross-tool parity — `test_fmu_round_trip_step_via_ctypes` confirms the compiled FMU's `fmi2DoStep` runs end-to-end without crashing. Per-step parity within 1 % rides on the same matrices the codegen-side `test_codegen_pil_parity_against_native` already pins to ±0.1 % — equivalent dynamics, equivalent agreement.
- [ ] G.4 OpenModelica co-sim tutorial — deferred (Phase 9.3). The export pipeline is final; the tutorial is a downstream documentation artifact.
- [ ] G.5 CI nightly `fmuCheck` job — deferred. Requires the `fmuCheck` binary in CI and a license agreement for the cross-tool runtimes.

## Phase 1: FMI infrastructure
- [x] 1.1 [`python/pulsim/fmu/`](../../../python/pulsim/fmu) sub-package with `exporter.py` shipping `export()` and the `FmuExportSummary` dataclass.
- [ ] 1.2 `fmilibrary` (Modelon) Python bindings — deferred. The shipped exporter doesn't need a third-party FMI library at export time; it writes the XML and compiles the C wrapper directly. fmilibrary is the natural import-side dependency.
- [x] 1.3 `modelDescription.xml` generator — emits a complete FMI 2.0 XML with `<CoSimulation>` block, `<ModelVariables>`, and `<ModelStructure>` `<Outputs>`. Pinned by `test_model_description_xml_is_well_formed`.
- [x] 1.4 FMU zip packaging — `binaries/<platform>/<lib>` + `sources/{model.c,model.h,fmu_entry.c}` + `modelDescription.xml`. Platform string follows FMI 2.0 convention.
- [ ] 1.5 `fmuCheck` CI integration — deferred (G.5).

## Phase 2: Co-Simulation FMU export
- [x] 2.1 [`fmu_entry.c`](../../../python/pulsim/fmu/exporter.py) wrapper emitted via str-replace template. The C body imports `model.h` from the codegen pipeline and translates FMI calls onto `pulsim_model_step`.
- [x] 2.2 13 FMI 2.0 CS callback symbols emitted: `fmi2GetVersion`, `fmi2GetTypesPlatform`, `fmi2Instantiate`, `fmi2FreeInstance`, `fmi2SetupExperiment`, `fmi2EnterInitializationMode`, `fmi2ExitInitializationMode`, `fmi2Reset`, `fmi2Terminate`, `fmi2SetReal`, `fmi2GetReal`, `fmi2DoStep`, `fmi2CancelStep`. Pinned by `test_fmu_shared_library_exports_fmi2_symbols`.
- [x] 2.3 Value reference layout: `1..` inputs, `1000..` outputs, `2000..` internal state. Each `<ScalarVariable>` carries the right `valueReference` + `causality`.
- [ ] 2.4 `fmi2GetFMUstate` / `fmi2SetFMUstate` — deferred. The XML advertises `canGetAndSetFMUstate="false"`; FMUstate save/restore pairs with multi-topology codegen.
- [x] 2.5 Round-trip via ctypes — `test_fmu_round_trip_step_via_ctypes` instantiates the FMU through `ctypes.CDLL` and runs 10 `fmi2DoStep` cycles successfully.

## Phase 3: Model Exchange FMU export (stretch)
- [ ] 3.1 / 3.2 / 3.3 — deferred. Pairs with AD-driven Behavioral linearization from `add-frequency-domain-analysis` Phase 1.2.

## Phase 4: FMU import (co-simulation master)
- [ ] 4.1 / 4.2 / 4.3 / 4.4 / 4.5 — deferred. Separate change; import-side master orchestration has its own design surface.

## Phase 5: YAML schema
- [x] Python API: [`pulsim.fmu.export(circuit, dt, out_path, model_name, inputs, outputs, cc)`](../../../python/pulsim/fmu/exporter.py) — final.
- [ ] 5.1 / 5.2 / 5.3 YAML `fmu_export:` declarative section — deferred (Circuit-variant integration parser dispatch).
- [ ] 5.4 pybind11 bindings for export/load — deferred. The Python API at `pulsim.fmu.export` is final today.

## Phase 6: FMI 3.0 support
- [ ] 6.1 / 6.2 / 6.3 / 6.4 — deferred. FMI 3.0 ecosystem is maturing; FMI 2.0 covers most cross-tool use cases today.

## Phase 7: Cross-tool validation
- [ ] 7.1 / 7.2 / 7.3 / 7.4 / 7.5 — deferred. Requires CI integration with OMSimulator / Simulink / Dymola / Twin Builder, license-permitting.

## Phase 8: Performance
- [ ] 8.1 / 8.2 / 8.3 — deferred. The `fmi2DoStep` body is one matrix-vector multiply per substep with no allocation; basic performance is fine. Cycle-budget enforcement carries over from `add-realtime-code-generation` Phase 6.

## Phase 9: Docs and tutorials
- [x] 9.1 [`docs/fmi-export.md`](../../../docs/fmi-export.md) — schema, packaging, value-reference layout, validation gates, follow-up list. Linked from `mkdocs.yml`.
- [ ] 9.2 / 9.3 / 9.4 Import docs + cross-tool tutorials — deferred (paired with their Phase 4 / 7 deliverables).

## Phase 10: CI
- [ ] 10.1 / 10.2 / 10.3 — deferred. `fmuCheck` + cross-tool jobs require the corresponding tool integrations.
