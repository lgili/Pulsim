Every numbered section is one or more PRs and ends with the
verification set: `ruff check`, `pytest python/tests`, `ctest`,
`openspec validate --all --strict`, the stub check, and
`core/tests/perf/test_zero_alloc.cpp`. "Goldens" = the fixed-engine
checkpoint hashes and the differential runner of
`restore-reference-benchmark-harness`; "matrix" = the device × engine
test of `add-engine-conformance-matrix`. Throttle builds on the shared
machine (`-j2`, `nice`).

## 0. Preconditions
- [ ] 0.1 `add-engine-conformance-matrix` and `restore-reference-benchmark-harness` merged; record the golden hashes and the trbdf2 step counters (`n_accept`, `n_reject`, `n_solves`) of the current `main`
- [ ] 0.2 Record the baseline: extension build time, ns/step of audit rows 10–13 (Shockley buck, MOSFET level 1), `grep -rn "StoredKind::" core/include python/*.cpp | wc -l` (123)

## 1. Fold the five history classes (no behaviour change)
- [ ] 1.1 `pwl/stateful_histories.hpp`: a tuple of the five history objects with `for_each(f)`, `init`, `seed_from_dc_op`, `commit(x, h, stage)`, `capture_gamma`, `snapshot_into`, `restore`, `to_flat`, `from_flat`
- [ ] 1.2 Replace the 56 named calls in `solver/run_transient.hpp` by the folds
- [ ] 1.3 Replace the 43 named calls in `solver/trbdf2_transient.hpp` by the folds
- [ ] 1.4 Oracle: goldens bit-identical; trbdf2 counters identical; `test_stateful_device_stamps_compose.py`, `test_snapshot_resume_stateful.py`, `test_trbdf2_stateful_with_diodes.py`, `test_trbdf2_stateful_devices.py` green
- [ ] 1.5 Docs: none (internal)

## 2. Kernel-built refresh
- [ ] 2.1 `pwl::build_device_refresh(graph, pool, histories, StepCtx&)`: the static diode / MOSFET / IGBT stamps (kernel-built since `add-engine-conformance-matrix`) plus one typed pass per history, in one function; both drivers call it
- [ ] 2.2 Delete the two wrapper chains (RT:471-600, TB:588-697) in favour of 2.1; keep `nl_refresh` as a deprecated override that replaces the base stamps only
- [ ] 2.3 Remove whatever `make_combined_diode_mosfet_refresh()` call sites remain in `python/bindings.cpp` (four on this branch); `enable_nonlinear_refresh=` emits a `DeprecationWarning`
- [ ] 2.4 C++ tests: the Phase 0 clamp test (0.594 V with no callback) still passes; every pair of stateful kinds in one circuit keeps each kind's single-kind result
- [ ] 2.5 Oracle: goldens bit-identical; matrix unchanged
- [ ] 2.6 Docs: CHANGELOG deprecation note; `docs/internals` section on the refresh

## 3. Flat device-state vector and snapshot
- [ ] 3.1 `device/device_state.hpp`: `DeviceState`, `DeviceStateLayout`, signature
- [ ] 3.2 The five histories and `HistoryState` store into blocks of one `DeviceState`; their arithmetic is untouched; rollback is one copy into a preallocated twin
- [ ] 3.3 `SolverSnapshot`: add `dev_state`, `mask`, `layout_signature`; the seven old fields become computed views; resume refuses a layout mismatch by block name
- [ ] 3.4 Pickle version 3; reader for the 10- and 11-field tuples that converts at resume time
- [ ] 3.5 Tests: resume pwl → trbdf2 and trbdf2 → pwl on a circuit with every stateful kind; a 2.0 pickle resumes; a snapshot of another circuit is refused by name
- [ ] 3.6 Oracle: goldens bit-identical; `test_snapshot_resume.py`, `test_snapshot_resume_stateful.py` unchanged; zero-alloc tests green
- [ ] 3.7 Docs: snapshot page, stubs, CHANGELOG

## 4. The interface, unused
- [ ] 4.1 `ad/ad_scalar.hpp`: `atan`, `atan2`, `pow`, `sin`, `cos`, `chain(x, value, dvalue_dx)`; unit tests against finite differences
- [ ] 4.2 `device/device.hpp`: `Device`, `ModeDevice`, `SelfCommutated`, `StaticDevice`, `Formula`, `StepCtx`, `Out`, the optional-member detectors
- [ ] 4.3 `device/stamp_fq.hpp` with `history_term` for the four formulas; unit tests: a linear capacitor and a linear inductor written as `Device` reproduce the companion stamps of `stamping/` entry by entry for trapezoidal, backward Euler and the BDF2 stage
- [ ] 4.4 `device/registry.hpp`: `TypeList`, `Builtin`, `DeviceSet`, `Instance`, `for_each_newton_kind`; `StoredKind` derived from the list with the existing values asserted
- [ ] 4.5 `static_assert`: a kind is never in both the legacy refresh and the registry's Newton list
- [ ] 4.6 Oracle: nothing uses it; goldens bit-identical; record the build time

## 5. Port the algebraic kinds and the charge-only kind
- [ ] 5.1 `FromCurrentModel<T>` adapter; port `ShockleyDiode`; delete its legacy stamp path
- [ ] 5.2 Port `IdealDiode`, `MosfetLevel1`, `IgbtLevel1` (one PR each)
- [ ] 5.3 Port `NonlinearCapacitor` as `q`-only; delete `nonlinear_capacitor_history.hpp` and its refresh
- [ ] 5.4 Per kind: finite-difference Jacobian test to 1e-12 of the operating range; monotonicity sweep
- [ ] 5.5 Oracle per PR: the kind's existing tests green; maximum deviation on the harness circuits containing the kind stated in the PR (budget 1e-10 relative), Newton iteration counts equal; goldens without the kind bit-identical; matrix unchanged
- [ ] 5.6 Measure ns/step on audit rows 10–13 before and after; not above before

## 6. Port the internal-charge kinds
- [ ] 6.1 `LauritzenDiode` with `q_M` as an internal unknown appended after the existing unknowns; delete `lauritzen_diode_history.hpp` and its refresh
- [ ] 6.2 IGBT tail with its charge as an internal unknown; delete `igbt_tail_history.hpp` and its refresh
- [ ] 6.3 Oracle: `test_trbdf2_stateful_devices.py`, `test_trbdf2_stateful_with_diodes.py`, the tail-energy and recovery tests within their existing tolerances on pwl and trbdf2; pwl and trbdf2 agree on the double-pulse energies to 0.1 %
- [ ] 6.4 Docs: note that `res.states` gains one column per such device; name-based accessors unchanged

## 7. Port the flux family and the PMSM
- [ ] 7.1 Saturable inductor (atan law), then `FluxTable` and `GappedCore` through `ad::chain`
- [ ] 7.2 Jiles–Atherton core: `State` + `accept`; the saturable transformer's magnetising branch
- [ ] 7.3 `PmsmMna` (three current unknowns, torque on the mechanical node)
- [ ] 7.4 Unit test: the stamped flux-row diagonal equals `−a·L(i)` for trapezoidal and the BDF2 stage (conductance form)
- [ ] 7.5 Oracle: `test_saturable_inductor_flux_closure.py`, the JA loop tests, the PMSM tests green; the strict xfail `test_engines_agree_below_saturation` keeps its status; delete `saturable_inductor_history.hpp`, `pmsm_mna_history.hpp` and their refreshes

## 8. Delete the legacy plumbing
- [ ] 8.1 Delete `NonlinearRefreshFn`, `nl_refresh`, `enable_nonlinear_refresh`, `has_nonlinear_devices()`, `nonlinear_refresh_device.hpp`, `models/device_model.hpp`
- [ ] 8.2 The conformance matrix's capability table is computed from the concept traits; delete the hand-kept table
- [ ] 8.3 Remove the snapshot views and the legacy pickle reader if the owner's answer to Open Question 4 allows (**BREAKING**)
- [ ] 8.4 Test-only kind `TestRcDevice` registered only in the test build: it stamps, integrates, snapshots and reports its current on every engine with no edit under `solver/`, `analysis/` or `python/`
- [ ] 8.5 `test_stateful_device_stamps_compose.py` extended to every pair of stateful kinds
- [ ] 8.6 Oracle: goldens; matrix; record `StoredKind::` site count and build time (budget +15 %)

## 9. Generic outputs
- [ ] 9.1 Kernel: pin currents and power for every category at commit; on-demand evaluation for `f`-only kinds; per-sample table for kinds with a `q` row
- [ ] 9.2 Python `res.i` / `res.p` use the kernel for every kind; delete the ladder at `python/pulsim/__init__.py:1004-1188` and `_shockley_params`
- [ ] 9.3 `device_loss_summary` and `device_thermal_summary` list the Newton kinds, integrating `p` over `t`
- [ ] 9.4 Tests: `∫v·i` against a 1 mΩ sense resistor within 1e-4 for each Newton kind on pwl and trbdf2; PWL and linear `res.i` unchanged to 1e-12; the adaptive-grid loss equals the fixed-grid loss within 0.5 %
- [ ] 9.5 Docs: `AI_GUIDE.md` (remove the "add a 1 mΩ resistor" advice), `docs/known-limitations.md`, CHANGELOG, stubs

## 10. AC and DC from f and q
- [ ] 10.1 `analysis/linearize.hpp`: `Linearization{G, C}` at `(x0, mask)`
- [ ] 10.2 `run_mna_sweep` on `linearize`; remove the Newton-kind refusal of `fix-mna-sweep-correctness`
- [ ] 10.3 DC operating point through `Formula::Dc` for every Newton kind; `PmsmMna` DC test
- [ ] 10.4 Tests: the five analytic cases (coupled transformer k = 0.99, RC through a closed switch, Shockley at 1 A, saturable L, nonlinear capacitor) at 100 Hz, 1 kHz, 10 kHz within 1e-3 relative; DC operating points of the harness unchanged to 1e-9
- [ ] 10.5 Docs: AC analysis page, CHANGELOG

## 11. Static and mode kinds through the registry
- [ ] 11.1 `StaticDevice::stamp_static` for R, L, C, sources, VCVS, ideal transformer; `ModeDevice::stamp_mode` for `Switch` and `SwitchedDiode`
- [ ] 11.2 `assemble.hpp`, `dc_assemble.hpp`, `history_state.hpp`, the three source overlays iterate the type list
- [ ] 11.3 Oracle: `J`, `C`, `b` identical entry by entry for every mask of every harness circuit; goldens bit-identical; `StoredKind::` sites outside `device/` and the binding's `kind_of` reduced to zero

## 12. Field tables
- [ ] 12.1 `fields[]` for every kind; builder validation errors name device and field
- [ ] 12.2 `components()` params, YAML keys and the device reference page generated from the tables; YAML gains the missing kinds (`shockley_diode`, `lauritzen_diode`, `nonlinear_capacitor`, `pmsm_mna`, flux-table inductor)
- [ ] 12.3 Oracle: `components()` output unchanged for existing kinds; `test_shipped_yaml_examples.py`, `test_yaml_composite_devices.py` green; stub check
- [ ] 12.4 Docs: device reference, YAML reference, CHANGELOG

## 13. Close
- [ ] 13.1 Update `openspec/CORE_ROADMAP.md` status and `docs/mental-model.md`
- [ ] 13.2 Final measurement table in the PR: dispatch sites, files touched to add a kind, build time, ns/step rows 10–13
