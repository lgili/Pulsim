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
- [x] 1.1 `pwl/stateful_histories.hpp`: a tuple of the five history objects with `for_each(f)`, `init`, `seed_from_dc_op`, `commit(x, h, stage)`, `capture_gamma`, `snapshot_into`, `restore`, `to_flat`, `from_flat`
- [x] 1.2 Replace the 56 named calls in `solver/run_transient.hpp` by the folds
- [x] 1.3 Replace the 43 named calls in `solver/trbdf2_transient.hpp` by the folds
- [x] 1.4 Oracle: goldens bit-identical; trbdf2 counters identical; `test_stateful_device_stamps_compose.py`, `test_snapshot_resume_stateful.py`, `test_trbdf2_stateful_with_diodes.py`, `test_trbdf2_stateful_devices.py` green
- [x] 1.5 Docs: none (internal)

## 2. Kernel-built refresh
- [x] 2.1 `pwl::build_device_refresh(graph, pool, histories, StepCtx&)`: the static diode / MOSFET / IGBT stamps (kernel-built since `add-engine-conformance-matrix`) plus one typed pass per history, in one function; both drivers call it
- [x] 2.2 Delete the two wrapper chains (RT:471-600, TB:588-697) in favour of 2.1; keep `nl_refresh` as a deprecated override that replaces the base stamps only
- [x] 2.3 Remove whatever `make_combined_diode_mosfet_refresh()` call sites remain in `python/bindings.cpp` (four on this branch); `enable_nonlinear_refresh=` emits a `DeprecationWarning` — the transient call sites went in slice 1; the DC-only `resolve_refresh` is the kernel's `pwl::dc_device_refresh` since slice 2 (DECISIONS #20)
- [x] 2.4 C++ tests: the Phase 0 clamp test (0.594 V with no callback) still passes; every pair of stateful kinds in one circuit keeps each kind's single-kind result — written as Python tests (`python/tests/test_stateful_kinds_pairwise.py`, 12 ordered pairs on pwl and on TR-BDF2); the clamp test is in the ctest suite
- [x] 2.5 Oracle: goldens bit-identical; matrix unchanged
- [x] 2.6 Docs: CHANGELOG deprecation note; `docs/internals` section on the refresh

## 3. Flat device-state vector and snapshot
- [x] 3.1 `device/device_state.hpp`: `DeviceState`, `DeviceStateLayout`, signature — the state is `SolverSnapshot::dev_state` + `DeviceStateLayout`; a separate `DeviceState` struct had no user and was dropped (DECISIONS #8)
- [ ] 3.2 The five histories and `HistoryState` store into blocks of one `DeviceState`; their arithmetic is untouched; rollback is one copy into a preallocated twin — PARTIAL (DECISIONS #15): the in-run rollback of the stateful kinds is one flat vector carrying every real incl. the stage scratch (bit-identical); the live storage stays in each kind's entries, and `HistoryState` keeps its own rollback
- [x] 3.3 `SolverSnapshot`: add `dev_state`, `mask`, `layout_signature`; the seven old fields become computed views; resume refuses a layout mismatch by block name
- [x] 3.4 Pickle version 3; reader for the 10- and 11-field tuples that converts at resume time
- [x] 3.5 Tests: resume pwl → trbdf2 and trbdf2 → pwl on a circuit with every stateful kind; a 2.0 pickle resumes; a snapshot of another circuit is refused by name
- [x] 3.6 Oracle: goldens bit-identical; `test_snapshot_resume.py`, `test_snapshot_resume_stateful.py` unchanged; zero-alloc tests green
- [x] 3.7 Docs: snapshot page, stubs, CHANGELOG

## 4. The interface, unused
- [x] 4.1 `ad/ad_scalar.hpp`: `atan`, `atan2`, `pow`, `sin`, `cos`, `chain(x, value, dvalue_dx)`; unit tests against finite differences (`core/tests/layer5_v3/test_device_interface.cpp`)
- [ ] 4.2 `device/device.hpp`: `Device`, `ModeDevice`, `SelfCommutated`, `StaticDevice`, `Formula`, `StepCtx`, `Out`, the optional-member detectors — PARTIAL (DECISIONS #19): `Device` (with `Params<S>` for sensitivities), `Formula`, `StepCtx`, `Out`, `seed_params`; `device::StatefulKind` for the shipped stateful kinds (`device/kind.hpp`); not `ModeDevice` / `SelfCommutated` / `StaticDevice` nor the optional-member detectors
- [x] 4.3 `device/stamp_fq.hpp` with `history_term` for the four formulas; unit tests: a linear capacitor and a linear inductor written as `Device` reproduce the companion stamps of `stamping/` entry by entry for trapezoidal, backward Euler and the BDF2 stage — in `device/device.hpp` (dense `stamp_fq`, `commit_fq`); `J` is compared with `stamping/`, `F` with the companion's hand formula
- [ ] 4.4 `device/registry.hpp`: `TypeList`, `Builtin`, `DeviceSet`, `Instance`, `for_each_newton_kind`; `StoredKind` derived from the list with the existing values asserted — PARTIAL (DECISIONS #14): `device::TypeList` and the stateful-kind list `pwl::StatefulKinds` with its compile-time checks (concept, block widths, each block claimed once, Newton row of `kKindCaps`); `StoredKind` is not derived from a list
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

<!-- Re-centred scope, 2026-11-13: group 14 is appended; groups 1–13 are unchanged. -->

## 14. Complementarity form on the PWL category (re-centred scope, D10)
- [ ] 14.1 `ComplementarityPort` and the `ComplementarityDevice` concept; `SwitchedDiode` implements it in both forms
- [ ] 14.2 The mode selector reads the ports from the registry; no device restates the law
- [ ] 14.3 Tests: the ports of a conductance-form diode (`g_on = 1e3`, `g_off = 1e-9`, `V_f = 0.7`) are `R_on = 1e-3`, `g_off = 1e-9`, `V_f = 0.7`; the sparsity pattern of every mask is unchanged by the concept

<!-- Slice 3, 2026-11-18: the dual path (DECISIONS #23). Steps 5–7 run
     behind device_path='interface'; the hand stamps stay the default and
     are deleted at the release (items 5.1–7.5 "delete …" wait for it). -->

## 15. Dual path (DECISIONS #23)
- [x] 15.1 `device::DevicePath`; kernel `SimulationOptions::device_path`, `TrBdf2Options::device_path`, `dc_device_refresh(pool, enable, path)`; Python `simulate(device_path=)`, `compute_dc_op(device_path=)`; `engine='dsed'` refuses `'interface'`
- [x] 15.2 `Device` contract: optional per-instance `Aux` (law variant, table, Jiles–Atherton base state), increment form in `stamp_fq`, internal-row commit rule (`q̇_n ← −f`), `kHasQ` / `kHasF`
- [x] 15.3 Laws (`device/laws.hpp`): smooth / Shockley diode, MOSFET L1, IGBT L1 (steady state) — the model's own `current<S, P>` body; nonlinear capacitor (junction, VDMOS C_gd, table); saturable inductor (atan, table, Jiles–Atherton)
- [x] 15.4 Sparse adapters (`pwl/interface_stamps.hpp`): stamp, commit, stage capture per ported kind, compile-time structural pattern; `StatefulHistories::set_path` refuses an unported kind by name
- [x] 15.5 Gates: `test_device_path_equivalence.py` (pwl ≤ 1e-12 of scale, trbdf2 ≤ 10·rtol at common accepted times, DC ≤ 1e-12); `test_device_laws.cpp` (every law variant — junction / VDMOS / table Coss, atan / table / Jiles–Atherton inductor — stamps, commits and captures the stage point to a few ulp of the hand path; ∂/∂p against central differences; `param_tangent` false for table and Jiles–Atherton); registry-driven cells in `test_engine_conformance.py`
- [x] 15.6 Port the Lauritzen diode, the IGBT tail, the MNA PMSM and the induction machine — Lauritzen and IGBT tail (local rows, DECISIONS #28); PMSM, flux-map PMSM and induction machine (flux as q, increment form, DECISIONS #42): pwl ≤ 2.5e-14 of scale; speed: PMSM within 3 %, the induction machine 1.04–1.09× (over the gate, lever recorded in #42)
- [x] 15.11 Readings follow the stamped law: `for_each_static_reading` takes the path (DECISIONS #43)
- [x] 15.7 Speed within 3 % on every case — measured reason (DECISIONS #38): instructions retired per accepted step on 'interface' are 0.86–1.02× legacy on the six law circuits (same Newton counts); per call every stamp but Lauritzen 1.19× / tail 1.10× is faster (#27); wall time on this shared machine cannot resolve 3 % — a quiet-machine timing stays on the release checklist; RE-MEASURED at slice 6 (#41): 0.90–0.98× on every case but the saturable inductor, 1.023–1.031× over three runs — at the gate's edge, its `commit_fq` λ re-evaluation is the open lever
- [ ] 15.8 At the release: flip the default, delete the hand stamps (list in DECISIONS #25)
- [x] 15.9 MOSFET L1 law C¹ through V_DS = 0 on 'interface' (the model's leak f(a, 0)·e^{−κ|V_DS|} dropped; DECISIONS #39): derivative-continuity, monotonicity (to 1e-13 V) and law-minus-leak tests; the boost with 0.3 µs gate ramps is a recorded path difference, its 1 ns-edge control and an always-on AC reverse-conduction circuit gate at 1e-12; the sub-threshold band and the triode/saturation corner stay on the release list
- [x] 15.10 Reviewer fixes (DECISIONS #40): the Tanh (permeance) law's tangent is the model's sech² (AD 1 − tanh² cancelled in saturation)
