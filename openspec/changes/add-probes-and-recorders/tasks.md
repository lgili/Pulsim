Every numbered section is one or more PRs and ends with the
verification set: `ruff check`, `pytest python/tests`, `ctest`,
`openspec validate --all --strict`, the stub check, and
`core/tests/perf/test_zero_alloc.cpp`. "Goldens" = the fixed-engine
checkpoint hashes and the differential runner of
`restore-reference-benchmark-harness` (the full suite stands in until it
lands). Any cross-engine comparison uses `res.avg` / `res.integral` or
states at equal times.

## 0. Preconditions and baseline
- [ ] 0.1 `add-unified-device-interface` section 9 merged (generic pin currents, power, `outputs()`)
- [ ] 0.2 Record: stored bytes per sample and ns/step for audit rows 1 and 20; `res.i('D1')` first-call time at 100k samples (43 ms); the count of result-attribute assignments in `__init__.py` (19)

## 0b. Kernel probes and recorders, stored-array feeding (track feat/core-p3-probes-recorders; see DECISIONS.md)
- [x] 0b.1 `record/probe.hpp`: `Probe::compile` cuts the readings `v(n)`, `v(a,b)`, `i(dev)`, `p(dev)` out lexically and compiles the rest with `ir::Expr` (one parser); normalised, de-duplicated atoms
- [x] 0b.2 `record/recorder.hpp`: `Recorder` (keep every / every n-th / none; event rows with both limits; declared and per-period windows with integral, avg, exact-line RMS, min / max with times; exact clipping, right limit at a window start on an event, left limit at its end); `feed_table` (kernel-side pass over stored arrays); `attach(StepHooks&)` (in-loop, state columns)
- [x] 0b.3 Python: `simulate(recorders=[pulsim.Recorder(...)])` on every engine, `res.recorded[name]` (`pulsim.Recording`, named arrays); `store_every > 1`, `resume_from`, `p()` readings, bad windows refused by name before the run
- [x] 0b.4 Tests: C++ closed forms (triangle on a clustered grid, windows mid-step, periods; square wave with both limits; decimation; capacity unchanged over 2e5 samples; in-loop on the fixed engine: no per-step allocation, windows equal the stored-array pass); Python: RC charge closed form on pwl / trbdf2 / exact, rectified sine with commutations inside the window, both limits against `right_limit_states`, the bridge truth 311.343881428885841697 V on the three engines, `time_average` / `time_rms` agreement to 1e-12
- [ ] 0b.5 (waits for the Simulation-object integration; DECISIONS 10) `recorders` row in `sim::kRunOptionTable` (every engine) when `add-kernel-simulation-object` is integrated (its test compares the table to simulate()'s parameters)
- [ ] 0b.6 (waits for stepper R1-R3) In-loop feeding for Python on the stepper's per-step hook (DECISIONS requests R1-R3); then lift the `store_every` refusal
- [ ] 0b.7 Recorder state in the snapshot (new pickle index) so `resume_from` works with recorders; refused by name until then (not started in slice 2: needs `Recorder` state export / import of the partial windows and the start row, a SolverSnapshot field at index 17 with a reader branch and the five field-count tests; no loop edit)
- [x] 0b.8 `Times` / `t_eval` sampling (`keep='times'`, linear, `land=false`; landing waits for the event queue in the loop), `RingSink` (`Recorder::stream_to(LiveRing*)`, C++), `at(t)` (`res.at`) (DECISIONS 16)

## 1. One constructible result type
- [x] 1.1 `SimulationResult` default-constructible with fill methods used by kernel adapters only; binding exposes construction for adapters and tests
- [x] 1.2 The DSED adapter fills a `SimulationResult`; delete `_PEDSimulationResult` (`_dsed_dispatch.py:330-476`); `n_accept`, `n_reject`, `n_events`, `n_rk45_steps`, `n_bdf2_steps`, `cpu_time_seconds` become read-only properties
- [x] 1.3 Oracle: every dsed test passes unmodified; `isinstance(res, pulsim.SimulationResult)` on the three engines; `device_loss_summary` on a switch + diode buck unchanged on dsed
- [ ] 1.4 Docs: result reference; CHANGELOG (**BREAKING**: class identity) — CHANGELOG done; the result reference page not written

## 2. Grid kind and typed statistics
- [x] 2.1 `record/grid.hpp`: `GridKind`; `RunStats`; `res.grid`, `res.stats`
- [x] 2.2 (the alias kept; `_trbdf2_warn_stats` keeps reading its stats argument, the same numbers as `res.stats`, because tests drive it with edited dicts) `_trbdf2_stats` alias dict for one release; `_trbdf2_warn_stats` reads `res.stats`
- [x] 2.3 Oracle: suite green; identical warnings

## 3. Event rows on every engine
- [ ] 3.1 (loop freeze, DECISIONS 18) `EventRows` in the result; the fixed and variable-step drivers store the right-limit state and the next-step mask at every instant where the mask or a source changed
- [ ] 3.2 `_result_views.step_start_states` reads `res.events` for every engine
- [ ] 3.3 Tests: DCM boost diode average current 0.0535 A on pwl, trbdf2 and dsed; `times`, `states` and the mode record bit-identical to before
- [ ] 3.4 Oracle: goldens bit-identical; zero-alloc tests green (event rows pre-reserved from the gate-event estimate, chunked otherwise)

## 4. Measures on the result
- [x] 4.1 (as Python methods over the C++ recorder pass, DECISIONS 16) `avg`, `rms`, `integral`, `at`, `minmax` (piecewise-linear, right-limit aware, exact `h·(a² + ab + b²)/3` for the square)
- [ ] 4.2 (methods done; the `stats.time_average` / `pl_integral` delegation not done) Python methods; `stats.time_average`, `stats.time_rms`, `_result_views.pl_integral*` delegate
- [ ] 4.3 (done: rectified sine on three engines equals time_average / time_rms to 1e-12; not done: the PR #140 six-run check) Tests: analytic triangle and PWM waveforms on a uniform and on a clustered grid give the same `avg` / `rms` to 1e-12; the PR #140 circuit: six runs agree within 0.3 % through `res.avg`
- [ ] 4.4 Docs: `AI_GUIDE.md` trap on `.mean()`; measurement page; examples use `res.avg`

## 5. Probe set and post-hoc evaluation
- [ ] 5.1 (precondition 0.1, DECISIONS 18) `record/probe.hpp`: `ProbeKind`, `ProbeDecl`, `ProbeSet::compile`, `ProbeSet::evaluate`, name grammar (`v(n)`, `v(a,b)`, `i(dev)`, `i(dev.pin)`, `p(dev)`, `dev.channel`, `sw.on`)
- [ ] 5.2 `res.i` / `res.p` / `res[...]` call `ProbeSet` for resistor, switch, PWL diode (with `V_f`), source branches, inductor and transformer windings; delete the corresponding rungs of `_result_i`
- [ ] 5.3 Unknown probe name → `NetlistError` naming it and the nearest valid names; unrecordable-after-the-fact probe → `UnsupportedError` naming the probe to add
- [ ] 5.4 Oracle: `res.i` unchanged to 1e-12 for each ported kind on the harness; `test_device_loss_summary.py`, `test_loss_window.py` green

## 6. Capacitor current (own PR, **BREAKING**, pending the owner's answer)
- [ ] 6.1 `i(C)` from the companion law; `res.derived.i_smooth(name)` keeps the previous derivative form
- [ ] 6.2 Tests: `∫i dt = C·Δv` to 1e-12 per step on both engines
- [ ] 6.3 Docs: CHANGELOG migration note; known-limitations entry on trapezoidal ripple

## 7. In-kernel recording
- [ ] 7.1 (loop freeze / stepper hook, DECISIONS 18) `record/recorder.hpp`: `RecordOptions`, `Selection`, `Recorder`, `Sink`, `MemorySink`; `StepView`
- [ ] 7.2 Call `Recorder::on_step` / `on_event` at the push sites of the drivers (four in `run_transient.hpp`, three in `trbdf2_transient.hpp`, two in `bdf1.hpp`), or at the stepper's single hook if it has landed
- [ ] 7.3 With `All`: pin currents of devices with a `q` row recorded beside the state; replaces the interim table of `add-unified-device-interface`
- [ ] 7.4 `builder.add_probe(...)`; YAML `probes:` key in the loader; declared probes listed by the builder (`probes()`), ready for `to_yaml()` of `add-immutable-circuit-ir`
- [ ] 7.5 `RecordOptions.what = Probes{...}`: table has only the selected columns; `res.states` raises `UnsupportedError` naming `what`
- [ ] 7.6 Tests: recorded and recomputed values bit-equal for every recomputable probe; Newton-device `i()` against a 1 mΩ sense resistor within 1e-6 at every sample; stored bytes with three probes on the 12-node inverter equal `3 · 8` per row
- [ ] 7.7 Oracle: default runs bit-identical (`times`, `states`, modes); zero-alloc tests green on the uniform grid; ns/step rows 1 and 20 not above 0.2
- [ ] 7.8 Docs: probes page, YAML reference, stubs

## 8. Sampling policies and sinks
- [ ] 8.1 (waits for 0b.6) `Decimate{n}`: fixed grid identical to `store_every`; accepted grid keeps every n-th step and all event rows; lift the `store_every` refusal on the variable-step engine
- [ ] 8.2 (`land=false` done as 0b.8; landing waits for the event queue in the loop) `Times{t}` / `Period{T}`: landing through `OutputSample` events when `add-unified-event-queue` is present; linear interpolation with the `interpolated` flag otherwise or when `land = false`
- [ ] 8.3 (C++ `RingSink` done; `live_stream` on every engine waits for 0b.6) `RingSink` over `LiveRing`; `live_stream` on every engine
- [ ] 8.4 Tests: `store_every` goldens on the fixed engine; a decimated accepted-grid record gives the same `avg` as the full record to 1e-12; `t_eval` samples equal the states of a run with gate edges at those times; a `live_stream` consumer sees the same rows on pwl and trbdf2
- [ ] 8.5 Oracle: conformance matrix cells for `store_every` / `live_stream` updated in this PR only

## 9. Guard the unweighted mean
- [ ] 9.1 (not started, DECISIONS 18) `Signal` (`ndarray` subclass with `t`, `avg`, `rms`) returned for non-uniform grids; `GridWarning` on `.mean()` / `.std()`
- [ ] 9.2 Oracle: suite with `GridWarning` as error except in the tests that assert it; `np.asarray(signal)` is a plain array sharing memory

## 10. Remove the Python reconstruction
- [ ] 10.1 (after section 5) Delete the remaining `_result_i` ladder (`__init__.py:861-1188`), `_shockley_params` consumers, and the per-branch attribute assignments (fields set once in the finalisation step)
- [ ] 10.2 `losses.py` and `thermal.py` read currents from probes; `_diode_state.DiodeStateTrace` keeps its law for the per-step energy integrals
- [ ] 10.3 Oracle: loss and thermal summaries unchanged to their test tolerances; `res.i('D1')` first call measured (target: below 1 ms at 100k samples; an estimate, to verify)
- [ ] 10.4 Docs: `AI_GUIDE.md`, `docs/known-limitations.md`, `openspec/CORE_ROADMAP.md` status
