Reference values (50 digits, `probes/references_50_digits.py`;
circuit of `numerics/p2_buck.py`: 24 V, 100 µH, 47 µF, `g_on = 1e3`,
`g_off = 1e-9`, 100 kHz):

- CCM (R = 1 Ω, D = 0.5, edges from t = 0), state at 40 T:
  `i_L = 11.8015131102418 A`, `v_C = 12.2288664641139 V`.
- DCM (R = 50 Ω, D = 0.3), periodic orbit at the rising edge:
  `i_L = 6.00750582388986e-9 A`, `v_C = 8.99624708728943 V`, diode
  turn-off at `τ/T = 0.799830698444468`.

## 1. Characterisation (no behaviour change)
- [x] 1.1 Commit the references above as test constants; add a test that the current `dsed` meets CCM to 1e-9 and DCM (4000 periods) to 1e-8 (today: 12.228866464, 8.996247087) — `python/tests/test_dsed_exact_propagator.py` (CCM 1e-10, DCM 5e-12 after the fix); before/after in `probes/dsed_baseline.py`
- [ ] 1.2 Split-measure the 3.6 µs dsed step with the allocation counter and `/usr/bin/sample`: bytes and calls per step in `exact_advance`, `locate_event_exact_`, `ped_push_sample`, the binding conversion; record the table in design.md — deferred: measured instead with a Python profile of the sync buck (C++ loop ≈ 1 ms, binding conversion ≈ 2.3 ms, per-sample `np.asarray` 0.6 ms of 17.4 ms); no allocation counter in this slice
- [x] 1.3 Reproduce the rest-state abort (`probes/vth0_rest_state.py`) as an expected-failure test; log `v_D`, `i_D` and both derivative signs per settle round — reproduced on the untouched build (`probes/dsed_baseline.py`), kept as a passing test after 5.x, no xfail added
- [x] 1.4 Reproduce the DCM idle chatter (DCM buck, `V_th = 1e-6`, L = 100 µH·(1 ± δ), δ ∈ {1e-5, 1e-4, 1e-2}; at least one sign of each δ aborts today) as expected-failure tests and log the signal and both derivative signs at each of the 32 commutations — reproduced (−1e-5, +1e-4, +1e-2 aborted), kept as passing tests after 5.x
- [ ] 1.5 Conditioning census: κ₂(V) in energy coordinates and `‖A·h‖₁` for every mask visited by the analytic reference set of `restore-reference-benchmark-harness`; record how many masks each tier of D2 would take — deferred: needs the analytic reference set of restore-reference-benchmark-harness

## 2. Propagator kernel
- [x] 2.1 `phi1`, `phi2` through `expm1`; test relative error ≤ 1e-14 on `|s| ∈ [1e-12, 50]` including `s = 1.01e-6` (today 6.5e-11) and purely imaginary `s` — φ₂ added with 8.2 (series below |s| = 1/2), same test (`Exact: phi2 is accurate to 1e-14 at every |s|`)
- [x] 2.2 Energy-coordinate scaling before the eigenproblem; κ₂ by SVD; `ExactLTI` carries `kappa`, `tier`, `reason`
- [x] 2.3 Tier 2: Padé [13/13] of the augmented matrix, cached per `(mask, h)`; acceptance `u·‖A·h‖₁ ≤ 1e-10` — cached per h, 16 entries per mask; bound evaluated at dt_max
- [ ] 2.4 Tier 3 hand-off to the implicit stepper for that mask; strict option that refuses instead — deferred (Phase 3): tier 3 runs the scheduler's numeric path for the mask and is recorded (DECISIONS.md OQ2)
- [x] 2.5 Tests: series RLC at `R/R_crit − 1 ∈ {1e-3, 1e-9, 1e-13, 0}` within 1e-10 of a 50-digit reference, the recorded tier satisfying its D2 bound (the exactly critical case must not be tier 1); buck idle mode `Φ₂₂ = 0.99914929813728288` within 1e-14 on tier 1 — C++: `core/tests/dsed/test_exact_lti.cpp`; Python: critical RLC tiers

## 3. Modal stepping and allocation removal
- [ ] 3.1 Cache `w = V⁻¹b` per mask; keep `z` across steps of a segment; rebuild `z` at events and every 1024 steps — partial: `w = V⁻¹b` cached per mask; `z` is not kept across steps (rebuilt each step)
- [ ] 3.2 Predicate rows `cᵀV` cached per `(mask, predicate)`; Illinois evaluates in O(n) — deferred: predicates are opaque functions today
- [ ] 3.3 Preallocated workspaces; samples into the contiguous trajectory buffer; zero-copy arrays to Python — partial: workspaces preallocated, states handed to Python as one 2-D array; samples still in `std::vector<Vector>` (contiguous buffer is Phase 3)
- [x] 3.4 Counters of D10 in the result — `result.exact`: n_eig, n_pade, n_phi_cache_hits, n_predicate_evals; `n_alloc_steady` not available (no counter)
- [ ] 3.5 Measurement gate (counters, sync buck of `perf/p9_dsed.py`, 2000 periods): 4000 ± 1 steps; eigendecompositions = masks visited; `n_alloc_steady = 0`; predicate evaluations ≤ 40 per located event (Illinois cap 30 plus the scan); results within 1e-12 of task 1.1's run. Report ns/step before and after as information, not as the gate — partial: 4000 steps, n_eig = masks = 2, n_pade = 0 asserted; predicate evals ≤ 2·steps + 40·located asserted (the scan step dominates); n_alloc_steady not measured

## 4. Event location
- [x] 4.1 Stop rule `|t_b − t_a| ≤ max(4u·|t|, 1e-12·h_seg)`; remove the `|g| < 1e-10` test
- [x] 4.2 Scan step `min(h_seg, π/(4·ω_max))` — over every under-damped mode, ≤ 1024 scan points
- [x] 4.3 Tests: DCM turn-off instant within 1e-12·T of `0.799830698444468·T`; a 10 MHz buck (period 1e-7 s) has duty error below 1e-9; an under-damped LC whose diode voltage crosses zero twice inside one gate segment reports both events

## 5. Zero-band mode selection
- [x] 5.1 Derivative rule (D4) for signals inside the event tolerance; Gauss–Seidel settle; `%.9e` time in the error — the look-ahead on the exact OFF-mode trajectory replaces the derivative (DECISIONS.md)
- [x] 5.2 Tests: task 1.3 now passes and equals the `V_th = 1e-9` run to 1e-12; the audit's DCM buck with the gate held off until t0 = 2 µs reads 13.309497 V at t0 + 40 T, as `V_th = 1e-9` does today; a diode bridge at rest with `V_th = 0` settles; the six runs of task 1.4 complete, and the central difference of `v_C(40T)` in L at δ = 1e-5 is 1758.1254 V/H within 1e-6 relative (1.75e-7 today at δ = 1e-6); the half-wave scenario of `add-switched-diode-forward-drop` is unchanged — bridge at rest not added as its own test; the V_f clamp (`add-switched-diode-forward-drop`) now runs and matches pwl

## 6. Initial conditions
- [x] 6.1 Per-device map of `c0` / `i0` onto the reduced state; mass-norm projection for capacitor loops, reported in diagnostics; refusal for a non-state node voltage — refusal for a non-state node voltage not added: capacitor loops are refused by the extractor before the map runs
- [x] 6.2 Remove the refusal at `python/pulsim/__init__.py:2054-2082`
- [x] 6.3 Tests: series RL with `i0 = 100 A` decays as `100·e^{−tR/L}` to 1e-10; a floating capacitor with `c0 = 5 V` reads 5 V at t = 0 on both terminals' difference; two paralleled capacitors (1 µF at 10 V, 3 µF at 2 V) start at 4.0 V with the adjustment reported; agreement with `engine='trbdf2'` (rtol 1e-9) to 1e-6 — parallel capacitors: refused by name by the extractor (capacitor loop), not 4.0 V; recorded in KNOWN_LIMITATIONS.md

## 7. Coupled inductors
- [ ] 7.1 Inductance-matrix block in the extraction; refusal for `|k| = 1`
- [ ] 7.2 Tests: two coupled inductors (L₁ = L₂ = 1 mH, k = 0.9), each winding in series with 1 Ω, 10 V step on the primary, secondary loop closed: `i₁,₂ = 5·[(1 − e^{−t/τ₊}) ± (1 − e^{−t/τ₋})]` with `τ± = (1 ± k)·L/R = 1.9 ms, 0.1 ms`, within 1e-10; flyback agreement with `trbdf2` to 1e-5

## 8. Sources through the event queue (needs `add-unified-event-queue`)
- [x] 8.1 Piecewise-constant sources: breakpoints as time events, `b` constant between — done in the dsed scheduler without the unified queue: `next_source_corner_after(t)` joins the gate-edge min (DECISIONS: group 16 slice)
- [x] 8.2 φ₂ term for piecewise-linear sources — `exact_advance_affine_into` (tier 1 φ₁/φ₂, tier 2 augmented exponential); C++ test on a defective (tier 2) and a diagonal (tier 1) block against closed forms
- [x] 8.3 Tests: the pulse-driven RC of audit item 11 (trbdf2 floors at 1.8e-4 V on 10 V today, whatever the rtol) within 1e-10 of the closed form at every sample; a ramp source into an RL within 1e-10 of the closed form — `test_exact_sources.py`: the instant-edge pulse RC (3.3 µs start) within 1e-10 at every sample, the ramp RL within 1e-10

## 9. Strategy in the unified stepper (needs `refactor-unified-transient-stepper`)
- [ ] 9.1 `ExactPropagator` integrator behind the stepper's integrator interface, event-driven controller
- [ ] 9.2 Applicability rule and refusal (`UnsupportedError` with the failing rule); route reason in the result
- [ ] 9.3 Conformance-matrix rows for the strategy: every device kind either runs or is refused by name

## 10. Fixed-step Φ/Γ
- [ ] 10.1 `(Φ_m, Γ_m)` cache keyed on `(mask, dt)`, byte-budgeted; dense update loop
- [ ] 10.2 Exact sub-steps for diode events between grid points
- [ ] 10.3 Tests: CCM buck at dt = T/10, T/50, T/100 within 1e-10 of the task 1.1 reference (prototype: 7e-14, 1.3e-12, 2.5e-12); DCM orbit within 1e-9; identical to the event-driven form to 1e-11
- [ ] 10.4 Measurement gate: flops per step = n² + n (counted), zero allocations, zero factorizations after the first visit of each mask; ns/step reported

## 11. Hybrid with Newton devices (needs `add-unified-device-interface`)
- [ ] 11.1 `affine_region` on the device interface; mode key `(mask, regions)`; region boundaries as state events
- [ ] 11.2 Implicit step whenever a device has no affine region; counters `n_exact_steps`, `n_implicit_steps`
- [ ] 11.3 Tests: flux-table inductor below its first knee equals the linear inductor to 1e-12 and crosses the knee as an event; a Shockley rectifier runs with `n_exact_steps = 0` and matches `trbdf2`; a buck with a PWL `Coss` table matches `trbdf2` (rtol 1e-9) to 1e-6

## 12. Retirement, docs, verification
- [ ] 12.1 Deprecate `integrator='rk45'|'bdf2'` on `engine='dsed'` (owner decision); remove after one release
- [ ] 12.2 Measure the dense / sparse crossover on the RLC ladder (n = 5 … 482) with the counter harness and replace the `n_state ≤ 64` placeholder
- [ ] 12.3 Docs: engine guide ("when the exact strategy is selected, when it is refused"), `docs/dsed.md`, AI_GUIDE, CHANGELOG with the 1e-10 result movement — partial: dsed chapters 11 and 15, `docs/known-limitations.md`, `docs/internals/layer5-v2-ideal-diode.md`, KNOWN_LIMITATIONS.md, CHANGELOG; `docs/dsed.md` does not exist; AI_GUIDE untouched (at its 32 KiB limit, nothing in it went stale)
- [ ] 12.4 ruff, pytest, ctest (`-j2`), `openspec validate --all --strict`, stub check

<!-- Re-centred scope, 2026-11-13: groups 13–19 are appended; groups 1–12 are unchanged. -->

## 13. Default route (re-centred scope, D11)
- [ ] 13.1 Segment eligibility test (D11 rules 1–4) and route record `result.exact['segments']` with the reason for every implicit segment. Report-only slice (DECISIONS, group 13 slice): one row per mask on every exact run (`exact`, `reason` 'device' / 'source' / 'extraction' / 'cost', `device`, `kind`, `detail`, `tier`, `n_segments`) and `pulsim.exact_route(builder, t_end, ...)`; open: implicit segments (group 14) and the route of 'auto'
- [x] 13.2 `engine='exact'` canonical, `'dsed'` alias; refusal of a non-eligible segment with `UnsupportedError` naming the rule, unless `allow_implicit_segments=True`. Done: the name and alias (`simulate()` maps above the dispatch; `_require_supported` takes either; results and refusals report 'dsed'). Refusal and `allow_implicit_segments` done (DECISIONS: group 13 slice): default refuse, `'source'` / `'cost'` / `'extraction'` by rule; conformance cells in `test_exact_route.py`
- [ ] 13.3 Tests: `auto` routes the DCM buck, the boost, the single- and three-phase bridges (resistor-pair and, after `add-branch-form-ideal-switch` group 8, ideal) to exact for every segment; a Shockley rectifier is routed implicit with reason `'device'`; `auto` and explicit `'exact'` give bit-identical states on the exact-only circuits. Report-only part (`python/tests/test_exact_route.py`): buck, boost, single-phase bridge, three-phase C-filter bridge and the three-phase rectifier example classify exact on every mask; Shockley → 'device' (Dsh, ShockleyDiode); coupled inductors, an inductor cut-set and a floating capacitor loop → 'extraction'; a PWM source → 'source'; 'exact' and 'dsed' bit-identical; `result.exact['auto_would_route']` reported on every exact run and surveyed on `examples/` (13 of 25 loadable would route exact, DECISIONS group 13). Open: `auto` does not route yet

## 14. Hybrid segments (D12; needs groups 9 and 11)
- [ ] 14.1 Classification at events; implicit segment to the next event; `affine_region` re-test at each accepted implicit step scheduling a re-entry event; SDIRK×3 on entry to an implicit segment
- [ ] 14.2 Flapping hold (8 re-entries per period); counters `n_exact_segments`, `n_implicit_segments`, `t_exact_fraction`, hand-over list
- [ ] 14.3 Tests: a buck whose switch is a MOSFET Level 1 and whose diode is PWL matches `trbdf2` (rtol 1e-9) within 1e-6 on `v_C`, and `t_exact_fraction` is at least the fraction of each period the MOSFET spends in cutoff (minus the event tolerance); a Shockley-diode rectifier runs with `n_exact_segments = 0`

## 15. Scaling rule (D13)
- [ ] 15.1 Action-of-the-exponential kernel (truncated Taylor with scaling and the 1-norm estimate) on the sparse `A` of a mode, augmented with the source states
- [ ] 15.2 Cost model and route choice per mode; counters in `result.exact['routes']`
- [ ] 15.3 Tests on the RLC ladder of `probes/expm_action_ladder.py` (h = 1 µs): n = 2000 non-stiff → action, 20 ± 2 matvecs per step, states within 1e-13 of the dense kernel on n = 200; n = 300 with snubbers and a mode visited for 10⁴ steps → dense; the same with 50 steps per visit → hand-over with reason `'cost'`
- [ ] 15.4 Calibrate the flop constants, `m_max` and the dense budget against C++ timings with the counter harness; record them here

## 16. Exact sources (D14)
- [x] 16.1 Augmented source states: polynomial chain, exponential, shared oscillator pairs; PWL corners as time events. Sine: shared oscillator pairs (before this change). Pulse / PWM: level + ramp through φ₁ / φ₂ between corners, each corner a time event (DECISIONS: group 16 slice). Exponential and higher-polynomial sources: no such source exists in the builder; table sources: none exist (only `add_saturable_inductor_table` takes a table)
- [x] 16.2 Tests: 100 V 50 Hz sine into 1 Ω, 1 mH at `h = T/20` within 1e-10 A of the closed form over 5 periods (prototype 1.6e-12); a 0 → 10 V ramp over 1 ms into an RC within 1e-12 of the closed form; a PWL source's corners appear as events. `python/tests/test_exact_sources.py`: sine 3.2e-13 A; ramp RC 2.7e-13 V with a sample exactly at 1 ms; pulse train, PWM source and coincident gate edge: every corner a sample

## 17. Output samples and controller ticks as events (D15; needs `add-unified-event-queue`)
- [ ] 17.1 Samples evaluated from the last event (modal O(n), or a cached `Φ(dt_out)`); ticks as queue events with zero-order hold
- [ ] 17.2 Tests: the DCM buck's samples at `dt_out = T/7` equal the exact flow within 1e-12 and the event count is the same as with no samples; a 20 kHz controller over 1 ms produces 21 ticks at the instants of `fix-control-tick-semantics`

## 18. Ideal switches native (D16; with `add-branch-form-ideal-switch` group 8 and `add-complementarity-mode-selection` group 6)
- [ ] 18.1 Exact route on branch-form ideal switches with reduced state per mode; mode by the selection service; D4 zero-band rule retired when the service is the default
- [ ] 18.2 Tests (the prototype oracles): DCM buck 4000 periods from rest, `v_C` within 1e-9 V of 8.9963970815477 V and turn-off within 1e-11·T; boost CCM 20 000 periods within 1e-9 V and 1e-9 A of the mpmath orbit; three-phase overlap angle 15.479928° within 1e-6°

## 19. Default flip (in the single number-changing release)
- [ ] 19.1 `engine='auto'` default route as D11; legacy names warn where they change meaning
- [ ] 19.2 Re-baseline through the reference harness with the other number-changing defaults; CHANGELOG
