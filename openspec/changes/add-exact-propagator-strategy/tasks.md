Reference values (50 digits, `probes/references_50_digits.py`;
circuit of `numerics/p2_buck.py`: 24 V, 100 µH, 47 µF, `g_on = 1e3`,
`g_off = 1e-9`, 100 kHz):

- CCM (R = 1 Ω, D = 0.5, edges from t = 0), state at 40 T:
  `i_L = 11.8015131102418 A`, `v_C = 12.2288664641139 V`.
- DCM (R = 50 Ω, D = 0.3), periodic orbit at the rising edge:
  `i_L = 6.00750582388986e-9 A`, `v_C = 8.99624708728943 V`, diode
  turn-off at `τ/T = 0.799830698444468`.

## 1. Characterisation (no behaviour change)
- [ ] 1.1 Commit the references above as test constants; add a test that the current `dsed` meets CCM to 1e-9 and DCM (4000 periods) to 1e-8 (today: 12.228866464, 8.996247087)
- [ ] 1.2 Split-measure the 3.6 µs dsed step with the allocation counter and `/usr/bin/sample`: bytes and calls per step in `exact_advance`, `locate_event_exact_`, `ped_push_sample`, the binding conversion; record the table in design.md
- [ ] 1.3 Reproduce the rest-state abort (`probes/vth0_rest_state.py`) as an expected-failure test; log `v_D`, `i_D` and both derivative signs per settle round
- [ ] 1.4 Reproduce the DCM idle chatter (DCM buck, `V_th = 1e-6`, L = 100 µH·(1 ± δ), δ ∈ {1e-5, 1e-4, 1e-2}; at least one sign of each δ aborts today) as expected-failure tests and log the signal and both derivative signs at each of the 32 commutations
- [ ] 1.5 Conditioning census: κ₂(V) in energy coordinates and `‖A·h‖₁` for every mask visited by the analytic reference set of `restore-reference-benchmark-harness`; record how many masks each tier of D2 would take

## 2. Propagator kernel
- [ ] 2.1 `phi1`, `phi2` through `expm1`; test relative error ≤ 1e-14 on `|s| ∈ [1e-12, 50]` including `s = 1.01e-6` (today 6.5e-11) and purely imaginary `s`
- [ ] 2.2 Energy-coordinate scaling before the eigenproblem; κ₂ by SVD; `ExactLTI` carries `kappa`, `tier`, `reason`
- [ ] 2.3 Tier 2: Padé [13/13] of the augmented matrix, cached per `(mask, h)`; acceptance `u·‖A·h‖₁ ≤ 1e-10`
- [ ] 2.4 Tier 3 hand-off to the implicit stepper for that mask; strict option that refuses instead
- [ ] 2.5 Tests: series RLC at `R/R_crit − 1 ∈ {1e-3, 1e-9, 1e-13, 0}` within 1e-10 of a 50-digit reference, the recorded tier satisfying its D2 bound (the exactly critical case must not be tier 1); buck idle mode `Φ₂₂ = 0.99914929813728288` within 1e-14 on tier 1

## 3. Modal stepping and allocation removal
- [ ] 3.1 Cache `w = V⁻¹b` per mask; keep `z` across steps of a segment; rebuild `z` at events and every 1024 steps
- [ ] 3.2 Predicate rows `cᵀV` cached per `(mask, predicate)`; Illinois evaluates in O(n)
- [ ] 3.3 Preallocated workspaces; samples into the contiguous trajectory buffer; zero-copy arrays to Python
- [ ] 3.4 Counters of D10 in the result
- [ ] 3.5 Measurement gate (counters, sync buck of `perf/p9_dsed.py`, 2000 periods): 4000 ± 1 steps; eigendecompositions = masks visited; `n_alloc_steady = 0`; predicate evaluations ≤ 40 per located event (Illinois cap 30 plus the scan); results within 1e-12 of task 1.1's run. Report ns/step before and after as information, not as the gate

## 4. Event location
- [ ] 4.1 Stop rule `|t_b − t_a| ≤ max(4u·|t|, 1e-12·h_seg)`; remove the `|g| < 1e-10` test
- [ ] 4.2 Scan step `min(h_seg, π/(4·ω_max))`
- [ ] 4.3 Tests: DCM turn-off instant within 1e-12·T of `0.799830698444468·T`; a 10 MHz buck (period 1e-7 s) has duty error below 1e-9; an under-damped LC whose diode voltage crosses zero twice inside one gate segment reports both events

## 5. Zero-band mode selection
- [ ] 5.1 Derivative rule (D4) for signals inside the event tolerance; Gauss–Seidel settle; `%.9e` time in the error
- [ ] 5.2 Tests: task 1.3 now passes and equals the `V_th = 1e-9` run to 1e-12; the audit's DCM buck with the gate held off until t0 = 2 µs reads 13.309497 V at t0 + 40 T, as `V_th = 1e-9` does today; a diode bridge at rest with `V_th = 0` settles; the six runs of task 1.4 complete, and the central difference of `v_C(40T)` in L at δ = 1e-5 is 1758.1254 V/H within 1e-6 relative (1.75e-7 today at δ = 1e-6); the half-wave scenario of `add-switched-diode-forward-drop` is unchanged

## 6. Initial conditions
- [ ] 6.1 Per-device map of `c0` / `i0` onto the reduced state; mass-norm projection for capacitor loops, reported in diagnostics; refusal for a non-state node voltage
- [ ] 6.2 Remove the refusal at `python/pulsim/__init__.py:2054-2082`
- [ ] 6.3 Tests: series RL with `i0 = 100 A` decays as `100·e^{−tR/L}` to 1e-10; a floating capacitor with `c0 = 5 V` reads 5 V at t = 0 on both terminals' difference; two paralleled capacitors (1 µF at 10 V, 3 µF at 2 V) start at 4.0 V with the adjustment reported; agreement with `engine='trbdf2'` (rtol 1e-9) to 1e-6

## 7. Coupled inductors
- [ ] 7.1 Inductance-matrix block in the extraction; refusal for `|k| = 1`
- [ ] 7.2 Tests: two coupled inductors (L₁ = L₂ = 1 mH, k = 0.9), each winding in series with 1 Ω, 10 V step on the primary, secondary loop closed: `i₁,₂ = 5·[(1 − e^{−t/τ₊}) ± (1 − e^{−t/τ₋})]` with `τ± = (1 ± k)·L/R = 1.9 ms, 0.1 ms`, within 1e-10; flyback agreement with `trbdf2` to 1e-5

## 8. Sources through the event queue (needs `add-unified-event-queue`)
- [ ] 8.1 Piecewise-constant sources: breakpoints as time events, `b` constant between
- [ ] 8.2 φ₂ term for piecewise-linear sources
- [ ] 8.3 Tests: the pulse-driven RC of audit item 11 (trbdf2 floors at 1.8e-4 V on 10 V today, whatever the rtol) within 1e-10 of the closed form at every sample; a ramp source into an RL within 1e-10 of the closed form

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
- [ ] 12.3 Docs: engine guide ("when the exact strategy is selected, when it is refused"), `docs/dsed.md`, AI_GUIDE, CHANGELOG with the 1e-10 result movement
- [ ] 12.4 ruff, pytest, ctest (`-j2`), `openspec validate --all --strict`, stub check
