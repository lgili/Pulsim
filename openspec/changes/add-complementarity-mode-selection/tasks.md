Reference values (all from `probes/lcp_cases.out` unless stated).

- DCM buck (24 V, 100 µH, 47 µF, 50 Ω, D = 0.3, 100 kHz), ideal S and D:
  orbit `v_C = 8.9963970815477 V`, diode off at `0.79983739563545·T`
  (40-digit, `../add-branch-form-ideal-switch/probes/ideal_dcm_buck.py`);
  `v_C(40T)` from rest 13.316240022 V.
- Boost (12 V, 100 µH, 47 µF, D = 0.5, R = 50 Ω): CCM orbit
  `v_C = 24.02019756285 V`, `i_L = 0.6596805174426 A` (mpmath).
- Single-phase bridge (100 V, 50 Hz, 1000 µF ‖ 100 Ω, `V_f = 0.7`):
  conduction from 66.672808323° to 91.797645707° of each half cycle.
- Three-phase bridge, `L_s = 1 mH`, 10 A DC load, `V_f = 0`: overlap
  `μ = 15.479928°`.
- Three-phase bridge, 1000 µF ‖ 20 Ω, `V_f = 0.7`, C from 0 V:
  conduction 66.350991° to 98.970270°, inrush impulse 14.758493 J.

## 1. Oracle and characterisation (no behaviour change)
- [x] 1.1 Port `probes/lcp_core.py` (Lemke, enumeration, port model) to `python/tests/oracles/lcp_reference.py` as a test-only oracle; unit tests: on 200 random PSD instances (k ≤ 8) Lemke and enumeration agree; on the LCPs dumped from the five families both return the same mode
- [x] 1.2 Port `probes/today_dsed.py` as characterisation tests of today's kernel: `dsed` with `V_th = 0` fails "did not settle" at rest and "is chattering" at 0.996 ms on the DCM buck; marked to flip at the default stage. Re-measured on the exact-propagator kernel (slice 1): both now run; the remaining failure is silent — the boost with `V_th ≤ 1e-6` ends 1.01 V low (R = 500) and 0.13 V low (R = 50) after 400 periods on `dsed` while `pwl`/`trbdf2` match the reference (`python/tests/test_mode_selection_today.py`)

## 2. ModeSelector in C++ (dense, standalone)
- [x] 2.1 `core/include/pulsim/events/mode_selector.hpp`: dense Lemke with the lexicographic ratio test and warm start from a given basis; enumeration for `k ≤ 12`; circuit-scale tolerances (D4)
- [x] 2.2 Catch2 tests on the dumped `(M, q)` instances of task 1.1: same mode as the oracle, same pivot count within ±1, secondary ray reported on the infeasible instance; zero pivots when `q ≥ 0`. Measured: the 218 LCPs give the oracle's pivot count exactly (max 6), z and w within 1e-9 of scale (`core/tests/modes/test_mode_selector.cpp`, `python/tests/test_mode_selector.py`)

## 3. Port model on the conductance form, and the `h_e` band
- [x] 3.1 Build `(M, q)` from the event's companion matrix (BE at `h_e`) through the cached factor: one solve for `q`, one per requested column; hybrid coordinates from the previous mode
- [x] 3.2 Measure, on the five families with resistor-pair diodes (`R_on = 1/g_on`), the range of `h_e/T_ref` over which every event returns the same mode; set the default `h_e` inside it and record the band in this file. Band (`probes/h_e_band.out`, mismatches against the prototype mode per `h_e/T_ref` = 1e-12…1e-2): buck at rest and DCM buck exact from 1e-10; boost exact 1e-11…1e-8, one event (t = 0, the R_on current share, DECISIONS #7) from 1e-7; single-phase bridges exact from 1e-9 (V_f = 0.7 one miss at 1e-2); three-phase overlap exact everywhere; three-phase C filter exact 1e-9…1e-6, 12 misses from 1e-5. Common band [1e-9, 1e-6]; default `h_e = 1e-7·T_ref`
- [x] 3.3 Tie widening (100×, 1e4×), OFF on a remaining tie, counted; record the current of every classified tie on the five families and count the ties decided differently with a 10× smaller `tol_w`. At `h_e = 1e-7·T_ref`: DCM buck 20 first-pass ties up to 0.72 A (the σ·C_max term), all resolved by widening; bridges max tie current 3.1e-2 A; 0 decided differently at `tol/10` in every family (at 1e-6·T_ref the three-phase C filter had 12)
- [x] 3.4 Hysteresis (D8): port held OFF, one bounded re-solve when `v ≥ V_th`, `v − V_th` as its crossing signal; Lemke-level test on the half-wave rectifier of the device-models scenario (turn-on at 15.9155 µs, off at T/2)

## 4. Shadow mode (first user-visible slice; numbers unchanged)
- [x] 4.1 `mode_selection='shadow'` on `dsed` settle: after today's settle, the selector runs on the same held state; disagreements in `result.diagnostics['mode_shadow']`
- [ ] 4.2 Same on the fixed-step event iteration, TR-BDF2 settle and DC seeding. Open: `'shadow'` is refused by name on `pwl`/`trbdf2`/`auto` (DECISIONS #11). What remains: a hook after the mask iteration in `solver/run_transient.hpp`'s event loop and after the settle in `solver/trbdf2_transient.hpp`, each building `(Z, u)` from its own factor through `cache.solve_at(mask, 2h_e)` and the BE history `−(C/h_e)·x(t⁻)` (the dsed adapter's `diode_port_problem_full` is the template) and calling `ModeSelector::select`
- [x] 4.3 Tests: on the buck, boost, single-phase and three-phase bridges with `V_th = 1e-2` (runs today on every engine) the shadow reports 0 disagreements; with `V_th = 1e-6` it reports each event where today's settle needed more than one round; states are bit-identical to `mode_selection` unset. Measured (`probes/shadow_survey.out`, `python/tests/test_mode_shadow.py`): banded buck, boost and single-phase bridge 0 disagreements; the three-phase C filter reports 30 at `V_th = 0` and `1e-2` alike, the LCP's tie rule wrong against the gauge (see the survey); `V_th = 0` boost: every disagreement is the settle's error (judge, DECISIONS #9); bit-identical on all 19 survey circuits and the six of `probes/bit_identity.py`

## 5. Passivity and refusals
- [ ] 5.1 Structural passivity classification at build time (device kinds, parameter signs, coupled-inductor matrix PSD). Slice 2: `CircuitBuilderAdapter::passivity()` names the first VCVS or negative R/C/L (DECISIONS #14). Open: the coupled-inductor matrix PSD test and a gyrator / ideal-transformer class (no such builder device reaches the dsed adapter today)
- [x] 5.2 Non-passive (including the passive-but-non-reciprocal gyrator and ideal transformer, which keep Lemke but report non-unique voltages): Lemke, then enumeration to `k = 12` (no solution → typed error; several → fewest flips, counted), then `UnsupportedError(engine, 'non_passive_ports', device, kind)` naming the controlled source and its device kind. Slice 2: Lemke, then `enumerate_lcp` to `k = 12` (counted `enumerations`, `nonunique_modes`), then `UnsupportedError(engine, 'non_passive_ports', device, kind)`; Catch2 "Named refusals of the mode selection"
- [x] 5.3 No consistent mode on a passive circuit (secondary ray) → typed error naming the diodes and `t`; test with a current source forced into two blocking diodes. Slice 2: `UnsupportedError(engine, 'consistent_mode', 'D1, D2', 'Diode')` with `t` (`events/mode_refusal.hpp`); tested on a synthetic ray (`H = −I`): in the conductance form `g_off > 0` makes `H` positive definite, so a current source into two blocking diodes has a solution (5e8 V) and no ray

## 6. Opt-in `mode_selection='lcp'`
- [x] 6.1 `dsed`: settle replaced by the selector; the chatter guard stays as an assertion that never fires. Slice 2: `mode_selection='lcp'` decides at t = 0 and every event on the dsed event integrator; the settle is not run (DECISIONS #18)
- [x] 6.2 Tests on `dsed` (resistor pair, `V_th = 0`): buck at rest runs and stays at 0; DCM buck 4000 periods runs; boost `R = 50` and `500` with `V_th = 0` run; L·(1 ± δ), δ ∈ {1e-5, 1e-4, 1e-2}, all six complete; single- and three-phase bridges with `V_th = 0` run. Slice 2 (`python/tests/test_mode_lcp.py`): every item runs and matches trbdf2 at rtol 1e-10 (bridges `p − n`); boost 20/50/500 Ω to 1.5e-6–1.24e-5 V; L·(1 ± δ) to 1.1e-6 V, 1178 events each; also the half-wave RL rectifier and the double-crossing tank with V_th = 1e-3 (exact instants); slice 3: the same tank at V_th = 0 after the locator fix (DECISIONS #19), four edges within 3e-16 s of solve_ivp. Event counts: `probes/guard_survey.out`, `probes/extra_circuits.out`
- [ ] 6.3 Fixed engine and TR-BDF2 with `mode_selection='lcp'`; the same tests; mask-cycle and round caps unused (counter 0)
- [ ] 6.4 Hysteresis end to end: the half-wave rectifier with `V_th = 0.5 V` on every engine; the banded DCM buck (`V_th = 1e-2`) equals today's run within 1e-9. Slice 2: on dsed only — the V_th = 0.5 V half-wave rectifier under 'lcp' is bit-identical to the default and within 2e-6 V of trbdf2; the banded DCM buck under 'lcp' equals the default exactly (0.0). Other engines wait for 6.3
- [ ] 6.5 Ideal elements (with `add-branch-form-ideal-switch` task 3): the prototype oracles on `dsed` — DCM buck `v_C` within 1e-9 V of 8.9963970815477 V and turn-off within 1e-11·T; boost CCM within 1e-9 V; bridge event times within 1e-12 s of the closed form; inrush record 14.758493 J within 1e-9 relative

## 7. Gauge islands
- [ ] 7.1 Gauge detection per mode; gauge-free predicates; island potential at the centre of the allowed interval; `diagnostics['gauge_nodes']`
- [ ] 7.2 Test: single-phase bridge with ideal diodes, `v(p)` during the off interval is the centre value on every engine (the three engines agree within 1e-9 V; today 1.9e-10 V against 47.586 V)

## 8. Cost gate
- [x] 8.1 Counters in `result.diagnostics`: `lcp_count`, `lcp_pivots_total`, `max_pivots_per_event`, `ties`, `widenings`, `enumerations`. Slice 2 adds `leak_pins`, `pin_judgements`, `unfixed`, `enumerations`, `nonunique_modes` (`mode_selection` and `mode_shadow`)
- [ ] 8.2 On the five families: pivots per event within the table of design D9 (+1); solves per event against today's settle rounds on the same events; ns per event reported
- [ ] 8.3 60-diode rectifier bank: max pivots per event and wall time per event reported; gate: no event over 4·k pivots

## 4b. Default-mode guard (slice 2)
- [x] 4b.1 After each settle on dsed the settled mode is judged (DECISIONS #9) from the projected state; an inconsistent one is decided by the LCP from the held state and recorded in `diagnostics['mode_guard']`; an LCP refusal there names the diode (DECISIONS #15)
- [x] 4b.2 Consistent runs bit-identical with the guard on and off (`probes/guard_survey.out`: 24 correct circuits; `probes/bit_identity.py`: 32 of 36 identical, the 4 others are the repaired boost and single-phase bridge); repairs: boost V_th = 0 (16 at 50 Ω, 31 at 500 Ω, 7 at 20 Ω), half-wave RL rectifier (2), single-phase bridges (19–20, ≤ 1.2e-12 V), three-phase rectifier example (9, 1.1e-12 V)
- [ ] 4b.3 Same guard on pwl / trbdf2 (needs the 4.2 hooks)

## 9. Default flip (in the single number-changing release)
- [ ] 9.1 Default `mode_selection='lcp'` on every engine; `'legacy'` warns
- [ ] 9.2 Re-baseline through the reference harness; every changed number listed in the CHANGELOG with its cause
- [ ] 9.3 One release later: delete the settle loops, caps and chatter guards; `'legacy'` refused

## 10. Docs and verification
- [ ] 10.1 Device guide: the diode law, what `V_th` means now, the gauge convention; AI_GUIDE trap list (bands are no longer needed)
- [ ] 10.2 ruff, pytest, ctest (`-j2`), `openspec validate --all --strict`
