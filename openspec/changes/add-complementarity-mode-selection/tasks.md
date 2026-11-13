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
- [ ] 1.1 Port `probes/lcp_core.py` (Lemke, enumeration, port model) to `python/tests/oracles/lcp_reference.py` as a test-only oracle; unit tests: on 200 random PSD instances (k ≤ 8) Lemke and enumeration agree; on the LCPs dumped from the five families both return the same mode
- [ ] 1.2 Port `probes/today_dsed.py` as characterisation tests of today's kernel: `dsed` with `V_th = 0` fails "did not settle" at rest and "is chattering" at 0.996 ms on the DCM buck; marked to flip at the default stage

## 2. ModeSelector in C++ (dense, standalone)
- [ ] 2.1 `core/include/pulsim/events/mode_selector.hpp`: dense Lemke with the lexicographic ratio test and warm start from a given basis; enumeration for `k ≤ 12`; circuit-scale tolerances (D4)
- [ ] 2.2 Catch2 tests on the dumped `(M, q)` instances of task 1.1: same mode as the oracle, same pivot count within ±1, secondary ray reported on the infeasible instance; zero pivots when `q ≥ 0`

## 3. Port model on the conductance form, and the `h_e` band
- [ ] 3.1 Build `(M, q)` from the event's companion matrix (BE at `h_e`) through the cached factor: one solve for `q`, one per requested column; hybrid coordinates from the previous mode
- [ ] 3.2 Measure, on the five families with resistor-pair diodes (`R_on = 1/g_on`), the range of `h_e/T_ref` over which every event returns the same mode; set the default `h_e` inside it and record the band in this file
- [ ] 3.3 Tie widening (100×, 1e4×), OFF on a remaining tie, counted

## 4. Shadow mode (first user-visible slice; numbers unchanged)
- [ ] 4.1 `mode_selection='shadow'` on `dsed` settle: after today's settle, the selector runs on the same held state; disagreements in `result.diagnostics['mode_shadow']`
- [ ] 4.2 Same on the fixed-step event iteration, TR-BDF2 settle and DC seeding
- [ ] 4.3 Tests: on the buck, boost, single-phase and three-phase bridges with `V_th = 1e-2` (runs today on every engine) the shadow reports 0 disagreements; with `V_th = 1e-6` it reports each event where today's settle needed more than one round; states are bit-identical to `mode_selection` unset

## 5. Passivity and refusals
- [ ] 5.1 Structural passivity classification at build time (device kinds, parameter signs, coupled-inductor matrix PSD)
- [ ] 5.2 Non-passive: Lemke, then enumeration to `k = 12`, then `UnsupportedError(engine, 'mode_selection', device, 'non_passive_ports')` naming the controlled source
- [ ] 5.3 No consistent mode on a passive circuit (secondary ray) → typed error naming the diodes and `t`; test with a current source forced into two blocking diodes

## 6. Opt-in `mode_selection='lcp'`
- [ ] 6.1 `dsed`: settle replaced by the selector; the chatter guard stays as an assertion that never fires
- [ ] 6.2 Tests on `dsed` (resistor pair, `V_th = 0`): buck at rest runs and stays at 0; DCM buck 4000 periods runs; boost `R = 50` and `500` with `V_th = 0` run; L·(1 ± δ), δ ∈ {1e-5, 1e-4, 1e-2}, all six complete; single- and three-phase bridges with `V_th = 0` run
- [ ] 6.3 Fixed engine and TR-BDF2 with `mode_selection='lcp'`; the same tests; mask-cycle and round caps unused (counter 0)
- [ ] 6.4 Hysteresis (D8): DCM buck with `V_th = 0.5 V`, `V_f = 0`: the diode turns on at `v = 0.5 V` (located within the event tolerance) and off at `i = 0`; result equals today's within 1e-9 where today runs
- [ ] 6.5 Ideal elements (with `add-branch-form-ideal-switch` task 3): the prototype oracles on `dsed` — DCM buck `v_C` within 1e-9 V of 8.9963970815477 V and turn-off within 1e-11·T; boost CCM within 1e-9 V; bridge event times within 1e-12 s of the closed form; inrush record 14.758493 J within 1e-9 relative

## 7. Gauge islands
- [ ] 7.1 Gauge detection per mode; gauge-free predicates; island potential at the centre of the allowed interval; `diagnostics['gauge_nodes']`
- [ ] 7.2 Test: single-phase bridge with ideal diodes, `v(p)` during the off interval is the centre value on every engine (the three engines agree within 1e-9 V; today 1.9e-10 V against 47.586 V)

## 8. Cost gate
- [ ] 8.1 Counters in `result.diagnostics`: `lcp_count`, `lcp_pivots_total`, `max_pivots_per_event`, `ties`, `widenings`, `enumerations`
- [ ] 8.2 On the five families: pivots per event within the table of design D9 (+1); solves per event against today's settle rounds on the same events; ns per event reported
- [ ] 8.3 60-diode rectifier bank: max pivots per event and wall time per event reported; gate: no event over 4·k pivots

## 9. Default flip (in the single number-changing release)
- [ ] 9.1 Default `mode_selection='lcp'` on every engine; `'legacy'` warns
- [ ] 9.2 Re-baseline through the reference harness; every changed number listed in the CHANGELOG with its cause
- [ ] 9.3 One release later: delete the settle loops, caps and chatter guards; `'legacy'` refused

## 10. Docs and verification
- [ ] 10.1 Device guide: the diode law, what `V_th` means now, the gauge convention; AI_GUIDE trap list (bands are no longer needed)
- [ ] 10.2 ruff, pytest, ctest (`-j2`), `openspec validate --all --strict`
