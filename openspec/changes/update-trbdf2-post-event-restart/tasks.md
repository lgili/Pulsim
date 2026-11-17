## 1. Characterisation (committed first)
- [x] 1.1 `python/tests/test_trbdf2_post_event_restart.py`: pin `p6_ratchet.py` — buck CCM, 400 periods, default tolerances: 7241 accepted steps (18 per period), 16262 solves, rising-edge post-edge step 8.26e-13 s at every listed period; strict xfail of the targets in 4.1
- [x] 1.2 Pin `p5_trbdf2_edges.py` DCM: the run of consecutive steps between 4e-14 and 1e-12 s; strict xfail of "no accepted step below 1e-11 s"
- [x] 1.3 Record the output error against the `expm` reference at rtol 1e-4, 1e-6, 1e-8 (8e-4, 9e-5, 2e-6 V) as the accuracy "before" — `probes/trbdf2_baseline.json` (max of the i_L and v_C end errors: 8.46e-3, 1.31e-4, 2.09e-6)
- [x] 1.4 Reproduce the `h_min` stall (`test_saturable_transformer.py::test_engines_agree_below_saturation`) and record the step sequence of the last 20 steps before the abort — `probes/hmin_stall.py`: on this branch it aborts at t = 24 µs (not 4 µs), `||dx||` = 3.729e-7 at T1.pm, residual 6.1e-5 on T1.m (3.2e-15 relative); last 20 steps rise from 2.2e-8 to 4.1e-7 s then shrink to 2.5e-7 s before the Newton rejects walk h to h_min

## 2. Derivative refresh, linear L/C
- [x] 2.1 `HistoryState::refresh_derivatives(x_probe, h_probe)`; C++ test on an RL and an RC with a mask change: the refreshed slot equals the analytic `v_L(t⁺)` and `i_C(t⁺)`
- [x] 2.2 Call it after the cascade of a gate-edge landing, after a diode-event landing and after the initial settle; take one probe when no cascade ran; every such solve goes through `trap_solve` — **gate-driven events only** (landed edge, boundary settle): at diode events and the initial settle it was implemented, measured harmful and removed (DECISIONS.md 6)
- [x] 2.3 `TrBdf2Stats`: `n_derivative_refreshes`, `n_refresh_probes`, `h_post_event_min`; binding and stub (the stats are a dict: no stub change)

## 3. Derivative refresh, stateful devices
- [x] 3.1 Nonlinear capacitor and saturable inductor slot refresh, with a test each — the Coss case is `test_gate_edge_refreshes_a_stateful_device[coss]`; the saturable-inductor case could not be tested end to end: as a buck's inductor on TR-BDF2 it loses its whole current at a turn-off edge with and without this change (pre-existing; the edge cascade's Newton probe fails silently), reported
- [x] 3.2 Lauritzen diode, IGBT tail, PMSM slot refresh, with a test each — implemented for all three; tested: Lauritzen (`[lauritzen]`, which aborted at h_min before the change and now completes). IGBT tail: the gated test circuit aborts at h_min at the first turn-off with and without the change (the documented stall); PMSM: no gated PMSM circuit was built (its diode-event test still passes)
- [x] 3.3 `test_trbdf2_stateful_with_diodes.py` still passes; add one case per device with a gate edge (two of five, see 3.1/3.2)
- [x] 3.4 Roundoff: buck with a 10 mF, 400 V capacitor at rtol 1e-8; record whether the lengthened refresh probe of design D1 is used and whether the step count changes without it — `probes/cap_roundoff.py` (800 V buck, 10 mF, 40 periods; v_out only reaches 32 V): the lengthened probe is not implemented; at rtol 1e-8 1054 accepted steps (1712 before) with the same v_out error against pwl (5.8e-4 V, the reference's own)
- [x] 3.5 (not implemented; see DECISIONS.md 5) Gate: only if 4.1 or 4.3 fails for a circuit because no usable probe exists, implement the SDIRK(2,2) first step (design D3) for that case

## 4. Proposal memory and acceptance
- [x] 4.1 Store the controller's proposal in `h_mem_gate`; un-xfail 1.1 with: post-edge step above 1e-8 s at every listed period of `p6_ratchet.py`; at most 8 accepted steps per period (prototype 4.1); output error at the end not above 2× the value recorded in 1.3
- [x] 4.2 Before/after measurement: accepted steps, solves and factorisations for the 400-period buck (the 3–4× fewer solves is an estimate; the prototype gives 3.5×) — 7241 → 1636 accepted steps (4.1 per period), 16262 → 5018 solves (3.2×, probes included), 82 → 65 rejects, error 9.84e-4 → 7.18e-5; factorisations are not counted by the engine
- [ ] 4.3 DCM buck: no accepted step below 1e-11 s; `i_L` error stays at or below 1e-9 A (1e-10 today) — **not met**: smallest step still 4e-14 s (after diode turn-offs, not refreshed); 15242 → 8932 accepted steps, error 2.39e-4 → 1.62e-5; strict xfail kept
- [x] 4.4 Cross-engine: buck CCM and DCM, flyback, bridge rectifier: time-weighted `v_out` (`np.trapezoid`) of `'trbdf2'` against `'pwl'` at T/800 and `'dsed'`, agreement not worse than before this change — `tw_err` column of the baseline (CCM 5.94e-4 → 1.51e-5, DCM 6.08e-4 → 9.84e-5, bridge unchanged; dsed refuses the V_th = 0 diodes at t = 0, pwl at fine dt used); flyback via `test_engine_auto.py::test_flyback_snubber_parity_and_spike` (passes)
- [x] 4.5 (diode events have no memory and no refresh; the bridge and the MOSFET half bridge are bit-for-bit unchanged) The flyback's diode-event count does not grow (diode events still have no memory)

## 5. The stall (reported, not promised)
- [x] 5.1 Re-run 1.4. Report: does it still abort, at which step size, and with which `||dx||` and residual. Delete the known-limitations row and un-xfail the test only if it passes — still aborts, now at t = 10 µs at h_min (last steps 4.13e-9 s, then rejected to 1e-14 s), `||dx||` = 2.286e-7 at T1.pm, residual 7.6e-6 on T1.m (1.2e-15 relative); row and xfail kept

## 6. Documentation
- [x] 6.1 `CHANGELOG.md` **BREAKING (step sequence)** with the before/after table
- [x] 6.2 `docs/solvers.md`: the refresh and the memory; `AI_GUIDE.md` trap 13 unchanged; `docs/known-limitations.md` per 5.1 (row kept unchanged)

## 7. Verification
- [x] 7.1 `ruff check`, `pytest` (throttled), `ctest`, stub check; reference harness on the variable-step circuits — ruff clean; pytest 2016 passed, 4 skipped, 9 xfailed; ctest 749/749; stubs up to date
- [x] 7.2 `openspec validate update-trbdf2-post-event-restart --strict` and `openspec validate --all --strict`
