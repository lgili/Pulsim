## 1. Characterisation (committed first)
- [x] 1.1 `python/tests/test_trbdf2_post_event_restart.py`: pin `p6_ratchet.py` — buck CCM, 400 periods, default tolerances: 7241 accepted steps (18 per period), 16262 solves, rising-edge post-edge step 8.26e-13 s at every listed period; strict xfail of the targets in 4.1
- [x] 1.2 Pin `p5_trbdf2_edges.py` DCM: the run of consecutive steps between 4e-14 and 1e-12 s; strict xfail of "no accepted step below 1e-11 s"
- [x] 1.3 Record the output error against the `expm` reference at rtol 1e-4, 1e-6, 1e-8 (8e-4, 9e-5, 2e-6 V) as the accuracy "before" — `probes/trbdf2_baseline.json` (max of the i_L and v_C end errors: 8.46e-3, 1.31e-4, 2.09e-6)
- [x] 1.4 Reproduce the `h_min` stall (`test_saturable_transformer.py::test_engines_agree_below_saturation`) and record the step sequence of the last 20 steps before the abort — `probes/hmin_stall.py`: on this branch it aborts at t = 24 µs (not 4 µs), `||dx||` = 3.729e-7 at T1.pm, residual 6.1e-5 on T1.m (3.2e-15 relative); last 20 steps rise from 2.2e-8 to 4.1e-7 s then shrink to 2.5e-7 s before the Newton rejects walk h to h_min

## 2. Derivative refresh, linear L/C
- [ ] 2.1 `HistoryState::refresh_derivatives(x_probe, h_probe)`; C++ test on an RL and an RC with a mask change: the refreshed slot equals the analytic `v_L(t⁺)` and `i_C(t⁺)`
- [ ] 2.2 Call it after the cascade of a gate-edge landing, after a diode-event landing and after the initial settle; take one probe when no cascade ran; every such solve goes through `trap_solve`
- [ ] 2.3 `TrBdf2Stats`: `n_derivative_refreshes`, `n_refresh_probes`, `h_post_event_min`; binding and stub

## 3. Derivative refresh, stateful devices
- [ ] 3.1 Nonlinear capacitor and saturable inductor slot refresh, with a test each
- [ ] 3.2 Lauritzen diode, IGBT tail, PMSM slot refresh, with a test each
- [ ] 3.3 `test_trbdf2_stateful_with_diodes.py` still passes; add one case per device with a gate edge
- [ ] 3.4 Roundoff: buck with a 10 mF, 400 V capacitor at rtol 1e-8; record whether the lengthened refresh probe of design D1 is used and whether the step count changes without it
- [ ] 3.5 Gate: only if 4.1 or 4.3 fails for a circuit because no usable probe exists, implement the SDIRK(2,2) first step (design D3) for that case

## 4. Proposal memory and acceptance
- [ ] 4.1 Store the controller's proposal in `h_mem_gate`; un-xfail 1.1 with: post-edge step above 1e-8 s at every listed period of `p6_ratchet.py`; at most 8 accepted steps per period (prototype 4.1); output error at the end not above 2× the value recorded in 1.3
- [ ] 4.2 Before/after measurement: accepted steps, solves and factorisations for the 400-period buck (the 3–4× fewer solves is an estimate; the prototype gives 3.5×)
- [ ] 4.3 DCM buck: no accepted step below 1e-11 s; `i_L` error stays at or below 1e-9 A (1e-10 today)
- [ ] 4.4 Cross-engine: buck CCM and DCM, flyback, bridge rectifier: time-weighted `v_out` (`np.trapezoid`) of `'trbdf2'` against `'pwl'` at T/800 and `'dsed'`, agreement not worse than before this change
- [ ] 4.5 The flyback's diode-event count does not grow (diode events still have no memory)

## 5. The stall (reported, not promised)
- [ ] 5.1 Re-run 1.4. Report: does it still abort, at which step size, and with which `||dx||` and residual. Delete the known-limitations row and un-xfail the test only if it passes

## 6. Documentation
- [ ] 6.1 `CHANGELOG.md` **BREAKING (step sequence)** with the before/after table
- [ ] 6.2 `docs/solvers.md`: the refresh and the memory; `AI_GUIDE.md` trap 13 unchanged; `docs/known-limitations.md` per 5.1

## 7. Verification
- [ ] 7.1 `ruff check`, `pytest` (throttled), `ctest`, stub check; reference harness on the variable-step circuits
- [ ] 7.2 `openspec validate update-trbdf2-post-event-restart --strict` and `openspec validate --all --strict`
