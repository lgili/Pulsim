## 1. Characterisation (committed first)
- [x] 1.1 `python/tests/test_newton_roundoff_floor.py`: port `p7b_roundoff.py` — 100 µH runs at 3e-11 s and aborts at 1e-11, 1e-12, 1e-13 s; 10 mH runs at 1e-8 s and aborts at 1e-9 … 1e-11 s; `‖dx‖` and the row residual are the powers of two of design D1; strict xfail of "completes at every dt" — ported as `test_newton_roundoff_floor.py` with the fix (DECISIONS 6); fails without it
- [x] 1.2 Instrument the abort (test-only hook or message parse) to compute `ω` in the engine at the stalled iterate; record it for each case of 1.1 (expected from the reported numbers: 0.30–0.47 ε) — the abort message now reports ω when the acceptance declines; accepted: worst ω 0.60–0.73 ε (1.34e-16 … 1.63e-16)
- [x] 1.3 Reproduce the documented stall (`test_engines_agree_below_saturation`): record `h`, `‖dx‖`, the residual, the row, `ω` and `‖dx‖/‖x‖` at the abort. If `ω > 8ε` or `‖dx‖/‖x‖ > 1e-6`, report that this change does not address it and continue with 2–4 for the `p7b` case only — on this branch it aborted at t = 10 µs (||dx|| 2.29e-7 at T1.pm, residual 7.6e-6); ω ≤ 8ε and dx/|x| ≤ 1e-6 hold there
- [ ] 1.4 Record the five DC fall-through tests' current operating points (3 Python, 2 C++) as the reference for 3.4 — not recorded separately: the DC callers pass no acceptance and compile to the old loop (DECISIONS 4); the existing tests pass unchanged

## 2. Acceptance in the Newton loop
- [x] 2.1 `NewtonRoundoffAcceptance` argument (default off) and a result record on `solve_with_newton_b_extra`; the last three `‖dx‖` are kept; the criterion is evaluated only at the points that raise for non-convergence today (budget exhausted; the LM "no improving step" exit) — budget exhaustion only (DECISIONS 4)
- [x] 2.2 Row scale `s_i` and `ω`, computed once at that point; rows with `s_i = 0`
- [x] 2.3 Transient callers pass it (`run_transient.hpp:989`, `:1008`; `trbdf2_transient.hpp:716`); DC callers and the static path do not; a C++ test greps or asserts the DC call sites — follow-up: `Roundoff acceptance: the DC call sites pass none` scans the six DC sources (DECISIONS 8)
- [x] 2.4 `SimulationOptions::newton_roundoff_acceptance` (default true), `SimulationResult::newton_roundoff_accepts`; binding, stub, one Python warning per run naming the row — no options switch (DECISIONS 4); result fields count / worst ω / row; warning

## 3. Acceptance tests
- [x] 3.1 Un-xfail 1.1: every dt completes; `i_L` at the end equals the dt = 1e-8 run's value to 1e-6 relative; `newton_roundoff_accepts.count > 0` exactly for the dts that aborted before
- [x] 3.2 A run that converged before: `newton_roundoff_accepts.count == 0` and states bit-identical (buck with a Shockley diode at dt = 1e-7; saturable inductor flux-closure test)
- [x] 3.3 Controls that must still fail: the PSFB mask-chatter circuit at 10 ns still aborts with the chatter message; a solve capped at 5 iterations on a hard diode chain is not accepted
- [x] 3.4 The five DC fall-through tests of 1.4 are unchanged, operating points bit-identical
- [x] 3.5 `test_saturable_inductor_flux_closure.py`, the Jiles–Atherton and Langevin picoampere tests unchanged
- [ ] 3.6 (not measured) Measure Newton iterations per step on the cases of 3.1 (expected: the full budget of 50 on every accepted solve) and the wall time of the stall reproducer; report whether early acceptance is worth a follow-up

## 4. Step floor
- [ ] 4.1 (not done, DECISIONS 3 and 9) `h_roundoff` and `h_noise` from the committed inductor currents (linear and saturable); C++ test against `p7b_roundoff.py`: `h_roundoff` = 4.0e-11 s and 4.0e-9 s, `h_noise` = 4.0e-15 s and 4.0e-13 s
- [ ] 4.2 (not done, DECISIONS 3) Variable-step engine: committed steps of a Newton circuit not below `max(h_min, h_noise)`; `TrBdf2Stats::n_roundoff_floor`; probes exempt
- [ ] 4.3 (not done, DECISIONS 3) Fixed engine: the dt-retry ladder stops above `h_noise`; a user `dt` below it is refused by name (10 mH at 1e-13 s)
- [ ] 4.4 (not done, DECISIONS 3) Test: two series 10 mH inductors and a Shockley diode on `engine='trbdf2'` — no committed step below 4e-13 s, the run completes, and a 1 H / 100 A / 400 V circuit keeps a floor below 2e-10 s

## 5. The documented stall (reported, not promised)
- [x] 5.1 Re-run 1.3 with the change. If it completes: un-xfail `test_engines_agree_below_saturation`, delete the "TR-BDF2 stalls at `h_min`" row of `docs/known-limitations.md`, and update the abort message at `trbdf2_transient.hpp:1268-1280`. If it does not: leave all three and write down what was measured — does not complete: runs to 494 µs (213 accepts), then a roundoff limit cycle (DECISIONS 5); all three left; follow-up: completes once failed cascade probes are handled (DECISIONS 7) — 15.3104 V vs 15.31 V; xfail flipped, row deleted, message updated

## 6. Documentation
- [x] 6.1 `CHANGELOG.md`: transient Newton accepts an iterate at roundoff level and reports it; step floor; the DC cascade and the tolerances are unchanged
- [x] 6.2 `docs/gotchas.md` and `AI_GUIDE.md` §6.4 (gotchas only: AI_GUIDE is 32 bytes under its 32 KiB limit): what a roundoff-acceptance warning means; "do not loosen `tol_newton_dx`" stays
- [x] 6.3 `solver/options.hpp` comments on `tol_newton_dx`: add the roundoff bound and this acceptance

## 7. Verification
- [ ] 7.1 `ruff check`, `pytest` (throttled), `ctest`, stub check
- [ ] 7.2 `openspec validate add-newton-roundoff-acceptance --strict` and `openspec validate --all --strict`

## 8. Follow-up: honest acceptance (DECISIONS.md 10)
- [x] 8.1 Scale and dx cap from the call's start state and sources
- [x] 8.2 Two-sided stagnation test; C++ test: a doubling ||dx|| is declined
- [x] 8.3 The accepted state is the checked iterate x_k
- [x] 8.4 Run budget (130 consecutive, 10× the worst honest run) raises `NewtonRoundoffBudgetExceeded` naming row and count; C++ test; Eigen 3.4.0 run of the Newton tests
- [x] 8.5 ll16: cause found (compute_dc_op on a source across series inductors, `probes/ll16_dc_op.py`), known-limitations row; it stays within the budget
- [x] 8.6 Python warning text; test parameter renamed `needs_acceptance`
