## 1. Characterisation (committed first)
- [ ] 1.1 `python/tests/test_newton_roundoff_floor.py`: port `p7b_roundoff.py` — 100 µH runs at 3e-11 s and aborts at 1e-11, 1e-12, 1e-13 s; 10 mH runs at 1e-8 s and aborts at 1e-9 … 1e-11 s; `‖dx‖` and the row residual are the powers of two of design D1; strict xfail of "completes at every dt"
- [ ] 1.2 Instrument the abort (test-only hook or message parse) to compute `ω` in the engine at the stalled iterate; record it for each case of 1.1 (expected from the reported numbers: 0.30–0.47 ε)
- [ ] 1.3 Reproduce the documented stall (`test_engines_agree_below_saturation`): record `h`, `‖dx‖`, the residual, the row, `ω` and `‖dx‖/‖x‖` at the abort. If `ω > 8ε` or `‖dx‖/‖x‖ > 1e-6`, report that this change does not address it and continue with 2–4 for the `p7b` case only
- [ ] 1.4 Record the five DC fall-through tests' current operating points (3 Python, 2 C++) as the reference for 3.4

## 2. Acceptance in the Newton loop
- [ ] 2.1 `NewtonRoundoffAcceptance` argument (default off) and a result record on `solve_with_newton_b_extra`; the last three `‖dx‖` are kept; the criterion is evaluated only at the points that raise for non-convergence today (budget exhausted; the LM "no improving step" exit)
- [ ] 2.2 Row scale `s_i` and `ω`, computed once at that point; rows with `s_i = 0`
- [ ] 2.3 Transient callers pass it (`run_transient.hpp:989`, `:1008`; `trbdf2_transient.hpp:716`); DC callers and the static path do not; a C++ test greps or asserts the DC call sites
- [ ] 2.4 `SimulationOptions::newton_roundoff_acceptance` (default true), `SimulationResult::newton_roundoff_accepts`; binding, stub, one Python warning per run naming the row

## 3. Acceptance tests
- [ ] 3.1 Un-xfail 1.1: every dt completes; `i_L` at the end equals the dt = 1e-8 run's value to 1e-6 relative; `newton_roundoff_accepts.count > 0` exactly for the dts that aborted before
- [ ] 3.2 A run that converged before: `newton_roundoff_accepts.count == 0` and states bit-identical (buck with a Shockley diode at dt = 1e-7; saturable inductor flux-closure test)
- [ ] 3.3 Controls that must still fail: the PSFB mask-chatter circuit at 10 ns still aborts with the chatter message; a solve capped at 5 iterations on a hard diode chain is not accepted
- [ ] 3.4 The five DC fall-through tests of 1.4 are unchanged, operating points bit-identical
- [ ] 3.5 `test_saturable_inductor_flux_closure.py`, the Jiles–Atherton and Langevin picoampere tests unchanged
- [ ] 3.6 Measure Newton iterations per step on the cases of 3.1 (expected: the full budget of 50 on every accepted solve) and the wall time of the stall reproducer; report whether early acceptance is worth a follow-up

## 4. Step floor
- [ ] 4.1 `h_roundoff` and `h_noise` from the committed inductor currents (linear and saturable); C++ test against `p7b_roundoff.py`: `h_roundoff` = 4.0e-11 s and 4.0e-9 s, `h_noise` = 4.0e-15 s and 4.0e-13 s
- [ ] 4.2 Variable-step engine: committed steps of a Newton circuit not below `max(h_min, h_noise)`; `TrBdf2Stats::n_roundoff_floor`; probes exempt
- [ ] 4.3 Fixed engine: the dt-retry ladder stops above `h_noise`; a user `dt` below it is refused by name (10 mH at 1e-13 s)
- [ ] 4.4 Test: two series 10 mH inductors and a Shockley diode on `engine='trbdf2'` — no committed step below 4e-13 s, the run completes, and a 1 H / 100 A / 400 V circuit keeps a floor below 2e-10 s

## 5. The documented stall (reported, not promised)
- [ ] 5.1 Re-run 1.3 with the change. If it completes: un-xfail `test_engines_agree_below_saturation`, delete the "TR-BDF2 stalls at `h_min`" row of `docs/known-limitations.md`, and update the abort message at `trbdf2_transient.hpp:1268-1280`. If it does not: leave all three and write down what was measured

## 6. Documentation
- [ ] 6.1 `CHANGELOG.md`: transient Newton accepts an iterate at roundoff level and reports it; step floor; the DC cascade and the tolerances are unchanged
- [ ] 6.2 `docs/gotchas.md` and `AI_GUIDE.md` §6.4: what a roundoff-acceptance warning means; "do not loosen `tol_newton_dx`" stays
- [ ] 6.3 `solver/options.hpp` comments on `tol_newton_dx`: add the roundoff bound and this acceptance

## 7. Verification
- [ ] 7.1 `ruff check`, `pytest` (throttled), `ctest`, stub check
- [ ] 7.2 `openspec validate add-newton-roundoff-acceptance --strict` and `openspec validate --all --strict`
