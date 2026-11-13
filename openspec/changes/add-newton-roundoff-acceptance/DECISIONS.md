# Decisions taken while implementing (no owner answer yet)

None of the open questions was marked "proposed"; the most
conservative option was taken.

1. **Q1: thresholds configurable?** → Fixed in code
   (`NewtonRoundoffAcceptance{omega_max = 8 eps, dx_rel_max = 1e-6}`,
   stagnation = two non-contracting steps), not exposed to Python.
   Reverse: add the fields to `SimulationOptions` / `TrBdf2Options`.
2. **Q2: stall reproducer assertion** → Moot: the reproducer does not
   pass (see 5). `test_engines_agree_below_saturation` stays a strict
   xfail and the known-limitations row stays.
3. **Q3/Q4: the step floor (`h_noise`, refusal of a user `dt` below
   it)** → Not implemented (tasks 4.x). The floor needs the fixed
   engine's dt-retry ladder and a refusal in `run_transient.hpp`
   (another track's file), and the acceptance alone already completes
   every case of `p7b_roundoff.py`. Reverse: n/a (follow-up).
4. **Where it applies.** On in the variable-step engine for every
   Newton solve (stages, probes, landings) and in the fixed engine's
   dynamic path (nominal and retry dt: two call sites in
   `run_transient.hpp`, a small hunk in another track's file). Never
   in the DC cascade (`continuation.hpp`, `dc_strategy.hpp`,
   `dc_assemble.hpp`) or the fixed engine's static path: they pass no
   `NewtonRoundoffAcceptance` and run the old loop. No
   `SimulationOptions` switch (task 2.4 asks for one, default true):
   the acceptance fires only where the run would otherwise abort.
   The LM "no improving step" exit is not covered (it throws before
   the budget is reached). Reverse: pass `nullptr` at the call sites.
5. **The documented h_min stall is not cured; finding.** The
   reproducer (`update-trbdf2-post-event-restart/probes/hmin_stall.py`,
   snubbed 48 V flyback with a saturable transformer, rtol 1e-6) used
   to abort at t = 10 µs. With this change 213 solves are accepted at
   roundoff (worst ω = 7.6 ε) and it runs to t = 494 µs of 1 ms, where
   it aborts at h_min with ω = 0.28 ε at the KCL row of T1.pm and
   ||dx|| = 1.9e-7 (within 1e-6·|x|), but the last three ||dx|| are
   3.16e-7, 5.03e-7, 1.87e-7: Newton is in a roundoff limit cycle, and
   the design's stagnation test (each of the last two steps at least
   half the previous one) reads the last step as contraction. A test
   on the net contraction over two steps (`‖dx_k‖ ≥ ¼‖dx_{k−2}‖`) would
   accept it; that is a change to the criterion the design fixed, so
   it is reported, not made. The abort message now says why the
   acceptance declined (ω, the dx ceiling, stagnation and the last
   three ||dx||), which is how this was measured.
6. **Characterisation and fix in one commit.** The p7b port fails
   without the change (every "aborted before" case raises) and passes
   with it; it was not committed separately as a strict xfail first.

## Follow-up (maintainer decisions, 2026-11-13)

7. **The documented stall is cured, by the probe handling, not by the
   acceptance.** With every failed Newton probe of the zero-time diode
   cascades retaken or refused (update-trbdf2-post-event-restart
   DECISIONS 8), `probes/hmin_stall.py` completes: 15.3104 V against
   15.31 V on pwl at 1e-8 s. The strict xfail is now a test (0.3 %),
   the known-limitations row is deleted and the `h_min` abort message
   no longer cites it (task 5.1). See 8 for the criterion change.
8. **Stagnation is NET contraction over two steps (maintainer decision
   2).** `stagnated = ||dx_k|| >= ¼·||dx_{k−2}||` replaces "each of the
   last two steps at least half the previous": quadratic or linear
   convergence cuts ||dx|| by far more than 4 in two steps, while the
   limit cycle 3.16e-7, 5.03e-7, 1.87e-7 of item 5 was read as
   converging. Transient only: the DC cascade still passes no
   acceptance, asserted by a C++ test that scans the six DC sources
   (absolute dx there is load-bearing). C++ test: a deterministic
   whole-ulp cycle (dx 6, 10, 4 ulps at a constant residual) is
   declined by the old rule (1.19e-6 vs 7.15e-7, 4.77e-7) and accepted
   by the new one at iteration budgets 9, 10, 11. Measured: the stall
   reproducer gives 15.3104 V with 431 roundoff accepts (ω ≤ 1.47e-15
   = 6.6 ε), 4 forced accepts (6), 9 probe retries (15), 109942 steps
   (109979). Converged runs: 0 accepts on all ten baseline circuits;
   the baseline table and the pwl/dsed hashes are unchanged.
9. **No step floor `h_noise` (maintainer decision 6).** The stall that
   motivated it was the failed cascade probe read as a state
   (post-event-restart DECISIONS 8), not a step below the noise floor:
   with that fixed the reproducer completes with its smallest step at
   1e-15 s, and no baseline circuit aborts at h_min. A floor would
   change every run's step sequence for no measured gain; the h_min
   abort already names the roundoff check that declined.
