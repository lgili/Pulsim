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
