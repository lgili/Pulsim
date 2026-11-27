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

10. **Honest acceptance (2026-11-13).**
   (a) The row scale |J||x_0| + |b| + |f_nl(x_0)| + |J_nl||x_0| and the
   dx cap 1e-6·max(1, |x_0|) are taken once per call, at the start
   state (iteration 0) and the step's sources, never at the iterate
   being checked: read at a diverging iterate they grew with it.
   (b) Stagnation is two-sided: no net contraction by 4 over two steps
   AND no growth (the largest ||dx|| of the last three is at most twice
   the largest of the three before). A doubling ||dx|| at roundoff
   residual is declined ("growing"; C++ test with a whole-ulp
   doubling sequence).
   (c) The state kept is x_k, the iterate whose residual was checked,
   not x_k + dx_k.
   (d) Run budget: each engine keeps one acceptance object per run (the
   fixed engine created one per step, so no count could span a run).
   More than `max_consecutive` = 130 accepts with no converged solve
   between them throws `NewtonRoundoffBudgetExceeded`, naming the row,
   the run length and the run's total. No caller retries it at a
   smaller step. It reaches Python as `SimulationAborted`. The budget
   is 10× the worst honest run. Measured over the full suite: longest
   run 13 and total 129 of 200 solves (10 mH at dt = 1e-10 s,
   `test_newton_roundoff_floor.py`). The saturable transformer at
   dt = 1e-8 s has a longest run of 4 and a total of 436 of 226,005
   solves.
   ll16_flyback_closed_loop on pwl: longest run 5, total 1,773 of 7,999
   solves. That is inside every budget an honest case allows: its
   longest run is below the honest worst, and its total is 4× the
   transformer's. ll16 therefore stays quiet under this check. Its
   cause is not the acceptance. The v1 reader starts it from
   `compute_dc_op`, and a voltage source across series inductors (Vin,
   Lleak, Lm) returns 2.4e13 A instead of a refusal
   (`probes/ll16_dc_op.py`). The solves at that state really are at
   roundoff. The defect is in the DC operating point; KNOWN_LIMITATIONS
   records it.
   (e) The Python warning no longer says the acceptance "replaces an
   abort that no tolerance could have passed". 100 µH at dt = 1e-11 s
   now converges on the in-house LU. The warning names the start-state
   scale, says that a wrong start state can be accepted all the way,
   and gives the budget.

11. **Stagnation is judged over WINDOWS, not two steps (orchestrator
   decision, 2026-11-20; amends 8 and 10b).** Release finding F-S-5
   (`update-number-changing-release/fixes/f-s-5.md`): a half bridge with
   charge-law output capacitances aborted at h_min on 9 of 30 run lengths
   (trbdf2) and 1 of 30 (esdirk43), and at rtol 1e-7 / 1e-8 / 1e-10 /
   1e-9. Traced per iteration: at the failing step Newton cycles EXACTLY
   (period 6 at t_end 1.3 µs, 7 at 2.7 µs, 10 at 5.23 µs), ||dx||
   3.1e-7 … 1.1e-5, residual 4.7e-7 … 1.4e-5 A, row backward error
   0.08 … 2.44 eps at every iterate — the companion's representation floor
   eps·Q_Ch/(γ·h_min) ≈ 2–5e-6 A against the absolute 1e-9 A. The solve
   is converged to what double precision gives; refusing it shrinks h, and
   a smaller h only RAISES that floor, so the refusal could only end at
   h_min. Rule 8's two-step test read the phases where ||dx_k|| < ¼
   ||dx_{k−2}|| (e.g. 9.1e-6, 8.7e-7, 5.2e-7) as "still contracting"; rule
   10b's three-vs-three maxima read a period-10 cycle as "growing".
   New rule, in the one place the solver decides it
   (`pwl/nonlinear_solve.hpp`, shared by both engines): with k =
   min(16, iterations/2), the newer window holds the last k ||dx||, the
   older one the k before. NO IMPROVEMENT: min(newer) ≥ ¼·min(older).
   NOT GROWING: max(newer) ≤ 2·max(older). Both, plus the unchanged
   omega ≤ 8 eps (at x_k, against the start state's row scale) and
   ||dx_k|| ≤ 1e-6·max(1, |x_0|), accept x_k.
   WHAT SEPARATES A CYCLE AT THE FLOOR FROM SLOW CONVERGENCE: in a cycle of
   period ≤ k both windows contain a whole period, so their best ||dx||
   is the SAME (ratio 1 in all three traces); in convergence the best
   iterate keeps improving, so the newer window's best is below the
   older's by the rate to the k-th power. A geometric sequence of rate r
   is declined as long as r^k < ¼; two steps declined only r² < ¼, so
   the window is STRICTER on every monotone convergence than rule 8 was
   (rate 0.7 at a roundoff residual: rule 8 accepted it, the window
   declines it, C++ test "slow convergence near the floor is declined").
   Not covered: a cycle of period > k (it may still be declined; budget
   50 gives k = 16), and a slowly drifting iterate at rate r ≥ 4^(−1/16)
   = 0.917 per iteration with a residual already at ≤ 8 eps: accepted,
   as rule 8 accepted it — its backward error is the guarantee, not the
   iterate's distance to the root.
   Unchanged: transient only (the DC cascade passes no acceptance; the C++
   scan test and `test_newton_stall_and_mask_chatter.py`'s spurious-root
   test stay green); the absolute tol_dx is untouched; the run budget
   (10d) and the state kept (10c) are as before.
   AMENDMENT (2026-11-20, session 3, ll13 regression): the comparison is
   made for every split k from min(8, kw) to kw, and the solve counts as
   stagnated if ANY split shows no improvement and no growth (k = kw alone
   was rule 11 as first written; every solve it accepted is still
   accepted). Traced on ll13_active_clamp_forward (trbdf2): the first solve
   the two rules decided differently (#85, t = 5.500006 us, h = 3.2e-13 s)
   CONVERGED (||dx|| 3.1e-2 -> 1.8e-5) and then sat at 1.19e-8 V for 15
   iterations at omega 0.31 eps; its older window of 16 held the
   convergence, so the floor read as "still contracting" and the decline
   walked the run into a genuine Newton failure at 41 us. A second shape
   (#112, #266): a cycle at 1.2-1.4e-6 with ONE step of 5.4e-8 inside the
   newer window. Both are accepted by the split of 8. Monotone protection:
   at every k >= 8 a geometric rate r is declined while r^8 < 1/4, i.e.
   r < 0.84 (rule 8 accepted r >= 0.5; the rate-0.7 C++ test is still
   declined). Not changed: the 8 eps limit, the absolute tol_dx, the dx cap
   1e-6·max(1,|x_0|) (largest ||dx_k||/cap over ll13's accepts 0.04: not
   the cause), the state kept (x_k: it was the best-omega iterate of its
   window in 291 of 345 accepts; where not, both were below 7 eps).
   The decline message now prints the windows' minima and maxima (and
   "too few for two windows" under four iterations).

12. **A floating-point fixed point is also judged at its own magnitudes
   (gate-edge line, 2026-11-20; amends 10a for this case only).**
   ll13_active_clamp_forward on trbdf2 aborted at h_min on 14f at rtol
   1e-7 (45 µs) and esdirk43 rtol 1e-6 (34 µs), and on the gate-edge merge
   at the defaults (21 µs). Each abort has the same signature: a dead time
   drives ~40 A into roff = 1e7, the drain and secondary reach 3.8e8 V
   within the solve, ||dx|| is constant at 4.66e-9 V for 32 iterations
   (below one ulp of the state, 6e-8 V: x_k + dx_k == x_k), and a 1 kS
   diode row between two such nodes leaves 2.4e-5 A of cancellation. That
   is 0.1 eps of the row's magnitudes at x_k but 1e6 eps of the START
   state's, taken before the spike: declined, h shrank to h_min, abort.
   Which tolerance met such a solve was a lottery of the step sequence
   (env-switched trace build: without this rule every variant of the
   merge aborted at some tolerance).
   Rule: when the windows say stagnated, the dx cap holds, and every
   component of the last Newton step is within the absolute tol_dx or
   within 4 eps of the iterate's own value, the row backward error is
   taken against max(start-state scale, scale at x_k) per row. 10a stands
   for every other iterate: a diverging iterate never steps at its own
   rounding. The absolute dx test is unchanged.
   Measured: ll13 runs at trbdf2 default / rtol 1e-7 / esdirk43 rtol 1e-6
   with 1.09e-4 / 9.4e-6 / 1.02e-4 of range against pwl + restart at
   dt 5e-10 at 25 smooth instants (14f's default: 1.09e-4).
   Not covered: a genuine Newton cycle at h_min (||dx|| 1.6e-4 V at
   5.6e6 V, 1.4e3 eps, dx above the cap) still aborts; it appeared on
   ll13 only under trace-build variants that are not shipped.
