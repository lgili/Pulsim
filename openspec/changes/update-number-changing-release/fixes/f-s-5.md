# F-S-5 — trbdf2 / esdirk43 abort on a half bridge with nonlinear C_oss (case A)

Branch `fix/core-trbdf2-coss-abort`, base cedac36b (`feat/core-integration`).
Test: `python/tests/test_release_findings_fs5.py` (case A of
`test_soft_switching_validation.py` on feat/core-integration-14t, copied with its
closed forms, plus rtol 1e-4 / 1e-8, `dt` 1e-10 and a 50 ns run).

## Reproduction (stored extensions)

| extension | trbdf2 | esdirk43 |
|---|---|---|
| b9e46778 | runs (books 1381.7 uJ: energy cell fails) | runs; rtol 1e-8 aborts at 1 us |
| be5454f3 | aborts at 1 us | runs |
| cedac36b (PR head) | aborts at t = 0 | aborts at t = 0 |
| D3 gate-edge branch (p5-igbt 6dffe5f4 extension) | aborts at t = 0 | aborts at t = 0 |

Reduced (no switch, no diode): `Vdc` ramp 0 -> 400 V in 100 ns, 1 mOhm vdc -> sw,
ONE nonlinear capacitor sw -> gnd, optional 10 A ramp out of sw. Runs at b9e46778,
aborts at t = 0 on cedac36b on both integrators, for m = 0.5 AND m = 0 (a linear
charge law). No dependent states: the t = 0 abort is not the capacitor loop.

Amplitude / stiffness scan of the reduced case (m = 0, linear law; cedac36b):
ramp 1 V and 4 V run; 40 / 400 / 4000 V abort (residual at h_min 2.9e-6 / 2.8e-6 /
3.6e-6 A, dx 3e-13 .. 3e-12); R = 1 mOhm and 10 mOhm abort, 100 mOhm runs; a DC
400 V source (inconsistent start) runs; the same ramp delayed by 10 ns runs; a dt
ceiling of 1e-13 .. 1e-16 s still aborts (residual 1.0e-9 .. 1.3e-9 against tol 1e-9).

Hand solution at t = 0+ (case A, Sh ON = 1 mOhm across Ch, all else OFF):
vdc = v_sw = 0 and I = 0 at t = 0, so the switch carries no current and the two
capacitors split the bus slope: C0 (dvdc/dt − v') = C0 v' ⇒ v'(0+) = 2e9 V/s,
i_Ch = i_Cl = 2 A, I(Vdc) = 2 A. In the reduced case (one capacitor) i_C(0+) = 0:
the zero derivative slots ARE consistent there, and it still aborts.

## Cause (b), the t = 0 abort since cedac36b: the charge law's cancellation near 0 V

Not the slots and not the dependent states (above). The abort signature is a Newton
that converged in dx (6e-13 .. 3e-12, halving) while the residual at node sw sits
at 2.8e-6 .. 3.5e-6 A, a backward error of 2e9 eps against the start state: the
residual does not move when v moves. With m = 0 the law is linear, so the only
non-linearity left is how Q is computed:

    Q = C0·V0/(1−m)·[pow(1 + v/V0, 1−m) − 1]

which cancels near 0 V: Q is held to C0·V0·eps absolute (1e-9·25·2.2e-16 = 5.5e-24 C;
m = 0.5: 1.1e-23 C), whatever v is. The companion divides it by the step
(i = (Q − Q_n)/(γs·h)), so the residual floor is C0·V0·eps/(γs·h): 1.9e-8 A at
h = 2e-15 s, 1e-5 A at h_min = 3e-18 s — above tol_res = 1e-9 A at every
h < 3.8e-14 s.

Why only since cedac36b: the run's first step is now the SDIRK restart step and it
carries the q-row estimate (its embedded first-order estimate). From rest the charge
grows as t² (the ramp behind a 1 ps RC), so that estimate is O(1) relative to Q at
every h and only the absolute part atol·C(0) = 1e-17 C accepts it: h ≈ 2e-15 s
(hand model of the stepper on the reduced case, `.scratch/f5/model2.py`: the step
sequence 3e-10 → … → 3.2e-15 (err 2.0) → 2.0e-15 (err 0.81, accepted), then TR-BDF2
steps of 2e-15 .. 1e-13 s, all with err < 1). There the law's floor (1.9e-8 A)
fails Newton; each Newton failure quarters h down to h_min. At b9e46778 the first
step was a TR step with no q estimate: no femtosecond step, no floor met.

Decisive experiment (cedac36b extension, reduced case, both integrators): the
cancellation quantum is C0·V0·eps, so with the LINEAR law (m = 0) V0 changes
nothing but the quantum. V0 = 2500 / 25 V abort; V0 = 2.5 / 0.25 / 0.025 /
0.0025 V run (m = 0.5: the same split). `.scratch/f5/v0scan.py`.

Fix: Q = C0·V0/(1−m)·expm1((1−m)·log1p(u)) — relative to Q to a few ulp at every
bias — in the model (`models/nonlinear_capacitor.hpp`, every engine's hand stamp)
and the interface law (`device/laws.hpp`, the same form, tangent C0·exp(−m·log1p(u))).

Trace with the debug build (per attempt: t, h, SDIRK?, Newton ok; per q row the
estimate ee and its scale; `PULSIM_OLDLAW=1` restores the old law), reduced case,
m = 0.5, trbdf2:

| law | first-step attempts (h: q-row err, row 0 = the capacitor) |
|---|---|
| pow − 1 (cedac36b) | 3e-12: 5.8e4; 3e-13: 1.3e4; 3e-14: 177; 3e-15: Newton fails (residual stuck at 3.0e-9 A at node sw, dx 2.6e-12), then 7.5e-16 … 3e-18 = h_min: Newton fails → abort |
| expm1(log1p) | 3e-12 … 3e-14 as above; 3e-15: 1.80; 2.0e-15: 0.81, accepted; then TR-BDF2 at err 1.3e-4 and growing; runs (1069 samples) |

The first-step sequence is the hand model's to the digit (`.scratch/f5/model2.py`:
err 5.845e4 at 3e-12). With the law fixed, the reduced case runs on both integrators
(m = 0 and 0.5), and case A runs on esdirk43 at the defaults, rtol 1e-4, rtol 1e-8
(which aborted at 1 us even at b9e46778) and dt 1e-10. TR-BDF2 now aborts at 1 us:

## Cause (a), the 1 us abort since be5454f3: the probe's slot after a hard turn-on

Trace (law fixed, trbdf2 defaults), the steps after the high-side turn-on at 1 us
(400 V across Ch shorted by Sh's 1 mOhm; Cl at −0.47 V). All TR-BDF2 steps (s = 0),
rows 0 = Ch (q 1.56e-7 C, scale 1.56e-12), 1 = Cl (q −4.7e-10 C, scale 4.7e-15):

| h | ee row 0 | ee row 1 | err (row 1 / its scale) | |
|---|---|---|---|---|
| 6.0e-15 | −4.92e-13 | −4.80e-13 | 32.7 | reject |
| 1.9e-15 | −1.48e-13 | −1.48e-13 | 31.4 | reject |
| 6.1e-16 | −4.72e-14 | −4.72e-14 | 10.0 | reject |
| 2.9e-16 | — | — | — | Newton fails |
| 7.1e-17 | −5.53e-15 | −5.53e-15 | 1.18 | reject |
| 6.8e-17, 2.2e-17 = h_min | — | — | — | Newton fails → abort |

The estimate is proportional to h (ee/h ≈ 80 A, constant): not a truncation error
(that falls as h³) but a wrong derivative slot f_n, the O(h) term
2·clte·h·f_n/γ. Its source: at a gate edge `restart_after_gate_event` commits a
TRAPEZOIDAL probe (or rewrites the slots by the trapezoidal rule), so a kind's
slot becomes 2·i+ − i− instead of the post-edge current i+, off by the jump; TR-BDF2's
first stage then integrates from it. Row 1 names it first only because Cl's charge
(near 0 V) gives it the smaller scale; both rows carry the same ee (the loop current
Vdc–Ch–Cl). At h ≈ 1e-16 the companion's representation floor eps·Q_h/(γh)
(3.5e-23 C / 4e-17 s ≈ 1e-6 A) is above Newton's 1e-9 A: abort. Before be5454f3 no
q row existed, nothing shrank the step, and the run booked 1381.7 uJ for 62.8 uJ.
Not the dependent states: per row the estimates are each right for a wrong slot.

Not this finding (recorded): two nonlinear capacitors in a loop with the bus and a
10 A source out of their mid node, NO diode or switch (`.scratch/f5/loop2.py`)
aborts at 54 ns on every build (b9e46778 too): the source drags Cl to −22.29 V,
the law's floor (u = −0.9, −22.5 V), where Q is flat by design (DECISIONS #26) and
nothing can sink the current. Case A's diodes clamp the node at −0.7 V.

## Fixes

- (b) `models/nonlinear_capacitor.hpp` and `device/laws.hpp` (model and interface
  path, Real and AD): Q = C0·V0/(1−m)·expm1((1−m)·log1p(u)). Same function, relative
  to Q to a few ulp at every bias; C++ test in `test_device_laws.cpp`.
- (a) `trbdf2_transient.hpp`: a mode-changing gate edge in a run with a stateful kind
  makes the next step the SDIRK restart (no slot read), TR-BDF2 only. On ESDIRK4(3)
  forcing it broke the defaults at 1.5 us: the SDIRK estimate reads the 700 A ZVS
  spike (0.7 V on Cl through Sl's 1 mOhm, tau 1 ps) as an O(jump) error, h fell to
  ~1e-16 s, below Ch's companion floor, Newton stalled. ESDIRK4(3) keeps its own
  restart (explicit stage from the committed probe), which ran with fix (b) alone.

Case A after both (2.2 us): trbdf2 crossings −12.34 / −0.51 / +1.28 ps,
E 62.8490 uJ; esdirk43 −7.62 / +0.54 / +15.39 ps, E 62.9259 uJ (closed form
62.8335; b9e46778 62.9246: 1.3 nJ further, within the 5 % cell). Knobs: all pass
except trbdf2 rtol 1e-8.

## Remaining: trbdf2 at rtol 1e-8 (strict xfail)

Aborts at the 1.5 us edge (Sl on across conducting Dl). The post-edge h is the
remembered same-parity h (`h_mem_gate`), 2.2e-17 s = h_min, because resolving the
1 ps edge to rtol 1e-8 needs it. There Ch's charge (1.56e-7 C) gives the companion a
floor eps·Q/(γh) ≈ 1e-6 A against Newton's absolute 1e-9 A. Roundoff acceptance
sees omega 0.23 eps (would accept) but declines on "still contracting": a period-~7
limit cycle whose last ||dx|| (5.4e-6, 5.2e-7, 3.1e-7) contract by chance. Next
step: a stagnation test over the cycle (e.g. a longer window) in
`pwl/nonlinear_solve.hpp`, a change to add-newton-roundoff-acceptance DECISIONS 8/10b
(maintainer criterion), so not made here.

## Fresh-context review (ef0ff76d) and state

Reviewer (scripts in `.scratch/review/`), sweeping t_end 1.1–4.0 us by 0.1 at the
defaults: trbdf2 aborts at 9 of 30 lengths (1.3, 2.7, 2.8, 3.2–3.5, 3.8, 4.0 us),
esdirk43 at 1 (3.3 us); be5454f3 13/30 and 0/30; b9e46778 0/30. At 2.2 us also
trbdf2 rtol 1e-7 / 1e-10 and esdirk43 rtol 1e-9 / 1e-10 abort. Every abort is the
same signature as "Remaining": Newton at h_min, row backward error 0.23–2.2 eps
(under 8 eps), roundoff acceptance declined "still contracting" or "growing". So
the companion floor eps·|Q|/(γh) vs absolute 1e-9 A, with the acceptance's
two-step stagnation window, is the main open cause at the defaults, not a 1e-8
corner. Also: a saturable inductor (1 mH, I_sat 5 A) + 1 mH cut-set opened by a
conductance switch at 200 us aborts on esdirk43 at every rtol (from cedac36b, not
this diff) and trbdf2 at rtol 1e-8 / 1e-10 (unattributed between cedac36b and the
flag); a ramp source across a capacitor reports the wrong-sign current sample at
the 100 ns corner on every build (slot 2·i+ − i−; esdirk43 source energy −2e-3).
Law (b): exact to 4.5e-16 vs mpmath on every branch, AD path included.

NEXT STEP: make the roundoff acceptance recognise a roundoff limit cycle (a window
covering the cycle, or the best iterate of the tail at omega <= 8 eps and dx within
the cap), which changes add-newton-roundoff-acceptance DECISIONS 8/10b, keep the
DC cascade excluded (absolute dx), then rerun the reviewer's t_end sweep
(`.scratch/review/q3_caseA3.py`) and remove the strict xfail. Second: commit the
probe's right-limit current to the slot instead of 2·i+ − i− (cause of (a) and of
the corner sample), which would make the TR-BDF2-only flag unnecessary.

## Session 2 (2026-11-20): the roundoff limit cycle — evidence and the window rule

STATE: WIP. `pwl/nonlinear_solve.hpp` carries the window rule plus TEMPORARY
instrumentation (`PULSIM_RO_TRACE`, `PULSIM_RO_OLD` = the old two-step rule,
`<cstdio>/<cstdlib>`, `ro_trace_buf`), all marked `TEMP`: remove before the
final build. Baseline for the byte compares (14b0c81a extension) recorded in
`.scratch/b14/base` (`.scratch/b14/rec.sh base record`; compare with
`rec.sh new compare`). Scripts: `.scratch/f5w/{ev.py,sweep.py,cut.py,cxx.sh}`.

Evidence (old rule, trbdf2 defaults, the last budget-exhausted solve, h = h_min
= span·1e-11): Newton cycles EXACTLY.
- t_end 1.3 us (h_min 1.3e-17 s): period 6, ||dx|| 1.11e-5, 9.06e-6, 8.74e-7,
  5.24e-7, 3.12e-7, 3.70e-7; residual 4.7e-7 … 1.4e-5 A; omega 0.08 … 2.44 eps.
  Floor eps·Q_Ch/(γh) = 2.2e-16·1.56e-7/(0.586·1.3e-17) = 4.5e-6 A.
  Two-step window: "still contracting" (5.24e-7 < ¼·9.06e-6).
- t_end 2.7 us (h_min 2.7e-17 s): period 7, ||dx|| 5.36e-6 … 1.50e-8; residual
  2.8e-7 … 6.6e-6 A (half of 1.3 us: amplitude ∝ 1/h); omega 0.10 … 2.44 eps;
  floor 2.2e-6 A. Two-step: "still contracting".
- t_end 5.2324 us (random sweep, first window rule k = 8): period 10; the
  8-vs-8 window read it as "growing" (max 2.76e-6 vs 2.19e-7) — hence k = 16.
Window minima were equal in both windows in all three (ratio 1.00).

Rule as in the code: k = min(16, iterations/2); accept x_k iff
min(last k ||dx||) ≥ ¼·min(k before) AND max(last k) ≤ 2·max(k before) AND
omega ≤ 8 eps AND ||dx_k|| ≤ 1e-6·max(1,|x_0|). 16 covers periods ≤ 16 at the
50-iteration budget (observed 6, 7, 10). DECISIONS 11 of
add-newton-roundoff-acceptance has the full statement. New C++ tests in
`test_newton_roundoff_acceptance.cpp`: a period-7 cycle accepted at budgets
30–36; slow convergence (rate 0.7, residual at 1.6 eps) declined "still
contracting" at budgets 20/25/30 (the two-step rule ACCEPTED it).

Results with k = 8 (before the period-10 case): run-length sweep 0/30 trbdf2,
0/30 esdirk43 (old rule, same build: 9/30 and 1/30, the reviewer's lengths);
tolerance sweep 1e-4 … 1e-10 no abort (old: trbdf2 1e-7/1e-8/1e-10 and esdirk43
1e-9/1e-10 abort); crossings within 15.6 ps of the closed form at every
rtol, → 0.01 ps at 1e-10; E_hard trbdf2 62.8490 (default) → 62.8367 uJ,
esdirk43 62.9259 (default, unchanged from session 1) → 62.8368 uJ, cf 62.8335
(esdirk43 rtol 1e-8 63.0082 is the same under the old rule). Random 200
lengths: trbdf2 1 abort (5.2324 us, the period-10 cycle), esdirk43 0.
Running: build with k = 16, then the three sweeps again (expect 0 aborts).

Results with k = 16 (final code, TEMP switches removed after these sweeps):
- run-length sweep 1.1–4.0 us: trbdf2 0/30, esdirk43 0/30 (b9e46778 0/30).
- tolerance sweep 2.2 us, rtol 1e-4…1e-10: no abort on either. Crossings
  (300/200/100 V) vs the charge closed form, ps: trbdf2 1e-4 −14.82/−0.07/
  +14.00, 1e-5 −12.34/−0.51/+1.28, 1e-8 −0.12/−0.01/+0.10, 1e-10 −0.01/0/+0.01;
  esdirk43 1e-4 −15.58/−0.51/+8.47, 1e-5 −7.62/+0.54/+15.39, 1e-10
  −0.02/0/+0.02 (all within 50 ps). E_hard vs 62.8335 uJ: trbdf2 62.8937 (1e-4),
  62.8490 (1e-5), 62.8367 (1e-9, 1e-10); esdirk43 63.1770 (1e-4), 62.9259
  (1e-5; b9e46778 62.9246, the +1.3 nJ is session 1's, the window rule leaves it
  unchanged), 63.0082 (1e-8, identical under the old rule), 62.8368 (1e-10).
  Tight-tolerance limit of both integrators: 62.8367–62.8368 (+3.2 nJ = 5e-5
  over the closed form, both agree). Roundoff accepts at 2.2 us: trbdf2 61
  (default) … 2679 (1e-10, of 10752 steps), esdirk43 2 … 1819; the run budget
  (130 consecutive) never fires.
- random 200 lengths in [0.2, 6] us, seed 20261120: 0 aborts on both (k = 8
  had 1, the period-10 cycle). 40 of them are a test
  (`test_case_a_runs_at_every_run_length`).
- strict xfail trbdf2 rtol 1e-8 removed: passes.
- Saturable cut-set (L1 atan 1 mH/5 A/0.1 mH + L2 1 mH, S opens at 200 us):
  esdirk43 now RUNS at the default and rtol 1e-8 (was the floor cycle).
  Flux-conserving closed form (mpmath) i+ = 0.908907 A from i1- = 1.882674,
  i2- = 1.9e-4; i(R) at 200.1 us = 0.909368 on esdirk43 (both rtol) and trbdf2
  (default, 1e-10): within 4.6e-4 (drift ≤ 4e-4 in 100 ns). The pwl+restart
  reference refuses by name (the restart re-initialises linear L only).
  trbdf2 rtol 1e-8 still aborts and is NOT the floor cycle: last ||dx||
  3.648e-9, 3.643e-9, 3.639e-9 at node b, omega 0.12 eps, declined "growing"
  (the newer window's max over 2x the older's): strict xfail with these
  numbers. Seen in passing: trbdf2's first post-edge row (200.0000276 us)
  reads 0.9412/0.9405 A against 0.9089 — recorded, not chased.

Integration note (step 5): `hs_edge_restart` (cause (a)'s workaround) is
untouched. When the branch that commits the probe's right-limit current to
the slot meets this one: remove the flag, then re-run `.scratch/f5w/sweep.py
len|tol|rand 200` and this file's tests; the window rule is independent of it.

## Session 2 verification (tip after 6e5241a6) and the OPEN REGRESSION

- C++ on Eigen 5 (fresh `.scratch/build-ctest`, targets only): layer5_v4 24
  cases (incl. the period-7 cycle and the protected slow-convergence case:
  declined "still contracting" at budgets 20/25/30), layer5_v3 35 cases (incl.
  session 1's law test in `test_device_laws.cpp`), step 32, sim 8, dsed 142:
  all pass.
- `test_release_findings_fs5.py` 27 passed + 1 strict xfail (cut-set trbdf2
  rtol 1e-8); `test_newton_stall_and_mask_chatter.py` (the 10.67 V spurious-root
  DC test) passes.
- Byte compares vs the 14b0c81a build: goldens ALL IDENTICAL, stateful ALL
  IDENTICAL, engines ALL IDENTICAL, pwl_identity 379/379, dsed bit-identity
  280/280, trbdf2_identity 369/371:
  - ll08_buck_mosfet (trbdf2): moved — 2020 → 1998 accepted steps, roundoff
    accepts 138 → 135 (solves that were declined and shrank h are now
    accepted). Not yet checked against a truth.
  - **ll13_active_clamp_forward (trbdf2): REGRESSION — ran at 14b0c81a (345
    roundoff accepts, 1538 steps), now aborts at t = 41 us**: Newton at h_min,
    ||dx|| 1.56e-4 at node sw_out, row backward error 3.4e5 eps (NOT a
    roundoff case; declined correctly). The new path reaches a state where
    Newton genuinely fails. Cause not traced.
- Fresh-context reviewer (opus): could not make the acceptance return a wrong
  answer (the 8 eps residual gate bounds any accepted state; the window is
  stricter than rule 8 on monotone convergence); its sweeps 60 lengths / 14
  rtols / 102 law variants: 0 aborts, crossings ≤ 2.9 ps. It found: trbdf2
  cut-set also aborts at rtol 1e-9 ("growing", not in the test); the decline
  message still prints the last three ||dx||, not the window minima/maxima
  (and prints inf / "still contracting" when kw < 2); test comments fixed.
  Note: trbdf2 rtol 1e-9 case A has 2000 accepts of 19557 solves (100k of 169k
  Newton iterations are spent in budget-exhausted solves).

NEXT STEPS (in order):
1. ll13 regression: rebuild with a trace of every ACCEPT (t, h, omega, row,
   ||dx_k||, kw, window min/max) and run ll13 on both builds (the old rule:
   restore the two-step lines locally); find the first accept the old rule
   declined, and whether the state it kept is wrong (compare v(sw_out),
   i(Lout) against pwl at small dt / trbdf2 rtol 1e-9). Candidates: an accept
   with ||dx_k|| near the cap 1e-6·|x_0| (|x_0| large in a forward converter
   ⇒ a cap of volts), or a cycle whose x_k is the worst phase. A fix would be
   to keep the best-residual iterate of the window, or to cap ||dx_k|| by
   tol_dx·k.
2. Then re-run: `.scratch/f5w/sweep.py len|tol|rand 200`, the byte compares
   (`.scratch/b14/rec.sh new compare`), ll08 against a truth, full pytest
   (one started on 6e5241a6+, with -x: `.scratch/logs/full_pytest.log`).
3. Decline message: print the window minima/maxima (the h_min abort text).
4. Cost (instructions per accepted step, stateless trbdf2/pwl, min of six):
   NOT run. The rule adds a 32-entry shift per Newton iteration only when the
   acceptance is on (transient Newton solves), nothing on linear circuits.
- Full pytest on the 1428a5e3 tree (extension of 6e5241a6): 6434 passed, 114 skipped, 18 xfailed, 0 failed (6 min 39 s). ll13 is not covered by a pytest; only trbdf2_identity sees it.

## Session 3 (2026-11-20): the ll13 regression traced and fixed (WIP at this tip)

Trace build (TEMP, removed; env-switched old two-step rule vs window rule,
both evaluated on every budget-exhausted solve). ll13_active_clamp_forward,
trbdf2 defaults: old rule 382 budget-exhausted solves (342 both accept, 3
only old, 3 only window); the old-rule run reproduces 14b0c81a (345 accepts,
1538 steps).
- FIRST DIVERGING SOLVE #85: t = 5.500006 us, h(dt_use) = 3.24e-13 s, 50
  iterations. Newton CONVERGED (||dx|| 3.1e-2 -> 1.8e-5) then sat at
  1.19e-8 V (node drain) for 15 iterations, omega 0.31 eps at every one.
  Window kw = 16: newer min 1.143e-8 / max 1.77e-5, older min 3.35e-5 /
  max 3.10e-2 -> "still contracting": the window rule DECLINED, the old
  rule ACCEPTED. The kept iterate is the window's best-omega iterate
  (omega 0.31 eps, relative dx 2e-10): a good state. The decline walked the
  run into a genuine Newton failure at 41 us.
- Second shape (#112 t = 11.5 us, #266 t = 37.5 us): a cycle at 1.2-1.4e-6
  with ONE step of 5.4e-8 (resp. 1.95e-8) 8-9 iterations back: the newer
  window's min fell 23x -> declined.
- Not the cause: the dx cap (largest ||dx_k||/cap over accepts 0.04; the
  x_0 of 1e6 V are the linear-solve retry starts, re-measured by the
  engine's second solve), the worst phase (x_k not the window's best-omega
  iterate in 54 of 345 accepts, all < 7 eps).
- Reference: engine='pwl' + discontinuity_restart, dt 5e-10 (dsed refuses
  the Shockley diodes). Its own error vs dt 1e-9: 1.6e-8 of range on the
  states at 25 smooth instants (4k+1, 4k+3 us). The trbdf2 run has ~1e8 V
  spikes of ~1e-13 s at the dead times (inductor current into roff 1e7) that
  pwl steps over; states agree away from the edges.

FIX (in the tree, `pwl/nonlinear_solve.hpp`): stagnated if, for SOME split
k from min(8, kw) to kw, the newer k ||dx|| are not 4x below the older k's
minimum and not over 2x their maximum (k = kw alone was rule 11; every solve
it accepted is still accepted). Monotone rate r declined while r < 4^(-1/8)
= 0.84 (rate-0.7 test still declined). Decline message prints the windows'
min/max ("too few for two windows" under 4 iterations). DECISIONS 11
amended. Two new C++ tests (converge-then-floor; one dip inside a cycle).
With the fix (trace build, same arithmetic): ll13 RUNS, 1555 steps, 359
roundoff accepts; 0 solves the old rule accepted are declined; state error
at the 25 instants vs the reference 1.9738e-5 = the 14b0c81a run's
1.9738e-5 (identical to 5 digits).

RUNNING at this tip: the final extension build (TEMP removed), then C++
layer5_v4/v3/step/sim/dsed (`.scratch/f5w/cxx.sh`).
NEXT: ll08 vs pwl+restart dt 5e-9 (own error 1.5e-5 vs 1e-8 on i_L1);
sweeps `.scratch/f5w/sweep.py len|tol|rand 200`; `.scratch/b14/rec.sh new
compare`; `.scratch/ll13/heavy.py` (circuits > 50 accepts -> pytest);
Python ll13 regression test; cut-set rtol 1e-9 strict xfail; cost; stateful
and error-norm test files; full pytest; reviewer.

Results on the FINAL build (TEMP removed):
- ll13 trbdf2: runs, 1555 accepted steps, 359 roundoff accepts; state error
  at 25 smooth instants vs pwl+restart dt 5e-10: 1.9738e-5 of range;
  cedac36b extension (= 14b0c81a numbers: 1538 steps, 345 accepts):
  1.9738e-5. Reference's own error 1.6e-8.
- ll08_buck_mosfet trbdf2: 1993 steps / 117 roundoff accepts (cedac36b and
  14b0c81a: 2020 / 138; e5310710: 1998 / 135). vs pwl+restart dt 5e-9 at 200
  instants: tip 2.2730e-6, cedac36b 2.2730e-6 (max over v_out, i_L1); the
  reference's own error (dt 1e-8 vs 5e-9) is 2.3e-6, so both are at the
  reference's resolution: not worse.
- Sweeps of case A on the final build: run lengths trbdf2 0/30, esdirk43
  0/30; rtol 1e-4…1e-10 no abort on either (crossings and E_hard identical
  to session 2's table); random 200 lengths 0 aborts on both.
- C++ (Eigen 5): layer5_v4 26 cases (incl. the two new ones and the
  rate-0.7 protected case), layer5_v3 35, step 32, sim 8, dsed 142: pass.
- Roundoff-heavy reference circuits on trbdf2 (all 108 triage "runs"
  circuits, `.scratch/ll13/heavy.py`): only ll13 (359) and ll08 (117) have
  > 50 accepts; ll11_llc_resonant_converter aborts at 5 us as at cedac36b
  (the known trbdf2 error cell). Both now have a pytest
  (`test_roundoff_heavy_reference_circuit_runs_on_trbdf2`, bound 1e-3 of
  range against pwl+restart).
- Saturable cut-set on trbdf2: rtol 1e-8 AND 1e-9 now RUN (i(R) at 200.1 us
  0.909368 at both, closed form 0.908907): the strict xfail is removed and
  1e-9 is a passing case (the reviewer's 1e-9 abort is gone, so it does not
  join an xfail).
- Byte compares vs the 14b0c81a recordings: goldens ALL IDENTICAL, stateful
  ALL IDENTICAL, engines ALL IDENTICAL, pwl_identity 379/379, dsed
  bit-identity 280/280, trbdf2_identity 368/371. Moved (all three; none
  newly aborting): ll13 (runs again; vs reference 1.9738e-5 = 14b0c81a),
  ll08 (1993 steps / 117 accepts vs 2020 / 138; vs reference 2.2730e-6 =
  cedac36b, at the reference's own error 2.3e-6), ll11_llc_resonant_converter
  (aborts at 5 us before and after; only the abort message's decline note
  changed text).
- Test files (d2, de, saturable transformer incl. the 15.310393 V flyback,
  trbdf2 stateful error norm incl. its 15-row list, fs5, test_newton_*,
  test_trbdf2_*, test_esdirk43_* incl. the 1.3e-7 V nonlinear-capacitor
  start, stateful, test_dc_*): 465 passed, 5 xfailed. ruff clean.
