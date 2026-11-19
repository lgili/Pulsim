# Decisions taken while implementing (no owner answer yet)

Open questions of `design.md`: none was marked "proposed"; the most
conservative option was taken unless the design states a decision.

1. **Q1: thinner default output on short runs, or keep `span/1000` in
   the ceiling minimum?** → Kept `span/1000` in the minimum (the
   conservative option). The default ceiling is still
   `min(span/1000, period/20, pulse width/3)`; only the floors derived
   from it stopped depending on the span. Consequence: the scenario
   "same steps at 40 T whatever `t_end`" holds when the ceiling is
   fixed (`dt` given, or a source period/20 below `span/1000`), not for
   a span-bound ceiling. Measured: buck with `dt = T/20`, `t_end` 40 T
   vs 400 T: identical times and states up to 39 T (they diverged at
   sample 24 before). A learned or oracle gate pulse does not set the
   default ceiling (it still caps `h` at twice the narrowest pulse, as
   before). Reverse: drop `span/1000` from `h_default` and set
   `h_max_basis` accordingly.
2. **Q2: double the Python default tolerances?** → No (design D1
   states the defaults are not changed). Users get the same density as
   before by passing `rtol=2e-5, atol=2e-8`; measured, that reproduces
   the previous step sequence of the diode bridge exactly (1242 steps,
   123 rejects, 22 diode events). Reverse: set the defaults in
   `simulate` to 2e-5 / 2e-8.
3. **Q3: stateful devices in the error estimate?** → Left to the
   unified stepper (not in this change).
4. **Floors bit-identical on a span-bound ceiling.** The design's
   floors `h_ref·1e-8`, `h_ref·1e-9`, ... equal the old `span·1e-11`,
   ... only up to roundoff. That roundoff alone (the factor 2 was
   ruled out by running at doubled tolerances, which must and did
   reproduce the old sequence once the floors were exact) tipped the
   325 V diode bridge into a `max_steps` grind at 4.46 ms. The floors
   are now the old span fractions of `span_ref`, which is the span
   itself while `h_max == span/1000` and `1000·h_max` otherwise. The
   probe retry cap is `span_ref·1e-6` (= `h_ref·1e-3`): design D3's
   table says `h_ref·1e-2`, which is not today's value (today's is
   `min(span·1e-6, h_max·1e-2)` = `h_ref·1e-3` on a span-bound
   ceiling); the table's own rule "every value equals today's" won.
   Reverse: compute the floors from `h_ref` directly.
5. **Stiff filter (D2) deferred.** It needs a solve on an existing
   `(mask, dt)` factor with a caller-supplied right-hand side in
   `PwlStateSpaceCache` (`pwl/cache.hpp`, owned by another track) and
   the Newton factor on the nonlinear path. Tasks 4.x are not done and
   the "TR-BDF2 Stiff Error Filter" requirement was removed from this
   change's spec delta; `h_chatter` stays. Reverse: n/a (follow-up).
6. **One commit instead of four.** The migration plan asks for one
   commit per part (breakpoints, bounds, filter, scale). With the
   filter deferred, the three remaining parts were verified and
   committed together; the before/after table separates them where it
   can (the bridge rows). Reverse: n/a.
7. **Breakpoint restart = the gate-edge restart of
   update-trbdf2-post-event-restart.** A step landed on a source
   corner commits a probe under the post-jump source (or refreshes the
   slots only, at `t_end` or a tick) and restores the pre-clamp step
   proposal. Not done when a gate edge was landed at the same instant
   (the edge cascade already committed its probe). PWM voltage sources
   contribute their edges; sine sources have none; `b_extra_fn` is
   opaque.
8. **Pre-existing crash fixed on the way.** The initial diode settle
   ignored a failed (Newton) probe and read `x_trial`, which is unsized
   before the first successful solve: a PMSM beside a rectifier with
   `dt = 4e-6`, `t_end = 2 ms` died with SIGSEGV once the floors moved
   (`test_pmsm_beside_a_diode_runs_on_trbdf2` crashed the pytest
   process). The settle now stops at a failed probe and keeps the bits.
9. **Tests re-baselined to the previous density, not loosened.**
   (a) `test_gate_edge_refreshes_a_stateful_device` runs TR-BDF2 at
   `rtol=2e-5, atol=2e-8`: at the defaults (now twice as strict) the
   Lauritzen buck hits the known h_min stall at t = 15 µs — it did the
   same before this change at `rtol=5e-6`. (b) The strict xfail
   `test_no_kilovolt_artefact_on_the_internal_node` runs at
   `rtol=2e-6, atol=2e-9`: at 1e-6 the new step sequence happens to
   miss the spike (20 V) although the defect remains (1460 V at 2e-6,
   1897 V at the defaults), so the xfail would have passed by accident.
10. **The stateful kinds' internal states in TR-BDF2's estimate (Q3
   answered; 2026-11-19).** Defect: the norm had rows for the linear
   capacitors and inductors only. A stateful kind's charge or flux (the
   Lauritzen q_M, the IGBT tail Q, a nonlinear capacitor's Q, a saturable
   inductor's λ, a machine's three λ, an external device's q rows) had no
   estimate, so a circuit whose dynamics live there sat at its ceiling at
   any rtol (the Lauritzen recovery 2e-1 off its closed form at every rtol,
   add-high-order-stiff-integrator DECISIONS 17).

   **The estimate of a q row.** On TR-BDF2 a kind is not advanced by the
   ESDIRK stage form (βh, ψ): it commits under the method's own two
   stages — trapezoidal over γh, then BDF2 over h — through its own law
   (K0 + K1·s(x) for the charge kinds, λ(i) for the inductor, λ(i, θ, λ_r)
   in increment form for the machines, q(x) for an external model). Bank's
   estimate needs the derivative at the three points; none of the kinds
   evaluates f at the stage points, but each stage rule implies it exactly:

       trapezoidal:  q_γ − q_n = (γh/2)(f_n + f_γ)   ⇒  f_γ = 2Δq_γ/(γh) − f_n
       BDF2:         f_1 = (c1 q_1 + c2 q_γ + c3 q_n)/h = (c1 Δq_1 + c2 Δq_γ)/h

   (c1 + c2 + c3 = 0, so only the increments Δq = q − q_n enter: no
   eps·|q| cancellation on a flux riding a large λ_pm). f_n is the kind's
   committed derivative slot, the one the trapezoidal stage integrated
   from. At a converged solve the implied f_γ and f_1 equal the law's
   derivative at q_γ and q_1 (the stage equation IS that equality), so the
   row's estimate is exactly Bank's on the device's own ODE:

       est = 2·C·h·(f_n/γ − f_γ/(γ(1−γ)) + f_1/(1−γ)),   in the unit of q.

   For a linear capacitor (q = C·v) this is C times the node-voltage row's
   expression; for a kind it is NOT that expression with v replaced, since
   q_γ and q_1 come from the kind's own law at the stage points
   (`trbdf2_increments`: q_γ from the stage-1 capture, q_1 under the BDF2
   stage's K0/K1 — the value `commit` will store). ESDIRK keeps its
   embedded difference Σ(b − b̂)·(q(Y_j) − ψ_j) + e_0·h·q̇_0 (entry 17).

   **Scale.** atol·s + rtol·max(|q_n|, |q_1|) with s the device's own
   conversion from the electrical unknown its state follows
   (`q_scales`): τ for the Lauritzen q_M (= τ·i on its steady-state curve),
   k·τ_tail for the IGBT tail (Q = k·τ·i_ss), C(0) for a nonlinear
   capacitor, L(0) for a saturable inductor, max(L_d, L_q) / L_ls + L_m for
   the machines, the largest |∂q_r/∂x| of an external row (1 where a law
   gives none). So the row's absolute tolerance is the error in volts or
   amperes the electrical rows tolerate, expressed in coulombs or webers —
   never atol in volts applied to a charge. A row that stays exactly 0 has
   a finite scale (atol·s), which settles entry 17's doubt (the esdirk43
   floor 1e-3·max|q| + numeric_limits::min() was a vanishing scale for a
   row that never moved; `ed_sqmax` is gone with it).

   **One implementation.** The q block sits once in the shared error phase
   (`tb_error`); the two integrators differ only in the per-row estimate.
   Not on an SDIRK restart step (esdirk43's forced restart; TR-BDF2 takes
   none with a stateful kind): its stages are not the method the estimate
   is for.

   **Joined by max, not one RMS.** err = √max(RMS² of the entries, RMS²
   of the q rows). One RMS over both (esdirk43's rule until now) let a
   well-resolved q row dilute the entries' error: the saturable LC took
   fewer steps than before and got 4–6 % worse at rtol 1e-4/1e-5. With
   the max a q row can only tighten the step. Without q rows the
   expression is the entries' RMS bit for bit (the round-off floor too).

   **From the first committed step.** A fresh run (no snapshot) starts
   with the kinds' derivative slots at init's zeros. From x = 0 behind a
   source that is a stale slot: the row sees an O(h) error, correctly, and
   the controller drove h to ~1e-16 s, where Newton on a nonlinear
   capacitor cannot converge (the 60 V coss case aborted at t = 0; it
   aborted on esdirk43 before this change too, through the vanishing
   scale). A linear capacitor takes the same 1e-16 s start and survives
   only because it is linear. The rows join after the first accepted step
   (its commit writes the method's own derivative); a resume joins at
   once (the snapshot carries real slots). The first step stays
   unestimated on the stateful rows, as before this change.

   **Proof per kind (TR-BDF2, atol = rtol·1e-3; probe
   `probes/q_rows_order.py`).** Error against the truth / accepted steps,
   base → tip:

   | case | rtol 1e-4 | 1e-6 | 1e-8 | 1e-10 | slope (controlled range) |
   |---|---|---|---|---|---|
   | Lauritzen waveform (closed form) | 2.0e-1/28 → 2.3e-4/128 | 2.0e-1/28 → 9.9e-6/521 | 2.0e-1/28 → 2.5e-6/2347 | 2.0e-1/28 → 4.0e-6/10829 | 0.69 (1e-4..1e-7; then the 3e-6 source-edge floor, esdirk43's is 3.4e-6) |
   | Lauritzen Q_rr (closed form) | 2.67/28 → 6.7e-4/128 | 2.67/28 → 3.1e-5/521 | 2.67/28 → 4.0e-6/2347 | 2.67/28 → 4.1e-6/10829 | 0.63 (1e-4..1e-7) |
   | IGBT tail waveform (closed form) | 5.2e-3/48 → 7.0e-5/107 | 5.2e-3/48 → 3.1e-6/361 | 5.2e-3/48 → 1.3e-7/1548 | 5.2e-3/48 → 4.4e-8/7062 | 0.67 (1e-4..1e-9) |
   | IGBT tail energy (closed form) | 4.9e-4/48 → 9.8e-6/107 | 4.9e-4/48 → 7.8e-7/361 | 4.9e-4/48 → 1.6e-7/1548 | 4.9e-4/48 → 1.2e-7/7062 | 0.44 (the samples' trapezoid floors it) |
   | PMSM locked rotor (closed form) | 9.3e-8/1004 → same | same | same | 9.3e-8/1004 → 8.2e-8/1037 | at the step ceiling: λ_d rides ψ_pm, so rtol·|λ| is loose for L_d·i_d |
   | PMSM free acceleration (ω_end vs esdirk43 1e-11) | 2.1e-5/1007 → 2.1e-5/1007 | 2.6e-5/1103 → 2.4e-5/1249 | 7.7e-6/2516 → 1.5e-6/5322 | 5.9e-7/11491 → 6.6e-8/24800 | 0.68 (1e-7..1e-10) |
   | induction machine DOL (pinned SciPy, ω) | 5.9e-6/10003 → 5.9e-6/10003 | 5.8e-6/10066 → 5.8e-6/10135 | 4.3e-6/17909 → 2.5e-6/23000 | 2.5e-7/83376 → 1.7e-7/107350 | 0.58 (1e-8..1e-10) |
   | saturable LC (end state, SciPy) | 1.56e-3/1053 → same | 4.3e-4/2486 → 3.0e-4/2744 | 2.2e-5/11066 → 1.4e-5/12394 | 1.1e-6/50920 → 6.7e-7/57216 | 0.66 (1e-6..1e-10) |
   | nonlinear capacitor through R (SciPy) | 8.6e-7/1003 → same | 8.6e-7/1003 → 8.7e-7/1006 | 8.6e-7/1003 → 6.8e-7/1118 | 8.6e-7/1003 → 3.1e-8/2783 | 0.67 (1e-8..1e-10; 3 ns ceiling above) |

   The DC machine has no stateful kind (its windings are linear inductors
   already in the norm): unchanged, as the PSFB.

   **esdirk43, same probe** (its rows changed scale and join by max):
   PMSM free acceleration 0.76–0.90× the base error at rtol 1e-6..1e-10,
   IGBT waveform and energy and Lauritzen Q_rr within 1 % or better
   (0.92–1.04), the locked PMSM and the nonlinear capacitor bit-equal in
   error. Larger ratios only inside a case's floor band, where the base is
   itself non-monotone in rtol: Lauritzen waveform 0.41–2.55× around the
   3e-6 source-edge floor (base 8.3e-7, 3.2e-6, 3.5e-6, 3.4e-6 at
   1e-7..1e-10), induction machine 1.7e-8..7.4e-8 against base
   2.3e-9..7.2e-8 (the pinned SciPy reference resolves ω to ~6e-8:
   add-high-order-stiff-integrator DECISIONS 17), saturable LC
   7.4e-8..8.9e-8 against 8.0e-8..8.2e-8 at its 1e-3-of-period ceiling
   (1e-6..1e-8), then 0.83–0.93× below.

   **THE LIST: every moved case against its truth** (one baseline at the
   merged base 93aef2ef: goldens with counters, 41 stateful cases,
   7 engines × 5 circuits, step identity, pwl_identity, trbdf2_identity
   goldens/reference/extra with modes '', lcp and shadow, reference_status,
   the exact scoreboard on six configurations). Error = the worst column's
   time-weighted RMS distance to the truth over the run's own samples,
   divided by the truth's range; truth = the same circuit on esdirk43 at
   rtol 1e-11 (`11`) or 1e-10, its distance to the next tolerance (or to
   TR-BDF2 at 1e-10) in brackets: a difference below it is not resolved.

   | case (trbdf2, its test's settings) | base | tip | truth [its own spread] |
   |---|---|---|---|
   | saturable inductor (resume set) | 8.28e-6 | 9.38e-6 | esdirk 1e-11 [9.6e-6]: not resolved |
   | gapped inductor | 2.40e-4 | 9.45e-6 | esdirk 1e-10 [7.4e-6] |
   | Lauritzen diode | 4.28e-5 | 5.94e-6 | esdirk 1e-11 [5.4e-6] |
   | nonlinear capacitor, 60 V sine from x = 0 | 1.99e-5 | 1.43e-5 | esdirk 1e-11 [2.6e-6] |
   | IGBT tail, clamped turn-off | 1.663e-2 | 1.663e-2 | esdirk 1e-11 [7.7e-3]: unchanged |
   | rectifier, atan saturable core | 2.89e-3 | 2.14e-3 | esdirk 1e-10 [5.1e-5] |
   | rectifier, gapped core | 5.38e-3 | 2.59e-3 | esdirk 1e-10 [5.7e-5] |
   | gated buck + nonlinear C | 1.83e-5 | 9.78e-6 | esdirk 1e-11, samples ≥ 50 ns from a gate edge [6.6e-7] |
   | gated buck + Lauritzen | 5.80e-7 | 5.51e-7 | same [5.7e-7]: at the truth's spread |
   | everything (all kinds) | 2.06e-2 | 1.69e-3 | esdirk 1e-10 [1.6e-3] |
   | example hysteretic_inductor_demo (J-A) | 7.57e-4 | 8.10e-5 | trbdf2 1e-11 [6.9e-8] |
   | example boost_saturable_inductor | 5.29e-3 | 6.16e-3 | esdirk 1e-11 [3.3e-5]; see below |
   | IM locked rotor (reference_status, analytic) | 8.23e-6 | 4.32e-6 | closed form |
   | IM free acceleration (reference_status, analytic) | 4.25e-5 | 2.34e-5 | closed form |

   The two rectifier rows of trbdf2_identity --extra are the same runs as
   the stateful set's. Unchanged: the PMSM case of the stateful set
   (bit-identical once the q rows join by max), every golden, every
   engines case (the saturable-inductor buck on trbdf2 included), the step
   identity, pwl_identity (390/390), trbdf2_identity 367/371 (the four
   above), the exact scoreboard (81/81 records, all six configurations),
   reference_status counts (trbdf2 106/2/1/0/0); REFERENCE_STATUS.md moves
   in the two induction-machine analytic cells only.

   The boost example (switches open, IGBT level 1 driven by a pulse,
   2 ms): the worst column is the switch node `sw`, an algebraic node held
   only by the open IGBT and diode while the inductor current is zero (an
   L·g_off = 1e-13 s mode). On every STATE the tip is far closer to the
   truth (max error over the run / range at rtol 1e-8: v_out 5.9e-3 →
   6.9e-6, inductor current 3.2e-3 → 4.6e-5); `sw` moves 2.7e-1 → 6.3e-1.
   Neither build converges on `sw` in rtol (base 2.7e-3 / 2.9e-3 /
   2.7e-3, tip 6.2e-3 / 6.4e-3 / 6.4e-3 at 1e-6 / 1e-8 / 1e-10), and the
   fixed engine is 8.5e-2 off at dt 5 ns: no tool resolves that node.
   Counted as a node no step resolves, not a regression of the fix; the
   cost is real (accepted steps 20.6 k → 32.8 k at the default rtol,
   45 k → 589 k at 1e-10: the flux row now asks for the method's order).

   Not moved by truth: the nonlinear-capacitor order probe at rtol 1e-6 /
   1e-7 (max error 8.62e-7 → 8.73e-7 / 9.23e-7, both runs at the 3 ns
   step ceiling there): the tip's error after the edge is smaller
   (−1.0e-8 vs −6.0e-8 at 20 ns, −3.0e-7 vs −5.1e-7 at 200 ns), the
   ceiling-bound drift that follows is the same, and the base's larger
   early offset cancelled part of it at the maximum (0.8 µs). Below
   1e-8 the tip falls as rtol^(2/3) (3.1e-8 at 1e-10; the base stays at
   8.6e-7).

   **Tests that moved (with their truths).** (a) `test_loss_port_seams::
   test_stateful_stamps_compose_on_trbdf2_with_and_without_its_loss_port`
   pinned the old run to itself (17.08541163 V, 3.39222120 A, rel 1e-5).
   Truth, esdirk43 at rtol 1e-11: 17.08836174 V, 3.38753530 A (esdirk43
   and trbdf2 at 1e-10 within 3e-9). Base error 1.7e-4 / 1.4e-3 relative,
   tip 2.3e-5 / 1.5e-4 (1067 → 7337 steps); now pinned to the truth at
   1e-4 / 5e-4, which the base fails. (b) `test_device_state_snapshot::
   test_a_snapshot_resumes_on_the_other_engine[pwl-trbdf2]`: the seam error
   i(D1) 7.8e-4 → 4.1e-3 against a 4e-3 bound. The continuous TR-BDF2
   reference is now 3.5e-6 from its rtol-1e-10 self (base 4.0e-5), and at
   rtol 1e-10 the continuation from the pwl snapshot is 6.2e-3 from the
   continuous run: the bound measures the fixed engine's own error at the
   split, which the base's reference shared. 1e-2 into TR-BDF2; 4e-3 kept
   into the fixed engine. A dropped Coss block reads 0.82 (next test).

   **Review (fresh-context reader).** Fixed: (1) the new rejected-steps
   test rejected only on the first step, where TR-BDF2's rows are not yet
   live; it now resumes from a snapshot (rows live at once) under a dt
   ceiling, and compares against the clean resume with an ulp yardstick
   (the rebuilt h1 is exact only to an ulp; measured tried vs clean
   1.0e-9, one-ulp spread 4–5e-10). (3) External-device rows with no q
   entry were counted in the q RMS and diluted it; `q_scales` marks them −1
   and the estimate skips them. (5) The seam bound is kept at 4e-3 into
   the fixed engine. Left: (2) on a circuit with stateful kinds only, the
   first TR-BDF2 step is unestimated and so bounded only by h_max /
   dt_init, as before this change (a DC-seeded start could estimate it);
   (4) an external device's atol_q is taken from the model's evaluation
   when the estimate first runs (deterministic, can depend on dt_init);
   (6) the boost `sw` node and the nonlinear-capacitor probe at 1e-6/1e-7
   are explained above, not truth-improved.

11. **The PSFB on trbdf2, re-measured with the forced-accept trace
   (2026-11-19).** `examples/phase_shift_full_bridge.yaml`, mean of
   v(vout) − v(rect_neg) over 1.8–2.0 ms against `engine='exact'`
   24.0661693 V; low-level `run_transient_trbdf2(trace_forced=True)` (a new
   diagnostic: per forced accept {t, h, err, unknown, ratio}). The circuit
   has no stateful kind: this change does not touch it.

   | integrator, floor | rtol 1e-5 | 1e-6 | 1e-8 |
   |---|---|---|---|
   | TR-BDF2, 0 | −9.14e-2 / 23797 / 2336 | −7.56e-2 / 20996 / 3884 | −1.02e-1 / 33613 / 6583 |
   | TR-BDF2, 1 | −8.92e-2 / 24379 / 2210 | −7.02e-2 / 20871 / 3779 | −1.11e-1 / 39319 / 6478 |
   | esdirk43 (floor always on) | −1.02e-1 / 9634 / 112 | −9.98e-2 / 9653 / 180 | −1.06e-1 / 27996 / 13775 |

   (error V / accepted / forced). Every forced accept is blamed on an
   inductor current of the primary cut-set: L_leak, T1.p, T1.s (L_out
   ≤ 2 %), never a capacitor voltage. Two populations: h = 2e-14 s exactly
   at a gate edge (phase 0 and 2.4 µs of the 2.5 µs half-leg grid), and
   h = 2e-10 s (the floor there) inside the 100 ns dead time that follows a
   turn-off, where the body diodes commutate the primary current at zero
   current. The ratio estimate / scale is 20–3000 (median), not round-off:
   in those modes the cut-set current changes within any step by the
   L·g_off femtosecond mode (DECISIONS #21 of update-trbdf2-post-event-
   restart, H1/H3). The estimate is right; no local change to the norm can
   help (the floor changes the result by < 1 %, H5's exclusion was
   refuted). The lever is the event: the "all OFF" successor of a
   zero-current diode turn-off, or physical device capacitance. Recorded in
   docs/known-limitations.md; not fixed here.

12. **Goal 3 (commutation energy as a thermal state jump) was not started:
   hand-off.** Budget went to the verification of entry 10 and to entry 11.
   State on this tip: trbdf2's `LossTracker` (solver/loss_tracker.hpp)
   queues an event's energy and `advance(h_next)` turns it into a
   rectangular current E/h_next into the deposit node over the NEXT step
   (one-step lag, exact in sum; add-in-kernel-event-losses DECISIONS 5
   and its open item (i)). The exact engine already takes the same queue
   as an impulse through `take_heat()`. Candidate for trbdf2: at the
   accept that books an event, apply the jump Δv = E / C_node on the
   deposit node's capacitor (the walk's nodes and the capacitors' history
   re-derived as in that open item) instead of `advance()`'s rectangle,
   then restart the step (the post-event SDIRK restart already exists).
   Accept: the RL chopper's single RC node in closed form, buck and VSI
   T_j against trbdf2 rtol 1e-10 within 1e-4 of the rise, energy in =
   loss booked. Interaction with entry 10: a thermal capacitor is an
   entry row, not a q row, so the joining rule is not involved.
