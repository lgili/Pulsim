# Decisions taken while implementing (no owner answer yet)

The open questions of `design.md` had no answer and none was marked
"proposed"; the most conservative option was taken.

1. **One-release opt-out flag?** → Kept, C++ only:
   `TrBdf2Options::refresh_after_event` (default `true`). `false`
   restores the old step sequence exactly (no refresh, accepted-step
   memory). Not exposed to Python. Why: the cheapest rollback and
   bisect handle, and it lets a C++ test compare old and new on the
   same build. Reverse: delete the field and the two `if` it guards.
2. **Delete the known-limitations row if the stall reproducer passes?**
   → The row stays unless the reproducer passes (task 5.1); see the
   result recorded in `tasks.md` 5.1. Reverse: n/a.
3. **Split the initial-settle refresh into its own change?** → Moot:
   the initial-settle refresh is not shipped (see 6).
4. **Lengthened refresh probe at tight tolerances (design D1,
   `h_p = max(h_probe, h·0.14·ε/rtol)`)** → Not implemented; task 3.4
   measures whether it is needed (see `tasks.md`).
5. **SDIRK(2,2) fallback (D3)** → Not implemented. Its gate did
   trigger in a form the design did not foresee: 4.3 fails because at
   a diode turn-off the probe exists but holds a stiff mode (see 6),
   not because no probe exists. SDIRK needs a fourth stage form in
   five stateful devices; left as the follow-up for diode events.
6. **Material deviation: the refresh is applied at GATE-driven events
   only** (a landed gate edge, and a gate edge settled at a step
   boundary), not at diode events and not at the initial settle. It
   was implemented at all five places first and measured with
   `probes/trbdf2_baseline.py` (rtol 1e-5):

   | circuit | before | refresh everywhere | gate events only |
   |---|---|---|---|
   | 325 V bridge, 1 mH line | 22 diode events, 1242 steps, 2906 solves | 5814 events, 7157 steps, 78796 solves | unchanged (no gates) |
   | Level-1 MOSFET half bridge (no gates; body diodes) | err 4.14e-4 | err 4.34e-2 | unchanged |
   | bridge, initial-settle refresh only | completes | `max_steps` exhausted at 33.5 ms | — |
   | buck CCM | 7241 steps | 1636 | (see tasks 4.2) |

   Cause, measured on the bridge: a diode turn-off at the zero
   crossing of the line current leaves the 1 mH inductor on a
   g_off-only path. The probe then holds the stiff L·g_off mode
   (picosecond time constant; v(ac1) swings by 100 V in 1e-12 s) and
   the refreshed `v_L` feeds it to the trapezoidal stage, which does
   not damp it: ringing, diode flips, chatter. The stale slot is not
   right either, but it is bounded. A guard on the refreshed voltage
   (skip it when it leaves the circuit's voltage range, 1× and 3×)
   changed nothing. The initial-settle refresh only perturbs the
   bridge's state by roundoff-sized amounts, yet that was enough to
   turn its later event sequence into a grind: the bridge's event
   handling is sensitive to perturbations at this level, which is
   itself worth a look (not in this change).

   The proper cure for the diode-event case is an L-stable first step
   after the event (design D3, SDIRK(2,2)), which needs a fourth stage
   form in five stateful devices; not done here. Consequence: task
   4.3 (DCM buck, no accepted step below 1e-11 s) is not met — those
   steps follow the diode turn-off — and stays a strict xfail. The
   spec delta was narrowed to what ships. Reverse: call
   `refresh_after_event` after the diode cascades and the initial
   settle (the call sites are the cascades marked in the engine).
7. **The edge probe is committed as a recorded step (landed gate
   edges).** The slot-only refresh (states kept, slots replaced) passed
   the step targets, but 17 cross-engine loss/charge/recovery tests
   failed: post-processing integrates each quantity across samples
   with the per-sample switch state, and a 4e-6 s first post-edge step
   made that integral ramp across the edge with the wrong mask. The
   cascade's last probe is a valid trapezoidal step of `h_probe` under
   the post-edge mask, so it is now committed (history, Coss and the
   stateful devices updated, sample and mask recorded) and the
   derivative slots follow from that update. The slot-only refresh is
   kept where no step can be committed: a gate edge settled at a step
   boundary, and a probe that would reach `t_end`. Cost: one extra
   recorded sample per edge (not in `n_accept`); the buck table is the
   same within 1e-6 (1636 steps, 5018 solves, error 7.18e-5). 13 of the
   17 then passed unchanged; the other 4 pin trbdf2/auto conduction
   losses to 1e-6 relative and were re-pinned (CHANGELOG). Reverse:
   pass `commit=false` at the landed-edge call site.
   Two consequences of the commit, fixed: (a) the opaque-`switch_fn`
   pulse learner read the toggle at the next step's start, `h_probe`
   after the edge, and `h_probe` differs per edge (`gate_pulse_min`
   came out 2.999995e-6 instead of 3e-6); the commit now records the
   toggle at the edge. (b) A controller tick inside the probe would
   fire `h_probe` late (9e-12 s); the engine falls back to the
   slot-only refresh when a tick or `t_end` lies within the probe.

## Follow-up (maintainer decisions, 2026-11-13)

8. **Every probe failure is handled (silent wrong answer ended).** The
   zero-time cascades (gate edge, boundary settle, diode event, diode
   landing) ignored `probe_solve`'s result; a failed Newton probe left
   `x_trial` at its warm start and the diode bits were decided on it.
   `probe_solve` now retakes a Newton failure at 100x the step up to the
   probe cap (only a singular factor is learned as the run's probe
   floor: a Newton failure depends on the state), and a cascade with no
   converged probe throws a named error (`SimulationAborted`). The
   landing sub-step rejects on a Newton failure; a localisation trial
   that fails ends the refinement. The initial settle keeps its bits on
   failure (the start state is consistent with them) and the restart
   keeps the old slots. Measured: saturable buck inductor, roundoff
   acceptance off: 11.13 V → 12.85 V (pwl 12.849 V); with acceptance on
   both builds are right (probes/sat_buck.py). C++-only
   `TrBdf2Options::newton_roundoff_acceptance` (default true) exists to
   run that test.
9. **One L-stable SDIRK(2,2) step after each diode event on circuits
   with a Newton device law; none after the initial settle; none on
   piecewise-linear circuits.** Alexander SDIRK(2,2), γs = 1−1/√2 = γ/2, so
   C/(γs h) = 2C/(γh): both stages are the trapezoidal companion at
   dt = γh with the derivative slots zeroed and the state slots set to
   the stage base (stage 2 base x0 + ((1−γs)/γs)(x1−x0)). Same factor,
   no new device stage form. Error estimate: the embedded
   h(1−γs)(k1−k2), unfiltered, with exponent −1/2 in the controller
   (design D3). Circuits with Coss, Lauritzen, IGBT tail, PMSM or a
   saturable inductor take TR-BDF2 steps as before. Measured, rtol
   1e-5, diode × N / settle × M (err, time-weighted err, steps,
   rejects; `probes/trbdf2_baseline.py`):

   | circuit | today (0/0) | diode ×1 | diode ×2 | diode ×3, settle ×3 |
   |---|---|---|---|---|
   | MOSFET half bridge | 4.66e-4, 5.73e-3, 1555, 772 | 3.30e-4, 2.0e-4, 1280, 173 | —, 1.5e-4, 1312 | — |
   | DCM buck | 1.25e-5, 1.07e-4, 9457, 4745 | 1.26e-5, 1.07e-4, 9373, 4432 | | |
   | 325 V bridge | tw 5.57e-2, 1437, 501, 34 events | tw 2.87e-2, 1672, 378, 163 events | tw 5.71e-2, 1375 | `max_steps` grind |
   | boost | 1482, 223 rejects | 1494, 270 | | |
   | VSI on RL | 1235, 30 rejects | 1300, 77 | | |

   Only the MOSFET half bridge is a Newton circuit with diode events;
   the other rows are piecewise linear and were measured with the step
   forced on there (they keep TR-BDF2 in the shipped default; the
   Shockley rectifier has a Newton law but no diode events and is
   unchanged). Forced on, errors were equal or better on every row at
   the same rtol, at +47 rejects on the boost and on the VSI. Applied to every
   circuit, the full suite moved 9 pins, all on piecewise-linear diode
   circuits: bridge loss energies 0.6 % (V_f = 0 identity pin), a
   reference bridge's D2 RMS current 2.4 % from pwl (2 % allowed), the
   doubled-tolerance density 1242 → 1682 steps, a V_th-hysteresis
   event count 612 → 608, a reverse-recovery turn-off on a step longer
   than t_rr, forced accepts in a held-diode run. None of those rows
   gained accuracy, so the step is limited to circuits where a Newton
   law is present (the MOSFET half bridge is the measured gain);
   `sdirk_on_linear_circuits` (default false) turns it on elsewhere. A settle SDIRK step
   turns the bridge into a `max_steps` grind (its chaotic event
   sequence, item 6), so the settle keeps its TR-BDF2 restart. The
   derivative refresh at diode events stays off (item 6: 5814 bridge
   events). Not cured: the DCM steps below 1e-11 s (strict xfail kept;
   they are the gate-edge probe steps and the diode-landing sub-steps,
   not the first step) and the kilovolt landing artefact (SDIRK: 11919
   → 168 V at default tolerances, 1460 → 989 V at rtol 2e-6: the
   2L/h·(residual µA) is in the landing state, not in the stage
   order). C++-only `TrBdf2Options::sdirk_after_diode_event` (default
   1) and `sdirk_after_settle` (default 0); 0 restores the old run.
10. **Hosea–Shampine filter: implemented, off by default.** `solve_at`
   allows it on linear circuits: the estimate goes into the companion
   as a state increment (derivative slots zero), and the source terms
   cancel by linearity, x_f = solve(b_f) − solve(0). With the SDIRK
   step: DCM buck 9457 → 3301 steps, 4745 → 862 rejects; boost 223 →
   57 rejects. But the 325 V bridge then reported a time-weighted
   error of 13 V: one diode turned off 4334 times in 2 ms at t = 33.4
   ms (the filter lets the step grow onto the L·g_off mode). TR-BDF2
   steps alone filtered: DCM rejects 4745 → 14178. The kilovolt
   artefact is not a filter question: filtered, its peak is 20 V at
   rtol 2e-6 and at default tolerances but 1029 V at rtol 1e-6, so the
   strict xfail stays. C++-only `TrBdf2Options::filter_error_estimate`.
11. **The wrong-way rows of the baseline, explained (maintainer decision
   4).** None is an engine defect of this change; one is a separate
   defect, listed last.
   - *VSI on RL, final i_L error 6.28e-4 → 6.73e-4 A.* The reference
     scatters: `engine='pwl'` samples the opaque sine-PWM `switch_fn`
     on its grid, so each edge is quantised to dt with a random sign.
     Final i_L at dt = 1e-7, 5e-8, 2.5e-8, 1.25e-8, 6.25e-9 s:
     −4.57519, −4.58104, −4.58043, −4.58099, −4.58057 A (±3e-4, not
     monotone). TR-BDF2 bisects each edge to h_min and gives −4.58037
     (rtol 1e-5) … −4.58042 (rtol 1e-8): flat in rtol and inside the
     reference's scatter. The share of time a diode conducts beside its
     switch is 14.54 % on both engines.
   - *325 V bridge, final error 0.618 V, flat in rtol.* It is the
     floating DC bus's common mode, not the output: v(dp) − v(dn)
     agrees to ±0.01 V at every 2 ms sample, while after each
     conduction interval pwl puts dn at −0.59 V and TR-BDF2 at 0.000 V.
     With every diode off, dn is set by 1 MΩ and the 1e-9 S leakages,
     and a V_th = 0 diode at zero current is on or off within that
     leakage; both are solutions of the ideal-diode problem. The
     baseline measures v(dp) alone; v(dp) − v(dn) would not see it.
   - *Shockley rectifier error not shrinking from rtol 1e-5 to 1e-6*
     (3.73e-5 → 7.77e-5 final; then 1.64e-5, 3.30e-6 at 1e-7, 1e-8; tw
     5.5e-5, 8.5e-5, 1.9e-5, 3.8e-6). The steps barely move between the
     first two (1079 → 1273), so the error is the sum of signed local
     errors on almost the same grid; it converges from 1e-6 on.
   - *Ripple tolerance 5e-3 → 7e-2* (`test_fft_and_rms_survive_the_
     irregular_grid`). The 5.8 % is the uniform resampler at the median
     spacing (T/8) clipping the ripple cusp, not the engine: the raw
     grid gives 67.98 mV against 68.2 mV from (1−D)V_out/(8LCf²). The
     engine's own peak-to-peak is now pinned at the old 5e-3; the
     resampler keeps 7e-2 (a time-weighted AC RMS through the
     resampler is worse, −14.6 %).
   - *Separate defect, not fixed:* the 325 V bridge at rtol ≤ 1e-6
     exhausts `max_steps` (1e7) at t = 10.18 ms (1e-6) and 10.05 ms
     (1e-7), on this branch and on the build before it. That is the
     first all-diodes-off interval, the common-mode node of the
     previous point; the bridge's sensitivity to roundoff-sized changes
     (items 6 and 9) is the same mechanism. Left for a change of its own.
12. **The SEPIC grind: the error estimate at its own roundoff floor, and
   diode events that commutate nothing (2026-11-13).** `sepic.yaml` at
   rtol 1e-8 / atol 1e-12 exhausted `max_steps` (1e7) at t = 0.776 ms.
   Engine trace: D1 turns off at its current zero at 776.1222 µs while
   S1 is also off, so `sw` and `mid` are held only by g_off (2 × 1e-6
   S). Their elimination error is eps·(2C/h)·V/g_off, and the estimate
   reads it through v_L/L as h·δv/L: constant in h, 5e-11 to 1.4e-10 A
   against a scale of 1.5e-11 A (atol 1e-12 + rtol 1e-8 × 1.4 mA).
   Every h was rejected down to h_min = 1e-14 s, where KCL roundoff
   (1e-5 A) put 5 V of forward bias on the OFF diode; the crossing
   detector fired, the zero-time cascade changed no bit but recorded an
   event, three times, and the chatter guard then decided ON from the
   noise. A second path: an ON diode at +9e-10 A crossing within
   dt_floor, whose probe at t did not flip (all 77 zero-time events of
   the trace changed no bit).
   Fix: (a) a step about to be rejected on a piecewise-linear circuit
   adds the estimate's roundoff floor, eps·J⁻¹(|J||x| + |b|) read through
   each entry's estimate formula, to that entry's scale, and is accepted
   when the estimate is within it (`n_roundoff_floor_accepts`); steps
   that pass are unchanged. (b) A zero-time cascade whose probe at t
   flips nothing is rerun with its first probe dt_floor long (the
   post-crossing side); if that flips nothing either, no event is
   recorded and the step is retaken at the same h under the chatter
   guard (accepting it moved the SEPIC's V_out by 7.5e-5 V; localising
   it landed a bridge diode every 1.6e-12 s). (c) A diode-event landing
   whose cascade undoes its flip records no event and starts the next
   step from the landed x, not from the cascade's last probe: on the
   floating 325 V bridge that probe read D1 at −212 V (roundoff on a
   g_off-held node at h_zero) against +0.06 V in the landed state, and
   the detector re-found the same crossing every 4.25e-11 s (found when
   (a)+(b) moved the default-tolerance bridge onto that path).
   Measured (V_out, accepted steps, rejects, events):

   | run | before | after |
   |---|---|---|
   | SEPIC rtol 1e-8 / atol 1e-12 | `max_steps` at 0.776 ms | 17.31570342 V, 28,539, 1,085, 90 |
   | SEPIC 1e-8 / 1e-10 | 37,073 samples, 16,596 rejects | 27,499 accepted, 721 rejects |
   | SEPIC 1e-5 / 1e-7 | | 17.31574723 V, 3,507 |
   | 325 V bridge rtol 1e-5 (baseline row) | tw 5.57e-2, 1,437, 501 | tw 2.73e-2, 1,356, 129 |
   | 325 V bridge rtol 1e-6 | `max_steps` at 10.18 ms | 1,971, 80; bus within 0.007 V of pwl |
   | 325 V bridge rtol 1e-7 | `max_steps` at 10.05 ms | 3,241, 86; bus within 0.0015 V of pwl |
   | Ćuk (`cuk_converter.yaml`) default | 78,414, 122,804 rejects | 12,828, 371 rejects; V_out equal to 4e-7 V |
   | Ćuk rtol 1e-8 / atol 1e-12 | 989,930, 3,163 forced accepts | 88,284, 1 forced accept |

   Every other row of `probes/trbdf2_baseline.py` and every trajectory of
   the golden probe is byte-identical; the floor's two extra solves
   raise `cache.event_hits` (buck 3,569 → 4,779). The bridge's common
   mode v(dp) stays undetermined (all diodes off, tied by 1 MΩ): it
   differs from pwl by up to 41 V at rtol 1e-5 (22 V before). The Ćuk's
   node `mid` still reaches 3.4e6 V (3.5e6 V before).
   The 16× sample growth of `sepic` on the c1 build (409,476 samples
   against 4,701 at baseline) is the same regime: 390,959 of those
   samples lie in switch-and-diode-off intervals, 387,085 of them with
   |i_L1| < 10 mA. At the default tolerances the run takes 3,721
   samples (17.31575 V).
   The DCM buck (task 4.3) is a different mechanism and stays an xfail:
   0 floor accepts, 246 forced accepts, smallest step 4e-15 s, 4,325
   accepted steps below 1e-11 s. Those steps are the stiff L·g_off mode
   after a turn-off with residual current (τ ≈ 2e-13 s, `sw` swings to
   5.6 kV), which the raw estimate reads as error, plus the gate-edge
   probe steps that are committed and short by design. The
   Hosea–Shampine filter (item 10) does not remove them without the
   SDIRK step.
   Two warnings lost the runs that exercised them. The snubbed boost
   (6 ms, `engine='auto'`) took 5 chatter breaks and warned that D1
   rode its conduction boundary; it now takes 0, with the same tail
   average (53.6778 V both builds) and 603 forced accepts instead of
   670. The floating bridge with a 1 mH choke at rtol 2e-5 held a
   diode (9 events, 2 chatter breaks, 12 forced accepts); it now holds
   none (2 events, 0, 0), dn stays within 344 V and the bus is
   unchanged (329.0959 V). At the default tolerances that run also
   went from 50,566 samples to 1,132. The boundary-ride warning is now
   pinned by calling `_trbdf2_warn_stats` with the count the boost had.
   The held-diode warning's subnet naming has no run left that reaches
   it; `test_diode_state_reconstruction.py` still covers the warning
   itself.
