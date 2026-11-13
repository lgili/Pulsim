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
