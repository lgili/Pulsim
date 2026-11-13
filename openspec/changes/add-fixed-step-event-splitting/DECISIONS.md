# Decisions taken while applying add-fixed-step-event-splitting

Rule for every entry: the open question's proposed option, else the
most conservative one. Format: question → option → why → how to reverse.

The proposal was written for the backward-Euler-pair restart; the split
was re-derived for the SDIRK(2,2)×3 rule (restart change, design D0):
trapezoidal `dt1`, then one SDIRK step over `dt2`, then the window's
two grid steps. A first implementation on the pair (never committed)
was read for what it learned and ported, not applied.

## Open questions of design.md

1. **Flip with the restart rule or one release later?** → Neither in
   this change: the default stays `false` (Migration step 1), as the
   restart rule's own default did. → The flip is the maintainer's,
   together with the rule's; task 7.4 is left open. → Reverse: n/a.

2. **Alternative B (one refactor) from the start?** → No, alternative
   A (sub-step 1 solved), the design's choice. → The cost gate (task
   6.2) did not fire: +6.6 % ns/step with a moving duty, +10.4 % with
   a fixed off-grid one, against 25 %. → Reverse: replace sub-step 1 by
   an interpolation of the state in `split_step` (`run_transient.hpp`).

3. **`max_events_per_step` = 8 and the 1 % snap.** → Accepted as
   proposed (`kMaxStepEvents`, `snap = 0.01·dt`). → No measured case
   needed more. → Reverse: the two constants in `run_transient.hpp`.

4. **Bisect an opaque `switch_fn`?** → No (most conservative): the split
   uses `next_edge_after` only; an opaque callable keeps the restart
   rule's start-of-step sampling and its edges are taken at the next
   grid point. → An opaque function's meaning between grid points is
   unknown: the grid-indexed idiom of the probes and the guide
   (`int(round(t/dt)) % N`) changes value at the HALF step, so a
   bisection would split every such run at t_{k-1} + dt/2, a phantom
   edge that moves results the user wrote on the grid on purpose. It
   would also cost ~10 GIL round-trips per edge. A Python schedule can
   declare its edges with a `next_edge_after(t)` method instead (read
   by the `pwl` binding only when the split is on; `trbdf2` is
   untouched). → Reverse: in the event collection of
   `run_transient.hpp`, bisect `switch_fn` on (t_prev, t] when
   `switch_fn(t_prev) != switch_fn(t)` and no oracle is given.

## Choices the design left open

- **Refusal.** `enable_substep_state_correction=True` without
  `discontinuity_restart=True` raises
  `UnsupportedError("pwl", "enable_substep_state_correction")` (a
  `ValueError`) naming both options, at the top of `run_transient`, so
  every caller (simulate, YAML, sweep, raw binding) gets it. The rule's
  refusal of the pair is gone; `test_restart_and_substep_correction_
  are_refused_together` became `test_substep_correction_requires_the_
  restart_rule`. Two C++ tests that set the option alone now set both.
- **Window after a split.** Any split sets `restart_left =
  max(left, 2)`: the remainder `dt2` is the window's first step, the
  next two grid steps are SDIRK. Their masks equal the post-event one,
  so no trigger would arm them otherwise.
- **Diode location only in a trapezoidal interval.** Only the first
  interval of a step can be trapezoidal. A crossing inside an SDIRK
  interval (after a gate edge, or a step of the window) is resolved per
  stage, as the rule alone does; its record keeps the grid
  interpolation.
- **Crossing within snap of an end.** Near the start: the step is
  re-taken as an SDIRK step (the rule alone) and the window covers the
  next two. Near the end: the pre-event solve is kept, the bits flip at
  the end and the window covers the next three steps.
- **A diode already on the far side at the step start** (a decision
  that `decide(x_prev)` would also flip): crossing at the start, i.e.
  the rule alone.
- **Several diodes.** All diodes whose crossing lies within `snap` of
  the earliest flip together at it; the others are re-decided by the
  SDIRK stages.
- **Gate edge within snap after the step start.** The step samples its
  gate at the edge (post side), i.e. the edge is taken at t_{k-1}.
- **Gate sample of a later interval.** `switch_fn(t_e)`: every schedule
  with an oracle snaps its edges (`pwm_phase`), so `t_e` reads the post
  side.
- **Source jump within snap before the step end.** The step's last solve
  reads the source at its left limit (`value_at(tau - left_by)`), and the
  next step restarts; the restart trigger for built-in sources becomes
  "a jump in [t_{k-1} − snap, t_{k-1} + snap)" with the split on.
- **No discarded `b_extra_fn` call.** A step with located events does
  not accumulate the nominal (t, dt) right-hand side first; it calls
  `b_extra_fn` at dt1's end and at both SDIRK stages (pinned).
- **dt-retry inside a split step.** The ladder rolls back to the step
  start and re-takes the step without the split (the gate keeps the
  step's start sample, interior events go to the next step, which
  restarts on the new mask). Retries are rare; a `shortcut:` comment.
  Pinned on the mains rectifier at dt = 1 ms (retries at 21/22/23 ms,
  edge at 22.4 ms: no split, 6 restart steps).
- **The V3 correction path** (direct `cache.solve_at`, Newton devices
  solved linearly, crossing read from the post-flip state) is deleted
  (design D5); it was reachable only with the option on, which now
  implies the split.
- **Sweeps.** `sweep.py` did not forward `discontinuity_restart`
  (pre-existing gap of the restart change); it does now, else a sweep
  with the split would always be refused.
- **Characterisation as pins, not strict xfails.** As in the restart
  change: with the default off both behaviours coexist, so each test
  asserts the rule-alone ("before") and the split ("after") numbers.
- **Readout of convergence tests at a common grid point.** An event off
  the grid makes "te + 0.3 ms" an off-grid time; reading the nearest
  sample adds an O(dt) readout error that looks like first order. The
  stateful-device tests read at a grid point of every dt.

## Measurements (engine, `probes/split_probe.py`)

- Task 1.3 (rule alone, before the split): DCM buck relative error at
  T/100 / T/101 / T/400 = 8.6e-7 / 6.9e-6 / 2.1e-7 (default engine
  −2.3e-4 / −5.0e-3 / −2.2e-4). **The restart rule alone removes the
  T/101 outlier**; the split takes it to 7.2e-8 / 7.3e-8 / 1.3e-9 and
  places the turn-off records within 1e-4 dt (rule alone: grid index,
  0.98 dt early).
- Factorisations (task 6.1): the off-nominal factors go through the
  8-entry event LRU of `solve_at`, keyed on the exact `(mask, dt)`. A
  fixed off-grid duty still refactors 0.76 times per period, because
  `t_e − t_{k−1}` differs by ulps from period to period; a moving duty
  refactors at every split (2.96 per period: trapezoidal `dt1`, and
  `2γ·dt2` under each mask the remainder visits). A small per-mask
  cache keyed on `dt2` would not pay: its keys never repeat exactly
  with a moving duty, and with a fixed one they repeat only up to an
  ulp. Keying on a rounded `dt2` would change numbers at 1e-16
  relative; not done.
