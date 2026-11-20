# Release findings, group D3: TR-BDF2's restart at a gate edge and at the run start

> **STATUS: the five failing tests are fixed; step 2 re-measured on the
> final variant (section "Successor session" at the end). The sections in
> between are the predecessor's record (measured at dd75634d / 2d062bc5).**

Branch `fix/core-release-d3-trbdf2-gate-edge-restart`, base cad6ab6b
(`fix/core-release-d2-trbdf2-stateful-restart`). Tests:
`python/tests/test_release_findings_d3.py`; the strict xfail on
`test_release_findings_de.py::test_coss_conduction_energy_to_1pc_at_defaults`
is removed (it passes), and the E_oss turn-on band there is 1e-2 (was 8e-2).
Only `core/include/pulsim/solver/trbdf2_transient.hpp` changes in C++; no
binding changes.

## Cause

A landed gate edge (or source corner) committed the zero-time cascade's last
probe: one trapezoidal step of h_p = max(h_probe_floor, 1e-6 h_pre). Its
first slot was the PRE-edge derivative. Across the C_oss loop of the release
C_oss buck (tau = 0.6 ps), the probe stopped at 38.94 V where the ODE is at
30.67 V after 2.8e-13 s. The loss port then booked it from the right limit's
2.3 MW. Result: +3.9 % conduction energy at the defaults.

## Fix

The probe is recorded, not committed. The step after the edge starts from a
consistent x0+: the states are continuous, and the algebraic unknowns and
slots are those of the new mode.

- **With a stateful kind** (`hs.any()`): the next step is the SDIRK(2,2)
  restart from the committed x, as after a diode event (`sdirk_gate`).
  - Its backward-Euler stages read no slot.
  - Its commit stores the new mode's slots.
  - **At the tip** this applies only when the probe's mask differs from
    `x_mask`. A source corner or an edge that changes no mask takes the
    probe-slot path below, which also refreshes the q rows
    (`hs.refresh_derivatives`).
- **Without one:** the linear slots are replaced by the probe's own companion
  values (`history.refresh_derivatives(x_probe, h_p)`, which KCL balances
  under the new mode). The next step is TR-BDF2 from x. Three things are
  attached to that path:
  - esdirk43's explicit stage may read those slots (`edge_slots_t/m` guard).
  - The first post-edge Newton is warm-started from the probe.
  - An SDIRK step there was rejected for three reasons:
    - +40 to +90 % accepted steps (buck 2471 → 4207, VSI 2347 → 4426).
    - esdirk43's two_phase_interleaved_boost and long_run_drift_buck left
      their class.
    - On every Newton circuit, ll12 and ll10 went agree → disagree.
- **A probe that Newton retook ×100 longer stays committed**, as before.
  - Its slots belong to t + dtp, about 20 time constants of the new mode
    past the edge (ll13: off-state drain tau ≈ 2e-13 s, probe 4.6e-12 s).
  - A TR-BDF2 step from x that read them was rejected down to h_min (the
    error did not shrink with h).
- **A step that ends within h_min/2 short of a source corner now lands on it.**
  - With t no longer advanced by the probe, steps stay on a grid of the PWM
    period, and a step reached a corner 3.4e-18 s early by rounding.
  - That step was not clamped, and the next step's search skips the corner
    (it looks past t + h_min/2).
  - So the step read the snapped source's right limit before the jump: ll12
    and ll10 went agree → disagree on V(ctrl).
  - Now it lands and reads the left limit, as every landed step does.
- **Start (task 2b):** a start whose slots already balance the circuit to
  tolerance takes no restart step.
  - The check probes h_min, refreshes the linear entries and the q rows, and
    requires |Δslot|·h ≤ atol + rtol·|y|.
  - Before, a stateful kind forced one SDIRK step at every start.

What the probe keeps:
- the zero-time diode cascade's mode decision at the edge;
- the post-edge sample with its switch state (loss and charge integration
  across the edge);
- the hooks and the loss tracker's `event`;
- the gate toggle learning;
- the linear path's slots and warm start;
- the slots-only replacement at t_end, a tick, or a boundary settle;
- the commit, when the probe had to be retaken longer.

## C_oss conduction energy (booked / closed form − 1, `_conduction_truth`)

| call | cad6ab6b | tip |
|---|---|---|
| defaults | +3.906 % | +0.057 % |
| rtol 1e-6 | +3.425 % | +0.013 % |
| rtol 1e-8 | +2.198 % | +0.007 % |
| dt 4e-9 | +0.121 % | +0.021 % |
| esdirk43 | +5.143 % | +0.140 % |
| esdirk43, rtol 1e-6 | +4.640 % | −0.014 % |
| esdirk43, dt 4e-9 | +0.160 % | +0.023 % |

## Truth-backed rows (`.scratch/d3_bench.py`, base vs tip)

- **Exact scoreboard, trbdf2:** 81 cells [80, 30, 9] → [80, 30, 9].
- **Exact scoreboard, trbdf2 + esdirk43:** [81, 65, 13] → [81, 65, 14].
  - synchronous_flyback improves class (2.28e-8 → 7.27e-10).
  - No cell leaves its class.
  - Per-cell ratios: 60 better, 32 worse, 60 the same.
  - long_run_drift_buck esdirk43: 8.54e-10 → 9.86e-10, same class.
- **reference_status, trbdf2:** 106 / 2 / 1 → 106 / 2 / 1; analytic 27 / 27
  agree.
  - The 2 disagreements (high_q_rlc_ringdown, ll13) are unchanged.
  - The error (ll11's Newton abort) is unchanged.

The moved set, max error against truth, at the defaults:

| cell | old | new | class |
|---|---|---|---|
| synchronous_flyback esdirk43 | 2.28e-8 | 7.27e-10 | better |
| cascaded_boost_buck esdirk43 | 3.96e-8 | 1.09e-9 | same |
| boost_classic / erickson boost esdirk43 | 1.53e-8 | 2.44e-9 | same |
| two_phase_interleaved_boost esdirk43 | 1.65e-8 | 1.68e-9 | same |
| synchronous_boost esdirk43 | 5.66e-9 | 2.07e-8 | same |
| ll09 half-bridge trbdf2 | 3.24e-6 | 2.65e-6 | same |
| ll10 full-bridge trbdf2 | 4.91e-5 | 5.55e-5 | same |

## Byte compares vs cad6ab6b

- **pwl_identity:** 390 / 390 identical.
- **Engines oracle:** dsed, exact, pwl, pwl+restart, bdf1 and adaptive are
  identical. Only trbdf2 lines differ (buck, satind, mosl1, syncbuck).
- **trbdf2_identity:** 221 / 371 identical.
- **Stateful oracle:** 9 trbdf2 cases differ.

## Step counts and cost

Accepted / rejected steps, and instructions per accepted step (min of 3,
interleaved with base):

| case | base | tip |
|---|---|---|
| buck | 2471 / 201, 14865 | 2471 / 201, 14724 |
| sync buck, dead time | 2987 / 209, 15336 | 2987 / 209, 14949 |
| three-phase VSI | 2347 / 94, 46130 | 2349 / 95, 45734 |
| flyback | 28626 / 600, 12106 | 23651 / 541, 12454 |
| C_oss buck | 7420 / 187, 73193 | 7516 / 192, 83528 |
| bridge | 1199 / 58, 12932 | 1199 / 58, 12926 |

- **Flyback:** −15 % instructions per run.
  - Diode events 199 → 166.
  - V(out) agrees to 1e-7 relative.
  - Its smallest post-event step is 1.45e-14 s (was 5.8e-13).
- **C_oss buck:** +16 % per run. It takes 94 SDIRK steps (was 55).
- **ll13:** 1624 accepted / 11 Newton retries (was 1538 / 7).
- **esdirk43:** buck 1675 (was 1676); flyback 7487 (was 7291).

## Task 2: the predecessor's "away" list

The jump-aware metric (`.scratch/d3_jump.py`) compares against pwl + restart
at t_end / 1e6 and splits all samples, samples farther than 20 truth dt from
an event, and the time-integrated L1. Runs: d85 = 85637327 (before D2),
cad6ab6b (after D2), and the tip.

**(a) rect_gapped, diode node: metric artefact.**
- Away-from-events rms: 1.079e-4 / 1.076e-4 / 1.076e-4.
- All-sample rms: 1.39e-2 / 1.49e-2 / 1.48e-2.
- The 8.8e-3 → 1.1e-2 move is samples at the 6 events. Their timing is
  within one truth dt.

**(a) everything, input-branch column (col 8): metric artefact.**
- Away-from-events rms: 5.0e-3 / 6.1e-3 / 6.1e-3. Away max: 0.132 on all
  three.
- The tip's L1 is 8.6e-6 → 1.5e-5, all-sample max 0.79 → 1.64.
  - All of the L1 growth is within 20 truth dt of an event: the near-event
    share goes 6.1e-6 → 1.3e-5, and the away share is unchanged at 2.5e-6.
  - The worst sample sits 1.6e-10 s from an edge whose engine and truth
    times differ by less than one truth dt (2e-10 s).
- The states got better at D3:
  - cols 2 / 3: 8.5e-6 → 3.6e-6 rms;
  - col 11: 1.05e-5 → 5.5e-6.
- D2 had already taken col 10 from 3.7e-2 to 3.1e-6 away from events.

**(a) gbuck_lauritzen: metric artefact.**
- Away-from-events rms, col 1: 7.2e-2 / 3.1e-2 / 3.1e-2. Col 4: 2.5e-2 on
  all three.
- The states (cols 2, 3, 5) are 1.4e-7 to 1.6e-7 on all three.
- The algebraic L1 rises near events only (col 1: 4.4e-6 / 8.3e-6 / 9.8e-6).

**(b) satind +5 %:**
- The start is now detected as consistent: the sine from 0 into R + a
  saturable L gives slot v_L(0) = 0.
- n_sdirk_steps = 0 on trbdf2 and esdirk43. The first step is TR-BDF2's
  (ESDIRK's) own, sized by the controller from h_init.
- The error against truth is unchanged within step noise. esdirk43's
  early-time error is 1e-15 → 1e-20 absolute.
- Inconsistent starts keep the restart step (both are tested):
  - a current-driven nonlinear capacitor (F-S-1);
  - a gated buck with a Lauritzen diode. Without the step, its first
    TR-BDF2 step was rejected to 4e-13 s.

**(c) Saturable boost, i_L at its own rtol: tolerance noise, not a loss.**

Error against the esdirk43 rtol 1e-10 truth (c6 = i_L, c3 = v_out):

| rtol | d85 | cad6ab6b | tip |
|---|---|---|---|
| 1e-5 (defaults), i_L | 1.543e-5 | 1.626e-5 | 1.641e-5 |
| 1e-5 (defaults), v_out | 1.222e-4 | 1.157e-4 | 1.166e-4 |
| 1e-4, i_L | 3.66e-5 | 3.17e-5 | 3.14e-5 |
| 1e-6, i_L | 4.22e-6 | 4.23e-6 | 4.36e-6 |

- The changes go both ways across tolerances and between i_L and v_out.
- All of them stay rtol-proportional. pwl + restart at 2e-10 agrees with
  the truth to 1 %.

## Open at this tip: the failing tests and the mode-change variant

**At dd75634d (SDIRK after every edge of a stateful circuit), 12 tests
failed.** Each was checked on cad6ab6b without the conftest, against
`.scratch/d3base`; all of them pass there. (Note: `build/python` is a
symlink to `python/pulsim`, so the conftest always imports the worktree
build.)

- `test_trbdf2_stateful_error_norm.py::test_an_idle_stateful_device_takes_no_extra_steps`
  ×4.
  - Cause: the pulse source's corners on an RC beside an idle IGBT tail
    each took an SDIRK step, which the tail-less twin did not.
- `test_esdirk43_stateful_kinds.py`: `lauritzen` / `igbt_tail` /
  `pmsm_locked_rotor` `_is_fourth_order` ×3.
  - Cause: the order loss came from the SDIRK step at source corners.
- `test_esdirk43_engine.py::test_in_kernel_losses_agree_with_trbdf2_at_tight_tolerance[_loss_buck]`
  and `[_loss_vsi]`.
- `test_esdirk43_engine.py::test_stateful_kinds_through_diode_events_agree_with_a_fine_trbdf2`.
- `test_trbdf2_resume_at_gate_edge.py::test_an_edge_the_first_run_committed_is_not_landed_again`
  `[8e-05-None]` and `[8e-05-esdirk43]`.

**The tip's variant** (uncommitted before this WIP commit):
- `lin = !sdirk_on_linear_circuits && !(hs.any() && m_probe != x_mask)`.
- The probe-slot path also calls `hs.refresh_derivatives`.
- The long-probe commit also calls `hs.commit` (Trapezoidal).

It fixes the 7 tests in the first two groups. The 5 in the last three still
fail. Diagnosis so far:

- **`_loss_buck` on esdirk43:** the run counts 39 gate events and 39
  derivative refreshes, where cad6ab6b counts 40. S1 books n_on 19 vs 20.
  - The switch transitions in the record are the same 40 on both. Every
    edge has a landing sample and a +1e-12 post-edge sample on both.
  - So one edge is landed without being counted, or without a restart.
  - At 2.2e-4 the step before the edge is 1.29e-8 s (1e-6 on base), and
    both runs have a sample at edge + 2.7e-20.
  - Suspect: a step that ends one ulp past the edge. The edge is then
    handled by the boundary settle or not landed (`landed_on_edge` false),
    so neither `n_gate_events` nor `lt->event` fires. This is the
    gate-edge analogue of the source-corner fix in dd75634d. Unconfirmed.
- **`_loss_vsi`:** E_cond differs by 5.1e-9 J of 1.8e-4. The limit is
  2e-5 of scale, 3.6e-9. Likely the same missed edge or restart.
- **Resume test and stateful DCM buck:** not looked at.

The truth-backed rows (scoreboard, reference_status, byte compares, cost,
C_oss energies) were NOT re-run on the mode-change variant.

## OpenSpec

- `update-trbdf2-post-event-restart`: the spec requirement now says the
  probe is recorded, not committed, with the SDIRK / probe-slot split. A
  C_oss scenario is added, and DECISIONS 22 records the change.
- Task 4.3 is NOT ticked. The DCM buck's smallest post-event step is still
  3e-13 s (the diode turn-offs), so it is not closed by this change.

## Verified, and not

**At dd75634d** (SDIRK after every edge of a stateful circuit):
- Extension build.
- The 5 C++ binaries, run directly on Eigen 5: dsed (142 cases), step
  (32), sim (8), thermal (9), layer5_v2 (37). All pass.
- Touched pytest files (44 files matching trbdf2 / restart / loss /
  esdirk / stateful / release_findings): 466 passed, 12 failed (listed
  above).
- Byte compares, scoreboard and reference rows, step counts, cost, C_oss
  energies, jump metrics, boost.
- ruff passes.

**At the tip** (mode-change variant):
- Extension build.
- The 4 failing files plus test_release_findings_d3 / de: 85 passed,
  5 failed.

**Not run:**
- The full pytest.
- Eigen 3.4.0, g++, the full ctest.
- The reviewer.
- On the mode-change variant: the scoreboard / reference / byte
  compares / C++ binaries.

## Next step (successor)

1. Find the uncounted gate edge on `test_esdirk43_engine._loss_buck`
   (esdirk43, rtol 1e-8; tip 39 events vs base 40).
   - Add a print at `++st.n_gate_events` in `tb_commit` and at the entry
     of `restart_after_gate_event`, then diff the edge times against the
     40 expected (k·5e-5 and k·5e-5 + 2e-5).
   - If a step ending within h_min/2 past or before a located gate edge
     is the cause, land it as dd75634d does for source corners.
2. Re-check `_loss_vsi`, `test_stateful_kinds_through_diode_events_agree_with_a_fine_trbdf2`
   and `test_trbdf2_resume_at_gate_edge`. Compare each with cad6ab6b
   (copy python/tests without conftest.py, PYTHONPATH=.scratch/d3base/python).
3. Re-run on the final tip:
   - `.scratch/d3_bench.py <dir> both`, `.scratch/d3_idc.sh compare`,
     `.scratch/d3_steps.py coss` / `instr 1` ×3 interleaved;
   - the 5 C++ binaries;
   - the 44 touched test files;
   - ONE full pytest (background).
   - Required: ref trbdf2 106/2/1, no scoreboard class worse, pwl / dsed
     identical, C_oss conduction within 1 %.
4. Run the fresh-context reviewer on the diff against cad6ab6b.
5. Clean `.scratch`: d3A, d3B, `bt`, the bench dirs except base and final.

## Successor session (tips c387433b, bb07661f)

### The five failures: causes and fixes

| test | cause | fix |
|---|---|---|
| `_loss_buck` (esdirk43) | The edge at t_end = 1e-3 was never landed. esdirk43's 1 µs steps ended at 0.00099999999999999815, 1.8e-18 s short of it. The clamp landed only `edge <= t + h`. With the probe no longer advancing t, steps stay on the PWM grid. Result: 39 gate events, S1 n_on 19 (base and trbdf2: 40 / 20). | ONE landing rule in `tb_clamp` for ticks, source corners and gate edges (`lands`). An instant at or before t + h + h_min/2 ends the step. The commit sets `t = t_land`, the queue's or the locator's own number, not `t + h`. The slack applies to the first landing only and never past t_end (review). The predecessor's corner-only slack is folded in. |
| `_loss_vsi` (esdirk43) | Not a missed edge. `LossTracker::accept_step` starts a step from the event's right limit only for devices that FLIPPED. An off switch whose v jumps with its leg kept its left-limit reading at the step's start weight. The base's committed probe step had hidden that (h_p long). esdirk43 E_cond drifted by about −1e-9 J/ms against the exact engine (−5.1e-9 J on INV__LSa; limit 3.6e-9). trbdf2 drifted +1e-10, and the error did not converge with rtol. | `LossTracker::event` also restarts the unflipped devices' readings from the right limit. The gate-edge restart reports every probe, also under an unchanged mask (a source corner). Against `engine='exact'`: esdirk43 is within 1e-12 J at 0.25 to 2 ms; trbdf2 equals the base's. |
| `test_an_edge_the_first_run_committed_is_not_landed_again` [8e-05-None], [8e-05-esdirk43] | Pass after the landing rule. Not traced separately. The split run ends on an edge at t_end, which the landing rule lands at exactly t_end. | the landing rule |
| `test_stateful_kinds_through_diode_events_agree_with_a_fine_trbdf2` | The EXPECTATION. At the first turn-on (1e-5) D still carries i_L (v_D = +4 mV). It turns off when C_oss lifts v_sw through 0, 1.9e-16 s after the edge. The 2 ns reference's SDIRK step after the edge was rejected ×10 from 2e-9 down to 2e-17 s (the 1 ps C_oss transient). That is still before the zero, so the chatter guard re-decided D ON at the edge and landed the turn-off at +1.87e-16 s: 21 diode events for the same 20 record transitions. esdirk43 and default trbdf2 count 19, as the base does. Traced with prints. | The test compares D's record transitions (values exact, times within 1 ns) instead of `n_diode_events`. The reason is written in the test. |

Tried and reverted, because neither was the cause:
- skipping `near_side_clear` after an edge;
- flooring the post-edge h at `h_ctrl`.

### Review (fresh context, opus, medium; static reading only)

1. **(high) Fixed in bb07661f.** On the SDIRK path the slots stayed pre-edge. A diode landing inside that first step commits a trapezoidal step from them. The probe's slots now replace them on both paths. The SDIRK stages and their estimate read none of them, so the C_oss numbers, steps and Newton count are unchanged.
2. **(medium) Not fixed.** A run split exactly ON an edge does not resume bit-identically. The snapshot does not carry `sdirk_left` / `sdirk_gate`, `edge_slots_t/m` or `x_edge`. A resume skips the start restart. The uninterrupted record also has the post-edge sample that the split lacks. cad6ab6b was not bit-identical there either: it committed the probe in the uninterrupted run and did slots-only at the split. The resume tests bound the final state at 2e-5 and pass. A split between edges is unaffected. Closing it needs a snapshot block (pickle field 19 or later).
3. **(medium-low) Fixed in bb07661f.** `lands` could lengthen h past an earlier landing, or past t_end.
4. **(low) Open.** `LossTracker::event` writes unflipped devices' readings with no rollback when the cascade undoes the event. No energy is booked for them. The next step's start reading is the uncommitted right limit.

The reviewer also named one pre-existing case, not D3's: `edge_at_resume`'s h_min step still commits a TR-BDF2 step from pre-edge slots under the post-edge mask.

### Step 2 on bb07661f

C_oss conduction energy (booked / closed form − 1). All of these are within 1 % and equal the predecessor's numbers:

| call | value |
|---|---|
| defaults | +0.0574 % |
| rtol 1e-6 | +0.0134 % |
| rtol 1e-8 | +0.0066 % |
| dt 4e-9 | +0.021 % |
| esdirk43 | +0.140 % |
| esdirk43, rtol 1e-6 | −0.014 % |
| esdirk43, dt 4e-9 | +0.023 % |

Exact scoreboard and reference rows:
- **Exact scoreboard, trbdf2:** [80, 30, 9], unchanged.
- **trbdf2 + esdirk43:** [81, 65, 13] → [81, 65, 16]. Three cells move up a class: boost_classic and textbook_erickson_boost_ccm (1.53e-8 → 6.78e-10), and synchronous_flyback (2.28e-8 → 7.27e-10).
- No cell leaves its class. Per cell: 58 better, 34 worse, 60 the same.
- The largest move without a class change: ll10 trbdf2 4.91e-5 → 6.87e-5 (×1.4, class 1e-3).
- **reference_status, trbdf2:** 106 / 2 / 1, analytic 27 / 27.
  - ll10 and ll12 agree. ll13 disagrees, as on the base.
  - The 2 disagreements are high_q_rlc_ringdown and ll13. The error is ll11.

Byte compares against cad6ab6b:
- pwl_identity 390 / 390 identical.
- Engines oracle: dsed, exact, pwl, pwl+restart and bdf1 are identical. Only the trbdf2 lines differ (buck, satind, mosl1, syncbuck).
- trbdf2_identity: 221 / 371 identical.
- Stateful oracle: 9 trbdf2 cases differ (saturable, gapped, lauritzen, igbt_tail, rect_atan, rect_gapped, gbuck_coss, gbuck_lauritzen, everything).

Steps and cost. Accepted / rejected steps, then instructions per accepted step (min of 2, interleaved):

| case | base | tip |
|---|---|---|
| buck | 2471 / 201, 14873 | 2471 / 201, 14774 |
| sync buck, dead time | 2987 / 209, 15363 | 2987 / 209, 15034 |
| three-phase VSI | 2347 / 94, 46333 | 2349 / 95, 45645 |
| flyback | 28626 / 600, 12067 | 23651 / 541, 12406 (−15 % per run) |
| C_oss buck | 7420 / 187, 73129 | 7516 / 192, 80246 |
| bridge | 1199 / 58, 12942 | 1199 / 58, 12926 |

The C_oss buck costs +11 % per run. The main cost is the SDIRK restarts after its 40 edges, which run from the pre-edge x into the 0.6 ps C_oss transient:
- Newton iterations (= factorisations) 51337 → 55122 (+7.4 %).
- 94 SDIRK steps (was 55).
- 15 forced accepts at h_min (was 0).
- The smallest post-event step is 4e-15 s (h_min; was 3.7e-14).

Resolving the transient is the price of the accuracy (+3.9 % → +0.06 %). The 15 forced accepts are waste: they are steps whose estimate could not shrink before h_min.

### Task 4.3 of `update-trbdf2-post-event-restart`: NOT ticked

On the DCM buck of `test_trbdf2_post_event_restart`, every interval below 1e-11 s is one post-edge record sample per edge (800 of 800). On this tip those samples are no longer steps. The smallest first post-event accepted step is 2.3e-7 s. The step half of the criterion therefore now holds on that circuit. The `i_L` error half (≤ 1e-9 A) was not measured. The 3e-13 s that the predecessor reported is the golden buck's (`d3_steps`, `hpe_min`), after a diode turn-off: a real step in an L·g_off mode, not a probe.

### Verified on bb07661f
- Extension build (Release).
- The 5 C++ binaries on Eigen 5, at c387433b: dsed 142, step 32, sim 8, thermal 9, layer5_v2 37, all passed. bb07661f changes only the gate-edge restart, and they were not re-run on it.
- 44 touched files: 478 passed, 1 xfailed (fixed-engine F-4-5).
- ruff passes.
- No binding change, so stubs were not needed.
- Full pytest (serial): 6220 passed, 110 skipped, 8 xfailed, **6 FAILED**.
  None of the 6 is in the 44 touched files. None was checked against cad6ab6b or 2d062bc5 (the predecessor never ran the full pytest).

**Not run:** Eigen 3.4.0, g++, the full ctest (integration batch).

### OPEN: 6 full-pytest failures on bb07661f — the branch is NOT mergeable

Suspected cause: the D3 design, not the landing or loss fixes. The edge's
probe is now a recorded sample, not an accepted step.

| test | failure | suspected cause |
|---|---|---|
| `test_external_device.py::test_device_output_covers_a_pwm_run_with_gate_edges_and_a_diode[kw1]`, `[kw2]` | `UnsupportedError`: "device_output('M', 'i(a)'): 39 stored sample(s) ... are not accepted steps of the run" | The 39 post-edge samples are not steps. The named-output recorder needs a value there: record the probe's device outputs at the post-edge sample, or record the sample as an event row. |
| `test_recorder_batch_combinations.py::test_rows_across_batch_boundaries_with_events[trbdf2]`, `[esdirk43]` | the recorder's rows lack 39 of the `res.times` rows (1019 vs 1058, 3266 vs 3305) | The same samples. The in-loop recorder sees no step hook for them. |
| `test_benchmark_harness.py::test_counter_gate[hard_switched_rc]` | n_reject 23 > frozen 17 | Re-freeze the counter only with the cause written down. |
| `test_benchmark_harness.py::test_counter_gate[flyback_converter]` | n_forced_accepts 744 > frozen 3; n_chatter_breaks 33 > frozen 0 | A regression signal: 744 steps forced at h_min. Trace it first: where they sit (post-edge SDIRK from the pre-edge x?), and whether 2d062bc5 already has them. Do not re-freeze it blindly. |

Next step for the successor:
1. Run these 6 against `.scratch/d3base` (cad6ab6b) and against 2d062bc5 to attribute them.
2. Fix the device-output / recorder feed at the post-edge sample: one rule for both.
3. Trace the flyback's forced accepts.
4. Rerun the full pytest and the step-2 rows.
