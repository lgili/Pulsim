# Release findings, group D2: TR-BDF2's SDIRK restart with stateful kinds

Branch `fix/core-release-d2-trbdf2-stateful-restart`, base 85637327
(`fix/core-release-de-trbdf2-tick`). Tests: `python/tests/test_release_findings_d2.py`,
`test_release_findings_de.py::test_branch_form_flyback_runs_on_trbdf2`.

## F-3-4 — trbdf2 aborts with branch-form switch + conductance diode next to a saturable transformer

Cause SHOWN (enabling the restart on TR-BDF2, rtol 1e-6, `.scratch/d2_land.py`):

| pair, run | base 85637327 | with the restart |
|---|---|---|
| conductance/conductance, 1 ms: smallest step after a diode landing | 2.2e-13 s (median 1.7e-12; 26 of 99 landings < 1e-12 s) | 1.7e-10 s (median 1.7e-10; 0 < 1e-12) |
| same: rejects / Newton retries | 3222 / 33 | 2359 / 0 |
| branch/conductance, 0.46 ms | 5.8e-14 s (10 of 45 < 1e-12), aborts at 0.47 ms | 1.7e-10 s (0 of 45) |

So the reject cascade after every landing was the missing restart (TR-BDF2's
stage read the pre-event derivative slots; `sdirk_possible = !hs.any()`
removed the SDIRK step whenever a stateful kind was present).

Fix (`trbdf2_transient.hpp`): the SDIRK(2,2) restart step runs the stateful
kinds' q rows through esdirk43's existing mechanism on both integrators —
`ed_sf = hs.any()` (was `esdirk && hs.any()`); the three `if constexpr (kE)`
guards around the restart's q-row stages (stage 1 BE of gs·h with an
`hs` rollback point, stage 2 base `ed_sdirk_base`, restore) dropped; the
commit `ed_hs_commit(x_1, gs·h, h)` taken on every SDIRK step
(`(kE || sdirk) && ed_sf`). `sdirk_possible` and its `dependent_states`
refusal deleted. The stages set the device step and stage through
`step_ctx` (BackwardEuler, gs·h), as esdirk43's restart does; each restart
starts from the committed (consistent) state.

All four pairs now run: 15.310393 V (esdirk43 15.31034 V; pwl dt 1e-8 15.3095 V).

## The `dependent_states` refusal becomes a run

trbdf2 + an ideal switch opening an inductor cut-set + ANY stateful kind
(here an unrelated Lauritzen diode): refused on the base on both
integrators; now runs. Closed form (flux conserved across the jump, then
i = i+ exp(−R t/(L + L2))): max error after the jump 3.8e-9 A (defaults),
2.2e-9 A (rtol 1e-6), esdirk43 3.4e-8 A, of i+ = −0.0604 A.

## F-4-5 remainder — the loss port's conduction energy +4.4 % with a nonlinear C_oss

TRACED, not fixed in this slice (cause outside the stateful restart; the fix
moves every trbdf2 gate edge, see below).

1. The booking, not the trajectory: on b85637327 defaults, booked E_cond
   4.3034e-5 J, closed-form truth (test's `_conduction_truth`) 4.1221e-5 J
   (+4.40 %), but the trapezoid of the run's OWN sampled v_S·i_S is
   4.1063e-5 J (−0.38 %). The excess lives in steps the samples do not show
   as power (`.scratch/d2_f45b.py`).
2. Per period (runs to (k+1)·20 us − 2 us with the full run's h_max, dt=4e-7,
   so every run is a prefix of the full one; the sum equals the full run to
   1e-9; `.scratch/d2_f45.py`): the excess sits at the turn-ons, +0.20 ..
   +0.25 E_oss on 13 edges and ≤ 0.08 E_oss on the 6 DCM edges (6–13) whose
   C_oss rings before the gate.
3. Those 13 edges follow a LONG step (2.8e-7 s) to the edge; the DCM ones a
   short one (1e-9 .. 1e-8 s). The landed edge commits a TRAPEZOIDAL probe of
   h_probe = max(h_probe_floor, 1e-6·h) (`trbdf2_transient.hpp`, the landed
   gate edge): 2.8e-13 s after a 2.8e-7 s step, i.e. 0.47 tau of the
   channel–C_oss loop (tau = R_on·C_oss(48 V) = 0.6 ps), against 1.8e-15 s
   after the DCM edges. Edge 1 (defaults): the committed probe lands at
   v_S = 38.94 V where the ODE C(v)·dv/dt = −v/R_on from 48.0043 V gives
   30.67 V at the same instant — half the decay, the trapezoid's first
   derivative slot being the PRE-edge capacitor current (≈ 0). The loss
   tracker books that step as h·(p_r + p_1)/2 from the right limit's
   2.3 MW: ≈ 5.3e-7 J while the capacitor released E_oss(48.00) −
   E_oss(38.94) = 2.4e-7 J. Later steps start from a KCL-consistent slot
   (the probe's own) and track the ODE shifted in time.
4. Control (same build, only h_max changed, so only the pre-edge step and
   therefore the probe shrink): E_cond vs truth

   | run | booked / truth − 1 |
   |---|---|
   | defaults (h_max 4e-7) | +4.40 % |
   | dt = 4e-8 | +0.62 % |
   | dt = 4e-9 (probe ≤ 4e-15 s) | +0.09 % |
   | rtol 1e-8 | +2.08 % |
   | rtol 1e-8, dt = 4e-9 | +0.08 % |
   | esdirk43 | +3.81 % |
   | esdirk43, dt = 4e-9 | +0.16 % |

   So the remaining excess is the committed gate-edge probe being a
   trapezoidal step across the derivative jump that is not short against
   the fastest mode it excites: its length scales with the PRE-edge step,
   not with anything the post-edge circuit sets. The SDIRK restart of
   this slice does not touch it (a gate edge restarts by the probe, not by
   SDIRK). Quantitatively the first-step estimate (≈ 3e-7 J/edge) over-
   predicts the measured per-edge excess (1.9e-7 J): the split between the
   probe and the first TR-BDF2 steps after it was not measured step by step.
5. Fix candidates (NOT done, owner of the trbdf2 gate-edge restart): make the
   committed probe short against the post-edge dynamics (h_probe_floor with
   the existing learned ladder for conditioning) or commit it as a
   backward-Euler step (no slot read). Either moves EVERY trbdf2 run with a
   gate edge (all trbdf2 goldens), which this slice cannot truth-back case
   by case. The fixed engine's E_oss booking stays a strict xfail
   (untouched, `run_transient.hpp` belongs to another agent).

## F-S-1 — esdirk43 aborts at t = 0 with a current-driven nonlinear capacitor

Minimal case (copied from `test_soft_switching_validation.py` on
`feat/core-soft-switching-validation` 45076ddb): 0 V source, 1 mΩ,
`add_nonlinear_capacitor(C0 = 1 nF, m = 0)` (a linear charge law), 10 A
into the node. Truth: v = −I·R·(1 − exp(−t/RC)), tau = 1 ps, end −0.01 V
(the brief's I·t/C does not apply: the 1 mΩ path carries the steady current).

On the base: esdirk43 aborts at t = 0 (Newton at h_min, ||dx|| 8.7e-14,
residual 9.3e-7 on KCL sw); TR-BDF2 runs (its q rows join only after the
first commit) with max error 1.8e-3 V in the first ps; `add_capacitor` runs
on both. Cause: the same inconsistent-slot start as F-3-4 — the stateful
kinds' derivative slots start at init's zeros, ESDIRK's explicit first
stage reads q' = 0 where the capacitor carries −10 A, and its q-row
estimate (live from the first step, DECISIONS 17) is O(h) at every h down to
h_min, where Newton stalls on round-off. Fix: a run that starts (not a
resume) with a stateful kind takes its first step as the SDIRK(2,2) restart
step of F-3-4 (`sdirk_left ≥ 1`), whose backward-Euler stages read no slot;
the commit stores the consistent one. After: both integrators run; end
−0.01 V; error after 50 ps < 1e-8 V. Left: the first step (h_init = 3e-12 s
= 3 tau) is not error-controlled on the q rows (an SDIRK step's estimate
covers the linear entries only), so the first ps is unresolved (max error
1.19e-3 V on both integrators, overshoot to −0.01069 V at 3 ps).
F-S-1's strict xfail on `feat/core-soft-switching-validation` turns XPASS
when that branch meets this one: remove the mark at the merge.

## Moved cases — first pass (tip 8508df9b), and why a third commit followed

`trbdf2_identity` (goldens + reference × modes '', lcp, shadow + extra) on
85637327 vs 8508df9b: 367/371 identical; `pwl_identity` 390/390 identical.
Moved: `stateful rectifier atan`, `stateful rectifier gapped`,
`example boost_saturable_inductor.yaml`, `example hysteretic_inductor_demo.yaml`.
Error at each engine's OWN samples against a dense truth interpolated there
(relative to the component's max; `.scratch/d2_pts.py`):

| case, component | truth | old rms | 8508df9b rms |
|---|---|---|---|
| boost_saturable_inductor v_out | esdirk43 rtol 1e-10 | 1.220e-4 | 1.798e-4 (AWAY) |
| boost_saturable_inductor i_L | same | 1.554e-5 | 5.335e-5 (AWAY) |
| rect_atan v_o / i_L | trbdf2 rtol 1e-10 | 4.671e-6 / 5.234e-6 | 4.695e-6 / 5.262e-6 |
| rect_gapped v_o / i_L | same | 1.069e-4 / 2.929e-4 | 1.066e-4 / 2.926e-4 |
| hysteretic demo (all states) | pwl dt 1e-9 | 1.293e-4 | 1.292e-4 |

The boost moved away: an SDIRK restart step had no q-row estimate (the rows
were skipped on `sdirk`), so the step after each landing was uncontrolled
on the saturable inductor's flux — its only state. Third commit: the q rows
take the entries' embedded first-order estimate on an SDIRK step,
(1−gs)/gs·(d_g/gs − d_1) with d = q(X) − q_n (`trbdf2_increments`), also at
the run's first step (it reads no slot). Re-measured below.

## Moved cases — final (tip after the q-row commit)

Identity at the tip vs 85637327: `trbdf2_identity` 367/371 identical (the
same four move), `pwl_identity` 390/390 identical (measured on 8508df9b;
pwl does not include the trbdf2 header's loop; not re-run after the last
two commits). The SDIRK step's q rows read their own stage-2 increment
q(x_1) − ψ5 before the rollback (the rollback also drops the γ capture:
with `trbdf2_increments` every flyback pair aborted again — measured, fixed
in 86cecb95).

Per-sample rms error relative to the component's max (old = 85637327, new = tip):

| moved case, component | truth | old | new |
|---|---|---|---|
| boost_saturable_inductor v_out | esdirk43 rtol 1e-10 | 1.220e-4 | 1.155e-4 |
| boost_saturable_inductor i_L | same | 1.554e-5 | 1.635e-5 |
| same at rtol 1e-4 (v_out / i_L) | same | 3.746e-4 / 3.682e-5 | 2.962e-4 / 3.180e-5 |
| same at rtol 1e-6 (v_out / i_L) | same | 3.516e-5 / 4.312e-6 | 3.392e-5 / 4.272e-6 |
| rect_atan v_o / i_L | trbdf2 rtol 1e-10 | 4.671e-6 / 5.234e-6 | 4.679e-6 / 5.244e-6 |
| rect_atan v_o / i_L | esdirk43 rtol 1e-10 | 6.576e-6 / 6.719e-6 | 6.518e-6 / 6.788e-6 |
| rect_gapped v_o / i_L | trbdf2 rtol 1e-10 | 1.069e-4 / 2.929e-4 | 1.067e-4 / 2.924e-4 |
| hysteretic demo (two states) | pwl dt 1e-9 | 1.293e-4 / 5.220e-5 | 1.282e-4 / 5.176e-5 |

Verdict: the boost moves toward the truth at three tolerances out of three on
v_out and two of three on i_L (+5 % at the example's own rtol 1e-5); the
rectifiers move by ≤ 1 % of their error, both ways (step-sequence noise);
the hysteretic demo moves closer. No case moves away systematically.

Test-suite cases that moved (pytest, touched files):

| case | old | new | truth |
|---|---|---|---|
| F-3-4 flyback, branch switch / conductance diode, rtol 1e-6 | aborts at 0.47 ms | 15.310393 V | esdirk43 15.31034, pwl dt 1e-8 15.3095 |
| F-3-4, the other three pairs | 15.310393 V | 15.310393 V | same |
| F-S-1 nonlinear C (m = 0) + 10 A, esdirk43 | aborts at t = 0 | end −0.01 V, max err 1.3e-7 V | −0.01·(1 − e^(−t/1 ps)) |
| same, TR-BDF2 | max err 1.8e-3 V (first ps) | 5.7e-7 V | same |
| dependent_states next to a Lauritzen diode (trbdf2 / esdirk43) | refused | max err 3.8e-9 A / 3.4e-8 A | flux-conservation closed form |
| Coss buck E_cond, trbdf2 defaults | +4.40 % | +3.91 % | closed form (`_conduction_truth`) |
| same, rtol 1e-6 / 1e-8 | +4.62 % / +2.08 % | +3.43 % / +2.20 % | same |
| same, esdirk43 defaults / rtol 1e-6 | +3.81 % / +4.75 % | +5.14 % / +4.64 % | same |
| same, esdirk43 dt 4e-9 / 4e-8 | +0.16 % / +1.16 % | +0.16 % / +0.98 % | same |

The Coss buck's scatter (both directions) is the F-4-5 probe defect riding on
the step sequence: with the probe made short (dt = 4e-9) old and new agree
to 1e-4 at +0.16 %. Its band in `test_coss_turn_on_energy_is_e_oss_on_the_variable_step_engine`
went 5 % → 8 % with the cause written next to it, and a NEW strict xfail
`test_coss_conduction_energy_to_1pc_at_defaults[trbdf2, esdirk43]` holds the
1 % the probe fix must reach.
