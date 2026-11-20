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
