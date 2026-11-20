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
