# R2 bucket 3 — engine route / result layout (77 failures, 20 files)

Branch `release/r2-bucket-3` from `origin/release/core-next-defaults` d2a15c91. Extension: the
R1 `.so` (`/Users/lgili/pulsim_wt/so_release_r1`). Baseline run of the 20 files: 77 failed /
756 passed / 3 skipped / 2 xfailed. The failing set equals R2_BUCKETS.md exactly (the strict XPASS
in `test_gated_device_seams.py` is one of the 77).

Note for anyone repeating this: `python/tests/conftest.py` prefers `<worktree>/build/python` when it
holds an extension. This worktree has one, and it was byte-identical to `python/pulsim` (`__init__.py`
and `.so` compared) while the runs were made. In a worktree whose `build/` is stale, the tests run
against the stale tree.

Rules: (a) physics against its truth, (b) legacy subject pins the legacy option, (c) an undriven
switch is driven, (d) the new route / layout / refusal is stated exactly. F = a release finding,
parked as `xfail(strict=True)`.

## Per file

| file | fixed | rule | notes |
|---|---:|---|---|
| test_trbdf2_robustness.py | 22 | b 21, d 1 | The subject is trbdf2: a module `TRB = {"engine": "trbdf2"}` is passed to every run whose subject is that engine. Two runs that passed on the exact engine (E2 bridges) were also pointed back at trbdf2, so their coverage is kept. The snubbed boost is parametrised: `auto` → `dsed` and `trbdf2`, both against pwl dt = 1e-8 54.48 V, **tolerance 0.1 → 0.01 V** (measured 54.4820 / 54.4829). The floating DC side also checks `auto` → `dsed` against the dt = 1e-6 truth (321.8520 vs 321.8520 V). The conditioning-floor island (`_buck_with_island`) pins `form='conductance'`, because its subject is the g_off stamp: in the branch form an OFF diode is open, the run has no floor, nothing is refused, and v_out agrees with the island-free run. |
| test_fixed_step_resolution.py | 11 | b | `_sim` sets `discontinuity_restart=False` by default. Every expected value (RC truncation estimates, the 11.11 V overshoot at 10 tau, CCM-buck LTE, ringing "without the restart rule") measures the restart-less step. The tests of the rule still pass True. |
| test_mode_pwl.py | 6 + 2 F | b, F-3-1 | `mode_selection='auto'` picks `lcp` on these circuits. The runs that are "the settle" now name `mode_selection='legacy'` (module `SETTLE`). The default guard exists only on the conductance form (`run_transient.hpp` turns it off for a branch-form diode), so the guard tests use a local `conductance` fixture that monkeypatches `add_switch` / `add_diode` (see *Needed from bucket 0*). The `CONFIGS` id "pwl" is the restart-less step again, now explicit (`discontinuity_restart=False`). The rounding fuzz of `bridge1_vf07` (default and lcp) is parked as F-3-1. |
| test_engine_conformance.py | 5 | b 4, d 1 | `CELLS` hold the conductance-form rows (`FORM_CELLS` hold the branch-form rows), so the Switch and Diode cells now name `form='conductance'` (bdf1 refuses the branch form by name). The guard test uses the settle of a conductance-form half-wave. The dependent-states pwl/False cell passes `discontinuity_restart=False`. `allow_implicit_segments`: `auto` with dt is still refused; `auto` without dt now routes to the exact engine, which honours the keyword (new test). |
| test_undriven_policy.py | 5 | d 2, b 3 | `USED["auto"]` = `dsed`. The tests of the per-engine None rule pin `_plan.DEFAULT_UNDRIVEN = None` through a fixture. Under that pin, `auto` without dt takes the exact engine's own rule (all closed, with the warning) and no longer refuses through trbdf2. That assertion now states the new behaviour. |
| test_exact_output_grid.py | 4 | d | The default is `output_rtol=1e-5`, asserted equal to the explicit value. `output_rtol=0` is the old grid and is no longer refused (removed from the bad-value list). Every run "without the rule" passes `output_rtol=0`. |
| test_branch_form_review.py | 2 + 1 F | d 1, b 1, F-3-2 | H4: `auto` without dt → `dsed`. M6: its subject is the restart-less step and the settle's loop release (`discontinuity_restart=False`, `mode_selection='legacy'`; under the LCP i_L is the same and there is no release to count). H1 is parametrised: `legacy` passes; `auto` is parked as F-3-2. |
| test_external_device.py | 3 | b | Both checks are identities of one trapezoidal step per stored sample: the junction charge (trapezoid of `i(pos)` = Q(400 V) = 312.31 nC) and the meter energy (rectangle rule over the stored grid). Only those runs pin `discontinuity_restart=False`. Under the restart: q = 292.19 nC by trapezoid of the samples while v(a) still reaches 400 V (the first step is backward Euler from the inconsistent start, and the t = 0 sample of 0 A was never integrated); the meter accepts sub-steps, and its energy is 7.8e-7 relative from the closed form 5e-5 J. Both are fine physically. The twin equalities pass unchanged under the new defaults. |
| test_control_tick_semantics.py | 2 | d, b | Two loops without dt → `dsed`, tick counts 200 / 20 unchanged. The multi-rate test's subject is trbdf2, so it names `engine='trbdf2'`. |
| test_gated_device_seams.py | 2 | b | The split follows the rule by default, so "the restart rule alone" passes `enable_substep_state_correction=False`. The event-queue xfail is strict again (it was XPASS). The plain run names `discontinuity_restart=False`. |
| test_mosfet_body_diode_default.py | 2 | c | The gate is driven OFF (`switch_fn=_all_off(b)`). |
| test_saturable_transformer.py | 2 F | F-3-3, F-3-4 | — |
| test_circuit_ir_integration.py | 1 | b | The non-branch switch names `form='conductance'`. |
| test_control_graph_blocks.py | 1 F | b + F-3-5 | Parametrised on `control_tick`: `label` passes; the default (`state`) is parked as F-3-5. |
| test_controller_state_isolation.py | 1 | c | Q is driven in all three runs. The `controller_state` refusal still fires. |
| test_dsed_auto_integrator.py | 1 F | F-3-6 | Parametrised: `output_rtol=0` passes; the default is parked as F-3-6. |
| test_ideal_loop_release.py | 1 | b | Mode None means "the default settle", so `_sim` maps it to `mode_selection='legacy'`. |
| test_linear_time_assembly.py | 1 | precondition | The claim (one structure, one symbolic analysis) is unchanged and passes. The run now visits 16 masks (16 with the restart rule alone, 32 restart-less), so the "many masks" precondition went from > 20 to > 10. This is a precondition, not a tolerance. |
| test_newton_stall_and_mask_chatter.py | 1 F | F-3-7 | — |
| test_shipped_yaml_examples.py | 1 | d | A YAML without a simulation block, run without dt → `dsed`. |

Re-run of the 20 files after the fixes: 0 failed / 829 passed / 3 skipped / 11 xfailed (2 pre-existing, the restored event-queue xfail, 8 findings).

Counts over the 77: (b) 54; (d) 11; (c) 3; precondition 1; F (parked strict xfail) 8 tests =
7 findings (F-3-1 covers 2 tests; F-3-2, F-3-5 and F-3-6 keep a passing legacy variant beside the
parked default). (a): no test failed on physics alone. The physics inside the route tests was
re-checked against its truth under the new route, and one tolerance was tightened. Nothing deleted, nothing skipped. Tests added: the `auto` variants of
the snubbed boost and the floating DC side, `test_allow_implicit_segments_takes_auto_to_the_exact_engine`,
and the legacy variants of H1, the rate-period refusal and the numeric-segment counters.

Tolerances tightened: snubbed boost end voltage 0.1 → 0.01 V (exact 54.4820, trbdf2 54.4829 V
against 54.48 V measured on pwl at dt = 1e-8).

## Findings

- **F-3-1** — The event split is not rounding-stable at a diode turn-on.
  - Call: `tests/test_mode_pwl.py` `_sim(_perturbed("bridge1_vf07", 5), 0.04, 1e-5)` vs `k = 0`. The floating single-phase bridge, V_f = 0.7, C or R moved by a few ulp.
  - Truth: the exact engine at dt = 1e-5. Both runs are about 3.4e-3 V off it after the turn-on at 13.71 ms.
  - Old (restart-less): the waveforms agree to 2.3e-11 V.
  - New (restart + split, any mode, any form): they differ by 3.07e-4 V at t = 13.71 ms. The difference alternates and decays over about 36 samples. The mode sequence is identical. With the split off (restart alone) the difference is 2.4e-11 V.
  - Reading: the split's located turn-on instant follows the root search, not rounding. The post-event error still alternates in sign (-3.4e-3, +3.8e-4, -4.6e-5 V), so the restart does not remove the ringing there either.
- **F-3-2** — An LCP refusal names a label as `.device`.
  - Call: `V(10 V) a-gnd; add_diode("D", "a", "gnd", form="branch"); R a-c 1 ohm; C c-gnd 1 uF`, then `simulate(b, 1e-4, dt=1e-6, engine="pwl")`.
  - Expected `.device == "D"` (the settle gives exactly that).
  - New, under `mode_selection='lcp'` (the release default here): `.device == "D (branch #1)"`.
  - Reading: the LCP refusal path passes `branch_label` where the device name belongs. `test_ideal_loop_release` had already accepted this with `startswith("D")`. Minor.
- **F-3-3** — The default fixed step turns a flyback's MOSFET off into its body diode.
  - Call: `tests/test_saturable_transformer.py` `_flyback(False, 1.0)` (linear coupled pair, `add_mosfet` with its default body diode), 100 kHz D = 0.5 PWM, `simulate(b, 1e-3, dt=1e-8)`.
  - Truth (exact engine, with and without dt): at turn-off i_p goes to 1.6e-7 A; peak 16.382 A; mean v_out 16.365 V.
  - Old (restart-less): body diode on, i_p = -15.48 A; peak 15.53 A; v_out 16.80 V.
  - New: body diode on, i_p 14.83 → -20.92 A, i_D1 = 68.7 A (lambda_s is conserved through the reversed primary); peak 20.92 A; v_out 17.15 V. `reinit_records` is empty.
  - Reading: both pwl variants pick an unphysical body-diode-ON mode at the cut-set the switch opens. The restart makes it larger. This is likely the open "ideal-switch flyback on pwl+restart" item in STATE.md.
- **F-3-4** — trbdf2 aborts when the switch and the diode are both branch form.
  - Call: `tests/test_saturable_transformer.py::test_engines_agree_below_saturation` circuit (snubbed saturable flyback), `engine='trbdf2', rtol=1e-6, atol=1e-9`.
  - Truth: pwl at dt = 1e-8 gives 15.3095 V.
  - Old (conductance form): 15.3104 V.
  - New (default branch form): `SimulationAborted`, Newton failed at h_min at t = 0.41 ms. With only the switch, or only the diode, in the conductance form the run completes at 15.3104 V.
- **F-3-5** — A control-graph build refusal loses its type under `control_tick='state'`.
  - Call: `simulate(rc, 1e-5, engine='pwl', dt=1e-7, step_observer=MixedDomainBlockChain(rate=Rate(5e-8))...make_step_observer(b))`.
  - Expected `ControlGraphBuildError` (a ValueError); `control_tick='label'` still gives that.
  - New (the default `state`): the same message as `SimulationAborted` (a RuntimeError).
  - Side note: the message prints the periods as "0.000000 s" (both ticks, pre-existing).
- **F-3-6** — The documented step counters read 0 under the default output rule.
  - Call: `tests/test_dsed_auto_integrator.py` pulse RC, `engine='dsed', b_extra_fn=zero, allow_implicit_segments=True`.
  - `output_rtol=0`: n_rk45_steps = 2022.
  - Default `output_rtol=1e-5`: n_exact = n_rk45 = n_bdf2 = 0, with the same 2023 samples and the same values (error 1.8e-7 of span vs pwl).
  - The counters are listed in `docs/result-reference.md`.
- **F-3-7** — The fixed step aborts on the saturable-primary PSFB under the restart rule.
  - Call: `tests/test_newton_stall_and_mask_chatter.py` `_run("saturable", 50e-9)`.
  - Old (restart-less): 6.913 V. For reference: linear 6.857 V (pwl), exact engine 6.8577 V.
  - New: `SimulationAborted` at t = 61.55 us; the step cannot be taken even at dt/64. The message names the diode-mask chatter inside Newton. Same result with `form='conductance'` and with `device_path='legacy'`.
  - Hysteretic primary: branch form runs (6.712 V); conductance form + restart aborts at 24.95 us.

## Needed from bucket 0

- `python/tests/oracles/lcp_pulsim_bridge.py`: `Family.builder` says "on pulsim's conductance form"
  but calls `add_switch` / `add_diode` without `form=`, so since the flip it builds the branch form.
  `test_mode_pwl.py` works around this with a local monkeypatch fixture (`conductance`) for the
  guard tests only. The right fix is in the oracle: build with `form='conductance'`, or take a
  `form` argument.
