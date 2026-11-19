# Group A — impulsive events under the restart rule (F-5-1, F-3-3, F-3-7, F-3-1, F-5-10)

Branch `fix/core-release-a-impulsive` on the PR head 27c00520. Reproductions:
`python/tests/test_release_findings_a.py` (F-5-1, F-3-3 so far; F-5-10, F-3-7, F-3-1 to add).
Status marks: **[R]** = read in the code, **[M]** = measured on 27c00520, **[H]** = hypothesis, not yet
shown by a trace.

## Reproduced on 27c00520 (explicit options) [M]

| finding | call | value | truth |
|---|---|---|---|
| F-5-1 | IGBT level1, no FWD, L = 1 mH i0 = 100 A, `engine='pwl', dt=1e-9` | restart: i_L 100 -> -141.42135 A in one step, v_sw 8.571 V, no warning, 0 records. restart-less: 100 -> 0 -> -99.99999 A, v_sw -2e8 V at sample 1, the "conduction path opens" warning | no path: i_L -> 0 by an impulse (5 J), named |
| F-3-3 | linear flyback R_L = 1, `dt=1e-8` | restart: min i_p = peak = -20.9199 A, v_out 17.146 V, 0 records. restart-less: -15.483 A, v_out 16.801 V (+ voltage-sanity warning, 1.9e4 V) | `engine='dsed', mode_selection='auto'`: peak 16.38199 A, min i_p +1.6e-7 A |
| F-5-10 | Vin 48 V, L 1 mH, S (g_on 1e3, g_off 1e-9), 10 kHz PWM, `engine='dsed'` | i_L 2.39994 -> 4.8e-8 A at each turn-off, v_sw max 48 V, 0 records, no warning (same with form='branch', output_rtol=1e-5, mode_selection='auto') | the state is right (48 V · g_off); the ½·L·2.4² = 2.88 mJ left in the g_off transient is neither recorded nor booked |

Side observation (NOT group A, group C): the F-3-3 flyback on `engine='dsed'` with the PR line's
default `mode_selection` (legacy settle) gives v_out = 6e-13 V, peak i_p 2.06 A — the secondary diode
never conducts. With `mode_selection='auto'` it is right.

## Where things happen [R]

Fixed engine, `core/include/pulsim/solver/run_transient.hpp`:
- Restart trigger and window: `step_begin`/take phase, ~l.2267-2285: `restart_left = kRestartSteps (3)`
  on a mask change at t_prev, an unknown start mask (`!stamped_known`), a source / b_extra jump; a diode
  flip inside a step re-arms it (`kRestartSteps - 1`). `kGamma = 1 - 1/sqrt(2)`.
- `run_restart(t_from, h)` (lambda, ~l.2314): ONE Alexander SDIRK(2,2) step. Stage 1 = backward-Euler
  solve of γh (`run_event_iteration(t_from+γh, 2γh, true)`, the trapezoidal factor at 2γh), diode event
  iteration inside it; then `history.extrapolate_previous(x, c)` with c = (1-γ)/γ builds the stage-2
  pseudo-state x0* = x0 + c(x1 - x0); stage 2 = a BE-form solve of γh from x0*, with its OWN diode
  event iteration (mode decided on the stage-2 solve). Stage-2 fallback ("on_to_off"): only when a diode
  conducting in stage 1 turns off in stage 2 and no diode turns on fresh; then stage 2 is re-taken as
  BE of (1-γ)h from x1, kept iff BE keeps stage 1's bits. A diode OFF at x0 and in stage 1 that turns ON
  in stage 2 is explicitly EXCLUDED from that check (`!rs_bits[q] && rs_bits2[q] && !rs_bits0[q]` ->
  `on_to_off = false`): the SDIRK stage 2 is kept unconditionally.
- Split (`enable_substep_state_correction`): `split_step` (~l.2464): a diode crossing in the first
  (trapezoidal) solve is located by LINEAR interpolation of the crossing signal (`interp_commutation_time`),
  three branches by `snap`: at a (restart from a), at b (keep the pre-event solve, flip at b, arm window),
  inside (trap to t_d, `run_restart(t_d, b - t_d)`).
- Re-initialisation service on the fixed engine: `pwl::ReinitTracker` (`core/include/pulsim/pwl/reinit.hpp`).
  It does NOT project: the SDIRK x3 restart IS the projection; `ReinitTracker::record` runs AFTER the
  committed step (run_transient ~l.3145, in the record phase) and writes a `ReinitRecord` computed from
  x_prev / x for each STRUCTURAL dependent structure (`BranchStructureChecker`, ideal / branch-form
  elements only) the step entered by a gated change. A conductance-form switch (g_off > 0), a
  MOSFET's R_off, or a smooth Newton device that blocks (IGBT level 1) is never a dependent structure,
  so such an interruption gets no record whatever happens. `refuse_dependent` refuses restart-less runs
  that enter a structural dependent mode (`dependent_states`).
- `on_impulse`: `SimulationOptions::on_impulse_error` (solver/options.hpp); pwl/trbdf2 raise in
  `ReinitTracker::record`; on the exact engine Python raises/warns from `res.reinit_records`
  (python/pulsim/__init__.py ~l.4419). 'record' (default) only appends records. The voltage-sanity
  guard (`_auto_check_voltage_sanity`, solver/voltage_sanity.hpp) reads STORED samples only.

Exact engine, `core/include/pulsim/dsed/scheduler.hpp` + `builder_adapter.hpp`:
- `fire_event_` -> `decide_modes_` (settle or LCP) -> `project_(t, x)` (scheduler ~l.1521):
  1. `reinit_to_constraint` (builder_adapter): energy-weighted projection onto K·x = k for STRUCTURAL
     dependent modes, writes `ReinitRecord`s (lambda, dW = ½ rᵀλ).
  2. `project_slow(t, x, thr)` (builder_adapter -> `exact_project_slow`, exact_lti.hpp): every mode with
     Re λ < -thr goes to its quasi-static value. NO record, NO energy. This is where F-5-10's 2.88 mJ goes
     (the g_off mode, τ = L·g_off = 1 ps). thr = 100/dt_max, or 1/(kParasiticFraction·scale) under
     output_rtol.
- The mode is decided BEFORE the projection and re-settled on the projected state
  (`settle_and_project_`, `resettle_projected_`); trial-mode records are truncated (`truncate_reinit_log_`)
  and the state restored when a trial mode's projection left records — so adding records INSIDE
  `project_` would change control flow (and numbers); a record must be added after the decision.
- `PEDSimulatorAuto` (scheduler_auto.hpp) only calls `reinit_to_constraint`, no fast projection.

## Hypotheses per finding

- **F-5-1 / F-3-3 [H, numbers consistent]**: the stage-2 pseudo-state. Stage 1 (BE of γh) relaxes the
  stiff interruption (i ≈ 0); x0* = x0 + c(x1 - x0) ≈ (1 - c)·x0 = -√2·x0 (1 - 2.41421 = -1.41421).
  Stage 2's mode / device region is decided on a solve from that pseudo-state: the reversed current
  finds a path the physical state never had — the IGBT's forward channel (F-5-1: -141.42135 A =
  -√2·100 + γh·v/L), the MOSFET body diode (F-3-3: -20.92 ≈ -√2·14.8). In a linear fixed mode
  stage 2 would cancel it (L-stable, x2 ≈ 0); across a region change it does not. So the mode at the
  event is not decided on an inconsistent PHYSICAL state but on the SDIRK's non-physical stage-2
  history. F-3-3 is a diode bit (Q1_body) turning on fresh in stage 2 — the case the stage-2 fallback
  explicitly excludes. F-5-1 has no bit at all (smooth IGBT). Trace pending (env-guarded fprintf in
  `run_restart`, PULSIM_TRACE_RESTART, not to be committed).
- **F-3-7 [H]**: the same pseudo-state overshoot feeding the Newton loop's per-iteration diode re-decision
  (`make_combined_diode_mosfet_refresh`) with a saturable primary: no self-consistent mask for the
  stage-2 solve -> chatter. To trace.
- **F-3-1 [H]**: the post-event error decays by about -0.11 per sample = R(z) of the TRAPEZOID at
  z ≈ -2.5 (SDIRK(2,2) gives -0.012 there), so samples after the turn-on are trapezoidal, i.e. the
  restart window does not cover the steps after this event, and the 1-ulp sensitivity is a branch flip
  between the split's `snap` cases (event inside / at b / at a) — a discontinuous decision, not the
  projection. To trace.
- **F-5-10 [R]**: `project_slow` drops the fast mode with its energy, by design, without a record. Fix in
  progress (uncommitted): after `decide_modes_` record ½·ΔxᵀEΔx of the event's state jump (E = C, L with
  mutuals; for a charge/flux-conserving jump this equals the projection formula's ½rᵀλ, sources' work
  included, >= 0) minus the dependent-state records already written, as a `ReinitRecord` kind
  `fast_mode` (booked by `losses.impulse_losses`, raised by on_impulse='error').

## Default path (restart-less fixed step) [M]
F-3-3 restart-less is also wrong (-15.48 A against ~0): the trapezoid reflects the stiff interruption
(R(∞) = -1) and the body diode then carries the reflection. It warns (voltage sanity) on this circuit.
Not covered by a restart-rule fix; recorded as a default-path finding of group A.

## Next steps
1. Trace F-5-1 / F-3-3 with the env-guarded print (built in .scratch/build-ext) to prove the stage-2 mechanism.
2. Fix at `run_restart`: decide stage 2's mode change (any bit, incl. a fresh turn-on) by the BE re-take
   from x1; for smooth devices find an equivalent test or report the stage-1 impulse to the guard.
3. Build F-5-10 record, check records on the dsed byte-compare set, then F-3-7 / F-3-1 traces.
