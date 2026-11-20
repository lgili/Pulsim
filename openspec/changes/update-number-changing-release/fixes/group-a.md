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

## Traces (env-guarded fprintf in `run_restart`, x0 / stage 1 / stage 2 / end; not committed) [M]

- F-5-1, step t = 0, h = 1 ns, x = (v_g, v_sw, ., i_L):
  `x0: 15 0 0 100` -> `s1: v_sw = -3.41421e8, i_L = 0` -> `s2: v_sw = 8.57106, i_L = -141.421`.
  -141.421 = 100 + 2.41421·(0 - 100) = x0* exactly: stage 2 found the IGBT's forward channel at
  the pseudo-state and stayed there. PROVEN. No diode bits (the IGBT level 1 is a smooth Newton device).
- F-3-3, first turn-off t = 5 µs, dt = 10 ns, bits (Q1_body, D1), x = (v_in, v_sw, ., ., ., i_p, -i_p, i_s):
  `x0: bits 00, v_sw 0.0206, i_p -2.061` -> `s1: bits 01, v_sw 3567 V, i_p -7.1e-6, i_s 3.964` ->
  `s2: bits 11 (body diode ON), v_sw -0.703, i_p +2.887` (≈ -√2·i_p(x0) = +2.915). The body diode is
  OFF at x0 and in stage 1 and turns on in stage 2 only — the case the stage-2 check excluded.
  With the exclusion removed: `RS retry keep_be=1`, end `bits 01, v_sw 48.01, i_p -9.6e-8`. PROVEN.
- F-3-7 (saturable PSFB, dt = 50 ns): the failing restart step at t = 21.5 µs prints `x0` for every
  halving (50 ns ... 0.78 ns) and never `s1`: Newton fails in STAGE 1 (||dx|| 1.1e4 V at vout).
  Not the stage-2 pseudo-state; the stage-2 fix leaves it unchanged. PROVEN negative.
- F-3-1 (bridge1_vf07, k = 0 vs k = 2 ulp): both take the split's "inside" branch;
  k = 0: `t_d = 33.70395 ms, x0 bits 0010` (one diode), stage 1 turns the second on (iters 2),
  end i = 14.2994 A; k = 2: `t_d = 33.7040466 ms, x0 bits 0110` (both within snap), end 14.1918 A.
  The two series diodes of the floating bridge can only conduct together; the split locates the
  event at the FIRST single diode's interpolated crossing, a tie decided by leakage/rounding. Not an
  impulse; the stage-2 fix leaves it unchanged. PROVEN.
- F-5-10 (exact engine): the event projection is OFF by default (`enable_event_projection_ = false`),
  so the 1 ps g_off mode is integrated exactly; `right_limit_states` at each turn-off holds
  v_sw = 2.39994e9 V = i_L/g_off and i_L = 2.39994 A (continuous). The energy is dissipated in g_off
  in that picosecond: nothing is dropped. The defect is the voltage-sanity guard
  (`_auto_check_voltage_sanity`, python/pulsim/__init__.py) reading only the stored samples, not the
  right limits. (Also seen, group B: `device_loss_summary` integrates that right-limit spike as a
  1 µs trapezoid: S's E_total = 5760 J, P_avg 1.9e7 W.) The reading "drops 2.4 A" in r2/bucket-5.md
  is a misreading of the sampled trace. PROVEN.

## Hypotheses per finding (written before the traces; see above for what held)

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

## Fixes landed (130fcb70) [M]

- **F-3-3** — `run_restart`: any stage-2 diode-mode change goes through the existing backward-Euler
  re-take from x1 (kept iff it keeps stage 1's bits; else the SDIRK stage 2 is restored bit for bit).
  The removed exclusion (fresh turn-on in stage 2) was the F-3-3 case. Side effect (reviewer): the old
  loop's `break` also cleared a real on->off flip seen earlier in the same loop; now it counts.
  Truth: exact engine (`mode_selection='auto'`) peak i_p 16.381990162 A, min +1.6e-7 A.
  Restart pwl peak: 16.3819859 / 16.3819892 / 16.3819901 at dt 10 / 5 / 2.5 ns (errors 4.2e-6,
  9.5e-7, 1.0e-7; order 2.2 then 3.2); min i_p +1.57e-7 A (was -20.92 A); v_out 16.3651 (was 17.146).
- **F-5-10** — `_auto_check_voltage_sanity` also scans the right-limit samples (`right_limit_index` /
  `right_limit_states`, only the exact engine has them). Truth: v_sw(0+) = i_L(0-)/g_off (closed
  form; matches to 1e-9). The warning now fires on `engine='dsed'`.
- Parked with traced causes (strict xfail in `test_release_findings_a.py`): F-5-1 (smooth Newton
  device: no bit for the stage-2 check — needs a region test for Newton devices or the BE re-take on
  a stage-1 impulse, plus the stage-1 voltage for the guard), F-3-7 (stage-1 Newton diode chatter at
  a gate edge — needs the in-Newton diode re-decision fixed for the BE stage), F-3-1 (the split
  locates a series diode pair's commutation at one diode's crossing — needs the split to locate the
  decided post-event mode's commutation).

## Byte compares vs the PR head (same machine) [M]
DEFAULT options: identical everywhere — goldens (all non-restart cases), stateful (all but the two
`*_restart` cases), engines, steps (all but restart/split), pwl_identity default config,
trbdf2_identity 371/371. Moved, all with the restart rule ON (the option, not the default):
golden `buck_pwl_restart` (stage2_retries 21 -> 29, stage2_be_kept 0 -> 6), stateful
`compose_all_restart`, `everything_pwl_restart`, engines `buck_pwl_restart`, steps
`buck_pwl_restart`, `vsi_pwl_split`, pwl_identity 18 of 379 restart/split entries (mostly the
stage2_retries counter; boost_pfc, active_clamp_forward, interleaved_buck_3ph commutation counts).
Checked against truth (the reviewer's risk: turn-ons at a zero start / buck CCM from rest): the
golden buck (24 V, 100 kHz, from rest, 2 ms) on pwl+restart vs the exact engine at output_rtol 1e-9,
p99 |dv_out| / p99 |di_L| over all samples:
dt 100 ns: 1.96e-5 / 7.2e-6 -> 9.9e-6 / 3.5e-6; 50 ns: 3.8e-6 / 9.0e-7 -> 2.4e-6 / 1.2e-6;
25 ns: 1.42e-6 / 5.6e-7 -> 1.34e-6 / 3.9e-7 (before -> after). Equal or better except di at 50 ns
(1.35x). Cost: the change runs only inside `run_restart` (option ON); it removes one bit snapshot
per restart and adds a BE re-take on the newly included case (counted in stage2_retries). The
restart-less loop is untouched, so the four-case instruction count was not re-measured.

## Default-path finding of group A (not fixed)
F-3-3 restart-less: -15.48 A against ~0 (trapezoid R(∞) = -1 reflects the cut current, the body
diode carries the reflection; v_out 16.80 vs 16.365). It warns (voltage sanity). Not covered by the
restart-rule fix; the release's default (restart ON) now gets it right.
