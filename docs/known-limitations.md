# Known limitations of Pulsim 2.1

This page lists the defects and gaps that Pulsim 2.1 ships with (the
rows written for 2.0 were re-checked on the 2.1 code base; the fixed
ones are gone). Each row gives the
measured effect and one workaround. Where a test pins the defect, the
row names it. These tests are strict xfails, so the suite fails as soon
as the defect is fixed and the row has to be updated.

The usage traps are not repeated here. They are behaviour by design
that returns a plausible wrong number: a diode `V_th` that is not a
forward drop, `.mean()` on the variable-step grid, `SwitchStateMask(n)`
taking a size, MOSFET bit numbering, undriven switches that close, a
broad `except` around `simulate`. They are in the
[AI usage guide, section 5](ai-guide.md#5-traps) (`python -m
pulsim.guide`). How to check a result is in its
[section 6](ai-guide.md#6-how-to-verify-a-result). Newton failures and
their messages are covered in [Gotchas](gotchas.md).

## Opt-in features of 2.1 with open findings

These options default off; with default options none of this is reached.
The numbers are in the `Known limitations` part of the 2.1.0 entry of the
[changelog](https://github.com/lgili/Pulsim/blob/main/CHANGELOG.md).

* **Restart rule** (`discontinuity_restart=True`): the ideal-switch
  flyback on the fixed engine gets worse under step refinement; an edge
  off the `dt` grid stays first order without event splitting.
  **2.1.0 (the released version) can also return a wrong steady state
  with the rule on:** a diode that turns off inside the restart step's
  second stage is not located, and the stage starts from an extrapolated
  capacitor voltage. Measured on a boost with a 1 nH residual inductance:
  51.34 V with the rule on, against 42.11 V (an independent ODE solve;
  `trbdf2` reads 42.111 V, the fixed engine with the rule off 41.86 V).
  This line fixes it: the turn-off is located and the restart step is cut
  there (counted in `result.counters["n_restart_cuts"]`; the same boost
  reads 42.75 V; re-measured on this line with the 0.5001 h gate sample:
  42.750 V 0.1 us after the edge, 42.114 V at 0.4 us against 41.48 V). On 2.1.0 the rule is off by default; with it on, compare
  against `trbdf2` before trusting it on a converter whose diode commutates
  right after a gate edge.
  Without event splitting, a diode that turns off inside a restart step's
  first stage is not located either: the step keeps a first-order kick
  (`half_bridge_transformer_dc`, dt = 100 ns: up to 3.3e-6 V on V(out) per
  such turn-off; with `enable_substep_state_correction=True` it is located).
  Without event splitting, a sample of an algebraic signal that is
  discontinuous at a rounded edge is the pre-edge value: `scr_halfwave_r`
  on `pwl` with the rule reads 0.0271 against its analytic truth (1.47e-16
  with the rule off).
* **Restart rule, switching energy**: the PSFB reference's S3 switching
  energy reads about −10 % of the exact truth (the C_oss collapse row;
  strict xfail in `python/tests/test_soft_switching_kpi.py`,
  [Soft switching](SOFT_SWITCHING.md)).
* **Restart rule, recorders** (finding F-Z-12): the in-loop feed and the
  stored-array pass agree bit for bit on every sample-type output (kept
  rows, `min` / `max` and their times) and on the event rows. The
  integral-type readings (`integral`, `avg`, `rms`) are two different
  quadratures: the in-loop record is the method's own SDIRK quadrature,
  which keeps `∫i_C dt = C·Δv` and the energy ledgers closed to round-off;
  the stored-array pass is the trapezoid of the state values. They are
  equal off restart steps and differ on a restart step by the leading-order
  estimate (smooth piece) or the bound (mixed-mode step) derived in the
  recorders section of [API reference](api-reference.md). A window that
  ends inside a mixed-mode restart step is first order in that end
  (the charge `∫i_C` against `C·Δv` of the read states: 2.6e-5 relative at
  1 µs on a PWM buck, half that at 0.5 µs; 1.7e-3 before the fix).
  On the measured DCM buck (restart rule, no split, dt = 1 µs) the stored
  trapezoid was the closer of the two to the truth (the exact engine at
  dt = 25 ns): the window integral of i(L) is +2.9e-5 relative from the
  stored arrays, −1.97e-4 in the loop (on a smooth gate-switched RL cell
  the two agree to 1e-9). The gap is a cancellation, not a better
  quadrature: it comes from the steps whose mode changes without a
  located event (a diode turn-off inside the step), where the two
  quadratures are equally far from the truth with opposite signs (sum
  −1.14e-4 in the loop, +1.10e-4 stored, over 160 such steps); the stored
  sign happens to cancel the other steps' −7.8e-5. The feeds' agreement is
  asserted in `python/tests/test_release_findings_b2.py`.
* **Event splitting** (with the restart rule): a retried step is re-taken
  without its interior events; an even number of edges of an opaque
  `switch_fn` inside one step is not seen. A diode flip in the restart
  window after an event is located (turn-on and turn-off), except: a flip
  in the first stage of a sub-step that starts at an event (taken at the
  event), a second flip in the rest of an already cut restart step, and a
  flip inside the run's first step (taken at the step end, up to 1·dt late;
  `python/tests/test_split_locator_reconduction.py`). The rule alone dates
  a diode flip inside a step at the step end (no locator): on
  `forward_converter` the D_fw hand-over after the switch turn-on costs the
  exact grade 1.5e-4 / 3.3e-5 / 3.5e-5 at dt = 100 / 50 / 25 ns (first
  order, phase dependent); with the split 1.0e-6 / 3.2e-7 / 2.7e-7 (the
  waveform's own level). (F-R-4, fixed: the D_rect re-conduction 6.1 ns
  after the turn-off was held off for a step because the edge's projection
  fitted one of the two winding modes; it now conducts from the edge.)
  Locating an event costs trial solves, each a factorisation: 4–7 trials per located event.
  A freewheeling diode that conducts for less than the rest of the step
  after a switch turn-off is not located: the step records a `cut_set`
  re-initialisation and drops the inductor's whole current as an impulse
  instead of delivering it to the output (`ll15_boost_closed_loop` at
  dt = 0.5 us: 60 mA, conducting 84 ns in `trbdf2`; `ll14_buck_closed_loop`:
  35 mA, 164 ns; a dt/100 run conducts it). The edge's right limit itself
  is correct (D on, the inductor current kept).
* **Exact engine output rule** (`engine='exact', output_rtol=...`): a
  thermal summary read from stored samples depends on the output grid.
  The first line after an event (within 2 ps of it) can exceed the rule:
  59× the bound on a DCM switch node, 1491 such lines on `cl_buck_pi`.
  Closed-loop units sit at the promise (ratio ~1.0, `cl_buck_pi` 1.02),
  not clearly under it.
* **Sensitivities** are refused by name at a recorded cut-set jump (a
  DCM buck on `pwl` with splitting that returned a number before).
* **Exact engine**: twelve paralleled bridges are refused by name; the
  phase-shifted full bridge does not run with device capacitances nor in
  ideal form.

## On default options: nonlinear output capacitance on the variable-step engine

A MOSFET with a nonlinear output capacitance on `engine='trbdf2'` (and on
`auto` without `dt`, which lands there): in 2.1.0 the conduction energy after
a gate edge read +3.91 % against its closed form (a Coss buck). Since
[Unreleased] it reads +0.057 % at the defaults (+0.14 % with
`integrator='esdirk43'`, +0.007 % at rtol 1e-8): the step after the edge
starts from the consistent post-edge point. For a tighter number, lower
`rtol` or pass `dt` (fixed step).

## Engine capability matrix

Which engine simulates which device kind. This table is the kernel's
(`pulsim.engine_capabilities()`); `test_engine_conformance.py` checks
that it equals this page and that every "yes" agrees with `engine='pwl'`
within 1 % of the observable's span and every "no" raises
`pulsim.UnsupportedError` naming the engine and the branch. `ac_sweep` is
the kernel AC sweep (`run_mna_sweep` with `freqs=`): a "yes" Newton kind is
linearised at the DC operating point from its device law
(`test_ac_linearize.py` checks each against its closed form and a
small-signal transient). `NonlinearDiode` is retired: `add_nonlinear_diode` refuses
unless called with `allow_non_passive=True`.

<!-- engine-capabilities:begin -->
| Kind | Class | pwl | trbdf2 | dsed | bdf1 | ac_sweep |
|---|---|---|---|---|---|---|
| Resistor | Linear | yes | yes | yes | yes | yes |
| VoltageSource | Source | yes | yes | yes | yes | yes |
| Switch | Switch | yes | yes | yes | yes | yes |
| Capacitor | Linear | yes | yes | yes | yes | yes |
| Inductor | Linear | yes | yes | yes | yes | yes |
| Diode | PwlDiode | yes | yes | yes | no | yes |
| NonlinearDiode | NewtonStatic | yes | yes | no | no | yes |
| CurrentSource | Source | yes | yes | yes | yes | yes |
| PWMVoltageSource | Source | yes | yes | yes | yes | yes |
| SineVoltageSource | Source | yes | yes | yes | yes | yes |
| PulseVoltageSource | Source | yes | yes | yes | yes | yes |
| MosfetLevel1 | NewtonStatic | yes | yes | no | no | yes |
| IgbtLevel1 | NewtonStatic | yes | yes | no | no | yes |
| VCVS | Linear | yes | yes | yes | yes | yes |
| SaturableInductor | NewtonStateful | yes | yes | no | no | yes |
| NonlinearCapacitor | NewtonStateful | yes | yes | no | no | yes |
| ShockleyDiode | NewtonStatic | yes | yes | no | no | yes |
| LauritzenDiode | NewtonStateful | yes | yes | no | no | yes |
| PmsmMna | NewtonStateful | yes | yes | no | no | yes |
| IdealTransformer | Linear | yes | yes | yes | yes | yes |
| VCCS | Linear | yes | yes | yes | yes | yes |
| CCCS | Linear | yes | yes | yes | yes | yes |
| CCVS | Linear | yes | yes | yes | yes | yes |
| Gyrator | Linear | yes | yes | yes | yes | yes |
| SineCurrentSource | Source | yes | yes | yes | yes | yes |
| PulseCurrentSource | Source | yes | yes | yes | yes | yes |
| PwlCurrentSource | Source | yes | yes | yes | yes | yes |
| InductionMachineMna | NewtonStateful | yes | yes | no | no | no |
| ExternalDevice | NewtonStateful | yes | yes | no | no | yes |
| ExcessLoss | NewtonStatic | yes | yes | no | no | yes |
| TransmissionLine | LinearDelay | yes | yes | no | no | yes |
<!-- engine-capabilities:end -->

## Option capability matrix

Which engine honours which `simulate()` keyword. This table is the
kernel's (`sim::kRunOptionTable`, `pulsim.option_capabilities()`);
`test_simulation_object.py` checks that it equals this page and that
every "refused" cell raises `pulsim.UnsupportedError` naming the engine
and the keyword (`feature`) when the keyword is set to a value that asks
for something (`None`, `False`, `store_every=1`, `max_dt_halvings=0`,
`mode_selection='legacy'`, `control_tick='label'` ask for nothing). No
engine accepts a keyword and ignores it. The defaults are
`sim::Defaults` (`core/include/pulsim/sim/defaults.hpp`).

<!-- option-capabilities:begin -->
| Keyword | pwl | trbdf2 | exact (dsed) | default |
|---|---|---|---|---|
| `engine` | yes | yes | yes | 'auto' |
| `rtol` | refused | yes | yes | 1e-5 (trbdf2), 1e-6 (exact) |
| `atol` | refused | yes | yes | 1e-8 (trbdf2), 1e-9 (exact) |
| `dt_init` | refused | yes | yes | kernel's choice (trbdf2), 1e-9 (exact) |
| `integrator` | refused | yes | yes | 'auto' (exact), TR-BDF2 (trbdf2) |
| `stiffness_threshold` | refused | refused | yes | 10.0 |
| `h_bdf2` | refused | refused | yes | 1e-6 |
| `t_start` | yes | yes | yes | 0.0 |
| `switch_fn` | yes | yes | yes | None |
| `b_extra_fn` | yes | yes | yes | None |
| `step_observer` | yes | yes | yes | None |
| `start_from_dc_op` | yes | refused | refused | False |
| `enable_nonlinear_refresh` | yes | yes | refused | None (stamp the Newton devices when there are any) |
| `max_newton_iterations` | yes | yes | refused | 50 |
| `max_event_iterations` | yes | yes | refused | 16 |
| `strict_event_iterations` | yes | refused | refused | False |
| `max_dt_halvings` | yes | refused | refused | 6 |
| `voltage_sanity_factor` | yes | yes | yes | 100 |
| `store_every` | yes | refused | refused | 1 |
| `auto_regularize` | yes | yes | yes | True |
| `tol_newton_dx` | yes | yes | refused | 1e-9 |
| `tol_newton_res` | yes | yes | refused | 1e-9 |
| `enable_newton_line_search` | yes | yes | refused | False |
| `enable_newton_lm` | yes | yes | refused | False |
| `newton_roundoff_budget` | yes | yes | refused | 130 |
| `enable_substep_state_correction` | yes | refused | refused | False |
| `discontinuity_restart` | yes | refused | refused | False |
| `split_snap` | yes | refused | refused | 1e-6 (2.1.0: 0.01) |
| `resolution_diagnostics` | yes | refused | refused | False |
| `on_impulse` | yes | yes | yes | 'record' |
| `inductor_freeze_di_max` | yes | refused | refused | None (off) |
| `inductor_abs_clamp` | yes | refused | refused | None (off) |
| `progress` | yes | refused | yes | False |
| `initial_state` | yes | yes | yes | None (the circuit's ICs) |
| `should_continue` | yes | yes | refused | None |
| `closed_loops` | yes | yes | yes | None |
| `live_stream` | yes | refused | refused | None |
| `mmc_arms` | yes | refused | refused | None |
| `on_cache` | yes | refused | refused | None |
| `controller_period` | yes | yes | yes | None |
| `resume_from` | yes | yes | yes | None |
| `solver` | yes | yes | yes | None |
| `sensitivities` | yes | yes | yes | None |
| `mode_selection` | yes | yes | yes | None ('legacy') |
| `allow_implicit_segments` | refused | refused | yes | None |
| `output_rtol` | refused | refused | yes | None (today's samples) |
| `undriven` | yes | yes | yes | None (each engine's own) |
| `recorders` | yes | yes | yes | None |
| `control_tick` | yes | yes | yes | 'label' |
| `device_path` | yes | yes | refused | 'legacy' |
| `t_end` | yes | yes | yes | required |
| `dt` | yes | yes | yes | None (pwl: required; trbdf2: kernel ceiling; exact: 1e-5) |
<!-- option-capabilities:end -->

The charge laws `ChargeLaw.table` / `.vdmos_cgd` are the
NonlinearCapacitor row. `add_power_mosfet` is a composite of the
MosfetLevel1, Capacitor, NonlinearCapacitor, Resistor, Inductor and
Shockley / Lauritzen diode rows: it runs on `pwl` and `trbdf2` and is
refused on `dsed`, `bdf1` and the kernel AC sweep, naming the channel.
With `fq_device=True` (opt-in) it is ONE f/q device (channel, C_gd,
C_ds, body diode: the ExternalDevice row) plus the Resistor, Inductor
and Capacitor rows of R_d, R_g, L_s, C_gs: it runs on `pwl`, `trbdf2`,
the DC operating point and the kernel AC sweep, and is refused on `dsed`
and `bdf1`, naming the device.

The branch form of a switch or diode (`add_ideal_switch`,
`add_diode(form='branch')`, YAML `form: branch`; opt-in) has its own
column; "no" raises `UnsupportedError(feature='branch_form')`. A mode
whose ideal elements (`R_on = 0` or `g_off = 0`) make a state dependent
(an inductor cut-set, a capacitor loop) runs on `pwl` with
`discontinuity_restart=True` and on `trbdf2` (always: three SDIRK(2,2)
steps after the entry), and an impulsive entry is recorded in
`res.reinit_records` (`on_impulse='error'` refuses it with
`feature='impulse'`; on `dsed` a jump is recorded only above 1e-9 of the
circuit's current or voltage scale, so a tiny cut in a circuit with a
very large current scale is projected without a record); on `pwl` without the rule, and on `trbdf2` with a
Coss, Lauritzen, IGBT-tail, PMSM or saturable-inductor history, it is
refused with `feature='dependent_states'`; a node bounded only by open ideal elements with
`g_off = 0` is refused with `feature='floating_node'`, a loop of closed
ideal elements and voltage sources with `feature='ideal_voltage_loop'`.

<!-- branch-form-capabilities:begin -->
| Form | pwl | trbdf2 | dsed | bdf1 | ac_sweep |
|---|---|---|---|---|---|
| Switch | yes | yes | yes | no | yes |
| Diode | yes | yes | yes | no | yes |
<!-- branch-form-capabilities:end -->

The gated PWL devices are a form of the Diode row with their own column
(`add_igbt(reverse_blocking=True)` and `add_gto`: GateHeld;
`add_thyristor` and `add_triac`: Latching); "no" raises
`UnsupportedError(feature='gated_diode')`. Not modelled: turn-off time
`t_q` and commutation failure, dv/dt and break-over firing, a latching
current distinct from `I_H`, gate power, reverse recovery and switching
energies. `I_H > 0` chops an inductive current into `R_off` (a voltage
`I_H·R_off` without a snubber).

<!-- gated-capabilities:begin -->
| Form | pwl | trbdf2 | dsed | bdf1 | ac_sweep |
|---|---|---|---|---|---|
| GateHeld | yes | yes | no | no | yes |
| Latching | yes | yes | no | no | yes |
<!-- gated-capabilities:end -->

Circuit-level limits, also raised as `UnsupportedError` (with `feature`
set): coupled inductors (`add_transformer`, `add_inductor_coupling`) on
`dsed` and `bdf1`; PWL diodes with `integrator='bdf2'` on `dsed`;
capacitor loops (parallel capacitors) on `dsed`; C blocks, an every-step
`step_observer` (no `controller_period`), `device_path='interface'`,
`start_from_dc_op`, `store_every`,
`live_stream`, `max_dt_halvings`, `mmc_arms`, `on_cache`, the inductor
guards, `enable_substep_state_correction` and `discontinuity_restart` on
`dsed`; the fixed-step keywords (`discontinuity_restart` among them) on
`trbdf2`; a control-graph `Rate` with a `compute_delay` time
(`compute_delay`) on `trbdf2`; a `b_extra_fn.next_jump_after` declaration is read by `pwl`
only (a no-op on `trbdf2` / `dsed`, which control the step on their own
error estimate); `enable_nonlinear_refresh=False` with a smooth /
Shockley diode, MOSFET or IGBT on every transient engine;
`sensitivities=` on every engine but `dsed`, and on `dsed` a parameter
that is not an R, L, C, DC-source value or the duty of a single-switch
trailing-edge PWM (`sensitivity_parameter`; a dead-time pair's duty is
refused), a time-varying source, a Newton device, an opaque gate or
`b_extra_fn`, a mode without an exact propagator, an RK45 step, a
grazing diode event (`grazing_event`), and a KPI window edge on a moving
event where the signal jumps.

## Engines

| Limitation | Measured | Workaround |
|---|---|---|
| **`trbdf2` on the active-clamp forward converter (`ll13_active_clamp_forward`): every MOSFET turn-off edge costs hundreds of steps below 1e-11 s, and some run lengths abort there.** The post-edge mode has a parasitic mode of tau ~ 2e-13 s; the committed edge probe is a trapezoidal step of dtp >> tau, which leaves that mode ringing (amplification ~ -0.99), and the step controller then resolves it. | Up to 389 steps below 1e-11 s per turn-off at rtol 1e-8 (1807 at rtol 1e-10); aborts "Newton failed ... at h_min" at rtol 1e-6 / 1.6 us, rtol 1e-8 / 3 us (at the 1.5 us turn-off) and rtol 1e-10 / 100 us. The default (rtol 1e-5) run and the graded rtol 1e-8 runs of other lengths complete. A backward-Euler edge probe was measured and not adopted (one more abort). | Use another run length or tolerance, or `engine='pwl'` with an explicit `dt`. |
| **`dsed` can settle a zero-band diode into the wrong mode; two guards stop it being returned.** The complementarity guard (default) judges each settled mode and repairs the wrong ones with the LCP; the stored-sample guard raises `DiodeModeInconsistentError` for what the kernel guard misses. | Boost 12 V, 100 µH, 47 µF, 100 kHz, D = 0.5, V_th = 0 at 20 / 50 / 500 Ω: repaired (7 / 16 / 23 events), ends on trbdf2 rtol 1e-10. LC tank into a series diode with V_th = 0 (a double crossing inside one step): the kernel guard returns v = -5.18 V at 10 µs (trbdf2 9.50 V); the stored-sample guard raises. | `V_th = 1e-3` for the double crossing, or `pwl` / `trbdf2`. |
| **The exact engine with the default mode selection loses a flyback's magnetising energy at every turn-off.** At the gate-off the secondary diode's current is a state at 0, so the settle reads its voltage as 0 and keeps it OFF; the forward bias exists only inside the 1e-14 s `R_off` transient in which the coupling would carry the current over, and the guard's judge (horizons past that transient) confirms the mode. A default-path defect; `mode_selection='auto'` (the next defaults) does not have it. Pinned as a strict xfail: `test_release_findings_c.py::test_the_flyback_secondary_conducts_on_the_exact_engine[legacy]`; notes: `openspec/changes/update-number-changing-release/fixes/group-c.md` (F-3-3 side observation). | `_flyback(False, 1.0)` of `test_saturable_transformer.py` (100 kHz, D = 0.5, `engine='dsed'`, 1 ms): v_out over 0.8-1 ms is 0.000 V (also at `rtol=1e-9`); the primary current drops from 2.06 A to 1e-7 A at each gate-off. `mode_selection='auto'` gives 16.357 V; truth 16.365 V (the fixed engine with the restart rule at dt 10 / 5 / 2.5 ns, and the exact engine under `'auto'`). | `mode_selection='auto'` or `'lcp'` on the exact engine, or `engine='pwl'` with `dt` and `discontinuity_restart=True`. |
| **Fixed-step controller ticks land on the `dt` grid.** Source j ticks at `t_start + k*T_j` on both engines; `trbdf2` lands each instant, `pwl` delivers it at the first grid point at or after it. | `T = 2.5*dt` over 100 µs at `dt` = 1 µs: 40 ticks, each late by less than `dt` (`result.control_tick_jitter_max`), with a warning; `T < dt` raises. | `dt = T/n`, or omit `dt` (`trbdf2`). |
| **By default the fixed engine is first order after a discontinuity.** The start (ICs included), every gate edge and every source jump leave a stale trapezoidal history (`v_prev`, `i_prev`). Guide trap 9; pinned by `test_first_step_order.py` and `test_fixed_step_discontinuity_order.py`. | RL decay, R = 1 Ω, L = 1 mH, `i0` = 5 A: relative error at 5 τ 1.997e-3, 9.99e-4, 5.00e-4 at `dt` = 4, 2, 1 µs (order 1.00); with `discontinuity_restart=True` (three SDIRK(2,2) steps per discontinuity) 6.66e-6, order 2.00. Buck CCM `i_L` at T/50: 1.18e-2 A, 1.17e-7 A with the rule. An edge off the `dt` grid stays first order unless `enable_substep_state_correction=True` (needs the rule) splits the step at it: buck D = 0.4567 `v_C` at T/50 8.08e-2 V → 1.07e-6 V, order 2. An undeclared `b_extra_fn` jump stays first order either way. Sample 0 of `v(a)` reads 0 V either way; the consistent value is -5 V. | `discontinuity_restart=True` (gate sampled at the step start); keep edges on the grid. |
| **Diode-mask chatter on the fixed engine.** With PWL diodes and a Newton device in the same circuit, the diode mask is re-decided at every Newton iteration and no mask is self-consistent at a small step. Pinned by `test_newton_stall_and_mask_chatter.py`. | PSFB with a saturable primary and a floating secondary. At `dt` = 20, 10, 5 and 2 ns it aborts near t = 1.5 µs, even split into 64 sub-steps. At 50 ns it runs. V_out = `v(vout) - v(rect_neg)`, averaged over 1.5-2 ms, is 6.771 V. With a linear primary it is 6.765 V at 50 ns and 6.938 V at 10 ns (6.937 V at 5 ns). So 50 ns reads 2.4 % low. | Run at 50 ns and accept about 2.5 % low; check it against the linear-primary circuit at 10 ns. `add_shockley_diode` runs at 10 ns but changes the model: each junction drops about 0.7 V and a PWL diode drops none (trap 1). V_out falls to 5.785 V (-16.6 %). Use it only if you want that drop. |
| **TR-BDF2 femtosecond landing step after a diode turn-off.** Pinned by `test_trbdf2_stateful_with_diodes.py::test_no_kilovolt_artefact_on_the_internal_node`. | L-D-RC rectifier, 20 V source, constant 111 µH. On TR-BDF2 the L-D node peaks at 1539 V on a 3e-14 s step; on `pwl` it peaks at 23.1 V. The output peak is the same on both: 19.640 V and 19.641 V. | Do not take `max()` of internal nodes from a TR-BDF2 run. Check peaks on `engine='pwl'` with a `dt`. |
| **A floating diode bridge has an undetermined common mode on TR-BDF2.** A V_th = 0 bridge whose DC bus is tied to ground only through 1 MΩ has no defined common mode once every diode is off. | 325 V, 1 mH line, 470 µF, 100 Ω: v(dp) − v(dn) agrees with `pwl` (`dt = 1e-6`) to 0.017 V at rtol 1e-5, 0.007 V at 1e-6 and 0.0015 V at 1e-7, but v(dp) alone differs by up to 41 V, 265 V and 32 V. (It ground to `max_steps` at rtol 1e-6 and 1e-7 before the TR-BDF2 roundoff-floor fix.) | Measure the bus as v(dp) − v(dn), or ground one DC rail. |
| **TR-BDF2 can miss PWL-diode commutations.** Guide trap 13. | 230 V rms bridge, 0.1 Ω, 470 µF, 1 kΩ. With `V_th = 0` it runs half-wave: 318.79 V and 12.91 V pp, against 322.06 V and 6.38 V pp with `dt=1e-6`. With `V_th = 0.7` and 1 mH of line, D2 and D4 are held OFF while forward-biased for one sample at 75.92 ms; V_dc is 321.80 V against 321.82 V. Snubbed boost (24 V, 220 µH, 50 kHz, D = 0.5, 1 nF across D1, 100 µF, 30 Ω), V_out over 2-3 ms: with `V_th = 0`, 24.64 V on `trbdf2` and `auto`, against 54.89 V with `dt = 1e-8`. With `V_th = 0.7` on `auto`, 56.18 V against 54.89 V (+2.3 %), and D1 is held ON while reverse-biased on 106 samples. | Pass `dt`. Rerun with `dt` whenever `simulate` warns that a diode "rode its conduction boundary", or `res.i` warns that one was held "in a state its commutation rule forbids". |
| **TR-BDF2 does not converge on a converter with no capacitance at its switch / rectifier nodes and a leakage inductance in series with a winding.** Each OFF device in series with an inductance leaves a femtosecond mode, and the node between the leakage inductance and the winding carries only inductors, so its voltage is solved to round-off at the smallest steps. It shows as many forced-accept warnings and a result that does not settle as `rtol` tightens. Every forced accept is blamed on a current of the primary's inductor cut-set (`L_leak`, `T1.p`, `T1.s`), at a gate edge (h = 2e-14 s) or inside the 100 ns dead time that follows it (h = 2e-10 s, the body diodes' zero-current commutation), with the estimate 20 to 3000 times its tolerance: the femtosecond mode makes that current change within any step, so the error estimate is right and cannot be fixed there. Record: DECISIONS #21 of `update-trbdf2-post-event-restart`, DECISIONS 11 of `update-trbdf2-error-control`. | `examples/phase_shift_full_bridge.yaml`, V_out averaged over 1.8-2.0 ms (2026-11-19): 23.975, 23.991 and 23.964 V at `rtol` 1e-5, 1e-6 and 1e-8, with 2.3 k to 6.6 k forced accepts; `integrator='esdirk43'` gives 23.964, 23.966 and 23.961 V. `engine='exact'` gives 24.0661693 V. | Model the devices' output / junction capacitance and the winding capacitance (with both, the run settles as `rtol` tightens), or use `engine='exact'`, which handles the capacitance-free circuit. |
| **TR-BDF2 allocates on a step whose Newton solve is retried.** Accepted and rejected steps do not allocate; the retry builds the failure's error message (DECISIONS 27 of `refactor-unified-transient-stepper`). | About 10 to 16 allocations per retried step, held by a test as a bound. | None needed for results; it costs time only on runs with many Newton retries. |
| The sliding-mode warning says `engine='auto'` even when `engine='trbdf2'` was requested. | Seen on the `V_th = 0` bridge above. | None needed: the warning describes the TR-BDF2 run. |
| **`trbdf2` and `auto` can crash the Python process.** | The saturable-primary PSFB of the mask-chatter row, with no `dt`. With `t_end = 0.2 ms` it raises `SimulationAborted` at t = 0 (`h_min`). With `t_end = 2 ms` the process dies with a segmentation fault (exit code 139). | Pass `dt`. Run variable-step jobs you cannot vet in a subprocess. |
| **A rectifier whose DC side floats cannot start on `engine='auto'`.** Guide trap 8. | 100 V bridge with the DC rails not connected to ground. It aborts with `factorize failed (numerically singular) ... (dt=6e-14) ... node dcn`. A 1 MΩ or 1 GΩ resistor from `dcn` to ground does not help. | Ground one DC rail (92.83 V on TR-BDF2), or pass `dt` (92.54 V at `dt=1e-5`). |
| **TR-BDF2 can step over the first pulse of a Python `switch_fn`.** Before a gate bit has closed a pulse the engine samples it at 4 equal points of each step plus one walking point, at most `h_max/4` apart, and more coarsely while another bit toggles with wider pulses. Once it has closed a pulse, no later pulse at least half as wide as the bit's narrower recent width (quiet: the run's narrowest pulse) is stepped over (`simulate` docstring, `switch_fn`). Pinned by `test_trbdf2_post_event_restart.py`. | Tick buck, 100 ns pulses every 30 µs, then every 10 µs, 1 ms run (h_max = 1 µs): the four regular samples are 250 ns apart and phase-locked to the 10 µs period, so only the walking sample can hit a pulse before one is seen. Both device forms stepped over the 30 µs pulse; the branch form stepped over the 60 µs one too, and saw every pulse from 90 µs on. | Pass `dt` of at most 4x the first pulse's width (one gate bit), start that pulse on a source breakpoint or a controller tick, or declare the edges (`p.NativePwm2Switch`, `next_edge_after`). |
| **DSED polls a Python `switch_fn`, so it misses gate intervals shorter than the polling period.** The poll runs every `dt_max/10` (1 µs on a 2 ms run). The warning calls this "correct but slower". | 48 V synchronous buck, 100 kHz, D = 0.4, 200 ns dead time, 50 Ω. With no `dt`, the body diodes are never ON in the record over 1-2 ms (31 % of the samples on `pwl`). V_out over that window is 19.73 V against 20.35 V on `pwl` (-3.1 %). With `dt=1e-7` DSED gives 20.350 V. | Pass a `dt` of at most half the shortest gate interval, or use `engine='pwl'`. |
| `simulate(engine='dsed')` with no `switch_fn` warns that it will poll "this plain Python switch_fn". | RC plus diode, no `switch_fn`: the run is correct but prints that warning. | Ignore it. |
| `start_from_dc_op=True` works only on the fixed engine. | Without `dt` it raises `ValueError: ... start_from_dc_op is a fixed-step entry point`. | Pass `dt`. |

## AC analysis

| Limitation | Measured | Workaround |
|---|---|---|
| **The kernel AC sweep (`run_mna_sweep` with `freqs=` and `v_in_branch_id=`) linearises at the DC operating point only the kinds marked `ac_sweep` in the table above.** It covers R, L, C, coupled and multi-winding transformers, ideal transformers, VCVS, VCCS, CCCS, CCVS, gyrators, sources (waveform current sources add no AC excitation), switches / PWL diodes held at the `mask=` you pass, and the Newton devices (smooth, Shockley and Lauritzen diodes, level-1 MOSFET/IGBT with the tail, saturable inductor with the atan or table law, nonlinear capacitor, PMSM at rest, external devices) through their device law at the DC point of `mask`. Refused with `UnsupportedError` naming the device: the induction machine; a Jiles-Atherton inductor (no unique incremental inductance); a nonlinear permeance on a magnetic node (no DC point yet); a PMSM held at speed or not in torque balance (no equilibrium). Switches without `mask=` are refused too. | Shockley diode at 1 A behind 1 Ω: `|H| = 0.0252` (it used to read 1, then was refused). | Time domain: `run_ac_sweep`, `pulsim.fra.run_fra`, `frequency_response`, or `run_mna_sweep(..., use_kernel=False, dt=, t_end=)`. |

## Periodic steady state

| Limitation | Measured | Workaround |
|---|---|---|
| A slow mode limits the orbit's accuracy to the map's rounding over one minus the Floquet radius. | Zero-mean sine current into 1 µF ∥ 1 TΩ (Floquet radius 1 - 1e-9): v(0) = 0.15726 V against 0.159155 V. | Add the real loss; check `floquet_radius`. |
| `engine='pwl'` / `'trbdf2'` shooting differentiates the map by finite differences (`n` period runs per Jacobian, Broyden updates between them), so it costs more runs than the exact stepper and has no orbit sensitivities (`sensitivities=` refused). `pwl` needs a gate edge at `t = 0` and `T/dt` an integer; `trbdf2` needs a schedule with `next_edge_after` and fixes the step ceiling at `dt` (default `T/100`); nonlinear devices are refused on both. | DCM buck from zero: 12 period runs on `pwl` (6 on the exact stepper). Orbit vs the engine's own 4000-period transient: 1.2e-11 (`pwl`) and 2.6e-9 (`trbdf2`) of the state scale. | `engine='exact'` when the circuit is in its coverage. |

## Design evaluation (`pulsim.design`)

| Limitation | Measured | Workaround |
|---|---|---|
| The `pss` route's thread scaling needs the kernel batch: a circuit-form evaluator with declarative KPIs (`Kpi.*`, `Kpi.formula`) runs in one GIL-free call; a factory (`build=`), a callable KPI, `gradients=` or a thermal loop take the Python path (named in `counters["batch"]`). | Buck Pareto, 4096 designs, load 15-19 on 10 cores: kernel batch 10022 / 21621 / 25369 evaluations/s on 1 / 4 / 8 threads; the Python thread path 3597 / 2286 / 2405. | `Evaluator(circuit=..., fields=..., schedule=...)` with `Kpi.formula`; else `backend="process"`. |
| `method="transient"` on the fixed-step `pwl` engine is first order at gate edges (the step ending at an edge takes the new mask at its end), and its KPIs with it; the energy balance closes on the engine's own trajectory, so it does not flag this. | DCM buck at T/200: `p_out` 0.87 % low, T/800 0.083 % (probe `openspec/changes/add-design-optimization-workflow/probes/pin_quadrature.py`). | `method="pss"` (2e-7 at T/200), or `options={"discontinuity_restart": True}` (3.5e-6). |
| A device whose power no loss model covers (Lauritzen diode, level-1 MOSFET/IGBT, controlled sources) refuses the efficiency. | Buck with a Lauritzen diode: efficiency would read 0.99989 without the diode's 1.36 W of 28.4 W; it is `refused` (`feature="losses"`). | Evaluate without `loads=` for the other KPIs, or use the PWL diode / switch. |
| `method="transient"` settles to the engine's own orbit. | DCM buck on `trbdf2`: estimated distance 1.4e-10, true error 2.2e-5 (the engine's accuracy). A PWM voltage source on `trbdf2` never settles to 1e-6 (period noise 2.7e-4). | `method="pss"` (exact stepper), or a switch with a schedule instead of a PWM source. |

## Event losses in the transient (`set_device_losses`)

| Limitation | Measured | Workaround |
|---|---|---|
| `engine="pwl"` and `engine="trbdf2"` read loss tables inside the run. `dsed` / `exact` reads them at a fixed `Tj=` (switching energy at each event, conduction as the exact ∫v·i of each mode, impulsive energy booked to the switch) runs the junction temperature loop (a `junction=` / `heat_node=` node: each step's heat as the network's exact response, each event's energy as an impulse; the reconstructed current of a branch the heat flows through at once, e.g. the ambient source's, excludes that heat current), and refuses by name a deposit node an algebraic node voltage reads instantly, a `V_on` conduction table, time-varying sources, and `integrator='bdf2'` (a `resume_from` continues the ledgers, field 17; a fresh `t_start` run books the zero-start run's ledgers with event times shifted by `t_start`); `bdf1` refuses every loss port (`UnsupportedError`, feature `device_losses`, the device). | Conformance cell `test_device_losses_cell`. | `engine="pwl"` or `"trbdf2"`, or the post-hoc `device_loss_summary` / the periodic-orbit evaluator on the exact engine. |
| The event energy is not drawn from the circuit: the PWL switch is ideal, the table energy only heats the junction. | By design, as in PLECS and PSIM. | Model the transition (`add_power_mosfet`) when the electrical side must see it. |
| `T_j` does not feed back into the device (`V_f`, `g_on` stay fixed). | Task 4.1 (optional) not done. | Re-run with `V_f` / `g_on` at the settled `T_j`. |

## Node domains

| Limitation | Measured | Workaround |
|---|---|---|
| Result traces carry no unit. `Circuit.unit_of(node)` knows it (`degC` on a thermal node, `rad/s` / `rad` on a PMSM's speed / angle node); `SimulationResult` does not. | — | `circuit.unit_of(name)` when labelling a plot. |
| Only declared tags, PMSM speed/angle pins, the induction machine's speed pin and the magnetic (winding / permeance) pins set a domain. A DC machine's `omega_node`, the mechanical helpers and user-built mechanical networks stay electrical unless declared. | DC machine with `omega_node` declared mechanical: `node_domain` reads `mechanical`, samples bit-identical to the untagged run. | `builder.node(name, "mechanical")` before `build()`. |

## Devices

| Limitation | Measured | Workaround |
|---|---|---|
| **External devices (`add_external_device`) are refused by the exact engine (`dsed`) and `bdf1`.** `dsed` propagates each topology mode in closed form and admits no Newton device; a law known only through `f`, `q` and their Jacobians at sampled points has no closed-form propagator. No sandbox: a model library is native code. Event functions (`n_events > 0`) are refused: no engine locates an external zero crossing yet. A YAML `external_device` loads only a library listed in `allow_native_code=[...]`; inline `code:` is not accepted. | `UnsupportedError(kind='ExternalDevice')` on `dsed` / `bdf1`. | `pwl`, `trbdf2`, the DC operating point and the kernel AC sweep run it (`docs/external-devices.md`). |
| **`add_nonlinear_diode` is retired.** Its law `logistic(κ(v − V_F0))·(v − V_F0)/R_d` sources power for `0 < v < V_F0`. It now raises `RetiredDeviceError` unless called with `allow_non_passive=True` (YAML: `allow_non_passive: true`). | Default parameters, 0.636 V forced: −1.3923 A. 20 V / 1 kHz sine clamp through 10 Ω: `pwl` at `dt = 1 µs` aborts; `trbdf2` RMS 9.634 V against 10.01 V for the PWL and Shockley diodes. | `add_diode(..., V_f=)` or `add_shockley_diode`. |

## Switching helpers

| Limitation | Measured | Workaround |
|---|---|---|
| **`make_pwm_switch_fn` edges jitter by one `dt` step.** It compares a floating-point phase. Guide trap 10. | 100 kHz, D = 0.5, `dt = 1e-7`: 49, 50 or 51 on-steps per period, where 50 were asked. `make_dead_time_pwm_pair_fn` with 200 ns dead time gives the high side 48 or 49 on-steps, where 48 are expected. | Make `D/f_sw` a whole number of steps and compute the PWM in steps (`grid_pwm`, guide 3.2), or split the steps at the edges: `discontinuity_restart=True, enable_substep_state_correction=True` (the helpers carry `next_edge_after`). |

## Sources

| Limitation | Measured | Workaround |
|---|---|---|
| **No behavioural sources.** The controlled sources (`add_vcvs`, `add_vccs`, `add_cccs`, `add_ccvs`, `add_gyrator`) are linear with constant gains; there is no expression source (SPICE `B`, PSIM math block) and no polynomial or table-driven gain. | `spice_to_builder` refuses `B` sources. | A nonlinear law of node voltages or branch currents: `b_extra_fn` or a closed loop. |
| **The table source is a current source only.** There is no PWL voltage source. | `add_pwl_current_source` exists; `add_pwl_voltage_source` does not. | A voltage table: drive a 1 Ω load with the current table and use it through a VCVS (`add_vcvs`), or a `b_extra_fn`. |
| **A controlled source on the fixed engine shows the engine's first-order start** (guide trap 9), like any reactive circuit. | Gyrator (g = 0.01 S) with 1 µF, 1 V through 10 Ω, `dt = 1e-6`: 63.19 mA at 1 ms against the exact 63.21 mA, error `dt/(2τ)`. An identical 10 mH inductor gives the same trace to 1e-9. `trbdf2` is within 1.5e-8, `dsed` within 1e-15. | Halve `dt` and compare, or run without `dt`. |
| `dsed` runs all seven new kinds. | A circuit with every controlled kind: `trbdf2` and `dsed` agree to 1e-7 after 10 µs. Sample 0 differs by design: `dsed` reports the consistent algebraic state, the other engines 0 V. | None needed. |


| Limitation | Measured | Workaround |
|---|---|---|
| **`res.i()` and `device_loss_summary` do not cover the Lauritzen diode, the nonlinear capacitor or an IGBT with a tail.** Their current needs a stored charge the states do not give the law. The static Newton kinds (Shockley, smooth diode, level-1 MOSFET, level-1 IGBT without a tail) and the saturable inductor are covered: against a 1 mΩ sense resistor within 2e-12 of the peak current and 4e-13 in energy on `pwl` and `trbdf2`. | `res.i("X")` raises `NotImplementedError` naming the recipe. | Put a 1 mΩ resistor in series and read `res.i` of the resistor (guide trap 1). |
| **`device_loss_summary` and `device_thermal_summary` have no time window.** Both average the whole record, start-up included. | Bridge of trap 13 (`V_th = 0.7`, `dt = 1e-6`, 0.1 s): the conduction model (`V_f0 = 0.7`, `r_on = 0.02`) on `BR__D1` reads 2.338 W over the run and 0.136 W over 80-100 ms (17.2x). | Apply the model to `res.i(name)` with `p.time_average(t, ..., t_from=...)` (guide 3.4). |
| **`p.time_rms` reads pulsed currents low on a fixed grid.** `time_rms` joins the samples with straight lines, so each hard edge becomes a one-step ramp. Guide trap 14. | Buck 48 V to 12 V, D = 0.25, 100 kHz, 4 Ω. MOSFET RMS over 4-6 ms: 1.4905, 1.4956 and 1.4981 A at `dt` = 1e-7, 5e-8 and 2.5e-8, against 1.5007 A from `sqrt(time_average(i**2))` at all three. The `i_rms` that `device_loss_summary` gives a switch or a PWL diode integrates `i**2` exactly in each step (1.50522 A at `dt` = 1e-7, 4 Ω, 47 µF). | `np.sqrt(p.time_average(t, i**2, t_from=...))`. |
| **The missed-commutation warning of `res.i` repeats once per call site.** Python's default filter keys on the calling line. | Six reads of one diode from four lines print it four times. | `warnings.filterwarnings("once", message=".*commutation rule forbids")` prints it once. It matches both the "OFF while forward-biased" and the "ON while reverse-biased" form. |

## Importers

| Limitation | Measured | Workaround |
|---|---|---|
| **`add_power_mosfet` (default composite): no `res.i` of the device and no loss row.** | `res.i("Q1")` raises like `mosfet_level1`; `device_loss_summary` omits it. | `fq_device=True`, `p.double_pulse_test` (sense resistor), or a 1 mΩ resistor in series. |
| **`add_power_mosfet(fq_device=True)`'s loss row is `v_DS · i_D` of the external drain / source pins.** The gate-drive energy is not in it, and at a switch event elsewhere in the circuit the device current's right limit is taken as its sample (the device itself is continuous). | `res.i("Q1")` is the drain-terminal current: within 1e-9 of a 1 mΩ sense resistor on `pwl`; on `trbdf2` 4.35e-9 of the peak (strict xfail in `python/tests/test_power_mosfet.py`). | For gate loss, read `res.i("Q1.rg")`. |
| **The level-1 MOSFET has a negative slope near threshold.** The triode polynomial at small or negative overdrive makes dI/dV_DS < 0 at V_DS = 0 for V_OV in (≈ −0.72/κ, 0), and just past saturation onset (down to −0.040 S for K = 1.43, κ = 15); there `i·v` reaches −0.29 mW. The zero-voltage leak `α(1−β)·K·V_OV²` is gone since 2026-11-19: zero current at V_DS = 0, C¹ across it (add-unified-device-interface DECISIONS #44). | K = 1.43, κ = 15: min dI/dV_DS −0.040 S; min i·v −0.29 mW at V_OV = −0.20 V, V_DS = −0.21 V. | Drive the gate well past threshold (κ·V_OV ≫ 1), or raise `kappa`. |
| **`add_power_mosfet` non-goals:** p-channel, live T_j inside the law (T_j is run-constant), drain inductance and package networks, avalanche, SiC/GaN effects; a table channel (`channel=p.ChannelTable(...)`) is taken at the T_j it was measured at (temperature coefficients refused with it). | - | - |
| **A gate transient on `pwl` needs a step that resolves it.** | The 400 V / 10 A double pulse with a 65 ns plateau: 7501 steps at `dt` = 0.2 ns; `trbdf2` takes about 1200. | `engine="trbdf2"`. |
| **VDMOS import: absent capacitances are 0 (LTspice), not ngspice's defaults** (CGDMAX 2 nF, CGDMIN 20 pF, CGS 1.4 nF, CJO 0.5 nF). CJO uses the junction law without SPICE's FC linearisation. | - | Write every charge parameter in the card; `charge="drop"` restores the ideal-gate device. |
| **`spice_to_builder` reads a subset of SPICE.** | `Q` (BJT) and `K` (coupled inductors) raise `NotImplementedError`. A `{...}` expression takes the parameter grammar (no unit suffixes inside the braces). `B` sources raise `ValueError`. `.subckt` is flattened correctly. | Build those parts with `CircuitBuilder` (`add_transformer` for coupled inductors) or a YAML netlist. |

## Machines

| Limitation | Measured | Workaround |
|---|---|---|
| **Native machines: DC machine, PMSM and induction machine are solved in the matrix; the observers are not.** `b.add_dc_machine`, `b.add_pmsm_mna`, `b.add_induction_machine_mna` are MNA-native on `pwl` and `trbdf2`; `p.add_pmsm`, `p.add_bldc`, `p.add_induction_motor` are explicit observers on `pwl` only (first order). The DC operating point holds a machine's speed and angle at `omega0` / `theta0` (no torque balance; a speed or angle node another device drives is refused by name); the PMSM and induction machine have no AC-sweep stamp; `dsed` refuses them (speed-dependent, not LTI between events). No machine model is validated against measurements: the tests check analytic limits of the equations and independent ODE integrations. | DC motor observer, sine drive, speed error 13.6 % at `dt` = 100 µs (first order); `add_dc_machine` 0.48 % (second order). Induction machine free acceleration: 1.0e-5 of synchronous speed at 20 µs on `pwl` with `discontinuity_restart=True`. | Use the native machines; halve `dt` to check an observer run. On `pwl` pass `discontinuity_restart=True` for a three-phase start (first order otherwise). |
| **Flux-map PMSM: DC point only at rest, no YAML form, defined on its grid plus a 10 % band.** `p.FluxMapDQ.from_tables` fits one co-energy spline to both flux tables (reciprocal by construction) and `b.add_pmsm_flux_map` stamps it on `pwl` and `trbdf2`. The held-mechanics DC point refuses `omega0 != 0` by name (the stator law is nonlinear in the current when spinning); `yaml.emit` refuses the map; the law is defined on the grid plus 10 % of each axis span (checked positive definite there; an accepted state outside is refused by name, Newton iterates pass through a smooth unmeasured extension). Reciprocity is judged from the tables' cross derivatives (central differences). The induction machine has no saturation. | Synthetic cross-coupled map: reciprocity defect 0.19 % (reciprocal, 1 A grid), 9.1 % with λ_q × 1.1 (refused at the default 2 %). | Tabulate the whole operating range; start a flux-map machine from zero state or at rest. |
| No `components()` entry for a composite machine as a whole (its sub-elements are listed). | — | — |

## Passives

| Limitation | Measured | Workaround |
|---|---|---|
| **The π-section line overshoots an ideal step by 27 % at any section count.** `b.add_transmission_line(model="pi")` (the default) is a lumped ladder. `model="bergeron"` is the exact travelling-wave line ([Transmission line](transmission-line.md)). | 50 Ω, 100 ns, source-matched, open end: far-end peak 1.271 V with 50 sections (exact 1.000 V). With a 20 ns edge and `recommend_line_sections` (55) the peak is 1.046 V. | `model="bergeron"`. |
| **Bergeron line: the variable step does not estimate the line's interpolation error.** On `trbdf2` the step is capped at τ and set by the reactive states; a feature recorded at one end is landed a delay later at the other, but nothing refines a smooth wave below the step that first recorded it. `esdirk43`, the exact engine, dsed, bdf1 and `periodic_steady_state` refuse the line by name. | A resistive circuit with a 100 ns line driven by a 3.3 MHz sine: 6.6 mV error on 0.5 V at any `rtol`. | Set `dt=` (the step ceiling) to resolve the waves, or use `pwl` with τ a multiple of `dt`. |
| **`rac:` takes ladder cells only.** YAML `transmission_line` is the Bergeron line (`model="pi"` is Python only). `rac: {R_dc, L_hf, R: [...], L: [...]}` is read; an `(f, R_ac)` curve is refused by name (fitting needs Python). `nonlinear_capacitor` takes `law:` or `dc_bias: {v, C}` (no residual check: `dc_bias` points become knots as given). Other unknown per-device keys of the older types are still ignored without a warning. | — | Fit with `p.passives.fit_rac_ladder` / `fit_mlcc_dc_bias` in Python and write the cells or the table. |
| One ESR value (no frequency dependence), single-pair line (no coupled three-phase cable). The DC-bias capacitor is a Newton device (`pwl` / `trbdf2` and the AC sweep, linearised at the DC point; dsed and bdf1 refuse it by name) and integrates the small-signal curve: no hysteresis, no ageing, no temperature. | — | Read ESR at the switching frequency; fit `R_ac(f)` of windings with `fit_rac_ladder`; the MLCC curve with `fit_mlcc_dc_bias`. |
| **ESL resonance on the fixed step.** `esl` in series with `C` resonates at `1/(2π√(esl·C))`; a fixed step that does not resolve it rings at the Nyquist rate. | — | `dt <= 1/(20·f_res)`, or `engine="trbdf2"`. |
| **A winding with an `rac=` ladder cannot be coupled.** `add_inductor_coupling` refuses it with `UnsupportedError` (feature `rac`): coupling the main inductor alone would leave the cells' flux uncoupled. | — | Couple windings built with `esr=`, or use `add_transformer`. |

## Magnetic networks

| Limitation | Measured | Workaround |
|---|---|---|
| **No fringing and no core-shape library.** `add_air_gap` is `P = μ₀·A/l` on the given area; every section takes its own area, length and permeability. Shipped parameters are synthetic. | — | Enlarge the gap area by your fringing factor; enter the sections of the core yourself. |
| **Rate-dependent loss is eddy loss only, and not validated against measurement.** `add_magnetic_resistance` gives `R_m·(dΦ/dt)²`, `add_hysteretic_permeance` the Jiles-Atherton loop and `add_excess_loss` Bertotti's `c·sign(u)·|u|^½`, linear below the flux rate `eps` (sine loss (3/4)·(eps/Û)² below the square-root law; the exact engine refuses it by name). `fit_core_loss_elements` maps loss data to `R_m` and `c`; a fit is not a validation. | — | `add_hysteretic_core_inductor` for one flux path; Steinmetz / iGSE after the run. |
| **DC operating point: a saturable permeance on a node reached only through permeances is refused by name** (its flux row needs the Newton DC refresh). Linear permeances get flux rows (per node, or per group joined by magnetic resistances and winding ports); the DC flux is solved from zero. | — | Start the transient from zero or from `initial_state=` (tutorial 09). |
| `set_initial` ignores a saturable permeance (it records capacitor and inductor values only), without a warning. | — | Write its MMF into `initial_state` at its node. |
| A saturable permeance runs on `pwl` and `trbdf2` only; `dsed`, `bdf1` and the AC sweep refuse it by name. | — | Linear permeances run on every engine. |

## Found by the examples audit (2026-11-13)

Reproducers are in `probes/` (run them against the tree to re-measure);
each example affected says so where it prints the number. Fixed since
the audit (see the CHANGELOG): the `pseudo_trans` divergence,
`run_transient_adaptive` reading low, the forward-Euler induction-motor
flux and `run_ac_sweep` reading inside the start-up. The `b_extra` sign
and the moving `branch_var_id_for_*` index are contracts, documented in
`simulate` and `docs/api-reference.md`.

| Limitation | Measured | Workaround |
|---|---|---|
| **`device_loss_summary`'s resistor `P_avg` is first order in `dt` on PWM currents** (each edge becomes a one-step ramp in the stored samples; the switch `P_avg` is exact). Owner: `python/pulsim/losses.py`, `device_loss_summary` (line 1556). Probe: `probes/loss_resistor_first_order_edges.py`. | R in series with a PWM switch: 1114.24 / 1121.77 / 1125.54 W at 50 / 100 / 200 steps per period against the exact 1129.30 W; M1 11.2926 W against 11.2930 W. | `p.time_average(t, R * i**2)` with `i` from `res.i`, or halve `dt`. |
| **`compute_bh_loop` can return a negative loop area (a negative core loss).** A closed sine loop on the `permalloy` J-A set has ∮H dB < 0 above ~300 A/m, and the `ferrite_n87` area shrinks as the drive rises into saturation. Owner: `python/pulsim/hysteresis.py`, `JilesAthertonModel.update`, the irreversible slope at line 336. Probe: `probes/ja_negative_loop_area.py`. | permalloy 29.1 / −12.4 / −14.8 J/m³ at 100 / 300 / 1000 A/m; ferrite_n87 42.8 / 28.8 / 13.9 J/m³ at 300 / 1000 / 3000 A/m. The δ_M guard and a ≥ 0 clamp on the slope do not remove it. | Check the sign of `energy_per_cycle_per_m3`, and that the J-A B matches the circuit's B (`run_psfb_ja_loss_analysis.py` does both). |
| **Tightly coupled half bridge: `trbdf2` aborted and a coarse fixed step read low without warning** (audit report). Owner: `core/include/pulsim/solver/trbdf2_transient.hpp`. Probe: `probes/half_bridge_tight_coupling.py`. | Audit: `refactorize failed (numerically singular)` at h ≈ 5e-14 s for k = 0.999 / 0.99999; at k = 0.99999 `dt` = 2e-8 gave 4.17 V against 4.99 V at 1e-8. NOT reproduced on this tree with the project's half bridge (3 ms, mean over 2–3 ms): k = 0.99999 reads 4.869 / 4.870 V at `dt` = 2e-8 / 1e-8 and `trbdf2` runs (4.924 V); k = 0.999: 4.685 / 4.688 / 4.695 V. The configuration behind the audit's numbers was not recorded. | `engine='pwl'`; halve `dt` and compare. |

## Older entries

[`KNOWN_LIMITATIONS.md`](https://github.com/lgili/Pulsim/blob/main/KNOWN_LIMITATIONS.md)
at the repository root was written for 1.3. It covers the schematic
renderer, MMC, M3C, the sparse LU and LiveScope, and it was not
re-verified for this page. Its 2.0 engine section covers the
`h_min` stall and the mask chatter above.

The library follow-ups are tracked in
`openspec/changes/add-ai-usage-guide/tasks.md`, section 12.
