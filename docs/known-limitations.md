# Known limitations of Pulsim 2.0

This page lists the defects and gaps that Pulsim 2.0 ships with. Each
one was reproduced on the 2.0 code base, including the unreleased fixes
listed under `[Unreleased]` in the changelog. Each row gives the
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

## Engine capability matrix

Which engine simulates which device kind. This table is the kernel's
(`pulsim.engine_capabilities()`); `test_engine_conformance.py` checks
that it equals this page and that every "yes" agrees with `engine='pwl'`
within 1 % of the observable's span and every "no" raises
`pulsim.UnsupportedError` naming the engine and the branch. `ac_sweep` is
the kernel AC sweep (`run_mna_sweep` with `freqs=`), which refuses every
Newton device. `NonlinearDiode` is retired: `add_nonlinear_diode` refuses
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
| NonlinearDiode | NewtonStatic | yes | yes | no | no | no |
| CurrentSource | Source | yes | yes | yes | yes | yes |
| PWMVoltageSource | Source | yes | yes | yes | yes | yes |
| SineVoltageSource | Source | yes | yes | yes | yes | yes |
| PulseVoltageSource | Source | yes | yes | yes | yes | yes |
| MosfetLevel1 | NewtonStatic | yes | yes | no | no | no |
| IgbtLevel1 | NewtonStatic | yes | yes | no | no | no |
| VCVS | Linear | yes | yes | yes | yes | yes |
| SaturableInductor | NewtonStateful | yes | yes | no | no | no |
| NonlinearCapacitor | NewtonStateful | yes | yes | no | no | no |
| ShockleyDiode | NewtonStatic | yes | yes | no | no | no |
| LauritzenDiode | NewtonStateful | yes | yes | no | no | no |
| PmsmMna | NewtonStateful | yes | yes | no | no | no |
| IdealTransformer | Linear | yes | yes | yes | yes | yes |
| VCCS | Linear | yes | yes | yes | yes | yes |
| CCCS | Linear | yes | yes | yes | yes | yes |
| CCVS | Linear | yes | yes | yes | yes | yes |
| Gyrator | Linear | yes | yes | yes | yes | yes |
| SineCurrentSource | Source | yes | yes | yes | yes | yes |
| PulseCurrentSource | Source | yes | yes | yes | yes | yes |
| PwlCurrentSource | Source | yes | yes | yes | yes | yes |
<!-- engine-capabilities:end -->

The charge laws `ChargeLaw.table` / `.vdmos_cgd` are the
NonlinearCapacitor row. `add_power_mosfet` is a composite of the
MosfetLevel1, Capacitor, NonlinearCapacitor, Resistor, Inductor and
Shockley / Lauritzen diode rows: it runs on `pwl` and `trbdf2` and is
refused on `dsed`, `bdf1` and the kernel AC sweep, naming the channel.

Circuit-level limits, also raised as `UnsupportedError` (with `feature`
set): coupled inductors (`add_transformer`, `add_inductor_coupling`) on
`dsed` and `bdf1`; builder initial conditions on `dsed`; PWL diodes with
`integrator='bdf2'`, or with `V_f > V_th` (use `V_th = V_f + 1e-3`), on
`dsed`; C blocks, controllers
(`step_observer`, `closed_loops`), `start_from_dc_op`, `store_every`,
`live_stream`, `max_dt_halvings`, `mmc_arms`, `on_cache`, the inductor
guards, `enable_substep_state_correction` and `discontinuity_restart` on
`dsed`; the fixed-step keywords (`discontinuity_restart` among them) on
`trbdf2`; a `b_extra_fn.next_jump_after` declaration is read by `pwl`
only (a no-op on `trbdf2` / `dsed`, which control the step on their own
error estimate); `enable_nonlinear_refresh=False` with a smooth /
Shockley diode, MOSFET or IGBT on every transient engine.

## Engines

| Limitation | Measured | Workaround |
|---|---|---|
| **Fixed-step controller ticks land on the `dt` grid.** Source j ticks at `t_start + k*T_j` on both engines; `trbdf2` lands each instant, `pwl` delivers it at the first grid point at or after it. | `T = 2.5*dt` over 100 µs at `dt` = 1 µs: 40 ticks, each late by less than `dt` (`result.control_tick_jitter_max`), with a warning; `T < dt` raises. | `dt = T/n`, or omit `dt` (`trbdf2`). |
| **By default the fixed engine is first order after a discontinuity.** The start (ICs included), every gate edge and every source jump leave a stale trapezoidal history (`v_prev`, `i_prev`). Guide trap 9; pinned by `test_first_step_order.py` and `test_fixed_step_discontinuity_order.py`. | RL decay, R = 1 Ω, L = 1 mH, `i0` = 5 A: relative error at 5 τ 1.997e-3, 9.99e-4, 5.00e-4 at `dt` = 4, 2, 1 µs (order 1.00); with `discontinuity_restart=True` (three SDIRK(2,2) steps per discontinuity) 6.66e-6, order 2.00. Buck CCM `i_L` at T/50: 1.18e-2 A, 1.17e-7 A with the rule. An edge off the `dt` grid, and an undeclared `b_extra_fn` jump, stay first order either way. Sample 0 of `v(a)` reads 0 V either way; the consistent value is -5 V. | `discontinuity_restart=True` (gate sampled at the step start); keep edges on the grid. |
| **TR-BDF2 stalls at `h_min` after a hard switching edge when a Newton device is present.** Pinned by `test_saturable_transformer.py::test_engines_agree_below_saturation`. | Snubbed 48 V flyback with a saturable transformer. `engine='trbdf2'` and `'auto'` abort at t = 4 µs with `||dx|| = 8.787e-08`. `engine='pwl', dt=1e-8` gives V_out = 15.31 V. | Use `engine='pwl'` with a `dt`. Do not loosen `tol_newton_dx` / `tol_newton_res` ([Gotchas](gotchas.md)). |
| **Diode-mask chatter on the fixed engine.** With PWL diodes and a Newton device in the same circuit, the diode mask is re-decided at every Newton iteration and no mask is self-consistent at a small step. Pinned by `test_newton_stall_and_mask_chatter.py`. | PSFB with a saturable primary and a floating secondary. At `dt` = 20, 10, 5 and 2 ns it aborts near t = 1.5 µs, even split into 64 sub-steps. At 50 ns it runs. V_out = `v(vout) - v(rect_neg)`, averaged over 1.5-2 ms, is 6.771 V. With a linear primary it is 6.765 V at 50 ns and 6.938 V at 10 ns (6.937 V at 5 ns). So 50 ns reads 2.4 % low. | Run at 50 ns and accept about 2.5 % low; check it against the linear-primary circuit at 10 ns. `add_shockley_diode` runs at 10 ns but changes the model: each junction drops about 0.7 V and a PWL diode drops none (trap 1). V_out falls to 5.785 V (-16.6 %). Use it only if you want that drop. |
| **TR-BDF2 femtosecond landing step after a diode turn-off.** Pinned by `test_trbdf2_stateful_with_diodes.py::test_no_kilovolt_artefact_on_the_internal_node`. | L-D-RC rectifier, 20 V source, constant 111 µH. On TR-BDF2 the L-D node peaks at 1539 V on a 3e-14 s step; on `pwl` it peaks at 23.1 V. The output peak is the same on both: 19.640 V and 19.641 V. | Do not take `max()` of internal nodes from a TR-BDF2 run. Check peaks on `engine='pwl'` with a `dt`. |
| **TR-BDF2 can miss PWL-diode commutations.** Guide trap 13. | 230 V rms bridge, 0.1 Ω, 470 µF, 1 kΩ. With `V_th = 0` it runs half-wave: 318.79 V and 12.91 V pp, against 322.06 V and 6.38 V pp with `dt=1e-6`. With `V_th = 0.7` and 1 mH of line, D2 and D4 are held OFF while forward-biased for one sample at 75.92 ms; V_dc is 321.80 V against 321.82 V. Snubbed boost (24 V, 220 µH, 50 kHz, D = 0.5, 1 nF across D1, 100 µF, 30 Ω), V_out over 2-3 ms: with `V_th = 0`, 24.64 V on `trbdf2` and `auto`, against 54.89 V with `dt = 1e-8`. With `V_th = 0.7` on `auto`, 56.18 V against 54.89 V (+2.3 %), and D1 is held ON while reverse-biased on 106 samples. | Pass `dt`. Rerun with `dt` whenever `simulate` warns that a diode "rode its conduction boundary", or `res.i` warns that one was held "in a state its commutation rule forbids". |
| The sliding-mode warning says `engine='auto'` even when `engine='trbdf2'` was requested. | Seen on the `V_th = 0` bridge above. | None needed: the warning describes the TR-BDF2 run. |
| **`trbdf2` and `auto` can crash the Python process.** | The saturable-primary PSFB of the mask-chatter row, with no `dt`. With `t_end = 0.2 ms` it raises `SimulationAborted` at t = 0 (`h_min`). With `t_end = 2 ms` the process dies with a segmentation fault (exit code 139). | Pass `dt`. Run variable-step jobs you cannot vet in a subprocess. |
| **A rectifier whose DC side floats cannot start on `engine='auto'`.** Guide trap 8. | 100 V bridge with the DC rails not connected to ground. It aborts with `factorize failed (numerically singular) ... (dt=6e-14) ... node dcn`. A 1 MΩ or 1 GΩ resistor from `dcn` to ground does not help. | Ground one DC rail (92.83 V on TR-BDF2), or pass `dt` (92.54 V at `dt=1e-5`). |
| **DSED polls a Python `switch_fn`, so it misses gate intervals shorter than the polling period.** The poll runs every `dt_max/10` (1 µs on a 2 ms run). The warning calls this "correct but slower". | 48 V synchronous buck, 100 kHz, D = 0.4, 200 ns dead time, 50 Ω. With no `dt`, the body diodes are never ON in the record over 1-2 ms (31 % of the samples on `pwl`). V_out over that window is 19.73 V against 20.35 V on `pwl` (-3.1 %). With `dt=1e-7` DSED gives 20.350 V. | Pass a `dt` of at most half the shortest gate interval, or use `engine='pwl'`. |
| **DSED refuses a three-phase VSI with a floating star neutral.** The message blames floating capacitors and nonlinear devices, although the circuit has no capacitor, and it points to `notes/DSED_BRIDGE_DESIGN.md`, an internal design note. | SPWM VSI with an R-L star load, `engine='dsed'`: `NotImplementedError`. | Run it on `engine='pwl'` with a `dt` (guide 3.7). |
| `simulate(engine='dsed')` with no `switch_fn` warns that it will poll "this plain Python switch_fn". | RC plus diode, no `switch_fn`: the run is correct but prints that warning. | Ignore it. |
| `start_from_dc_op=True` works only on the fixed engine. | Without `dt` it raises `ValueError: ... start_from_dc_op is a fixed-step entry point`. | Pass `dt`. |

## AC analysis

| Limitation | Measured | Workaround |
|---|---|---|
| **The kernel AC sweep (`run_mna_sweep` with `freqs=` and `v_in_branch_id=`) does not linearise anything.** It covers R, L, C, coupled and multi-winding transformers, ideal transformers, VCVS, VCCS, CCCS, CCVS, gyrators, sources (waveform current sources add no AC excitation), and switches / PWL diodes held at the `mask=` you pass. A Newton device (smooth, Shockley or Lauritzen diode, level-1 MOSFET/IGBT, saturable inductor, junction capacitor, PMSM) is refused with `UnsupportedError` (a `ValueError`) naming the branch and its kind; switches without `mask=` are refused too. | Shockley diode biased at 1 A behind 1 Ω: refused (it used to read `|H| = 1`; the small-signal value is 0.0252). | Time domain: `run_ac_sweep`, `pulsim.fra.run_fra`, `frequency_response`, or `run_mna_sweep(..., use_kernel=False, dt=, t_end=)`. |

## Periodic steady state

| Limitation | Measured | Workaround |
|---|---|---|
| **`periodic_steady_state` cannot solve a boost whose diode has `V_th` at or near 0.** It fails loudly ("did not settle after 64 rounds"); the exact stepper's event handling is the cause (exact-propagator work). | Plain boost, PWL diode: fails at `V_th` = 0 and 1e-3, solves at 0.05 (integration review, 2026-11-13). | `V_th = 0.05`, or a long transient. |
| A slow mode limits the orbit's accuracy to the map's rounding over one minus the Floquet radius. | Zero-mean sine current into 1 µF ∥ 1 TΩ (Floquet radius 1 - 1e-9): v(0) = 0.15726 V against 0.159155 V. | Add the real loss; check `floquet_radius`. |

## Devices

| Limitation | Measured | Workaround |
|---|---|---|
| **`add_nonlinear_diode` is retired.** Its law `logistic(κ(v − V_F0))·(v − V_F0)/R_d` sources power for `0 < v < V_F0`. It now raises `RetiredDeviceError` unless called with `allow_non_passive=True` (YAML: `allow_non_passive: true`). | Default parameters, 0.636 V forced: −1.3923 A. 20 V / 1 kHz sine clamp through 10 Ω: `pwl` at `dt = 1 µs` aborts; `trbdf2` RMS 9.634 V against 10.01 V for the PWL and Shockley diodes. | `add_diode(..., V_f=)` or `add_shockley_diode`. |

## Switching helpers

| Limitation | Measured | Workaround |
|---|---|---|
| **`make_pwm_switch_fn` edges jitter by one `dt` step.** It compares a floating-point phase. Guide trap 10. | 100 kHz, D = 0.5, `dt = 1e-7`: 49, 50 or 51 on-steps per period, where 50 were asked. `make_dead_time_pwm_pair_fn` with 200 ns dead time gives the high side 48 or 49 on-steps, where 48 are expected. | Make `D/f_sw` a whole number of steps and compute the PWM in steps (`grid_pwm`, guide 3.2). |
| **`switch_specs` charges the soft edges of a bidirectional PWL switch at the full bus when there is no dead time.** | Guide 3.7 VSI (600 V; 4 mJ / 3.5 mJ at 600 V, 50 A). `INV__HSa` switching loss is 22.64 W with no dead time and 11.29 W with 1 µs. | Model the dead time. |

## Sources

| Limitation | Measured | Workaround |
|---|---|---|
| **No behavioural sources.** The controlled sources (`add_vcvs`, `add_vccs`, `add_cccs`, `add_ccvs`, `add_gyrator`) are linear with constant gains; there is no expression source (SPICE `B`, PSIM math block) and no polynomial or table-driven gain. | `spice_to_builder` refuses `B` sources. | A nonlinear law of node voltages or branch currents: `b_extra_fn` or a closed loop. |
| **The table source is a current source only.** There is no PWL voltage source. | `add_pwl_current_source` exists; `add_pwl_voltage_source` does not. | A voltage table: drive a 1 Ω load with the current table and use it through a VCVS (`add_vcvs`), or a `b_extra_fn`. |
| **A controlled source on the fixed engine shows the engine's first-order start** (guide trap 9), like any reactive circuit. | Gyrator (g = 0.01 S) with 1 µF, 1 V through 10 Ω, `dt = 1e-6`: 63.19 mA at 1 ms against the exact 63.21 mA, error `dt/(2τ)`. An identical 10 mH inductor gives the same trace to 1e-9. `trbdf2` is within 1.5e-8, `dsed` within 1e-15. | Halve `dt` and compare, or run without `dt`. |
| `dsed` runs all seven new kinds. | A circuit with every controlled kind: `trbdf2` and `dsed` agree to 1e-7 after 10 µs. Sample 0 differs by design: `dsed` reports the consistent algebraic state, the other engines 0 V. | None needed. |


| Limitation | Measured | Workaround |
|---|---|---|
| **`res.i()` does not reconstruct a Shockley diode's current, and `device_loss_summary` omits Shockley diodes.** | `res.i("D")` raises `NotImplementedError`; the summary of a V-R-D circuit lists only `R`. | Put a 1 mΩ resistor in series and read `res.i` of the resistor (guide trap 1). |
| **`device_loss_summary` and `device_thermal_summary` have no time window.** Both average the whole record, start-up included. | Bridge of trap 13 (`V_th = 0.7`, `dt = 1e-6`, 0.1 s): the conduction model (`V_f0 = 0.7`, `r_on = 0.02`) on `BR__D1` reads 2.338 W over the run and 0.136 W over 80-100 ms (17.2x). | Apply the model to `res.i(name)` with `p.time_average(t, ..., t_from=...)` (guide 3.4). |
| **`p.time_rms`, and the `i_rms` of `device_loss_summary`, read pulsed currents low on a fixed grid.** `time_rms` joins the samples with straight lines, so each hard edge becomes a one-step ramp. Guide trap 14. | Buck 48 V to 12 V, D = 0.25, 100 kHz, 4 Ω. MOSFET RMS over 4-6 ms: 1.4905, 1.4956 and 1.4981 A at `dt` = 1e-7, 5e-8 and 2.5e-8, against 1.5007 A from `sqrt(time_average(i**2))` at all three. `i_rms` of the summary is 1.5969 A against `sqrt(P_avg/R_on)` = 1.6078 A. `P_avg` is right. | `np.sqrt(p.time_average(t, i**2, t_from=...))`. |
| **The missed-commutation warning of `res.i` repeats once per call site.** Python's default filter keys on the calling line. | Six reads of one diode from four lines print it four times. | `warnings.filterwarnings("once", message=".*commutation rule forbids")` prints it once. It matches both the "OFF while forward-biased" and the "ON while reverse-biased" form. |
| `ThreePhaseVsiResult` and `BridgeRectifierResult` are not exported as `pulsim.X`. | `hasattr(pulsim, "ThreePhaseVsiResult")` is False. | `from pulsim.topology import ThreePhaseVsiResult, BridgeRectifierResult`. |

## Importers

| Limitation | Measured | Workaround |
|---|---|---|
| **`add_power_mosfet` is a composite of Newton devices: no `res.i` of the device and no loss row.** Generic outputs arrive with the unified device interface (Phase 3). | `res.i("Q1")` raises like `mosfet_level1`; `device_loss_summary` omits it. | `p.double_pulse_test` (sense resistor), or a 1 mΩ resistor in series. |
| **`add_power_mosfet` non-goals:** p-channel, live T_j inside the law (T_j is run-constant), drain inductance and package networks, avalanche, SiC/GaN effects, a table channel (constant-K Level-1 only), AC analysis. | - | - |
| **A gate transient on `pwl` needs a step that resolves it.** | The 400 V / 10 A double pulse with a 65 ns plateau: 7501 steps at `dt` = 0.2 ns; `trbdf2` takes about 1200. | `engine="trbdf2"`. |
| **VDMOS import: absent capacitances are 0 (LTspice), not ngspice's defaults** (CGDMAX 2 nF, CGDMIN 20 pF, CGS 1.4 nF, CJO 0.5 nF). CJO uses the junction law without SPICE's FC linearisation. | - | Write every charge parameter in the card; `charge="drop"` restores the ideal-gate device. |
| **`spice_to_builder` reads a subset of SPICE.** | `Q` (BJT) and `K` (coupled inductors) raise `NotImplementedError`. A `.param` expression such as `{R*2}` raises `KeyError`. `B` sources raise `ValueError`. `.subckt` is flattened correctly. | Build those parts with `CircuitBuilder` (`add_transformer` for coupled inductors) or a YAML netlist. |

## Machines

| Limitation | Measured | Workaround |
|---|---|---|
| **Only the DC machine is solved in the matrix.** `b.add_dc_machine` is MNA-native; the PMSM observer, BLDC and induction motor are still forward-Euler observers on `pwl` only (`b.add_pmsm_mna` is native). No machine model is validated against measurements: the tests check analytic limits of the equations. | DC motor observer, sine drive, speed error 13.6 % at `dt` = 100 µs (first order); `add_dc_machine` 0.48 % (second order). | Use `b.add_dc_machine`; halve `dt` to check an observer run. |
| No angle node on `add_dc_machine` (`theta_node` raises `UnsupportedError`); no `components()` entry for the machine as a whole (its sub-elements are listed). | — | Integrate `res.v(omega_node)`, or build the angle: `add_vccs` from the speed node into a 1 F capacitor. |

## Passives

| Limitation | Measured | Workaround |
|---|---|---|
| **The π-section line overshoots an ideal step by 27 % at any section count.** `b.add_transmission_line(model="pi")` is a lumped ladder; the exact (Bergeron) line is not implemented and `model="bergeron"` is refused. | 50 Ω, 100 ns, source-matched, open end: far-end peak 1.271 V with 50 sections (exact 1.000 V). With a 20 ns edge and `recommend_line_sections` (55) the peak is 1.046 V. | Give the source a realistic rise time and use `p.passives.recommend_line_sections(delay, t_rise)` sections. |
| **YAML has no `rac:` ladder or `transmission_line` type.** `rac:` on an inductor is refused by name; `esr`, `esl`, `c_par` are read. Other unknown per-device keys are still ignored without a warning. | — | Build ladders and lines from Python. |
| One ESR value (no frequency dependence), single-pair line (no coupled three-phase cable). The DC-bias capacitance is `odd_charge_table` only: no `fit_mlcc_dc_bias`, no YAML, and it is a Newton device (`pwl` / `trbdf2`). | — | Read ESR at the switching frequency; fit `R_ac(f)` of windings with `fit_rac_ladder`; integrate the datasheet C(V) curve into knots yourself. |
| **A winding with an `rac=` ladder cannot be coupled.** `add_inductor_coupling` refuses it with `UnsupportedError` (feature `rac`): coupling the main inductor alone would leave the cells' flux uncoupled. | — | Couple windings built with `esr=`, or use `add_transformer`. |

## Older entries

[`KNOWN_LIMITATIONS.md`](https://github.com/lgili/Pulsim/blob/main/KNOWN_LIMITATIONS.md)
at the repository root was written for 1.3. It covers the schematic
renderer, MMC, M3C, the sparse LU and LiveScope, and it was not
re-verified for this page. Its 2.0 engine section covers the
`h_min` stall and the mask chatter above.

The library follow-ups are tracked in
`openspec/changes/add-ai-usage-guide/tasks.md`, section 12.
