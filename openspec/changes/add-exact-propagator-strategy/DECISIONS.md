# add-exact-propagator-strategy — decisions for the first slice (2026-11-13)

The owner has not answered the Open Questions of design.md. Each was
taken as the option marked "proposed", else the most conservative one.

| Question | Option taken | Why | How to reverse |
|---|---|---|---|
| OQ1 Retire `integrator='rk45'` / `'bdf2'` on dsed | Not done in this slice (task 12.1 deferred) | Removal is an owner decision and needs the Phase-3 implicit strategy as the replacement fallback | — |
| OQ2 Tier 3 inside a run: record and continue, or refuse | Record and continue (proposed). On today's structure "continue" means the scheduler's existing numeric path for that mask (RK45 on `integrator='rk45'`, the detector's choice on `'auto'`), not an implicit strategy, which does not exist inside the dsed schedulers yet. The mask, its tier, κ₂, ‖A·h‖₁ and the reason are in `result.exact["masks"]` | The numeric path is what ran before for a mask without a propagator; it is slow on stiff masks, not wrong. The implicit hand-off and the strict option (task 2.4) need the unified stepper | Raise `UnsupportedError` in `NativeCircuitBuilderAdapter::has_exact_step()` when `exact.tier == 3` |
| OQ3 `n_state ≤ 64` auto limit | Not applied (task 9.2/12.2 deferred) | Needs the stepper's applicability rule and the measured crossover | — |
| OQ4 Block Schur–Parlett | Not started | Non-goal of this change | — |
| OQ5 Engine name for fixed-step Φ/Γ | Not decided (task 10 deferred, Phase 3) | — | — |
| OQ6 Zero- vs first-order hold | Not decided (task 8 deferred, Phase 3) | — | — |

Choices made where the design left room:

| Question | Option taken | Why | How to reverse |
|---|---|---|---|
| Where the energy scale √C, √L comes from | A `state_mass` vector added to `PwlStateSpaceCache::ContinuousLTI`, filled from the extraction's own `caps[i].C` / `inds[i].L` (9-line hunk in `pwl/cache.hpp`, the only change outside `dsed/`) | The reduced state does not map back to devices without the extraction's ordering; the extraction already has the values | Drop the member and pass an empty scale (identity) |
| Which `h` tier 2's bound uses | The scheduler's `dt_max` (`set_max_step`), the longest step it takes | The bound must hold for every step of the run; Illinois probes use shorter `h` | — |
| Padé implementation | Eigen's `unsupported/Eigen/MatrixFunctions` `.exp()` (Higham 2005 scaling and squaring, Padé 13) on the energy-scaled augmented matrix, cached per `h` (16 entries per mask) | Exactly the design's method; Eigen ships it in every distribution the build accepts | — |
| φ₂ | Not added | Only the piecewise-linear source term (task 8.2, Phase 3) uses it | — |
| D4 "first non-zero derivative" | Read on the exact OFF-mode trajectory at δ = 10⁻¹²…1·dt_max: ON iff the signal `v_D − V_on` leaves the band upwards first; a signal that never leaves it is OFF. The band is 1e-12 × max(largest node voltage, |V_th|, |V_f|). The look-ahead decides every diode that is in the band OR whose value law wants to flip; a diode the value law leaves alone keeps its state | A derivative `rᵀ(A·x + b)` is dominated by rounding of the stiff g_off mode (λ ≈ −5e12): at the buck's rest state the true `diL/dt` is −1e-12 A/s and the rounding term 1e-11 A/s. Propagating relaxes the stiff modes first, so the sign that decides is the physical one. Confirming every flip was needed at V_th = 0: at a current zero the OFF-mode raw `v_D` is the ON-mode rounding current divided by g_off, and the value law flipped the diode back and forth for 64 rounds | Restore `SwitchedDiode::decide` for every diode in `settle_diode_cascade_` |
| Settle and projection order | `settle → project onto the slow manifold → re-settle`, up to 4 passes, at the start and at every event | The old order (settle, then project) moved the state after the mask was chosen. In an LC tank driving a series diode from rest the projection put the start past the diode's turn-on crossing, the predicate never changed sign, and the diode never conducted: v_C = 3.3e-6 V instead of 13.73 V, silently, on the untouched build with V_th > 0 (finding recorded in the test `test_series_diode_from_rest_conducts`) | — |
| Builder initial conditions (D5) | Per device from `CircuitBuilder::initial_conditions()`: each inductor state takes its `i0`; capacitor states are the C-weighted least-squares fit of the declared branch voltages (charge-conserving when a capacitor loop makes them inconsistent: 1 µF at 10 V ∥ 3 µF at 2 V → 4 V, `result.initial_condition_adjustment` = 6 V); undeclared capacitors are free (minimum norm) | `builder.initial_state()` writes a capacitor IC onto its positive node only and parallel capacitors overwrite each other, so it does not identify branch voltages | Restore the `UnsupportedError("dsed", "initial_conditions")` in `simulate()` |
| Settle order | Gauss–Seidel (the mask follows each flip), as D4 states | — | Re-apply the overlay once per round |
| Root returned by the event search | The bracket end on the post-crossing side | The state at the event must be past the crossing. With the old value test the side was random, and an ON-mode residual of 1e-12 A reads as hundreds of volts of forward bias in the OFF mode (1/g_off), which is the chatter of findings (1) and (2) | — |
| An event at the very end of a step | Fired (was dropped when within 1e-15 s of the step end) | The next step starts past the crossing and could never see it again | — |
| Scan-step mode filter (D3) | Every under-damped mode (`|Re λ| ≤ |Im λ|`), not only those the predicate sees; at most 1024 scan points per step | Predicates are opaque functions today, so `cᵀV` is not available (task 3.2) | — |
| Allocation removal | Workspaces in `ExactLTI` and the scheduler; `exact_advance_state_into`; the slow-manifold projection reuses the cached tier-1 basis instead of an `EigenSolver` per event; samples still go into `std::vector<Vector>` (the contiguous buffer is task 3.3, Phase 3 recorders) | Counter-gated: see the measurement table | — |
| `event_log` on the builder runners | New `event_log=True` keyword; `simulate()` passes `False` and reads the new `event_times` / `event_types` arrays (exposed as `result.event_times`, `result.event_types`) | The per-event dict with two pybind mask objects was most of the binding cost on a 4000-event run, and nothing in `simulate()` read it | Pass `event_log=True` |
| `V_f > V_th` refusal on dsed (`solver/unsupported.hpp`) | Removed (22-line deletion in another track's file) | The conformance check decides: the sine clamp with `V_th = 0, V_f = 0.7` and with `V_th = V_f = 0.7` matches pwl within the matrix tolerance on dsed now. `V_th = V_f` was never refused and ran silently wrong on the untouched build (minimum 0.0 V against −9.98 V) | Restore the block in `require_supported` |
| Inconsistent capacitor ICs (parallel capacitors) | Left refused by the extractor (`capacitors form a cycle`, an `UnsupportedError`); the charge-conserving fit is in `reduced_initial_state` but no capacitor loop reaches it | Capacitor loops are an extraction change in `pwl/cache.hpp` (other track) | — |

## Re-centred scope, group 13 slice (2026-11-13; report only)

| Question | Option taken | Evidence | How to reverse |
|---|---|---|---|
| Granularity of the route record | One row per mask visited (`result.exact['segments']`, `pulsim.exact_route`), with `n_segments` = the maximal runs of that mask in the recorded switch states | Eligibility is a property of the mask (its `A`, its sources); one row per segment would repeat it thousands of times (the boost visits 2 masks in 969 events) | Emit per-event rows from the event log |
| Reason vocabulary | `'device'`, `'source'`, `'extraction'`, `'cost'` — D11's names. A tier-3 mask whose reason is not a source (near-defective eigenbasis and a stiff `‖A·h‖`) is `'cost'`; there is no `'size'` rule: D11 retired `n_state ≤ 64` into the cost question | D11 text; no state-count limit is implemented on this branch | Rename in `_exact_route.py` |
| Source rule decided without a run | Only a `b_extra_fn` makes every mask tier 3, so `exact_route` answers `'source'` (naming it) before running. PWM / pulse voltage sources left this rule on 2026-11-13 (group 16 corner slice, below): they are exact | Before the slice such a run was numeric and could fail on its own account: an RC behind a PWM source failed `PIController: 6 consecutive rejections`, a diode behind one chattered at the 50 µs edge (that circuit now runs 301 / 301 exact steps, `v(o)` 4.999998216 V = pwl at dt = 1e-8) | Run first, classify from the tiers |
| Which run `exact_route` classifies | `simulate(engine='exact', integrator='rk45')` (the event integrator, which steps masks exactly); the user's kwargs pass through | A diode-free circuit routes to the numeric `auto` scheduler, which keeps no tier record | — |
| `engine='exact'` vs `'dsed'` | `simulate()` maps `'exact'` to `'dsed'` above the engine dispatch (one run, bit-identical); `_VALID_ENGINES` lists both | Owner decision 5 of the Open Questions; above the dispatch so no early return is missed (memory: simulate-engine-dispatch-order) | Drop the mapping |
| Which name results and refusals report | `'dsed'` (`result.engine_used`, every `UnsupportedError.engine`) until the default flip (task 19.1) | It is the capability table's column (`engine_capabilities()`), and callers match on it: `thermal.py` reads `engine_used == 'dsed'`, the conformance suite and PulsimGUI match refusals by `(engine, feature)`. Renaming the reported name is a breaking change with no accuracy gain; it belongs with the release that flips the defaults | Report `'exact'` in `simulate()` and add the column |
| The refusal of a non-eligible segment (13.2), and its default | `engine='exact'` (and `'dsed'`: one engine) refuses by default; `allow_implicit_segments=True` lets the segment run numerically. The scheduler asks the adapter at the step it would take numerically (`refuse_implicit_step`, both the RK45 and the auto scheduler): `UnsupportedError('dsed', 'source', '', '')` for a user `b_extra_fn`, `'cost'` for a tier-3 mode (the tier reason in the message); a circuit the native extractor rejects, which only the numeric Python-adapter bridge could run, is `'extraction'`. Not refused: `integrator='bdf2'` (an explicit, deprecated numeric request), the judge's and the selector's hypothetical modes (they never reach the hook). `allow_implicit_segments` on pwl / trbdf2 / auto is refused by name | Design D11 and Open Question 2 ("with explicit engine='exact' it is refused unless allow_implicit_segments=True"). Measured cost of the default: in the full suite 5 runs reached a numeric segment, all with a user `b_extra_fn` written to test the numeric path (they now pass the keyword); `bit_identity.py` 36/36 and the mode survey unchanged (no reference run has a tier-3 segment). Gain: ten bridges paralleled on one bus ground for minutes and raised "no meaningful progress" at 1 µs; it is now `('dsed', 'cost')` at its first segment, naming κ₂(V) = 2.3e6 and ‖A·h‖ = 1.5e6 | Default `allow_implicit_segments=True` in `simulate()` |
| `exact_route` under the refusal | Passes `allow_implicit_segments=True`: it classifies, it does not refuse | Its rows for 'source' / 'cost' need the run to continue | — |
| What `auto` would route to exact (13.3, report only) | `result.exact['auto_would_route']`: `True` when the run visited at least one mask and every mask was exact; read by nothing, the default route does not change | `probes/auto_route_survey.out` on the 26 YAMLs under `examples/` (`benchmarks/circuits/` is 126 pulsim-v1 netlists the 2.0 loader refuses, so it is not a 2.0 benchmark set; `examples/refrigerator_compressor_cc.yaml` is v1 too). `auto` picks `pwl` on all 25 loadable ones today. Would route exact: 13 — buck, buck_converter, half_bridge_pwm, induction_motor_dol, rc_circuit, rc_pulse, rl_circuit, rlc_circuit, rlc_step_response, stress_buck_switching, stress_large_rc_ladder, three_phase_diode_rectifier, three_phase_inverter. Would not: 12 — 7 Newton devices refused by the capability table (MosfetLevel1 ×4, IgbtLevel1, SaturableInductor ×2) and 5 rejected by the extractor (flyback, half_wave_rectifier, phase_shift_full_bridge, stress_large_resistive_ladder, voltage_divider; the last two have no state). No `'source'` or `'cost'` row: no example has a `b_extra_fn` or a tier-3 mode. `test_auto_would_route_is_reported_and_read_by_nothing` | Drop the field |

## Sources, group 16 slice (2026-11-13)

| Question | Option taken | Evidence | How to reverse |
|---|---|---|---|
| Is the sine class exact today? | Yes, pinned: sine sources enter as shared oscillator pairs (`build_augmented_exact_`) and every step is closed form | `python/tests/test_exact_sources.py`: series RLC from rest under 100 V 50 Hz against the phasor + eigen closed form, 1.0e-14 (v_C) and 2.7e-13 (i_L) relative, all 60 000 steps exact, tier 1; task 16.2's 100 V 50 Hz into 1 Ω + 1 mH at h = T/20 over 5 periods, 3.2e-13 A (gate 1e-10) | — |
| Pulse / PWM sources exact with corner events | Done (task 16.1 / 8.1 / 8.2). Between corners a pulse / PWM source is level + ramp; `exact_advance_affine_into` steps `ẋ = Ax + b + c0 + c1·τ` as `e^{Ah}x + hφ₁(Ah)(b + c0) + h²φ₂(Ah)c1` (tier 1), or the exponential of the (n+2)-augmented matrix (tier 2, uncached: c0, c1 change at every corner). c0 / c1 are each source's value and slope at t⁺ times its column of `-b_projection` (`LTIEntry::pwl_cols`). Each corner is a time event: `next_gate_edge_` returns min(gate edge, `next_source_corner_after(t)`), and a corner fires as a gate-edge event, so the diode modes are re-decided there; the landing assigns `t = t_gate` exactly | `python/tests/test_exact_sources.py` (all fail on 25ceafec, 7 / 9; the 2 sine tests pass on both): ramp into RC 2.7e-13 V and ramp into RL to 1e-10 against the closed forms, a sample exactly at the 1 ms corner; trapezoidal pulse train into a series RLC, a PWM-source buck (20 periods) and audit item 11's instant-edge pulse into an RC within 1e-10 relative of a numpy scaling-and-squaring affine propagation, a sample on every corner, `n_exact_steps` = every step. `probes/source_exactness.out`: ramp RC 8.97e-8 V, 0 exact steps → 2.682e-13 V, 5000 / 5000, tier 1 | Drop `collect_pwl_sources_()`; the masks go back to tier 3 |
| A corner on a gate edge | One event: the min of the two times lands once; `at_corner` is `t_gate < next_switch_edge_`, so when they coincide the gate path runs and the corner is consumed by it (same t, same re-decision); the landing writes `t = t_gate` so floating-point `t + h` never stops one ulp short and re-fires | `test_source_corner_coinciding_with_a_gate_edge_is_landed_once`: 100 kHz source corners on every 50 kHz gate edge, every corner sampled, no duplicate time, 1e-10 against the exact propagation | — |
| Which dsed runs change | Only runs with a pulse / PWM voltage source (they were tier 3 numeric). pwl and trbdf2 are not touched: `bit_identity.py` 36 / 36 and the mode survey 90 / 90 against 25ceafec | Old → new, each against a reference: tutorial 01's 10 V step into 1 kΩ / 1 µF 3.4e-14 → 2.9e-13 V off the closed form (5006 numeric → 5000 exact steps; both within the 1e-10 gate, the new one roundoff accumulated over 5000 exact steps); ramp RC end 9.884223079460 → 9.884223081104 V (closed form 9.884223081104); PWM source → R → diode → C raised a chatter error at 50 µs → runs, 4.999998216 V = pwl dt = 1e-8 | — |
| `project_slow` with a ramp | The cached slow projection is used only when no PWL source exists; otherwise the full affine advance | The cache holds `hφ₁(Ah)b` for a fixed b; a ramp changes c0 at every step | — |
| Table sources | None to make exact | The builder has no table / PWL voltage source; only `add_saturable_inductor_table` takes a table | — |
| Single-shot pulse: which piece holds at a corner landed on exactly (reviewer finding) | `value_at` / `slope_at` compare `t < t_start + x` (the rounded sum `next_corner_after` returns) when `period == 0`; the periodic branch keeps `cycle_position`'s snap | Fresh-context review: `(t_start + e) - t_start` rounds below `e` for about 43 % of (t_start, e); the step landed on the corner then read the old piece, and `next_corner_after(corner)` already pointed past it, so the next exact step ran on the stale level. A 10 V, 300 ns pulse from 1 ms into 1 kohm / 1 nF peaked at 7.27 V instead of 2.59 V; the earlier tests all started at t = 0 or were periodic. Now `test_delayed_single_shot_pulse_reads_the_new_piece_at_its_corner` (instant and ramped edges, corners at 300 / 200 / 600 ns that round down) is within 1e-10; on the old build it was 6.3 V and 36 V off. pwl / trbdf2 bit-identical vs 450314ff, `bit_identity.py` 36/36 vs 25ceafec | Restore the `elapsed < x` comparison |

## Merge of `feat/core-integration` (2026-11-13)

The merge (integration 1aef1a52 into this branch at 876757ab) met four seams. Each was resolved
by intent; pwl / trbdf2 are the merged tree's (the bit-identity baseline of every later step).

| Question | Option taken | Evidence | How to reverse |
|---|---|---|---|
| Integration's sine / pulse / table CURRENT sources on the exact engine (they were numeric there; 13.2 refuses a numeric segment by default) | Exact: a sine current is a shared oscillator pair (`build_augmented_exact_`, entering by its node stamp); a pulse / table current is level + ramp between corners (`PwlSource::PulseI` / `TableI`, `pwl_cols` = `b_projection · stamp`), each corner a time event (`next_source_corner_after`). A table's slope is read at the midpoint to its next knot, so a knot landed on exactly reads its new piece whatever `fmod` rounded | The refusal broke the capability table (3 conformance cells said supported) and the periodic-steady-state solver (4 tests). `test_current_waveforms_are_exact_on_the_exact_engine`: every step exact, every corner a sample; `test_waveform_sources_agree_on_trbdf2_and_dsed` (trbdf2 rtol 1e-10) within 1e-6 on all three, unchanged | Refuse them as `'source'` in `refuse_implicit_step` |
| A stored event sample at a source corner | It reads the sources' LEFT limit (`build_time_varying_b_extra_(t, left_by)` with half the last step, in `reconstruct_full_states` for the pre-event samples); the right limit keeps the new value | Integration's stored-sample guard named it: a PWM → R → D → C run paired D ON with the new 0 V level at the 50 µs edge (i(D) = −0.4966 A, `DiodeModeInconsistentError`). The sample was neither limit. `test_a_left_limit_sample_reads_its_sources_left_limit` | Pass `left_by = 0` |
| The C++ `pulsim::ConvergenceFailure` in Python | Left as pybind11's default translation, `RuntimeError` (the class derives from `std::runtime_error`). No `pulsim.ConvergenceError` exists on the merged tree; inventing a name would compete with the branch that defines it. When that branch meets this one, register the translator there (`py::register_exception<pulsim::ConvergenceFailure>` onto its class); callers catching `RuntimeError` keep working only if its base stays a `RuntimeError` | `grep ConvergenceError` on the merged tree: only the comment in `errors/convergence.hpp` | — |
| How the reference harness runs the exact engine | `v1_reader.BenchCase.run` passes the netlist's `dt` to the exact engine as its output grid (`dt_max`), as it does to pwl; trbdf2 adapts and is unchanged. The gate and the metric are unchanged | The exact engine's samples are exact, but its default grid is 10 µs: two samples per 20 µs period, and the strict gate interpolates linearly between them. On the 15 dsed cells that disagreed after the merge, the error AT the engine's own samples was 1.8e-10 … 5.9e-5 relative (`exact_triage.py`), while the gate read 1.1e-3 … 0.36. Re-measured independently (`.scratch/s5/undersample_check.py`, the merged tree run with and without `dt` against the same trbdf2 rtol 1e-8 reference): 13 cells disagree without `dt`; with it 10 agree (gate 1.1e-3 … 0.288 → ≤ 8.8e-5; error at the engine's own samples ≤ 4.8e-5 of the span; values at instants both grids share equal to ≤ 2.3e-13 on these 10 cells). Caveat (fresh-context review 2026-11-17): `dt` is the exact engine's `dt_max`, not a pure output grid — it also sets the event-projection threshold (100/dt_max), the poll of a switch_fn without edges (dt_max/10), the chatter window and the zero-band probe ladder, so in general it can move the trajectory; the measurement above shows it did not on the moved cells. A separate output-grid argument would remove the coupling (open) and 3 stay (chopper, full bridge, flying capacitor) | Drop `"dsed", "exact"` from the tuple |

Cells moved by the merge (dsed column of `REFERENCE_STATUS.md`, strict gate, the integration
tree's numbers → the merged tree's with the harness change; agree = within 1e-3):

| circuit | integration 1aef1a52 | merged (err_rms) | cause |
|---|---|---|---|
| linear/ll02_rc_pulse_charge | disagree 1.14e-3 | agree 4.5e-7 | merge code (agrees without `dt` too) |
| linear/ll03_rl_step | disagree 1.61e-3 | agree 5.6e-7 | merge code |
| linear/pulse_voltage_rl | disagree 1.97e-3 | agree 5.1e-6 | merge code |
| resonance/ll04_rlc_underdamped | disagree 1.59e-3 | agree 7.3e-7 | merge code |
| magnetics/core_loss_steinmetz_sine | disagree 1.06e-2 | agree 3.5e-6 | output grid (1.06e-2 without `dt`) |
| stress/stiff_rc_high_freq_switching | disagree 1.10e-3 | agree 1.1e-5 | output grid (1.12e-3) |
| buck/buck_with_current_sense | disagree 1.23e-3 | agree 1.3e-6 | output grid (1.24e-3) |
| buck/cascaded_buck_buck | disagree 1.69e-3 | agree 1.5e-6 | output grid (1.84e-3) |
| buck/high_freq_gan_buck | disagree 5.38e-2 | agree 5.1e-6 | output grid (1.53e-2) |
| buck/sync_buck_500khz | disagree 1.14e-3 | agree 2.0e-6 | output grid (1.15e-3) |
| buck/synchronous_buck | disagree 1.22e-3 | agree 1.4e-6 | output grid (1.24e-3) |
| buck_boost/bidirectional_buck_boost | disagree 1.22e-3 | agree 1.3e-6 | output grid (1.24e-3) |
| inverter/half_bridge_lc_filter | disagree 1.21e-3 | agree 1.5e-6 | output grid (1.22e-3) |
| inverter/lcc_resonant_inverter | disagree 9.92e-2 | agree 8.8e-5 | output grid (0.288) |
| isolated/full_bridge_dc | disagree 0.148 | disagree 0.0815 | output grid lowers it (0.356 without `dt`); the rest = pwl's own error (pwl+restart 0.0815) |
| multilevel/flying_capacitor_3level | disagree 0.287 | disagree 0.050 | as above (0.294 without `dt`; pwl+split 0.05) |
| basic/chopper_with_rc_snubber | error (did not settle) | disagree 0.0146 | settle fixed by the merge; residue: a PWM-edge sample stores the right limit at some edges (0.7, 0.9 ms read 5.38 V, the left limit 19.99 V) and the left at others — the trajectory is identical, the label is not |
| boost/boost_classic, boost_pfc_open_loop, cascaded_boost_buck, textbook_erickson_boost_ccm, two_phase_interleaved_boost; buck/buck_dcm, buck_hard_switching_reference, buck_switching, textbook_erickson_buck_ccm; buck_boost/buck_boost (10) | error (did not settle) | agree 2.7e-7 … 2.2e-6 | the complementarity selector (this branch) |

Totals (agree / disagree / error / refused / unverified): 39 / 16 / 11 / 40 / 2 → 63 / 3 / 0 / 40 / 2;
`reference_status` exit 0. The plan's "15 undersampled" over-counted: 10 cells moved by the grid,
4 by the merged code. Neither `metrics.py` nor `differential_runner.py` differs from 1aef1a52; the
harness diff is the one `dt` line.

## Refusal causes on the reference set, and the DC start (2026-11-17)

The 40 dsed refusals of the strict gate after the merge (`.scratch` triage of the merged tree,
messages classified): 23 extractor rejections, 6 `step_observer`, 6 DC start (the harness passes a
full-MNA DC operating point), 5 Newton devices (MOSFET level 1 ×3, Shockley ×2). The 23
extractor rejections are reported through the bridge-fallback message, which drops the
extractor's own reason; the previous agent's draft split them 10 static circuits / 7 coupled
inductors / 6 `G_aa` singular — not re-derived here. `step_observer` and the Newton devices are
correct refusals.

| Question | Option taken | Evidence | How to reverse |
|---|---|---|---|
| A full-MNA `initial_state` on the exact engine (the DC start, 6 cells) | Accepted when its length is the MNA size and differs from the reduced size: every capacitor's branch voltage and inductor's current are read from it and projected onto the starting mode's reduced state by the same charge-conserving least squares as `c0=` / `i0=` (`_NativeCircuitBuilderAdapter.reduced_state_from_full`); a vector whose capacitor voltages need an adjustment > 1e-9·max(1, max\|x\|) is refused with a ValueError, never silently moved; any other length keeps the old error | `test_dsed_full_initial_state.py`: the DC point of an R–L–C circuit stays at the closed form (8 V, 0.4 A) to 1e-9 on 'dsed' and 'exact'; off equilibrium the run is within 7.7e-7 V of trbdf2 rtol 1e-10. Strict gate, dsed: ll01_resistive_divider_dc refused → agree 1.1e-9, stiff_rlc refused → agree 3.6e-9; ll06 / ll07 / ll09 / ll10 (vcswitch) refused → disagree 0.092 / 0.126 / 0.069 / 0.047. Those four are NOT caused by the start: with no initial_state ll06 already reads 0.083 against pwl at dt/40 (263 vs 264 switch edges); the refusal only hid them. Frozen as known disagreements with that note; pwl disagrees there too (0.05–0.12) while pwl converges to the reference as dt → 0 (0.031 → 0.016) and dsed does not — an open vcswitch defect of the exact engine. Totals 63/3/0/40/2 → 65/7/0/34/2 | Restore the shape check in `_dsed_dispatch.py` |

Fresh-context review of this step and of DECISIONS #30 (complementarity) — acted on: a user
`b_extra_fn` can inject a net current into an island, so it now disables the island correction
(the KCL sum would need that source term); a full-MNA vector whose size equals the reduced size
is accepted only where both readings give the same state, otherwise refused
(`test_equal_sizes_with_different_readings_are_refused`); the consistency scale uses node voltages
only. Not acted on: the inconsistency refusal is defensive — the circuits that could trip it
(a capacitor across a source) are refused by the extractor first, so no test reaches it; inductor
currents that are not states and algebraic node voltages are dropped by the projection, as for
`c0=` / `i0=`.

## Samples at a discontinuity, and the vcswitch cells (2026-11-17)

The seven dsed disagreements of the strict gate (ll06 0.0918, ll07 0.126, ll09 0.0692, ll10
0.0465, chopper_with_rc_snubber 0.0146, flying_capacitor_3level 0.050, full_bridge_dc 0.0815)
had one class of cause, not seven: the exact engine's SAMPLES were exact (error at its own
samples ≤ 1.8e-5 of the span on all seven) but the trace the strict measure interpolates was
not the trajectory. Three mechanisms, measured by inserting each fix alone (`.scratch/vc/`):

1. **Jumps drawn as ramps.** A stored sample at an event is the left limit; the right limit
   lives in `right_limit_*`, which no interpolating consumer read. The harness drew each jump
   as a ramp over the next output step, while the trbdf2 reference stores the left limit at
   t_e and the right limit one tiny step later (8.8e-14 s after a 40 µs edge on ll06).
2. **Missing or wrong right limits.** (a) A source corner leaves the reduced state alone, so
   `reconstruct_right_limits` skipped it although the source-driven nodes jump (V(ctrl) on
   every vcswitch cell); (b) the `integrator='auto'` scheduler (every circuit without diodes)
   pushed its gate-edge samples with no pre-event state, so those samples read the sources'
   RIGHT limit and recorded no right limit — "the right limit at some edges and the left at
   others" of the chopper note; (c) a run ending on a corner stored the new level at t_end
   (never integrated); (d) summed grid steps stopped 3 ulps short of an edge and a sliver
   step followed, whose sample read the source's snapped new piece (20 slivers on ll10).
3. **A fast mode read through the output step.** ll06's switch node rises with τ ≈ 14 ns
   (68 µH against 2·1e-4 S of g_off) and ll09's load current collapses with τ ≈ 1 ns during
   the dead time; the netlist's 200 / 250 ns grid drew each as one ramp (ll06 V(sw) 0.0296
   with 1 and 2 fixed; ll09 I(Lload) 0.038).

The vcswitch hypothesis (a control-node threshold crossing sampled on the output grid) is
refuted: Pulsim has no voltage-controlled switch device. The v1 reader maps a vcswitch whose
control node is a source waveform to a scheduled gate (`_on_windows`: the crossing of a PWM
with zero rise time IS its edge, duty − dead_time) and refuses a circuit-driven control node
as Unmapped; the engine lands those edges exactly as time events. "263 vs 264 edges" was the
edge at t_end (stored right on pwl, left on dsed). No predicate was added, so no rounding fuzz
applies; a native vcswitch with hysteresis is a device feature for another change.

| Question | Option taken | Evidence | How to reverse |
|---|---|---|---|
| The convention for a sample at a discontinuity | The result grid stays strictly increasing; the stored sample at an event is the LEFT limit everywhere (gate edges and source corners on both schedulers, and the final sample when a corner lands on t_end); the RIGHT limit is recorded in `right_limit_*` whenever the full state jumps (state, mask, or a source); `pulsim._result_views.with_right_limits(result)` returns the two-sided trace with the right limit at `nextafter(t_e)`, and the harness's `traces()` reads every result through it. A repeated time was rejected: `np.interp` returns the later duplicate AT t_e while the reference's sample at t_e is its left limit, so the trapezoid on the interval before t_e would carry the whole jump² (≈ 0.02 of span on ll06, estimated from its trbdf2 spacing); `np.gradient` and every Δt divider would divide by zero. The ulp sample is an exact sample of the trajectory (the propagator over one ulp is the identity to rounding), not a moved one | Gate with the two-sided trace only (fix 1): chopper 0.0146 → 2.9e-5, full_bridge_dc 0.0815 → 2.7e-5, flying_capacitor_3level 0.050 → 3.7e-5, ll10 legs 0.040 → 1.1e-4, ll07 V(sw) 0.018 → 3.5e-5; V(ctrl) cells needed fix 2. pwl / trbdf2 results carry no right limits, so their cells are untouched; no sample of any trace is dropped | Drop the `with_right_limits` call in `differential_runner.traces` |
| Resolving a mode faster than the output step | After an event that changes the mode or a source's piece, the exact path also stops at t_e + s·√2^k, s from max(1/(8ρ_max), 1e-5·dt_max) to min(30/ρ_min, dt_max), ρ the decay rates −Re λ > 8/dt_max of the new mode (`exact_fast_samples`); modes gone in less than 1e-5·dt_max add nothing (a DCM buck's 1e-13 s g_off modes). Extra exact samples only: no state, mask or event decision is touched | ll06 V(sw) 0.0296 → 1.7e-4 (11 298 samples, was 10 152); ll09 0.0692 → 1.7e-4 (15 587, was 12 022); closed form `test_a_fast_mode_after_an_event_is_sampled` (τ = 10 ns, dt 1 µs: interpolant within 5 % of i0 of the exponential, was a ramp 0.5·i0 off at mid-step) | Return early in `plan_fast_samples_` (both schedulers) |
| Is a source corner without a state change an event sample? | Yes, on both schedulers: the `auto` scheduler now records the state before a fired gate edge (it never modifies x, so it is x) — the sample reads its sources' left limit and gets a right-limit row | `test_a_source_edge_is_two_sided` (dsed / exact): the 50 µs sample reads 10 V, the ulp sample 0 V, the capacitor continuous and at the closed form 10(1 − e⁻⁵) | Pass nullptr again in `scheduler_auto.hpp` |

Cells moved (dsed, strict gate, old → new): chopper_with_rc_snubber 0.0146 → 2.9e-5,
flying_capacitor_3level 0.050 → 3.7e-5, full_bridge_dc 0.0815 → 2.7e-5, ll06 0.0918 → 1.7e-4,
ll07 0.126 → 1.8e-4, ll09 0.0692 → 1.7e-4, ll10 0.0465 → 1.3e-4; all seven freezes dropped.
Totals 65 / 7 / 0 / 34 / 2 → 72 / 0 / 0 / 34 / 2. Against pwl at dt/40 the seven read
0.003 … 0.020: that is pwl's own error — pwl against the same trbdf2 reference converges at
half order in this measure (ll06 and full_bridge_dc alike: dt/10 0.031, dt/40 0.016, dt/160
0.0079; its sample at an edge is the right limit and its trace ramps the jump over the previous
step), while dsed sits at 1.7e-4 / 2.7e-5 at the netlist's dt. pwl and trbdf2 cells are
unchanged (their results record no right limits; bit identity below).

## The refusal table by the extractor's own reason (2026-11-17)

The bridge refusal now leads with the C++ extractor's reason (it sat past the triage's
600-character cut, behind the generic "floating capacitors / nonlinear devices" text, which
was true of neither), and `exact_triage.py cause()` classifies it. `probes/exact_triage.out`
after the step above (34 refusals): 10 static circuits ("no dynamic elements"), 7 coupled
inductors, 6 `step_observer` (correct), 6 `G_aa` singular, 3 MOSFET level 1 + 2 Shockley
(correct). The previous agent's draft split (10 / 7 / 6) is confirmed cell by cell.

## Static circuits (2026-11-17)

| Question | Option taken | Evidence | How to reverse |
|---|---|---|---|
| A mode with no capacitor and no inductor | `compute_lti_state_space` returns an EMPTY state: A 0×0, recovery x = −G⁻¹(b + b_extra(t)) through the same sparse LU as the structural path (a singular G is refused with the `G_aa` message); `make_exact_lti` marks the empty stepper tier 1 (the step is the identity on nothing). No state is faked: the scheduler, its event location (gate edges, source corners, diode predicates read through the recovery map; a sine source's oscillator pair is still augmented, so its zero crossings are located on the exact trajectory) and the mode guard are unchanged | Closed forms, `test_exact_static_circuits.py` (dsed / exact): divider 7.5 V to 1e-12; gated resistor on / off levels to 1e-12 with the 30 µs edge stored left (on) and its right limit off; sine → R → diode: samples within 1e-9 of the piecewise closed form and events within 1e-9 of 10 / 20 / 30 ms. C++ `compute_lti_state_space gives a static circuit an empty state` (was: rejects it) | Restore the throw at `n_state == 0` |

Cells moved (dsed, strict gate, refused → agree): back_to_back_diodes_sine 1.1e-5,
diode_clamp_overvoltage 6.0e-12, bridge_resistive_pure 1.3e-13, current_divider_3way 0,
voltage_divider_3stage 0, transformer_step_up 5.1e-6, vcswitch_resistor 3.2e-6,
cascaded_h_bridge_5level 5.7e-5, mmc_4cell_chain 5.7e-5,
electrothermal_resistor_self_heating 0. Against pwl at dt/40: the sine cells 1.3e-8; the
switched ones 3.6e-3 … 4.3e-3, pwl's own edge ramp (step 1).
