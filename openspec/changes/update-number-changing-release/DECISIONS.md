# DECISIONS: update-number-changing-release

## 1. The exact route's output grid without dt (`output_rtol`), 2026-11-19

Branch `feat/core-auto-route-exact` (from `feat/core-release-rehearsal` bcadf4cd, merged with
`feat/core-integration` 061e4099). It closes blockers 1.1-1.3 and owner item 2.5.

### Causes, found by trace

- **1.1 `ll06_buck_vcswitch`** (exact engine, no dt, `exact` 1.1e-2):
  - The event trace showed the D1 turn-off right limit was already projected: `I(L1)` went from +4.8e-3 to -4.77e-3 A, and `V(sw)` from 0 to 47.6 V, in zero time.
  - `project_` cuts every mode faster than `100/dt_max`. With the default `dt_max` of 10 us, that removed a real ~100 ns transient.
  - Proof: `enable_event_projection=False` scores 8.9e-12.
- **1.2 `cl_buck_aggressive_pi`** (strict 3.95e-3): no engine defect.
  - Two independent closed-loop references agree within 8.4e-6: trbdf2 at rtol 1e-9 with `control_tick='state'`, and pwl with split at dt 2e-8.
  - The dsed samples agree with both to 1.2e-6.
  - The strict failure came from the 10 us output grid alone.
- **1.3** The 10 us grid between exact samples.

### The rule

Applies only with `simulate(engine='exact', output_rtol=r)` and no `dt`, `store_every` or recorders.

- **Event instants.** Every event instant is a sample, with both limits as before.
- **Step cut (chord bound).** Each exact step is cut so that the chord bound δ²·max|ẍ|/8 ≤ tol_j holds for every state j. The bound uses the mode's closed form:
  - ẍ(τ) = V e^{Λτ} V⁻¹ ẍ(0);
  - two rigorous bounds are computed, and the cancellation-preserving one is solved by bisection in log δ;
  - oscillator (sine) states and PWL source slopes are included.
- **Tolerance.** tol_j = r · max(range of state j so far, 1e-3·its peak, atol).
  - The range is used, not |x|: a ripple on a 24 V level is what a plot reads. `ll07_boost_vcswitch` disagreed at 5.2e-3 while the scale was |x|.
- **Event projection.** With the rule on, the projection removes only the parasitic cluster, 1/(1e-6·resolved scale), instead of everything faster than 100/dt_max.
- **Post-event points.** The geometric post-event points are the existing `exact_fast_samples` plan.
- **Cap.** 4096 cut steps per interval. Hits are counted in `diagnostics['output_grid']['n_output_capped']`.
- **Unbounded steps.** Steps without a tier-1 basis are counted in `n_output_unbounded` and keep the dt_max grid.
- **Warnings.** Both counters raise a RuntimeWarning.
- **Refused by name:** on pwl and trbdf2, with `integrator='bdf2'`, with sensitivities, and on the non-native fallback.

### The exact-zero root (`boost_classic` / `textbook_erickson_boost_ccm` under the rule)

**Symptom.** At `output_rtol=1e-4`, both runs were refused with DiodeModeInconsistentError: Drect ON carrying -8.7e-2 A at 1.06e-3 s.

**Trace.** I added temporary fprintf instrumentation, since removed:
- `locate_root_bracket` hit an exact 0 of Drect's ON current at t = 1.0594845721750604e-3 and returned that instant.
- The OFF mode read the same state as +9.98e-8 V forward. Its bound was 4.6e-14, plus 7.97e-8 of root error.
- The zero-band ladder therefore kept the diode ON.
- The two runs reached the root from different step starts. Their states differed by 5e-13 A, which is rounding.

**Fix.** An exact 0 carries no sign, so it now stays on ta's side, and the returned state is past the crossing, as the function's contract says.
- The fix is behind `zero_is_root=false`, which the exact path passes only under the output rule.
- With it on unconditionally, 19 of the 111 benchmark circuits change their default samples (byte compare). That change is left to the release flip.

### Profile value: 1e-5

Rehearsal `auto+next` against `auto` (trbdf2 today), using the tool's own `judge`:

| `output_rtol` | outcomes (circuits) | better | worse | worse that still agree |
|---|---|---:|---:|---:|
| 1e-4 | 106 / 2 / 1 | 47 | 30 | 29 |
| 1e-5 | 106 / 2 / 1 | 73 | 6 | 5 |

None of the 6 remaining worse cells comes from the output grid. The strict error is the same at 1e-4 and 1e-5:

- **`cascaded_h_bridge_5level`, `mmc_4cell_chain`, `back_to_back_diodes_sine`:**
  - the trbdf2 run is the harness reference, so `auto` scores 0;
  - the exact samples are 1e-13 to 2e-16 off the truth.
- **`ll06_buck_vcswitch` and `ll09_half_bridge_inverter_leg`:**
  - the strict error against the trbdf2 reference is 1.7e-4, below the tolerance (1e-3);
  - against the truth the exact samples are 8.8e-12 and 4.8e-11;
  - the error is the reference's.
- **`thermal/electrothermal_resistor_self_heating`:**
  - disagrees at 1.78e-3 with `exact` 0, the same as `auto:route` without the rule;
  - cause: the trbdf2 reference starts V(vin) at 0 (the uic left limit) and ramps to 5 V by 2e-5 s, and the 1 us samples of the opaque switch_fn poll put that ramp inside the gate window;
  - suggested fix: the reference reads the right limit at t=0, or the gate excludes the t=0 limit;
  - the harness gate was not loosened.

Under `dsed:output_grid` / `dsed+next` (netlist dt), the counts are unchanged at 100/0/0/9/0.

**Cost.** Measured on the 100 circuits `auto+next` routes to dsed, rule alone, wall time best of 2, all 100 run, cap hit 0 times, unbounded steps 0:

| `output_rtol` | samples | wall |
|---|---|---|
| off | 2,982,860 | 1.97 s |
| 1e-4 | 3,128,984 (+4.9 %) | 2.41 s |
| 1e-5 | 3,600,621 (+20.7 %) | 3.44 s |

The biggest sample ratios at 1e-4 were lcc ×120, t_type ×23 and steinmetz ×21.

**Rehearsal regeneration.** The cell signature hashes the whole profile, so every cell was stale after the value change.
- Only the configurations that read `output_rtol` were re-run: `auto+next`, `dsed:output_grid`, `dsed+next`.
- All other cells were re-signed from the run on this branch's code.
- This is valid because the rule is off in them: the dsed default and dt runs compare byte-equal, 222/222, base 9640041a against this tip.

### Functions changed in `core/include/pulsim/dsed/`

- `exact_lti.hpp`: new `exact_chord_step`.
- `builder_adapter.hpp`: new `BuilderAdapter::exact_output_step`.
- `event_predictor.hpp`: `locate_root_bracket` gains the parameter `zero_is_root` (default `true`, today's behaviour).
- `scheduler.hpp`, no code moved or reformatted:
  - new `set_output_rule`, `n_output_capped`, `n_output_unbounded`, and `output_step_` with its members (after `enable_event_projection_`);
  - the exact-path call site in `simulate()` (after the `fast_t_` clamp) and the run-start reset;
  - the `project_` threshold under the rule;
  - the two `locate_root_bracket` calls in `locate_event_exact_` pass `output_rtol_ == 0`.

Outside dsed:
- `sim/options.hpp`: the `output_rtol` row.
- `python/dsed_bindings.cpp`: `run_ped_native_builder` gains `output_rtol` and `output_cap`.
- Python files: `__init__.py`, `_plan.py`, `_dsed_dispatch.py`.

`loss_port.hpp` and the snapshot payload are untouched.

### Left open (reviewer items not done)

- `exact_output_step` and its `fits` allocate per step: there is a per-step augmented vector, and an `advance_` for the interval end. This is part of the +75 % wall time at 1e-5.
- Tier-2 (Padé) segments are not bounded. They keep the dt_max grid and are counted as unbounded.
- The tolerance is per state, not per observable. An observable that is a difference of large states gets that absolute error.
