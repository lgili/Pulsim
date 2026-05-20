# Design — `close-electrothermal-loop-and-promote-thermal-traits`

## Two trackers, one closed loop

The audit (`docs/thermal-and-loss-models-audit.md` § 1) identified that
Pulsim has two parallel `T` trackers that don't talk to each other:

```
                ┌──────────────────────────────┐
                │  Newton solve step k         │
                │  (uses scale_i in stamp)     │
                └──────────────────────────────┘
                              │
                              ▼
                ┌──────────────────────────────┐
   dev.last_power()  ◀──── │  device accumulators         │
   dev.junction_temperature() ◀│  device.T_j_ ← static T_amb  │
                              │  (never updated)             │
                              └──────────────────────────────┘
                              │
                              ▼
                ┌──────────────────────────────┐
                │  DefaultThermalService        │
                │  T_i(t) ← T_i + dt·... ODE   │
                │  scale_i = 1 + α·(T_i−T_ref) │
                └──────────────────────────────┘
                              │
                              ▼
                ┌──────────────────────────────┐
                │ circuit_.set_device_temp_    │
                │  scales(scale_i)             │
                └──────────────────────────────┘
                              │  (feeds back into stamp)
                              └──────────────────────────────┘
```

The integrated `T_i` flows back into the **stamp** via `scale_i`, but
not into the **device's own `T_j_`**. Two consequences:

1. `dev.junction_temperature()` lies — returns `T_amb`, not `T_i(t)`.
2. `Rds_on_at_Tj()` (used in `accumulate_loss` to compute power
   correctly when `R_th_ja > 0`) reads the stale `T_j_`, so the
   conduction loss is computed at T_amb-anchored resistance instead of
   T_i(t)-anchored resistance. The closed loop is correct in the
   STAMP path (via scale_i) but wrong in the LOSS path (via T_j_).

## The fix

Add a second `std::visit` walker inside
`DefaultThermalService::commit_accepted_segment`, immediately after
`refresh_scales`. The walker dispatches `set_T_j_init(T_i)` on every
device that has `has_thermal_model = true`. Detection via concept:

```cpp
template <typename T>
concept HasThermalInit = requires(T& t, Real v) {
    { t.set_T_j_init(v) } -> std::same_as<void>;
};
```

Devices that don't satisfy the concept (motors, magnetics, sources)
compile away to no-op. Future motor/magnetics thermal pipelines will
add `set_T_j_init` and join the loop without re-touching this
dispatch.

The walker iterates `circuit_.devices()` in index order, indexes into
the existing `thermal_states_[i].temperature`, and calls the setter.
Reuses the same `device_index → thermal_state_index` mapping that
`refresh_scales` already uses.

## Why the trait flip on Diode / R / L / C

All four expose:
- `R_th_ja`, `T_amb`, `T_j_` private members
- `set_T_j_init`, `junction_temperature()`,
  `steady_state_junction_temperature()` public methods
- `accumulate_loss(v, dt)` that integrates conduction energy with
  T_j-corrected resistance via `R_at_Tj()` / `DCR_at_Tj()` /
  `ESR_at_Tj()` / `V_F0_at_Tj()`

But `device_traits<X>::has_thermal_model = false` (see e.g.
`ideal_diode.hpp:700`, `resistor.hpp:185`, `inductor.hpp:231`,
`capacitor.hpp:246`). The trait gates `DefaultThermalService::reset`
(`transient_services.cpp:1158-1162`) and decides whether the device
enters the closed loop. Today the trait is wrong for these four — the
machinery is built but unused. Flipping it joins them to the loop with
zero new physics.

## Why no behaviour change on `R_th_ja = 0`

The `DefaultThermalService::commit_accepted_segment` Euler update is
`T_i ← T_i + dt·(P·R_th − (T_i − T_amb))/τ`. With `R_th = 0`, the
update collapses to `T_i ← T_i + dt·(0 − (T_i − T_amb))/τ`. If
`T_i(0) = T_amb` (default initial state), the ODE keeps `T_i(t) =
T_amb` for all t. The closed-loop `set_T_j_init(T_amb)` is a no-op
relative to the device's construction state. Verified by inspection of
the existing flow.

## Why YAML expansion is in scope

Without YAML support for thermal config on Diode / R / L / C, the new
trait flips would only be reachable via the C++ API. Users that drive
their circuits from YAML (the majority for batch / sweep workflows)
would still see the gap. The YAML parser block is a 30-LOC additive
change that mirrors the existing `mosfet`/`igbt` blocks — no schema
break, just an allow-list expansion.

## Test strategy

The closure regression test
(`test_electrothermal_loop_closure.cpp`) drives a buck converter with
`R_th_ja = 1 K/W` and `T_amb = 25 °C` to steady state. Three
assertions guard the closure:

1. **Walks**: T_j(t_end) ≠ T_amb. Today fails — returns 25.0 for the
   whole sim.
2. **Matches predicted steady state**: T_j(t_end) within 5 % of
   `steady_state_junction_temperature()`. Today fails — they're both
   the same wrong constant.
3. **Per-step invariant**: `dev.junction_temperature() ==
   thermal_summary.device_temperatures[i]` within 1e-9 °C at every
   accepted step. Today both equal T_amb, so the assertion passes
   trivially. After closure, both equal T_i(t), still trivially true
   — but breakage of the dispatch would diverge them.

## Risks

1. **Existing electrothermal tests** that asserted T_j stays near
   T_amb will fail with the closure landed. Audit them and update
   tolerances. Identified set is small (~3 tests in `[thermal]` /
   `[regression]`).
2. **IdealDiode default `R_th_ja = 25 K/W`** (`ideal_diode.hpp:35-96`
   — the only device with a non-zero R_th_ja default) means
   trait-flipping the diode would cause every existing diode-using
   sim to start computing `T_j(t)` instead of seeing the constant
   `T_amb`. If the sim has any current through the diode, T_j will
   walk. For most tests this is invisible (T_j(t) → finite steady
   state slightly above T_amb) but for high-current rectifier
   benchmarks the T_j could climb 5-20 °C and the diode's R_d /
   V_F0 would track. Flagged for explicit audit in 1.2.1.
3. **Numerical drift on PWL state-space tests** — switching benchmarks
   measure end-of-period waveform shapes. If T_j now walks during the
   warm-up portion, the steady-state waveform might shift by a few
   percent on first-period KPIs. Tolerances likely need widening on
   the affected tests.

## What this proposal explicitly does NOT do

- **Does NOT add iron / core loss.** Steinmetz wiring is the scope of
  the follow-up `magnetics-iron-loss-and-unified-naming` OpenSpec.
- **Does NOT touch motors.** Motor thermal pipeline is the scope of
  the follow-up `motor-loss-thermal-pipeline` OpenSpec.
- **Does NOT change diode's `R_th_ja` default** from 25 K/W to 0. That
  behaviour change is flagged for a separate decision (it would
  improve uniformity at the cost of breaking every diode-using sim
  that implicitly relied on a non-trivial R_th).
- **Does NOT add Cauer / Foster multi-node thermal networks.** Still
  flat R_th_ja single-resistance.
- **Does NOT add per-event switching-loss breakdown** (turn-on vs
  turn-off vs Qrr buckets separately). Audit Priority 7 → future.
- **Does NOT add gate-drive loss** (Q_g · V_gs · f_sw). Audit
  Priority 10 → future.

These are scoped out by design to keep this change surgical: ~250-400
LOC modified, ~200-300 LOC added, ~1 day of focused work + 0.5 day of
test audit. Larger fidelity work follows in two clearly-separated
OpenSpecs.
