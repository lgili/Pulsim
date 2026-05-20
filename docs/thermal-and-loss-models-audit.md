# Thermal + Loss Models — Comprehensive Audit

**Date**: 2026-05-20
**Scope**: every Pulsim device class — semicondutores, passivos, magnéticos, motores —
plus the unified electrothermal pipeline.
**Goal**: producible roadmap for getting to PSIM / PLECS-grade loss & temperature
fidelity. The audit was run by 4 parallel sub-agents reading the actual source
end-to-end; line-number citations are inline.

---

## TL;DR

| Domain | State | Top issue |
|---|---|---|
| 🟡 Semiconductors (MOSFET, IGBT, Diode) | partially complete | MOSFET/IGBT have accumulators + closed-loop scaling; **`T_j_` feedback into `Rds_on_at_Tj()` is OPEN-LOOP** — the device's internal junction temp never updates during the sim. VCSwitch + IdealSwitch lie via `has_loss_model=true` with zero instrumentation. |
| 🟡 Passives (R, L, C) | conduction-loss only | Accumulators exist + work, but `has_thermal_model=false` on all three → they don't participate in the thermal service. ESR(f), skin/proximity, dielectric loss all missing. |
| 🔴 Magnetics (Transformer, HysteresisInductor, SaturableTransformer) | mostly dead-code | `magnetic/bh_curve.hpp::steinmetz_loss_density` + `iGSE` helpers exist but are **never called by any device**. Hysteresis loop is integrated for telemetry only — its area is never converted to power. Transformer model is purely ideal (no R_pri / R_sec / leakage). |
| 🔴 Motors (DC, PMSM, BLDC, IM, PSC) | **zero loss accumulators** | `R_s_tc`, `T_ref_winding`, `R_th_winding_to_ambient` are **dead code** — landed as params but never used by stamping or accumulation. Iron loss completely absent. Friction power not counted as loss. `*_steady_state_winding_temperature` accessor is the only loss-related API; user must supply `I_rms` from hand-computed post-processing. |
| 🟡 Unified pipeline | electrothermal scaling works for MOSFET+IGBT only | Walker dispatches `accumulate_loss` on {R, C, L, IdealDiode, MOSFET, IGBT}; thermal service tracks T(t) only for MOSFET+IGBT (`has_thermal_model=true`). All other categories are siloed: their losses are read but their temperature is never integrated. |

---

## 1. The most impactful bug — T_j feedback is OPEN-LOOP

There are **two separate `T` trackers** in the codebase that don't talk to each other:

1. **Device-internal `T_j_`** (`mosfet.hpp:823`, `igbt.hpp:632`, `ideal_diode.hpp:683`):
   - Initialized from `params.T_amb` at construction.
   - Only mutated by `set_T_j_init()`.
   - `Rds_on_at_Tj()` reads this `T_j_` to compute the temperature-corrected
     conduction resistance.
   - **The simulator never calls `set_T_j_init()` during a transient run.**
   - Net effect: `Rds_on_at_Tj()` returns its t=0 value for the whole simulation.

2. **`DefaultThermalService` state** (`transient_services.cpp:1201`):
   - Tracks `T_i(t) ← T_i + dt·(P·R_th − (T_i−Tamb))/τ` per accepted step.
   - Pushes `scale_i = clamp(1+α·(T_i−T_ref), 0.05, 4)` into the stamp via
     `circuit_.set_device_temperature_scales(scale_i)`.
   - Stamp uses `scale_i` to scale `kp`, `g_on`, `g_off` at
     `runtime_circuit.hpp:6645, 7367` (MOSFET) and `:6647` (IGBT).

So the **closed loop** flows: `P(t)` → `T_i(t)` → `scale_i(t)` → modified `g_on` /
`kp` in the next stamp. But the **device-side `T_j_`** (which the user reads via
`mosfet_junction_temperature(name)`) stays frozen at `T_amb`. The two diverge
silently. A warning at `runtime_circuit.hpp:849-865` acknowledges this.

**Fix** (Top 1 priority): at the end of each accepted step inside
`DefaultThermalService::commit_accepted_segment`, dispatch `set_T_j_init(T_i)`
on every device with `has_thermal_model=true`. This closes the loop AND makes
`mosfet_junction_temperature(name)` actually reflect the integrated state.

---

## 2. Per-domain findings

### 2.1 Semiconductors

| Device | Cond params | Switching params | Thermal params | Accumulator | Wired? | Circuit accessors |
|---|---|---|---|---|---|---|
| **MOSFET** `mosfet.hpp:35-133` | `g_on`, `g_off`, `Rds_on_tc=5e-3 1/K` (no explicit `Rds_on` — implied `1/g_on`) | `Eon_25`, `Eoff_25`, `I_ref=10A`, `V_ref=400V`, `Esw_tc=3e-3 1/K` | `R_th_ja=0`, `T_amb=25°C` | full ladder | ✅ `runtime_circuit.hpp:5001-5024` | full (`mosfet_total_energy`, `_average_power`, `_peak_power`, `_junction_temperature`, ...) |
| **IGBT** `igbt.hpp:32-119` | `g_on`, `g_off`, `v_ce_sat=1.5V`, `Rce=0.02Ω`, `Rce_tc=5e-3`, `V_ce_tc=2e-3 V/K` | `Eon_25`, `Eoff_25`, `I_ref=50A`, `V_ref=600V`, `Esw_tc=3e-3` | `R_th_ja=0`, `T_amb=25°C` | full ladder | ✅ `:5025-5048` | full |
| **IdealDiode** `ideal_diode.hpp:35-96` | `g_on=1e3`, `g_off=1e-9`, `V_F0=0`, `R_d=-1`, `V_F0_tc=-2e-3 V/K` | `Qrr=0`, `Erec_shape=0.5` — **different model** (no `Eon`/`Eoff`, no `I_ref`/`V_ref`, no `Esw_tc`) | `R_th_ja=25 K/W` **(non-zero default, divergent)**, `T_amb=25°C` | full ladder (uses `was_conducting_` instead of `was_on_`) | ✅ `:4979-5000` (BUT `accumulate_loss(v_diode, dt)` has **no `is_on` arg** → cannot honor forced switch state) | full |
| **VoltageControlledSwitch** | `v_threshold`, `g_on`, `g_off`, `hysteresis` | NONE | NONE | NONE | ❌ no dispatch branch | NONE — `has_loss_model=true` is a **lie** |
| **IdealSwitch** | `g_on`, `g_off`, `initial_state` | NONE | NONE | NONE | ❌ | NONE — same lie |

#### Issues
- **IdealDiode reverse-recovery model is dimensionally inconsistent with MOSFET/IGBT.**
  Diode uses `E = Qrr·V_r·shape`; FETs use `E = E_25·(I/I_ref)·(V/V_ref)·tc_factor`.
  No temperature dependence on diode Qrr. No current scaling. Datasheet importers must special-case.
- **Body diode (MOSFET) + antiparallel diode (IGBT) losses are folded into the channel accumulator** — no separate `e_body_diode_` / `e_freewheel_`. Once `body_diode_enable=true` the diode current is summed with channel current and the loss bucket is wrong.
- **No gate-drive loss** anywhere — Q_g·V_gs·f_sw missing on both MOSFET and IGBT.
- **No per-event breakdown** — `e_sw_` is one bucket for `E_on + E_off` (MOSFET/IGBT) or `Qrr·V_r·shape` (diode). Cannot distinguish turn-on, turn-off, recovery.
- **No IGBT tail-current model** — comment at `igbt.hpp:19` admits the deferral.
- **`device_traits::has_thermal_model` is `false` on IdealDiode** even though IdealDiode has `R_th_ja`/`T_amb`/`junction_temperature()` — the trait is inverted, so diodes never enter the closed electrothermal loop.

### 2.2 Passives

| Device | DC loss | AC/freq loss | Thermal | Accum wired? | Accessor | Closed loop |
|---|---|---|---|---|---|---|
| **Resistor** `resistor.hpp:28-42` | ✅ `resistance`, `TCR`, `T_ref` | ❌ | ✅ `R_th_ja`, `T_amb` | ✅ `:4828-4830` | ✅ `resistor_*` | ❌ `has_thermal_model=false` |
| **Inductor** `inductor.hpp:23-44` | ✅ `DCR`, `DCR_tc`, `T_ref` | ❌ no skin, no Steinmetz | ⚠ winding-only `R_th_ja` | ✅ Cu only `:4812-4814` | ✅ `inductor_*` | ❌ |
| **Capacitor** `capacitor.hpp:24-48` | ✅ `ESR`, `ESR_tc`, `T_ref` | ❌ no ESR(f), no tan-δ | ✅ `R_th_ja`, `T_amb` | ✅ `:4786-4788` | ✅ `capacitor_*` | ❌ |
| **Transformer (ideal)** | ❌ no R_pri / R_sec | ❌ | ❌ | ❌ no `accumulate_loss` | ❌ | ❌ |

#### Asymmetries (same concept, three names)
- Temperature coefficient: `TCR` (Resistor) / `DCR_tc` (Inductor) / `ESR_tc` (Capacitor).
- Method name: `R_at_Tj()` / `DCR_at_Tj()` / `ESR_at_Tj()`.
- `reset_loss` doesn't zero `v_last_`/`i_last_` consistently across the three.

### 2.3 Magnetics (mostly dead code)

| Device | DC loss | AC/freq loss | Thermal | Wired? |
|---|---|---|---|---|
| **HysteresisInductorDevice** `hysteresis_inductor_device.hpp:44-57` | ❌ no DCR field | ⚠ Jiles-Atherton loop integrated, **area never converted to P_hyst** | ❌ | `update_history_impl` is empty `:139-143` |
| **`magnetic::SaturableInductor`** | ❌ math object only, not in `DeviceVariant` | ❌ | ❌ | ❌ |
| **`magnetic::SaturableTransformer`** | ❌ no R_w per winding | ❌ | ❌ | ❌ |
| **`magnetic::bh_curve.hpp` Steinmetz / iGSE helpers** | n/a | helpers EXIST `:205-280` | n/a | **never called from any device** — only used by `test_magnetic_phase{1,6}*.cpp` |
| **`CatalogCore::steinmetz`** in YAML `core_catalog.hpp:66-67` | n/a | parsed from YAML | n/a | **value is discarded** — no factory wires it into a device |

#### What PSIM / PLECS have that we don't
- Capacitor ESR-vs-frequency LUT (PLECS 1-D table), tan δ dielectric loss, lifetime/L₀ + Arrhenius wear-out.
- Inductor: AC winding R (skin + proximity F_R(f)), Steinmetz core loss wired to peak B from flux-linkage history.
- Transformer: R_pri / R_sec copper resistance, leakage inductances, Steinmetz core loss, inter-winding capacitance. Pulsim's `Transformer.hpp` Params even declares `L_m` but `stamp_impl` ignores it.
- Cauer / Foster 2-node thermal network (R_jc + R_ca + C_th_j + C_th_c) per device. Pulsim has only flat junction-to-ambient.
- Per-device exposed B(t), H(t), and B_peak waveforms for downstream Steinmetz post-processing.

### 2.4 Motors (the biggest hole)

| Motor | Cu loss params | Iron loss | Thermal | Accumulator | API | Closed loop |
|---|---|---|---|---|---|---|
| **DC** `dc_motor.hpp:25-53` | `R_a`, `R_s_tc`, `T_ref_winding` **(dead — never stamped at `:99`)** | ❌ | `R_th_winding_to_ambient`, `T_amb` | **none** — no `e_copper_`, no `T_winding_(t)` | only `dc_motor_steady_state_winding_temperature(name, i_a_rms)` `runtime_circuit.hpp:3932` (closed-form, user-supplied I_rms) | ❌ `has_thermal_model=false` |
| **PMSM** `pmsm.hpp:32-63` | `Rs`, `R_s_tc`, `T_ref_winding` **(dead at `pmsm_device.hpp:204`)** | ❌ | same | **none** | `pmsm_steady_state_winding_temperature` only | ❌ |
| **BLDC** `bldc_motor.hpp:71-100` | `R_s`, `R_s_tc`, `T_ref_winding` **(dead at `:246`)** | ❌ | same | **none** | `bldc_motor_steady_state_winding_temperature` only | ❌ |
| **Induction (3φ)** `induction_motor.hpp:76-100` | `R_s` stamped (T-independent), `R_r` (not accumulated), `R_s_tc` dead | ❌ Krause αβ is lossless by construction | same | **none** | `induction_motor_steady_state_winding_temperature` only | ❌ |
| **Single-φ IM PSC** `single_phase_induction_motor.hpp:61-88` | `R_s_main`, `R_s_aux`, `R_r`, shared `R_s_tc` (all dead) | ❌ | shared `R_th_winding_to_ambient` | **none** | `single_phase_induction_motor_steady_state_winding_temperature` only | ❌ |

#### Critical motor gaps (impact-ordered)

1. **No iron / core loss anywhere.** Largest fidelity miss. For a refrigeration compressor (PSC, steady 60 Hz, light-load operation), iron loss is ≥ copper at part-load — efficiency reports out 5-10 percentage points too high. For EV traction PMSM at high speed, hysteresis + eddy in the stator stack often exceeds copper. PSIM exposes `K_h·f·B^α + K_e·f²·B²` (Steinmetz) on every machine; PLECS exposes iron loss on a separate output node. Pulsim has **zero hooks**.
2. **`R_s_tc` / `T_ref_winding` are dead code.** Declared on every motor Params, never read by the stamp. Only path to T-dependent R is to re-create the motor with a hand-computed R after each `_steady_state_winding_temperature` call.
3. **No loss accumulators at all.** Even `e_copper_` doesn't exist. `efficiency = P_mech / P_elec` is impossible from a single simulation run.
4. **No `T_winding_(t)` ODE state.** Only a closed-form quiescent solve. Transient thermal protection or cycling-fatigue studies are blocked.
5. **Friction power is NOT a loss.** `b·ω + τ_C·sign(ω) + tau_load_quad·|ω|³` is subtracted from `τ_em` in every motor's `advance_state` (`dc_motor_device.hpp:131-135`, `pmsm_device.hpp:226-231`, etc.) but `P_fric = friction_torque · ω` is never integrated. Affects efficiency calc even after the above gaps are closed.
6. **No multi-mass thermal network.** Hermetic compressors need at least winding → frame → ambient. Currently single-resistance lumped.

#### Motor field-name inconsistencies
- Stator/armature R: `R_a` (DC) / `Rs` (PMSM) / `R_s` (BLDC, IM) / `R_s_main` + `R_s_aux` (PSC). Five spellings.
- Viscous friction: `b` (DC) / `b_friction` (all others).
- Back-EMF: `K_e` + `K_t` (DC) / `K_e_peak` (BLDC) / `psi_pm` (PMSM). No common nomenclature.
- `friction_coulomb` default is `0` everywhere except single-φ IM (which defaults to 0.05). Surprise during sweeps.
- 3φ IM `slip(omega_sync_electrical)` vs PSC `slip(omega_sync_mechanical)` — same name, different units.

---

## 3. Unified pipeline issues

### Pipeline (per accepted step)

```
Newton solve x_n  (stamp uses scale_i = clamp(1+α(T_i−T_ref), 0.05, 4)
                   in MOSFET/IGBT.kp,g_on,g_off — runtime_circuit.hpp:6645, 7367)
       │
       ▼
process_accepted_step_events   ← turn-on/off edges
       │
       ▼
accumulate_conduction_losses(x_n, dt)            simulation_step.cpp:394
   └─▶ assemble_state_space walker per device     runtime_circuit.hpp:4770-5049
        dev.accumulate_loss(...)  ← {R, C, L, IdealDiode, MOSFET, IGBT}
   └─▶ DefaultLossService::commit_accepted_segment   transient_services.cpp:893
        reads dev.last_power() into last_device_power_[i]
       │
       ▼
update_thermal_state(dt)                          simulation_step.cpp:404
   └─▶ DefaultThermalService::commit_accepted_segment  transient_services.cpp:1201
        T_i ← T_i + dt·(P·R_th − (T_i−Tamb))/τ        (only for has_thermal_model=true)
        refresh_scales() → circuit_.set_device_temperature_scales(scale_i)
   └─▶ ⚠ NO dev.set_T_j_init(T_i) — device-side T_j_ stays at T_amb
       │
       ▼
t += dt; circuit_.update_history(x_n)
```

### Rough edges in the public API
- **`LossResult.rms_current` field exists but is left at 0** by `finalize()` (`losses.hpp:179`). Users must integrate `result.states` themselves. The `*_steady_state_winding_temperature` accessors take `i_rms` as an *input* argument instead of computing it.
- **`junction_temperature()` returns a stale value** for the entire sim (see §1). User-visible inconsistency with `thermal_summary.device_temperatures`.
- **`options.enable_losses=false` doesn't gate device accumulators.** Walker always calls `dev.accumulate_loss(...)`. Only `DefaultLossService` aggregation respects the flag. So `dev.average_power()` is non-zero with losses "disabled".
- **Switching-energy is duplicated source-of-truth.** `options.switching_energy` map + device-internal `Eon_25/Eoff_25/Qrr` params — `transient_services.cpp:1052-1059` falls back to one if the other is zero, but both paths can fire and double-count.
- **YAML thermal enabling restricted to MOSFET/IGBT/BJT only** (`yaml_parser.cpp:88-93`). Diode/R/L/C `R_th_ja` set in C++ have no YAML route.
- **`compute_totals()` doesn't populate `input_power / output_power`** — efficiency is silently 0 unless the user post-processes V·I (`transient_services.cpp:1097-1099`).
- **`thermal_scale_` is silently clamped to `[0.05, 4.0]`** — no warning for extreme α·ΔT.

---

## 4. Cross-cutting uniformity issues

| Concept | MOSFET | IGBT | IdealDiode | Resistor | Inductor | Capacitor | DC motor | PMSM | BLDC | IM | PSC |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Conduction resistance | `1/g_on` | `1/g_on` + `Rce` | `R_d` | `resistance` | `DCR` | `ESR` | `R_a` | `Rs` | `R_s` | `R_s` | `R_s_main` / `R_s_aux` |
| Temperature coefficient | `Rds_on_tc` | `Rce_tc`+`V_ce_tc` | `V_F0_tc` | `TCR` | `DCR_tc` | `ESR_tc` | `R_s_tc` (dead) | `R_s_tc` (dead) | `R_s_tc` (dead) | `R_s_tc` (dead) | `R_s_tc` (dead) |
| T_ref field | `T_ref` | `T_ref` | `T_ref` | `T_ref` | `T_ref` | `T_ref` | `T_ref_winding` | `T_ref_winding` | `T_ref_winding` | `T_ref_winding` | `T_ref_winding` |
| `accumulate_loss(...)` signature | `(v_ds, dt, is_on)` | `(v_ce, dt, is_on)` | `(v_diode, dt)` ⚠ no `is_on` | `(v_res, dt)` | `(v_ind, dt)` | `(v_cap, dt)` | ❌ | ❌ | ❌ | ❌ | ❌ |
| `was_*_` transition flag | `was_on_` | `was_on_` | `was_conducting_` | n/a | n/a | n/a | ❌ | ❌ | ❌ | ❌ | ❌ |
| `has_thermal_model` | ✅ true | ✅ true | ❌ false (despite having T_j) | ❌ false | ❌ false | ❌ false | ❌ false | ❌ false | ❌ false | ❌ false | ❌ false |
| `has_loss_model` | ✅ true | ✅ true | ✅ true | ✅ true | ✅ true | ✅ true | ❌ false | ❌ false | ❌ false | ❌ false | ❌ false |

Five naming buckets for "stator resistance", three for "temperature coefficient",
two for "transition flag", and two different sigs for `accumulate_loss`.

---

## 5. Top 10 priority improvements

Ordered by impact for a user who relies on thermal + loss heavily:

### 🔴 Priority 1 — Close the T_j feedback loop
At the end of `DefaultThermalService::commit_accepted_segment`
(`transient_services.cpp:1201`), dispatch `set_T_j_init(T_i)` on every device
with `has_thermal_model=true`. Today `Rds_on_at_Tj()` reads a stale `T_amb` for
the whole sim. **Single highest-ROI change in the whole audit.**

### 🔴 Priority 2 — Promote `has_thermal_model = true` on IdealDiode, R, L, C
All four have working accumulators that compute `last_power()` but
`has_thermal_model=false` keeps them out of the thermal service. One-line
trait flips per device + YAML support beyond `mosfet/igbt/bjt_*`
(`yaml_parser.cpp:88-93`).

### 🔴 Priority 3 — Wire iron / core loss
Inductor + HysteresisInductorDevice are screaming for this. Add `core_volume`,
`A_e`, `l_e`, `turns`, `SteinmetzLoss{k, α, β}` params (Steinmetz already
parsed from YAML at `core_catalog.hpp:66`, just discarded). Compute
`B_peak = λ_peak / (N · A_e)` from the flux-linkage history, call
`magnetic/bh_curve.hpp::steinmetz_loss_density(B_peak, f) · V_e`, integrate
into `e_iron_`. Promote `LossBreakdown` (`losses.hpp:35-49`) to track
`iron / copper / dielectric` separately.

### 🔴 Priority 4 — Motor loss + thermal accumulators
- Add `e_copper_`, `e_iron_`, `e_mech_friction_`, `T_winding_`, `t_sim_`,
  `p_peak_*` accumulators + `T_winding_(t)` ODE state to each motor's
  device wrapper.
- Stamp temperature-dependent `R_s(T_winding_)` in the trapezoidal companion
  (uses the existing dead `R_s_tc`, `T_ref_winding`).
- Add Steinmetz iron-loss params for the 3φ machines + PSC.
- Specialize `device_traits<...>::has_thermal_model = true` for every motor.
- Expose `<motor>_total_energy`, `<motor>_copper_loss`, `<motor>_iron_loss`,
  `<motor>_mechanical_loss`, `<motor>_winding_temperature`, `<motor>_efficiency`
  on Circuit.

### 🟠 Priority 5 — Auto-compute `rms_current`, `avg_current`, `efficiency`
`LossResult.rms_current` is left at 0; `efficiency` is silently 0. Each
device's `accumulate_loss` has access to `last_current()` — integrate I² and
|I| online and feed `LossResult::finalize`. Add a real `efficiency = (Σ
P_out) / (Σ P_in)` accumulator with terminal V·I.

### 🟠 Priority 6 — Add VCSwitch + IdealSwitch loss instrumentation
Both declare `has_loss_model=true` (`voltage_controlled_switch.hpp:280`,
`ideal_switch.hpp:103`) but have no accumulator. Either implement the
standard ladder (mirror MOSFET's pattern) or flip the trait to `false` and
update the catalog manifest. The current state is misleading API.

### 🟠 Priority 7 — Track body diode + antiparallel diode losses separately
When `MOSFET::body_diode_enable=true` (or IGBT antiparallel), the diode
current is summed with channel current and the loss bucket loses sight of
the freewheel path. Add `e_body_diode_` (MOSFET) / `e_freewheel_` (IGBT)
sub-accumulators.

### 🟡 Priority 8 — Unify `accumulate_loss` signature + naming
- All semicondutores take `(v_xx, dt, is_on)` — including IdealDiode (today it
  has no `is_on`, so forced-state overrides are silently dropped).
- Pick one name: `R_s_tc` for stator/armature resistance TC, `R_tc`-style for
  passives. Migrate `TCR` / `DCR_tc` / `ESR_tc` / `Rds_on_tc` / `Rce_tc` to a
  single convention.
- Single transition flag name: `was_on_` (drop `was_conducting_`).
- Single stator-R name across motors: `R_s` (deprecate `R_a`, `Rs`,
  `R_s_main`/`R_s_aux` → keep aux as a separate winding override).

### 🟡 Priority 9 — Cauer 2-node thermal network
Add `R_th_jc + R_th_ca + C_th_j + C_th_c` to MOSFET/IGBT/Diode Params.
Mirror PSIM's "Thermal Network" block + PLECS's Cauer/Foster export. The
flat `R_th_ja` single-resistance is fine for a quick sizing pass but blocks
transient-thermal-protection studies.

### 🟡 Priority 10 — Mechanical friction-as-loss + gate-drive loss
- `P_fric = (b·ω + τ_C·sign(ω) + tau_load_quad·|ω|³) · |ω|` integrated into
  `e_mech_loss_` on each motor.
- `P_gate = Q_g · V_gs · f_sw` added to MOSFET / IGBT accumulators (param
  `Q_g` default 0 to keep back-compat).

---

## 6. Suggested OpenSpec scaffolding

Recommended split into 3 OpenSpec changes (in priority order):

1. **`close-electrothermal-loop-and-promote-thermal-traits`** — Priorities 1+2
   above. Small, surgical, high ROI. Touches `transient_services.cpp` +
   `device_traits` specializations + YAML parser. Should land first; it
   unlocks accurate T_j-dependent loss reporting for everything currently
   wired.

2. **`motor-loss-thermal-pipeline`** — Priority 4 + parts of 5. Adds copper /
   iron / friction accumulators + closed-form + ODE thermal to all 5 motor
   families. Largest LOC count, biggest user-visible value for compressor +
   EV / drive workflows.

3. **`magnetics-iron-loss-and-unified-naming`** — Priority 3 + 8. Wires
   Steinmetz into Inductor + HysteresisInductor + (future) SaturableTransformer.
   Unifies `accumulate_loss` signatures and field naming as a back-compat
   alias layer (old names still resolve via deprecated typedefs).

Priorities 5 (rms/efficiency), 6 (switches loss), 7 (body-diode split), 9
(Cauer 2-node), 10 (friction-as-loss + gate-drive) are best landed as
follow-ups inside (1)-(3) above rather than independent changes.

---

## 7. Files touched at a glance

Key reference files for any follow-up:
- `core/include/pulsim/v1/components/{mosfet,igbt,ideal_diode,voltage_controlled_switch,ideal_switch,resistor,inductor,capacitor,transformer,hysteresis_inductor_device}.hpp`
- `core/include/pulsim/v1/components/{dc_motor,pmsm,bldc_motor,induction_motor,single_phase_induction_motor,mechanical}_device.hpp`
- `core/include/pulsim/v1/motors/{dc_motor,pmsm,bldc_motor,induction_motor,single_phase_induction_motor,mechanical}.hpp`
- `core/include/pulsim/v1/magnetic/{bh_curve,saturable_inductor,saturable_transformer,hysteresis_inductor,core_catalog}.hpp`
- `core/include/pulsim/v1/runtime_circuit.hpp` — accessors `:825-1162` (passives + switches) and `:3915-3984` (motor steady-state)
- `core/include/pulsim/v1/simulation.hpp:259-281` — `ThermalDeviceConfig`
- `core/include/pulsim/v1/losses.hpp:35-193` — `LossBreakdown` + `LossResult`
- `core/src/v1/transient_services.cpp:893-1208` — `DefaultLossService` +
  `DefaultThermalService` walkers
- `core/src/v1/simulation_step.cpp:394-495` — per-step loss/thermal dispatch
- `core/src/v1/yaml_parser.cpp:88-93` — YAML thermal enable
