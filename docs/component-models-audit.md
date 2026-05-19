# Component Models Audit — Pulsim vs PSIM / PLECS

Date: 2026-05-18. Sources: full reads of every file in
`core/include/pulsim/v1/components/` (20 files, ~5400 LOC) plus the magnetic
helpers in `core/include/pulsim/v1/magnetic/` and the motor math objects in
`core/include/pulsim/v1/motors/`.

This audit answers three questions:

1. **Are our component models competitive with PSIM and PLECS?** — fidelity
   vs the commercial state of the art.
2. **Are they uniform among themselves?** — same `Params` POD shape, same
   `accumulate_loss` / thermal / switching-mode contract across the family.
3. **Where can we improve, especially for Newton convergence in extreme
   cases?** — concrete bug list with line numbers.

---

## TL;DR — Executive Summary

| Category | Status vs PSIM/PLECS | Confidence |
|----------|-----------------------|------------|
| **Passives (R, C, L)** | ✅ **Ahead** — TCR / ESR / DCR + thermal binding on every passive. PSIM stock has no thermal. | High |
| **MOSFET (PWL Ideal + auto-snubber)** | 🟢 **On par** — competes with PLECS Level-1, exceeds PSIM "ideal MOSFET". | Medium |
| **MOSFET (Behavioral / Shichman-Hodges)** | 🟡 **Behind** — teaching-level Level-1 model with `kp`/`λ`; no body diode; not datasheet-driven. | High |
| **IGBT** | 🔴 **Behind** — `V_CE_sat` is in the loss accumulator but **NOT in the stamp**. User sees 5 mV drop where datasheet says 1.5 V. | High |
| **IdealDiode (RealisticDiode)** | 🟢 **Matches PLECS, exceeds PSIM** — V_F0 + R_d Norton, Qrr reverse recovery, tiered C_j auto-default. | High |
| **Transformer** | 🔴 **Behind** — only ideal-N-turns. No Lm / leakage / saturation / losses. Saturable math exists in `magnetic/` but is not wired to the device wrapper. | High |
| **HysteresisInductor** | 🟡 **Half-done** — Jiles-Atherton math runs as a *side-channel state*, but `L_eff` in the MNA stamp is **constant**, so saturation isn't in the loop. | High |
| **Motors (DC, PMSM, BLDC, 3φ IM)** | 🟡 **On par** for the base models, but no winding thermal, no automatic shaft coupling, no nameplate-to-equivalent-circuit helpers, no DC field-winding variants. | High |
| **Single-phase PSC motor** | 🟡 **Pure PSC only** — no CSCR / centrifugal-switch variants, no symmetrical-components torque dip near synchronous speed. | Medium |
| **Auto-parasitics + snubber advisor** | 🟢 **Ahead** — `Circuit.auto_configure_parasitics()` is a clear differentiator vs PSIM (which leaves snubber sizing to the user). | High |

**Bottom line.** If we put Pulsim head-to-head with PSIM/PLECS on a realistic
350 W LLC converter + flyback + 3φ motor drive benchmark today, we'd win on
loss/thermal reporting and snubber automation, lose on **transformer
fidelity** (the #1 gap) and on **IGBT V_CE_sat in the stamp** (the #2 gap),
and tie on most everything else.

---

## 1. Switching Devices

Full agent report cross-checked against my own reading of the files. Source
line numbers cited inline.

### 1.1 MOSFET (`components/mosfet.hpp`, 717 LOC)

**Fidelity.** Two stamp paths:
- **PWL Ideal** (line ~644): `g_on` / `g_off` keyed by `pwl_state_` — *structurally* equivalent to PLECS "MOSFET Level 1" and PSIM "Switch". With the `boost-pfc-auto-parasitics` pre-flight we ship today, `C_oss` is auto-sized to keep the Tustin LC tank bounded. **Competitive.**
- **Behavioral** (line ~532): Shichman-Hodges Level-1 with a sigmoid blend across cutoff/triode/saturation (`κ = kSmoothRegionSharpness = 50`, line 397). This is closer to LTspice Level-1 than to a power-MOSFET model — datasheets give `R_ds_on`, `V_GS(th)`, `g_fs`, `C_oss`, `Q_g`, not `kp` and `λ`. **Teaching-level model.**

**Critical gaps:**
- **No body diode** (acknowledged in lines 17–20 header). PLECS and PSIM both bundle one with the MOSFET primitive. Synchronous-rectification waveforms in Pulsim show V_sw going to −V_dc instead of clamping at one diode drop.
- **Datasheet usability 2/5:** `kp` and `λ` are SPICE-Level-1 fields, not datasheet fields. Rename to `R_ds_on` + `g_fs` + `V_th` as primary, derive `kp` internally.

**Convergence — V_gs = 0 cutoff trace.** The audit confirmed the cutoff-region failure is **not** a missing stamp on the gate row per se — the `jacobian_pattern_impl` (line 193) reserves all 9 entries. The actual two failure modes:

1. **In float-precision builds**, `sigma_g = 1/(1+exp(κ·(V_th − V_gs)))` underflows to denormal/flush-to-zero by `V_gs = 0, V_th = 4`, killing the derivative chain.
2. **In any precision**, the gate-row *diagonal* is never anchored — `J(n_gate, n_gate)` is only stamped if some other device (PWM source, gate driver R) provides it. With a Thevenin-source gate driver of high impedance, the gate row is ill-conditioned.

**Fixes (high-priority, low-cost):**
- Reduce `κ` from 50 to 20 (or clamp `sigma_g = max(sigma_g, 1e-30)` post-exp).
- Stamp `J(n_gate, n_gate) += g_gate_leak ≈ 1e-9` unconditionally to anchor the gate node.

### 1.2 IGBT (`components/igbt.hpp`, 562 LOC)

**Fidelity — the biggest single fidelity bug in the library.**

The `v_ce_sat` field (line 36) is consumed *only* in `accumulate_loss` (line 174); it never appears in `stamp_jacobian_behavioral` (line 343). So the MNA solution gives `V_CE = I_C / g_on = 50 A / 1e4 = 5 mV` everywhere — physically wrong by ~300×. A user putting this into a 3φ inverter and reading line-line voltages sees 10 mV switch drops where they expect ~3 V (= 2 IGBTs in series).

**Fix.** Mirror the diode's Norton-shift pattern:

```
i_C(t) = (V_CE(t) − V_CE_sat(T_j)) / R_CE_on   when ON
```

with `R_CE_on` already a Params field (line 44) and Norton current source `I_0 = V_CE_sat / R_CE_on`. This both makes the IGBT current zero at V_CE = V_CE_sat (matches datasheets) **and** gives Newton a finite-slope landing zone instead of a thin sigmoid edge.

**Secondary gaps:** no tail-current model (PSIM/PLECS have it — IGBT turn-off losses are dominated by it), no antiparallel diode (every real bridge has one), no shoot-through current clamp (two ON IGBTs in a half-bridge give `g_on² ≈ 1e8 S` → Newton diverges).

### 1.3 IdealDiode / RealisticDiode (`components/ideal_diode.hpp`, 705 LOC)

**The best-modeled device in the library.** `V_F0 + R_d` Norton-shifted forward fit (line 619) is exactly the PLECS "thermal diode" model. T_j-correction on V_F0 (line 185), `Qrr` reverse-recovery on ON→OFF (lines 289–301), tiered `C_j` auto-snubber by trigger (5 nF for `Qrr > 0`, 500 pF for `V_F0 > 0`, 0 otherwise, line 139). **Matches PLECS; exceeds PSIM** (PSIM lacks Qrr in its default diode).

**Uniformity gaps with MOSFET/IGBT:**
- Stores bare fields (`V_F0_`, `R_d_`, …) instead of a `Params params_` aggregate. Cosmetic but breaks codegen.
- Uses `Qrr` / `Erec_shape` for switching energy where MOSFET/IGBT use `Eon_25`/`Eoff_25`/`I_ref`/`V_ref`/`Esw_tc`. Both are physically correct but tooling that aggregates "switching energy across all switches" has to special-case the diode.
- `R_th_ja` defaults to 25 K/W (line 72), not 0 — thermal is ON by default for the diode but OFF for MOSFET/IGBT.

**Convergence gap (single-line fix):** `event_hysteresis_ = 1e-2 V` (line 653) — at 10 mV, ESL ringing on a 400 V bus regularly exceeds the band and the diode chatters every Newton iteration. **Bump default to 50 mV.**

### 1.4 IdealSwitch + VoltageControlledSwitch — minor uniformity outliers

- `IdealSwitch` has *no* thermal binding, *no* `Params + name` constructor, `g_on = 1e6` default (three orders tighter than MOSFET).
- `VCSwitch` has `event_hysteresis_ = 1e-9 V` (line 268) — 10⁷ tighter than the rest of the family. **Bump to 1e-3 V for consistency.**

### 1.5 Cross-switch uniformity matrix

| Feature                          | MOSFET | IGBT | Diode | IdealSw | VCSw |
|----------------------------------|:------:|:----:|:-----:|:-------:|:----:|
| `Params + name` ctor             | ✅ | ✅ | ✅ | ❌ | ❌ |
| `R_th_ja` + `T_amb`              | ✅ | ✅ | ✅ | ❌ | ❌ |
| `accumulate_loss` (trapezoidal)  | ✅ | ✅ | ✅ | ❌ | ❌ |
| `Eon_25 + Eoff_25 + Esw_tc`      | ✅ | ✅ | ❌ (Qrr) | ❌ | ❌ |
| Auto-promote to Ideal            | ✅ | ✅ | ❌ | ❌ | ❌ |
| `C_oss` / `C_j` auto-default     | 10 nF | 20 nF | tiered | ❌ | ❌ |
| `*_user_set()` flag              | ✅ | ✅ | ✅ | ❌ | ❌ |
| `set_switching_mode`             | ✅ | ✅ | ✅ | ✅ | ✅ |
| `event_hysteresis` default (V)   | 1e-2 | 1e-2 | 1e-2 | — | **1e-9** |
| Norton-shifted ON-stamp          | ❌ | ❌ | ✅ | n/a | n/a |
| Body / antiparallel diode        | ❌ | ❌ | n/a | ❌ | ❌ |

---

## 2. Passives & Magnetics

### 2.1 Resistor / Capacitor / Inductor — the gold standard

`resistor.hpp` (189 LOC), `capacitor.hpp` (250 LOC), `inductor.hpp` (235 LOC).
**Uniformity 5/5** — same `Params` POD shape, same `accumulate_loss(x, dt)`
signature, same thermal accessor block. Trapezoidal companion stamps are
textbook-correct. `history_initialized_` flag (cap line 218, inductor line 205)
correctly arms BDF1 fallback on the first step from initial conditions.

**Ahead of PSIM** because PSIM's stock R/C/L has no thermal model. **Datasheet
usability 4/5** — missing voltage/current ratings, ESL on cap, saturation
current on inductor.

### 2.2 Transformer — the #1 fidelity gap

`transformer.hpp` (122 LOC). **Ideal-only — `Params::magnetizing_inductance`
exists but is never read in `stamp_impl`.** No leakage, no winding resistance,
no saturation, no core loss. Line 82 has a literal placeholder
`G.coeffRef(br_s, br_s) += 0.0` — the secondary branch row appears incomplete.

**The shocker:** the math IS there.
`core/include/pulsim/v1/magnetic/saturable_transformer.hpp` (168 LOC) ships a
multi-winding saturable transformer with `λ_m` state, per-winding leakage, and
core-loss integration. **It's just not wired to a `components/` device
wrapper.**

This is the highest-ROI fidelity fix in the audit: hooking the existing
`magnetic/saturable_transformer.hpp` into a new `SaturableTransformer` device
variant (mirroring how `HysteresisInductorDevice` wraps
`magnetic/hysteresis_inductor.hpp`) costs maybe 200 LOC + a Python binding.

### 2.3 HysteresisInductor — half-finished

`hysteresis_inductor_device.hpp` (179 LOC). Jiles-Atherton state advances
(line 130) but the MNA stamp uses a **constant** `L_eff` (line 104). So
hysteresis is *tracked* but not in the loop. Plus no `accumulate_loss` /
`total_energy()` integration with the unified loss pipeline.

**Fix.** Recompute `L_eff = ∂B/∂H · A·N²/path` at the current J-A operating
point each Newton step. PLECS's "Saturable Inductor" does exactly this — call
the math object's incremental-permeability accessor and re-stamp.

### 2.4 Steinmetz core loss — exists, but disconnected

`magnetic/bh_curve.hpp` (lines 198–280) implements both **Steinmetz**
(`P = k_h · f · B^α + k_e · f² · B²`) **and iGSE** (improved generalized
Steinmetz for non-sinusoidal flux — better than PSIM stock). Neither is
exposed through the device-level `accumulate_loss` path. `Inductor.hpp`'s
header comment (lines 33–36) explicitly acknowledges the gap.

**Fix.** Add `InductorParams::k_h`, `α`, `β` (Steinmetz triplet) +
`core_volume` + `effective_area` and wire the iGSE accumulator from
`magnetic/bh_curve.hpp` into `inductor.hpp::accumulate_loss`.

### 2.5 Passive uniformity matrix

| Feature                         | R  | C  | L  | Xfmr | HystL |
|---------------------------------|:--:|:--:|:--:|:----:|:-----:|
| `Params + name` ctor            | ✅ | ✅ | ✅ | ❌ | ✅ |
| `R_th_ja` + `T_amb`             | ✅ | ✅ | ✅ | ❌ | ❌ |
| Parasitic (TCR / ESR / DCR)     | ✅ | ✅ | ✅ | ❌ | ❌ |
| `accumulate_loss` + family quartet | ✅ | ✅ | ✅ | ❌ | ❌ |
| `history_initialized_` IC guard | n/a | ✅ | ✅ | n/a | ❌ |
| Saturation / B-H in stamp       | n/a | n/a | ❌ | ❌ | ❌ |
| Voltage / current rating field  | ❌ | ❌ | ❌ | ❌ | ❌ |
| Polarity flag                   | n/a | ❌ | n/a | n/a | n/a |

---

## 3. Motors

### 3.1 Per-motor fidelity

- **PMSM** (302 LOC) — **best in class.** Stationary-frame stamping with
  `L_s = (L_d + L_q)/2` average + proper salient dq torque
  `τ_em = 1.5·p·(ψ_pm·i_q + (L_d − L_q)·i_d·i_q)`. Shortcut for high-saliency
  IPM (electrical dynamics use the average L) but exact for SPM. **Gap:** no
  cogging torque, no saturation `L_d(i_d)`, no iron-loss resistor `R_c`.
- **InductionMotor (3φ)** (372 LOC) — proper Krause αβ-frame with rotor flux
  `ψ_r` as state and `σ·L_s` transient leakage in the per-phase stamp. **Gap:**
  no skin-effect / deep-bar `R_r(slip)` — under-estimates DOL start torque on
  motors > 50 kW. Forward-Euler on `ψ_r` should be trapezoidal (cross-coupling
  `ω_e · ψ_rβ` is the dominant rate at high slip).
- **BLDC** (317 LOC) — pure trapezoidal back-EMF, energy-balance torque. Clean
  but no Hall sensor outputs and no back-EMF shape parameter (PSIM has both).
- **DCMotor** (178 LOC) — permanent-magnet equivalent only. **No field-winding
  variants** (separately-excited, shunt, series, compound) — that's the gap vs
  PSIM's "DC Machine" which exposes `i_f, R_f, L_f, L_af(i_f)`.
- **Single-phase PSC** (262 LOC) — pure PSC. No CSCR (Capacitor-Start-
  Capacitor-Run), no centrifugal switch, no symmetrical-components
  decomposition into forward + backward fields (so the textbook torque dip
  near synchronous speed isn't visible). Forward-Euler on `V_cap` introduces
  a half-step phase lag — should be trapezoidal.

### 3.2 Motor uniformity matrix

| Field / Method                  | DC | PMSM | BLDC | IM(3φ) | 1φIM |
|---------------------------------|:--:|:----:|:----:|:------:|:----:|
| `omega` accessor                | ✅(1) | ✅ | ✅ | ✅ | ✅ |
| `theta` accessor                | ✅(1) | ✅ | ✅ | ✅ | ✅ |
| `tau_em` accessor               | ❌ | ✅ | ✅ | ✅ | ✅(2) |
| `R_th_ja` winding thermal       | ❌ | ❌ | ❌ | ❌ | ❌ |
| Circuit `<motor>_omega(name)`   | **❌** | ✅ | ✅ | ✅ | ✅ |
| Mechanical port (auto-shaft)    | ❌ | ❌ | ❌ | ❌ | ❌ |
| Quadratic load coefficient      | ✅ | ❌ | ✅ | ✅ | ✅ (via math) |
| Coulomb friction                | ❌ | ✅ | ✅ | ✅ | ✅ |
| `b` vs `b_friction` naming      | `b` | `b_friction` | `b_friction` | `b_friction` | `b_friction` |
| `R_a` vs `R_s` vs `Rs` naming   | `R_a` | `Rs` | `R_s` | `R_s` | `R_s_main/aux` |

(1) DC uses `omega()` / `theta()` not `omega_m()` / `theta_m()`.
(2) PSC uses `electromagnetic_torque()` not `tau_em()`.

**No motor has winding thermal binding** — every other power device does.
That's the most glaring family-wide gap.

### 3.3 Motor convergence gaps

- **High-slip start on InductionMotor** — forward-Euler on ψ_r at `dt = 100 µs`
  is barely stable when `ω_e · ψ_r` cross-coupling is ~314 rad/s. Replace with
  trapezoidal (one fixed-point iteration).
- **PMSM with sub-µH L_s** — high-frequency surface-magnet machines need
  `l_s_safe = max(L_s, 1e-9)` guard in the companion conductance (line ~154).
- **`V_cap` forward-Euler in 1φ PSC** at start — phase-shift instability
  during inrush. One-line fix: trapezoidal averaging.

---

## 4. Top 10 Prioritized Improvements

Ranked by **user-visible impact × inverse cost**. Each item lists scope and
estimated effort.

1. **Wire `magnetic/saturable_transformer.hpp` into a `SaturableTransformer`
   device variant.** Single largest fidelity gap. Math exists. ~200 LOC + py
   binding + Catch2 test. **2 days.**
2. **Add Norton-shift V_CE_sat to IGBT stamp.** Mirror diode's
   `i = g·(v − V_F0)` pattern with `V_CE_sat` and `R_CE_on` (both already in
   Params). Eliminates the 5 mV-instead-of-1.5 V fidelity bug. ~30 LOC + test.
   **Half day.**
3. **Stamp `J(n_gate, n_gate) += g_gate_leak ≈ 1e-9` on MOSFET + IGBT.**
   Anchors floating-gate node, eliminates singularity at deep cutoff. ~10 LOC
   per device. **2 hours.**
4. **Add body diode opt-in flag to MOSFET + IGBT (parallel to VSI's
   `add_body_diodes`).** Solves synchronous-rect reverse conduction. ~50 LOC.
   **Half day.**
5. **Reduce smooth-blend `κ` from 50 → 20 (or clamp `sigma_g ≥ 1e-30`).**
   Prevents float underflow in deep cutoff. Trade-off: transition window
   widens from ~120 mV to ~300 mV (still narrow vs PWM amplitude). **1 hour.**
6. **Bump diode `event_hysteresis_` default from 10 mV to 50 mV** (and align
   `VCSwitch` from 1 nV to 1 mV). Eliminates threshold chatter on noisy buses.
   **15 minutes.**
7. **Steinmetz / iGSE core loss on `Inductor` + new `SaturableTransformer`.**
   Math exists in `magnetic/bh_curve.hpp`. ~100 LOC of integration. **1 day.**
8. **Trapezoidal integration on `ψ_r` (InductionMotor) and `V_cap` (1φ PSC).**
   Two one-line fixes that improve high-slip start convergence. **1 hour.**
9. **Add `R_th_winding_to_ambient` + `T_winding` to all 5 motor families.**
   Mirror the diode/MOSFET thermal pattern. R_s and R_r' scale with T_winding
   so over-rated start pulses warm up the winding. ~30 LOC per motor. **1
   day.**
10. **Add a `Circuit::couple_shaft(motor_name, mechanical_name,
    gear_ratio=1.0)` declarative API** so users don't have to manually do
    `mech.set_tau_input(motor.tau_em()); motor.set_tau_load(mech.reaction())`
    each step. Wire `motors::GearBox::reflect_load` (currently dead code).
    ~30 LOC. **Half day.**

**Total estimated effort for the full list: ~9 days of focused C++ work**, of
which items 1, 2, 3 alone account for the bulk of the fidelity / convergence
deficit vs PSIM/PLECS.

---

## 5. Unification Gaps (cross-cutting)

### 5.1 `Params + name` constructor

Missing on: `Transformer`, `VoltageSource`, `CurrentSource`, `IdealSwitch`,
`VCSwitch`. Adopt everywhere so builders/templates pass a single struct.

### 5.2 Thermal binding (`R_th_ja`, `T_amb`)

Present: every passive (R/C/L), every switching device (MOSFET/IGBT/Diode).
Missing: **every motor**, `Transformer`, `HysteresisInductor`, `IdealSwitch`,
`VCSwitch`. The motors are the biggest miss — winding thermal model is the
standard PSIM/PLECS expectation.

### 5.3 `*_user_set()` flag for auto-defaulted parasitics

Present on the three switching devices that have an in-ctor auto-default for
`C_oss` / `C_j`. Should extend to **every** in-ctor auto-default in the
codebase (e.g. `Capacitor.ESR` defaulting to 0, `Inductor.DCR` defaulting to
0 — these are technically opt-in, not defaulted, but the pattern should be
consistent).

### 5.4 `accumulate_loss` family quartet

`total_energy` / `peak_power` / `average_power` /
`steady_state_junction_temperature` — present on R/C/L/Diode/MOSFET/IGBT. NOT
present on `Transformer`, `HysteresisInductor`, any motor, `IdealSwitch`,
`VCSwitch`. Promote to a `ThermalLossMixin` CRTP base (the three passive
implementations are line-for-line identical) and adopt across the family.

### 5.5 Stamp convention

MOSFET/IGBT use Norton-companion residual `f -= i_eq`. Diode/VCSwitch use
standard form `f += i_sw`. Both correct but mixed conventions are a
regression hazard — pick one and migrate.

### 5.6 Field naming

- `R_a` (DC motor) vs `R_s` (BLDC/IM) vs `Rs` (PMSM) vs `R_s_main/aux` (1φ).
- `b` (DC motor) vs `b_friction` (everyone else).
- `omega()` / `theta()` (DC) vs `omega_m()` / `theta_m()` (PMSM/BLDC/IM).
- `tau_em()` (PMSM/BLDC/IM) vs `electromagnetic_torque()` (1φ PSC).

Pick one convention per field and migrate the rest. The BLDC motor's naming
(`R_s`, `L_s`, `K_e_peak`, `b_friction`, `friction_coulomb`, `tau_load_const`,
`tau_load_quad_coeff`, `omega_m`, `theta_m`, `tau_em`) is the most complete —
**use BLDC as the template the other motors converge to.**

---

## 6. Solver-Side Levers Already in Place

Worth noting that the solver layer already has hooks the model layer can lean
on harder:

- **`ModelRegularizationOptions`** (`simulation.hpp:106-123`) — already defines
  per-device-class `g_off_min` floors (`mosfet_g_off_min = 1e-7`,
  `diode_g_off_min = 1e-9`, etc.). Today it's gated on `apply_only_in_recovery
  = true`. **A more aggressive default — apply at first Newton step, not just
  retry — would mask the floating-gate singularity automatically.** No code
  change needed; just flip the default.
- **`BackendTelemetry.simultaneous_event_groups`** (`simulation.hpp:158`) was
  recently added to count steps where ≥ 2 device commutations are coalesced
  into a single atomic Newton solve. This is exactly the diagnostic needed
  for the 3φ inverter case — combined with `pwl_event_bisections`, we can
  measure how often event coalescing is needed and what % of steps need it.
- **`auto_parasitics`** pre-flight already exists and works. Extending it to
  detect "switch → R → L → return" patterns (not just switch → L directly)
  would auto-fix VSI-style topologies without forcing users to call
  `set_mosfet_C_oss` manually.

---

## 7. What This Means for Convergence

The four highest-leverage convergence-side improvements **don't change any
physics** — they're numerical hardening:

1. Gate-row anchoring on MOSFET / IGBT (item 3 above).
2. `κ = 20` (or `sigma_g` clamp) on the smooth blend.
3. Aggressive `g_off_min` floor (flip `apply_only_in_recovery` default).
4. Trapezoidal on `ψ_r` and `V_cap` (motor side).

Combined, these eliminate the failure modes we've hit during the
`boost-pfc-auto-parasitics` work: V_pole = 0 stuck (gate floating), Behavioral
MOSFET conducting at cutoff (sigma_g underflow), threshold chatter (event
hysteresis), motor start-up instability (FE on flux state).

The fidelity improvements (saturable transformer, IGBT Norton-shift, body
diodes) are independent of these — they don't help convergence but they're
what users compare against PSIM/PLECS waveforms.

---

## See also

- [Convergence Tuning Guide](convergence-tuning-guide.md) — practical
  recipes for the convergence failures cited in §1 (Behavioral cutoff,
  PWL Ideal threshold flicker, motor flux integration).
- [Electrothermal Workflow](electrothermal-workflow.md) — the existing
  YAML-level configuration for thermal coupling on switching devices.
  Motor + transformer thermal binding (recommendation §5.2) extends
  this surface.
- [PWL Switching Engine](pwl-switching-migration.md) — design notes on
  the segment-state-space resolution that the gate-anchoring +
  `g_off_min` recommendations build on top of.
