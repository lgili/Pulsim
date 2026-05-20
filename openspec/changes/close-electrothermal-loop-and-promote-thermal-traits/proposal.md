## Why

The May 2026 thermal + loss audit (`docs/thermal-and-loss-models-audit.md`)
found that Pulsim's electrothermal pipeline is built but **not closed**.
Two separate temperature trackers run in parallel without talking:

1. **Device-internal `T_j_`** (e.g. `mosfet.hpp:823`, `igbt.hpp:632`,
   `ideal_diode.hpp:683`) is initialised from `params.T_amb` at
   construction and mutated only via `set_T_j_init()`. It is what
   `Rds_on_at_Tj()`, `V_ce_sat_at_Tj()`, and the AC-side switching-energy
   `tc_factor` actually read.

2. **`DefaultThermalService` state** (`transient_services.cpp:1201`)
   integrates a per-device `T_i(t) ← T_i + dt·(P·R_th − (T_i−Tamb))/τ`,
   then pushes `scale_i = clamp(1 + α·(T_i − T_ref), 0.05, 4)` into the
   stamp via `circuit_.set_device_temperature_scales(scale_i)`.

The simulator **never calls `set_T_j_init()`** during a transient run.
So `T_j_` stays frozen at its construction value (typically 25 °C),
`mosfet_junction_temperature(name)` returns garbage for the whole sim,
and `Rds_on_at_Tj()` is dead code in any practical workflow. The two
trackers diverge silently — a warning at `runtime_circuit.hpp:849-865`
acknowledges the bug.

Compounding the issue, four device families that have working
`accumulate_loss` + `R_th_ja` + `T_amb` + `junction_temperature()`
accessor — IdealDiode, Resistor, Inductor, Capacitor — declare
`has_thermal_model = false` in their `device_traits` specialisations.
`DefaultThermalService::reset()` (`transient_services.cpp:1158-1162`)
uses that trait to gate which devices it tracks, so the closed loop is
restricted to MOSFET + IGBT today. The trait flip is a one-line change
per device that would expose four additional device families to the
electrothermal coupling that they're already 95 % wired for.

And the YAML parser tightens the same screw from the user side: it
accepts thermal config only for `mosfet`, `igbt`, and `bjt_*`
(`yaml_parser.cpp:88-93`). Users that set `R_th_ja` on a Diode / R / L
/ C in C++ have NO YAML route to express the same config — silent
asymmetry between the two configuration paths.

This proposal closes all three holes in one focused, surgical change.
No new physics, no new params. Pure pipeline plumbing + trait flips.

## What Changes

**The closed loop.** At the end of
`DefaultThermalService::commit_accepted_segment` (post the existing
`T_i` Euler update + `refresh_scales()` call), dispatch
`set_T_j_init(T_i)` on every device with `has_thermal_model = true`.
The same `std::visit` walker that today calls `set_device_temperature_scales`
gets a second branch that pushes the integrated `T_i` back into the
device-internal `T_j_`. Devices that don't expose `set_T_j_init`
(motors today, magnetics today) silently skip — adding the closed loop
to them is the scope of the follow-up OpenSpec `motor-loss-thermal-pipeline`.

**Trait flips on IdealDiode, Resistor, Inductor, Capacitor.** Each
already has `R_th_ja`, `T_amb`, `T_j_`, `set_T_j_init`,
`junction_temperature()`, `steady_state_junction_temperature()`. Flip
`device_traits<X>::has_thermal_model` from `false` to `true` in each
header. Each gets a per-device-name `*_T_j_init` setter on `Circuit`
(today exposed for MOSFET / IGBT only) so users can seed initial
junction temperatures from YAML or from a hot-start scenario.

**YAML schema expansion.** Extend the YAML thermal block parser
(`yaml_parser.cpp:88-93`) to accept `R_th_ja`, `T_amb`, `T_j_init` on
`diode`, `resistor`, `inductor`, `capacitor`. Schema unchanged for the
existing keys; new keys are additive and default to "thermal model OFF"
when omitted (back-compat).

**Telemetry consistency.** The device-side `junction_temperature()`
accessors will now reflect the integrated state, so
`thermal_summary.device_temperatures[i]` and
`mosfet_junction_temperature(name)` agree at every accepted step.
Update the existing electrothermal regression tests to assert that
agreement; they currently pass only because both report the same
constant.

**Status.** No API breaks. No behavior change on circuits that don't
configure thermal at all (`R_th_ja = 0` shorts the entire loop). For
circuits that DO configure thermal:
- MOSFET / IGBT: `Rds_on_at_Tj()` / `V_ce_sat_at_Tj()` now see the real
  T_j(t) instead of T_amb. **Behaviour change** — but the change brings
  the device into agreement with PSIM/PLECS, which is the stated goal.
- IdealDiode / R / L / C: their existing `accumulate_loss` accumulates
  on the T_amb-anchored resistance today; under this proposal it
  accumulates on T_j(t)-corrected resistance, exactly like FETs.
  **Same kind of behaviour change**, same direction.

The change ships with a sanity-check `test_electrothermal_loop_closure.cpp`
that runs a buck converter with `R_th_ja > 0` and asserts the device's
`mosfet_junction_temperature()` walks up from `T_amb` to a non-trivial
final temperature — today it returns `T_amb` for the whole sim.

## Impact

- **Affected specs**: `device-models` (trait surface on Diode, R, L, C +
  promoted thermal API surface on each), `kernel-v1-core` (electrothermal
  service contract: closed-loop T_j dispatch).
- **Affected code** (estimated 250-400 LOC modified, 200-300 LOC added):
  - `core/src/v1/transient_services.cpp` — extend
    `DefaultThermalService::commit_accepted_segment` walker with a
    second `set_T_j_init` dispatch branch (~30 LOC).
  - `core/include/pulsim/v1/components/{ideal_diode,resistor,inductor,
    capacitor}.hpp` — flip `device_traits<X>::has_thermal_model` (~4
    lines per file).
  - `core/include/pulsim/v1/runtime_circuit.hpp` — add per-device-name
    `*_T_j_init`, `*_steady_state_junction_temperature`, and verify
    `*_junction_temperature` accessors exist on diode/R/L/C (~80 LOC,
    mostly copy-paste from MOSFET / IGBT patterns).
  - `core/src/v1/yaml_parser.cpp` — extend the thermal block parser to
    accept the new device types (~30 LOC).
  - `core/tests/test_electrothermal_loop_closure.cpp` — new (~150 LOC).
  - Documentation: a paragraph in `docs/electrothermal-workflow.md`
    describing the closed loop + which devices participate.
- **Migration**: none required for users — existing circuits with
  `R_th_ja = 0` see no change. Users with `R_th_ja > 0` see the
  physically-correct temperature trajectory instead of a flat curve.
  This counts as a bug fix, not a breaking change.
- **Risk**: medium. The behaviour change on existing electrothermal
  tests (PWM-driven buck with `R_th_ja > 0`) needs careful audit —
  numbers will shift, tolerances may need widening. The change is
  back-compat-preserving via the `R_th_ja = 0` short-circuit, so any
  test that previously had `R_th_ja = 0` is bit-identical.
