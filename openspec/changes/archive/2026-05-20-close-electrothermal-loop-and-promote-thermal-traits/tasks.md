## Phase 1 — Close the T_j feedback loop (highest-ROI, ~0.5 days)

### 1.1 Wire `set_T_j_init` dispatch in `DefaultThermalService`
- [x] 1.1.1 In `core/src/v1/transient_services.cpp`, locate
      `DefaultThermalService::commit_accepted_segment` (~line 1201)
      where the per-device `T_i` Euler update + `refresh_scales()`
      runs. After `refresh_scales`, add a second `std::visit` walker
      that pushes `T_i` back into the device-internal `T_j_` via
      `set_T_j_init`. Gate the dispatch on
      `device_traits<T>::has_thermal_model = true`.
- [x] 1.1.2 The walker MUST be in the SAME order as the existing
      `refresh_scales` walker so `T_i[device_index]` aligns to the
      correct device. Reuse `circuit_.devices()` iteration index.
- [x] 1.1.3 Add a `set_T_j_init` declaration check via SFINAE / concept
      so devices that don't expose the setter (motors, magnetics)
      compile away to no-op. Future motor/magnetics thermal will fill
      these in without re-touching this dispatch.

### 1.2 Audit the existing electrothermal tests
- [x] 1.2.1 Identify every Catch2 test that uses `R_th_ja > 0` on a
      MOSFET / IGBT today. Run them under the new closed-loop dispatch
      and record numerical shifts. Expected shifts: `mosfet_junction_temperature`
      walks instead of staying flat; `Rds_on_at_Tj()` evaluates with
      higher T_j → slightly higher conduction loss → marginally hotter
      steady-state. Order of magnitude check: T_j at 1 kW continuous,
      R_th_ja = 1 K/W should land 50-80 °C above ambient, not at ambient.
- [x] 1.2.2 Update assertion tolerances in
      `test_v1_kernel.cpp::"v1 electro-thermal coupling emits device telemetry"`
      and any other tests in `[thermal]` / `[regression]` tags. Add a
      regression block that explicitly asserts T_j ≠ T_amb at the end
      of the simulation when `R_th_ja > 0`.

### 1.3 Add the closure regression test
- [x] 1.3.1 New `core/tests/test_electrothermal_loop_closure.cpp`.
      Topology: simplest buck (V_dc → L → MOSFET → Diode → C → R_load,
      PWM on the gate, MOSFET with `R_th_ja = 1 K/W`, `T_amb = 25 °C`).
      Run for several thermal time constants (≥ 1 s). Assert:
      (a) `mosfet_junction_temperature("M1")` ≠ `T_amb` at t = tstop.
      (b) The final `mosfet_junction_temperature("M1")` matches
      `mosfet_steady_state_junction_temperature("M1")` within 5 %.
      (c) The closed loop is monotone: T_j(t) is non-decreasing in the
      first ~0.5·τ and monotonically converges thereafter.
- [ ] 1.3.2 Mirror the same scenario for IGBT (after Phase 2 promotes
      its trait, today it's already true).

## Phase 2 — Promote `has_thermal_model = true` on Diode, R, L, C (~0.5 days)

### 2.1 IdealDiode
- [x] 2.1.1 In `core/include/pulsim/v1/components/ideal_diode.hpp`
      (search for `template<> struct device_traits<IdealDiode>`),
      flip `has_thermal_model` from `false` to `true`.
- [x] 2.1.2 Verify `set_T_j_init` exists on IdealDiode. If missing,
      add it as a noexcept setter on the device class (mirror MOSFET's
      `set_T_j_init` at `mosfet.hpp:295`).
- [ ] 2.1.3 Verify `R_th_ja` default. Today it is `25 K/W` (non-zero,
      divergent from MOSFET/IGBT which default to 0). Flag it for a
      follow-up discussion but DO NOT change the default in this
      proposal — that would be a behaviour change on every existing
      diode-using sim that has no thermal config.

### 2.2 Resistor
- [x] 2.2.1 Flip trait in `core/include/pulsim/v1/components/resistor.hpp`.
- [x] 2.2.2 Add `set_T_j_init` if missing.
- [ ] 2.2.3 Confirm `Resistor::R_at_Tj()` honors the new T_j(t) (it
      uses the temperature coefficient `TCR`).

### 2.3 Inductor
- [x] 2.3.1 Flip trait in `core/include/pulsim/v1/components/inductor.hpp`.
- [x] 2.3.2 Add `set_T_j_init` if missing.
- [ ] 2.3.3 Confirm `Inductor::DCR_at_Tj()` honors the new T_j(t).

### 2.4 Capacitor
- [x] 2.4.1 Flip trait in `core/include/pulsim/v1/components/capacitor.hpp`.
- [x] 2.4.2 Add `set_T_j_init` if missing.
- [ ] 2.4.3 Confirm `Capacitor::ESR_at_Tj()` honors the new T_j(t).

### 2.5 Per-device Circuit accessors
- [x] 2.5.1 In `core/include/pulsim/v1/runtime_circuit.hpp`, ensure the
      four devices have `<dev>_junction_temperature(name)`,
      `<dev>_steady_state_junction_temperature(name)`, and
      `<dev>_T_j_init(name, T)` accessors (mosfet pattern at lines
      926-989). Today some exist, some don't — audit and fill the
      gaps. Specifically:
      - Diode: `diode_junction_temperature` exists, add
        `diode_T_j_init` if missing.
      - Resistor: add all three if missing.
      - Inductor: add all three if missing.
      - Capacitor: add all three if missing.

## Phase 3 — YAML schema expansion (~0.5 days)

### 3.1 Extend `yaml_parser.cpp` thermal block
- [x] 3.1.1 Locate the `thermal_devices` parser in
      `core/src/v1/yaml_parser.cpp` (~line 88-93 — the dispatch by
      device type). Add `diode`, `resistor`, `inductor`, `capacitor`
      to the allow-list.
- [x] 3.1.2 For each added type, the YAML block accepts:
      `R_th_ja`, `T_amb`, `T_j_init`, `cth` (thermal capacitance),
      `temp_ref`, `alpha` (TC of the temperature-scaled parameter).
      Defaults match the existing C++ defaults (R_th_ja = 0 → thermal
      OFF). Reject unknown keys with a `ParseError` describing the
      schema.
- [x] 3.1.3 Add a small YAML regression test: a buck circuit defined
      entirely in YAML with `R_th_ja > 0` on the MOSFET AND the
      output capacitor. Assert both devices see T_j(t) walking from
      T_amb to a non-trivial steady-state value.

## Phase 4 — Telemetry consistency check (~0.25 days)

### 4.1 Cross-validate device.junction_temperature() vs thermal_summary
- [x] 4.1.1 In `test_electrothermal_loop_closure.cpp`, add an assertion
      that at every accepted step, `dev.junction_temperature()` equals
      the corresponding `thermal_summary.device_temperatures[i]` within
      1e-9 °C. They are currently the same constant T_amb (both wrong);
      after closure they should be the same T(t) (both right).
- [x] 4.1.2 Document the contract in
      `docs/electrothermal-workflow.md`: a paragraph stating that the
      device-side `<dev>_junction_temperature(name)` returns the
      simulator-integrated T_j(t), not the static `T_amb`.

## Phase 5 — Documentation + migration notes (~0.25 days)

- [ ] 5.1 Update `docs/thermal-and-loss-models-audit.md` § "Top 10
      priority improvements" to mark Priority 1 + 2 as ✅ landed in
      this change.
- [x] 5.2 Update `docs/electrothermal-workflow.md` with the closed-loop
      description + the list of devices participating
      (MOSFET / IGBT / Diode / R / L / C).
- [ ] 5.3 Migration note in `docs/migration-guide.md` (if exists, else
      `docs/release-notes-vX.md`): on circuits with `R_th_ja > 0`, the
      Rds_on / V_F0 / DCR / ESR values now track the integrated T_j(t)
      instead of being clamped at T_amb. Numerical drift up to ~5 % on
      steady-state telemetry. Back-compat preserved via `R_th_ja = 0`.

## Phase 6 — Validation

- [x] 6.1 Full Catch2 suite must stay green:
      `pulsim_tests` + `pulsim_simulation_tests` zero new failures.
- [x] 6.2 The new `[electrothermal_closure]` tag must have ≥ 6 cases
      / ≥ 20 assertions covering: MOSFET buck closure, IGBT inverter
      closure, diode rectifier closure, R-L-C tank thermal walk,
      YAML-driven closure, and the per-step
      `dev.junction_temperature() == thermal_summary.T_i` invariant.
- [ ] 6.3 Notebook visual smoke test: run
      `examples/notebooks/03_thermal_modeling.ipynb` and confirm the
      temperature plots no longer flatline at T_amb.
