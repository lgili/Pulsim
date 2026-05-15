## 1. Three-phase grid source — SHIPPED (Pulsim v0.10.0a1, commit 12634a8)

- [x] 1.1 (Pragmatic shortcut) `Circuit::add_three_phase_source` ships as a
      helper that decomposes a three-phase source into three internal
      `SineVoltageSource` branches between each line node and the shared
      neutral. **No new device-variant**, no changes to the ~10 `if constexpr`
      stamping ladders. Lives at `runtime_circuit.hpp:1984` (after
      `add_sine_voltage_source`). The real `grid::ThreePhaseSource` math class
      remains uncalled — it can be wired through a proper device-variant in a
      future iteration when programmable / non-sinusoidal waveforms are
      needed.
- [x] 1.2 Branch equations come for free from the existing `SineVoltageSource`
      stamping path (one per leg = three branches total). The helper records
      branch names `<name>__A`, `<name>__B`, `<name>__C` so per-phase
      probe-current bindings keep working.
- [x] 1.3 `add_three_phase_source(name, n_a, n_b, n_c, n_neutral, params)` +
      convenience overload taking `(v_line_to_line_rms, frequency_hz)`. Params
      struct `Circuit::ThreePhaseSourceParams` (line-to-line RMS, frequency,
      phase A, sequence direction, unbalance factor) is a POD with
      `static_assert(std::is_trivially_copyable_v<...>)`.
- [x] 1.4 Source is already in the `Electrical` domain (no scheduler changes
      needed since it uses existing primitives).
- [x] 1.5 Backend tests in `core/tests/test_three_phase_source.cpp` (3 cases,
      16 asserts): balanced positive sequence peak, negative-sequence
      B/C swap, unbalance factor scales B and C. Full suite: 141 cases,
      1106 asserts, all green. Python smoke tests in
      `python/tests/test_three_phase_source.py` (6 cases): params class +
      writable fields + balanced peaks + negative sequence flip +
      unbalance scaling + convenience overload.

**Lessons learned for Tracks 2/3 — concrete device-variant integration map:**

When a new device cannot be decomposed into existing primitives (PMSM, FOC,
DC motor, saturable transformer all need true device-variant status), the
following ladders in `runtime_circuit.hpp` must be extended (one
`else if constexpr (std::is_same_v<T, NewDevice>)` branch per location):

- `using DeviceVariant = std::variant<...>` declaration at line **56–71**
- `add_node` branch-index re-numbering hook at line **130** (only matters if
  the device reserves MNA branch rows like VoltageSource / Inductor do)
- Tagging / capability filters at lines **366, 436, 1591, 2255, 2302, 2328,
  2417, 2595, 2860** — these are the per-device class ladders (filter
  capacitor/inductor for the history-advance hooks, filter sources for the
  branch-row reservation, etc.)
- `stamp_device_dc` at line **3744** — DC operating-point stamping
- `stamp_device_jacobian` at line **3818** — transient Jacobian stamping
- `update_history` hooks at lines **2603, 2624, 2647, 2989, 3061** — advance
  the internal state ω/θ/φ_flux at the end of each accepted timestep
- `replace_device` at line **1860** — only if the device participates in
  hot-swap (e.g., MOSFET model upgrade); motors and magnetics don't.

The CRTP `stamp_impl` from `core/include/pulsim/v1/components/*.hpp` is
**not called** by the runtime path — only the `if constexpr` ladders. The
runtime treats every device by value inside the variant. New devices should
follow `Inductor` (history + state) for motors and `Transformer` (multiple
branch rows) for the saturable transformer.

## 2. Motors — device integration

- [ ] 2.1 Add a new `Mechanical` domain tag in `core/include/pulsim/v1/scheduler.hpp`
      (or wherever `mixed_domain_phase_order` is declared).
- [ ] 2.2 Create `core/include/pulsim/v1/devices/motor_devices.hpp` wrapping
      `motors::pmsm`, `pmsm_foc`, `dc_motor`, `mechanical` as device variants. PMSM
      contributes 2 electrical branches + 2 mechanical state vars (ω, θ).
- [ ] 2.3 Add `Circuit::add_pmsm(...)`, `add_pmsm_foc(...)`, `add_dc_motor(...)`,
      `add_mechanical(...)`.
- [ ] 2.4 Hook the new devices into `mixed_domain_phase_order` so mechanical updates
      run after the corresponding electrical solve.
- [ ] 2.5 Backend test: PMSM-FOC no-load spin-up reaches reference speed within
      configured settling time.

## 3. Saturable magnetics — device integration

- [ ] 3.1 Create `core/include/pulsim/v1/devices/saturable_magnetic_devices.hpp`
      wrapping `magnetic::saturable_transformer` and `magnetic::hysteresis_inductor`
      as device variants.
- [ ] 3.2 Implement nonlinear mutual-inductance Jacobian stamping for the saturable
      transformer (uses `bh_curve` for the flux-current map).
- [ ] 3.3 Add `Circuit::add_saturable_transformer(...)` and `add_hysteresis_inductor(...)`.
- [ ] 3.4 Backend test: drive a saturable transformer beyond knee voltage and confirm
      flux flat-tops vs. the linear ideal-transformer reference.

## 4. pybind11 exposure

- [ ] 4.1 Expose `grid::ThreePhaseSource` and `grid::ProgrammableThreePhaseSource`
      plus `PhaseSequence` enum in `python/bindings.cpp`.
- [ ] 4.2 Expose `motors::PMSM`, `PMSM_FOC`, `DC_Motor`, `Mechanical` (params + state
      structs).
- [ ] 4.3 Expose `magnetic::SaturableTransformer`, `HysteresisInductor`, `BHCurve`,
      `CoreCatalog` + the catalog entries.
- [ ] 4.4 Expose the new `Circuit::add_*` methods alongside the existing ones.
- [ ] 4.5 Update `python/pulsim/__init__.py` `__all__` and add module-level type
      stubs.
- [ ] 4.6 Python smoke tests: drop each component, run a 1 ms transient, assert
      shape of result.

## 5. Release

- [ ] 5.1 Run the full existing benchmark / validation suite — none of the existing
      transient tests should regress.
- [ ] 5.2 Bump Pulsim `0.9.x → 0.10.0`.
- [ ] 5.3 Tag `v0.10.0`, push to PyPI, monitor smoke imports across py3.10/3.11/3.12/3.13.

## 6. Hand-off

- [ ] 6.1 Open the wave-5 OpenSpec on the PulsimGui side that depends on Pulsim
      v0.10.0; reopen the deferred sub-wave-C palette + dialog work using the new
      `Circuit::add_*` hooks.
