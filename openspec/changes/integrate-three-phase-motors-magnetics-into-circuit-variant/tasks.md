## 1. Three-phase grid source — device integration

- [ ] 1.1 Create `core/include/pulsim/v1/devices/three_phase_source_device.hpp`
      wrapping `grid::ThreePhaseSource` + `grid::ProgrammableThreePhaseSource` as a
      `Circuit` device variant.
- [ ] 1.2 Implement residual contribution + Jacobian stamping for three branch
      equations (one per phase).
- [ ] 1.3 Add `Circuit::add_three_phase_source(name, nodes_abc, source_params)`.
- [ ] 1.4 Register the device under the existing `Electrical` domain tag in
      `Circuit::mixed_domain_phase_order`.
- [ ] 1.5 Backend test: drop a 3φ source, sample 1 ms transient, verify three
      phase-shifted waveforms.

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
