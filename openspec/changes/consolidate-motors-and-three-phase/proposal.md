## Why

A motor-and-three-phase audit uncovered code-level duplication and four overlapping in-flight changes — three of which were planning to reimplement, in YAML primitives, what the spec already requires as first-class device classes. Without consolidation, Pulsim would ship two PMSM models, two three-phase source representations, and three competing tracks of motor benchmark work.

Concretely:
- `PmsmSteadyStateParams` (`runtime_circuit.hpp:2857`) is a POD aggregate with its own stamping path that the source comment itself flags as a "temporary placeholder pending PmsmDevice adoption". `PmsmDevice` (`components/pmsm_device.hpp:73`) is the canonical DeviceVariant-integrated path required by the `motor-models` spec.
- `ThreePhaseSourceParams` (`runtime_circuit.hpp:2525`) duplicates the field set of `grid::ThreePhaseSource` (`grid/three_phase_source.hpp:35`) instead of composing it. The grid math object is unused outside its own tests.
- `add-three-phase-control-blocks` shipped 15/15 tasks (Clarke/Park/PLL/SVM virtual blocks); it is already done and should be archived.
- `add-motor-drive-benchmarks` proposes "build motors from existing primitives (voltage sources for back-EMF, coupled inductors for stator windings)". The `motor-models` spec already requires `PmsmDevice`, `BldcMotorDevice`, `InductionMotorDevice` as proper C++ classes. The benchmark change is rebuilding what should already exist.
- `add-three-phase-grid-tied-suite` proposes a manual PLL composition from `integrator + gain + sum`. The `pll` virtual block from `add-three-phase-control-blocks` is already shipped.
- `integrate-three-phase-motors-magnetics-into-circuit-variant` is 5/24 done and is the only change actually doing the right work: wrapping the math objects in DeviceVariant entries and registering them in the mixed-domain scheduler.
- The `motor-models` spec lists `BldcMotorDevice` and `InductionMotorDevice` as shall-provide. Neither is implemented.

This change folds the surviving work from those four proposals into a single track, removes the duplications, fills the two missing motor classes, and ships the benchmark suite against the consolidated device API.

## What Changes

### Code-level deduplication (Phase A)

- **REMOVE** `PmsmSteadyStateParams`, `Circuit::add_pmsm_steady_state()`, the matching pybind11 binding, and the seven existing C++/Python tests built around it. Migrate the test intent to `PmsmDevice` with the rotor mechanical state held fixed via initial-condition (the steady-state operating point is the degenerate case of the dynamic dq model).
- **REFACTOR** `ThreePhaseSourceParams` to embed (compose) `grid::ThreePhaseSource` rather than duplicate its field set. Existing API surface (`Circuit::add_three_phase_source(name, nodes, ThreePhaseSourceParams)`) is preserved; the field accessors continue to work through the composed object. A new overload `Circuit::add_three_phase_source(name, nodes, grid::ThreePhaseSource)` accepts the math object directly, both from C++ and Python.

### Device-variant integration completion (Phase B)

Finishes the 19/24 remaining tasks of `integrate-three-phase-motors-magnetics-into-circuit-variant`:
- Wraps `grid::ThreePhaseSourceProgrammable` and `grid::ThreePhaseHarmonicSource` into DeviceVariant entries (currently only the basic `ThreePhaseSource` reaches Circuit through the params struct).
- Wraps the remaining motor math objects (`PMSM_FOC`, `Mechanical`) into the DeviceVariant.
- Wraps the magnetics math objects (`SaturableTransformer`, `HysteresisInductor`) into DeviceVariant entries.
- Adds the `Mechanical` domain tag to the mixed-domain scheduler so motor + mechanical-load coupling can be authored from YAML.
- Exposes everything via pybind11 (`Circuit::add_pmsm_foc`, `add_mechanical`, `add_saturable_transformer`, `add_hysteresis_inductor`).

### Missing motors (Phase C)

- **ADD** `BldcMotorDevice` — trapezoidal back-EMF (120° flat-top profile), six-step commutation aware via the gate-signal input. Same `DynamicDeviceBase` integration as `PmsmDevice`.
- **ADD** `InductionMotorDevice` — squirrel-cage induction in stationary αβ frame with rotor flux as state; slip-dependent torque computation. Same DeviceVariant integration.

### Benchmark suite on consolidated devices (Phase D)

- **ADD** `benchmarks/circuits/motor_bldc_six_step.yaml` exercising `BldcMotorDevice` via the existing six-switch inverter primitives and the new 6-step commutation gate pattern.
- **ADD** `benchmarks/circuits/three_phase_inverter_svpwm.yaml` — 6-switch sine-PWM inverter feeding a wye RL load. Reuses the shipped `svm` virtual block.
- **ADD** `benchmarks/circuits/back_to_back_rectifier_inverter.yaml` — AC-DC-AC pair sharing a DC link, exercises grid-side rectifier + load-side inverter.
- The previously-shipped `motor_dc_brush_step_load`, `motor_pmsm_dq_open_loop`, `motor_induction_locked_rotor` YAMLs continue to work and validate the consolidated device API rather than ad-hoc primitive macros.

### Spec ownership consolidation

- The `motor-models` capability gains the BLDC and induction requirements (closes the existing spec gap) and drops the temporary steady-state-via-POD requirement.
- The `three-phase-grid` capability gets a clear "composition over duplication" note in the source-device requirement.
- The `magnetic-models` capability gains device-variant requirements for the saturable transformer / hysteresis inductor.
- The `benchmark-suite` capability gains the new motor/grid-tied benchmark requirements.
- The `python-bindings` capability gets the new `add_*` exposure.

### Abandoned/archived changes

The following are deleted as **superseded by this change** — captured here for traceability, the original proposal narratives are absorbed where relevant in this `proposal.md`:
- `openspec/changes/add-motor-drive-benchmarks/`
- `openspec/changes/add-three-phase-grid-tied-suite/`
- `openspec/changes/integrate-three-phase-motors-magnetics-into-circuit-variant/`

The fourth in-flight change, `add-three-phase-control-blocks`, **is fully shipped (15/15)** and is archived independently of this consolidation (Phase A.0 below).

## Impact

- **Affected specs**: `motor-models`, `three-phase-grid`, `magnetic-models`, `python-bindings`, `benchmark-suite`.
- **Affected code**:
  - `core/include/pulsim/v1/runtime_circuit.hpp` — removes `PmsmSteadyStateParams`, refactors `ThreePhaseSourceParams`, wires new DeviceVariant entries for BLDC / induction / FOC / mechanical / magnetics, exposes new `add_*` methods.
  - `core/include/pulsim/v1/components/{bldc_motor_device,induction_motor_device,saturable_transformer_device,hysteresis_inductor_device,mechanical_device,pmsm_foc_device}.hpp` — new device wrappers.
  - `core/include/pulsim/v1/motors/{bldc_motor,induction_motor}.hpp` — new math objects.
  - `python/bindings.cpp` — removes `PmsmSteadyStateParams` binding, adds bindings for the new devices and overloads.
  - `python/pulsim/__init__.py` — updates `__all__`.
  - `core/tests/test_pmsm_steady_state.cpp`, `python/tests/test_pmsm_steady_state.py` — **deleted** (intent migrated to existing dynamic tests).
  - `core/tests/test_bldc_motor_device.cpp`, `core/tests/test_induction_motor_device.cpp`, `core/tests/test_saturable_transformer_device.cpp`, `core/tests/test_hysteresis_inductor_device.cpp`, `core/tests/test_pmsm_foc_device.cpp`, `core/tests/test_mechanical_device.cpp` — new.
  - `benchmarks/circuits/motor_bldc_six_step.yaml`, `three_phase_inverter_svpwm.yaml`, `back_to_back_rectifier_inverter.yaml` — new.
- **Backward compat**: existing `Circuit::add_three_phase_source(name, nodes, ThreePhaseSourceParams)` and `Circuit::add_pmsm(...)` continue to work bit-for-bit. The breaking removal is `add_pmsm_steady_state`, which had no production callers (only test fixtures) and was already commented in source as temporary.

## Success Criteria

1. **Zero duplicate model**: `grep` for `PmsmSteadyStateParams` returns 0 hits across `core/`, `python/`, `openspec/specs/`, `core/tests/`, `python/tests/` after this PR. `ThreePhaseSourceParams` composes `grid::ThreePhaseSource` (no parallel field set).
2. **Spec gap closed**: `BldcMotorDevice` and `InductionMotorDevice` are present in `core/include/pulsim/v1/components/` and registered in `DeviceVariant`, with dedicated unit + transient tests.
3. **Benchmark coverage**: all four motor-family benchmarks (DC brush, PMSM dq, BLDC six-step, induction locked-rotor) run against the consolidated devices, plus the three grid-tied benchmarks (SVPWM inverter, single-phase PLL grid-tied, back-to-back rectifier).
4. **Test suite**: 100% of the in-tree C++ + Python tests pass after the migration (post-deletion of the obsolete steady-state suite).
5. **One PR**: this change ships as a single PR — no partial-state intermediate releases.
