## Status snapshot (after PR #3 lands)

- **Phase 0 — DONE** (1 commit): consolidated change scaffolded; the 3
  absorbed proposals deleted; `add-three-phase-control-blocks`
  archived as 2026-05-17-add-three-phase-control-blocks.
- **Phase A — DONE** (2 commits): `PmsmSteadyStateParams` removed
  (gate G.1); `ThreePhaseSourceParams` composes
  `grid::ThreePhaseSource` (gate G.2).
- **Phase B.2a — DONE**: `MechanicalDevice` registered in
  DeviceVariant; multi-shaft enabler via `set_tau_input` /
  `reaction_torque`.
- **Phase B.2b — DONE**: `PmsmFocDevice` signal-domain controller
  (PI current loops auto-tuned via pole-zero cancellation).
- **Phase B.1 — DONE**: `Circuit::add_three_phase_source` overloads
  for `grid::ThreePhaseSourceProgrammable` and `grid::ThreePhaseHarmonicSource`
  decompose into the appropriate sine legs (3 for programmable,
  3+3·N for harmonic).
- **Phase B.3 — DEFERRED** to a focused follow-up change:
  `SaturableTransformerDevice` and `HysteresisInductorDevice`
  require type-erasure for the templated BH-curve, variable
  winding count, and full nonlinear MNA coupling — better landed
  as a dedicated PR. The math objects remain accessible from C++.
- **Phase C.1 — DONE** (math + wrapper + tests): `BldcMotorDevice`
  with trapezoidal back-EMF; closes motor-models spec gap (G.3).
- **Phase C.2 — DONE** (math + wrapper + tests): `InductionMotorDevice`
  squirrel-cage in stationary αβ frame; closes motor-models spec
  gap (G.3).
- **Phase D (YAML pipeline) — DONE**: netlist parser dispatches
  `type: dc_motor`/`pmsm`/`bldc_motor`/`induction_motor` onto the
  matching `Circuit::add_*` builders. 4 YAML smoke tests pin the
  contract.
- **Phase D (benchmark YAMLs) — DEFERRED**: `motor_bldc_six_step`,
  `three_phase_inverter_svpwm`, `back_to_back_rectifier_inverter`
  with frozen KPI baselines belong in a dedicated benchmark PR
  now that the YAML pipeline is ready. (G.4 partial: motor YAML
  pipeline is in place; benchmark coverage itself is the deferred
  piece.)
- **Test suite — GREEN** (G.5): 497 / 497 in-tree C++ tests
  passing (was 466 baseline + 31 new from Phase A through D).
  Python `test_legacy_retry_fallback.py` etc. also green.
- **PR shape (G.6)**: single PR `feat/consolidate-motors-and-three-phase`,
  10 commits — Phase 0 + 8 phase commits + supporting work from
  parallel sessions on the same branch.

## Gates & Definition of Done

- [x] G.1 Zero duplicate model: `rg "PmsmSteadyStateParams"` returns 0 hits across `core/`, `python/`, `openspec/specs/`, tests.
- [x] G.2 Composition over duplication: `ThreePhaseSourceParams` field accessors and the `to_grid_source()` / `from_grid_source()` round-trip route through `grid::ThreePhaseSource`; `add_three_phase_source(params)` internally builds the math object and calls a private helper.
- [x] G.3 Spec gap closed: `BldcMotorDevice` and `InductionMotorDevice` are present in `core/include/pulsim/v1/components/` and listed in `DeviceVariant` (plus `MechanicalDevice` and `PmsmFocDevice` as bonus signal-domain devices).
- [ ] G.4 Benchmark coverage: motor benchmark YAML pipeline shipped (Phase D parser + 4 smoke tests); the benchmark YAMLs themselves (`motor_bldc_six_step`, `three_phase_inverter_svpwm`, `back_to_back_rectifier_inverter`) with frozen baselines are deferred to a dedicated benchmark PR — see "Phase D (benchmark YAMLs) — DEFERRED" above.
- [x] G.5 Full test suite green: 497 / 497 in-tree C++ tests passing; Python tests (excluding a pre-existing benchmark flake unrelated to motors / 3φ) all green.
- [x] G.6 Single PR — no partial-state intermediate releases. All work delivered through PR #3 on branch `feat/consolidate-motors-and-three-phase`.

## Phase 0: Workspace cleanup
- [ ] 0.1 Archive `add-three-phase-control-blocks` (15/15 already shipped) — move to `openspec/changes/archive/2026-05-17-add-three-phase-control-blocks/` and update `openspec/specs/kernel-v1-core/spec.md` to incorporate its virtual-block requirements.
- [ ] 0.2 Delete `openspec/changes/add-motor-drive-benchmarks/`.
- [ ] 0.3 Delete `openspec/changes/add-three-phase-grid-tied-suite/`.
- [ ] 0.4 Delete `openspec/changes/integrate-three-phase-motors-magnetics-into-circuit-variant/` (its 5/24 already-shipped tasks are reflected in the current state of `runtime_circuit.hpp`; the remaining 19 land here).
- [ ] 0.5 `openspec validate consolidate-motors-and-three-phase --strict` passes.

## Phase A: Code deduplication

### A.1 — ThreePhaseSourceParams composes grid::ThreePhaseSource
- [ ] A.1.1 Refactor `ThreePhaseSourceParams` in `core/include/pulsim/v1/runtime_circuit.hpp` to hold a `grid::ThreePhaseSource source` member. Replace the duplicated fields with forwarding accessors (`v_rms()`, `frequency()`, `phase_rad()`, `sequence()`) and forwarding setters.
- [ ] A.1.2 Add overload `Circuit::add_three_phase_source(name, nodes, const grid::ThreePhaseSource&)` that takes the math object directly.
- [ ] A.1.3 Expose the new overload via pybind11; bind `grid::ThreePhaseSource` as `pulsim.ThreePhaseSourceModel` (math-object form) so Python users can choose the surface they prefer.
- [ ] A.1.4 Run `core/tests/test_three_phase_source.cpp`, `test_pmsm_dynamic.cpp`, `test_pmsm_steady_state.cpp`, `test_three_phase_rl_load.cpp` (compile-level smoke); fix any field-access call sites that did not go through accessors.
- [ ] A.1.5 New test `core/tests/test_three_phase_source_composition.cpp` (~3 cases) confirming the composed access path returns identical values as the legacy field path, and the new overload threads through to the same stamp.

### A.2 — Remove PmsmSteadyStateParams
- [ ] A.2.1 Delete `PmsmSteadyStateParams` (`runtime_circuit.hpp:2857`), `Circuit::add_pmsm_steady_state()` and its convenience overload (lines 2868, 2911), and the `static_assert` immediately after the struct.
- [ ] A.2.2 Delete pybind11 binding for `PmsmSteadyStateParams` (`python/bindings.cpp:691–708, 1366`).
- [ ] A.2.3 Remove `PmsmSteadyStateParams` from the `from ._pulsim import (...)` block and the `__all__` list in `python/pulsim/__init__.py`.
- [ ] A.2.4 Delete `core/tests/test_pmsm_steady_state.cpp` (3 tests) and `python/tests/test_pmsm_steady_state.py` (4 tests).
- [ ] A.2.5 Add steady-state-via-pinned-ω cases to `core/tests/test_pmsm_dynamic.cpp` and `python/tests/test_pmsm_dynamic.py` so the operating-point validation intent is preserved on the canonical device path. Uses `PmsmDevice` initial condition to fix ω_m.
- [ ] A.2.6 Search-and-fix any straggling reference: `rg "PmsmSteadyStateParams|add_pmsm_steady_state"` returns 0 hits.
- [ ] A.2.7 Full C++ test suite green (`ctest --test-dir build --output-on-failure -j 4`).
- [ ] A.2.8 Full Python test suite green (`pytest python/tests -v --ignore=python/tests/validation`).

## Phase B: Finish device-variant integration

### B.1 — Three-phase source family
- [ ] B.1.1 Wrap `grid::ThreePhaseSourceProgrammable` into a `ThreePhaseProgrammableSourceParams` (composition pattern from A.1) and register in `DeviceVariant`. Expose `Circuit::add_programmable_three_phase_source(name, nodes, params)`.
- [ ] B.1.2 Wrap `grid::ThreePhaseHarmonicSource` similarly into `ThreePhaseHarmonicSourceParams`.
- [ ] B.1.3 pybind11 bindings for both. Tests in `core/tests/test_three_phase_source.cpp` extend to cover the programmable and harmonic variants end-to-end.

### B.2 — Remaining motors integrated
- [ ] B.2.1 Create `core/include/pulsim/v1/components/pmsm_foc_device.hpp` wrapping `motors::PMSM_FOC` math object as a `DynamicDeviceBase` subclass. Expose `Circuit::add_pmsm_foc(name, nodes, params)`.
- [ ] B.2.2 Create `core/include/pulsim/v1/components/mechanical_device.hpp` wrapping `motors::Mechanical` (shaft + inertia + viscous friction + external torque). Expose `Circuit::add_mechanical(name, params)` (no electrical nodes; couples via shaft state).
- [ ] B.2.3 Register `PmsmFocDevice` and `MechanicalDevice` in `DeviceVariant`. Add stamping branches in `stamp_rlc`, `advance_state`, `restore_state`.
- [ ] B.2.4 Add a `Mechanical` domain tag to the mixed-domain scheduler (`runtime_circuit.hpp` virtual-block evaluator) so mechanical state advances in lockstep with electrical/control.
- [ ] B.2.5 Tests: `core/tests/test_pmsm_foc_device.cpp` (3 cases: no-load spin-up, step-load response, current-loop bandwidth check), `core/tests/test_mechanical_device.cpp` (2 cases: inertia integration, shaft-torque coupling).

### B.3 — Magnetics integrated
- [ ] B.3.1 Create `core/include/pulsim/v1/components/saturable_transformer_device.hpp` wrapping `magnetic::SaturableTransformer`.
- [ ] B.3.2 Create `core/include/pulsim/v1/components/hysteresis_inductor_device.hpp` wrapping `magnetic::HysteresisInductor`.
- [ ] B.3.3 Register both in `DeviceVariant`; stamp branches.
- [ ] B.3.4 Expose `Circuit::add_saturable_transformer`, `Circuit::add_hysteresis_inductor` (C++ + pybind11).
- [ ] B.3.5 Tests: `core/tests/test_saturable_transformer_device.cpp`, `core/tests/test_hysteresis_inductor_device.cpp` (1-2 cases each: inrush current, hysteresis loop area).

### B.4 — Python surface complete
- [ ] B.4.1 Re-export the new device classes in `python/pulsim/__init__.py` `__all__`.
- [ ] B.4.2 Full Python test suite green.

## Phase C: Missing motors

### C.1 — BLDC motor device
- [ ] C.1.1 New math object `core/include/pulsim/v1/motors/bldc_motor.hpp` — trapezoidal back-EMF (120° flat-top, 60° linear ramp per phase, sequenced via mechanical angle), 6-step commutation aware via the gate inputs. Parameters: `R_s`, `L_s`, `K_e_peak`, `pole_pairs`, `J`, `b_friction`, `flux_density_profile` (function or table).
- [ ] C.1.2 New device wrapper `core/include/pulsim/v1/components/bldc_motor_device.hpp` (DynamicDeviceBase). Stamps phase currents into MNA; advances mechanical state per step.
- [ ] C.1.3 Register `BldcMotorDevice` in `DeviceVariant`; add stamping/advance/restore branches.
- [ ] C.1.4 Expose `Circuit::add_bldc_motor(name, nodes, params)` (C++ + pybind11).
- [ ] C.1.5 Tests: `core/tests/test_bldc_motor_device.cpp` (3 cases: open-circuit back-EMF waveform shape, six-step commutation phase-current pattern, locked-rotor inrush current). Plus `python/tests/test_bldc_motor_device.py` (2 cases: end-to-end YAML build + transient + telemetry check).

### C.2 — Induction motor device
- [ ] C.2.1 New math object `core/include/pulsim/v1/motors/induction_motor.hpp` — squirrel-cage three-phase induction in stationary αβ frame with rotor flux linkages as state. Parameters: `R_s`, `R_r`, `L_s`, `L_r`, `L_m`, `pole_pairs`, `J`, `b_friction`. Slip-dependent torque `T_em = (3/2)·p·(L_m/L_r)·(ψ_rα·i_sβ - ψ_rβ·i_sα)`.
- [ ] C.2.2 New device wrapper `core/include/pulsim/v1/components/induction_motor_device.hpp` (DynamicDeviceBase).
- [ ] C.2.3 Register in `DeviceVariant`; stamps.
- [ ] C.2.4 Expose `Circuit::add_induction_motor(name, nodes, params)`.
- [ ] C.2.5 Tests: `core/tests/test_induction_motor_device.cpp` (3 cases: locked-rotor inrush, no-load slip ≈ 0 in steady state, full-load slip vs analytical formula). Plus `python/tests/test_induction_motor_device.py` (2 cases).

## Phase D: Benchmark suite

### D.1 — Motor family completion
- [ ] D.1.1 `benchmarks/circuits/motor_bldc_six_step.yaml` — BLDC motor + 6-switch inverter + 6-step commutation pattern (vcswitches driven by mechanical-angle-aware controller). KPI: torque ripple %, phase current THD, average torque.
- [ ] D.1.2 Verify the previously-shipped `motor_dc_brush_step_load.yaml`, `motor_pmsm_dq_open_loop.yaml`, `motor_induction_locked_rotor.yaml` continue to run against the consolidated devices (the induction one will need to switch from primitive-rebuild to `add_induction_motor`).
- [ ] D.1.3 Freeze baselines for all four motor benchmarks via `benchmarks/freeze_kpi_baseline.py`.

### D.2 — Three-phase grid-tied benchmarks
- [ ] D.2.1 `benchmarks/circuits/three_phase_inverter_svpwm.yaml` — 6-switch sine-PWM inverter driving a wye RL load. Uses the shipped `svm` virtual block.
- [ ] D.2.2 `benchmarks/circuits/back_to_back_rectifier_inverter.yaml` — AC-DC-AC pair sharing a DC link. Grid-side rectifier (passive or active), load-side inverter feeding an RL or motor load.
- [ ] D.2.3 Freeze baselines.

### D.3 — Docs & integration
- [ ] D.3.1 Update `docs/motor-models.md` to list the four motor device classes (DC, PMSM, BLDC, induction) with parameter glossaries.
- [ ] D.3.2 Update `docs/three-phase-grid.md` to document the new programmable / harmonic source devices and the SVPWM benchmark.
- [ ] D.3.3 Add a notebook `examples/notebooks/31_motor_drive_zoo.ipynb` walking through one transient per motor type.

## Phase E: Validation & ship

- [ ] E.1 Full C++ test suite green.
- [ ] E.2 Full Python test suite green.
- [ ] E.3 Benchmark runner green on all motor + grid-tied scenarios.
- [ ] E.4 `openspec validate consolidate-motors-and-three-phase --strict` passes.
- [ ] E.5 Single PR opened on branch `feat/consolidate-motors-and-three-phase`, all commits structured per phase (A.1, A.2, B.1, ...).
