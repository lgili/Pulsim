# Tasks — add-induction-motor-squirrel-cage

## 1. Kernel C++

- [ ] 1.1 Create `core/include/pulsim/motors/induction_motor.hpp` with `InductionMotor` template class, 5-state dq model, configurable reference frame (stationary default).
- [ ] 1.2 Stamp the device on `CircuitBuilder` via the existing motor-device infrastructure (mirror `pmsm.hpp` glue).
- [ ] 1.3 Add `InductionMotorParams` POD with defaults for a 4-pole 4 kW 50 Hz reference machine.
- [ ] 1.4 Wire mechanical coupling through the existing `Mechanical` block (shared with PMSM/BLDC).
- [ ] 1.5 Output channels: `i_a`, `i_b`, `i_c`, `psi_dr`, `psi_qr`, `T_e`, `slip`, `omega_m`.
- [ ] 1.6 Unit tests in `core/tests/test_induction_motor.cpp`:
  - Locked-rotor input impedance vs analytical formula (5 % tol).
  - No-load: rotor current < 5 % of stator at synchronous speed.
  - Slip-torque steady-state sweep matches `T(s)` analytical curve.

## 2. Python bindings + helpers

- [ ] 2.1 pybind binding in `python/bindings.cpp` exposing `add_induction_motor`, `InductionMotorParams`.
- [ ] 2.2 Extend `python/pulsim/motors.py`: `make_induction_motor_observer`, `im_parameters_from_nameplate`.
- [ ] 2.3 Re-export from `python/pulsim/__init__.py` (`__all__` entries).
- [ ] 2.4 pytest covering the analytical-impedance regression and observer wiring.

## 3. YAML support

- [ ] 3.1 Extend `python/pulsim/yaml_loader.py` (or kernel YAML parser) for `device_type: induction_motor`.
- [ ] 3.2 Schema validation + descriptive error when required fields missing.
- [ ] 3.3 Round-trip test: YAML → builder → simulate → expected steady-state.

## 4. Examples

- [ ] 4.1 `examples/scripts/run_im_vf_open_loop.py` — V/f ramp 0 → 50 Hz, plot stator current / rotor flux / torque / speed.
- [ ] 4.2 `examples/yaml/induction_motor_vf.yaml` — equivalent YAML showcase.
- [ ] 4.3 `examples/scripts/run_im_ifoc_closed_loop.py` — indirect FOC with torque-step response using `MixedDomainBlockChain`.

## 5. Docs

- [ ] 5.1 New page `docs/v2/motors-induction.md` covering model equations, parameter identification recipe, IFOC tuning.
- [ ] 5.2 Link from the motors index page; cross-reference PMSM page.

## 6. Validation + release

- [ ] 6.1 Run `openspec validate add-induction-motor-squirrel-cage --strict` after every spec edit.
- [ ] 6.2 Full pytest pass on Linux + macOS + Windows (CI).
- [ ] 6.3 Update `__version__` minor bump to 1.5.0 (motor library extension is a feature add, not a patch).
- [ ] 6.4 PR + merge + tag + publish.
- [ ] 6.5 Archive the change after deployment: `openspec archive add-induction-motor-squirrel-cage --yes`.
