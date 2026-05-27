# Tasks — add-induction-motor-squirrel-cage

> Status (Phase 2.1 — v1.5.0 RC): Python-side complete, C++ port deferred.
> The Python observer pattern matches the existing PMSM / BLDC scaffolding
> and runs at v2's typical sub-µs dt without measurable overhead. A native
> C++ port (Task 1.x) is queued for a follow-up release once the API
> surface stabilizes.

## 1. Kernel C++ — deferred to a follow-up release

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

- [x] 2.1 No new pybind binding needed — the Python `make_induction_motor_observer` wires through the existing `step_observer` / `b_extra_fn` callbacks already exposed by the kernel.
- [x] 2.2 Extend `python/pulsim/motors.py`: `InductionMotor` dataclass, `add_induction_motor`, `make_induction_motor_observer`, `im_parameters_from_nameplate`.
- [x] 2.3 Re-export from `python/pulsim/__init__.py` (`__all__` entries).
- [x] 2.4 pytest covering nameplate helper, DOL smoke run, no-load acceleration, locked-rotor input impedance — all 4 pass locally (`python/tests/test_induction_motor.py`).

## 3. YAML support — deferred

- [ ] 3.1 Extend the YAML parser for `device_type: induction_motor`.
- [ ] 3.2 Schema validation + descriptive error when required fields missing.
- [ ] 3.3 Round-trip test: YAML → builder → simulate → expected steady-state.

## 4. Examples

- [x] 4.1 `examples/scripts/run_im_direct_online_start.py` — DOL start at 400 V / 50 Hz, plots stator current / rotor flux / torque / speed. Validated locally: motor reaches synchronous speed (1500 rpm) at 0 % slip with no load in 1.5 s sim time / 0.2 s wall.
- [ ] 4.2 `examples/yaml/induction_motor_dol.yaml` — equivalent YAML showcase (deferred with task 3).
- [ ] 4.3 `examples/scripts/run_im_ifoc_closed_loop.py` — indirect FOC with torque-step response using `MixedDomainBlockChain` (deferred for the sensorless-observer change which depends on it).

## 5. Docs — deferred to v1.5.0 final

- [ ] 5.1 New page `docs/v2/motors-induction.md` covering model equations, parameter identification recipe, IFOC tuning.
- [ ] 5.2 Link from the motors index page; cross-reference PMSM page.

## 6. Validation + release

- [x] 6.1 Run `openspec validate add-induction-motor-squirrel-cage --strict` after every spec edit — clean.
- [x] 6.2 pytest passes locally (4/4). Full cross-platform CI pending the PR.
- [ ] 6.3 Update `__version__` minor bump to 1.5.0 (motor library extension is a feature add, not a patch).
- [ ] 6.4 PR + merge + tag + publish.
- [ ] 6.5 Archive the change after deployment: `openspec archive add-induction-motor-squirrel-cage --yes`.
