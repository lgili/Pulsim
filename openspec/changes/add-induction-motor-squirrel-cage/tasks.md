# Tasks — add-induction-motor-squirrel-cage

> Status (Phase 2.1 — v1.5.0): **Python AND C++ paths complete**.
> The C++ BlockChain adapter at
> `motor_adapters.hpp::add_induction_motor_to_chain` mirrors the existing
> PMSM / BLDC pattern (header-only, captures live Mechanical + 2-state
> rotor flux in shared_ptr, writes back-EMF via ctx.b_extra). Validated
> against the Python observer to < 5% RMS error over a 4 kW DOL start.

## 1. Kernel C++ — done (header-only BlockChain adapter pattern)

- [x] 1.1 Added `add_induction_motor_to_chain` in
      `core/include/pulsim/motors/motor_adapters.hpp` (header-only,
      mirrors `add_three_phase_motor_to_chain`). 5-state Krause
      stationary-αβ model.
- [x] 1.2 Stamped via the existing per-phase Rs + σ·Ls + dummy-source
      pattern (user calls Python `pulsim.add_induction_motor` to build
      the branches; the C++ adapter resolves indices).
- [x] 1.3 Parameters passed positionally (matching existing motor
      adapter convention); `im_parameters_from_nameplate` provides a
      4-pole 4 kW 50 Hz reference set.
- [x] 1.4 Mechanical coupling via the shared `Mechanical` struct (same
      as PMSM/BLDC/DC).
- [x] 1.5 Output channels: `omega`, `theta`, `psi_alpha`, `psi_beta`,
      `torque`, `slip` (all optional via empty-string fallback).
- [x] 1.6 Integration test
      `python/tests/test_induction_motor_cpp_adapter.py` — compares the
      C++ adapter vs the Python observer on a 4 kW DOL start at 25 Hz.
      ω_m / |ψ_r| / T_em agree to < 5% RMS over 100 ms.
      Pure-C++ GoogleTest deferred (the BlockChain doesn't have C++
      tests for any of the other motor adapters either; Python
      integration is the established pattern).

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
