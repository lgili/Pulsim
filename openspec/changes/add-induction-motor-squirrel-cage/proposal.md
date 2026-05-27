## Why

The `motor-models` spec already declares an `InductionMotorDevice` requirement, but no implementation ships in v1.4.2. Pulsim therefore covers DC, PMSM, BLDC — but **not** the squirrel-cage induction motor, which represents roughly 70 % of installed industrial-drive horsepower worldwide. Customers evaluating Pulsim against PSIM / PLECS for VFD / inverter-fed-drive workflows hit this gap immediately.

The PSIM Motor Drive Library treats IM as a first-class block (DQ-frame state model, slip-ring or squirrel-cage variants, V/f and FOC templates). PLECS Electrical Machines library does the same. To close the gap, Pulsim needs a numerically-stable IM device, the standard analytical helpers around it (no-load / locked-rotor parameter ID, slip / torque curves), and at least two showcase scripts: an open-loop V/f ramp and a vector-controlled (IFOC) drive.

## What Changes

- **C++ kernel** — add a `InductionMotor` device in `core/include/pulsim/motors/induction_motor.hpp` with:
  - 5th-order squirrel-cage model in the stationary dq (α-β-0) frame: stator d/q current states, rotor flux d/q states (referred to stator), mechanical speed coupled through an external `Mechanical` block (already in place for PMSM/BLDC).
  - Parameters: `Rs`, `Ls`, `Lr`, `Lm`, `Rr` (rotor resistance referred to stator), `pole_pairs`, optional `core_loss_resistance`.
  - Two reference frames selectable at build time: stationary (default — keeps voltages physical for direct grid attachment) and synchronous (cheaper for steady-state at fixed slip).
  - Output channels: per-phase stator currents `i_a/i_b/i_c`, rotor flux magnitude, electrical torque, slip, mechanical speed.
- **Python binding** + helper API:
  - `CircuitBuilder.add_induction_motor(name, terminal_nodes, params, mechanical)` returning a handle for observer wiring.
  - `pulsim.motors.InductionMotorParams` POD with sensible defaults for a 4-pole 4 kW 50 Hz machine.
  - `pulsim.motors.make_induction_motor_observer(handle, mechanical)` matching the existing PMSM/BLDC observer pattern so the device snaps into the standard `step_observer` chain.
  - `pulsim.motors.im_parameters_from_nameplate(P_rated, V_LL, f, ...)` analytical helper that fills the equivalent-circuit RLs from nameplate + locked-rotor / no-load test data.
- **YAML support** — `device_type: induction_motor` keyed entry mirroring PMSM, with all params.
- **Examples**:
  - `examples/scripts/run_im_vf_open_loop.py` — V/f ramp 0 → 50 Hz on a 4 kW machine, plots stator current, rotor flux, torque, mechanical speed.
  - `examples/scripts/run_im_ifoc_closed_loop.py` — indirect FOC (slip-frequency-based) using the existing `MixedDomainBlockChain` building blocks, demonstrating step response to a torque reference.
- **Cross-validation** — one Python regression test comparing locked-rotor input impedance to the analytical formula `Rs + jωσLs + (Rr/s + jωLr) || jωLm` and a steady-state slip-torque curve sweep matching the analytical T-slip relation within 5 %.
- **Docs** — page in the existing motor tutorial describing the IM model, parameters, and the IFOC tuning recipe.

## Impact

- **Affected specs**: `motor-models` (expanded IM requirements + scenarios), `netlist-yaml` (new device_type).
- **Affected code**: new `core/include/pulsim/motors/induction_motor.hpp` (~600 LOC), pybind binding (~80 LOC), `python/pulsim/motors.py` extensions (~200 LOC), YAML parser path (~40 LOC).
- **Backward compatibility**: PURE ADDITION — no existing API changes. The empty `InductionMotorDevice` requirement in `motor-models` becomes implemented, not redefined.
- **Performance**: 5 extra ODE states per IM. At typical 4 kHz PWM with dt = 10 µs, adds ~1 % of solve time vs an open-loop converter. No nonlinearity issues — the dq form is linear in states for constant speed.
- **Risk**: numerical instability when running at exactly synchronous speed (slip = 0). Mitigated by a small `s_floor = 1e-6` clamp documented in the helper.
