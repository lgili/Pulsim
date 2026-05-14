## Why

Phase 3+ landed math-object headers for three-phase grid sources (`grid/three_phase_source.hpp`),
motors (`motors/{pmsm, pmsm_foc, dc_motor, mechanical}.hpp`), and saturable magnetics
(`magnetic/{saturable_transformer, hysteresis_inductor, bh_curve}.hpp`). Each header
explicitly flags a deferred follow-up: *"The downstream simulator integration (registering
them as Circuit devices that stamp three branch equations) lands once the Circuit-variant
integration follow-up arrives."* (three-phase) / *"Phase 4+ wires it into Circuit's device
variant."* (saturable transformer) / *"Phase 3"* (PMSM dq frame, with no `Circuit::add_pmsm`
hook yet).

The PulsimGui frontend reached the point in its wave-4 plan where it wanted to expose these
as palette items. Inspection of the runtime confirmed there is no `Circuit::add_three_phase_source()`,
`Circuit::add_pmsm()` or `Circuit::add_saturable_transformer()`, and `python/bindings.cpp`
has no pybind exposure either. The wave-4 sub-wave C (palette + bindings + dialog wiring)
was therefore moved out of scope and is blocked on this change.

## What Changes

Three coordinated tracks on the C++ side, then a pybind exposure track. Each math-object
class gets:

1. **Device-variant registration.** A new `pulsim::v1::devices::*Device` wrapper that satisfies
   the existing `Circuit` device variant contract (residual contribution, Jacobian stamp,
   state advance, event participation). For the three-phase source this means three branch
   equations (one per phase); for the PMSM, two electrical branches + two mechanical state
   variables (ω, θ); for the saturable transformer, mutual-inductance + nonlinear flux
   coupling.
2. **`Circuit::add_*` API.** A new `Circuit::add_three_phase_source(...)` /
   `add_pmsm(...)` / `add_pmsm_foc(...)` / `add_dc_motor(...)` / `add_mechanical(...)` /
   `add_saturable_transformer(...)` / `add_hysteresis_inductor(...)`. Each returns the
   device handle so callers can wire it up the same way they wire ideal switches today.
3. **Mixed-domain scheduler hooks.** Motors and mechanical loads need to participate in the
   mixed-domain ordering already defined by `Circuit::mixed_domain_phase_order`. The hooks
   add a new domain tag (`Mechanical`) and slot the new devices into the deterministic
   execution order.
4. **pybind exposure.** Once the C++ API stabilises, `python/bindings.cpp` exposes each
   class with the same shape as existing devices (`MOSFETParams` etc.), and `Circuit`'s
   pybind methods grow `.def("add_three_phase_source", &Circuit::add_three_phase_source, ...)`
   entries.

The deliverable is **Pulsim v0.10.0** on PyPI with the new API + bindings, which then
unblocks PulsimGui wave-5 to reopen the sub-wave-C palette work.

## Impact

- **Affected specs:**
  - `three-phase-grid` — new device-class requirement and `Circuit::add_three_phase_source` API
  - `motor-models` — new device-class requirements (PMSM / PMSM-FOC / DC motor /
    mechanical load) and corresponding `Circuit::add_*` API
  - `magnetic-models` — saturable transformer + hysteresis inductor device-class
    requirements and `Circuit::add_*` API
  - `python-bindings` — pybind exposure of all of the above
- **Affected code (Pulsim):**
  - `core/include/pulsim/v1/devices/three_phase_source_device.hpp` (new)
  - `core/include/pulsim/v1/devices/motor_devices.hpp` (new)
  - `core/include/pulsim/v1/devices/saturable_magnetic_devices.hpp` (new)
  - `core/include/pulsim/v1/circuit.hpp` — new `add_*` overloads + device-variant updates
  - `core/include/pulsim/v1/scheduler.hpp` — new `Mechanical` domain tag
  - `python/bindings.cpp` — new class_ + Circuit method exposures
  - `python/pulsim/__init__.py` — `__all__` re-exports
- **Release plan:** Land C++ tracks 1–3 first, validate via existing benchmarks
  (add-three-phase-grid-tied-suite, add-motor-drive-benchmarks, add-magnetic-saturation-fidelity
  — all already authored as proposals); land pybind exposure last; cut Pulsim v0.10.0.
  PulsimGui wave-5 picks up the GUI palette work after the PyPI release.
