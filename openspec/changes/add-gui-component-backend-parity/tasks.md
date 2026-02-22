## 1. Scope Lock and Inventory

- [x] 1.1 Freeze the missing-component inventory snapshot (GUI catalog vs backend coverage) and keep it as a tracked artifact for this change.
- [x] 1.2 Define canonical backend component identifiers and GUI/YAML aliases for all missing components.
- [x] 1.3 Define parameter normalization rules (units, defaults, min/max validation, enum values) for every new component.

## 2. Kernel Runtime Foundation

- [x] 2.1 Extend runtime component abstraction to support mixed domains: electrical devices, control blocks, and virtual instrumentation/routing nodes.
- [x] 2.2 Add deterministic execution ordering between electrical solve, control update, and event-driven state transitions.
- [x] 2.3 Add structured diagnostics for unsupported/invalid component descriptors with stable error codes.

## 3. Power Semiconductors and Switching

- [x] 3.1 Implement `BJT_NPN` and `BJT_PNP` backend models and parameter structs.
- [x] 3.2 Implement `THYRISTOR` and `TRIAC` latching models with gate-trigger and holding-current behavior.
- [x] 3.3 Close `SWITCH` parity so GUI switch semantics map directly to backend model behavior.
- [x] 3.4 Add nonlinear convergence and limiting aids specific to new switching/latching devices.

## 4. Protection Components

- [x] 4.1 Implement `FUSE` behavior using I²t trip logic and open-state transition.
- [x] 4.2 Implement `CIRCUIT_BREAKER` overcurrent + trip-delay behavior with deterministic state changes.
- [x] 4.3 Implement `RELAY` coil/contact model (`COM/NO/NC`) with pickup/dropout thresholds and contact resistance states.

## 5. Magnetic and Preconfigured Network Components

- [x] 5.1 Implement `SATURABLE_INDUCTOR` with current-dependent inductance model.
- [x] 5.2 Implement `COUPLED_INDUCTOR` with mutual inductance/coupling coefficient support.
- [x] 5.3 Implement `SNUBBER_RC` as canonical backend macro model (explicit R-C branch realization).

## 6. Analog and Control Blocks

- [x] 6.1 Implement `OP_AMP` and `COMPARATOR` behavioral models including output saturation and comparator hysteresis.
- [x] 6.2 Implement control-law blocks: `PI_CONTROLLER`, `PID_CONTROLLER`, `MATH_BLOCK`, `PWM_GENERATOR`.
- [x] 6.3 Implement signal-processing blocks: `INTEGRATOR`, `DIFFERENTIATOR`, `LIMITER`, `RATE_LIMITER`, `HYSTERESIS`.
- [x] 6.4 Implement advanced blocks: `LOOKUP_TABLE`, `TRANSFER_FUNCTION`, `DELAY_BLOCK`, `SAMPLE_HOLD`, `STATE_MACHINE`.

## 7. Instrumentation and Signal Routing

- [x] 7.1 Implement virtual probe models: `VOLTAGE_PROBE`, `CURRENT_PROBE`, `POWER_PROBE`.
- [x] 7.2 Implement scope-channel models: `ELECTRICAL_SCOPE`, `THERMAL_SCOPE` (channel metadata + extracted waveforms).
- [x] 7.3 Implement routing blocks: `SIGNAL_MUX`, `SIGNAL_DEMUX` with deterministic channel mapping.

## 8. YAML and Python APIs

- [x] 8.1 Extend YAML parser schemas for all newly supported component types.
- [x] 8.2 Add strict validation diagnostics for missing pins, invalid parameter ranges, and incompatible block wiring.
- [x] 8.3 Expose Python bindings for new component creation/configuration APIs and result access for probes/scopes.
- [x] 8.4 Preserve backward compatibility of existing `Circuit.add_*` methods.

## 9. Validation and Benchmarking

- [x] 9.1 Add one smoke circuit per newly supported component type.
- [x] 9.2 Add behavioral reference tests per component family (switching, magnetic, control, protection, instrumentation).
- [x] 9.3 Add regression gate ensuring no GUI-declared component type returns “unsupported component type” in supported mode.
- [x] 9.4 Add performance checks for mixed-domain execution overhead.

## 10. Finalization

- [ ] 10.1 Update docs and capability matrix (GUI component -> backend support status).
- [ ] 10.2 Run `openspec validate add-gui-component-backend-parity --strict`.
- [ ] 10.3 Prepare migration notes for PulsimGui converter integration.
