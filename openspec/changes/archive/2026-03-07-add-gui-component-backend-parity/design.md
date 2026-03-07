## Context

This change targets backend parity for all GUI components currently not implemented end-to-end. The gap was computed from PulsimGui library definitions and conversion support (snapshot date: **2026-02-22**).

Current gap summary:

- GUI components in library: 47
- Fully converted/supported today: 13
- Missing from backend conversion path: 34
  - 27 unsupported physical/behavioral components
  - 7 instrumentation/routing components currently skipped in conversion

## Goals / Non-Goals

- Goals:
  - Ensure every GUI component has a backend representation path (physical or virtual).
  - Standardize schemas and APIs so GUI, YAML, and Python bindings stay aligned.
  - Preserve deterministic runtime behavior for mixed electrical/control/instrumentation circuits.
- Non-Goals:
  - Full HDL/RTL-style digital simulation.
  - Full semiconductor physics TCAD-level models.
  - Automatic control-loop tuning or synthesis.

## Missing-Component Inventory and Target Strategy

### Unsupported Physical/Behavioral Components (27)

| GUI Component | Category | Backend Status Today | Target Backend Strategy | Phase |
|---|---|---|---|---|
| `BJT_NPN` | Semiconductors | Unsupported | Add BJT model/params and nonlinear stamp | P1 |
| `BJT_PNP` | Semiconductors | Unsupported | Add complementary BJT polarity model | P1 |
| `THYRISTOR` | Semiconductors | Unsupported | Add latching SCR model with gate/holding current | P1 |
| `TRIAC` | Semiconductors | Unsupported | Add bidirectional latching triac model | P1 |
| `SWITCH` | Semiconductors | Unsupported in GUI converter | Normalize mapping to backend switch model | P0/P1 |
| `OP_AMP` | Analog | Unsupported | Behavioral op-amp macro with gain/slew/rails | P3 |
| `COMPARATOR` | Analog | Unsupported | Comparator with hysteresis and delay | P3 |
| `FUSE` | Protection | Unsupported | I²t trip logic driving open switch state | P2 |
| `CIRCUIT_BREAKER` | Protection | Unsupported | Overcurrent + delay + reset/trip state | P2 |
| `RELAY` | Protection | Unsupported | Coil-driven `COM/NO/NC` contact model | P2 |
| `PI_CONTROLLER` | Control | Unsupported | Discrete/continuous PI block model | P3 |
| `PID_CONTROLLER` | Control | Unsupported | PID with anti-windup options | P3 |
| `MATH_BLOCK` | Control | Unsupported | Arithmetic/logic block ops | P3 |
| `PWM_GENERATOR` | Control | Unsupported | Carrier/compare PWM behavior block | P3 |
| `INTEGRATOR` | Control | Unsupported | Integrator block with saturation clamp | P3 |
| `DIFFERENTIATOR` | Control | Unsupported | Filtered differentiator | P3 |
| `LIMITER` | Control | Unsupported | Upper/lower clamp | P3 |
| `RATE_LIMITER` | Control | Unsupported | Slew-rate constrained output | P3 |
| `HYSTERESIS` | Control | Unsupported | Two-threshold stateful block | P3 |
| `LOOKUP_TABLE` | Control | Unsupported | 1D interpolation with selectable method | P3 |
| `TRANSFER_FUNCTION` | Control | Unsupported | LTI block realization from num/den | P3 |
| `DELAY_BLOCK` | Control | Unsupported | Time-delay block with interpolation | P3 |
| `SAMPLE_HOLD` | Control | Unsupported | Triggered sample-and-hold | P3 |
| `STATE_MACHINE` | Control | Unsupported | Deterministic finite-state block runtime | P4 |
| `SATURABLE_INDUCTOR` | Magnetic | Unsupported | Current-dependent inductance model | P2 |
| `COUPLED_INDUCTOR` | Magnetic | Unsupported | Mutual inductance/coupling matrix stamp | P2 |
| `SNUBBER_RC` | Networks | Unsupported | Canonical RC snubber macro expansion | P2 |

### Skipped Instrumentation/Routing Components (7)

| GUI Component | Category | Backend Status Today | Target Backend Strategy | Phase |
|---|---|---|---|---|
| `VOLTAGE_PROBE` | Measurement | Skipped | Virtual signal extractor (`V(node+)-V(node-)`) | P4 |
| `CURRENT_PROBE` | Measurement | Skipped | Virtual branch-current extractor | P4 |
| `POWER_PROBE` | Measurement | Skipped | Virtual `P=V*I` derived signal | P4 |
| `ELECTRICAL_SCOPE` | Measurement | Skipped | Channel container + signal binding metadata | P4 |
| `THERMAL_SCOPE` | Measurement | Skipped | Thermal channel container + signal binding metadata | P4 |
| `SIGNAL_MUX` | Measurement | Skipped | Signal-domain mux routing block | P4 |
| `SIGNAL_DEMUX` | Measurement | Skipped | Signal-domain demux routing block | P4 |

## Key Decisions

### Decision 1: Mixed-Domain Runtime Graph

Introduce a runtime graph with three component classes:

- Electrical stamped devices (MNA/DAE participation)
- Behavioral control blocks (signal-domain state update)
- Virtual instrumentation/routing blocks (no MNA stamp, but deterministic signal extraction/routing)

Why:

- Prevents forcing probes/scopes into fake electrical devices.
- Enables control blocks without compromising existing electrical solver pipeline.

### Decision 2: Deterministic Execution Order Per Step

For each accepted timestep:

1. Solve electrical state.
2. Update behavioral control blocks.
3. Apply event-driven state transitions (trip/latch/contact changes).
4. Emit instrumentation signals and channel snapshots.

Why:

- Predictable and reproducible coupling between electrical and control paths.

### Decision 3: Canonical Parameter Schema with Alias Layer

Use canonical backend parameter names and maintain alias translation for GUI/YAML compatibility (example: `lambda_` -> `lambda`, `turns_ratio` -> `ratio`).

Why:

- Keeps backend internals clean while preserving external compatibility.

### Decision 4: Family-by-Family Validation Gates

Require at least:

- 1 smoke test per component type
- 1 behavioral/reference test per family (power, protection, magnetic, control, instrumentation)

Why:

- Avoids regressions where component type is “accepted” but behavior is non-functional.

## API and Schema Impact

### YAML

- Add component type schemas for all missing components.
- Add strict pin-count and parameter-range validation for each.
- Add dedicated schema for instrumentation/routing blocks with signal-binding semantics.

### Python Bindings

- Expose creation/configuration paths for all new component families.
- Expose virtual probe/scope outputs and channel metadata in results.
- Preserve existing `add_*` APIs and procedural compatibility.

## Risks / Trade-offs

- Risk: Control-domain integration can destabilize timestep handling.
  - Mitigation: deterministic scheduler + bounded update order + dedicated mixed-domain regressions.
- Risk: Latching/trip models may increase convergence failures.
  - Mitigation: event localization, robust state transitions, and model-specific limiting.
- Risk: API expansion complexity.
  - Mitigation: canonical descriptor approach + compatibility wrapper methods.

## Migration Plan

1. Land kernel/runtime primitives and schemas first.
2. Land bindings + YAML parser support.
3. Land validation/benchmark gates.
4. Integrate GUI converter mappings once backend contracts are stable.

## Open Questions

- Should control blocks run strictly discrete-time (sampled) or support continuous-time mode per block?
- Should thermal scopes read directly from thermal solver outputs or from unified signal registry abstraction?
- Should `STATE_MACHINE` transitions be evaluated only on events or every accepted timestep?
