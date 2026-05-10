## Why

Building a buck/boost/flyback/LLC/DAB/PFC from primitives in a YAML netlist is repetitive and error-prone. Every user re-derives the same topology, often with subtle mistakes (wrong winding polarity, missing snubber, wrong control loop sign). PSIM and PLECS ship libraries of pre-validated converter blocks. To compete on time-to-first-result, Pulsim needs the same.

A "converter template" is a parameterized macro that expands into a netlist when instantiated, with internal control optionally embedded. A user writes:

```yaml
- type: buck_template
  name: PSU1
  parameters:
    Vin: 48
    Vout: 12
    Pout: 240
    fsw: 100e3
    inductor: { value: 47e-6, dcr: 0.020 }
    output_cap: { value: 470e-6, esr: 0.012 }
    mosfet_high: catalog/wolfspeed/C3M0065090J
    mosfet_low:  catalog/wolfspeed/C3M0065090J
    control:
      type: voltage_mode_pi
      kp: 0.5
      ki: 1500
      crossover_hz: 5000
```

…and gets a fully-wired buck with synchronous rectification, gate driver behavior, output cap ESR, and PI compensator already connected. They run a transient or AC sweep and immediately get a Bode plot. **That's the experience PSIM/PLECS deliver.**

## What Changes

### Converter Template DSL
- New top-level component type with `_template` suffix that expands at parse time into a sub-netlist.
- Expansion is deterministic: same parameters → same netlist (verifiable via golden expansion test).
- Templates compose: a `three_phase_inverter_template` may instantiate three `half_bridge_template` blocks.

### First-Wave Templates (10 blocks)
- **Topologies**: buck, boost, buck-boost, flyback (DCM/CCM), forward, two-switch forward, half-bridge, full-bridge, LLC half-bridge, dual-active-bridge (DAB), totem-pole PFC, interleaved-2φ buck.
- **Control variants**: voltage-mode PI, peak current-mode, average current-mode, type-II/III compensator.
- **Optional features**: snubbers (RC, RCD), gate driver model, output ESR, dead-time generator.

### Auto-Designed Defaults
- For each template, sensible defaults computed from `(Vin, Vout, Pout, fsw)` if the user omits component values.
- Inductor sized for ≤30% ripple; output cap sized for ≤1% ripple voltage; control loop tuned to the requested crossover.
- These are starting points, not final designs — user is expected to override.

### Validation Matrix
- Each template has: default-parameters transient, default-parameters AC sweep, parameter-sweep over 5 representative operating conditions.
- Cross-validation against published reference designs (e.g., TI WEBENCH-style numbers, vendor application notes).

### Python Builder API
- Programmatic equivalent: `pulsim.templates.buck(vin=48, vout=12, ...)` returning a `Circuit` with all wiring.
- Useful for parameter sweeps and Monte Carlo.

### Documentation
- Each template has a docs page with: schematic, default values, design equations, when to use vs alternatives, parity reference.

## Impact

- **New capability**: `converter-templates`.
- **Affected specs**: `converter-templates` (new), `python-bindings` (Python builder API).
- **Affected code**: `core/include/pulsim/v1/templates/`, expansion logic in `core/src/v1/yaml_parser.cpp`, `python/pulsim/templates/`.
- **Backward compat**: existing primitive-only netlists unchanged; templates are additive.

## Success Criteria

1. **First-wave coverage**: 10 templates implemented and validated.
2. **Default-design quality**: each default config produces a stable simulation that meets specified `Vout` within 5% steady-state and crosses over at requested `crossover_hz` within 20%.
3. **Reference parity**: at least 5 templates match published reference designs (TI/STM AN/Infineon AN) within 10% on key waveforms.
4. **Documentation**: each template has a docs page with schematic, equations, and at least one tutorial notebook.
5. **Performance**: template expansion time ≤10 ms per instance for largest template (DAB).
