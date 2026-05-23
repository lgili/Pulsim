## Phase 1 — Loader header (~0.5 days)

- [x] 1.1 New `pulsim/v2/yaml/loader.hpp` with
      `LoadedCircuit`, `load_file`, `load_string`.
- [x] 1.2 Implement device dispatch by `type:` string
      for all 11 device types.
- [x] 1.3 Implement optional `simulation:` block parsing.
- [x] 1.4 Validation: required-field checks + clear
      error messages.

## Phase 2 — C++ tests (~0.4 days)

- [x] 2.1 Each device type round-trips (12+ unit tests).
- [x] 2.2 simulation: block populates SimulationOptions.
- [x] 2.3 Missing required field throws.
- [x] 2.4 Unknown device type throws.
- [x] 2.5 Integration: load buck.yaml + run simulation.
- [x] 2.6 Direct vs YAML equivalence test.

## Phase 3 — Python bindings + tests (~0.2 days)

- [x] 3.1 `pulsim.v2.load_yaml_file(path)` binding.
- [x] 3.2 `pulsim.v2.load_yaml_string(text)` binding.
- [x] 3.3 Python smoke tests for both.

## Phase 4 — Sample YAMLs (~0.2 days)

- [x] 4.1 `examples/v2/half_wave_rectifier.yaml`.
- [x] 4.2 `examples/v2/buck.yaml`.
- [x] 4.3 `examples/v2/flyback.yaml`.

## Phase 5 — Regression + docs (~0.1 days)

- [x] 5.1 All previous v2 tests stay green.
- [x] 5.2 `openspec validate pulsim-v2-yaml-parser
      --strict` passes.
- [x] 5.3 `docs/pulsim-v2/layer8-yaml-parser.md`.
