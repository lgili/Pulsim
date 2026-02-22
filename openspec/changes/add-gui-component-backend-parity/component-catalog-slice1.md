# GUI Component Catalog (Slice 1)

This file freezes canonical IDs, aliases, and normalization rules introduced in the first implementation slice.

## Canonical IDs and alias strategy

- Canonical IDs use `snake_case` and are stored in backend/runtime descriptors.
- GUI/YAML aliases are normalized with `lowercase + alnum only` (punctuation/underscore ignored) before lookup.
- Examples:
  - `BJT_NPN`, `bjt-npn` -> `bjt_npn`
  - `CIRCUIT_BREAKER`, `circuit-breaker`, `breaker` -> `circuit_breaker`
  - `SIGNAL_DEMUX`, `demux` -> `signal_demux`

## Parameter normalization rules

### Shared

- Numeric fields accept engineering suffixes (`k`, `milli`, `micro`, `u`, etc.).
- Top-level parameter fields and `params.*` are merged, with top-level taking precedence.
- Alias parameters:
  - `lambda_` -> `lambda`
  - `target_device` -> `target_component`

### Surrogate-mode families (implemented)

- `bjt_npn`, `bjt_pnp`
  - Required pins: 3 (`base, collector, emitter`)
  - `beta` must be > 0 (validation error otherwise)
  - Uses MOSFET surrogate (`kp ~= beta * 1e-3`)
- `thyristor`, `triac`
  - Required pins: 3 (`gate, terminal1, terminal2`)
  - Uses VCSwitch surrogate (`gate_threshold`, `g_on`, `g_off`)
- `fuse`, `circuit_breaker`
  - Required pins: 2
  - Uses switch surrogate (`initial_state`, `g_on`, `g_off`)
- `saturable_inductor`
  - Required pins: 2
  - `inductance` must be > 0
  - Uses linear inductor surrogate for slice 1
- `coupled_inductor`
  - Required pins: 4
  - Turns ratio resolution: `ratio` > `turns_ratio` > `sqrt(l1/l2)`
  - If `l1/l2` path used, both must be > 0

### Virtual runtime families (registered, non-stamping)

- `relay`, `op_amp`, `comparator`, `pi_controller`, `pid_controller`, `math_block`, `pwm_generator`,
  `integrator`, `differentiator`, `limiter`, `rate_limiter`, `hysteresis`, `lookup_table`,
  `transfer_function`, `delay_block`, `sample_hold`, `state_machine`, `voltage_probe`,
  `current_probe`, `power_probe`, `electrical_scope`, `thermal_scope`, `signal_mux`, `signal_demux`
- Stored in `Circuit.virtual_components()` with:
  - canonical `type`
  - normalized pin/node indices
  - extracted numeric params (`numeric_params`)
  - metadata (`metadata`)

## Diagnostics policy (slice 1)

- Unsupported component type: `PULSIM_YAML_E_COMPONENT_UNSUPPORTED`
- Invalid pin count: `PULSIM_YAML_E_PIN_COUNT`
- Invalid parameter range: `PULSIM_YAML_E_PARAM_INVALID`
- Registered virtual component: `PULSIM_YAML_W_COMPONENT_VIRTUAL`
- Surrogate mapping applied: `PULSIM_YAML_W_COMPONENT_SURROGATE`
