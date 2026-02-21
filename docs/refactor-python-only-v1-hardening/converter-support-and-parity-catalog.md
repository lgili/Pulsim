# Converter Support Matrix and LTspice Parity Catalog

This matrix is the acceptance baseline for converter-focused simulations in
`refactor-python-only-v1-hardening`.

## Converter Support Matrix

Legend:
- `yes`: implemented and validated in active path.
- `partial`: available but missing full converter-grade validation/coupling.
- `no`: missing in declared scope.

| capability | runtime v1 | yaml | python bindings | validation coverage | status |
|---|---|---|---|---|---|
| Resistor / Capacitor / Inductor | yes | yes | yes | analytical + benchmark | yes |
| Voltage / Current sources (dc, pulse, sine, pwm) | yes | yes | yes | benchmark + regression | yes |
| Diode | yes | yes | yes | nonlinear tests + benchmarks | yes |
| Ideal switch / vc switch | yes | yes | yes | switching regressions | yes |
| MOSFET | yes | yes | yes | nonlinear tests + converter tests | yes |
| IGBT | yes | yes | yes | device tests; limited parity coverage | partial |
| Transformer | yes | yes | yes | limited converter validation | partial |
| Loss accumulation telemetry | yes | yes (options/results) | yes | v1 regression tests | yes |
| Thermal network simulation | yes | n/a (runtime utility) | yes | dedicated thermal tests | partial |
| Coupled electro-thermal converter run | partial | partial | partial | missing end-to-end converter gate | partial |
| Periodic steady-state (shooting / HB) | yes | yes | yes | steady-state tests + benchmark | yes |

## Required Gaps Before Gate G3/G4

- Complete coupled electro-thermal converter validation (not only standalone thermal utilities).
- Increase parity-grade validation for IGBT and transformer converter scenarios.
- Add explicit YAML coupling controls for thermal-enabled converter workflows.

## LTspice Parity Catalog (Acceptance Draft)

Each mapped case requires:
- Pulsim YAML netlist.
- LTspice netlist/source.
- Observable mapping.
- Thresholds (`max_error`, `rms_error`, optional steady-state metrics).

| benchmark_id | pulsim_netlist | ltspice_source | observables | gate |
|---|---|---|---|---|
| rc_step | `benchmarks/circuits/rc_step.yaml` | `benchmarks/ltspice/rc_step.asc` (or exported `.cir`) | `V(out)` | tier_a |
| rl_step | `benchmarks/circuits/rl_step.yaml` | `benchmarks/ltspice/rl_step.asc` | `V(out)` | tier_a |
| rlc_step | `benchmarks/circuits/rlc_step.yaml` | `benchmarks/ltspice/rlc_step.asc` | `V(out)` | tier_a |
| diode_rectifier | `benchmarks/circuits/diode_rectifier.yaml` | `benchmarks/ltspice/diode_rectifier.asc` | `V(out)`, `I(D1)` | tier_b |
| buck_switching | `benchmarks/circuits/buck_switching.yaml` | `benchmarks/ltspice/buck_switching.asc` | `V(out)`, `V(sw)`, `I(L1)` | tier_b |
| stiff_rlc | `benchmarks/circuits/stiff_rlc.yaml` | `benchmarks/ltspice/stiff_rlc.asc` | `V(out)` | tier_b |
| boost_converter | `benchmarks/circuits/boost_converter.yaml` (to add) | `benchmarks/ltspice/boost_converter.asc` | `V(out)`, `I(L1)`, switch current | tier_c |
| flyback_converter | `benchmarks/circuits/flyback_converter.yaml` (to add) | `benchmarks/ltspice/flyback_converter.asc` | primary/secondary currents, `V(out)` | tier_c |
| thermal_buck | `benchmarks/circuits/thermal_buck.yaml` (to add) | `benchmarks/ltspice/thermal_buck.asc` | electrical + junction temperature | tier_c |

## Acceptance Criteria by Tier

- `tier_a`: analytical and LTspice parity thresholds, deterministic results.
- `tier_b`: robust convergence and fallback telemetry, LTspice waveform parity.
- `tier_c`: large/stiff converter stress, coupled electro-thermal checks, deterministic status fields.
