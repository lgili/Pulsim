# Averaged Converter Modeling

This guide defines the backend contract for averaged-converter transient mode.

## Scope and MVP Limits

Current backend support:

- topologies: `buck`, `boost`, `buck_boost`
- operating modes: `ccm`, `dcm`, `auto`
- execution mode: transient (`Simulator.run_transient(...)`)
- duty command: constant scalar (`simulation.averaged_converter.duty`)
- envelope policy: `strict` or `warn` (CCM-only check)

Current explicit non-goals:

- no automatic averaged-model synthesis for arbitrary topologies
- no implicit fallback to switched transient on averaged failures
- no backend consumption of controller/PWM duty channels in averaged mode yet

## YAML Contract

Configure averaged mode in `simulation.averaged_converter`:

```yaml
simulation:
  tstart: 0.0
  tstop: 350e-6
  dt: 0.2e-6
  averaged_converter:
    enabled: true
    topology: boost            # buck | boost | buck_boost
    operating_mode: auto       # ccm | dcm | auto
    envelope_policy: warn       # strict | warn
    vin_source: Vin             # voltage_source component name
    inductor: L1                # inductor component name
    capacitor: C1               # capacitor component name
    load_resistor: Rload        # resistor component name
    output_node: out            # output node name
    duty: 0.4
    duty_min: 0.0
    duty_max: 0.95
    switching_frequency_hz: 100000.0
    initial_inductor_current: 0.0
    initial_output_voltage: 0.0
    ccm_current_threshold: 0.0
```

Strict parser/runtime checks include:

- missing mapped fields when `enabled=true`
- invalid `topology`, `operating_mode`, or `envelope_policy`
- invalid duty bounds (`0 <= duty_min <= duty <= duty_max <= 1`)
- invalid `ccm_current_threshold`
- invalid `switching_frequency_hz` (`> 0`)
- invalid mapped component types (for example mapping `inductor` to a resistor)

## Python API Surface

```python
import pulsim as ps

parser = ps.YamlParser(ps.YamlParserOptions())
circuit, options = parser.load("examples/10_buck_averaged_mvp_backend.yaml")

options.newton_options.num_nodes = int(circuit.num_nodes())
options.newton_options.num_branches = int(circuit.num_branches())

result = ps.Simulator(circuit, options).run_transient(circuit.initial_state())
print(result.success, result.diagnostic.name, result.message)
```

Related typed enums/options:

- `AveragedConverterOptions`
- `AveragedConverterTopology`
- `AveragedOperatingMode`
- `AveragedEnvelopePolicy`

## Diagnostics

Canonical averaged runtime diagnostics:

- `AveragedInvalidConfiguration`
- `AveragedUnsupportedConfiguration`
- `AveragedOutOfEnvelope`
- `AveragedSolverFailure`

Behavior is deterministic:

- `operating_mode=ccm` + `strict`: fails with `AveragedOutOfEnvelope`
- `operating_mode=ccm` + `warn`: run succeeds with out-of-envelope warning in message
- `operating_mode=dcm|auto`: uses DCM-capable equations and does not raise CCM-envelope failure

## Result Channels and Metadata

Successful averaged runs publish canonical virtual channels:

- `Iavg(<inductor_name>)`
- `Vavg(<output_node>)`
- `Davg`

Current metadata contract for averaged channels (`result.virtual_channel_metadata[...]`):

- `component_type = "averaged_converter"`
- `source_component = "averaged_state"`
- `domain = "time"`
- unit: `A`, `V`, or `ratio`

All channel series are sample-aligned with `result.time`.

## Frontend Responsibilities

Frontend must:

- expose explicit averaged-mode setup UX for all required mapping fields
- show selected envelope policy (`strict`/`warn`) before running
- plot `Iavg(...)`, `Vavg(...)`, `Davg` directly from backend channels
- route channels using metadata first, then canonical names as fallback
- surface typed diagnostics/messages without log regex heuristics

Frontend must not:

- synthesize averaged channels when backend did not emit them
- silently replace averaged mode by switched mode in UI behavior
- infer physical validity outside backend envelope diagnostics
- hide out-of-envelope warnings returned in backend message/diagnostic

## Migration From Switched Transient

Recommended migration path for control-design iteration:

1. Start from a switched converter YAML (`buck`, `boost`, or `buck_boost`) that already has mapped `Vin`, `L`, `C`, `Rload`.
2. Add `simulation.averaged_converter` and keep `duty` fixed for first parity pass.
3. Compare averaged-vs-switching with paired benchmark validation.
4. Select `operating_mode` (`ccm`, `dcm`, or `auto`) based on expected load regime.
5. Use averaged mode for rapid controller sweeps in supported envelope.
6. Keep switched/electrothermal runs as final validation stage.

Reference files:

- switched pair: `benchmarks/circuits/buck_switching_paired.yaml`
- averaged pair: `benchmarks/circuits/buck_averaged_mvp.yaml`
- deterministic expected-failure case: `benchmarks/circuits/buck_averaged_expected_failure.yaml`

## Runnable Examples

YAML:

- `examples/10_buck_averaged_mvp_backend.yaml`

Python:

```bash
PYTHONPATH=build/python python3 examples/run_buck_averaged_mvp.py
```

Benchmark subset + KPI gate:

```bash
PYTHONPATH=build/python python3 benchmarks/benchmark_runner.py \
  --only buck_switching_paired buck_averaged_mvp buck_averaged_expected_failure \
  --output-dir benchmarks/phase14_averaged_artifacts/benchmarks

python3 benchmarks/kpi_gate.py \
  --baseline benchmarks/kpi_baselines/averaged_converter_phase14_2026-03-07/kpi_baseline.json \
  --bench-results benchmarks/phase14_averaged_artifacts/benchmarks/results.json \
  --thresholds benchmarks/kpi_thresholds_averaged.yaml \
  --report-out benchmarks/phase14_averaged_artifacts/reports/kpi_gate_averaged.json \
  --print-report
```
