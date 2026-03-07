# Supported Components Catalog

This page is the canonical user-facing inventory of YAML component types currently supported by PulsimCore v1.

- Catalog snapshot: `2026-03-07`
- Canonical source in backend parser: `component_node_arity()` (`core/src/v1/yaml_parser.cpp`)
- Regression gate: `python/tests/test_supported_components_catalog.py`

## Electrical Components

| Category | Canonical Type | Notes |
| --- | --- | --- |
| Passive | `resistor` | Linear resistor |
| Passive | `capacitor` | Capacitor with optional initial condition |
| Passive | `inductor` | Inductor with optional initial condition |
| Sources | `voltage_source` | Supports waveform definitions (`dc`, `pulse`, `sine`, `pwm`) |
| Sources | `current_source` | DC current source |
| Switching/Power | `diode` | Ideal diode model |
| Switching/Power | `switch` | Ideal controlled switch |
| Switching/Power | `vcswitch` | Voltage-controlled switch |
| Switching/Power | `mosfet` | MOSFET model |
| Switching/Power | `igbt` | IGBT model |
| Magnetics | `transformer` | Coupled ideal transformer |
| Switching/Power | `snubber_rc` | RC snubber macro |
| Switching/Power | `bjt_npn` | Surrogate behavior using backend switch/power path |
| Switching/Power | `bjt_pnp` | Surrogate behavior using backend switch/power path |
| Switching/Power | `thyristor` | Event/latch controlled device |
| Switching/Power | `triac` | Bidirectional event/latch controlled device |
| Protection | `fuse` | Event-based trip with I²t logic |
| Protection | `circuit_breaker` | Event-based overcurrent + delay trip |
| Protection | `relay` | Coil/event + contact switching model |
| Magnetics | `saturable_inductor` | Nonlinear inductance controller over electrical branch |
| Magnetics | `coupled_inductor` | Coupled inductor controller over two branches |

## Control and Signal Components

| Category | Canonical Type | Notes |
| --- | --- | --- |
| Control | `op_amp` | Virtual control block |
| Control | `comparator` | Virtual control block |
| Control | `pi_controller` | Virtual control block |
| Control | `pid_controller` | Virtual control block |
| Control | `gain` | Virtual control block |
| Control | `sum` | Virtual control block |
| Control | `subtraction` | Virtual control block |
| Control | `math_block` | Virtual control block |
| Control | `pwm_generator` | Virtual control block |
| Control | `integrator` | Virtual control block |
| Control | `differentiator` | Virtual control block |
| Control | `limiter` | Virtual control block |
| Control | `rate_limiter` | Virtual control block |
| Control | `hysteresis` | Virtual control block |
| Control | `lookup_table` | Virtual control block |
| Control | `transfer_function` | Virtual control block |
| Control | `delay_block` | Virtual control block |
| Control | `sample_hold` | Virtual control block |
| Control | `state_machine` | Virtual control block |
| Signal Routing | `signal_mux` | Virtual routing block |
| Signal Routing | `signal_demux` | Virtual routing block |

## Probes and Scopes

| Category | Canonical Type | Notes |
| --- | --- | --- |
| Instrumentation | `voltage_probe` | Virtual instrumentation channel |
| Instrumentation | `current_probe` | Virtual instrumentation channel |
| Instrumentation | `power_probe` | Virtual instrumentation channel |
| Instrumentation | `electrical_scope` | Virtual grouped electrical channels |
| Instrumentation | `thermal_scope` | Virtual grouped thermal channels |

## Thermal-Port Eligible Components

`component.thermal.enabled: true` is supported for:

- `resistor`
- `diode`
- `mosfet`
- `igbt`
- `bjt_npn`
- `bjt_pnp`

See [Electrothermal Workflow](electrothermal-workflow.md) for the complete runtime contract (`T(...)` channels, summaries, and consistency checks).

## Regression and CI Gate

Use the catalog regression test to guard support drift:

```bash
PYTHONPATH=build/python pytest -q python/tests/test_supported_components_catalog.py
```

