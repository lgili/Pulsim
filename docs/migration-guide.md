# Migration Guide: Python-Only v1 Runtime

This guide documents the migration to the unified v1 kernel with a Python-only user-facing runtime.

## 1. What Changed

### Removed / Unsupported User Surfaces

- Legacy CLI execution flow
- Legacy gRPC server/client workflow
- JSON netlist loading path

### Supported User Surface

- Python package `pulsim`
- YAML netlists (`schema: pulsim-v1`)
- Python benchmark/parity/stress tooling in `benchmarks/`

## 2. Netlist Migration (JSON -> YAML)

JSON loaders are no longer part of the supported runtime path.

Use versioned YAML:

```yaml
schema: pulsim-v1
version: 1
components:
  - type: resistor
    name: R1
    nodes: [in, out]
    value: 1k
```

## 3. Runtime Migration

### Before (removed)

- `pulsim` CLI command flows
- Remote gRPC client/server product workflow

### After (supported)

```python
import pulsim as ps

parser = ps.YamlParser(ps.YamlParserOptions())
circuit, options = parser.load("circuit.yaml")

options.newton_options.num_nodes = int(circuit.num_nodes())
options.newton_options.num_branches = int(circuit.num_branches())

sim = ps.Simulator(circuit, options)
result = sim.run_transient(circuit.initial_state())
```

### Legacy transient backend keys

No runtime suportado, a escolha de caminho transiente é canônica por modo:

- `simulation.step_mode: fixed`
- `simulation.step_mode: variable`

As chaves legadas `simulation.backend` / `simulation.sundials` (e equivalentes em
`simulation.advanced`) são tratadas apenas para diagnóstico de migração.

## 4. Removed API/Workflow Mapping

| Removed workflow | Replacement |
| --- | --- |
| CLI `run/validate/info/sweep` | Python runtime + `benchmarks/*.py` runners |
| gRPC remote simulation docs | Local Python runtime usage |
| JSON netlist loader docs/tests | YAML parser (`YamlParser`) |
| Planned placeholder high-level suites | Active runtime/benchmark/validation suites |

## 5. Versioned Deprecation Timeline

| Version | Status | Notes |
| --- | --- | --- |
| `v0.2.0` | Deprecation window | Python-only surface declared; legacy docs marked stale |
| `v0.3.0` | Removal | Legacy CLI/gRPC/JSON user-facing guidance removed from primary docs |
| `v0.4.0` | Enforcement | Supported workflows restricted to Python + YAML + benchmark/parity/stress toolchain |

## 6. Migration Notes: PulsimGui Converter Integration

### Canonicalização de tipos

O conversor do PulsimGui deve emitir tipos que o parser normaliza para IDs
canônicos (ex.: `OP_AMP` -> `op_amp`, `PI_CONTROLLER` -> `pi_controller`,
`CIRCUIT_BREAKER` -> `circuit_breaker`).

Recomendação: sempre serializar o tipo canônico em minúsculo para reduzir
ambiguidade no pipeline GUI -> YAML -> backend.

### Regras de modelagem no backend

- `bjt_npn` e `bjt_pnp`: surrogate interno baseado em `mosfet`.
- `thyristor`, `triac`, `fuse`, `circuit_breaker`: composição com `switch` e
  controlador virtual/event-driven.
- `relay`: composição com dois `switch` (`NO`/`NC`) + controlador virtual da bobina.
- `saturable_inductor`: `inductor` elétrico + controlador virtual de indutância efetiva.
- `coupled_inductor`: dois ramos `inductor` + controlador virtual de acoplamento.
- `voltage_probe/current_probe/power_probe/scope/mux/demux`: componentes
  virtuais (não estampam MNA).

### Pinagem e validação

Ative strict mode no parser durante integração:

```python
opts = ps.YamlParserOptions()
opts.strict = True
parser = ps.YamlParser(opts)
```

Isso garante diagnóstico estável para:

- pinagem inválida (ex.: `relay` sem `COM/NO/NC`);
- parâmetros fora de faixa (ex.: `duty_min > duty_max`);
- blocos de controle com configuração inconsistente.

### Gate mínimo para CI do conversor

```bash
PYTHONPATH=build/python pytest -q python/tests/test_gui_component_parity.py
PYTHONPATH=build/python pytest -q python/tests/test_runtime_bindings.py
./build-test/core/pulsim_simulation_tests "[v1][yaml][gui-parity]"
```

## 7. Upgrade Checklist

1. Replace any JSON netlist assets with YAML `pulsim-v1` netlists.
2. Remove CLI automation and migrate to Python runners.
3. Remove gRPC-dependent user scripts from active workflows.
4. Update CI jobs to run Python benchmark/parity/stress scripts.
5. Add GUI parity regression gate (`test_gui_component_parity.py`) in CI.
6. Validate with `openspec validate refactor-python-only-v1-hardening --strict`.
