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

### Before/After: legacy backend -> canonical mode

Before (legacy, removed in strict migration path):

```yaml
simulation:
  backend: auto
  sundials:
    enabled: true
    family: ida
  adaptive_timestep: true
  dt: 1e-7
```

After (canonical fixed-step native core):

```yaml
simulation:
  step_mode: fixed
  dt: 1e-7
  dt_min: 1e-9
  dt_max: 1e-7
```

After (canonical variable-step native core):

```yaml
simulation:
  step_mode: variable
  dt: 1e-7
  dt_min: 1e-9
  dt_max: 2e-6
  timestep:
    preset: power_electronics
```

If `strict = True`, legacy backend keys produce parser diagnostic
`PULSIM_YAML_E_LEGACY_TRANSIENT_BACKEND`.

### Schema Evolution Policy (v1)

- Deprecated (migration window): `simulation.adaptive_timestep`
  - Accepted in schema `pulsim-v1`, but emits warning
    `PULSIM_YAML_W_DEPRECATED_FIELD` with replacement guidance to
    `simulation.step_mode: fixed|variable`.
- Removed (strict migration path): `simulation.backend`,
  `simulation.sundials`, `simulation.advanced.backend`,
  `simulation.advanced.sundials`
  - In strict mode, parser fails deterministically with
    `PULSIM_YAML_E_LEGACY_TRANSIENT_BACKEND` and migration guidance.

### Before/After: procedural compatibility and canonical runtime

Before (procedural path, still supported in migration window):

```python
import pulsim as ps

times, states, ok, msg = ps.run_transient(
    circuit, 0.0, 1e-3, 1e-6, circuit.initial_state()
)
```

After (canonical class/runtime surface):

```python
import pulsim as ps

opts = ps.SimulationOptions()
opts.tstart = 0.0
opts.tstop = 1e-3
opts.dt = 1e-6

sim = ps.Simulator(circuit, opts)
result = sim.run_transient(circuit.initial_state())
```

Both paths share the same v1 kernel semantics; prefer `Simulator` for new code.

### Before/After: expert override location

Before (mixed top-level knobs):

```yaml
simulation:
  step_mode: variable
  integrator: trbdf2
  solver:
    order: [gmres]
```

After (canonical mode + explicit expert section):

```yaml
simulation:
  step_mode: variable
  advanced:
    integrator: trbdf2
    solver:
      order: [gmres]
      iterative:
        preconditioner: ilut
        max_iterations: 300
        tolerance: 1e-8
```

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

## 8. Numerical Surface — v0.10 → v0.11

The `simplify-and-harden-numerical-surface` change collapses the
50-field tuning surface into a single `Preset` enum + four
profiles, deprecates dead integrators + redundant booleans, and adds
four automatic convergence aids (Armijo line search, simultaneous
event coalescence, iterative refinement, homotopy continuation) that
all engage by default.

Full reference: [Numerical Configuration](numerical-configuration.md).
The migration recipes below cover only the changes that affect
existing user code.

### 8.1 `make_robust_options(...)` → `SimulationOptions.from_preset(...)`

```python
# Before (still works):
opts = ps.make_robust_options(circuit, 0.0, 1e-3, 1e-6,
                                ps.NewtonOptions(),
                                ps.LinearSolverStackConfig.defaults())

# After (recommended):
opts = ps.SimulationOptions.from_preset(ps.Preset.Robust,
                                          dt=1e-6, tstop=1e-3)
```

The two paths produce numerically identical configurations for the
Robust profile. `from_preset()` adds three more named profiles
(`Auto`, `Fast`, `HighFidelity`) on the same surface.

### 8.2 `adaptive_timestep` → `step_mode`

YAML:
```yaml
# Before (emits PULSIM_YAML_W_DEPRECATED_FIELD):
simulation:
  adaptive_timestep: true

# After:
simulation:
  step_mode: variable        # or "fixed"
```

Python:
```python
# Before (still works, will be removed in next major release):
opts.adaptive_timestep = True

# After:
opts.step_mode = ps.StepMode.Variable   # or StepMode.Fixed
```

### 8.3 `direct_formulation_fallback` → just remove it

The DAE fallback is now **unconditionally on internally**. The field
is a no-op.

YAML:
```yaml
# Before (emits PULSIM_YAML_W_DEPRECATED_FIELD):
simulation:
  direct_formulation_fallback: true

# After: just remove the field.
```

### 8.4 Deprecated integrators

| Removed (warning in v0.11, error in next major) | Replacement |
|---|---|
| `Integrator.BDF3` / `BDF4` / `BDF5` | `Integrator.BDF2`, `TRBDF2`, or `RosenbrockW` |
| `Integrator.Gear` | `Integrator.BDF2` (literal alias) |
| `Integrator.SDIRK2` | `Integrator.TRBDF2` |

YAML:
```yaml
# Before:
simulation:
  integrator: bdf5

# After:
simulation:
  integrator: trbdf2     # for stiff dynamics with order ≥ 2
  # or: bdf2 / rosenbrockw
```

### 8.5 New telemetry counters (additive — no migration needed)

Four new diagnostic counters appear in `result.backend_telemetry` /
`result.linear_solver_telemetry` / `result.dc_result`:

| Counter | Where | Meaning |
|---|---|---|
| `simultaneous_event_groups` | `backend_telemetry` | Steps where ≥ 2 PWL devices commuted atomically |
| `linear_refinement_steps` | `linear_solver_telemetry` | Direct-solve residual exceeded threshold; one round of iterative refinement fired |
| `homotopy_steps` | `dc_result` | Lambda increments executed in the homotopy continuation |
| `homotopy_ladder_completed` | `dc_result` | True if homotopy reached `λ = 1` |
| `line_search_backtracks` | `newton_result.telemetry` | Already existed; now reflects Armijo backtracks |

If you have CI gates that pin telemetry-counter values, audit them
against the new fields.

### 8.6 Convergence improvements are silent (no migration needed)

The four convergence aids (Armijo line search inside Newton,
simultaneous PWL event coalescence, KLU iterative refinement,
homotopy DC OP) all engage **automatically** when their trigger
conditions are met. No user-facing knobs to flip. Circuits that
previously failed convergence on a cold-start may now succeed; that's
the intended improvement.

If a benchmark of yours regresses because the new behavior produces
a numerically slightly-different trajectory, set
`opts.newton_options.armijo_line_search = False` and
`opts.dc_config.homotopy_config.enable = False` to reproduce the
v0.10 behavior. Reach out via the issue tracker so we can pin the
regression as a parity test.
