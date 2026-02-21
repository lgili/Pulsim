# Python API Reference

A superfície suportada para usuários é a API Python do módulo `pulsim`.

## Fluxo principal de uso

1. `YamlParser` carrega netlist e opções.
2. Ajuste fino de `SimulationOptions` em runtime.
3. `Simulator` executa DC/transiente/periódico.
4. `SimulationResult` entrega sinais e telemetria.

## Tipos principais

| Tipo | Papel |
| --- | --- |
| `YamlParserOptions`, `YamlParser` | Parse/validação de YAML `pulsim-v1`. |
| `Circuit` | Grafo/circuito pronto para simulação. |
| `SimulationOptions` | Configuração completa da simulação. |
| `Simulator` | Execução de `dc_operating_point`, `run_transient`, `run_periodic_shooting`, `run_harmonic_balance`. |
| `SimulationResult` | Sinais (`time`, `states`), eventos, telemetria e resumos de perdas/térmico. |

## Exemplo completo

```python
import pulsim as ps

parser = ps.YamlParser(ps.YamlParserOptions())
circuit, options = parser.load("benchmarks/circuits/buck_converter.yaml")

options.newton_options.num_nodes = int(circuit.num_nodes())
options.newton_options.num_branches = int(circuit.num_branches())
options.linear_solver = ps.LinearSolverStackConfig.defaults()
options.integrator = ps.Integrator.TRBDF2

sim = ps.Simulator(circuit, options)
result = sim.run_transient(circuit.initial_state())

print("success:", result.success)
print("steps:", result.total_steps)
print("newton_total:", result.newton_iterations_total)
print("linear_fallbacks:", result.linear_solver_telemetry.total_fallbacks)
```

## Enums importantes

- `Integrator`: `Trapezoidal`, `BDF1..BDF5`, `TRBDF2`, `RosenbrockW`, `SDIRK2`
- `LinearSolverKind`: `SparseLU`, `EnhancedSparseLU`, `KLU`, `GMRES`, `BiCGSTAB`, `CG`
- `PreconditionerKind`: `None_`, `Jacobi`, `ILU0`, `ILUT`, `AMG` (quando disponível)
- `SolverStatus`, `DCStrategy`, `SimulationEventType`, `FallbackReasonCode`

## Configurações que mais importam

### Solver linear

- `LinearSolverStackConfig.order`
- `LinearSolverStackConfig.fallback_order`
- `LinearSolverStackConfig.auto_select`
- `IterativeSolverConfig.preconditioner`, `max_iterations`, `tolerance`

### Newton e convergência

- `NewtonOptions.max_iterations`
- `NewtonOptions.enable_limiting`
- `NewtonOptions.max_voltage_step` / `max_current_step`

### Timestep e integrador

- `SimulationOptions.integrator`
- `SimulationOptions.adaptive_timestep`
- `SimulationOptions.timestep_config`
- `SimulationOptions.lte_config`

### Térmico e perdas

- `SimulationOptions.enable_losses`
- `SimulationOptions.switching_energy`
- `SimulationOptions.thermal`
- `SimulationOptions.thermal_devices`

## Resultados para análise

Campos úteis de `SimulationResult`:

- `time`, `states`
- `events`
- `success`, `final_status`, `message`
- `newton_iterations_total`, `timestep_rejections`
- `linear_solver_telemetry`, `fallback_trace`
- `loss_summary`, `thermal_summary`
