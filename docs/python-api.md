# Python API (superfície suportada)

A forma suportada de usar a biblioteca é via Python (`import pulsim`) com netlists YAML.

## Blocos principais

- `YamlParserOptions`, `YamlParser`: carregam e validam netlists.
- `SimulationOptions`: opções de transiente, integrador, solver linear, fallback, térmico.
- `Circuit`: circuito pronto para simulação (gerado do YAML).
- `Simulator`: DC e transiente (`dc_operating_point`, `run_transient`, `run_periodic_shooting`, `run_harmonic_balance`).
- `SimulationResult`: séries temporais, eventos, telemetria de solver, perdas, térmico.

## Exemplo completo

```python
import pulsim as ps

parser = ps.YamlParser(ps.YamlParserOptions())
circuit, options = parser.load("benchmarks/circuits/buck_converter.yaml")

options.newton_options.num_nodes = int(circuit.num_nodes())
options.newton_options.num_branches = int(circuit.num_branches())

# Exemplo: solver stack e integrador stiff
options.linear_solver = ps.LinearSolverStackConfig.defaults()
options.integrator = ps.Integrator.TRBDF2

sim = ps.Simulator(circuit, options)
result = sim.run_transient(circuit.initial_state())

print("ok:", result.success)
print("steps:", result.total_steps)
print("newton_iters:", result.newton_iterations_total)
print("fallbacks:", result.linear_solver_telemetry.total_fallbacks)
```

## Solvers lineares (runtime stack)

`LinearSolverStackConfig` permite ordem preferida e fallback:

- ordem direta/iterativa (`order`, `fallback_order`);
- auto seleção por tamanho (`auto_select`, `size_threshold`, `nnz_threshold`);
- precondicionador iterativo (`Jacobi`, `ILU0`, `ILUT`, `AMG` quando disponível).

## Térmico e perdas

- habilite em `options.thermal` e `options.thermal_devices`;
- perdas de chaveamento em `options.switching_energy`;
- resultado em `result.loss_summary` e `result.thermal_summary`.
