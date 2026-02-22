# Pulsim User Guide (Python-Only)

Pulsim é um simulador de eletrônica de potência com kernel unificado v1 e runtime
suportado em Python.

## Escopo suportado

- Python runtime: `import pulsim`
- Netlist versionado em YAML (`schema: pulsim-v1`)
- Ferramentas de benchmark/paridade/stress em `benchmarks/`

Não faz parte da superfície suportada:

- workflows legacy de CLI
- gRPC como caminho principal de uso
- carregamento de netlist JSON

## Fluxo recomendado

1. Compile as bindings Python (`cmake ... -DPULSIM_BUILD_PYTHON=ON`).
2. Exporte `PYTHONPATH=build/python`.
3. Carregue YAML com `YamlParser`.
4. Rode `Simulator(...).run_transient(...)`.
5. Valide com benchmark/paridade/stress.

## Exemplo mínimo

```python
import pulsim as ps

parser = ps.YamlParser(ps.YamlParserOptions())
circuit, options = parser.load("benchmarks/circuits/rc_step.yaml")

options.newton_options.num_nodes = int(circuit.num_nodes())
options.newton_options.num_branches = int(circuit.num_branches())

sim = ps.Simulator(circuit, options)
result = sim.run_transient(circuit.initial_state())

print("ok:", result.success, "steps:", result.total_steps)
```

## Onde continuar

- Formato YAML completo: `netlist-format`
- API Python: `python-api`
- Benchmark/paridade/LTspice: `benchmarks-and-parity`
- Matriz GUI -> backend: `gui-component-parity`
- Migração e compatibilidade: `migration-guide`
