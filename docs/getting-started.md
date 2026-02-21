# Getting Started

## 1) Build local das bindings

```bash
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DPULSIM_BUILD_PYTHON=ON
cmake --build build -j
```

## 2) Rodar via Python usando o build local

```bash
PYTHONPATH=build/python python3 - <<'PY'
import pulsim as ps

parser = ps.YamlParser(ps.YamlParserOptions())
circuit, options = parser.load("benchmarks/circuits/rc_step.yaml")

options.newton_options.num_nodes = int(circuit.num_nodes())
options.newton_options.num_branches = int(circuit.num_branches())

sim = ps.Simulator(circuit, options)
result = sim.run_transient(circuit.initial_state())

print("success:", result.success)
print("steps:", result.total_steps)
PY
```

## 3) Rodar testes principais

```bash
# C++/kernel tests
ctest --test-dir build-test --output-on-failure

# Python runtime tests
PYTHONPATH=build/python pytest python/tests -v --ignore=python/tests/validation
```

## 4) Próximos passos

- Veja o formato YAML completo em `netlist-format`.
- Veja exemplos e notebooks em `notebooks`.
- Valide fidelidade com SPICE em `benchmarks-and-parity`.
