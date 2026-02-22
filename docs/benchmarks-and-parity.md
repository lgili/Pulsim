# Benchmarks, Paridade e Stress

## Benchmark base (runtime Python)

```bash
PYTHONPATH=build/python python3 benchmarks/benchmark_runner.py \
  --output-dir benchmarks/out
```

## Casos complexos de conversores (foco em convergencia)

```bash
PYTHONPATH=build/python python3 benchmarks/benchmark_runner.py \
  --only buck_switching boost_switching_complex interleaved_buck_3ph buck_mosfet_nonlinear \
  --output-dir benchmarks/out_converters
```

Atalhos no Makefile:

```bash
make benchmark-converters BUILD_DIR=build
make benchmark BUILD_DIR=build LTSPICE_EXE=/Applications/LTspice.app/Contents/MacOS/LTspice
```

## Matriz de cenários (variações de solver/integrador)

```bash
PYTHONPATH=build/python python3 benchmarks/validation_matrix.py \
  --output-dir benchmarks/matrix
```

## Paridade externa com ngspice

```bash
PYTHONPATH=build/python python3 benchmarks/benchmark_ngspice.py \
  --backend ngspice \
  --output-dir benchmarks/ngspice_out
```

## Paridade externa com LTspice

```bash
PYTHONPATH=build/python python3 benchmarks/benchmark_ngspice.py \
  --backend ltspice \
  --ltspice-exe "/Applications/LTspice.app/Contents/MacOS/LTspice" \
  --output-dir benchmarks/ltspice_out
```

## Stress suite por tiers

```bash
PYTHONPATH=build/python python3 benchmarks/stress_suite.py \
  --output-dir benchmarks/stress_out
```

## Gate de paridade GUI -> backend

```bash
PYTHONPATH=build/python pytest -q python/tests/test_gui_component_parity.py
PYTHONPATH=build/python pytest -q python/tests/test_runtime_bindings.py
./build-test/core/pulsim_simulation_tests "[v1][yaml][gui-parity]"
```

## Artefatos de saída

- benchmark: `results.csv`, `results.json`, `summary.json`
- paridade: `parity_results.csv`, `parity_results.json`, `parity_summary.json`
- stress: `stress_results.csv`, `stress_results.json`, `stress_summary.json`

Esses artefatos são o caminho recomendado para validação contínua (CI) e comparação de regressão.
