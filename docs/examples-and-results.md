# Examples and Results

Esta seção mostra como rodar exemplos reais e quais resultados observar para validar a simulação.

## Exemplo 1: RC step (sanity check)

```bash
PYTHONPATH=build/python python3 benchmarks/benchmark_runner.py \
  --only rc_step \
  --output-dir benchmarks/out_rc
```

Sinais esperados:

- `V(out)` sobe exponencialmente;
- erro analítico baixo (`max_error` pequeno);
- poucas ou zero rejeições de timestep.

## Exemplo 2: Buck converter

```bash
PYTHONPATH=build/python python3 benchmarks/benchmark_runner.py \
  --only buck_converter \
  --output-dir benchmarks/out_buck
```

Checagens:

- regime estacionário no duty esperado;
- ripple de saída coerente com L/C e frequência;
- sem explosão de `fallbacks`.

## Exemplo 3: Electro-thermal

Use um netlist com bloco térmico ativo e perdas habilitadas.

Indicadores:

- temperatura sobe com perdas;
- `thermal_summary.max_temperature` dentro do esperado;
- eficiência em `loss_summary.efficiency`.

## Resultado estruturado

`results.json` (benchmark) e `parity_results.json` (paridade) são os arquivos principais para automação.

Exemplo simplificado:

```json
{
  "benchmark_id": "buck_converter",
  "status": "passed",
  "runtime_s": 0.42,
  "steps": 19876,
  "max_error": 0.0018,
  "telemetry": {
    "newton_iterations": 53211,
    "timestep_rejections": 21,
    "linear_fallbacks": 3
  }
}
```

## Notebooks recomendados

- `examples/notebooks/02_buck_converter.ipynb`
- `examples/notebooks/03_thermal_modeling.ipynb`
- `examples/notebooks/10_benchmarks.ipynb`

## Comparação com SPICE

```bash
PYTHONPATH=build/python python3 benchmarks/benchmark_ngspice.py \
  --backend ngspice \
  --output-dir benchmarks/ngspice_out
```

```bash
PYTHONPATH=build/python python3 benchmarks/benchmark_ngspice.py \
  --backend ltspice \
  --ltspice-exe "/Applications/LTspice.app/Contents/MacOS/LTspice" \
  --output-dir benchmarks/ltspice_out
```

Compare:

- `max_error` / `rms_error`
- erro de fase (`phase_error_deg`)
- erro em regime permanente (`steady_state_*`)
- tempo total de execução.
