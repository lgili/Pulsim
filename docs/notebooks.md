# Notebooks

Os notebooks ficam em `/examples/notebooks` e usam a biblioteca Python (`import pulsim`).

## Executar um notebook

```bash
PYTHONPATH=build/python python3 -m jupyter notebook
```

Depois abra, por exemplo:

- `examples/notebooks/01_getting_started.ipynb`
- `examples/notebooks/02_buck_converter.ipynb`
- `examples/notebooks/03_thermal_modeling.ipynb`
- `examples/notebooks/10_benchmarks.ipynb`

## Executar em modo não interativo (CI/local)

```bash
PYTHONPATH=build/python MPLBACKEND=Agg python3 -m nbconvert \
  --to notebook \
  --execute \
  --inplace \
  --ExecutePreprocessor.timeout=600 \
  examples/pulsim_tutorial.ipynb
```

## Dicas

- Use o mesmo Python ABI do módulo compilado (`cp313`, `cp314`, etc.).
- Se houver erro de import, confirme o `PYTHONPATH=build/python`.
