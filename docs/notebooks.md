# Notebooks

Os notebooks ficam em `/examples/notebooks` e usam a biblioteca Python (`import pulsim`).

## Executar um notebook

```bash
PYTHONPATH=build/python python3 -m jupyter notebook
```

Depois abra, por exemplo:

- `examples/notebooks/00_notebook_index.ipynb`
- `examples/notebooks/01_getting_started.ipynb`
- `examples/notebooks/02_buck_converter.ipynb`
- `examples/notebooks/03_thermal_modeling.ipynb`
- `examples/notebooks/forward_converter_design.ipynb`
- `examples/notebooks/10_benchmarks.ipynb`
- `examples/notebooks/15_new_components_catalog.ipynb`
- `examples/notebooks/16_control_blocks_mixed_domain.ipynb`
- `examples/notebooks/17_protection_magnetics_probes.ipynb`

## Novos notebooks (componentes GUI/backend)

- `00_notebook_index.ipynb`: hub com trilha recomendada (base -> conversores -> componentes novos -> benchmark/validacao).
- `15_new_components_catalog.ipynb`: valida o parser com todos os componentes novos.
- `16_control_blocks_mixed_domain.ipynb`: exemplos de blocos de controle e roteamento de sinais.
- `17_protection_magnetics_probes.ipynb`: eventos de protecao/chaveamento, magneticos e probes.

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
