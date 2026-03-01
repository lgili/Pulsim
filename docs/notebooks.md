# Notebooks

Jupyter notebooks live in `examples/notebooks` and use the same backend runtime (`import pulsim`) used by scripts and CI.

## Launch Locally

```bash
PYTHONPATH=build/python python3 -m jupyter notebook
```

Recommended sequence:

- `examples/notebooks/00_notebook_index.ipynb`
- `examples/notebooks/01_getting_started.ipynb`
- `examples/notebooks/02_buck_converter.ipynb`
- `examples/notebooks/03_thermal_modeling.ipynb`
- `examples/notebooks/10_benchmarks.ipynb`

Advanced catalog/coverage notebooks:

- `15_new_components_catalog.ipynb`
- `16_control_blocks_mixed_domain.ipynb`
- `17_protection_magnetics_probes.ipynb`
- `18_buck_mosfet_pwm_block.ipynb`

## Execute Headless (CI/local automation)

```bash
PYTHONPATH=build/python MPLBACKEND=Agg python3 -m nbconvert \
  --to notebook \
  --execute \
  --inplace \
  --ExecutePreprocessor.timeout=600 \
  examples/pulsim_tutorial.ipynb
```

## Notebook Environment Tips

- Keep notebook Python ABI aligned with the built extension (e.g., `cp313`, `cp314`).
- If import fails, validate `PYTHONPATH=build/python` first.
- Prefer deterministic seeds and fixed benchmark manifests when publishing figures.
