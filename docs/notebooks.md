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
- `examples/notebooks/28_electrothermal_component_telemetry.ipynb`
- `examples/notebooks/29_electrothermal_yaml_validation_modes.ipynb`
- `examples/notebooks/30_electrothermal_kpi_gate_consistency.ipynb`
- `examples/notebooks/31_frequency_analysis_ac_sweep_spec_walkthrough.ipynb`
- `examples/notebooks/10_benchmarks.ipynb`

Advanced catalog/coverage notebooks:

- `15_new_components_catalog.ipynb`
- `16_control_blocks_mixed_domain.ipynb`
- `17_protection_magnetics_probes.ipynb`
- `18_buck_mosfet_pwm_block.ipynb`

Magnetic-core addition:

- `35_magnetic_core_mvp_tutorial.ipynb`

Electrothermal-focused additions:

- `28_electrothermal_component_telemetry.ipynb`
- `29_electrothermal_yaml_validation_modes.ipynb`
- `30_electrothermal_kpi_gate_consistency.ipynb`

Frequency-analysis addition:

- `31_frequency_analysis_ac_sweep_spec_walkthrough.ipynb`

Post-processing and C-Block additions:

- `33_post_processing_tutorial.ipynb`
- `34_cblock_runtime_validation.ipynb`

## Execute Headless (CI/local automation)

```bash
PYTHONPATH=build/python MPLBACKEND=Agg python3 -m nbconvert \
  --to notebook \
  --execute \
  --inplace \
  --ExecutePreprocessor.timeout=600 \
  examples/pulsim_tutorial.ipynb
```

Magnetic-core tutorial headless run:

```bash
PYTHONPATH=build/python MPLBACKEND=Agg python3 -m nbconvert \
  --to notebook \
  --execute \
  --inplace \
  --ExecutePreprocessor.timeout=600 \
  examples/notebooks/35_magnetic_core_mvp_tutorial.ipynb
```

## Notebook Environment Tips

- Keep notebook Python ABI aligned with the built extension (e.g., `cp313`, `cp314`).
- If import fails, validate `PYTHONPATH=build/python` first.
- Prefer deterministic seeds and fixed benchmark manifests when publishing figures.
