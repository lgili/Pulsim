# Pulsim Python Examples

Runnable Python scripts that exercise every user-facing feature shipped
in the May 9–10 roadmap (Fase 0 → Fase 4). Each script is a single file
with a TL;DR docstring at the top, sensible defaults, and an optional
matplotlib plot guarded by `PULSIM_EXAMPLE_NOPLOT=1` so it runs cleanly
in headless CI.

## Running

```bash
# Use the version of pulsim already installed in your environment:
python 01_ac_sweep_rc.py

# Or, if running against a local build tree:
PYTHONPATH=build_py/python python3 01_ac_sweep_rc.py

# Skip plot generation (useful for headless CI):
PULSIM_EXAMPLE_NOPLOT=1 python 01_ac_sweep_rc.py
```

## Index

| # | Script | Demonstrates | Doc page |
|---|---|---|---|
| 01 | `01_ac_sweep_rc.py` | Analytical AC sweep + Bode plot, validates `-3 dB / -45°` at the RC corner. | [`docs/ac-analysis.md`](../../docs/ac-analysis.md) |
| 02 | `02_fra_vs_ac_sweep.py` | Empirical FRA (transient + DFT) overlaid on the analytical AC sweep — confirms the `≤ 1 dB / ≤ 5°` agreement contract. | [`docs/fra.md`](../../docs/fra.md) |
| 03 | `03_buck_template.py` | `pulsim.templates.buck(...)` auto-design + DC OP + transient on a synchronous buck. | [`docs/converter-templates.md`](../../docs/converter-templates.md) |
| 04 | `04_boost_buckboost_templates.py` | Boost and buck-boost templates side by side, comparing auto-designed `L / C / D`. | [`docs/converter-templates.md`](../../docs/converter-templates.md) |
| 05 | `05_codegen_pil.py` | `pulsim.codegen.generate(...)` → C99 → gcc → PIL parity diff against the in-Python A_d/B_d evolution (rel. error ≤ 1e-6). | [`docs/code-generation.md`](../../docs/code-generation.md) |
| 06 | `06_fmu_export.py` | `pulsim.fmu.export(...)` to FMI 2.0 Co-Simulation `.fmu`, inspects the zip layout + `modelDescription.xml`, ctypes-loads the lib and confirms 13/13 FMI 2.0 CS callback symbols. | [`docs/fmi-export.md`](../../docs/fmi-export.md) |
| 07 | `07_parameter_sweep_lhs.py` | 64-point LHS over `(L, C)` with steady-state / peak / settling-time metrics, P5/P50/P95 percentiles. | [`docs/parameter-sweep.md`](../../docs/parameter-sweep.md) |
| 08 | `08_monte_carlo_yield.py` | 256 Monte-Carlo draws under ±5 % L/C, ±1 % R tolerance — yield analysis vs a `Vout ∈ 5 V ± 2 %` spec window. | [`docs/parameter-sweep.md`](../../docs/parameter-sweep.md) |
| 09 | `09_robustness_wrapper.py` | `pulsim.run_transient(robust=True)` retry loop + Newton/linear-solver auto-tune. | [`docs/robustness-policy.md`](../../docs/robustness-policy.md) |
| 10 | `10_linear_solver_cache.py` | `BackendTelemetry.linear_factor_cache_*` hit-rate readout on a switching converter (≥ 97 % hit rate after warm-up). | [`docs/linear-solver-cache.md`](../../docs/linear-solver-cache.md) |
| 11 | `11_yaml_ac_analysis.py` | Load circuit from YAML via `pulsim.YamlParser` then dispatch a Python-built `AcSweepOptions` (the `analysis:` block parser is shipped C++-side; the Python binding for `options.ac_sweeps` is a deferred follow-up). | [`docs/ac-analysis.md`](../../docs/ac-analysis.md) |

## Outputs

Each example prints results to stdout and (unless `PULSIM_EXAMPLE_NOPLOT`
is set) writes a PNG next to the script. PNG outputs are git-ignored;
they're regenerated on every run.

## Dependencies

| Required | When |
|---|---|
| `pulsim` (the package itself) | always |
| `numpy` | always (used by `pulsim` internally) |
| `scipy` | examples 05, 07, 08 (for `expm` and `qmc`) |
| `matplotlib` | optional — gated by `PULSIM_EXAMPLE_NOPLOT` |
| `pandas` | optional, used by `SweepResult.to_pandas()` in 07, 08 |
| A C compiler (`gcc`/`cc`) | examples 05, 06 (PIL parity + FMU compile) |

## C++ examples

Header-only features (magnetic models, motors, three-phase grid,
robustness profile) don't have Python bindings yet. Their runnable
counterparts live in [`../cpp/`](../cpp/).
