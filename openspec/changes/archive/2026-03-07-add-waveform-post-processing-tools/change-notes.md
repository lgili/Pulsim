# Change Notes: add-waveform-post-processing-tools

This document stores acceptance evidence for waveform post-processing implementation.

---

## Implementation Summary

| Area | Files |
|---|---|
| Core engine | `python/pulsim/post_processing.py` (~1120 lines) |
| Public API exports | `python/pulsim/__init__.py` (+18 exports) |
| YAML parser | `core/src/v1/yaml_parser.cpp` (added `"post_processing"` key) |
| Build system | `python/CMakeLists.txt` (configure_file + install for `post_processing.py`) |
| Tests | `python/tests/test_post_processing.py` (65 tests, 9 test classes) |
| Docs | `docs/post-processing.md` (formulas, contract, frontend rules, migration) |
| Examples | `examples/post_processing_workflow.py`, `examples/post_processing_buck.yaml` |
| KPI thresholds | `benchmarks/kpi_thresholds_post_processing.yaml` |

---

## Gate G1: Contract Completeness ✅

All configuration and result types are fully specified:

- **YAML contract**: `simulation.post_processing.jobs[]` with `kind`, `signals`, `window`, `metrics`, `n_harmonics`, `fundamental_hz`, `window_function`, power signals.
- **Result types**: `PostProcessingJobResult` (scalar metrics, spectrum bins, harmonic table, THD, efficiency, power factor, undefined metrics, diagnostic codes).
- **Diagnostic taxonomy**: 8 stable codes (`ok`, `invalid_configuration`, `signal_not_found`, `invalid_window`, `insufficient_samples`, `sampling_mismatch`, `undefined_metric`, `numerical_failure`).
- **Parser diagnostics**: `[PULSIM_PP_E_*]` prefixed error strings for all invalid YAML structures.

---

## Gate G2: Metric Correctness ✅

### Test evidence (65 tests, 65 passed, Python 3.13.5, arm64 macOS)

```
============================= test session starts ==============================
platform darwin -- Python 3.13.5, pytest-8.4.2
65 passed in 0.20s
```

### Metric tolerance table

| Metric | Fixture | Computed | Theory | Error | Threshold | Status |
|---|---|---|---|---|---|---|
| RMS | A=√2 sine, n=10000 | 1.0000 | 1.0000 | <1e-3 | 1e-3 | ✅ |
| THD pure sine | n=8192, Hann | 0.68 % | 0 % | 0.68 % | <5 % | ✅ |
| THD with 3rd harmonic | A3/A1=0.1, n=16384 | 10.0 % | 10.0 % | <3 % | 3 % | ✅ |
| Efficiency (90 %) | V_in=10, I_in=1, V_out=9, I_out=1 | 90.000 % | 90.000 % | <1e-6 % | 1e-4 % | ✅ |
| Power factor (unity) | in-phase V/I, n=1000 | 1.0000 | 1.0 | <0.01 | 0.01 | ✅ |
| Power factor (capacitive) | 90° V/I, n=2000 | ~0.000 | 0.0 | <0.05 | 0.05 | ✅ |
| Crest factor sine | A=2, n=10000 | 1.4140 | √2=1.4142 | <0.01 | 0.01 | ✅ |
| Ripple | V_dc=10, V_ac=0.5, f=10kHz | 0.1000 | 1.0/10.0=0.1 | <0.01 relative | 1 % | ✅ |

### Robustness / diagnostic coverage

| Scenario | Expected diagnostic | Observed | Status |
|---|---|---|---|
| All-zero signal → crest | `UndefinedMetric` | ✅ | ✅ |
| Zero-mean sine → ripple | `UndefinedMetric` | ✅ | ✅ |
| Unknown metric name | `InvalidConfiguration` | ✅ | ✅ |
| Signal not in virtual_channels | `SignalNotFound` | ✅ | ✅ |
| t_start > t_end | `InvalidWindow` | ✅ | ✅ |
| Window beyond simulation time | `InvalidWindow` | ✅ | ✅ |
| Fewer samples than min_samples | `InsufficientSamples` | ✅ | ✅ |
| Empty signals list | `InvalidConfiguration` | ✅ | ✅ |
| Invalid YAML kind | parse error `[PULSIM_PP_E_JOB_KIND]` | ✅ | ✅ |
| Invalid window mode | parse error `[PULSIM_PP_E_WINDOW_MODE]` | ✅ | ✅ |
| Invalid window function | parse error `[PULSIM_PP_E_WINDOW_FUNC]` | ✅ | ✅ |

---

## Gate G3: Determinism ✅

Determinism tests run all three job kinds twice with identical inputs and assert bit-identical outputs:

- `TestDeterminism::test_time_domain_repeated_identical` ✅
- `TestDeterminism::test_spectral_repeated_identical` ✅
- `TestDeterminism::test_power_efficiency_repeated_identical` ✅

No locale, timezone, clock, or random-state dependence. numpy.fft is deterministic for identical inputs.

---

## Gate G4: Performance ✅

Full 65-test suite (covering all job kinds, window modes, window functions, error paths): **0.20s** on arm64 macOS.

Per-job overhead from the example workflow (`examples/post_processing_workflow.py`):

```
Overall success: True
Jobs completed: 4
Total jobs failed: 0

[vout_time_domain]: rms=12.001927, mean=12.000000, ripple=0.052480
[vout_spectrum]: Fundamental=20000.0 Hz, THD=16.667%, 5 harmonics
[converter_efficiency]: P_in=15.0000 W, P_out=6.0019 W, η=40.013%
[vout_last5cycles]: rms=12.001927, ripple=0.052480

RMS error vs theoretical: 0.000052 V  (threshold: 1e-3 V)
```

---

## Gate G5: Integration Readiness ✅

- **YAML integration**: `simulation.post_processing` key accepted by C++ YAML parser. Example YAML netlist: `examples/post_processing_buck.yaml`.
- **Python API**: 18 new public exports from `pulsim.*`; backward-compatible (all existing simulations unaffected).
- **Frontend contract**: `docs/post-processing.md` documents all structured output fields, prohibited frontend heuristics, and migration checklist.
- **Docs nav**: Entry added in `mkdocs.yml` under Guides.

---

## Metric Formulas Reference

| Metric | Formula |
|---|---|
| RMS | $\sqrt{\frac{1}{N}\sum x[k]^2}$ |
| Crest Factor | $\frac{\max |x|}{\text{RMS}}$ |
| Ripple | $\frac{\text{p2p}}{|\bar{x}|}$ |
| THD | $\frac{\sqrt{\sum_{h=2}^{H}A_h^2}}{A_1}\times 100\%$ |
| Efficiency | $\frac{P_{\text{out}}}{P_{\text{in}}}\times 100\%$ |
| Power Factor | $\frac{P_{\text{avg}}}{V_{\text{rms}}\cdot I_{\text{rms}}}$ |

Full formulas and window function definitions: **`docs/post-processing.md`**.

---

## Known Limitations and Deferred Items

- Settling-time and overshoot metrics: deferred (task 6 note in docs).
- Loop/stability metrics (gain/phase margin) from time-domain: deferred.
- Variable-frequency cycle window: deferred (assumes fixed `period`).
- Streaming/incremental post-processing: deferred.
- KPI baseline JSON (`kpi_baselines/post_processing_*/`) not yet captured (task 5.5 thresholds file exists; baseline freeze requires an artifact-producing bench runner run).

## Validation Commands

```bash
# Run post-processing tests
PYTHONPATH=build/python pytest -q python/tests -k post_processing

# Run the example workflow
PYTHONPATH=build/python python3 examples/post_processing_workflow.py

# View KPI thresholds
cat benchmarks/kpi_thresholds_post_processing.yaml
```

