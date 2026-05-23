## Why

Small-signal frequency-domain (AC) analysis is the standard workflow for **control-loop design** in switching converters: linearise the plant around a DC operating point, sweep frequency, plot Bode (magnitude/phase), tune compensator. v2 today has only **transient** simulation. Designing a Type-II / Type-III compensator for a buck/boost requires either an external tool or trial-and-error transient probes.

v1 has `frequency_analysis_phase*` infrastructure. v2 should ship the equivalent for its devices.

## What Changes

- **ADD** a Layer 11 `pulsim::v2::ac` namespace containing:
  - `ac::linearise_at(...)` — given a DC operating point `x_op`, returns `(A, B, C, D)` state-space matrices of the linearised system around `x_op`.
  - `ac::AcSweepResult run_ac_sweep(graph, pool, x_op, freqs, input_node, output_node)` — sweeps frequency, returns complex transfer function `H(jω)` at each frequency.
  - `ac::bode_data(result)` — extracts `(magnitude_dB, phase_deg)` arrays for plotting.
- **ADD** small-signal linearisation hooks in nonlinear devices (`MosfetLevel1`, `IgbtLevel1`, `IdealDiode`, `SaturableInductor`) — each returns its `∂I/∂V` Jacobian at the operating point, which feeds the A matrix.
- **ADD** Python bindings: `pulsim.v2.run_ac_sweep(...)`, returns numpy arrays for easy plotting.
- **ADD** YAML schema `analysis: ac_sweep` block with `f_start`, `f_end`, `n_points`, `points_per_decade`, `input`, `output`.
- **ADD** Showcase: AC sweep of a buck converter open-loop control-to-output transfer function, verify the LC double-pole + ESR zero analytically.

## Impact

- **Affected specs:** new `pulsim-v2-ac-analysis` capability.
- **Affected code:**
  - `core/include/pulsim/v2/ac/linearise.hpp` (new)
  - `core/include/pulsim/v2/ac/run_ac_sweep.hpp` (new)
  - `core/include/pulsim/v2/ac/bode.hpp` (new)
  - `core/include/pulsim/v2/yaml/loader.hpp` (extend with `analysis:` block)
  - `python/bindings_v2_kernel.cpp`
  - `core/tests/v2/layer11/*` (new test target)
  - `examples/v2/buck_ac_sweep.yaml` (new)
- **Risk:** Linearisation around a DC OP requires `compute_dc_op` to work, which currently fails for V11/V12 time-varying sources. Mitigation: provide an explicit `x_op` override path; DC OP fix can be a follow-up.
