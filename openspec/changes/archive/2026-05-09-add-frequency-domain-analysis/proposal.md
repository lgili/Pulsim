## Why

Pulsim today provides time-domain transient simulation and periodic steady-state (shooting / harmonic balance) but **no frequency-domain analysis**. Power-electronics engineers need:

1. **AC small-signal analysis** — linearize the system around an operating point, sweep frequency, return Bode/Nyquist data. This is how control loops are designed.
2. **Frequency Response Analysis (FRA)** — inject a sinusoidal perturbation at a specific node and measure the transfer function in the time domain. This is how converters are characterized in the lab and what closed-loop validation expects.

PSIM has `.AC` analysis. PLECS has `Analysis Tools → AC Sweep` and impedance analyzer blocks. Without these, Pulsim is unusable for control-loop design — which is the primary task power-electronics engineers spend their time on.

## What Changes

### AC Small-Signal Analysis
- New `Simulator::run_ac_sweep(AcSweepOptions)` method.
- Operating point obtained via existing DC solver (or supplied externally).
- Linearize: `(jω E − A) X(ω) = B U(ω)`, where `E`, `A`, `B` come from the same MNA decomposition the PWL engine uses.
- Frequency sweep: log/linear, `f_start`, `f_stop`, `points_per_decade`.
- Output: `AcSweepResult` with magnitude (dB), phase (deg), real/imag parts at each frequency, per measurement node.
- Multi-input support: independent sweeps per source for transfer-function matrix (e.g., `H(s) = V_out / V_in`).

### Frequency Response Analysis (FRA)
- New `Simulator::run_fra(FraOptions)` method.
- Injection: small-signal sinusoid summed onto a chosen source (e.g., reference voltage of a control loop) at one frequency at a time.
- Steady-state response measurement: integrate over `n_cycles` cycles, FFT for fundamental component, compute `mag/phase`.
- Frequency sweep with the same `f_start/f_stop/points_per_decade` knobs as AC sweep.
- Validates the AC analysis result: AC = analytical small-signal; FRA = simulated equivalent.
- Required when nonlinearities, saturation, or PWM interaction matter (PLECS-style approach).

### Operating-Point Linearization
- `Simulator::linearize_around(x_op, t_op) -> (E, A, B, C, D)` matrices for downstream analysis (e.g., user computing eigenvalues, designing observers).
- Numerical differentiation for non-AD-friendly devices; AD-derived where supported.

### Bode/Nyquist Helper API
- Python helpers `pulsim.bode_plot(ac_result, ax=None)`, `pulsim.nyquist_plot(...)` using matplotlib.
- CSV export `export_ac_csv(result, path)`, JSON export `export_ac_json(...)`.

### YAML Schema
- New top-level `analysis:` section (parallel to `simulation`):
```yaml
analysis:
  - type: ac
    f_start: 1
    f_stop: 1e6
    points_per_decade: 20
    perturbation_source: Vref
    measurement_nodes: [vout, ierr]
  - type: fra
    f_start: 100
    f_stop: 1e5
    points_per_decade: 5
    perturbation_amplitude: 0.01
    n_cycles: 50
```
- Multiple `analysis` entries supported in one netlist.

### Telemetry
- `AcSweepResult.condition_numbers[f]` — system conditioning per frequency.
- `FraResult.thd[f]` — total harmonic distortion at perturbation frequency (sanity check for small-signal regime).

## Impact

- **New capability**: `ac-analysis`.
- **Affected specs**: `ac-analysis` (new), `python-bindings` (new methods + result types), `netlist-yaml` (analysis section).
- **Affected code**: new files `core/include/pulsim/v1/ac_analysis.hpp`, `core/src/v1/ac_analysis.cpp`, additions to `python/bindings.cpp`, and `python/pulsim/plot_helpers.py`.
- **Performance**: AC sweep is one linear solve per frequency point — fast. FRA is one transient simulation per frequency — slow but parallelizable in Phase 3 via `parameter-sweep`.

## Success Criteria

1. **AC vs FRA parity**: agreement within 1 dB / 5 deg on a buck converter open-loop transfer function `V_out / d` from 100 Hz to 100 kHz.
2. **AC vs analytical**: small-signal Bode of an RLC tank matches analytical model within 0.1 dB / 1 deg up to 10× resonant frequency.
3. **FRA on closed-loop**: reproduces the published loop gain / phase margin for a known PI-compensated buck within 2 dB / 5 deg.
4. **Performance**: AC sweep with 200 frequency points runs in ≤2× the time of a single-step DC operating point on a typical converter.
5. **Plotting**: Bode/Nyquist helpers produce publication-quality plots with one Python call.
