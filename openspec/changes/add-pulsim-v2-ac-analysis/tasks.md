## 1. Linearisation

- [ ] 1.1 Write `ac::linearise_at(graph, pool, x_op, switch_mask)` — returns dense `(A, B, C, D)` for the linearised state-space
- [ ] 1.2 For each nonlinear device, expose a `linearise_at(x_op)` method that returns the Jacobian (reuse existing AD machinery)
- [ ] 1.3 Build the full A matrix by stamping linear contributions + nonlinear Jacobians at x_op

## 2. Frequency sweep

- [ ] 2.1 Write `ac::run_ac_sweep(...)` — for each `ω = 2π·f`, solve `(jωI - A) X = B` and extract `H(jω) = C·X + D`
- [ ] 2.2 Support log-spaced frequency grid (`f_start`, `f_end`, `points_per_decade`)
- [ ] 2.3 Return `AcSweepResult` with `freqs`, `H` (complex array)

## 3. Bode utilities

- [ ] 3.1 `ac::bode_data(result)` returns `(mag_dB, phase_deg)` numpy-ready arrays
- [ ] 3.2 Optional: ASCII plot for quick CLI inspection

## 4. YAML + Python

- [ ] 4.1 YAML `analysis: ac_sweep` block parsed by loader, returned as a separate `AcSweepOptions` struct alongside `LoadedCircuit`
- [ ] 4.2 Python bindings: `run_ac_sweep(...)` returns dict with `freqs`, `H`, `mag_dB`, `phase_deg`
- [ ] 4.3 Example Jupyter-style usage in docs

## 5. Showcase: Buck AC sweep

- [ ] 5.1 YAML: open-loop buck with explicit `x_op` override (since V11/V12 don't have DC OP yet)
- [ ] 5.2 Sweep 10 Hz → 1 MHz, verify LC double-pole at `f_LC = 1/(2π√LC)` and slope -40 dB/dec past resonance
- [ ] 5.3 Compare against analytical control-to-output transfer function

## 6. Validation + commit

- [ ] 6.1 Build + run all targets
- [ ] 6.2 `openspec validate add-pulsim-v2-ac-analysis --strict`
- [ ] 6.3 Commit and push
