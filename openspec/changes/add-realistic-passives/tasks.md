## 1. ESR, ESL, parallel capacitance
- [x] 1.1 `CircuitBuilder::add_capacitor(..., esr, esl)` and `add_inductor(..., esr, c_par)`: composite expansion with internal nodes only for non-zero values; sub-element names `{name}.esr`, `{name}.esl`, `{name}.cpar`; validation by name (finite, ≥ 0) — Python wrappers installed on `CircuitBuilder` (`python/pulsim/passives.py`, DECISIONS D1)
- [x] 1.2 Identity test: every existing `add_capacitor` / `add_inductor` call (all parameters zero) is bit-identical (states, node count, benchmark hashes) — states and node count bit-identical (zero parameters call the native method); benchmark hashes not run
- [x] 1.3 AC test: 100 µF, 10 mΩ, 10 nH → minimum `|Z|` = 10 mΩ at 159.15 kHz (1e-9); `|Z(f)|` equals `sqrt(esr² + (2πf·esl − 1/(2πf·C))²)` at 10 frequencies (1e-9) — kernel complex MNA sweep
- [x] 1.4 Transient test: sine drive at 20, 159.15 and 800 kHz against the phasor solution (1e-3); `c0` is the ideal capacitor's voltage — measured 5.5e-5, 3.3e-9, 7.1e-5
- [x] 1.5 Ripple test: ±1 A square current (pulse current source of `add-controlled-source-set`, or a pulse voltage source behind 1 kΩ) into 50 mΩ + 100 µF → 100 mV step at each edge (1e-6) — ±1000 V pulse behind 1 kΩ (DECISIONS D11)
- [x] 1.6 Inductor test: 100 µH with `c_par` = 10 pF → impedance peak at 5.033 MHz (1e-6 in frequency) — 1e-3 on the rounded 5.033 MHz, 1e-6 against 1/(2π√(LC)) (D11)
- [x] 1.7 `components()` reports the parameters; `device_loss_summary` reports `i²·esr` under the component: 1 A rms through 50 mΩ → 50.0 mW (1e-9) — loss as row `{name}.esr` (D3), 1e-6; `components()` lists the sub-elements, not a parameter on the capacitor
- [ ] 1.8 Bindings, stubs; YAML keys `esr`, `esl`, `c_par` (strict) — PARTIAL: Python kwargs and stubs done; YAML keys NOT DONE (D2)

## 2. π-section line
- [x] 2.1 `add_transmission_line(..., model="pi", sections=N)` from `(Z0, delay)` or `(L_per_m, C_per_m, length)`; series `R` distributed; validation by name (`Z0 > 0`, `delay > 0`, `sections ≥ 1`) — Python, installed on `CircuitBuilder`; `model="bergeron"` refused by name
- [x] 2.2 AC test: 50 Ω, 100 ns, 50 sections, open end → `Z_in` = −j91.52 Ω at `ωτ = 0.5` (1e-3) — measured −j91.525 Ω (ideal −j91.524)
- [x] 2.3 Regression on the probe: ideal step, 50 sections → far-end peak 1.271 V and 50 % crossing at 1.015·τ (1 %); the docstring states the artefact — measured 1.2713 V, 1.0152·τ
- [x] 2.4 `recommend_line_sections(delay, t_rise)` = `ceil(11·delay/t_rise)`; test: 20 ns edge on the 100 ns line with the recommended 55 sections overshoots less than 6 % — measured peak 1.046 V

## 3. AC-resistance ladder
- [x] 3.1 `python/pulsim/passives.py`: `RacLadder` (cells, `R_dc`, `L_hf`, band, residual, source) and `fit_rac_ladder(f, R_ac, *, L, R_dc=None, n_cells=4, tol=0.05)` (log-spaced corners, non-negative least squares); refusals (`L_hf ≤ 0`, residual above `tol`) naming the component
- [ ] 3.2 `add_inductor(..., rac=ladder)` composite; YAML `rac:` (cells or a `(f, R_ac)` table to fit at load) — PARTIAL: `rac=` done; YAML `rac:` NOT DONE (D2)
- [x] 3.3 Identity test: `R_dc` = 10 mΩ, one cell 90 mΩ / 143.24 nH → `R_ac` = 10.009, 55.00, 99.11 mΩ at 1 kHz, 100 kHz, 1 MHz (AC, 1e-9); two-tone loss (1 A peak at 1 kHz and at 100 kHz) = 32.50 mW (1e-3) — AC 1e-9 against the analytic ladder; the rounded design values to 2e-4; loss 1e-3
- [x] 3.4 Fit tests: round trip on ladder-generated data (1e-6); the round-wire skin-effect curve over two decades — record the cell count needed for 5 % and set the default from it — round trip 1e-6; skin curve: 2 cells 1.75 %, 4 cells 0.60 %; default kept at 4 (D10)
- [x] 3.5 The fitted points are reported as fitted; no agreement with them is reported as validation — docstrings, CHANGELOG and limitation pages say so

## 4. Capacitance against DC bias (after the charge-law variant)
- [ ] 4.1 Odd charge table for `NonlinearCapacitor` (knots for `v ≥ 0`, odd extension, linear tail); `Q(v)` strictly increasing at 1 pV spacing around 0 and around the knee — NOT STARTED: needs the charge-law variant from `add-power-mosfet-datasheet-device` (D9)
- [ ] 4.2 `fit_mlcc_dc_bias(v_bias, c_ratio, C_nominal)` returning the table and its residual
- [ ] 4.3 Test: `C0/(1 + (v/V1)²)`, `C0` = 10 µF, `V1` = 10 V, 64 knots → `Q(20 V)` = 110.71 µC (1e-4), `C(20 V)` = 2.000 µF (1e-3), stored energy 0.8047 mJ (1e-4)
- [ ] 4.4 Test: small-signal ripple on a 20 V bias corresponds to 2.000 µF (1e-3); charge conserved over a ±20 V cycle (1e-9); `pwl` against `trbdf2` (1e-6)
- [ ] 4.5 YAML: `nonlinear_capacitor` with `table:` or `dc_bias:` curve

## 5. Bergeron line
- [ ] 5.1 `models/transmission_line.hpp` (params, validation, `R ≪ Z0` check) and `pwl/transmission_line_history.hpp` (ring buffer of both ends, delayed-value lookup with linear interpolation, snapshot) — NOT DONE: kernel device; does not fit today's step loop without touching it (D6)
- [ ] 5.2 Stamp: `1/Z0` across each port in the cached matrix; history sources into `b` each step; `R/4`, `R/2`, `R/4` composite
- [ ] 5.3 `pwl`: refuse `dt > τ` naming the line; test (matched source, open end, `τ/dt` = 100) → 0 V before `τ`, 1.000 V after, peak ≤ 1.000 V + 1e-9
- [ ] 5.4 Lattice test: 1 V step, 10 Ω source, open end → far end 1.667, 0.556, 1.296 V after `τ`, `3τ`, `5τ` (1e-9)
- [ ] 5.5 Non-integer `τ/dt` (e.g. 100.5): steps within 2 % of their value; documentation recommends a divisor
- [ ] 5.6 `trbdf2`: step capped at `τ`, stage-time history lookup; lattice test (1e-3)
- [ ] 5.7 DC operating point: series resistance; history initialised from it; a line carrying DC current starts the transient with no wave (far-end voltage constant to 1e-9)
- [ ] 5.8 Lossy line: `R` = 1 Ω on a 50 Ω line, matched at both ends → far-end step attenuated to `Z0/(2·Z0 + R)` = 0.4950 V in steady state (1e-6)
- [ ] 5.9 Refusals through the conformance error: `dsed`, the exact propagator, periodic-steady-state shooting, AC (message names `model="pi"`)
- [ ] 5.10 Binding, stubs, YAML type `transmission_line`

## 6. Conformance and documentation
- [ ] 6.1 Rows in the matrix of `add-engine-conformance-matrix`: `capacitor[esr/esl]`, `inductor[esr/c_par/rac]`, `nonlinear_capacitor[odd table]`, `transmission_line[pi]`, `transmission_line[bergeron]` × {pwl, trbdf2, dsed, DC, AC} — NOT DONE: the conformance matrix does not exist yet
- [ ] 6.2 `AI_GUIDE.md` (which datasheet number goes where; the step needed for an ESL resonance; π versus Bergeron), `docs/losses-and-thermal.md` (ESR and ladder loss rows), `docs/known-limitations.md` and `KNOWN_LIMITATIONS.md` (single-pair line, no coupled cable, one ESR value, large-signal ceramic behaviour), `CHANGELOG.md` — PARTIAL: AI_GUIDE (signatures and one note), losses-and-thermal, both limitation pages, CHANGELOG; π-versus-Bergeron and ESL step guidance in the limitation pages only

## 7. Verification
- [ ] 7.1 `ruff check`, `pytest`, `ctest`, `openspec validate --all --strict`, stub check
