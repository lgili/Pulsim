## 1. Documentation

- [x] 1.1 Write `proposal.md` listing all 5 phases (A–E)
- [x] 1.2 Write this `tasks.md` checklist for Phase A + roadmap pointers
- [x] 1.3 Write delta spec under `specs/pulsim-v2-parity/spec.md`
- [x] 1.4 `openspec validate add-pulsim-v2-v1-parity-roadmap --strict`

## 2. Phase A.4 — MovingAverageFilter (trivial)

- [x] 2.1 Add `MovingAverageFilter` class to `python/pulsim/v2_control.py`
- [x] 2.2 Re-export from `pulsim.v2`
- [x] 2.3 Catalog + example entries

## 3. Phase A.1 — 18 virtual blocks as Python classes (Stage 1)

All blocks follow the dataclass + `reset()` + `update(...)` template.

### 3.1 Math blocks
- [x] 3.1.1 `Gain(k)` — `y = k · x`
- [x] 3.1.2 `Sum(weights)` — N-input weighted sum
- [x] 3.1.3 `Subtract` — `y = a − b`
- [x] 3.1.4 `MathBlock(op)` — generic op ∈ {add, sub, mul, div, abs, neg, sqrt, pow2}

### 3.2 Standalone control blocks
- [x] 3.2.1 `Integrator(gain, output_min, output_max)` with anti-windup
- [x] 3.2.2 `Differentiator(filter_alpha)` with IIR derivative filter
- [x] 3.2.3 `TransferFunction(num_coeffs, den_coeffs)` — direct-form II
- [x] 3.2.4 `StateMachine` — Mealy machine (toggle / level / SR-latch)
- [x] 3.2.5 `OpAmp(gain, rail_min, rail_max)` — saturating ideal op-amp

### 3.3 Signal-shaping blocks
- [x] 3.3.1 `Limiter(min, max)` — hard clamp
- [x] 3.3.2 `DelayBlock(samples)` — FIFO buffer
- [x] 3.3.3 Hysteresis already covered by existing `Comparator` (noted in docstrings)

### 3.4 Modulation blocks
- [x] 3.4.1 `PwmGenerator(frequency, phase)` — sawtooth + duty comparator (output 0/1)
- [x] 3.4.2 `SpaceVectorModulator(v_dc)` — α-β → 3 duty cycles (centered SVM)

### 3.5 Transform blocks
- [x] 3.5.1 `ClarkeTransform` — abc → αβ0
- [x] 3.5.2 `InverseClarkeTransform` — αβ → abc
- [x] 3.5.3 `ParkTransform(theta)` — αβ → dq with rotor angle
- [x] 3.5.4 `InverseParkTransform(theta)` — dq → αβ

### 3.6 Synchronization
- [x] 3.6.1 `PLL(f_nominal, Kp, Ki)` — cross-product phase-detector PLL on αβ

### 3.7 Routing
- [x] 3.7.1 `SignalMux(selector)` — select one of N inputs
- [x] 3.7.2 `SignalDemux(n_outputs)` — broadcast one input to multiple outputs

### 3.8 Tests + catalog
- [x] 3.8.1 Unit tests for each block (smoke test via inline script) — all 19 verified
- [x] 3.8.2 Re-export every class from `pulsim.v2`
- [x] 3.8.3 Update `p.catalog()` with 6 new categories: Math/signal, Standalone control, Modulation, Transforms, Synchronization, Routing
- [ ] 3.8.4 Add 3 `p.example(...)` snippets: `"pll"`, `"foc_loop"`, `"transfer_function"` — deferred (catalog already shows usage)

## 4. Phase A.1 — YAML wire-up (Stage 2)

- [ ] 4.1 Extend `core/include/pulsim/v2/yaml/loader.hpp` with handlers for each new `type:`
- [ ] 4.2 Implement `MixedDomainBlockChain` — topologically sorted Python-side container, evaluated inside `step_observer`
- [ ] 4.3 Parse `signal_from_channel:` / `signal_to_channel:` metadata (matches v1)
- [ ] 4.4 Add `core/tests/v2/yaml/test_mixed_domain.cpp`
- [ ] 4.5 Write 1 YAML showcase using a 3-block chain (e.g. setpoint → PI → PWM) + 1 Python equivalent in `examples/v2/scripts/`

## 5. Phase A.2 — DC strategy

- [ ] 5.1 New header `core/include/pulsim/v2/pwl/dc_strategy.hpp` exporting `DCStrategy`, `GminConfig`, `SourceSteppingConfig`, `PseudoTransientConfig`
- [ ] 5.2 Add `compute_dc_op_with_strategy(graph, pool, mask, opts)` overload
- [ ] 5.3 Python binding: `p.compute_dc_op(builder, strategy=...)`
- [ ] 5.4 Unit tests: 3 "hard" circuits where the naive solve fails but the strategy succeeds (Gmin ramp, source step, pseudo-transient)

## 6. Phase A.3 — MNA-based AC sweep

- [ ] 6.1 New header `core/include/pulsim/v2/ac/mna_sweep.hpp` with `run_mna_sweep(graph, pool, x_op, freqs, input, output)`
- [ ] 6.2 Linearise the cached MNA at `x_op` (nonlinear devices contribute their `∂I/∂V` Jacobian)
- [ ] 6.3 Solve `(jωE − A) X = B u` per frequency via the existing KLU factorisation (per-frequency complex factorisation)
- [ ] 6.4 Python binding: `p.run_mna_sweep(builder, x_op, freqs, input_node, output_node)`
- [ ] 6.5 Verification: same buck plant should match the swept-sine result within 0.5 dB / 2° on the linear range (validates both paths)

## 7. Phase A wrap-up

- [ ] 7.1 Update `docs/v2/api-reference.md` with the 18 new blocks
- [ ] 7.2 Update `docs/v2/helpers.md` with examples of YAML mixed-domain chains
- [ ] 7.3 Update `examples/v2/scripts/README.md`
- [ ] 7.4 Run the full v2 regression suite — must pass
- [ ] 7.5 Commit + push

## 8. Future work tracker (Phases B–E — separate proposals)

Each gets its own proposal when scheduled. Tracked here so we don't lose them.

### Phase B — Convergence (~8 pw)
- [ ] B.1 Variable-step + LTE (`StepDoubling`, `Richardson`, `SolutionHistory`)
- [ ] B.2 BDF1, BDF2, TRBDF2, RosenbrockW integrators

### Phase C — Domain (~13 pw)
- [ ] C.1 Thermal coupling (Foster/Cauer + T_j → R_DS_ON feedback)
- [ ] C.2 Three-phase grid helpers
- [ ] C.3 Magnetic fidelity (B-H + Steinmetz + core catalog)
- [ ] C.4 Switchgear & protection (Thyristor/Triac/BJT/Relay/Fuse/CB/VCSwitch/SnubberRC)
- [ ] C.5 HVAC (CompressorLoad + refrigerant tables)

### Phase D — Motors (~10 pw)
- [ ] D.1 DcMotor + Mechanical
- [ ] D.2 PMSM + PmsmFoc
- [ ] D.3 BLDC, Induction, SinglePhaseInduction

### Phase E — Tooling & ecosystem (~30 pw)
- [ ] E.1 C99 PIL code generation
- [ ] E.2 FMU 2.0 export
- [ ] E.3 Parameter sweep + Monte Carlo
- [ ] E.4 SPICE parity harness
- [ ] E.5 KPI gates + baselines
- [ ] E.6 Validation framework (5 graded levels)
- [ ] E.7 Schematic render
- [ ] E.8 Snubber advisor
- [ ] E.9 Frequency analysis polish (Bode/Nyquist/FRA overlays, CSV/JSON)
- [ ] E.10 SPICE-style `.cir` netlist parser
- [ ] E.11 Device catalogs (GaNSystems, Infineon, Vishay, Wolfspeed + ferrite cores)
