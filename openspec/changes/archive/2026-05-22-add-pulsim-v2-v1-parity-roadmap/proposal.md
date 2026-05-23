## Why

After the Tier-0 work (saturable magnetics, ergonomics, OpenSpec backlog, tutorial docs, AC analysis, UX helpers), the v2 kernel is solid for SMPS prototyping but **not yet a drop-in replacement for v1**. An audit (transcript reference: `agentId: a19300e1f9f1f87ac`) identified ~40–55 person-weeks of gaps across solver numerics, component models, YAML schema, Python ecosystem, tooling, and the mixed-domain virtual-block library.

This proposal **documents the full v1→v2 parity roadmap as five phases** so the team can sequence the work without forgetting items, and **starts implementation with Phase A** — the smallest, highest-impact bucket.

## What Changes

### Roadmap (documentation only, no code)

Five phases, sequenced by impact-per-week:

- **Phase A — Solver + YAML virtual blocks (~10 pw)**
  - A.1: 18 virtual blocks in YAML (op_amp, gain, sum, math, integrator, differentiator, limiter, hysteresis, transfer_function, delay, state_machine, pwm_generator, signal_mux/demux, clarke/park transforms + inverses, pll, svm). Each block ALSO available as a standalone Python class so users can wire loops in either YAML or Python.
  - A.2: DC operating-point real strategies (Gmin ramp, source-stepping, pseudo-transient continuation). Today v2 only snapshots the transient at `t_eval`.
  - A.3: MNA-based AC sweep — linearise around DC OP, sweep `(jωE − A)⁻¹ B`. 100× faster than swept-sine for linear and weakly-nonlinear plants.
  - A.4: `MovingAverageFilter` — trivial gap, included for completeness.

- **Phase B — Convergence (~8 pw)**
  - B.1: Variable-step + LTE (step doubling, Richardson extrapolation, solution history).
  - B.2: BDF1 / BDF2 / TRBDF2 / RosenbrockW — stiff-friendly integrators alongside Trapezoidal.

- **Phase C — Domain (~13 pw)**
  - C.1: Thermal coupling (Foster / Cauer networks + T_j → R_DS_ON feedback).
  - C.2: Three-phase grid helpers (`ThreePhaseSource`, `ThreePhaseVSI`, `ThreePhaseRLLoad`, `GridFollowingInverter`, `GridFormingInverter`, `BridgeRectifier`, symmetrical-components).
  - C.3: Magnetic fidelity (`HysteresisInductor` with B-H + Steinmetz, `SaturableTransformer`, `CoreCatalog` — TDK/EPCOS/Ferroxcube/Magnetics).
  - C.4: Switchgear & protection (`Thyristor`, `Triac`, `BJT_NPN/PNP`, `Relay`, `Fuse`, `CircuitBreaker`, `VoltageControlledSwitch`, dedicated `SnubberRC` device).
  - C.5: HVAC (`CompressorLoad`, refrigerant tables R134a/R600a).

- **Phase D — Motors (~10 pw)**
  - D.1: `DcMotor` + `Mechanical` (J+B+τ_load) — entry point.
  - D.2: `PMSM` + `PmsmFoc` (built-in field-oriented control) — drives showcase.
  - D.3: `BLDC`, `Induction`, `SinglePhaseInduction`.

- **Phase E — Tooling & ecosystem (~30 pw)**
  - E.1: C99 PIL code generation (port `codegen/generator.py`).
  - E.2: FMU 2.0 export (port `fmu/exporter.py`).
  - E.3: Parameter sweep + Monte Carlo (Cartesian, LHS, Sobol, Halton + metrics).
  - E.4: SPICE parity harness (ltspice/ngspice cross-validation).
  - E.5: KPI gates (`kpi_gate.py`, baselines, thresholds).
  - E.6: Validation framework (5 graded levels).
  - E.7: Schematic render (layout, ELK, netlistsvg, native, symbols).
  - E.8: Snubber advisor (`snubber.py`).
  - E.9: Frequency analysis polish (Bode/Nyquist/FRA overlays, CSV/JSON export).
  - E.10: SPICE-style `.cir` netlist parser.
  - E.11: Device catalogs (GaNSystems, Infineon, Vishay, Wolfspeed + ferrite cores).

### Phase A implementation (code)

This proposal also IMPLEMENTS Phase A:

1. **`MovingAverageFilter`** — exponential-moving-average block in `python/pulsim/v2_control.py`. ~30 lines, trivial.
2. **18 Python control blocks** added to `pulsim.v2_control`:
   - Math: `Gain`, `Sum`, `Subtract`, `MathBlock`
   - Control (standalone): `Integrator`, `Differentiator`, `TransferFunction`, `StateMachine`, `OpAmp` (with rail saturation)
   - Signal: `Limiter`, `Hysteresis` (already partial — Comparator), `DelayBlock`
   - Modulation: `PwmGenerator`, `SpaceVectorModulator` (SVM)
   - Transforms: `ClarkeTransform`, `ParkTransform`, `InverseClarkeTransform`, `InverseParkTransform`
   - Synchronization: `PLL`
   - Routing: `SignalMux`, `SignalDemux`
   Each class follows the `PIController` template: stateful dataclass + `reset()` + `update(...)` returning the output.
3. **DC strategy** — port `pulsim::v1::numerical::DCStrategy` to `pulsim::v2::pwl::DCStrategy`. New header `core/include/pulsim/v2/pwl/dc_strategy.hpp`. Add `compute_dc_op` overload that takes a `DCStrategyOptions` struct.
4. **MNA-based AC sweep** — new `pulsim::v2::ac::run_mna_sweep(graph, pool, x_op, freqs, input, output)` in `core/include/pulsim/v2/ac/mna_sweep.hpp`. Linearise the cached MNA matrix around `x_op`, sweep `(jωE − A)⁻¹ B`, return complex `H(jω)`.

### YAML wire-up

For each new Python block, extend `core/include/pulsim/v2/yaml/loader.hpp` to:
- Parse the corresponding `type:` entry
- Resolve cross-block signal dependencies via `signal_from_channel:` / `signal_to_channel:` metadata (matching v1's convention)
- Build a `MixedDomainBlockChain` that's evaluated inside `step_observer` at every simulation step
- Order blocks topologically so dependencies are resolved before consumers

## Impact

- **Affected specs**: new `pulsim-v2-parity` capability documenting which Phase A items are part of the public surface.
- **Affected code**:
  - `python/pulsim/v2_control.py` (+~600 lines for the 18 blocks + MovingAverageFilter)
  - `python/pulsim/v2.py` (re-export the new classes)
  - `core/include/pulsim/v2/yaml/loader.hpp` (new `type:` handlers + signal channel resolver)
  - `core/include/pulsim/v2/pwl/dc_strategy.hpp` (new, ~300 lines)
  - `core/include/pulsim/v2/ac/mna_sweep.hpp` (new, ~250 lines)
  - `core/tests/v2/control/` (new test target for the 18 blocks)
  - `core/tests/v2/yaml/test_mixed_domain.cpp` (new)
  - `examples/v2/` (a YAML showcase using the block chain + a Python equivalent)
- **Risk**: low for the Python blocks (additive). Medium for the YAML mixed-domain executor (needs topological sort + careful event ordering with the existing solver). Medium for the DC strategy (must keep backward compatibility with the current snapshot path).
- **Phases B–E**: NOT implemented here — they're tracked as future work in this proposal's `tasks.md`. Each phase will get its own proposal when scheduled.
