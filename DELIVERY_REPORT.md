# Pulsim Roadmap Delivery Report

> **Status: Roadmap Complete (Fase 0 → Fase 4).**
> 16 OpenSpec changes archived between 2026-05-08 and 2026-05-10.
> 5,152 C++ test assertions + 41 Python tests green across both
> default and AD builds. 24 user-facing docs pages in mkdocs.

This document is the authoritative reference for what shipped during
the roadmap execution. It complements [`ROADMAP.md`](ROADMAP.md) (the
strategic plan) by listing everything actually delivered — feature-by-
feature, with paths to source files, tests, and docs, plus an explicit
list of deferrals and their rationale.

---

## Executive summary

| Metric | Value |
|---|---|
| OpenSpec changes archived (Fase 0–4) | **16** |
| C++ test assertions (default mode) | **5,111** (4,021 simulation + 1,090 header-only) |
| C++ test cases | **411** |
| Python tests | **41** (covering codegen, FMU, sweep, properties, frequency analysis, templates) |
| User-facing docs pages added | **15+** (linked from `mkdocs.yml`) |
| Headline benchmark | **351×** speedup (Behavioral → Ideal+cache, 10-PWM-cycle buck) |
| Reference vendor parts | **6** catalog devices + **4** magnetic cores |

**What now works end-to-end that didn't before**:

- PWL state-space switching engine (351× faster than Newton-DAE on a buck).
- AD-validated nonlinear stamps (MOSFET, IGBT, IdealDiode, VCSwitch).
- Linear-solver LRU cache with 98.6 % hit rate on PWM cycling.
- AC small-signal sweep + FRA empirical Bode + Python plotting helpers.
- 4 vendor magnetic cores + saturable inductor / transformer / hysteresis.
- 6 vendor catalog MOSFET / IGBT / diode parts with `Coss(Vds)` etc.
- 3 converter templates (buck / boost / buck-boost) + PI compensator.
- PMSM + DC motor + Park/Clarke + PMSM-FOC current loop.
- 3-phase sources + 3 PLLs + Fortescue + grid-following / grid-forming inverters.
- Python C99 codegen pipeline + FMI 2.0 CS export (PIL parity ±0.1%).
- Monte-Carlo / LHS / Sobol parameter sweep with metric library.
- Hypothesis-driven property tests covering KCL / energy / passivity.
- `RobustnessProfile` typed knob bundle + `build_bench.py` baseline.

---

## Fase 0 — Killer feature trio (3 changes)

The "PSIM-killer" trio. Delivers the speedup story that makes Pulsim
competitive with PSIM and PLECS on switching converters.

### `refactor-pwl-switching-engine` (archived 2026-05-08)

**Delivered**: PWL state-space segment-primary stepper. When all switches
are in `Ideal` mode and the topology is admissible, every step is one
linear solve of the Tustin discretization — no Newton iteration.

- Devices stamp `M·dx/dt + N·x = b(t)` directly.
- Topology bitmask drives a per-(topology, dt) cache.
- VCSwitch event scheduler with bisection-to-event for accurate
  commutation timing.

**Headline numbers**: 240–374× speedup vs Newton-DAE on
`buck_switching.yaml` after switching mode is set to `Ideal`.

**Key files**: `core/include/pulsim/v1/transient_services.hpp`,
`core/src/v1/transient_services.cpp`,
`core/tests/test_pwl_segment_primary.cpp`,
`core/tests/test_pwl_speedup_benchmark.cpp`.

### `add-automatic-differentiation` (archived 2026-05-08)

**Delivered**: Eigen `AutoDiffScalar`-derived Jacobians for nonlinear
device stamps. Build-flag toggle (`PULSIM_USE_AD_STAMP=ON`) selects
between manual and AD path. Validation layer agrees within 1e-12
between the two on every operating point exercised.

- `pulsim::v1::ad::seed_from_values(...)` for ADReal seeding.
- `validate_nonlinear_jacobians(circuit, op_points, abs_tol)` cross-
  validation harness.
- AD path implemented for `MOSFET`, `IGBT`, `IdealDiode`,
  `VoltageControlledSwitch`. Linear devices opt out via SFINAE.

**Performance budget**: AD path ≈ 3.81× slower per stamp than manual
(848 ns vs 222 ns on AppleClang 17 / Release+LTO MOSFET). Default flag
keeps the manual path; AD is opt-in for new-device authoring + bug
hunting.

**Key files**: `core/include/pulsim/v1/ad/*.hpp`,
`core/include/pulsim/v1/components/{mosfet,igbt,diode,vcswitch}.hpp`
(both manual and AD paths via `#ifdef`),
`core/tests/test_ad_*.cpp`, `docs/automatic-differentiation.md`.

### `refactor-linear-solver-cache` (archived 2026-05-09)

**Delivered**: per-`matrix_hash` LRU cache of analyzed-and-factorized
linear solvers in the segment-primary stepper. Default capacity 64
entries (chosen for typical ≤ 16-topology converters × few dt values).

- `LinearFactorCache` class wraps `std::list` (LRU ordering) +
  `std::unordered_map` (O(1) lookup). Each entry owns a
  `RuntimeLinearSolver` plus a `shared_ptr` to the underlying
  `SegmentLinearStateSpace` (keeps `last_matrix_` non-dangling).
- Phase 5 typed `CacheInvalidationReason` enum (`None`,
  `TopologyChanged`, `StampParamChanged`, `GminEscalated`,
  `SourceSteppingActive`, `NumericInstability`, `ManualInvalidate`).
- `BackendTelemetry` per-reason counters + `last_typed` mirror for
  programmatic inspection.

**Gates**: G.1 cache hit rate **99.6 %** on stable-topology RC; G.3
**351×** speedup vs Newton-DAE baseline; G.5 every existing test
passes unmodified.

**Key files**: `core/src/v1/transient_services.cpp` (anon-namespace
`LinearFactorCache`), `core/tests/test_linear_cache_*.cpp`,
`docs/linear-solver-cache.md`.

---

## Fase 1 — Fidelity layer (3 changes)

Closes the model-fidelity gap with PSIM/PLECS. Engineers evaluate
simulators by realism of device + magnetic + control-loop models.

### `add-frequency-domain-analysis` (archived 2026-05-09)

**Delivered**: full AC analysis stack — linearization, AC sweep, FRA,
multi-input transfer matrix, Python plotting, YAML schema.

- `Simulator::linearize_around(x_op, t_op)` returns
  `LinearSystem { E, A, B, C, D }` in descriptor form.
- `Simulator::run_ac_sweep(opts)` solves `(jω·E − A) X = B` per
  frequency via complex `SparseLU`. `analyzePattern` runs once,
  `factorize`+`solve` per ω. 200-point sweep ≈ **600 µs** on a 4-state
  LC.
- `Simulator::run_fra(opts)` injects `ε·cos(2π·f·t + φ)` on a named
  source via `Circuit::set_ac_perturbation`, runs trapezoidal
  transient, Goertzel single-bin DFT. Half-step phase correction
  (`e^{-jω·dt/2}`) accounts for trapezoidal midpoint vs sample-end
  timing. Agrees with AC sweep within ≤ 1 dB / ≤ 5° (gate G.1).
- Phase 4 multi-input matrix: `perturbation_sources` list ⇒ N×M
  measurements per `(source, node)` pair.
- Phase 5 Python helpers: `pulsim.bode_plot`, `pulsim.nyquist_plot`,
  `pulsim.fra_overlay`, `export_ac_csv/json`, `load_ac_result_csv`.
- Phase 7 YAML `analysis:` array with strict-mode validation.

**Headline numbers**: 9 Python tests green; 925 + 3,752 C++ assertions.

**Key files**: `core/include/pulsim/v1/frequency_analysis.hpp`,
`core/src/v1/simulation.cpp` (run_ac_sweep + run_fra),
`python/pulsim/frequency_analysis.py`, `docs/ac-analysis.md`,
`docs/fra.md`.

### `add-magnetic-core-models` (archived 2026-05-09)

**Delivered**: 7-phase header-only magnetic primitives + saturable
devices + 4 vendor cores.

- `BHCurveTable` (lookup), `BHCurveArctan`, `BHCurveLangevin` —
  templated curve types with forward / inverse / `dbdh`.
- `SteinmetzLoss { k, α, β }` + iGSE for non-sinusoidal flux.
- `JilesAthertonState` + `jiles_atherton_step()` ODE with `±Ms` clamp
  for numerical stability.
- `SaturableInductor<Curve>` with `current_from_flux` /
  `differential_inductance` (= `N²·A_e·dB/dH/l_e`).
- `SaturableTransformer<Curve>` with N windings + per-winding leakage
  + shared magnetizing flux λ_m.
- `HysteresisInductor` wrapping J-A around the geometry/turns layer.
- 4 reference cores at `devices/cores/`: TDK N87, Ferroxcube 3C90,
  Magnetics MPP_60u, EPCOS N97. YAML loader at `core_catalog.hpp`.

**Gates**: G.1 inrush within ±20 % of analytical Faraday; G.2 iGSE
within ±10 % of Steinmetz on sinusoidal flux 25–500 kHz; G.4 4 of 4
catalog cores load.

**Key files**: `core/include/pulsim/v1/magnetic/*.hpp`,
`core/tests/test_magnetic_phase{1..6}_*.cpp`,
`devices/cores/<vendor>/<material>.yaml`, `docs/magnetic-models.md`.

### `add-catalog-device-models` (archived 2026-05-09)

**Delivered**: catalog-tier device library with vendor-published
parameters.

- `LookupTable2D` bilinear interp, monotone-axis validation.
- `MosfetCatalog` with `Coss/Ciss/Crss(V_ds)` 1D + `Eon/Eoff(I_c, V_ds)`
  2D + `R_ds_on(T_j)` + `V_th(T_j)`.
- `IgbtCatalog` with `V_ce_sat(I_c, T_j)` + exponential tail current
  `i_tail(t) = I_c0 · I_tail_fraction · exp(-t/τ_tail)` + `Eon/Eoff/Erec`.
- `DiodeCatalog` with `V_f(I_f, T_j)` + `Q_rr(I_f, di/dt)` + reverse-
  recovery shape factor `s_rec`.
- `device_catalog_yaml.hpp` loader returning
  `std::variant<MosfetCatalogParams, IgbtCatalogParams, DiodeCatalogParams>`.
- 6 reference parts at `devices/catalog/`:
  Infineon IPP60R190P7 (Si CoolMOS), Wolfspeed C3M0065090J (SiC),
  GaN Systems GS66508T, Infineon IKW40N120T2 (Si IGBT TrenchStop2),
  Wolfspeed C4D20120D (SiC Schottky), Vishay VS-30CTH02 (fast-recovery).

**Gates**: G.1 Eon scales linearly per-catalog within 10 %; G.2
conduction loss within ±5 % over 25–125 °C; G.3 Si Q_rr ramps with
di/dt vs SiC Schottky stays flat; G.4 6 of 6 catalog YAMLs load.

**Key files**: `core/include/pulsim/v1/catalog/*.hpp`,
`core/tests/test_catalog_phase{1..8}*.cpp`,
`devices/catalog/<vendor>/<part>.yaml`, `docs/catalog-devices.md`.

---

## Fase 2 — Domain library (3 changes)

Library that takes engineers from "I have a Pulsim simulator" to
"I'm designing a converter / motor drive / grid inverter."

### `add-converter-templates` (archived 2026-05-10)

**Delivered**: parametric converter factory with auto-design + PI
compensator.

- `ConverterRegistry` singleton with Levenshtein "did you mean"
  suggestion on typos.
- `expand_buck`, `expand_boost`, `expand_buck_boost` (3 of 10 spec'd
  topologies — bridges / LLC / DAB / PFC are the deferred follow-ups).
- Auto-design: L sized for ≤ 30 % current ripple, C for ≤ 1 % voltage
  ripple. User-supplied `L`, `C`, `Rload` override.
- `PiCompensator` with trapezoidal integrator + anti-windup back-
  calculation. `from_crossover(f_c, K_plant)` default tune.
- Python: `pulsim.templates.buck(Vin=24, Vout=5, Iout=2, fsw=100e3)`.

**Key files**: `core/include/pulsim/v1/templates/*.hpp`,
`python/pulsim/templates.py`, `core/tests/test_converter_templates.cpp`,
`python/tests/test_converter_templates.py`,
`docs/converter-templates.md`.

### `add-motor-models` (archived 2026-05-10)

**Delivered**: PMSM + DC motor + Park/Clarke + PMSM-FOC current loop.

- Mechanical primitives: `Shaft { J, b_friction, friction_coulomb,
  omega }`, `GearBox { ratio, efficiency }`, `ConstantTorqueLoad`,
  `FanLoad`, `FlywheelLoad`.
- Frame transforms (`motors/frame_transforms.hpp`): amplitude-
  invariant Clarke + Park + composite `abc_to_dq` / `dq_to_abc`.
- `Pmsm` device in dq frame with saliency support `(L_d − L_q)·i_d·i_q`.
  Pinned by no-load + locked-rotor gates.
- `DcMotor` (separately excited) with closed-form `steady_state_omega`
  + `mechanical_time_constant`.
- `PmsmFocCurrentLoop` cascaded id / iq PI tuned via pole-zero
  cancellation `K_p = ω_c · L_axis`, `K_i = K_p · R_s / L_axis`.

**Deferrals**: Induction motor (waits on `add-three-phase-grid-library`
+ AD), BLDC commutation, encoder/Hall/resolver sensors.

**Key files**: `core/include/pulsim/v1/motors/*.hpp`,
`core/tests/test_motor_models.cpp`, `docs/motor-models.md`.

### `add-three-phase-grid-library` (archived 2026-05-10)

**Delivered**: 3φ utility-side library — sources + PLLs + symmetrical
components + inverter templates.

- `ThreePhaseSource` (balanced sinusoidal), `ThreePhaseSourceProgrammable`
  (per-phase scale + `evaluate_with_sag` step helper),
  `ThreePhaseHarmonicSource` (fundamental + harmonic list).
- 3 PLLs: `SrfPll` (single-PI on V_q), `DsogiPll` (dual SOGI bank +
  SrfPll for unbalance robustness), `MafPll` (1-period MAF kills
  integer harmonics).
- Symmetrical components: `fortescue` + `inverse_fortescue` +
  `unbalance_factor` (Fortescue decomposition into 0/+/− sequence).
- `GridFollowingInverter` (SrfPll + dq decoupled current loops + P/Q
  → id*/iq*) auto-tuned via `K_p = 2·ζ·ω_pll/V_pk`.
- `GridFormingInverter` with P-f and Q-V droop control.

**Gates**: G.1 SrfPll locks within 50 ms / ±3°; G.4 grid-forming Q-V
droop produces V_loaded/V_no_load ≈ 0.95 at 5 % rated reactive demand.

**Key files**: `core/include/pulsim/v1/grid/*.hpp`,
`core/tests/test_grid_library.cpp`, `docs/three-phase-grid.md`.

---

## Fase 3 — Differentiation (5 changes)

Features no other open-source simulator has — opens up HIL / MIL /
compliance / yield / property-test workflows.

### `add-realtime-code-generation` (archived 2026-05-10)

**Delivered**: Python `pulsim.codegen.generate(circuit, dt, out_dir)`
producing self-contained C99 model module from any PWL-admissible
circuit.

- Pipeline: `Simulator::linearize_around` → reduce descriptor form
  (drop algebraic V-source rows) → discretize via van Loan augmented-
  matrix `expm(A · dt)` → emit `model.c / model.h / model_test.c`.
- Stability check: discrete spectral radius `rho(A_d) < 1` enforced;
  fails loud with `RuntimeError` recommending smaller dt.
- ROM/RAM estimate: `4·(|A_d| + |B_d| + |C| + |D|)` bytes for ROM,
  `4·2·state_size` for RAM. RC = 56 B / 8 B; LC = 192 B / 16 B.
- PIL parity test compiles generated C with system gcc, runs binary,
  confirms ±0.1 % per-step agreement vs Python-side reference (gate G.1).

**Deferrals**: ARM Cortex-M7 / Zynq target profiles (CMSIS-DSP +
qemu-arm), multi-topology codegen (`switch(topology)` body), CLI
wrapper.

**Key files**: `python/pulsim/codegen/*.py`,
`python/tests/test_codegen.py`, `docs/code-generation.md`.

### `add-fmi-export` (archived 2026-05-10)

**Delivered**: Python `pulsim.fmu.export(circuit, dt, out_path)`
producing FMI 2.0 Co-Simulation `.fmu` archive.

- 4-step pipeline: codegen → emit `fmu_entry.c` (FMI callback wrapper)
  → emit `modelDescription.xml` → compile shared library → zip to
  `binaries/<platform>/<lib>` + `sources/` + `modelDescription.xml`.
- 13 FMI 2.0 CS callback symbols: `fmi2GetVersion`,
  `fmi2GetTypesPlatform`, `fmi2Instantiate`, `fmi2FreeInstance`,
  `fmi2SetupExperiment`, `fmi2EnterInitializationMode`,
  `fmi2ExitInitializationMode`, `fmi2Reset`, `fmi2Terminate`,
  `fmi2SetReal`, `fmi2GetReal`, `fmi2DoStep`, `fmi2CancelStep`.
- Value reference layout: `1..` inputs, `1000..` outputs, `2000..`
  internal state.
- ctypes-loaded round-trip: `fmi2Instantiate` + 10 `fmi2DoStep` cycles
  succeed.

**Deferrals**: Model Exchange export (needs AD), FMU import master,
FMI 3.0, cross-tool validation in OMSimulator/Simulink/Dymola.

**Key files**: `python/pulsim/fmu/*.py`,
`python/tests/test_fmu_export.py`, `docs/fmi-export.md`.

### `add-monte-carlo-parameter-sweep` (archived 2026-05-10)

**Delivered**: Python `pulsim.sweep.run(circuit_factory, parameters,
metrics, ...)` with 5 sampling strategies + metric library + dual
executors.

- Strategies: `cartesian` (full enumeration), `monte_carlo` (IID),
  `lhs` (Latin Hypercube), `sobol`, `halton` (low-discrepancy QMC).
  All seeded for bit-identical reruns.
- Distributions: `Distribution.uniform / normal / loguniform /
  triangular`. Each carries an `inverse_cdf(u)` so QMC samplers drive
  every distribution type uniformly.
- Metrics: `steady_state(channel, t_window)`, `peak`, `rms`,
  `settling_time(target, tolerance)`, `custom(name, fn)`.
- Executors: `serial`, `joblib` (loky pool, default `n_workers =
  cpu_count() - 1`).
- `SweepResult` with `parameters`, `metrics`, `failed`, `percentile(name,
  q)`, `to_pandas()`.

**Deferrals**: GPU backend (CuPy/Numba), Sobol sensitivity indices,
Optuna optimization wrapper, Dask cluster executor.

**Key files**: `python/pulsim/sweep/*.py`,
`python/tests/test_sweep.py`, `docs/parameter-sweep.md`.

### `add-property-based-testing` (archived 2026-05-10)

**Delivered**: Hypothesis-driven property test suite with 5 invariant
families.

- `python/tests/properties/strategies.py`: `gen_passive_rc`,
  `gen_passive_rlc`, `gen_resistor_divider` with physical bounds (R: 1
  Ω – 1 MΩ, C: 1 pF – 1 mF, L: 1 nH – 100 mH, V: 0.1 – 100 V).
- `test_kcl.py`: KCL holds at DC OP + after transient on randomized
  voltage dividers.
- `test_energy.py`: RC steady-state V_C → V_src; charging
  monotonicity (catches sign-error regressions).
- `test_passivity.py`: resistor `P_R(t) = (V_in − V_C)²/R ≥ 0` per
  step.
- `test_pwl_invariants.py`: cache hit rate ≥ 92 % + no DAE fallback
  on linear PWL.

**Deferrals**: C++ RapidCheck integration, reciprocity (linear AC),
KVL via incidence matrix, periodicity gate, full energy-balance
integral.

**Key files**: `python/tests/properties/*.py`,
`docs/property-based-testing.md`.

### `add-closed-loop-benchmarks` (archived prior to roadmap drive)

**Delivered**: 3 closed-loop YAML circuits (`ll14_buck_closed_loop.yaml`,
`ll15_boost_closed_loop.yaml`, `ll16_flyback_closed_loop.yaml`) with
PI controllers + PWM generators registered in
`benchmarks/manifests/local_limit.yaml`.

**Key files**: `benchmarks/circuits/{ll14, ll15, ll16}_*.yaml`.

---

## Fase 4 — Hygiene (2 changes)

### `refactor-unify-robustness-policy` (archived 2026-05-10)

**Delivered**: `RobustnessProfile` typed knob bundle as the canonical
source of truth.

- `enum class RobustnessTier { Aggressive, Standard, Strict }`.
- `RobustnessProfile { newton_max_iters, newton_tol_residual, ...,
  enable_source_stepping, allow_dae_fallback, ... }` (14 fields).
- Factory `RobustnessProfile::for_tier(tier)` derives knobs per tier.
- `parse_robustness_tier(string)` for YAML/CLI dispatch.

**Deferrals**: Removing legacy `apply_robust_*_defaults` from
`bindings.cpp` + `_tune_*_for_robust` from `python/__init__.py` is
the mechanical follow-up.

**Key files**: `core/include/pulsim/v1/robustness_profile.hpp`,
`core/tests/test_robustness_profile.cpp`, `docs/robustness-policy.md`.

### `refactor-modular-build-split` (archived 2026-05-10)

**Delivered**: build-time bench harness establishing the baseline for
the (deferred) bindings + library split.

- `scripts/build_bench.py` measures clean + incremental rebuild
  wallclock, emits JSON artifact for CI ratchet.
- Baseline on Apple Silicon / AppleClang 17 / Release+LTO /
  `pulsim_tests`: 39.16 s clean, 26.13 s incremental, 66.7 % ratio.

**Deferrals**: The actual `python/bindings.cpp` (2,857 lines) split
into `bindings/{devices,control,simulation,parser,solver}.cpp` +
library partition into `pulsim_core / pulsim_simulation / pulsim_periodic`.
Mechanical refactor that benefits from the bench baseline above for
objective wall-clock proof.

**Key files**: `scripts/build_bench.py`, `docs/build-system.md`.

---

## File-level inventory

### New C++ headers (header-only libraries)

```
core/include/pulsim/v1/
├── frequency_analysis.hpp              (ac_sweep, fra, linearize)
├── robustness_profile.hpp              (RobustnessProfile + tiers)
├── magnetic/
│   ├── bh_curve.hpp                    (BHCurve* + Steinmetz + iGSE + J-A)
│   ├── core_catalog.hpp                (YAML loader for vendor cores)
│   ├── hysteresis_inductor.hpp         (J-A wrapper)
│   ├── saturable_inductor.hpp
│   └── saturable_transformer.hpp
├── catalog/
│   ├── lookup_table_2d.hpp             (bilinear LUT)
│   ├── mosfet_catalog.hpp
│   ├── igbt_catalog.hpp
│   ├── diode_catalog.hpp
│   └── device_catalog_yaml.hpp         (YAML loader)
├── templates/
│   ├── registry.hpp                    (ConverterRegistry)
│   ├── buck_template.hpp
│   ├── boost_template.hpp
│   ├── buck_boost_template.hpp
│   └── pi_compensator.hpp
├── motors/
│   ├── mechanical.hpp                  (Shaft, GearBox, loads)
│   ├── frame_transforms.hpp            (Park / Clarke)
│   ├── pmsm.hpp
│   ├── dc_motor.hpp
│   └── pmsm_foc.hpp                    (cascaded PI current loop)
└── grid/
    ├── three_phase_source.hpp
    ├── pll.hpp                         (SrfPll, DsogiPll, MafPll)
    ├── symmetrical_components.hpp      (Fortescue)
    └── inverter_templates.hpp          (grid-following / forming)
```

### Modified C++ files

- `core/src/v1/transient_services.cpp` — LinearFactorCache, Phase 4.3
  workspace hoisting.
- `core/src/v1/simulation.cpp` — `linearize_around`, `run_ac_sweep`,
  `run_fra`.
- `core/include/pulsim/v1/simulation.hpp` — LinearSystem, AcSweep*,
  Fra* structs.
- `core/include/pulsim/v1/runtime_circuit.hpp` —
  `set_ac_perturbation`, `find_connection`/`find_device` promoted to
  public.
- `core/CMakeLists.txt` — registered ~20 new test files.

### New Python modules

```
python/pulsim/
├── frequency_analysis.py               (bode_plot, nyquist_plot, fra_overlay,
│                                         export_*_csv/json, load_ac_result_csv)
├── templates.py                        (buck, boost, buck_boost factories)
├── codegen/
│   ├── __init__.py
│   └── generator.py                    (generate, discretize_state_space)
├── sweep/
│   ├── __init__.py
│   ├── distributions.py                (Distribution, sample, strategies)
│   ├── metrics.py                      (steady_state, peak, rms, ...)
│   └── runner.py                       (SweepResult, run with executors)
└── fmu/
    ├── __init__.py
    └── exporter.py                     (FMI 2.0 CS export)
```

### New Python tests

```
python/tests/
├── test_frequency_analysis.py          (9 tests)
├── test_converter_templates.py         (5 tests)
├── test_codegen.py                     (3 tests + PIL parity)
├── test_sweep.py                       (13 tests)
├── test_fmu_export.py                  (4 tests + ctypes round-trip)
└── properties/
    ├── strategies.py                   (Hypothesis strategies)
    ├── test_kcl.py                     (2 tests)
    ├── test_energy.py                  (2 tests)
    ├── test_passivity.py               (1 test)
    └── test_pwl_invariants.py          (2 tests)
```

### New C++ tests

```
core/tests/
├── test_linear_cache_numeric_lru.cpp           (Phase 3 LRU)
├── test_linear_cache_phase6_benchmarks.cpp     (gate G.1 + buck speedup)
├── test_frequency_analysis_phase{1..9}*.cpp    (linearize + AC + FRA + perf)
├── test_magnetic_phase{1..6}*.cpp              (B-H, J-A, saturable, validation)
├── test_catalog_phase{1..8}*.cpp               (LUT, MOSFET, IGBT, diode, validation)
├── test_converter_templates.cpp                (registry + buck/boost/buck-boost)
├── test_motor_models.cpp                       (mechanical + Park/Clarke + PMSM + DC + FOC)
├── test_grid_library.cpp                       (sources + PLLs + Fortescue + inverters)
└── test_robustness_profile.cpp                 (tier knob bundle + parse)
```

### Reference data files

```
devices/
├── cores/                              (4 magnetic cores)
│   ├── TDK/N87.yaml
│   ├── Ferroxcube/3C90.yaml
│   ├── Magnetics/MPP_60u.yaml
│   └── EPCOS/N97.yaml
└── catalog/                            (6 vendor parts)
    ├── Infineon/IPP60R190P7.yaml       (Si CoolMOS P7)
    ├── Infineon/IKW40N120T2.yaml       (Si IGBT)
    ├── Wolfspeed/C3M0065090J.yaml      (SiC MOSFET)
    ├── Wolfspeed/C4D20120D.yaml        (SiC Schottky)
    ├── GaNSystems/GS66508T.yaml        (GaN HEMT)
    └── Vishay/VS-30CTH02.yaml          (fast-recovery Si)
```

### New docs pages (linked from `mkdocs.yml`)

```
docs/
├── linear-solver-cache.md
├── ac-analysis.md
├── fra.md
├── magnetic-models.md
├── catalog-devices.md
├── converter-templates.md
├── motor-models.md
├── three-phase-grid.md
├── code-generation.md
├── parameter-sweep.md
├── property-based-testing.md
├── fmi-export.md
├── robustness-policy.md
└── build-system.md
```

### New scripts

```
scripts/
└── build_bench.py                      (clean + incremental wallclock)
```

---

## OpenSpec spec deltas

Specs created (under `openspec/specs/`):

- `code-generation`
- `converter-templates`
- `fmi-export`
- `magnetic-models`
- `motor-models`
- `parameter-sweep`
- `three-phase-grid`
- `ac-analysis`

Specs updated (existing specs gaining new requirements):

- `kernel-v1-core`
- `linear-solver`
- `device-models`
- `python-bindings`
- `netlist-yaml`
- `benchmark-suite`

---

## Deferrals tracked across changes

These are formally documented in each change's archived `tasks.md`.
None block production use of the shipped surface; they're follow-ups
that pair naturally with future changes.

| Deferral | Pairs with | Why deferred |
|---|---|---|
| AD-driven Behavioral linearization (Phase 1.2 of frequency-domain) | Future change | Needs AD residual through `assemble_residual` — PWL path is final |
| Multi-topology codegen (`switch(topology)` body) | `add-realtime-code-generation` follow-up | Topology-bitmask exposure work |
| Heap-allocation zero-count assertion (G.2 of linear-solver-cache) | Future hardening | Needs custom allocator wrapper |
| Phase 3 catalog Coss(Vds) caps stamping | Circuit-variant integration | Same Circuit-DeviceVariant lift |
| Saturable devices on MNA stamp surface | Circuit-variant integration | Same Circuit-DeviceVariant lift |
| FMI 2.0 ME export + import + 3.0 | `add-fmi-export` follow-ups | Needs AD + master orchestration + FMI 3.0 ecosystem maturity |
| GPU sweep backend / Sobol indices / Optuna | `add-monte-carlo-parameter-sweep` Phase 5/7/8 stretch | Each is its own large change |
| C++ RapidCheck property tests | `add-property-based-testing` Phase 8 | FetchContent + GoogleTest dep |
| `bindings.cpp` mechanical split | `refactor-modular-build-split` Phase 2-3 | Risk-bearing refactor; baseline now established |
| Robustness `apply_robust_*_defaults` legacy deletion | `refactor-unify-robustness-policy` Phase 2 | Mechanical call-site updates |
| Bridge / LLC / DAB / PFC converter templates | `add-converter-templates` Phase 4-5 | Each topology family is its own engineering surface |
| Induction motor + BLDC + sensor models | `add-motor-models` Phase 4-6 | Each is well-known math but bench-test heavy |
| Anti-islanding (IEEE 1547) | `add-three-phase-grid-library` Phase 7 | Compliance certification has its own lifecycle |
| Tutorial notebooks | Per-change Phase 10 | Depend on the Circuit-variant integration that surfaces YAML for new device types |
| Cross-tool FMU validation in OMSimulator/Simulink | `add-fmi-export` Phase 7 | License + CI-environment requirement |

---

## Migration / API guidance

**The shipped surface is purely additive.** No existing code path
changes signature or behavior. Specifically:

- All new C++ types live in their own headers (`magnetic/`,
  `catalog/`, `templates/`, `motors/`, `grid/`, `frequency_analysis.hpp`,
  `robustness_profile.hpp`). Existing includes don't pull them in
  unless explicitly added.
- `Simulator::linearize_around`, `run_ac_sweep`, `run_fra` are new
  public methods — opt-in, no impact on the existing
  `dc_operating_point`, `run_transient`, `run_periodic_shooting`
  signatures.
- `BackendTelemetry` gained per-reason cache-invalidation counters +
  symbolic-factor-cache field. Existing fields retained verbatim.
- `Circuit::set_ac_perturbation` is new and opt-in; circuits that
  never call it behave identically to before.
- All Python additions sit under new submodules (`pulsim.codegen`,
  `pulsim.fmu`, `pulsim.sweep`, `pulsim.templates`,
  `pulsim.frequency_analysis`). Existing `pulsim.Simulator` API
  unchanged.

The only behavioral change is the linear-solver cache itself — the
segment-primary stepper now uses an LRU map rather than a single-slot
cache. Every existing test passes unmodified, which validates the
contract.

---

## How to verify locally

```bash
# C++ default build
cmake -B build && cmake --build build --target pulsim_tests pulsim_simulation_tests
build/core/pulsim_tests
build/core/pulsim_simulation_tests

# C++ AD build
cmake -B build_ad -DPULSIM_USE_AD_STAMP=ON && cmake --build build_ad
build_ad/core/pulsim_tests
build_ad/core/pulsim_simulation_tests

# Python (after building build_py)
python3.14 -m pytest python/tests/test_frequency_analysis.py
python3.14 -m pytest python/tests/test_converter_templates.py
python3.14 -m pytest python/tests/test_codegen.py
python3.14 -m pytest python/tests/test_sweep.py
python3.14 -m pytest python/tests/test_fmu_export.py
python3.14 -m pytest python/tests/properties

# Build wallclock bench
python3 scripts/build_bench.py --build-dir build --target pulsim_tests
```

---

## See also

- [`ROADMAP.md`](ROADMAP.md) — original strategic plan that drove the
  16 changes.
- [`docs/`](docs/) — per-feature user-facing documentation.
- [`openspec/changes/archive/`](openspec/changes/archive/) — full
  change-by-change history with `proposal.md`, `design.md`,
  `tasks.md`, and spec deltas for every archived change.
