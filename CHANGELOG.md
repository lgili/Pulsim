# Changelog

All notable changes to Pulsim are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.6.1] — 2026-05-28

### Fixed

* **`pip install pulsim` was broken on v1.6.0** because
  `pulsim/__init__.py` re-exports `wire_chain_from_yaml`, which forced
  a top-level `import yaml as _yaml` in `pulsim.yaml_chain`. PyYAML is
  only listed under the `dev` optional extra (it's not a runtime
  dependency), so the cibuildwheel smoke test (`python -c "import
  pulsim"`) failed on all platforms (Linux/macOS/Windows) and the
  PyPI publish workflow rejected the v1.6.0 wheels.

  Moved the `import yaml` to a lazy import inside
  `wire_chain_from_yaml` — only triggered when the caller passes a
  YAML *string*. Python list/dict chain specs continue to work
  without PyYAML installed. If the YAML-string path is taken without
  PyYAML, the user gets an actionable `ModuleNotFoundError` pointing
  at `pip install 'pulsim[dev]'`.

  No behavioural change for anyone who already had PyYAML
  installed via `pulsim[dev]` or as a transitive dep.

## [1.6.0] — 2026-05-28

### Highlights — Path-Based Event-Driven (DSED) engine + native PWM switch_fn

Lands the **Path-Based Event-Driven (PED)** simulation engine
([PR #62](https://github.com/lgili/Pulsim/pull/62)) — Pulsim's
alternative to the default fixed-step trapezoidal + PWL cache loop.
DSED predicts the next event time analytically (gate edges, body
diode commutation, voltage thresholds), integrates with adaptive
step control between events (DOPRI5 or BDF2 per-mode dispatch), and
handles mask transitions instantaneously without aliasing.

End-result for canonical buck CCM (24V→12V, 100 kHz, 5 ms window,
1007 vs 50001 steps): **DSED is 24× faster than PWL in wall-clock**.
Geo-mean speedup across 6 converter topologies (buck/boost/buck-boost/
half-bridge/floating-cap RLC/NPC split-bus): **14.5× vs PWL**.

#### What you can do now

```python
import pulsim as p

b = p.CircuitBuilder()
b.add_voltage_source("Vin", "in", "gnd", 24.0)
# ... build a buck ...

# DSED engine — variable-step, event-driven, ~24× faster than 'pwl'
sf = p.NativePwm2Switch(T_sw=1e-5, D=0.5, n_switches=2)
res = p.simulate(b, t_end=5e-3, engine='dsed', switch_fn=sf)
```

The `engine='dsed'` API is fully wired through Python with all the
familiar kwargs (rtol, atol, integrator='rk45'/'bdf2'/'auto',
b_extra_fn, switch_fn, initial_state). No user code change needed
beyond the `engine='dsed'` opt-in.

#### Topologies supported

LTI-per-mask circuits: buck, boost, buck-boost, flyback, forward,
half-bridge, full-bridge, NPC split-bus (floating caps), MMC SM-stacks
(floating caps), PFC with AC input, grid-tied inverters with sine
V_grid. Plus rejection of pathological cases (floating caps with
both terminals on ground, parallel caps, inductor loops) with
actionable error messages.

#### Architecture (Bridges 1–13)

* **Bridges 1–5** — algorithm gates: DOPRI5 + PI controller +
  Illinois root finder, BDF2 + Crank-Nicolson bootstrap, stiffness
  detector, per-mode RK45↔BDF2 auto-dispatch. Initial Python
  prototype + C++23 port.
* **Bridge 5.1b** — T^T·M·T congruence transform for **floating
  capacitors** (NPC split bus, MMC SM stacks, half/full-bridge
  differential output caps).
* **Bridges 6/7** — Time-varying source overlay (sine / PWM / pulse)
  + user `b_extra_fn` callback via per-mask projection matrix B.
* **Bridge 8** — Real-converter end-to-end validation (buck, boost,
  NPC split-bus, half-bridge + sine V_in).
* **Bridge 9** — Inductor-loop rejection (parallel L, all-L cycles)
  with clear "merge into L_eq" pointer.
* **Bridge 10** — Pybind11 scheduler wrappers (C++ inner loop,
  Python callbacks). 2.7× per-step over pure Python.
* **Bridge 11** — Native C++ `CircuitBuilderAdapter` — eliminates
  GIL roundtrips on the hot loop. Brought DSED to 13.3× faster
  than PWL.
* **Bridge 12** — Native PWM switch_fn classes (`NativePwm2Switch`,
  `NativeMultiMaskPwm`) detected at construction; scheduler calls
  them in pure C++ without GIL. Brought DSED to 24.3× faster than
  PWL.
* **Bridge 13** — PWL engine also detects native PWM and dispatches
  through a C++ lambda. PWL itself becomes 2× faster on PWM-driven
  circuits.

#### Speedup breakdown (buck CCM, 5 ms, 100 kHz)

| Layer | Wall-clock | per-step | vs PWL (old) |
|---|---:|---:|---:|
| PWL (C++ trap, 50001 steps)               | 52.7 ms | 1.05 µs | 1.0× (baseline) |
| **PWL + native PWM** (Bridge.13)           | **26.2 ms** | **0.52 µs** | **2.0×** |
| DSED Python scheduler (Bridge.5)           | 61.3 ms | 60.8 µs | 0.85× |
| DSED Bridge.10 (Python adapter)            | 22.4 ms | 22.2 µs | 2.3× |
| DSED Bridge.11 (native adapter)            |  3.8 ms |  3.80 µs | 13.3× |
| **DSED Bridge.12 (+ native PWM)**          |  **2.2 ms** |  **2.19 µs** | **24.3×** |

Final v_C = 12.0000 V exact (bit-for-bit match with analytical
D·V_in steady state) across all layers.

#### Tests

* **549/549 C++ tests pass** (Catch2; added 6 LTI extractor tests
  for the congruence transform + inductor-cycle rejection)
* **14 new Python end-to-end tests** (`python/tests/test_dsed_end_to_end.py`)
  covering buck/boost/NPC/floating-cap/sine-input/Bridge.12-vs-Python
  bit-for-bit agreement
* **Total Python test suite: 950 pass**

#### API surface added

```python
# DSED engine
pulsim.simulate(b, engine='dsed', integrator='auto'|'rk45'|'bdf2',
                 rtol=..., atol=..., dt_init=..., h_bdf2=...,
                 stiffness_threshold=..., switch_fn=..., b_extra_fn=...)

# Native PWM (DSED + PWL both detect these)
pulsim.NativePwm2Switch(T_sw, D, n_switches, hs_first=True)
pulsim.NativeMultiMaskPwm(T_sw, phase_boundaries, masks)

# Advanced — direct PED scheduler access (user-defined LTI system)
pulsim.dsed.PEDSimulator(...)
pulsim.dsed.PEDSimulatorBDF2(...)
pulsim.dsed.PEDSimulatorAuto(...)
pulsim.dsed.PIController(...)
pulsim.dsed.StiffnessDetector(...)
pulsim.dsed.BDF2State / bdf2_step(...)
pulsim.dsed.RK45State / rk45_step(...) / interpolate(...)
pulsim.dsed.EventPredictor / EventPredicate / illinois / brent_fallback

# C++ control blocks (advanced, embedded export):
# pulsim._pulsim._NativePIController, _NativePIDController,
# _NativeFirstOrderLowPass — bit-for-bit identical to the Python
# pulsim.control classes but for native step_observer use cases.
```

#### Known limitations / out of scope

* Nonlinear devices (diode Shockley, MOSFET Vth, saturable L) still
  use `pulsim.dsed.PEDSimulator` directly with a user-defined LTI
  system. The PED engine does not model per-operating-point
  linearization.
* Inductor cycles (parallel L) and parallel capacitors are rejected
  with clear errors pointing to the merge-equivalent workaround.

See `notes/DSED_BRIDGE_DESIGN.md` for the full design (~700 lines:
math, MNA→state-space reduction, T^T·M·T congruence for floating
caps, projection matrix B for time-varying sources, native C++
adapter, dispatch hierarchy) and
`openspec/changes/add-path-based-dsed-engine/` for the proposal +
specs.

## [1.5.0] — Unreleased

### Highlights — Phase 2 physics-parity push + PSIM-equivalent loss/thermal/control panels

This release closes **Phase 2 (Physics Parity)** of the v1.x roadmap
and lands a four-way upgrade to the post-hoc analysis surface so
users get the loss / thermal / control workflows they expect from
PSIM and PLECS without leaving Python.

**A. Phase 2 — Physics parity in the C++ kernel**
([PR #51](https://github.com/lgili/Pulsim/pull/51) →
[#52](https://github.com/lgili/Pulsim/pull/52) →
[#53](https://github.com/lgili/Pulsim/pull/53) →
[#54](https://github.com/lgili/Pulsim/pull/54) →
[#55](https://github.com/lgili/Pulsim/pull/55) →
[#56](https://github.com/lgili/Pulsim/pull/56))

* **2.1 — Squirrel-cage induction motor**: header-only C++ port at
  `core/include/pulsim/motors/induction_motor.hpp` + pybind
  `CxxBlockChain.add_induction_motor(...)`. 5-state Krause αβ
  model running at kernel speed.
* **2.2 — Jiles-Atherton hysteretic inductor**: C++ port at
  `core/include/pulsim/magnetics/jiles_atherton.hpp` +
  `CxxBlockChain.add_hysteretic_inductor(...)`. Sign convention
  fix on `v_M` in the b_extra row matches the Python observer.
* **2.3 — Sensorless rotor observers**: C++ port of
  `SlidingModeObserver` (PMSM Utkin + LPF + PLL) and
  `FluxMRASObserver` (IM Schauder voltage + current with
  bootstrap-fixed normalised cross-product) at
  `core/include/pulsim/observers/sensorless.hpp`. New
  `CxxBlockChain.add_sliding_mode_observer(...)` /
  `.add_flux_mras_observer(...)`.
* **2.4 — Adaptive Runge-Kutta**: `DormandPrince5` and `RadauIIA3`
  shipped (Python in v1.4.x, C++ port for standalone use in v1.5).
  `simulate(integrator=)` schema landed; kernel wiring deferred to
  v1.6 — see "v1.6 deferred" note below.

**B. YAML composite devices + `chain:` wiring**
([PR #54](https://github.com/lgili/Pulsim/pull/54),
[#56](https://github.com/lgili/Pulsim/pull/56))

* New device kinds in `circuit:` — `induction_motor` and
  `hysteretic_inductor`. The loader expands them into
  deterministic branch-id schemes (`IM_Lsig_{a,b,c}`,
  `IM_E_{a,b,c}`, `L_core_L0`, `L_core_V_M`).
* New `pulsim.wire_chain_from_yaml(loaded, chain_spec)` resolves
  the deterministic branch names and stamps a `CxxBlockChain`.
  Four block types: `induction_motor`, `hysteretic_inductor`,
  `sliding_mode_observer`, `flux_mras_observer`.
* See [docs/yaml-chain.md](yaml-chain.md).

**C. PSIM-style loss + thermal pipeline**
([PR #58](https://github.com/lgili/Pulsim/pull/58))

* `device_loss_summary` extended to cover **resistor + inductor +
  ideal-switch + switched-diode** in one pass, with optional
  per-device datasheet annotations:
  - `diode_specs={"D1": {"Q_rr": ...}}` or
    `{"E_rr_ref": ..., "V_R_ref": ...}` → reverse-recovery energy
    per turn-off event, accumulated from `commutation_events`.
  - `switch_specs={"M1": {"E_on_ref": ..., "E_off_ref": ...,
    "V_ref": ..., "I_ref": ...}}` → PSIM-style turn-on / turn-off
    energy scaled by `(V_blocking, I_load)` at each `switch_fn`
    edge.
  - `core_loss_specs={"L1": {"material": "N87", ...}}` →
    Steinmetz / iGSE core loss from `pulsim.magnetic`.
* New **`device_thermal_summary(builder, result,
  thermal_specs=...)`** pipes the loss output through a per-device
  Foster network and returns per-device `T_j(t)` traces, plus
  `T_j_avg`, `T_j_peak`, `P_total_avg`, `R_th_total`.
* Strict spec validation — unknown device names in any `*_specs`
  raise `KeyError`; non-positive geometry on core loss raises
  `ValueError`. No silent zeros.
* Shared `_result_views` helpers between `losses.py` and
  `thermal.py` eliminate duplicated result-walk code.
* See [docs/losses-and-thermal.md](losses-and-thermal.md).

**D. PSIM/PLECS-style "C block" via Numba JIT**
([PR #59](https://github.com/lgili/Pulsim/pull/59))

* New `@pulsim.fast_block` decorator turns a Python control
  function into a Numba-LLVM-compiled native callable. Same
  authoring contract as PSIM's Custom C Block (read inputs,
  mutate `state` in-place, return scalar) without runtime `cc`
  invocation, cross-OS compiler dance, or `.so` plumbing.
* `pip install pulsim[fast]` enables the JIT path; the optional
  dep keeps the base install lean. Without numba, `@fast_block`
  raises a clear `ImportError` with the install hint.
* See [docs/fast-block.md](fast-block.md) and the runnable
  showcase
  [`examples/scripts/run_fast_block_pi_buck.py`](../examples/scripts/run_fast_block_pi_buck.py).

### Added

* `pulsim.SlidingModeObserver` / `FluxMRASObserver` — C++ kernel
  adapters via `CxxBlockChain.add_*` (Phase 2.3).
* `pulsim.wire_chain_from_yaml(loaded, chain_spec)` — Python
  glue between the YAML loader and `CxxBlockChain`.
* `SimulationOptions.integrator` / `rtol` / `atol` / `dt_init`
  fields + matching YAML `simulation:` block keys (Phase 2.4
  schema, kernel wiring deferred to v1.6).
* `simulate(integrator=, rtol=, atol=, dt_init=)` kwargs —
  `"kernel"` default unchanged; `"dopri5"` / `"radau"` raise
  `NotImplementedError` with a v1.6 pointer.
* `pulsim.device_loss_summary` extended (see Highlights C).
* `pulsim.device_thermal_summary` — new (see Highlights C).
* `pulsim.FastBlock`, `pulsim.fast_block` — new (Highlights D).
* `pulsim.magnetic` Steinmetz / iGSE helpers + N87 / 3F4 / 3C90
  built-in material catalogue used by `device_loss_summary`'s
  core-loss path.
* Optional dep: `pulsim[fast]` → `numba>=0.58`.

### Changed

* `device_loss_summary` previously skipped switches / diodes /
  magnetic devices silently; now they're reported with the
  datasheet annotations described above. The signature gains
  `switch_specs`, `diode_specs`, `core_loss_specs` kwargs.
* **(Breaking)** `device_loss_summary` now **raises `KeyError`**
  when any `*_specs` mapping references a device name / branch_id
  that isn't in the builder. Was a silent skip in v1.4. Update
  YAMLs and test fixtures to use the actual device names.
* `KNOWN_LIMITATIONS.md` § "Per-device loss reporting" rewritten
  to reflect the v1.5 coverage — what's actually covered today
  vs the sub-`dt` switching-transient waveform shapes that the
  fixed-`dt` kernel still doesn't resolve.

### Fixed

* MRAS bootstrap fix: the normalised cross-product
  `ε / (|ψ_ref|·|ψ_adj|)` is now the default
  (`normalise_eps=True`) in `FluxMRASObserver` — resolves
  cold-start convergence for IM sensorless on `ω̂_init=0`.
* JA observer `v_M` sign in `b_extra` was inverted on the
  v1.4.x release branch — now matches the Python observer's
  `+N·A·µ₀·dM/dt` convention.
* `device_thermal_summary` previously computed `P_core_avg`
  internally but omitted the field from the output dict —
  fixed, users now see the core contribution that drove `T_j`.
* `_inductor_core_loss` previously returned silent zeros for
  invalid geometry (`N_turns ≤ 0` etc.); now raises `ValueError`.
* Narrowed `except Exception` around iGSE fall-back to
  `except ValueError` so unrelated bugs surface instead of
  hiding.

### v1.6 deferred

The Phase 2.4 schema for `simulate(integrator="dopri5"|"radau")`
landed, but actual execution waits on a `PwlStateSpaceCache`
refactor (the cache stores `J = G + (2/dt)·M` in trap-companion
form; adaptive RK needs `(G, M, b)` separately and DAE-aware
Radau — augmented MNA's mass matrix is structurally singular).
Same blocker postpones in-kernel `R_DS_on(T_j)` live feedback
and the stiff-thermal Radau example. See the v1.6 milestone for
the cache refactor work-item.

## [1.4.0] — 2026-05-24

### Highlights — In-house complex sparse LU + generalised path-based update framework

This release bundles **two algorithmic contributions** that were
originally scoped as separate releases but ship together as
v1.4.0 since neither had been tagged yet:

**A. In-house complex sparse LU** (per
[`openspec/changes/add-pulsim-complex-sparse-lu/`](openspec/changes/add-pulsim-complex-sparse-lu/)) —
templates `PulsimSparseLuSolver` on `Scalar` and migrates
`run_mna_sweep` to the new `PulsimComplexSparseLuSolver`
(= `PulsimSparseLuSolverT<std::complex<Real>>`). Completes the
v1.3.0 "no third-party LU in production" agenda — the AC sweep
code path no longer compiles `Eigen::SparseLU<complex>`.
`Backend::Eigen` is retained as the IEEE TPEL §VI.B
paper-comparison baseline.

**B. Generalised path-based update framework** (per
[`openspec/changes/add-generalised-path-refactor/`](openspec/changes/add-generalised-path-refactor/)) —
generalises the v1.3.0 single-bit path-based partial refactor to
**three SMPS-relevant use cases** that no open-source
power-electronics simulator currently exploits:

1. **Multi-bit switch transitions** (Part A) — SPWM with multiple
   legs commutating simultaneously, multilevel commutation patterns.
   v1.3.0 unconditionally routed those to full `factorize()`; v1.4.0
   attempts the union of etree paths when the union covers ≤
   `MAX_PATH_LENGTH_RATIO` (default `0.6`) of the matrix.
2. **Parametric value changes** (Part B) — `R`, `L`, `C`, source `V`
   updates for sweep / Monte Carlo / design-optimisation workloads.
   v1.3.0 forced a fresh `analyze + factorize` rebuild per sweep
   point (~100 µs/point cold path); v1.4.0 reuses both the symbolic
   factor AND most of L+U via the same path-union machinery.
3. **Single-bit Gray-code flips** (preserved from v1.3.0) — same
   2.7-2.9× speedup at n_state ≥ 12 documented in
   `RANK1_RESULTS.md`.

User-facing Python helpers `sweep_path_aware` /
`monte_carlo_path_aware` ship as drop-in replacements for `sweep` /
`monte_carlo`. Auto-fallback to the legacy path when the swept
parameter name is unknown to the builder; the user sees a
`RuntimeWarning` and the same `SweepResult` shape.

### Performance — Part A multi-bit microbench

Captured 2026-05-24 on macOS Apple Silicon (see
[`artigos/02_tpel_methods/benchmarks/MULTI_BIT_RESULTS.md`](artigos/02_tpel_methods/benchmarks/MULTI_BIT_RESULTS.md)).
Pulsim path-union speedup vs the v1.3.0 emulation (Eigen sliding
solver = full factorize per flip):

| n_state | δ = 1 | δ = 2 | δ = 3 | δ = 4 |
|--------:|------:|------:|------:|------:|
| 10      | 3.12× | 1.62× | 1.61× | 1.42× |
| 14      | 1.72× | 1.58× | 1.58× | 1.42× |
| 18      | 1.56× | 1.28× | 1.51× | 1.25× |
| 22      | 1.36× | 1.42× | 1.54× | 1.51× |
| 26      | 1.55× | 1.46× | 1.33× | 1.42× |

Multi-bit hit rate decays gracefully with δ:
~40–50 % of 2-bit transitions take the path-union path,
~20–25 % at δ = 3, ~8–19 % at δ = 4. The remainder gracefully
fall back to full factorize without regression vs v1.3.0.

### Performance — Part B parametric microbench

Captured 2026-05-24 on the same hardware (see
[`artigos/02_tpel_methods/benchmarks/PARAMETRIC_RESULTS.md`](artigos/02_tpel_methods/benchmarks/PARAMETRIC_RESULTS.md)).
Pulsim `refactor_parametric` speedup vs the legacy rebuild-the-
cache-from-scratch-per-sweep-point pattern (current
`pulsim.sweep.sweep(...)` semantics):

| n_state | 50 pts | 100 pts | 500 pts | 1000 pts |
|--------:|-------:|--------:|--------:|---------:|
| 8       | 5.18×  | 3.29×   | 3.55×   | 3.68×    |
| 14      | 3.57×  | 3.02×   | 3.51×   | 3.35×    |
| 26      | 3.53×  | 3.31×   | 3.38×   | 3.40×    |

**Zero fallbacks across all 12 (n_switches × n_sweep_points)
cells** — every refactor_parametric call took the path-based
update successfully on this fixture family.

### Added

- **`pulsim::sparse::MAX_PATH_LENGTH_RATIO`** — compile-time
  tunable (default `0.6`). Path-based update is skipped when the
  union-path length exceeds this fraction of `n`. See
  `openspec/changes/add-generalised-path-refactor/design.md`
  Decision 2 for the empirical break-even rationale.
- **`DirectSolverT<Scalar>::partial_refactor_count_path(changed_cols)`**
  — virtual query method. Returns the length of the union path that
  `partial_refactor` would walk **without executing the refactor**.
  Used by `solve_rank1` to consult `MAX_PATH_LENGTH_RATIO` before
  attempting path-based update on multi-bit transitions.
  Default implementation returns 0; `PulsimSparseLuSolverT<Scalar>`
  overrides with the real walk.
- **`PulsimSparseLuSolverT<Scalar>::partial_refactor_count_path`**
  — production implementation. Walks the etree path of each column
  in the **hypothetical union** of `varying_set_ + changed_cols`,
  deduplicates via an in-path bitmap. Pure read-only — does not
  mutate solver state. Companion `partial_refactor_path_ratio`
  wraps `count_path / n` for the common comparison expression.
- **`pulsim::pwl::CacheMetrics::multi_bit_rank1_hits`** — new
  counter for multi-bit successes via path-union `partial_refactor`.
  v1.3.0 routed all multi-bit transitions to `full_refactor_hits`;
  v1.4.0 splits them between this new counter (success path) and
  `full_refactor_hits` (path too long → fallback).
  Invariant: `rank1_hits + multi_bit_rank1_hits + full_refactor_hits
  + fallbacks == N`.
- **`pulsim::pwl::DevicePool::columns_affected_by_switch(sw_idx,
  graph)`** — new helper returning the MNA columns affected by
  toggling switch `sw_idx`. Mirrors the
  `branch_var_id_for_source` access pattern. Used by
  `compute_changed_columns_` and (in a future cycle) by Python
  bindings exposing the switch→column map.
- **`core/tests/benchmarks/test_bench_multi_bit_rank1.cpp`** —
  3-backend microbench across `(N, δ) ∈ {8,12,16,20,24} × {1,2,3,4}`.
  1000 random transitions per cell.
- **`artigos/02_tpel_methods/benchmarks/MULTI_BIT_RESULTS.md`** +
  `multi_bit_microbench.csv` — full writeup mirroring
  `RANK1_RESULTS.md`'s structure.
- **7 new C++ unit tests** in `core/tests/layer0/test_pulsim_lu_solver.cpp`
  (4 spec-mandated v1.4.0 scenarios: multi-col `partial_refactor`,
  monotone `count_path`, empty-input no-op, `MAX_PATH_LENGTH_RATIO`
  range gate) and `core/tests/layer4/test_pwl_cache_rank1.cpp`
  (3 cache-level scenarios: 2-bit transition routing, telemetry
  invariant under mixed Hamming workload, 4-bit transition correctness).

#### Part B — parametric refactor

- **`pulsim::pwl::ParametricRefactorResult`** + **`ParametricRefactorMode`**
  + **`ParametricUpdate`** — new public types in `cache.hpp`.
  Result invariant: `path_refactor_hits + fallback_hits ==
  masks_processed`.
- **`PwlStateSpaceCache::refactor_parametric`** — new C++ API
  with two overloads:
  - Single-param: `refactor_parametric(branch_id, new_value, mode)`
  - Batch: `refactor_parametric(span<const ParametricUpdate>, mode)`
  Pushes parameter updates through the pool, walks every active
  mask (or just the rank-1 mask in `Mode::CurrentOnly`),
  re-assembles `(J, b)` at the new values, and calls
  `partial_refactor(new_J, affected_cols)` for each segment.
  Falls back to fresh `factorize()` when path too long
  (gated by `MAX_PATH_LENGTH_RATIO`) or backend lacks
  `partial_refactor` support.
- **`DevicePool::columns_affected_by_branch(branch_id, graph)`**
  — returns the MNA columns that depend on a branch's stored
  parameter value(s). Resistor/Switch/Capacitor → both endpoint
  cols; Inductor → its branch-current col; VoltageSource →
  empty (RHS-only). Unsupported device kinds → empty (falls back).
- **`DevicePool::update_resistor_R / update_inductor_L /
  update_capacitor_C / update_voltage_source_V`** — value
  mutators dispatching on the stored variant via
  `std::get_if`. Throws `std::out_of_range` on kind mismatch.
- **`CircuitBuilder::branch_id_of(name)`** — inverse of
  `name_of(branch_id)`. Throws on unknown name.
- **`CircuitBuilder::update_resistor_R(name, R_ohms)` (+ inductor /
  capacitor / voltage_source variants)** — convenience wrappers
  that combine `branch_id_of` + the pool mutator. Designed for
  the user-facing parametric refactor pattern:
  ```python
  b.update_resistor_R("R_load", 3.0)
  cache.refactor_parametric(b.branch_id_of("R_load"), 3.0)
  ```
- **pybind11 bindings** for all of the above — `ParametricRefactorResult`
  + `ParametricRefactorMode` enum + cache methods + builder helpers
  exposed to Python via `python/bindings.cpp`. Smoke-tested
  end-to-end on the `pulsim 1.4.0` wheel.
- **`core/tests/layer4/test_pwl_cache_parametric.cpp`** — 6 new
  test cases / 57 assertions covering: single-param sweep parity
  vs fresh-rebuild within 1e-10, two-param simultaneous parity,
  empty-updates no-op, unsupported-kind throws, `Mode::CurrentOnly`
  processes 1 mask, telemetry invariant over 10 sweep points.
- **`core/tests/benchmarks/test_bench_parametric_sweep.cpp`** —
  3-backend microbench across `(n_switches, n_sweep_points) ∈
  {2,4,8} × {50,100,500,1000}` on parallel-leg buck fixtures.
- **`artigos/02_tpel_methods/benchmarks/PARAMETRIC_RESULTS.md`**
  + `parametric_microbench.csv` — full writeup.

#### Part C — In-house complex sparse LU (AC sweep migration)

- **`pulsim::sparse::PulsimSparseLuSolverT<Scalar>`** — the
  templated class. Backward-compat type aliases keep every Layer 1-9
  call site source-compatible:
  ```cpp
  using PulsimSparseLuSolver        = PulsimSparseLuSolverT<Real>;
  using PulsimComplexSparseLuSolver = PulsimSparseLuSolverT<std::complex<Real>>;
  ```
- **`pulsim::sparse::DirectSolverT<Scalar=Real>`** — the templated
  abstract base. `DirectSolver = DirectSolverT<Real>` for backward
  compat. Same pattern for `SparseLuSolverT<Scalar=Real>` /
  `SparseLuSolver = SparseLuSolverT<Real>`.
- **`pulsim::sparse::make_default_solver_t<Scalar>(n, hint)`** —
  template factory. The non-template
  `make_default_solver(n, hint)` is now a shim that dispatches to
  `make_default_solver_t<Real>(n, hint)`.
- **`pulsim::VectorT<Scalar>`** and **`pulsim::sparse::MatrixT<Scalar>`** /
  **`pulsim::sparse::TripletT<Scalar>`** templates with
  `Vector` / `Matrix` / `Triplet` backward-compat aliases for `Real`.
- **`core/tests/layer0/test_pulsim_lu_solver_complex.cpp`** — 5 new
  test cases / 31 assertions covering the complex specialisation:
  Hermitian PD identity, asymmetric MNA 8×8 (forces partial pivoting
  at the zero-diagonal voltage-source row), single-column
  partial_refactor parity, solve-before-factorize lifecycle, factory
  dispatch.
- **`core/tests/analysis/test_mna_sweep.cpp`** — 2 integration
  tests through `run_mna_sweep`: RC low-pass within 0.1 dB / 1°
  of `1/(1+jωRC)` across 50 frequencies; series RLC peak within
  1.5 % of `1/(2π√(LC))` (Q ≈ 5).
- **`core/tests/benchmarks/test_bench_ac_sweep.cpp`** — 2-backend
  AC sweep microbench across `n ∈ {8, 16, 32, 64, 128}`,
  100 log-spaced frequencies from 1 Hz to 1 MHz.
- **`artigos/02_tpel_methods/benchmarks/AC_SWEEP_RESULTS.md`** +
  `ac_sweep_microbench.csv` — full writeup of the
  Pulsim-vs-Eigen parity story.

### Changed

- **`core/include/pulsim/analysis/mna_sweep.hpp`** —
  `Eigen::SparseLU<ComplexSparseMatrix, COLAMDOrdering<Index>>`
  replaced with `sparse::PulsimComplexSparseLuSolver`. Lifecycle:
  `analyze(M)` → `factorize(M)` → `solve(b, x)`, all returning
  `bool` (vs Eigen's `info()` enum). `ComplexSparseMatrix` switched
  from RowMajor to ColMajor to match the in-house solver's CSC
  input format (no transpose-and-copy per frequency).
  `#include <Eigen/SparseLU>` removed — no longer needed on the
  production path.
- **`PwlStateSpaceCache` constructor** — `pool` parameter changed
  from `const DevicePool&` to `DevicePool&`. Existing callers that
  pass a non-const builder pool continue to compile unchanged.
  Required so `refactor_parametric` can drive `pool.update_*`.
- **`PwlStateSpaceCache::try_make_segment`** — segments are now
  built with `Backend::Auto` (= Pulsim in-house LU) by default,
  not the Eigen baseline. Numerically bit-identical on real-scalar
  SPD matrices; enables `refactor_parametric` to use
  `partial_refactor` on the cached segment factors without an
  explicit `set_segment_backend` step.
- **`PwlStateSpaceCache::solve_rank1`** — multi-bit routing logic
  rewritten. For `delta_bits >= 2`, the cache now:
  1. Computes the deduped `changed_cols` set via
     `compute_changed_columns_`.
  2. Queries `solver.partial_refactor_count_path(changed_cols)`.
  3. If `path_length / n ≤ MAX_PATH_LENGTH_RATIO`, calls
     `partial_refactor`; on success counts `multi_bit_rank1_hits++`.
  4. Otherwise calls `factorize()` and counts `full_refactor_hits++`.
  Single-bit transitions (`delta_bits == 1`) keep v1.3.0 behavior:
  always try `partial_refactor` without the ratio gate.
- **`compute_changed_columns_`** now deduplicates via `std::set<Index>`
  before returning. Switches sharing a node (common in half/full-bridge
  topologies) previously produced duplicate column entries. v1.4.0's
  `partial_refactor_count_path` requires a canonical input to give
  a meaningful answer to the ratio gate, so dedup happens at the call
  site.

### Removed

- **Implicit production dependency on `Eigen::SparseLU<std::complex<Real>>`**.
  The Eigen complex SparseLU instantiation is no longer compiled
  into the AC sweep code path. `Backend::Eigen` keeps the path
  explicitly available for paper-comparison purposes.

### Migration notes

- **`CacheMetrics` ABI**: the struct grew a new field
  (`multi_bit_rank1_hits`). Existing callers reading
  `rank1_hits` / `full_refactor_hits` / `fallbacks` continue to
  compile + work. Code that pinned `full_refactor_hits == N` for
  N multi-bit transitions will see those calls land in
  `multi_bit_rank1_hits` instead — update test fixtures to use the
  telemetry invariant `rank1 + multi_bit + full + fallbacks == N`.
- **`solve_rank1` behavior on Eigen backend** (no
  `partial_refactor` support): unchanged. Every transition falls
  back to full factorize and counts under `full_refactor_hits`.
- **Pulsim backend behavior on a 4-switch n_state=6 fixture**: a
  small fraction (~3-5 %) of multi-bit transitions now hit
  `fallbacks` instead of `full_refactor_hits` because the pivot
  threshold check on the wider path rejects more often. Telemetry
  invariant still holds; numerical correctness still within 1e-10
  vs fresh-factorise. Documented in `MULTI_BIT_RESULTS.md`.
- **`PulsimSparseLuSolverT<Real>`** is bit-identical to v1.3.0's
  `PulsimSparseLuSolver` modulo the rename — every Layer 1-9
  consumer that uses the unparameterised name keeps compiling
  unchanged.

### Regression test summary

- **498 / 498 C++ tests pass** (up from 478 in v1.3.0; +20 new
  tests: 5 complex-solver unit + 2 mna_sweep integration +
  7 multi-bit spec scenarios + 6 parametric refactor cases).
- **6 / 6 Python tests pass** (`test_sweep_path_aware.py` —
  KPI parity vs legacy, two-param sweep, unknown-name fallback,
  MC, result shape).
- **`pulsim 1.4.0`** Python wheel builds + imports clean.
  `cache.refactor_parametric(b.branch_id_of("R_load"), 3.0)`
  smoke-tested end-to-end.
- Existing rank-1 microbench (single-bit Gray-code, all N ∈ {4..24})
  shows the same 2.7-2.9× speedup as v1.3.0 — the complex solver
  templatisation, multi-bit routing, and parametric refactor all
  change dispatch logic without regressing the v1.3.0 hot path.

## [1.3.0] — 2026-05-24

### Highlights — In-house sparse LU + path-based partial refactorization

This release replaces the V8 KLU-backed `partial_refactor` with a
**fully in-house C++23 sparse LU stack** (`pulsim::sparse::PulsimSparseLuSolver`),
implementing the path-based partial refactorisation algorithm
(Chan/Brandwajn/Tinney, *IEEE Trans. Power Syst.* 1, 1986;
Dinkelbach et al., *Energies* 14:7989, 2021, §3) from scratch on
top of Eigen sparse-matrix containers. **Zero third-party LU
dependency** — neither SuiteSparse KLU (V8) nor the dpsim-simulator
fork (the rejected V8.1-vendoring approach).

Per the project owner's 2026-05-24 architectural decision
(documented in
[`openspec/changes/replace-klu-with-pulsim-sparse-lu/`](openspec/changes/replace-klu-with-pulsim-sparse-lu/)):
the algorithmic novelty of the planned IEEE TPEL methods paper
must be ours, not a thin wrapper around someone else's C patch.

### Performance

3-backend microbench captured 2026-05-24 on macOS Apple Silicon
(see [`artigos/02_tpel_methods/benchmarks/RANK1_RESULTS.md`](artigos/02_tpel_methods/benchmarks/RANK1_RESULTS.md)):

| n_state | baseline solve | Pulsim path-based | speedup |
|--------:|---------------:|------------------:|--------:|
| 6       | 6.7 µs         | 2.3 µs            | 2.93×   |
| 14      | 10.0 µs        | 3.6 µs            | **2.81×** |
| 18      | 12.2 µs        | 4.3 µs            | **2.82×** |
| 26      | 16.4 µs        | 6.1 µs            | **2.68×** |

**Zero fallbacks across all 1999 single-bit Gray-code flips per N**
— every transition exercised the path-based fast path successfully.
The per-call cost stays nearly flat (3.6 → 6.1 µs from n_state=14
to n_state=26) while the baseline scales linearly — the textbook
signature of O(path) per call vs O(nnz·log n) for fresh factorize.

### Added

- **`pulsim::sparse::PulsimSparseLuSolver`** — in-house sparse LU
  in pure C++23, ~900 lines header-only. Implements the full
  `DirectSolver` lifecycle:
  - `analyze()` — Reverse Cuthill-McKee column ordering (George 1971),
    elimination tree (Davis 2006 §4.10 / Liu 1986), symbolic L+U
    pattern
  - `factorize()` — Gilbert-Peierls left-looking with partial
    pivoting (Gilbert & Peierls, *SIAM J. Sci. Stat. Comput.* 9,
    1988). Handles the asymmetric MNA + zero-diagonal patterns
    characteristic of voltage-source constraint rows.
  - `solve()` — forward + back substitution with `Prow`/`Pcol`
    permutations
  - `partial_refactor()` — **path-based** re-elimination over the
    etree, with lazy union of varying columns + pivot-threshold
    fault detection. ~2.7-2.9× speedup vs baseline at the
    n_state ≥ 14 regime.
- **`pulsim::sparse::Backend::Pulsim`** — new enum value (replaces
  `Backend::KLU` from v1.2.0). Default for `Backend::Auto`.
- **CSV bench `rank1_microbench.csv`** — 3-backend, 8-row capture
  for direct citation in the TPEL §VI table.

### Removed (BREAKING at the C++ kernel-builder level)

- **`pulsim::sparse::KluSolver`** — replaced by PulsimSparseLuSolver
- **`pulsim::sparse::Backend::KLU`** — replaced by `Backend::Pulsim`
- **`find_package(KLU)` block in `CMakeLists.txt`** — KLU is no
  longer a dependency at all
- **`PULSIM_HAVE_KLU` + `PULSIM_ENABLE_KLU` compile defs / build
  options** — no longer applicable
- **`libsuitesparse-dev` from CI** — no longer needed; `apt install
  libsuitesparse-dev` removed from all Linux CI matrix entries,
  `brew install suite-sparse` removed from macOS

**Migration:** any out-of-tree caller that constructed `KluSolver`
directly or passed `Backend::KLU` to `make_default_solver(n, hint)`
must switch to `PulsimSparseLuSolver` / `Backend::Pulsim`. The
standard `make_default_solver()` / `make_default_solver(n,
Backend::Auto)` entry points continue to work transparently — the
factory returns PulsimSparseLuSolver by default.

### Not changed

- **Public Python API** — `pp.simulate(builder, ...)` keeps working.
  Wiring `solve_rank1` into Layer 5's `run_transient` + Python
  bindings is out of scope of this release; tracked as
  `add-pwl-rank1-runtime-integration` (TBD).
- **All 8 reference projects under `projects/`** — bit-identical
  output (verified via the 17,279 layer4/4_v1/5/5_v1/5_v4
  assertions across 135 test cases).
- **Build prerequisites** — just **Eigen 3.4+ and a C++23 compiler**
  now; no SuiteSparse install needed.

### Fixed — pre-release cleanup

- **`pulsim.device_loss_summary`** now walks both **inductor** and
  **resistor** branches. Resistor entries report `P_avg` and
  `E_total` in addition to `i_avg`/`i_rms`/`i_peak` — current is
  reconstructed from the node-voltage difference and the stored
  `R_ohms`. Switches and diodes remain deferred (see
  [`KNOWN_LIMITATIONS.md`](KNOWN_LIMITATIONS.md) § *Post-hoc
  analysis*). Previously the summary covered inductors only and
  the module docstring advertised a function that the curated
  `pulsim.*` surface never re-exported.
- **`pulsim.LossAccumulator`, `pulsim.EfficiencyCalculator`,
  `pulsim.device_loss_summary`, `pulsim.average_power_at_node`**
  are now wired into `pulsim.__all__` and importable from the
  top level. The functions existed in `pulsim/losses.py` from the
  start but were not exposed, so callers following the module
  docstring (`p.LossAccumulator()`) hit `AttributeError`.
- **`pulsim.schematic.render`** — removed the `position_hints=`
  keyword. Neither backend (`netlistsvg`, `python_native`) ever
  shipped an implementation; both raised `NotImplementedError` on
  non-empty input. The auto-layout path is unchanged. The
  follow-up renderer is tracked as
  [`add-schematic-renderer-v2`](openspec/changes/add-schematic-renderer-v2/).
- **`KNOWN_LIMITATIONS.md`** — added at the repository root,
  cataloguing every deliberately-deferred item carried into v1.3
  and linking each one back to its OpenSpec proposal or follow-up
  task.

## [1.2.0] — 2026-05-24

### Highlights — PWL rank-1 cache update path (Layer 4 V8)

This release ships the algorithmic contribution that backs the
planned IEEE TPEL methods paper on Pulsim's PWL state-space cache
(see [`artigos/02_tpel_methods/`](artigos/02_tpel_methods/)). The
full design + decisions + delta specs live in
[`openspec/changes/add-pwl-rank1-update/`](openspec/changes/add-pwl-rank1-update/)
and the captured benchmark in
[`artigos/02_tpel_methods/benchmarks/RANK1_RESULTS.md`](artigos/02_tpel_methods/benchmarks/RANK1_RESULTS.md).

**No BREAKING changes.** Every existing caller produces bit-identical
output. The rank-1 path is purely additive performance. 17,839 layer5
assertions across 89 test cases pass unchanged.

### Added

- **`pulsim::sparse::KluSolver`** — `DirectSolver` implementation
  wrapping SuiteSparse KLU (Davis & Natarajan, *ACM TOMS* 37(3), 2010,
  Algorithm 907). Purpose-built for circuit MNA matrices. Header-only,
  gated on `PULSIM_HAVE_KLU` (set by the root `CMakeLists.txt`'s
  `find_package(KLU CONFIG)` block). When KLU is absent the kernel
  builds and runs identically using `Eigen::SparseLU`.
- **`pulsim::sparse::Backend` enum + factory hint** — new overload
  `make_default_solver(Size n, Backend hint = Backend::Auto)` lets the
  caller request `Backend::KLU`, `Backend::Eigen`, or auto-pick by
  matrix size. Default `Backend::Auto` picks KLU when n ≥ 100
  (`PULSIM_KLU_AUTO_THRESHOLD`, tuneable at build).
- **`DirectSolver::supports_partial_refactor()`** + **`partial_refactor(M, changed_cols)`** —
  new virtual methods on the base interface, with default impls
  returning `false` so existing solvers transparently fall back.
- **`PwlStateSpaceCache::solve_rank1(mask, b_extra, x)`** — sliding-
  solver fast path. On single-bit Gray-code mask flips it calls
  `partial_refactor` instead of rebuilding the cache segment; falls
  back transparently to full re-factor on multi-bit flips, unsupported
  backends, or numerical singularities.
- **`PwlStateSpaceCache::set_rank1_backend(Backend)`** — pre-`solve_rank1`
  override for the rank-1 sliding solver's backend (useful for
  benchmarks that want to exercise KLU even at small n).
- **`PwlStateSpaceCache::metrics()`** + **`pulsim::pwl::CacheMetrics`** —
  `{rank1_hits, full_refactor_hits, fallbacks}` atomic monotonic
  counters for benchmark attribution. Thread-safe sampling via
  `std::memory_order_relaxed`.
- **Microbenchmark `core/tests/benchmarks/test_bench_pwl_rank1.cpp`** —
  Catch2 binary in the opt-in `pulsim_benchmarks` target. Sweeps
  N ∈ {4, 6, 8, 10, 12} switches, times `solve` vs `solve_rank1`,
  writes CSV to `${PULSIM_BENCH_RESULTS_DIR}/rank1_microbench.csv`.
- **CI matrix** updated to install `libsuitesparse-dev` on Linux +
  `suite-sparse` via brew on macOS across every existing entry
  (Clang 17/18, GCC 13, Debug sanitizers, coverage).
- **README "Build prerequisites"** section documenting the new
  optional dependency with install commands for macOS / Debian /
  Fedora and the `-DPULSIM_ENABLE_KLU=OFF` opt-out.

### Performance

Captured microbench on macOS 26.5 / Apple Silicon / AppleClang 17:

| N | n_state | µs/solve | µs/rank1 | speedup |
|--:|--:|--:|--:|:--:|
| 4  | 6  | 4.67 | 10.29 | 0.45× (overhead dominates at tiny n) |
| 6  | 8  | 2.57 | 2.79  | 0.92× (break-even) |
| 8  | 10 | 2.57 | 2.90  | 0.89× (break-even) |
| 10 | 12 | 4.60 | 2.73  | 1.69× (rank-1 wins) |
| 12 | 14 | 9.69 | 3.08  | **3.15×** (headline finding) |

Per-call rank-1 cost stays ~3 µs across the sweep while per-call
`solve` cost grows linearly with n — the textbook signature of
amortising the symbolic factorisation across all calls. The
V0 MVP delegates to `klu_refactor`; the V8.1 follow-up will replace
that with path-based partial re-elimination per Chen et al.,
IEEE TPEL 2024 §III, extending the speedup to 5-10× at n=200.

### Not changed

- **Public Python API** — `pp.simulate(builder, …)` continues to use
  the existing per-mask cache path via `cache.solve(mask)`. Wiring
  `solve_rank1` into Layer 5's `run_transient` + Python bindings is
  out of scope of this release; tracked as
  `add-pwl-rank1-runtime-integration` (TBD).
- **All 8 reference projects under `projects/`** — bit-identical
  output (verified via the layer5 / layer5_v1..v4 / showcase regression
  test suite, 17,839 assertions across 89 test cases).

## [1.1.0] — 2026-05-23

### Highlights — JOSS submission release

This release marks the first version of Pulsim accompanied by a
peer-reviewed publication. The accompanying paper has been submitted
to the [Journal of Open Source Software (JOSS)](https://joss.theoj.org/);
the source lives in [`artigos/01_joss_tool_paper/`](artigos/01_joss_tool_paper/).
Once the JOSS paper is accepted, this version's DOI will be the
canonical software citation.

### Added

- **`LICENSE`** at repo root — MIT text. The licence was previously
  only declared in `pyproject.toml`; JOSS (and most academic
  citation tools) require the licence file at the root.
- **`CITATION.cff`** at repo root — Citation File Format v1.2.0
  metadata for automatic citation generation by GitHub and tools
  like `cffconvert`.
- **`artigos/` directory** — paper sources for the Pulsim publication
  campaign, with `README.md` documenting the 4-paper strategic plan
  (JOSS tool paper → EPE-ECCE Europe 2026 conference →
  IEEE Open Journal of Power Electronics methods paper →
  IEEE TPEL / JESTPE application paper).

### Fixed

- **README quick-start example** — `p.scope(...)` updated to
  `p.plot.scope(...)` to match the actual location of the plot
  helper in the current Pulsim 1.x API. Verified end-to-end
  against the installed package.

## [0.10.0] — 2026-05-19

### Highlights

The 0.10.0 release closes the alpha cycle that started with `0.10.0a1`
and adds a **switched-mode closed-loop control surface** that brings
Pulsim into PSIM/Simulink territory for power-electronics controller
design and verification.

### Added — Switched-Mode Closed-Loop

- **`Simulator.run_transient(x0, circuit, callback)`** — new binding
  overload that accepts a Python callback invoked after every accepted
  timestep. The callback can call back into the circuit
  (`circuit.set_pwm_duty(name, new_duty)`, `circuit.set_pmsm_foc_references(...)`,
  …) to close the loop. Single transient run, full state preservation,
  Python in control — same architectural pattern as PSIM / Simulink.
- **GIL-safe streaming binding** — `run_transient_streaming` now
  releases the GIL around the C++ integration loop, lets callbacks
  re-enter pybind11 safely, and survives `None` callbacks. The
  `py::call_guard<py::gil_scoped_release>` race that crashed on every
  invocation is fixed.
- **`RuntimeCircuit::has_any_dynamic_history()`** — kernel helper that
  lets `Simulator::run_transient_native_impl` discriminate fresh-circuit
  vs. continuation calls. Continuations now preserve cap `i_prev` and
  inductor `v_prev` on the same Circuit instance (the per-period
  closed-loop pattern no longer collapses the dynamic state).
- Periodic shooting `run_periodic_shooting` retains "fresh-state-per-
  shooting-iteration" semantics — explicit `update_history(guess, true)`
  reset before each `run_transient(guess)` call.

### Added — Teaching Notebooks

- `examples/notebooks/vsi_inverter_design.ipynb` — end-to-end design
  of a 3φ Voltage Source Inverter (SPWM, 16 kHz, 6 SiC MOSFETs).
- `examples/notebooks/boost_pfc_vsi_design.ipynb` — full AC → DC → 3φ AC
  cascade (220 V_rms in, 400 V DC bus, 230 V_rms 3φ out).
- `examples/notebooks/boost_pfc_closed_loop.ipynb` — switched-mode
  closed-loop PFC using `Simulator.run_transient(x0, ckt, callback)`.
  V_dc converges (architecture proof-of-concept; PI tuning is iterative
  follow-up work — cascaded ACMC is the next milestone).

### Fixed

- `run_transient(x0)` no longer ping-pongs voltage-source nodes between
  `0` and `2·V_src` when `x0 = zeros` (consistent initialization fix
  in `a2cb883`).
- `run_transient_streaming` no longer aborts the process with
  `pybind11::handle::inc_ref` GIL assertions when any callback is
  passed (including `None`).
- Per-period closed-loop boost: cap state is preserved across
  Simulator constructions sharing the same Circuit, removing the
  divergence-to-0V symptom on continuation runs.
- 95 `ruff` errors across `python/` brought to zero — E702 multi-stmt
  semicolons split onto separate lines, F401 unused imports added to
  `__all__` or removed, E402 imports-after-importorskip ignored at the
  per-file level for property tests.
- `mkdocs build --strict` is green again — removed dangling refs to
  retired loss-params classes (`MOSFETLossParams`, `IGBTLossParams`,
  `DiodeLossParams`, `ConductionLoss`, `SwitchingLoss`), switched
  cross-tree file links to absolute GitHub URLs, added `: Any`
  annotations on `circuit` params that griffe was flagging.
- Stress benchmark suite no longer aborts on `periodic_rc_pwm` —
  added the missing entry to `benchmarks/benchmarks.yaml` (with no
  SPICE netlist, since the periodic-analysis bench has no parity
  baseline).
- `test_fmu_*` skip cleanly on Windows (ctypes.CDLL holds the DLL
  handle across `TemporaryDirectory` cleanup → PermissionError).
- `test_bode_plot_rejects_failed_result` skips when matplotlib is
  not installed (Windows CI).
- `test_shooting_uses_warm_start_retry_for_pwm_case` marked `xfail`
  pending shooting-solver re-tune for dead-time PWM (regression
  pre-dates this release; tracked separately).

### Notebooks — also revalidated

- `boost_converter_design.ipynb` runs end-to-end on the new kernel
- `flyback_converter_design.ipynb` runs end-to-end on the new kernel
- `vsi_inverter_design.ipynb` / `boost_pfc_vsi_design.ipynb` —
  `np.trapz` → `np.trapezoid` compat for NumPy 2.x

### Removed

- (No public API removals in this release. The loss-params classes
  documented in earlier alpha series were already replaced by
  device-side params during the alpha cycle.)

### Migration

- The new closed-loop pattern is **opt-in via a new binding overload**.
  Existing single-shot transient calls (`Simulator.run_transient()` /
  `Simulator.run_transient(x0)`) behave exactly as before.
- Per-period closed-loop users that reused the same Circuit across
  Simulator constructions now get **correct state preservation** by
  default. If your code depended on the old "reset on every call"
  behaviour, call `circuit.update_history(x, True)` explicitly before
  each `run_transient` to force the reset.

### Internal

- Kernel test suite: **304 cases / 4214 assertions** green.
- Python lint: **`ruff check python/`** zero errors.
- Docs build: **`mkdocs build --strict`** green.

### Notable commits

- `fc3c686` — kernel: preserve dynamic-device history + streaming GIL fix
- `ed879af` — bindings: `Simulator.run_transient(x0, ckt, callback)`
- `9062c78` — notebook: closed-loop PFC switched-mode proof-of-architecture
- `cef7981` — notebook: AC → DC → 3φ AC cascade design
- `663e3be` — notebook: 3φ VSI design walkthrough
- `9806df5` — chore: zero ruff errors
- `c5d7699` — fix: docs strict + benchmark index
- `1b9d01d` — fix: restore periodic shooting + Windows test gates

---

## Earlier Releases

See [GitHub Releases](https://github.com/lgili/Pulsim/releases) for
0.9.0 and earlier.
