## 1. Phase 1 — Reorganization (no behavior change)

- [ ] 1.1 Create `core/include/pulsim/v1/numerical/` directory.
- [ ] 1.2 Add `numerical/integrator.hpp` re-exporting from
      `integration.hpp` (move the enum + helpers; leave forwarder
      stub at `integration.hpp` with `#pragma message`).
- [ ] 1.3 Add `numerical/newton.hpp` extracting `NewtonOptions` +
      Newton driver from `convergence_aids.hpp`.
- [ ] 1.4 Add `numerical/linear_solver.hpp` extracting
      `LinearSolverKind` + `LinearSolverStackConfig` from
      `high_performance.hpp`.
- [ ] 1.5 Add `numerical/dc_strategy.hpp` extracting `DCStrategy` +
      `DCConvergenceConfig` from `convergence_aids.hpp`.
- [ ] 1.6 Add `numerical/timestep_control.hpp` extracting
      `TransientStepMode` + `AdvancedTimestepConfig` +
      `RichardsonLTEConfig` + `BDFOrderConfig` from
      `transient_services.hpp`.
- [ ] 1.7 Add `numerical/stiffness.hpp` extracting `StiffnessConfig`.
- [ ] 1.8 Add `numerical/formulation.hpp` extracting `FormulationMode`.
- [ ] 1.9 Update every internal `#include` to use the new paths.
- [ ] 1.10 Add deprecated-header forwarder stubs at old paths
      (`integration.hpp`, etc.) with a `#pragma message` warning.
- [ ] 1.11 Run `cmake --build build -j 2 --target pulsim_tests
      pulsim_simulation_tests _pulsim` and verify green.
- [ ] 1.12 Run `./build/core/pulsim_tests` + `pulsim_simulation_tests`
      — expect zero behavior change (same assertion count, same pass
      rate as baseline).

## 2. Phase 2 — `Preset` enum + `from_preset(...)` factory

- [ ] 2.1 Add `core/include/pulsim/v1/numerical/preset.hpp` defining
      `enum class Preset { Auto, Fast, Robust, HighFidelity }`.
- [ ] 2.2 Implement `SimulationOptions::from_preset(Preset, dt,
      tstop)` static factory in `simulation.hpp`.
- [ ] 2.3 Materialise each preset's profile (see design D1):
      - `Auto`     → equivalent to today's `make_robust_options(...)`
      - `Fast`     → PWL Ideal + Trapezoidal + KLU + fixed step
      - `Robust`   → TRBDF2 + KLU + adaptive + stiffness + retries
      - `HighFidelity` → TRBDF2 + step-doubling LTE + tighter
        tolerances + dt_max small
- [ ] 2.4 Add pybind11 binding for `pulsim.Preset` enum.
- [ ] 2.5 Add pybind11 binding for
      `pulsim.SimulationOptions.from_preset(...)`.
- [ ] 2.6 Add YAML parser support for `simulation.preset: auto |
      fast | robust | high_fidelity` (case-insensitive).
- [ ] 2.7 When `simulation.preset:` is present, all explicit overrides
      under `simulation.*` apply ON TOP of the preset.
- [ ] 2.8 Add `test_preset_cpp.cpp` (5 cases): each preset produces a
      valid `SimulationOptions`, materialised fields match design
      table, raw `SimulationOptions{}` still constructs.
- [ ] 2.9 Add `test_preset_python.py` (5 cases): same coverage via
      Python.
- [ ] 2.10 Add `test_preset_yaml.cpp` (3 cases): YAML `preset: robust`
      round-trips; explicit override wins over preset; unknown
      preset string rejected.

## 3. Phase 3 — `AdvancedOptions` namespace + deprecation aliases

- [ ] 3.1 Add `numerical/advanced_options.hpp` aggregating
      `NewtonOptions`, `AdvancedTimestepConfig`, `RichardsonLTEConfig`,
      `BDFOrderConfig`, `DCConvergenceConfig`, `StiffnessConfig`,
      `FallbackPolicy`, `FormulationMode`, `LinearSolverStackConfig`.
- [ ] 3.2 Add `AdvancedOptions advanced{};` field to
      `SimulationOptions`.
- [ ] 3.3 Add deprecated top-level field aliases (forward to
      `advanced.*` via `[[deprecated("use opts.advanced.newton")]]`
      tags on get/set). Emit one warning log per process on first
      access.
- [ ] 3.4 Update Python bindings: `opts.advanced` exposes the
      sub-struct; `opts.newton_options`, `opts.timestep_config`,
      etc. become Python `@property` shims with `DeprecationWarning`.
- [ ] 3.5 Update YAML parser: `simulation.advanced.{newton, timestep,
      dc, stiffness, formulation, linear_solver}.*` is the canonical
      path; old keys still parse but emit
      `PULSIM_YAML_W_DEPRECATED_FIELD`.
- [ ] 3.6 Migrate every example, notebook, and docs snippet to the
      `opts.advanced.*` namespace.
- [ ] 3.7 Add `test_advanced_options.cpp` (3 cases): nested struct
      reachable, deprecated alias still works + warns, YAML round-
      trip preserves both paths.

## 4. Phase 4 — Damped Newton with Armijo line search

- [ ] 4.1 Add `numerical/line_search.hpp` implementing the Armijo
      backtracking loop (see design D4).
- [ ] 4.2 Wire line search into the Newton driver in
      `numerical/newton.hpp`. Trigger on `||r_trial|| ≥ ||r||`.
- [ ] 4.3 Expose tuning under `opts.advanced.newton.line_search.{enable,
      sigma, alpha_min}`. Default: enabled, `σ = 1e-4`, `α_min =
      2^-8`.
- [ ] 4.4 Add `line_search_backtracks` counter to Newton telemetry.
- [ ] 4.5 Add `test_line_search.cpp` (4 cases):
      - Pathological 1D scalar problem where naïve Newton diverges
        but line-search converges
      - Multilevel-style 3-level NPC cold start — converges with
        line search, diverges without (currently fails)
      - Counter reports backtracks
      - `enable=false` reproduces today's behavior

## 5. Phase 5 — Simultaneous event detection in PWL engine

- [ ] 5.1 Add `numerical/event_detector.hpp` with
      `find_simultaneous_crossings(...)` implementing the sort-and-
      group algorithm (design D5).
- [ ] 5.2 Refactor the PWL event-handling loop in `runtime_circuit.hpp`
      (or wherever the per-step PWL crossing scan lives) to:
      - Collect ALL pending crossings into a vector per step.
      - Group by crossing instant within `ε = 1e-12 · dt`.
      - Apply the group atomically, do ONE Newton at the group's
        instant.
- [ ] 5.3 Add `simultaneous_event_groups` counter to telemetry.
- [ ] 5.4 Add `test_simultaneous_events.cpp` (3 cases):
      - 6-switch 3φ inverter — all H/L pairs commutate at the same
        PWM edge, only ONE Newton solve fires per group
      - MMC half-arm with 9 submodules, all submodules turn on at
        identical phase angle — converges (currently hangs)
      - Single-switch event still behaves identically to today

## 6. Phase 6 — Iterative refinement on KLU

- [ ] 6.1 Add `numerical/iterative_refinement.hpp` with
      `refine_if_needed(A, x, b, threshold)`.
- [ ] 6.2 Hook the refinement check into the linear-solver
      back-solve in `numerical/linear_solver.hpp` (after KLU solve,
      compute `r = b - A·x`; if `||r||/||b|| > 10·ε_machine`, do
      one round of `A·δ = r; x ← x + δ`).
- [ ] 6.3 Skip when the active linear solver is iterative (GMRES
      already does this internally).
- [ ] 6.4 Add `linear_refinement_steps` counter to telemetry.
- [ ] 6.5 Add `test_iterative_refinement.cpp` (3 cases):
      - Construct a synthetic 100×100 ill-conditioned MNA (cap-to-
        cap loops) — KLU alone gives 10⁻⁴ residual, refinement
        recovers 10⁻¹²
      - Refinement counter reports correct number of trigger events
      - Well-conditioned RC circuit triggers ZERO refinements

## 7. Phase 7 — Homotopy continuation as last-resort DC

- [ ] 7.1 Add `numerical/homotopy.hpp` with `solve_homotopy_dc(
      circuit, ladder_steps)`.
- [ ] 7.2 Implement the λ-stepping loop (design D7): at λ=0 all
      nonlinear devices replaced by `g_off` linear conductance; at
      λ=1 full nonlinear model. 5 increments default, 10 for
      `Preset.HighFidelity`.
- [ ] 7.3 Add `DCStrategy::Homotopy` enum value.
- [ ] 7.4 Update `DCStrategy::Auto` orchestrator to add Homotopy as
      the 5th fallback (after Direct → Source → Gmin →
      PseudoTransient).
- [ ] 7.5 Expose `opts.advanced.dc.homotopy.{enable, ladder_steps}`.
- [ ] 7.6 Add `homotopy_ladder_completed` boolean + `homotopy_steps`
      counter to telemetry.
- [ ] 7.7 Add `test_homotopy_dc.cpp` (3 cases):
      - 3-level NPC cold start — fails on all four prior strategies,
        converges via Homotopy
      - Simple RC cold start — Direct wins on iteration 1, Homotopy
        never invoked
      - Explicit `strategy_override = Homotopy` skips the ladder

## 8. Phase 8 — `LinearSolverKind` + `DCStrategy` user surface collapse

- [ ] 8.1 In `numerical/linear_solver.hpp`, add a public-facing
      enum `LinearSolverKind { Auto, Direct, Iterative }`. The
      internal 6-value enum becomes `internal::LinearSolverImpl`.
- [ ] 8.2 The auto-selector maps `Auto → Direct (if N < 5000) else
      Iterative`. `Direct → best-of(KLU, EnhancedSparseLU, SparseLU)`.
      `Iterative → best-of(GMRES, BiCGSTAB)`.
- [ ] 8.3 Replace the preconditioner enum with `solver_quality:
      Fast|Default|Best` on `opts.advanced.linear_solver`.
- [ ] 8.4 Same collapse for `DCStrategy` — public enum becomes
      `{Auto, Override}`; internal 5-value enum stays for the
      auto-selector and for `opts.advanced.dc.strategy_override`.
- [ ] 8.5 Update pybind11 + YAML parser accordingly.
- [ ] 8.6 Update every example / notebook / doc.
- [ ] 8.7 Run full test suite — must stay green.

## 9. Phase 9 — Deprecate / remove dead integrators

- [ ] 9.1 Mark `Integrator::{BDF3, BDF4, BDF5, Gear, SDIRK2}` as
      `[[deprecated("removed in v2 — use BDF2 / TRBDF2 /
      RosenbrockW")]]` in the C++ enum.
- [ ] 9.2 YAML parser emits `PULSIM_YAML_W_DEPRECATED_FIELD` when one
      of these integrators is requested.
- [ ] 9.3 Python binding emits `DeprecationWarning`.
- [ ] 9.4 Update `docs/numerical-modes-audit.md` — mark these as
      "deprecated in v1, removed in v2".

## 10. Phase 10 — Deprecate `adaptive_timestep` + `direct_formulation_fallback`

- [ ] 10.1 Mark `SimulationOptions::adaptive_timestep` and
      `::direct_formulation_fallback` as
      `[[deprecated]]` on get/set; first access logs warning.
- [ ] 10.2 YAML parser emits `PULSIM_YAML_W_DEPRECATED_FIELD`.
- [ ] 10.3 Document the migration in
      `docs/migration-guide.md`.

## 11. Phase 11 — Flip `SwitchingMode::Auto` resolution to Ideal

- [ ] 11.1 Change `Circuit::default_switching_mode_` initial value
      from `Auto` (which resolved to Behavioral) to a path that
      resolves `Auto → Ideal`.
- [ ] 11.2 On first transient run, log a one-time INFO
      `"SwitchingMode::Auto resolving to Ideal (was Behavioral prior
      to v0.11). Set SwitchingMode::Behavioral explicitly to preserve
      old behavior."`
- [ ] 11.3 Update `docs/pwl-switching-migration.md` status block to
      "shipped — default flip landed".
- [ ] 11.4 Run the full benchmark suite — gather a diff of
      successes / failures vs prior behavior. Expected: more passes
      (PWL is faster + more accurate); no new failures.
- [ ] 11.5 If any benchmark regresses, document the workaround
      (explicit `Behavioral`) in the benchmark's README.

## 12. Phase 12 — MMC topology template

- [ ] 12.1 Add `core/include/pulsim/v1/templates/mmc.hpp` exposing
      `templates::mmc(MmcParams)` where `MmcParams` covers:
      `num_submodules_per_arm`, `V_dc`, `f_arm_carrier`,
      `f_output`, `m_modulation`, `L_arm`, `R_arm`,
      `C_submodule`, `R_load`, `L_filter`, `C_filter`.
- [ ] 12.2 Build 6 arms × N submodules (3 phases × upper + lower).
- [ ] 12.3 Wire each submodule as a half-bridge MOSFET pair +
      floating cap.
- [ ] 12.4 Add capacitor-balancing controller as a signal-domain
      block (round-robin sort-and-pick).
- [ ] 12.5 Add `templates::mmc_example_yaml()` returning the
      string of a 9-submodule reference YAML for users to copy.
- [ ] 12.6 Add `test_mmc_template.cpp` (3 cases): 4-submodule build
      stamps successfully, full transient runs to completion, cap
      voltages stay balanced within ±5% under nominal load.

## 13. Phase 13 — Multilevel benchmark suite + PLECS / PSIM parity

- [ ] 13.1 Add `benchmarks/multilevel/3level_npc.yaml` + golden CSV
      from PLECS (one-time export, version-tagged).
- [ ] 13.2 Add `benchmarks/multilevel/5level_flying_cap.yaml` +
      golden CSV from PLECS.
- [ ] 13.3 Add `benchmarks/multilevel/ttype_3level.yaml` + golden
      CSV from PLECS.
- [ ] 13.4 Add `benchmarks/multilevel/mmc_9sub.yaml` + golden CSV
      from PSIM (MMC is PSIM's strong suit).
- [ ] 13.5 Add `test_multilevel_npc.cpp` gating on ≤ 0.5% RMS error
      vs PLECS golden across V_phase_A, I_load_A, V_cap_neutral.
- [ ] 13.6 Add `test_multilevel_flying_cap.cpp` same gate.
- [ ] 13.7 Add `test_multilevel_ttype.cpp` same gate.
- [ ] 13.8 Add `test_multilevel_mmc.cpp` gating on ≤ 1% RMS error
      vs PSIM golden (looser because MMC controller details vary).
- [ ] 13.9 Add `tools/multilevel_bench_runner.py` — wall-clock
      comparison vs PLECS / PSIM on the same circuit.
- [ ] 13.10 Document the benchmark gates in
      `docs/benchmarks-and-parity.md`.

## 14. Phase 14 — Docs + examples + migration guide

- [ ] 14.1 Promote `docs/numerical-modes-audit.md` from "audit"
      to `docs/numerical-configuration.md` as the user-facing
      guide. Lead with `Preset`.
- [ ] 14.2 Update `docs/getting-started.md` — replace
      `SimulationOptions()` with `SimulationOptions.from_preset(
      Preset.Auto, dt, tstop)` in the first example.
- [ ] 14.3 Update `docs/convergence-tuning-guide.md` — re-anchor
      around `Preset.Robust` + `Preset.HighFidelity`.
- [ ] 14.4 Update `docs/configuration.md` — table of every YAML
      `simulation.*` key, marked as `top-level` / `advanced` /
      `deprecated`.
- [ ] 14.5 Add `docs/migration-guide.md` section "v0.10 → v0.11
      numerical surface" with per-deprecation migration recipe.
- [ ] 14.6 Add `docs/multilevel-converters.md` covering the 4
      reference topologies, the `templates::mmc(...)` builder, and
      the benchmark gates.
- [ ] 14.7 Update `mkdocs.yml` nav to add the two new pages.
- [ ] 14.8 Migrate `examples/python/*.py` (all 13 scripts) +
      every notebook to use `Preset` + `opts.advanced.*`.

## 15. Phase 15 — Validation + release notes

- [ ] 15.1 Run full `ctest --test-dir build --output-on-failure`
      — must be green (excluding the pre-existing
      `test_switching_phase4` failure which is unrelated).
- [ ] 15.2 Run the 4 multilevel benchmarks — must hit the RMS-error
      gates.
- [ ] 15.3 Run wall-clock comparison vs PLECS / PSIM on the
      multilevel set — Pulsim must be within 2× of the slower of the
      two competitors.
- [ ] 15.4 Run a parameter sweep on `Preset.Auto`/`Fast`/`Robust`/
      `HighFidelity` across the existing benchmark suite — verify
      `Robust` matches today's `make_robust_options` numerically
      within machine precision.
- [ ] 15.5 Draft release notes covering: new `Preset` API,
      deprecations, BREAKING `SwitchingMode::Auto` flip, multilevel
      benchmark wins.
- [ ] 15.6 `openspec archive simplify-and-harden-numerical-surface
      --yes` after release.
