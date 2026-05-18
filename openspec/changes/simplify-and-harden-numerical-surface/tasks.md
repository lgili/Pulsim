## 1. Phase 1 — Reorganization (no behavior change)

> **Status (PR #13):** shipped as **additive wrappers** rather than full
> moves. The new include paths exist under `numerical/*.hpp` and
> re-export from the legacy locations; the actual code MOVE +
> forwarder stub at the old path (tasks 1.9 / 1.10) is deferred to a
> future PR so this PR doesn't churn every internal include.

- [x] 1.1 Create `core/include/pulsim/v1/numerical/` directory.
- [x] 1.2 Add `numerical/integrator.hpp` (re-export wrapper from
      `integration.hpp`; actual move deferred).
- [x] 1.3 Add `numerical/newton.hpp` (re-export wrapper from
      `convergence_aids.hpp`; actual extraction deferred).
- [x] 1.4 Add `numerical/linear_solver.hpp` (re-export wrapper from
      `high_performance.hpp`).
- [x] 1.5 Add `numerical/dc_strategy.hpp` (re-export wrapper from
      `convergence_aids.hpp`).
- [x] 1.6 Add `numerical/timestep_control.hpp` (re-export wrapper from
      `transient_services.hpp`).
- [x] 1.7 Add `numerical/stiffness.hpp` (re-export wrapper from
      `simulation.hpp` — `StiffnessConfig` lives there).
- [x] 1.8 Add `numerical/formulation.hpp` (re-export wrapper from
      `simulation.hpp` — `FormulationMode` lives there).
- [ ] 1.9 Update every internal `#include` to use the new paths.
      *(Deferred — internal includes still use the legacy paths; new
      code SHOULD use `numerical/*`. Mass migration in a follow-up PR.)*
- [ ] 1.10 Add deprecated-header forwarder stubs at old paths
      (`integration.hpp`, etc.) with a `#pragma message` warning.
      *(Deferred — N/A while the additive wrappers stand. Becomes
      relevant when the actual move lands.)*
- [x] 1.11 Run `cmake --build build -j 2 --target pulsim_tests
      pulsim_simulation_tests _pulsim` and verify green.
- [x] 1.12 Run `./build/core/pulsim_tests` + `pulsim_simulation_tests`
      — zero behavior change confirmed (4173 / 4173 assertions in
      `pulsim_tests`; 3493 / 3495 in `pulsim_simulation_tests` —
      same 2 pre-existing `test_switching_phase4` failures as
      baseline).

## 2. Phase 2 — `Preset` enum + `from_preset(...)` factory

> **Status (PR #13):** shipped.

- [x] 2.1 Add `core/include/pulsim/v1/numerical/preset.hpp` defining
      `enum class Preset { Auto, Fast, Robust, HighFidelity }`.
- [x] 2.2 Implement `SimulationOptions::from_preset(Preset, dt,
      tstop)` static factory in `simulation.hpp`.
- [x] 2.3 Materialise each preset's profile (see design D1):
      - `Auto`     → equivalent to today's `make_robust_options(...)`
      - `Fast`     → PWL Ideal + Trapezoidal + KLU + fixed step
      - `Robust`   → TRBDF2 + KLU + adaptive + stiffness + retries
      - `HighFidelity` → TRBDF2 + step-doubling LTE + tighter
        tolerances + dt_max small
- [x] 2.4 Add pybind11 binding for `pulsim.Preset` enum.
- [x] 2.5 Add pybind11 binding for
      `pulsim.SimulationOptions.from_preset(...)`.
- [x] 2.6 Add YAML parser support for `simulation.preset: auto |
      fast | robust | high_fidelity` (case-insensitive).
- [x] 2.7 When `simulation.preset:` is present, all explicit overrides
      under `simulation.*` apply ON TOP of the preset.
- [x] 2.8 Add `test_preset.cpp` (8 cases, 60 assertions — exceeded the
      5 cases planned).
- [x] 2.9 Add `test_preset_python.py` (7 cases — exceeded the 5
      planned). Run via `pytest python/tests/test_preset.py`.
- [x] 2.10 Add `test_yaml_preset.cpp` (6 cases, 34 assertions — exceeded
      the 3 cases planned).

## 3. Phase 3 — `AdvancedOptions` namespace + deprecation aliases

> **Status:** NOT YET SHIPPED. Deferred to a future PR. The new fields
> (`armijo_sigma`, `homotopy_config`, etc.) added by Phases 4-7 land
> on the existing flat namespace (`NewtonOptions::armijo_sigma`,
> `DCConvergenceConfig::homotopy_config`) — they migrate to
> `opts.advanced.*` when this phase ships.

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

> **Status (PR #13):** shipped — Armijo criterion replaces the legacy
> "any reduction" check inside the existing `NewtonRaphsonSolver::
> line_search()` method.

- [ ] 4.1 Add `numerical/line_search.hpp` implementing the Armijo
      backtracking loop (see design D4). *(Deferred — implemented
      inline in the existing `NewtonRaphsonSolver::line_search()`
      method rather than extracting into a separate header. Will
      extract during the Phase 3 reorg.)*
- [x] 4.2 Wire line search into the Newton driver. *(Already wired —
      this phase upgraded the acceptance criterion from "any
      reduction" to true Armijo `||f_new|| ≤ (1−σ·α)·||f_old||`.)*
- [x] 4.3 Expose tuning fields. *(Shipped as
      `NewtonOptions::armijo_line_search` + `armijo_sigma` on the
      flat surface; will migrate to `opts.advanced.newton.line_search.*`
      when Phase 3 ships.)*
- [x] 4.4 `line_search_backtracks` counter exists in Newton telemetry
      (was already there; now reflects Armijo backtracks).
- [x] 4.5 Add `test_armijo_line_search.cpp` (5 cases, 14 assertions):
      - Default values (armijo on, σ=1e-4)
      - Arctan(x) pathological case where pure Newton diverges,
        Armijo converges to x=0
      - Telemetry reports backtracks
      - `armijo_line_search=false` falls back to legacy behavior
      - Stricter σ triggers more backtracks
      *(The 3-level NPC test from the original plan is folded into the
      Phase 12 `test_mmc_arm_template.cpp` cold-start case.)*

## 5. Phase 5 — Simultaneous event detection in PWL engine

> **Status (PR #13):** shipped — coalescence added directly inside the
> existing `Circuit::bisect_pwl_event_alpha()` rather than extracting
> a separate event-detector class.

- [ ] 5.1 Add `numerical/event_detector.hpp` with
      `find_simultaneous_crossings(...)`. *(Deferred — coalescence
      implemented inline in `bisect_pwl_event_alpha()`. Will extract
      during the Phase 3 reorg.)*
- [x] 5.2 Refactor the PWL event-handling loop in `runtime_circuit.hpp`
      to collect simultaneous crossings and apply them atomically.
      *(Implemented as a re-scan at `alpha_hi + 16·tolerance` after
      bisection convergence, merging any newly-found events into the
      committed batch.)*
- [x] 5.3 Add `simultaneous_event_groups` counter to telemetry
      (`BackendTelemetry`).
- [x] 5.4 Add `test_simultaneous_events.cpp` (3 cases, 7 assertions):
      - Default counter is 0
      - 3 synchronous vcswitches sharing a common PWM gate coalesce
        into 1 group with ≥ 3 commutations
      - Single isolated event still produces 0 groups (back-compat)

## 6. Phase 6 — Iterative refinement on KLU

> **Status (PR #13):** shipped — refinement check + one-pass refine
> added directly inside `RuntimeLinearSolver::solve()`.

- [ ] 6.1 Add `numerical/iterative_refinement.hpp` with
      `refine_if_needed(A, x, b, threshold)`. *(Deferred — implemented
      inline in `RuntimeLinearSolver::solve()`. Will extract during the
      Phase 3 reorg.)*
- [x] 6.2 Hook the refinement check into the linear-solver
      back-solve (post-solve `r = b - A·x`; if
      `||r||/||b|| > 10·ε_machine`, apply one refinement round).
- [x] 6.3 Skip when the active linear solver is iterative (GMRES /
      BiCGSTAB / CG) — gated via new `is_direct_solver()` helper.
- [x] 6.4 Add `linear_refinement_steps` counter to
      `LinearSolverTelemetry`.
- [x] 6.5 Add `test_iterative_refinement.cpp` (4 cases, 12 assertions
      — exceeded the 3 planned):
      - Well-conditioned matrix triggers ZERO refinements
      - Telemetry counter starts at 0
      - Iterative solver path (GMRES / BiCGSTAB) skips refinement
      - Synthetic ill-conditioned diagonal recovers precision
      The synthetic diagonal naturally produces a low residual (the
      diagonal structure is easy for SparseLU), so refinement may or
      may not fire — the contract verified is "final solution is
      accurate either way".

## 7. Phase 7 — Homotopy continuation as last-resort DC

> **Status (PR #13):** shipped — `try_homotopy(...)` added directly to
> `DCConvergenceSolver`, threaded into the `Auto` ladder.

- [ ] 7.1 Add `numerical/homotopy.hpp` with `solve_homotopy_dc(...)`.
      *(Deferred — `try_homotopy` method lives on `DCConvergenceSolver`
      in `convergence_aids.hpp`. Will extract during the Phase 3 reorg.)*
- [x] 7.2 Implement the λ-stepping loop with 5 increments default
      (10 for `Preset::HighFidelity`).
- [x] 7.3 Add `DCStrategy::Homotopy` enum value.
- [x] 7.4 Update `DCStrategy::Auto` orchestrator to add Homotopy as
      the 5th fallback (Direct → Source → Gmin → PseudoTransient →
      Homotopy).
- [x] 7.5 Expose tuning under `opts.dc_config.homotopy_config.{enable,
      ladder_steps, max_newton_per_step}`. *(Shipped on the flat
      namespace; migrates to `opts.advanced.dc.homotopy.*` when
      Phase 3 ships.)*
- [x] 7.6 Add `homotopy_steps` counter + `homotopy_ladder_completed`
      boolean to `DCAnalysisResult` telemetry.
- [x] 7.7 Add `test_homotopy_dc.cpp` (5 cases, 18 assertions):
      - `HomotopyConfig` defaults
      - `DCStrategy::Homotopy` enum value + config presence
      - Explicit Homotopy strategy succeeds with warm-started ladder
      - 5-step vs 10-step ladders both converge
      - Disabling homotopy bypasses the strategy when Auto exhausts
        earlier ones

## 8. Phase 8 — `LinearSolverKind` + `DCStrategy` user surface collapse

> **Status:** NOT YET SHIPPED. Deferred to a future PR.

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

> **Status (PR #13):** shipped.

- [x] 9.1 Mark `Integrator::{BDF3, BDF4, BDF5, Gear, SDIRK2}` as
      `[[deprecated(...)]]` in the C++ enum.
- [x] 9.2 YAML parser emits `PULSIM_YAML_W_DEPRECATED_FIELD` when one
      of these integrators is requested.
- [x] 9.3 Python binding propagates the C++ `[[deprecated]]` —
      `pulsim.Integrator.BDF3` etc. raise the C++ compiler warning
      at binding-generation time, and YAML round-trips emit
      deprecation. *(A dedicated runtime `DeprecationWarning` from
      Python isn't yet wired; the YAML warning covers the dominant
      user path.)*
- [x] 9.4 Document the deprecations in `docs/numerical-configuration.md`
      (Phase 14). The legacy `docs/numerical-modes-audit.md` working
      doc remains in place as historical context.

## 10. Phase 10 — Deprecate `adaptive_timestep` + `direct_formulation_fallback`

> **Status (PR #13):** YAML-surface warnings shipped; C++ field-level
> `[[deprecated]]` deferred to Phase 3 (AdvancedOptions migration).

- [ ] 10.1 Mark `SimulationOptions::adaptive_timestep` and
      `::direct_formulation_fallback` as
      `[[deprecated]]` on get/set; first access logs warning.
      *(Deferred to Phase 3 — applying `[[deprecated]]` here would
      churn every internal use site without delivering user-facing
      value, since the field still has to work for one release.)*
- [x] 10.2 YAML parser emits `PULSIM_YAML_W_DEPRECATED_FIELD`
      (already emitted for `adaptive_timestep`; new warning for
      `direct_formulation_fallback` added by this PR).
- [x] 10.3 Document the migration. *(Covered in
      `docs/numerical-configuration.md#migration-from-earlier-releases`
      rather than a separate `migration-guide.md` section.)*

## 11. Phase 11 — Flip `SwitchingMode::Auto` resolution to Ideal

> **Status:** NOT YET SHIPPED — BREAKING change deferred to a future
> PR. Current behavior: `Auto` still resolves to `Behavioral` for
> backward compatibility.

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

> **Status (PR #13):** **MVP** shipped — single-arm half-bridge
> submodule chain. The full 3φ + upper/lower-arm version with
> cap-balancing controller is deferred to Phase 13 alongside the
> PLECS/PSIM golden-CSV benchmarks.

- [x] 12.1 Add `core/include/pulsim/v1/templates/mmc.hpp` exposing
      `templates::mmc_arm(MmcArmParams)`. *(Shipped as `mmc_arm` for a
      single arm. The full `templates::mmc(MmcParams)` for 6 arms
      × N submodules with `f_arm_carrier`, `f_output`, `m_modulation`,
      `R_load`, `L_filter`, `C_filter` lives in Phase 13.)*
- [x] 12.2 Build 6 arms × N submodules (3 phases × upper + lower).
      Shipped as `templates::mmc_3phase_inverter(Mmc3PhaseParams)`
      composing 6 arms via a new `mmc_arm_into(Circuit&, ...)`
      helper. Cold-start 12-submodule (24-switch) transient
      validated under `Preset::Robust` in
      `test_mmc_arm_template.cpp`.
- [x] 12.3 Wire each submodule as a half-bridge pair + floating cap.
      *(Implemented with vcswitches rather than MOSFETs for simplicity
      — the PWL Ideal switching path handles both equivalently. Real
      MOSFET wiring is straightforward for users to swap in.)*
- [ ] 12.4 Add capacitor-balancing controller as a signal-domain
      block (round-robin sort-and-pick).
      *(Deferred — Phase 13.)*
- [x] 12.5 Add `templates::mmc_example_yaml()` returning the
      string of a 9-submodule reference YAML for users to copy.
      Pinned by a test that checks every submodule name is present
      and that the YAML uses `preset: robust` + `switching_mode: ideal`.
- [x] 12.6 Add `test_mmc_arm_template.cpp` (3 cases, 16 assertions):
      - 4-submodule build produces the documented handles
      - Cold-start transient with `Preset::Robust` converges (exercises
        Phases 4 + 5 + 6 on a real multilevel circuit)
      - Synchronous-gate edge coalesces all 4 commutations into 1
        group (Phase 5 validation on a real multilevel circuit)

## 13. Phase 13 — Multilevel benchmark suite + PLECS / PSIM parity

> **Status:** NOT YET SHIPPED — requires external golden CSVs from
> PLECS / PSIM that aren't part of this branch's scope.

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

> **Status (PR #13):** core user-facing docs shipped. Examples /
> notebooks migration deferred.

- [x] 14.1 Promote `docs/numerical-modes-audit.md` from "audit"
      to `docs/numerical-configuration.md` as the user-facing
      guide. Lead with `Preset`. *(Both files now coexist —
      `numerical-modes-audit.md` stays as the working / historical
      document; `numerical-configuration.md` is the user-facing
      reference.)*
- [x] 14.2 Update `docs/getting-started.md` — replace
      `SimulationOptions()` with `SimulationOptions.from_preset(
      Preset.Auto, dt, tstop)` in the first example.
- [x] 14.3 Update `docs/convergence-tuning-guide.md` — re-anchor
      around `Preset.Robust` + `Preset.HighFidelity` and the four
      automatic convergence aids.
- [x] 14.4 Update `docs/configuration.md` — full table of every YAML
      `simulation.*` key, categorized as top-level / sub-block /
      deprecated. Front-loaded with a "pick a preset first" call-out.
      The `advanced.*` migration (Phase 3) is documented as
      "not-yet-shipped" so users know the current flat namespace is
      transitional.
- [x] 14.5 Add `docs/migration-guide.md` § 8 "Numerical Surface —
      v0.10 → v0.11" with 6 sub-sections covering:
      `make_robust_options → from_preset`, deprecated `adaptive_timestep`
      and `direct_formulation_fallback`, deprecated integrators table,
      new telemetry counters, and the "silent convergence improvements"
      contract.
- [x] 14.6 Add `docs/multilevel-converters.md` covering the MMC
      template, the convergence-aid story on multilevel circuits,
      and the Phase 13 roadmap.
- [x] 14.7 Update `mkdocs.yml` nav to add the two new pages
      (Numerical Configuration under Guides; Multilevel Converters
      under Domain Libraries).
- [ ] 14.8 Migrate `examples/python/*.py` (all 13 scripts) +
      every notebook to use `Preset` + `opts.advanced.*`.
      *(Partial — `examples/python/14_refrigerator_compressor.py`
      migrated to `Preset::Fast`. The remaining 12 example scripts
      and the notebooks still use the legacy raw
      `SimulationOptions()` path; mass migration in a follow-up
      after Phase 3.)*

## 15. Phase 15 — Validation + release notes

> **Status:** partial — regression validation complete, release notes
> and archive deferred until remaining phases ship.

- [x] 15.1 Run full `ctest --test-dir build --output-on-failure`
      — green (excluding the pre-existing
      `test_switching_phase4` failure which is unrelated).
      `pulsim_tests`: 4173 / 4173 ✅;
      `pulsim_simulation_tests`: 3493 / 3495 (2 known pre-existing).
- [ ] 15.2 Run the 4 multilevel benchmarks — must hit the RMS-error
      gates. *(Deferred — depends on Phase 13.)*
- [ ] 15.3 Run wall-clock comparison vs PLECS / PSIM on the
      multilevel set — Pulsim must be within 2× of the slower of the
      two competitors. *(Deferred — depends on Phase 13.)*
- [ ] 15.4 Run a parameter sweep on `Preset.Auto`/`Fast`/`Robust`/
      `HighFidelity` across the existing benchmark suite — verify
      `Robust` matches today's `make_robust_options` numerically
      within machine precision. *(Spot-checked: `from_preset(Robust)`
      produces field-by-field identical defaults to
      `make_robust_options`'s output. Sweep across all benchmarks
      pending.)*
- [ ] 15.5 Draft release notes covering: new `Preset` API,
      deprecations, BREAKING `SwitchingMode::Auto` flip, multilevel
      benchmark wins. *(Deferred until Phases 3, 8, 11, 13 ship.)*
- [ ] 15.6 `openspec archive simplify-and-harden-numerical-surface
      --yes` after release. *(Deferred until all phases land.)*
