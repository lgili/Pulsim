# Tasks — add-adaptive-runge-kutta-solvers

## 1. Integrator scaffolding

- [ ] 1.1 Define `IntegratorStep` C++ concept in `core/include/pulsim/integrators/integrator.hpp` capturing `step(...)`, `interpolant(...)`, `order()`, `is_stiff()`.
- [ ] 1.2 Refactor existing Tustin + BDF1 to implement the concept (compile-time polymorphism, no virtual cost).
- [ ] 1.3 Add `Integrator` enum (Tustin / Bdf1 / DormandPrince54 / RadauIIA3) and integrator-selector dispatch in `run_transient`.
- [ ] 1.4 Expose enum on `SimulationOptions::Advanced::Timestep`.

## 2. Dormand-Prince 5(4)

- [ ] 2.1 Implement `DormandPrince54.hpp` — 7-stage FSAL explicit RK with Butcher tableau hard-coded.
- [ ] 2.2 4th-order embedded estimate → `err_norm` for the step controller.
- [ ] 2.3 4th-order Hermite-type dense-output formula using existing stage values (no extra evaluations).
- [ ] 2.4 Unit tests: Lorenz manufactured solution vs scipy oracle within 1e-5, order-of-convergence verification.

## 3. Radau IIA(3)

- [ ] 3.1 Implement `RadauIIA3.hpp` — 2-stage implicit RK, Butcher tableau hard-coded, L-stable.
- [ ] 3.2 Newton inner solver using existing sparse Jacobian + Kelley line-search.
- [ ] 3.3 3rd-order Radau interpolant for dense output.
- [ ] 3.4 Step-controller feedback: Newton iteration count + LTE both feed `dt` adjustment.
- [ ] 3.5 Unit tests: stiff RC discharge — converges in ~10 steps vs ~10 k for fixed-step Tustin at matched tol.

## 4. Step controller wiring

- [ ] 4.1 Have DOPRI5 and Radau emit `err_norm` consumed by the existing LTE-based controller (already specced in `transient-timestep`).
- [ ] 4.2 Reuse the existing event detector — pass dense-output interpolant for sub-step localisation.
- [ ] 4.3 Event-time bisection precision regression test: half-wave rectifier zero-crossing within 1 ns.

## 5. Python API

- [ ] 5.1 Bind `Integrator` enum in pybind.
- [ ] 5.2 `simulate(integrator="dopri5"|"radau"|"tustin"|"bdf1", step_mode="variable"|"fixed")` shorthand in `python/pulsim/__init__.py`.
- [ ] 5.3 pytest covering Python-shorthand → C++ enum mapping and a smoke `simulate(integrator="dopri5", step_mode="variable")` run.

## 6. Benchmarks + examples

- [ ] 6.1 `examples/scripts/run_psfb_adaptive_dopri5.py` — variable-step DOPRI5 on the existing PSFB benchmark; print accepted/rejected counts + speedup vs Tustin baseline.
- [ ] 6.2 `examples/scripts/run_stiff_thermal_radau.py` — thermal-electrical coupled circuit benefiting from Radau L-stability.
- [ ] 6.3 Add an entry to `benchmarks/regression/` comparing wall-clock between integrators at matched accuracy.

## 7. Docs

- [ ] 7.1 New page `docs/v2/solvers.md` covering all four integrators with order, stability, dense output, when-to-use.
- [ ] 7.2 Cross-link from the mental-model / getting-started pages.
- [ ] 7.3 Document the `rtol` / `atol` / `dt_min` / `dt_max` tuning recipe.

## 8. Validation + release

- [ ] 8.1 `openspec validate add-adaptive-runge-kutta-solvers --strict` clean.
- [ ] 8.2 Full CI green on Linux + macOS + Windows.
- [ ] 8.3 Ship in 1.5.0 with the rest of Phase 2.
- [ ] 8.4 Archive: `openspec archive add-adaptive-runge-kutta-solvers --yes`.
