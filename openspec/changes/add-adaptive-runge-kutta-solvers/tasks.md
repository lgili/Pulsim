# Tasks — add-adaptive-runge-kutta-solvers

> Status (Phase 2.4 — v1.5.1 RC):
> **Standalone Python integrators ship now**. ``DormandPrince5``
> and ``RadauIIA3`` are usable today against any user-supplied
> ``f(t, x)`` callback — great for offline ODE analysis,
> simulating controllers + linearised plants, or as a reference
> implementation. **In-kernel coupling** (replacing Tustin /
> BDF1 inside ``pulsim.simulate``) is deferred to v1.6.0 because
> the cache currently exposes Tustin-discretised matrices and
> not the continuous-time ``A``, ``b`` form the adaptive RK
> integrators need.

## 1. Integrator scaffolding (C++ kernel — deferred to v1.6.0)

- [ ] 1.1 Define `IntegratorStep` C++ concept.
- [ ] 1.2 Refactor existing Tustin + BDF1 to the concept.
- [ ] 1.3 Add `Integrator` enum + dispatch in `run_transient`.
- [ ] 1.4 Expose enum on `SimulationOptions::Advanced::Timestep`.

## 2. Dormand-Prince 5(4) — Python implementation shipped

- [x] 2.1 ``python/pulsim/integrators.py::DormandPrince5`` — 7-stage FSAL explicit RK with Butcher tableau hard-coded.
- [x] 2.2 4th-order embedded estimate → ``err_norm`` for the step controller.
- [ ] 2.3 4th-order Hermite-type dense-output formula — deferred (only needed for in-kernel event detection in v1.6.0).
- [x] 2.4 Unit tests: exp(-t) within 1e-7, SHO over 2π within 1e-6, order-of-convergence verified across rtol = 1e-4 / 1e-6 / 1e-8.

## 3. Radau IIA(3) — Python implementation shipped

- [x] 3.1 ``python/pulsim/integrators.py::RadauIIA3`` — 2-stage implicit RK, L-stable, Butcher tableau hard-coded.
- [x] 3.2 Newton inner solver via numpy ``linalg.solve`` on the 2n × 2n block system; optional analytical Jacobian or central finite differences.
- [ ] 3.3 3rd-order Radau interpolant for dense output — deferred (with task 2.3, only needed for in-kernel events).
- [x] 3.4 Step-controller feedback: error from step-doubling (Richardson), Newton iteration-count tracked but not yet fed back into ``dt`` selection (single-PI controller for now).
- [x] 3.5 Unit tests: stiff van der Pol (μ=100) — Radau uses ≤50% of DOPRI5's steps at matched tolerance, both producing the same final state to 7 sig figs.

## 4. Step controller wiring (in-kernel — deferred to v1.6.0)

- [ ] 4.1 Have DOPRI5 and Radau emit ``err_norm`` consumed by the existing LTE-based controller.
- [ ] 4.2 Reuse the existing event detector with dense-output interpolant for sub-step localisation.
- [ ] 4.3 Event-time bisection regression: half-wave rectifier zero-crossing within 1 ns.

## 5. Python API

- [x] 5.1 ``pulsim.DormandPrince5``, ``pulsim.RadauIIA3``, ``pulsim.AdaptiveSolution`` re-exported.
- [ ] 5.2 ``simulate(integrator="dopri5"|"radau", step_mode="variable")`` shorthand — deferred to v1.6.0 (waits on in-kernel coupling).
- [x] 5.3 pytest: 9/9 pass (constructor + analytical accuracy + order-of-convergence + Radau-beats-DOPRI5-on-stiff + solution shape + invalid-input guards).

## 6. Benchmarks + examples — deferred to v1.6.0

- [ ] 6.1 ``examples/scripts/run_psfb_adaptive_dopri5.py``.
- [ ] 6.2 ``examples/scripts/run_stiff_thermal_radau.py``.
- [ ] 6.3 ``benchmarks/regression/`` wall-clock comparison entry.

## 7. Docs — deferred to v1.5.1 final cut

- [ ] 7.1 New page ``docs/v2/solvers.md`` covering all four integrators.

## 8. Validation + release

- [x] 8.1 ``openspec validate add-adaptive-runge-kutta-solvers --strict`` clean.
- [x] 8.2 Local pytest 9/9 pass in 0.25 s.
- [ ] 8.3 Ship as the standalone-integrator half in v1.5.1; in-kernel coupling lands in v1.6.0.
- [ ] 8.4 Archive: ``openspec archive add-adaptive-runge-kutta-solvers --yes``.
