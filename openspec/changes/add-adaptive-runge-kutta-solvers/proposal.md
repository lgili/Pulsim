## Why

Today Pulsim's transient kernel offers two integrators — Tustin (2nd-order trapezoidal) and BDF1 (1st-order implicit Euler) — both **fixed-step**. Variable stepping exists only at the Python wrapper layer (`run_transient_adaptive`), which restarts the kernel in coarse segments and uses `||dx/dt||` as a heuristic instead of a real LTE estimate. PSIM and PLECS expose embedded-pair Runge-Kutta DOPRI5 (Dormand-Prince 4-5) and L-stable implicit Radau IIA(3) as standard adaptive choices.

Two concrete pains today:

- **Soft-switching transients** (PSFB / LLC ringing during ZVS) need ~10 ns dt at the transition and ~10 µs dt at steady-state. Fixed-step Tustin at 10 ns over a 50 ms slot is 5 M steps — most of them wasted. Real adaptive stepping would cut this by 10×.
- **Stiff biological / thermal / magnetics coupling** wants L-stable behaviour at large steps; BDF1 is only 1st-order, so accuracy at large step sizes is poor.

The existing `transient-timestep` spec already declares an LTE controller, Richardson extrapolation, and an event-driven step controller — so the infrastructure for "variable step" exists at the API level. What's missing is the **actual integrators** behind it: a non-stiff embedded RK pair (DOPRI5(4)) for the common case, and an implicit Radau IIA(3) for stiff systems.

## What Changes

- **C++ kernel** — add two new integrators in `core/include/pulsim/integrators/`:
  - `DormandPrince54.hpp` — explicit RK 5(4) embedded pair. Stages s = 7, FSAL (first-same-as-last) → 6 effective evaluations. Local error estimate from the embedded 4th-order solution. Used for non-stiff plants (open-loop converters, motor drives in light-load regimes).
  - `RadauIIA3.hpp` — implicit Radau IIA, s = 2 stages, order 3. L-stable, A-stable. Newton iteration with the existing sparse Jacobian backend. Used for stiff plants (snubber-clamped IGBT modules, JA core hysteresis, thermal-electrical co-sim).
- **Step controller plumbing** — wire the new integrators behind the existing `opts.step_mode = TransientStepMode::Variable` switch:
  - `opts.advanced.timestep.integrator = Integrator::DormandPrince54 | Integrator::RadauIIA3 | Integrator::Tustin | Integrator::Bdf1` (default unchanged: Tustin / fixed).
  - The existing LTE-based + Newton-feedback step controller (from `transient-timestep` spec) consumes the per-step error norms from DOPRI5 and Radau directly.
  - The existing event-driven step trimming (`transient-timestep`) cooperates with the variable-step integrator: when an event lands inside a step, the step is bisected and resumed at the event time using the integrator's dense-output formula.
- **Dense output** — both integrators expose a continuous interpolant so the existing `step_observer` can be evaluated at exact event times without a redundant solve.
- **Python API**:
  - `SimulationOptions.advanced.timestep.integrator` enum exposed as a Python `IntEnum` with values `DormandPrince54`, `RadauIIA3`, `Tustin`, `Bdf1`.
  - `pulsim.simulate(..., integrator="dopri5", step_mode="variable")` shorthand.
- **Tests**:
  - Lorenz / Van-der-Pol non-stiff manufactured-solution test against analytical / scipy `solve_ivp` reference within 1e-5 relative tolerance.
  - RC discharge (stiff at small `R*C`) showing Radau IIA(3) converges in 5–10 large steps where Tustin would need thousands.
  - PSFB benchmark: variable-step DOPRI5 vs fixed-step Tustin, demonstrate ≥ 5× wall-clock speedup at equivalent accuracy.
  - Event-detection regression: switched-mode diode commutation precisely lands at zero-crossing time within 1 ns tolerance.
- **Docs** — new page `docs/v2/solvers.md` describing all four integrators, when to use each, and a tuning recipe for `rtol` / `atol`.

## Impact

- **Affected specs**: `transient-timestep` (new requirements for DOPRI5 and Radau IIA integrators, integrator selection enum).
- **Affected code**: ~1500 LOC new C++ (~600 each for the two integrators + step-controller wiring + dense-output formulas), ~80 LOC pybind, ~50 LOC Python wrapper.
- **Backward compatibility**: PURE ADDITION. The default `opts.step_mode = TransientStepMode::Fixed` + `Integrator::Tustin` keeps every existing user's simulation byte-identical. New integrators are opt-in.
- **Performance**: 5–10× speedup on transient settling / steady-state tails (matches PSIM's DOPRI5 vs Tustin numbers). Stiff regimes see proportionally larger gains.
- **Risk**:
  - **Newton-failure on Radau**: implicit stage requires Newton convergence each step; if the Jacobian goes singular the step must be rejected. Mitigated by reusing the existing `enable_nonlinear_refresh` infrastructure.
  - **Event-time precision**: dense-output interpolants are 4th-order accurate, sufficient for switching-event localisation under any realistic dt budget.
  - **Two integrator regimes split user mental model**: the docs page above must be unambiguous about default behaviour to avoid surprise breakage.
