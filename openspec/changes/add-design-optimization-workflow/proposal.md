## Why

The first external users the roadmap expects are a doctoral
optimisation group: they size converters by running thousands of
simulations inside a multi-objective optimiser (efficiency against
volume, loss against cost) and need each simulation to be a function
call that returns numbers, returns them the same way every time, and
fails in a way an optimiser can use. Pulsim has the parts and not the
contract:

- **Sweeps exist, evaluation records do not.** `pulsim.sweep` and
  `monte_carlo` run jobs on a thread pool when the schedule is native
  (`python/pulsim/sweep.py:148-220`, `update-sweep-threading-and-import-time`,
  done on `feat/core-integration`); a raising job is recorded as
  failed, but a configuration refusal (`UnsupportedError`) aborts the
  whole sweep (`sweep.py:151-156`), which is right for a sweep and
  wrong for an optimiser whose designs differ structurally.
- **Losses need `T_j`, and `T_j` needs the losses.** Switching-energy
  tables are read after the run at a junction temperature the caller
  supplies (`python/pulsim/losses.py:1315-1325`,
  `add-in-kernel-event-losses`); switching loss reaches the thermal
  network as a constant (`thermal.py:622-626`). An optimiser that wants
  `T_j ≤ 125 °C` as a constraint has to close the loop itself.
- **Gradients by finite differences are wrong on the default engine.**
  `∂v_C(40T)/∂L` of the DCM buck, central differences at relative steps
  1e-6 … 1e-2 (`../add-parameter-sensitivities/proposal.md`):

  | Engine | Relative error |
  |---|---|
  | `pwl`, dt = T/100 | −9.3, −9.5, −12.9, +2.0, −0.46 |
  | `pwl`, dt = T/400 | +9.1, +9.1, +2.6, −0.040, +0.041 |
  | `trbdf2`, default tolerances | −1.1e-4 … +2.9e-3 |
  | `dsed` | +1.75e-7 at 1e-6; three of five steps abort ("D1 is chattering") |

  The fixed engine places a diode commutation on its grid, so any KPI
  that depends on an event time is a staircase in the parameters: a
  gradient-based optimiser on it follows noise. With ideal switches,
  complementarity mode selection and exact propagation the same
  difference is smooth (prototype: all six `L·(1 ± δ)` runs complete,
  `∂v_C(40T)/∂L = 1731.970476 V/H` for the ideal buck,
  `../add-complementarity-mode-selection/probes/lcp_cases.out` (iii)),
  but the default for an optimiser must not depend on which engine the
  user picked.
- **Steady state is the expensive part.** Running 400 periods to read
  one steady period is the common workaround; the periodic-steady-state
  solver's first slice reaches the DCM buck orbit, 8.996247087236 V
  against the 50-digit 8.99624708728943 V, in 5 map evaluations
  (`add-periodic-steady-state-shooting`, slice 1 on
  `feat/core-integration`).

## What Changes

- **`pulsim.design.evaluate(design) -> EvaluationRecord`**: a
  documented contract around a user `build(design) -> circuit`
  function. The record holds the design, a status (`ok`, `refused`,
  `failed`, `infeasible`), the KPIs (efficiency; loss breakdown per
  device and per category; junction temperatures; output ripple;
  voltage and current stresses; a user volume or cost model), the
  constraint violations, the route and solver counters, and the
  provenance (Pulsim version, build hash, options).
- **Deterministic and thread-safe**: the same design and options give a
  bit-identical record on any thread of any batch size; evaluations
  share no mutable state.
- **Failures are records, not exceptions**: `UnsupportedError` →
  `refused` with its four fields; non-convergence of the steady state
  or of Newton → `failed` with the reason and the counters; a violated
  constraint or a thermal runaway → `infeasible` with the violation
  magnitudes. An adaptor maps each status to an optimiser's penalty or
  constraint form.
- **A fixed-point junction-temperature loop** until
  `add-in-kernel-event-losses` closes it in the kernel: losses at
  `T_j⁽ᵏ⁾`, `T_j⁽ᵏ⁺¹⁾` from the thermal network, damped, to 0.1 K; a
  re-simulation only when the electrical solution depends on `T_j`.
- **Steady state** through `periodic_steady_state` where it applies,
  otherwise a transient with a settle criterion; KPIs on one steady
  period, time-weighted.
- **Batch evaluation** `evaluate_many(designs, workers=…)` on the thread
  backend of `pulsim.sweep`; order kept; a design-specific refusal
  does not stop the batch.
- **A worked Pareto example**: synchronous buck, efficiency against
  box volume, over switching frequency, inductance, output capacitance
  and a part choice, with an OPTIONAL optimiser dependency (pymoo when
  installed; otherwise a Latin-hypercube sample and a non-dominated
  filter in numpy).
- **Derivative-free by default**. Gradients only through
  `add-parameter-sensitivities`, never finite differences through the
  fixed engine.

## Impact

- Affected specs: `design-optimization` (new).
- Affected code: new `python/pulsim/design.py` (evaluator, records,
  thermal fixed point, batch, penalty adaptors);
  `examples/optimization/buck_pareto.py` and a notebook; docs page
  "Using Pulsim inside an optimiser". No kernel change in this change.
- Uses: `pulsim.sweep` thread backend (done), `losses.device_loss_summary`,
  `thermal.device_thermal_summary`, `periodic_steady_state` (slice 1),
  `UnsupportedError` and the capability table (done).
- Needs from other changes, in the order they make the workflow better:
  - `add-in-kernel-event-losses`: replaces the fixed-point loop by
    `T_j` closed inside the run (one simulation per design instead of
    two to four).
  - `add-immutable-circuit-ir` (`with_params`): re-binding parameters
    without rebuilding the circuit, so a batch shares one symbolic
    analysis and the build cost leaves the per-design budget.
  - `add-parameter-sensitivities`: gradients for gradient-based
    optimisers, with the event jump correction.
  - `add-periodic-steady-state-shooting` completion: steady state for
    every engine and for nonlinear devices.
  - The two pillars (`add-complementarity-mode-selection`,
    `add-branch-form-ideal-switch`, `add-exact-propagator-strategy`):
    KPIs that are smooth in the design variables.
- The parts in `lib/data` are synthetic and refused by MPN unless
  `allow_synthetic=True` (owner note *shipped-parts-are-synthetic*); the
  worked example uses them with that flag and says so on every plot.
  Which real parts ship is the owner's decision.
