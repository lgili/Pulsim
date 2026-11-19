# How a run is planned

`pulsim.simulate()` is three steps (add-kernel-simulation-object, Stage A):

```python
def simulate(builder, t_end=None, dt=None, **kw):
    plan = _plan_run(builder, t_end, dt, **kw)       # resolve everything once
    return _finalize(_ENGINES[plan.engine](plan), plan)
```

| step | function | what it does |
|---|---|---|
| plan | `_plan_run` | every keyword resolved (`LoadedCircuit`, the `solver=` bundle, validation); the preflight ties (`_apply_ties`, the one place a run changes its circuit); `resume_from` and the initial state; the drive (`_compose`, then `_compose_for` the engine landed on); the route (`_route`: the option x engine table, the variable-step blockers, the `'auto'` rule); the engine's kernel options |
| run | `_run_pwl` / `_run_trbdf2` / `_run_dsed` | build the kernel object or cache, attach what needs the run's own cache, run; they read **only** the plan |
| finalize | `_finalize` | the result's handles (`_builder`, `plan`, `engine_used`, `engine_route_reason`), the post-run checks and warnings |

The plan is an immutable `pulsim._plan.SimPlan`: `engine`, `engine_asked`,
`route_reasons`, `features`, `step` (`FixedStep` / `VariableStep` /
`ExactStep`), `newton`, `events`, `initial`, `drive`, and `resolved_from`
(one entry per keyword: `explicit`, `solver`, `yaml` or `default`). It is
not `pulsim.RunPlan`, the circuit record a result carries (`result.plan`);
the plan holds that one as `circuit_plan`.

Every record of the plan is a `typing.NamedTuple`: a tuple with no
`__dict__`, so a field cannot be set or deleted and no attribute can be
added (`AttributeError`). Not frozen dataclasses: their `__init__` pays one
`object.__setattr__` per field, and the plan is built on every
`simulate()` call (task 7.2 keeps that call's fixed overhead near its
pre-Stage A value; `test_simulate_overhead.py` is a local bytecode
ratchet). `simulate()` builds the records with `tuple.__new__` and every
field in order, and passes its keywords to `_plan_run` and `_route`
positionally; `test_run_plan.py` checks each of those name lists against
the signature and each record's arity. The objects the plan holds (the
builder, the control plan, the user's callables) are not immutable.

## The rule: put it in the plan

`simulate()` used to be one 1 800-line function with three early returns,
one per engine. A check, a default or an initial condition written below
the first return applied to one engine only, silently: builder ICs, a
`resume_from` check and a preflight report were each found missing on one
engine this way. Now:

- Anything that must hold for every engine goes in `_plan_run`, before the
  route.
- An engine runner reads only `plan` (`test_engine_branches_read_only_the_plan`
  parses them with `ast`); a new `simulate()` keyword fails
  `test_plan_resolves_every_kwarg` until it is planned.
- Behaviour that legitimately differs per engine is an explicit per-engine
  entry of the plan or of `_finalize`, never a position in the code.

## What still differs per engine, and how the release makes it one

Stage A moved code without changing a result, a warning or a refusal. The
blocks that were per engine (DECISIONS 23 of `add-kernel-simulation-object`)
now each have ONE release behaviour behind a mechanism whose default is
today's (DECISIONS 26): a keyword resolved in the plan (`None` = today), a
library default in `pulsim._plan` the number-changing release flips
(`benchmarks/tools/next_defaults.py`, section `PLAN`), or "kept per engine"
where the difference is the engine's own.

| block | today: pwl | today: trbdf2 | today: exact (dsed) | release | mechanism |
|---|---|---|---|---|---|
| controlled switches nothing drives | all closed + one warning (in the run, after the cache hooks) | refused by name (`feature='switch_fn'`); `'auto'` without `dt` refuses, with `dt` it is pwl | all closed + one warning (inside the exact dispatcher; the `sensitivities=` run refuses) | `'refuse'`: `UnsupportedError` naming every switch, on every engine, above the route (only when nothing drives any switch: a partial drive is the drive's) | `simulate(undriven=None\|'refuse'\|'open'\|'closed')`; `None` reads `_plan.DEFAULT_UNDRIVEN` (`None` today); `'open'` / `'closed'` run that mask on every engine and through `'auto'`, one entry in `result.diagnostics['undriven']` and `plan.diagnostics` |
| count checks (`store_every`, `max_event_iterations`, `max_dt_halvings`) | Python, after the `mmc_arms` combinations | `max_event_iterations` by the kernel (its own wording); the others refused by name | all three refused by name | one check and one message per option on every engine that reads it, before the `mmc_arms` combinations | `_plan.UNIFORM_ENGINE_RULES` (False today) |
| `on_impulse='warn'/'error'` | refused at the event by the kernel (ideal branch-form devices) | the same | recorded; `'error'` refuses after the run (the scheduler can retract a trial projection's record); `integrator='bdf2'` never detects one and ignores the keyword | `integrator='bdf2'` refuses `on_impulse` by name; the exact engine's refusal names the device kind as the kernel's | `_plan.UNIFORM_ENGINE_RULES` |
| `mode_selection='auto'` | resolved in `_route` | resolved in `_route` (the settle: no rule yet) | resolved in `_route` (was: inside the exact dispatcher) | the same (done) | one function, `_resolve_mode_auto`, one rule per engine |
| voltage check | one function `_auto_check_voltage_sanity` on every engine (was: an inline copy on pwl) | same | same | the same (done) | none needed |
| `resume_from` clock | starts at the snapshot's `t` when `t_start` is left 0.0; an explicit `t_start` wins | same | refused by name | the same | one rule in `_plan_run`; `plan.resolved_from['t_start'] == 'resume_from'` |
| `mmc_arms`, `on_cache`, `progress`, `live_stream` | composed / wired in the run (they need the run's own cache or per-step ring) | refused / routed by the option table | refused (`progress` runs) | kept per engine | the option table refuses them by name |
| loss port on the exact engine | honours `integrator`, `t_start` | honours both | refuses `integrator='bdf2'` and `t_start != 0` by name (its events run on its own clock) | kept per engine | refusal by name |

Stage B moves the plan into the kernel (`sim::RunOptions`, `sim::Simulation`).
