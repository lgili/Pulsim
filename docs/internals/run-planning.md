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

## What still differs per engine (today's behaviour, kept)

Stage A moved code without changing a result, a warning or a refusal. These
blocks are per engine on purpose until the number-changing release decides
them (DECISIONS 23 of `add-kernel-simulation-object`):

| block | pwl | trbdf2 | dsed |
|---|---|---|---|
| no `switch_fn` (`_undriven`) | all closed + warning, decided in the run after the cache hooks | all open, silent, decided while planning | all closed + warning, decided inside the exact dispatcher |
| `mmc_arms`, `on_cache` | composed in the run (they need the run's cache) | refused by the route | refused by the route |
| `progress`, `live_stream` | wired in the run | refused / routed | refused |
| finalize | breaches, dt retries, guard, resolution and unlocated-edge warnings; voltage check | stats warnings; voltage check | `on_impulse='error'` refuses after the run; voltage check |
| `mode_selection='auto'` | resolved by the route | resolved by the route (always the settle) | resolved inside the exact dispatcher |

Stage B moves the plan into the kernel (`sim::RunOptions`, `sim::Simulation`).
