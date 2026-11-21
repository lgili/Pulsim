## Context

An optimiser calls `f(design)` thousands of times, often from several
threads or processes, and expects: the same answer for the same input,
a finite answer or an explicit failure, and a cost per call it can
budget. Pulsim's pieces today answer a different question (a sweep
over a grid with one configuration), and its default fixed engine
makes event-dependent KPIs piecewise constant in the parameters
(proposal table). No prototype in this change: everything it composes
exists and is tested; the numbers it cites come from the changes it
uses.

Owner rules that apply: typed refusals, never a silent fallback
(owner note *gui-swallows-backend-refusals*: a broad `except` turned
refusals into wrong answers — here a refusal becomes a record, never a
default value); time-weighted averages on adaptive grids (owner note
*time-weight-averages-on-adaptive-grids*); synthetic parts flagged.

## Goals / Non-Goals

- First slice: runs on today's kernel (resistor-pair switches, `dsed`
  and `trbdf2`, `periodic_steady_state` slice 1); its tests use only
  what is integrated. The ideal-switch and exact-route numbers improve
  the KPIs' smoothness later without changing the contract.
- Goals: one evaluation contract; deterministic, thread-safe records;
  typed failures an optimiser can consume; a closed `T_j` loop now and
  a path to the in-kernel loop; batch evaluation; one worked example a
  research group can copy.
- Non-Goals: shipping an optimiser (the example's dependency is
  optional); magnetic and capacitor design models beyond a documented
  illustrative volume law; gradient-based optimisation before
  `add-parameter-sensitivities`; distributed (multi-machine) batches.

## Decisions

### D1. The contract

```python
@dataclass(frozen=True)
class EvaluationRecord:
    design: Mapping[str, float | str]       # as given; hashed into design_id
    design_id: str                           # sha256 of the canonical design + options
    status: Literal["ok", "refused", "failed", "infeasible"]
    kpis: Mapping[str, float]                # empty unless status in ("ok", "infeasible")
    losses: Mapping[str, Mapping[str, float]]   # device → {conduction, switching, recovery, core, copper, esr}
    t_j: Mapping[str, float]                 # device → °C
    violations: Mapping[str, float]          # constraint → amount (≥ 0), in the KPI's unit
    reason: str | None                       # refusal / failure text
    refusal: tuple[str, str, str, str] | None   # (engine, feature, device, kind)
    counters: Mapping[str, float]            # route, PSS iterations, events, solves, thermal iterations
    provenance: Mapping[str, str]            # pulsim version, build hash, options digest
    wall_s: float                            # informational; never part of equality

class Evaluator:
    def __init__(self, build: Callable[[Mapping], Circuit], *, period: float,
                 kpis: Sequence[Kpi] = DEFAULT_KPIS, constraints: Mapping[str, Bound] = {},
                 thermal: ThermalSpec | None = None, steady: SteadySpec = SteadySpec(),
                 engine: str = "auto", options: Mapping = {}): ...
    def evaluate(self, design: Mapping) -> EvaluationRecord: ...
    def evaluate_many(self, designs: Sequence[Mapping], *, workers: int | None = None,
                      backend: str = "thread") -> list[EvaluationRecord]: ...
```

Default KPIs, each computed on one steady period with `np.trapezoid`
on the run's own time grid: efficiency `P_out/P_in`; total and
per-device losses by category (`device_loss_summary`); `T_j` per device
(D4); peak-to-peak output voltage and inductor current ripple; peak and
RMS current and peak blocking voltage per semiconductor; any user
`Kpi(name, fn(result) -> float)`.

### D2. Determinism and thread safety

- Same `design` + same options + same Pulsim build → equal records
  except `wall_s`. The evaluator fixes everything that could vary:
  engine route and options are resolved once in `__init__`; no RNG is
  used (a user model that needs one receives `seed = int(design_id,
  16) % 2**32`); thread counts inside the kernel are fixed to one per
  evaluation.
- Evaluations share no mutable state: `build` is called per evaluation
  (until `with_params` exists, D7), each simulation has its own
  workspace (the kernel's thread-local workspaces), and the record is
  immutable.
- Python callbacks (a `switch_fn`, a user KPI) hold the GIL; the
  evaluator records `counters['python_callbacks']` so a user sees why a
  batch did not scale. Native schedules (`make_pwm_switch_fn`, the
  `Modulator` objects of `add-unified-event-queue`) release it.
- Tested: a batch of 64 designs on 1 and 8 workers gives equal
  records, element by element (D1 equality).

### D3. Failures are records

| Raised during evaluation | Status | Record |
|---|---|---|
| `pulsim.UnsupportedError` | `refused` | `refusal = (engine, feature, device, kind)`, message |
| non-convergence: Newton, steady state, step floor | `failed` | reason, counters (iterations, residual, last `t`) |
| a constraint bound exceeded | `infeasible` | KPIs kept; `violations[name] = amount` |
| thermal fixed point diverges or `T_j > thermal.t_limit` | `infeasible` | `violations['thermal_runaway'] = T_j − t_limit` or the last increment |
| any other exception | re-raised | a bug, not a design outcome |

`evaluate` never returns a default KPI value in place of a failure,
and never catches a bare `Exception` to do so.

`evaluate_many` keeps going on design-specific outcomes. It stops and
raises when the first three evaluations are all `refused` with the same
four fields (the configuration, not the design, is unsupported), which
is the batch analogue of `pulsim.sweep`'s rule.

Penalty adaptors (`pulsim.design.adapters`):

- `as_objectives(record, names, penalty=1e30)`: the KPI vector, or
  `penalty` in every component for `refused`/`failed`;
- `as_constraints(record, names)`: the violation vector (`≤ 0` means
  feasible, the convention of NSGA-II-style optimisers), `+inf`-free:
  `refused`/`failed` give `penalty` in every component;
- the adaptor never hides the status: the caller can count them.

### D4. Junction temperature: a fixed point now

Until `add-in-kernel-event-losses` lands, `T_j` is closed outside the
run:

```
T⁽⁰⁾ = T_amb
repeat k = 0, 1, …:
    P⁽ᵏ⁾ = losses(waveforms, T⁽ᵏ⁾)                 tables read at T⁽ᵏ⁾ per device
    T̂ = T_amb + R_th·P⁽ᵏ⁾                           steady thermal network (device_thermal_summary)
    T⁽ᵏ⁺¹⁾ = T⁽ᵏ⁾ + ω·(T̂ − T⁽ᵏ⁾)                    ω = 1, halved when the increment grows
until ‖T⁽ᵏ⁺¹⁾ − T⁽ᵏ⁾‖_∞ < 0.1 K, at most 20 iterations
```

- The waveforms are re-simulated only when the electrical solution
  depends on `T_j` (a temperature-dependent `R_on` or `V_f` in the
  circuit); otherwise one simulation serves every iteration and only
  the table look-ups repeat.
- With a loss linear in temperature, `P = P₀·(1 + α·(T − 25))`, the
  fixed point is `T* = (T_amb + R_th·P₀·(1 − 25α)) / (1 − R_th·P₀·α)`
  and exists only for `R_th·P₀·α < 1`; this closed form is the test
  oracle, and `R_th·P₀·α ≥ 1` must give `infeasible`
  (`thermal_runaway`).
- Recorded: `counters['thermal_iterations']`, the final increment.

Supersession: when `add-in-kernel-event-losses` is the default,
`ThermalSpec(mode='kernel')` becomes the default and the loop is kept
as `mode='fixed_point'` for comparison.

### D5. Steady state

- `SteadySpec(method='pss')`: `periodic_steady_state(builder,
  period=…)`; its failure is a `failed` record with the solver's
  reason. Slice 1 runs on the exact stepper and refuses circuits it
  cannot take (typed); the refusal becomes `refused`, and the evaluator
  does not fall back to a transient silently.
- `SteadySpec(method='transient', max_periods=N, tol=…)`: periods are
  simulated until the period-to-period change of the state at the
  period start is below `tol` (relative, on the circuit scale) or `N`
  is reached (`failed`, reason `'not_settled'`).
- `SteadySpec(method='auto')` (default): `pss` when the capability
  check accepts the circuit, otherwise `transient`; the choice is in
  `counters['steady_method']`. This is a documented route, not a
  fallback after a failure.

### D6. Batch evaluation

`evaluate_many` uses `pulsim.sweep`'s job runner with the thread
backend (`backend='process'` accepted for Python-heavy designs). The
result list is in input order. Per-design wall time is recorded; the
batch's wall time is not part of any record.

### D7. Re-binding and gradients (later, from other changes)

- With `add-immutable-circuit-ir`, `Evaluator` accepts a `Circuit` with
  named parameters instead of `build`, and evaluates with
  `circuit.with_params(design)`: no rebuild, one symbolic analysis for
  the batch when the topology does not change.
- With `add-parameter-sensitivities`, `evaluate(design,
  gradients=True)` adds `∂KPI/∂p` for the KPIs that change supports.
  Finite differences are not offered by the evaluator: on the fixed
  engine their relative error is −12.9 … +9.1 (proposal table).

### D8. The worked example

`examples/optimization/buck_pareto.py`: synchronous buck, 48 V → 12 V,
10 A.

| Variable | Range |
|---|---|
| `f_sw` | 50 kHz … 500 kHz (log) |
| `L` | 2.2 µH … 100 µH (log) |
| `C_out` | 22 µF … 470 µF (log) |
| switch part | two synthetic MOSFET entries of `lib/data` (`allow_synthetic=True`) |

Objectives: maximise efficiency, minimise box volume. Constraints:
output ripple ≤ 1 % of 12 V, `T_j ≤ 125 °C` at `T_amb = 40 °C` with a
heatsink sized to the losses. Volume law, stated as illustrative
(not a magnetics design): inductor volume from the area-product
scaling `V_L = k_L·(L·I_pk·I_rms)^{3/4}`, capacitor volume
`k_C·C·V²`, heatsink volume `k_hs/R_th,sa` with `R_th,sa` the largest
that meets `T_j ≤ 125 °C`; constants in the example with their
source or marked as assumptions.

Optimiser: `pymoo` NSGA-II if importable (an optional extra
`pulsim[opt]`), otherwise 512 Latin-hypercube designs and a
non-dominated filter in numpy. The example prints the status counts
(ok / refused / failed / infeasible) and plots the front with the
synthetic-part warning in the title.

LLC is the follow-up example once `periodic_steady_state` covers
resonant converters on the exact route (Phase 4 PSS completion).

## Risks / Trade-offs

- Fixed-point loop cost: two to four simulations per design when
  `R_on(T)` is in the circuit → recorded; removed by the in-kernel
  losses.
- KPIs that are staircases on the fixed engine → `engine='auto'`
  default, the exact route after the pillars land; the docs page shows
  the proposal table.
- Users treating `penalty` as a real value → the adaptor documents the
  status counts; the example prints them.
- Synthetic parts in a published front → flagged in every output; the
  real-part decision is the owner's.

## Open Questions

None for the owner. To settle by measurement in tasks: the default
settle tolerance of `method='transient'`, and the batch size at which
the thread backend stops scaling for the example (Python callbacks).
