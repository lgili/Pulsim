## Context

Paths: `core/…` = `core/include/pulsim/…`; RT = `core/solver/run_transient.hpp`,
TB = `core/solver/trbdf2_transient.hpp`, PY = `python/pulsim/__init__.py`.

The fixed engine (`pwl`) and the variable-step engine (`trbdf2`) are
one kernel in two loops:

| Piece | Fixed (RT) | Variable (TB) | Same? |
|---|---|---|---|
| Linear solve | `cache.solve(mask)` at the cache's `dt` | `cache.solve_at(mask, γh)` for both stages | same factor cache |
| Companion history | `HistoryState` physical `(v, i)` | same object, plus `compute_b_extra_trbdf2_stage2` / `commit_trbdf2` | same |
| Device stamps | wrapper chain, `refresh_dt` | wrapper chain, `(dev_h, dev_stage)` | same stamps, two plumbings |
| Sources | `accumulate_b_extra` (RT:777-793) | `accumulate_sources` (TB:480-500) | copies |
| Diode decisions | re-solve the step until the mask repeats | sign change + Illinois + settle | different policies, one law |
| Step size | integer counter `t0 + k·dt` | LTE on differential variables, I-controller | different controllers |
| Failed Newton step | halving ladder (`max_dt_halvings`) | reject and shrink | different retry policies |
| Features | twelve only here | — | divergence |

The audit's judgement, which this change adopts: one engine is the
right end state for the two MNA drivers; DSED is a different
formulation (dense reduced state space) whose exact LTI advance is
worth keeping as a strategy, which `add-exact-propagator-strategy`
does.

What must be preserved (audit §d, `raw/numerics.md` §3, `raw/perf.md`):

- the `(G, C, b)` split with `solve_at(mask, dt)` and the lazy LRU
  factor cache;
- single-factor TR-BDF2 (γ = 2 − √2, one factorisation per `(mask, h)`);
- trapezoidal inside smooth segments (exact LC energy conservation);
- physical `(v, i)` history, which makes a variable step legal;
- the integer step counter on the fixed grid;
- per-direction diode signals and Illinois on trial solves;
- exact snapshot resume on the fixed engine;
- honest records: `dt_retries`, event breaches, forced accepts;
- zero allocations in the step loop, thread-local workspaces.

Owner notes that are requirements:

- *fixed-engine-first-order-start*: no trapezoid assembled at `2h`; the
  elapsed time of a solve is the time its commit integrates. The arch
  audit adds that the first-order start and the sub-step correction are
  load-bearing: they are reproduced, not "fixed", by this refactor.
- *trbdf2-probe-solves-set-the-device-step*: one choke point for every
  solve.
- *nonlinear-refresh-wrapper-chain*: no per-kind wrapper in the loop.
- *flux-row-keep-conductance-form*, *newton-absolute-dx-is-load-bearing*:
  the Newton solve and its criteria are not touched here.
- *time-weight-averages-on-adaptive-grids*: equality of two steppers is
  judged on states at equal times or on time-weighted measures, never
  on sample means.

## Goals / Non-Goals

Goals:

- One step loop for the MNA formulation, parameterised by an integrator
  and a step controller.
- `Fixed` + `Trapezoidal` proven bit-identical to today's fixed engine
  before any user-visible switch; `Lte` + `TrBdf2` proven
  step-identical to today's variable-step engine.
- Every feature either engine-independent or refused for a stated
  structural reason.
- Retire `bdf1`, the V0 overload and `integrators/`.
- Extension points for the exact propagator, a higher-order stiff
  integrator and forward sensitivities.

Non-Goals:

- No change to the numerical method of either engine. Restart rules,
  event splitting, the LTE scale factor and the roundoff acceptance are
  Phase 2 changes; they arrive here as policies, in whatever state they
  are when this change starts.
- No change to the Newton solver or the factor cache (Phase 1).
- No retirement of the DSED schedulers.
- No new public engine names beyond the two `-legacy` aliases.

## Decisions

### D1. Shape: thin template over a non-template core

```cpp
namespace pulsim::step {

// One solve, fully described. The only way to ask for a solve.
struct StageSpec {
    device::Formula formula;   // BackwardEuler / Trapezoidal / Bdf2Stage2
    Real a;                    // d/dt q ≈ a·q + r  → matrix G + a·C
    Real factor_dt;            // key for cache.solve_at(mask, factor_dt); a == 2/factor_dt
    Real t_end;                // time the stage solves at
    Real h_elapsed;            // time the stage spans (commit integrates this)
    Real c2, c3;               // BDF2 weights
};

class StepperCore {            // non-template: everything that is not stage sequencing
public:
    bool  solve_stage(const Mask&, const StageSpec&, Vector& x_out);   // the single solve point
    void  commit(const Vector& x, const StageSpec& last);              // history + device state + accept()
    void  snapshot_into(StateCopy&) const; void restore(const StateCopy&);   // one copy each
    bool  process_instant(Real t);        // event queue order: ticks, edges, breakpoints, settle
    events::EventQueue& queue(); events::ModeSet& modes();
    // cache, DeviceSet + DeviceState, HistoryState, sources, workspace, hooks, stats
};

template <class I>
concept Integrator = requires(const I& i, Real h, int s) {
    { I::name }      -> std::convertible_to<std::string_view>;
    { I::order }     -> std::convertible_to<int>;
    { I::n_stages }  -> std::convertible_to<int>;
    { i.stage(h, s, /*t0*/ Real{}) } -> std::same_as<StageSpec>;
};
template <class I>
concept HasLte = Integrator<I> && requires(const I& i, const StepperCore& c, Real h) {
    { i.lte(c, h) } -> std::convertible_to<Real>;      // normalised error; ≤ 1 accepts
};

template <class C>
concept Controller = requires(C& c, Real t, Real err, bool ok) {
    { c.propose(t) }         -> std::convertible_to<Real>;   // h before event clamping
    { c.on_result(ok, err) } -> std::same_as<Decision>;      // Accept / Retry{h} / Abort
    { C::uses_lte }          -> std::convertible_to<bool>;
    { c.save() }             -> std::same_as<ControllerState>;
    { c.load(ControllerState{}) };
};

template <Integrator I, Controller C>
class Stepper {
public:
    StepStatus step();            // one accepted step (or abort)
    Result     run();
private:
    StepperCore& core_; I integ_; C ctrl_;
};
}
```

- The template contains the stage loop, the LTE call and the accept /
  retry branch; with `C::uses_lte == false` the error vector, the
  rollback copy and the reject branch are not compiled.
- Shipped instantiations: `Stepper<Trapezoidal, Fixed>` (`pwl`),
  `Stepper<TrBdf2, Lte>` (`trbdf2`, `auto`),
  `Stepper<BackwardEuler, Fixed>` (replaces `bdf1`; also the engine of
  a critically damped restart). `Stepper<TrBdf2, Fixed>` is available
  to tests. Selection at run time is a `std::variant` of these inside
  `sim::Simulation`.

- Alternatives considered.
  - *Run-time strategy objects (virtual `Integrator` / `Controller`).*
    A close call: two virtual calls per step are a few nanoseconds
    against a 380 ns step. Rejected because the fixed path must compile
    to today's loop (no LTE storage, no rollback copy) for the perf gate
    to hold on counters and for bit-identity to be easy to prove, and
    because the stage table is tiny. The public selection is run-time
    either way.
  - *Make the fixed engine a special case of the variable one with
    `h_min = h_max = dt`.* Rejected: the LTE controller's arithmetic
    would produce `t += h`, not `t0 + k·dt`, and the grid would drift;
    and the fixed engine's event policy is different by design.
  - *Keep two drivers and extract shared helpers only.* This is steps
    1–3 of the migration; stopping there leaves the feature divergence,
    which is the defect source.

### D2. Integrators are stage tables

```cpp
struct Trapezoidal {                 // order 2, 1 stage
    StageSpec stage(Real h, int, Real t0) const
    { return {Formula::Trapezoidal, 2/h, h, t0 + h, h, 0, 0}; }
};
struct BackwardEuler {               // order 1, 1 stage; BE over h has a = 1/h = trap's a at 2h,
    StageSpec stage(Real h, int, Real t0) const        // so its factor key is 2h
    { return {Formula::BackwardEuler, 1/h, 2*h, t0 + h, h, 0, 0}; }
};
struct TrBdf2 {                      // order 2, 2 stages, single factor
    static constexpr Real gamma = 2 - std::numbers::sqrt2;
    StageSpec stage(Real h, int s, Real t0) const;     // s=0: TR over γh; s=1: BDF2 to t0+h
    Real lte(const StepperCore&, Real h) const;        // moved from TB:1286-1375
};
```

- `factor_dt` is the key into the existing cache: the matrix is
  `G + (2/factor_dt)·C`. For TR-BDF2 both stages return `γh`, which is
  the single-factor property stated as data. For backward Euler over
  `h` the key is `2h`: this is a statement about which cached factor is
  reused, not about elapsed time. `h_elapsed = h` in every case, so
  nothing integrates over a doubled interval (the rejected "trapezoid
  at 2h" doubled the elapsed time).
- The restart rule of `add-discontinuity-restart-rule` (two backward
  Euler half-steps) is `BackwardEuler::stage(h/2, …)` twice: factor key
  `h`, the nominal factor, as that proposal requires.
- Extension points: `add-high-order-stiff-integrator` adds an
  `Esdirk43` type (five stages, one `factor_dt`, an embedded `lte`).
  `add-exact-propagator-strategy` adds `ExactPropagator` behind the
  integrator interface with an event-driven controller. To leave room
  for it, an integrator is not required to express its advance as
  `StageSpec`s: it may instead provide `advance(core, mask, h)` (exact
  for an LTI segment), which the stepper calls in place of the stage
  loop. The hook exists and is unused in this change.

### D3. The single solve point

```cpp
bool StepperCore::solve_stage(const Mask& m, const StageSpec& s, Vector& x_out) {
    const device::StepCtx ctx = to_ctx(s);            // formula, a, h, t — nothing ambient
    history_.rhs(s, b_stage_);                        // linear companions, formula-aware
    sources_.accumulate(s.t_end, b_src_);             // the one copy of source accumulation
    b_stage_ += b_src_;
    ++stats_.n_solves;
    return devices_.has_newton()
        ? newton_.solve(m, s.factor_dt, b_stage_, ctx, x_out)   // stamps via stamp_fq
        : (cache_.solve_at(m, s.factor_dt, b_stage_, x_out), true);
}
```

Stage solves, event probes, the zero-time settle, gate landings, retry
sub-steps and restart half-steps all call it. There is no other call
into the cache or the Newton solver from the stepper. This is
`trap_solve` (TB:741-768) and `run_event_iteration`'s "single point"
(RT:930-940) merged, with the step context as an argument instead of
captured variables.

### D4. One step, with policies

```
step():
  1. poll cancellation
  2. core.process_instant(t)              ticks → gate edges → breakpoints → settle   [event queue]
       if the mask or a source changed: restart.arm()
  3. h = ctrl.propose(t); h = min(h, queue.next_landing(t) − t)     (Fixed: h = dt; landing per EventPolicy)
  4. core.snapshot_into(rollback)          only when the controller or the event policy can reject
  5. plan = restart.armed() ? restart.stages(integ, h) : integ stages
     for each stage: ok &= core.solve_stage(mask, stage, x)
  6. err = uses_lte ? integ.lte(core, h) : 0
     switch ctrl.on_result(ok, err):
        Retry{h'}: core.restore(rollback); retry.policy; goto 5
        Abort:     raise ConvergenceError with the partial result
  7. events.after_solve(x):
        IterateAtStepEnd: modes.settle(re-solve this step) with cycle report; optional sub-step correction
        Localize:         bracket sign changes → locate() → truncate the step to the event → goto 5 with h = t_event − t
  8. core.commit(x, last stage); guards.after_commit(); hooks.on_step(view)
  9. t = ctrl.advance()                    Fixed: t0 + (++k)·dt ; Lte: t + h
```

Policies and what they reproduce:

| Policy | Values | Reproduces |
|---|---|---|
| `EventPolicy` | `IterateAtStepEnd{max_iter, strict, substep_correction}` | RT:930-1040 and RT:1310-1410 |
| | `Localize{t_tol, chatter_guard, held_redecide}` | TB:1375-1710 |
| `RestartPolicy` | `None` | both engines today |
| | `Cda` | `add-discontinuity-restart-rule` |
| | `DerivativeRefresh` / `Sdirk22First` | `update-trbdf2-post-event-restart` |
| `RetryPolicy` | `HalvingLadder{max_halvings}` | RT:1075-1160, with `dt_retries` records |
| | `ShrinkAndRepeat` | TB Newton retries, `n_newton_retries` |
| `GuardPolicy` | inductor freeze / clamp | RT's post-commit guards, with `inductor_guard_actions` |

Each policy is a small struct held by the controller's configuration;
the default set for `Fixed` is today's fixed engine and for `Lte`
today's variable-step engine. `add-fixed-step-event-splitting` is
`Fixed` + `Localize` with the linear-interpolation trial evaluator.

### D5. Features as plugins

| Feature (keyword) | Today | Becomes | After the port |
|---|---|---|---|
| `store_every`, `progress`, `live_stream` | fixed only | `hooks.on_step` recorder / observer plugins | both controllers (`store_every` on an irregular grid is defined by `add-probes-and-recorders`; refused until then) |
| `start_from_dc_op` | fixed only | an initial-condition source | both |
| `strict_event_iterations` | fixed only | option of the event policy | both |
| `max_dt_halvings` | fixed only | `HalvingLadder` | `Fixed` only; refused on `Lte` with the reason "the LTE controller already retries with a smaller step" |
| `enable_substep_state_correction` | fixed only | option of `IterateAtStepEnd` | `Fixed` only; refused on `Lte` ("events are localised") |
| `inductor_freeze_di_max`, `inductor_abs_clamp` | fixed only | `GuardPolicy` | both |
| `step_observer` without a cadence | fixed only | tick at `Rate{dt}` | `Fixed` only; refused on `Lte` ("no grid to ride") |
| C blocks, `closed_loops`, `BlockChain` | fixed, or both | tick handlers | both |
| `on_cache` (bidirectional thermal) | fixed only | tick handler with cache access (`refactor_parametric`) | both |
| `mmc_arms` | fixed only | source-term provider + parametric refactor at ticks | both, after its own test passes |
| `should_continue`, `b_extra_fn`, `initial_state`, `resume_from` | both | core | both |

The conformance matrix of `add-engine-conformance-matrix` is the
ledger: a feature × engine cell changes only in the PR that ports the
feature, with its test.

### D6. Controllers and snapshots

```cpp
struct Fixed { Real dt; Size k = 0; Real t0;                 // t = t0 + k·dt
               RetryPolicy retry; EventPolicy events; RestartPolicy restart; GuardPolicy guards;
               static constexpr bool uses_lte = false; };
struct Lte   { Real rtol, atol, h, h_min, h_max; StepMemory memory;   // moved from TB:1710-1810
               RetryPolicy retry; EventPolicy events; RestartPolicy restart; GuardPolicy guards;
               static constexpr bool uses_lte = true; };
struct ControllerState { std::array<Real, 8> v; std::uint8_t kind; };   // last h, memory slots, k
```

- The snapshot of `add-unified-device-interface`
  (`{t, x, dev_state, mask, …}`) gains an optional `ControllerState`.
  With it, a variable-step resume continues with the same next step and
  is bit-exact; without it (an older snapshot, or one taken on a
  different controller) the controller starts from its initial step and
  the resume is exact in state and "to tolerance" in trajectory, as
  today.
- A snapshot is strategy-independent in its physical part, so a run
  may be resumed by a different integrator or controller.

### D7. Equality proofs

Two hidden engine names exist during the migration, `pwl-next` and
`trbdf2-next`, selectable only through a private keyword used by the
harness.

- **Variable step.** `trbdf2-next` is built by moving code, not by
  rewriting it, so the claim is equality of arithmetic: for every
  harness circuit and every test circuit, the sequence of accepted
  times, the states, the mode record, `n_accept`, `n_reject`,
  `n_solves`, `n_gate_events`, `n_diode_events` are identical to
  `trbdf2`.
- **Fixed step.** For every harness circuit, every circuit of the test
  suite that runs on `pwl`, and one circuit per fixed-only feature, the
  checkpoint hash of the full state trajectory, the mode record,
  `commutation_events`, `event_iteration_count`, `dt_retries` and
  `inductor_guard_actions` of `pwl-next` equal those of `pwl`.
  Circuits containing a Newton kind that
  `add-unified-device-interface` ported with a stated deviation are
  compared against goldens re-baselined in that port, never in this
  change.
- The differential runner of the harness runs both names nightly until
  the legacy drivers are deleted. A difference is a defect in the new
  stepper by definition.

## Performance constraints

- `Fixed` + `Trapezoidal`: no LTE vector, no rollback copy unless the
  event or retry policy can reject, one mask lookup reused between
  edges, the primary cached segment; solves per step equal to the
  legacy engine on every harness circuit.
- `Lte` + `TrBdf2`: one factorisation per `(mask, γh)`; the factor-cache
  behaviour of `update-trbdf2-factor-reuse` (byte-budgeted event cache,
  geometric step grid) is the controller's, unchanged.
- Zero heap allocations per accepted, rejected or retried step; the
  five cases of `core/tests/perf/test_zero_alloc.cpp` are ported to the
  stepper and a variable-step case is added.
- No `std::function` in the stage loop: hooks are stored in a small
  vector and called once per accepted step; Newton stamps are the
  static loops of the device interface.
- Workspaces (`b_stage`, `b_src`, rollback copy, Newton system) belong
  to `StepperCore` and are sized once; thread-local pools are reused.
- Gate: the counter-based perf gate of the harness (`n_solves`,
  factorisations, cache hits and misses) equal between each `-next`
  engine and its legacy twin; ns/step on audit rows 1–3 and 5–9 not
  above the legacy value, best of five.
- Compile time: extension build measured before step 4 and after step
  9; budget +10 %.

## Risks / Trade-offs

- **Highest-risk change of the phase** (the audit's own ranking). →
  Nothing user-visible changes until two equality proofs pass; the
  legacy drivers remain selectable for one release; every port is a
  single-feature PR.
- **Bit-identity is fragile** (operation order in source accumulation,
  the order of history commits, `-ffp-contract=off` already set). →
  Steps 1–3 move code without reordering and are verified bit-identical
  one extraction at a time, before any loop is rewritten.
- **The load-bearing quirks** (first-order start from an inconsistent
  IC, the sub-step correction's inline half-commit, the saturable
  inductor's refusal on the retry ladder) are reproduced deliberately
  under `RestartPolicy::None` and the default policies; removing them is
  Phase 2 work with its own goldens.
- **Policy explosion.** Four policy axes could produce untested
  combinations. → Only the two default sets and the Phase 2 sets are
  constructible from the public options; other combinations are
  test-only.
- **Three steppers alive at once** (legacy, next, DSED). → A fixed end
  date: the legacy drivers are deleted one release after the switch.
- **`bdf1` users.** Its binding is used by `_dsed_dispatch.py` and two
  core tests. → Alias kept for one release; its tests are the oracle of
  step 8.
- **Pending change `add-adaptive-runge-kutta-solvers`** extends
  `core/integrators/`. → Owner decision required before step 9.

## Migration Plan

Preconditions: goldens and differential runner
(`restore-reference-benchmark-harness`), conformance matrix,
`add-unified-device-interface` steps 1–3, `add-unified-event-queue`
steps 2, 5 and 8. Each row is one or more PRs and is shippable alone.

| # | Step | Invariant |
|---|---|---|
| 1 | Extract `SourceAccumulator` (one copy of RT:777-793 / TB:480-500 / the DSED adapter's); both drivers call it. | Fixed goldens bit-identical; TR-BDF2 step sequence identical. |
| 2 | Extract `StepperCore::solve_stage` from `trap_solve` + `solve_dispatch` (TB) and the fixed engine's solve path (RT); both drivers call it for every solve. | Same; `n_solves` identical; `test_trbdf2_stateful_with_diodes.py` green. |
| 3 | Extract `commit`, `snapshot_into` / `restore` and the IC / resume seeding (RT:345-467, TB:420-585) into `StepperCore`; both drivers call them. | Same; `test_snapshot_resume*.py`, `test_initial_conditions_across_engines.py` green. |
| 4 | `Integrator` types; TB's two stages and RT's one stage are driven from `stage()`; `TrBdf2::lte` moved from TB. | Same. |
| 5 | `Lte` controller (TB:1710-1810) and `Fixed` controller (the integer counter) as types; policies `Localize`, `IterateAtStepEnd`, `ShrinkAndRepeat`, `HalvingLadder`, `GuardPolicy` extracted in place. | Same. |
| 6 | `Stepper<TrBdf2, Lte>` assembled from 1–5 as `trbdf2-next`; equality proof of D7. Then `engine='trbdf2'` and `'auto'` (no `dt`) select it; the old function stays as `trbdf2-legacy`. | Step-identical on the harness and the suite; conformance matrix equal for both names. |
| 7 | `Stepper<Trapezoidal, Fixed>` as `pwl-next` for plain runs; then one PR per fixed-only feature of D5, in this order: `start_from_dc_op`; recorder hooks (`store_every`, `progress`, `live_stream`); `strict_event_iterations`; `HalvingLadder`; sub-step correction; inductor guards; observers, closed loops, `BlockChain`; C blocks; `on_cache`; `mmc_arms`. | For each PR: the feature's test files produce bit-identical results on `pwl-next` and `pwl`; the `pwl-next` cell of the matrix flips from refused to supported in that PR only. |
| 8 | Equality proof of D7 over the whole harness and suite; then `engine='pwl'` and `'auto'` (with `dt`) select the stepper; the old function stays as `pwl-legacy` and behind the raw `run_transient` binding. | Fixed goldens bit-identical; counter gate equal; zero-alloc tests green on the stepper. |
| 9 | Features that are not structurally fixed-only are enabled on `Lte` (one PR each, with a cross-engine agreement test): `start_from_dc_op`, `strict_event_iterations`, inductor guards, `on_cache`, C blocks; recorder features when `add-probes-and-recorders` defines them. | Each new cell agrees with the fixed engine within the variable-step tolerance at equal times. |
| 10 | `Stepper<BackwardEuler, Fixed>` behind `run_transient_bdf1`; delete `core/solver/bdf1.hpp`. | `test_dc_operating_point.py`, `core/tests/layer4/test_gmin.cpp`, `core/tests/dsed/test_switch_state_record.cpp` green; first-order convergence test; bdf1 cells of the matrix now supported for Newton kinds and PWL diodes. |
| 11 | Delete the V0 overload (RT:179-265) after moving its tests; deprecate `core/integrators/`, `integrators.py`, `adaptive.py` (**BREAKING** after one release). | Suite green; stub check. |
| 12 | Controller state in the snapshot; variable-step resume exact. | A split variable-step run is bit-identical to the continuous one; older snapshots resume to tolerance. |
| 13 | One release after step 8: delete `pwl-legacy`, `trbdf2-legacy`, the bodies of `run_transient` and `run_transient_trbdf2`, and the hidden names. | Matrix; harness; suite. |

Rollback: until step 13 a single switch in the engine table returns
`pwl` and `trbdf2` to the legacy functions.

## Open Questions

1. `add-adaptive-runge-kutta-solvers` is an open change that builds on
   `core/integrators/`. Withdraw it in favour of
   `add-high-order-stiff-integrator`, or keep `integrators/` alive?
2. Naming: keep `engine='pwl'` for "fixed-step trapezoidal MNA" (the
   audit notes the name is misleading), or introduce
   `engine='fixed'` / `'variable'` with the old names as aliases? Not
   proposed here.
3. `max_dt_halvings` and the sub-step correction: refuse them on the
   LTE controller for the structural reasons in D5 (proposed), or map
   them to no-ops with a diagnostic?
4. Should `Stepper<TrBdf2, Fixed>` (an L-stable fixed step, no Nyquist
   ringing) be offered publicly, e.g. `integrator='trbdf2'` with `dt`?
   It falls out of the design at no cost but is a new public surface.
5. How long do `pwl-legacy` and `trbdf2-legacy` live: one minor
   release (proposed) or until 3.0?
6. Is step-identity (not only tolerance agreement) the right bar for
   the variable-step switch? It forbids incidental clean-ups during the
   move, which is the intent.

## Re-centred scope, 2026-11-13

> Appended; D1–D7 stand except where named below.

### D8. The kernel is event-driven first

```
run():
  t = t0; mode = selector.select(gates(t0), x, none, t0); x = reinit.apply(mode, x, t0)
  while t < t_end:
     e = queue.peek()                               time events: gate edges, breakpoints,
                                                    ticks, samples, t_end
     seg = route(mode, x, t)                        exact | implicit (add-exact-propagator-strategy D11–D13)
     (t_hit, x) = seg.advance(x, t, e.t, watch)     exact flow, or the step loop of D4 as the inner loop;
                                                    watch = crossing signals of the mode; returns at e.t
                                                    or at the first located state event before it
     t = t_hit
     handle every event at t (D2 order of add-unified-event-queue):
        samples → record (no mode change)
        ticks → controller outputs; gate edges; breakpoints
        mode = selector.select(gates(t⁺), x, mode, t)          one LCP
        x = reinit.apply(mode, x, t)                           projection, record
        seg.restart()                                          SDIRK(2,2) × 3 on the implicit strategies
```

The implicit inner loop is D4 with two changes: step 2
(`process_instant`) moves to the outer loop, and step 7's
`IterateAtStepEnd` / `Localize` keep only the location of a crossing
(the step is truncated to the located time and the outer loop takes
the event).

### D9. Integrators are segment strategies

| Strategy | Used for | Restart | Error control |
|---|---|---|---|
| Exact propagator | linear / PWL segments (default) | none (exact flow; the projection is explicit) | none needed; tier and route recorded |
| Trapezoid, fixed grid | `engine='pwl'`, and implicit segments of a fixed-`dt` run | three consecutive Alexander SDIRK(2,2) steps after every discontinuity | none (fixed grid) |
| TR-BDF2 | `engine='trbdf2'`, and implicit segments of an `auto` run | post-event restart (`update-trbdf2-post-event-restart`) | LTE (`update-trbdf2-error-control`) |
| Backward Euler | the `bdf1` entry point (D4 table) | — | — |

`StageSpec` already expresses an SDIRK stage: `E/(γh) − J` is the
trapezoidal matrix at `factor_dt = 2γh`, so the restart uses the
factor cache with no new factor kind.

### D10. Two shared services

- `ModeSelector` (`add-complementarity-mode-selection` D11): called
  once per event instant, after the gates and sources of the instant
  are applied. Replaces `ModeSet::settle` of `add-unified-event-queue`
  D6 at the default flip; until then `settle` is the legacy policy and
  the selector can run in shadow.
- `ReinitService` (`add-branch-form-ideal-switch` D4): called after the
  selector; on the exact strategy it projects explicitly, on the
  implicit ones it arms the restart and records the impulse.

No strategy contains a decision loop or a projection of its own.

### D11. Engine names as strategy selectors

| Name | Strategy | Status |
|---|---|---|
| `'auto'` (default) | exact where eligible, TR-BDF2 elsewhere | default after the flip |
| `'exact'` | exact; non-eligible segments refused unless allowed | new |
| `'dsed'` | alias of `'exact'` | legacy, warns after the flip, removed one release later |
| `'pwl'` | trapezoid on the fixed grid + SDIRK×3 restart | kept |
| `'trbdf2'` | TR-BDF2 with error control | kept |
| `'pwl-legacy'`, `'trbdf2-legacy'` (D7) | the legacy drivers | removed per task group 13 |

`integrator='rk45'|'bdf2'` on `'dsed'` follow
`add-exact-propagator-strategy` task 12.1 (deprecated, then removed).
