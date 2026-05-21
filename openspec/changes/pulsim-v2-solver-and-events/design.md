# Design — `pulsim-v2-solver-and-events` (Layer 5)

## What Layer 5 V0 does

Layer 4 gave us **`cache.solve(mask, b_extra, x)`** — one cached
triangular solve per switch state. Layer 5 V0 is the **smallest
loop** that calls `solve` repeatedly to produce a transient
waveform:

```
for k = 0..N:
    t      = t_start + k · dt          # integer step counter
    mask   = switch_fn(t)              # user-supplied schedule
    b_ex   = b_extra_fn ? b_extra_fn(t) : 0
    cache.solve(mask, b_ex, x)         # the PLECS-style hot path
    record(t, x)
```

That's it. The entire architectural value of v2 is the
`cache.solve` line — one map lookup + one O(nnz) triangular solve.
Layer 5 V0 wraps it in fixed-dt time-stepping, output recording,
and input validation. Nothing else.

## Why V0 is intentionally small

Modern time-domain solvers do much more than what Layer 5 V0 does:

| Capability                       | V0 | Where it lands |
|----------------------------------|:--:|----------------|
| Fixed dt                          | ✅ | here           |
| Externally-scheduled switching    | ✅ | here           |
| Time-varying source RHS (callback)| ✅ | here           |
| Event detection (zero-crossing)   | ❌ | V1 follow-up   |
| Adaptive dt + LTE                  | ❌ | V1 follow-up   |
| Trapezoidal integrator (C/L)      | ❌ | needs Layer 4 V1 |
| Newton for nonlinear              | ❌ | needs Layer 4 V2 |
| Strided output / down-sampling    | ❌ | V1 add         |
| Python bindings                    | ❌ | Layer 6        |

Each "❌" requires its own OpenSpec; piling them into V0 would
make the loop unrecognisable and bury the architectural pivot
(Layer 4's `cache.solve` doing all the heavy lifting per step)
under integrator scaffolding.

V0 with this scope is **the smallest viable end-to-end demo of
v2**: build a graph → register devices → build cache → call
`run_transient` → get back `(t, x(t))` waveforms.

## Why external switching, not auto-events

The natural design impulse is "the solver should detect when a
diode's current crosses zero and update the switch state". That's
event detection, and it's a SIGNIFICANT piece of machinery:

1. Define what an event IS (a sign change in a watched scalar
   between `t_n` and `t_n + dt`).
2. Bisect/Brent's-method to find the precise event time.
3. Trigger a switch state update + a cache lookup at the new
   state.
4. Validate the new state's consistency (no instantaneous
   discontinuities in node voltages).

For V0 we **outsource** all of that to the user via the
`SwitchScheduleFn` callback. The user is expected to:
- Pre-compute the schedule (e.g., a PWM waveform table)
- Use a gate-driver model that decides switch states
- Or just compute it analytically every call

This is also the model used by PLECS for "ideal switching"
simulations driven by gate-driver signals — perfectly sufficient
for the chopper-PWM workload Layer 5 V0 needs to demonstrate.

V1 will add an auto-event scheduler that runs INSIDE the loop:
detect a sign change in a watched signal (e.g., a switch's
current), bisect, fire `switch_fn` updates, continue. The API
boundary (`switch_fn` as a `std::function`) doesn't change —
auto-events just become "yet another `switch_fn` implementation".

## Why fixed dt

Adaptive integration (RKF45-style LTE estimation with dt halving
on excess error) is essential for stiff circuits with widely
separated time constants. But:

1. **It needs caps/inductors first.** Adaptive dt is meaningless
   for static circuits — the solution is exact at every step.
2. **The cache invalidates on dt changes.** When dt changes
   adaptively, the trapezoidal companion `g_eq = 2C/dt` changes,
   the MATRIX changes, every cached factor becomes stale. That's
   the cache-invalidation problem, and it's the right time to
   address it — in the V1 trapezoidal follow-up where caps/L
   land.
3. **The chopper test doesn't need it.** PWM at fixed dt is the
   natural model; the user picks dt to resolve the switching
   transients.

So V0 takes fixed dt as a hard scope choice and defers adaptive
dt to the integrator V1.

## Why all-zero initial state

With Layer 4 V0 restricted to **Resistor + VoltageSource +
Switch**, there's no state carried between steps. The solution
`x(t)` is purely a function of `(switch_state, b_extra)` — it's
static. So `x_init = Vector::Zero(state_size)` is fine; the first
cached solve produces the correct `x(t_0)`.

When caps/inductors land in the trapezoidal V1 follow-up, the
solver will need a real initial-condition API:
- `Vector x0` parameter (DC operating point or user-supplied)
- A history term `i_hist` carried in `b_extra` from step to step

That's the same callback API we have today — `b_extra_fn(t)` can
trivially incorporate history terms. V0 doesn't expose it because
it's not yet needed.

## Output recording strategy

V0 records `(t, x)` every step into `std::vector<Vector>`. Cost
per step: one `Vector` copy (a few dozen bytes for typical PE
circuits) + one `push_back` (amortized O(1)).

For a typical PWM simulation:
- 1 µs dt × 1 ms span = 1000 steps
- 50-node circuit = ~400-byte state vectors
- Total memory: ~400 KB

That fits comfortably. For larger simulations (1 s span, 10 µs
dt = 100k steps × 400 B = 40 MB), it's still fine. For
"acquisition-class" simulations (10 s span, 10 ns dt = 1 G steps),
we need strided output recording — V1 add.

V0 doesn't optimise this. The user can always coarsen `dt` if
they don't care about transient detail.

## The `BExtraFn` callback

Why a callback and not a pre-computed vector? Three reasons:

1. **Generality.** A sinusoidal source `V(t) = A · sin(2πft)`
   is trivially expressed as a lambda. Pre-computing requires
   storage proportional to the step count.
2. **Composition.** Multiple time-varying sources combine into
   one `b_extra` lambda by adding their contributions. The user
   builds the composition; the solver doesn't care.
3. **History terms (V1 forward-compatibility).** When caps/L
   land, the history contribution `−i_hist` lives in `b_extra`.
   That's stateful — the solver passes the previous-step's `x`
   into a stateful lambda that computes `i_hist` and adds it.
   The current callback signature already supports this (the
   lambda closes over solver state).

V0 defaults to no callback (`b_extra_fn = {}`), in which case
the solver uses a pre-allocated zero buffer to avoid per-step
allocation.

## Numerical care: integer step counter vs `t += dt`

Floating-point accumulation over thousands of steps drifts. After
1000 steps of `dt = 1e-6`, `t = 1000 · 1e-6 = 0.001` exactly in
infinite precision, but `t += dt` in double can drift by several
ULP. For PWM, even small drift causes spurious switching at the
period boundary.

V0 uses an integer step counter and recomputes
`t = t_start + k · dt`. This is one extra multiply per step
(negligible cost) and preserves exact reproducibility.

## Lifetime + thread safety

`run_transient` takes:
- `const PwlStateSpaceCache&` — borrowed reference; caller owns
  the cache, must outlive the call (and indeed any retained
  result that references its state vectors).
- `const SwitchScheduleFn&` + `const BExtraFn&` — borrowed
  references to `std::function`s; caller owns them.
- `const SimulationOptions&` — borrowed; ditto.

The function is **not** thread-safe with respect to mutation of
the cache or the callbacks during the call. Concurrent reads
are fine (the cache is logically `const`, the callbacks are
expected to be re-entrant). V0 does not run parallel time-stepping
— that lives in V1 if needed (typically only useful for
parameter sweeps, not within a single simulation).

The returned `SimulationResult` owns its vectors and is
safely returnable by value (NRVO + move semantics).

## Test strategy

V0 tests focus on **the loop machinery**, not numerical
integrator correctness (no caps/L yet, so there's no integrator
to validate):

1. **Input validation** — bad options, bad state size, empty
   `switch_fn` all throw.
2. **Step count + time grid** — `expected_step_count` matches the
   observed `num_steps`; times are exactly `t_start + k · dt`.
3. **Callback wiring** — `switch_fn` is consulted every step;
   `b_extra_fn` is consulted every step when supplied; default
   path uses zero `b_extra`.
4. **End-to-end correctness** — chopper at 10 kHz PWM, mean v_out
   matches `V_dc · duty`, square wave shape matches the analytical
   ON/OFF values.

This is enough to ship V0 with confidence. The integrator-quality
tests come with the trapezoidal V1 follow-up.

## Risks

1. **`std::function` overhead per step.** Each call is a virtual
   dispatch + small allocation if the lambda doesn't fit in the
   small-buffer optimisation. For 10k steps/ms this could be
   measurable — but the integration test caps wall time at < 100 ms
   for 1000 steps, leaving plenty of headroom. If this becomes
   a real bottleneck in V1+ workloads, we templatise the callback
   types (V1 follow-up: header-only generic `run_transient`).

2. **Output vector copies.** Each step copies the state vector into
   the result. For 50-node circuits and 100k-step simulations this
   is 50 × 100k × 8B = 40 MB of copy traffic. Acceptable for V0;
   strided output is the V1 add.

3. **Cache invalidation on dt change.** V0 doesn't change dt, so
   this isn't exercised. When the trapezoidal V1 follow-up lands,
   the cache will need an `invalidate()` method or a
   "build_with_dt(dt)" API. The current `cache.solve` signature
   doesn't take dt — that's fine for V0, and V1 will extend it
   in a backwards-compatible way (new overload + deprecation note
   on the old signature, or a different cache class that
   composes with the existing one).

## API affordances

```cpp
namespace pulsim::v2::solver {

struct SimulationOptions {
    Real t_start = 0;
    Real t_end   = 0;
    Real dt      = 0;

    [[nodiscard]] bool valid() const noexcept;
    [[nodiscard]] Size expected_step_count() const noexcept;
};

struct SimulationResult {
    std::vector<Real>   times;
    std::vector<Vector> states;

    [[nodiscard]] Size num_steps() const noexcept;
    [[nodiscard]] bool empty()     const noexcept;
    void reserve(Size n);
};

using SwitchScheduleFn =
    std::function<topology::SwitchStateMask(Real)>;
using BExtraFn =
    std::function<Vector(Real)>;

SimulationResult run_transient(
    const pwl::PwlStateSpaceCache& cache,
    Size state_size,
    const SimulationOptions& opts,
    const SwitchScheduleFn& switch_fn,
    const BExtraFn& b_extra_fn = {});

}  // namespace pulsim::v2::solver
```

That's the entire V0 public surface. Layer 6 (frontend) will wrap
this in a Python binding + a YAML/schematic-driven runner.

## What V0 hands to Layer 5 V1

V1's job is to start replacing manual switching with auto-event
detection while keeping the same `run_transient` signature. The
sketch:

```cpp
// V1 sketch — same signature, fancier `switch_fn` internally.
auto auto_event_switch_fn = make_event_driven_schedule(
    /*initial state*/      SwitchStateMask{n_switches},
    /*watch signals*/      {{branch_id_diode, &v2::topology::current_view}},
    /*on threshold cross*/ [](auto& mask, auto branch_id){
        mask.toggle(branch_id);
    });
auto result = run_transient(cache, state_size, opts,
                             auto_event_switch_fn);
```

V0's loop just calls `switch_fn(t)` — whatever the user puts in.
V1 supplies a richer `switch_fn` factory that observes the running
state and fires its own updates. No change to the loop. That's
the architectural payoff of keeping V0 small.
