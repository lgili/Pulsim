# Layer 5 — solver + events (the time-stepping orchestrator)

Layers 0-4 gave the v2 kernel everything it needs to **solve a
single switch-state snapshot** of a piecewise-linear circuit. The
Layer 4 cache pre-factorises one matrix per switch combination
and exposes the hot-path `solve(mask, b_extra, x)` that does a
hash-map lookup + a triangular solve — ~µs per call.

**Layer 5 is the orchestrator** that runs an actual transient
simulation: advance `t` from `t_start` to `t_stop`, look up the
switch state, call `cache.solve` once per step, record `(t, x(t))`.

For Layer 5 **V0**, the scope is intentionally tight:
- **Fixed `dt`** — no adaptive stepping, no LTE estimation.
- **User-supplied switching** via a `SwitchScheduleFn` callback.
  Event detection (zero-crossing on diode currents, etc.) is V1.
- **Optional time-varying RHS** via a `BExtraFn` callback for
  sinusoidal sources etc.
- **All-zero initial state** — static circuits only (no carryover
  between steps).
- **Output every step** — no down-sampling.

That's the simplest loop that demonstrates v2 end-to-end. The
chopper-PWM integration test in this OpenSpec is the proof: a 10
kHz PWM signal driving a chopper, 1 ms simulated at 1 µs steps,
mean output voltage equals `V_dc · duty` to within 1 %.

## The V0 surface

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
using BExtraFn = std::function<Vector(Real)>;

SimulationResult run_transient(
    const pwl::PwlStateSpaceCache& cache,
    Size state_size,
    const SimulationOptions& opts,
    const SwitchScheduleFn& switch_fn,
    const BExtraFn& b_extra_fn = {});

}  // namespace pulsim::v2::solver
```

Three header-only files. No `.cpp`, no `.so`, no special linking.

## The run_transient loop

```
┌─────────────────────────────────────────────────────────────┐
│  run_transient(cache, state_size, opts, switch_fn,          │
│                b_extra_fn):                                  │
│                                                              │
│    validate(opts, state_size, switch_fn)                    │
│    result.reserve(opts.expected_step_count())               │
│    x = Vector::Zero(state_size)                              │
│    zero_b_extra = Vector::Zero(state_size)                  │
│                                                              │
│    for k = 0 .. expected_step_count - 1:                    │
│      t       = t_start + k · dt    ← integer counter        │
│      mask    = switch_fn(t)                                  │
│      b_extra = b_extra_fn ? b_extra_fn(t) : zero_b_extra    │
│      cache.solve(mask, b_extra, x)   ← Layer 4 hot path      │
│      result.times.push_back(t)                               │
│      result.states.push_back(x)      ← copy of x             │
│                                                              │
│    return result                                             │
└─────────────────────────────────────────────────────────────┘
```

The `cache.solve(mask, b_extra, x)` line is the architectural
payoff. One `unordered_map` probe + one triangular solve on a
pre-factorised LU. ~µs per call, **NO** assemble, **NO** factorize,
**NO** Newton iteration.

For 1000 PWM steps that's ~ms of wall time. The same workload
with a v1-style Newton-per-step solver would run 10-50× slower.

## Numerical care: integer step counter

Naïve `t += dt` accumulates several ULP of drift over thousands
of steps. For a 10 kHz PWM at 1 µs `dt`, the drift over 1 ms is
roughly 100 steps × machine-epsilon = ~1e-14, which can move a
period boundary by a fraction of a sample. The next time the
schedule transitions, it might transition one step later than the
ideal model expects.

V0 uses `t = t_start + k · dt` with `k` an integer counter. One
extra multiply per step (negligible) + exact reproducibility.

## Worked example: 10 kHz PWM chopper

```cpp
#include "pulsim/v2/pwl/cache.hpp"
#include "pulsim/v2/pwl/device_pool.hpp"
#include "pulsim/v2/solver/options.hpp"
#include "pulsim/v2/solver/run_transient.hpp"
#include "pulsim/v2/topology/graph.hpp"

using namespace pulsim::v2;

// Build the chopper: V_dc → Switch → R → GND
topology::Graph g;
auto vin  = g.add_node("vin");
auto vout = g.add_node("vout");
g.add_branch(vin,  g.ground(), topology::BranchKind::Source);
g.add_branch(vin,  vout,       topology::BranchKind::Switch);
g.add_branch(vout, g.ground(), topology::BranchKind::PassiveLinear);

pwl::DevicePool pool;
pool.add_voltage_source(0, {.V = 12.0});
pool.add_switch(1, /*g_on=*/1e3, /*g_off=*/1e-9);
pool.add_resistor(2, {.G = 0.1});

pwl::PwlStateSpaceCache cache(g, pool);
cache.build();   // 2 segments (1 switch)

// 10 kHz, 50 % duty PWM schedule
solver::SwitchScheduleFn pwm = [](Real t) {
    const Real T = 1e-4;      // 100 µs period
    topology::SwitchStateMask mask(1);
    mask.set(0, std::fmod(t, T) < 0.5 * T);
    return mask;
};

solver::SimulationOptions opts{
    .t_start = 0,
    .t_end   = 1e-3,          // 1 ms simulation
    .dt      = 1e-6,          // 1 µs steps
};

auto result = solver::run_transient(
    cache, pool.state_size(g), opts, pwm);

// result.num_steps() == 1001
// mean(result.states[k][vout]) ≈ 6 V (V_dc · duty)
// Every sample is either at the ON value (≈ 11.9988 V) or at
// the OFF value (≈ 1.2e-7 V), depending on the PWM schedule.
```

That's the entire driver. Six API calls + one lambda. The cache
build runs once (sub-millisecond), the time-stepping loop runs
in another ~ms. **`cache.solve` is doing all the heavy lifting.**

## What V0 explicitly does NOT do

| Capability                          | Why deferred                            | Lands in                                |
|-------------------------------------|------------------------------------------|------------------------------------------|
| Event detection (auto zero-crossing)| Needs watch-signal API + bisection       | Layer 5 V1 follow-up                     |
| Adaptive dt + LTE estimation        | Needs error estimator + cache invalidate | Layer 5 V1 follow-up                     |
| Trapezoidal integrator (C / L)      | Needs `g_eq = 2C/dt` + `−i_hist`         | `pulsim-v2-trapezoidal-companion` (L4 V1)|
| Newton for nonlinear devices        | Needs per-segment Newton on cached factor| `pulsim-v2-nonlinear-segment-newton`     |
| Strided / down-sampled output       | Two-line loop change                     | Layer 5 V1                                |
| Templatised callback dispatch       | `std::function` overhead is fine for V0  | Layer 5 V1 if benchmarks demand          |
| Multi-rate / parallel time-stepping | Out of scope for any single-circuit sim  | Layer 6+                                 |

Each deferred capability gets its own OpenSpec. V0 is the
smallest viable end-to-end loop.

## Why external switching, not auto-events

Event detection is significant machinery:
1. Define what an event IS (sign change in a watched scalar).
2. Bisect (or Brent) to find the precise event time.
3. Update the switch state + look up the new segment.
4. Validate continuity at the transition.

For V0 we **outsource** all of that to the user via the
`SwitchScheduleFn` callback. The user supplies:
- A pre-computed schedule (PWM table)
- A gate-driver model
- A heuristic on the running state

This is also the model PLECS uses for ideal switching simulations
driven by gate-driver signals — perfectly sufficient for the
chopper-PWM workload V0 is designed to demonstrate.

V1's auto-event scheduler will plug into the same
`SwitchScheduleFn` slot: a factory function `make_event_driven_schedule(...)`
returns a `std::function` that, when called by `run_transient`,
observes the running state and emits its own switch updates. The
loop machinery in V0 doesn't change.

## Why fixed dt

Adaptive integration is essential for stiff circuits — but only
**once you have caps / inductors**. For Layer 5 V0's static-only
scope (no integrators), the cached solve gives the exact answer
at every step regardless of `dt`. Adaptive dt is meaningless.

When trapezoidal-companion caps/L land in `pulsim-v2-trapezoidal-companion`,
the Layer 4 cache becomes `dt`-dependent (the `g_eq = 2C/dt`
contribution lives in the MNA matrix). Adaptive dt then has to
invalidate stale factors. That's the right time to address it —
inside that follow-up.

## What V0 hands to V1

```cpp
// V1 sketch — same run_transient signature, fancier switch_fn.
auto auto_event_fn = make_event_driven_schedule(
    /*initial mask*/        topology::SwitchStateMask{n_switches},
    /*watch signals*/       {{branch_id_diode, &current_view}},
    /*on threshold cross*/  [](auto& mask, auto branch_id){
        mask.toggle(branch_id);
    });
auto result = run_transient(cache, state_size, opts,
                             auto_event_fn);
```

The loop is the same. The intelligence lives in the `switch_fn`
the user (or a V1 factory) supplies. That's the architectural
payoff of keeping V0 small.

## Validation

`pulsim_v2_layer5_tests` covers:
- **SimulationOptions** (8 cases): default invalid, normal valid,
  negative/zero dt, t_end ≤ t_start, NaN, infinity, PWM-sim step
  count.
- **SimulationResult** (3 cases): default empty, reserve doesn't
  change size, push N samples reports N steps.
- **run_transient** (6 cases): bad options/state_size/switch_fn
  throw, static circuit gives constant state, time grid is exact,
  switch_fn drives schedule, b_extra_fn is honoured.
- **Integration chopper PWM** (3 cases): 1001 samples + wall-clock
  smoke, mean v_out matches `V_dc · duty`, waveform is a clean
  square wave between analytical ON/OFF values.

Current: **2069 assertions / 21 test cases, all green** (the
chopper-PWM tests iterate over 1001 samples each, hence the
assertion count).

Total v2 surface after Layer 5: **2487 assertions / 152 test
cases**.
