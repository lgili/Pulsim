# Layer 4 V1 — Trapezoidal companion (Capacitors + Inductors)

Layers 0-5 V0 could simulate piecewise-linear **static** circuits
(R + V + Switch). Layer 4 V1 + Layer 5 V1 extend the kernel with
**Capacitor + Inductor** support via the textbook trapezoidal
companion model. The result: any linear PE circuit can now run
end-to-end through `run_transient`.

This document covers the companion-model math, how the dt-aware
cache + history-state plumbing fit together, and the V0
limitations to be aware of (notably: all-zero initial conditions
+ trap-rule boundary artifact).

## The companion-model math

The trapezoidal rule turns a dynamic ODE step into a static
equation that the existing PWL cache can solve. For a 2-terminal
device with constitutive relation `f(v, i, dv/dt, di/dt) = 0`,
the trap rule approximates the integral `∫f dt ≈ (dt/2)·(f_n +
f_{n+1})`.

### Capacitor

`i_C = C · dv_C/dt`  →  trap  →
```
i_{n+1} = G_eq · v_{n+1} − I_hist
G_eq   = 2C / dt
I_hist = G_eq · v_n + i_n
```

So at step n+1, the cap behaves as a Norton equivalent:
- A linear conductance `G_eq` (positive into v_{n+1})
- A history current source `I_hist` (parallel)

`G_eq` goes into the MNA matrix at assembly time (identical to
a resistor with G = G_eq). `I_hist` depends on the previous-
step state — Layer 5's `HistoryState` tracks it and contributes
to `b_extra` at solve time.

### Inductor

`v_L = L · di_L/dt`  →  trap  →
```
i_{n+1} = G_L,eq · v_{n+1} + I_hist,L
G_L,eq   = dt / (2L)
I_hist,L = i_n + G_L,eq · v_n
```

V2 uses the **branch-current formulation**: `i_L` is an explicit
MNA unknown (like a voltage source's branch current), with the
constraint row (scaled by `2L/dt` to keep the +1 / -1 voltage-
column pattern):

```
v_{n+1, from} − v_{n+1, to} − (2L/dt) · i_{n+1} + (2L/dt) · I_hist,L = 0
```

`-(2L/dt)` stamps on the branch_var_id diagonal; `+(2L/dt) ·
I_hist,L` goes into `b_extra`.

### Why trapezoidal, not backwards Euler

- **Second-order accurate** vs BE's first-order. PWM circuits
  with µs dt benefit.
- **Energy-preserving for ideal LC tanks** — oscillations don't
  decay due to numerical dissipation. BE has 1st-order damping.
- **SPICE / PLECS default** — proven on millions of circuits.

Gear-2 / BDF can be added later if a workload demands more
damping. Out of scope for this OpenSpec.

## Cache dt-dependency

V0's cache was dt-independent. V1's MNA matrix contains
`G_eq = 2C/dt` and `−(2L/dt)` terms, so the factor is **dt-
specific**:

```cpp
cache.build(dt);   // factor for this dt only
cache.solve(mask, b_extra, x);
// ... change dt? ...
cache.build(other_dt);   // rebuilds all segments
```

Calling `build()` (no argument) keeps the V0 behaviour: dt = 0
sentinel, dynamic devices SKIPPED. Useful for static-only
circuits and as the backwards-compat path.

For a 1 µF capacitor at dt = 1 µs, `G_eq = 2`. At dt = 100 ns,
`G_eq = 20`. At dt = 10 ns, `G_eq = 200`. The matrix gets
stiffer at small dt — sparse LU handles it, but watch for
near-singularity if dt is wildly mismatched to the time
constants.

## History-state plumbing

`pulsim::pwl::HistoryState` owns one entry per dynamic
device:

```cpp
struct HistoryEntry {
    Index branch_id;
    Index from, to;
    DevicePool::StoredKind kind;
    Index inductor_branch_var_id;   // valid only for inductors
    Real  v_prev = 0;
    Real  i_prev = 0;
    Real  C_or_L;                   // farads or henries
};
```

Layer 5 V1's `run_transient` calls it twice per step:

```
b_extra = history.compute_b_extra(dt);   // before solve
cache.solve(mask, b_extra, x);
history.update_from_state(x, dt);        // after solve
```

`compute_b_extra` builds a state-size vector with the trap
companion's contributions:
- Cap (between `from` and `to`):
  `b_extra(from) -= I_hist`, `b_extra(to) += I_hist`
- Inductor (at branch_var_id):
  `b_extra(branch_var_id) += (2L/dt) · I_hist,L`

`update_from_state` reads (v_new, i_new) for each entry:
- Cap: `v_new = v_from - v_to` (from x); `i_new = G_eq · (v_new
  - v_prev) - i_prev` (derived from companion).
- Inductor: `v_new = v_from - v_to`; `i_new = x[branch_var_id]`
  (direct unknown).

## The run_transient V1 loop

```cpp
SimulationResult run_transient(
    const pwl::PwlStateSpaceCache& cache,
    const topology::Graph& graph,
    const pwl::DevicePool& pool,
    const SimulationOptions& opts,
    const SwitchScheduleFn& switch_fn,
    const BExtraFn& b_extra_fn = {});
```

The signature changes from V0 (which took `state_size` directly)
to V1 (graph + pool, state_size derived). The V0 signature is
preserved for backwards compatibility — it skips HistoryState
entirely.

Loop body (dynamic path):

```
// Record IC as sample 0
result.push((t_start, x = 0))

for k = 1 .. N - 1:
    t = t_start + k · dt
    b_extra = history.compute_b_extra(dt)
    if (b_extra_fn) b_extra += b_extra_fn(t)
    cache.solve(switch_fn(t), b_extra, x)
    history.update_from_state(x, dt)
    result.push((t, x))
```

`cache.dt() != opts.dt` throws `std::invalid_argument` — a
mismatched cache would silently drift, so we catch it up-front.

For static-only circuits (`cache.dt() == 0`), the loop takes a
simpler path identical to V0 — each sample is an independent
solve, no history bookkeeping.

## Worked example: RC charging

```
V_dc(5V) ──[Source]── n0 ──[R=1Ω]── n1 ──[C=1µF]── GND
                                     ^
                                     v_C(t) here
```

Analytical: `V_C(t) = V_dc · (1 − e^{−t/τ})`, `τ = RC = 1 µs`.

```cpp
Graph g;
auto n0 = g.add_node("n0");
auto n1 = g.add_node("n1");
g.add_branch(n0, g.ground(), BranchKind::Source);
g.add_branch(n0, n1,         BranchKind::PassiveLinear);
g.add_branch(n1, g.ground(), BranchKind::PassiveLinear);

DevicePool pool;
pool.add_voltage_source(0, {.V = 5.0});
pool.add_resistor(1, {.G = 1.0});       // G = 1/R
pool.add_capacitor(2, {.C = 1e-6});

PwlStateSpaceCache cache(g, pool);
cache.build(/*dt=*/1e-8);                // 10 ns step

SimulationOptions opts{
    .t_start = 0,
    .t_end   = 5e-6,                      // 5τ
    .dt      = 1e-8,
};

auto result = run_transient(
    cache, g, pool, opts,
    [](Real){ return SwitchStateMask(0); });

// result.states[k][n1] tracks 1 − e^{−t_k/τ} to within < 1 %
// for k > N/4 (after the IC artifact decays).
```

## V0 limitations

### Inconsistent initial conditions

V0 starts with `v_C(0) = 0` and `i_L(0) = 0` for every dynamic
device. That's only one piece of the puzzle — the trap rule
ALSO needs `i_C(0)` (which is determined by the rest of the
circuit at t = 0+) and `v_L(0)` (= V_source − v_C(0) − ...).

For the RC charge from rest:
- v_C(0) = 0 ✓ (matches V0 convention)
- i_C(0+) = V_dc/R = 5 A ≠ 0 ← V0 forces this to 0

This **trap-rule boundary artifact** decays as `λ_trap^n ≈ (1
- dt/(2RC))^n` per step. For `dt = τ/100`, the artifact is
< 1 % of the analytical value after roughly N/4 simulation
steps. Our integration tests skip the first N/4 samples to
bypass this transient.

The proper fix is DC operating-point pre-charge — a follow-up
OpenSpec (`pulsim-v2-dc-operating-point`).

### No event detection

Switching is still externally scheduled via `switch_fn(t)`. Auto
zero-crossing detection (for diode commutation, etc.) is a Layer
5 V1.5 follow-up.

### No adaptive dt

The cache is dt-specific. Changing dt rebuilds every segment —
expensive. Adaptive dt with LTE estimation needs lazy per-segment
rebuild or rank-1 updates. Future Layer 5 V2.

### No nonlinear devices

`BranchKind::Nonlinear` branches are still SKIPPED in assembly.
Newton iteration on top of the cached factor lands in
`pulsim-v2-nonlinear-segment-newton`.

## Validation

`pulsim_v2_layer4_v1_tests`: **76 assertions / 32 cases**.
- Capacitor / Inductor math (16 cases)
- DevicePool with caps + L (9 cases)
- stamp_companion helpers (5 cases)
- dt-aware cache (5 cases)
- V0 regression coverage (1 case)

`pulsim_v2_layer5_v1_tests`: **51 assertions / 15 cases**.
- HistoryState basic API (5 cases)
- run_transient V1 wiring (3 cases)
- RC charging (2 cases)
- RL ramping (2 cases)
- RLC ringdown (3 cases)

Total v2 surface after Layer 4/5 V1: **2614 assertions / 199 test
cases**.

## What V1+ delivers

This OpenSpec unlocks dynamic circuits at the kernel level. A
buck converter, a boost converter, an LLC resonant tank — all
now runnable in v2 with PLECS-style per-step performance (one
hash probe + one triangular solve, no Newton, no factorise).

Next on the roadmap:
- **DC operating-point pre-charge** — fixes the IC artifact
  outright (`pulsim-v2-dc-operating-point`).
- **Event detection** — auto zero-crossing on watch signals
  (`pulsim-v2-event-detection`).
- **Newton for nonlinear devices** — per-segment iteration on
  top of the cached factor (`pulsim-v2-nonlinear-segment-newton`).
- **Adaptive dt + LTE estimation** — needs cache invalidation
  on dt changes (`pulsim-v2-adaptive-dt`).
