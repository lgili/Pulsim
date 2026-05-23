# Design — `pulsim-v2-trapezoidal-companion`

## The companion-model math

For a 2-terminal device with constitutive relation `f(v, i, dv/dt,
di/dt) = 0`, the **trapezoidal rule** approximates time
derivatives as:

```
dv/dt ≈ (v_{n+1} − v_n) / dt          (forward difference)
        but trapezoidal evaluates at the midpoint:
        dv/dt|_{n+1/2} ≈ (v_{n+1} − v_n) / dt
```

The trapezoidal rule for `∫_{t_n}^{t_{n+1}} f(τ) dτ ≈ (dt/2)·(f_n
+ f_{n+1})` applied to the cap/inductor relations yields the
companion-model contributions below.

### Capacitor

The constitutive relation:  `i_C = C · dv_C/dt`.

Integrating with the trap rule over `[t_n, t_{n+1}]`:

```
v_{n+1} − v_n = (1 / C) · ∫_{t_n}^{t_{n+1}} i_C(τ) dτ
              ≈ (dt / 2C) · (i_n + i_{n+1})
```

Solving for `i_{n+1}`:

```
i_{n+1} = (2C/dt) · (v_{n+1} − v_n) − i_n
        = (2C/dt) · v_{n+1} − [(2C/dt) · v_n + i_n]
        ↑                       ↑
        G_eq · v_{n+1}          I_hist
```

So at step n+1, the cap behaves as a Norton equivalent:
- **Linear conductance** `G_eq = 2C/dt` (positive into v_{n+1})
- **History current source** `I_hist = G_eq · v_n + i_n`
  (flows in the same direction as the device current).

The MNA stamp is identical to a resistor with `G = G_eq`, plus a
contribution `+I_hist` to the from-node's RHS row and `−I_hist`
to the to-node's RHS row (current-source convention: positive
current flows INTO the from-node from the device).

### Inductor

Constitutive: `v_L = L · di_L/dt`. Dual derivation:

```
i_{n+1} − i_n = (1 / L) · ∫ v_L(τ) dτ
              ≈ (dt / 2L) · (v_n + v_{n+1})
```

This is naturally formulated with the inductor's current as an
extra unknown (the same way a voltage source's branch current is
an unknown). Two equivalent formulations exist:

**(a) Branch-current formulation** (used in this OpenSpec):
- Add a branch-current unknown `i_L` to the state vector.
- Add a KVL constraint row: `v_{n+1, from} − v_{n+1, to} −
  (2L/dt) · (i_{n+1} − I_hist,L) = 0` where `I_hist,L = i_n +
  (dt/2L) · v_n`.
- This keeps the stamping symmetric with VoltageSource (which
  already adds branch-current unknowns).

**(b) Norton-companion formulation**:
- `G_eq = dt/(2L)`, `V_hist = v_n + (dt/2L) · i_n`, treat the
  inductor as a Norton-equivalent admittance.
- Simpler stamping (no extra unknown), but loses an algebraic
  guarantee that |i_L| ≤ history bound.

We pick (a) because:
1. It composes naturally with the existing branch-current
   plumbing for voltage sources (DevicePool already has the
   relative-branch-id machinery).
2. The branch-current unknown `i_L` is directly inspectable in
   the output — the user gets `i_L(t)` as part of `x`.
3. Numerical stability is slightly better for stiff L circuits.

### Why trapezoidal, not backwards Euler

- **Trapezoidal is second-order accurate** vs backwards Euler's
  first-order. For PWM circuits with µs-scale dt, this matters.
- **Trapezoidal is energy-preserving for ideal LC tanks** —
  oscillations don't decay due to numerical dissipation. Backwards
  Euler has 1st-order damping, which corrupts ringing analysis.
- **Trapezoidal is the SPICE / PLECS default** — proven on
  millions of circuits. The "ringing" critique (oscillation at
  Nyquist when stepping discontinuities) is a real issue but
  matters only when post-discontinuity dt is very fine; PE
  workloads with PWM-imposed dt don't hit it.

Gear-2 / BDF can be added later if a workload demands extra
damping (e.g., for chattering-prone IGBT models). Out of scope
here.

## Cache dt-dependency

V0's cache was dt-independent: `cache.build()` factorised once,
solved many times. Now the MNA matrix contains `G_eq = 2C/dt`
entries, so:

```
J_cap(dt=1µs) ≠ J_cap(dt=2µs)
```

**The factor is invalid when dt changes.** Three design choices:

**A. Rebuild on dt change** (chosen):
- `cache.build(dt)` stores dt and factorises all segments.
- `cache.build(dt2)` with different dt invalidates + rebuilds.
- Pro: simple, predictable.
- Con: adaptive-dt must rebuild every step it changes dt
  (significant cost). V1+ optimization: lazy per-segment
  rebuild, or rank-1 updates between adjacent dt's.

**B. Cache per (mask, dt) pair**:
- Memory grows by O(num_dt_values) — bad for adaptive dt with
  continuous dt evolution.
- Not chosen for V0.

**C. Decompose J = J_static + (1/dt) · J_dynamic**:
- Mathematically clean, but the LU factor of a sum is NOT the
  sum of factors — can't avoid refactorisation.
- Not chosen.

V0 picks **A**, fixed dt throughout the simulation. Adaptive dt
in V1+ will need a smarter strategy — likely lazy per-segment
rebuild when an adaptive controller halves dt.

## History-state plumbing

`HistoryState` is owned by the `run_transient` loop. It maintains
one (v_prev, i_prev) entry per dynamic branch (capacitor or
inductor):

```cpp
struct HistoryEntry {
    Index branch_id;
    StoredKind kind;     // Capacitor or Inductor
    Real v_prev = 0;     // v_{n} (across the device)
    Real i_prev = 0;     // i_{n} (through the device)
};

class HistoryState {
public:
    HistoryState(const Graph&, const DevicePool&);  // builds entries
                                                     // from pool's
                                                     // dynamic devices

    /// Returns the b_extra contribution at step n+1, given the
    /// previous-step state vector and current dt.
    Vector compute_b_extra(const Vector& x_prev, Real dt) const;

    /// Updates entries from x_prev (called after each cache.solve).
    void update_from_state(const Vector& x_prev);

private:
    std::vector<HistoryEntry> entries_;
    // (graph + pool stored for lookups during compute_b_extra)
};
```

The Layer 5 loop becomes:

```cpp
HistoryState history(graph, pool);  // all zeros initially
Vector x = Vector::Zero(state_size);

for k = 0 .. N - 1:
    t = t_start + k * dt

    // 1. History from previous step
    Vector b_extra = b_extra_fn ? b_extra_fn(t) : 0;
    b_extra += history.compute_b_extra(x, opts.dt);

    // 2. Switch state
    auto mask = switch_fn(t);

    // 3. Cached solve
    cache.solve(mask, b_extra, x);

    // 4. Update history for next step
    history.update_from_state(x);

    // 5. Record
    result.times.push_back(t);
    result.states.push_back(x);
```

`HistoryState` knows about the graph (for branch endpoints, to
read v across each device) and the pool (for C / L values).

## Why all-zero initial conditions

For V0, every capacitor starts at v_C(0) = 0 and every inductor
at i_L(0) = 0. This is correct for:
- **Power-on transients** — the natural "circuit starts dead and
  is powered up" model.
- **Most academic examples** — RC charge curve, RL turn-on, RLC
  ring-down from rest.

For **steady-state initialisation** (the PSIM "DC analysis"
button), the user has to either:
- Manually charge the capacitors via a transient long enough for
  the natural response to decay.
- Wait for the `pulsim-v2-dc-operating-point` follow-up which
  will compute the DC steady state analytically and seed
  HistoryState from it.

V0 doesn't ship DC OP because the math is non-trivial for
switched circuits (the steady state is a limit cycle, not a
point). That's a real research / engineering effort and deserves
its own OpenSpec.

## Static-only backwards compatibility

The new `cache.build(dt)` overload is additive. The old
`cache.build()` (no-arg) is kept and:
- Sets internal dt to 0 (a sentinel).
- The assemble loop **skips** Capacitor / Inductor branches
  entirely (logs nothing, just doesn't stamp them).
- Behaves identically to V0 for static-only circuits.

This means every Layer 4 V0 test continues to pass without any
edit. The chopper-PWM Layer 5 V0 integration test also passes
unchanged (the chopper has no caps / inductors).

The transition path for static-only callers that want to add a
capacitor is one line: call `cache.build(dt)` instead of
`cache.build()`.

## Test strategy

The companion-stamp math has well-known analytical solutions for
RC, RL, and RLC. V0 tests use those as ground truth:

### RC charge (V_C from 0 to V_dc through R)
- `V_C(t) = V_dc · (1 − e^{−t/τ})`, `τ = RC`
- Simulate at `dt = τ/100`, over `t ∈ [0, 5τ]` (5 time
  constants = 99.3 % of final value).
- Tolerance: < 1 % at every sample (trap is 2nd-order, dt=τ/100
  gives ~10^-4 LTE).

### RL ramp (I_L from 0 to V/R through R)
- `I_L(t) = (V/R) · (1 − e^{−t/τ})`, `τ = L/R`
- Same validation pattern. Inductor branch-current unknown is
  directly readable from `x`.

### RLC ring-down (V_C from V0 to 0 with damped oscillation)
- Underdamped case `ζ < 1`: `V_C(t) = V_0 · e^{−ζω_n t} · cos(ω_d t + φ)`
- `ω_n = 1/√(LC)`, `ω_d = ω_n · √(1 − ζ²)`, `ζ = (R/2)·√(C/L)`
- Verify zero-crossings of V_C(t) occur at the analytical times
  within < 1 ULP at trapezoidal precision.
- Edge case: undamped (R = 0) — period MUST be exactly 2π√(LC).

### Static-circuit regression
- Re-run all Layer 4 V0 tests with the new cache.hpp.
- All Layer 5 V0 tests, including the 10 kHz PWM chopper.
- Expect identical numbers (V0 cache.build() shim must be
  bit-identical to the original implementation).

## Risks

1. **Sign convention drift**. The history-term sign depends on
   the device's terminal labelling AND the MNA's convention for
   "current flowing into the from-node". The test `test_capacitor`
   has 4 cases (cap between two non-ground nodes / between v_n
   and ground / from terminal swap / inductor variant) that lock
   this in.

2. **Index plumbing for inductor branch-current**. Like voltage
   sources, inductors need a relative branch-id offset. The
   DevicePool already has machinery for sources; adding inductors
   reuses the same `num_branch_current_unknowns` counter. Care
   needed in `state_size` computation — V1 vector grows.

3. **Numerical artifacts with very small dt or very stiff L/C**.
   `G_eq = 2C/dt` blows up at small dt for small C; the LU factor
   can lose accuracy. We don't ship a heuristic in V0 — if users
   hit this, the work-around is dt scaling or iterative refinement
   (which Layer 0's DirectSolver already supports).

4. **Caching scaling at N > 8 switches with dynamic devices**.
   Cap/L history is per-device, not per-segment, so this doesn't
   add to the 2^N memory cost. But every segment now has the
   same shape (matrix dim doesn't change), so cache size is
   unchanged — risk is just rebuild time when dt changes.

5. **dt-change in mid-simulation** is NOT a V0 use case — the
   loop assumes dt is fixed. If a future Layer 5 V1.5 adds
   adaptive dt, it'll need to call `cache.build(new_dt)` between
   steps. That's a significant cost; the implementation can
   detect and short-circuit when new_dt == current_dt.

## Performance expectations

- **Cache build**: one factorisation per segment per dt change.
  Same cost as Layer 4 V0 build.
- **Per-step solve**: one map lookup + one triangular solve +
  one history-vector computation (O(num_dynamic_branches)). The
  history computation is O(D) where D is the number of
  caps/inductors — typically D < 20 for PE circuits, so this is
  a few-dozen-µs overhead per step at most.
- **Memory**: HistoryState size O(D · 2 doubles) ≈ 320 bytes for
  20-device circuit. Negligible.

For the RC reference test (one cap, dt = 100 ns, 5000 steps), we
expect simulation wall-clock < 50 ms on the test host.

## API surface summary

```cpp
namespace pulsim::v2::models {

struct Capacitor {
    struct Params { Real C; };  // farads
    static constexpr BranchKind kind = BranchKind::PassiveLinear;
    static constexpr Size num_terminals = 2;
    static constexpr bool is_dynamic = true;

    /// Returns (g_eq, _) for the static stamp.
    static Real g_eq(Real dt, const Params& p);

    /// Returns the history-term scalar: g_eq · v_prev + i_prev.
    static Real history_term(Real v_prev, Real i_prev,
                              Real dt, const Params& p);
};

struct Inductor {
    struct Params { Real L; };  // henries
    static constexpr BranchKind kind = BranchKind::PassiveLinear;
    static constexpr Size num_terminals = 2;
    static constexpr bool is_dynamic = true;

    /// Inductor uses a branch-current unknown (extra row),
    /// analogous to VoltageSource. g_eq goes on the constraint
    /// row (the 2L/dt term in v_n+1 − v_n = (2L/dt)(i_n+1 −
    /// I_hist,L)).
    static Real g_eq_inv(Real dt, const Params& p);  // dt/(2L)

    static Real history_term(Real v_prev, Real i_prev,
                              Real dt, const Params& p);
};

}  // namespace pulsim::v2::models

namespace pulsim::v2::pwl {

class DevicePool {
public:
    // ... existing methods ...
    void add_capacitor(Index branch_id, models::Capacitor::Params);
    void add_inductor(Index branch_id, models::Inductor::Params);

    // Layout helpers extended: state_size grows by 1 per inductor
    // (the branch-current unknown).
    Index branch_var_id_for_inductor(Index branch_id,
                                      const Graph&) const;
    Size  num_dynamic_branches() const;  // C + L total
};

class PwlStateSpaceCache {
public:
    // V0 static-only shim (deprecated for circuits with C/L).
    void build();

    // V1 dt-aware build.
    void build(Real dt);

    Real dt() const noexcept;

    // solve unchanged.
};

class HistoryState {
public:
    HistoryState(const Graph&, const DevicePool&);
    Vector compute_b_extra(const Vector& x_prev, Real dt) const;
    void   update_from_state(const Vector& x_prev);
    void   reset();   // zeros all entries
};

}  // namespace pulsim::v2::pwl

namespace pulsim::v2::solver {

// run_transient signature unchanged — HistoryState is built
// internally from the cache's graph + pool.
SimulationResult run_transient(
    const pwl::PwlStateSpaceCache& cache,
    Size state_size,
    const SimulationOptions& opts,
    const SwitchScheduleFn& switch_fn,
    const BExtraFn& b_extra_fn = {});

}  // namespace pulsim::v2::solver
```

## What V1 hands to follow-ups

- **DC operating-point pre-charge**: `pulsim-v2-dc-operating-point`.
  Compute steady-state V_C and I_L given a fixed switch state,
  seed HistoryState from it.
- **Adaptive dt + LTE estimation**: `pulsim-v2-adaptive-dt`.
  Cache invalidates on dt change.
- **Event detection**: `pulsim-v2-event-detection`. Auto
  zero-crossing on watch-signals to replace user switch_fn.
- **Newton for nonlinear devices**:
  `pulsim-v2-nonlinear-segment-newton`. Per-segment Newton on top
  of the cached factor.
- **Sherman-Morrison rank-1 update** for Gray-code-adjacent
  segments. Cuts build cost roughly in half at large N.
