# Layer 5 V2 — Ideal-diode auto-commutation

After Layer 5 V1.5 proved the v2 kernel on a synchronous buck
(both switches controlled by user PWM), Layer 5 V2 adds the
**ideal-diode auto-commutation** feature. The kernel now handles
diodes that switch ON/OFF based on circuit conditions — exactly
what real rectifiers, asynchronous buck/boost converters, body
diodes, and freewheeling diodes do.

**Result: half-wave rectifier integration test passes — all 16
test cases / 35 assertions green.** The boost converter
integration is intentionally deferred to the follow-up
`pulsim-v2-event-detection` OpenSpec (per-step state decisions
chatter at the DCM/CCM boundary; sub-step bisection is the right
fix).

## The diode state machine

```
        ┌────── v_anode − v_cathode ≥ V_th ──────┐
        │                                         │
        ▼                                         │
    ┌──────┐                                   ┌──────┐
    │ OFF  │                                   │ ON   │
    │ g_off│                                   │ g_on │
    └──────┘                                   └──────┘
        ▲                                         │
        │                                         │
        └───────────── i_diode ≤ 0 ───────────────┘
```

For an ideal diode `V_th = 0`. For a Si-behavioral diode
`V_th = 0.7`. Both are supported.

At the topology level, the diode IS a Switch — same combinatorial
axis in the Layer 4 cache (an N-switch graph with M diodes has
2^N segments). The new piece is the **per-step state decision
logic** that decides each diode's ON/OFF bit based on the
just-computed `(v_diode, i_diode)`.

## API

```cpp
namespace pulsim::v2::models {

struct SwitchedDiode {
    struct Params {
        Real g_on;   // forward conductance (typ. 1e3)
        Real g_off;  // reverse conductance (typ. 1e-9)
        Real V_th;   // forward threshold (0 for ideal, 0.7 for Si)
    };

    // Topology: looks like a switch.
    static constexpr topology::BranchKind kind =
        topology::BranchKind::Switch;
    static constexpr Size num_terminals = 2;
    static constexpr bool is_dynamic = false;

    /// Decide the NEXT step's state based on the current step's
    /// state + (v_diode, i_diode) computed from the just-solved x.
    static bool decide_next_state(bool currently_on, Real v_diode,
                                    Real i_diode, const Params& p);
};

}  // namespace pulsim::v2::models

namespace pulsim::v2::pwl {

class DevicePool {
public:
    void add_diode(Index branch_id, Real g_on, Real g_off,
                    Real V_th = 0.0);
    const models::SwitchedDiode::Params& diode_params(Index) const;
    std::span<const Index> diode_branches() const;
    Size num_diodes() const noexcept;
};

class DiodeEventState {
public:
    DiodeEventState(const Graph&, const DevicePool&);
    topology::SwitchStateMask current_diode_mask() const;
    topology::SwitchStateMask diode_owned_bits() const;
    bool update_from_state(const Vector& x);
    void reset() noexcept;
};

}  // namespace pulsim::v2::pwl
```

`run_transient` is unchanged signature-wise. It internally
constructs a `DiodeEventState` (empty if no diodes), combines
the user's `switch_fn(t)` output with the diode's auto-mask each
step, and updates the diode state after each solve.

## The Layer 5 V2 loop (dynamic path)

```cpp
HistoryState history(graph, pool);      // V1 — cap/inductor
DiodeEventState diodes(graph, pool);    // V2 — diodes
const auto diode_owned = diodes.diode_owned_bits();
const bool has_diodes = diodes.num_diodes() > 0;

result.push((t_start, x = 0));

for (k = 1; k < N; ++k):
    t = t_start + k · dt

    // 1. History from previous step
    b_extra = history.compute_b_extra(dt)
    if (b_extra_fn) b_extra += b_extra_fn(t)

    // 2. Combine user switches + diode auto-state
    user_mask = switch_fn(t)
    if has_diodes:
        diode_mask = diodes.current_diode_mask()
        combined = combine_masks(user_mask, diode_mask, diode_owned)
    else:
        combined = user_mask

    // 3. Cached solve
    cache.solve(combined, b_extra, x)

    // 4. Update history + diodes for next step
    history.update_from_state(x, dt)
    if has_diodes:
        diodes.update_from_state(x)

    // 5. Record
    result.push((t, x))
```

## Worked example: half-wave rectifier

```
V_sine(10 V · sin(2π·60·t)) ──[Diode]── R(10 Ω) ── GND
```

```cpp
Graph g;
auto n0 = g.add_node("n0");
auto n1 = g.add_node("n1");
g.add_branch(n0, g.ground(), BranchKind::Source);
g.add_branch(n0, n1,         BranchKind::Switch);   // diode IS a switch
g.add_branch(n1, g.ground(), BranchKind::PassiveLinear);

DevicePool pool;
pool.add_voltage_source(0, {.V = 0});  // sinusoidal via b_extra_fn
pool.add_diode(1, /*g_on=*/1e3, /*g_off=*/1e-9, /*V_th=*/0);
pool.add_resistor(2, {.G = 0.1});

PwlStateSpaceCache cache(g, pool);
cache.build(/*dt=*/10e-6);

solver::SimulationOptions opts{
    .t_start = 0, .t_end = 1./30,  // 2 cycles
    .dt = 10e-6
};

const Index src_var = pool.branch_var_id_for_source(0, g);
const auto b_extra_fn = [src_var, state_size = pool.state_size(g)](Real t) {
    Vector b = Vector::Zero(state_size);
    const Real V_sine = 10 * std::sin(2 * M_PI * 60 * t);
    b[src_var] = -V_sine;
    return b;
};

const auto schedule = [](Real){ return SwitchStateMask(1); };

auto result = run_transient(cache, g, pool, opts, schedule, b_extra_fn);

// During positive half-cycles: result.states[k][n1] tracks V_sine.
// During negative half-cycles: result.states[k][n1] ≈ 0.
// Mean output power: V_amp² / (4·R) = 2.5 W within 5 %.
```

## V0 limitations

### Per-step (not sub-step) state decisions

The diode flips its state at MOST ONCE per dt. If the actual
commutation should happen partway through a step (e.g., at
t = t_n + 0.3·dt), V0 will catch it at step n+1 — a 0.7·dt
delay. For typical PE simulations at dt = T_sw/100, that's < 1 %
of a PWM period.

For most workloads this is invisible. For boost converters
operating at the DCM/CCM boundary — where the diode commutates
EXACTLY when i_L hits zero — the V0 logic can chatter because
the per-step solve can't accurately resolve i_L = 0 (the L
pumps voltage to enormous values during the DCM dead-time).

The fix is sub-step bisection (find the exact commutation time
within a step, take a partial step to that time, switch, then
continue). That lands in `pulsim-v2-event-detection`.

### Half-wave rectifier works because the commutation is "slow"

The sinusoidal source's zero-crossing happens over many dt's
(at f = 60 Hz, dt = 10 µs, a half cycle is 833 samples). The
diode's per-step decision catches the commutation comfortably
within one or two samples of the analytical zero-crossing
time. Mean output power matches the analytical formula to
< 0.5 %.

### Initial diode state: all OFF

V0 starts every diode OFF at t = 0. If a diode SHOULD be ON at
t = 0+ (e.g., a half-wave rectifier with V_sine starting at the
peak), the first step's solve uses OFF; the next step's
update_from_state catches the forward bias and flips to ON.
Worst-case latency: 1 dt. Documented limitation.

DC operating-point pre-charge — which would compute consistent
diode states at t = 0 — is the `pulsim-v2-dc-operating-point`
follow-up.

## What V2 hands to follow-ups

- `pulsim-v2-event-detection` — sub-step bisection to find the
  exact diode commutation time, fixing the DCM/CCM boundary
  chatter that breaks the boost converter test.
- `pulsim-v2-nonlinear-segment-newton` — real diode I-V curves
  (exponential model), reverse recovery, junction capacitance.
- `pulsim-v2-dc-operating-point` — consistent diode-state IC,
  eliminates the V0 1-dt latency at t = 0.

## Status of the layered v2 surface

| Layer    | Cases  | Assertions | New?  |
|----------|--------|------------|-------|
| 0        | 19     | 80         |       |
| 1        | 36     | 126        |       |
| 2        | 36     | 93         |       |
| 3        | 16     | 61         |       |
| 4 V0     | 24     | 58         |       |
| 5 V0     | 21     | 2069       |       |
| 4 V1     | 32     | 76         |       |
| 5 V1.5   | 17     | 59         |       |
| **5 V2** | **16** | **35**     | **NEW** |
| **Total**| **217**| **2657**   |       |

v1 regression: `pulsim_tests` 304 cases / 4214 assertions — all
green.
