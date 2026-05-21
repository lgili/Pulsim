# Design — `pulsim-v2-ideal-diode-auto-commutation`

## The diode state machine

An ideal diode has two states:

```
        ┌─────────────────── v_anode − v_cathode ≥ V_th ─────────────┐
        │                                                             │
        ▼                                                             │
    ┌──────┐                                                       ┌──────┐
    │ OFF  │                                                       │ ON   │
    │      │                                                       │      │
    │ g_off│                                                       │ g_on │
    └──────┘                                                       └──────┘
        ▲                                                             │
        │                                                             │
        └─────────────────────── i_diode ≤ 0 ─────────────────────────┘
```

For a perfectly ideal diode `V_th = 0`. For a Si-behavioural
ideal diode `V_th = 0.7 V`. Layer 5 V2 supports both.

Within the v2 cache architecture, a diode IS a switch — its
two states are two of the 2^N combinatorial bits. The
ON / OFF stamps are identical to `IdealSwitch`: `g_on` between
the two terminals when ON, `g_off` when OFF.

What's new is the **state-decision logic**: at the end of each
step, look at `v_diode` and `i_diode` from the just-computed
state vector `x`, decide the next step's diode bit.

## Per-step decision, not sub-step

Modern simulators (PSIM, PLECS) do **sub-step event detection**:
when a diode's state should flip mid-step, they bisect to find
the exact time, step back to that point, switch, and continue.
This produces zero-error commutation timing.

V0 of this OpenSpec takes a simpler approach: **per-step
decisions**. After each cache.solve, we look at `(v_diode,
i_diode)` and decide whether to flip the diode bit for the NEXT
step. Worst-case timing error: one dt.

Trade-offs:
- **Pro**: simple. No bisection, no step-back / step-forward
  machinery. The loop stays clean.
- **Pro**: matches PLECS's default behaviour ("ideal switching"
  mode with PWM-resolution dt).
- **Con**: < dt of timing error per commutation. For typical PE
  workloads at dt = T_sw/100, that's < 1 % of the PWM period —
  invisible at the converter output.
- **Con**: in pathological cases (very slow dynamics, very fast
  diode transitions) it could miss a commutation entirely. We
  document this limitation; the user can pick a smaller dt.

Sub-step bisection is a Layer 5 V2.5 follow-up
(`pulsim-v2-event-detection`).

## State-decision logic

After each `cache.solve` produces `x`:

```cpp
for each diode in pool.diodes():
    v_a = x[diode.anode_node]       (or 0 if anode == GND)
    v_k = x[diode.cathode_node]     (or 0 if cathode == GND)
    v_diode = v_a − v_k

    // i_diode is harder to extract: it's NOT a state-vector
    // unknown unless the diode is in series with an inductor.
    // For the static ON case, we compute it from Ohm's law on
    // the diode's conductance:
    //   i_diode = g_on · v_diode    (ON state)
    //   i_diode = g_off · v_diode   (OFF state — should be tiny)
    g = diode.is_on ? g_on : g_off
    i_diode = g * v_diode

    next_state = diode.is_on
    if (diode.is_on && i_diode <= 0)        next_state = OFF
    if (!diode.is_on && v_diode >= V_th)    next_state = ON

    diode.is_on = next_state
```

The combined mask for the NEXT step is:
- User-supplied switch bits from `switch_fn(t_next)`
- Diode bits from the just-updated `DiodeEventState`

## Why this avoids commutation cycling

A naive implementation could oscillate: at step n the diode is
OFF, gets turned ON by `v_diode > 0`. At step n+1 with the diode
ON, `i_diode < 0` (because the trap rule hadn't seen the new
state yet) → turn OFF. Repeat forever.

Two safeguards:
1. **Decide for NEXT step, not current**. The current step's
   solve uses the diode state determined at the END of the
   previous step. So between consecutive steps the diode state
   changes at most once (no oscillation within a step).
2. **Use the post-solve `(v, i)` to decide**. With the trap
   rule's energy-preserving property, the just-computed state
   represents the actual circuit response. If `i_diode ≤ 0` at
   the just-computed state, the diode physically wants to
   commutate — flipping it is correct.

The boost converter test is the natural verification: the
diode commutates once per PWM cycle (when the switch turns
off), and the cycle should be ~100 commutations / ms with no
chatter.

## State-vector layout (unchanged)

A diode does NOT add an MNA branch-current unknown (it's a
2-terminal "resistor-like" element in either state). So
`pool.state_size(graph)` is unchanged by adding diodes.

The only state that exists for the diode is the binary `is_on`
bit, which is part of the `SwitchStateMask` (and indirectly the
cache key).

## DevicePool API

```cpp
class DevicePool {
public:
    // ... existing methods ...

    /// Register an ideal diode on `branch_id`.
    /// The branch MUST have been added to the graph with
    /// `BranchKind::Switch` (the diode IS a switch from the
    /// topology's perspective).
    ///
    /// `anode` is the from-terminal, `cathode` is the to-
    /// terminal. The diode conducts when v_anode > v_cathode.
    ///
    /// Default V_th = 0 (perfectly ideal). Pass a non-zero
    /// value for a Si-behavioral diode.
    void add_diode(Index branch_id,
                    Real g_on,
                    Real g_off,
                    Real V_th = Real{0});

    /// Returns the list of branch ids registered as diodes
    /// (in branch order). Used by DiodeEventState construction.
    std::span<const Index> diode_branches() const;
};
```

Internally the diode stores in the same `Entry` variant as
the other devices (extending `StoredKind` with a new value).
The runtime check in `assemble_segment` still treats it as a
`Switch` (using its g_on / g_off).

## DiodeEventState

```cpp
class DiodeEventState {
public:
    DiodeEventState(const Graph& graph, const DevicePool& pool);

    /// Returns the mask of diode bits in the current state. The
    /// mask is `pool.num_switches()` wide — diode bits set per
    /// the tracker's state, non-diode bits left at 0 (caller
    /// combines with user's switch_fn output).
    [[nodiscard]] SwitchStateMask current_diode_mask() const;

    /// Returns the bits that diodes ALSO claim (so the caller
    /// can detect bit overlap with user-controlled switches if
    /// any — that's a configuration error).
    [[nodiscard]] SwitchStateMask diode_owned_bits() const;

    /// Re-decide each diode's state based on the just-computed
    /// state vector `x`. Returns true if any diode flipped (so
    /// the caller knows the next step's cache.solve uses a
    /// different segment).
    bool update_from_state(const Vector& x);

    /// Reset all diodes to OFF (V0 initial state).
    void reset() noexcept;

private:
    struct DiodeEntry {
        Index branch_id;
        Index switch_idx;      // bit position in mask
        Index from, to;
        Real g_on, g_off, V_th;
        bool is_on = false;
    };

    const Graph& graph_;
    std::vector<DiodeEntry> diodes_;
    Size num_switches_;
};
```

## Layer 5 V2 run_transient overload

```cpp
SimulationResult run_transient(
    const pwl::PwlStateSpaceCache& cache,
    const topology::Graph& graph,
    const pwl::DevicePool& pool,
    const SimulationOptions& opts,
    const SwitchScheduleFn& switch_fn,
    const BExtraFn& b_extra_fn = {});
```

The signature is **the same as Layer 5 V1**. The diode
awareness is hidden inside the implementation — if the pool
has no diodes, behaviour is identical to V1. If it does, the
loop internally maintains a DiodeEventState.

Loop body (dynamic path with diodes):

```cpp
HistoryState history(graph, pool);    // V1
DiodeEventState diodes(graph, pool);  // V2 — empty if no diodes
diodes.reset();

result.times.push_back(t_start);
result.states.push_back(x);

for k = 1 .. N - 1:
    t = t_start + k * dt
    b_extra = history.compute_b_extra(dt) + (b_extra_fn ? ... : 0)

    // Combine user mask + diode mask
    auto user_mask = switch_fn(t);
    auto diode_mask = diodes.current_diode_mask();
    auto combined_mask = combine_masks(user_mask, diode_mask,
                                         diodes.diode_owned_bits());

    cache.solve(combined_mask, b_extra, x)
    history.update_from_state(x, dt)
    diodes.update_from_state(x)         // V2

    record(t, x)
```

`combine_masks` replaces the user's bits at diode-owned
positions with the diode's own bits.

## Why V0 starts every diode OFF

Two reasons:
1. **Simplicity**. No need for a DC operating point at t=0.
2. **Conservative**. An OFF diode at t=0 is the
   "uncharged-rectifier" power-on state — exactly what real
   hardware does.

If the diode SHOULD be ON at t=0+ (e.g., a freewheeling diode
in a buck with an already-charged inductor), the first step
will detect `v_diode > 0` after solve and flip it for step 2.
Worst-case latency: 1 dt.

A future `pulsim-v2-dc-operating-point` OpenSpec will compute
consistent diode states at t=0 alongside cap voltages and
inductor currents.

## Test strategy

### Half-wave rectifier (the minimum-viable diode test)

```
V_sine(10V·sin(2πf t)) ──[Diode]── R(10Ω) ── GND
```

Expected output: V_out tracks V_sine when V_sine > 0, V_out ≈ 0
when V_sine < 0. Over a full cycle, the average output power
= V_amp² / (4·R) = 100/40 = 2.5 W.

Validation:
- Sample V_out during a positive half: V_out within 0.5 V of
  V_sine.
- Sample V_out during a negative half: |V_out| < 1 V (diode OFF
  has tiny g_off → tiny but nonzero current).
- Mean power over a full cycle within 5 % of analytical.

`V_sine(t)` enters via `b_extra_fn` — we modulate the source's
constraint-row b_extra by the sine wave.

### Boost converter (the real PE workload)

```
   V_in(12V) ──[L(100µH)]── v_sw ──┬──[Q(switch)]── GND
                                    │
                                    └──[D(diode)]── V_out ──┬─[C(100µF)]── GND
                                                              │
                                                              └─[R_load(20Ω)]── GND
```

PWM controls Q at 100 kHz, D_PWM = 0.5 → V_out = V_in / (1 - D_PWM)
= 24 V.

During Q ON: v_sw = 0, L charges (V_L = V_in, di_L/dt > 0).
              Diode is reverse-biased (V_out > 0, v_sw = 0 → V_D = 0 - V_out < 0). OFF.
During Q OFF: v_sw rises, L discharges (V_L = V_in - V_out, di_L/dt < 0).
              Diode is forward-biased (v_sw > V_out → V_D > 0). ON.
              i_L flows through D to charge C.

Diode commutation events: 1 per PWM cycle (twice — turn ON at
Q's OFF transition, turn OFF when i_L tries to reverse if in
DCM, OR when Q turns ON again in CCM).

Validation:
- mean V_out ≈ 24 V within 10 % (boost step-up factor).
- Energy balance: P_in = V_in · mean(I_L) ≈ P_out = V_out² / R.
- No state oscillations (diode bit doesn't flip every step).

## Tolerances (looser than buck)

The boost is harder to settle than the buck — the inductor
current ripple is comparable to mean I_L (we're closer to DCM
than the buck was). 10 % tolerance reflects:
- The trap-rule numerical error at our dt.
- The 1-dt timing error on diode commutation (worst case 1 %
  of a switching period).
- The boundary IC artifact (1-2 dt's).

These three combine to ~5-10 % steady-state error in our V0
parameter regime. The 10 % tolerance is comfortable.

## Risks

1. **State chatter**. If the per-step decision logic oscillates
   the diode between ON and OFF every step, the simulation
   produces garbage. Mitigation: extensive logging in the V0
   tests; an INFO line counts transitions per cycle and asserts
   it's a small number.

2. **Wrong diode polarity**. If `add_diode` swaps anode and
   cathode, the diode never conducts. Test catches this — the
   half-wave rectifier output would be flat-zero.

3. **CCM/DCM boundary**. The boost can run in CCM (continuous
   inductor current) or DCM (discontinuous — i_L hits 0). The
   diode state decision must handle both. Specifically: in DCM
   when Q is OFF and i_L reaches 0, the diode must turn OFF.
   The `i_diode ≤ 0` check handles this — when i_L = 0 and
   tries to go below 0, the diode opens.

4. **Cache size growth**. Each new diode doubles the
   combinatorial space. For 4 diodes, the cache has 16
   segments — still trivial. For 8+ diodes (e.g., 3-phase
   inverter with body diodes), cache grows to 256+. Layer 4 V1
   supports up to N=16 (65k segments). Past that we need lazy
   building, which is a separate OpenSpec.

## Validation gates

- `pulsim_v2_layer5_v2_tests` MUST pass with at least 25
  assertions / 8 test cases.
- All Layer 0-5 V0/V1 tests stay green.
- v1 pulsim_tests stays green.
- `openspec validate pulsim-v2-ideal-diode-auto-commutation
  --strict` passes.

## What V2 hands to follow-ups

- `pulsim-v2-event-detection` — sub-step bisection for precise
  commutation timing.
- `pulsim-v2-nonlinear-segment-newton` — real diode I-V curves
  with exponential model, reverse recovery, capacitance.
- `pulsim-v2-dc-operating-point` — consistent diode-state IC.
