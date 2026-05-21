# Layer 5 V2.2 — Sub-step commutation diagnostics

Layer 5 V2.1 caught the chatter in event-iteration but the
recorded `(t, x)` pairs are still at the dt grid — a diode that
flips between `t_n` and `t_{n+1}` is recorded as "flipped at
`t_{n+1}`", with O(dt) timing error.

Layer 5 V2.2 ships **sub-step commutation diagnostics**: a
linearly-interpolated estimate of the actual commutation time
for every diode flip, recorded in
`SimulationResult.commutation_events`. The recorded state
vectors are still at the dt grid (no sub-step state correction
— that requires multi-dt cache support, future
`pulsim-v2-multi-dt-cache` OpenSpec).

## Result format

```cpp
struct CommutationEvent {
    Real  t_estimated;   // linear-interp zero crossing
    Index branch_id;     // which diode flipped
    bool  new_state;     // true = ON, false = OFF
};

struct SimulationResult {
    // ... existing fields ...
    std::vector<CommutationEvent> commutation_events;
};
```

For each diode flip, `t_estimated` is computed by linear
interpolation of `(v_diode − V_th)` between `t_n` (`x_prev`) and
`t_{n+1}` (`x`):

```
t* = t_n + (t_{n+1} - t_n) · s_prev / (s_prev - s_curr)
```

where `s = v_diode − V_th`. The same formula works for both
OFF→ON (s changes from negative to positive) and ON→OFF (s
changes from positive to negative).

## Verified

- **Diode-free circuit → empty list**: no events recorded
  when the pool has no diodes.
- **Step source + diode**: V_dc switches from −2 V to +2 V at
  `t = 0.5 s`; the OFF→ON event is recorded with
  `t_estimated` within `2·dt` of the analytical 0.5 s.

## V0 limitation: no state correction

The recorded `result.states[k]` is still at `t_k = k · dt`. The
sub-step timing only affects `result.commutation_events[i].t_estimated`.

Why we can't easily correct the state mid-step: the
trap-companion cache is dt-specific (each segment's matrix
contains `g_eq = 2C/dt`). A partial step at `dt_partial < dt`
needs a separate factored matrix per partial-dt value. That's
a multi-dt cache architecture — significant work, future
OpenSpec.

For workloads where the user only needs the timing info (switch
loss estimation, EMC analysis, gate-drive sequencing), the
diagnostic-only V0 is enough.

## Status

| Layer | Cases | Assertions |
|---|---|---|
| 0 | 19 | 80 |
| 1 | 36 | 126 |
| 2 | 36 | 93 |
| 3 | 16 | 61 |
| 4 V0 | 24 | 58 |
| 5 V0 | 21 | 2069 |
| 4 V1 | 32 | 76 |
| 5 V1 | 17 | 59 |
| 5 V2.2 | **20** | **46** ← +2 / +4 substep tests |
| 4 V2 | 9 | 520 |
| 4 V3 | 5 | 13 |
| 5 V4 | 4 | 60 |
| **Total** | **239** | **3261** |
