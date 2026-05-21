# Layer 5 V2.1 — Event detection (iteration to consistency)

Layer 5 V2 added auto-commutating diodes but couldn't simulate
a boost converter cleanly — the per-step diode logic chattered
at the DCM/CCM boundary, and v_sw blew up to 1000+ V during the
brief "both switches OFF" dead-time at startup.

**Layer 5 V2.1 fixes this** by iterating the switch state to
consistency at each step:

1. Solve with the current mask.
2. Check if any diode would flip (`diodes.update_from_state`).
3. If yes, re-solve with the new mask.
4. Repeat until stable OR `max_event_iterations` is hit.
5. Record the sample.

This is the V0 of event detection — a fixed-point iteration on
switch state. **Sub-step bisection** (finding the exact mid-step
commutation time) is a future enhancement; the current loop is
"single events per step at full dt accuracy".

## Result: boost converter passes

| Metric | Analytical | Numerical | Error |
|---|---|---|---|
| mean V_out | 24.0 V | 24.014 V | **0.06 %** |
| mean I_L | 2.40 A | 2.327 A | 3.0 % |
| Wall clock | < 30 s | 1.2 s | — |
| max iterations | ≤ 4 | **1** | — |

Only `1` extra solve was needed at each commutation moment
(1999 of 100k steps), so the overhead is ~2 %. Per-step cost
still dominated by the cached triangular solve.

## API changes (additive)

```cpp
struct SimulationOptions {
    // ... existing fields ...
    Size max_event_iterations = 16;   // NEW (default 16)
};

struct SimulationResult {
    // ... existing fields ...
    std::vector<Size> event_iteration_count;   // NEW
};
```

`max_event_iterations = 0` disables iteration (matches V2
behaviour, useful for regression testing).

`event_iteration_count[k]` is the count of EXTRA solves at step
k (0 = first solve was consistent).

## What V2.1 fixes that V2 couldn't

The boost converter at startup goes through several scenarios:
- **Q ON, D OFF**: L charges. Clean.
- **Q OFF, D ON (CCM)**: L discharges into the cap. Clean.
- **Q OFF, D OFF (DCM dead-time)**: i_L → 0, both switches
  open, v_sw is unbounded (only g_off paths). **This is the
  bad case.**

V2's per-step decision saw v_sw blowing up to 1000+ V → forced
the diode ON for the NEXT step. But the current step already
had bogus values that corrupted the L's history.

V2.1's iteration loop catches this WITHIN the step:
- Solve with [Q=OFF, D=OFF]. v_sw goes huge → diode flips.
- Solve with [Q=OFF, D=ON]. v_sw clamps, i_L flows correctly.
- update_from_state confirms diode stays ON.
- Record the GOOD sample.

## Limitations (V0 of event-detection)

1. **No sub-step accuracy**. Events get caught WITHIN a step;
   the exact crossing time isn't bisected. For typical PE
   workloads at dt = T_sw/100, the error is < 1 % of a
   switching period — negligible.
2. **One commutation per device per step max**. If two events
   really need to happen mid-step, the iteration will resolve
   them sequentially (or hit the limit and throw).
3. **Iteration cost**. Each commutation costs ~1 extra
   triangular solve (~1 µs in Debug). For 100k-step
   simulations with 2k commutations, total overhead ~2 %.

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
| **5 V2.1** | **18** | **42** ← boost re-enabled |
| **Total** | **219** | **2664** |

v1 regression: 304 cases / 4214 assertions, all green.
