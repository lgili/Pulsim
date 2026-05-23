# Design — `pulsim-v2-event-detection` (V0)

## The chatter problem

Layer 5 V2 decides each diode's state ONCE per step, AFTER the
step's cache.solve. The mask used at step n+1 reflects whatever
state was set at the end of step n. This is fine when
commutations are "slow" (sinusoidal zero-crossings spread over
many dt's) but FAILS when:

- A diode is OFF and the next step would push v_diode huge
  positive (so it "should" be ON for that step).
- The cache.solve at step n+1 uses the OLD state (OFF), producing
  numerically catastrophic values (v_sw → 1000s of volts during
  DCM dead-time).
- The diode flips for step n+2, but the L's energy has already
  been numerically dissipated.

This is exactly what killed the V2 boost converter test.

## The V0 fix: iterate to consistency

Algorithm at each step:

```
do:
    user_mask = switch_fn(t)
    diode_mask = diodes.current_diode_mask()
    combined = combine(user_mask, diode_mask, diode_owned)
    cache.solve(combined, b_extra, x)
    flipped = diodes.update_from_state(x)
    iterations++
while (flipped && iterations < max_event_iterations)

if (flipped): throw  // non-convergence
record (t, x)
```

The loop bounds the work: ~M diodes × 2 states each = 2^M
possible mask configurations at a given step. In practice 2-3
iterations is enough for the boost (the diode flips ONCE and
the next iteration confirms stability).

The diagnostic `event_iteration_count` per sample lets the user
see WHERE the simulation needed iteration:
- Zero everywhere → no events (e.g., resistive circuits).
- Spikes at PWM transitions → expected (Q switches drive the
  diode to flip).
- Frequent non-zero values → the dt is too coarse for the
  circuit's dynamics (consider smaller dt, or move to sub-step
  bisection in a future OpenSpec).

## Why this isn't classical event detection

Classical event detection (PLECS, PSIM, modern SPICE):
1. After each step, check if any watch signal crossed zero.
2. If yes, bisect to find t* exactly.
3. Step back to t*, switch, continue from t* with fresh dt.

This requires variable dt during commutation events, which has
ripple effects on the cache structure (the trap-companion's
g_eq depends on dt, so a different dt means a different cache).

**V0 sidesteps all that** by sticking with fixed dt and just
iterating the SWITCH state at each step. The cost: ~ dt of
timing error per commutation (the trap rule averages over
[t_n, t_n+1]; the switch state we use is "what the diode
SHOULD be in by the end of the step"). For PE workloads where
dt is typically `T_sw / 100`, this is < 1 % of a switching
period — negligible.

## When iteration fails to converge

The `max_event_iterations` safety throws
`std::runtime_error` if a step doesn't stabilize. Causes:
1. **Two diodes that need each other's state to decide** — A
   wants to be ON iff B is OFF and vice versa. Rare in real
   circuits.
2. **Numerical noise at exact transition** — `i_diode` hovers
   right at 0 ± epsilon. Default 16 iterations easily covers
   this.
3. **dt too coarse for the circuit** — multiple events in one
   step. The user fix is smaller dt OR (future) sub-step
   bisection.

## SimulationOptions extension

```cpp
struct SimulationOptions {
    Real t_start = 0;
    Real t_end   = 0;
    Real dt      = 0;
    Size max_event_iterations = 16;  // NEW (default 16)

    bool valid() const noexcept;
    Size expected_step_count() const noexcept;
};
```

`max_event_iterations == 0` disables event iteration entirely
(V2 behaviour). Useful for regression testing and circuits
that don't need it.

## SimulationResult extension

```cpp
struct SimulationResult {
    std::vector<Real>   times;
    std::vector<Vector> states;
    std::vector<Size>   event_iteration_count;   // NEW (parallel to times)
    // ... existing methods ...
};
```

`event_iteration_count[k]` = number of iterations that step k
needed. 0 means "step solved on the first try, no flips".
Default for existing tests (no diodes) → always 0.

## What the boost converter test will look like

```cpp
TEST_CASE("Boost converter: V_out = V_in/(1-D) at steady state",
          "[v2][layer5_v2][integration][boost]") {
    BoostConverter b(1e-7);   // dt = 100 ns
    SimulationOptions opts{
        .t_start = 0, .t_end = 10e-3, .dt = 1e-7,
        .max_event_iterations = 16,
    };

    auto result = run_transient(*b.cache, b.g, b.pool, opts,
                                  &BoostConverter::pwm_schedule);

    // Expected: V_out converges to V_in/(1-D) = 24 V within
    // 10 %. mean(I_L) within 10 % of 2.4 A.
    // event_iteration_count: should spike to 2-3 at PWM
    // transitions (Q→OFF → diode turns ON), zero otherwise.
}
```

## Risks

1. **Existing tests might trigger event iteration accidentally**.
   Verified by running the full regression sweep — all RC/RL/
   RLC/buck/chopper/half-wave tests should produce
   `event_iteration_count == 0` everywhere (no diodes →
   no events, or diodes commutate cleanly without iteration).
2. **Performance regression on diode-heavy circuits**. The
   iteration adds an extra cache.solve per commutation. For
   the boost at 100 kHz over 10 ms = 1000 commutations, each
   needing ~1-2 extra solves = ~2000 extra triangular solves
   over 100k total steps. ≈ 2 % overhead. Negligible.
3. **Adding `event_iteration_count` to `SimulationResult`**
   changes the struct layout. All existing tests should still
   compile and pass (we're only adding a new field).
