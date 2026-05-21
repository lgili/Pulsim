## Why

Layer 5 V1.5 proved v2 on the synchronous buck — every metric
within 0.1 % of analytical. But **synchronous** is the easy
case: both switches are user-controlled. **Real PE circuits use
diodes** that auto-commutate based on circuit conditions:
- Boost converter: power switch + freewheeling diode (NOT a
  second controlled switch).
- Buck-boost: same pattern.
- Full-bridge rectifier: 4 diodes, no controlled switches.
- DCM operation of any synchronous converter: when i_L would go
  negative, the freewheeling diode opens (synchronous Q2 in our
  buck test cannot, by design).

Without a diode model, v2 can simulate maybe 30 % of real PE
circuits. **With ideal-diode auto-commutation, that jumps to
~80 %** (the remaining 20 % needs true nonlinear semiconductor
models — that's a separate OpenSpec).

This OpenSpec adds the `IdealDiode` device with per-step
auto-commutation logic. The diode IS a switch (from Layer 1's
perspective — same combinatorial space, same cache structure)
but its state is determined by v_diode / i_diode each step
instead of by the user.

## What Changes

**Scope decision — Layer 5 V2**:

- **NEW device class `IdealDiode`**. Same g_on / g_off
  parameters as `IdealSwitch`. The new bit is the per-step
  state-decision logic:
  - If OFF and `v_a − v_k ≥ V_th` (anode-cathode forward
    bias): transition to ON.
  - If ON and `i_diode ≤ 0` (current attempting to reverse):
    transition to OFF.
  - `V_th` defaults to 0 V for the "perfectly ideal" diode;
    user can override for a Si (~0.7 V) or Ge (~0.3 V) behavior.

- **At the topology level, an IdealDiode is a `Switch`**.
  Layer 1's switch enumerator includes it in the combinatorial
  space. For an N-switch circuit with M of those being diodes,
  the cache still has 2^N segments. The diode's ON/OFF state is
  one of the bits, just like a user-controlled switch — the
  difference is who chooses the bit.

- **Layer 5 V2 owns the diode-state combination**. The user's
  `switch_fn(t)` returns a mask for ALL switches in the graph
  (including diodes), but the diode bits are IGNORED and
  OVERWRITTEN by Layer 5's per-step decisions. Internally Layer
  5 maintains a `DiodeEventState` that tracks the current state
  of each diode and updates it after each cache.solve.

- **Per-step decisions, no sub-step bisection**. V0 takes the
  trap-rule's O(dt²) accuracy as good-enough: if the actual
  commutation happens partway through a step, we'll miss it by
  < dt and the next step gets the right state. For typical PE
  workloads at dt = T_sw/100 = ~100 ns, that's < 100 ns of
  timing error per commutation — invisible at the converter
  output. Sub-step bisection (PLECS-style) lands in
  `pulsim-v2-event-detection` (a follow-up).

- **All-zero initial diode state (OFF)**. At t = 0, every diode
  starts OFF. The very first cache.solve uses that. If the
  diode SHOULD be ON at t = 0+ (e.g., a half-wave rectifier with
  V_sine starting positive), the second step will flip it.
  Worst-case latency = 1 dt. Documented limitation.

**New files** (estimated 500-700 LOC):

```
core/include/pulsim/v2/models/
└── ideal_diode.hpp                   # NEW — diode device + decide_state

core/include/pulsim/v2/pwl/
├── device_pool.hpp                   # MODIFIED — add_diode method
└── diode_event_state.hpp             # NEW — per-diode state tracker

core/include/pulsim/v2/solver/
└── run_transient.hpp                 # MODIFIED — diode-aware loop variant

core/tests/v2/layer5_v2/              # NEW test directory
├── test_main.cpp
├── test_ideal_diode.cpp              # device-class unit tests
├── test_diode_event_state.cpp        # state-tracker tests
├── test_integration_half_wave_rectifier.cpp  # simplest diode use
└── test_integration_boost.cpp        # real PE workload
```

**Validation (V0 scope)**:

- **Half-wave rectifier**: V_sine(amp=10 V, f=60 Hz) → Diode →
  R(10 Ω) → GND. Simulate 2 cycles at dt = 10 µs.
  - During positive half: V_out ≈ V_sine (diode ON, R sees the
    source). > 99 % of samples within 0.5 V.
  - During negative half: V_out ≈ 0 (diode OFF, R isolated). >
    99 % of samples within 0.1 V.
  - Mean output power within 5 % of V_amp²/(4R) = 2.5 W.

**Deferred to follow-up `pulsim-v2-event-detection`**:

- **Boost converter**: tried during V0 development, but the
  per-step diode state-decision chatters during the DCM/CCM
  boundary transient at startup (with idealized switches). v_sw
  blows up during DCM dead-times because the L pumps voltage and
  the V0 logic can't resolve the commutation sub-step. The fix
  is bisection-based event detection — that's a separate
  OpenSpec.

## Impact

- **Affected specs**:
  - MODIFIED `kernel-v2-dynamic-devices` (ADDED Requirement for
    IdealDiode model).
  - MODIFIED `kernel-v2-solver` (ADDED Requirement for diode-
    aware run_transient).
  - MODIFIED `kernel-v2-pwl-cache` (ADDED Requirement: DevicePool
    can register diodes).

- **Affected code** (estimate):
  - NEW headers: ~300 LOC.
  - MODIFIED headers: ~150 LOC (device_pool.hpp,
    run_transient.hpp).
  - NEW tests: ~400 LOC across 5 files.
  - NEW CMake target: `pulsim_v2_layer5_v2_tests`.

- **Migration**: none. Pure additive. Existing tests / users
  unaffected.

- **Risk**: medium-low.
  - Auto-commutation logic is well-understood (every PE simulator
    has it).
  - The per-step decision strategy is a known V0 trade-off (PLECS's
    "ideal switching" mode works this way by default).
  - Risk concentrate: state-transition logic must NOT cause
    cycling (e.g., turning ON-OFF-ON-OFF every other step). The
    Layer 5 loop must accept the current step's state and decide
    the NEXT step's state — no in-step state changes.

- **What this proposal explicitly does NOT do**:
  - **No sub-step bisection**. The exact commutation time within
    a step is not found. Drops to `pulsim-v2-event-detection`.
  - **No real nonlinear diode (exponential I-V)**. Just the
    binary ON/OFF model. Falls under
    `pulsim-v2-nonlinear-segment-newton`.
  - **No reverse recovery, no capacitance**. Both lands in
    nonlinear-segment-Newton.
  - **No PWM controller models**. User-supplied schedule for
    controlled switches.
  - **No DCM-CCM transition handling** in tests beyond the
    boost example (which naturally exercises the boundary).
