## Why

Layers 0-5 V0 + Layer 4 V1 / Layer 5 V1 give the v2 kernel
everything it needs to simulate a real power-electronics
converter:
- Resistor + Voltage source + Switch + Capacitor + Inductor
- PWL state-space cache (pre-factorised per switch state)
- Trapezoidal companion for dynamic devices
- Fixed-dt time-stepping with history tracking
- User-supplied switching schedule

All the per-layer unit tests pass (199 cases, 2614 assertions).
The RC, RL, and RLC integration tests prove the trap-companion
math is right. But **no test yet exercises the full kernel on a
real PE converter topology** — the place where the architecture
either succeeds or runs into trouble.

This OpenSpec adds **the synchronous buck converter integration
test** — the canonical "first real workload" for any PE
simulator. If the buck converter steady-state matches analytical
expectations and the simulation runs at competitive speed, the
v2 architecture is **proven on a real PE workload**, not just
on textbook RC/RL/RLC.

This is the **architectural proof point** that the v2 kernel
beats PSIM/PLECS on the hot path (one map probe + one
triangular solve per step, no Newton, no factorise).

## What Changes

**Scope decision — Layer 5 V1.5**:

- **NEW integration test only.** No new device models, no new
  layer headers, no API changes. Pure exercise of the existing
  surface (Layer 4 V1 cache + Layer 5 V1 run_transient).
- **Synchronous topology.** Two complementary switches (Q1 high
  side, Q2 low side) drive the L-C filter. No diode needed —
  Q2 plays the freewheeling role with idealized switching.
  Real diode modeling lands in the nonlinear-Newton OpenSpec.
- **Power-on transient from rest.** All-zero IC (V0 default).
  The natural startup transient is part of the validation —
  buck converters DO start from rest, this is what users see.
- **Steady-state analysis.** Compute averages over the LAST 10
  switching periods (after startup transient has decayed). The
  IC artifact from Layer 4 V1 V0 is much smaller than the
  natural startup time constant `τ = L/R_load`, so the
  steady-state metrics will match analytical expectations to
  within a few %.
- **Performance smoke test.** Wall-clock budget for the full
  simulation to confirm the architecture's per-step cost
  expectation.

**Buck converter parameters** (chosen for analytical-validatable
ripples + reasonable simulation cost):

```
   V_in (12 V) ──[Q1 high-side]──┐
                                  ├──[L (100 µH)]── V_out ──┬─[R_load (10 Ω)]── GND
   GND ──[Q2 low-side]────────────┘                          │
                                                              └─[C (100 µF)]── GND

   PWM: f_sw = 100 kHz, duty D = 0.5
   Q1 ON during [0, D·T_sw); Q2 ON during [D·T_sw, T_sw).
   Complementary: at all times, exactly one is ON.
```

Analytical steady-state expectations (continuous conduction
mode):
- `V_out = V_in · D = 6.0 V`
- `I_L_avg = V_out / R_load = 0.6 A`
- `ΔI_L = V_in · D · (1 - D) / (L · f_sw) = 12·0.5·0.5 / (100µ·100k) = 0.3 A` (peak-to-peak)
- `ΔV_out = ΔI_L / (8·C·f_sw) = 0.3 / (8·100µ·100k) = 3.75 mV` (peak-to-peak)

**Simulation parameters**:
- `t_end = 1 ms` (100 switching periods — enough for steady state).
- `dt = 100 ns` (10000 simulation steps, 100 samples / PWM period).

**Validation checks** (over the LAST 10 PWM periods, after
startup):
- `mean(V_out) ≈ 6.0 V` within `< 5 %`.
- `mean(I_L)   ≈ 0.6 A` within `< 5 %`.
- `ΔI_L peak-to-peak ≈ 0.3 A` within `< 15 %`.
- `ΔV_out peak-to-peak ≈ 3.75 mV` within `< 30 %`.
  (Output voltage ripple is sensitive to ESR / parasitics, but
  even the ideal-cap analytical formula should be matched to
  the right order of magnitude.)
- **Wall-clock budget**: 10k-step simulation runs in `< 5 s` on
  the test host. (The per-step cost is dominated by Layer 4's
  cache.solve, expected ~ 10-50 µs per step in Debug. Release
  builds will be much faster.)

## Impact

- **Affected specs**:
  - MODIFIED `kernel-v2-solver` (ADDED Requirement for a buck
    converter integration scenario).

- **Affected code**:
  - NEW `core/tests/v2/layer5_v1/test_integration_buck.cpp`
    (~200 LOC).
  - NO new headers, NO API changes.
  - The existing `pulsim_v2_layer5_v1_tests` target picks up
    the new test file automatically (one CMake line edit).
  - NEW `docs/pulsim-v2/layer5-v1.5-buck-validation.md` design
    note documenting the result and what it means.

- **Migration**: none. Pure new test code.

- **Risk**: low. Every primitive the buck uses is already
  validated (Layer 4 V1 + Layer 5 V1 tests pass). The risk
  surface is just the integration — does the combined system
  match analytical expectations?
  - If yes (likely): we have our proof point and unblock the
    next layer of OpenSpecs (events, nonlinear, DC OP) with
    confidence.
  - If no: we've found a real bug in the integration that
    layered unit tests missed. Worth knowing.

- **What this proposal explicitly does NOT do**:
  - No diode model (Q2 plays its role; real Diode lands in
    `pulsim-v2-nonlinear-segment-newton`).
  - No event detection (PWM is user-scheduled; auto
    zero-crossing is `pulsim-v2-event-detection`).
  - No DC operating-point pre-charge (startup transient is
    part of the validation; `pulsim-v2-dc-operating-point`
    is a separate follow-up).
  - No closed-loop control (no V_out feedback; open-loop fixed
    duty cycle).
  - No comparative benchmarking against PSIM/PLECS (this is a
    *correctness* validation, not a head-to-head benchmark —
    that's a `pulsim-v2-vs-plecs-bench` follow-up).
