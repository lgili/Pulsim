## Why

Layers 0-4 gave the v2 kernel everything it needs to solve a single
"snapshot" of a piecewise-linear circuit:
- Layer 0: sparse direct solver with cached factorisation.
- Layer 1: graph + switch-state combinatorics.
- Layer 2: AD-driven device models.
- Layer 3: generic MNA stamping.
- Layer 4: per-switch-state factor cache (the PLECS-style pivot).

Layer 5 is the **time-stepping orchestrator** that runs a real
simulation by gluing those pieces together: advance `t` from
`t_start` to `t_stop`, update the switch state per step, do one
cached `solve` per step, record the result.

A working Layer 5 V0 — even with a tight scope (fixed dt, externally-
scheduled switches, static-only circuits) — is the first
end-to-end demonstration that the v2 architecture works: build a
graph → register devices → build cache → call `run_transient` →
get back `(t, x(t))` for every step. The chopper-PWM integration
test in this OpenSpec is the proof: ~10 thousand `cache.solve`
calls per millisecond of simulated time, every one of them an
O(nnz) triangular solve on a cached factor. **That's the
architecture that beats PSIM/PLECS.**

## What Changes

**Scope decision — Layer 5 V0**:

- **Fixed dt only.** Adaptive dt + LTE estimation lands in a V1
  follow-up; V0 demonstrates the loop pattern with the simplest
  possible time-stepper.
- **User-controlled switching.** The caller supplies a callback
  `mask(t) → SwitchStateMask` that returns the switch state at
  any time `t`. This decouples event detection (V1 follow-up)
  from this layer entirely — V0 just consumes whatever schedule
  the user provides (PWM generators, gate drivers, pre-recorded
  tables).
- **Optional time-varying source RHS via callback.** `b_extra(t)
  → Vector` lets V0 simulate sinusoidal sources by recomputing
  the constraint-row RHS each step.
- **All-zero initial state.** With Layer 4 V0 restricted to
  Resistor + VoltageSource + Switch (no caps / inductors), there's
  no carryover state between steps. `x_init = Vector::Zero` is
  fine; the cached `solve` produces the right `x(t)` purely from
  `(switch_state, b_extra)`.
- **Output recording every step.** V0 doesn't down-sample; large
  simulations should be run with a coarser dt if memory matters.
  Strided output is a V1 add.

**New directory `core/include/pulsim/v2/solver/`** with three
headers:

```
pulsim/v2/solver/
├── options.hpp        # SimulationOptions struct (t_start, t_end, dt)
├── result.hpp         # SimulationResult struct (times + states)
└── run_transient.hpp  # main entry point function
```

**Main entry point**:

```cpp
namespace pulsim::v2::solver {

using SwitchScheduleFn =
    std::function<topology::SwitchStateMask(Real t)>;
using BExtraFn = std::function<Vector(Real t)>;  // returns b_extra(t)

SimulationResult run_transient(
    const pwl::PwlStateSpaceCache& cache,
    Size state_size,
    const SimulationOptions& opts,
    const SwitchScheduleFn& switch_fn,
    const BExtraFn& b_extra_fn = {});

}  // namespace pulsim::v2::solver
```

The function:
1. Validates inputs (`opts.dt > 0`, `opts.t_end > opts.t_start`).
2. Pre-allocates output vectors based on `(t_end - t_start) / dt`.
3. Initialises `x = Vector::Zero(state_size)`.
4. Loops `t = t_start; t ≤ t_end; t += dt`:
   - `mask = switch_fn(t)`
   - `b_extra = b_extra_fn ? b_extra_fn(t) : Vector::Zero(state_size)`
   - `cache.solve(mask, b_extra, x)`
   - Append `(t, x)` to the result.
5. Returns `SimulationResult`.

**Integration test — chopper at 10 kHz PWM**:

Build the V_dc-Switch-R-GND chopper from Layer 4's integration
test. Drive the switch with a 10 kHz PWM signal (`switch_fn`).
Simulate 1 ms at `dt = 1 µs` — that's 1000 simulation steps.

Verify:
- Number of recorded steps matches `(t_end - t_start) / dt + 1`.
- The average of `v_out(t)` across the 10 full PWM periods equals
  `V_dc · duty` within < 1 % (`12 · 0.5 = 6 V`).
- The `v_out(t)` waveform is a clean square wave between
  `V_dc · g_on / (g_on + G_R)` and `V_dc · g_off / (g_off + G_R)`.
- Total simulation wall-clock time should be < 100 ms on the
  test host (no benchmark; just a smoke test that 1000 cached
  solves are fast).

## Impact

- **Affected specs**:
  - NEW capability `kernel-v2-solver` (run_transient + supporting
    types).
- **Affected code** (this proposal — estimated 700-1000 LOC added,
  0 LOC modified):
  - NEW `core/include/pulsim/v2/solver/` (3 headers).
  - NEW `core/tests/v2/layer5/` (5 test files + main).
  - NEW CMake test target `pulsim_v2_layer5_tests`.
  - NEW `docs/pulsim-v2/layer5-solver-and-events.md` design note.
- **Migration**: none. Pure new code in `pulsim::v2`. v1 untouched.
- **Risk**: low. V0 is intentionally narrow — fixed dt,
  user-controlled switching, no caps/inductors. The risk
  surface is just the loop machinery + correct callback wiring,
  both of which the integration test exercises end-to-end.
- **What this proposal explicitly does NOT do**:
  - No event detection (zero-crossing on switch currents, gate-
    voltage threshold crossings). V1 follow-up will add an
    auto-event scheduler that fires `switch_fn` updates from
    inside the loop based on signal observations.
  - No trapezoidal integrator for capacitors / inductors. Needs
    history-term stamping in Layer 4 — both arrive together in
    a follow-up `pulsim-v2-trapezoidal-companion` OpenSpec.
  - No adaptive dt / LTE estimation. V1 follow-up.
  - No Newton iteration for nonlinear devices. Needs the
    `Nonlinear` branch handling currently SKIPPED by Layer 4 V0.
    Follow-up `pulsim-v2-nonlinear-segment-newton` OpenSpec.
  - No Python bindings, no YAML, no schematic loader (Layer 6).
