# Design — `pulsim-v2-buck-converter-validation`

## Why this OpenSpec exists

Every layer of the v2 kernel has unit + integration tests. RC,
RL, RLC all validate the trap companion against analytical
solutions. But the **real test** of a PE simulator is whether
it can run a real converter and produce numerically sensible
output that a power-electronics engineer would recognise.

This OpenSpec adds **the buck converter integration test** —
the canonical "first real workload". If it passes, the
architecture is proven on a real PE workload. If it fails, we've
found a real integration bug.

## Topology

The synchronous buck:

```
   V_in (12 V) ──[Q1 high-side]──┐
                                  ├──[L (100 µH)]── V_out ──┬─[R_load (10 Ω)]── GND
   GND ──[Q2 low-side]────────────┘                          │
                                                              └─[C (100 µF)]── GND
```

Four nodes (plus GND):
- `v_in`: top of Q1, fixed at V_in by the source.
- `v_sw`: between Q1 and L (the switching node).
- `v_out`: between L and the output cap + load.

Five branches:
- 0: V_in source (v_in → GND).
- 1: Q1 (v_in → v_sw), Switch.
- 2: Q2 (v_sw → GND), Switch.
- 3: L (v_sw → v_out), Inductor.
- 4: R_load (v_out → GND), Resistor.
- 5: C (v_out → GND), Capacitor.

State vector layout (`pool.state_size(graph)` = 6):
```
[v_in, v_sw, v_out, i_src, i_L]
   0     1     2      3     4
```

Wait — that's only 5. Let me recount. 3 nodes + 1 source + 1
inductor = 5. ✓

The state vector layout actually depends on insertion order in
the DevicePool. For our test:
1. add_voltage_source(0)  → branch_var = num_nodes + 0 = 3
2. add_switch(1) / add_switch(2) → no branch unknowns
3. add_inductor(3)  → branch_var = num_nodes + num_sources + 0 = 4
4. add_resistor(4) → no branch unknowns
5. add_capacitor(5) → no branch unknowns

So state_size = 5: `[v_in, v_sw, v_out, i_src, i_L]`.

## Switch state combinatorics

Two switches → 4 combinations (cache builds 4 segments):
- `00`: both OFF — invalid (no current path; nominally an open
  circuit but the very-small `g_off` keeps the matrix non-singular).
- `01`: Q1 OFF, Q2 ON — freewheel state (current circulates
  through L and Q2 in normal CCM operation).
- `10`: Q1 ON, Q2 OFF — power state (V_in supplies the L).
- `11`: BOTH ON — **shoot-through**. Catastrophic in a real
  hardware design (V_in → GND through tiny resistance).
  Numerically still computable (the two Switch `g_on`'s in
  series give a finite path), but produces large currents.
  Our PWM schedule MUST never visit this state.

The cache builds all 4 segments at `build(dt)` time. The
schedule callback chooses among them. With complementary PWM,
we only visit `01` and `10`.

## Complementary PWM schedule

```cpp
auto pwm = [V_in, D, f_sw](Real t) {
    const Real T_sw = 1.0 / f_sw;
    const Real phase = std::fmod(t, T_sw);
    const bool q1_on = phase < D * T_sw;
    SwitchStateMask mask(2);
    mask.set(0, q1_on);     // Q1
    mask.set(1, !q1_on);    // Q2 = complement
    return mask;
};
```

No dead-time. Real hardware needs ~50 ns dead-time to avoid
shoot-through, but our idealised model can switch
instantaneously. Adding dead-time = adding a tiny "both OFF"
interval = visiting state `00` briefly. With `g_off = 1e-9`,
state `00` is well-defined numerically (very high impedance)
but produces an unwanted brief discontinuity in i_L. For V0
validation, no dead-time is fine.

## Analytical expectations

The buck in continuous conduction mode (CCM) at duty `D`:

| Quantity | Formula                            | Value (V_in=12, D=0.5, L=100µ, C=100µ, R=10Ω, f_sw=100k) |
|----------|------------------------------------|--------------------------------------------------------|
| V_out    | V_in · D                           | **6.0 V**                                              |
| I_L_avg  | V_out / R_load                     | **0.6 A**                                              |
| ΔI_L     | V_in · D · (1−D) / (L · f_sw)      | **0.3 A** (peak-to-peak)                               |
| ΔV_out   | ΔI_L / (8 · C · f_sw)              | **3.75 mV** (peak-to-peak)                             |
| CCM check| ΔI_L < 2·I_L_avg                   | 0.3 < 1.2 ✓ (well inside CCM)                         |

For our parameters, the natural settling time constant is
`τ = L/R_load = 100µ/10 = 10 µs`. After 100 µs (one PWM period
× 10) the startup transient is mostly settled. We measure
steady-state over the LAST 10 PWM periods (t ∈ [0.9 ms, 1 ms]).

## Why all-zero IC is OK here

The startup transient (rising V_out from 0 to 6 V over ~50 µs)
DOMINATES the IC artifact (which decays in 1-2 dt's). By the
time the IC artifact has decayed, the buck is still in startup;
by the time the buck reaches steady state, both transients are
gone.

A test measuring **steady-state** (last 10 periods, average +
ripples) is essentially immune to the IC limitations of V0.
That's the deliberate design choice: validate where the V0
limitations don't matter, defer the DC OP to its own OpenSpec
when needed.

## Measurement methodology

After `run_transient` returns, sample indices in the last 10
PWM periods:
```cpp
const Real t_meas_start = opts.t_end - 10 * T_sw;
const Size k_start = static_cast<Size>(
    (t_meas_start - opts.t_start) / opts.dt);
```

Mean V_out and I_L over the measurement window:
```cpp
Real sum_v_out = 0, sum_i_L = 0;
for (k = k_start; k < N; ++k) {
    sum_v_out += states[k][v_out_idx];
    sum_i_L   += states[k][i_L_idx];
}
const Real mean_v_out = sum_v_out / (N - k_start);
const Real mean_i_L   = sum_i_L   / (N - k_start);
```

Peak-to-peak ripples over the measurement window:
```cpp
Real min_v_out = +inf, max_v_out = -inf;
Real min_i_L   = +inf, max_i_L   = -inf;
for (k = k_start; k < N; ++k) {
    min_v_out = std::min(min_v_out, states[k][v_out_idx]);
    max_v_out = std::max(max_v_out, states[k][v_out_idx]);
    min_i_L   = std::min(min_i_L,   states[k][i_L_idx]);
    max_i_L   = std::max(max_i_L,   states[k][i_L_idx]);
}
const Real delta_v_out = max_v_out - min_v_out;
const Real delta_i_L   = max_i_L   - min_i_L;
```

## Tolerance choices

| Metric | Tolerance | Rationale |
|--------|-----------|-----------|
| mean V_out  | 5 %  | Steady-state error is dominated by trap-rule numerical accuracy at dt=100ns (2nd order, well below 1 % per period). 5 % is a generous bound for the combined effect of trap error + finite settling. |
| mean I_L    | 5 %  | Same logic. |
| ΔI_L p-p    | 15 % | Output ripple in CCM is determined by the linear V_L·dt/L slope. Trap rule with dt=100ns resolves it well, but the peak detection can shift by ±1 dt at the corners, causing ~ a few % envelope error. |
| ΔV_out p-p  | 30 % | Output voltage ripple is the integral of (I_L − I_load) through C. Sensitive to phase of measurement. 30 % covers FP noise + sampling phase. The ANALYTICAL formula (ΔI_L/(8Cf_sw)) is itself an approximation that ignores higher-order harmonics — the simulation captures them, so the simulated ripple can be slightly larger. |

These tolerances are loose enough to pass even with V0
limitations (IC artifact, no DC OP, fixed dt) but tight enough
to catch a real integration bug.

## Why this is the architectural proof point

If this test passes, we've shown the v2 kernel:
1. Can build a multi-switch graph (Layer 1).
2. Stamps R, V, Switch, L, C correctly (Layers 2/3/4).
3. Pre-factorises 4 segments for the 2-switch combinatorial
   space (Layer 4).
4. Drives the trap companion correctly through 10 000 steps
   (Layer 4 V1).
5. Manages history state across switch transitions (Layer 5 V1).
6. Produces numerically meaningful PE waveforms.

That's the entire stack working end-to-end. The architecture
is proven.

If this test fails, the failure mode tells us exactly which
layer to debug:
- Wrong mean V_out → algebraic sign error or matrix-assembly
  bug somewhere.
- Wrong ripple amplitude → trap-companion timing / history
  plumbing bug.
- Way-off transient → integrator stability or state-vector
  layout bug.
- Crashes / NaN → cache lookup or matrix-factor failure.

## What we deliberately DON'T validate here

- **Comparison with PSIM/PLECS output**. That's
  `pulsim-v2-vs-plecs-bench` — generate matched PLECS netlists,
  run side-by-side, compare numerical agreement. Bigger scope,
  needs PLECS-licensed harness.
- **Performance benchmarking**. Layer 4 V0 has a 10k-lookup
  performance smoke test. The buck integration test has a
  conservative wall-clock budget but no per-step
  benchmark. Real benchmarking lands in
  `pulsim-v2-vs-plecs-bench` too.
- **Closed-loop control**. No V_out feedback loop. Adding a
  PI controller requires either Python bindings (Layer 6) or
  the BExtraFn callback to act as a controller (= hacky). Open-
  loop test is fine for kernel validation.
- **Boundary cases**. No DCM (discontinuous conduction mode) at
  light load. No shoot-through (forbidden by the PWM schedule).
  These edge cases are valuable but out of scope for the
  "happy-path proof point".

## Risks

1. **Trap rule's L+C system stability**. Trap is A-stable but
   the combined L+C+R+switching at dt = 100ns might exhibit
   numerical artifacts (chattering at the switching boundary).
   Mitigation: dt is small relative to L/R = 10 µs (factor 100
   margin) and small relative to T_sw / 100 (also factor 100).
   Should be fine.

2. **Cache build for 4 segments**. The matrix conditioning
   varies wildly between the `00` (both off, high impedance)
   and `11` (both on, shoot-through, low impedance) states.
   `00` has the L looking into very-high-impedance, so the L's
   constraint row has very-tiny conductance. KLU should handle
   it. If not, we'll need to add a small "off-state leakage
   resistance" — that's V1 of the buck OpenSpec if needed.

3. **Sign convention for Q1 vs Q2**. Q1 connects v_in to v_sw
   (high side); Q2 connects v_sw to GND (low side). I chose
   `from = v_in` and `from = v_sw` consistently. The switch
   bit-order (Q1 → mask bit 0, Q2 → mask bit 1) matches the
   branch-iteration order. Verified at test time.

## Test structure

One Catch2 TEST_CASE with 4-6 REQUIRE blocks (one per metric).
INFO blocks print the numerical values for diagnostics. The
test file is < 200 LOC including the helper struct.

## What happens next

If this OpenSpec passes:
- Move on to `pulsim-v2-event-detection` to drop the user-PWM
  burden (auto zero-crossing for natural diode commutation).
- Or `pulsim-v2-dc-operating-point` to eliminate the startup
  transient for steady-state-only analyses.
- Or `pulsim-v2-nonlinear-segment-newton` to add real diode /
  MOSFET models.

If this OpenSpec fails:
- We've found a real bug. Whatever it is, fix-then-validate
  becomes the immediate priority.
- The detailed analytical expectations make the failure mode
  diagnostic — we'll know what to look at.
