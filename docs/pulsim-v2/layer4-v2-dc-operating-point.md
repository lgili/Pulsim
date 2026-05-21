# Layer 4 V2 — DC operating-point initialization

Layers 4 V1 and 5 V1 simulated dynamic circuits with the trap-
rule companion, but every simulation started from
`v_C = 0, i_L = 0`. For circuits with a meaningful DC steady
state (most real ones), this caused a "startup transient" that
took several time constants to decay before the simulation
matched physics.

Layer 4 V2 adds the **DC operating-point** capability: compute
the steady-state solution at t = 0 with caps as OPEN circuits
and inductors as SHORT circuits, then seed `HistoryState` /
`DiodeEventState` from it. The trap-rule simulation can then
either:
- **Skip the startup**: simulation starts AT steady state, only
  the ripple is visible.
- **Use as a more-accurate IC**: simulation starts at the
  steady state of the t=t_start switch configuration, which is
  the right IC for most engineering analyses.

## API surface

```cpp
namespace pulsim::v2::pwl {

/// DC MNA assembly: caps SKIPPED, inductors as v=0 shorts.
inline void dc_assemble(const Graph&, const DevicePool&,
                         const SwitchStateMask&,
                         sparse::Matrix& J, Vector& b);

/// Solves the DC system, returns the state vector.
inline Vector compute_dc_op(const Graph&, const DevicePool&,
                              const SwitchStateMask&);

/// Build a HistoryState seeded from a DC OP vector.
inline HistoryState make_seeded_history(
    const Graph&, const DevicePool&, const Vector& dc_x);

/// Build a DiodeEventState with each diode set per the DC bias.
inline DiodeEventState make_seeded_diodes(
    const Graph&, const DevicePool&, const Vector& dc_x);

}  // namespace pulsim::v2::pwl

namespace pulsim::v2::solver {

SimulationResult run_transient(
    const PwlStateSpaceCache& cache,
    const Graph& graph,
    const DevicePool& pool,
    const SimulationOptions& opts,
    const SwitchScheduleFn& switch_fn,
    const BExtraFn& b_extra_fn = {},
    bool start_from_dc_op = false);   // NEW V3 flag

}  // namespace pulsim::v2::solver
```

`start_from_dc_op = false` (default) preserves Layer 5 V2.1
behaviour exactly. `true` makes the loop:

1. Get `initial_mask = switch_fn(t_start)`.
2. Iterate diode consistency on the DC system.
3. Compute `dc_x = compute_dc_op(graph, pool, mask)`.
4. Seed HistoryState from `dc_x` (cap: v_prev=v_C, i_prev=0;
   L: v_prev=0, i_prev=i_L).
5. Record sample 0 as `(t_start, dc_x)`.
6. Continue with the trap-rule loop from sample 1.

## DC stamping rules

| Device       | DC behaviour                                              |
|--------------|-----------------------------------------------------------|
| Resistor     | Same stamp as transient (G between terminals).            |
| Voltage src  | Same constraint (v_from − v_to = V).                      |
| Switch       | Same conductance (g_on or g_off per mask bit).            |
| **Capacitor**| **SKIPPED** (open circuit, G = 0).                        |
| **Inductor** | Constraint row `v_from − v_to = 0`, branch unknown i_L.   |
| Diode        | Like Switch (per the mask bit).                           |

A tiny ε on the inductor's branch-row diagonal prevents
structural singularity in degenerate topologies.

## Examples

### RC with DC OP

```cpp
PwlStateSpaceCache cache(g, pool);
cache.build(/*dt=*/10e-9);
SimulationOptions opts{ .t_start=0, .t_end=5e-6, .dt=10e-9 };
auto fn = [](Real){ return SwitchStateMask(0); };

// V0: starts at v_C=0, traces exponential charge to V_dc.
auto result_v0 = run_transient(cache, g, pool, opts, fn);

// V3: starts at v_C=V_dc (DC OP), no transient.
auto result_v3 = run_transient(cache, g, pool, opts, fn,
                                /*b_extra_fn=*/{},
                                /*start_from_dc_op=*/true);
```

### LC tank — undamped

With DC OP, an LC tank with V_dc applied starts at the DC
steady state (v_C = V_dc, i_L = 0). The trap-rule simulation
then produces a TINY oscillation (numerical noise only) — the
system is already at equilibrium.

### Buck converter

With DC OP at the t=0 switch state, the buck starts at the
steady-state v_out for THAT switch config. PWM then drives the
small-signal ripple around it. Eliminates the 4-period
startup transient that "from-rest" simulations show.

## What V0 does NOT do

- **No DC OP for nonlinear devices**. The smooth-blend
  `IdealDiode` in Layer 2 would need Newton iteration to find
  its DC point. Lands in `pulsim-v2-nonlinear-segment-newton`.
- **No "cold-start with charge transient"**. The DC OP gives
  the steady state; the user who wants the from-rest transient
  uses the default (`start_from_dc_op = false`).
- **No multi-segment DC sweep**. Pick one switch state at
  t = t_start; the DC OP is for that one. If the user needs the
  DC for a different state (e.g., the buck's "off" state during
  startup analysis), they call `compute_dc_op` directly.

## Test coverage

- `pulsim_v2_layer4_v2_tests`: 520 assertions / 9 cases.
  - DC OP for V-R, V-R-C, V-R-L, chopper.
  - HistoryState seeding (caps + inductors).
  - DiodeEventState seeding.
  - RC integration with DC OP (stays at V_dc throughout).
  - Regression: from-zero IC still works.

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
| 5 V2.1 | 18 | 42 |
| **4 V2** | **9** | **520** ← NEW |
| **Total** | **228** | **3184** |

v1 regression: 304 / 4214, all green.
