# Design — `pulsim-v2-dc-operating-point` (Layer 4 V2)

## The IC inconsistency

For the trapezoidal companion at step n+1:

- Cap:      `i_{C,n+1} = G_eq · v_{C,n+1} − I_hist`
              where `I_hist = G_eq · v_{C,n} + i_{C,n}`.
- Inductor: `i_{L,n+1} = G_L,eq · v_{L,n+1} + I_hist,L`
              where `I_hist,L = i_{L,n} + G_L,eq · v_{L,n}`.

If the user supplies `v_C(0) = 0` and `i_L(0) = 0` (the V0
default), then `I_hist` at step 1 is 0 — but the REAL i_C(0+)
and v_L(0+) determined by the rest of the circuit are NOT zero.

The system is over-constrained: the user gives one piece of IC
(v_C(0), i_L(0)), but trap needs four numbers per device
(v_n, i_n on both terminals across an interval). Without the
companion-aware history terms, the first step's solve uses the
WRONG history → wrong x_1.

## The DC operating point

A consistent IC is the DC steady-state solution of the circuit
under the t=t_start switch configuration. At DC:
- `dv_C/dt = 0` → `i_C = 0` (open circuit).
- `di_L/dt = 0` → `v_L = 0` (short circuit).

In MNA terms:
- Cap: stamp `G = 0` between its terminals. No conductance, no
  contribution to `b`. The cap's `v_C` is determined by the
  rest of the circuit's KCL — typically pulled to whatever the
  resistive/source network demands.
- Inductor: stamp a constraint row `v_from − v_to = 0` with `i_L`
  as a branch unknown. The inductor IS in the state vector but
  drops zero voltage at DC.

Once the DC system is solved, we have `dc_x` with:
- v_C values at every cap (from the cap's terminals in `dc_x`).
- i_L values at every inductor (read directly from `dc_x`).
- Consistent i_C(0+) and v_L(0+) values (also derivable from
  the DC solve).

We seed:
- `HistoryState[cap_k].v_prev = v_C from dc_x`
- `HistoryState[cap_k].i_prev = 0` (DC cap current is zero by
  definition)
- `HistoryState[ind_k].v_prev = 0` (DC inductor voltage is zero
  by definition)
- `HistoryState[ind_k].i_prev = i_L from dc_x`

Then the first trap step's `I_hist = G_eq · v_prev + i_prev`
matches what physics requires.

## DC assembly

`dc_assemble` is a sibling of `assemble_segment`. Same dispatch
loop, different stamping:

```cpp
inline void dc_assemble(const topology::Graph& graph,
                         const DevicePool& pool,
                         const topology::SwitchStateMask& mask,
                         sparse::Matrix& J,
                         Vector& b) {
    // Resistor / VoltageSource / Switch / Diode: stamp as
    // usual (their stamps are dt-independent).
    // Capacitor: SKIP (g_eq = 0 contribution).
    // Inductor: stamp a short-circuit constraint:
    //   v_from − v_to − 0·i_L = 0      (just the v constraint)
    //   plus branch-current KCL contributions on terminal rows.
    //
    // Equivalent to inductor's branch-current formulation with
    // G_L,eq = ∞ (i.e., infinite slope) → just enforce v_L = 0.
}
```

The inductor stamp at DC reuses the same branch-current unknown
as the trap case, with the constraint coefficient simplified.

## DC solve

```cpp
inline Vector compute_dc_op(const topology::Graph& graph,
                              const DevicePool& pool,
                              const topology::SwitchStateMask& mask) {
    const Size n = pool.state_size(graph);
    sparse::Matrix J;
    Vector b;
    dc_assemble(graph, pool, mask, J, b);
    sparse::compress_in_place(J);

    auto solver = sparse::make_default_solver();
    if (!solver->analyze(J) || !solver->factorize(J)) {
        throw std::runtime_error(
            "compute_dc_op: DC matrix is singular for this "
            "switch state. Circuit may have redundant inductors "
            "or other algebraic degeneracy.");
    }

    Vector x;
    solver->solve(-b, x);   // J·x = -b solves the residual to 0
    return x;
}
```

## Layer 5 V3 integration

The V3 `run_transient` overload takes a new optional flag:

```cpp
SimulationResult run_transient(
    const PwlStateSpaceCache& cache,
    const Graph& graph,
    const DevicePool& pool,
    const SimulationOptions& opts,
    const SwitchScheduleFn& switch_fn,
    const BExtraFn& b_extra_fn = {},
    bool start_from_dc_op = false);   // NEW
```

When `start_from_dc_op = true`:
1. Get `initial_mask = switch_fn(opts.t_start)`.
2. Iterate diode consistency at the DC operating point (the DC
   solve might want a different diode state than V0's
   all-OFF):
   ```
   diodes.reset();   // V0 default
   do {
       mask = combine(initial_mask, diodes.current_diode_mask(),
                       diode_owned);
       dc_x = compute_dc_op(graph, pool, mask);
       flipped = diodes.update_from_state(dc_x);
   } while (flipped && iters < max);
   ```
3. Seed HistoryState from `dc_x`.
4. Record sample 0 as `(t_start, dc_x)`.
5. Continue with the V2.1 loop from sample 1.

Default `false` preserves V2.1 behaviour exactly.

## Tests

1. **RC with DC OP**: V_C(0) = ? Analytical: at t=0+ with V_dc
   applied through R, v_C(0+) = 0 (the cap can't change
   instantaneously, but it starts at 0). DC OP gives v_C = V_dc
   (the steady state). With this, the trap rule from step 1
   onwards traces the EXPONENTIAL DISCHARGE from V_dc → V_dc
   (no change, already at steady state).

   Wait — that's not what we want. We want the CHARGE
   transient from 0 → V_dc. Hmm. Let me re-think.

   Actually for RC charge from REST (V_dc applied at t=0), the
   physically-correct IC is v_C(0) = 0 AND i_C(0+) = V_dc/R
   (the instantaneous current driven by V_dc through R).

   The DC OP I described above gives v_C = V_dc (steady state),
   not v_C = 0 (initial rest). These are DIFFERENT.

   The user might want EITHER:
   - "Power-on transient from rest" → v_C(0) = 0, i_C(0+)
     consistent with the rest of the circuit.
   - "Steady-state simulation" → start from DC OP, no
     transient.

   For V0 of THIS OpenSpec, I'll provide the **DC steady-state
   IC**. Users get the "skip the startup transient" behaviour.
   The "consistent power-on rest" IC is a follow-up (it
   requires solving a special "rest IC" system: v_C frozen at
   user's value, i_C is the algebraic answer).

2. **LC tank from DC OP**: with V_dc = 10 V and R = 0, the DC OP
   gives v_C = 10, i_L = 0. Starting from there, the trap rule
   should produce a TINY oscillation (numerical noise around
   the equilibrium). Test: amplitude < 1 % of V_dc.

3. **Buck from DC OP**: at switch state ON, DC OP gives
   v_out = V_in (steady state with no R load... wait, with R
   load it gives v_out = V_in). Starting from there with PWM,
   the system oscillates around V_in · D. Should converge
   FASTER than from-rest.

Hmm actually I realize the RC test analysis above isn't quite
right. Let me reconsider what "DC operating point" means in
practice.

For RC with a DC source V_dc applied: the user is asking "what
is the steady-state of this circuit?" The answer is v_C = V_dc.
Starting from that, the trap rule should produce v_C(t) = V_dc
for all t (no transient, we're already at steady state).

So the DC OP "from t=0" with the source already applied gives
the FINAL value, not the initial. The "charge transient" only
appears if the user toggles a switch or changes the source at
t > 0.

For the buck converter: starting from DC OP (= steady state)
would mean v_out already = V_in · D, i_L already at I_L_avg.
The simulation jumps straight to the ripple-only behaviour.
That's REALLY useful for engineering analysis.

OK so DC OP is best for STEADY-STATE analyses, not for
"power-on transient from cold" analyses. Both are useful;
they're different starting conditions. V0 ships the DC OP one.

## What V1 might add

- "Consistent power-on" IC: v_C(0) = user-specified (typically
  0), i_L(0) = 0, AND i_C(0+) / v_L(0+) computed to satisfy
  KCL/KVL at t=0+. This is the "true cold start" IC and
  produces the charge transient cleanly.
- Newton iteration for nonlinear DC (e.g., real diodes with
  exponential I-V).
