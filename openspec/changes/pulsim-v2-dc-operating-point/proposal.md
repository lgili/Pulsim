## Why

The Layer 4 V1 trapezoidal companion has a "V0 IC limitation":
all caps start at v_C = 0, all inductors at i_L = 0, AND the
companion's history terms assume i_C(0) = 0, v_L(0) = 0 — which
is **inconsistent** for any non-trivial circuit. Effects:

- RC charge: numerical V_C at step 1 is V/2 instead of V (the
  ODE-trap analytical value). Error decays as λ_trap^n but is
  significant for the first ~τ.
- LC tank: amplitude at step 1 is half the expected value
  (numerically attenuated). Recovers over many steps.
- Boost converter (with diodes): the inconsistent IC drives the
  system into anomalous transients that delay convergence to
  steady state.

The fix is the **DC operating point**: compute the steady-state
solution at t = 0 with caps modelled as OPEN and inductors as
SHORT, then seed HistoryState + DiodeEventState from it. The
trap rule then starts from a consistent IC and produces the
correct transient.

## What Changes

**Scope decision — Layer 4 V2** (the DC operating-point
capability):

- New header `pwl/dc_assemble.hpp` — `dc_assemble(graph, pool,
  mask, J, b)` builds a special MNA matrix where:
  - Capacitors are stamped as `G = 0` (open circuit, no
    conductance, no history).
  - Inductors are stamped as a branch-current constraint with
    `v_from − v_to = 0` (short circuit, but `i_L` is still an
    unknown).
  - All other devices stamp as usual.
- New header `pwl/dc_operating_point.hpp` — `compute_dc_op(
  graph, pool, mask) → Vector` solves the DC system for the
  given switch state. Returns the state vector at the DC
  operating point.
- New header `pwl/seeding.hpp` — `seed_history_from_dc_op(
  history, dc_x, graph, pool)` and
  `seed_diodes_from_dc_op(diodes, dc_x, graph, pool)` —
  utility helpers to initialise HistoryState and
  DiodeEventState from a DC op vector.

- **Layer 5 V3 `run_transient` overload** with an additional
  optional `bool start_from_dc_op = false` flag (or
  alternatively a `Vector x_init` parameter). When set:
  1. Pick the initial switch state from `switch_fn(t_start)`.
  2. Iterate diodes to consistency for the DC op (some diodes
     may flip when the DC system is solved).
  3. Compute DC op, seed HistoryState + DiodeEventState.
  4. Record sample 0 from `dc_x` (NOT from all-zero IC).
  5. Continue with normal trap-rule stepping from sample 1.

- **Tests**:
  - RC charge with `start_from_dc_op = false` (V0 behaviour)
    reproduces V2.1 baseline.
  - RC charge with `start_from_dc_op = true` matches analytical
    `1 − e^{−t/τ}` from sample 0 (no boundary artifact).
  - Boost converter with DC op pre-charge converges faster.

## Impact

- **Affected specs**: ADDED capability `kernel-v2-dc-op` (new
  delta spec).
- **Affected code**: ~400 LOC new (3 headers), ~50 LOC
  modified (run_transient overload).
- **Migration**: zero. Default `start_from_dc_op = false`
  preserves V2.1 behaviour exactly.
- **Risk**: medium. The DC assembly is a different stamping
  flavour, and inductors-as-shorts can produce singular
  systems if the topology has redundant inductors (e.g., two
  inductors in parallel with no resistance). Mitigation: throw
  on singular DC, document the case, ship a regression test
  that confirms RC / RL / RLC / buck all work.

## What this proposal does NOT do

- **No DC OP for nonlinear devices**. The smooth-blend ideal
  diode (Layer 2 `IdealDiode`) and other nonlinear devices
  would need Newton iteration to find their DC. Stays in
  `pulsim-v2-nonlinear-segment-newton`.
- **No closed-loop DC OP** (e.g., "find the duty that makes
  V_out = 10 V"). That's a separate analysis mode.
- **No transient extrapolation from a non-DC IC**. The user
  CAN'T supply an arbitrary `x_init` in V0 — only "all zero"
  or "compute DC op". V1 might add the manual override.
