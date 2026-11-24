## Context

Owner decision (2026-11): the PWL diode follows PSIM, an ideal switch
plus a series forward drop `V_f` while ON; the OFF state has no offset;
the VSI's `V_f` becomes a real drop; body and freewheel diodes are
unchanged.

## Decisions

Notation: `v = v_a − v_k`, `V_on = (V_f == 0) ? V_th : max(V_th, V_f)`.

| | Rule |
|---|---|
| ON law | `i = g_on·(v − V_f)` |
| OFF law | `i = g_off·v` (no offset) |
| OFF → ON | `v ≥ V_on` |
| ON stays | `g_on·(v − V_f) > 0` |
| Crossing signal | ON: `v − V_f`; OFF: `v − V_on` |
| Power | ON: `V_f·i + i²/g_on`; OFF: `g_off·v²` |
| Hysteresis band | `max(V_th − V_f, 0)` |
| Validation | `V_f` finite and ≥ 0, else `std::invalid_argument` naming the diode |

- **`max(V_th, V_f)`.** Turning on below `V_f` turns on into negative
  current and off again at the next decision: a guaranteed mask cycle.
  `V_on = V_f + V_th` was rejected: a datasheet 0.7 V in both fields
  would turn on at 1.4 V.
- **The `V_f == 0` guard** keeps negative `V_th` (accepted today)
  bit-identical.
- **No OFF offset** (PSIM's open switch): an all-OFF mask stamps what it
  stamped before, and a zero initial condition stays consistent because
  every diode starts OFF. The cost is a `g_off·V_f` (≈ 0.7 nA) jump at
  turn-off.
- **Bit-identity at `V_f = 0`:** the RHS term is skipped when
  `V_f == 0`, so even the sign of zero in `b` is unchanged; Python keeps
  the legacy `g·v` expressions when `V_f == 0`.
- **Stamp, not `b_extra`.** The per-mask constant lives in the Switch
  stamp (`stamp_switch_fixed(..., v_drop_on)`), shared by the transient
  and DC stampers through `pwl::detail::stamp_switch_branch`, so every
  cached segment, event entry, rank-1 / parametric entry, DSED LTI mode
  and trbdf2 Newton solve inherits it. `source_scale` never scales it.
- **pwl sub-step watcher** watches `v − V_on` for both directions (exact
  when `V_th ≤ V_f`); the per-direction watcher is a separate change,
  because it would change `commutation_events` for existing `V_th > 0`
  users.
- **DSED** has no hidden band: with `V_th ≤ V_f` a zero-current turn-on
  cannot settle (Jacobi settle); the error suggests
  `V_th = V_f + 1e-3`. A Gauss-Seidel settle is a follow-up.
- **Python** reads `V_f` from `components()`, and one law in
  `_diode_state.DiodeStateTrace` (`current`, `current_start`,
  `conducting`, `v_off`) feeds `res.i`, the loss and thermal summaries.
  Same-stamp ON steps integrate the power in current form,
  `h·[V_f·(i_a + i_c)/2 + (i_a² + i_a·i_c + i_c²)/(3·g_on)]` (exact for
  a linear current); flip steps keep the trapezoid of the end powers
  `v·i`. Window cuts split at the interpolated current (same stamp) or
  power (flip), so windows still add up to the whole run.

## Risks / Trade-offs

- A decision site that misses the law lets an ON diode conduct
  backwards up to `g_on·V_f` without a warning. Mitigation: one helper,
  designated initialisers at every `Params` construction, tests that hold
  at `V_f ± ε` and assert `min(i_D) ≥ −1e-9` on every engine.
- `g_on·V_f` enters `max|b_constant|` in the Newton acceptance scale;
  this applies only after the `dx` criterion passed.
- `V_th ≤ V_f` gives a zero-width band, exactly as `V_th = 0` today:
  bridges and DSED want `V_th > V_f`.

## Migration Plan

`add_three_phase_vsi(V_f=0.0, V_th=0.7)` restores the old VSI behaviour.
