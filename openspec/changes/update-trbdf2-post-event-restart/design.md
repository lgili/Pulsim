## Context

TR-BDF2 (γ = 2 − √2) takes a trapezoidal stage over `γh` and a BDF2
stage to `t + h`, both with the factor of `(mask, γh)`. Its history is
physical: per capacitor `(v_n, i_n)`, per inductor `(i_n, v_n)`, and a
state plus a derivative slot for each stateful device. The local error
estimate (Bank et al. 1985) is

`est = C·h·(f_n/γ − f_γ/(γ(1−γ)) + f_1/(1−γ))`,

with `f_n = i_n/C` or the inductor `di/dt` built from `v_n`.

An event is handled in zero time: a landing commits the state at the
event, then probe solves (trapezoidal, `h_probe ≥ span·1e-12`, under
the candidate post-event mask) ask the diodes which way they conduct
until the bits settle. The probes commit nothing. The next step starts
from the landed history, whose derivative slots belong to the
pre-event circuit.

What is excellent and stays: the single-factor core, the physical
history, per-direction diode signals with Illinois on trial solves, the
exact gate landing (1e-14 s), the absence of memory for diode events.

Evidence: `raw/numerics.md` item 2; `p6_ratchet.py`,
`p5_trbdf2_edges.py`; the numpy model `p20_trbdf2_restart_proto.py`
(state-space buck, the controller constants of the engine: exponent
−1/3, growth clamp 0.2 … 5, reject clamp 0.1 … 0.9, `h_max` = span/1000,
two-slot gate memory).

## Goals / Non-Goals

- Goals: the first step after an event is as accurate as any other
  step and is sized by a true error estimate; no committed step far
  below the circuit's time scales after a gate edge; fewer solves.
- Non-Goals: the estimator's scale and filter and the step bounds
  (`update-trbdf2-error-control`); factor reuse and the geometric step
  grid (`update-trbdf2-factor-reuse`); a promise about the `h_min`
  stall; the unified stepper; memory for diode events.

## Decisions

### D1. Refresh the derivative slots from the last probe

After the cascade settles, the last probe solution `x_p` was solved
under the final post-event mask at `t + h_p`. From it:

- inductor: `v_n ← v_from − v_to` (node voltages, no cancellation);
- capacitor: `i_n ← (2C/h_p)(v_p − v_n) − i_n` (the probe's own
  companion current), `v_n` unchanged;
- saturable inductor: `V_n ← v_from − v_to`;
- PMSM: `dλ_n ← v − R·i`;
- Lauritzen diode and IGBT tail: `f_n ←` the law at the probe voltage
  and the stored charge (`q_E/T_M − A·q_M`, `k·i_ss − Q/τ`);
- nonlinear capacitor: `i_n ← (2/h_p)(Q(v_p) − Q_n) − i_n`.

The states are not touched: the probe exists to pin them. The probe
passes through `trap_solve`, which sets the device step and stage, so
the stateful devices stamp the probe at `h_p` as today.

When an event has no cascade (a gate edge in a circuit without
diodes, a diode landing that flips nothing further), one probe is
taken under the post-event mask for the refresh. The initial settle at
`t_start` refreshes the slots in the same way, which also gives the
variable-step engine a consistent start.

Roundoff of the capacitor current. It is a difference quotient: its
absolute error is about `ε·|v|·2C/h_p`, so `f_n = i/C` carries
`2ε|v|/h_p`, and the estimate carries `(|C_lte|/γ)·h·2ε|v|/h_p` with
`|C_lte|/γ = 0.069`. Against a tolerance `rtol·|v|` this is harmless
while `h/h_p ≤ rtol/(0.14·ε)`: 3.3e12 at rtol = 1e-4 and 3.3e8 at
rtol = 1e-8. With `h_p = span·1e-12` and `h ≤ span/1000` the ratio is
at most 1e9, so the bound holds down to rtol ≈ 3e-8 and not below.
(Derived, not measured.) Below that the engine uses a longer probe for
the refresh, `h_p = max(h_probe, h·0.14·ε/rtol)`; task 3.4 measures
whether that is ever needed.

- Alternatives considered:
  - *SDIRK(2,2) first step* (D3): no `f_n`, no extra solve. Prototype:
    8.1 steps per period against 4.1 for the refresh, because its
    embedded estimate is first order and conservative.
  - *A pinned-state solve* (capacitors as voltage sources): exact, and
    needs an assembly the code does not have.
  - *Accept the first step without an error test*: the O(h) error of
    the stale slot is real, not only the estimate (prototype, proposal
    memory with a stale derivative: 3.7e-4 V against 4.8e-5 V).

### D2. The memory stores a proposal

After the first accepted post-edge step of size `h` with error `e`,
the slot stores `min(h·clamp(0.9·e^(−1/3), 0.2, 5), h_max)`. An
edge-clamped step still stores nothing. With a true estimate the
controller's proposal is the right restart for the same corner one
period later, and it can grow as well as shrink.

Prototype, first post-edge step: today 9.4e-10 s (engine: 8.26e-13 s
on the rising slot); refresh with the accepted-step memory 3.0e-7 s;
refresh with the proposal memory 4.0e-6 s (`h_max`).

- Alternatives considered: no memory at all (the first step after
  every edge then starts from the pre-edge `h`, which is the
  edge-clamped remainder and arbitrary); a memory for diode events
  (measured in the engine's own comment to quintuple a flyback's event
  count; not reopened).

### D3. SDIRK(2,2) as the specified fallback

Alexander's two-stage method with `α = 1 − 1/√2`:

`(C/(αh) + G)·X_1 = …`, `X_2 = x_n + h((1−α)k_1 + α k_2)`,
`x_{n+1} = X_2`.

`α = γ/2`, so `C/(αh) = 2C/(γh)`: the matrix is the one TR-BDF2
factors for this `h`. It is stiffly accurate and L-stable, it needs no
derivative at `t_n`, and `k_2` is the derivative at `t + h`, which
fills the slots for the next TR-BDF2 step. Its error is estimated by
the embedded first-order solution, `h(1−α)(k_1 − k_2)`, with the
controller exponent −1/2 for that step. Each stateful device would
need a fourth stage form.

It is not implemented unless the refresh fails its acceptance (task
3.5), because it costs a stage form in five devices and, in the
prototype, twice the steps per period.

### D4. What is recorded

`TrBdf2Stats` gains `n_derivative_refreshes`, `n_refresh_probes` (the
probes taken only for the refresh) and `h_post_event_min`.

## Risks / Trade-offs

- One extra probe solve per event without a cascade, at an off-nominal
  `(mask, h_p)`: a factorisation unless the probe step is reused.
  Mitigation: the refresh probe uses the learned probe step so the
  event cache can serve it; measured in task 4.2.
- The capacitor-current roundoff at tight tolerances (D1).
- Larger first steps expose later steps to the gate sampling of an
  opaque `switch_fn`; the pulse-width cap is unchanged.
- The stall is not known to be cured. If the reproducer still aborts,
  the row in the known limitations stays and the result is reported.
- Tests that pin variable-step step counts move.

## Migration Plan

1. Land the refresh behind `TrBdf2Options::refresh_after_event`
   (default true in the release that ships it; false reproduces the
   old sequence for one release).
2. Land the proposal memory after the refresh's acceptance passes.
3. CHANGELOG **BREAKING (step sequence)** with the before/after table
   of `p6_ratchet.py` and `p5_trbdf2_edges.py`.
- Rollback: `refresh_after_event=false`.

## Open Questions

1. Is a one-release opt-out flag wanted, or should the old sequence
   simply go?
2. If the reproducer of the `h_min` stall passes after this change,
   is that enough to delete the known-limitations row, or must
   `add-newton-roundoff-acceptance` land first?
3. Should the initial settle's refresh (a consistent start for the
   variable-step engine) be split into its own change?
