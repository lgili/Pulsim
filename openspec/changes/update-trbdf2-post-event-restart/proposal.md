## Why

After every event the variable-step engine (`engine='trbdf2'`,
`solver/trbdf2_transient.hpp`) collapses its step and climbs back five
times per step. Two causes compound (core review 2026-11,
`raw/numerics.md` item 2; `p6_ratchet.py` and `p5_trbdf2_edges.py`,
re-run on this branch):

- **A false error estimate.** The estimate at
  `trbdf2_transient.hpp:1327` uses `f_n` from the companion history,
  which after an event still holds the derivative of the pre-event
  topology. The same stale slot enters the trapezoidal stage. The first
  step after an edge therefore has an O(h) error, not O(h³), and the
  controller is right to refuse it: on the buck the falling-edge step
  is 0.86–0.92 ns.
- **A one-way memory.** `h_mem_gate` (`:943`, written at `:1756`, read
  at `:1795`) stores the accepted post-edge step, which is the
  remembered step itself whenever it is accepted first time, so it can
  only shrink. In a 400-period buck the rising-edge slot is
  **8.26e-13 s for the whole run**, with 82 rejects in total.

Measured cost: 7241 accepted steps and 16262 solves for 400 periods,
18 steps per period on segments that are almost linear; the DCM buck
shows 12 consecutive steps of 4e-14 … 1e-12 s. A step of 1e-12 s is
also below the roundoff floor of a Newton solve with an ordinary
inductor (`add-newton-roundoff-acceptance`), which is the signature of
the documented `h_min` stall; that link is an inference, the audit did
not run the stall's reproducer.

A numpy model of the controller (`p20_trbdf2_restart_proto.py`, buck
CCM, 400 periods, rtol 1e-5, atol 1e-8) reproduces the mechanism (14.1
steps and 28.2 solves per period, post-edge step 9.4e-10 s) and gives,
with the two remedies below, 4.1 steps and 8.1 solves per period with
a post-edge step of 4e-6 s: 3.5× fewer solves at an output error of
7.2e-5 V against 4.8e-5 V. Prototype figures; the audit's own estimate
for the engine is 3–4× fewer solves on PWM circuits.

## What Changes

> As implemented: the refresh runs at gate-driven events only; at a
> diode event and at the initial settle it was measured to hurt (a
> stiff L·g_off mode in the probe) and is not applied. See
> `DECISIONS.md` item 6.

- **Derivative refresh after an event.** After a gate-edge landing, a
  diode-event landing and the initial settle, the derivative slots of
  the companion history are re-read from a solve under the post-event
  mask at `t⁺`: the last probe of the zero-time cascade, which already
  holds `v_L(t⁺)` and `i_C(t⁺)` and is discarded today. When no cascade
  ran, one probe is taken for this purpose. The five stateful devices
  refresh their slot from the same solution.
- **The memory stores the controller's proposal.** The post-gate-edge
  slot keeps the step the controller proposes after the first accepted
  post-edge step, clamped to `h_max`, instead of the accepted step.
  Diode events keep having no memory.
- **Order of landing.** The memory change is only correct after the
  refresh: in the prototype, the proposal memory on top of a stale
  derivative raises the output error from 4.8e-5 V to 3.7e-4 V.
- **Alexander's SDIRK(2,2) first step** is specified as the fallback
  for an event after which no usable probe exists: its diagonal
  `1 − 1/√2` equals `γ/2`, so it uses the factor of the step and needs
  no `f_n`. It is implemented only if the acceptance of the refresh
  fails (design.md D3).
- **BREAKING (step sequence).** The accepted step sequence, the step
  count and, within the tolerance, the results of every variable-step
  run with an event change.

## Impact

- Affected specs: `transient-timestep`.
- Affected code: `solver/trbdf2_transient.hpp` (landing paths, settle,
  memory, statistics), `pwl/history_state.hpp` (slot refresh),
  `pwl/{nonlinear_capacitor,saturable_inductor,lauritzen_diode,igbt_tail,pmsm_mna}_history.hpp`
  (slot refresh), `python/bindings.cpp` (new statistics), docs,
  `CHANGELOG.md`.
- Depends on: `restore-reference-benchmark-harness`. Interacts with
  `update-trbdf2-factor-reuse` (steps on a geometric grid: the stored
  proposal is snapped to it) and `update-trbdf2-error-control`.
- Golden baselines: the fixed engine and `dsed` are untouched.
  Variable-step results move within their tolerance; step counts and
  `_trbdf2_stats` change. Tests that pin a step count or a
  variable-step number to more digits than the tolerance are re-pinned.
  Users are told in the CHANGELOG; nothing in the public API changes.
- `docs/known-limitations.md` keeps the "TR-BDF2 stalls at `h_min`"
  row unless task 5.1 shows the reproducer passes.
