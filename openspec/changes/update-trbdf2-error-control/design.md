## Context

The variable-step engine controls the local error of the differential
variables of the linear companions (capacitor voltages, inductor
currents) with an RMS norm and an elementary controller of exponent
−1/3. Measured behaviour (`raw/numerics.md` §1): second order, events
landed to 1e-14 s, global error following `rtol` weakly (buck output
error 8e-4, 9e-5, 2e-6 V at rtol 1e-4, 1e-6, 1e-8).

Evidence for this change: `p10_lte.py`, `p17_span.py`, `p13.py`, and
written for this proposal `p13b.py` (tolerance sweep and step sequence
around a source jump, in the engine) and `p25_trbdf2_errctl_proto.py`
(numpy: the estimator's scale and filter on `y' = λy`; a pulse-driven
RC with and without landing). The prototype's RC reproduces the
engine's error to four digits (2.692e-3, 1.824e-4, 8.683e-6 V).

## Goals / Non-Goals

- Goals: a tolerance that means what it says; an estimate that stays
  bounded for a stiff mode; the same answer at a time `t` whatever
  `t_end` is, when the circuit declares its own time scales; source
  jumps reached by a clamped step.
- Non-Goals: a PI or predictive controller; an error estimate over the
  stateful devices' states; dense output; removing `h_chatter` blind;
  the derivative refresh (`update-trbdf2-post-event-restart`); factor
  reuse; a new integrator.

## Decisions

### D1. Factor 2

`est = 2·C·h·(f_n/γ − f_γ/(γ(1−γ)) + f_1/(1−γ))`, with `C` the
constant already in the code (−0.04044). Prototype: true / estimate
goes from 1.97 to 0.98 at hλ = −0.05.

- Alternatives considered: keep the estimate and halve the default
  tolerances (same steps for default users, but an explicit `rtol`
  still means twice itself); document the factor. Rejected: the
  defect is the meaning of the user's number.
- The Python defaults (`rtol` = 1e-5, `atol` = 1e-8) are not changed.

### D2. Hosea–Shampine filter, applied once

With `W = I − (γ/2)·h·J` the iteration matrix of both stages, the
filtered estimate is `W⁻¹·est`. In companion form the raw estimate
`e` (one entry per capacitor voltage and inductor current) is put on
the right-hand side as a state increment, `(2C/(γh))·e_C` on the
capacitor's node rows and `(2L/(γh))·e_L` on the inductor's row, with
no sources, and solved with the factor of `(mask, γh)` that the two
stages just used. The filtered entries are read back from the solution
(capacitor voltage difference, inductor current). The cache gains one
entry point: a solve on an existing factor with a caller-supplied
right-hand side (today `solve_at` always adds `b_constant`).

Prototype, estimate against true error for one step from `y = 1`:

| hλ | true | ×2 | ×2, filtered |
|---|---|---|---|
| −0.05 | −4.83e-6 | 4.91e-6 | 4.84e-6 |
| −0.5 | −3.27e-3 | 3.85e-3 | 3.36e-3 |
| −5 | −1.83e-1 | 8.32e-1 | 3.38e-1 |
| −50 | −8.05e-2 | 2.07e+1 | 1.32 |
| −5e3 | −9.64e-4 | 2.35e+3 | 1.61 |
| −5e6 | −9.66e-7 | 2.36e+6 | 1.61 |

The filtered estimate is bounded by 1.61 times the amplitude of the
stiff component at any stiffness; the raw one grows without limit. A
stiff mode that is present at an amplitude below the tolerance
therefore stops forcing the step down. A stiff mode excited above the
tolerance is still resolved by small steps; that is error control
doing its job.

- Newton circuits: the filter needs a factor of `J_lin + J_nl`. The
  Newton loop builds and discards one per iteration today, so the
  filter is applied there only once `update-newton-persistent-solver`
  keeps the last factor; until then the estimate of a Newton circuit
  is unfiltered and the statistics say so.
- `h_chatter` stays. Task 4.4 runs its regression circuits with the
  floor disabled and removes it only if they all pass.
- Alternatives considered: applying the filter twice (the estimate
  then decays as the true error does; a second back-solve; not
  prototyped); filtering only when the raw estimate rejects (saves the
  back-solve on accepted steps and makes the accepted estimate, which
  sets the next step, inconsistent with the rejected one).

### D3. Bounds without the span

Two span-free scales replace the span:

- `h_ref`, the resolved step ceiling;
- `t_ulp = 16·ε·max(|t_start|, |t_end|)`, the smallest step that still
  advances the clock.

| Constant | Today | Proposed |
|---|---|---|
| `h_min` | `span·1e-11` | `max(t_ulp, h_ref·1e-8)` |
| `h_probe_floor` | `min(h_min, span·1e-12)` | `min(h_min, h_ref·1e-9)` |
| `h_chatter` | `max(h_min, span·1e-7)` | `max(h_min, h_ref·1e-4)` |
| gate pulse floor | `max(16·h_min, span·1e-9)` | `max(16·h_min, h_ref·1e-6)` |
| probe retry cap | `min(span·1e-6, h_max·1e-2)` | `h_ref·1e-2` |
| chatter window | `1e3·h_chatter` | unchanged form, follows `h_chatter` |

Every proposed value equals today's when `h_ref = span/1000`, so a run
whose ceiling is still the span fraction keeps its step sequence.

The default ceiling: the smallest of the caps the circuit declares
(20 steps per period of a periodic source, a third of the narrowest
pulse, twice the narrowest gate pulse known from an edge oracle or
learned from an opaque `switch_fn`). `span/1000` is used only while no
such cap exists. `TrBdf2Stats::h_max_basis` records `user`, `source`,
`gate` or `span`. A user `dt` is still the ceiling when given.

- Consequence: a circuit with a periodic source or schedule gets the
  same steps up to a time `t` whatever `t_end` is. An opaque
  `switch_fn` is span-bound until its first pulse is learned.
- Consequence: a short run of a slow circuit gets fewer samples than
  `span/1000` gave it (a 1 ms run of a 50 Hz source has a ceiling of
  1 ms). The guide says to pass `dt` as the output ceiling.
- Alternatives considered: keep `span/1000` in the minimum always
  (the measured coupling stays); a ceiling from the circuit's
  eigenvalues (an eigen-solve per mask); dense output on a user grid
  (`add-probes-and-recorders`).

### D4. Source breakpoints

One table per run: for each pulse source its start, end of rise, start
and end of fall, per period; for each PWM voltage source its edges.
`next_breakpoint_after(t)` is combined with the gate edge
(`min`), and the existing landing clamps the step to it. A landed step
whose end is a jump reads that source's left limit; the jump applies
from the next step. Ramp corners are landed too, so the estimate never
straddles a kink. `b_extra_fn` is opaque and has no breakpoints.

After the landing the engine restarts as it does after a gate edge.
With `update-trbdf2-post-event-restart` the derivative is refreshed
and the next step is sized by a true estimate; without it the post-jump
steps still collapse as they do after a gate edge today.

## Risks / Trade-offs

- More steps at a given tolerance (about 26 %, estimate). Stated in
  the release notes with the equivalence "double the tolerances".
- One more back-solve per attempted step on the linear path (two
  stage solves become three). Measurement gate in tasks.
- The ceiling rule changes the step sequence of every run with a
  periodic source or schedule and thins the output of short runs of
  slow circuits.
- Removing `h_chatter` is not promised.
- The filter's benefit is shown on a scalar prototype, not in the
  engine.

## Migration Plan

Each of the four parts lands as its own commit, in this order: source
breakpoints (no numerical change beyond the step sequence near jumps),
bounds, filter, scale. Each re-runs the reference harness on the
variable-step circuits; no reference error may grow at a given
tolerance, except by the definition change of D1, which must lower it.
CHANGELOG **BREAKING** for the scale and for the bounds. No flag: the
old behaviour is recovered by doubling the tolerances and passing
`dt`.

## Open Questions

1. Is thinner default output on short runs of slow circuits
   acceptable, or should `span/1000` stay in the minimum?
2. Should the Python default tolerances be doubled to keep today's
   step density for users who never set them?
3. The stateful devices' states are not in the error estimate. Open a
   follow-up, or leave it to the unified stepper?
