## Context

`run_transient` advances `x` from `t_{k−1}` to `t_k = t_start + k·dt`
(integer step counter) with the trapezoidal companion. Per dynamic
device it carries a state and a derivative slot:

| Device | State | Derivative slot | Trapezoidal law at step `h` |
|---|---|---|---|
| Capacitor | `v_n` | `i_n` | `i = (2C/h)(v − v_n) − i_n` |
| Inductor | `i_n` | `v_n` | `v = (2L/h)(i − i_n) − v_n` |
| Mutual `M` | `i_n` of both windings | — | `(2M/h)(i_other − i_other,n)` on each row |
| Nonlinear capacitor | `Q_n` | `i_n` | `i = (2/h)(Q(v) − Q_n) − i_n` |
| Saturable inductor (flux row, conductance form) | `λ_n` | `V_n` | `v = (2/h)(λ(i) − λ_n) − V_n` |
| Lauritzen diode | `q_M,n` | `f_n` | `q_M(1 + hA/2) = q_M,n + (h/2) f_n + (h/2) q_E/T_M` |
| IGBT tail | `Q_n` | `f_n` | `Q(1 + h/(2τ)) = Q_n + (h/2) f_n + (hk/2) i_ss` |
| MNA-native PMSM | `λ_n` (abc) | `dλ_n` | `(v − Ri) + dλ_n − (2/h)(λ − λ_n) = 0` |

After a topology change, a source jump or an inconsistent start, the
derivative slot describes a circuit that no longer exists. The
trapezoidal rule then averages a stale derivative with a new one: an
O(h) local error (the edge lands at the midpoint of the step) and an
undamped alternation of every algebraic quantity.

Owner decisions that bind this design (lessons-learned notes):
no "trapezoid at 2h" start; the flux row stays in conductance form;
every solve passes through the one point that sets the device step;
each refresh wrapper captures the accumulated chain; the absolute `dx`
rule stays.

Evidence: `raw/numerics.md` item 1 and the probes named there, plus the
numpy prototypes written for this proposal (all in
`/Users/lgili/pulsim_wt/core_review_2026-11/numerics/`):
`p18_cda_proto.py` (linear circuits, a dense re-implementation of the
step with the engine's event iteration), `p22_cda_stateful_proto.py`
(nonlinear capacitor, saturable inductor), `p23_cda_pairs_proto.py`
(how many pairs). Prototype numbers are labelled as such.

## Goals / Non-Goals

- Goals: second-order accuracy of the fixed engine across mask changes
  and source jumps that fall on the grid, and from any start; no
  undamped alternation after an event; no new factorisation; the
  trapezoidal rule untouched in smooth segments.
- Non-Goals:
  - Events and edges off the grid. They stay first order here and are
    the subject of `add-fixed-step-event-splitting`.
  - Sample 0. A node whose consistent value at `t_start` is −5 V still
    reads 0 V; that needs the consistent-initialisation solve
    (capacitors as voltage sources) the note on the first-order start
    describes. Not in this change.
  - The `½CV²` of a hard-switched capacitor. It is dissipated inside
    one half-step and `i²R` still does not see it (prototype: 0.0500 W
    of conduction, nothing for the 0.5 W); that is
    `add-in-kernel-event-losses`.
  - The Newton-loop diode-mask chatter and the `h_min` stall.
  - Any global damping, θ-method or Gear-2 mode.

## Decisions

### D0. Rework: three SDIRK(2,2) steps replace the backward-Euler pair

The pair of D1 was implemented (41711488) and failed a review on
resonant modes. Backward Euler over `h/2` damps a mode of frequency ω
by about (ωh)²/2 per event, and a converter has two events per period.

Engine, Q = 50 tank driven at resonance by a ±50 V PWM, peak current
error against T/1600:

| Variant | T/50 | T/100 |
|---|---|---|
| flag off | −0.855 % | −0.055 % |
| BE pair | −11.9 % | −3.1 % |
| SDIRK×3 | −0.730 % | −0.040 % |

Ideal LC beside an electrically isolated PWM source, energy after 100
periods: flag off 0.9961, BE pair 0.019, SDIRK×3 0.9973.

**The rule now.** The step containing the discontinuity and the next
two are Alexander SDIRK(2,2) steps, `γ = 1 − 1/√2` (L-stable, stiffly
accurate, order 2):

- stage 1: a backward-Euler-like solve of `γh` from `x0`, ending at
  `t0 + γh`;
- stage 2: the same solve of `γh` from the pseudo previous state
  `x0* = x0 + c(x1 − x0)`, `c = (1 − γ)/γ`, ending at `t0 + h`. This is
  the second SDIRK stage rewritten as a backward-Euler step: its
  residual `x2 − x0 − h(1−γ)f1 − hγf2` with `hγf1 = x1 − x0`.

Details:
- **Factor.** The stage conductance `C/(γh)` is the trapezoidal
  companion at `2γh`, so both stages use one factor per mask at that
  fixed step. The cache's event table (`solve_at`, one entry per mask
  under the segment byte budget since update-trbdf2-factor-reuse)
  keeps them; the 8-entry LRU it replaced refactored on every restart
  step of a VSI: 3172 builds in 3157 restart steps at dt = 1e-6. (A
  second lazy cache served this until the integration of
  update-trbdf2-factor-reuse; see DECISIONS.)
- **Stateful devices.** `HistoryState` and the five devices get
  `extrapolate_previous(x1, c)`: capacitor `v_prev`, inductor `i_prev`,
  Coss `q_prev`, Lauritzen `q_M`, IGBT tail `Q`, PMSM `λ` (abc) and
  saturable `λ`. Each is extrapolated from the stage-1 solution through
  the device's own law; the flux row stays in conductance form. For
  Jiles–Atherton the stage-1 flux is evaluated from the committed JA
  state (an O(h) error at a loop tip, documented).
- **Commit.** The commit is once per step, from stage 2: the
  backward-Euler commit at `γh` from `x0*` refreshes the derivative
  slots with `(x2 − x0*)/(γh)`, the stage-2 derivative. That is `f(x2)`
  for a linear circuit, so the trapezoid that follows starts
  consistent.
- **Why three steps.** The window length is the review's decision; it
  came from the prototype `p31_restart_sdirk_multi.py`. The step that
  contains the jump carries the impulse. The steps after it damp what
  is left by L-stability before the trapezoid resumes. Three meets
  every acceptance target in the engine, including the 1 ps mode of the
  capacitor loop at dt = 1e-9. No shorter window was re-measured in the
  engine.
- **Window state.** It is a counter. Each trigger sets it to 3, each
  restart step consumes one, and a diode flip in either stage sets it
  to `max(left, 2)`. `SolverSnapshot::restart_steps_left` carries it
  (pickle field 12; a legacy bool `True` loads as 3).
- **Diodes per stage, measured.** The alternative decides the diodes
  once per step, with stage 1 solved under the old bits and the step
  redone on a flip. On the buck it collapses: CCM `v_C` error −11.9 V,
  DCM output 0.83 V instead of 13.3 V. Per stage: CCM `i_L` error
  1.17e-7 A at T/50, order 2; DCM idle 0.05 %.
- **`b_extra_fn` calls.** Called once per stage time: `t0 + γh` (off
  the grid) and `t0 + h`, with no discarded evaluation. A callback that
  jumps declares it with `fn.next_jump_after(t)` and is then read at
  its left limit and restarted like a built-in source. Composed
  callbacks (C-blocks, hooks) carry the minimum of their parts.
  Undeclared jumps are not restarted.

Engine acceptance with the flag on:

| Case | BE pair | SDIRK×3 |
|---|---|---|
| RL edge, dt = 2 µs | −5.05e-6 A | +1.01e-6 A, order 1.98 |
| buck CCM `i_L`, T/50 | 2.20e-5 A | 1.17e-7 A, order 2.0 |
| RC source step, τ/10 | 2.19e-2 V | 2.63e-3 V |
| cap loop, dt = 1e-9 | 9.821 … 10.178 A | 9.994 … 10.004 A |

D1, D6 and D8 below describe the pair, kept as the record of what was
replaced; D7 is updated in place.

### D1. Two backward-Euler half-steps reuse the nominal matrix (superseded by D0)

Backward Euler over an interval `h_e` reads `dX/dt ≈ (X − X_n)/h_e`.
With `h_e = h/2`:

- capacitor `i = (C/h_e)(v − v_n) = (2C/h)(v − v_n)`;
- inductor `v = (2L/h)(i − i_n)`; mutual term `(2M/h)(…)` unchanged;
- nonlinear capacitor `i = (2/h)(Q(v) − Q_n)`, `∂i/∂v = (2/h)C(v)`;
- flux row `v = (2/h)(λ(i) − λ_n)`, diagonal `−(2/h)L(i)`;
- Lauritzen `q_M(1 + hA/2) = q_M,n + (h/2) q_E/T_M`, so `K1` and the
  denominator are the trapezoidal ones and `K0 = q_M,n/den`;
- IGBT tail likewise, `K0 = Q_n/den`;
- PMSM `(v − Ri) − (2/h)(λ − λ_n) = 0`.

In every row the coefficient of the unknown is the trapezoidal one at
`h`; only the history term loses the derivative slot. So the linear
path calls `cache.solve(mask, …)` with the nominal factor, the Newton
path uses the nominal segment `cache.lookup(mask)`, and no
`(mask, dt)` key is added. The right-hand side is the companion
history without the slot plus the sources at the half-step's end time.

- Alternatives considered:
  - *One backward-Euler step over `h`.* Conductance `C/h`: a second
    factor per mask, and it is the rejected variant below.
  - *Exact derivative refresh* (re-evaluate the slots under the new
    topology at `t⁺` with the states pinned). Slightly more accurate
    (prototype: +1.0e-6 A against −5.1e-6 A on the RL edge) and it
    would also fix sample 0, but it needs an assembly with capacitors
    as voltage sources and one more factor per mask. Kept as a
    follow-up.
  - *Global damping.* Rejected: an ideal LC conserves energy exactly
    over 1000 periods today (`raw/numerics.md` §3).
  - *Leave the engine alone and tell users to put edges on the
    half-grid.* It would fix timing and not the alternation.

### D2. Why this is not the rejected "trapezoid at 2h" start

The rejected variant took one step that advanced the clock by `h` with
the companion assembled at `2h`. That is backward Euler over `h`, and
it has three properties the owner's note records: every stateful device
was handed `2h` for a step whose elapsed time was `h`; it needs the
factor keyed at `2h`; and it re-scaled the dt-retry ladder (21 tests).

Here each solve advances the clock by `h/2` and the devices are told
`h_e = h/2` with the rule `BackwardEuler`. The interval a device
integrates over is the interval the clock advanced, the matrix is the
one already factored at `h`, and the retry ladder keeps its step sizes
(D7).

One point must be put to the owner plainly (Open Questions). The note
says the `2h` start moved
`test_nonlinear_capacitor.py::test_charge_is_conserved_exactly` "from
its half-step charge offset to a full-step one (4.8 % at sample 11)".
`p22_cda_stateful_proto.py` models that device's stamp alone, charged
at 6 A from rest:

| Start | `Q/(I·t)` at samples 1, 2, 11, 40 | distance from the test's `I·(t − h/2)` at sample 11 |
|---|---|---|
| today | 0.500, 0.750, 0.955, 0.988 | 0.00 % |
| companion at `2h` (rejected) | 1.000 at all four | 4.76 % |
| two BE half-steps (this change) | 1.000 at all four | 4.76 % |

In the prototype both starts give `Q = I·t`, the statement in the
test's own docstring, and 4.76 % is the distance from `I·t` to the
reference `I·(t − h/2)` that the test documents as the defect. This
change will therefore move that test by the same 4.8 %, and it treats
that as the fix (`q_ref` becomes `I·t`). The prototype is not the
engine: task 5.2 measures the sign in the engine before the reference
is edited.

### D3. An explicit `BackwardEuler` stage

`pwl::TrBdf2Stage` gains `BackwardEuler`: `dX/dt = (X − X_n)/h_e`,
where the `h` argument is the interval this solve advances (as it is
for `Trapezoidal`; for `Bdf2Stage2` it stays the full step). `CossStage`
gains the same value. `HistoryState` gains
`compute_b_extra_backward_euler(h_e, out)` and a backward-Euler commit
(`i_C = (C/h_e)(v − v_n)`; inductor commit unchanged).

Per device, the stamp is the `Trapezoidal` branch evaluated at `2·h_e`
with the stored derivative omitted, and the commit stores the
derivative the rule produced:

| Device | Stamp change | Commit |
|---|---|---|
| Nonlinear capacitor | drop `− i_n` | `i_n = (1/h_e)(Q − Q_n)` |
| Saturable inductor, gapped core, Jiles–Atherton | drop `V_n`; row stays in conductance form; JA integrates from `(H_n, M_n)` | `V_n = v_from − v_to`, `λ_n` from the law (unchanged) |
| Lauritzen diode | `K0 = q_M,n/den` | `f_n = q_E/T_M − A·q_M` (unchanged: a law evaluation) |
| IGBT tail | `K0 = Q_n/den` | `f_n = k·i_ss − Q/τ` (unchanged) |
| PMSM | drop `dλ_n` | `dλ_n = v − R·i` (unchanged) |

- Alternatives considered: *zero the derivative slots, then run the
  ordinary trapezoidal solve at `h` twice.* Algebraically identical for
  all five devices (each is in derivative-slot form; checked by reading
  the stamps) and about six one-line methods. Rejected as the
  mechanism: it tells the device `h` for a solve that advances `h/2`,
  which is the confusion D2 exists to avoid, and
  `pwl/trbdf2_stage.hpp` states that the stage is an explicit argument
  everywhere so that a wrong history term cannot hide behind a correct
  matrix. It is kept as the test oracle: the `BackwardEuler` result
  must equal the zero-slot trapezoidal result to 1e-14.

`run_transient`'s `refresh_dt` gets a companion `refresh_stage`,
assigned at the same single point (`run_event_iteration`), and each of
the five wrappers passes it on. The wrappers keep capturing
`nl_refresh_effective`.

### D4. The gate mask of a step is sampled at its start

Today the step ending at `t_k` is stamped with `switch_fn(t_k)`. With a
right-continuous closure (`phase < D`; the `numeric/pwm_phase.hpp`
helpers behind `NativePwm2Switch` and `make_pwm_switch_fn` make an
edge on a grid point take effect at that point) the step that ends on the edge already
carries the new mask, and the stale-derivative trapezoid places the
edge at that step's midpoint. With the restart rule the edge is placed
at the start of the first step that carries the new mask, so
end-sampling would put it a full step early. Prototype, RL edge on the
grid, error at dt = 4, 2, 1 µs:

| Variant | Error [A] | Order |
|---|---|---|
| today, closure closed for `t > te` | −1.214e-2, −6.067e-3, −3.032e-3 | 1.00 |
| today, closure closed for `t ≥ te` | +1.210e-2, +6.057e-3, +3.030e-3 | 1.00 |
| restart, mask sampled at the step END | +2.418e-2, +1.211e-2, +6.058e-3 | 1.00 |
| restart, mask sampled at the step START | −2.022e-5, −5.050e-6, −1.258e-6 | 2.00 |

So, with the rule on, step `(t_{k−1}, t_k]` uses `switch_fn(t_{k−1})`.
`SimulationResult::switch_state_bits` keeps recording the mask each
sample was solved under.

- Alternatives considered: midpoint sampling (an edge on the grid is
  then half a step late); bisecting `switch_fn` inside the step (that
  is event splitting).
- Consequences: a closure written for the old convention with edges on
  the grid (`kk = round(t/dt) % N < Don`, guide 3.2) keeps its duty and
  shifts by one step; a left-continuous closure (`t > te`) becomes one
  step late. Python consumers that re-derive a sample's mask from
  `switch_fn(times[k])` (`losses.py`, `_diode_state.py`) must read
  `switch_state_bits` instead (task 6.3).
- The step observer keeps today's arguments `(t_k, x_{k−1})` and still
  fires before the mask is read. Whether its time label becomes
  `t_{k−1}` belongs to `fix-control-tick-semantics`.

### D5. Source jumps

A jump table is built once from the sources that have them: pulse
sources with zero rise or fall time (start, start + width, every
period) and PWM voltage sources. `b_extra_fn` is opaque and contributes
none. A step that ends on a jump evaluates that source's left limit;
the jump applies from the next step, which restarts. A jump strictly
inside a step makes that step a restart step and is seen by whichever
half-step ends after it (a timing error of at most `h/2`, first order,
until event splitting).

Prototype, 10 V step on the grid into an RC, dt = τ/10, τ/20, τ/40,
τ/80: today 0.476, 0.244, 0.124, 0.062 V (order 0.98); with the rule
2.19e-2, 5.85e-3, 1.51e-3, 3.84e-4 V (order 1.95).

### D6. The event iteration runs per half-step (superseded by D0: per stage)

The two half-steps are not one atomic solve. If the diodes are decided
only on the state after the second half-step, a gate that opens an
inductor's path produces its kick in the first half-step, backward
Euler damps it completely, and at the end no diode is asked to conduct.
Prototype, buck CCM at T/100: `v_out` = 0.286 V against 12.229 V, the
inductor current chopped at every turn-off. So each half-step runs the
existing event iteration from its own start state (solve, let the
diodes re-decide, re-solve), with its own mask-cycle set.

Two half-steps are needed under the settled mask. The half-step in
which a jump happens carries the jump's impulse in its derivative; only
the one after it is clean. If a diode flips while the second half-step
is iterating, the next step is therefore a restart step too. Prototype,
DCM buck idle interval, `v(sw)` (true value `v_out` = 13.317 V):

| Variant | T/100 | T/400 |
|---|---|---|
| today | 0.360 … 26.268 V | 0.900 … 25.745 V |
| restart, one half-step after the flip | 0.843 … 25.793 V | 3.558 … 23.089 V |
| restart, two half-steps after the flip | 13.311 … 13.323 V | 13.311 … 13.324 V |

A diode flip found by an ordinary trapezoidal step's event iteration
is a discontinuity inside that step: the step is rolled back to its
snapshot (the per-step snapshots already exist) and re-taken as a
restart step.

### D7. Interactions

- **dt-retry ladder.** Unchanged in depth, sub-step sizes and
  `DtRetry` records. At level `j` the step is re-taken from the
  snapshot as `2^j` sub-steps of `dt/2^j`; if the outer step was a
  restart step EVERY sub-step is an SDIRK step of `dt/2^j` (its stages
  on the event solver at `2γ·dt/2^j`). A flip in a trapezoidal sub-step
  arms the window. A step that was not a restart step retries exactly
  as today.
- **Sub-step commutation correction**
  (`enable_substep_state_correction`). Refused together with the rule
  (`std::invalid_argument` naming both): its V3 path commits a
  trapezoidal history from an interpolated state that the SDIRK window
  cannot see. `add-fixed-step-event-splitting` replaces it.
- **Inductor freeze guard.** Applied to the final state of the step,
  before the commit, as today.
- **Snapshots and resume.** `final_snapshot` gains a counter,
  `restart_steps_left` (read-only `restart_pending` = counter > 0). A
  resumed run compares its first mask with the snapshot's
  `stamped_mask` and continues the window; it does not restart
  otherwise, which keeps
  `test_a_resumed_run_keeps_its_exact_first_step`.
- **`start_from_dc_op`.** The DC solution has `i_C = 0` and `v_L = 0`
  exactly, so the zero slots are the right ones. The first step skips
  the restart when its mask equals the DC mask and no source jumps at
  `t_start`; such a run is bit-identical to today's until its first
  event. Otherwise the mask or jump trigger fires.
- **Other starts** (zero state, builder `i0=`/`c0=`,
  `initial_state=`): the first step restarts. A start that happens to
  be consistent pays an O(h²) backward-Euler error: prototype, the
  sine-driven RL control case goes from 4.28e-7 to 4.98e-7 relative at
  dt = 4 µs and stays second order.
- **Static path** (`cache.dt() == 0`): no dynamics, so no restart, but
  the same sampling: sample `k > 0` reads the gate at `t_{k−1}` and
  jumping sources (and declared `b_extra_fn` jumps) at their left limit,
  so adding a capacitor does not shift the recorded mask by one sample.

### D8. One pair, as published (superseded by D0: three SDIRK steps)

Marti & Lin use one pair. It removes the alternation exactly when the
jump completes inside the first half-step. A fast mode that is finite
but unresolved is attenuated by `1/(1 + h/(2τ))` per half-step, and
the remainder, about `4·C·τ·ΔV/h²` of current, rings on.
`p23_cda_pairs_proto.py`, 1 nF across 1 mΩ (τ = 1 ps), 100 V, on-state
switch current (exact 9.999 A):

| dt | today (engine, `p8_caploop.py`) | 1 pair | 2 pairs | 3 pairs |
|---|---|---|---|---|
| 1e-8 | −9.8 … +29.8 A | 9.995 … 10.003 | 9.999 | 9.999 |
| 1e-9 | −79 … +99 A | 9.61 … 10.39 | 9.999 | 9.999 |
| 1e-10 | not measured | −19.0 … +40.2 | 9.987 … 10.011 | 9.999 |

The cost of a second pair is accuracy: buck CCM error at T/100 is
5.50e-6 A with one pair, 1.10e-5 A with two, 1.64e-5 A with three, all
order 2.00. This change implements one pair. The residual is the case
`add-fixed-step-resolution-diagnostics` reports.

### D9. Opt-in, then default

`SimulationOptions::discontinuity_restart` (bool) selects the rule,
start-sampling and the left-limit source evaluation together. Off is
today's engine, bit for bit.

## Risks / Trade-offs

- Every golden baseline with an event moves. Mitigation: the flag, the
  reference harness, one re-baselining commit with the error table.
- A missed stage in one device gives a right matrix, a converging
  Newton and a wrong answer. Mitigation: per-device Jacobian identity
  and zero-slot oracle tests; the existing composition test extended to
  a restart step.
- Deciding diodes on a half-step state adds decisions. A turn-off whose
  zero crossing lies in the second half is chopped up to `h/2` early;
  the charge error per event is O(h²) (prototype: DCM buck `v_C` error
  −9.56e-4 … −3.71e-6 V at T/50 … T/800, order 2.0; −2.33e-4,
  −2.16e-4, −2.23e-4, −1.99e-4 V at N = 100, 101, 103, 107).
- A restart step costs two solves. Buck at T/100 with two edges per
  period: about 2 % more solves (estimate from the step count).
- Backward Euler damps: a restart at a point where nothing is
  discontinuous costs O(h²) once.
- Unresolved fast modes still ring after one pair (D8).
- User closures written against end-sampling shift by one step (D4).

## Migration Plan

1. Land with `discontinuity_restart = false`. Harness: every fixed-step
   golden hash unchanged.
2. Run the harness with the flag on. For every circuit with an analytic
   or `expm` reference the error must not grow; attach the before/after
   table to the pull request.
3. Flip the default in the next release that may change numbers.
   CHANGELOG **BREAKING** entry: what moves, the table, how to get the
   old numbers (`discontinuity_restart=False`), the gate-sampling shift
   and how to rewrite a grid closure.
4. Regenerate the fixed-engine goldens in one commit that touches
   nothing else. Re-pin Python tests that hold fixed-step numbers, each
   with its old and new value in the commit message.
5. Keep the flag for one release after the flip. Its removal belongs to
   `refactor-unified-transient-stepper`.
- Rollback: the flag.

## Open Questions

1. Coss charge test (D2): is the reading "the half-step offset is the
   defect, `Q = I·t` is the fix" accepted? If the engine shows a
   full-step offset instead, work stops and the result is reported.
2. One pair or two (D8)? One follows the paper and costs half the
   damping; two removes the residual at dt = 1e-9 on the capacitor
   loop.
3. Slot-zeroing instead of an explicit stage (D3)? It is the smaller
   diff and algebraically identical.
4. Which release flips the default, and does the flag-off path survive
   until the unified stepper?
5. Step-observer time label under start-sampling: keep `t_k` here and
   decide in `fix-control-tick-semantics`?
6. Should the consistent-initialisation solve (sample 0, exact
   derivative refresh) be opened as its own change?
