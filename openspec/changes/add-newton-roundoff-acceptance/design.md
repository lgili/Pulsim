## Context

`solve_with_newton_b_extra` (`pwl/nonlinear_solve.hpp`) iterates
`J(x)·dx = −f(x)`, `f = J_lin·x + b_constant + b_extra + f_nl(x)`, and
returns when `‖dx‖∞ < tol_dx` and `‖f‖∞ < tol_res` (`:394`). A second
branch (`:416`) accepts a residual that is small against the largest
entry of `x` and the right-hand side, but only once the `dx` test has
passed. Both tolerances are absolute on purpose (owner decision; five
tests pin the DC cascade's fall-through).

At a transient step an inductor's row is
`v_a − v_b − (2L/h)·i + (2L/h)·i_n + v_n = 0`. Its terms are of size
`(2L/h)·|i|`; their sum is rounded to a multiple of
`ulp((2L/h)·|i|) ≈ ε·(2L/h)·|i|`. A node voltage that is determined
through that row is therefore known only to that quantum. When the
quantum exceeds 1e-9 V the absolute `dx` test cannot pass.

Evidence: `p7b_roundoff.py` (engine), and for this proposal
`p21_roundoff_proto.py` (numpy). What the prototype shows and does not
show is stated in D5.

## Goals / Non-Goals

- Goals: a transient solve whose iterate is exact to working precision
  is accepted and says so; neither engine commits a step below the
  size at which the solve is noise; the DC cascade is unchanged.
- Non-Goals: changing `tol_newton_dx` or `tol_newton_res`; a relative
  step test; the diode-mask chatter inside Newton (a bouncing iterate
  of 0.76 V at a small residual, a different defect); Newton
  engineering (factor reuse, chord iterations, Armijo), which is
  `update-newton-persistent-solver`; a promise about the documented
  stall.

## Decisions

### D1. The criterion

The criterion is evaluated only where the solve raises today: when
the iteration budget is exhausted without the absolute test passing.
With `k` the last iteration and `dx_k` computed from `f(x_k)`, the
solve returns `x_k + dx_k` instead of raising when all of these hold:

1. **Stagnation.** `‖dx_k‖ ≥ ½‖dx_{k−1}‖` and
   `‖dx_{k−1}‖ ≥ ½‖dx_{k−2}‖`: the last two iterations did not
   contract.
2. **Row-wise backward error.**
   `ω = max_i |f_i(x_k)| / s_i ≤ 8·ε`, with
   `s_i = (|J_lin|·|x_k|)_i + |b_constant,i| + |b_extra,i| +
   (|J_nl|·|x_k|)_i + |f_nl,i|`,
   the sum of the magnitudes of the terms row `i` is made of. A row
   with `s_i = 0` must have `f_i = 0`.
3. **Sanity ceiling on the step.**
   `‖dx_k‖∞ ≤ 1e-6·max(1, ‖x_k‖∞)`.

Condition 2 is the Oettli–Prager componentwise backward error: `x_k`
solves exactly a system whose every coefficient is perturbed by at
most `8ε` relative. Equivalently it is the residual test on the system
with each row divided by its own scale, the row equilibration the
owner's note asks for.

The numbers:

| Case (engine, `p7b_roundoff.py`) | row residual | `ω` |
|---|---|---|
| 100 µH, h = 1e-11 | 3.725e-9 = 2^-28 | 0.47 ε |
| 100 µH, h = 1e-12 | 2.980e-8 = 2^-25 | 0.37 ε |
| 100 µH, h = 1e-13 | 2.384e-7 = 2^-22 | 0.30 ε |
| 10 mH, h = 1e-9 | 3.725e-9 = 2^-28 | 0.47 ε |
| 10 mH, h = 1e-11 | 2.384e-7 = 2^-22 | 0.30 ε |

(`ω` computed as residual / `(2·(2L/h)·|i|)`.) The threshold `8ε` is
16 times the largest of them. In the prototype's control, an iterate
that is still converging sits at `ω ≥ 1e14 ε`.

Why condition 3. A near-singular Jacobian can turn a roundoff-sized
residual into a large `dx`. The mask-chatter failure is such a case
(`‖dx‖` = 0.76 V, residual 5.4e-10, on a node whose row contains
`g_on·v` terms of about 1e5: `ω` there could be as low as 24 ε, only
three times the threshold; an estimate from the reported numbers, not
a measurement). The ceiling keeps that failure a failure: 0.76 V
against 1e-6·100 V. The stalls this change is for have `‖dx‖` of at
most 1.2e-7 V on a 10 V circuit and 8.8e-8 V on a 48 V circuit.

Evaluating at the budget, not as soon as the conditions hold, is what
makes a solve that converged before return exactly what it returned
before. Its price is the full budget (50 iterations) on every solve
that ends this way; task 3.6 measures it.

- Alternatives considered:
  - *Accept as soon as the three conditions hold.* A few iterations
    instead of 50, but a solve that used to reach the absolute test
    after a short stagnation would return a different iterate, and the
    goldens would move. Left as a follow-up if 3.6 shows the cost
    matters.
  - The three scale-aware step tests of the owner's note. Refuted by
    measurement; none is used. This criterion differs from all three:
    it does not rescale `tol_dx`, it is per row and at roundoff level,
    and it is not offered to the DC cascade.
  - A relative residual test (`‖f‖ ≤ tol·scale`). Exists at `:416`,
    gated behind the absolute `dx` test, so it cannot fire here.
  - Scaling the Jacobian's rows before factoring. Changes the pivots
    of every solve, including the ones that converge today.
  - Iterative refinement in extended precision. Fixes the quantum and
    costs a second residual evaluation in `long double` per iteration;
    not portable to MSVC's 64-bit `long double`.

### D2. Where it applies

`solve_with_newton_b_extra` takes a new trailing argument,
`NewtonRoundoffAcceptance{bool enabled = false; Real omega_max =
8ε; Real dx_rel_max = 1e-6;}`, and reports through an output record
whether the return was a roundoff acceptance. Callers:

| Caller | Passes it |
|---|---|
| `run_transient.hpp:989`, `:1008` (dynamic path, nominal and retry) | yes |
| `trbdf2_transient.hpp:716` (stages, probes, landings) | yes |
| `run_transient.hpp:1647` (static path, no companion rows) | no |
| `pwl/continuation.hpp:88`, `pwl/dc_strategy.hpp:200`, `pwl/dc_assemble.hpp:565` (DC) | no |

Default off means the DC cascade compiles to the code it has today.

### D3. The step floor

Three regions, by the size of the row quantum
`q(h) = ε·(2L/h)·|i|`:

| Region | Condition | Behaviour |
|---|---|---|
| A | `q ≤ tol_newton_dx`, i.e. `h ≥ h_roundoff = 2·ε·L·|i|/tol_newton_dx` | the absolute test is reachable; nothing changes |
| B | `tol_newton_dx < q ≤ 1e-6·max(1, ‖x‖∞)` | the absolute test may stall; D1 accepts and records |
| C | `q > 1e-6·max(1, ‖x‖∞)`, i.e. `h < h_noise = 2·ε·L·|i|/(1e-6·max(1, ‖x‖∞))` | the solution is noise above 1e-6 relative; D1's ceiling refuses it; no step is committed here |

`h_roundoff` is where the measured failures start: 100 µH at 0.9 A
predicts 4.0e-11 s and the run fails between 3e-11 and 1e-11 s; 10 mH
predicts 4.0e-9 s and fails between 1e-8 and 1e-9 s. `h_noise` is the
floor. For the 10 mH case (`‖x‖∞` = 10 V) it is 4.0e-13 s, and the
engine's own output there shows the noise: at 1e-13 s the divider node
jitters by 6e-5 V, 6e-6 of the solution.

The floor is `h_noise` and not `h_roundoff` because `h_roundoff`
grows with the stored flux `L·|i|`: 1 H at 100 A gives 4.4e-5 s, which
would force accepts at steps a 20 kHz converter needs. `h_noise` for
the same inductor on a 400 V circuit is 1.1e-10 s.

- Variable-step engine: the lower bound of a committed step of a
  Newton circuit becomes `max(h_min, h_noise)`, with `h_noise` from
  the committed state (linear and saturable inductors, `L(i)` for the
  latter). A step the controller wants below it is taken at it; if its
  estimate exceeds 1 it is a forced accept, already counted in
  `n_forced_accepts`, and `n_roundoff_floor` counts how often the
  floor was the reason. Probe solves commit nothing and are exempt.
- Fixed engine: the dt-retry ladder stops at the first level whose
  sub-step is below `h_noise`. A user `dt` below `h_noise` at the start
  of the run is refused with a message that names the inductor and the
  bound.
- Only Newton circuits are concerned. A linear circuit has no
  convergence test and its floor stays `h_min`.

### D4. What is recorded

`SimulationResult::newton_roundoff_accepts`:
`{count, t_first, worst_dx, worst_row}`, and the Python layer warns
once per run when the count is non-zero, naming the row
(`pwl::row_label`). `TrBdf2Stats::n_roundoff_floor`. An acceptance is
never silent.

### D5. What the prototype shows

`p21_roundoff_proto.py` re-implements the circuit of `p7b_roundoff.py`
with a dense pivoted LU. It does **not** reproduce the stall: the
iteration reaches a floating-point root and converges at every step
size. The stall depends on the engine's own order of summation and
elimination. The prototype therefore only supports two statements: the
`ω` of a converged iterate is at most 5.3 ε once the step is below the
predicted floor, and the `ω` of an iterate that is still moving is at
least 1e14 ε. The criterion has to be validated in the engine, which
is what tasks 1 and 3 are for.

## Risks / Trade-offs

- Accepting a wrong iterate. Mitigation: three conditions, of which
  the backward error is at roundoff level; the record; the control
  tests of task 3.3 (mask chatter must still abort; a slowly
  converging solve must not be accepted).
- Disturbing the DC cascade. Mitigation: off by default; the five
  fall-through tests are in the verification set; a test asserts the
  DC callers do not pass the argument.
- Cost. A solve accepted this way has spent its whole iteration
  budget. The row scale needs `|J|·|x|`, one sparse product, computed
  once at the end.
- The floor forces accepts. A forced accept is counted and warned;
  `h_noise` is far below any step a tolerance asks for (D3).
- The documented stall may have another cause. Then this change still
  fixes `p7b_roundoff.py` and the stall's row stays in the known
  limitations.

## Migration Plan

Land the acceptance and the floor as two commits. Both are on by
default for transient solves: a solve that converged before does not
reach the new branch. `SimulationOptions::newton_roundoff_acceptance`
(default true) turns the acceptance off for a bisect. No golden moves
unless a previously aborting circuit is in the harness.

## Open Questions

1. Thresholds: `8ε`, the `1e-6` ceiling and two iterations of
   stagnation are set from the measured `ω ≤ 0.47 ε` and
   `‖dx‖/‖x‖∞ ≤ 1.2e-8`. Does the owner want them configurable or
   fixed?
2. If the stall's reproducer passes, should
   `test_engines_agree_below_saturation` assert agreement between the
   engines (15.31 V on the fixed engine) or only completion?
3. A user `dt` below `h_noise` is refused here ("refusals instead of
   silent fallbacks"). The 10 mH case at 1e-13 s completes today, with
   6e-5 V of jitter; is a refusal there acceptable, or should it be a
   warning?
4. The review asked for the floor at `2·L·ε·|i|/tol`. This design puts
   it at `h_noise`, 1000× lower on a 1 V circuit, for the reason in
   D3. Is that the intended reading?
