# Decisions — add-parameter-sensitivities (slice on the exact path)

Delegated design decisions, each as question → choice → evidence → how
to reverse. Slice: groups 1, 2, 3 and 7 on today's `dsed` exact stepper
(`python/pulsim/sensitivity.py`). Groups 4, 5, 6 and 8 wait for the
changes the proposal lists; until then every route that cannot give a
gradient refuses with `UnsupportedError` (cells in
`test_engine_conformance.py::test_sensitivities_cell`).

## 1. Where the exact path lives

→ Python (numpy) over the native adapter, as a post-pass of one engine
run, not a C++ header. The run is unchanged; the sensitivity reads the
run's samples and event log and re-derives every step. Evidence: the
DCM buck over 40 periods with five parameters costs one engine run and
12 ms; the migration plan of `design.md` puts this slice "in Python over
the `dsed` adapter". No hook in a forbidden file was needed. → Reverse:
port `_Mode` and `_propagate` to `dsed/sensitivity.hpp` when the
unified stepper (group 5) carries the recurrence.

## 2. `∂A/∂p`, `∂b/∂p` and the recovery derivative

→ From assemblies, exactly: every entry of `A`, `b` and of the
node-voltage recovery is a linear-fractional (Möbius) function of an R,
L, C or source value (a rank-one stamp inside a Schur complement, or a
row scaled by `1/C`, `1/L`). For a Möbius function the harmonic
combination of the two one-sided differences, `2ab/(δ(b − a))`, is the
derivative itself. It is formed at `δ = p/8` and `p/16` (four
assemblies per parameter) and the two must agree to 1e-7 of the block's
scale, else the parameter is refused (`feature="sensitivity_parameter"`).
Evidence: DCM buck `∂v_C/∂L`, `∂R`, `∂D` within 3.1e-8, 3.4e-8, 3.2e-8 of
the prototype (whose last printed digit is 3e-8). Alternative: a C++
derivative of `compute_lti_state_space` (`pwl/cache.hpp`, not
forbidden) — more code and a second extraction to keep in step. →
Reverse: add `compute_lti_state_space_derivative(mask, branch_id)` and
replace `_param_derivative`.

## 3. The propagator of the sensitivity, and its tiers

→ Augmented `Ab = [[A, b], [0, 0]]` in energy coordinates (`√C`, `√L`,
and the equilibrium's norm for the augmented row). Tier 1: eigenbasis
with `u·κ₂(V) ≤ 1e-10`, Daleckii–Krein with `F_ij = h e^{a} φ₁(b − a)`
(`a` the eigenvalue with the larger real part: no overflow on a stiff
mode, no cancellation for close ones). Tier 2: the block exponential
`exp([[Ab, 0], [dAb, Ab]] h)` (Van Loan 1978) by Padé [13/13] when
`u·‖Ab h‖₁ ≤ 1e-10`, with the coupling blocks balanced to `‖Ab‖` (a
similarity: unbalanced, `dA/dC ≈ 1e12` set the Padé scaling and the
state block lost 1e-7). Tier 3: refused naming the mode. Evidence:
critically damped series RLC (engine tier 2, κ = 1.9e8), 1e-3 above it
and 50 % above it (tier 1): `∂v_C/∂R, ∂L, ∂C` equal central differences
of the exact map within 1e-7. → Reverse: none needed; a third tier
(Schur–Parlett) would replace the refusal.

## 4. Which map is differentiated

→ The exact stepper with events located (`rtol = 1e-11`) and **no event
projection**: the map `periodic_steady_state` differentiates. Every
step is re-derived with the propagator of decision 3 and compared with
the engine's sample (1e-7 of the state's range); a step it does not
reproduce (a numeric step of a mode without an exact propagator) refuses
the run. Consequence: a `simulate(..., sensitivities=…)` run can differ
from a plain `dsed` run where the projection would act (fast modes at an
event); it is documented. → Reverse: differentiate the projection as a
reset map `R_x`, `R_p` (D3) and run with it on.

## 5. Events and cascades

→ The engine merges a zero-time cascade into one logged event (the
buck's gate-off and its freewheel diode's turn-on). `f⁻` is the old
mode's field, `f⁺` the final mode's, and `dτ/dp` comes from the
triggering event: the modulator for a gate edge (`dτ/dD = T` on a
falling edge of a trailing-edge PWM, 0 otherwise), the predicate for a
diode, `dτ/dp = −(∇g·S⁻ + g_p)/(∇g·f⁻)` with `g_p` from the recovery
derivative of decision 2. `S⁺ = S⁻ + (f⁻ − f⁺)·dτ/dp`.

## 6. The event rule: grazing refused, near misses flagged

→ A gradient exists only while the event pattern is locally constant.
Grazing (`|∇g·f⁻| ≤ 1e-9·‖∇g‖·‖f⁻‖` at a located event) is refused
(`feature="grazing_event"`, the diode named): one side of the parameter
removes the event and the derivative is one-sided. A near miss — an
armed predicate that comes within 1e-6 of its own range of zero without
crossing (an interior minimum, or a stretch end the diode did not cause,
such as the CCM valley that sits on the gate edge) — is computed (it is
exact for the pattern taken) and flagged: `sensitivity.flags` and a
`UserWarning`. Evidence: the buck bisected to the CCM/DCM boundary
(valley 1e-7 A) is flagged at the section; 5 Ω (CCM) and 50 Ω (DCM) are
not. → Reverse: refuse near misses too (`flags` becomes an error).

## 7. Parameter names

→ `"<component>.<field>"` with the short field (`R`, `L`, `C`, `V`) or
the `components()` name (`R_ohms`, `L_henries`, `C_farads`), and
`"<switch>.duty"`, accepted only when the switch's schedule is a
fixed-period trailing-edge PWM (checked from its edges). Switch
conductances, diode thresholds, integers and unknown fields are refused
naming the component and the field. More than 64 parameters are
refused. → Reverse: named parameters with `add-immutable-circuit-ir`.

## 8. KPIs

→ Closed forms of the exact trajectory per step (modal integrals with
φ₁ and its divided differences, the second divided difference written as
a Taylor series about the midpoint when the points are within 1e-3), plus
the event terms `(y⁻ − y⁺)·dτ/dp` of a signal that jumps. Value and
gradient are of the same quantity, so a central difference of the value
agrees to rounding. RMS, peak, valley and ripple over a tier-2 mode are
refused in this slice (they need the eigenbasis). Peak / ripple carry
`non_unique` when the extremum is attained at two instants. → Reverse:
the Kronecker block exponential for the quadratic KPIs on tier 2.

## 9. Eigenproblem order

→ The eigen-decomposition of tier 1 is computed on the matrix permuted
by descending row norm (a similarity, undone on `V`). Evidence: the DCM
buck's idle mode (rows -425 and -5e12 1/s) had its slow eigenvalue
2.7e-4 off (`u·‖A‖`, LAPACK deflating from the bottom with the stiff row
last), which cost 1e-9 of `v_C` per step: invisible to the 1e-7 step
check, but central differences of the KPI values were noise at 3.6e-6.
Graded: exact to 1e-15 against a 50-digit `expm`, and every KPI of the
DCM buck within 7e-9 of central differences. → Reverse: a Schur-based
propagator (tier 3) would not need the order.

## 10. Extremum search

→ Sub-grid scan (16 cells per step) with the end cells included, Newton
on `y'` within the neighbouring cells, a root kept only when it
converges inside the step; the step ends are always candidates (with
`y'·dτ/dp` when they are events). Evidence: the orbit run (`T/50`
steps) put the `v_out` peak and valley in end cells; read at the step
end, the ripple's duty gradient was 1.8e-3 off; scanned, within 1e-6 of
central differences of the PSS. → Reverse: none.

## 11. `Evaluator(gradients=…)`

→ A map `{design variable: "<component>.<field>"}` (a list of names that
are both is accepted), not `True`: the evaluator cannot know how
`build` turns a variable into a circuit value. A variable whose value
differs from the circuit value (a scaled map, e.g. µH) is refused rather
than given a scaled gradient. Gradients of `v_out`, `ripple_v` and
`ripple_i.<L>` only (the losses and the efficiency go through
`device_loss_summary`, which is not differentiated), on the `pss` route
only; those KPI values become the closed forms the gradients belong to.
Transient route and thermal loop: `UnsupportedError(feature=
"gradients")`. → Reverse: chain rule through a user `d value/d variable`
when `add-immutable-circuit-ir` names parameters.

## 12. Review (fresh-context reviewer, read-only) and what changed

The event-time formula, the jump, the predicate rows, `g_p`, the masks of
`f-`, `f+` and the Moebius / Daleckii-Krein / Van Loan algebra were
confirmed. Found and fixed (each with a test in
`test_sensitivity_review.py` unless noted):

1. `.duty` accepted a dead-time pair (the duty also moves the other
   switch's edge): now refused unless no other switch has edges.
2. Extremum search: the sub-grid follows the fastest oscillation (8
   points per period); a strict sub-grid extremum whose stationary point
   is not found is refused; a bracketed one is found by bisection on
   `y'` (Newton left the cell on the stiff idle mode).
3. Numeric (RK45) steps of the stepper are refused
   (`n_exact_steps != n_accept`); the step check is 1e-10, not 1e-7.
4. The Moebius check is per row, against 1024 u of the row's assembly
   (64 u refused a 20 mOhm series resistor; one scale for the whole
   vector let a 1 V non-Moebius entry through).
5. The orbit's sensitivity period must reproduce the solver's events
   and `x(T) = x*` (1e-7 of the state scale), else refused.
6. Near misses: interior minima of `|g|` are refined by Newton on `g'`
   (tier 1) and the scan follows oscillations; tier-2 modes are scanned
   with the block exponential (`test_sensitivity_exact`).
7. `Evaluator(gradients=)` drops a `non_unique` KPI's gradient
   (`counters["grads_refused"]`); the two ends of an orbit are one point,
   not a tie.
8. `Evaluator(gradients=)` refuses a variable that moves more than its
   one field (rebuild with the variable moved, compare `components()`).
9. A window edge or `final` on an event that moves with the parameters,
   where the signal jumps, is refused (one-sided). The grazing cosine is
   taken in energy coordinates. Two events with different `dtau` on one
   sample refuse an extremum sitting there.

## Open questions of `design.md`

1. Forward mode with a 64-parameter guard: kept (the optimisation use
   case is a handful of design variables).
2. Naming: strings now (decision 7).
3. Sensitivities in the error norm: moot on the exact path (no step
   controller error); decide with group 5.
4. Temperature as a parameter: with `add-in-kernel-event-losses`.
5. No `pulsim.optimize`: `value_and_grad` plus the worked example.

## Hand-off (2026-11-19)

- Saltation checked against the formulation: `_jump` forms
  dτ/dp = −(∇g·S + g_p)/(∇g·f⁻) and S⁺ = S⁻ + (f⁻ − f⁺)·dτ/dp (f⁻ under
  the old mask, f⁺ under the cascade's final one); a trailing gate edge has
  dτ/dD = T. Orbit gradients solve (I − M)·dx* = ∂P/∂p with the solver's M.
  `g_p` has no −∂V_th/∂p term (V_th is refused as a parameter).
- Validated against 50-digit truths (shooting probes
  `boost_dcm_gradient_truth.py`, `buck_dcm_gradient_truth.py`): every
  orbit gradient within 2.7e-8 relative (C) and 1.3e-11 (L, R, D).
- A state-event turn-on with a field jump (V_th = V_f + 1e-3, resonant
  rectifier) matches central differences of the exact map to 1.0e-7.
- Open, with the reason: 4.x (`Params<S>` on the device interface), 5.x
  and the TR-BDF2 / fixed-engine parts of 6.x (unified stepper, loop
  freeze), 6.1 in the stepper (event queue), 8.x (controller gains through
  the control graph: not started), 9.x.

## 13. Hand-off (2026-11-19, after the merge of the exact engine 2b1d808e)

- 4.1 / 4.2 are done on the device-interface track (`Params<S>` on every
  law, `device::seed_params`, `check_param_tangent` at 1e-8 against a
  fourth-order central difference for every parameter of every kind).
  Marked done here with the references.
- 4.4 (new, open): a circuit gradient THROUGH a Newton law needs an
  engine that runs the law and differentiates its step. The exact path
  refuses Newton kinds (`kKindCaps`: `FixedTrap | TrBdf2` only); the
  design's path is D1 in the stepper (group 5): LOOP FREEZE. Nothing on
  this branch differentiates a Newton device; the reference
  (`_sens_reference.shockley_run`, trapezoidal + Newton in numpy) stays
  the only one.
- 8.x open. The tick saltation is u⁺ = g(x, x_d, p) ⇒
  S_u⁺ = ∂g/∂x·S_x + ∂g/∂x_d·S_xd + ∂g/∂p, with the held source value u
  an extra state of the segment propagator (u' = 0). `g` is the compiled
  graph's tick; it has no derivative today (`block_laws.h` doubles only).
  REQUEST to the control-graph track: block laws instantiated on the AD
  scalar (as `device/laws.hpp` did), or a per-kind Jacobian of the tick
  map, exposed to Python. A finite difference of the tick would be the
  low-order FD of the same code this change exists to replace.
  Also needed here: `sensitivity._run` passing `tick_fns` /
  `tick_periods` to `run_ped_native_builder`, and modes keyed by
  (mask, held source values).
- `orbit_sensitivities` runs its period at dt_max = T (was T/50): the
  exact stepper's settle at T/50 picks the wrong diode branch near the
  boost DCM orbit (shooting DECISIONS 22, request to the exact track).
- 9.1 sweep guide waits for group 6; 9.2 is the verification of each
  push (ruff, pytest, the touched C++ targets; full ctest on the
  integration tree).
