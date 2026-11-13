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

## Open questions of `design.md`

1. Forward mode with a 64-parameter guard: kept (the optimisation use
   case is a handful of design variables).
2. Naming: strings now (decision 7).
3. Sensitivities in the error norm: moot on the exact path (no step
   controller error); decide with group 5.
4. Temperature as a parameter: with `add-in-kernel-event-losses`.
5. No `pulsim.optimize`: `value_and_grad` plus the worked example.
