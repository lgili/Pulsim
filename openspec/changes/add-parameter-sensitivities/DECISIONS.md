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

## 14. Closed-loop sensitivities on the exact path (2026-11-19)

- Scope: one rate group driving graph PWM gates (`bind_pi_to_switch`,
  `ControlGraph` gates) through duty registers; probes read node
  voltages. Parameters: `<block>.<field>` of the graph (kind "control")
  plus every plant parameter as before. Refused by name: `<switch>.duty`
  in a loop (the duty is a graph output), a graph gate without its ticks
  (would run open), a tick-moving event, a live kink within 1e-9.
- At tick kT (before the sample's events): S_c⁺ = J·[S_c; S_sig;
  R S + dR xb; e_p] with J from `tick_jacobian` (control-graph G34); the
  duty rows give S_d. At the gate's OFF edge t = phase + (k + d)T the jump
  uses dτ = (∂t_off/∂d)·S_d = T·S_d (`CarrierPwm::edge_duty_derivative`,
  rising edge 0), with the existing S⁺ = S⁻ + (f⁻ − f⁺)·dτ. Sign: a larger
  duty delays the OFF edge, dτ > 0. Control: setting edge_dtau to 0, or
  blinding the probe columns, breaks agreement by > 1% (tests).
- Discrete-time saturation has no instant to shift: a clamp is decided at
  a tick, and its derivative is one-sided (0 for Kp, Ki, setpoint while
  clamped, the limit's own column 1). Central differences of the exact map
  agree on both sides (ticks 3 / 6 / 8 / 50 / 150 of a run clamped for
  ticks 0–4).
- `pi.setpoint` moves the pi block's constant input only; the
  `error` subtract's `a` is a separate parameter (`error.a`).
- Measured (macOS arm64): closed-loop buck to 1 ms, AD vs central
  differences of the exact map, max relative error at h/p = 1e-2…1e-7:
  Kp 6.1e-5, 6.1e-7, 6.3e-9, 4.8e-9, 7.5e-8, 9.2e-7 (O(h²) then
  roundoff); at 1e-4 / 1e-5: Ki 2.8e-9 / 6.8e-10, setpoint 4.1e-10 /
  1.9e-10, L 7.7e-9 / 2.0e-9, R 9.5e-9 / 3.3e-9. Tests pin 1e-5, assert
  1e-6. Steady state (20 ms, tick 1999): v = 5.000000000000, dV/dVref =
  1.000, dV/dKp = −4.2e-14, dV/dKi = −7.3e-19, dV/dL = −3.6e-9, dV/dR =
  −3.5e-15 (closed form 1, 0, 0, 0, 0).
- Open: 8.2 (a graph-driven held source value as a sensitivity state —
  the exact path closes the loop through gate edges only); closed-loop
  `orbit_sensitivities` / Floquet multipliers with the controller state
  (goal 4, not started: the period map must carry (x, x_c) and the tick
  Jacobian is the controller block of M); TR-BDF2 closed-loop
  sensitivities (stepper group 5).
- Stage recurrence for Newton laws (not in scope, recorded): the exact
  path refuses Newton kinds; a closed-loop gradient through one needs the
  stepper's per-stage S recurrence S_{n+1} = −J⁻¹(∂F/∂x_n S_n + ∂F/∂p)
  with the device Jacobian at the converged stage (`Params<S>` seeds
  exist), the stage's own tick and edge terms ordered as here, and the
  converged Newton J factor reused for the back-substitutions.
- Fresh-context review (2026-11-19). Fixed: (1) coinciding OFF edges of
  several legs added their shifts (2T·S_d for two switches on one duty):
  now applied once when identical, refused when they differ (the event
  splits); (2) the kink test was relative only, so `PL_GT(err, 0)` at
  err = 3e-17 passed: floor 1e-12 absolute. Open: the PLL wrap notes only
  the post-wrap side (θ just below 2π is not flagged; the fix
  `PL_KINK(two_pi - fabs(theta), 0)` after the fmod waits for a rebuild —
  the machine was out of disk); "live" means any seeded column separates
  the operands, not the requested parameters (a sample_hold on its own
  period is refused loudly, never wrong); a kink inside `pl_clamp` is
  reported as `pl_clamp`, not the calling law; the replay ticks at
  offset + kT, not the run's jittered instant (the OFF-edge check catches
  a drift).

## 15. The stage recurrence on pwl and trbdf2 (group 5, 4.4; 2026-11-20)

- Where: one header, `core/include/pulsim/solver/step/sensitivity.hpp`
  (`step::StageSensitivity`). Every scheme describes a stage by its ψ
  coefficients only (`StagePsi`: q̇ = a·(q − ψ), ψ a combination of q_n,
  q̇_n and the earlier stages' q, q̇): `psi_trapezoidal`,
  `psi_backward_euler_on`, `psi_bdf2_stage2`, `psi_sdirk_stage2`,
  `psi_esdirk43`. The engines call `start / begin_step / stage / commit /
  finish`; they never hand over their history objects, so the engines'
  state is not touched (states bit-identical, tested). `core/tests/step/
  test_sensitivity.cpp` checks each ψ against the scheme's own stage
  derivative (ESDIRK: the Butcher rows on y' = λy).
- Per stage: J·S = a(dψ/dp − ∂q/∂p) − ∂f/∂p; dq = q_x·S + ∂q/∂p,
  dq̇ = a(dq − dψ). q_x = C_split/2 from `SegmentAssembler::split` (the
  engine matrix is G + C_split/factor_dt, a = 2/factor_dt). ∂f/∂p and
  ∂q/∂p: R (−G²·v), C (v), uncoupled L (−i, sign read from q_x), and the
  four static Newton laws through `device::seed_params` (the law's AD
  tangent, field names `L::kParamNames`). Start: S = 0, dq = ∂q/∂p(x0),
  dq̇ = 0 (every engine starts from zero derivative slots).
- No added factorization (5.5): a linear stage solves on the cache's
  resident factor (`SegmentCache::resident_factor`, new: no build, no
  refactor, no LRU touch); a Newton stage on the workspace factor of its
  last iteration (J(x_k), X = x_{k+1}) and refines r = rhs − (J_lin +
  J_nl(X))·S to `tol_newton_dx` relative (5.2); J_nl(X) from the
  interface static refresh into a private matrix.
- Counters (5.3): `n_sens_solves` and `n_sens_iterations` count the
  accepted steps' back-substitutions, so `n_sens_solves = n_p × stages ×
  accepted steps` holds on any linear run; a rejected attempt's work goes
  to `n_sens_discarded` (TR-BDF2 RC at rtol 1e-8: 28 = 7 rejects × 2
  stages × 2 params). Measured: Shockley rectifier on pwl, 20 000 solves,
  20 620 iterations (≤ 2 per stage holds, 1.03 mean).
- Contiguity, not a list of cases: `begin_step` and `finish` require the
  step to start from the bits this object committed. Any step the
  recurrence did not produce (an event landing, a zero-time settle, a
  split, a retry, a restart, the freeze guard) fails that check and is
  refused by name (`UnsupportedError(engine, "sensitivities")`, "group
  6"); a mode change between stages is refused the same way. On pwl the
  step's own flags (split, restart, retry, events, iters ≠ 1) refuse
  before. Events are group 6.
- Snapshot: the sensitivity state is NOT carried; `resume_from` with
  sensitivities is refused by name on both engines (tested). If the
  integration batch wants it, it is a new appended block (S, dq_n, dq̇_n,
  t_last, x_last) — the block index is left to that batch. Also refused:
  pwl `start_from_dc_op` (x0 would depend on p), the static pwl path,
  stateful kinds (nonlinear C, saturable L, Lauritzen, machines,
  external, IGBT L1 — a tail history), a coupled winding's L, > 64
  parameters, and in Python a run that does not go through `Simulation`
  (callbacks, closed loops, control ticks).
- Truths (macOS arm64, Eigen 5):
  RC at t = RC on TR-BDF2 rtol 1e-8 vs the closed form: 8.1e-8 (R and
  C); on ESDIRK43: 2.1e-13 / 3.0e-13. Shockley rectifier on pwl (dt
  1e-5, 40 ms) vs central differences of the same scheme: R 4.6e-9, C
  3.7e-11, Rl 4.2e-10, I_S 1.27e-7, n 4.3e-9 (prototype worst 3.4e-7;
  test 1e-6). Shockley on TR-BDF2 rtol 1e-9, 10 ms, vs central
  differences of scipy Radau rtol 1e-12: R 6.8e-6, C 3.0e-8, Rl 3.6e-9,
  I_S 1.3e-7, n 6.8e-7; ESDIRK43: 1.2e-7, 1.8e-10, 1.0e-10, 3.8e-8,
  3.7e-9 (test 1e-4: TR-BDF2's own truncation error of the state is
  the floor there, not the recurrence). MOSFET L1 (K, V_T) on pwl vs
  central differences: within 1e-6.
- 4.4 partial: the saturable inductor (L_0, I_sat) keeps a flux history
  and is refused by name; Shockley and MOSFET L1 are covered.
- Review (fresh-context, opus, read-only, 2026-11-20). Fixed: in-kernel
  losses (heat currents and capacitor-slot writes from earlier states),
  `enable_newton_lm` and an LM auto-promotion (the factor is J + λI or
  null: `newton_factor` refuses), a caller's `nl_refresh` override, the
  trbdf2 slot-only derivative refresh near t_end / a tick, `mmc_arms` and
  `on_cache` (Python), and the kernel `Simulation(engine='dsed')`
  silently ignoring the names — each now refused by name and tested
  (losses, LM, on_cache). Each of these changed the stage equations or
  the slots without moving (t, x), which is why contiguity alone missed
  them. The pwl Newton hook reads the J_lin of the solve it follows
  (it no longer touches the cache's LRU clock). Open: an event probe of
  `ed_stages` overwrites the stage buffers and the later refusal names
  "taken from a state the recurrence did not produce" rather than the
  event; `enable_substep_state_correction` refuses every step.
- Off-path cost (instructions per accepted step, proc_pid_rusage, min
  over 3 alternating processes × 8 runs, base b9e46778 and tip both
  built LTO off — the project default turns LTO on, and an LTO base
  read as a false 1–2 % regression): buck_pwl +0.18 % (min of 6
  processes; 3 processes gave +0.54 %, the process-to-process spread is
  ≈ 0.5 %), buck_trbdf2 +0.20 %, rlc152_pwl +0.33 %, rlc152_trbdf2
  −0.35 %, shockrect51_pwl +0.09 %, shockrect51_trbdf2 0.00 %,
  mosl1_pwl +0.04 %, mosl1_trbdf2 +0.03 %. The hooks are `[[unlikely]]`
  calls into out-of-line members.

## 16. Events on the time-stepping engines (group 6, 2026-11-20)

Derivation (checked here, not copied). Each mode is the index-1 DAE
`d/dt q(x) + f(x, t) = 0` in the engines' conductance form (`f = G·x − b(t)`,
`q = C/2·x` for the linear stamps the recurrence admits; the engine matrix is
`M = G + a·C/2`). At an event instant τ(p) the charges and fluxes q are
continuous (no reset: identity for a PWL diode or a switch with R_on > 0).
For t > τ, `q(t, p) = Φ⁺(t; τ, q(τ⁻, p))`, so

    dq/dp(τ⁺) = dq/dp(τ⁻) + (q̇⁻ − q̇⁺)·dτ/dp,        (Φ⁺_τ = −q̇⁺ at t → τ⁺)

and from the event's condition `g(x(τ), τ, p) = 0`

    dτ/dp = −(g_x·S⁻ + g_p)/(g_x·ẋ⁻ + g_t).

What the recurrence carries is (dq/dp, dq̇/dp) of the step start, so the jump
is applied to dq/dp; dq̇/dp after the event is not formed (every engine's
first post-event stage is backward-Euler-like, ψ = q_n with no q̇_n term — a
stage that would read it is refused); the algebraic components of S follow
from the first post-event stage solve, as the design says.

- q̇⁻ = −f⁻(x⁻, τ) exactly: the stage relation `q̇_stage = a(q(X) − ψ)` IS the
  residual of the stage equation, so the object mirrors the engine's own
  charge slots (a value recurrence next to the sensitivity one, no solve).
- q̇⁺ = −f⁺(x⁺, τ) at the CONSISTENT post-event point x⁺ = x⁻ + δ, with
  `C/2·δ = 0` and `f⁺(x⁺) ∈ range(C/2)`. With `δ = M⁺⁻¹(−f⁺(x⁻) + C/2·w)`
  the second condition holds for every w, and the first is the small bordered
  system `(C/2·M⁺⁻¹·C/2)·w = C/2·M⁺⁻¹·f⁺(x⁻)` on the rows C touches (dense,
  column-pivoted QR). `f⁺(x⁻) = f⁻(x⁻) + (G⁺ − G⁻)·x⁻ − (b_c⁺ − b_c⁻)`: the
  common time-varying sources cancel, `f⁻(x⁻) = −q̇⁻`. M⁺ is the first
  post-event stage's own factor (any a works: the formula is exact for every
  a). The same bordered solve gives ẋ⁻ (all components, algebraic ones
  included): `C/2·y = q̇⁻`, `G⁻·y − ḃ ∈ range(C/2)`, on a pre-event factor;
  ḃ(τ) by a central difference of the source rows (ε = 1e-7·τ).
- Cost at an event with dτ/dp ≠ 0: (rows of C) + 1 back-substitutions per
  bordered solve and one dense QR of that size; NO factorization of the
  circuit matrix (the gate). An event with dτ/dp = 0 costs nothing.
- Kinds: a gate edge of a fixed schedule has dτ/dp = 0 (no jump; carried
  even without event splitting: the mask sequence does not move with p, so
  the discrete map is smooth). `<switch>.duty` of a fixed-period
  trailing-edge PWM (checked in Python by `_duty_period`, passed to C++ as
  `<switch>.duty#T=<period>`): dτ/dD = T at its OFF edge, 0 at its ON edge
  (g_x = 0). A PWL diode crossing: g = v_a − v_k − V_on (OFF) or v − V_f /
  i_s (ON), slope 1 in the read voltage; g_p = 0 (V_th, V_f are not
  parameters).
- Location accuracy. pwl with event splitting locates the crossing by its
  linear interpolation (O(h²) in τ) and steps to it; the jump is formed at
  that τ. Without splitting an event sits on the grid: x(p) is a staircase,
  the jump would carry an O(h) error of τ that no correction with a measured
  sub-step fraction removes consistently (the state itself was computed as if
  the event were at the grid point), so a mode change that was not located is
  REFUSED naming the device. TR-BDF2 lands its steps on the located event.
- S⁺ is re-initialised consistently too (one more bordered solve per
  parameter on the same QR): C/2·S⁺ = dq⁺ − ∂q/∂p, G⁺S⁺ + ∂f/∂p ∈ range, and
  dq̇/dp⁺ = −(G⁺S⁺ + ∂f/∂p), so a trapezoid after the event (trbdf2's
  gate-edge probe) reads a consistent slot. An event with dτ/dp = 0 is NOT
  re-initialised: the recurrence then mirrors the engine's own discrete map.
- Measured (macOS arm64, Eigen 5), all within the tests' 1e-5 relative:
  diode turn-on (V_th 0.7, V_f 0) vs its closed form (complex step on the
  piecewise solution), R and C, on pwl+split (dt 1e-8) and trbdf2 (rtol
  1e-10): pass; the dτ/dp term is > 50 % of each answer (asserted;
  ≈ 94 % for R). One RC pulse, `S.duty`, `R.R`, `C.C` vs the closed form on
  pwl+split (dt 1e-6) and trbdf2: pass; d v/d D is entirely the jump term.
  DCM buck (task 6.3 table, 40 T) on pwl+split at dt = T/1000: ∂v/∂L, ∂R,
  ∂D within 1e-5: pass. Fixed gate edges without splitting vs central
  differences of the same scheme: 1e-6, pass.

### State at hand-off (2026-11-20, context budget reached)

Branch feat/core-sensitivity-events. OPEN, each a strict xfail or a
refusal by name:
1. trbdf2 DCM buck: its turn-off is an INTERPOLATED landing (secant between
   two bracket states, no solve): refused by name. Next step: differentiate
   the interpolation (S_land = S_a + θ(S_b − S_a) with S_b a trap stage on
   the trial's factor, captured when the trial is solved) — or land with a
   solve when sensitivities are on.
2. ~~`test_moving_event_with_a_newton_device_is_refused` returned a
   number~~ — CLOSED, not a hole (see "Audit" below): its Shockley diode
   (I_S 1e-14, n = 0.05, n·V_T = 1.3 mV) clamped v_c at 36 mV, the PWL
   diode never reached 0.7 V, no event happened, and the number was right
   (3.4e-9 vs central differences of the same scheme). With n = 10 the
   event happens and the refusal fires, naming D.
3. trbdf2 crossing within the conditioning floor (probe flip at t),
   ESDIRK landings, Newton-circuit moving events: refused by name.
4. 4.4 saturable inductor: its flux λ(i) is a nonlinear q with its own
   history (`sat_history`), outside the recurrence's linear q = C/2·x. It
   needs q(X) and ∂q/∂x from the law at each stage, its history checked to
   follow the scheme's ψ on every integrator, and ∂λ/∂(L_0, I_sat) by
   `seed_params`; not started.
Not run: cost gate, byte compares, full pytest, C++ targets, stubs (no
binding signature changed), openspec validate, the fresh-context reviewer.

### Audit (2026-11-20, second session): no third outcome

The rule: a step with sensitivities commits only through the recurrence (+
the jump) or a refusal. Why it holds, by construction, in
`StageSensitivity::stage`: a stage whose mode differs from the last one
the object saw is accepted only if an event was REGISTERED before it
(`time_event` / `state_event`), else it is refused ("not located"); a
registered event with dτ/dp ≠ 0 in a circuit holding any Newton kind is
refused (`has_newton_` is the DEVICE SET — Shockley, smooth diode,
MOSFET L1, excess loss — not the device that fired); a stateful kind is
refused at construction; a step not started from the committed bits is
refused (contiguity). So a hole can only be (a) a mode change that does not
show in the stamped mask, or (b) a registered event with a wrong dτ/dp. (a):
the admitted kinds without a mask bit are linear or the four static laws
(smooth: no event); time-varying sources have no parameters (dτ/dp = 0).
(b) is what the truth cells check.

`python/tests/test_sensitivity_event_audit.py` enumerates 5 event kinds
(none, PWL diode turn-on, fixed gate edge, duty edge, sine-fed turn-on AND
turn-off) × 3 device classes (linear, + a static Newton law, + a stateful
kind) × 5 engine configurations (pwl, pwl+restart, pwl+split, trbdf2,
trbdf2+esdirk43) = 75 cells; each is a number within 1e-5 of central
differences of scipy Radau (rtol 1e-12, event functions) or an
`UnsupportedError` naming the reason, pinned per cell. Plain pwl's numbers
are checked against central differences of the same scheme (1e-6): its
state has its own O(dt) first-order-start error (1.3e-4 here, the same as
its sensitivity's gap to the truth; with restart both are ≤ 1e-8).

| kind | linear | Newton | stateful |
|---|---|---|---|
| none | number on all 5 | number on all 5 | refused (history) |
| diode turn-on | pwl, pwl+restart: refused (not located); split, trbdf2: number; esdirk43: refused (landing) | refused (Newton device) on split / trbdf2; else as linear | refused |
| fixed gate edge | number on all 5 | number on all 5 | refused |
| duty edge | pwl, pwl+restart: refused (staircase); split, trbdf2, esdirk43: number | refused (Newton device) where located | refused |
| sine on/off | split: number; trbdf2: refused (turn-off within the conditioning floor); esdirk43: refused (landing); pwl, +restart: refused | refused | refused |

Worst truth gaps of the number cells: diode turn-on 1.9e-7 (split) /
6.0e-8 (trbdf2); duty 5e-10 / 5e-9 / 4e-12 (esdirk43) — d v/d D is entirely
the jump term, so the sign and the factor T of dτ/dD are checked there;
sine on/off on split 8.2e-8 (two turn-ons, two turn-offs, the ḃ term).

### Review (fresh-context, opus, read-only, 2026-11-20) and what changed

Asked for an event configuration that returns a number without having
differentiated the event, and for the sign and factor of dτ/dp. Found, each
reproduced against a Radau truth and fixed:
1. ḃ HAD THE WRONG SIGN in `state_event`'s ẋ⁻: `src` writes the engines'
   b_extra rows, whose sign is the opposite of b in f = G·x − b (the
   engine's rhs is −(b_const + b_extra)). Every diode crossing driven by a
   time-varying source got a wrong dτ/dp: a sine-fed rectifier with a series
   R2 (the turn-on's jump survives to t_end) returned 1.787e-2 vs 1.611e-2
   (11 %), a sine current source into a diode 2.7e-6 vs 1.19e-4. The audit's
   sine cell had passed because its turn-on jump decayed in 0.1 µs and its
   turn-off jump is zero; the cell now has the series R2.
2. AN EVENT INSIDE THE FIRST STEP of plain pwl returned a number (a duty
   edge before dt: 0.0 vs 9.95; a diode turn-on: −4.5e-4 vs 1.85e-4): the
   gate baseline was taken at the end of step 1, and the first stage adopted
   any mode. Now the gate baseline is `gate_at(t_start)` and the engines hand
   their start mode to `start_mode` (when it is known: a seeded start
   decides its mode from x0 at step 1, not an event); the first stage is
   checked like any other. Both are audit cells.
Confirmed right by the reviewer's own derivation and runs: the sign and the
factor T of dτ/dD (one-pulse RC, entirely the jump term: 2.5e-8 pwl+split,
5e-9 trbdf2; with a diode cascading at the OFF edge 1e-10 / 2e-8), a
DC-driven turn-on on an algebraic node (jump 58 % of the answer, ≤ 1e-8).
Not probed by the reviewer: events at t_end, ties with differing dτ/dp,
ESDIRK / Newton refusals beyond the tests.

### TR-BDF2 turn-off landings (6.3, 2026-11-20)

- The INTERPOLATED landing (a turn-off at its bracket floor, no solve) is
  carried: `StageSensitivity::carry` moves dq/dp by dt_land·dq̇/dp (dt_land is
  ps; first order, the scheme's own order there), keeps the slots, and
  re-initialises S consistent with dq/dp at the landed x by one bordered
  solve per parameter on ANY resident factor of the mode
  (`PwlStateSpaceCache::any_resident_factor`: the event entry's, else the
  segment's; read-only, no build — no factorization with or without
  sensitivities; the off path is untouched). The event that follows uses
  the same factor for ẋ⁻. ESDIRK interpolated landings stay refused.
- A slot-only derivative refresh AT t_end no longer refuses (no step follows).
- DCM buck (40 T) on TR-BDF2 vs the exact path's table: rtol 1e-10 L 5.2e-6,
  R 2.9e-8, D 5.2e-8 (pass, test); rtol 1e-11 L 1.1e-6. At rtol 1e-9 a
  turn-off falls within the conditioning floor of a step start (the bits are
  flipped by probes at t, nothing landed): REFUSED by name (strict xfail).
  Tried and reverted: carrying that flip as an event at t from the committed
  state — 3.6e-4 off a Radau truth on the sine-fed rectifier at rtol 1e-10,
  1.7e-4 at 1e-11, 7.6e-5 at 1e-12 (a bias that converges slowly, not traced:
  candidates are τ = t instead of the crossing up to dt_floor later, and the
  probe-decided cascade). NEXT STEP for 6.3 at rtol 1e-9: trace that bias on
  the audit's sine cell (trbdf2), e.g. by forming the jump at the probe's
  located t_star instead of t.
- The 2 T buck of `test_sensitivity_exact.py` now runs on trbdf2 and agrees
  with the exact engine within 1e-5 at rtol 1e-10 (it was a refusal test).

### Verification at e149fc47 (2026-11-20) and the state for the next agent

- Byte compares vs 41860812 (sensitivities off, base built LTO off like the
  tip, `.scratch/g6cmp.sh`): goldens 21/21 (counters incl. cache / Newton
  factorizations unmoved), stateful 40, engines 35, steps 28, pwl_identity
  390/390, trbdf2_identity 371/371 — all identical.
- States bit-identical with and without sensitivities, engine stats and
  Newton factorizations equal, n_sens_solves within [rows, 5·rows+8] and
  n_sens_iterations 0 (linear) / ≤ 2 per solve (Newton): asserted in every
  number cell of the audit.
- COST GATE (instructions per accepted step, min over 6 alternating
  processes × 3 runs): bridge_trbdf2 −0.07 %, buck_trbdf2 0.00 %, vsi_pwl
  −0.07 %, buck_pwl +0.59 % (min of 12: 3106.4 vs 3088.2; means +0.53 %) —
  buck_pwl FAILS the 0.5 % gate by a hair. Not traced: the off path of the
  pwl loop gained per-step `if (sens.on())` blocks (gate tracking, the
  inline-stage flag) and new lambdas; candidates: hoist the step's two
  sens blocks into one out-of-line call, check code layout.
- Full pytest at e149fc47: 6286 passed, 110 skipped, 9 xfailed (0 failed). ruff clean; `openspec validate add-parameter-sensitivities --strict` valid.
- NOT RUN: the C++ step-test target (pulsim_core_step_tests) — no build
  of the test tree this session; full ctest (track agent rule).
- NEXT: (1) buck_pwl cost; (2) 6.3 at rtol 1e-9 (the floor-path bias
  above); (3) 4.4 saturable inductor (item 4 of the hand-off list).

### Cost gate, step tests, landing review (2026-11-20, third session; WIP)

- WHAT COST THE +0.59 % on buck_pwl (sensitivities off). Measured with a
  standalone C++ driver (`.scratch/c7/rtbench.cpp`: diode buck, pwl, dt
  1e-7, differential instructions per step (I(4 ms) − I(2 ms))/ΔN, min of 3;
  30 s per variant against a header tree) — it reproduces the Python gate:
  base 41860812 2845–2850 (PWM) / 2831–2833 (gate held on), tip e149fc47
  2863–2868 / 2851 (+18). Bisection by deleting hook groups from the tip:
  restart hooks only −0; split hooks only −3; per-step pre-block only −3;
  pre-block+split −0…−10 (noisy); pre-block+restart −5; ALL of them
  −18 (2844–2853 / 2833: parity). No single hook: the cost is the closures
  `run_restart` and `split_step`, defined inside step_take and therefore
  BUILT ON EVERY STEP (whether called or not): each by-reference capture is
  one load + one store per step, and the hooks added 8 captures (sens,
  sens_begin, sens_stage to run_restart; sens, sens_gate_event, sens_begin,
  sens_stage, sens_diode_events to split_step) = 16 instructions, plus the
  pre-step `if (sens.on())` block. Seen in the disassembly (store pairs
  before the split_step call in the step body).
- CHANGED (run_transient.hpp, uncommitted→this commit): (1) the pre-step
  block is gone — the step's gate change at t_prev is taken lazily by
  `sens_open()` at the step's first call into the recurrence (sens_begin,
  or the closing block of a plain step; `gate_t0`, `t_prev_step`); one
  `sens.on()` test per step, as in the base; (2) the inner closures reach
  the recurrence through ONE reference `sx` (a const struct: cached `on`,
  sens, the hook lambdas); (3) `split_step` is defined inside `if (split)`
  where it runs, so a run without the split builds no closure for it.
  Bench after: (1)+(2) alone +7…+10; with (3) 2798–2801 / 2785–2791 —
  −1.6 % against the base, and parity (±3) against the base with the same
  move applied (2802 / 2785): the off path costs nothing measurable beyond
  the move's own gain. Real-extension four-case gate and byte compares:
  PENDING (extension build started 14:19; see the next entry or the report).
- REVIEW of e66913b3..e149fc47 (fresh-context opus, scripts in
  `.scratch/c7/review/`): carry's sign and side correct; the bordered
  re-init is independent of which resident factor (linear). dt_land is NOT
  ps: 0.2–5 ns on the 40 T DCM buck at rtol 1e-10 (the floor bounds the
  bracket width, not its distance from t) — comment fixed in
  sensitivity.hpp. No case found where the displacement is most of the
  answer (share ≤ 2e-3; everything coarser refuses at the floor). FINDING
  (medium): an event within ~3e-13 s of t_end left PENDING returned the
  left limit — duty edge on the pulse RC: dv/dD 0.000 vs truth 7.38; diode
  turn-on: dv/dR −6.7e-4 vs +4.4e-3. FIX: `StageSensitivity::finish`
  refuses by name when an event with dτ/dp ≠ 0 is still pending ("whose
  jump in the derivative was not applied"). Audit cells for it: NOT YET
  ADDED (needs the rebuilt extension to pin each engine's outcome;
  reproducers `edge_at_tend2.py`, `diode_at_tend2.py`). Not reachable,
  noted: `qdv_n_` is left at x_n by carry (exact only for V_f = 0).
- pulsim_core_step_tests: NOT built this session (`.scratch/build-ctest`,
  LTO on, has a 10:37 binary of older code).
- NEXT: (a) when the extension is built: four-case gate (`.scratch/bl/cost.sh
  6`), g6cmp byte compares vs 41860812 (`.scratch/g6cmp.sh compare`), the
  reproducers → audit cells `*_at_t_end` (expected: refusal on trbdf2),
  audit + sensitivity tests, full pytest, ruff, openspec validate; (b) build
  and run pulsim_core_step_tests; (c) 6.3 at rtol 1e-9 unchanged.
- GATE on the rebuilt extension at 93292618 (LTO off, `.scratch/bl/cost.sh
  6 norec`, instructions per accepted step, min of 6 alternating processes
  × 3 runs, sensitivities off, vs 41860812): buck_pwl 3053.4 vs 3102.5
  (−1.58 %), vsi_pwl 4721.9 vs 4771.1 (−1.03 %), buck_trbdf2 11060.9 vs
  11058.8 (+0.02 %), bridge_trbdf2 12836.0 vs 12829.8 (+0.05 %). All ≤ +0.1 %.
  The pwl gain is the split_step move (both cases run without the split).
  Cumulative vs b9e46778 NOT measured (no LTO-off build of it).
- NOT RUN on this tip: byte compares (g6cmp), audit / sensitivity tests,
  full pytest, ruff, openspec validate, pulsim_core_step_tests. The move
  and the lazy gate event change code paths on the default pwl loop: the
  byte compares are the first thing the next agent runs.

### Verification of the tip (2026-11-20, fourth session): 0d518f42

- BYTE COMPARES vs 41860812, sensitivities off (`.scratch/g6cmp.sh compare`
  against `.scratch/g6base`): at 713229ce (the closure move, lazy gate
  event, `sx`) AND again on the final extension at 0d518f42 — goldens with
  counters, stateful, engines, steps all identical; pwl_identity 390/390,
  trbdf2_identity 371/371. Logs `.scratch/logs/g6cmp_713229ce.txt`,
  `g6cmp_0d518f42.txt`. The refactor is behaviour-preserving.
- THE REVIEWER'S FINDING, re-run (`.scratch/c7/review/edge_at_tend2.py`,
  `diode_at_tend3.py` = diode_at_tend2 catching refusals; logs
  `.scratch/logs/repro_*`): the diode turn-on was refused at every gap
  ≤ 2.2e-13 s and right above it (4.399967e-3, 1.8e-8). The duty edge was
  NOT fixed by the pending-event refusal: gaps 3e-17 … 3e-14 s still read
  dv/dD = 0.000 (truth 7.38) — the edge never became pending. Cause:
  trbdf2/esdirk43's no-final-sliver rule (tb_clamp) leaves an edge within
  64 h_min of t_end to the next run, and on pwl+restart / pwl+split the
  restart rule reads the gates at t_prev, so an edge AT the last grid point
  is opened by no step (sens_open) — on both, x(t_end) is the pre-edge
  side. FIX (0d518f42): `StageSensitivity::edge_past_end(τ, before,
  after)` refuses by name ("whose jump in the derivative was not applied:
  the run ends at it") when a duty parameter's switch turns OFF there;
  called in the no-final-sliver branch (trbdf2_transient.hpp) and after
  the pwl driver (run_transient.hpp, gates at gate_t vs t_reached; plain
  pwl reads the gates at t, no-op). Both sens-gated. After it every gap
  ≤ 3e-14 s is refused, ≥ 3e-13 s right (7.380810).
- AUDIT CELLS `test_an_event_at_t_end_is_refused_by_name_or_true` (20):
  duty edge / diode turn-on at τ(1 + 1e-11) and 1.5 steps before t_end,
  five engine configurations, pwl grid ending at t_end. At t_end: refused
  by name everywhere (pwl STAIRCASE, pwl+restart/split/trbdf2/esdirk43
  NOT_APPLIED; diode: pwl+restart NOT_LOCATED, esdirk43 LANDING), except
  plain pwl's diode, whose own trajectory (first order from the start) has
  not crossed yet: its number is its own map's derivative (central
  difference h = 1e-6 R; h = 1e-4 R steps over its crossing). Before
  t_end: numbers within 2e-8 of the closed forms on pwl+split / trbdf2 /
  esdirk43 (duty) — the refusal hides no working case. Audit 107 passed,
  every sensitivity file 261 passed / 1 xfailed (`.scratch/logs/
  audit_step2.txt`, `sens_tests_step2.txt`).
- pulsim_core_step_tests built (Eigen 5.0, /opt/homebrew/include/eigen3,
  `.scratch/build-ctest`, LTO on as that tree's cache has it) and run:
  811 assertions in 34 test cases pass (`.scratch/logs/
  steptests_0d518f42.txt`). Nothing to fix.
- FULL PYTEST at 0d518f42: 6306 passed, 110 skipped, 9 xfailed, 0 failed
  (`.scratch/logs/pytest_full_0d518f42.log`). ruff check python/ clean;
  `openspec validate add-parameter-sensitivities --strict` valid. Stubs
  not run (no binding signature changed).
- COST GATE on the final extension (LTO off, `.scratch/bl/cost.sh 6
  norec`, min of 6, sensitivities off, vs 41860812): buck_pwl 3044.9 vs
  3095.3 (−1.63 %), vsi_pwl 4726.9 vs 4783.6 (−1.19 %), buck_trbdf2
  11068.9 vs 11061.7 (+0.07 %), bridge_trbdf2 12843.8 vs 12831.2
  (+0.10 %). The pwl gain is confirmed; trbdf2 reads +0.07/+0.10 against
  the earlier +0.02/+0.05 (one sens-gated branch added to tb_clamp's
  no-final-sliver arm, plus run-to-run spread): all within +0.1 %.
  Log `.scratch/logs/cost_0d518f42.txt`.
- NOT RUN: full ctest, Eigen 3.4.0, g++-15, a fresh-context review of
  0d518f42's fix. Still open as before: 6.3 at rtol 1e-9 (strict xfail),
  4.4 saturable inductor.

### On the PR line (2026-11-23): merge onto 7188a9e1, branch feat/core-sensitivity-events-pr

The PR line had rebuilt the restart region under this branch (run_restart /
restart_lined built once per run, `consistent_start` with the jump's vector
series, `decide_edge_mode`, `locate_islands`). `run_transient.hpp` is the PR
head's file with the calls into the recurrence re-applied by meaning
(`.scratch/sev/apply.py`); `trbdf2_transient.hpp` merged textually (one
hunk: `edge_past_end` in the no-final-sliver arm, next to the line rule).

- Hooks (`sens_stage`, `sens_gate_event`, `sens_open`, `sens_begin`,
  `sens_diode_events`, the one-reference struct `sx`): defined ONCE per run,
  before `decide_edge_mode` / `consistent_start` / `run_restart`.
- `run_restart`: `sx.begin` at entry; stage 1 fed right after its
  `run_event_iteration` — after `decide_edge_mode` chose the mode, BEFORE
  `consistent_start` (the probe's solves replace a Newton stage's factor;
  every false return of the probe restores x, histories, bits, mask); stage
  2 as the branch had it (SDIRK psi, the kept backward-Euler re-take, a
  mode-changing stage 2 refused).
- NEW REFUSAL: `consistent_start` returning true (the histories restart
  from x0 + J and stage 1 is solved again) → `refuse_step(t, "restarted
  from a state its event moved (an impulsive event …)")`. dJ/dp is not
  carried: J is the Krylov series of the probe's second differences (tried
  whenever the circuit has ≥ 2 C / L states), and without it the recurrence
  integrates the fast component of dq/dp by the SDIRK stages (relative
  error R(z), the error the series removed from the state). Tests:
  `test_sensitivity_restart_jump.py` (two capacitors closed on each other:
  refused, with the reinit record as the control; the same circuit before
  the closing: 1e-6 of central differences).
- CONSEQUENCE, a cell LOST: the DCM buck on pwl + split (dt = T/1000, 40 T)
  was within 1e-5 of the table on the branch's base; on the PR line the
  turn-off at 29.524 T is recorded as a cut-set jump of L (1.73e-10 V·s,
  1.5e-16 J: the located crossing's residual current) and the run is
  refused. `test_dcm_buck_transient_against_the_table[pwl]` is a strict
  xfail. NEXT STEP: carry dJ/dp — replay the probe's stages as
  backward-Euler stages of the recurrence (same factor, one back-substitution
  per parameter and stage), then the tangent of the series (a pure function
  of the stage readings: differentiate it with the fit's order held), and
  add C/2·dJ/dp to the re-solved stage 1's ψ.
- `split_step` (inside `if (split)`), the closing block and the end-of-run
  `edge_past_end`: the branch's lines at the same places. `locate_islands`
  needs nothing: a group of ≥ 2 diodes located on a summed signal registers
  ≥ 2 state events at one instant, which `state_event` refuses as a tie.
- Review (fresh context, fable, read-only): no path found where a step is
  accepted with a state the fed stages do not produce. Fixed from it:
  `sens_open` passed `located = split && !ev_overflow` with an opaque
  `switch_fn` (edges sampled, not located) — now also requires
  `next_edge_fn` (Python already refuses a duty without the PWM helper, so
  no Python test reaches it). Recorded, not fixed: a sensitivity solve that
  does not converge inside a restart / split step throws runtime_error,
  which the dt-retry ladder catches — the run is still refused, as "retried
  at a smaller dt"; the inductor freeze guard's snap-back is refused one
  step late by the contiguity check; `mode_guard_site` / `mode_lcp_site`
  not opened by the reviewer.
- Release finding F-Y-1 (pwl sensitivities refused at the first split
  step): on this tip they run with the restart rule and the split ON
  (`test_sensitivity_release_defaults.py`): RC vs central differences
  1.9e-10 / 3.4e-11, vs the closed form 1.7e-7 (plain pwl 1e-3); Shockley
  rectifier worst 1.3e-8; MOSFET K, V_T within 1e-6; states and
  factorizations identical with and without sensitivities.
- Verification seen (logs `.scratch/sev/`): sensitivities OFF vs the PR
  head (so_4367f887 + python at 7188a9e1): goldens 21/21, stateful 41,
  engines 35, steps 28, pwl_identity 396/396, trbdf2_identity 376/377 (the
  one difference is a file path inside a YAML loader message), AC 59/59,
  res.i 212 keys, recorder 12/12, dsed bit-identity 36/36 (+ shadow), loss
  dump 81 calls, sensitivity / PSS dump 836 calls, exact dump 441 calls —
  identical. Cost (instructions per accepted step, min of six, LTO off on
  both sides, measured before the `located` fix, which is inside a
  sens-only lambda): buck_pwl +0.10 %, vsi_pwl +0.04 %, buck_trbdf2
  +0.01 %, bridge_trbdf2 +0.02 %.

### TR-BDF2 turn-off landings commit their diode (2026-11-26, PR #150 CI)

- The rtol-1e-9 (macOS) / 1e-10 (Linux, Windows) refusal "a diode crossing
  within the conditioning floor of its start" was not a sensitivity defect:
  the step before was an interpolated turn-off landing (1e-6 of its bracket
  past the zero, by construction), and the landing's round-off bound held
  the diode ON at -1.4e-24 A (DCM buck) / -8.9e-16 A (sine cell). The next
  step re-found it on its forbidden side and flipped it by probes at the
  same t. Which path ran was decided by libm.
- Fix (`trbdf2_transient.hpp`, landing): when the bound held every flip at a
  landing, the located diode, on its post-crossing side by any amount,
  commits there; the regular landed-event path (and its saltation) follows.
  Not when another diode flipped (a bridge pair crosses together).
- Sensitivities now: sine cell vs Radau rel -2.1e-7 / 6.5e-9 / -1.4e-8 at
  rtol 1e-9 / 1e-10 / 1e-11 (the carried-at-t attempt: 7.4e-4 / 3.6e-4 /
  1.65e-4); DCM buck L -2.9e-5 / -5.4e-6 / -1.1e-6 (the method's error).
  The 1e-9 strict xfail and the audit's FLOOR cell are numbers now.
- Runs without sensitivities (21 trbdf2 runs, Eigen-3.4 build of a9d6a83a,
  shown bit-identical to its Eigen-5 build on the 7 sensitivity runs): 18
  bit-identical, most with fewer solves. Three differ:
  llc_half_bridge_zvs at rtol 1e-10 (D_lo turned off one 2.2e-15 s step
  later, under the ON mask, without a commutation record; now at the
  landing; final state 4.2e-7 off, n_steps +4 of 52 556, n_solves -43);
  diode_bridge_rectifier at defaults / rtol 1e-10 (the located D4 turns off
  at the landing, its series partner D1, then at zero current inside the
  bound, stays ON up to 0.3 ms instead of both flipping at the probe pass;
  V(out) the same to 7e-14 at t_end, only the source's floating common mode
  ac_p / ac_n held by g_off differs; n_solves +11 / -5).
