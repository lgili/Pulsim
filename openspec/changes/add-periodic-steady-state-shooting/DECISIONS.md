# Decisions taken while implementing slice 1

Slice 1 = tasks 1–4 and the docs of 8.1 (the solver on today's `dsed`
exact stepper). The owner was away; each open question took the option
marked "proposed", else the most conservative one. Each entry: question
→ option → why → how to reverse.

## Open questions of design.md

1. **Name of the public function.** No option was marked proposed.
   → New function `periodic_steady_state` (additive), and
   `steady_state` keeps its name and its legacy result when `dt` is
   given; without `dt` it returns the new `PeriodicSteadyStateResult`.
   `SteadyStateResult.history` stays on the legacy path. → Nothing
   existing breaks in this slice; the BREAKING rename belongs with
   task 5 when the fixed engine moves to the true state. → Reverse:
   make `steady_state` return the new type always and drop
   `SteadyStateResult` (task 5.3).
2. **Remove or rename `run_harmonic_balance`.** → Not acted on:
   `run_periodic_shooting` and `run_harmonic_balance` delegate only in
   task 5.3 (fixed engine, Phase 2), so both are untouched. → Most
   conservative. → Decide at 5.3.
3. **Unstable orbit.** → Proposed option: returned with
   `stable = False` and `unstable_multipliers` naming each multiplier
   and its type (flip / fold / complex pair). → Reverse: raise in
   `periodic_steady_state` unless `allow_unstable=True`. The slice has
   no test circuit with an unstable orbit (the peak-current-mode case
   needs a comparator-driven gate, task 6.3), so the verdict is tested
   on stable orbits only.
4. **`period_multiple=n`.** → Not part of this change (conservative).
   → Reverse: pass `period=n*T`; the schedule check already accepts any
   period that is a multiple of the schedule's.
5. **Unknown period.** → Out of scope for this change (conservative).
6. **Default section.** Proposed: the first rising gate edge. → Slice 1
   uses `t = 0`, which IS the rising edge of the zero-phase built-in
   PWM helpers. The native scheduler starts every run at `t = 0`, so a
   different section would need a time-shifted schedule wrapper whose
   edges are no longer bit-exact. → Reverse: a `section=` argument that
   shifts the schedule once Phase 3's event queue owns the clock.

## Choices the slice needed

7. **Delegation is keyed on `dt`, not on coverage** (task 4.1).
   `steady_state(..., dt=None)` and `frequency_response(..., dt=None)`
   call the new solver; with `dt` the existing fixed-engine code runs
   unchanged, as D7 says ("`dt` selects the fixed engine"). → Every
   existing caller passes `dt`; delegating them silently would have
   swapped the engine they asked for (the fixed engine's orbit differs
   from the exact one by O(dt²), 0.023 % on the DCM buck at T/100) and
   the type of `snapshot` (a pwl `SolverSnapshot` cannot be built from
   the true state before the restart rule lands). Refusing instead of
   falling back: the new path raises by name what it does not cover.
   → Reverse: in `steady_state`, try the new solver first when
   `switch_fn` names its edges, and fall back on its refusal.
8. **Event projection off inside the map.** The dsed scheduler projects
   the state onto the new mode's slow manifold at every event (a reset
   map `R`). Map evaluations run with `enable_event_projection=False`,
   so the state is continuous and `S` has no `R_x` term; the exact
   stepper has no stability limit, so nothing grinds. The verification
   period and `orbit` use `simulate(engine='dsed')` (projection on) and
   must agree with the map to 1e3× the last Newton residual. →
   Reverse: replicate the projection's Jacobian (`V·diag(slow)·V⁻¹`,
   threshold 100/dt_max) as `R_x` in `_event_jump`.
9. **Lossless loop refused although `μ ≠ 1`.** D6 refuses a multiplier
   equal to 1; the spec scenario "ideal LC loop → not isolated" has
   multipliers `e^{±jωT}`, so `I − M` is regular and Newton converges.
   → The solver refuses any multiplier with `| |μ| − 1 | < 1e-10`
   (damping slower than 1e10 periods), naming the states of its
   eigenvector: no transient reaches such an orbit, so returning it
   would contradict the "equals a long transient" requirement. `I − M`
   singular (μ = 1) is refused separately during Newton. → Reverse:
   return such orbits with `stable = False` and a "marginal" type.
10. **Event instants in `pattern`** are refined by one Newton step on
   the predicate, `τ − g/(∇g·f⁻ + g_t)`. The dsed locator stops at
   `|g| < 1e-10 V`, which on the DCM buck leaves the diode turn-off
   2.3e-13 s (2.3e-8·T) early; the state is unaffected (the field is
   continuous there) but the spec asks the instant to 1e-9·T. →
   Reverse: report the raw logged instant.
   **Reversed (integration batch 4, after add-exact-propagator-strategy):**
   the stepper now brackets events in time to `max(4u|t|, 1e-12·h)`;
   the Newton step moved the DCM turn-off by −3.1e-22 s and the DCM
   boost's by 0, so `pattern` reports the logged instant (DCM turn-off
   0.7998306984492 T, reference 0.7998306984445 T).
11. **`B` in `frequency_response(dt=None)`** is a central difference of
   the true-state map in `u` (`du = 1e-5·max(1, |u0|)`, refused if it
   changes the commutation pattern): `switch_fn_of` is an opaque family,
   so `∂τ/∂u` is unknown. `sampled_model` builds `B` analytically from
   the same propagators for the two inputs it can parameterise
   (`("duty", switch)`: trailing edge, `∂τ/∂D = T`; `("source", name)`:
   a DC source value, by an augmented state `(x, u)` so the saltation
   carries `g_u`). → Reverse: give `frequency_response` an `input=`
   spec routed to `sampled_model`.
12. **PWM and pulse voltage sources are refused**: their edges are not
   events of the exact stepper (it falls back to RK45 and does not
   land on them). DC and sine sources are accepted; a sine source must
   satisfy `f·T` integer within 1e-12.
   Integration (review of 4f55aabc): the sine / pulse / table CURRENT
   sources of add-controlled-source-set were accepted unchecked, and a
   source that is not T-periodic returned an orbit that does not exist
   (`stable=True`). Now every waveform source is checked and refused
   with `UnsupportedError(feature="period", device=<source>)`: sine
   (voltage and current) `f·T` a positive integer; pulse current source
   period > 0 dividing T with its first pulse inside its first period;
   table current source `periodic=True`, table period dividing T, first
   knot at t <= 0. The sine-voltage case changed type from ValueError to
   UnsupportedError (a ValueError subclass), for one rule.
13. **Scaled norm**: `atol_i = rtol·max_t|x_i(t)|` over all period runs
   so far (D4 says "from the state scale of the base run"; the running
   maximum is the base run's scale once the iterate is near the orbit,
   and it never lets a component sitting at zero at the section — the
   DCM inductor current, 6e-9 A — demand relative accuracy).
14. **Binding**, the minimum task 2.1 needs: `(A, b)` per mask and the
   event log already existed (`_NativeCircuitBuilderAdapter.A_matrix /
   b_vector`, `run_ped_native_builder(...)["event_log"]`). Added
   `_NativeCircuitBuilderAdapter.node_pair_voltage(from, to, t, x)` (the
   predicate's recovery row, read as the gradient of an affine map) and
   the sine frequency in `CircuitBuilder.components()` (to name an
   incommensurate source by frequency).

## Group 5: the fixed and TR-BDF2 engines (slice 4)

Measured with `.scratch` probes on the DCM buck (24 V, 100 µH, 47 µF,
50 Ω, D = 0.3, T = 10 µs) unless stated; `python/pulsim/_transient_shooting.py`.

15. **The map is one `simulate()` per period from `X(x)`, the MNA vector
   carrying the true state `x` (minimum norm, the rest zero).** Question:
   can the fixed engine's map be a function of the true state alone? Its
   trapezoidal history is not in `x`. Evidence: with
   `discontinuity_restart=True` and `enable_substep_state_correction=True`
   the period opens with the restart rule's SDIRK steps, which read no
   history: one period from the engine's full vector at `4000T` and from
   `X` of its true state give the same `v_C` bit for bit
   (8.99624640380659). That is the transient's own map only where the
   transient restarts too, i.e. at a gate edge at `t₀`; without one the
   fixed engine is refused (`feature='section'`). `trbdf2` keeps a step
   controller that a one-period run restarts from `h_init`; its default
   step ceiling is `t_end/1000`, so a one-period run and a long transient
   would be different integrations. The shooting fixes the ceiling at
   `dt` (default `T/100`) and the oracle runs the transient with the same
   `dt`. Opaque gates: sampled on the grid on `pwl` (checked periodic on
   it), refused on `trbdf2`. Nonlinear devices are refused (their state
   is not in `x`). → Reverse: carry device states in `x` (group 6.1).
16. **`M` by forward differences (`h = 1e-6·(scale_j + |x_j|)`), Broyden
   rank-one updates after accepted steps, refreshed when an accepted step
   reduces the scaled residual by less than 10×, when no backtracking
   step is accepted with a stale `M`, and always before the stopping rule
   reads `M`.** Evidence (period runs from zero, refresh threshold 0.5 /
   0.3 / 0.2 / 0.1 / 0.05): DCM `pwl` 14 / 14 / 13 / 13 / 17, DCM `trbdf2`
   21 / 21 / 16 / 13 / 18; CCM unaffected (8 / 9). The final multipliers
   are those of a fresh difference `M` at the orbit, never a Broyden
   estimate. → Reverse: `_SLOW` in `_transient_shooting.py`.
17. **No separate verification period on these engines.** A period run
   is a fresh deterministic `simulate()` from `X(x)`; re-running the
   accepted iterate reproduces it bit for bit, so the accepted run is the
   orbit run (`result.orbit`, `result.full_state`). Saves one run: DCM
   buck `pwl` 13 → 12 (the 5.5 gate). → Reverse: re-run `x` in `solve`.
18. **Delegation (5.3)**: `run_periodic_shooting` keeps its signature
   (full MNA vectors in and out; `x_steady_state` is the engine's vector
   at the section after the orbit period); `b_extra_fn` / `step_observer`
   are refused by name; `tol` / `max_iterations` / `fd_step` are accepted
   and ignored; a non-convergence is `converged=False` with an all-NaN
   vector and the reason in `message` (it used to return the last
   iterate). `run_harmonic_balance` takes the spectrum of `result.orbit`.
   Both warn `DeprecationWarning`. → Reverse: the v1 code is in git
   (`python/pulsim/periodic.py` at 3ea998c6).
19. **Evaluator**: `SteadySpec(method='pss')` runs on the Evaluator's
   engine (`route='pss/pwl'`, `'pss/trbdf2'`; counters
   `pss_jacobians`, `pss_broyden_updates`); the KPIs read the second of
   two periods from `X(x₀)` with the same restart rule. `method='auto'`
   still takes `pss` on the exact stepper only: on `pwl` it would change
   the route, cost and numbers of existing evaluations. `gradients=` on
   these routes is refused (`feature='gradients'`). → Reverse:
   `Evaluator._steady`.

Measured (orbit vs the engine's own 4000-period transient, of the state
scale; period runs from zero; max|μ| against the exact stepper's):

| circuit | engine | v_out | orbit − transient | runs | max|μ| (exact) |
|---|---|---|---|---|---|
| buck CCM | pwl T/100 | 11.9877290506 | 1.5e-14 | 7 | 0.899035 (0.899035) |
| buck CCM | trbdf2 | 11.9877289446 | 2.9e-9 | 10 | 0.899035 |
| buck DCM | pwl T/100 | 8.9962464037 | 1.2e-11 | 12 | 0.988977 (0.988977) |
| buck DCM | pwl T/101 | 8.9962464251 | 1.8e-11 | 12 | 0.988977 |
| buck DCM | trbdf2 | 8.9962471311 | 2.6e-9 | 15 | 0.988979 |
| boost CCM | pwl T/100 | 24.0536315741 | 4.6e-10 | 22 | 0.994645 (0.994645) |
| boost CCM | trbdf2 | 24.0536300200 | 4.2e-9 | 25 | 0.994645 |
| boost DCM | pwl T/100 | 36.5944019107 | 4.5e-12 (40000 periods) | 34 | 0.998942 (0.998942) |
| boost DCM | trbdf2 | 36.5943972503 | 5.1e-9 (4000 periods from the pwl orbit) | 33 | 0.998942 |

The boost DCM transient has not settled in 4000 periods (μ = 0.99894,
period-to-period change 6.2e-6 of scale): the difference there is the
transient's, not the shooting's; at 40000 periods it is 4.5e-12 on
`pwl`. `trbdf2` exhausts `max_steps` (10⁷) before 40000 periods, so its
transient starts from the `pwl` orbit (5.1e-9 after 4000 periods).
(Table measured with decision 20's steps.)

20. **The difference step and the marginal band are per engine.** A
   variable-step run moves by its tolerance when `x` moves (its step
   sequence changes), so a difference over `h` carries `~rtol/h` of
   noise: max|μ| of the trbdf2 orbit against the exact stepper's, at
   `h` = 1e-6 / 1e-5 / 1e-4 / 1e-3 of the scale: buck CCM 3.0e-5 / 4e-7
   / 1e-7 / 3e-7, DCM 2.6e-5 / 1.4e-5 / 2e-6 / 1.4e-5. `trbdf2` takes
   `h = 1e-4`, `pwl` keeps 1e-6 (deterministic to rounding; no change
   between 1e-7 and 1e-5). The 1e-10 marginal test of the exact path
   cannot be applied to a difference Jacobian: an undamped LC beside a
   switched branch (multipliers on the unit circle, a transient that
   never settles) read 1 − 1.8e-9 on `pwl` (returned as a stable orbit)
   and 1 + 4.2e-5 on `trbdf2` at `h = 1e-6` (returned as an unstable
   one). The band is `100 × run noise / h` (1e-14 per run on `pwl`,
   `rtol` on `trbdf2`): 1e-6 and 1e-4; it also bounds the singular test
   of `I − M`. Both now refuse that LC as not isolated. An orbit with a
   genuine `| |μ| − 1 |` below the band is refused, never returned. →
   Reverse: `_FD_REL`, `TransientMap.marginal`.
