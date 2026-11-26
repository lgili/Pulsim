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
Re-measured on the integration tree (batch 8, after the TR-BDF2 track and
the control graph merged): every buck/boost CCM/DCM row above reproduces
digit for digit, pwl and trbdf2 (v_out, orbit − transient, runs, max|μ|);
the boost DCM rows were not re-run. A trbdf2 section inside a conduction
stretch (gate edge at 0.3 T): 7.5e-10 (CCM buck, 12 runs) and 7.8e-9
(DCM buck, 18 runs) from the transient at 4000 T. The pwl map refuses it
(no edge at t = 0). The pwl map also refuses `options` that switch off
the restart rule or event splitting, and every engine refuses a gate or
a source that follows a controller (`feature='closed_loop'`).

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

## The stop rule and the stuck diode (2026-11-19)

21. **Newton stops on the orbit ERROR ESTIMATE, and a diode at its own
   turn-off crossing is decided by the motion of its OFF signal.**
   Question: why did the boost DCM orbit (12 V, 100 µH, 47 µF, 500 Ω,
   D = 0.5; `test_periodic_orbit_kernel`) come back 1.07e-6 V off on one
   build (36.594399763447576 V, 25 period runs, macOS arm64 + Eigen
   3.4.0) and 1e-10 V off on others (21 runs)? Truth:
   `probes/boost_dcm_truth.py` (mpmath, 50 digits) v_C* =
   36.594400837798 V, diode off at t/T = 0.743942.
   Evidence, two defects:
   (a) *The period map.* One period from 25 starts within ±3e-10 V of
   the orbit (`.scratch` probe, event projection off as in the map):
   6 of 25 ended at i_L = −0.63 A instead of −1.26e-8 A, the event log
   showing "D1 turns OFF" with old mask = new mask. At the located
   turn-off the ON-mode current is a rounding residual (±2e-12 A, v_D
   ~2e-15 V on 1e3 S); the settle's OFF-mode probe (`zero_band_on_`)
   amplifies it through g_off = 1 nS into ±1e-3 V of starting signal,
   and its first rung (δ = 1e-12·dt_max = 1e-17 s, where the true drift
   is −1.2e14 V/s·δ = −1.2e-3 V) read the residual's sign and turned the
   diode back ON; its DiodeOff predicate then started negative and
   never fired again. Fix (scheduler.hpp, one hunk): when the diode's
   ON-mode signal is in the zero band, a rung decides only once the
   probe has moved further than the OFF voltage's distance from the
   crossing voltage (the amplified rounding; a `V_th > V_f` band is not
   counted in it).
   After: 0 of 25. The Newton line search had been halving its steps
   against that discontinuity (errors 2.1e-6, 1.07e-6, 5.4e-7 … V).
   (b) *The stop rule.* It stopped one step after the residual met
   `rtol`, without checking that step: with a slow multiplier the error
   is |G|/(1 − |μ|) (μ = 0.998942: 1.14e-9 V of residual is 1.07e-6 V
   of state), so the halved step above was returned. Now it stops when
   every component of the residual AND of the next correction
   `(I − M)⁻¹·G` is within `rtol·(scale_i + |x_i|)`, the estimate is
   `orbit_error_estimate` on the result, and a full Newton step from a
   converged residual whose correction does not halve is the map's
   rounding over (1 − |μ|): `ConvergenceError` (C++ status
   `noise_floor`) instead of an orbit. Damped steps and relaxation
   rounds disarm that check (they contract linearly); on `pwl` /
   `trbdf2` (difference M, linear convergence) it takes three such
   steps in a row.
   Numbers (old / new / truth, boost DCM v_C*): 36.594399763447576
   (25 runs, the Eigen 3.4.0 build) / 36.5944008378685 (21 runs, kernel
   and Python loops; error 7.0e-11 V, estimate 2.7e-11 V) /
   36.594400837798. Orbit gradients against
   `probes/boost_dcm_gradient_truth.py` (50-digit central differences):
   dv/dL 9.1e-13, dv/dC 2.7e-8, dv/dR 1.1e-11, dv/dD 2.6e-12 relative.
   Moved elsewhere: the zero-mean sine into 1 µF ‖ 1 TΩ (|μ| = 1 − 1e-9)
   used to return v(0) 1.2 % off at the default rtol = 1e-9; it now
   raises the noise floor and returns within `rtol=0.05`.
   `test_example_boost_gradient`'s FD ladder stays (1e-2, 1e-3): its
   1.15e-6 at (1e-3, 1e-4) is unchanged to the last digit (that CCM
   orbit never took the old polish exit; a 1e-4 step over a 1e-9-rtol
   orbit is up to 1e-5 of noise). → Reverse: the `polished` flag in
   `periodic_orbit.hpp`, `periodic_steady_state.py`,
   `_transient_shooting.py`; `sig0` / `on_tie` in `zero_band_on_`.

## On the exact engine's settle (2026-11-19, merge of 2b1d808e)

22. **The scheduler hunk of 21(a) is gone; the stop rule stands on the
   exact engine's own fix, with a rounding floor in the estimate.**
   The merge takes `dsed/scheduler.hpp` whole from
   `feat/core-a-complementarity` 2b1d808e (settle at located diode
   events, significance ladder; `sig0` / `on_tie` removed).
   Re-proved on the merged build (macOS arm64, Eigen 5):
   - 25 period runs within ±3e-10 V of the boost DCM orbit, dt_max = T
     (the solver's map): 0 wrong-branch (all end at i_L = −1.259e-8 A).
   - Boost DCM v_C* = 36.594400837864811 V, 21 runs, kernel = Python
     loop; error +6.7e-11 V against the 50-digit 36.594400837798
     (estimate 5.7e-11 V). The other DCM cases: buck −5.2e-11, boost
     V_th 1 mV +7.2e-11 V. `test_periodic_orbit_kernel` keeps
     max(1e-8, 2·estimate): 140× over the worst measured error.
   - `test_design_thermal_kernel` 125 °C inverter: green at 1e-9.
   - Gradient truths (boost and buck DCM): green at their bounds.
   Two things moved, each with its mechanism:
   (a) *A zero residual is not an exact orbit.* The merged map is far
   less noisy: the sine into 1 µF ‖ 1 TΩ (|μ| = 1 − 1e-9) now ends
   Newton with G = 0.0 exactly, the estimate (I − M)⁻¹·G = 0, and the
   solve returned v(0) 5.2e-7 relative off (8.3e-8 V against a 3.2e-10 V
   tolerance) as converged. The estimate is now
   |(I − M)⁻¹·G| + |(I − M)⁻¹|·(u·scale): one ulp of each state's scale
   is a floor of the map's rounding, amplified. Here it is 9.8e-8 V
   (actual 8.3e-8 V) and the solve raises the noise floor again; at
   rtol = 1e-6 it returns (estimate 1.8e-7, error 2.2e-7 V). Same
   formula in the kernel loop (`periodic_orbit.hpp`), the Python loop
   (`_rounding_floor`) and `_transient_shooting.py`. The floor is a
   LOWER bound of the map's rounding (one ulp; the real rounding of a
   period run is larger), so the estimate is not an upper bound of the
   error (2.2e-7 V error against 1.8e-7 V estimate at rtol = 1e-6
   above); its job is that a zero residual cannot claim an exact orbit.
   `scale` is the running max over the period runs (0.44 V here, the
   start-up runs), as in the rest of the rule. On the boost DCM
   (|μ| = 0.99894) it raises the estimate from 3.4e-11 to 5.7e-11 V,
   still 1000× inside rtol; the run counts do not move.
   (b) *The orbit's sensitivity period runs at dt_max = T*
   (`orbit_sensitivities(n_steps=1)`, was T/50). At dt_max = T/50 the
   exact stepper's settle kept the boost DCM diode ON past its own
   turn-off in 18 of 25 runs within ±3e-10 V of the orbit (event log
   "D1 turns OFF" with old mask = new mask, −0.049 A at the next
   samples; with the diode-law check on, the run raises
   `DiodeModeInconsistentError` instead). At dt_max = T: 0 of 25. The
   propagators are exact at any h, so only the sample grid changes;
   the gradient truths and every orbit-KPI test are unchanged in
   their bounds. Trade-off (reviewer): `_Mode` picks its tier from
   u·‖A‖·h_max, so at h_max = T a near-defective mode that was tier 2 at
   T/50 can be tier 3 and the orbit gradient is refused by name; at
   T/50 the wrong branch is refused too (the `per > 1e-7` check). Both
   are refusals, not wrong numbers; the DCM case is the common one.
   → REQUEST to the exact-engine track (owner of
   `zero_band_on_` / `decision_cap_`): the same settle at dt_max = T/50.
   Reproducer: `.scratch/p6/sens_run.py` in the p4-pss worktree (boost
   DCM of `test_periodic_orbit_kernel`, x within ±3e-10 V of
   36.594400837798, `run_ped_native_builder(..., rtol=1e-11,
   atol=1e-14, dt_init=T·1e-6, dt_max=T/50, enable_event_projection=
   False)`).
   FD ladder of `test_example_boost_gradient` re-measured: dcost/dL
   3.75e-7 off at (1e-3, 1e-4) (was 1.15e-6), 2.7× inside the 1e-6
   bound: not the ≥ 5× a finer ladder needs, so (1e-2, 1e-3) stays
   (8.2e-8 L, 7.1e-12 C). → Reverse: `_rounding_floor` and its two
   twins; `n_steps=1` back to 50 once the request lands.

## On controller states in `x0` (2026-11-24, Task 7.1 & 7.2)

23. **Compiled control-graph states `x_ctrl` sit in `x0 = [x_circuit; x_ctrl]`,
    seeded and read via `CxxBlockChain.dump_state` / `load_state` with no
    C++ binding changes; opaque Python closures/stateful callables are
    refused with `UnsupportedError("periodic_steady_state", "hidden_controller_state", ...)`.**
    - *Normalization & graph discovery:* `python/pulsim/_control_pss.py`
      (`normalize_control_args`, `ControlGraphStateMapper`) accepts
      `closed_loops=`, `step_observer=`, `control=`, or `switch_fn=`
      gates produced by `MixedDomainBlockChain` (`make_pwm_switch_fn`,
      `make_multi_pwm_switch_fn`, `bind_pi_to_duty_callable`). `steady_state`
      with `dt=None` forwards `closed_loops`, `step_observer`,
      `controller_period`, and `control` to `periodic_steady_state`.
    - *State extraction & seeding without caller mutation:* `ControlGraphStateMapper`
      snapshots initial `dump_state()` blobs, seeds `x_ctrl` into the compiled
      `CxxBlockChain` slots (clearing per-run tick counters / recordings), runs
      one period under `suppress_sweep_points()`, and reads back `[x_circuit(T);
      x_ctrl(T)]`. On failure the initial controller blob is restored; on
      convergence the shooting state sits cleanly on the result via
      `PeriodicSteadyStateResult.controller_state`, `controller_labels`, and
      `circuit_state`, alongside combined `x0` and `state_labels`.
    - *Scale-invariant singularity check & `pwl` tick phase:* Mixed
      circuit+controller Jacobians `I - M` can have small singular values from
      controller scaling (e.g. `Ki * T = 1e-3`) while all Floquet multipliers stay
      far from 1; `_transient_shooting.py` pairs a scaled SVD check
      `sig[-1] <= 1e-10 * max(1.0, sig[0])` with the eigenvalue check
      `min|eigvals(I - M)| <= marginal`, plus a scale floor for near-zero states
      such as `prev_error`. On `engine="pwl"` with compiled control, shooting
      defaults `control_tick` to `"state"` so 1-period shooting matches
      `simulate(..., control_tick="state")` on `[0, N*T]`.
    - *Refusal boundary:* `check_hidden_controller_state` inspects `switch_fn`,
      `step_observer`, `closed_loops`, `control`, and `PythonBlock` nodes,
      raising `UnsupportedError("periodic_steady_state", "hidden_controller_state", ...)`
      for closures with mutable cell contents, `nonlocal`/global mutations,
      stateful callable objects, or uninspectable `PythonBlock` / `CBlock`
      callbacks, while allowing stateless pure-time schedules (`make_pwm_switch_fn`,
      `SwitchSchedule`, stateless `lambda t: ...`).
    - *Measured voltage-mode buck + PI at `f_sw = 100 kHz` (`V_in = 24 V`,
      `V_ref = 12 V`, `L = 100 uH`, `C = 47 uF`, `R = 5 ohm`, `Kp = 0.03`,
      `Ki = 100.0`; switching-rate sampled error `|v(out)[0] - V_ref|`; 6000-period
      settled transient comparison across `[i_L, v_C, integrator_state, prev_error]`):*
      - `exact`: 4 steps, 12 runs, `|v0 - V_ref| = 4.086e-14 V`
        (`|mean - V_ref| = 5.678e-5 V` was a trapezoid over the orbit's 3 stored
        samples, not the mean; the time-averaged ripple offset is 5.458e-5 V on
        `pwl` T/100 and 5.459e-5 V on `trbdf2`, equal to the long transient's —
        correction 2026-11-25, task 7.2 reworded to assert the sampled error), `max|x0 - x_tr| = 3.554e-10`
        (`ctrl_diff = 1.765e-12`).
      - `pwl` (`dt = T/100`, `control_tick="state"`): 4 steps, 12 runs,
        `|v0 - V_ref| = 3.375e-14 V`, `max|x0 - x_tr| = 3.523e-10`
        (`ctrl_diff = 1.270e-12`).
      - `trbdf2` (`dt = T/100`): 5 steps, 13 runs,
        `|v0 - V_ref| = 3.197e-14 V`, `max|x0 - x_tr| = 3.777e-9`
        (`ctrl_diff = 3.777e-9`, `< 1e-8` across all states).

Port note (2026-11-25): entry 23 came from another tool (gemini/wip-2026-11-24
78d06266 + dd72a62e) and was re-verified on 527e644d: its tests pass, every
existing PSS / shooting / sensitivity call is byte-identical, and the exact
orbit's sampled error 4.1e-14 V was measured again. Task 7.2 was reworded
and closed 2026-11-25: its "zero mean error within 1e-9" is not what a
once-per-period PI regulates; the test asserts the sampled error, the orbit
against the long transient, and the mean error as the ripple offset (the
"5.68e-5 V" quoted before was a trapezoid over the exact orbit's 3 samples).
Open-loop note: `check_hidden_controller_state` also runs on open-loop
`switch_fn` gates, so a callable object with a `__dict__`, a function
attribute or a closure over mutable state is refused even without a
controller (2.1.0 took it as open-loop) — listed under Changed in the
CHANGELOG. On success the caller's control graph is left seeded at the
orbit's controller state (tick counters and recordings cleared).
