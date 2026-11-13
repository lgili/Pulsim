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
