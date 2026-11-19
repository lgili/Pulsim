# Decisions — add-in-kernel-event-losses

Format: question → option → evidence → how to reverse.

## 1. Group 1 alone (slice 3 of the optimisation track)

→ `thermal/loss_table.hpp` (`LossTable3D`, `ConductionTable`), the
bindings, `LossTable.to_kernel()`; no kernel caller yet (group 2 hooks
the reading rule into `solver/run_transient.hpp`, a file another change
is rewriting). Evidence: 10 000 random points inside and outside the
grid equal `LossTable.__call__` with a maximum relative difference of
0 (the operations are the same, in the same order); the temperature
table returns 1.5 mJ at 75 °C and 2.5 mJ at 175 °C to 1e-12
(`python/tests/test_loss_table_kernel.py`, `core/tests/thermal/`).
Same result against Eigen 3.4.0 (the header uses no Eigen).
→ Reverse: delete the header and the two bindings.

## 2. A non-finite query is refused, not NaN

`LossTable.__call__` returns NaN for a NaN coordinate (numpy
propagates it, `np.maximum(NaN, 0)` is NaN). In the kernel a NaN energy
would be deposited into a thermal node and spread through the network
before anything reads it, so the kernel table raises
`std::invalid_argument` naming the axis ("i query is not finite"). An
infinite axis value is refused at construction for the same reason
(Python accepts it and returns NaN weights).
→ Reverse: drop the `isfinite` check in `detail::axis_weight`.

## 3. `ConductionTable` has no Python twin

`V_on(i, T_j)` is read with the same rule; its oracle is a `LossTable`
with a single-point `v` axis (the dimension collapses), 2 000 random
points within 1e-12.

## 4. Group 2 on `pwl`: one tracker, three calls

→ `solver/loss_tracker.hpp`: `advance(h)` at the start of each outer
step turns the energy the last accepted step queued into a power;
`inject(b)` adds it to the RHS of every solve of the step
(`accumulate_b_extra`: retries, dt halvings, split sub-steps, restart
stages all see the same power); `accept()` runs once, after the history
commit, and is the only place events and conduction energy are read.
A rolled-back attempt never reaches `accept()`, so nothing is counted
twice. The ledger moves the energy from pending to injected when the
step it was owed to is accepted; `heat_injected + heat_pending` equals
the recorded heat within 1e-12 (an aborted run keeps the unaccepted
step's energy pending). The static linear path refuses a loss port.
→ Reverse: delete the header and the four `lt->` lines.

## 5. Deposition is a rectangular power, received over two steps

The heat of step k is `E/dt` over step k+1. On trapezoidal steps the
capacitor history carries half of each step's injection into the next,
so the network receives E as h/2 + h/2; the run sum is exact (the
adiabatic test: 1.000 K within 1e-6), a BE/SDIRK restart step is within
one step's heat. Not a state jump: a jump would need a per-network
impulse response, which `pwl` does not have.

## 6. Conduction energy per step mirrors `_branch_energy`

A same-stamp step is the exact product of the step's two lines; a step
whose stamp flips keeps the trapezoid of the end powers, as
`losses._branch_energy` does (the line product reads a freewheel diode
negative). With the line product on the flip step, a 10 Ω chopper put
1.7 µJ of fictitious heat per edge into the junction and the diode rig
disagreed with `device_loss_summary` by 2e-5 J over the run; with the
rule above both agree within 1e-12.

## 7. A deposit node must have a capacitor and no voltage source

The spec asks to refuse a junction reached only through resistors. The
check is local: a capacitor on the node (else its rise depends on dt)
and no voltage-type source on it (the heat would leave through it; the
documented ambient node `amb` would otherwise pass). Refused at run
start by name, device and node. A typo naming an electrical node with
a capacitor is not detectable.

## 8. `trbdf2`, `dsed`, `bdf1`, `auto` and the adaptive driver refuse

No unified event queue yet (task 3.1). trbdf2 changes `h` per attempt
and has several solve sites; adding the hook there now would be undone
by 3.1, and another agent edits that file. Each engine refuses at entry
by name: `trbdf2_transient.hpp` and `bdf1.hpp` one line each
(`refuse_device_losses`), dsed / exact through `require_supported` (the
adapter constructor and Python's `_require_supported`). What 3.1
replaces: the trbdf2 line becomes `advance(h)` at each attempt start,
`inject` in `accumulate_sources`, `accept` at the accepted and landing
pushes; the dsed one `accept` at each event. `run_transient_adaptive`
refuses too: it resumes in segments and would drop the heat in flight.
The shooting route of `periodic_steady_state` (pwl / trbdf2) refuses
for the same reason (integration batch 12 review): each period is a
fresh run from a seeded state, which drops the turn-on at the period
start and the last step's heat.

## 9. One reading rule on both routes; the transient settles on the orbit

`thermal::commutation_point` is the rule; `_period_kpis.commutations`
now calls it through `_pulsim._commutation_points`, so the orbit route
and the transient share one function. Slice-6 buck, R_th = 20 K/W: the
long transient's period mean T_j minus the orbit fixed point is
-2.8e-3 K at C_th = 3e-4 J/K and -8.5e-4 K at 1e-3; -0.084 K at 1e-5.
The residual scales with E/C_th: the orbit holds T_j constant over the
period, the transient reads T_j at x⁻, the low point of the junction's
sawtooth. It does not shrink with dt (T/200 to T/1600: -0.0859 to
-0.0872 K at C_th = 1e-5), so it is not the reading.

## 10. Open loop with a heat node

`Tj=` alone records; `Tj=` with `heat_node=` reads at a fixed T_j and
still heats that node (test 2.6 needs a deposited, open-loop heat).
`junction=` with `heat_node=` is refused.

## 11. The Evaluator refuses kpis / constraints changed after a batch

`evaluate`, `evaluate_many` (every backend) and `plan_of` refuse a
`kpis=` that was reassigned or a `constraints=` that changed since
construction (`ValueError` naming them). Not a config digest: a
schedule closure that records its calls changes its digest and is
legitimate (the first attempt refused `test_schedule_per_design_and_
parameters`), and a digest per `evaluate` broke the Python-call bound
(912 against 500). An `Evaluator(circuit=...)` whose circuit carries a
loss port is refused by name: `value_hash` (design ids, the process
backend's cache) does not see loss ports, and the evaluator's losses
come from `loss_options=`.

## 12. Evaluator options: honoured or refused, never dropped (batch-11 #7)

`Evaluator(options=)` was passed to `simulate` on the transient route
and to the period runs on pss/pwl and pss/trbdf2, and silently dropped
on pss/dsed (and by the kernel batch). Now: pss/dsed refuses any option
by name (`refusal[1] == "options"`; the exact stepper has no step size
or tolerance to honour), the kernel batch sends such an evaluator to
the Python path (which refuses it), and the keywords simulate() only
warns an engine ignores (rtol / atol / dt_init / integrator /
stiffness_threshold / h_bdf2 on pwl; the last three on trbdf2) are
refused by an explicit list before the run (after it for `auto`, against
`engine_used`). Not by turning the warning into an error: that edits the
process-wide filter list, which `evaluate_many`'s threads would race. Rows tested: `test_evaluator_options_are_honoured_or_refused_
on_every_route` (dsed: dt, rtol, discontinuity_restart; pwl transient
and pss: dt honoured, rtol / atol refused; trbdf2 pss: rtol honoured,
integrator refused on pss and transient). Unknown keywords were already
a TypeError from simulate.

## 13. trbdf2: one tracker, the integrator's quadrature, the engine's limits

The trbdf2 engine runs the same `LossTracker` as pwl (decision 8's plan):
`advance(h)` at each attempt's stage-1 solve, `inject` at the end of
`accumulate_sources` (every solve: stages, probes, landings, records),
`accept_step` at the three commit sites, `event` beside `hooks->event` at
the two sites that report left / right limits (gate-edge settle, diode
landing). Conduction per committed step uses the integrator's weights:
TR-BDF2 `(w, w, d)` at `(0, γh, h)`, `w = 1/(2(2-γ))`; SDIRK(2,2)
`(1-γs, γs)` at `(γs h, h)`; the trapezoid on landings and committed
probes. The event is read by `thermal::commutation_point` on the two
limits, with the same table call as pwl; the first step under the right
side starts its quadrature from the right limit, so the flip is not read
again; a flip no event reported falls back to the step ends (pwl's rule).
The heat owed is consumed in proportion to the committed length (a
landing shorter than the attempt keeps the rest, rescaled by the next
`advance`). pwl's `accept()` is unchanged (its numbers byte-identical).
Reviewer (fresh context): an event re-reported at a coincident gate edge
after a landing (stale left mask) was booked twice — skipped now; a
pending event whose right side was never committed is taken back
(`unbook_`).

Truth (closed-form RL chopper, `test_device_losses_variable_step.py`,
40 commutations per device, E = k·v·i tables), relative error:

| run | S1 event | S1 cond | D1 event | D1 cond |
|---|---|---|---|---|
| pwl dt 1e-7 | -1.29e-4 | -1.82e-4 | -2.06e-4 | -2.37e-4 |
| pwl dt 2.5e-8 | -3.21e-5 | -4.52e-5 | -5.17e-5 | -5.93e-5 |
| trbdf2 default (h_max = span/1000) | +1.50e-6 | +7.68e-6 | -7.5e-8 | +6.66e-5 |
| trbdf2 rtol 1e-6 | +1.27e-6 | +5.93e-6 | -3.0e-7 | +6.63e-5 |
| trbdf2 rtol 1e-6, dt (h_max) 2.5e-7 | | +4.73e-7 | | +4.12e-6 |

pwl is first order at the edges (ratio 4.0 per dt/4); trbdf2's conduction
error is set by h_max, not rtol, here (second order: 16× per h_max/4).

## 14. Impulsive energy is a loss of the element that carried it

`impulses(result.reinit_records)` after each `reinit.record` (pwl and
trbdf2) books `dW` to the device named by `cause` (`E_impulse`) and to its
junction. The step that carried it conducts its end power only (as
`device_loss_summary` reads the pulse step): its quadrature read the
impulse's mean current against the voltage it collapsed (the capacitor
loop's port read -38.6 µJ, the junction cooled). An E_on table AND an
impulse on one commutation are two categories (table: not drawn from the
circuit; impulse: drawn); a table measured with the device's own Coss
would count that discharge twice if the circuit also models Coss as an
ideal capacitor — documented, not detected. The summary's and simulate()'s
warnings are gone (`impulse_in_junction=True`).

## 15. Resume carries the tracker (pickle field 17)

`SolverSnapshot::loss_state` = `LossTracker::save()`: heat owed (queue and
the current power), ledgers (Neumaier pairs), each device's last reading,
its pending event and its events. A resume restores it in place of
`begin()`, so the resumed result's records are the uninterrupted run's.
Pickle version 5 gains field 17 (18 fields); 17-field tuples still load.
Loss ports are hashed in the circuit IR when present (structure: names,
nodes, conduction kind, which tables; values: Tj and the table arrays).

## Not done / known gaps

The exact engine (`dsed` / `exact`) still refuses a loss port (its exact
quadratic-form conduction integral and the event booking from its exact
limits are not wired; `scheduler.hpp` untouched). Task 3.3's dsed column,
4.x (electrical feedback), 5.1 (summaries reading kernel records), 6.2
(tutorial). A device that flips twice inside one pwl split step records no
event; a cancelled run (`analysis::Cancelled`) carries no records.

Hand-off for the exact engine (not started): the scheduler reports only
`hook_accepted_(t0, t, x_left, mask)` (`dsed/scheduler.hpp:525`, `:713`);
the right limit is the post-`fire_event_` state there. The smallest entry
point is an `on_event`-style call beside `fire_event_` with
(t, x_pre_event, mask_before, x, current_mask). A PWL device's conduction
on an LTI segment is `g·∫(w·x)² − g·V_f·∫w·x`, which
`period_integrals.hpp` already computes exactly for linear observables
(`PeriodObservable`, the augmented exponential); the heat input needs a
piecewise-constant source the propagator integrates exactly.
Not measured: a hard-switched half-bridge with output capacitances (the
requested energy-balance case); the capacitor loop is the balance test.
