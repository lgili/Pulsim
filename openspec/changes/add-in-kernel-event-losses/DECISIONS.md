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

## 16. trbdf2 with a junction ground to max_steps: the forcing jumps at every step (diagnosis)

Reported: the three-phase VSI with six RC junctions (`_vsi_case` of
`test_exact_junction_loop.py` on `feat/core-a-complementarity`) raised
`SimulationAborted` (max_steps 1e7) at t = 20 µs on trbdf2, rtol 1e-9 and
1e-10; the same VSI without the junctions runs.

Smallest stalling circuit (bisected on the netlist): ONE switch held ON
(no switching, no event tables), 10 V into 1 Ω, conduction loss into one
RC junction (2 K/W, 200 µF to a 25 V ambient source), 2 ms, atol 1e-12:

| rtol | no junction acc / rej / forced | with the junction |
|---|---|---|
| 1e-6 | 1152 / 6 / 1 | 111 972 / 156 830 / 20 570 |
| 1e-8 | 1656 / 6 / 1 | SimulationAborted (grinding) |
| 1e-10 | 3983 / 6 / 1 | SimulationAborted (grinding) |

The VSI with ONE junction (on INV__HSa, 40 µs, rtol 1e-6): 14 926
accepted, 11 100 rejected, 2085 forced accepts (without: 1021 / 7 / 1).
With the junction alone (an RC node, no loss port) or with losses at a
fixed T_j the counts are the junction-free ones: the FEED causes it.

Trace (a temporary per-attempt print in the error phase, not committed):
every rejected or forced step names a junction capacitor as the largest
component — 13 177 of 13 185 on the one-junction VSI, 2 444 121 of
2 444 130 (2 206 611 rejects + 237 519 forced, 25 µs) on the six-junction
VSI at rtol 1e-9 (the rest are phase inductors). A cascade at t = 1.5004e-5 s (one junction, rtol 1e-6):

| h attempted | heat power E/h | estimate (K) | scale (K) | verdict |
|---|---|---|---|---|
| 4.0e-8 | 1.47e-3 W | 2.400e-8 | 7.7e-11 | reject |
| 1.0e-9 | 5.65e-2 W | 4.018e-8 | 7.7e-11 | reject |
| 2.2e-11 | 2.64 W | 4.060e-8 | 7.7e-11 | reject |
| 4.0e-16 (h_lo) | 1.47e5 W | 4.061e-8 | 7.7e-11 | FORCED |
| 4.0e-16 | 0.205 W | -4.061e-8 | 7.8e-11 | FORCED (f_n = 1.47e5 A) |

The estimate does not move with h. Cause: the tracker delivers the heat
E owed by the last accepted step as the power E/h over the step being
attempted, so the injected current jumps at every step boundary and a
retry with a smaller h raises it as 1/h. The capacitor's history current
i_prev is the LEFT limit (it embeds the previous step's power), and both
the trapezoidal stage and the estimate's f_n read it: TR-BDF2's estimate
of a jump ΔJ at the step start is 2|c_lte|/γ · h·ΔJ/C = 0.138·E/C,
independent of h (measured 4.061e-8 K = 0.138 × 5.88e-11 J / 2e-4 F).
Every h down to h_lo fails, the step is forced, and the forced step's own
power E/h_lo becomes the next step's f_n (the second forced accept). Not
the estimator alone: the trapezoid integrates (J_left + J_now)/2 over the
first stage, so the jump is a real O(E/C) error of that stage, not noise.
The scale makes it worse but is not the cause: the capacitor's voltage is
the temperature RISE (0 at the start), so its scale is atol (here 1e-12
K) for the first microseconds; a kelvin-sized atol would only move the
threshold — the estimate is independent of h, so some h_attempt always
fails once E/C > tolerance/0.138, and tightening rtol makes it worse.
(c), the impulsive / event energy at a commutation, is the same jump with
a larger E (the reject cascade at each edge); (d) and (e) were not seen.

## 17. trbdf2: the heat power changes at the step start, with its right limit in the history

Fix of 16, in the shared tracker and the two `advance` sites of
`trbdf2_transient.hpp` (TR-BDF2 / SDIRK take, ESDIRK take) plus the
slots-only refresh; the error estimate is NOT changed.

1. RIGHT LIMIT. `advance(h, history, right_limit)` computes, per deposit
   node, ΔJ = (power of this attempt) − (power the last commit was solved
   under, `left_`) and adds each capacitor's share of ΔJ to its stored
   current `i_prev`. The share is the capacitors' response to a current
   step with every state held: Kc·y = e_d on the nodes reached from the
   deposit node through capacitors (nodes held by an independent voltage
   source to ground end the walk), share_k = C_k·(y_from − y_to) — 1 for an
   RC, 1 and 1 for a Foster pair, 1 and 0 for a Cauer ladder. Computed once
   (`bind_`, dense LU of a few nodes). The trapezoidal stage then integrates
   the attempt's own constant power from the step start, TR-BDF2 is exact
   for it on a capacitor, and the estimate measures the network's response
   instead of the boundary jump (a real error of the first stage, not only
   an estimator artefact: 16). Bookkeeping (`applied` per capacitor) undoes
   the previous attempt's share on a retry; a commit (`accept_step`) or the
   slots-only refresh (`slots_rewritten`) zeroes it and records `left_`. An
   SDIRK restart attempt passes `right_limit = false`: its backward-Euler
   stages read no stored current, the history is its zeroed copy, and the
   snapshot it restores still holds the share. Not corrected (the old rule,
   the run unchanged): a walk that meets anything but resistors,
   capacitors, inductors, PWL switches / diodes, current sources and
   grounded independent voltage sources; a walk that reaches no fixed node
   (Kc singular); more than 256 nodes.
2. NO LAG FOR CONDUCTION. With a corrected network, the conduction power at
   the step start (`pred_`: v·i, or V_on(|i|, T_j)·|i| on the table rule,
   at the last accepted reading or the event's right limit) is delivered
   over the step and the step's own energy, booked at its commit as
   before, owes the queue only the difference (which may be negative for a
   moment). A constant loss reaches the junction with no lag; a smooth
   loss leaves P'·h²/2 per step for the next one: T_j is second order in h
   for conduction. A commutation's energy and an impulse still reach the
   junction as E/h over the first step after the edge (first order in that
   step: a steady offset ≈ P_events·h/(2C)); a landing shorter than the
   attempt keeps the rest at the SAME power (`h_now_ -= h`).
3. Conservation unchanged: `heat_injected + heat_pending` = every energy
   booked (the delivered prediction is moved to injected at the commit and
   taken off the queue).
4. Snapshot: the loss state is version 2 when `left_` / `pred_` hold
   something (two more lists at the end of the queue lists); version 1 —
   pwl, fixed T_j, every older snapshot — still loads (no field-count change).
5. Untouched: pwl (`accept` / `advance(dt)`; `inject` adds an empty list),
   fixed-T_j loss ports (no deposit node: no binding, no prediction), runs
   without loss ports (`lt` null). Byte compares vs 061e4099 identical:
   goldens 21/21, stateful 41, engines 35/35, steps 28/28, pwl_identity
   390/390, trbdf2_identity 371/371. Instructions per accepted step
   (interleaved, min of 3): buck pwl 3099.5 → 3099.1, vsi pwl 4764.0 →
   4769.3 (+0.11 %), buck trbdf2 11033.9 → 11019.5, bridge trbdf2 12784.0 →
   12772.3.

Steps (accepted / rejected / forced), atol 1e-12, before → after:

| case | rtol | before | after |
|---|---|---|---|
| held switch + RC (2 ms) | 1e-6 | 111 972 / 156 830 / 20 570 | 1137 / 6 / 1 |
| | 1e-8 | aborted (grinding) | 1596 / 6 / 1 |
| | 1e-10 | aborted (grinding) | 4665 / 6 / 1 |
| buck, 2 junctions (202.5 µs) | 1e-6 | 17 055 / 17 312 / 2777 | 1035 / 7 / 1 |
| | 1e-8 | 26 492 / 30 905 / 4692 | 1106 / 7 / 1 |
| | 1e-9 | 46 824 / 78 451 / 7848 | 1410 / 7 / 1 |
| | 1e-10 | 95 289 / 145 131 / 15 494 | 2846 / 26 / 1 |
| VSI, 6 junctions (2.03 ms) | 1e-6 | 38 354 / 34 169 / 6469 | 1488 / 90 / 1 |
| | 1e-8 | max_steps at 20 µs | 5781 / 183 / 1 |
| | 1e-9 | max_steps at 20 µs | 12 160 / 172 / 1 |
| | 1e-10 | max_steps at 20 µs | 25 869 / 244 / 1 |

VSI 1e-9 → 1e-10: ×2.13 steps (TR-BDF2's rtol^(-1/3) = 2.15); the buck is
h_max-bound (span/1000) above 1e-9. The forced accept left is the start's.

Truths (`test_trbdf2_junction_feed.py`). A switch closing at 10 µs into a
resistor (a constant loss, no electrical state) feeding an RC, a Foster
pair and a Cauer ladder: max |T_j − closed form| / rise at rtol 1e-4 / 1e-6
/ 1e-8 / 1e-10 = RC 3.7e-7 / 3.7e-7 / 3.0e-7 / 2.6e-8, Foster 1.0e-5 /
6.7e-6 / 4.3e-7 / 2.2e-8, Cauer 9.2e-6 / 4.5e-6 / 3.2e-7 / 1.7e-8 (before:
aborted at 1e-6 and below); esdirk43 at 1e-10 ≤ 1.1e-8 (before: the same
stall). Ledger exact (0 to the last bit); the RC's own energy C·θ + ∫θ/R
equals heat_injected within 1e-6 (the trapezoid on samples). The RL
chopper into one RC (closed form with impulses, test_exact_junction_loop's
`_rc_truth`): max error / rise 9.0e-4 / 2.6e-4 / 3.1e-5 at rtol 1e-6 / 1e-8
/ 1e-10 (1031 / 3315 / 15 278 steps); before: 3.5e-4 at 1e-6 with 25 649
steps (the forced accepts at h_lo delivered each commutation's energy over
femtoseconds), aborted at 1e-8 and 1e-10. Richardson on trbdf2 (T_j at ten
times vs rtol 1e-12 / 1e-11): buck 4.1e-4 (h_max-bound) … 8.3e-5 (1e-10),
6.0e-6 (1e-11); VSI 3.9e-2 (1e-6), 1.2e-2 (1e-9), 8.4e-5 (1e-10) — its
large values at loose rtol are samples within 1e-7 s of a diode
commutation that moves with rtol (a 20 K event energy on one side of the
sample or the other). The old engine's buck at rtol 1e-10 (95 289 steps)
and the new one at 1e-12 agree within 2.9e-6 of the rise.

Test moved: `test_trbdf2_resume_continues_the_loss_record` runs at rtol
1e-8 (was 1e-6, the old engine aborts at 1e-8 on this circuit): at 1e-6
the new junction is 6.4e-3 K (full) / 3.2e-3 K (resumed) from the closed
form at t_end (the old one, with 25× more steps, 9.6e-5 K), so two step
sequences cannot meet the test's 1e-4 K; at 1e-8 they differ by 1.2e-5 K
and the energies by < 1e-9. No other existing test moved.

Open (hand-off): (i) a commutation's energy as a STATE JUMP at the edge
(y·E on the walk's nodes and the capacitors' v_prev, the slots re-derived
from Kc⁻¹(−G·y·E)) would remove the first-order offset of 2 — the data are
in `bind_`; its sites are next to the event hooks (1885, 3774), left for the
merge of the recorder branch. Not estimated today: the estimate sees a
constant power over the step, so that offset is an estimator blind spot,
bounded by one post-edge step's heat. (ii) The first SDIRK step of a run
books 64.6 % of its conduction (p(x_g) reads half the power; seen as a
-3.5e-6 relative conduction deficit on the 2 ms RC), pre-existing, not
chased. (iii) The PSFB's non-convergence on trbdf2 has no loss port and is
not this cause.

Review (fresh-context reader) and what was done with it:

- S1 fixed: a run ending on an attempt it did not commit (an End sliver)
  left that attempt's share in the history the final snapshot saved, so a
  resume added it twice. `retract()` takes it back before `hs.save`.
- S2 fixed: the chopper's closed form is now a test
  (`test_the_switching_chopper_junction_follows_its_closed_form`, rtol 1e-8,
  1.5e-3 of the rise; 2.6e-4 measured).
- N1 fixed: with no correctable network `slots_rewritten` does nothing,
  and the prediction is made only for nodes that have shares (a network
  `bind_` skips keeps the old rule). Left: in a circuit with both kinds,
  a landing shortens `h_now_` for the old-rule nodes too (the rest of their
  energy keeps its power instead of spreading over the attempted length).
  No test has such a circuit.
- N2 fixed: after a version-1 restore the history's power is unknown, so
  the correction waits for the first commit (`left_known_`).
- A branch with no registered device kind counts as plain in
  `classify_nodes_` (a controlled source would be read as passive); no
  shipped circuit puts one on a thermal node.

## 18. Python summaries read `res.loss_events` when present (task 5.1)

Ported from gemini/wip-2026-11-24 38a6e535 + 899b5ff5 (another tool's work, reviewed
and re-verified on 527e644d).

- `device_loss_summary` maps the result's kernel records by device name; a circuit
  without `set_device_losses` gets an empty map and the post-hoc path is untouched.
- A device with a kernel record takes its switching counts and energies from
  `res.loss_events(name)` over the half-open window `[t_from, t_to)` (adjacent windows
  partition the events; a resumed run's inherited events fall outside its window),
  summed with `math.fsum`. It does so with or without `switch_specs` / `diode_specs`:
  passing both never double-counts, and the ignored spec warns (named device). A port
  with neither an `E_on` nor an `E_off` table (`DeviceLossRecord.has_e_on` /
  `has_e_off`, new, false) leaves the device to the post-hoc path: the caller's spec
  if given, else no switching keys (never a 0 J that nothing measured). A zero-valued
  table or a ZVS edge keeps the kernel's 0 J.
- `device_thermal_summary` takes `P_sw_avg` from `device_loss_summary`, so it follows.
- Evidence: `test_loss_summary_kernel_records.py` (28 cases): fixed-`Tj` parity with
  the post-hoc path on a twin circuit within 1e-9 (pwl, pwl + restart rule, dsed) and
  1e-5 (trbdf2); the restart-rule cases were added in the port (STATE lesson: parity
  runs with the rule on too). A 1e-6 error injected into the kernel read fails 15.
- Reverse: remove `_kernel_loss_records`, `_use_kernel_switching`,
  `_kernel_switching_loss` and their three call sites in `losses.py`.

## Not done / known gaps

The exact engine (`dsed` / `exact`) still refuses a loss port (its exact
quadratic-form conduction integral and the event booking from its exact
limits are not wired; `scheduler.hpp` untouched). Task 3.3's dsed column,
4.x (electrical feedback), 6.2
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
