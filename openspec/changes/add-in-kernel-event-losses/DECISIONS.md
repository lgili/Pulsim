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

`plan_of` compares the evaluator's config digest with the one taken at
construction (which also keys every design id); a change after the
first batch raises `ValueError` naming `kpis=` / `constraints=`.

## 12. Evaluator options: honoured or refused, never dropped (batch-11 #7)

`Evaluator(options=)` was passed to `simulate` on the transient route
and to the period runs on pss/pwl and pss/trbdf2, and silently dropped
on pss/dsed (and by the kernel batch). Now: pss/dsed refuses any option
by name (`refusal[1] == "options"`; the exact stepper has no step size
or tolerance to honour), the kernel batch sends such an evaluator to
the Python path (which refuses it), and every engine call of the
evaluator turns simulate()'s "ignored by the ... engine" warning (rtol /
atol / dt_init / integrator on pwl, integrator on trbdf2) into the same
refusal. Rows tested: `test_evaluator_options_are_honoured_or_refused_
on_every_route` (dsed: dt, rtol, discontinuity_restart; pwl transient
and pss: dt honoured, rtol / atol refused; trbdf2 pss: rtol honoured,
integrator refused on pss and transient). Unknown keywords were already
a TypeError from simulate.

## Not done / known gaps

Task 3.1-3.3 (variable-step hooks), 4.x (electrical feedback), 5.1
(summaries reading kernel records), 6.2 (tutorial). A resumed run drops
the step in flight; the IR and content hash omit loss ports; a device
that flips twice inside one split step records no event; a cancelled
run (`analysis::Cancelled`) carries no records (the exception has no
result).
