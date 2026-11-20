# Release findings, group B — event samples and result integrals

Branch `fix/core-release-b-event-samples` (from the PR head 27c00520). Tests:
`python/tests/test_release_findings_b.py` (the release options passed explicitly:
`engine='pwl', discontinuity_restart=True, enable_substep_state_correction=True`;
`engine='dsed', output_rtol=1e-5`).

## One design gap, three faces (the orchestrator's reading, checked)

1. **An event instant has two values.** Confirmed with a nuance per engine.
   - Exact engine: the right limit of every event IS stored
     (`right_limit_index` / `right_limit_states`; trace: the 230 V bridge's
     turn-on rows 14, 24621, 44623 ... all in `right_limit_index`, left 7e-10 A,
     right 13.6 A). The defect is in the consumers: `res.i()` is the left
     limits, and `time_average(res.times, res.i())`, a trapezoid, `max|i|`
     read them alone (F-6-2, F-5-6, F-5-4 exact).
   - Fixed engine under the restart rule: no right limit is stored at all,
     and an event located INSIDE a step (the split) has no row.
2. **The quadrature of a restart step.** Confirmed: the stored samples and
   the loss port integrate an SDIRK(2,2) step as the trapezoid of its ends;
   the trapezoidal engine's identity "trapezoid of the samples = the
   integrator's quadrature" (charge closing to 3e-12) is gone.
3. **The source value at an edge row (F-6-5).** The restart rule solves the
   row AT a source jump with the left limit (`value_at(t − dt·1e-3)`), the row
   being the left limit; `res.i(source)` evaluated the waveform at t (its
   right value): rows 3e-4, 6e-4, 8e-4 of the pulse source.

## Storage convention (decided once, all engines)

A stored sample at an event is its LEFT limit. Right limits live beside the
samples (`result.right_limit_index` / `right_limit_states` /
`right_limit_switch_states`, the exact engine's existing record, now a field
of the kernel result: None = no record). An event inside a fixed step is
`result.inner_events` (its time, left limit, right limit). `res.times` does
not change on any engine (no layout change, no option). The documented
accessor `pulsim.with_event_limits(result)` inserts the inner events as rows
and each right limit one ulp after its event; `steady_state_stats(res, name)`,
`device_loss_summary`, `device_thermal_summary` read through it.

On the fixed engine the "right limit" of a restart (sub)step is the start of
the line whose trapezoid is the step's own quadrature,
`2·w1·X1 + (1 − 2·w1)·X2` (w1 = 1 − γ; γ for the backward-Euler re-take):
exact for the integral of any quantity linear in the state (a charge, KCL),
second order for a product (v·i), equal to the right limit to O(h²) (for a
linear trajectory `x_s = x_n` exactly: 4γ − 2γ² − 1 = 0). The loss port
integrates the same lines (one representation for the kernel and the
summaries).

## Per finding (trace, cause)

| id | trace | cause |
|---|---|---|
| F-6-2 | turn-on rows in `right_limit_index`; time_average of the two-limit trace = summary i_avg to 5e-15 | consumer read left limits |
| F-5-6 | 466.664 A = right limit at a turn-on (1.4 V / 3 mΩ), in the two-limit trace | consumer read left limits |
| F-5-4 exact | charge: left limits −1.23e-3, two limits +1.1e-4 (output_rtol interpolation) | consumer read left limits |
| F-5-4/5 pwl | charge +3.1e-3 (restart only), +4.9e-3 (split); restart-less −2.9e-12 | no right limit / inner row; trapezoid ≠ SDIRK quadrature |
| F-2-2 | boost DCM D1 charge 5.6e-3 off | same |
| F-5-7 | DCM duty 0.2180 vs exact 0.2210 (V_th = 0.7) | the split's commutation has no row: right-hold duty misses the sub-step before it |
| F-6-4 | RL chopper S1 cond +1.86e-3 (dt 1e-7), +9.3e-4 (5e-8): first order; restart off 1.8e-4 | LossTracker::accept: a flipping step's trapezoid of end powers |
| F-6-5 | rows 3e-4/6e-4/8e-4: Ip 0/1/0, Rsq 1/0/1 | res.i evaluated the source's right value at a left-limit row |
| F-2-4 | ratio 0.9777/1.0109 (rule grid) vs 0.9942/0.9945 (event grid) | compute_temperature resampled the event trace onto a uniform grid whose spacing fell between the edges (first order); P_avg itself right to 1e-11 |
| F-5-2 | column 6 (S current, OFF 2.4e-8 A) read 1.43e-10 off at k=192: landing at kT − 7.4e-18, np.interp mixed the next (ON) sample | interpolation across the edge's jump |
| F-4-4 | — | the in-loop recorder sees the restart step's ends only; capacitor readings are the companion's END current |

## Hand-off (2026-11-19, branch tip see `git log`)

Measured on this branch with the release options explicit
(`test_release_findings_b.py`, 14 pass + 1 strict xfail):

| id | status | number before -> after, truth |
|---|---|---|
| F-6-2 | fixed, verified | time_average over D1: left limits 0.160684 -> two-limit 0.161027 = summary i_avg to 5e-15 (truth: the summary's step-start integration; pwl dt=1e-7 0.161005) |
| F-5-6 | fixed, verified | summary 466.664 A = max of the two-limit trace (rel 1e-12) and of the source's |
| F-5-4 exact | fixed, verified | bridge charge -1.23e-3 -> +1.1e-4 (KCL charge from node voltages; the output rule's tolerance) |
| F-5-4 / F-5-5 pwl | fixed, verified | +4.9e-3 / -1.3e-5 -> <= 1e-9 (KCL identity, same trace) at both V_th |
| F-2-2 | fixed, verified | boost D1 charge vs load charge (CCM and DCM): trapezoid of the two-limit trace 1e-9, summary i_avg 1e-6 |
| F-5-7 | fixed, verified | DCM duty 0.2180 vs exact 0.2210 -> within 2e-3 (V_th 0.7 and 0) |
| F-6-4 | fixed, verified | RL chopper cond S1 1.86e-3 -> <= 2.5e-4, error ratio dt/(dt/2) >= 3 (second order; closed form) |
| F-6-5 | fixed, verified on pwl | Rsq vs Ip rows 3e-4/6e-4/8e-4 now equal (1e-9); the exact engine takes the same left read — NOT checked against its rows |
| F-2-4 | fixed, verified | ratio 0.9777/1.0109 -> 0.9947/0.9943 (pwl T/1000 0.9946/0.9943) |
| F-5-2 | fixed, verified | settle 1.3e-4 "not_settled" -> ok |
| F-5-5 | estimator limit REFUTED: with the restart lines the sample trapezoid closes to round-off (<= 1e-9) at V_th = 0 too |
| F-4-4 | not started (strict xfail) | in-loop recorder at restart steps |

Functions changed: `LossTracker::accept` (+ `accept_lines_`, `StepLines`),
`run_transient` (`restart_lined`, `record_lines`, the step phases' clear /
retry / accept / record calls), `SimulationResult` event-limit fields and
their bindings (`right_limit_*` get/set, `inner_events`),
`_result_views.with_inner_events` / `with_event_limits` / `_ResultView`
(nested views), `losses.device_loss_summary` and
`thermal.device_thermal_summary` (read inner events; thermal's PL Foster on
event results), `thermal.compute_temperature(piecewise_linear=)`,
`stats.steady_state_stats`, `design._settle_error`, `res.i` of waveform
current sources.

NOT verified: full pytest, ctest, the instruction-count cost of the four
cases (the change adds work only on restart steps: one state copy at stage 1
and one axpy), stubs `--check` (new properties: regenerate
`_pulsim.pyi`), reviewer pass, Eigen 3.4.0. Byte compares: see the commit
after this note.

Next three steps for a successor:
1. F-4-4: feed the LoopFeed the step's lines. In `step_record`, when
   `q_ok && !lines.empty()`, call `hooks->step` per line (start right limit
   as the step's start state = the feed's "jump" row; one hook step per inner
   event) instead of the one step; the feed's Capacitor reading returns the
   companion's END current for any x, so pass the capacitor currents of each
   line point (stage 1: C·(X1 − x0)/(γh); the line start 2·w1·i(X1) +
   (1 − 2·w1)·i(X2)) — e.g. an `i_cap` override span in `StepHooks::step`.
   Truth: the window charge = C·Δv (test_f44).
2. Run the full pytest and list every existing test whose number moved
   (restart-rule pwl summaries, dsed thermal T_j, design on restart runs);
   regenerate stubs; reviewer.
3. Cost: instructions per accepted step on buck/vsi pwl and buck/bridge
   trbdf2 (rule off must be unchanged: `lines` is touched only when
   `restart_rule`).
