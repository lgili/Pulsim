# Group A — impulsive events under the restart rule (F-5-1, F-3-3, F-3-7, F-3-1, F-5-10)

> **THIRD SESSION (2026-11-20) — the impulse detector is ONE rule [M].** The second session's stack
> (energy ratio 0.81, 1e-6 stored-energy floor, 1 % of the kind's magnitude, 1e-4 A/V floor) is gone.
> - **Probe: two readings, a jump must show in BOTH.** x2 = a second BE stage of γh from x1 (stage 1's
>   bits frozen, same factor), x0+ = 2·x1 − x2, read twice: (a) sources frozen at t1, (b) BUILT-IN sources
>   moved to t2 = t + 2γh (unless one jumps in (t1, t2]; a `b_extra_fn` keeps its t1 value — called once
>   per solve time). (a) alone fails on a fast mode that TRACKS its forcing (it extrapolates to x0 + γh·ẋ:
>   `test_dt_retry_of_a_step_that_holds_an_edge`, rectifier from rest, frozen d = 43.1 / 40.2 / 30.5 V on C
>   at t = 0 / 1 / 2 ms, histories moved, dt-retries at 4, 5, 6 ms; moved d = −0.31 / 2.3 / 3.7 V against
>   the bound 149 / 144 / 102 V, retries 21, 22, 23 ms, test passes). (b) alone fails on a source-driven
>   device kink: the MOSFET level-1 gate ramp crossing V_th at t = 0 (`mos_l1` golden, pwl+restart) gave
>   a 1.8e-14 J record; with both readings 0 records and the engines compare is identical again.
> - **Rule** (per state: capacitor voltage, inductor current; linear algebra and constants in the
>   `consistent_start` comment): impulsive iff some state has
>   |x0+ − x0| > h·max(|ẋ(0−)|, |x2 − x1|/γh) + 1e-8·(1 + |x0| + |x1|) [+ G_off·(1 + V_max) for currents].
>   The rate term: a consistent start has x0+ − x0 = O(γ²h²ẍ), bounded by the state's motion over the
>   restart step h at its rate before (history slot i/C, v/L) or after (x2 − x1) the event. A linear mode
>   relaxed from a jump J gives |d| = (r/(1+r))²|J| against r/(γ(1+r)²)|J|, r = |λ|γh: a jump when
>   r > 1/γ (τ < γ²h ≈ 0.09 h) — derived, not tuned. The tolerance: 1e-8 relative with a one-unit floor is
>   the event services' noise (`events::held_tolerance`), which holds the Newton tolerance (1e-9) and the
>   round-off; for currents also its leakage scale, G_off = Σ g_off of the conductance-form switches and
>   diodes times the node voltages (the CCM buck's zero start: i_L 0 -> 2.4e-8 A through two g_off at 24 V
>   was a "jump" of 2.9e-20 J that re-routed the start: 8 restart factorizations instead of 4 and one
>   stage-2 retry; under the tolerance 5e-8 A it is not). Ideal-element (structural) modes stay with the
>   tracker's projection and record (one service per jump, not two records): routing, not a threshold.
> - **Device histories (reviewer item).** Stage 1 is re-solved with the stateful device histories at x0
>   (restored), not x1: their states (junction / tail charge, machine flux) are continuous through the
>   event, and their laws would read x0+'s extrapolated algebraic entries (F-5-1: v_sw −6.8e8 V). The O(h)
>   lag of "start from x1" is gone. Not measured on a circuit with a stateful device AT an impulse.
> - **Saturable inductors / nonlinear capacitors (reviewer item): refused by name.** Their state (core
>   current, capacitor voltage) is probed by the same rule; the rate before the event comes from the law
>   (V_L_old / L(i_L_old), i_prev / C(v_prev)) — without it the `everything` circuit's S turn-off kink at
>   5.02 us was refused (Cj −1.57e-3 V); with it only the 10 us turn-on, which shorts Cj (1 nF) through
>   R_on 1 mΩ (τ = 1 ps vs dt 20 ns: a real −24 V discharge), is refused; their
>   histories cannot be moved to x0+ (λ(i), q(v) read x0+), so a jump on one raises
>   `UnsupportedError(feature='impulse', device=<name>, kind='saturable_inductor'|'nonlinear_capacitor')`
>   under the rule ("run this circuit without discontinuity_restart"). Tests: a saturable core cut by its
>   switch; a nonlinear capacitor across a buck switch (no refusal through the 5 us turn-off, refused at
>   the 10 us turn-on). The stateful compare's `everything_pwl_restart` therefore raises now.
> - **Split: summed signal only on a FLOATING island (reviewer item).** Diodes the decided mode flips that
>   touch the same node island (union of every branch but the OFF diodes) with no path to ground share
>   one summed signal (the island's common mode is set by the OFF diodes' leakage alone: F-3-1's
>   floating bridge). Every other diode keeps its own crossing. Measured: two grounded rectifiers whose
>   turn-offs fall 0.1 dt apart in one step — D1 located at 59.822078 dt, as alone (bit-identical o1
>   waveform); the summed rule put both at a common time. D2 (inside the restart remainder, not located,
>   pre-existing: one located event per step) error vs dt/32: 1.61e-3 V combined vs 1.52e-3 V alone.
>   F-3-1 still passes (1-ulp sensitivity 1.8e-12 .. 2.4e-11 V).
> - **Numbers on this rule** (F-5-1 / F-3-3 unchanged): F-5-1 i_L 100 -> −3.8e-5 A, one record dW
>   4.99999881 J (truth 5 J); F-3-3 peak 16.381985087 / 16.381988999 / 16.381990009 A at dt 10 / 5 / 2.5 ns
>   (exact 16.381990162), min +1.5e-7 A. 9 restart files + release_findings_a: all pass (F-3-7 xfail).
> - **Verified at this tip [M]** (ext build of the committed header): the 41 test files that use
>   discontinuity_restart / event_split: 1128 passed, 3 skipped, 6 xfailed; test_release_findings_a 9 passed
>   1 xfailed. Byte compares vs the PR head, DEFAULT options: goldens all identical (buck/vsi_pwl_restart
>   identical, only `cache.event_hits` counters move), stateful identical except `everything_pwl_restart`
>   (now refused, above), engines ALL identical, steps ALL identical, trbdf2_identity 371/371,
>   pwl_identity 347/379 — the 32 that differ are all pwl+restart / pwl+split. Classified: 9 counter-only
>   (the probe's Newton solves: ll15 split, ll16 ×2, buck_mosfet_nonlinear ×2, ll08 ×2, ll11 ×2); the 23
>   others carry reinit records (impulsive events at commutations: ll14, ll15 restart, half_bridge_transformer_dc,
>   stiff_two_timescale_rc (Ca at t = 0), boost_pfc_open_loop, ll12, ll09, active_clamp_forward, flyback,
>   forward, ll13, synchronous_flyback). Three LOSE the `_implausible_voltage` warning the head had
>   (half_bridge_transformer_dc, ll13, synchronous_flyback). Cost (instructions per accepted step, min of 3,
>   same tree, two runs): buck_pwl 1.0053 / 1.0047, vsi_pwl 1.0032 / 1.0034, buck_trbdf2 0.9999 / 0.9993,
>   bridge_trbdf2 0.9983 / 1.0008 (within 0.5 %, at the edge for buck_pwl; an earlier build of the same
>   default path measured 1.0006: layout noise); rule on: buck 1.0553 / 1.0544, vsi 1.0207 / 1.0210.
>   One full pytest on aa5d99f0: 6100 passed, 110 skipped, 9 xfailed; ruff clean; stubs up to date.
>   On the next tip (B1 + A1 xfail + F-3-7 parked; extension built with the env-gated trace, removed
>   before commit — the committed header is that build minus inert prints): 41 restart/split test files
>   1129 passed, 3 skipped, 7 xfailed; byte compares exactly as above (pwl_identity the same 32 entries,
>   trbdf2 371/371); reference_status identical cell for cell to aa5d99f0's run (666 cells; pwl+restart
>   115/20/1, pwl+split 118/17/1, the same as the run before the two readings; flyback 0.110 -> 2.3e-6
>   and forward 0.092 -> 2.6e-6 still agree, no cell worse).
>   Clean rebuild of 77fa64b0 (no trace): the 41 files 1129 passed, 3 skipped, 7 xfailed; stubs up to date.
>   NOT run: full ctest; the full pytest on the final tip (it ran on aa5d99f0); cost not re-measured
>   after aa5d99f0 (the later changes touch only the rule-on probe and the split).
> - **Group B field for the event's peak voltage (step 5; their branch NOT merged).** On
>   `fix/core-release-b-event-samples` (36836230) the result carries `right_limit_index` /
>   `right_limit_states` (filled in `record_lines` from `lines.x_start` when `lines.start`) and
>   `inner_event_times` / `inner_event_right` (split-interior events). The impulse's spike lives in the
>   algebraic rows of x0+ = `event_rl_x` (F-5-1: v_sw −6.8e8 V; the cut current's L·di/dt), and nowhere
>   else in the samples. Carry it as: an impulse at a grid step start (`event_rl_t == t_prev_step`):
>   `lines.x_start = event_rl_x`, i.e. the `right_limit_states` row whose `right_limit_index` is that
>   step's start sample; an impulse inside a split (`event_rl_t` == `inner_event_times[j]`):
>   `inner_event_right[j]`. Then `_check_right_limit_voltage` (already scanning those rows on the exact
>   engine) must also run on the fixed engine's finalize path to report it. Nothing is read by group A
>   from those fields; `event_rl_t` / `event_rl_x` stay as the hand-off locals.
> - **Reviewer (opus, fresh context, read-only) on a324a2db.** Reproduced by its scripts (scratchpad p1..p5):
>   - **A1 (high) OPEN, strict xfail `test_a_cut_smaller_than_h_times_the_old_rate_is_an_impulse`.** The
>     bound's old-mode rate h·|v_L(0−)/L| = 10 A masks an 8 A cut (1 µH at 100 V, dt 0.1 µs, S opens with
>     its body diode reverse biased): no record, i_L 8 -> −8.38 A (truth 0, pwl dt 1e-10). Tried: drop the
>     old rate when the event is AT the step start (gate mask / unknown start / snapped source jump,
>     t_from == t_prev). A1 then passes (one record ½·L·8² J, i_L -> <1e-3 A), but the DCM buck's 47 uF
>     output C is flagged at its gate edges (36 µV, 3e-14 J, cause ''), and reference_status moves on 31
>     cells, status unchanged, errors up to +48 % (buck_dcm 3.4e-6 -> 5.0e-6, cascaded_boost_buck,
>     two_phase_interleaved_boost, sync_buck_500khz, long_run_drift_buck, stiff_rc_high_freq_switching).
>     Reverted. Needs a jump-vs-kink test that does not read the old rate's magnitude (the stage-1
>     increment is not it: at an extremum |x1 − x0| = |d| too).
>   - **A2 (high per reviewer) OPEN, by design.** A snubber Rs 1 Ω + Cs 1 nF (τ = 1 ns) at dt 100 ns:
>     τ < γ²h, so Cs is an impulse by the rule (record 4.37 µJ, cause '', kind 'capacitor_loop' though Rs
>     is in series). The waveform is right; `on_impulse='error'` would refuse it. The rule is dt-scaled by
>     construction; `cause`/`kind` naming for a series-R loop is not done.
>   - **A3 (medium) OPEN.** A source edge in (t1, t2] leaves only the frozen reading; a rectifier from rest
>     plus an unrelated 1 V pulse with its edge at 0.4·dt gives a 0.96 mJ record at t = 0 (waveform matches
>     the fine run to 1e-4 V). Needs the moved reading at the edge's left limit.
>   - **B1 (medium) FIXED.** A flipped diode bridging two floating islands now joins them (the isolated
>     bridge's D1 on the transformer side and D4 on the output side). Not reproduced numerically (1-ulp
>     sensitivity 1.1e-10..1.8e-10 V before); F-3-1 and the split tests pass.
>   - **B2 (low) OPEN.** Current sources / VCCS outputs / OFF conductance switches count as galvanic
>     paths in the island union (an island grounded only through them is not "floating").
>   - Also from the reviewer: every dt-retry sub-step under the rule runs the probe (1–2 extra solves);
>     `event_rl_t`/`event_rl_x` are written but read by nothing (hand-off for group B, above).
> - **F-3-7 traced, PARKED for group C (mode selection; branch fix/core-release-c-mode-selection).**
>   Truth: restart-less 6.911 V, linear primary under the rule 6.890 V. Per-iteration trace (temporary
>   env-gated prints in `run_event_iteration` and the Newton loop, removed): the abort is restart STAGE 1
>   (BE of γh) at the HS_A turn-off t = 44.90 µs, solved under the step-start diode mask
>   `0010|0010|1001` (switches HS_A..LS_B | D_HS_A..D_LS_B | D1..D4): D_LS_A still OFF, so the saturable
>   L_leak's current is cut into g_off. Newton iteration 0 drives mid_a to −1.26e4 V; the primary then
>   converges (residual 1e-9), but the rect_neg/vout iterate bounces 0.2–3.3 V at residual 8e-10..2.9e-9
>   for all 50 iterations (round-off at |x| = 1.26e4, roundoff acceptance declined). The event
>   iteration never reaches its first re-decision (`it=0` on every attempt), so D_LS_A is never asked to
>   turn on. Each dt halving doubles |x| (2.5e4 ... 8.1e5 V at dt/64) and the bounce (803 V at the end).
>   The linear primary passes because its stage is a direct LU solve, after which the event iteration
>   flips D_LS_A. Cause: the mode a gate edge is first solved in (the step-start diode bits, an
>   inconsistent mode that cuts an inductor) — mode selection, not the impulse rule. What it needs: the
>   edge's mode decided before the Newton solve (D_LS_A ON at the HS_A turn-off), e.g. the LCP/settle
>   decision at the gate edge applied to the Newton path too. The strict xfail stays.

> **Second session — results (2026-11-19) [M]** (measured on 43c76065 + review fixes, energy ratio 0.25; the byte-compare, pwl_identity and reference_status numbers below are from that build, NOT from the tip):
> - **F-5-1 FIXED.** Restart: i_L 100 -> -141.42135 A before; after 100 -> -3.8e-5 A (max |i_L[1:]| 3.3e-4 A),
>   ONE record t=0 cut_set states ['L'], cause 'QL', dW 4.99999881 J (truth ½·L·i0² = 5 J), lambda -0.1 V·s;
>   on_impulse='error' raises naming 'QL', 'warn' warns. NOT done: the voltage-sanity guard does not see the
>   stage-1 voltage (-3.4e8 V) — it needs a result field (group B owns the result type); the record is the report.
> - **F-3-3 FIXED.** Flyback min i_p -20.92 A -> +1.5e-7 A; peak 16.381985 / 16.381989 / 16.381990 A at dt
>   10 / 5 / 2.5 ns vs exact engine 16.381990162 (errors 5.1e-6, 1.2e-6, 1.5e-7: order 2.1, 2.9). The CCM
>   buck zero start (`test_buck_ccm_stays_second_order_to_t_over_1600`) and `test_one_factor_per_mask_and_no_refactor`
>   pass unchanged (4 factorizations, stage2_be_kept 0). `test_a_failed_stage2_retry_...`: its failing-retry
>   variant changed (the rectifier's zero-start 0 -> 21 V charge of Cout, τ = 5 µs < γ·dt = 15 µs, is now an
>   impulse taken from x0+): 10 -> 12 failed retries, overshoot 37.203 -> 28.016 V (working fallback: 23.999 V);
>   re-pinned. The stage-2 fallback / exemption is still exercised (16 retries on the golden buck), kept.
> - **F-3-1 FIXED.** The split locates the DECIDED mode's commutation: when the event iteration's final bits
>   flip >= 2 diodes that crossed, all one way, the crossing is interpolated on the SUM of their signals (the
>   floating bridge's series pair). 1-ulp sensitivity 3.1e-4 V -> 2.4e-11 V (k = 1..5); event at 33.7040 ms,
>   exact engine 33.704043-33.704054 ms (before: 33.70395 ms at k = 0). Error vs exact after the event:
>   -3.12e-3, +3.2e-4, -3.6e-5, +1.6e-5 V (before -3.35e-3, +3.6e-4, -4.3e-5, +1.9e-5). It still ALTERNATES:
>   that is the stability function of the conduction mode (τ = 2·C/g_on = 2 µs, z = -5: SDIRK(2,2)
>   R = -0.18 for z < -2.414, trapezoid -0.43), not the event location; the "no alternation" criterion
>   cannot hold for this circuit at dt = 10 µs. Shortcut: two INDEPENDENT diodes crossing one way in one
>   step are located on their summed signal (reviewer: gate on series connection).
> - **F-3-7 OPEN.** The consistent start moves the abort from t = 21.5 µs to 31.55 µs (Newton, ||dx|| 60 V at
>   rect_neg, residual 6e-6). Not traced further (budget). Needs the per-iteration diode-bit trace in the
>   stage's Newton (saturable primary + nonlinear refresh), restart-less 6.911 V, linear primary 6.890 V.
> - **Impulse test** (dt-dependent by design: an impulse "on the grid's scale"): jump energy > ¼ of the stage
>   increment's AND some state moved > 1 % of its kind's largest |state| + 1e-6. A first energy-only test
>   flagged zero-start leakage (2.9e-20 J, broke the factor count) and a bridge capacitor's residual
>   relaxation at window steps (2.5e-7 J records, error 3.6e-3 V for 3 samples); both gone.
> - **Byte compares vs PR head:** DEFAULT options identical everywhere (goldens, stateful, engines, steps,
>   pwl_identity default column, trbdf2_identity 371/371). Moved, all with the rule ON: goldens
>   `buck_pwl_restart` (vs exact, p99 |dv_out| 1.96e-5 -> 1.04e-5 V, p99 |di_L| 7.2e-6 -> 3.5e-6 A; 12
>   records of 2e-11..9e-10 J at DCM diode turn-offs, the late-located zero crossing's residual cut) and
>   `vsi_pwl_restart` (vs exact unchanged at p99 2.4e-4 V / 1.1e-2 A; 1 record 5e-15 J at 14.3 µs);
>   stateful `everything_pwl_restart`; engines `buck_pwl_restart`; steps `buck/vsi_pwl_restart`,
>   `vsi_pwl_split`; pwl_identity 64 of the restart/split entries (zero-start rectifier charges, DCM
>   turn-offs, the bridge pair; three lose their `_implausible_voltage` warning: half_bridge_transformer_dc,
>   ll13_active_clamp_forward, synchronous_flyback — the spike was the pseudo-state's).
> - **reference_status** strict differential: pwl+restart 89/19/1, pwl+split 91/17/1 (agree/disagree/error) —
>   equal to the PR head's counts given in the brief. Against the previous session's run (87/21/1, 89/19/1)
>   flyback_converter (0.110 -> 3.3e-5) and forward_converter (0.092 -> 2.9e-3) moved disagree -> agree.
> - Reviewer (opus, fresh context) fixed: impulse records now booked by the loss tracker on every circuit
>   (`lt->impulses` outside the structural branch); rs_bits0 comment; test re-pinned. LEFT: Newton-device
>   circuits — the discarded probe moves the Newton round-off acceptance counters; device histories (hs) start
>   at x1, not x0+ (O(h) per impulsive event with stateful Newton devices: first order there); `devices`/`cause`
>   are the switching branches at the cut state's terminals (a non-adjacent cut gives cause ''); saturable
>   inductors are not in the history entries (their jumps are neither detected nor recorded); a failed probe
>   falls back silently; `projection_gap` not set; `event_rl_x`/`event_rl_t` written for group B, not read yet.

> **For group C (mode selection), second session, 2026-11-19 [M]:** the restart rule now starts every
> SDIRK step from a CONSISTENT state (`consistent_start` in `run_transient.hpp`, beside `run_restart`):
> after stage 1, x0+ = 2·x1 − S(x1) (S = the same stage re-solved from x1 on the same factor, bits
> frozen); when the jump x0+ − x0 holds > 1/4 of the stage increment's energy (and > 1e-6 A/V) the
> event is impulsive: recorded (`ReinitRecord`, any device kind), histories moved to x0+, stage 1
> re-solved from x0+ with its full event iteration (the mode is decided there, from a consistent
> start). F-5-1 and F-3-3 pass with it. If F-2-3 / F-5-9's forbidden OFF diode sits in a restart
> stage-2 right after an impulsive stage 1 (x1 − x0 = O(1)), it is this defect and this branch
> changes it; a site outside `run_restart` (plain trapezoidal step, settle/LCP choice on a consistent
> state) is not. The right limit x0+ is exposed for group B as `event_rl_x` at `event_rl_t`
> (NaN when the step had no impulsive event), locals of the run beside `x_prev`.

Branch `fix/core-release-a-impulsive` on the PR head 27c00520. Reproductions:
`python/tests/test_release_findings_a.py` (F-5-1, F-3-3 so far; F-5-10, F-3-7, F-3-1 to add).
Status marks: **[R]** = read in the code, **[M]** = measured on 27c00520, **[H]** = hypothesis, not yet
shown by a trace.

## Reproduced on 27c00520 (explicit options) [M]

| finding | call | value | truth |
|---|---|---|---|
| F-5-1 | IGBT level1, no FWD, L = 1 mH i0 = 100 A, `engine='pwl', dt=1e-9` | restart: i_L 100 -> -141.42135 A in one step, v_sw 8.571 V, no warning, 0 records. restart-less: 100 -> 0 -> -99.99999 A, v_sw -2e8 V at sample 1, the "conduction path opens" warning | no path: i_L -> 0 by an impulse (5 J), named |
| F-3-3 | linear flyback R_L = 1, `dt=1e-8` | restart: min i_p = peak = -20.9199 A, v_out 17.146 V, 0 records. restart-less: -15.483 A, v_out 16.801 V (+ voltage-sanity warning, 1.9e4 V) | `engine='dsed', mode_selection='auto'`: peak 16.38199 A, min i_p +1.6e-7 A |
| F-5-10 | Vin 48 V, L 1 mH, S (g_on 1e3, g_off 1e-9), 10 kHz PWM, `engine='dsed'` | i_L 2.39994 -> 4.8e-8 A at each turn-off, v_sw max 48 V, 0 records, no warning (same with form='branch', output_rtol=1e-5, mode_selection='auto') | the state is right (48 V · g_off); the ½·L·2.4² = 2.88 mJ left in the g_off transient is neither recorded nor booked |

Side observation (NOT group A, group C): the F-3-3 flyback on `engine='dsed'` with the PR line's
default `mode_selection` (legacy settle) gives v_out = 6e-13 V, peak i_p 2.06 A — the secondary diode
never conducts. With `mode_selection='auto'` it is right.

## Where things happen [R]

Fixed engine, `core/include/pulsim/solver/run_transient.hpp`:
- Restart trigger and window: `step_begin`/take phase, ~l.2267-2285: `restart_left = kRestartSteps (3)`
  on a mask change at t_prev, an unknown start mask (`!stamped_known`), a source / b_extra jump; a diode
  flip inside a step re-arms it (`kRestartSteps - 1`). `kGamma = 1 - 1/sqrt(2)`.
- `run_restart(t_from, h)` (lambda, ~l.2314): ONE Alexander SDIRK(2,2) step. Stage 1 = backward-Euler
  solve of γh (`run_event_iteration(t_from+γh, 2γh, true)`, the trapezoidal factor at 2γh), diode event
  iteration inside it; then `history.extrapolate_previous(x, c)` with c = (1-γ)/γ builds the stage-2
  pseudo-state x0* = x0 + c(x1 - x0); stage 2 = a BE-form solve of γh from x0*, with its OWN diode
  event iteration (mode decided on the stage-2 solve). Stage-2 fallback ("on_to_off"): only when a diode
  conducting in stage 1 turns off in stage 2 and no diode turns on fresh; then stage 2 is re-taken as
  BE of (1-γ)h from x1, kept iff BE keeps stage 1's bits. A diode OFF at x0 and in stage 1 that turns ON
  in stage 2 is explicitly EXCLUDED from that check (`!rs_bits[q] && rs_bits2[q] && !rs_bits0[q]` ->
  `on_to_off = false`): the SDIRK stage 2 is kept unconditionally.
- Split (`enable_substep_state_correction`): `split_step` (~l.2464): a diode crossing in the first
  (trapezoidal) solve is located by LINEAR interpolation of the crossing signal (`interp_commutation_time`),
  three branches by `snap`: at a (restart from a), at b (keep the pre-event solve, flip at b, arm window),
  inside (trap to t_d, `run_restart(t_d, b - t_d)`).
- Re-initialisation service on the fixed engine: `pwl::ReinitTracker` (`core/include/pulsim/pwl/reinit.hpp`).
  It does NOT project: the SDIRK x3 restart IS the projection; `ReinitTracker::record` runs AFTER the
  committed step (run_transient ~l.3145, in the record phase) and writes a `ReinitRecord` computed from
  x_prev / x for each STRUCTURAL dependent structure (`BranchStructureChecker`, ideal / branch-form
  elements only) the step entered by a gated change. A conductance-form switch (g_off > 0), a
  MOSFET's R_off, or a smooth Newton device that blocks (IGBT level 1) is never a dependent structure,
  so such an interruption gets no record whatever happens. `refuse_dependent` refuses restart-less runs
  that enter a structural dependent mode (`dependent_states`).
- `on_impulse`: `SimulationOptions::on_impulse_error` (solver/options.hpp); pwl/trbdf2 raise in
  `ReinitTracker::record`; on the exact engine Python raises/warns from `res.reinit_records`
  (python/pulsim/__init__.py ~l.4419). 'record' (default) only appends records. The voltage-sanity
  guard (`_auto_check_voltage_sanity`, solver/voltage_sanity.hpp) reads STORED samples only.

Exact engine, `core/include/pulsim/dsed/scheduler.hpp` + `builder_adapter.hpp`:
- `fire_event_` -> `decide_modes_` (settle or LCP) -> `project_(t, x)` (scheduler ~l.1521):
  1. `reinit_to_constraint` (builder_adapter): energy-weighted projection onto K·x = k for STRUCTURAL
     dependent modes, writes `ReinitRecord`s (lambda, dW = ½ rᵀλ).
  2. `project_slow(t, x, thr)` (builder_adapter -> `exact_project_slow`, exact_lti.hpp): every mode with
     Re λ < -thr goes to its quasi-static value. NO record, NO energy. This is where F-5-10's 2.88 mJ goes
     (the g_off mode, τ = L·g_off = 1 ps). thr = 100/dt_max, or 1/(kParasiticFraction·scale) under
     output_rtol.
- The mode is decided BEFORE the projection and re-settled on the projected state
  (`settle_and_project_`, `resettle_projected_`); trial-mode records are truncated (`truncate_reinit_log_`)
  and the state restored when a trial mode's projection left records — so adding records INSIDE
  `project_` would change control flow (and numbers); a record must be added after the decision.
- `PEDSimulatorAuto` (scheduler_auto.hpp) only calls `reinit_to_constraint`, no fast projection.

## Traces (env-guarded fprintf in `run_restart`, x0 / stage 1 / stage 2 / end; not committed) [M]

- F-5-1, step t = 0, h = 1 ns, x = (v_g, v_sw, ., i_L):
  `x0: 15 0 0 100` -> `s1: v_sw = -3.41421e8, i_L = 0` -> `s2: v_sw = 8.57106, i_L = -141.421`.
  -141.421 = 100 + 2.41421·(0 - 100) = x0* exactly: stage 2 found the IGBT's forward channel at
  the pseudo-state and stayed there. PROVEN. No diode bits (the IGBT level 1 is a smooth Newton device).
- F-3-3, first turn-off t = 5 µs, dt = 10 ns, bits (Q1_body, D1), x = (v_in, v_sw, ., ., ., i_p, -i_p, i_s):
  `x0: bits 00, v_sw 0.0206, i_p -2.061` -> `s1: bits 01, v_sw 3567 V, i_p -7.1e-6, i_s 3.964` ->
  `s2: bits 11 (body diode ON), v_sw -0.703, i_p +2.887` (≈ -√2·i_p(x0) = +2.915). The body diode is
  OFF at x0 and in stage 1 and turns on in stage 2 only — the case the stage-2 check excluded.
  With the exclusion removed: `RS retry keep_be=1`, end `bits 01, v_sw 48.01, i_p -9.6e-8`. PROVEN.
- F-3-7 (saturable PSFB, dt = 50 ns): the failing restart step at t = 21.5 µs prints `x0` for every
  halving (50 ns ... 0.78 ns) and never `s1`: Newton fails in STAGE 1 (||dx|| 1.1e4 V at vout).
  Not the stage-2 pseudo-state; the stage-2 fix leaves it unchanged. PROVEN negative.
- F-3-1 (bridge1_vf07, k = 0 vs k = 2 ulp): both take the split's "inside" branch;
  k = 0: `t_d = 33.70395 ms, x0 bits 0010` (one diode), stage 1 turns the second on (iters 2),
  end i = 14.2994 A; k = 2: `t_d = 33.7040466 ms, x0 bits 0110` (both within snap), end 14.1918 A.
  The two series diodes of the floating bridge can only conduct together; the split locates the
  event at the FIRST single diode's interpolated crossing, a tie decided by leakage/rounding. Not an
  impulse; the stage-2 fix leaves it unchanged. PROVEN.
- F-5-10 (exact engine): the event projection is OFF by default (`enable_event_projection_ = false`),
  so the 1 ps g_off mode is integrated exactly; `right_limit_states` at each turn-off holds
  v_sw = 2.39994e9 V = i_L/g_off and i_L = 2.39994 A (continuous). The energy is dissipated in g_off
  in that picosecond: nothing is dropped. The defect is the voltage-sanity guard
  (`_auto_check_voltage_sanity`, python/pulsim/__init__.py) reading only the stored samples, not the
  right limits. (Also seen, group B: `device_loss_summary` integrates that right-limit spike as a
  1 µs trapezoid: S's E_total = 5760 J, P_avg 1.9e7 W.) The reading "drops 2.4 A" in r2/bucket-5.md
  is a misreading of the sampled trace. PROVEN.

## Hypotheses per finding (written before the traces; see above for what held)

- **F-5-1 / F-3-3 [H, numbers consistent]**: the stage-2 pseudo-state. Stage 1 (BE of γh) relaxes the
  stiff interruption (i ≈ 0); x0* = x0 + c(x1 - x0) ≈ (1 - c)·x0 = -√2·x0 (1 - 2.41421 = -1.41421).
  Stage 2's mode / device region is decided on a solve from that pseudo-state: the reversed current
  finds a path the physical state never had — the IGBT's forward channel (F-5-1: -141.42135 A =
  -√2·100 + γh·v/L), the MOSFET body diode (F-3-3: -20.92 ≈ -√2·14.8). In a linear fixed mode
  stage 2 would cancel it (L-stable, x2 ≈ 0); across a region change it does not. So the mode at the
  event is not decided on an inconsistent PHYSICAL state but on the SDIRK's non-physical stage-2
  history. F-3-3 is a diode bit (Q1_body) turning on fresh in stage 2 — the case the stage-2 fallback
  explicitly excludes. F-5-1 has no bit at all (smooth IGBT). Trace pending (env-guarded fprintf in
  `run_restart`, PULSIM_TRACE_RESTART, not to be committed).
- **F-3-7 [H]**: the same pseudo-state overshoot feeding the Newton loop's per-iteration diode re-decision
  (`make_combined_diode_mosfet_refresh`) with a saturable primary: no self-consistent mask for the
  stage-2 solve -> chatter. To trace.
- **F-3-1 [H]**: the post-event error decays by about -0.11 per sample = R(z) of the TRAPEZOID at
  z ≈ -2.5 (SDIRK(2,2) gives -0.012 there), so samples after the turn-on are trapezoidal, i.e. the
  restart window does not cover the steps after this event, and the 1-ulp sensitivity is a branch flip
  between the split's `snap` cases (event inside / at b / at a) — a discontinuous decision, not the
  projection. To trace.
- **F-5-10 [R]**: `project_slow` drops the fast mode with its energy, by design, without a record. Fix in
  progress (uncommitted): after `decide_modes_` record ½·ΔxᵀEΔx of the event's state jump (E = C, L with
  mutuals; for a charge/flux-conserving jump this equals the projection formula's ½rᵀλ, sources' work
  included, >= 0) minus the dependent-state records already written, as a `ReinitRecord` kind
  `fast_mode` (booked by `losses.impulse_losses`, raised by on_impulse='error').

## Default path (restart-less fixed step) [M]
F-3-3 restart-less is also wrong (-15.48 A against ~0): the trapezoid reflects the stiff interruption
(R(∞) = -1) and the body diode then carries the reflection. It warns (voltage sanity) on this circuit.
Not covered by a restart-rule fix; recorded as a default-path finding of group A.

## Next steps
1. Trace F-5-1 / F-3-3 with the env-guarded print (built in .scratch/build-ext) to prove the stage-2 mechanism.
2. Fix at `run_restart`: decide stage 2's mode change (any bit, incl. a fresh turn-on) by the BE re-take
   from x1; for smooth devices find an equivalent test or report the stage-1 impulse to the guard.
3. Build F-5-10 record, check records on the dsed byte-compare set, then F-3-7 / F-3-1 traces.

## Fixes landed (130fcb70) [M]

- **F-3-3** — `run_restart`: any stage-2 diode-mode change goes through the existing backward-Euler
  re-take from x1 (kept iff it keeps stage 1's bits; else the SDIRK stage 2 is restored bit for bit).
  The removed exclusion (fresh turn-on in stage 2) was the F-3-3 case. Side effect (reviewer): the old
  loop's `break` also cleared a real on->off flip seen earlier in the same loop; now it counts.
  Truth: exact engine (`mode_selection='auto'`) peak i_p 16.381990162 A, min +1.6e-7 A.
  Restart pwl peak: 16.3819859 / 16.3819892 / 16.3819901 at dt 10 / 5 / 2.5 ns (errors 4.2e-6,
  9.5e-7, 1.0e-7; order 2.2 then 3.2); min i_p +1.57e-7 A (was -20.92 A); v_out 16.3651 (was 17.146).
- **F-5-10** — `_auto_check_voltage_sanity` also scans the right-limit samples (`right_limit_index` /
  `right_limit_states`, only the exact engine has them). Truth: v_sw(0+) = i_L(0-)/g_off (closed
  form; matches to 1e-9). The warning now fires on `engine='dsed'`.
- Parked with traced causes (strict xfail in `test_release_findings_a.py`): F-5-1 (smooth Newton
  device: no bit for the stage-2 check — needs a region test for Newton devices or the BE re-take on
  a stage-1 impulse, plus the stage-1 voltage for the guard), F-3-7 (stage-1 Newton diode chatter at
  a gate edge — needs the in-Newton diode re-decision fixed for the BE stage), F-3-1 (the split
  locates a series diode pair's commutation at one diode's crossing — needs the split to locate the
  decided post-event mode's commutation).

## Byte compares vs the PR head (same machine) [M]
DEFAULT options: identical everywhere — goldens (all non-restart cases), stateful (all but the two
`*_restart` cases), engines, steps (all but restart/split), pwl_identity default config,
trbdf2_identity 371/371. Moved, all with the restart rule ON (the option, not the default):
golden `buck_pwl_restart` (stage2_retries 21 -> 29, stage2_be_kept 0 -> 6), stateful
`compose_all_restart`, `everything_pwl_restart`, engines `buck_pwl_restart`, steps
`buck_pwl_restart`, `vsi_pwl_split`, pwl_identity 18 of 379 restart/split entries (mostly the
stage2_retries counter; boost_pfc, active_clamp_forward, interleaved_buck_3ph commutation counts).
Checked against truth (the reviewer's risk: turn-ons at a zero start / buck CCM from rest): the
golden buck (24 V, 100 kHz, from rest, 2 ms) on pwl+restart vs the exact engine at output_rtol 1e-9,
p99 |dv_out| / p99 |di_L| over all samples:
dt 100 ns: 1.96e-5 / 7.2e-6 -> 9.9e-6 / 3.5e-6; 50 ns: 3.8e-6 / 9.0e-7 -> 2.4e-6 / 1.2e-6;
25 ns: 1.42e-6 / 5.6e-7 -> 1.34e-6 / 3.9e-7 (before -> after). Equal or better except di at 50 ns
(1.35x). Cost: the change runs only inside `run_restart` (option ON); it removes one bit snapshot
per restart and adds a BE re-take on the newly included case (counted in stage2_retries). The
restart-less loop is untouched, so the four-case instruction count was not re-measured.

## Default-path finding of group A (not fixed)
F-3-3 restart-less: -15.48 A against ~0 (trapezoid R(∞) = -1 reflects the cut current, the body
diode carries the reflection; v_out 16.80 vs 16.365). It warns (voltage sanity). Not covered by the
restart-rule fix; the release's default (restart ON) now gets it right.

## REVERTED: the F-3-3 stage-2 change (after the full pytest) [M]
Full pytest on 130fcb70: 6089 passed / 4 failed. Three failures are the stage-2 change itself:
`test_buck_ccm_stays_second_order_to_t_over_1600` (stage2_be_kept 1, not 0: at the CCM buck's zero
start the BE re-take keeps stage 1's mode for a REAL stage-2 turn-on — the dt-independent 7e-10 A the
exclusion was written to prevent; the reviewer flagged exactly this risk), `test_one_factor_per_mask_
and_no_refactor` (6 restart factorizations, not 4), `test_a_failed_stage2_retry_...` (retries 38 vs
19). So the exclusion is load-bearing; `run_transient.hpp` is back to the PR head byte for byte and
F-3-3 is a strict xfail again. What the fix needs: a test that separates the pseudo-state's turn-on
(F-3-3: stage 1 itself commutated, iters_1 = 2, the cut current relaxed to ~0) from a real one (zero
start), e.g. include a fresh stage-2 turn-on only when stage 1 commutated — unmeasured; then the
three tests above plus the F-3-3 truth (peak 16.381990, min i_p ~ +1.6e-7) and the restart byte set.
The fourth failure was `test_simulate_overhead[pwl]` (2894 > 2890 bytecodes) from the guard edit
in the shared `_auto_check_voltage_sanity`; the right-limit scan moved to its own helper
`_check_right_limit_voltage`, called only on the exact engine's finalize path.

State of the branch now: C++ identical to the PR head (no extension change; the head's extension
`so_27c00520` is the one to use); Python: `_check_right_limit_voltage` (F-5-10). Test file: F-5-10
passes; F-5-1, F-3-3, F-3-7, F-3-1 strict xfails with traced causes. Re-run after the revert:
test_release_findings_a, test_fixed_step_discontinuity_order, test_restart_review_fixes,
test_simulate_overhead, test_engine_auto: 68 passed, 4 xfailed; ruff clean; stubs up to date.
The byte compares of the earlier section no longer apply: with the C++ reverted every case is the
PR head's own extension (identical by construction); the Python change only adds a warning on
`engine='dsed'` results whose right limits breach the voltage bound.

## Fourth session (2026-11-20): the jump test without the old rate (A1), A3 [M]

Branch `fix/core-release-a4-jump-test` from 1cec4e8b. Code tip 0bd28e03 (criterion below is IN the code).

**The orchestrator's J = 2·D(h/2) − D(h) is the probe's existing d.** With D1 = x1 − x0 (one BE stage of
γh) and D2 = x2 − x0 (two stages), d = 2·x1 − x2 − x0 = 2·D1 − D2 is the same Richardson estimate
(steps γh / 2γh instead of h/2 / h): per mode d = −J·(r/(1+r))², O(r²) slow, → −J fast. No new solve
gives J; the question was only the BOUND it is compared with. Compared with the noise floor alone it
flags every curved trajectory: the DCM buck's C at a gate edge has d = 6.4e-7 V = (γh)²ẍ (ẍ ≈ 3e9
V/s²) against tol 2.9e-7 V. So "J vs noise" is refuted by the buck_dcm trace.

**Criterion in the code (0bd28e03).** A third BE stage x3 (same factor, discarded) gives
d' = Δ2 − Δ3 (Δk = xk − x(k−1)); per mode d' = a·d (a = 1/(1+r)); on a smooth trajectory d' ≈ d.
Impulsive iff, in both readings (frozen / moved),
  |d| > max(|Δ2|/γ, (1 + 1/γ)·|d'|) + 1e-8·(1 + |x0| + |x1|) [+ G_off·(1 + V_max) for currents].
For one mode both terms give the predecessor's threshold τ < γ²h; |Δ2|/γ covers an inflection,
(1+1/γ)|d'| an extremum. No r0. Same rule for saturable L / nonlinear C (their r0 from the law removed).
Domain: the probe runs only when the restart's start may be inconsistent — a trigger at t_prev, a split
event, the first step, a stage-1 flip (breach_1), or after a non-restart (trapezoid) solve
(`prev_be`/`step_fresh`/`rs_fresh`). Not at window steps 2–3 that start from a restart solve's end.
A3: a source jump in (t1, T] is read at its left limit (`add_source_rows(out, t_jump, T, left_by)`);
a second jump in the interval -> frozen reading only. NB `source_jump_in(a, b)` counts [a, b).

**Builds.** b1: trace prints only (1cec4e8b numbers). b2: the d' criterion + third stage — segfault
(unsized `ip_s2/ip_s3`, found at b3). b3: [t1,T) interval fix. b4: vectors sized — A1, A3, DCM pass;
`test_a_nonlinear_capacitor_shorted_by_its_switch_is_refused_by_name` FAILED: refused at 5.02 µs,
Cj d = −1.529e-3 V, d' = −4e-8 V (a real 1 ps-mode residual the 5.0 µs restart step left: SDIRK
|R(z)| at z = 2e4 times the 24 V transient). b5: refuse_jump trace (diagnosis above). b6: the
fresh-start domain — all of test_release_findings_a passes (12 passed, F-3-7 xfail). b7: same code,
trace stripped (building when this was written).

**Measured (b4–b6 builds, the committed rule; trace d / d' / |Δ2|/γ):**
| case | d | d' | result |
|---|---|---|---|
| A1 (1 µH, 100 V, dt 0.1 µs, 8 A cut) | −8.000 A | −5.5e-7 A | 1 record 3.19976e-5 J; truth ½·L·7.9997² = 3.19976e-5 J (7.9997 A: closed form with R_on 1 mΩ); i_L -> 2.0e-7 A (was −8.38 A) |
| DCM buck C at gate edge t = 223.1 µs | 6.4e-7 V | 6.3e-7 V | no record (extremum: Δ2 = −1e-8 V) |
| DCM buck L at diode turn-offs (5 events) | −4e-4 .. −1.5e-3 A | ~−1e-8 A | recorded, 7.8e-12 .. 1.2e-10 J (real discrete cut; masked before by h·r0 = 7e-3 A). buck_dcm analytic err_rms pwl+restart 5.3158e-8 -> 5.3074e-8 (better) |
| snubber Rs 1 Ω + Cs 1 nF, τ = 1 ns, dt 100 ns, turn-on 1 µs | −93.8 V | −3.1 V | recorded (as before), 4.40 µJ of ½·C·100.36² = 5.04 µJ: the fraction (r/(1+r))² = 0.876 the grid cannot resolve; the rest is integrated |
| snubber Rs 100 Ω (τ = dt) | — | — | no record (r = 0.29 < 1/γ) |
| F-5-1 | −100 A | −4.7e-6 A | 1 record 4.999998812 J (unchanged) |
| A3 rectifier + 1 V pulse edge at 0.4 dt | −1.44e-2 V (moved) | 1.0e-3 V | no record; equals the pulse-free run to 5.7e-14 V (was 0.96 mJ record) |
Without the domain restriction (b4) the snubber also recorded the SDIRK residuals at the window steps
(1.1 µs: 0.27 V, 3.6e-11 J), and the waveform stopped ringing (−0.278, 0.0108, 0.0100 vs truth 0.0100);
with it (b6) those records and that improvement are gone (old waveform: 0.0227, 0.0094, 0.0105 ...).

Analytic set (27 cases × pwl+restart / pwl+split) on b4: identical to the predecessor's refstat_t7 except
buck_dcm restart (above). NOT re-run on b6/b7.

**Verified at 0bd28e03 (b6 build = committed code + inert trace):** test_release_findings_a 12 passed 1 xfailed.
**NOT verified yet:** the 48 restart/split test files; full pytest; byte compares (default path —
expected identical: the change is inside the rule's probe plus two bool stores per step); cost
(default path gains `prev_be = committed_be` and one `step_fresh` assignment per step — must be measured,
buck_pwl was at the 0.5 % gate); reference_status differential; group numbers F-3-3 / F-3-1 on this tip
(script `.scratch/group_nums.py`); dt-retry test, mos_l1, `everything` stateful case; reviewer; ruff on
the header-free files is clean.

**Next steps for a successor.** 1) build ext (`.scratch/build.sh ext`, ccache launcher now set);
2) `python .scratch/group_nums.py`; the 48 files `grep -l 'discontinuity_restart\|enable_substep_state_correction' python/tests/*.py`;
3) byte compares `.scratch/baseA.sh t8 compare` (baselines = PR head; the predecessor's 32 pwl_identity
restart/split entries already differ — classify any NEW differences); 4) `.scratch/cost.sh` (needs
`so_head_same.so` = 1cec4e8b build, not available: build 1cec4e8b ext once or compare to the PR head's
`so_<hash>`); 5) reference_status `--json` vs `.scratch/refstat_t7.json` cell by cell (save and restore
benchmarks/REFERENCE_STATUS.md, the tool rewrites it); 6) reviewer; 7) full pytest.
Open by design: a jump on a smooth Newton device region change at a window step without a diode flip
is no longer probed (no bits to see it); the window-step residual of a restart solve is integrated,
not recorded.
