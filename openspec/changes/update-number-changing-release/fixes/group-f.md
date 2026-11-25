# Group F — the exact engine's output rule and the `auto` route

Branch `fix/core-release-f-exact-output` = PR head 27c00520 + merge of
`fix/core-exact-output-rule-cost` (1e2b90ff; conflicts resolved as in wip 0b37922c).
Every number below is from an extension built from this tree (`.scratch/build-ext`), the
"before" column from the merge commit's build, the PR head's own build where named.

## F-6-7 — the trajectory depended on the output grid

**Trace.** Ten paralleled bridges, first mode, 0 to 1e-4 s: there is NO event in the window
(`event_times` empty), so the only thing that differs between grids is the number of exact
steps. i(L0a)(1e-4) − mpmath truth 0.12972284951463489 A, merge build:

| `output_rtol` | samples = steps | error |
|---|---:|---:|
| None | 101 | 7.7e-12 |
| 1e-4 | 255 | 5.3e-11 |
| 1e-5 | 792 | 1.6e-10 |
| 1e-8 | 65 160 | 9.5e-9 |

The error grows linearly with the step count (≈ 1.5e-13 A per step): each output sample was
a STEP, `x_{k+1} = Φ(δ_k)·x_k` (`scheduler.hpp` exact branch: `h_ex = output_step_(...)`
then `advance_(t, x, h_ex, x_new_)`, `x.swap(x_new_)`), so every sample added one rounding
of the mode's eigenbasis to the carried state, and the event locator worked from that state.

**Fix.** The exact branch takes the rule-free step (dt_max / time events / fast samples, as
with the rule off); the rule's samples inside it are pushed by `push_output_samples_` as
`x(τ) = Φ(τ − t0)·x0` from the step's start, read only. Steps, event location and the state
carried are the rule-free run's whatever is sampled.

**After.** Same table: every row 7.7e-12 and the final state bit-identical to `None`
(samples 101 / 333 / 926 / 29 014; 1e-8 wall 61 s → 10.7 s). Ten bridges to 3 ms (events):
event times and the states at them bit-identical for 1e-4 / 1e-5 / 1e-6 (test).

**Counters (F-3-6 / F-6-6).** `n_exact_steps` counts steps again, never samples; the event
integrator's DOPRI5 steps are `n_rk45_steps = n_accept − n_exact_steps` (they were read
nowhere on this integrator: 0 with or without the rule); the rule's samples are
`diagnostics['output_grid']['n_output_samples']`.

## F-5-8 — recovery energy on the exact route

**Trace.** Boost (`R_load` 20 Ω, CCM), exact engine, `output_rtol=1e-5`, `Q_rr` 100 nC:
E_sw 6.775e-4 J against pwl dt = 10 ns 6.9495e-4 J, 60 of 299 turn-offs "unresolved".
Per turn-off against pwl the 60 are the natural (current-zero) turn-offs of the start-up's
DCM stretch, e.g. t_off 7.4801e-4 s: pwl V_R 19.89 V, exact 11.45 V. The exact samples there:
t_off (left limit, 0 V; right limit −19.872 V), next sample 1 µs later −19.838 V, next the
gate edge 7.5e-4 (left −19.804 V, right −31.80 V). `_reverse_voltage_at` paired the LEFT
values into midpoints, so its second midpoint straddled the gate edge's jump, and the
back-extrapolation to t_off + t_rr along that line read 11.58 V. The step length was not the
cause: the samples are exact and the rule bounds their chords; the reader was the fixed
engine's.

**Fix** (`losses._diode_turn_offs`, `interpolable=` set when the result carries
`diagnostics['output_grid']`): v_D at t_off + t_rr on the step line from the step's start
(the right limit, `trace.v_start`) to its end; such a turn-off is never "unresolved".

**After.** CCM 6.9483e-4 J (pwl 10 ns 6.9495e-4, −1.8e-4 rel), DCM 9.9337e-4 (9.9390e-4,
−5.3e-4), no warning.

## F-X-1 — ten branch-form bridges refuse under the rule (OPEN, xfail)

**Reproduction.** Reproduces only with true branch-form diodes (`form='branch'`,
`R_on = 1/g_on`); `next_defaults.builder_form()` does not produce them on this line, the test
builds them explicitly. Rule off: runs (20 909 samples, 111 events). `output_rtol` 1e-5 and
1e30: `DiodeModeInconsistentError`, Db8b OFF 0.73 V forward at 19.969 ms. The release tip
(967c8bfe) refuses the same way with `mode_selection` legacy and lcp.

**Trace.** Toggling the two rule-gated event semantics one at a time: `zero_is_root` back to
the rule-off value still refuses; the projection threshold back to `100/dt_max` runs. With the
rule the projection keeps every mode except the parasitic cluster (1/(1e-6·1 µs) = 1e12 /s),
so the ten bridges' common mode L/R_g = 0.2e-3/1e6 = 2e-10 s survives. After Dt0c turns OFF at
0.019968955755 s, Db0b is located OFF 18 times in 43 ns (intervals shrinking ×0.85, a Zeno
approach to a sliding mode), then Db1b..Db7b likewise, and the settle ends one diode OFF and
forward-biased.

**Not fixed.** Restoring the old projection makes the cell pass but projects ll06's real
100 ns transient (1.1e-2 off truth, `test_exact_output_grid`), i.e. trades one wrong answer
for another. The cause-level fix is a sliding-mode rule in the event decisions (zero band /
settle), not in this group's scope. Test: `test_fx1_ten_branch_form_bridges_run_under_the_rule`
(strict xfail, cause in the reason).

## F-X-3 — ladder RC under the rule (session 1: PARTIAL; session 2: FIXED, see below)

**Trace.** n-section RC ladder (10 Ω, 1 nF), 2e-5 s, `output_rtol=1e-5`. The far nodes sit at
~0 V, so their tolerance is the floor rtol·atol = 1e-14, while `exact_chord_step`'s bound is a
triangle-inequality sum over modes (1e-3..1e-7 V for those nodes) of terms that cancel to
~1e-14 there. Every chord was cut to the floor: n=25 638 samples, n=50 1 357, n=100 197 755
(14.4 s); n=500 did not finish in 29 min (and every sample also advanced the state, F-6-7).

**Fix.** A component held only to the floor tolerance (it has not moved more than atol)
whose bound exceeds that tolerance even over the modes' own resolution step
(min(h, 1e-3/max|λ|)) cannot be certified by this bound at any useful step: it is not held to
it, the call is counted (`diagnostics['output_grid']['n_output_uncertified']`, no warning: a
start from rest leaves one such chord on an ordinary resonant tank). A tolerance above the
floor is always held: the first version (any component) let `test_exact_lti`'s resonant case
(constant tol 5e-5 against a 2e4 V/s state) through 2188× over its tolerance, and would have
dropped every moving state at `output_rtol` ≲ 1e-7. The state carried is the rule-free run's
(F-6-7), so this changes only where samples fall.

**After.** n=100: 6 215 samples, 0.55 s (test; one capped interval). n=500 (ladder500_rc):
70 873 samples, 162 s against 7.6 s rule off (≈21×, shared machine under load; 11 capped
intervals, so the run warns) — the 3× target is NOT met. Reviewer fix: the range is widened
over the step before any state is left uncertified (a state from rest that really moves is
then held; the first version let the resonant tank's first chord out unchecked and had
2 140 samples at n=100). The remaining cost is the O(n²)
per-sample bound on the near nodes. Experiment (not committed): treating `atol` as an absolute
floor, tol_j = max(rtol·|x_j|, atol), gives 7 160 samples, 14.8 s (≈2×). That changes what the
rule guarantees on every rule-on grid (it loosens the floor 1e-14 → 1e-9 V), so it is an owner
decision, not a fix made here.

## Group C cross-check (F-X-1)

Group C's tip `fix/core-release-c-mode-selection` d014f049 (its own built extension, read
only, rule at 1e30 so the projection is gated with almost no samples): the same refusal, Db8b
OFF 0.7296 V forward at 1.99690 e-2 s. Their tie-release / pin-judge fix is not this site.

## Reviewer (fresh context, read only) — fixed

1. A state from rest at the floor was left uncertified before the range was widened over the
   step: widen when any state is uncertified too (tank back to its pre-exclusion grid,
   209 / 325 samples; test `test_f67_chords_from_rest_are_within_the_rule`).
2. `n_rk45_steps = n_accept − n_exact_steps` was set for BDF2 results too (a default-path
   counter): now only on the two PEDSimulator bindings (test `test_f36_bdf2_reports_no_rk45_steps`).
3. `losses` read every rule-on result as interpolable: now only when no interval was capped,
   no mode unbounded and the budget not spent (`_rule_held`).
4./5. Comment and counter wording (a floor state that really moves is left uncertified too;
   the counter counts chords decided, not samples).

Left: no independent truth for the excluded far-node chords (F-X-3); the F-5-8 pwl reference
is one dt (10 ns), its convergence order not measured.

## Session 2 — F-X-3 fixed: the absolute floor, and a second-order chord bound

**Decision (orchestrator).** tol_j = max(output_rtol·range_j, atol), range_j = the state's range
so far, at least 1e-3 of its peak; atol = the run's ONE absolute tolerance (`simulate(atol=)`,
default 1e-9; the engine has no per-kind atol: event location uses a time tolerance only, so
the scalar is used for volts and amperes alike). The "uncertified" escape is REMOVED
(`ChordWork::floor / n_uncertified / tol_eff`, `n_output_uncertified` in the scheduler, the
binding, the dispatch and the docstring): every state is held to its tolerance; what the bound
cannot certify ends at the sample cap / budget, which warn.

**Trace after the floor alone.** ladder100: 5 491 samples, ONE capped interval (warns);
ladder500: 7 139 samples, one capped interval, 14.8 s vs 7.3 s rule off. Histogram per µs:
4 097 samples in the FIRST rule-free step (1 204 in the first 10 ns), then 653, 251 ... The
far nodes are no longer the cost; the first node near the start from rest is. Closed form
(numpy eigendecomposition of the ladder, ∫ sqrt(max_j |ẍ_j|/(8 tol_j)) dt over the first µs):
~600 samples needed, n = 100 and n = 500 alike. A numpy replica of `exact_chord_step` on the
same states: 5 349 samples (the C++ rule hit the 4 096 cap there) — the bound's first-order term
|ẍ_S(0)| + δ Σ m_k|λ_k| is the triangle inequality over every mode.

**Fix 2 (`exact_chord_step`).** A second-order form joins the min: |Σ V z̈| + δ|Σ V z̈ λ| +
δ²/2 Σ m|λ|² (the third derivative at τ = 0 exact, the remainder by the triangle inequality;
growing modes with their g_k like the other sums). Rigorous by Taylor with the Lagrange
remainder per component. Replica: 1 170 samples; C++: 1 170.

**After.** ladder100 2 083 samples, 0.18 s, no cap, no warning; ladder500 3 072 samples,
12.25 s vs 7.28 s rule off (1.7×), no cap, no warning. Test
`test_fx3_ladder_far_nodes_are_held_to_the_absolute_floor`.

### Rehearsal re-run (session 2, after the floor)

`next_defaults_rehearsal.py run --configs dsed:output_grid,dsed+next,auto+next --jobs 1`, cache
seeded from p1-linalg's cells (1e2b90ff, the table's tree) with these three cells dropped.
dsed:output_grid and dsed+next: 122 agree / 14 refused, every cell bit-equal to the base (they
run with a dt: the rule is not engaged). auto+next: 132 agree / 2 disagree / 2 error (base 133 /
2 / 1). Against the base, two cells moved:

* `switching/isolated/active_clamp_forward.yaml` agree (exact 2.5e-8) -> `DiodeModeInconsistentError`
  (D_fw ON at -1.02e-2 A, 16.6595 µs). NOT the floor: it refuses at output_rtol 1e-4 / 1e-5 / 1e-6
  alike (6 / 17 / 53 samples break the law) and runs at 1e30 (no sample there). Trace (law check
  off): D_fw stays ON from 16.64 to 16.77 µs carrying down to -1.4e-2 A with NO event; the mpmath
  truth (`benchmarks/truth/active_clamp_forward.json`) has D_fw OFF at 1.66354842e-5 and ON at
  1.67126165e-5 (77 ns). The rule-free step's scan (`locate_event_exact_`, D3: an eighth of the
  fastest under-damped period) misses this double crossing; before F-6-7 every rule sample was a
  step and acted as a scan point, so the rule located it. The rule-off run (and the 1e30 run) is
  silently wrong there; the rule's exact samples now reveal it and the law check refuses —
  correctly. Fix needed (localisation, not the output rule): a rigorous no-crossing test per scan
  interval — the chord bound of `exact_chord_step` applied to the predicate's row (|g − chord| ≤
  Σ_k |c·V_k| |z̈_k| ...): an interval whose end values exceed the bound in the same sign cannot
  hold a root, otherwise it is split. Left open (new finding F-X-5; F-X-4 is the orchestrator's id for the DC sample count below).
* `switching/basic/snubber_rc.yaml` err_rms against the reference 2.16e-8 -> 6.92e-8 (3.2×, err_max
  1.97e-7 -> 1.96e-7, tol 1e-3; no truth for this circuit). NOT the floor: with `atol=1e-14` (the
  old floor rtol·atol = 1e-14, exactly) the cell is bit-equal (1 177 samples, 6.918e-8). From the
  F-6-7 grid (samples read inside rule-free steps), at the reference's own noise level.

## F-X-4 (orchestrator's id) — a DC run stored 2 000 002 samples under the rule

**Trace.** `electrothermal_resistor_self_heating` (5 V across 1 Ω, 2 s, no dt): the exact engine
takes 2 000 001 steps of 1 µs with or without the rule, `n_output_samples` 0. The 1 µs is the
poll of a gate that names no edge (`next_gate_edge_`: an infinite `next_edge_after`, the default
all-closed mask included, lands every dt_max/10 and fires a same-mask "gate_edge" that returns
at "Drop spurious same-mask gate edges"), not the rule. Every step end was stored.

**Fix (rule on only; the steps are untouched).** A step end the chord from the last STORED
sample still covers (certified by `exact_chord_step`, not by the cap / budget / an unbounded
mode) is not stored; when a later chord needs it, it is stored first (`output_pend_`,
`output_flush_`). A chain breaks at any predicate event, a source corner, a control tick, a
gate change, a fire that moved x, and before any numeric step. Stored samples 2 000 002 -> 2,
V(vin) = 5 V exactly; steps 2 000 001 both ways. The junction temperature the benchmark derives
from the samples (benchmarks/kpi `compute_junction_temperature`) integrated the thermal RC by
explicit Euler with the sample spacing as its step (unstable past 2τ = 1 s: on 2 samples a
nonsense value); it now solves each interval exactly for P linear between samples: on 2 and on
1 004 samples T_j(2 s) = 25 + 125(1 − e^-4) = 147.7105 °C to 1e-13 (test). Interpolation error
of V(vin) against the closed form (constant): 0.

Not changed (owner decision, default path): the poll itself — every exact run whose gate names
no edge steps at dt_max/10 = 1 µs, not the documented 10 µs; treating an infinite answer as
"no edge" would move every such default run's step grid.

## F-T-2 — a gate oracle that does not advance hung the exact engine

**Trace.** `next_edge_after(t)` returning t: `EventQueue::next_after` re-asks a function source
whose pending event is not after t; the answer t gives `h_use = 0`, the scheduler fires the
"edge" at t (same mask, dropped), and asks again — forever (dsed). trbdf2 "returned" only
because its step controller forced steps at its h floor (a floor warning; the edges were never
located). periodic_steady_state refuses such a gate by name.

**Rule (one, shared).** The refusal by name, `events::require_after` (event_queue.hpp), called
wherever an oracle is asked: the event queue (`EventQueue::next_after`: the exact engine's RK45
scheduler and trbdf2's native oracles) and the exact engine's auto and BDF2 schedulers, which
ask the switch_fn directly (the hang reproduced on the auto scheduler, `integrator='auto'`): an
answer not strictly after t raises `UnsupportedError(feature='switch_fn')` naming
`next_edge_after(t)`, the answer and t — periodic_steady_state's rule. trbdf2 never asks a
Python gate's oracle (it bisects a Python gate's edges: sim_bindings "the variable step lands
native edges only"), so it runs this gate, correctly, with its usual floor warning. Test
`test_ft2_a_gate_oracle_that_does_not_advance_is_refused_by_name` (integrator auto / rk45 /
bdf2, milliseconds).

## F-X-1 — cause found (session 2)

**The decision depended on t_end.** Ten branch-form bridges under the rule: t_end = 0.02 refuses
at 19.969 ms; t_end = 0.0205 / 0.025 / 0.03 run (113 / 138 / 163 events). Cause:
`resolved_scale_(t) = next_time_event_(t) − last_time_event_t_`. The bridges have no gate
oracle (the default constant mask), so (i) `next_time_event_` is the run's end, and (ii) every
dt_max/10 poll landing fired a same-mask "gate_edge" that set `last_time_event_t_` before the
"drop spurious same-mask gate edges" return. So t_res = the time LEFT in the run: 31 µs at
19.969 ms of a 20 ms run, the parasitic cut 1/(1e-6·31 µs) = 3.2e10 /s, and the bridges' common
mode L/R_g (5e9 /s) left the cluster: neither projected nor settled over (the previous trace's
"1e12 /s" was this same quantity a µs before the end). Earlier in the run (t_res ≫) the same
mode is in the cluster.

**Fix.** A poll landing that changed nothing is not a time event: `fire_event_` restores
`last_time_event_t_` at the same-mask return when the landing is the poll boundary
(`poll_t_`, set in `next_gate_edge_`). t_res is then the span since the last real time event
(the run's start here): constant, not shrinking toward t_end.

**After.** Ten branch-form bridges, `output_rtol=1e-5`, t_end 0.02: runs, 111 events, the
19.969 ms one alone (19.96895575516 ms, no chatter), event times equal to the rule-off run's to
9.8e-13 s, end state to 1e-12 V (test `test_fx1_ten_branch_form_bridges_run_under_the_rule`).
The answer to the brief's question: the 2e-10 s mode IS part of the cluster by the engine's own
definition at every instant but the last 3 % of a run; the settle and the projection were right,
t_res was wrong. Default path: the settle reads the same t_res (`settled_`), so oracle-less runs
near their end may decide differently (measured in the byte compares below).

## F-X-6 (new, OPEN) — a trial mode's projection survives the settle under the rule (forward converter)

Found by item 3 (the six transformer circuits against their mpmath truths, conductance form,
no dt). `switching/isolated/forward_converter.yaml`, exact_scoreboard `exact` (RMS of the samples
against the truth): rule off 4.7e-8; `output_rtol` 1e-4 / 1e-5 / 1e-6: 0.0457 (V(out) up to
0.175 V off, from 48 µs to ~65 µs). The same at the PR head's extension (0.04567, so not this
session's change). Under the release profile (`auto+next`, branch form) the cell is 1.0e-9.

**Trace.** Switch-off event at 48 µs: before it (SW on, D_rect on, D_fw off), I(L_o) = 5.620787 A
(truth 5.620787). The truth then has D_fw ON 5e-14 s later; I(L_o) does not move. The engine's
first sample after the event (2.2e-13 s) has I(L_o) = 6.206321 A (+0.585 A in 0.2 ps through 47 µH).
The settle's first trial mode is (SW off, D_rect on, D_fw off), eigenvalues 1.12e11, 1.21e10,
1.0e3 ± 6.3e3j, 0 (truth's own eigenbasis). The rule's projection cut there is
1/(1e-6·12 µs) = 8.3e10: it projects out 1.12e11 and keeps 1.21e10. That partial spectral
projection, computed independently in the truth's mpmath eigenbasis of the trial mode, gives
I(L_o) = 6.206321 A and V(n_lo) = −3.08e6 V: the engine's number to 7 digits. D_fw then turns on
(mask changed), the final mode's fast mode (5.13e10) is below the cut, and nothing undoes the
trial mode's jump: `settle_and_project_` restarts from the pre-event state only when the reinit
log grew (`track && changed && reinit_log_size_() > mark`). The rule-off cut (100/dt_max)
projects both fast modes; the same truth-basis computation gives 0.776 A there, so rule-off is
right for another reason (not traced).

**Fix needed (not made: it changes `settle_and_project_` on the default path too).** A trial
mode's projection must not survive a mask change: on `changed`, restart from `settle_x0_` and
project in the settled mode only, whatever the reinit log says; or a cut that never splits a
cluster of g_off modes. Either needs the byte compares and the scoreboard on the rule-off path.

## Session 2 — item 3 and verification on 44ab0a81

* **Event-instant states across `output_rtol`** (`.scratch/item3/evstates.py`, every
  `benchmarks/circuits` netlist the exact engine runs without dt: 99 ran, 27 refused/unmapped):
  event times and the full states at them are bit-identical for 1e-4 / 1e-5 / 1e-6 on all 99,
  the six transformer / coupled-inductor circuits included. The brief's open point of
  DECISIONS (6.7e-8 on the flyback) is closed by F-6-7: the steps no longer depend on the grid.
* **Rule on vs rule off** (same netlists): equal event counts and states within 1e-12 except
  ll06_buck_vcswitch (7.9e-3 of scale: rule off is the less accurate one, exact 6.3e-6 vs 8.8e-12),
  cuk 3.3e-11, diode_rectifier 1.3e-11, sepic 3.2e-12, full_bridge_dc 1.6e-12,
  diode_clamp_overvoltage 1.4e-12; event COUNT differs on cl_boost_pi, half_bridge_transformer_dc,
  boost_pfc_open_loop, flyback, forward (the rule resolves modes rule-off projects).
  Against the mpmath truth (`exact`, RMS of the samples): half_bridge_transformer_dc 3.0e-6 off →
  8.5e-13 on; flyback 2.1e-6 → 3.9e-7; cuk 5.5e-15 → 7.1e-15; sepic 4.6e-14 → 4.5e-14;
  forward 4.7e-8 → 0.0457 (F-X-6, open, pre-existing at the PR head); active_clamp_forward
  refuses under the rule (F-X-5, open).
* **Default path**: goldens identical; dsed bit-identity 34/36 against `.scratch/bitid_pr.npz`
  with bridge1_vf07/dsed different — the PR head's own extension (so_27c00520) gives the same
  34/36 against that file, and the tip's six dsed bit-identity circuits are np.array_equal to the
  PR head's (`.scratch/b1cmp.py`): no default number moved, the F-X-1 `last_time_event_t_` change
  included.
* ladder500_rc: 13.1 s vs 8.1 s rule off (1.6×), 3 072 samples, no cap, no warning. ladder100:
  2 083 samples, 0.78 s. `benchmarks/tools/counters.py` finishes (exit 0).
* F-X-1 independent truth: the local mpmath solve of the ten bridges around 19.969 ms does not
  finish (61 diodes, 91-state augmented modes: mpmath QR fails to converge at 20 digits; > 2 h at
  30 digits). The event at 19.96895575516 ms is backed by the rule-off run (event times to
  9.8e-13 s, end state 1e-12), not by an independent truth.

## Session 2 reviewer (fresh context, read only) — fixed / left

1. FIXED (high): a deferred end could hide the loss port's kick (heat lands after the cover
   check). A kicked end is never deferred. Underneath it the reviewer's scenario found a
   session-1 regression: under the rule with heat the samples read INSIDE a step missed the
   step's heat (RL chopper + RC junction, closed form: 0.0399 K off on a 7.47 K rise; PR head
   1.35e-13). With a state kick the rule's samples are steps again (the PR head's behaviour for
   that case only): 1.35e-13 restored (test `test_fx4_a_heated_step_end_is_stored_under_the_rule`).
2. FIXED (high): the F-X-1 restore keyed on the oracle-less poll only; a plain Python gate
   (EdgeSearch, `has_edge_oracle()` false) lands on finite poll boundaries the same way. The
   restore now also covers `!has_edge_oracle()`. Test: the ten-bridge F-X-1 test with a Python
   gate. Default path: dsed Python-gate runs of buck_ccm / buck_dcm / boost / bridge1_vf07 /
   bridge3 np.array_equal to the PR head's extension (`.scratch/pygate_cmp.py`); goldens and the
   dsed bit-identity set as before.
3. LEFT (medium): with no edge known, t_res is still "t_end − last real event": the parasitic cut
   depends on t_end. Needs a schedule- or ceiling-based scale; only the rule-off run backs F-X-1.
4. FIXED (medium): a deferral is accepted only if the run's chained state at the step end is
   within 1e-3 of the tolerance of the flow from the last stored sample (one O(n²) advance per
   deferred step, under the rule only).
5. LEFT (low): the cover check does a full output_step_ (wasted work when it fails).
6. LEFT (low, design): the absolute floor is one scalar atol for every state kind.
7. LEFT (low): `require_after` lets NaN through as "no edge".
   The KPI change is now in the CHANGELOG (it moves every engine's T_j KPI to the closed form).

### Rehearsal re-run (session 2, on 44ab0a81: F-X-4 deferral in)

auto+next 123 agree / 11 disagree / 2 error (base 133 / 2 / 1); dsed:output_grid and dsed+next
120 / 2 disagree / 14 refused (base 122 / 0 / 14). The ACCEPTANCE "no cell worse by > 2×" IS NOT
MET, and the cause is the F-X-4 deferral (the after-floor run a5b9a485 had none of these):
* 9 DC / resistive cells (current_divider_3way, ll01, voltage_divider_3stage, bridge_resistive_pure,
  bode_rlc_low_pass, stiff_rlc, pulse_voltage_rl, analytic wheatstone, pwl_diode_drop) now store
  2 samples, err_max 0.0; the harness drops each trace's first sample, its err_rms is NaN and the
  cell is "disagree (undersampled)". A harness limit, not an error, but an outcome change.
* 25 cells agree with err_rms up to ~10× the base (rc_step 4.3e-7 → 5.9e-6, tol 1e-3): the base
  read the 1 µs poll grid's extra ends. Checked against the truth on diode_half_wave_rl_load:
  every sample exact (5.7e-16), the worst chord midpoint 9.96e-5 on a 10.3 V range =
  0.97·rtol·range: the rule's promise, met; the scoreboard `exact` there 4.9e-6 → 8.1e-4
  is dominated by the event jumps.
Decision needed (orchestrator): F-X-4 (store only what the tolerance needs) and the rehearsal's
2× bar cannot both hold; either the harness learns a 2-sample exact trace and the 2× bar is read
against the rule's tolerance, or F-X-4 is reverted to "every step end stored".

### Final verification (89b5d936)

* Full pytest: 6116 passed, 110 skipped, 8 xfailed (12:56); `-m reference` 383 passed;
  reference_status exit 0; ruff clean; openspec strict 79/79; stubs up to date.
* C++ `pulsim_core_dsed_tests` (project CMake, Eigen 5, Release): 142 cases / 41 265 assertions pass.
  Not built: Eigen 3.4.0, the other C++ targets, full ctest.
* Goldens identical; dsed bit-identity 34/36 as at the PR head; Python-gate default runs equal to
  the PR head's. Scoreboard dsed 81/81 < 1e-3, none worse than session 1's final; flips default
  100/109, branch 100/109, ideal 85/100 (15 refused), outcomes identical; banks 1/9/10 unchanged
  (12 refuses as before); bar scripts (fuzz10, fuzz, truths: ten banks iL0a 7.66e-12, orbit)
  identical to session 1's logs; dsedall 340 passed.
* ladder500: 22.6 s vs 11.4 s rule off (2.0×, the host loaded: rule off alone went 8.1 → 11.4 s);
  DC 2 s: 2 000 002 samples rule off, 2 under the rule.
* Instructions per accepted step, tip / PR head (min of 3, `/usr/bin/time -l` minus an
  import-only run, 0.04 s runs): buck pwl 0.968, buck trbdf2 0.956, buck dsed 1.010,
  bridge3 dsed 1.003. pwl/trbdf2 are not touched by this branch: their −3…−4 % is the method's
  noise, so the 0.5 % bar cannot be read from these numbers.

## Session 3 — branch `fix/core-release-f3-exact-leftovers` (from d59a18fe + merge of F-S-4 8366a3e5)

Base for every "before" below: the extension built at the merge commit b08f4e4e (`.scratch/base_py`).
Truth: mpmath `exact_truth.py` from `origin/feat/core-soft-switching-validation` fa4b5c71
(in `.scratch/f3/`, not committed) and the stored `benchmarks/truth/*.json`.

### F-X-6 — FIXED: a trial mode's projection never survives the settle

**Why the rule changes a trajectory at all.** The one piece of output-rule state the settle reads is
the projection threshold in `project_`: rule off `100 / dt_max`, rule on `1 / (1e-6 · t_res)` (only
the parasitic cluster is projected — by design, so the rule resolves modes rule-off projects). It does
not depend on the `output_rtol` VALUE (1e-4 / 1e-5 / 1e-6 stay bit-identical) but on whether the
rule is on. Trace at the 48 µs switch-off (temporary fprintf in `settle_and_project_`, states
[v_C, i_Lm, i_sec, I(L_o)]): rule off — one settle pass, mask 010, `changed = 0`, both fast modes
projected (I(L_o) 5.62 → 0.776), the guard then repairs the mode from the held state (final 5.6208,
right). Rule on — trial mode 010 projected PARTIALLY (1.12e11 /s out, 1.21e10 kept), D_fw turns on
(mask 011, `changed = 1`), no reinit record was written (a slow-manifold projection writes none),
so the restart (`track && changed && log grew`) did not run: I(L_o) 6.2063 A kept.

**Fix.** On any mask change in the settle the settled mode starts again from the pre-event state and
projects alone (`if (changed)`; the log is truncated when tracked). Before → after, forward converter,
I(L_o) at 48.0000553 µs (truth 5.620787) and full-run mean of V(out) − truth 6.6073535537897:
rule 1e-4 / 1e-5 / 1e-6: 6.2063 A, +0.131 V → 5.62078 A, −1.2e-4 / −1.1e-5 / −1.2e-6 V (the samples'
chord interpolation, ∝ rtol); scoreboard `exact` at 1e-5: 0.0457 → 1.0e-9. Rule off unchanged.
Test `test_fx6_forward_converter_keeps_the_inductor_current_at_the_switch_off`.

**Moved on the default path** (`.scratch/f3/dsedall.py`: every scoreboard circuit × dsed (with dt,
as the scoreboard), dsed+next, no dt rule off; hashes of times / states / events): 4 circuits, both
dsed and no-dt cells; truth `exact` before → after (dsed cell): half_bridge_transformer_dc 3.0e-6 →
1.49e-6, boost_pfc_open_loop 2.21e-7 → 2.57e-7 (1.16×, worse, same decade), active_clamp_forward
2.54e-8 → 5.16e-9, flyback_converter 2.12e-6 → 1.18e-6. No cell crosses a scoreboard threshold
(1e-3 / 1e-6 / 1e-9): the exact-engine row stays 81 / 77 / 70 cell by cell. Rule-on cells: flyback
3.9e-7 → 7.3e-7 (1.9×, worse), forward 0.0457 → 1.0e-9, others identical. Goldens identical; dsed
bit-identity 36/36 default + 36/36 shadow. Not traced: why boost_pfc / rule-on flyback get slightly
worse (both stay within 2× and far below 1e-3).

### F-X-5 — FIXED: a no-crossing certificate per scan interval

Between events a diode predicate is affine in the state, g = Re Σ q_k z_k(τ) + d(τ) (q_k = c·V_k,
probed once per mode and predicate from `p.value`, cached in `ExactLTI::scan_q`; d affine over the
step), and z̈_k(τ) = z̈_k(0) e^{λ_k τ}. On [a, a + δ]: |g − chord| ≤ E = Σ_k |q_k z̈_k(a)|
min(δ²/8 · e^{max(0, Re λ)δ}, 2/|λ|²) — the output rule's per-mode bound. Ends of one sign, both
outside the significance band, with min(|g_a|, |g_b|) > E hold no root; otherwise the interval is
halved (depth ≤ 40, ≤ 256 halvings per interval) until each piece is certified or holds a
significant sign change (then `locate_root_bracket`). Tier-1 modes without sine sources only
(otherwise the D3 scan alone, as before). The eigenvalue-count alternative (n − 1 real zeros) was
not needed: the bound decides directly.

active_clamp_forward, D_fw OFF/ON (truth 1.66354841887779847e-5 / 1.67126164738377463e-5 s):
rule off no dt — missed (nearest event 2.8 µs away), `exact` 2.74e-6 → found within 7e-17 s,
5.18e-9; rule 1e-5 — `DiodeModeInconsistentError` → runs, 5.16e-9; dt = 1e-7 unchanged (found
before). Test `test_fx5_a_double_crossing_inside_one_scan_interval_is_found` (None, 1e-4..1e-6).
Default path: only active_clamp_forward's no-dt cell moves (670 → 674 events); every other
scoreboard dsed / dsed+next / no-dt cell, the goldens and the bit-identity set are identical.
**Cost** (instructions per run minus an import-only run, min of 6 interleaved, same build flags,
base b08f4e4e): buck_ccm dsed 0.979, bridge3 dsed 0.998, forward_converter 1.001,
boost_pfc_open_loop 0.992, active_clamp_forward 1.111 (4 more events, 10 more samples, the halvings
around the double crossing). The per-step price is one O(n²) z̈ per exact step with an armed diode
predicate and O(n) per predicate per scan interval; nothing measurable on the circuits above.
**Cost, corrected (final tip)**: the single-run figures above are inside the import's noise
(±70 M instructions on a 3.2 G import). Ten runs per process, same method, base b08f4e4e, no-dt
route (rule off), final tip ffdf5007: buck_ccm 1.018, bridge3 1.000, boost_pfc_open_loop 0.999,
forward_converter 1.111, active_clamp_forward 1.134 (an intermediate tip, before the affinity change: 1.015 / 1.000 /
0.999 / 1.139 / 1.132; the base's own spread across the two runs is up to 2.7 % on the forward
converter). The price is the halvings where the certificate does not hold: each midpoint is a
state sample plus a predicate read at a new instant (a source b-vector build); predicate reads
+12 % (forward, 6 813 → 7 644) and +18 % (active clamp, 22 261 → 26 170). This misses the 0.5 %
cost gate on the two forward converters and on buck_ccm (+1.5..1.8 %); it is the stated price of F-X-5 (a tighter bound, e.g.
the exact |z̈| along the interval instead of its start value × e^{Re λ τ}, would cut the halvings;
not done).

### F-X-4 harness and the new bar — DONE (decision taken)

`metrics.first_kept`: dropping the initial sample is skipped when it would leave < 2 samples (the
reference is then read from its own first kept sample); a trace that cannot be compared raises with
the reason and `classify` records `undersampled: <why>`. The nine DC / resistive cells now agree
(current_divider_3way 0, ll01 1.3e-7, voltage_divider_3stage 0, bridge_resistive_pure 1.3e-13,
bode_rlc_low_pass 0, stiff_rlc 2.9e-9, ...). New bar for cells under the rule (DECISIONS 6):
`rule_ratio` = worst |interp(midpoint) − reference| / (output_rtol·range + atol) between events,
≤ 1, and no outcome change. Rehearsal counts: see "Rehearsal under the new bar" below.

### Docs — the opaque-gate poll (decision: stays at dt_max/10)

`simulate()` docstring (`dt`, `output_rtol`), `_dsed_dispatch` heuristics, KNOWN_LIMITATIONS, the
release spec scenario and the rehearsal text now say: 10 µs ceiling, 1 µs when the gate names no
edge. AI_GUIDE and the dsed overview already said `dt_max / 10`. No number moves.

### F-S-4 leftover — FIXED: a start projection of a given state is reported

`record_start_projection` (adapter), called once after the t = 0 decision: a given state that the
start's settle / projection moved by more than 1e-6 of max(|x0_i|, the circuit's scale of its kind)
gets a record at t0, kind `start_projection`, states = the moved ones, `projection_gap` = the largest
move, dW = ½ΔᵀWΔ (W the mode's energy weight, else diag(C, L) — mutual terms of coupled windings
left out there, a named shortcut). In the ideal limit ½ΔᵀWΔ is the jump's loss including the
sources' impulsive work (10 V onto 1 nF at 8 V: V·CΔv − ΔE = 2 nJ = ½C(2 V)²), as for the
dependent-state records. Not written when the start already wrote dependent-state records. The
threshold is 1e-6, not reinit's 1e-9: from rest a reverse-biased diode's g_off leakage (10 V · 1 nS =
1e-8 A in a 1 A circuit) is a projection too, and no report. `on_impulse='error'` refuses it by name;
`impulse_losses` / `device_loss_summary` book it to no device. Not refused by default (the precharged
capacitor's inrush through an ON diode runs and is reported). 1 A in L behind a reverse-biased diode:
0 A (+1e-8 leakage) with a record dW = 0.5 mJ = ½Li². Records on the default suites: none of the 81
scoreboard circuits (dsed, dsed+next, no dt) and none of the 18 examples the exact engine runs get a
start record; the exact test files (dsed*, exact*, release_findings*, mode_selection*,
branch_form_reinit, release_uniform_rules: 425 pass) are unchanged.

### Small ones

* `events::require_after` refuses NaN by name (`test_require_after_refuses_nan_by_name`, three
  integrators). Was: a NaN pending event compared false everywhere and the queue never asked again.
* t_end dependence — LEFT OPEN (strict xfail `test_parasitic_cut_does_not_depend_on_t_end`). The
  parasitic cut is 1/(1e-6 · t_res), t_res = next time event − last time event; with no gate edge the
  next time event is the run's end. Case: 10 V → ON diode → 0.1 Ω → 1 nF ∥ 1 kΩ under the rule:
  v_C(0) = 0 V in a 10 µs run (mode resolved), 9.999 V in a 1 ms run (projected, with a start
  record). It should depend on the mode's own spectrum (a gap in its decay rates), not on t_end; that
  moves every oracle-less run's settle (`settled_` reads `resolved_scale_` on the default path): not
  local. On the 51 oracle-less benchmark circuits event times are identical for t_end and 1.5·t_end
  (rule off and on), so the benchmark set does not show it.

### Session 3 reviewer (fresh context, opus, read only) — fixed / left

* M1 FIXED — the F-X-5 halving located a sign change in a piece that started past the
  threshold (a way back, no event in the scan); it now needs the admissible start `s·g_a > 0`.
* M2 FIXED — the certificate's predicate row is probed at x = 0 and the unit eigenvectors, which
  assumes g affine; an ideal branch diode's DiodeOn reads −1 inside its band. The row is used only if
  it reproduces g at the step start (|g(x0) − g(0) − Re Σ q_k z_k| ≤ 1e-6 · scale), else no
  certificate (the D3 scan alone). `exact_scan_modes` returns z with z̈.
* M3 LEFT, by the truth — the reviewer asked that the restart be re-checked by the cascade (a trial
  ON diode equalising two capacitors could be physics the settled mode needs). Tried: on the forward
  converter the restarted state reads D_fw off again at every switch-off (mask 11 → 10 at each switch-off printed, every 20 µs),
  so the check kept the iterated state and I(L_o) went back to 6.2063 A (truth 5.6208 A). A
  capacitor equalisation is a dependent-state projection that writes records, and the restart then
  re-records it in the settled mode; the no-record branch is the parasitic cut alone, which is not a
  constraint of the settled mode. `resettle_projected_` keeps the old rule (not on this path).
* M4 FIXED (part) — a NaN rule ratio (no interval between switching steps) fell to "not worse"; it
  now falls to the truth check (`exact` beyond 2×). The judge does not apply the truth check to a
  rule cell with a finite ratio: the decided bar is the rule's promise and the class (DECISIONS 6).
* L1/L5 FIXED — band checks before the row probe; the probe has its own buffer (no `x_tmp_`
  aliasing); `exact_scan_modes` reuses its pwl buffers (no heap per step).
* L4 LEFT — `n_predicate_evals` now counts the certificate's probes (2n+1 per mode and predicate,
  +1 per step that uses it); not a listed number. L7/L8 noted, no change.
* After the fixes: 243 default cells (81 circuits × dsed / dsed+next / no-dt) identical to the
  pre-review tip; goldens and bit-identity 36/36 + 36/36; the F3 tests 16 passed + 1 xfail.

### Rehearsal under the new bar (final C++ ffdf5007; harness with the two fixes below)

Cells of auto, auto:route, auto+next, dsed, dsed:output_grid, dsed+next re-measured (the other
configurations' cached cells carry an older profile signature and are not re-rendered; the
committed NEXT_DEFAULTS_REHEARSAL.md is left as it was; this render is in `.scratch`).

| configuration | agree | disagree | error | refused | worse vs base | rule cells | ratio > 1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| dsed:output_grid (vs dsed) | 100 + 22 | 0 | 0 | 9 + 5 | 0 | 22 | 0 |
| dsed+next (vs dsed) | 100 + 22 | 0 | 0 | 9 + 5 | 2 (truth) | 22 | 0 |
| auto+next (vs auto) | 107 + 27 | 1 | 1 | 0 | 53 | 122 | 54 (9 NaN → truth) |

(circuits + analytic). dsed+next's two: half_bridge_transformer_dc ex 1.5e-6 → 3e-6 and
active_clamp_forward ex 5.2e-9 → 2.5e-8 (dsed+next, not the rule; combination of singles). No
outcome change anywhere except auto+next high_q_rlc_ringdown disagree → agree (better).
auto+next ratios > 1: 34 in (1, 1.25], 6 in (1.25, 2], 14 above 2 — ll06_buck_vcswitch 596 (a 5.5 ns
chord across a fast ring: 32.76 V vs reference 33.05 V), ll10_full_bridge_inverter_lc 93.5,
ll09_half_bridge_inverter_leg 59.6, high_q_rlc_ringdown 12.4, ll07_boost_vcswitch 8.16,
diode_bridge_rectifier 5.39, half_wave_rectifier_sine 5.26, diode_clamp_overvoltage 4.87,
ll02_rc_pulse_charge 4.74, core_loss_steinmetz_sine 4.53, three_phase_diode_rectifier 3.65,
diode_half_wave_rl_load 3.37, sine_voltage_rc 3.11, diode_rectifier 3.09 (the chord ending on
the 5 µs source edge). The (1, 1.25] band is the chord bound being an estimate (z̈ at the
interval's start); the large ones are fast transients the bound under-reads. Every one of them
agrees at the samples (strict err_rms ≤ 4.6e-4) and the truth (`exact`) did not grow. These are
OPEN rule findings (the engine's chord bound), not harness artifacts. NOT fixed this session.

Harness fixes found by the re-run (committed): the trace's sample at an event is the left limit
(a jump's right limit lives in the right-limit arrays), so a chord leaving an event is excluded
(analytic hard_switched_rc 5.0e4 → 0.995); the reference's first step (its given, possibly
inconsistent, start) is excluded (vcswitch_resistor 2.5e4 → ≤ 1). Analytic cells pass their
event instants.

### Final verification (C++ ffdf5007; tip with docs/harness after it)

Full pytest 6226 passed, 110 skipped, 9 xfailed (0 failed); `-m reference` 383 passed;
reference_status exit 0; ruff clean; `openspec validate --all --strict` 79/79;
`update-number-changing-release --strict` valid; stubs up to date; `pulsim_core_dsed_tests`
(Eigen 5) 41265 assertions / 142 cases pass; goldens + bit-identity 36/36 default and shadow;
243 default scoreboard cells identical to the pre-review tip; event times/states bit-equal across
output_rtol 1e-4/1e-5/1e-6 on 100 benchmark circuits (26 skipped by the tool), forward converter
and active clamp included. Not run: Eigen 3.4.0, Windows/MSVC, a second reviewer pass on the
review fixes.

## Session 4 — branch `fix/core-release-f4-chord-bound` (from 6cb51c00, C++ ffdf5007)

### Item 1 — the output rule's promise: the diagnosis (evidence before code)

`.scratch/f4/both.py` (auto+next cell, the 100 circuit units with a rule cell): the
rule ratio against the harness's reference (trbdf2 rtol 1e-8, read linearly between its
steps) and against the engine's OWN exact flow (the same cell at output_rtol/100, event
times asserted bit-equal on all 100). Base tip ffdf5007:

| | ratio vs trbdf2 ref > 1 | ratio vs exact flow > 1 |
|---|---:|---:|
| cells | 54 | 6 (ll06 635, lcc_resonant_inverter, active_clamp_forward, half_bridge_inverter_lc 6.4, chopper_with_rc_snubber 1.42; flying_capacitor 1.0002) |

* **48 of the 54 are the reference's own error**, not the rule: against the exact flow
  they sit at 0.96..0.9997 (high_q_rlc_ringdown 12.4 → 0.9998: the trbdf2 reference had
  2 points in the chord the flow resolves with 80; sine_voltage_rc 3.11 → 0.19 (0 reference
  points in the chord); ll09 59.6 → 0.988; ll10 93.5 → 0.982; ll07 8.16 → 0.995; the
  rectifiers 3..5.4 → 0.96..0.998). The "34 cells within 1.25×" are this: the bound is
  TIGHT (0.98–0.9997 of its tolerance against the flow), so any reference error tips them
  over. Not a constant in d²/8, not the midpoint (the bound holds for every τ), not the
  range (the engine's range so far ⊆ the run's range: its tolerance is the smaller one).
* **The genuine violations are not the second derivative read at the interval start.** The
  bound in `exact_chord_step` was already the rigorous per-mode form the brief asks for
  (Σ_k |V_jk||z̈_k| max(1, e^{Re λ d}) with the exact-at-τ=0 refinements, Peano kernel ≤ d²/8)
  — but per STATE. The stored samples are the RECOVERED vector y = R·x + R_osc·(sin, cos) +
  R_b·b_pwl(t) (node voltages, branch currents). ll06, at the worst chord (28.38 µs, 5.47 ns):
  I(L1) 0.82 of its tolerance, V(out) 1e-5, but V(sw) = V(n_l) **596 ×**: with switch and
  diode both off V(sw) = (48 g_s − I(L1))/(g_s + g_d), 5 kΩ, so I(L1)'s admissible chord
  error 5.75e-5 A reads 0.29 V on a node whose tolerance is 4.8e-4 V (32.76 V chord vs
  33.0475 V flow; the dense flow's max|ÿ|·d²/8 = 723 × tol). Same cause on the inverter
  LC / snubber nodes.
* Two smaller defects found on the way: (a) a rule chord a sliver short of a step end was
  STRETCHED to it by `near_time`'s 1e-12 s floor (ll06, a 1.46e-10 s chord after a diode
  event: 1.007 ×); (b) the harness read the flow across a jump's right-limit row (one ulp
  after the event) and across source corners (pulse edges) — its own artifacts (104 × on
  the active clamp, 5e6 on a 1 ns pulse edge), not the engine.

### Item 1 — the fix

* `exact_chord_step` bounds the rows of a row map P = O·V cached on the mode
  (`ExactLTI::chord_P_re/im`; empty: P = V as before), O = [I; R, R_osc] built by the
  adapter (`LTIEntry::recover_osc` = R_b·stamp·amp·(cos φ, sin φ) per oscillator, next to
  q_sin / q_cos). The piecewise-linear sources' direct share is linear in τ (no chord
  error); the sines' direct share is an oscillator mode like any other (its |λ|² = ω²
  enters the same sums). The scheduler tracks each recovered row's range
  (`output_ylo_/yhi_`) and passes its tolerance in `ChordWork::ty` (same rule: rtol ×
  max(range, 1e-3·peak), floored at atol, and at the row's own evaluation error
  16·u·κ·(|R||x| + |R_osc|) — a balanced three-phase neutral, exactly 0 with 3.4e-7 V of
  rounding, held to atol spent the whole 2e6-sample budget).
* The deferred-end gap check (F-X-4) also on the recovered rows (floored at the rounding).
* `push_output_samples_`: no chord stretched past its bound to reach the step end — the
  rest is halved; same for the thermal path's rule-cut step at a time event.
* Harness: `metrics.rule_ratio_flow` — the rule's ratio against the exact flow: the same
  cell at output_rtol × 0.05 (× 0.2 when the per-interval cap cuts it; capped twice or other
  event times = not comparable, listed), read at each chord's midpoint by the quadratic
  through the dense samples INSIDE the chord and the chord's two ends (one smooth piece).
  The old reading against trbdf2 is kept as `rule_ratio_ref` (information).

### Session 4 reviewer (fresh context, opus, read only) — fixed / left

The reviewer fuzzed a numpy port of `exact_chord_step` (400 random systems with growing
modes, output rows up to 1e3, ramps: worst 0.9994 × tol; monotone in d on 3000 cases),
checked the sine stamp convention of `recover_osc` against `compute_sine_b_extra` /
`current_stamp_` (right), and found the rule-off path byte-identical by construction.
* FIXED (medium) — the sliver halving could still stretch: with near_time's 1e-12 s floor
  and d < h/2 it stepped h/2 > d (a 1e11 rad/s ring, d ≈ 9e-14 s). Now `min(d, h/2)`, also
  on the thermal path.
* FIXED (medium) — the first rounding floor (16·u·κ·|R||x|) was far too loose (κ is a
  worst case): it let ll06 / ll09 / rlc_parallel_step back to 16 / 11 / 3. The floor is now
  the row's actual evaluation magnitude, 16·u·Σ_k |P_jk||z_k| (computed in
  `exact_chord_step`, `ChordWork::tyf`); the deferral check reads the same floor.
* FIXED (low) — a bisection whose 1e-12·h probe does not fit returned it as a certified
  step; it is now `bounded = false` (counted in `n_output_unbounded`, the chord is the step).
* FIXED (harness) — a dense flow with `n_output_unbounded` > 0 is not a flow (not compared).
* LEFT (low) — the time-resolution floor `max(d, 64·u·max(1,|t|))` stretches a chord at
  large t (a 1 GHz ring at t = 100 s, rtol 1e-6) with no counter. Not seen on the set.
* LEFT (medium, stated) — when the rounding floor binds, the promise is
  max(rtol·range, atol, the row's own rounding); no counter reports it.
* LEFT (harness, decided bar) — `rule_ratio_flow` reads chord MIDPOINTS (the bar of
  DECISIONS 6); for a fast decaying mode the maximum is nearer the start (the midpoint reads
  0.5–0.62 of it). The property tool reads random instants, which covers it.

### Session 4 — state at hand-off (context budget reached)

Measured (auto+next, the 100 circuit units with a rule cell; `.scratch/f4/after2.jsonl`,
tip before the reviewer fixes, harness `rule_ratio_flow` with its first stencil):
* ratio vs the exact flow ≤ 1 on 97 / 100 comparable; the three over 1 were re-read after the
  stencil fixes (bracketing, linear on a lopsided stencil): ll06 0.998, rlc_parallel_step
  0.981 — and **ll09_half_bridge_inverter_leg 7.3 (OPEN, genuine)**: the chord
  [24.013458 µs, 25 µs] starts 13 ns after the 24 µs event, where a fast mode is still
  alive (V(out) rises 0.82 mV in the first 1.8 ns of the chord, then 0.16 mV over the rest);
  the chord reads 47.61979 → 47.62077 V straight, the flow 47.62061 V at +1.8 ns. The STATE
  row I(Lload) reads 6.9 × too, so this is not the recovered-row defect: the bound (or the
  path that stores the chord: the fast-sample step ends of `plan_fast_samples_`, the F-X-4
  deferral) let a 1 µs chord through with the fast mode's amplitude 2|γ| ≈ 1.6 mV above
  the tolerance (≈ 7e-5). NEXT STEP: print, in `push_output_samples_` / the deferral block,
  (t0, t1, d, output_certified_, out_defer) for ll09 between 24.0 and 24.1 µs; check whether
  the chord came from the deferral check (`output_step_(ts, states.back(), …, false)`) with
  a stale `ts` / tolerance, or from a step whose rule call returned h uncounted.
* sample counts after / before (ratio): min 1.00, median 1.00, p75 1.25, p90 4.65, max 14.2
  (19 of 100 above 2×: boost_pfc_open_loop 13 531 → 191 771, buck_dcm 891 → 8 590,
  textbook_erickson_buck_ccm 6 946 → 48 104, ll06 2 943 → 18 251, cl_buck_pi 20 228 →
  97 726 …: the switch-node / high-impedance rows that were never held before — the buck
  DCM's V(sw) ring binds at 0.991 of its tolerance). three_phase_balanced_rl_load 5 153 →
  8 038 (after the rounding floor; 2.1e6 without it). ladder500 NOT re-measured.
* property tool (`benchmarks/tools/output_rule_property.py`, dt grid = median chord / 100,
  200 random grid instants per circuit, events within 1e-12 s): on the earlier tip 26 of 34
  comparable, worst 0.9968, none over 1 (8 not comparable: event offsets 1e-17..1e-13 s —
  now accepted below 1e-12 s — and two with other events, half_bridge_transformer_dc /
  forward_converter, rule-off projection). The `--truth` (mpmath) run on six solved
  circuits (`.scratch/f4/truth_units.txt`) and the 30-circuit re-run on the final tip were
  NOT run. `.scratch/f4/run3.sh` runs all of it plus the output_rtol events gate
  (`.scratch/f4/evgate.py`).
* NOT run on the final tip: the rehearsal tool itself (its rule column now reads
  `rule_ratio_flow`, the old one is `rule_ratio_ref`), goldens / dsed bit-identity /
  scoreboard default cells (rule-off is unchanged by construction: every change sits behind
  output_rtol > 0 — the reviewer confirmed — but not measured), `pulsim_core_dsed_tests`
  (tree configured at `.scratch/build-ctest`, not built; the new C++ case is in
  test_exact_lti.cpp), full pytest, instructions per run.
* Item 2 (certificate cost, the sines in the certificate — coordinator's addition: the
  three-phase bridge with line inductance, D2 ON at 5.000000250912935 ms with D5 still ON,
  D5 OFF at 5.001350943090826 ms; the engine flips both at the D2 instant) and item 3 (the
  t_end-dependent parasitic cut): NOT started. For item 2 the shared object is ready: the
  certificate's row q_k = c·V_k is a `chord_P` row; `exact_chord_step`'s second-order form
  (|Σ q z̈| + δ|Σ q z̈ λ| + δ²/2 Σ|q z̈||λ|²) with z̈(a) = z̈(t0)e^{λτ} is the sharper E; for the
  sines the predicate's direct oscillator share is (recover_osc[from] − recover_osc[to]) ×
  its gain, which lets `exact_scan_modes` drop its `aug_omegas.empty()` refusal.

## Session 5 — output rule (branch `fix/core-release-f4-chord-bound`; stopped by the coordinator at its context budget)

### Item 1 — ll09's 7.3x: a HARNESS artifact, fixed (fe78e327)

* The chord is [24.013458151530692 µs, 25 µs] of V(out) / V(load_mid) (and the I(Lload) chord
  [1.050278676 ms, 1.074 ms]). The dense flow (output_rtol × 0.05) has exactly ONE sample inside
  it, at 24.0152271 µs (1.8 ns after the chord's start, where the post-edge fast mode is still
  alive). `metrics.rule_ratio_flow` applied its lopsided-stencil guard only when the chord held
  ≥ 2 dense samples; with one it fitted the quadratic through (start, +1.8 ns, end) and
  extrapolated the fast piece's curvature over 985 ns: 5.8 (V(out)) / 6.9 (I(Lload)).
* The engine's chord is within its promise: tolerance = 1e-5 × 1960 V (V(out)'s range includes the
  −979 V right after the 24 µs edge) = 0.0196 V; the chord's midpoint error against the dense flow
  is 4.1e-4 V (0.02 of it). Every ll09 chord read against the dense flow LINEARLY at 14 fractions
  (1e-3 … 0.997) of each chord: worst 0.989 (V(mid)/V(out)/V(load_mid)), 0.942 (I(Lload)).
  No engine path emitted an uncertified chord; no flag was wrong. (The predecessor's 10.8 in
  after2.jsonl was the tip before the reviewer's rounding-floor fix; on 075feac9 the engine part
  was already gone and only the stencil artifact remained.)
* Fix: the guard also for a one-sample stencil (`ts.size >= 3`); test
  `test_f4_rule_ratio_flow_never_extrapolates_one_short_dense_piece`.
* Rule vs exact flow on 075feac9 (after.py, auto+next, the 100 circuit units): 99 comparable,
  98 ≤ 1 (worst ≤ 0.99988), ll09 7.27 = the artifact above (now ≤ 0.989); 1 not comparable
  (three_phase_balanced_rl_load: the dense flow hit its 2e6-sample budget at 4.53 ms). NOT re-run
  with the fixed harness on the 100.
* Paths that emit a chord, audited BY READING `scheduler.hpp` (no hole found): regular step
  (`push_output_samples_`: each cut certified by `output_step_`, the last chord's certification
  returned; uncertified = cap / budget / unbounded → the end is stored and counted); the chord
  into an event (t1 = located root, end stored as the event sample, state = the same flow);
  the first chord after an event (starts at the stored post-settle state, which is `x`); slivers
  (halved, `min(d, h/2)`, also on the thermal path); deferred step ends (`output_pend_`: bound
  re-evaluated from the last STORED sample over [ts, t1] in the current mode, only when the step
  had no event, `out_transparent` (no mask/state change at a poll), no kick, not at t_end, plus the
  state and recovered-row gap checks; ticks / zero-length edges / numeric steps flush first);
  the last chord of the run (stored, never deferred near t_end); a resumed run's first chord
  (ranges restart, so the tolerance is tighter; no chord crosses the resume). Not checked by a
  test: the deferral after a `ControlTick` that fires without a mask change (flushed at the top
  of the loop by `tick_due_`).

### Item 3 — WIP in the tree at this tip (built, NOT verified)

* `chord_row_bound` (exact_lti.hpp): exact_chord_step's bound for ONE row (F split, three forms,
  growing modes); the event scan's certificate now uses it (z̈(a) = z̈(t0)e^{λτ}, complex). It is
  ≤ the old certificate bound everywhere (same F part, S part min of three forms instead of the
  triangle one), so fewer halvings, never a certificate the old bound refused for a wrong reason.
  C++ test "the scan certificate's row bound is the output rule's bound" (test_exact_lti.cpp)
  pins it to exact_chord_step's `w.eb` per row — NOT built / run.
* Sine sources: `exact_scan_modes` no longer refuses augmented modes (z, z̈ on (x, sin, cos));
  the predicate's direct sine share is `node_pair_osc(from, to)` (recover_osc rows; an ON
  branch-form diode reads its current row `var`), added to the probed row; the affine check
  subtracts that share's value at t0. Shortcut: assumes the diode predicate is v_D − const
  (gain 1, `crossing_signal`); a predicate of another gain would get a wrong sine share.
* THE THREE-PHASE BRIDGE IS NOT A CERTIFICATE DEFECT (evidence on this build, debug print since
  removed): truth (fa4b5c71's exact_truth, to 6 ms): events 0.3719356718626846 ms (D1),
  1.490258218668964 ms (D5), 5.000000250912935 ms (D2 ON), 5.001350943090825 ms (D6 OFF — the
  notes' "D5" is D6). The engine: the settle at 5.0000002509 ms keeps D6 ON (v_D = 1.6e-7 V,
  band 5.4e-10, next = ON), then the MODE GUARD repairs it: `diagnostics['mode_guard']` repairs 1
  at 0.005000000250912887, diode D6, violation 0.12 V — `judge_` reads the settled mode at
  1e2·h_e and 1e4·h_e (capped at the next time event) and calls D6's legitimate turn-off 1.35 µs
  later a violation of the mode, and the LCP turns D6 OFF at once. `mode_selection='lcp'` merges
  them too. The fix belongs in `judge_` / `guard_check_` (a law that holds significantly at t+
  and breaks later is a crossing the scan locates, not a violation) — settle territory, not done.
  Also seen: the engine's D1 event 0.3719354704753909 ms vs truth 0.3719356718626846 ms (2e-10 s).

### Build being waited for at the stop

The extension build of this tip (item 3 WIP, debug print removed). It was meant to check
`dsedall` (the 243 default cells) tip vs base — the base hashes are recorded in
`.scratch/f4/dsedall_base5.json` (base = 075feac9's extension saved in `.scratch/base075/`).
Also recorded with that base: goldens + dsed bit-identity (`.scratch/golden_s5base`,
`.scratch/bitid_s5base.npz`; check: `BASE=s5base zsh .scratch/bitid5.sh check`). Chain script for
the rest: `.scratch/f4/chain1.sh` (dsedall cmp, bitid, cost vs base075, run5.sh = rule vs flow on
100, property test on 100 + `--truth` on six with fa4b5c71's truth tool in `.scratch/f4/truthtool`,
events gate).

### Item 2 acceptance — status at this tip

| check | status |
|---|---|
| property test ≥ 30 circuits, 200 random instants | NOT run (tool ready: `.scratch/f4/prop_truth.py`, 100 units in `prop_units_all.txt`) |
| `--truth` on six | NOT run |
| rehearsal tool auto+next / dsed:output_grid / dsed+next, analytic 22 | NOT run |
| evgate 1e-4/1e-5/1e-6 on 100 | NOT run |
| goldens / dsed bit-identity 36/36 default + shadow | baseline recorded, check NOT run |
| scoreboard exact-engine row 81/77/70 | default cells RUN on e88a06dc's build: dsedall tip vs base (075feac9) — 243 same, 0 moved (81 circuits × dsed / dsed+next / no-dt rule off: times, states, events, reinit hashes); so the row cannot have moved. Scoreboard itself not re-rendered |
| samples before/after, ladder500 | NOT run |
| instructions (rule on, five circuits; rule off ±0.5 %) | NOT run |
| pulsim_core_dsed_tests (Eigen 5) | NOT built |
| exact test files, `-m reference`, full pytest | NOT run (only test_release_findings_f4.py: 3 passed) |
| ruff | run on the two harness files: clean |
| openspec validate --strict | NOT run |
| rule vs exact flow on 100 (075feac9, old harness) | run: 99 comparable, 98 ≤ 1, ll09 = harness artifact |

### Next steps (successor)

1. DONE on e88a06dc: `dsedall` 243 same / 0 moved (the sine certificate and the sharper bound move
   no default cell on the scoreboard set). Still to run: the goldens / bit-identity check
   (`BASE=s5base zsh .scratch/bitid5.sh check`).
2. Build + run `pulsim_core_dsed_tests` (tree `.scratch/build-ctest`): the new chord_row_bound case
   and the recovered-row case.
3. Cost (`.scratch/f3/cost10.sh <tip python> .scratch/base075`): the forward converters' certificate
   cost vs the old certificate; the ≤ 3 % target is vs pre-certificate (b08f4e4e), not built here.
4. The three-phase bridge: fix the mode guard's judge horizon (above), then the strict xfail of
   fix/core-exact-integrals-sources with the two truth instants.
5. Item 2's list above on the final tip; then item 4 only if context remains.

## Session 6 — acceptance of the output rule (verification only; logs in `.scratch/f7/`)

Tip at start dde945bd (extension = e88a06dc's C++, `build/python/pulsim` is a symlink to
`python/pulsim`, the same .so).

* **Rule off unchanged.** Goldens ALL IDENTICAL; dsed bit-identity 36/36 default and 36/36
  shadow (`bitid_goldens_ruleoff.log`). `dsedall` tip vs 075feac9: 243 same, 0 moved
  (`dsedall_cmp_vs_075.log`).
* **`pulsim_core_dsed_tests` (Eigen 5, `.scratch/build-ctest`)**: first build of the two new
  cases. The `chord_row_bound` case PASSES. The recovered-row case (c83fe230) FAILED at its
  precondition (`e.tier == 1`: 3 — `[0 1; -4e12 -2e5]` has a near-defective eigenbasis, the
  case had never been built). Fixed in the test (f8babadc: same ring, x2 scaled by 1/ω; the
  bound still binds, worst > 0.5 every step). After: 144 cases / 41 365 assertions pass
  (`ctest_dsed_run_eigen5.log`, `..._after_testfix.log`).
* **NEW, OPEN, blocks the full pytest — ten paralleled bridges under the rule.**
  `test_release_findings_f.py::test_f67_event_states_do_not_depend_on_the_output_grid` (ten
  bridges, 3 ms, output_rtol 1e-4/1e-5/1e-6) did not finish in 18 CPU-minutes (killed).
  `bridges10.py` (1e-4 s, output_rtol 1e-5): EVERY one of the 100 intervals capped
  (`n_output_capped` 100, 409 600 rule samples, 409 701 stored), 2.386e12 instructions —
  identical on base075 (075feac9, 2.386e12) and on the tip, so it is NOT e88a06dc; the test
  passed in session 3's full pytest on ffdf5007, so it came with session 4's recovered-row
  bound (`bridges10_1e-5.log`). Likely cause (not confirmed in source): a diode node between
  two OFF diodes reads i(L)/(2·g_off); the fast mode's F share 2|q z| is d-independent, so no
  chord certifies while it is above the node's tolerance. Not a few-line fix: reported.
* **e88a06dc STAYS**: no gate failed because of it (rule off 243/243 + 36/36, its C++ case
  passes, the two defects below are identical on base075 = 075feac9).
* **Rule vs exact flow, 100 units, fixed harness (`.scratch/f4/run5.sh f7`,
  `rule_vs_exact_flow_100.jsonl`)**: 99 comparable, 0 above 1, worst 0.99988
  (high_freq_gan_buck; then sync_buck_500khz 0.99983, stiff_rc_high_freq_switching 0.99983,
  high_q_rlc_ringdown 0.99982). Not comparable: three_phase_balanced_rl_load — and NOT only the
  dense flow: the RULE CELL ITSELF spends its 2 000 000-sample budget by 4.5295 ms
  (`output_budget_t`, 2 100 001 stored; 8 038 on the pre-reviewer tip of session 4, 5 153
  rule-free). It came with 075feac9's per-row rounding floor (after_t0.jsonl, 075feac9: same
  2 100 001). To be comparable the floating balanced neutral's row needs a floor that
  certifies a row that is 0 up to rounding (the old 16·u·κ floor did, at 8 038 samples) —
  the reviewer of session 4 found that floor too loose for ll06/ll09; a floor for rows whose
  exact value is constant is the open design question.
* **Property test, 100 units × 200 random grid instants (`property_100x200.jsonl`)**: 95
  comparable (5 not: event counts differ against the rule-off dt grid — cl_boost_pi,
  half_bridge_transformer_dc, boost_pfc_open_loop, flyback_converter, forward_converter).
  94 ≤ 0.9976. **ONE GENUINE VIOLATION: diode_clamp_overvoltage V(out) 9.64 × at 2.5162 ms**
  (`clamp_probe.log`): the rule stores 11 samples; the chord [1.02 ms, 2.533 ms] reads
  5.0004999 → 5.0000125 V straight across the pulse's fall corner at 2.52 ms, where V(out) is
  still 5.0005 V; at 2.5162 ms the chord reads 5.0000179 V (error 4.8e-4, tol 5.0e-5). The
  dense flow (×0.05) stores the SAME 11 samples, so `rule_ratio_flow` read 0 inside that chord:
  the exact-flow referee cannot see a defect the dense run shares. Same on base075. Likely
  path (by reading scheduler.hpp:600–765, not instrumented): the rule-off run steps every
  1 µs = the no-edge gate poll (dt_max/10); at 2.52 ms the poll and the corner coincide, so
  `at_corner = t_gate < t_sw` is false, the landing counts as `out_transparent`, the end is
  deferred, and the next deferral check bounds [ts, t1] in the post-corner piece. A fix
  (`out_transparent` also requires the landing not be a source corner — a tie test against
  the corner queue; rule-on only) is a few lines but needs an extension rebuild and a full
  re-acceptance: NOT done in this session.
* **`--truth` (mpmath, six circuits, fa4b5c71's tool verified byte-equal)**
  (`property_truth_mpmath_6.jsonl`): 6/6 comparable, worst 0.9406 vs the dt grid and
  0.9406 vs the truth (pwm_voltage_rc); the grid and the truth agree to 1e-6 of a tolerance.
* **Events gate** (`events_gate_output_rtol.log`): event times and states bit-equal across
  output_rtol 1e-4/1e-5/1e-6: 100 same, 0 diff, 0 skipped.
* **Stored samples, 6cb51c00 (before.jsonl) → tip** (`samples_before_after.log`): min 1.00,
  p25 1.00, median 1.00, p75 1.25, p90 4.84, max 407.5; 20 of 100 above 2×; total 879 412 →
  3 617 402. Ten largest: three_phase_balanced_rl_load 407.5× (5 153 → 2 100 001, the budget),
  boost_pfc_open_loop 14.2×, buck_dcm 9.6×, textbook_erickson_buck_ccm 6.9×,
  ll06_buck_vcswitch 6.2×, buck_boost 6.0×, two_phase_interleaved_boost 5.4×,
  buck_hard_switching_reference 4.9×, textbook_erickson_boost_ccm 4.9×, boost_classic 4.9×.
* **Instructions per run, rule OFF, e88a06dc's build vs base075** (`.scratch/f3/cost10.sh`, min
  of six, minus an import-only run; `cost_ruleoff_instructions_vs_base075.log`): buck_ccm
  +1.80 %, bridge3 +5.08 %, active_clamp_forward +4.26 %, forward_converter +2.28 %,
  boost_pfc_open_loop +7.65 % — the 0.5 % gate FAILS on all five, and the only C++ difference
  between the two builds is e88a06dc (no default number moved: the extra work is the sharper
  certificate / the sine modes now scanned). base075 already contains the certificate's own
  cost (+11–13 % on the two forward converters against the pre-certificate build); this
  measured only what came after it.
* **DECISION: e88a06dc REVERTED (7d7f3bf6)** — the gate failed because of it and the fix is
  not a few lines. core/ = 075feac9's + the test fix f8babadc; the extension in
  `python/pulsim` is base075's .so (byte-identical; e88a06dc's saved in
  `.scratch/so_e88a06dc/`), so the rule-off cost of the tip equals base075's by construction.
  `pulsim_core_dsed_tests` on the reverted tip (Eigen 5): 143 cases / 41 326 assertions pass
  (`ctest_dsed_run_eigen5_reverted.log`). The two defects above (ten bridges, diode clamp)
  were measured on base075 too, so they hold on the reverted tip. Re-run chain:
  `.scratch/f7/chain_reverted.sh` (logs `R_*`).
* **Re-run on the reverted tip (7d7f3bf6 / aaf0174c, .so = base075's)**: goldens ALL IDENTICAL,
  bit-identity 36/36 default + 36/36 shadow (`R_bitid_goldens_ruleoff.log`); dsedall vs
  075feac9 243 same / 0 moved; rule vs exact flow 99 comparable, 0 above 1, worst 0.99988,
  stored sample counts equal to e88a06dc's build on 100/100 (`R_after_f7r.jsonl`); property
  95/100 comparable, diode_clamp_overvoltage 9.64 × (same), all others ≤ 0.9976; `--truth`
  6/6, worst 0.9406 (truth 0.9406); events gate 100 same / 0 diff (`R_events_gate.log`).
* **ladder500_rc, rule on** (`R_ladder500.log`, `.scratch/f/fx3.py 500 2e-5`): finishes,
  3 093 stored (3 072 rule samples), n_output_capped 0, no warning; 2.923e11 vs 1.623e11
  instructions rule off (1.80 ×); wall 25.7 s vs 10.3 s (2.49 ×, loaded host).
* Not run: the rehearsal tool (auto+next / dsed:output_grid / dsed+next, analytic 22) — its
  cache was backed up to `.scratch/rehearsal_pre_f7` and `.scratch/f7/drop_rule_cells.py`
  drops the three rule configurations so `next_defaults_rehearsal.py run` re-measures them
  (render with `R.OUT` pointed into `.scratch`: `render()` writes the tracked
  NEXT_DEFAULTS_REHEARSAL.md).
* **Full pytest on the reverted tip (`R_pytest_full.log`, `--timeout=900`, still running at the
  hand-off, chain PID 43140; the `-m reference` run follows it in the same chain)**: 0 failures up
  to 80 %, then in test_release_findings_f.py TWO TIMEOUT FAILURES so far —
  `test_f67_event_states_do_not_depend_on_the_output_grid` and
  `test_fx1_ten_branch_form_bridges_run_under_the_rule[constant]` ([python] was running) — the
  ten-bridges cap above. The full pytest is NOT green.
* Also: ruff `python/` + the four harness tools clean (`ruff.log`); `openspec validate
  update-number-changing-release --strict` valid (`openspec_validate.log`); stubs not needed (no
  binding source changed since 6cb51c00).
* **Reviewer (fresh context, opus, read only)**: no under-reading of `exact_chord_step` /
  `chord_row_bound` found (numpy port, 20 000 random systems, worst 1.0001 = float cancellation
  in the test); sine share of the certificate correct for today's predicates. One plausible
  uncounted chord: `at_corner = t_gate < t_sw` false when a source corner ties with a gate
  edge / poll → `out_transparent` → the corner deferred (the diode-clamp violation above is
  this, with the no-edge poll). Harness blind spots: a NaN rule ratio passes as not-worse in the
  rehearsal; the flow referee is the engine against itself (a chord with no dense sample inside
  reads 0 — the diode clamp is exactly this); only declared observables are read; the first chord
  after a jump-free event is excluded; midpoint-only reading; the denominator rtol·rng + atol.
  Stated gaps: the 1e-3·peak range floor and y(t+h) in the range are not in the docstring; the
  docstring still says "every STATE".

### Session 6 verdict
NOT MERGEABLE. The promise holds against the exact flow on 99/99 comparable circuits and against
the rule-off grid on 94/95, but (1) diode_clamp_overvoltage violates it 9.64 × (a source corner
deferred; the flow referee cannot see it), (2) three_phase_balanced_rl_load spends the whole
2e6-sample budget (the promise is suspended after 4.53 ms), (3) ten paralleled bridges cap every
interval and three tests time out — (2) and (3) arrived with session 4's recovered-row bound /
075feac9's rounding floor. e88a06dc reverted (rule-off cost +1.8..+7.6 %).
* Hand-off correction: the re-run chain was terminated (SIGTERM, exit 143) during the full pytest
  at 80 % (after the two timeouts in test_release_findings_f.py; `[python]` unfinished). The
  full pytest did NOT complete and `-m reference` did NOT run on this tip.

## Session 7 — the three defects of session 6 (logs in `.scratch/f8/`, prefix `S7_`)

F7's background chain (PID 43140 + its pytest 48697) stopped by PID before the rebuild.

### D-1 — a source corner deferred (FIXED, d365c847)

* Instrumented (env-guarded print, removed): the rule-on run's landings near 2.52 ms —
  `t = 2.5199999999999966e-3, t_gate = t_sw = the poll, corner pending 2.52e-3, at_corner 0,
  out_defer 1, transparent 1` — the no-edge gate poll lands 3.4e-18 s SHORT of the pulse's fall
  corner (not on it): `at_corner = t_gate < t_sw` is false, the landing is "transparent", the
  end is deferred, and the next step treats the corner as passed. F7's reading confirmed (the
  tie is a near-tie, not an exact one).
* Fix: `corner_at_(t)` — the landing is never deferred when the corner pending in the queue is
  at its instant (`near_time`, finite only: `near_time(t, inf)` is true). Rule-on only (behind
  `out_defer`).
* Numbers (`.scratch/f8/clamp_check.py`, rule-on line vs a 1e-7 s rule-off grid): unfixed
  9.67 × at 2.52 ms, corner not stored; fixed: corner stored (10 samples), worst 0.000.
  **PR head 8c921c81 (so_8c921c81 + `git archive 8c921c81 python/pulsim`): YES, the same
  defect, 9.666 × at 2.52 ms, 9 samples, corner not stored.**
* Test: `test_f8_a_source_corner_is_always_a_stored_sample` (the four corners stored; the
  referee's ratio ≤ 1).

### The referee (benchmarks/tools/output_rule_referee.py, new)

* The flow is the same route with the rule off, run as a CHAIN of windows that tile the run at
  stored instants (`resume_from` the previous window's snapshot; one build, so a closed loop's
  Python controller continues); each window's dt = its shortest stored line / 6, so every line
  of non-zero length holds ≥ 5 reference instants wherever they fall. A line with none, a NaN,
  other events, a window without a snapshot, a suspended rule (cap / unbounded / budget), a
  floored column without a floor mapping: a harness FAILURE (`comparable` false, the reason).
  Denominator: `max(rtol·range, atol, floor of that column)` (range: the run's, ≥ 1e-3 of the
  peak; floor: `output_floor_by_column` mapped through the observable), no other slack. The
  first line after an event counts (both limits of a jump are in the trace; only reference
  instants within 1e-12 s of an event are skipped).
* On the clamp: unfixed engine 9.666 (`S7_referee_clamp_unfixed.jsonl`), fixed 3.6e-7
  (`S7_referee_clamp_fixed.jsonl`). Test `test_f8_the_referee_fails_a_line_it_cannot_read`.

### Session 7, as found by session 8 (F8 stopped before writing these)

* D-2 / D-3 are 6835fd6c (floor = 16·u·Σ|O||V||z|; rows certified by their floor counted):
  ten bridges 409 600 capped samples -> 926, three_phase_balanced_rl_load 2 100 001 -> 5 290;
  tests `test_f8_ten_bridges_rows_at_rounding_do_not_cut_the_samples`,
  `test_f8_a_balanced_neutral_does_not_spend_the_budget`; C++ case "a variation 1.2x the
  floor is refined, 0.25x is certified" (`S7_ctest_dsed_run_eigen5.log`: 144 cases pass).
* F8's chain logs (`.scratch/f8/S7_*`): bit-identity 36/36 + 36/36, dsedall 243 same, events
  gate 100 same, property 95 comparable worst 0.9976, truth 6/6, ladder500 27.8 s.
  **`S7_referee_100.jsonl` holds 38 of 100 units** (the process died at
  driven_lc_series_resonance, empty log) and one of them read **2904x**
  (diode_half_wave_rl_load): see session 8.

## Session 8 — F-X-7, the merge of the PR line, the referee (logs in `.scratch/f9/`, prefix `S8_`)

Tips: 7fc6ed97 (merge of origin/feat/core-integration 7188a9e1: only this file conflicted, both
sides kept; the dsed headers merged without conflict), b78d5a01 (F-X-7), bdabb1e9 (referee).

### F-X-7 — NOT the recovered-row bound: a jump drawn as a ramp (FIXED, b78d5a01)

* On the merged tip before the fix (`S8_fx7_merged_before.log`, bridge3_overlap branch form,
  bus = v(p) − v(n) by `np.interp` over `res.times` at 4001 instants, truth pwl + restart +
  split at dt 1e-7): rule off 0.010659 V; output_rtol 1e-5 **34.182 V at t = 0.12 ms**; 1e-7
  22.36 V. So this branch's bound on recovered rows did not fix it.
* Cause (`S8_fx7_twosided_before.log`): the sample stored at an event is its LEFT limit. The
  bus jumps 132.6096 -> 173.063 V at the commutation at 119.3798 us; with the rule on the next
  stored sample was 4 us later, so plain interpolation drew the jump as a 4 us ramp. With the
  rule off a fast sample 1e-10 s after the event hid it. Read two-sided
  (`with_right_limits`) the SAME rule-on run reads 0.010675 V (1e-5) / 0.010661 V (1e-7): the
  chord bound held on every line.
* Fix (rule on only, `output_jump_`): after an event that changed the mode or a source's piece
  the exact flow 2 ps later is a stored sample (evaluated from the step's start: no state,
  step or event moves); a step too short to hold it keeps its end. +1 sample per such event
  (the bridge: 3352 -> 3368).
* After (`S8_fx7_after.log`): 1e-5 **0.010674 V**, 1e-7 **0.01066 V**, rule off 0.010659 V.
  Every stored column against the rule-off run's own exact samples (`S8_fx7_columns.log`):
  worst 0.9971 of its tolerance at 1e-5, 0.44 at 1e-7 (floored columns 0.06). The bus is a
  difference of two columns: its bound is the sum of theirs (it read 1.11x ONE range's
  tolerance, which is not a violation).
* Test `test_fx7_a_jump_of_an_output_is_stored_on_both_sides` (the 4001-instant reading
  <= 3e-2 V at both tolerances; a sample 2 ps after every event; every stored column within
  its promise). Docstring and docs/solvers.md state the 2 ps line.
* The rule moves an event by picoseconds against the rule-off run (2.2e-12 s at 15 ms).

### The referee (bdabb1e9)

* diode_half_wave_rl_load **2904x** is not a chord: the rule's event projection settles the
  parasitic cluster at once (V(out) −0.31385 -> −3.1e-6 V at 10.0999 ms) while the rule-off
  reference lets that ~1 ns mode decay (−0.2994 V 47 ps after the event). Reference instants
  within `SETTLE` = 4e-5 of the run after an event are read apart: `ratio` 0.967,
  `ratio_incl_settle` 2904. **Open (design, not fixed): inside that window the rule-on result
  is the settled solution, not the exact one; the option's text says "removes only the
  parasitic cluster" but the promise's "exact solution" does not carry that exception.**
* Denominator: the run's range (as sessions 6–7). The promise's range is "so far, through
  the end of the exact step holding the line"; the stored trace does not show the steps, so
  the so-far reading through each line's end sample is reported beside it
  (`ratio_range_so_far`, stricter than the engine's: rc_step 3.51 at t = 1.4e-7 s against 0.957).
* Every stored column is read against its own floor when the dt reference stores the same
  columns (`ratio_stored_columns`); a dt run of several circuits stores one column fewer
  (6 vs 5 on the diode circuits): there only the declared observables are read (said in the row).
* One process per unit (`.scratch/f9/ref1.sh`): a crash is a recorded failure.

### The settle window — orchestrator's decision, and what is NOT done

Decision: (1) the promise is stated against the flow with the parasitic cluster projected at
the event — DONE in the option's docstring and docs/solvers.md (cut: time constants below
`kParasiticFraction` = 1e-6 of the run's resolved time scale, `exact_settle_length`);
(3) the referee reads the post-event window apart and reports the unprojected reading beside
it — DONE (bdabb1e9). **(2) NOT DONE, for a successor (needs C++ + binding + rebuild):** each
projected event writes a projection record — the instant, the largest projected amplitude per
recorded quantity (|y(before the projection) − y(after)| on the recovered vector, new mode),
the cluster's slowest time constant — into `diagnostics['output_grid']` (`n_projections`,
`projection_max`, the records under a cap). Where: `settle_and_project_` in
dsed/scheduler.hpp, rule-on path only (`output_rtol_ > 0`), next to the reinit records;
surfaced through dsed_bindings.cpp + `_dsed_dispatch.py` like `output_floor_by_column`; stubs
regenerated. Test to write: diode_half_wave_rl_load, the record at 10.0999172 ms carries
V(out) amplitude 0.31384 V (= −0.31384758 − (−3.138e-6), `.scratch/f9/hw.py`) and the
referee's `ratio_incl_settle` (2904) is explained by it. Also needed by scoreboard finding F-SB-3.


## Session 5 — branch `fix/core-release-f5-settle-decisions` (from 6cb51c00): two settle decisions

Base for every "before": the extension built at 6cb51c00 (`.scratch/f5/base_py`). Truth: the
circuits' own state equations solved per mode in mpmath (augmented [x; u; u'] matrix exponential,
40 digits; `python/tests/test_exact_settle_decisions.py`), cross-checked on the pulse circuits
against `exact_truth.py` (fa4b5c71, in `.scratch/f5/`, not committed): turn-off 59.2793718175008149 µs
(v0 = 0) and 58.9377110436850184 µs (v0 = −1), the two agree to 1e-24 s; V(o)(100 µs) 5.92758587765985 /
5.91795535047949 V. Circuit: pulse v0 → 10 V (5 µs, rise 1 µs, width 50 µs, fall 30 µs) → D1 (g_on 1e3,
g_off 1e-9) → 100 Ω → 0.1 µF ∥ 1 kΩ.

### (a) A source corner on an exact zero — FIXED

Cause (code read + the run): `fire_event_` dropped a same-mask gate edge before any decision
("Drop spurious same-mask gate edges silently"); a source corner is fired as one. The scan of the
next step (`locate_event_exact_`, `dio && ga == 0` branch) ladders the exact-zero start, finds it
leaving on its event side (sg = +1 for an OFF diode's DiodeOn) and `continue`s — "the mode
decision's case" — which no one runs. D1 OFF under 10 V; `DiodeModeInconsistentError` at 6 µs.

Rule: at a time event that keeps the mask, `undecided_at_time_event_` runs the settle cascade's own
first-round test per diode (in band → `zero_band_on_`, the sign of the first non-vanishing
derivative after t on the OFF-mode trajectory; past the threshold beyond the band → the value law
confirmed by `zero_band_on_`); if a diode would flip, the event runs the full decision
(`decide_modes_`, all three selectors). Which instants: the four callers that fire a time event
say what changed (`corner_event_`: 2 a source corner, 1 a tick, 0 neither). A corner checks every
diode; a tick (a new level) only diodes past their threshold beyond the band; a poll landing nothing
(it changes no input, and the scan located every crossing up to it; bridge3 lands one per step,
100k in 0.1 s, and the check there cost +1.8 % instructions). A first version re-decided in-band diodes
at every same-mask time event, and a poll landing in `bridge1_vf07` (bit-identity set) turned OFF
a leakage-level ON diode (D1 carrying D4's g_off current, 7e-10 A) 1 µs after its located partner's
turn-off; the mpmath truth (`exact_truth.py` patched for V_f in `.scratch/f5/exact_truth_vf.py`) keeps
one of the pair ON until 5.3663208617 ms (base 5.3660157816 ms, that version 5.10287 ms). Nothing
changes a rate at a poll, so continuity keeps the mode there.

Numbers: v0 = 0, all selectors, both forms: D1 ON at 5 µs exactly (truth 5 µs), turn-off
59.27937181726 µs (truth 59.2793718175008, 2.4e-16 s), V(o)(100 µs) 5.92758588 (truth
5.9275858777); before: `DiodeModeInconsistentError` on all three selectors and both forms.
Variants (all in the test file, against the truth): one ulp of 0 and of 10 V below the zero (ON at the
corner) and above it (the start decides ON, the corner is no event; base: the turn-off defect
below); a corner with ġ = 0 too (an RC prefilter: g = ġ = 0, g̈ > 0) — DECIDED by the next
derivative (the ladder's first significant rung), ON at 5 µs on all selectors (base: error); the
same drive from a PWL table current source into 1 Ω (base: error; tip: turn-off within 4e-17 s);
a ramp ending on a zero (resistive diode path; base: error; tip: OFF at 86 µs, current 0 after);
a sine's zero crossings on the diode's zeros (passed on base too, kept as a guard); a pulse whose
rise starts at t = 0 (start decision: ON from 0, turn-off 54.2793718 µs; checked, not a test).

### (b) A located turn-off kept ON — FIXED

Cause (trace of the earlier session + this one): at the located root the ON signal is zero only to
its significance bound B (~1e-15 V); the post-event decision's ladder (`zero_band_on_` → `ladder_sign_`)
already discounted the root's error, but only as a STATE error over t_tol (`loc_dx_`); the OFF
mode reads B amplified by 1 + R·g_on = 1e5 (+1.35e-10 V), which is a TIME error of the root times the
OFF signal's own rate (the source ramp, −2.77e5 V/s), and kept D1 ON.

Rule: `root_time_error_` — the root is known to t_tol + B/|ġ|, B = `signal_bound_` of the crossing
signal at the root, ġ its slope along the pre-event mode (the secant to the ladder's first
significant rung); `ladder_sign_(…, dt0)` treats the post-mode signal's own movement over that
window (sources included) as noise, max with the state term. No tuned constant: B, t_tol and the
ladder are the existing ones. It is read only where a decision in `zero_band_on_` would put the
FIRED diode back in its pre-event state (the crossing is the decision unless the contrary signal
is significant beyond the root's error), once per event, from the root state and pre-event mask
kept at the event (`loc_root_x_`, `loc_root_mask_`). Computed at every located event it cost
+1.9 % (bridge3) and +6.7 % (active_clamp_forward) instructions; an agreeing decision needs no
more noise. Trace on the pulse: t_tol 1e-18 s, B/|ġ| 2.5e-15 s, the OFF signal
then decides −1 at the 1e-15 s rung.

Numbers: v0 = −1, conductance form, all selectors: turn-on 5.090909086439 µs (truth
5.0909090864381347, < 1e-15 s), turn-off 58.93771104320 µs (truth 58.9377110436850, 4.9e-16 s),
V(o)(100 µs) 5.91795535 (truth 5.9179553505); before: `DiodeModeInconsistentError` (ON carrying
−2.64e-3 A) on all three selectors. Branch form was NOT affected (it passed on base): its ON signal
is the current itself, no 1e5 gain.

Selectors: (a) legacy, lcp, auto, both forms; (b) legacy, lcp, auto, conductance form only.

### Still open (found here, not fixed)

* t_end 1–4 ulps past a source corner (reviewer): the corner snaps onto t_end and never fires;
  the end sample reads 8.5e-15 V of forward bias as a law break. Same on base.
* The numeric path (`allow_implicit_segments`, `b_extra_fn`): `zero_band_on_` has no exact OFF-mode
  ladder and falls back to OFF in the band, so (a) is not decided there. Same on base.
* bridge1_vf07: the all-OFF recovery of a series pair splits 1.4 V with a 3.4e-8 V asymmetry
  (rounding of a g_off = 1e-9 / 0.01 S solve, above `signal_bound_`), so the wrong diode of the pair
  (D1, truth D4) keeps the leakage; the pair's turn-off is 3e-7 s early (5.3660157816 vs
  5.3663208617 ms). Pre-existing; unchanged here.
* Cost of (a) in the worst case: a diode resting at an exact zero at every corner runs the full ladder
  (~14 propagations) per corner (reviewer: 6 idle diodes × 800 corners, +35 % on that synthetic run).

### Must-not-move (final C++, `.scratch/f5/tip_py` vs `.scratch/f5/base_py` built at 6cb51c00)

* Goldens (golden_baseline compare): ALL IDENTICAL. dsed bit-identity 36/36 default and 36/36 shadow
  (the first, eager version moved `bridge1_vf07` dsed t/x: the poll re-decision above; reverted by
  scoping).
* Exact scoreboard cells (`.scratch/f5/dsedall.py`: 81 circuits × dsed / dsed+next / no-dt rule
  off, hashes of times, states, events, reinit records): 243 same, 0 moved — the exact-engine row
  stays 81 / 77 / 70.
* Event times and the states at events bit-equal across output_rtol 1e-4 / 1e-5 / 1e-6
  (`.scratch/f5/rtolinv.py`, no dt): 111 circuits invariant, 0 differ, 15 not loadable by v1_reader
  (base: the same 111 / 0 / 15).
* Moved vs base, rule ON only: boost_pfc_open_loop (all three output_rtol, invariant among
  themselves). Base fired a DiodeOff twice at 10.282917789558 ms; the re-decision changed the
  trajectory at rounding level onward. Truth (`grade.py`, scoreboard `exact`, rule 1e-5, no dt):
  7.36e-10 → 2.95e-12 (better). Its dsed / dsed+next / no-dt rule-off cells are identical
  (2.565e-7 / 2.195e-7 / 2.565e-7).
* Cost (instructions per run minus an import-only run, 10 runs per process, min of 6
  interleaved, same flags, `.scratch/f5/cost10.sh`), rule off: buck_ccm 0.998, bridge3 1.000,
  active_clamp_forward 0.992, forward_converter 0.974 (base spread), boost_pfc_open_loop 0.998.
* Loss port: no separate number compared; the full pytest (below) carries its tests.

### (c) The mode guard read the mode's next event as a violation — OPEN (fix tried, reverted; strict xfail)

Circuit: three-phase bridge, 325 V 50 Hz, 0.5 mH lines, D1..D6 (1e3 / 1e-9), 220 µF ∥ 40 Ω, 1 MΩ tie
(`_bridge_3ph`). Truth (`exact_truth.py`, 40 digits, 0..6 ms): D1 OFF 0.37193567186 ms, D5 OFF
1.49025821867 ms, D2 ON 5.00000025091293 ms with D6 still ON, D6 OFF 5.00135094309083 ms;
V(dp,dn)(6 ms) 624.647465499693 V.

Cause (coordinator's trace, confirmed): the settle at D2's turn-on keeps D6 ON; `guard_check_` →
`judge_` advances the mode 1e2 and 1e4 h_e (capped only by the next TIME event) and reads D6's own
turn-off 1.35 µs later as a violation (0.12 V); the LCP repair turns D6 OFF at once: 3 events.

Rule (`leaves_later_`, in `judge_`): a law broken at the horizon by a diode that is consistent once
the mode's parasitic transient is over (t + `exact_settle_length`, what the guard exists to see past)
and first moves INTO its admissible side from there (`ladder_sign_`, the decision's own lexicographic
test) is the mode's next event, which the scan locates; not a violation. Used by the guard, the
selector's pin arbitration and `pose_or_open_loops_` (all through `judge_`).

Numbers, legacy: 4 events; D2 ON 5.000000250912887 ms (truth …935, 4.8e-17 s), D6 OFF
5.001350943055478 ms (3.5e-14 s), D5 OFF 1.4902582186960 ms (2.7e-14 s); before: D6 merged into D2.
OPEN on the same circuit, base and tip alike: D1's turn-off 0.37193547048 ms is 2.0e-10 s early,
so V(dp,dn)(6 ms) is 4.8e-8 V off (624.6474655478).
lcp / auto: still merged (strict xfail): the LCP solve itself returns D6 OFF at D2's instant, no
ties (trace: `lcp_apply_` bits before `resettle_projected_`). With every upper diode OFF, D6
carries leakage-level current only; the selection does not resolve it. Needs: the LCP's
classification of a leakage-level port current (the tie band) read against the root's error, not
this guard.

Moved on the DEFAULT path by (c) (bit-identity set, dsed, default options): `bridge1_vf07` t/x
(34/36 arrays identical, default and shadow). Events 12 → 16; truth (`exact_truth_vf.py`, V_f
0.7) has 16: each conduction start is TWO turn-ons 2.2e-17..2.3e-12 s apart (e.g. 13.704043490641626
and …679 ms). Base merged them into one at 13.704043488297 ms (the guard's repair); tip has
13.704043488297 and 13.704043490643 ms (the second within 1.3e-15 s of the truth's). End states
identical to print precision. Goldens identical; the 243 scoreboard default cells identical (the
exact-engine row stays 81 / 77 / 70); rtol invariance 111 / 0 / 15 kept.
C++ `pulsim_core_dsed_tests` (Eigen 5) on the final header: 41265 assertions / 142 cases pass.

Rule-ON cells moved by the final tip (no dt, output_rtol 1e-4/1e-5/1e-6, each still invariant across
the three), scoreboard `exact` vs the mpmath truth at 1e-5, base → tip: forward_converter 1.02e-9 →
5.6e-12, boost_pfc_open_loop 7.36e-10 → 2.95e-12, flyback_converter 7.265e-7 → 7.262e-7. Their
dsed / dsed+next / no-dt rule-off cells are identical to base. Cost was measured before (c) (the
guard's extra ladder runs only on a judged violation); not re-measured on the final tip.

**(c) REVERTED (commit after 2bdc6598).** The exact-engine test files on the (c) tip: 4 failed —
`test_exact_guard_transient::test_the_forward_converter_rectifier_follows_the_transformer`
(0.6923 vs 0.7733), `test_exact_route::test_ten_paralleled_bridges_agree_with_pwl[None]` and
`test_release_findings_f::test_fx1_ten_branch_form_bridges_run_under_the_rule[constant|python]`
(`DiodeModeInconsistentError`, Db5a ON carrying −1.08e-6 A at 13.265 ms). `exact_settle_length`
is shorter than the transients the guard exists to see (the forward converter's rectifier is
consistent past it and still wrong), so "consistent past the parasitic transient, then admissible
first" excuses real violations. The bridge1_vf07 / rule-on improvements listed above came with
(c) and are reverted with it. What a fix needs: the guard's look-ahead bounded by the mode's next
LOCATED crossing (run the scan of the settled mode from the event up to 1e4·h_e and judge only up to
its first root), not by a transient length; and for lcp/auto the LCP's leakage-level port current
classification. Strict xfail `test_the_overlap_of_d2_and_d6_is_kept[legacy|lcp|auto]`.
The final tip's must-not-move numbers are those measured BEFORE (c) (the compares on 5613abbe, the
poll scoping): goldens identical, bit-identity 36/36 + 36/36, 243 cells identical, rtol 111/0/15,
rule-on boost_pfc 7.36e-10 → 2.95e-12; this revert restores that C++ exactly.
Verified on 738acf8f: full pytest 6246 passed / 110 skipped / 12 xfailed; `-m reference` 383 passed;
C++ dsed tests (Eigen 5) green; ruff check clean. Not run: Eigen 3.4.0, g++, full ctest, a second
reviewer on 79330ab7..5613abbe (scoping + lazy root error came after the review).

## Session 9 — the projection record, F-Y-3, the acceptance (logs in `.scratch/f10/`, prefix `S9_`)

* Merge of origin/feat/core-integration 527e644d (sensitivities through events): cbe1430f,
  `--no-ff`, no conflict (no dsed header touched by it; `_pulsim.pyi` did not conflict).
* Truth for the projection record (`.scratch/f10/hw_truth.py`, mpmath `exact_truth.solve`, no
  pulsim; `S9_hw_truth.log`): diode_half_wave_rl_load turns D1 off at 10.0999171950244 ms;
  V(out) just after it −0.313847548 V, then −0.28398 V at +0.1 ns, −0.11546 V at +1 ns,
  −2.05e-5 V at +10 ns, −6.27970e-6 V at +20 ns and −6.27968e-6 V at +40 ns: one ~1.0 ns mode,
  projected amplitude 0.31384127 V (the predecessor's 0.31384 confirmed; its settled value
  −3.1e-6 V is the engine's, the truth's is −6.28e-6 V).
* The other tool's dsed commits (origin/gemini/wip-2026-11-24, read only, not taken):
  4cbab009 replaced `exact_chord_step`'s bound by a per-mode cap for decaying modes
  (|V_jk z̈_k| / |λ_k|², ×2 for a complex pair: a stiff mode with a tiny residual no longer
  collapses the chord to rounding) plus a "peeled" cancellation set, and stopped a no-edge gate
  poll from resetting the per-interval sample cap; 0c9172e0 reverted the scheduler half only.

### Step 1 — the projection record (rule on only)

* `project_` (scheduler.hpp), under the rule: the recovered vector before and after
  `project_slow` in the settled mode; `note_projection_` keeps one record per instant (a settle
  round restarted from the held state replaces it; a round that re-projects the previous round's
  result adds its amplitude, an upper bound over the chain), the cluster's slowest time constant
  from `fast_cluster_tau` (builder_adapter.hpp: 1/min(−Re λ) over Re λ < −threshold). Committed
  when a projection at another instant comes, or read pending at the end.
  `diagnostics['output_grid']`: `n_projections`, `projection_max_by_column`, `projections` (first
  1000: `t`, `amplitude`, `column`, `tau`). Rule off: nothing computed (one `if` on `output_rtol_`).
* diode_half_wave_rl_load at output_rtol 1e-5: the record at 10.099917195024 ms, largest
  amplitude 0.3138476 V on the column of V(n_l) (the R–L midpoint), V(out)'s column
  0.31384 V (truth 0.31384127 V, within 2e-4), tau 9.9999e-10 s (truth: the ~1.0 ns mode).
  Test `test_f9_every_projection_of_the_parasitic_cluster_is_recorded`.

### Step 2 — F-Y-3, a window's min / max inside an exact step (rule on only)

* `SegmentMode::extremum` (segment_integrals.hpp): a degree-1 probe in the modal form,
  y(s) = Re Σ α_i e^{μ_i s}; past the constant and terms below 64u·Σ|α|, the stationary points
  are closed form for one real mode (none), two real modes (one root) or one conjugate pair
  (s_k = (π/2 − φ + kπ)/ω, first and last two). Else the step ends stay and the piece is COUNTED
  (`recorder_feed['closed_form']['n_extrema_ends']`; `n_extrema_closed` the others). The recorder
  asks the source (`SegmentSource::extrema`, default: none) after a closed piece; the exact
  engine's source answers only when the run has the rule on (`set_extrema`), so rule off is
  unchanged.
* Series RLC from rest (10 V, 2 Ω, 1 mH, 1 µF; `S9_fy3_rlc.log`): rule on max 19.053844735147337 V
  at 9.9395592885069e-5 s against the closed form 19.053844735147347 at 9.9395592885069e-5
  (−5.6e-16 relative); rule off (step ends) 19.053136 (−3.7e-5). Test
  `test_fy3_a_window_max_inside_one_exact_step_is_the_trajectory_peak`.
* The release's buck (`S9_fy3_buck.log`, test_tstart_time_base `_buck`, window [0.5, 0.9] ms):
  output_rtol 1e-5 **max 19.897853 / min 7.034291** (320 pieces, all closed); rule off
  19.836086 / 7.045884 (= the release's reading, unchanged); trbdf2 rtol 1e-10 19.897835 /
  7.034297 (its own step ends, so below the trajectory's max).

### Session 9 reviewer (fresh context, opus medium, read only; diff 527e644d..306efc60) — fixed / left

* FIXED — rule-off diagnostics keys: `recorder_feed['closed_form']` gained `n_extrema_closed` /
  `n_extrema_ends` (0) on every exact run with recorders; now emitted only under the rule (test:
  the rule-off run has no such key).
* FIXED — a projection under the rule on a step with no exact propagator (rk45 mode) was not
  recorded: every rule-on `project_slow` now records. Each event's decision commits the
  previous record (`decide_modes_`), so a restart inside one settle replaces only its own
  event's record, and two events at one instant keep two.
* FIXED — a projection row that is not a stored column reads `column: None` (was a negative
  index).
* FIXED (referee) — the settle window was 4e-5 of the run span after EVERY event (a 100 kHz,
  0.25 s run would read nothing and pass with ratio 0): now one window per projected event,
  [t, t + 40·tau] from the projection record; events without a record have none; past the
  record's cap (1000) the old window for the later events, flagged `settle_fallback`. Reading
  no instant raises (no vacuous pass). The tolerance's range is the rule run's own (projected)
  trace, not widened by the unprojected reference's parasitic excursion.
* LEFT (stated) — the verdict uses the run's range, not the range so far through the end of the
  exact step holding the line (the promise's); `ratio_range_so_far` (stricter: through the
  line's end sample) stays reported beside it. An exact judge needs the engine's per-line
  tolerance exported: not done.
* LEFT — `fast_cluster_tau` runs a dense eigensolve per projection when the mode has sine
  sources (rule-on cost only).
* Checked clean by the reviewer: the interior-extremum formula (2000 random damped pairs with a
  constant and an offset against a 1e5-point grid: 0 missed), the two-real-mode root, the
  dropped small terms' bound, every other rule-off path (no bit can change).
  Verdict: it attacked the chord collapse on stiff decaying modes (the ten-bridges / ladder
  symptom). This branch already covers the measured cases by D-2 / D-3 (a row at its rounding
  floor certified constant, counted: ten bridges 409 600 capped samples → 926, balanced neutral
  2 100 001 → 5 290) and F-X-3 (atol as an absolute floor); the per-mode cap is a valid,
  tighter bound (|e^{λτ} − chord| ≤ 1 real / 2 complex for Re λ ≤ 0) worth a later look as a
  sample-count item only — it moves every rule-on sample count and was written on a tree 377
  commits old, without the projection or floor changes; the scheduler half was wrong (the
  per-target cap reset) and its author reverted it. Nothing taken.
* FIXED (referee, found here) — S8's referee run (`.scratch/f9/S8_referee_100.jsonl`, never
  written up) read **1e5 ×** on pwm_voltage_rc / pwm_voltage_rl_filter, on the PWM source's
  stored column x[0] at 40 / 50 µs: the rule run stores the corner's two limits at
  3.9999999999999976e-5 (0 V) and 3.9999999999999983e-5 (10 V), the reference's own sample
  at 4e-5 is its LEFT limit (0 V), read inside the rule's post-jump line. A source corner is no
  event, so EVENT_TOL never excluded it. Now reference instants within 1e-12 s of a jump of
  either trace are not read (fa03bfcf): pwm_voltage_rc 0.9508 (V(out)), diode_half_wave_rl_load
  0.9674 with 3 per-projection settle windows (`ratio_incl_settle` 2904 kept beside it).

### Acceptance (tip fa03bfcf; .so built from it, LTO OFF; baseline = a build of 527e644d in `.scratch/build-base527`, .so saved in `.scratch/so_527e644d/`)

* (a) rule OFF against the PR head (`.scratch/f10/base.sh`, `S9_ruleoff_compare.log`): goldens
  ALL IDENTICAL; exact bit-identity 36/36 default + 36/36 shadow.
  The exact dumps of every `python/tests/test_exact_*.py` (the same files on both sides: `git
  diff 527e644d HEAD` is empty for them; `S9_ruleoff_exdump.log`): 312 passed both sides;
  **rule-off calls 0 diffs**; 420/441 calls identical, the 21 others are rule-on runs (stored
  samples: D-1's corners, F-X-7's 2 ps samples), 0 call-count mismatches.

### Step 3 — test_boost_dcm_recovery_voltage_is_v_out_minus_v_in[auto] on this tip (`S9_boost_dcm.log`)

* The default route (`auto`) runs trbdf2 on the PR line: 100 turn-offs after 2 ms, worst
  0.20 % (V_R 33.6306 against 33.6989 V): PASSES. engine='exact' rule off: worst 2.9e-5: passes.
* engine='exact', output_rtol 1e-5 (and 1e-7: identical): 99 of 100 turn-offs off by > 1 %,
  worst **16.8 % (V_R 27.8714 against 33.4977 V at 2.98679 ms)**. NOT the engine: the test calls
  the private reader `_diode_turn_offs(tr, t)` without `interpolable=True`, so it pairs the
  rule's samples into step midpoints across the event's left limit and its 2 ps sample; with
  `interpolable=True` — what `device_loss_summary` passes on a rule run (losses.py:2499,
  `_rule_held`) — the same run reads worst 5.8e-6, 0 above 1 %. The release's parked reading
  (39.49 against 34.30 V under the default rule) is this test-side call; the fix belongs in the
  test (pass `interpolable` from the run's `output_grid` as the summary does), not in the engine.
* (b) `pulsim_core_dsed_tests` (`S9_ctest_dsed.log`): Eigen 5 (`.scratch/build-ctest`) **152 test
  cases / 41 469 assertions pass**; Eigen 3.4.0 (`.scratch/build-e34ct`, new, 200 eigen-3.4.0
  include refs in its build.ninja) **152 / 41 469 pass**.
* The rest of the acceptance runs as ONE sequential chain, `.scratch/f10/chain_all.sh` (started
  09:38, PID 45028; progress lines in `S9_chain.log`, one per item): (g) cost → `S9_cost.log`;
  (h) exact-engine files + test_release_findings_f4 → `S9_pytest_exact.log`, ONE full pytest
  (`--timeout=900 --durations=40`, the ten-bridges walls are in its durations) →
  `S9_pytest_full.log`, ruff → `S9_ruff.log`; (c)(e)(f) `chain_s9.sh` → `S9_referee_100.jsonl`
  (summarise with `refsum.py`), `S9_samples.jsonl`, `S9_property_100x200.jsonl`,
  `S9_property_truth_6.jsonl`, `S9_events_gate.log`, `S9_ladder500.log`; (d) the rehearsal
  (rule cells dropped from `.scratch/rehearsal`, backup in `.scratch/f10/rehearsal_backup_pre_s9`)
  → `S9_rehearsal_run.log`, rendered into `S9_NEXT_DEFAULTS_REHEARSAL.md` (not the tracked file).
* openspec `validate --all --strict`: 79 passed, 0 failed (`S9_openspec.log`); stubs `--check`: up
  to date (`S9_stubs.log`).
* (g) rule-off cost, tip vs a build of 527e644d, both LTO OFF (`S9_cost.log`). Four standard
  cases, instructions per accepted step, min of 3 rounds: buck_pwl 3025.8 vs 3023.6 (+0.07 %),
  vsi_pwl 4687.5 vs 4696.4 (−0.19 %), buck_trbdf2 11113.7 vs 11121.0 (−0.07 %), bridge_trbdf2
  12878.9 vs 12864.6 (+0.11 %): within 0.5 %. Five exact-engine cases (instructions per run, min
  of 6 interleaved): buck_ccm −4.49 %, bridge3 −0.04 %, **active_clamp_forward +0.89 %**,
  forward_converter −6.67 %, boost_pfc_open_loop −2.29 %. The 0.5 % gate FAILS on
  active_clamp_forward. Not attributed: this session's rule-off work is one bool test per event
  decision and per projection (and a virtual call per closed recorder piece, no recorders here);
  the spread (−6.7 % … +0.9 %) on identical results looks like code layout of the branch's
  larger scheduler (sessions 4–8 included: the base is the PR head, not this session's start).
  To attribute it: build fd1765ab+merge (cbe1430f) and compare the three.
* (h) the exact-engine test files + test_release_findings_f4 (`S9_pytest_exact.log`): **327 passed,
  3 xfailed**.

* (h) full pytest on 6fca24a4 (`S9_pytest_full.log`, 618 s): **3 failed, 7215 passed, 137
  skipped, 42 xfailed**; ruff: all checks passed. Ten-bridges walls: f67 85.3 s, fx1 [constant]
  56.9 s, [python] 56.5 s. The three failures, all rule-on, all pass on the PR head 527e644d
  (`S9_3fail_at_527.log`) and fail on 6fca24a4 (`S9_3fail_at_tip.log`):
  * `test_dcdc_engine_matches_orbit[buck_R10-dsed]` (output_rtol 1e-6): **a defect of F-X-7**
    (b78d5a01). Its 2 ps sample was pushed before the step's chord samples, which still started
    at the event, so the rule stored the event's fast mode at +0.37, +0.74 … ps AFTER the +2 ps
    sample: t[1823] < t[1822] by 1.63e-12 s, once per period (4 of 8081 samples), and
    time_average refused the trace. Fixed in **9d6dc0c4**: the chords start at the stored 2 ps
    sample (still evaluated from the step's start). All 25 orbit cells pass.
  * `test_f54_exact_two_limit_trace_closes_the_bridge_charge`,
    `test_f62_exact_time_average_of_the_two_limit_trace_is_the_summary`: their control assertion
    ("the left limits alone miss the turn-on jump") is what F-X-7 changed on purpose: the flow
    2 ps past each turn-on is stored (12 of 12 bridge events, `f54why.py`), so the stored trace
    closes the charge to 3.5e-6 (was −1.2e-3) and reads the summary's i_avg within 4.3e-9 (was
    −0.21 %). The controls now assert that (9d6dc0c4).
  * After 9d6dc0c4: exact files + f4 + release_findings_b + textbook **389 passed, 6 xfailed**
    (`S9_pytest_fx7order.log`); ruff clean. The full pytest was NOT re-run on 9d6dc0c4.
* The chain was stopped by PID at the referee's 4th unit (its readings were of the pre-fix
  .so) and restarted on 9d6dc0c4 as `chain_rest.sh` (dsed ctest on Eigen 5, then (c)(e)(f),
  then (d)); progress in `S9_chain.log` after the "REST start" line.

* `chain_rest.sh` on 9d6dc0c4: dsed ctest Eigen 5 **152 / 41 469 pass**. Then two more
  defects found by its readings:
  * **SIGSEGV under the rule on stateless circuits** (session 9's projection record):
    `fast_cluster_tau` ran an EigenSolver of a 0×0 A (back_to_back_diodes_sine: a sine, R, two
    diodes; also transformer_step_up, driven_lc_series_resonance in the referee; the rehearsal's
    "worker exit -11" cells: back_to_back_diodes_sine, transformer_step_up, analytic halfwave_r,
    ideal_transformer, vcvs_gain2). Samples / property / events gate stopped at unit 4 of 100.
    Fixed in **3058a87c** (no state: no cluster), test
    `test_f9_a_stateless_circuit_runs_under_the_rule`; f4 file 11 passed.
  * **The referee refused 45 of 100 units** on 9d6dc0c4 (6 on the session 8 run,
    `.scratch/f9/S8_referee_100.jsonl`): fa03bfcf counted every pair ≤ 2 ps apart of either trace
    as a jump and skipped instants within 1 ps of it, so a reference grid finer than 2 ps read
    nothing ("N line(s) without a reference instant"). A jump is now a pair within 64 ulps
    (**70526766**): rlc_step 0.9646, pwm_voltage_rc 0.9508, pwm_voltage_rl_filter 0.9929,
    boost_classic 0.9997, stiff_two_timescale_rc 0.9454 (`.scratch/f10/refdbg/`). Of the 55
    comparable units on the refused run: above 1: none; incl. settle: diode_half_wave_rl_load
    2904 (the projected cluster, inside its window).
* `chain_rest2.sh` (3058a87c) and `chain_rest3.sh` (the referee, 3eef29a5 = 70526766 + docs):
  * (c) referee, 100 units (`S9_referee_100b.jsonl`): **89 comparable, above 1: none**, worst
    0.99991 (high_q_rlc_ringdown x[3]); incl. settle: diode_half_wave_rl_load 2904 (the projected
    1 ns cluster, inside its window); settle_fallback 0. Not comparable (11): events differ 4
    (half_bridge_transformer_dc, active_clamp_forward, flyback_converter, forward_converter),
    DiodeModeInconsistentError boost_pfc_open_loop, no floor mapping three_phase_balanced_rl_load,
    driven_lc_series_resonance (the referee process ended without a line; samples / property /
    `--truth` run it fine), and 4 with one or two ~2 ps lines left unread (ll02_rc_pulse_charge
    [2e-6, 2.000002e-6], ll03_rl_step, ll04_rlc_underdamped, snubber_rc): F-X-7's line after a
    source corner, its reference instants inside EVENT_TOL of the corner — a referee limit, open.
  * property 100×200 (`S9_property_100x200.jsonl`): **95 comparable, worst 0.99758, above 1:
    none**; 5 events differ (cl_boost_pi, half_bridge_transformer_dc, boost_pfc_open_loop,
    flyback_converter, forward_converter).
  * `--truth` 6/6 ran: rule ratio / truth ratio rlc_series_resonant 0.49558 / 0.49558,
    driven_lc 0.49862 / 0.49862, rl_natural 0.91659 / 0.91659, half_wave_rectifier_sine
    0.91264 / 0.91264, pwm_voltage_rc 0.94091 / 0.94090, chopper_with_rc_snubber 4.9e-4 / 4.9e-4.
  * (e) events gate: **same 100, diff 0, skipped 0**.
  * (f) samples: 100/100 ran, 1 554 426 in all; three_phase_balanced_rl_load 5 290 (609 rule
    samples, 1 projection); driven_lc 80 743. ladder500 (on 9d6dc0c4, untouched since): rule off 21
    samples 24.2 s 139 MB; rule 1e-5 3 093 samples 80.9 s 166 MB, 0 capped / unbounded / floored.
  * (d) rehearsal (`S9_NEXT_DEFAULTS_REHEARSAL.md`): no worker crash left. Engine-class rows:
    three auto+next **timeouts (300 s)** on the routed engine's profile (driven_lc_series_resonance,
    high_q_rlc_ringdown, three_phase_diode_rectifier — not the exact engine); **analytic
    hard_switched_rc on dsed:output_grid: disagree, err_rms 0.0676** (tracked rehearsal 2.5e-10).
    Cause, measured (`hsrc.py`): the closing switch gives the capacitor a 1 ps mode, which the
    rule's event projection settles at once; F-X-7's sample 2 ps past each closing stores the
    PROJECTED flow (0.0100 V) where the unprojected closed form reads 13.54 V, and err_rms's
    trapezoid weights that one sample over half the next 5 µs line. The promise is against the
    projected flow (decision in force) and the projection is recorded, so no engine change here;
    but the rehearsal gate now reads this cell as wrong — for the orchestrator.

### Session 9 — what is left (for the successor, if this session stops before the chain ends)

1. Read the chain's results (`.scratch/f10/S9_chain.log`, one line per item): the full pytest
   (`S9_pytest_full.log`; the ten-bridges walls in its `--durations`), ruff, the referee
   (`refsum.py S9_referee_100.jsonl`: none above 1 outside the per-projection windows, every
   unit not comparable named), property 200 instants, `--truth` 6/6, events gate 100 same,
   samples (three_phase_balanced_rl_load near 8 038?), ladder500 wall, the rehearsal render.
2. The cost gate on active_clamp_forward (+0.89 %): attribute it (see (g)) or decide.
3. test_diode_state_reconstruction's rule-on reading (step 3): pass `interpolable` in the test
   when the release flips the rule on (release branch work, not this line).
4. Clean up: `.scratch/build-e34ct` (Eigen 3.4.0 test tree), `.scratch/base527` +
   `.scratch/build-base527` (keep `.scratch/so_527e644d/` for cost reruns), `.scratch/f10/base`
   (537 MB of baseline dumps) once the orchestrator has read the logs.
