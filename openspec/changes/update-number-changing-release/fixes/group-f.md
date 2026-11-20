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
