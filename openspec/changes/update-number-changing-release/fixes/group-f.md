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
  hold a root, otherwise it is split. Left open (new finding F-X-4).
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
