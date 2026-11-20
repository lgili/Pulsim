# F-S-4 — the exact engine dropped an inductor's initial current at a rectifier start

Branch `fix/core-exact-initial-inductor-current` (from the PR head 77ce4a28).
Test: `python/tests/test_release_findings_fs4.py`.

## The case

11 elements (from the PSFB's power-transfer start, `test_soft_switching_validation.py`
on `feat/core-soft-switching-validation` fa4b5c71): 400 V -> L_lk 10 uH (0 A) -> ideal
4:1 primary (p, b); b to ground through S4 = 1 mOhm || D4 (gnd -> b) || C4 = 200 pF;
the secondary into a full bridge D5..D8 (g_on 1000 S, g_off 1 nS) feeding L_o 20 uH
(24 A) to ground. The given state is consistent: the four bridge diodes share L_o
(12 A each) and short the secondary; L_lk ramps from 0 under the whole bus.

Truths (independent of pulsim):

* `exact_truth.py` (fa4b5c71's version, mpmath dps 30, run from `.scratch`, not committed):
  first mode D4 OFF, D5..D8 ON; D6 turns off at 150.018002831 ns;
  i_Llk(150 ns) = 5.99923506514082 A, i_Llk(200 ns) = 6.06050914046011 A,
  i_Lo(0) = 24 A. The three variants (no C4, no D4, a diode b -> 400 V) agree
  with it to 4e-12 A.
* closed form of the first interval: the bridge shorts the secondary through
  R_s = 1 mOhm (L_o's current splits symmetrically and puts no voltage across the
  winding), reflected R_s / n^2 = 16 mOhm, plus S4: R = 17 mOhm,
  i_Llk(t) = (V/R)(1 - exp(-R t / L_lk)); at 150 ns 5.999235 A, within 1e-7 of the
  tool.

## Before (PR head, extension so_b9e46778), `engine='dsed'`

| selector | full | no C4 | no D4 | + diode b -> rail |
|---|---|---|---|---|
| legacy (today's default) | i_Lo(0) = -2.8e-9 A, i_Llk(200 ns) = 0.2424 A, silent | right | right | silent, same numbers |
| `'lcp'` | silent, same | silent, same | silent, same | silent, same |
| `'auto'` (the release default) | silent, same | silent, same | silent, same | silent, same |

`output_rtol=1e-5` changes nothing. (The finding's loud `DiodeModeInconsistentError`
for the rail-diode variant does not reproduce on this head: it is silent too.)

## Cause (trace, temporary fprintf at the t = 0 decision; state x = [v_C4, i_Llk, i_Lo])

```
auto:   lcp in    bits=00000 x=0 0 24
        sel status=Solution on=00000 ties=5 widenings=2 pins=0
        u = -6e-15, 1.19e10, 1.19e10, 1.19e10, 1.19e10 V      (port voltages, all-OFF companion)
        i = 0, 12, 12, 11.9997, 12.0003 A   w = 7e-8, 0, 0, 0, 0 V
        lcp projected bits=00000 x=2.5e-11 2.5e-08 0          <- the 24 A is gone, no record
        lcp resettled bits=01001 x=-1.9e-10 4.8e-08 -2.8e-09
legacy: settle -> 11111 x=... 24 (the settle keeps it), guard: the judge rejects D4 ON,
        the guard's LCP repair decides 00000 as above -> 01001, i_Lo = -2.8e-9
```

The LCP's solution is the right one (four bridge ports conducting 12 A each, gaps 0).
It is CLASSIFIED wrong: every port is a "tie" because the tie band is
tol_i = 1e-9 * iscale + tol_w * c_max / h_e with iscale = vscale * G and
vscale = 1.19e10 V. That vscale is `port_scale(t, x_held)` read under the mask `rec`
— at t = 0 the initial diode bits, all OFF: a GUESS, never integrated. Its recovery
map pushes L_o's 24 A through the open bridge's g_off and reads 1.2e10 V. With
vscale 1.2e10 the band is 1.2e4 A (plus 2e3 A from C4 at h_e = 2e-14 s): 12 A is
"rounding", the ties go OFF (the judge finds all-OFF "consistent": its trajectory
dumps the 24 A into g_off within ~1e-14 s), and the slow-manifold projection of the
all-OFF mode (`project_slow`, which writes no reinit record) removes the current.
After every later event `rec` is the mask the state was integrated in, so its
recovery is consistent: the defect is the start's.

Second defect, exposed once the mode is right (lcp / auto with C4 and D4): the LCP
decides 01111 (D4 OFF, w = +6.8e-6 V); `resettle_projected_` then reads the PROJECTED
state, v_b = -8e-9 V, and turns D4 ON (confirmed by `zero_band_on_` from that same
projected state), and D4 conducts -4 mA (`DiodeModeInconsistentError` at 0.2 ns; the
rail-diode variant chatters). -8e-9 V is the slow manifold's asymptote: with
tau = R C = 0.2 ps under di/dt = 40 A/us the manifold sits at R (i - tau i'), while
the trajectory from the given state is v_b(t) = R i' (t - tau (1 - e^(-t/tau))) >= 0
for all t: D4 is never forward-biased (the truth's first mode has D4 OFF).

Same defect as the flyback "v_out 0 V on the legacy selector" (OVERNIGHT_PLAN 23:02)?
See "Selectors and the flyback" below.

## Fix

1. `NativeCircuitBuilderAdapter::port_scale_state(x)` (new, builder_adapter.hpp): the
   circuit scale of the state alone — capacitor voltages, inductor currents, sources,
   drops, conductances — not of the node voltages a guessed mask makes of it.
   `port_scale_full` takes the two floors (defaults unchanged).
2. `PEDSimulator::run_selector_` (scheduler.hpp): the start's decision
   (`start_decision_`, set around the t = 0 `decide_modes_` in `simulate()` only;
   a resume's decision reads an integrated mask and is unchanged) reads that scale.
   Shared by `lcp_apply_` (lcp / auto) and the guard's repair (legacy), so one place
   fixes all three selectors.
3. `PEDSimulator::resettle_projected_` (scheduler.hpp): at the START decision only, a
   flip found on the projected state is confirmed on the mode's trajectory from the
   held (given) state. Tried first at every event: 8 of 162 scoreboard cells moved
   from `ran` to `DiodeModeInconsistentError` (forward_converter, half_bridge_
   transformer_dc, boost_pfc_open_loop, active_clamp_forward, flyback_converter):
   after an event the projected state IS the jump's physics (the forward converter's
   collapsed transformer current, the function's own comment), so the general form
   was wrong and was reverted to start-only.
4. NOT done — the second half of the rule ("a projection that changes a state the user
   supplied is never silent: the reported jump or a refusal"). A refusal was written
   (a nonzero given state moved by > 1e-6 of itself at t = 0 with no record ->
   `consistent_start`) and REMOVED after review: the slow projection also removes
   fast modes through REAL conductances, and the refusal fired on consistent starts
   on the default path (10 V -> diode ON -> 1 nF given 8 V: the 1 ps inrush through
   g_on moves it to 10 V; 200 pF given 1 V across 1 mOhm with any diode in the
   circuit). Distinguishing a lost given value from a physical fast relaxation needs
   the jump REPORTED, not refused: a start-projection record with its energy, and the
   energy must count the sources' impulsive work (as exact_truth.py's impulsive loss
   does). That changes `reinit_records` / `on_impulse='error'` for every start from
   rest with a parasitic cluster (13 tests of the exact files move by 1e-8..1.6e-5),
   so it is a separate, listed change. Still silent on this tip: a GIVEN state that
   no mode can keep (1 A in an inductor whose only onward path is a reverse-biased
   diode starts at 0 A on every selector; trbdf2 shows the 1 A decaying through
   g_off). With the mode choice fixed, F-S-4's case itself no longer reaches it.

## Functions touched (for the merge with the output-rule branch)

* `core/include/pulsim/dsed/builder_adapter.hpp`: `port_scale_state` (new),
  `port_scale_full` (two defaulted floor parameters).
* `core/include/pulsim/dsed/scheduler.hpp`: `simulate()` (the t = 0 block around
  `decide_modes_`, one reset next to `settle_decisions_ = false`), `run_selector_`
  (the scale), `resettle_projected_` (the confirmation's state at the start), one
  member `start_decision_`.
  No change in the output rule, `exact_lti.hpp`, the stepping or the sampling.

## Selectors and the flyback

* legacy (today's default): failed with C4 + D4 and with the rail diode, through the
  guard's LCP repair (the same scale). Fixed by 1.
* `'lcp'` and `'auto'`: failed on all four variants. Fixed by 1 (mode) + 3 (D4).
* The flyback "v_out 0 V on the legacy selector" (group C's side observation,
  `_flyback(False, 1.0)` of test_saturable_transformer, 100 kHz, 1 ms) is a DIFFERENT
  defect: it happens at every gate-off, not at the start, and the start's scale is
  not involved. On this tip: legacy v_out(0.8-1 ms) = -4.3e-14 V (unchanged),
  'auto' 16.365091 V (truth 16.365).

## After (this tip)

All three selectors, all four variants, with and without `output_rtol=1e-5`:
i_Lo(0) = 24 A, i_Llk(200 ns) = 6.0605091 A (truth 6.06050914046 A, within 1e-6
relative; i_Llk(100 ns) within 1e-6 of the closed form). No warning, no refusal.

## Moved cases

None on the default path or the scoreboard's exact cells: byte compares against the PR
head's extension (so_b9e46778): golden ALL IDENTICAL, stateful 40/40, engines 35/35,
bit-identity 36/36 default + 36/36 shadow, and the exact engine on every scoreboard
circuit (configs `dsed` and `dsed+next`, the scoreboard's own run_cell path) 162/162
bit-identical — so the scoreboard's dsed row (81 / 77 / 70) is unchanged. Only the
opt-in selectors' t = 0 decision (and the default guard's t = 0 LCP repair) can read
the new scale; on circuits where `port_scale_state` < `port_scale` at a CONSISTENT
initial mask (a step-up transformer's secondary above every source, voltage-source
branch currents) the t = 0 tie bands are tighter — no case in these suites (review
note, reasoned, not shown).

## Review (fresh context, opus medium, read-only)

* HIGH, fixed by removal: the start-state refusal fired on consistent starts (see 4).
* MEDIUM, left: `resume_landing_(decide=true)` (a snapshot of another engine, "decided
  at t as at a start") reads `rec` from that engine's stamped mask without the state
  scale. That mask is the other engine's integrated mode at the instant, not a guess;
  not changed, not shown either way.
* LOW, left: at the start `zero_band_on_` reads `x_held` under the decided mask without
  `reinit_to_constraint` (a mode with dependent states judges an unconstrained state).
* The scale change is at the cause; truth constants and the closed form checked.

## Verified at the tip (macOS, Eigen 5, own extension build; conftest imports it)

* extension build; `pulsim_core_dsed_tests` 41265 assertions / 142 cases pass;
* `test_release_findings_fs4.py` 16 pass (failing on the PR head: 13 of them);
  exact-engine test files (`test_dsed*`, `test_exact*`, `test_release_findings*`,
  `test_mode_selection*`) 346 pass;
* ONE full pytest: 6187 passed, 110 skipped, 8 xfailed, 0 failed;
* byte compares vs the PR head's extension: listed under "Moved cases" (all identical);
* ruff clean. Stubs: no binding signature changed (not run).

NOT run: full ctest, Eigen 3.4.0, g++, the scoreboard's `run`/`render` (its dsed cells
are byte-identical instead), openspec strict, cost tables (no time-loop change: the
new code runs once per run).
