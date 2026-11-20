# Exact engine vs trbdf2 / fixed engine on the buck window integrals (finding F-B2-1)

Suspicion (fixes/group-b2.md, "Numbers"): on the buck of
`test_recorder_device_readings._buck` the exact engine at `output_rtol=1e-10`
differs by 1.1e-4 to 2.0e-4 on ∫i(L), ∫v(out), ∫p(R) over [A, B] from
trbdf2 (rtol 1e-11) and from the fixed engine with the restart rule at dt/16,
which agree with each other to 3e-8; called "a modelling difference of the
exact route".

**Verdict: (c), an artefact of how the integral is taken — but not the
left-limit one. The exact engine's states are exact; its window integrals are
chords over its 10 us steps.** trbdf2 and the fixed engine are right.

Extension: `/Users/lgili/pulsim_wt/p3-devices/python/pulsim/_pulsim.cpython-313-darwin.so`
(branch `fix/core-release-b2-recorder-restart` at f0d3eaf9, as built).
Probes: `.scratch/exact_buck/` (`truth.py`, `vs_truth.py`, `integrals.py`,
`states.py`, `nearA.py`); `.scratch/ref44.py` is the report's own probe.

## Referee (independent of the three engines)

Two mpmath truths, no pulsim import (`.scratch/exact_buck/truth.py`):

1. `benchmarks/tools/exact_truth.py` from `origin/feat/core-soft-switching-validation`
   (fa4b5c71), the buck as a netlist (vcswitch driven by a 20 kHz, D = 0.4
   PWM source; diode g_on 1e3 / g_off 1e-9, V_th = V_f = 0; uic, zero IC).
2. By hand: x = [iL, vC], v_sw = (24 g_s − iL)/(g_s + g_d), x' = M x + c per
   (S, D) mode, x(t) = e^{Mt}(x0 + xp) − xp by eigendecomposition; diode
   OFF→ON iff v_sw ≤ 0, ON→OFF iff g_on·(−v_sw) ≤ 0 (bisection); window
   integrals by `mp.quad` on the closed form.

They agree to 18 digits on the integrals and 15 on the states. The one diode
turn-off (DCM at start-up) is at 147.784821853551 us in both and in the exact
engine's `event_times`. Same circuit everywhere: no diode-law, switch-form,
conductance or gate-timing difference (gate edges at k·50 us and k·50 + 20 us
on all engines).

## States at the window ends (a run that ENDS there; no integral, no interpolation)

Relative error vs the truth:

| engine | iL(A) | vC(A) | iL(B) | vC(B) |
|---|---|---|---|---|
| exact (default) | −1.1e-16 | +2.2e-16 | −6.1e-15 | −5.8e-15 |
| exact output_rtol 1e-10 | −3.2e-13 | −9.2e-13 | +1.4e-12 | +3.1e-12 |
| trbdf2 rtol 1e-11 | −4.4e-9 | +2.1e-10 | −4.5e-9 | −1.2e-9 |
| pwl rule dt/16 (interpolated) | −3.4e-7 | −1.1e-7 | −2.1e-7 | −2.8e-7 |

The truth sides with the exact engine to 14–15 digits.

## Window integrals over [A, B], relative error vs the truth (i(L), v(out), p(R))

| how | i(L) | v(out) | p(R) |
|---|---|---|---|
| recorder, exact, output_rtol 1e-10 (the report's call; feed in-loop, 101 steps) | −1.08e-4 | +1.60e-4 | +2.02e-4 |
| recorder, exact default (101 steps) | −1.08e-4 | +1.60e-4 | +2.02e-4 |
| stored record (102 rows), plain trapezoid = `with_event_limits` = `res.integral` | −1.08e-4 | +1.60e-4 | +2.02e-4 |
| stored record, output_rtol 1e-10 (165 269 rows, `with_event_limits`) | +8.9e-6 | −3.95e-4 | −1.40e-3 |
| recorder / stored, exact dt = 1e-7 | −1.08e-8 | +1.60e-8 | +3.30e-8 |
| recorder / stored, exact dt = 1e-8 | −1.08e-10 | +1.59e-10 | +3.29e-10 |
| recorder, trbdf2 rtol 1e-11 | −1.1e-9 | −1.9e-9 | −3.8e-9 |
| recorder, pwl rule dt/16 | +1.8e-9 | −1.3e-9 | −5.8e-9 |

- The event limits change nothing here: i(L), v(out), p(R) do not jump
  (`n_jumps` 0); the left-limit convention is not the cause.
- The error is the chord's: exactly h² (10 us → 0.1 us → 10 ns:
  1e-4 → 1e-8 → 1e-10, the same digits).
- The report's `output_rtol=1e-10` never reached the engine: `simulate()`
  drops it when `recorders` is given (`__init__.py`: "a recorder samples on
  its own grid: the output-grid rule stays off"), with no warning (checked
  with `-W always`). The feed then sees the exact engine's default steps:
  one per 10 us cap (`recorder_feed['n_steps'] = 101`).
- Even the stored record at output_rtol 1e-10 is 1.4e-3 off on ∫p(R): the
  4096-sample cap bit in 40 intervals (`diagnostics['output_grid']
  ['n_output_capped'] = 40`), leaving 10 us chords (e.g. 203.2 → 213.2 us,
  across A). That one warns (RuntimeWarning).

## What a user following the documentation gets

Silently the wrong one, on every route a user would take on `engine='exact'`
without `dt`:

- `Recorder(windows=...)` (`_recorders.py`): "statistics over each window,
  clipped exactly"; the module docstring sends the exact engine through the
  in-loop feed ("events at their exact instants … they see every accepted
  step"). Nothing says an accepted step is a 10 us chord.
- `res.avg` / `res.integral` (`_recorders.result_avg`): "windows clipped
  exactly. **Correct on any grid, unlike `.mean()`.**" — the misleading
  sentence: on the exact engine's default grid it is 1.1e-4 off on this buck.
- `output_rtol`'s docstring does say "Ignored with `dt` / `store_every` /
  recorders", but the run does not warn.

## Finding F-B2-1 and what should change

The exact engine (states exact to 1e-14) hands back window statistics with
the trapezoid error of 10 us steps, silently. Options, smallest first:

1. Documentation: `result_avg`'s "Correct on any grid" → "exact for the
   piecewise-linear interpolant of the samples; on `engine='exact'` without
   `dt` the samples are 10 us apart and the error is the chord's (h²)"; the
   Recorder `windows` doc and the AI guide the same; warn when `output_rtol`
   is dropped for recorders.
2. Engine: the exact engine integrates each segment in closed form (it has
   the per-mode decomposition; `exact_truth.py` does exactly that), so its
   recorder windows / periods are exact like its states.

Until then: on `engine='exact'`, pass `dt` (1e-8 gives 1e-10 here) for
window statistics.

The note in `test_release_findings_b2.py` (test of the window integrals vs
trbdf2: "a model difference of its own") is wrong on the cause; it is the
chord, not the model.

Test: `python/tests/test_release_findings_b2_exact.py` — strict xfail
(raises=AssertionError) on the recorder integrals within 1e-6 of the truth;
its control (states within 1e-12 of the truth) raises RuntimeError so a state
regression fails instead of being absorbed by the xfail.

## Run / not run

Run: `.scratch/ref44.py` (reproduces 1.08e-4 / −1.60e-4 / −2.02e-4 vs trbdf2
exactly), the probes above, the new test (1 xfailed; `--runxfail` fails on the
integral assertion with the numbers above), ruff on the test. Not run: the
full pytest, ctest (no C++ touched), any other circuit.
