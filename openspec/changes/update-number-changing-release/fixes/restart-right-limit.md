# F-SB-1 — an event's right limit is a state value (restart rule, fixed engine)

Branch `fix/core-restart-right-limit` from `origin/feat/core-integration-14r`
(6027f4fd). Finding and evidence: `restart-scoreboard.md` (branch
`docs/core-restart-scoreboard-investigation`, ad9ee191). Tests:
`python/tests/test_restart_right_limit.py`, the investigation's
`test_restart_scoreboard_findings.py` (its strict xfail removed: it passes).

## The decision implemented (the orchestrator's)

Two things were stored in one field. Now:

1. **The right limit is a state value** x(t_e+), recorded only where
   something happens. `StepLines::rl_*` (loss_tracker.hpp) → `record_lines`
   → `result.right_limit_*` / `inner_event_right*`.
2. **The quadrature start stays the lines' own** (`StepLines::x_start`,
   `right[]`, the held-pieces rows): the loss tracker and the in-loop
   recorder read them as before; they never reach the result.

## What x(t_e+) is, and how it is computed (`event_right_limit`, run_transient.hpp)

Each capacitor voltage / inductor current at its value at t_e+, every other
unknown solved in the event's mode there:

- continuous states: x0 exactly (the histories at the step start);
- a state an IDEAL element's constraint moves (branch form, the structural
  record): x0 + J, J the probe's jump (consistent_start's `ip_d`, flagged
  jumping) — right − left = the record's jump;
- a CONDUCTANCE-form element's unresolved fast mode that the impulse rule
  records (τ = L·g_off = 1 ps, a vcswitch cutting an inductor): the states are
  NOT relaxed. Its right limit is the model's own, i_L continuous and
  v = i_L/g_off — the exact engine's (F-5-10) — and the record reports the
  collapse that follows. This is the one reading of the decision's
  parenthetical "x0+ = consistent_start's" I did not take literally: with
  x0 + J there the row equals no engine's right limit, and the group-A
  hand-off (the impulse's peak visible in the right-limit row) cannot hold.
  For ideal elements both readings coincide.

The algebraic solve: the ε → 0 limit of a backward-Euler solve of ε from those
states in stage 1's mode (sources at t_e + ε, i.e. a source edge's right
value), as two single solves of ε and 2ε and Richardson 2·x(ε) − x(2ε)
(the O(ε) motion cancels; ε = 1e-9·γh). Exact to rounding for every mode
slower than ε; a mode with τ near ε keeps O((ε/τ)²) (the 1 ps cut at dt
0.5 µs: 4e-9 relative on i_L). Everything the solves touch is put back (x,
b_extra, histories, stateful histories, diode bits, iteration flags, the Newton
round-off budget and its counters); the step's trajectory does not change.

Where: `restart_lined(a, h, loc, ev)` — `ev` true at a trigger at the step
start that is an event (`rs_trig_ev`: a gate-mask change, a source / b_extra
edge; NOT an unknown start's stamp and NOT t = t_start), at a scheduled split
event (i > 0) and at a located split crossing; `run_restart` computes the
limit when `ev` or the probe found a jump (`ip_jump`: a recorded impulse at
any restart step, t = 0 included). Never at a window's later steps.

The held-pieces rows of a mixed-mode restart step (F-4-4: X1 held over w1·h,
then X2) are quadrature, not events: they stay in the lines and no longer
appear in `inner_events`.

## Numbers

| case | truth | before | after |
|---|---|---|---|
| 10 V → ideal S → 1 Ω → 1 µF, dt 0.5 µs: right rows | one, at the closing | 6 (t = 0, 0.5, 1.0, 2.5, 3.0, 3.5 µs) | 1 (2.5 µs; the sampled gate) |
| same, v_C right − left at the closing | 0 (continuous) | 0.163174 V | 4e-19 V |
| same, v_a / i right | 10 V / 10 A | 10 V / 9.84 A | 10 V / 10 A |
| C 1 µF (2 V) onto 10 V via ideal S | v_C+ = 10 V, J = −λ/C | line start | 10 V; J = record to 1e-9 (pwl and dsed) |
| L 10 µH cut by ideal S | i_L+ = 0, J = λ/L | line start | 0; J = record to 1e-9 (pwl and dsed) |
| 48 V, L 1 mH, switch g_off 1 nS (F-5-10 on pwl) | v_sw+ = i_L/g_off, i_L continuous | no right-limit row voltage | 2.4e9-class row, warning raised |
| ll09, every right row: i(Lload), V(mid) | continuous (V(mid) = f(i_L)) | V(mid) 48.98 V vs 99.99 V | right − left ≤ 1.2e-17 A / 7.7e-13 V |

(scoreboard cells: below)

## Voltage guard on the fixed engine (group A hand-off)

`solver::find_implausible_voltage(graph, pool, result)` now also scans the
result's right-limit rows and inner-event right rows (NaN rows skipped). The
fixed engine's finalize path already calls it (`_auto_check_voltage_sanity`
and `sim::Simulation::check_voltage_`), so the check costs no Python bytecode:
the overhead gate does not move. On a result without event limits the scan is
one bool test.
