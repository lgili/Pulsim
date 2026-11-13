# fix-control-tick-semantics — decisions (2026-11-13)

## Open Questions of design.md

| # | Question | Option taken | Why | How to reverse |
|---|---|---|---|---|
| 1 | Fixed engine, `T_j` not a multiple of `dt` | Run with a warning ("not a multiple of dt"); `result.control_tick_jitter_max` records the largest lateness (< `dt`) | Proposed option; the count stays exact (`ceil(span/T)`) and only the instant moves | Turn the `warnings.warn` in `_control_plan.FixedStepTicks.__init__` into a `ValueError` |
| 2 | Fixed engine, tick 0 | At `t_start` with the initial state (the kernel already primes the observer there) | Proposed; equal to `trbdf2` | Start `FixedStepTicks._k` at 1 for the first instant after `t_start` |
| 3 | `controller_period` without `step_observer` | Refuse (`ValueError`, "without a step_observer"), on every engine (raised above the dispatch) | Proposed; ignoring a keyword is the silent-wrong-answer pattern | Drop the `elif controller_period is not None` branch in `compose_control` |
| 4 | Order of coincident ticks | Registration order: closed loops in list order, then `step_observer` | Proposed; the order the fixed path already used | Sort `plan.ticks` by period (slowest first) in `compose_control` |
| 5 | Ship the fixed-engine count change now or with `add-discontinuity-restart-rule` | Now | Proposed; the old count lost ticks (198 of 200) | Revert the `pwl` hunk of `simulate()` (keep the `if closed_loops:` throttled composition) |

## Choices the design left open

| Choice | Taken | Why / reverse |
|---|---|---|
| Kernel interface | `TrBdf2Options::observer_periods` plus a new trailing parameter `tick_fn(j, t, x)`; the scalar `observer_period` + `observer_fn` stays as the one-source form | Keeps every existing call site and the single-source arithmetic (`t_start + k*T`, `min` over one element) identical |
| Binding | `observer_periods=[...]`, `observers=[...]` kwargs; lengths must match (`ValueError`) | `stats['control_tick_counts']`; `n_ctrl_ticks` stays the sum |
| A `step_observer` with no period next to a `ClosedLoop` on `trbdf2` | Router blocker (`auto` routes to `pwl`, or refuses without `dt`); explicit `trbdf2` refuses (`feature='step_observer'`) | Before, it silently rode the loop's rate. Design 4: `every_step` is refused on `trbdf2`. Reverse: tick it at the fastest source period |
| Period-less loops (`period == 0`) | Every-step on `pwl` (their own throttle), refused on `trbdf2` | As before for a single loop; previously a period-less loop next to a periodic one rode the other's rate on `trbdf2` |
| `FixedStepTicks` at `t_end` | The kernel calls the observer at `t = t_end`; no tick is due there (instants are `< t_end`) | Without it the count was `ceil + 1` (2001 calls -> 21 ticks) |

## Proposal numbers

Reproduced on the pre-change extension (`.scratch/c5`): probe7 `pwl`
40 PI ticks, 20001 monitor calls, 13.296 V; `auto` 40 / 0 / 13.033 V
without the monitor and 400 / 400 / 12.091 V with it; probe6 `auto`
raises `TypeError` at `|`; probe2 `pwl` 2001 observer calls.

After the change: probe6 `pwl` 200 / 20, `auto` (trbdf2) 200 / 20;
probe7 `pwl` 40 PI ticks, 400 monitor calls, 13.026 V (was 13.296 V
with 20001 every-step monitor calls), `auto` 40 / 0 / 13.033 V and
40 / 400 / 13.033 V (was 400 / 400 / 12.091 V). `pwl` and `trbdf2` now
agree within 0.007 V on probe7 (was 0.263 V).
