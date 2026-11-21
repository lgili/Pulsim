## Why

On small circuits the fixed-step (`pwl`) engine spends most of a step on
bookkeeping, not on the solve. Core review 2026-11, `raw/perf.md`
(numbers taken under `nice -n 15 taskpolicy -b`; trust the ratios):

| Cost today | Measurement | Where (this branch) | Probe |
|---|---|---|---|
| A 64-entry `SwitchStateMask` array (about 3.5 KB) is value-initialised inside the step loop | 21 % of the buck step is `memset`; 13 % on the VSI | `solver/run_transient.hpp:920`, `:1593` | `perf/p1_basic.py`, `perf/samp_buck_pwl.txt` |
| `switch_fn` is called twice per step with the same `t` | 7 % of the buck step with a C++ schedule; a Python callable costs 0.25–1.06 µs per call | `solver/run_transient.hpp:892`, `:956` (static path `:1615`) | `perf/p7_python.py` |
| Every solve searches the mask table | 4.5–7 % | `pwl/cache.hpp:298` (`try_lookup`), reached from `:343` and `run_transient.hpp:989` | `perf/samp_buck_pwl.txt` |
| Rollback snapshots are taken every step, whole 48-byte history entries and one `vector<bool>` per diode set | diode snapshots 7 % on the VSI | `solver/run_transient.hpp:878-888`, `pwl/history_state.hpp:474` | `perf/samp_vsi_pwl.txt` |
| With no `switch_fn`, `simulate()` passes a Python lambda | 704 vs 373 ns/step (1.89×) on an RLC with n = 5; with a Python `switch_fn` a 4-thread sweep runs at 0.21× of sequential | `python/pulsim/__init__.py:2673` (pwl), `:2304` (trbdf2) | `perf/p7_python.py`, `perf/p8_par.py` |
| Source passes zero a full vector and scan every branch each step; the history pass divides twice per inductor per step | sine scan 4.3 %, history 4.6 % at n = 1502 | `sources/sine_b_extra.hpp:41-64` (same shape in `pwm_b_extra.hpp`, `pulse_b_extra.hpp`), `pwl/history_state.hpp:160-195` | `perf/p2_scale.py`, `perf/samp_lad500.txt` |
| `res.times` is converted to a Python list on every access | 7.5 ms per access at 100 k samples (`res.states`: 10 µs) | `python/bindings.cpp:1910` | `perf/p11_res.py` |

The buck step costs 380–460 ns and the triangular solve is 13 % of it.

## What Changes

Every item below leaves the arithmetic of a step untouched. Nothing in
this change alters a state value or the step sequence.

- **Event-iteration scratch hoisted.** `masks_seen` moves out of the step
  loop (both the dynamic and the static path); a step resets only its
  count.
- **One `switch_fn` evaluation per solve time.** The gate mask read for
  `mask_pre` is reused by every event iteration at the same `t`; only a
  solve at a different time (dt-retry sub-step, commutation split) calls
  `switch_fn` again. Python callbacks per nominal step drop from two or
  more to one.
- **Last-segment memo.** The loop keeps the last mask, its segment
  pointer and the cache generation it was read under; a step whose mask
  is unchanged skips the table search. The LRU tick is still written, so
  eviction order is unchanged. `PwlStateSpaceCache` gains a generation
  counter bumped by every structural mutation.
- **Smaller rollback snapshots.** The history snapshot copies only the
  mutable pair `(v_prev, i_prev)`; diode on-bits are copied as packed
  words; all snapshots are skipped when neither rollback path can run
  (`max_dt_halvings == 0` or no dt-retry possible, and
  `enable_substep_state_correction == false`).
- **Native default schedule.** When `switch_fn` is `None`, `simulate()`
  passes a C++ constant `SwitchSchedule` holding the same mask the
  lambda returned. New binding `make_constant_switch_fn(mask)` over the
  existing `sources::constant_schedule`.
- **Sparse source passes.** Sine, PWM and pulse sources write their rows
  straight into `b_extra` from an index list built once per run; the
  history pass caches `2C/dt`, `dt/2L` and `2L/dt` per entry, recomputed
  only when `dt` changes.
- **Zero-copy `res.times`.** **BREAKING (type only):** `res.times`
  becomes a read-only 1-D NumPy view, like `res.states`; it was a Python
  list. Values are unchanged. Samples are written through a cursor into
  the reserved buffers.
- **Loop counters.** The run reports `switch_fn_calls`,
  `segment_searches` and `rollback_snapshots` (`res.counters`), the
  basis of the regression test.

Not in this change: the event queue and declarative modulators
(`add-unified-event-queue`), the dense small-n path
(`add-exact-propagator-strategy`), anything inside the solve
(`update-linear-time-assembly`).

## Impact

- Affected specs: `transient-timestep`, `python-bindings`.
- Affected code: `solver/run_transient.hpp`, `solver/result.hpp`,
  `pwl/cache.hpp` (generation counter), `pwl/history_state.hpp`,
  `pwl/diode_event_state.hpp`, `sources/{sine,pwm,pulse}_b_extra.hpp`,
  `python/bindings.cpp`, `python/pulsim/__init__.py`, stubs, docs.
- Bit-identity: states, switch-state record and event counts are
  identical on every circuit of the reference goldens. Depends on
  `restore-reference-benchmark-harness` for that oracle.
- Behaviour visible to users: the type of `res.times`; the number of
  `switch_fn` calls per step. A `switch_fn` that changes its own state
  on each call (none in the package does) sees fewer calls.
- Expected speed-up, **estimates from the time split, not prototyped**:
  1.5–1.8× on circuits with n < 20 (hoist, single call, memo,
  snapshots); 5–10 % at n ≥ 1500 (source and history passes); a few
  percent from the sample cursor. The 1.89× for runs without a
  `switch_fn` is a measured ratio between the two existing paths.
