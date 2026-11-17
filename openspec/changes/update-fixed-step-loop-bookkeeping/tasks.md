Throttle every build and run on the shared Mac: one build-heavy job at
a time.

Thresholds below are targets derived from the review's time split
(estimates, not prototyped). A missed threshold is reported to the owner
with the new time split; it is not waived.

## 1. Baseline (before any change)
- [x] 1.1 Confirm `restore-reference-benchmark-harness` is merged and its fixed-engine goldens pass on this branch; record the golden set revision — not on this branch; baseline = `probes/bookkeeping_baseline.py` at 662c137c (DECISIONS.md)
- [x] 1.2 Add loop counters `switch_fn_calls`, `segment_searches`, `rollback_snapshots` to the run result and `res.counters` (no behaviour change; goldens still pass)
- [x] 1.3 Record, best of 5: ns/step for buck and VSI (`perf/p1_basic.py`), RLC n = 5 / 152 / 1502 (`perf/p2_scale.py`), no-`switch_fn` vs constant schedule (`perf/p7_python.py`), `res.times` access (`perf/p11_res.py`); record the three counters for the buck and the VSI — `probes/perf_ab.py` (interleaved A/B, DECISIONS.md table)
- [x] 1.4 Re-take the `sample` split for the buck (`perf/samp.sh`) and note the `memset`, lookup, `switch_fn` and record shares — superseded by the A/B and counters

## 2. Event-iteration scratch
- [x] 2.1 Hoist `masks_seen` above the step loop on the dynamic path (`run_transient.hpp:920`)
- [x] 2.2 Same on the static path (`:1593`)
- [x] 2.3 Goldens: states, switch states, `event_iteration_count` `np.array_equal` — 69 probe runs, value + bytes

## 3. One `switch_fn` evaluation per solve time
- [x] 3.1 Read the gate mask once per step; reuse it for `mask_pre` and every event iteration at the same `t` (dynamic and static paths)
- [x] 3.2 Sub-step and dt-retry solves read the schedule once per distinct time
- [x] 3.3 Test with a counting schedule: `calls(2N) − calls(N) == N` on a run without retries; one extra call per retry sub-step time — `test_switch_fn_is_read_once_per_step`
- [x] 3.4 Goldens identical; closed-loop examples (observer writes, `switch_fn` reads) identical

## 4. Last-segment memo
- [x] 4.1 `PwlStateSpaceCache::generation()`, bumped by lazy insert, eviction, `refactor_parametric`, `clear` (and `build`, `build_lazy`, `build_one_segment`)
- [x] 4.2 Loop-local memo `{mask, segment, generation}`; LRU tick still written on a memo hit
- [x] 4.3 Tests: `segment_searches ≤ mask changes + 1` on the buck; eviction between two steps under a tight byte budget invalidates the memo; `segment_evictions` unchanged against baseline — bound is per solve, pinned 81 / 40; eviction test 2 evictions, 3 searches
- [x] 4.4 Goldens identical

## 5. Rollback snapshots
- [x] 5.1 List every consumer of `history_snap`, the five device-history snapshots, `diodes_snap` and `last_solved_bits`
- [x] 5.2 History snapshot as a flat `(v_prev, i_prev)` buffer; diode bits as packed words — diode bits already packed
- [x] 5.3 Skip the rollback snapshots when no rollback path is enabled; `rollback_snapshots == 0` in that configuration
- [x] 5.4 Goldens identical, including a dt-retry case and a sub-step-correction case

## 6. Native default schedule
- [x] 6.1 Bind `make_constant_switch_fn(mask)` over `sources::constant_schedule`
- [x] 6.2 `simulate()` uses it when `switch_fn is None` on the `pwl` route (`__init__.py:2673`) and the `trbdf2` route (`:2304`); warnings unchanged
- [x] 6.3 Goldens identical on both engines for circuits without a `switch_fn`; if `trbdf2` differs, pass the mask function without the edge oracle and record why — identical on both; no fallback
- [x] 6.4 Test: a run without `switch_fn` installs no Python callable in the step loop — `test_default_schedule_is_native`

## 7. Source and history passes
- [x] 7.1 Per-run index lists for sine, PWM and pulse sources; rows written directly into `b_extra`
- [ ] 7.2 Cached `2C/dt`, `dt/2L`, `2L/dt` per history entry, keyed on `dt` — not done (DECISIONS.md)
- [x] 7.3 Goldens identical by value and by byte hash (signed zeros), including a dt-retry case

## 8. Result plumbing
- [x] 8.1 `res.times` as a read-only zero-copy array; samples written through a cursor (no cursor: zero-copy over the vector)
- [x] 8.2 Sweep the package and tests for list-only use of `res.times`
- [x] 8.3 Test: `res.times.base` is the result, the array is read-only, values equal the previous list

## 9. Docs
- [x] 9.1 CHANGELOG (`res.times` type, callback contract, `res.counters`)
- [x] 9.2 `docs/migration-guide.md`, `docs/performance-tuning.md`, `docs/internals/layer4-pwl-state-space-cache.md`; KNOWN_LIMITATIONS only if 6.3 took the fallback (KNOWN_LIMITATIONS unchanged: 6.3 took no fallback)
- [x] 9.3 Stubs for `make_constant_switch_fn`, `res.counters`, `res.times`

## 10. Verification and acceptance
- [x] 10.1 `ruff`, `pytest`, `ctest`, `openspec validate --all --strict`, stub check, `test_zero_alloc`
- [x] 10.2 Counter regression test in CI (no wall-clock): `switch_fn_calls`, `segment_searches`, `rollback_snapshots` at the values fixed in 3.3, 4.3 and 5.3 — `python/tests/test_fixed_step_loop_bookkeeping.py`
- [x] 10.3 Re-run 1.3 on the same machine and throttle. Accept when: buck and VSI ns/step ≤ 0.75 × baseline; no-`switch_fn` RLC n = 5 within 10 % of the constant-schedule figure; RLC n = 1502 not slower than baseline; `res.times` access ≤ 0.1 ms at 100 k samples. A missed threshold is reported to the owner with the new time split, not waived — buck 0.76 (target 0.75: missed by 0.01, reported); the rest met
