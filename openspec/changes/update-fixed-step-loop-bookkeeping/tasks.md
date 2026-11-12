Throttle every build and run on the shared Mac: `-j2`,
`nice -n 15 taskpolicy -b`, one build-heavy job at a time.

Thresholds below are targets derived from the review's time split
(estimates, not prototyped). A missed threshold is reported to the owner
with the new time split; it is not waived.

## 1. Baseline (before any change)
- [ ] 1.1 Confirm `restore-reference-benchmark-harness` is merged and its fixed-engine goldens pass on this branch; record the golden set revision
- [ ] 1.2 Add loop counters `switch_fn_calls`, `segment_searches`, `rollback_snapshots` to the run result and `res.counters` (no behaviour change; goldens still pass)
- [ ] 1.3 Record, best of 5: ns/step for buck and VSI (`perf/p1_basic.py`), RLC n = 5 / 152 / 1502 (`perf/p2_scale.py`), no-`switch_fn` vs constant schedule (`perf/p7_python.py`), `res.times` access (`perf/p11_res.py`); record the three counters for the buck and the VSI
- [ ] 1.4 Re-take the `sample` split for the buck (`perf/samp.sh`) and note the `memset`, lookup, `switch_fn` and record shares

## 2. Event-iteration scratch
- [ ] 2.1 Hoist `masks_seen` above the step loop on the dynamic path (`run_transient.hpp:920`)
- [ ] 2.2 Same on the static path (`:1593`)
- [ ] 2.3 Goldens: states, switch states, `event_iteration_count` `np.array_equal`

## 3. One `switch_fn` evaluation per solve time
- [ ] 3.1 Read the gate mask once per step; reuse it for `mask_pre` and every event iteration at the same `t` (dynamic and static paths)
- [ ] 3.2 Sub-step and dt-retry solves read the schedule once per distinct time
- [ ] 3.3 Test with a counting schedule: `calls(2N) − calls(N) == N` on a run without retries; one extra call per retry sub-step time
- [ ] 3.4 Goldens identical; closed-loop examples (observer writes, `switch_fn` reads) identical

## 4. Last-segment memo
- [ ] 4.1 `PwlStateSpaceCache::generation()`, bumped by lazy insert, eviction, `refactor_parametric`, `clear`
- [ ] 4.2 Loop-local memo `{mask, segment, generation}`; LRU tick still written on a memo hit
- [ ] 4.3 Tests: `segment_searches ≤ mask changes + 1` on the buck; eviction between two steps under a tight byte budget invalidates the memo; `segment_evictions` unchanged against baseline
- [ ] 4.4 Goldens identical

## 5. Rollback snapshots
- [ ] 5.1 List every consumer of `history_snap`, the five device-history snapshots, `diodes_snap` and `last_solved_bits`
- [ ] 5.2 History snapshot as a flat `(v_prev, i_prev)` buffer; diode bits as packed words
- [ ] 5.3 Skip the rollback snapshots when no rollback path is enabled; `rollback_snapshots == 0` in that configuration
- [ ] 5.4 Goldens identical, including a dt-retry case and a sub-step-correction case

## 6. Native default schedule
- [ ] 6.1 Bind `make_constant_switch_fn(mask)` over `sources::constant_schedule`
- [ ] 6.2 `simulate()` uses it when `switch_fn is None` on the `pwl` route (`__init__.py:2673`) and the `trbdf2` route (`:2304`); warnings unchanged
- [ ] 6.3 Goldens identical on both engines for circuits without a `switch_fn`; if `trbdf2` differs, pass the mask function without the edge oracle and record why
- [ ] 6.4 Test: a run without `switch_fn` installs no Python callable in the step loop

## 7. Source and history passes
- [ ] 7.1 Per-run index lists for sine, PWM and pulse sources; rows written directly into `b_extra`
- [ ] 7.2 Cached `2C/dt`, `dt/2L`, `2L/dt` per history entry, keyed on `dt`
- [ ] 7.3 Goldens identical by value and by byte hash (signed zeros), including a dt-retry case

## 8. Result plumbing
- [ ] 8.1 `res.times` as a read-only zero-copy array; samples written through a cursor
- [ ] 8.2 Sweep the package and tests for list-only use of `res.times`
- [ ] 8.3 Test: `res.times.base` is the result, the array is read-only, values equal the previous list

## 9. Docs
- [ ] 9.1 CHANGELOG (`res.times` type, callback contract, `res.counters`)
- [ ] 9.2 `docs/migration-guide.md`, `docs/performance-tuning.md`, `docs/internals/layer4-pwl-state-space-cache.md`; KNOWN_LIMITATIONS only if 6.3 took the fallback
- [ ] 9.3 Stubs for `make_constant_switch_fn`, `res.counters`, `res.times`

## 10. Verification and acceptance
- [ ] 10.1 `ruff`, `pytest`, `ctest`, `openspec validate --all --strict`, stub check, `test_zero_alloc`
- [ ] 10.2 Counter regression test in CI (no wall-clock): `switch_fn_calls`, `segment_searches`, `rollback_snapshots` at the values fixed in 3.3, 4.3 and 5.3
- [ ] 10.3 Re-run 1.3 on the same machine and throttle. Accept when: buck and VSI ns/step ≤ 0.75 × baseline; no-`switch_fn` RLC n = 5 within 10 % of the constant-schedule figure; RLC n = 1502 not slower than baseline; `res.times` access ≤ 0.1 ms at 100 k samples. A missed threshold is reported to the owner with the new time split, not waived
