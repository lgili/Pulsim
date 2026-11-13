Throttle every build and run on the shared Mac: `-j2`,
`nice -n 15 taskpolicy -b`, one build-heavy job at a time.

Thresholds below are targets derived from the review's time split
(estimates, not prototyped). A missed threshold is reported to the owner
with the new time split; it is not waived.

## 1. Baseline (before any change)
- [x] 1.1 Confirm the reference goldens pass on this branch; record the revision
- [x] 1.2 Add counters `structure_builds` (sparse-structure builds of `G`/`C`/`J`) and `symbolic_analyses` to `CacheMetrics` (no behaviour change)
- [x] 1.3 Record with `perf/p6_build.py`, best of 3: µs per segment for the RLC ladder at n = 5 / 17 / 152 / 1502 / 15,002 and per mask for the switched ladders (n = 310, 3008), buck and VSI; record both counters. Expected today: `structure_builds` and `symbolic_analyses` equal to the number of segments built
- [x] 1.4 Record ns/step for the RLC ladder at n = 152 / 1502 / 15,002 (`perf/p2_scale.py`) and the `sample` split at n = 1502
- [x] 1.5 Record first-visit cost in steps for the VSI and the n = 3008 switched ladder (segment build time ÷ step time)

## 2. Tier 1 — reserve (step S)
- [x] 2.1 Per-column reserve for `G` and `C` from node degrees in `assemble_segment_split` and the static path
- [x] 2.2 Test: `J` and `b` byte-equal to the pre-change assembly for every mask of every golden circuit, both regimes
- [x] 2.3 Measure 1.3. Target: build(n = 15,002) ÷ build(n = 1502) ≤ 15 (91 today)

## 3. Tier 1 — stamp-slot map (step M)
- [x] 3.1 Pattern template per regime on the cache; later assemblies copy indices, zero values, stamp into existing entries
- [x] 3.2 Slot arrays `G → J`, `C → J`; `J[k] = G[k] + inv_dt·C[k]`; reuse by `try_make_segment`, the event entries and `solve_rank1`
- [x] 3.3 Test as 2.2, plus signed-zero byte hash
- [x] 3.4 Counter test: `structure_builds ≤ 2` per cache whatever the number of masks
- [x] 3.5 Template invalidation test: a pool whose structure changes gets a new cache; `refactor_parametric` keeps the template and updates values

## 4. Tier 1 — shared symbolic analysis
- [x] 4.1 Confirm no reader of the L/U index arrays between `analyze` and `factorize`; remove the dead fill or make it lazy
- [x] 4.2 `Symbolic` object (column permutation, inverse, elimination tree) held by `shared_ptr<const>`; `adopt_symbolic` on the in-house solver with the index-array check; default `DirectSolver` implementation analyses privately
- [x] 4.3 Cache owns one `Symbolic` per regime; segments, event entries and the rank-1 solver adopt it
- [x] 4.4 Test: shared ordering equals a private analysis for every golden mask; factors byte-equal
- [x] 4.5 Counter test: `symbolic_analyses ≤ 2` per cache on the VSI (87 masks) and on a 3008-unknown switched ladder; `symbolic_fallbacks == 0`
- [x] 4.6 Mismatch test: a matrix with another pattern is analysed privately and `symbolic_fallbacks` is 1
- [x] 4.7 Goldens: states `np.array_equal` on `pwl` and `trbdf2`

## 5. Tier 1 — solve passes
- [ ] 5.1 A row-permuted copy of `b_constant` per segment after `factorize` (the original stays for Newton and diagnostics); fused "negate, add, permute" pass
- [ ] 5.2 Run-local solve work vector passed by the engine loops; thread-local default kept
- [x] 5.3 Goldens identical; concurrent-solve test (two threads, one warm cache) still passes
- [x] 5.4 `solve_rank1`: no structure rebuild per call (`structure_builds` unchanged across 1000 calls); results unchanged

## 6. Tier 1 acceptance
- [ ] 6.1 Re-run 1.3–1.5. Accept when: build(n = 15,002) ÷ build(n = 1502) ≤ 12; per-mask build at n = 3008 ≤ 0.1 × baseline; first visit on the VSI ≤ 0.5 × baseline; no ns/step regression at any n — measured: ratio 10.6 ✓, n = 3008 0.042× ✓, ns/step within noise ✓, VSI first visit 0.62× ✗ (DECISIONS.md 11)
- [x] 6.2 CI regression test on `structure_builds`, `symbolic_analyses`, `symbolic_fallbacks` only; no wall-clock assertion

## 7. Tier 2 — neighbouring masks through `partial_refactor` (gated: owner sign-off; BREAKING rounding)
- [ ] 7.1 Baseline: first-visit cost after tier 1; path length of single-switch flips on the VSI and the switched ladder
- [ ] 7.2 `clone()` on the in-house solver; nearest-resident-mask search on a lazy miss; path-ratio gate; counters `flip_partial_refactors`, `flip_fallbacks`
- [ ] 7.3 Cache flag, default off; with the flag off the goldens are identical to tier 1
- [ ] 7.4 Gate with the flag on: states within 1e-9 relative on every golden; solve backward error not above baseline; `flip_fallbacks ≤ 0.5 × flip_partial_refactors` on the switched-ladder probe
- [ ] 7.5 Measure 1.5. Accept when a first visit at n = 3008 costs ≤ 0.5 × the tier 1 figure; then default on and regenerate the goldens

## 8. Tier 3 — reciprocal pivots (gated: evidence first; BREAKING rounding)
- [ ] 8.1 Standalone solve microbenchmark at n = 152 / 1502 / 15,002 (divide vs stored reciprocal); stop here unless the reciprocal gives ≥ 1.15× at n = 1502
- [ ] 8.2 Store `1/U[k,k]` at `factorize` and `partial_refactor`; multiply in back substitution
- [ ] 8.3 Gate: states within 1e-9 relative on every golden; LU residual tests at their existing tolerances; regenerate the goldens with the owner's sign-off
- [ ] 8.4 Measure 1.4. Accept when ns/step at n = 1502 is ≤ 0.85 × the tier 1 figure

## 9. Docs
- [x] 9.1 CHANGELOG (tier 1 under performance; tiers 2–3 under numerical changes)
- [x] 9.2 `docs/internals/layer0-numeric-and-sparse.md`, `layer3-stamping-pipeline.md`, `layer4-pwl-state-space-cache.md`: template, slot arrays, symbolic object and its invalidation; note whether the per-stamp search shows in the profile
- [ ] 9.3 KNOWN_LIMITATIONS: tier 2 fallbacks; stubs for the new metrics

## 10. Verification
- [x] 10.1 `ruff`, `pytest`, `ctest`, `openspec validate --all --strict`, stub check, `test_zero_alloc`
