Throttle every build and run on the shared Mac: `-j2`,
`nice -n 15 taskpolicy -b`, one build-heavy job at a time.

Thresholds below are targets derived from the review's time split and
from the edge counts of the probe circuits (estimates, not prototyped).
A missed threshold is reported to the owner with the new time split; it
is not waived.

## 1. Baseline (before any change)
- [ ] 1.1 Confirm the harness has `trbdf2` goldens (states and sample times) for the buck, the VSI and the RLC ladder, and an analytic reference set; add the three probe circuits from the pre-change build if missing
- [ ] 1.2 Record `event_hits`, `event_refactors`, `event_builds`, `n_accept`, `n_reject`, `n_solves` with `perf/p5_events.py`. Expected on this branch: buck 10 ms 20,885 / 25,115 / 5 / 20,413; VSI 20 ms 16,548 / 16,337 / 3,198 / 16,396; RLC n = 1502 2 ms 2,847 / 1,442 / 1 / 2,143
- [ ] 1.3 Record wall time per run, best of 3 (`perf/p1_basic.py` rows 5 and 7, `perf/p2_scale.py` row 8)
- [ ] 1.4 Record heap allocations per `solve_at` refactor with the `test_zero_alloc` harness
- [ ] 1.5 Record the error of each analytic reference at the default tolerances

## 2. Part 1 — byte-budgeted event table (bit-identical)
- [ ] 2.1 `EventEntry::approx_bytes()`; event table bounded by the segment byte budget; eviction after the throwing steps; counter `event_evictions`
- [ ] 2.2 Remove `kMaxEventEntries`; update `num_event_entries` / `num_alt_dt_values` docs and the tests that assume 8
- [ ] 2.3 Goldens: `trbdf2` states and sample times `np.array_equal`
- [ ] 2.4 Counter test: on the VSI, `event_builds` equals the number of distinct masks solved and `event_evictions == 0`
- [ ] 2.5 Tight-budget test: evictions occur and the result equals the unlimited-budget run

## 3. Part 2 — allocation-free refactor (bit-identical)
- [ ] 3.1 `G` and `C` scattered on the union pattern per entry; in-place `J[k] = G[k] + inv_dt·C[k]`; unit test: bytes equal to `G + (1/dt)·C` for 100 `dt` values per reference mask
- [ ] 3.2 `factorize` work vectors and L/U capacity kept as solver members
- [ ] 3.3 Zero-alloc test: the second and later refactors of a resident entry perform 0 heap allocations
- [ ] 3.4 Goldens identical; LU unit and scaling tests pass
- [ ] 3.5 Measure as in 1.3 (parts 1 + 2 only). Accept when the VSI wall time is ≤ 0.7 × baseline and no case is slower

## 4. Part 3 — grid and factor slots (BREAKING, off by default until 6)
- [ ] 4.1 `quantise_down(h, m)` with `frexp`, a literal table and `ldexp`; unit tests: result ≤ h, result > h / 2^(1/m), idempotent, identical bits for a fixed list of inputs
- [ ] 4.2 Option `step_grid_per_octave` (C++ options, binding, `simulate()` passthrough); 0 = continuous
- [ ] 4.3 Single quantisation point after the clamp; exemptions for `h_max`, `h_lo` and landing steps; stat `n_offgrid_steps`
- [ ] 4.4 Factor slots: four grid slots (LRU) and one scratch slot per entry; `solve_at` takes the on-grid hint; counters `event_slot_evictions`, `event_analyses`
- [ ] 4.5 With `step_grid_per_octave = 0`: goldens identical to parts 1 + 2
- [ ] 4.6 Counter test (structural): `event_builds + event_refactors` ≤ off-grid solves + distinct on-grid `(mask, dt)` pairs + `event_slot_evictions`

## 5. Part 3 — measurement and accuracy gate
- [ ] 5.1 Measure `m` ∈ {2, 4, 8} on the three probe circuits: `n_accept`, `event_refactors`, wall time, reference error; fix the default
- [ ] 5.2 Accuracy: every analytic reference within 1.25 × its baseline error at the default tolerances, and inside the engine's stated tolerance; `n_forced_accepts`, `n_chatter_breaks` and `n_held_diode_redecided` not above baseline
- [ ] 5.3 Step-count gate: `n_accept ≤ 1.20 × baseline` on each probe circuit
- [ ] 5.4 Reuse targets: `event_refactors ≤ 0.30 × n_accept` on the buck (1.23 today), `≤ 0.40 × n_accept` on the VSI (1.00 today), `≤ 0.10 × n_accept` on the RLC ladder (0.67 today)
- [ ] 5.5 Wall-time targets for the same `t_end`: buck ≤ 0.5 × baseline, VSI ≤ 0.5 × baseline, RLC n = 1502 ≤ 0.7 × baseline
- [ ] 5.6 Hand 5.1–5.5 to the owner for the accuracy sign-off the review asked for

## 6. Default flip (after sign-off)
- [ ] 6.1 Default `step_grid_per_octave = 4` (or the value fixed in 5.1)
- [ ] 6.2 Sweep tests and docs that pin `n_accept`, sample counts or sample times on `trbdf2`; regenerate the `trbdf2` references
- [ ] 6.3 Cross-engine agreement suites pass at their existing tolerances

## 7. Docs
- [ ] 7.1 CHANGELOG: parts 1–2 under performance, part 3 under numerical changes with the migration line
- [ ] 7.2 KNOWN_LIMITATIONS: landing steps still refactor; memory per mask
- [ ] 7.3 `docs/solvers.md`, `docs/performance-tuning.md`, `docs/internals/layer4-pwl-state-space-cache.md`; stubs for the option and the new counters

## 8. Verification
- [ ] 8.1 `ruff`, `pytest`, `ctest`, `openspec validate --all --strict`, stub check, `test_zero_alloc`
- [ ] 8.2 CI regression test on counters only (`event_builds`, `event_evictions`, the structural bound of 4.6, the ratios of 5.4); no wall-clock assertion
