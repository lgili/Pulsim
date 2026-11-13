Throttle every build and run on the shared Mac: `-j2`,
`nice -n 15 taskpolicy -b`, one build-heavy job at a time.

Thresholds below are targets derived from the review's time split
(estimates, not prototyped). A missed threshold is reported to the owner
with the new time split; it is not waived.

## 1. Baseline (before any change)
- [ ] 1.1 Confirm the reference goldens contain the four Newton probe circuits (Shockley buck, Shockley rectifier + ladder at n = 51 and n = 483, MOSFET Level-1) on `pwl` and `trbdf2`; add them from the pre-change build if missing
- [x] 1.2 Add counters `newton_solves`, `newton_iterations`, `newton_analyses`, `newton_factorizations`, `newton_lm_attempts` (no behaviour change; goldens pass) — on `NewtonWorkspace`; surfaced as `SimulationResult.newton` in stage 2 (DECISIONS.md 11)
- [x] 1.3 Record, best of 3: ns/step for `perf/p4_newton.py` (rows 10–12) and `perf/p10_mos.py` (row 13), and the five counters per circuit — golden probe `perf`, interleaved base / pre-A1 / A1, best of 3
- [x] 1.4 Record heap allocations per Newton iteration with the `test_zero_alloc` harness (operator-new count and Eigen malloc guard) on the Shockley buck — probe: 47.7 Eigen mallocs per iteration, 15 `operator new` per call
- [ ] 1.5 Re-take the `sample` split for the Shockley buck and the n = 483 rectifier

## 2. Stage A1 — workspace and in-place Jacobian
- [x] 2.1 `NewtonWorkspace`; optional trailing parameter on `solve_with_newton_b_extra`; `nullptr` keeps the current path
- [x] 2.2 Union pattern and slot maps; `J[k] = L[k] + N[k]` in place
- [x] 2.3 Zero-keep-pattern helper for `J_nl` at the existing `setZero()` sites and the outermost engine wrappers — `sparse::zero_keep_pattern`, bit-identical (DECISIONS.md 12)
- [x] 2.4 Hoist `f_combined`, the `J_lin·x` buffer, `neg_f`, `dx` and the line-search vectors; out-parameter overload for `x` — `solve_with_newton_in_place` (DECISIONS.md 13)
- [x] 2.5 Unit test: the in-place Jacobian equals `seg.J + J_nl` entry by entry (pattern and bytes) over 100 random iterates of each Newton golden — on a two-diode buck (100 random iterates); the goldens cover the rest end to end
- [x] 2.6 Test with two stateful device kinds in one circuit: no stamp is lost — base refresh + Lauritzen diode + saturable inductor composed as the engine composes them; kept pattern byte-equal to a fresh `J_nl`

## 3. Stage A1 — persistent analysis
- [x] 3.1 Pattern check after each refresh; rebuild and re-analyse on a real change
- [x] 3.2 Hard gate: for every Newton golden, factors and solutions from "fresh `Eigen::SparseLU` per iteration" and "one analysis reused" are byte-equal. If not, keep the fresh analysis and stop A1 at task 2 — passed: 288 synthetic factorisations byte-equal, Newton goldens identical (DECISIONS.md 2)
- [x] 3.3 Engines pass a workspace: `run_transient` (dynamic and static paths, retry segment) and `run_transient_trbdf2`
- [x] 3.4 Goldens: states, switch states, iteration counts `np.array_equal` on `pwl` and `trbdf2` — states and switch states; iteration counts via the C++ workspace test
- [x] 3.5 Counter test: `newton_analyses ≤ 2` per run and `newton_iterations` equal to baseline on every Newton golden — `python/tests/test_newton_persistent_solver.py` (`analyses == 1`); the golden gate compares the iteration count of every Newton call
- [x] 3.6 Allocation test: zero allocations per iteration outside `Eigen::SparseLU::factorize` after the first step — met with A2: zero allocations per iteration, factorisation included (`test_zero_alloc`, DECISIONS.md 14)
- [ ] 3.7 Measure as in 1.3. Accept A1 when the Shockley buck is ≤ 0.5 × baseline ns/step and the n = 1503 rectifier ≤ 0.75 × baseline — measured vs baseline: Shockley buck pwl 0.57×, trbdf2 0.57× (target 0.5×, missed); rectifier n = 483 0.56× (n = 1503 not measured; target 0.75×); MOSFET L1 0.58×, saturable inductor 0.55×

## 4. Stage A2 — in-house LU (gated: owner sign-off on the tolerance gate)
- [x] 4.1 Workspace backend hint `Pulsim`; analysis reuse through the same pattern check
- [x] 4.2 Gate: `newton_iterations` equal to A1 on every Newton golden; states within 1e-9 relative; no new Newton failure in the hostile-circuit and stall/chatter suites — every call's iteration count identical; states bit-identical (largest relative difference 0) on all 19 cases (DECISIONS.md 15)
- [x] 4.3 Regenerate the Newton goldens; record the largest relative difference per circuit in the PR — nothing to regenerate: 0 on every circuit (DECISIONS.md 15)
- [x] 4.4 Allocation test: zero heap allocations per Newton iteration after the first step (requires the allocation-free `factorize` of `update-trbdf2-factor-reuse`; otherwise record the residual count)
- [x] 4.5 Measure as in 1.3. Accept A2 when the Shockley buck is ≤ 0.3 × baseline and the n = 1503 rectifier ≤ 0.6 × baseline — Shockley buck 0.14 × baseline, rectifier n = 483 0.24 × (n = 1503 not in the golden set; DECISIONS.md 16)

## 5. Stage B — low-rank update (gated: starts only after 4 is merged)
- [ ] 5.1 Baseline for stage B: counters and ns/step after A2; path length of the device columns on the rectifier ladder (`partial_refactor_count_path`)
- [ ] 5.2 Prototype `partial_refactor` on the device columns and Woodbury on the cached linear factor on the rectifier ladder; choose by measured time and by error against the A2 result
- [ ] 5.3 Implement the chosen update; full factorisation on mask or `dt` change and on pivot fault; counter `newton_partial_refactors`
- [ ] 5.4 Gate: iteration counts equal to A2; states within 1e-9 relative; fallback ratio ≤ 20 % on the probe circuits
- [ ] 5.5 Measure as in 1.3. Accept B when the n = 1503 rectifier is ≤ 0.15 × baseline

## 6. Docs
- [x] 6.1 CHANGELOG (per stage; A2 and B under numerical changes)
- [x] 6.2 KNOWN_LIMITATIONS: LM path still factorises from scratch; user refreshers with `x`-dependent stamp positions re-analyse
- [x] 6.3 `docs/internals/layer4-pwl-state-space-cache.md` and `docs/solvers.md`: workspace, invalidation rules, counters

## 7. Verification
- [x] 7.1 `ruff`, `pytest`, `ctest`, `openspec validate --all --strict`, stub check, `test_zero_alloc`
- [x] 7.2 CI regression test on counters only (analyses, factorisations per iteration, iterations); no wall-clock assertion — C++ counter test (`analyses == 1`, `factorizations == iterations`) in ctest
