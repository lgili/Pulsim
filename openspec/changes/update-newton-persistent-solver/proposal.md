## Why

Circuits with Newton devices run 40–55× slower per step than linear
ones of the same size, and almost none of that is device evaluation.
Core review 2026-11, `raw/perf.md` (throttled machine; trust the
ratios):

| Case | Measured | Probe |
|---|---|---|
| Shockley-diode buck, `pwl`, n = 5 | 20.4 µs/step against 0.38 µs for the PWL-diode buck (53×) | `perf/p4_newton.py` |
| Same, `trbdf2` | 39.8 µs per accepted step; 17.9 µs per Newton solve call | `perf/p4_newton.py` |
| Shockley rectifier + ladder, `pwl`, n = 3 / 51 / 483 / 1503 | 10.6 µs / 47.7 µs / 453 µs / 1.80 ms per step (linear circuits: about 11 µs at n = 482, 33.7 µs at n = 1502) | `perf/p4_newton.py` |
| MOSFET Level-1, `pwl`, n = 7 | 14.5 µs/step | `perf/p10_mos.py` |

Time split (`perf/samp_shock.txt`, `perf/samp_shockrect.txt`):

| | malloc/free + memset | numeric factorise | analyse | sparse copy | triangular solve | device evaluation |
|---|---|---|---|---|---|---|
| n = 5 | 64 % | 13.5 % | 5.3 % | 8.8 % | 2.7 % | 0.3 % |
| n = 483 | 5 % | 41 % | 34 % | 11 % | 4.7 % | < 0.5 % |

The cause is in `pwl/nonlinear_solve.hpp`. Every Newton iteration:

- copies `seg.J` and merges `J_nl` into a new matrix (`:161-165`);
- allocates `f_combined` (`:170`) and `neg_f` (`:332`);
- constructs a new solver, runs the symbolic analysis (COLAMD and
  elimination tree) and factorises (`:311-322`). The solver is
  `sparse::make_default_solver()` with no arguments, which is
  `Eigen::SparseLU` (`sparse/solver.hpp:301-303`), not the in-house LU
  the linear path uses.

There is no counter for Newton iterations, analyses or factorisations
today, so the report inferred them from time.

## What Changes

Three stages. Stage A1 is bit-identical; A2 and B change rounding and
are gated separately.

- **Stage A1 — persistent workspace, same backend (bit-identical).**
  - A `NewtonWorkspace` owned by the run (one per `run_transient` /
    `run_transient_trbdf2` call) holds the combined Jacobian, the work
    vectors and one linear solver.
  - The combined Jacobian has the union pattern of `seg.J` and `J_nl`,
    built once; each iteration writes `J[k] = J_lin[k] + J_nl[k]` in
    place through two slot maps.
  - The symbolic analysis is done once per pattern and reused; a
    pattern check after each refresh triggers a new analysis only when
    the pattern really changed.
  - `f_combined`, `neg_f`, `dx`, the `J_lin·x` product and the
    line-search trial vectors are hoisted. `J_nl` keeps its pattern
    between iterations instead of being rebuilt entry by entry.
  - The backend stays `Eigen::SparseLU`, so pivots and rounding are the
    ones of today.
- **Stage A2 — in-house LU for the Newton path. BREAKING (rounding):**
  the workspace solver becomes `PulsimSparseLuSolver`. Iteration counts
  are expected to be unchanged and states to agree within 1e-9
  relative; they are not bit-identical because the two factorisations
  pivot and round differently. This removes the allocations inside
  `Eigen::SparseLU::factorize` and is the precondition of stage B.
- **Stage B — low-rank update (follow-up inside this change, gated
  separately). BREAKING (rounding):** between iterations only the
  columns touched by nonlinear devices change. Route them through the
  existing `partial_refactor` (path-based, pivot-fault fallback to a
  full factorisation). Woodbury on the cached linear factor is the
  alternative, decided by measurement at the start of stage B.
- **Counters:** `newton_solves`, `newton_iterations`,
  `newton_analyses`, `newton_factorizations`,
  `newton_partial_refactors`, `newton_lm_attempts` in `res.counters`.
- Unchanged: the convergence tests, the absolute `dx` rule, the line
  search and LM logic, the DC cascade (`dc_strategy.hpp`,
  `pseudo_transient.hpp`, `dc_assemble.hpp` keep their own solvers).

## Impact

- Affected specs: `linear-solver`, `transient-timestep`.
- Affected code: `pwl/nonlinear_solve.hpp`, the `J_nl.setZero()` sites
  of `pwl/nonlinear_refresh_*.hpp`, `solver/run_transient.hpp:989-1014`
  and `:1647`, `solver/trbdf2_transient.hpp:698-727`,
  `sparse/pulsim_lu_solver.hpp` (stage A2/B), `solver/result.hpp`,
  `python/bindings.cpp`, `core/tests/perf/test_zero_alloc.cpp`.
- Bit-identity: stage A1 — states, iteration counts and switch-state
  record identical to the goldens. Stages A2 and B — iteration counts
  identical, states within 1e-9 relative; goldens of Newton circuits
  are regenerated once, with the owner's sign-off.
- Depends on `restore-reference-benchmark-harness` (the goldens must
  include the four Newton probe circuits before any edit).
- Expected speed-up, **estimates from the time split, not prototyped**:
  stage A about 4–5× at small n and about 2× at large n; stage B 15–40×
  in total. At n = 5 stage A1 alone cannot reach the full figure:
  `Eigen::SparseLU::factorize` allocates internally and those
  allocations stay until A2.
- Roadmap note: the Phase 1 row describes this change as an in-house LU
  with bit-identical results. That holds for A1 only; see design.md.
