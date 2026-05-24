## Why

After v1.3.0 (`replace-klu-with-pulsim-sparse-lu`), the
path-based partial-refactor algorithm of Chan/Brandwajn/Tinney
1986 + Dinkelbach 2021 is fully implemented in-house — but
restricted to the **single use case the literature describes**:
a single switch bit flips → walk the etree path of the affected
column → update L+U in place. Captured speedup on that workload
is 2.7-2.9× (chapter 8 of `docs/how-pulsim-works/`).

The same etree-path machinery solves **two other SMPS-relevant
problems** that no open-source simulator currently exploits and
that we are not aware of in the published literature:

1. **Multi-bit switch transitions** (SPWM with multiple legs
   commutating in the same timestep, multilevel converter
   commutation patterns). Currently fall back to full
   `factorize()` via `PwlStateSpaceCache::solve_rank1`'s
   fast-path-vs-fallback gate. Could be handled by computing
   the **union of the etree paths** of all changed columns and
   re-eliminating once.

2. **Parametric value changes** for sweep / Monte Carlo /
   design-optimisation workloads. Today every sweep point pays
   a full `analyze + factorize` (~100 µs/point cold path; for a
   1000-point sweep that's 100 ms of pure setup before any
   simulation work). But when the user changes a single physical
   parameter (e.g. `L_out` from 100 µH to 101 µH), only the
   columns of $J$ that involve `L_out` change — same sparsity
   pattern, same etree — and path-based refactor applies
   identically to the switch case.

Both extensions reuse the same `partial_refactor` core. Both
target real pain points raised repeatedly in PE-simulator user
studies (long sweep times, slow multilevel-converter
simulations). Together they generalise the framework to:

> "**Any change to $J$'s values that preserves its sparsity
> pattern** can be amortised via etree-path walk."

This generalisation is the IEEE TPEL paper's strengthened §VI
contribution: not "we implemented Dinkelbach 2021" but
"**we extend the path-based update framework to three SMPS-
relevant use cases not covered in prior work**".

Expected speedups (extrapolated from chapter 8's microbench
data + first-principles cost analysis):

| Use case | Baseline | Path-based | Speedup |
|---|---:|---:|---:|
| Single-bit switch (already shipped) | 10.0 µs/call | 3.6 µs/call | **2.8×** |
| Multi-bit (3 legs commutating, NEW) | ~50 µs/call | ~10 µs/call | **5×** |
| 1-param sweep, 1000 points (NEW) | ~100 ms | ~5-10 ms | **10-20×** |
| Monte Carlo, 1000 samples × 5 params (NEW) | ~500 ms | ~10-20 ms | **25-50×** |

## What Changes

### Part A — Multi-bit switch transitions

- **MODIFIED** `PulsimSparseLuSolver::partial_refactor(new_M,
  changed_cols)` to accept a `std::span<const Index>` of
  arbitrary length and compute the **union of etree paths** for
  all changed columns. The current implementation already
  accumulates `changed_cols` into `varying_set_` across calls;
  this proposal extends the path-computation to handle the
  *single-call* multi-bit case efficiently (deduplicated
  via the existing `in_path` bitmap).
- **MODIFIED** `PwlStateSpaceCache::solve_rank1(mask, b_extra,
  x)` to **stop routing multi-bit transitions to
  full-factorize unconditionally**. Instead, compute the
  Hamming-distance set between the current mask and the
  previous mask, identify the corresponding column indices in
  $J$ via a new `DevicePool::columns_affected_by_switch(sw_id)`
  helper, and call `partial_refactor(new_J, changed_cols)`. If
  the path-union exceeds a (tunable) `MAX_PATH_LENGTH_RATIO`
  fraction of $n$ — meaning the path-based update would be no
  cheaper than a full factorise — fall back to `factorize()` as
  before.
- **NEW** `CacheMetrics::multi_bit_rank1_hits` counter to
  distinguish multi-bit successes from single-bit ones in
  diagnostic output.

### Part B — Parametric value changes for sweeps + Monte Carlo

- **NEW** `DevicePool::columns_affected_by_param(param_name)`
  helper: given a parameter symbol (e.g. `"L_out"`, the
  builder-time inductor handle), returns the set of column
  indices in $J$ whose values depend on that parameter. Cached
  internally after first build; rebuilt only on graph topology
  changes (rare).
- **NEW** `PwlStateSpaceCache::refactor_parametric(param_names,
  new_values, masks="active")` API:
  - Re-stamps only the affected columns of every active mask's
    cached $J$.
  - Invokes `solver.partial_refactor(new_J, affected_cols)` for
    each.
  - Returns a `ParametricRefactorResult` with per-mask
    success/fallback counts.
- **NEW** Python `pulsim.sweep.sweep_path_aware(builder, params,
  values, t_end, dt, ...)` helper: drop-in replacement for
  `pulsim.sweep.sweep(...)` that exploits the parametric refactor
  internally. Falls back transparently to the legacy
  `analyze + factorize` per-point path when:
  - The parameter being swept is not in the
    `DevicePool::columns_affected_by_param` table (e.g. a topology
    change, not a value change), OR
  - The path-union grows above the same `MAX_PATH_LENGTH_RATIO`
    threshold.
- **NOT changed**: the existing `pulsim.sweep.sweep(...)` and
  `pulsim.sweep.monte_carlo(...)` APIs continue to work. The
  new path-aware variant is **opt-in** initially; once stable
  (one minor version cycle) it becomes the default.

### Part C — Benchmarks and paper-bound artefacts

- **NEW** `core/tests/benchmarks/test_bench_multi_bit_rank1.cpp`:
  microbench covering 1-bit, 2-bit, 3-bit Hamming-distance
  transitions on an N-switch fixture. Captures speedup vs
  fall-back-to-factorise baseline.
- **NEW** `core/tests/benchmarks/test_bench_parametric_sweep.cpp`:
  microbench covering parameter sweeps of 50, 100, 500, 1000
  points on the 10 reference projects. Captures wall-clock
  speedup of `refactor_parametric` vs the per-point
  `analyze + factorize` baseline.
- **NEW** `artigos/02_tpel_methods/benchmarks/MULTI_BIT_RESULTS.md`
  + `PARAMETRIC_RESULTS.md`: writeups in the same style as the
  existing `RANK1_RESULTS.md`. Both CSVs committed under
  `artigos/02_tpel_methods/benchmarks/results/`.
- **MODIFIED** `docs/how-pulsim-works/08-benchmarks.md`: extend
  the 3-backend table to 5 columns (add multi-bit and
  parametric columns); update Fig 8.1 / 8.2 / 8.3 to the new
  data; add Figs 8.5 (parametric speedup curve) and 8.6
  (multi-bit Hamming-distance sensitivity).
- **NOT changed**: chapters 1-7 remain valid (the algorithm
  story extends naturally; no rewrites needed).

## Impact

- **Affected specs**:
  - `pulsim-sparse-lu` — MODIFIED requirement (partial_refactor
    handles arbitrary-length changed_cols cleanly with path-union
    semantics)
  - `pwl-rank1-update` — MODIFIED requirement (solve_rank1 routes
    multi-bit transitions to partial_refactor instead of full
    factorize); ADDED requirement (parametric refactor API)
  - `python-bindings` — ADDED requirement (`pulsim.sweep.sweep_path_aware`)
- **Affected code**:
  - `core/include/pulsim/sparse/pulsim_lu_solver.hpp` — already
    supports the union case via `varying_set_`; this proposal
    adds the *per-call* multi-bit codepath and tightens the
    path-length-fallback heuristic
  - `core/include/pulsim/pwl/cache.hpp` — `solve_rank1`
    multi-bit routing; new `refactor_parametric` API
  - `core/include/pulsim/pwl/device_pool.hpp` — new
    `columns_affected_by_*` helpers
  - `python/pulsim/sweep.py` — new `sweep_path_aware` helper
  - `core/tests/benchmarks/` — two new bench files
  - `artigos/02_tpel_methods/benchmarks/` — two new writeups
  - `docs/how-pulsim-works/08-benchmarks.md` — extension
- **Target release**: v1.5.0 (after `add-pulsim-complex-sparse-lu`
  ships in v1.4.0; both proposals are independent and can be
  developed in parallel)
- **Scope estimate**: ~3-4 weeks. Part A is small (~1 week) since
  most machinery exists. Part B is the bulk (~2 weeks) — new
  DevicePool tracking + Python integration + benchmarks.

## Out of scope

- GPU-parallel path-union refactor (would help when path is wide;
  out of scope as v1.x is single-threaded by design)
- Adaptive `PIVOT_THRESH` per sweep regime (deferred to a separate
  `add-adaptive-pivot-threshold` proposal)
- Path-based refactor for the *complex* solver (handled by the
  separate `add-pulsim-complex-sparse-lu` proposal — that one
  templates `PulsimSparseLuSolver` on Scalar; once both proposals
  ship, complex AC-sweep workloads get parametric updates for
  free)
- Cross-builder symbolic cache reuse (Monte Carlo on different
  topologies sharing analyze) — niche, deferred
- BTF block-triangular form integration (would compose with the
  path-union framework cleanly, but adds an orthogonal algorithm;
  deferred to `add-btf-block-triangular-ordering`)
