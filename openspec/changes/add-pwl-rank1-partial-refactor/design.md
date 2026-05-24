## Context

V0 (commit `e9707a3`, v1.2.0) shipped `KluSolver::partial_refactor`
as an MVP that delegates to `klu_refactor` — full numeric refactor
with the cached symbolic ordering. That gave **3.15× at n=14** in
the microbench but cannot deliver the spec's "path-based partial
refactorisation" claim because upstream SuiteSparse KLU does NOT
expose the elimination-tree-aware re-elimination path.

The dpsim-simulator/SuiteSparse fork
(https://github.com/dpsim-simulator/SuiteSparse) implements
Schumacher/Dinkelbach's path-based variant on top of KLU,
released as LGPL-2.1+. The C source files (
`klu_compute_path.c`,
`klu_partial_factorization_path.c`,
`klu_partial_refactorization_restart.c`) extend `klu_numeric`
with `path`, `pathLen`, `block_path`, `variable_block`, and
`variable_offdiag_{orig,perm}_entry` fields, and add a public
`KLU_PATH_INVALID` error code.

The audit (2026-05-24) confirmed:
- Algorithm is published — Chan/Brandwajn/Tinney 1985-86,
  Abusalah/Mahseredjian/Karaagac/Kocar 2020 (OAJ-PE), Chen et al.
  2024 (IEEE TPEL doc 11018472), Dinkelbach et al. 2021
  (*Energies* 14:7989, full algorithm + benchmarks).
- The DPsim fork's C functions are the ONLY public open-source
  implementation atop KLU. Inspecting the source shows clean
  public signatures (no internal symbol leaks); the wrapper is
  straightforward.
- License is compatible: LGPL-2.1+ allows static linking from
  MIT-licensed code (Pulsim) provided we ship the LGPL'd object
  files. JOSS-style provenance preserved via `LICENSES/`.

## Goals / Non-Goals

**Goals**

- Replace `KluSolver::partial_refactor`'s `klu_refactor` delegation
  with `klu_partial_factorization_path`, delivering true O(path)
  per single-bit switch flip.
- Add the necessary path-precomputation lifecycle
  (`klu_compute_path`) with internal caching so the public
  `partial_refactor(M, changed_cols)` signature stays unchanged —
  V8.1 is a drop-in performance upgrade.
- Vendor the DPsim fork via FetchContent so users don't need to
  install a special KLU build.
- Preserve the V0 contract verbatim: no breaking changes, no new
  public methods, no behaviour change visible to existing callers
  (other than speedup).
- Honour the pivot-fault semantics: when `klu_partial_factorization_path`
  reports `KLU_PIVOT_FAULT`, `partial_refactor` returns `false`
  and invalidates the path cache (pivots may have shifted), so the
  caller's full-refactor fallback path engages cleanly.

**Non-goals**

- Wiring `solve_rank1` into Layer 5 `run_transient`. Same scope
  boundary as V0 — that remains a separate proposal
  (`add-pwl-rank1-runtime-integration`).
- Path-based MULTI-bit flips. The DPsim functions handle
  multi-column varying sets, but `solve_rank1` currently only
  triggers the partial path on single-bit Gray-code flips
  (multi-bit goes to full `factorize`). V8.1 inherits this
  constraint; widening to multi-bit is a future optimisation.
- Use of `klu_analyze_partial` (the fork's alternative analyzer
  that orders the matrix specifically for partial refactor). The
  audit suggests it could shorten the path for typical switching
  patterns, but it changes the symbolic-factor lifecycle in ways
  that would force `solve_rank1` and `solve` to use separate
  symbolic factors. Deferred to V8.2.
- Removing `Backend::Eigen` or the V0 fallback paths. They remain
  for `PULSIM_ENABLE_KLU=OFF` and small-n cases.

## Decisions

### D1. Vendor via FetchContent (vs system install + patch)

Three options were considered:

1. **System `libklu` + cherry-pick patch into Pulsim's tree.**
   Rejected: the new C functions need access to internal KLU
   helpers (static functions in `klu_factor.c`, etc.) that aren't
   exported from a stock `libklu.so`. Cherry-pick wouldn't link.
2. **CMake FetchContent of the DPsim fork.** Chosen. Pins to a
   specific commit SHA for reproducibility. Builds a static
   `libklu_dpsim` as part of the Pulsim build. Removes the
   per-platform install instruction. Adds ~3 MB download at first
   configure.
3. **Git submodule of the DPsim fork.** Rejected — adds friction
   for contributors (must remember `git submodule update --init`)
   and complicates CI cache strategy. FetchContent is the idiomatic
   modern CMake answer.

### D2. Backwards-compatible API: hide path-cache behind `partial_refactor`

The DPsim fork's three-phase API (`analyze` → `compute_path` →
`partial_factorization_path`) is more granular than our V0 API
(`analyze` → `factorize` → `partial_refactor(M, changed_cols)`).

Two ways to integrate:

1. **Expose the three phases as separate public methods** on
   `KluSolver`. Honest to the upstream API but breaks the V0
   `DirectSolver` contract and forces every caller (incl.
   `PwlStateSpaceCache::solve_rank1`) to grow a new path-cache
   management protocol.
2. **Keep `partial_refactor(M, changed_cols)` as the single
   public entry, hide path-caching inside `KluSolver`.** Chosen.
   On each call, hash `changed_cols`, look up in
   `path_cache_`, compute-and-cache on miss, then call
   `klu_partial_factorization_path`. No caller change.

The cache is bounded: for the PWL `solve_rank1` workload it holds
at most N entries (one per switch's single-bit flip). For other
callers it's bounded by the number of distinct change-sets they
exercise. Empirically tiny.

### D3. Pivot-fault handling

`klu_partial_factorization_path` returns `FALSE` with
`common_.status` set to either `KLU_SINGULAR` (zero pivot, circuit
truly degenerate) or `KLU_PIVOT_FAULT` (pivot below
`common_.pivot_tol_fail`, recoverable via re-pivoting).

Our `partial_refactor` returns `bool`. Mapping:

- `TRUE && status == KLU_OK` → return `true`
- `FALSE && status == KLU_SINGULAR` → return `false` (caller's
  full-refactor fallback path engages, but it too will fail —
  signals genuinely degenerate input)
- `FALSE && status == KLU_PIVOT_FAULT` → **invalidate the path
  cache** (the pivot order encoded in the precomputed path is
  now stale) and return `false`. Caller's `factorize` re-establishes
  fresh pivots, and the next `partial_refactor` re-builds the path.

Invalidating the path cache on pivot-fault is conservative — we
could try to invalidate only the affected path entries, but the
KLU primitives don't make that easy. Bulk invalidate keeps the
logic simple and correct.

### D4. Tightened test 2.7.3 — L/U bit-identical instead of x within 1e-12

V0 test 2.7.3 checks `x_partial ≈ x_full` within 1e-12. The
path-based algorithm is mathematically equivalent to full refactor
when pivots are unchanged, so the test can be stricter: extract L
and U via `klu_extract`, compare element-wise within 1e-14 (machine
epsilon × small constant). This documents the bit-exact equivalence
contract that the open-source patch maintains.

### D5. License + provenance

DPsim KLU patches: LGPL-2.1+. Compatible with Pulsim's MIT for
static linking provided:
- LGPL license text is shipped (we add `LICENSES/LGPL-2.1.txt`)
- Modifications to LGPL'd code are themselves LGPL (we do NOT
  modify the vendored KLU sources — we only link)
- A note in the README points users at the LGPL terms

JOSS-style provenance: `LICENSES/SuiteSparse-DPsim-fork.md` records
the upstream URL, the specific commit SHA we vendored, the date
captured, and a hash of the downloaded tarball. Audit trail for
the publication.

## Risks / Trade-offs

| Risk | Mitigation |
|------|------------|
| DPsim fork is unmaintained / archived. | Pin to a specific commit SHA we know works (recorded in design.md + LICENSES/). Worst case we fork-the-fork in pulsim-org and maintain it ourselves. |
| Path cache grows unbounded for callers with many distinct change-sets. | Bound the cache size with an LRU eviction policy. Default cap: 256 entries (plenty for any single circuit's distinct switch combos). |
| FetchContent download fails in air-gapped CI / offline build. | Add `-DPULSIM_KLU_FETCH_SOURCE=local` build flag that takes a local path to a pre-downloaded SuiteSparse tarball. Document in README. |
| The fork's `klu_compute_path` mutates `numeric_` in non-obvious ways, breaking the per-call `partial_refactor` semantics if we cache the path. | The cache stores OWNED COPIES of the path arrays (deep-copied from `numeric_` after `compute_path`); on each `partial_refactor` we re-install the cached path onto `numeric_` before calling `klu_partial_factorization_path`. Defensive, slightly more code, but isolates concerns. |
| Bumping symbolic factor (e.g. via a future `analyze` call) invalidates all cached paths. | `KluSolver::analyze` already clears `numeric_`; extend it to also clear `path_cache_`. Documented in the new `ensure_path_for` helper. |
| LGPL vs MIT licence confusion for users distributing Pulsim. | Add a clear paragraph in README + LICENSES/ explaining: (a) Pulsim's own code is MIT, (b) the linked-in KLU patches are LGPL-2.1+, (c) distributing a Pulsim binary requires shipping the LGPL license text. JOSS reviewers will check this. |

## Migration Plan

Pure additive at the API level — every existing `KluSolver`
consumer continues to work without code change. Existing
simulations produce identical output (within numerical tolerance)
and gain speedup automatically on single-bit flip workloads.

**Rollback:** revert the patch series. The previous V0 MVP
(`klu_refactor` delegate) re-engages. No data format changes, no
test regressions.

## Open Questions

- Q: Should the path cache be configurable (size cap, eviction
  policy) via a `KluSolver` ctor arg, or fixed-default 256?
  Tentative: fixed default for V8.1; add a setter if the
  microbench shows the cap is binding.
- Q: For multi-bit flips (rare path through `solve_rank1` since
  Gray-code enumeration guarantees single-bit transitions), should
  V8.1 still try path-based with the union of all changed columns?
  Tentative: NO for V8.1 — keep the V0 fallback (full `factorize`).
  Revisit in V8.2 with empirical data.
- Q: Should we expose path-cache stats via `metrics()` (hit rate,
  evictions, total paths computed)? Useful for the TPEL paper's
  benchmark attribution. Tentative: YES — add to `CacheMetrics`
  as `path_cache_hits` and `path_cache_misses`.
