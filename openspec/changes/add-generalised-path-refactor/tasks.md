## 1. Part A — Multi-bit path-union in `partial_refactor`

- [ ] 1.1 Audit the existing `PulsimSparseLuSolver::partial_refactor`
      to confirm the per-call multi-bit case works correctly when
      `changed_cols.size() > 1`. The `varying_set_` already
      accumulates across calls; the per-call multi-bit path needs:
      - `compute_path_()` walks all `changed_cols` and dedupes via
        `in_path` bitmap → produces the union path
      - Path columns are processed ascending (already correct)
      - Pivot-threshold check still works per column (already correct)
- [ ] 1.2 Add new public method
      `PulsimSparseLuSolver::partial_refactor_count_path()` that
      returns the **length of the union path** without executing
      the refactor. Used by Part A.3's cost-vs-fallback gate.
- [ ] 1.3 Add tunable `MAX_PATH_LENGTH_RATIO` constant (default 0.6)
      in `pulsim_lu_solver.hpp`. If `path_length / n > ratio`, the
      caller is signalled to fall back to full `factorize()` rather
      than paying for a near-full path-walk that's no cheaper.
- [ ] 1.4 New unit-test cases in
      `core/tests/layer0/test_pulsim_lu_solver.cpp`:
      - 1.4.1 `partial_refactor` with 2-col `changed_cols` produces
        same solve output as fresh-factorise, within $10^{-12}$
      - 1.4.2 `partial_refactor` with 4-col `changed_cols` (k = n/2)
        same
      - 1.4.3 Path-union length is monotonic in `|changed_cols|`
        (more changes → at-least-as-long path)
      - 1.4.4 `partial_refactor_count_path()` returns
        non-zero, ≤ n, and matches the actual path length used
        on the next `partial_refactor` call

## 2. Part A — Multi-bit routing in `PwlStateSpaceCache::solve_rank1`

- [ ] 2.1 Refactor `solve_rank1(mask, b_extra, x)` to compute the
      Hamming distance `delta_bits` between `mask` and the
      previous mask:
      - `delta_bits == 0` → same mask, just refresh `b_constant`
        and triangular solve (existing fast path)
      - `delta_bits == 1` → single-bit, existing path-based call
      - `delta_bits > 1` → check `partial_refactor_count_path()`;
        if union path is ≤ `MAX_PATH_LENGTH_RATIO × n`, call
        `partial_refactor(new_J, changed_cols)`; otherwise fall
        back to full `factorize()`
- [ ] 2.2 New `DevicePool::columns_affected_by_switch(sw_id)`
      helper: given a switch id, return the column indices in $J$
      whose values change when that switch toggles. Walk
      `assemble.hpp::stamp_switch_*` to extract the affected cols.
      Result cached per (graph, sw_id).
- [ ] 2.3 Extend `CacheMetrics` with two counters:
      - `single_bit_rank1_hits` (rename of current `rank1_hits`,
        with backward-compat alias)
      - `multi_bit_rank1_hits` (NEW)
      Update the invariant:
      `single_bit + multi_bit + full_refactor + fallbacks == total`
- [ ] 2.4 Update `core/tests/layer4/test_pwl_cache_rank1.cpp`:
      - 2.4.1 Add a test case that uses a 2-bit Gray-code
        transition (mask 0b00 → 0b11). Assert `multi_bit_rank1_hits
        == 1` and `solve_rank1`'s output matches fresh-factorise
        within $10^{-12}$.
      - 2.4.2 Stress test: 64-mask sweep with random Hamming
        distances ∈ {1, 2, 3, 4}. Assert (a) solve outputs match
        fresh-factorise within $10^{-10}$ for every step; (b)
        sum of all counters = total step count; (c)
        `multi_bit_rank1_hits > 0` (path-based engaged on at least
        some multi-bit transitions).

## 3. Part B — `DevicePool::columns_affected_by_param`

- [ ] 3.1 New method
      `DevicePool::columns_affected_by_param(const std::string&
      param_name)` returning `std::set<Index>`. Parameter names
      are the builder-time handles (e.g. `"L_out"`, `"R_load"`,
      `"C_in"`). The method walks all stored device-param structs
      and asks each "which columns of $J$ do you contribute to?";
      this is the per-device equivalent of the existing
      `stamp_device<...>` dispatch but inverted (col → device →
      param vs param → device → col).
- [ ] 3.2 Implementation strategy: every device kind exposes a
      compile-time-constant `affected_columns(branch_endpoints)`
      static method. The pool aggregates these per registered
      device. Result is cached in
      `cols_by_param_: unordered_map<string, vector<Index>>` and
      invalidated only when the graph topology changes (not when
      values change — which is the whole point).
- [ ] 3.3 Edge cases:
      - Parameter name not in pool → returns empty set (caller
        is signalled to fall back to full re-analyse)
      - Parameter name appears in multiple devices (e.g. shared
        coupled-inductor `L_m`) → union of all affected cols
- [ ] 3.4 Unit tests in `core/tests/layer2/test_device_pool.cpp`:
      - 3.4.1 Buck fixture: `columns_affected_by_param("L_out")`
        returns the 2 columns for the inductor branch-current row
        + its endpoint conductance.
      - 3.4.2 Unknown param → empty set.
      - 3.4.3 Cache invariance: 1000 calls with the same param
        name compute the affected set exactly once (verified
        via internal hit counter).
      - 3.4.4 Topology change (add a new device) → cache is
        invalidated; next call recomputes.

## 4. Part B — `PwlStateSpaceCache::refactor_parametric` API

- [ ] 4.1 New method signature:
      ```cpp
      struct ParametricRefactorResult {
          std::size_t   masks_processed;     // active masks
          std::size_t   path_refactor_hits;  // ✓ via partial_refactor
          std::size_t   fallback_hits;       // had to factorize()
          double        wall_time_us;
      };

      ParametricRefactorResult
      PwlStateSpaceCache::refactor_parametric(
          std::span<const std::string> param_names,
          std::span<const Real>        new_values,
          /*mask_filter*/ Mode mode = Mode::AllActive
      );
      ```
      `param_names.size() == new_values.size()` precondition.
- [ ] 4.2 Implementation steps:
      1. Update each parameter via `pool_.update_param(name, value)`.
         (New `update_param` shim added to `DevicePool`; ~30 lines
         of `std::visit` dispatch over the device variant.)
      2. Build `affected_cols = union(columns_affected_by_param(p))`
         for each param `p`.
      3. For each `(mask, segment)` in `segments_`:
         - Re-stamp `J` for that mask at the changed columns only
           (cheap: ~5 entries per column on average)
         - If `affected_cols.size() / n_state > MAX_PATH_LENGTH_RATIO`,
           call `segment.solver.factorize(new_J)` and increment
           `fallback_hits`.
         - Otherwise call
           `segment.solver.partial_refactor(new_J, affected_cols)`
           and increment `path_refactor_hits` (or `fallback_hits`
           on `false` return).
- [ ] 4.3 `Mode::AllActive` (default) processes every cached
      segment. `Mode::CurrentOnly` processes just the most-recent
      segment from `solve(...)` calls — useful for hot-loop
      parameter perturbations (e.g. a control study where one
      gain trims live).
- [ ] 4.4 Unit tests in
      `core/tests/layer4/test_pwl_cache_parametric.cpp` (NEW):
      - 4.4.1 Single-param sweep `L_out ∈ [50µH, 200µH]`, 10
        points. After each `refactor_parametric` call, the
        cache's `solve(mask, ...)` output matches a fresh
        `analyze + factorize + solve` within $10^{-10}$.
      - 4.4.2 Two-param simultaneous change `(L_out, C_out)`. Same
        parity check.
      - 4.4.3 Parameter the pool doesn't know about → result has
        `fallback_hits > 0` for every affected mask.
      - 4.4.4 1000-point sweep timing: total wall time of the
        sweep < (1000 × per-point hot-path time) + (single cold-
        path build). I.e. the per-point cost is path-bounded, not
        re-analyze-bounded. Captured at the test-binary level as
        a wall-clock assertion.

## 5. Part B — Python `pulsim.sweep.sweep_path_aware` helper

- [ ] 5.1 New function in `python/pulsim/sweep.py`:
      ```python
      def sweep_path_aware(
          builder: CircuitBuilder,
          param_name: str,
          values: Sequence[float],
          t_end: float,
          dt: float,
          *,
          switch_fn: Optional[Callable] = None,
          **kwargs,  # forwarded to pp.simulate
      ) -> SweepResult:
          ...
      ```
      Builds a single `PwlStateSpaceCache` upfront, then iterates
      `values` calling `cache.refactor_parametric([param_name], [v])`
      between simulation runs. Returns the same `SweepResult` shape
      as the legacy `sweep(...)` so callers can swap without
      breakage.
- [ ] 5.2 New 2-param + N-param overloads via
      `sweep_path_aware_nd(builder, params_dict, ...)` accepting
      a Cartesian product or a callable that yields parameter
      tuples (for Monte Carlo).
- [ ] 5.3 Auto-fallback semantics: if `pool.columns_affected_by_param(p)`
      returns empty (param unknown), the helper logs a warning
      and silently routes to the legacy `sweep(...)`. No user-
      facing breakage.
- [ ] 5.4 Pybind11 wiring: expose
      `PwlStateSpaceCache.refactor_parametric` +
      `DevicePool.columns_affected_by_param` to Python via
      `python/src/bindings_pwl.cpp`. Add Python unit tests in
      `python/tests/test_sweep_path_aware.py`.

## 6. Part B — Monte Carlo helper

- [ ] 6.1 Add `pulsim.sweep.monte_carlo_path_aware(builder,
      params_distributions, n_samples, t_end, dt, ...)` —
      drop-in replacement for `pulsim.sweep.monte_carlo` that
      uses the parametric-refactor path. Each sample draws from
      the distributions, calls `refactor_parametric(all_params,
      drawn_values)`, then runs the transient.
- [ ] 6.2 Unit tests:
      - 6.2.1 1000-sample Monte Carlo on buck with
        `R_DS_on ~ Normal(20mΩ, 5mΩ)` and
        `L_out ~ Uniform(95µH, 105µH)`. Compare wall time of
        path-aware vs legacy. Assert path-aware ≥ 10× faster.
      - 6.2.2 KPI parity: 1000-sample MC, compute mean output
        voltage. The two paths should agree to 5 significant
        digits on the mean (verifying numerical equivalence).

## 7. Part C — Benchmarks

- [ ] 7.1 `core/tests/benchmarks/test_bench_multi_bit_rank1.cpp`
      (NEW): mirror the structure of
      `test_bench_pwl_rank1.cpp`. Same N-switch fixture, but
      drive the switch mask through transitions of Hamming
      distance ∈ {1, 2, 3, 4}. Capture per-Hamming-distance
      µs/call across N ∈ {8, 12, 16, 20, 24}. Output CSV.
- [ ] 7.2 `core/tests/benchmarks/test_bench_parametric_sweep.cpp`
      (NEW): for each of the 10 reference converter projects (or
      a representative subset: buck, NPC, MMC), sweep one
      well-known parameter through 100 points. Compare wall-clock
      of (a) the legacy `analyze + factorize`-per-point and (b)
      the new `refactor_parametric`. Output CSV with per-converter
      speedups.
- [ ] 7.3 Capture both benchmarks on macOS / Apple Silicon (the
      same harness used by chapter 8's existing microbench).
      CSVs land in
      `artigos/02_tpel_methods/benchmarks/results/multi_bit_microbench.csv`
      and `.../parametric_microbench.csv`.
- [ ] 7.4 Honest-limitations check: if any captured speedup is
      below 1× (i.e. path-based loses), document the cause in
      the writeup. Likely candidates: very small $n$ (path-
      construction overhead) or very large Hamming distance
      (path-union approaches full).

## 8. Part C — Paper artefacts

- [ ] 8.1 `artigos/02_tpel_methods/benchmarks/MULTI_BIT_RESULTS.md`
      (NEW): full writeup in the same style as `RANK1_RESULTS.md`.
      Sections: backends, captured table, per-Hamming-distance
      analysis, when does path-based lose, reproducibility recipe.
- [ ] 8.2 `artigos/02_tpel_methods/benchmarks/PARAMETRIC_RESULTS.md`
      (NEW): same structure for the parametric sweep + Monte
      Carlo. Includes a wall-clock-vs-sweep-size scaling figure.
- [ ] 8.3 Update `docs/how-pulsim-works/08-benchmarks.md`:
      - Extend the captured 3-backend table to a 5-row variant
        (single-bit / 2-bit / 3-bit / parametric-sweep / MC).
      - Add Fig 8.5 (parametric speedup curve) + Fig 8.6
        (multi-bit Hamming-distance sensitivity).
      - Update Fig 8.4 (pivot-fallback heatmap) to include the
        multi-bit + parametric workloads.
- [ ] 8.4 Update `docs/how-pulsim-works/07-rank1-partial-refactor.md`
      §7.7 ("What the algorithm does NOT do") to:
      - Remove the "multi-bit falls back" caveat (now handled)
      - Add the new caveat: "if `path/n > MAX_PATH_LENGTH_RATIO`,
        we still fall back" (less restrictive, still honest)
- [ ] 8.5 Update `docs/how-pulsim-works/10-paper-figures-index.md`
      with the 2 new figures + their paper-section mappings.
- [ ] 8.6 Update `artigos/02_tpel_methods/paper.md` (TPEL paper
      draft) §VI table to reflect the 5-row decomposition.
      §I.B is also updated to position the contribution as
      "generalised path-based update framework" rather than
      "implementation of Dinkelbach 2021".

## 9. Close-out + release

- [ ] 9.1 Bump version 1.4.0 → 1.5.0 in `pyproject.toml`,
      `python/pulsim/__init__.py`, `CITATION.cff`. (Assumes
      v1.4.0 = `add-pulsim-complex-sparse-lu` ships first; if
      this lands before v1.4.0, the bump is 1.3.x → 1.5.0.)
- [ ] 9.2 CHANGELOG `[1.5.0]` entry: highlights the generalised
      framework, includes the captured speedup table, references
      `MULTI_BIT_RESULTS.md` + `PARAMETRIC_RESULTS.md`.
- [ ] 9.3 `openspec validate add-generalised-path-refactor --strict`
      passes.
- [ ] 9.4 Open PR `feat/generalised-path-refactor` → main.
      Title: `feat(v1.5.0): generalised path-based updates —
      multi-bit + parametric`.
- [ ] 9.5 Post-merge: archive the change to
      `openspec/changes/archive/YYYY-MM-DD-add-generalised-path-refactor/`.

## Out of scope (future proposals)

- `add-btf-block-triangular-ordering` — would compose with the
  path-union framework (within-block path-based update) for an
  additional ~2× on the worst-case multi-bit workloads. Separate
  proposal because BTF is a substantial orthogonal algorithm.
- `add-adaptive-pivot-threshold` — PIVOT_THRESH currently fixed at
  $10^{-3}$. Adapt per-circuit based on observed pivot-magnitude
  distribution.
- `add-pwl-cache-cross-instance-symbolic-sharing` — Monte Carlo
  often spawns many `PwlStateSpaceCache` instances over the same
  topology. A global symbolic cache (etree + RCM) keyed by graph
  hash could amortise the symbolic phase across instances. Niche
  but useful for very-large-MC workloads (n_samples ≥ 10k).
- Speculative pre-factorisation parallel to the current solve
  (would help in real-time HIL contexts). Out of scope while v1.x
  is strictly single-threaded by design.
