## Why

The linear-solver factorization cache in `core/src/v1/transient_services.cpp:35-49` (`hash_sparse_numeric_signature`) uses an O(nnz) hash that mixes the **numeric values** of the Jacobian into the cache key. Because Jacobian numeric values change every Newton iteration (and every step), the hash effectively never matches — defeating the purpose of caching. Compounding this, [transient_services.cpp:207](core/src/v1/transient_services.cpp:207) allocates a fresh `SparseMatrix(n,n)` on every `build_model` call, violating the kernel's own "allocation-bounded steady-state stepping" requirement (`openspec/specs/kernel-v1-core/spec.md`).

KLU and Eigen `SparseLU` separate symbolic factorization (depends on sparsity pattern only) from numeric factorization (depends on values). For circuits with stable topology — the dominant case in switching converters once Phase-0 PWL engine lands — the symbolic factorization is reusable across thousands of steps. Numeric factorization itself can be reused when matrix values are unchanged (e.g., stable PWL topology between events).

The fix: re-key the cache on `(sparsity_pattern_hash, topology_bitmask)`, store factorizations per-topology, pre-allocate working buffers, and expose telemetry for cache effectiveness.

## What Changes

### Cache Key Decomposition
- **Sparsity-pattern hash**: cheap O(nnz) hash over `(row, col)` pairs only, no values. Computed once per topology.
- **Topology bitmask**: `std::uint64_t` (or `boost::dynamic_bitset` for >64 switches) over PWL-mode device states.
- **Numeric digest**: separate, optional fast hash of values for sanity-check or AD-validation invalidation. Not used as primary key.
- Cache layered as: `sparsity_pattern → symbolic factor`, `(sparsity, topology) → numeric factor`.

### Symbolic Factor Reuse
- KLU `klu_analyze()` runs once per sparsity pattern; result cached in `Simulator` lifetime.
- Eigen `SparseLU::analyzePattern()` similarly.
- Reused across Newton iterations within a topology and across steps within a topology.

### Numeric Factor Reuse for Stable PWL Topology
- When `Ideal` mode resolves and topology bitmask unchanged across steps, numeric factor reused.
- Invalidation triggers: topology change, source-stepping ramp, Gmin escalation, parameter change (rare).

### Pre-Allocation of Working Buffers
- `Simulator` ctor allocates `SparseMatrix jacobian_workspace_` and `Vector residual_workspace_` sized to circuit at construction.
- All segment-model builds and Newton iterations write into these buffers; no per-step allocation.
- Tracked in telemetry via `BackendTelemetry.time_series_reallocations` (existing field).

### Deterministic Invalidation Reasons
- Each cache invalidation tagged with reason: `topology_changed | stamp_param_changed | gmin_escalated | source_stepping_active | numeric_instability | manual_invalidate`.
- Reasons surfaced in `LinearSolverTelemetry.cache_invalidations` (list of records).

### Cache Bounds and Eviction
- Symbolic cache: bounded by distinct sparsity patterns (typically 1–2 for a circuit).
- Numeric cache: bounded by `simulation.linear_factor_cache_max` (default 4096), LRU evicted.
- Eviction emits telemetry counter `linear_factor_cache_evictions`.

## Impact

- **Affected specs**: `linear-solver`.
- **Affected code**: `core/include/pulsim/v1/transient_services.hpp`, `core/src/v1/transient_services.cpp`, `core/include/pulsim/v1/solver.hpp` (RuntimeLinearSolver), `core/src/v1/simulation.cpp` (workspace allocation in ctor).
- **Performance**: ≥3× speedup on Newton-iteration-heavy paths (Behavioral-mode switching circuits). Compounds with PWL engine for total ≥30× target on switching benchmarks.
- **Memory**: bounded by `linear_factor_cache_max`; default ≤16 MB on 4096 entries × 4 KB avg factor footprint.

## Success Criteria

1. **Cache hit rate**: ≥95% on stable-topology windows (post-warmup) measured via `linear_factor_cache_hits / total_solves`.
2. **No per-step allocation**: zero `SparseMatrix` heap allocations in the steady-state loop, verified by malloc tracker in tests.
3. **Performance**: ≥3× speedup on `mosfet_buck.yaml` and `boost_switching_complex.yaml` benchmarks vs current cache-broken baseline.
4. **Determinism**: identical invalidation trace across reruns of same netlist on same hardware.
5. **Backward compat**: existing tests pass without changes; default behavior more efficient, never less.
