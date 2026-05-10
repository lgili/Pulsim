# Linear-Solver Cache

> Status: shipped. Default-on. No flags to flip — every PWL run benefits.

Pulsim's segment-primary stepper (the PWL state-space fast path
introduced by `refactor-pwl-switching-engine`) keeps a per-key LRU cache
of analyzed-and-factorized linear solvers. When the simulator
re-encounters a `(topology, dt, parameters)` tuple it has already
factorized — for example, a buck converter cycling between Q-on and
Q-off at steady state — the cached factor is pulled instantly. Every
subsequent step pays one `solve(rhs)` and nothing else: no
`analyzePattern`, no `factorize`, no `SparseMatrix` allocation.

This document explains the cache structure, the telemetry surface, and
the (small) tuning knobs.

## TL;DR

- The cache is **default-on** and **invisible to user code**. PWL
  converters that ran before this change still run; they're just much
  faster.
- The headline number, measured on the 10-PWM-cycle buck benchmark
  (`test_linear_cache_phase6_benchmarks.cpp`): **351× wall-clock
  speedup** Behavioral mode → Ideal mode + cache, with **98.6 %**
  cache hit rate.
- The cache lives in the segment stepper; Newton-DAE has its own
  caching policy inside `RuntimeLinearSolver` and is unaffected.

## Why a per-key cache?

Two prior caches almost did the job:

1. **Symbolic-pattern cache** inside `EnhancedSparseLUPolicy` skipped
   `analyzePattern` when the sparsity pattern was unchanged.
2. **Single-slot numeric cache** in the segment stepper held the most
   recent factor for the current `(topology_signature, matrix_hash)`.

Both broke under power-electronics workloads. PWM cycling between two
topologies invalidates the single slot on every commutation: in (1) the
sparsity is unchanged but every step still re-factors; in (2) the slot
overflows so re-factoring is forced even when the *previous* topology's
factor is still numerically valid and would be reusable five steps
later.

The Phase-3 cache replaces the single slot with a small LRU map keyed on
the **value-aware matrix hash**. Two cache hits later, both topologies
sit in cache, and steady-state PWM looks like:

```
step 1:  build E → analyze → factorize → solve   (miss → entry A)
step 2:  build E → analyze → factorize → solve   (miss → entry B; topology flipped)
step 3:  build E → solve                         (HIT on A)
step 4:  build E → solve                         (HIT on B)
…
```

## Cache structure

`LinearFactorCache` lives in the anonymous namespace of
`core/src/v1/transient_services.cpp` — it's an implementation detail of
`DefaultSegmentStepperService`, not part of the public ABI.

```
key:     std::uint64_t  matrix_hash    // hash_sparse_numeric_signature(E)
entry:   { unique_ptr<RuntimeLinearSolver>  solver,
           shared_ptr<const SegmentLinearStateSpace>  matrix_holder }
storage: std::list<Entry>                 // front = MRU, back = LRU
index:   std::unordered_map<key, list_iter>
bound:   64 entries, LRU-evicted
```

Each entry holds:

- An independent `RuntimeLinearSolver` instance — it carries its own
  `analyzed_` flag plus the underlying `Eigen::SparseLU` (or KLU, etc.)
  factor in fully-baked state. A cache hit calls `solve(rhs)` directly;
  no symbolic or numeric work happens.
- A `shared_ptr<const SegmentLinearStateSpace>` keeping the matrix data
  alive. `RuntimeLinearSolver`'s fallback retry path can read
  `last_matrix_` for re-factorization on a different policy. The
  `shared_ptr` ensures that pointer never dangles.

### Why `matrix_hash` and not `(sparsity_hash, topology_signature)`?

In PWL mode the assembled `E = M + (dt/2)·N` is fully determined by
topology + dt + device parameters. A value-aware FNV-1a hash of `E`'s
non-zero entries collapses all three into one 64-bit key. Two distinct
topologies don't produce a colliding hash modulo astronomical
(~`2⁻⁶⁴`) FNV collision; tracking topology separately would add
defense-in-depth at zero algorithmic benefit.

For Newton-DAE / Behavioral mode the matrix changes per Newton iteration
so the hash is unique each step. The cache simply doesn't hit there —
wasted hash-table inserts, but no correctness issue. Newton-DAE
continues to rely on `EnhancedSparseLUPolicy`'s internal symbolic-only
reuse.

## Telemetry surface

`SimulationResult::backend_telemetry` exposes the cache observables:

| Counter | Meaning |
|---|---|
| `linear_factor_cache_hits` | Steps that pulled a cached factor and ran only `solve(rhs)`. |
| `linear_factor_cache_misses` | Steps that ran a fresh `analyze + factorize + solve` because no entry matched. |
| `linear_factor_cache_invalidations` | Total invalidation events recorded across the run (sum of typed counters below). |
| `linear_factor_cache_invalidations_topology_changed` | The previous step's topology bitmask differed. |
| `linear_factor_cache_invalidations_numeric_instability` | Matrix hash drifted within a stable topology, or a cached `solve` failed. |
| `linear_factor_cache_invalidations_stamp_param_changed` | Reserved for runtime parameter mutation. Currently 0. |
| `linear_factor_cache_invalidations_gmin_escalated` | Reserved for transient Gmin escalation. Currently 0. |
| `linear_factor_cache_invalidations_source_stepping_active` | Reserved for source-stepping homotopy. Currently 0. |
| `linear_factor_cache_invalidations_manual_invalidate` | Reserved for `Simulator::invalidate_linear_cache()`. Currently 0. |
| `linear_factor_cache_last_invalidation_reason` | String mirror of the most recent typed reason ("topology_changed", "numeric_instability", or empty). |
| `linear_factor_cache_last_invalidation_reason_typed` | The same value as a `CacheInvalidationReason` enum. Prefer this for programmatic inspection. |
| `symbolic_factor_cache_hits` | Reserved from the original Phase-2 single-slot design; the per-key LRU subsumes within-stepper symbolic reuse, so this counter stays 0 today. |

`SimulationResult::linear_solver_telemetry` aggregates analyze /
factorize / solve counts across both the shared linear-solve service
(Newton-DAE) and the segment stepper's LRU (segment-primary). The
`last_*` fields prefer the segment-primary path's most-recent values
when any segment-primary work happened during the run.

### Reading hit rate

```python
tel = result.backend_telemetry
total = tel.linear_factor_cache_hits + tel.linear_factor_cache_misses
hit_rate = tel.linear_factor_cache_hits / total if total else 0.0
print(f"PWL cache hit rate: {hit_rate:.1%}")
```

For the 10-cycle buck benchmark this prints `98.6 %`. For a 1000-step
passive RC it prints `99.6 %`.

## Invalidation reasons

`CacheInvalidationReason` (defined in
`core/include/pulsim/v1/transient_services.hpp`) is the typed
discriminator for why an accepted step did not reuse the previous step's
hot factor:

| Value | Fires when | Notes |
|---|---|---|
| `None` | The new step hit the LRU on the same or a previously-cached `(topology, dt)` tuple. | The default. Cache cycling through revisited entries lands here. |
| `TopologyChanged` | Previous step's topology bitmask or state size differs. | Set even if the new step subsequently hits the LRU on a previously-cached topology — this is informational. The cache still serves the request. |
| `NumericInstability` | Matrix hash drifted within an unchanged topology, or a cached `solve` failed and was discarded. | Bisection-induced fractional `dt` around VCSwitch commutations is the most common source in PWM converters. |
| `StampParamChanged` | Reserved. Future hook for runtime device parameter mutation. | Not exercised today. |
| `GminEscalated` | Reserved. Will fire when transient Gmin is bumped during recovery. | Tracked in the convergence-aids change. |
| `SourceSteppingActive` | Reserved. Source-stepping / homotopy ramp engaged. | Tracked in the convergence-aids change. |
| `ManualInvalidate` | Reserved. User / kernel forced a rebuild. | Public `Simulator::invalidate_linear_cache()` is a follow-up. |

The string mirror (`linear_factor_cache_last_invalidation_reason`) emits
the canonical `snake_case` form: `"topology_changed"`,
`"numeric_instability"`, etc. Code that already parses the string
continues to work.

## Tuning knobs

There aren't many — the cache is sized for the typical
power-electronics workload and the heuristic that picks `solve(rhs)`
versus `analyze + factorize + solve` is fully automatic.

### Capacity (compile-time, default 64)

```cpp
// core/src/v1/transient_services.cpp, anonymous namespace:
class LinearFactorCache {
    static constexpr std::size_t kDefaultCapacity = 64;
    …
};
```

Why 64 and not the originally-spec'd 4096?

- Each entry holds a fully-allocated `RuntimeLinearSolver` plus a
  `shared_ptr` to the underlying matrix. For a 100-node circuit that
  is ~12 KB per entry; 4096 entries would peak at ~50 MB. 64 entries
  is ~800 KB worst case.
- 64 covers every realistic converter: PWM topologies (2–4), resonant
  converters with up to ~12 segments, HB averaging with a handful of
  `dt` swaps. The LRU eviction kicks in only on pathological runs that
  produce a continuous stream of unique matrix hashes (e.g. adaptive
  `dt` paired with a chaotic switching pattern).

If you do hit eviction in production, that's a strong signal to
investigate the run rather than to crank the capacity. Check
`linear_factor_cache_misses` against the `linear_factor_cache_hits`
ratio — a steady-state hit rate below 70 % usually points at unstable
`dt` adaptation, not undersized cache.

### Solver choice (runtime, via `LinearSolverStackConfig`)

The cached `RuntimeLinearSolver` instances are constructed from the
shared service's `LinearSolverStackConfig`. Every `LinearSolverKind` —
`SparseLU`, `EnhancedSparseLU`, `KLU`, iterative variants — is
supported. The cache itself is policy-agnostic; pick the stack that
matches your circuit size.

```python
opts.linear_solver.order = [pulsim.LinearSolverKind.KLU,
                            pulsim.LinearSolverKind.SparseLU]
opts.linear_solver.allow_fallback = True
```

### Disabling the cache

There's no flag to turn the cache off. The cache *is* the fast path —
disabling it would mean reverting to per-step `analyze + factorize`,
which is what Behavioral mode (Newton-DAE) already does for nonlinear
devices. If you want to compare against the no-cache baseline, switch
the relevant devices to `SwitchingMode::Behavioral`; the cache stops
seeing those steps and the comparison is honest.

## Sample telemetry: the buck benchmark

Captured on AppleClang 17 / Release+LTO / Apple Silicon (M-series):

```
=== Phase 6 Buck benchmark (10 PWM cycles, dt=100 ns) ===

Behavioral mode (Newton-DAE, no segment cache)
  wall_seconds = 2.221
  total_steps  = 1063

Ideal mode (PWL state-space + numeric-factor LRU)
  wall_seconds       = 6.31 ms
  total_steps        = 1000
  cache_hits         = 986
  cache_misses       = 14
  cache_hit_rate     = 98.6 %
  invalidations      = 14 numeric_instability + 0 topology_changed (*)

  speedup vs Behavioral = 351×
```

(*) The 14 invalidations come from VCSwitch bisection-to-event
splitting at each PWM edge. Each edge produces a single fractional `dt`
that doesn't cycle back, so it counts as `numeric_instability` and
contributes one cache miss. Steady-state steps between edges share `dt`
and topology and hit.

## What's next

The Phase-3 LRU is the central deliverable; a few follow-ups remain
tracked but are not gating production use:

- **Heap-allocation zero-count assertion** (Phase 6.3): pinning the
  literal "no `SparseMatrix` allocation per step" contract via a
  custom allocator wrapper. The 351× speedup is incompatible with
  per-step heap pressure; a quantitative test is a hardening
  follow-up.
- **Configurable capacity**: surface `linear_factor_cache_max` through
  `SimulationOptions` and the YAML parser if any user actually hits
  eviction in practice.
- **Cross-entry symbolic reuse** (Phase 4 idea): two entries with the
  same sparsity but different values (e.g. same topology at two `dt`
  values) currently each pay their own `analyzePattern`. Sharing the
  symbolic factor across the LRU would shave the warmup cost — needs
  Eigen surgery to extract and reapply the elimination tree.
- **`Simulator::invalidate_linear_cache()`** public method, wired to
  the `ManualInvalidate` reason for users that mutate stamps mid-run.

## See also

- [`refactor-pwl-switching-engine`](../openspec/changes) — the segment
  engine that exposes the linear-solver hot path the cache lives on.
- [`backend-architecture.md`](backend-architecture.md) — broader linear
  solver stack (KLU, GMRES, fallback chains).
- [`performance-tuning.md`](performance-tuning.md) — SIMD, memory
  alignment, and other knobs.
