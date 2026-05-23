# Design — `pulsim-v2-lazy-cache-build` (Layer 4 V6)

## Why lazy?

For a 2^N combinatorial space with M actually-visited masks,
the build cost is `2^N · cost_factor`. If M << 2^N (typical
for PWM-controlled converters), the eager build wastes
`(2^N − M) · cost_factor`.

Lazy build flips this: pay `M · cost_factor`. The first call
for each unique mask is slow; subsequent calls are fast.

Memory savings: same ratio. A 6-switch circuit has 64 segments
× ~10 KB each = 640 KB cached. Lazy build with M=12 actually-
visited masks = 120 KB. For larger circuits this matters more.

## API

```cpp
class PwlStateSpaceCache {
public:
    // ... existing methods ...

    /// Lazy build — stores dt but does NOT factorise any
    /// segment. The first call to solve(mask, ...) for each
    /// mask builds-and-caches the factor on demand.
    void build_lazy(Real dt);

    /// Number of segments that have been physically built.
    /// For build(dt) (eager): equals 2^N immediately.
    /// For build_lazy(dt): grows as new masks are visited.
    Size num_built_segments() const noexcept;
};
```

`build()` (eager) and `build_lazy()` are mutually exclusive —
calling `build_lazy()` after `build()` clears the eager-built
segments and starts fresh.

## solve() with lazy support

```cpp
void solve(const SwitchStateMask& mask, ...) const {
    auto it = segments_.find(mask);
    if (it == segments_.end()) {
        // Not built yet — build on demand.
        if (!lazy_mode_) {
            throw std::out_of_range("lookup miss in eager mode");
        }
        // Mutate cache from a const method via `mutable`.
        build_one_segment(mask);
        it = segments_.find(mask);
    }
    // Standard solve path.
    ...
}
```

The `mutable` storage pattern preserves the const-correctness
of `solve` from the caller's perspective: solving doesn't
logically change the cache (the user sees the same `(mask, x)`
outputs), but it can populate the cache lazily.

## Test

The boost converter test from Layer 5 V2 visits exactly 2 of
4 possible switch states (PWM with no overlap). With lazy
build:
- Initial `num_built_segments() == 0`.
- After running run_transient, `num_built_segments() == 2`
  (only the two visited masks).

That's the V0 verification.

## Future work: pre-warm hints

V1 could add a `pre_warm(masks_set)` method that asks the cache
to build the listed masks NOW (in parallel?), so the first
simulation step doesn't pay the factor cost. Useful for
real-time simulations where build-on-first-use would add
unacceptable latency.

## Why NOT Sherman-Morrison

The original ask was for SM rank-1 updates between Gray-code-
adjacent switch states. The math is well-known:
`(A + uvᵀ)⁻¹ = A⁻¹ − (A⁻¹uvᵀA⁻¹) / (1 + vᵀA⁻¹u)`.

The blocker: SM updates the INVERSE, but our cache stores LU
FACTORS via KLU/COLAMD/SparseLU. KLU doesn't expose factor-
update primitives. Implementing SM at the factor level would
require:
1. Solving with the current factor for `u` to get `A⁻¹u`.
2. Computing the rank-1 outer product `(A⁻¹u)·(vᵀA⁻¹) / (1 + vᵀA⁻¹u)`.
3. Subtracting from `A⁻¹` to get the new inverse.

Step 3 stores the new INVERSE (dense), not factors. For sparse
matrices, this destroys the LU's sparsity benefit. There exist
specialized libraries (Cholmod for SPD, custom non-symmetric
update libs) that maintain factors under rank-1 updates, but
integrating one is a significant infrastructure change.

Lazy build achieves the SAME effective speedup (skip unused
factors) without changing the factor backend. It's the
pragmatic V0.
