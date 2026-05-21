# Design — `pulsim-v2-multi-dt-cache` (Layer 4 V7)

## API

```cpp
class PwlStateSpaceCache {
public:
    // Existing single-dt path (unchanged).
    void build(Real dt);
    void build_lazy(Real dt);
    void solve(const SwitchStateMask& mask,
                const Vector& b_extra, Vector& x) const;

    // V7: multi-dt support.

    /// Solve with the cache's PRIMARY dt if `dt == this->dt()`,
    /// else use an auxiliary cache keyed on dt. The auxiliary
    /// segment is built on demand (first time the (mask, dt)
    /// pair is requested).
    void solve_at(const SwitchStateMask& mask, Real dt,
                   const Vector& b_extra, Vector& x) const;

    /// Number of distinct dt values that have factors in the
    /// auxiliary cache. Excludes the primary dt (which is
    /// reported via num_built_segments()).
    [[nodiscard]] Size num_alt_dt_values() const noexcept;

    /// Number of segments factored at the given auxiliary dt.
    /// Returns 0 if the dt has no cached segments.
    [[nodiscard]] Size num_alt_segments_at(Real dt) const noexcept;
};
```

## Internal storage

```cpp
private:
    // Primary single-dt cache (V0-V6).
    std::unordered_map<SwitchStateMask, PwlSegment> segments_;
    Real dt_ = 0;
    bool lazy_mode_ = false;

    // Auxiliary multi-dt cache (V7).
    // Nested map: dt → (mask → segment).
    std::unordered_map<Real, std::unordered_map<
        SwitchStateMask, PwlSegment>> alt_segments_;
};
```

The auxiliary cache is keyed on `dt` as the outer key, with the
mask map inside. Lookups: `alt_segments_[dt][mask]`.

## solve_at logic

```cpp
void solve_at(const SwitchStateMask& mask, Real dt,
               const Vector& b_extra, Vector& x) const {
    if (dt == this->dt()) {
        // Primary cache — delegate to existing solve().
        solve(mask, b_extra, x);
        return;
    }
    // Auxiliary cache.
    auto& bucket = alt_segments_[dt];   // creates if missing
    auto it = bucket.find(mask);
    if (it == bucket.end()) {
        // Build segment at this (mask, dt).
        ... factor + insert ...
    }
    const PwlSegment& seg = bucket.at(mask);
    Vector rhs = -(seg.b_constant + b_extra);
    seg.solver->solve(rhs, x);
}
```

`mutable` qualifier on `alt_segments_` preserves `const`-
correctness from the caller's POV (logically, solving doesn't
mutate cache contents that any user can observe; we just
populate hidden state).

## How this enables sub-step state correction

Once `solve_at` exists, a sub-step state corrector in
`run_transient` can:

1. Detect a commutation between t_n and t_n+1.
2. Bisect/interpolate t*.
3. Call `solve_at(old_mask, t* - t_n, b_extra_partial, x_at_tstar)`.
4. Switch mask to new_mask.
5. Call `solve_at(new_mask, t_n+1 - t*, b_extra_remaining, x_at_t_n+1)`.

Each unique partial-dt value pays its factor cost once (per
mask). For a PWM converter with fixed commutation phases, the
partial-dt values repeat → cache hit rate is high.

V0 of this OpenSpec ships only the primitive. The state-
corrector itself is a future follow-up OpenSpec.

## Memory considerations

Each `alt_segments_[dt]` entry can hold up to 2^N segments. If
the user calls `solve_at` with many distinct dt values, memory
grows by O(2^N · #distinct_dts).

For a 6-switch circuit (64 segments) × 10 distinct partial-dts
× ~10 KB per segment = 6.4 MB. Acceptable for typical PE
simulations.

V0 doesn't add eviction. Future tuning can add LRU eviction
if memory becomes a concern.

## Why not "rebuild per commutation, no cache"

We COULD just rebuild the partial-dt segment on every
commutation event, discarding after use. That avoids the
memory growth.

But for PWM converters, the same partial-dt values recur
EVERY cycle (the commutation timing is locked to the PWM
phase). Caching across cycles is essentially free in memory
and massively cheaper in time.

V0 ships the cache; future tuning can add eviction or
"opportunistic discard" heuristics.
