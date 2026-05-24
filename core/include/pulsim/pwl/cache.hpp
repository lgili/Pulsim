#pragma once

// =============================================================================
// Pulsim — Layer 4: PwlStateSpaceCache (the PLECS-killer)
// =============================================================================
//
// `pulsim-v2-pwl-state-space-cache` Phase 4.
//
// THE architectural pivot. For each switch combination Layer 1
// enumerates, pre-build the per-segment MNA matrix (Layer 3
// stampers + Layer 2 device models) and pre-factorise via Layer 0's
// DirectSolver. Per-step hot loop: hash-map lookup + triangular
// solve. NO assemble, NO factorize, NO Newton iteration per step.
//
// Expected speedup vs v1: 10-50× on PE workloads. This is THE
// architectural reason v2 will compete with PLECS.
//
// V0 scope: static-only circuits (Resistor + VoltageSource +
// Switch). Capacitor/Inductor (trapezoidal companion + history) and
// nonlinear devices (per-segment Newton) are V1+ extensions.

#include "pulsim/numeric/dense.hpp"
#include "pulsim/numeric/dictionary.hpp"
#include "pulsim/numeric/types.hpp"
#include "pulsim/pwl/assemble.hpp"
#include "pulsim/pwl/device_pool.hpp"
#include "pulsim/pwl/segment.hpp"
#include "pulsim/sparse/matrix.hpp"
#include "pulsim/sparse/solver.hpp"
#include "pulsim/topology/enumerator.hpp"
#include "pulsim/topology/graph.hpp"
#include "pulsim/topology/switch_state.hpp"

#include <atomic>
#include <bit>
#include <cstdint>
#include <expected>
#include <format>
#include <stdexcept>
#include <utility>
#include <vector>

namespace pulsim::pwl {

/// Telemetry counters for the rank-1 cache update path
/// (openspec/changes/add-pwl-rank1-update). Monotonically incremented
/// from inside `PwlStateSpaceCache::solve_rank1`. Use
/// `PwlStateSpaceCache::metrics()` to snapshot.
///
/// * `rank1_hits`         — calls where the new mask differed from the
///                          previous by exactly one bit AND the underlying
///                          solver supports partial refactorization, so the
///                          fast `partial_refactor` path engaged.
/// * `full_refactor_hits` — calls that hit a full re-factorisation, either
///                          because of a multi-bit mask change, a
///                          first-encounter mask, or a no-mask-change
///                          re-solve (e.g. time-varying source values).
/// * `fallbacks`          — calls where the rank-1 fast path was attempted
///                          but had to fall back to a full re-factor (e.g.
///                          backend without partial-refactor support, or a
///                          numerical singularity reported by
///                          `partial_refactor`).
struct CacheMetrics {
    std::uint64_t rank1_hits         = 0;
    std::uint64_t full_refactor_hits = 0;
    std::uint64_t fallbacks          = 0;
};

/// Structured error returned by the non-throwing `try_*` cache API
/// (Layer 4 V4 ergonomics, C++23). Lets tooling and the Python
/// frontend report singular-matrix failures with the offending mask
/// and dt attached, rather than parsing a string out of `what()`.
///
/// The throwing API (`build`, `lookup`) is unchanged and still
/// the canonical entry point — `try_*` is purely additive for
/// callers that prefer expected-style flow.
struct CacheError {
    enum class Kind {
        StructurallySingular,
        NumericallySingular,
        MaskNotBuilt,
    };
    Kind                       kind;
    topology::SwitchStateMask  mask;
    Real                       dt = Real{0};   // 0 for static-only

    [[nodiscard]] std::string what() const {
        const char* k =
            kind == Kind::StructurallySingular ? "structurally singular"
          : kind == Kind::NumericallySingular  ? "numerically singular"
                                                : "no segment built";
        return std::format(
            "PwlStateSpaceCache: {} for mask {} (dt={})",
            k, mask.to_string(), dt);
    }
};

class PwlStateSpaceCache {
public:
    PwlStateSpaceCache(const topology::Graph& graph,
                        const DevicePool& pool) noexcept
        : graph_{graph}, pool_{pool} {}

    /// Build all 2^N segments and pre-factorize each (V1 dt-aware).
    ///
    /// `dt > 0` enables dynamic-device stamping (Capacitor /
    /// Inductor trap companion). `dt = 0` (or the no-arg overload)
    /// skips dynamic devices and produces a V0-identical static
    /// build.
    ///
    /// Calling build(dt) twice with different dt CLEARs and
    /// rebuilds every factor (the matrix is dt-dependent).
    ///
    /// Time:   O(2^N · (assemble + analyze + factorize))
    /// Memory: O(2^N · (nnz(J) + nnz(L) + nnz(U)))
    void build(Real dt) {
        dt_ = dt;
        lazy_mode_ = false;
        segments_.clear();
        const Size num_switches = graph_.num_switches();
        for (auto mask :
             topology::enumerate_switch_states(num_switches)) {
            build_one_segment(mask);
        }
    }

    /// V0 backwards-compat overload — static-only build (dt = 0).
    /// Caps and Inductors are silently skipped.
    void build() { build(Real{0}); }

    /// Lazy build (Layer 4 V6). Stores dt but defers segment
    /// factorisation. Each `solve(mask, ...)` call builds the
    /// requested mask's factor the first time it's seen, then
    /// caches it. Useful when only a few of the 2^N possible
    /// switch states are visited (typical for fixed-duty PWM
    /// converters).
    void build_lazy(Real dt) {
        dt_ = dt;
        lazy_mode_ = true;
        segments_.clear();
    }

    /// Currently-built dt. Returns 0 for static-only builds.
    /// Layer 5's run_transient checks this against opts.dt and
    /// throws on mismatch.
    [[nodiscard]] Real dt() const noexcept { return dt_; }

    /// O(1) segment lookup. In eager mode, throws if the mask
    /// wasn't pre-built. In lazy mode (`build_lazy(dt)`), builds
    /// the segment on demand and caches it before returning.
    [[nodiscard]] const PwlSegment& lookup(
        const topology::SwitchStateMask& mask) const {
        auto r = try_lookup(mask);
        if (!r) {
            throw std::out_of_range(r.error().what());
        }
        return **r;
    }

    /// Non-throwing companion to `lookup` (Layer 4 V4 ergonomics).
    /// Returns a pointer to the segment on success, or a
    /// `CacheError` carrying the offending mask + dt on failure.
    ///
    /// Lazy mode triggers a build on first access — if THAT
    /// build hits a singular matrix, the returned error reports
    /// the singularity kind directly.
    [[nodiscard]] std::expected<const PwlSegment*, CacheError>
    try_lookup(const topology::SwitchStateMask& mask) const {
        const auto it = segments_.find(mask);
        if (it != segments_.end()) {
            return &it->second;
        }
        if (lazy_mode_) {
            auto built =
                const_cast<PwlStateSpaceCache*>(this)
                    ->try_build_one_segment(mask);
            if (!built) {
                return std::unexpected(built.error());
            }
            return &segments_.find(mask)->second;
        }
        return std::unexpected(CacheError{
            .kind = CacheError::Kind::MaskNotBuilt,
            .mask = mask,
            .dt   = dt_,
        });
    }

    /// HOT LOOP entry point. Look up the segment for `mask` and
    /// solve `J · x = -(b_constant + b_extra)` via the cached
    /// factor. `b_extra` lets Layer 5 inject time-varying source
    /// values or history terms in V1; in V0 it's typically zero.
    ///
    /// Newton's convention is solve(J, -f, dx). For Layer 4's
    /// static-only V0 path the residual at convergence is
    /// f = J·x + b_total = 0, so the solution is x = -(L^-1 ·
    /// (U^-1 · b_total)) = solver.solve(-b_total). We compute that
    /// directly: rhs = -(b_constant + b_extra), then solve.
    void solve(const topology::SwitchStateMask& mask,
                const Vector& b_extra,
                Vector& x) const {
        const PwlSegment& seg = lookup(mask);
        Vector rhs = -(seg.b_constant + b_extra);
        seg.solver->solve(rhs, x);
    }

    /// Number of segments currently cached. In eager mode this
    /// is `2^N` after `build()`; in lazy mode it grows as
    /// `solve(mask, ...)` calls populate the cache.
    [[nodiscard]] Size num_built_segments() const noexcept {
        return segments_.size();
    }

    [[nodiscard]] Size num_segments() const noexcept {
        return segments_.size();
    }

    /// Multi-dt cache (Layer 4 V7). Solves with a dt that MAY
    /// be different from the primary `this->dt()`. Builds the
    /// (mask, dt) factor on demand in an auxiliary cache the
    /// first time each pair is seen.
    ///
    /// When `dt == this->dt()`, this delegates to the primary
    /// `solve(mask, b_extra, x)` (same fast path).
    void solve_at(const topology::SwitchStateMask& mask,
                   Real dt,
                   const Vector& b_extra,
                   Vector& x) const {
        if (dt == dt_) {
            solve(mask, b_extra, x);
            return;
        }
        auto& bucket = alt_segments_[dt];
        auto it = bucket.find(mask);
        if (it == bucket.end()) {
            PwlSegment seg =
                const_cast<PwlStateSpaceCache*>(this)
                    ->make_segment(mask, dt);
            auto inserted_it =
                bucket.emplace(mask, std::move(seg)).first;
            it = inserted_it;
        }
        const PwlSegment& seg = it->second;
        Vector rhs = -(seg.b_constant + b_extra);
        seg.solver->solve(rhs, x);
    }

    /// Number of distinct auxiliary-dt values currently in
    /// the multi-dt cache.
    [[nodiscard]] Size num_alt_dt_values() const noexcept {
        return alt_segments_.size();
    }

    /// Number of segments factored at the given auxiliary dt
    /// (0 if `dt` has no cached segments).
    [[nodiscard]] Size num_alt_segments_at(Real dt) const noexcept {
        const auto it = alt_segments_.find(dt);
        return it == alt_segments_.end() ? 0 : it->second.size();
    }

    // -------------------------------------------------------------------------
    // Layer 4 V8 — rank-1 cache update fast path
    // (openspec/changes/add-pwl-rank1-update)
    // -------------------------------------------------------------------------
    //
    // `solve_rank1` maintains its OWN sliding factorisation that is updated
    // incrementally across consecutive calls: on a single-bit Gray-code mask
    // flip AND a partial-refactor-capable backend (e.g. KLU), it calls
    // `DirectSolver::partial_refactor` to update the factor in O(path)
    // instead of O(nnz·log n) full refactorisation. Multi-bit flips,
    // unsupported backends, numerical singularities, and first-encounter
    // masks all fall back transparently to the same `factorize` cost the
    // existing `solve` path would pay.
    //
    // The sliding solver is INDEPENDENT of the per-mask `segments_` map
    // populated by `build` / `build_lazy` / `solve`. The two paths are
    // orthogonal:
    //   * `solve(mask)`       — best for sequences that revisit a small set
    //                           of masks (cache hit on revisit).
    //   * `solve_rank1(mask)` — best for sequences that march through many
    //                           single-bit-adjacent masks (2^N too large for
    //                           the per-mask cache to help).
    //
    // Per the proposal:
    //   * Backend without `partial_refactor` support → silent fallback to
    //     full re-factor; `metrics().fallbacks` increments.
    //   * `partial_refactor` returning false (numerical singularity) →
    //     silent fallback to full re-factor; `metrics().fallbacks`
    //     increments.
    //   * Multi-bit flip OR first call → full re-factor; the existing
    //     full path; `metrics().full_refactor_hits` increments.
    //   * Single-bit flip with partial-refactor success →
    //     `metrics().rank1_hits` increments.

    /// Rank-1 sliding-solver entry point. Builds J for `mask`, then either
    /// reuses, partial-refactors, or full-refactors the cached factor as
    /// the mask diff vs the previous call dictates. Falls back
    /// transparently on any unsupported / failed partial-refactor.
    ///
    /// Requires `build(dt)` or `build_lazy(dt)` to have been called first
    /// (to set `dt_`). Throws `std::runtime_error` on initial-factorize
    /// numerical singularity.
    void solve_rank1(const topology::SwitchStateMask& mask,
                      const Vector& b_extra,
                      Vector& x) const {
        sparse::Matrix J;
        Vector b;
        assemble_segment(graph_, pool_, mask, dt_, J, b);
        sparse::compress_in_place(J);

        if (!rank1_initialized_) {
            // First call: fresh analyze + factorize on a new solver
            // picked by the factory. Backend hint defaults to
            // Backend::Auto (which currently falls through to
            // SparseLuSolver — the Pulsim native LU lands in a
            // follow-up, see openspec/changes/replace-klu-with-pulsim-sparse-lu/).
            // The caller may override via set_rank1_backend(...).
            rank1_solver_ = sparse::make_default_solver(
                pool_.state_size(graph_), rank1_backend_hint_);
            if (!rank1_solver_->analyze(J)) {
                throw std::runtime_error(std::format(
                    "PwlStateSpaceCache::solve_rank1: initial analyze "
                    "failed (structurally singular) for mask {}",
                    mask.to_string()));
            }
            if (!rank1_solver_->factorize(J)) {
                throw std::runtime_error(std::format(
                    "PwlStateSpaceCache::solve_rank1: initial factorize "
                    "failed (numerically singular) for mask {} (dt={})",
                    mask.to_string(), dt_));
            }
            rank1_b_constant_ = b;
            rank1_mask_       = mask;
            rank1_initialized_ = true;
            full_refactor_hits_.fetch_add(1, std::memory_order_relaxed);
        } else if (mask == rank1_mask_) {
            // No mask change — solver still valid, just refresh b_constant
            // in case time-varying sources updated their baseline.
            rank1_b_constant_ = b;
            rank1_hits_.fetch_add(1, std::memory_order_relaxed);
        } else {
            // Mask changed: decide partial vs full refactor.
            const std::uint64_t diff = mask.bits() ^ rank1_mask_.bits();
            const int pop = std::popcount(diff);

            const bool partial_eligible =
                (pop == 1) && rank1_solver_->supports_partial_refactor();

            bool refactored_via_partial = false;
            if (partial_eligible) {
                const auto changed_cols =
                    compute_changed_columns_(rank1_mask_, mask);
                refactored_via_partial = rank1_solver_->partial_refactor(
                    J, std::span<const Index>{
                        changed_cols.data(), changed_cols.size()});
            }

            if (refactored_via_partial) {
                rank1_b_constant_ = b;
                rank1_mask_       = mask;
                rank1_hits_.fetch_add(1, std::memory_order_relaxed);
            } else {
                // Full refactor on the same symbolic. The sparsity pattern
                // is identical across all switch states (only conductance
                // values change), so analyze() does NOT need to re-run.
                if (!rank1_solver_->factorize(J)) {
                    throw std::runtime_error(std::format(
                        "PwlStateSpaceCache::solve_rank1: full refactor "
                        "fallback failed (numerically singular) for mask "
                        "{} (dt={})",
                        mask.to_string(), dt_));
                }
                rank1_b_constant_ = b;
                rank1_mask_       = mask;
                if (partial_eligible) {
                    // partial_refactor was attempted and returned false.
                    fallbacks_.fetch_add(1, std::memory_order_relaxed);
                } else if (pop == 1) {
                    // Single-bit flip but backend doesn't support
                    // partial_refactor (e.g. SparseLuSolver).
                    fallbacks_.fetch_add(1, std::memory_order_relaxed);
                } else {
                    // Multi-bit flip — genuine full refactor.
                    full_refactor_hits_.fetch_add(
                        1, std::memory_order_relaxed);
                }
            }
        }

        Vector rhs = -(rank1_b_constant_ + b_extra);
        rank1_solver_->solve(rhs, x);
    }

    /// Override the backend `solve_rank1` uses for its sliding solver.
    /// By default `solve_rank1` calls
    /// `make_default_solver(state_size, Backend::Auto)`. Call this
    /// before the first `solve_rank1` to force a specific backend —
    /// e.g. `Backend::Pulsim` for benchmarks that want the path-based
    /// partial-refactor path (once
    /// openspec/changes/replace-klu-with-pulsim-sparse-lu/ Sections 2-5
    /// land), or `Backend::Eigen` to force the Eigen::SparseLU
    /// fallback path for measurement.
    ///
    /// No-op once `solve_rank1` has been called — the solver has
    /// already been constructed.
    void set_rank1_backend(sparse::Backend hint) noexcept {
        if (!rank1_initialized_) {
            rank1_backend_hint_ = hint;
        }
    }

    /// Snapshot the rank-1 telemetry counters. Read-only from outside;
    /// monotonic across the cache's lifetime.
    [[nodiscard]] CacheMetrics metrics() const noexcept {
        return {
            .rank1_hits = rank1_hits_.load(
                std::memory_order_relaxed),
            .full_refactor_hits = full_refactor_hits_.load(
                std::memory_order_relaxed),
            .fallbacks = fallbacks_.load(
                std::memory_order_relaxed),
        };
    }

private:
    /// Non-throwing build of a single segment. Returns the
    /// segment by move on success, or a `CacheError` carrying
    /// the singularity kind + mask + dt on failure.
    [[nodiscard]] std::expected<PwlSegment, CacheError>
    try_make_segment(const topology::SwitchStateMask& mask, Real dt) {
        sparse::Matrix J;
        Vector b;
        assemble_segment(graph_, pool_, mask, dt, J, b);
        sparse::compress_in_place(J);

        auto solver = sparse::make_default_solver();
        if (!solver->analyze(J)) {
            return std::unexpected(CacheError{
                .kind = CacheError::Kind::StructurallySingular,
                .mask = mask,
                .dt   = dt,
            });
        }
        if (!solver->factorize(J)) {
            return std::unexpected(CacheError{
                .kind = CacheError::Kind::NumericallySingular,
                .mask = mask,
                .dt   = dt,
            });
        }

        PwlSegment seg;
        seg.J = std::move(J);
        seg.b_constant = std::move(b);
        seg.solver = std::move(solver);
        seg.state_size = pool_.state_size(graph_);
        return seg;
    }

    /// Throwing wrapper — the existing kernel hot path. Delegates
    /// to `try_make_segment` and translates the typed error into
    /// a `runtime_error` whose `what()` matches the V0 format
    /// callers (and tests) already grep for.
    [[nodiscard]] PwlSegment make_segment(
        const topology::SwitchStateMask& mask, Real dt) {
        auto r = try_make_segment(mask, dt);
        if (!r) {
            throw std::runtime_error(r.error().what());
        }
        return std::move(*r);
    }

    /// Non-throwing single-segment insert at the current dt.
    /// Used by `try_lookup`'s lazy-mode path.
    [[nodiscard]] std::expected<void, CacheError>
    try_build_one_segment(const topology::SwitchStateMask& mask) {
        auto seg = try_make_segment(mask, dt_);
        if (!seg) {
            return std::unexpected(seg.error());
        }
        segments_.emplace(mask, std::move(*seg));
        return {};
    }

    /// Insert a primary-cache segment at the current dt.
    /// Eager `build()` calls in a loop; lazy `lookup()` on
    /// demand.
    void build_one_segment(const topology::SwitchStateMask& mask) {
        segments_.emplace(mask, make_segment(mask, dt_));
    }

    /// Compute the list of MNA column indices that change between two
    /// switch masks. For each bit that differs, the corresponding switch's
    /// branch contributes its two terminal nodes (from, to). The result
    /// may contain duplicates if multiple switches share a node — that's
    /// fine, downstream `partial_refactor` consumers can deduplicate.
    ///
    /// Consumed by `DirectSolver::partial_refactor` overrides that
    /// implement path-based re-elimination (Chan/Brandwajn/Tinney
    /// 1986, Dinkelbach et al. 2021). The default `DirectSolver`
    /// returns false from `partial_refactor`, so this hint goes
    /// unused on backends that don't support the fast path —
    /// `solve_rank1`'s fallback to full `factorize` engages
    /// transparently.
    [[nodiscard]] std::vector<Index> compute_changed_columns_(
        const topology::SwitchStateMask& prev_mask,
        const topology::SwitchStateMask& curr_mask) const {
        std::vector<Index> cols;
        const std::uint64_t diff = prev_mask.bits() ^ curr_mask.bits();
        if (diff == 0) {
            return cols;
        }
        cols.reserve(static_cast<std::size_t>(std::popcount(diff)) * 2);
        Size switch_idx = 0;
        for (Index b_id = 0; b_id < graph_.num_branches(); ++b_id) {
            const auto& branch = graph_.branch(b_id);
            if (branch.kind == topology::BranchKind::Switch) {
                if (((diff >> switch_idx) & 1ULL) != 0ULL) {
                    cols.push_back(branch.from);
                    cols.push_back(branch.to);
                }
                ++switch_idx;
            }
        }
        return cols;
    }

    const topology::Graph& graph_;
    const DevicePool& pool_;
    numeric::Dictionary<topology::SwitchStateMask, PwlSegment>
        segments_;
    Real dt_ = Real{0};   // 0 = static-only build (V0)
    bool lazy_mode_ = false;

    // Layer 4 V7: auxiliary multi-dt cache for sub-step
    // bisection state correction (`solve_at`). Keyed first by
    // dt then by mask. `mutable` so `solve_at` (const) can
    // populate on demand.
    mutable numeric::Dictionary<Real, numeric::Dictionary<
        topology::SwitchStateMask, PwlSegment>> alt_segments_;

    // Layer 4 V8: sliding solver + state for the rank-1 fast-path
    // (`solve_rank1`). Independent of `segments_`. `mutable` so
    // `solve_rank1` (const) can update them.
    mutable std::unique_ptr<sparse::DirectSolver> rank1_solver_;
    mutable topology::SwitchStateMask rank1_mask_{};
    mutable Vector                     rank1_b_constant_;
    mutable bool                       rank1_initialized_ = false;
    mutable sparse::Backend            rank1_backend_hint_ = sparse::Backend::Auto;

    // Layer 4 V8 telemetry. Atomic so a background thread can sample
    // mid-simulation without locking; const-correct for use inside
    // const `solve_rank1` and const `metrics()`.
    mutable std::atomic<std::uint64_t> rank1_hits_{0};
    mutable std::atomic<std::uint64_t> full_refactor_hits_{0};
    mutable std::atomic<std::uint64_t> fallbacks_{0};
};

}  // namespace pulsim::pwl
