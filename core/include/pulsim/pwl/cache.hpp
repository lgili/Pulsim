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

#include <algorithm>
#include <atomic>
#include <bit>
#include <chrono>
#include <cstdint>
#include <expected>
#include <format>
#include <set>
#include <stdexcept>
#include <utility>
#include <vector>

namespace pulsim::pwl {

/// Telemetry counters for the rank-1 cache update path
/// (openspec/changes/add-pwl-rank1-update, extended by
/// openspec/changes/add-generalised-path-refactor Part A).
/// Monotonically incremented from inside
/// `PwlStateSpaceCache::solve_rank1`. Use
/// `PwlStateSpaceCache::metrics()` to snapshot.
///
/// * `rank1_hits`            — fast-path hits where the mask either did not
///                             change (b_constant refresh + triangular solve
///                             only) OR changed by exactly one bit AND the
///                             single-bit `partial_refactor` path engaged.
///                             v1.3.0 semantics preserved.
/// * `multi_bit_rank1_hits`  — (v1.4.0+) calls where the mask changed by ≥ 2
///                             bits AND the multi-bit path-union
///                             `partial_refactor` path engaged successfully.
///                             Previously these landed in `full_refactor_hits`
///                             (the v1.3.0 unconditional multi-bit fallback);
///                             v1.4.0 routes them through path-based update
///                             when the union path is short enough.
/// * `full_refactor_hits`    — fresh `factorize()` because the path-based
///                             update would be no cheaper than a full
///                             factorisation (`path_length / n >
///                             MAX_PATH_LENGTH_RATIO`), or the backend does
///                             not implement `partial_refactor`, or the
///                             first-encounter initial factor.
/// * `fallbacks`             — `partial_refactor` was attempted and returned
///                             `false` (numerical singularity / pivot fault
///                             in the path). Distinct from `full_refactor_hits`
///                             which is the "didn't even try" bucket.
///
/// Invariant per
/// `openspec/changes/add-generalised-path-refactor/specs/pwl-rank1-update`:
///     `rank1_hits + multi_bit_rank1_hits + full_refactor_hits + fallbacks
///      == total solve_rank1 calls`.
struct CacheMetrics {
    std::uint64_t rank1_hits            = 0;
    std::uint64_t multi_bit_rank1_hits  = 0;
    std::uint64_t full_refactor_hits    = 0;
    std::uint64_t fallbacks             = 0;
};

/// Result of `PwlStateSpaceCache::refactor_parametric` (v1.4.0,
/// `openspec/changes/add-generalised-path-refactor` Part B).
///
/// Per the spec invariant:
///   `path_refactor_hits + fallback_hits == masks_processed`.
///
/// * `masks_processed`     — total number of cached masks the call
///                           visited. For `Mode::AllActive` this is
///                           `segments_.size()`; for
///                           `Mode::CurrentOnly` it is at most 1 (the
///                           rank-1 sliding solver's current mask
///                           if rank-1 has been used).
/// * `path_refactor_hits`  — segments where `partial_refactor` was
///                           called AND returned `true`. The headline
///                           speedup bucket.
/// * `fallback_hits`       — segments where the path was either too
///                           long (`affected_cols/n > MAX_PATH_LENGTH_RATIO`),
///                           the backend doesn't support
///                           `partial_refactor`, or `partial_refactor`
///                           was attempted and returned `false`
///                           (pivot fault). In every case the
///                           segment was rebuilt via fresh
///                           `factorize(new_J)`.
/// * `wall_time_us`        — total wall-clock spent in the call
///                           (microseconds). Includes re-stamping +
///                           path-walks + factorizations.
struct ParametricRefactorResult {
    std::size_t masks_processed    = 0;
    std::size_t path_refactor_hits = 0;
    std::size_t fallback_hits      = 0;
    double      wall_time_us       = 0.0;
};

/// Selection mode for `PwlStateSpaceCache::refactor_parametric`.
///
/// `AllActive` (default) — process every cached `(mask, segment)`
/// entry. The right choice for sweep / Monte Carlo workloads where
/// downstream `simulate()` calls will visit multiple masks across
/// the transient.
///
/// `CurrentOnly` — process just the most-recent rank-1 sliding
/// solver's mask (if rank-1 has been used). Useful for hot-loop
/// parameter perturbations (e.g. a closed-loop control study that
/// trims one gain live) where the user knows only the active mask
/// matters. A no-op (returns `masks_processed == 0`) when the cache
/// has not been used via `solve_rank1`.
enum class ParametricRefactorMode {
    AllActive,
    CurrentOnly,
};

/// Single parameter update: which device (by `branch_id`, obtained
/// via `CircuitBuilder::branch_id_of(name)`) and its new value.
/// The pool's mutator dispatch (`update_resistor_R`,
/// `update_inductor_L`, `update_capacitor_C`,
/// `update_voltage_source_V`) is selected from the stored device
/// kind on the corresponding branch.
struct ParametricUpdate {
    Index branch_id;
    Real  new_value;
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
    /// v1.4.0: pool stored as a non-const reference so
    /// `refactor_parametric` can mutate it in place. All read paths
    /// remain unchanged; existing callers that pass a non-const
    /// `DevicePool` keep compiling. A `const DevicePool&` argument
    /// no longer binds — callers that need read-only safety should
    /// `std::cref(pool)` upstream, but in practice every Layer 1-9
    /// consumer passes a freshly built non-const pool.
    PwlStateSpaceCache(const topology::Graph& graph,
                        DevicePool& pool) noexcept
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
            //
            // v1.4.0 (openspec/changes/add-generalised-path-refactor Part A):
            // multi-bit flips are no longer unconditionally routed to
            // full factorize. Instead we compute the affected columns for
            // EVERY toggled bit (deduplicated), query the solver for the
            // would-be path length, and:
            //   - if path-length / n ≤ MAX_PATH_LENGTH_RATIO → try
            //     partial_refactor on the multi-column changed set
            //   - else → fall back to full factorize (path-based wouldn't
            //     be cheaper anyway)
            const std::uint64_t diff = mask.bits() ^ rank1_mask_.bits();
            const int pop = std::popcount(diff);

            const auto changed_cols =
                compute_changed_columns_(rank1_mask_, mask);
            const std::span<const Index> changed_cols_span{
                changed_cols.data(), changed_cols.size()};

            const bool can_try_partial =
                rank1_solver_->supports_partial_refactor();
            // Routing rule per
            // `openspec/changes/add-generalised-path-refactor`
            // §pwl-rank1-update spec:
            //   * pop == 1 (single-bit, v1.3.0 case) — always try
            //     partial_refactor when the backend supports it. The
            //     captured microbench shows 100% success rate on
            //     single-bit Gray-code flips; the path is always
            //     short. No ratio gate needed.
            //   * pop >= 2 (multi-bit, v1.4.0 NEW) — query
            //     partial_refactor_count_path() and skip the
            //     path-based attempt only when the union path covers
            //     > MAX_PATH_LENGTH_RATIO of the columns (where
            //     path-based wouldn't save work vs a fresh factorise).
            bool try_path = can_try_partial;
            if (try_path && pop > 1) {
                const Index n_state = static_cast<Index>(
                    pool_.state_size(graph_));
                const Index path_len =
                    rank1_solver_->partial_refactor_count_path(
                        changed_cols_span);
                const Real path_ratio = n_state > 0
                    ? (static_cast<Real>(path_len) /
                       static_cast<Real>(n_state))
                    : Real{1};
                try_path = (path_ratio <= sparse::MAX_PATH_LENGTH_RATIO);
            }

            bool refactored_via_partial = false;
            if (try_path) {
                refactored_via_partial = rank1_solver_->partial_refactor(
                    J, changed_cols_span);
            }

            if (refactored_via_partial) {
                rank1_b_constant_ = b;
                rank1_mask_       = mask;
                if (pop == 1) {
                    rank1_hits_.fetch_add(1, std::memory_order_relaxed);
                } else {
                    multi_bit_rank1_hits_.fetch_add(
                        1, std::memory_order_relaxed);
                }
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
                if (try_path) {
                    // partial_refactor was attempted and returned false
                    // (numerical/pivot fault) — fall back.
                    fallbacks_.fetch_add(1, std::memory_order_relaxed);
                } else {
                    // Either the backend can't do partial OR the path
                    // ratio was above MAX_PATH_LENGTH_RATIO → didn't
                    // even try. Counted under full_refactor_hits, not
                    // fallbacks (the "didn't try" vs "tried and failed"
                    // distinction matters for diagnostics).
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
            .multi_bit_rank1_hits = multi_bit_rank1_hits_.load(
                std::memory_order_relaxed),
            .full_refactor_hits = full_refactor_hits_.load(
                std::memory_order_relaxed),
            .fallbacks = fallbacks_.load(
                std::memory_order_relaxed),
        };
    }

    // -------------------------------------------------------------------------
    // v1.4.0 Part B — parametric refactor for sweep / Monte Carlo workloads
    // -------------------------------------------------------------------------
    //
    // Pushes a batch of `(branch_id, new_value)` updates into the pool,
    // then walks every active mask in `segments_` (or just the
    // rank-1 sliding solver's mask, per `mode`) and re-stamps + path-
    // refactors each cached segment at the minimum column set.
    //
    // The key win vs the legacy "build_lazy + sweep" pattern: we
    // **reuse the symbolic factorization** (`analyze`) and the
    // **majority of L+U entries** across sweep points. Only the
    // columns of J that depend on the changed parameters need re-
    // elimination, and only via the etree path of those columns.
    //
    // Result invariant: `path_refactor_hits + fallback_hits ==
    //                    masks_processed`.
    //
    // Numerical contract: post-refactor `solve(mask, ...)` matches a
    // fresh `analyze + factorize + solve` on the updated pool within
    // 1e-10 (the same tolerance the multi-bit path-union gate
    // enforces).
    //
    // Pre-conditions:
    //   * `build(...)` or `build_lazy(...)` has been called.
    //   * Every `updates[i].branch_id` is a registered device whose
    //     kind matches the `update_*` mutator semantics (Resistor,
    //     Inductor, Capacitor, VoltageSource). Other kinds silently
    //     fall back to full-rebuild for every affected mask.
    //   * Topology is unchanged (graph node/branch structure intact).
    //     Updating values of a switch is not supported; switches are
    //     toggled via the mask, not via parametric refactor.
    [[nodiscard]] ParametricRefactorResult refactor_parametric(
        std::span<const ParametricUpdate> updates,
        ParametricRefactorMode mode = ParametricRefactorMode::AllActive) {
        ParametricRefactorResult result;
        const auto t0 = std::chrono::steady_clock::now();

        if (updates.empty()) {
            // No-op. Still report wall_time for telemetry callers.
            const auto t1 = std::chrono::steady_clock::now();
            result.wall_time_us =
                std::chrono::duration<double, std::micro>(t1 - t0).count();
            return result;
        }

        // 1. Push the updates into the pool. Each `update_*` mutator
        //    dispatches on the stored device kind; mismatched kinds
        //    throw std::out_of_range which propagates to the caller.
        for (const auto& upd : updates) {
            apply_parametric_update_(upd);
        }

        // 2. Build the deduplicated union of columns affected by the
        //    parameter changes. An empty affected-cols set on any
        //    branch (e.g. VoltageSource — RHS-only) means that
        //    branch's change doesn't touch J; only b_constant needs
        //    re-stamping, which happens during the per-segment
        //    re-assemble step inside refactor_one_parametric_.
        std::set<Index> affected_cols_set;
        for (const auto& upd : updates) {
            const auto cols =
                pool_.columns_affected_by_branch(upd.branch_id, graph_);
            for (Index c : cols) affected_cols_set.insert(c);
        }

        const std::vector<Index> affected_cols(
            affected_cols_set.begin(), affected_cols_set.end());
        const std::span<const Index> affected_cols_span{
            affected_cols.data(), affected_cols.size()};

        // 3. Pick which segments to process.
        std::vector<topology::SwitchStateMask> masks_to_process;
        if (mode == ParametricRefactorMode::CurrentOnly) {
            if (rank1_initialized_) {
                masks_to_process.push_back(rank1_mask_);
            }
        } else {
            // AllActive — every cached primary segment AND, if the
            // rank-1 sliding solver is in use, its mask too.
            masks_to_process.reserve(segments_.size() +
                                      (rank1_initialized_ ? 1 : 0));
            for (const auto& [m, _seg] : segments_) {
                masks_to_process.push_back(m);
            }
            if (rank1_initialized_) {
                // Avoid double-processing if the rank-1 mask is also
                // in segments_ (e.g. the user did a mixed
                // solve + solve_rank1 workflow).
                if (std::find(masks_to_process.begin(),
                              masks_to_process.end(),
                              rank1_mask_) == masks_to_process.end()) {
                    masks_to_process.push_back(rank1_mask_);
                }
            }
        }

        // 4. For each selected mask, re-assemble J + b at the new
        //    pool values, then path-refactor (or fall back to full
        //    factorize when the path covers > MAX_PATH_LENGTH_RATIO
        //    of n, or when the backend doesn't support partial_refactor).
        for (const auto& mask : masks_to_process) {
            ++result.masks_processed;
            if (refactor_one_parametric_(mask, affected_cols_span)) {
                ++result.path_refactor_hits;
            } else {
                ++result.fallback_hits;
            }
        }

        const auto t1 = std::chrono::steady_clock::now();
        result.wall_time_us =
            std::chrono::duration<double, std::micro>(t1 - t0).count();
        return result;
    }

    /// Convenience overload for the very common one-parameter case:
    /// `cache.refactor_parametric(branch_id, new_value)`.
    [[nodiscard]] ParametricRefactorResult refactor_parametric(
        Index branch_id, Real new_value,
        ParametricRefactorMode mode = ParametricRefactorMode::AllActive) {
        ParametricUpdate one{branch_id, new_value};
        return refactor_parametric(
            std::span<const ParametricUpdate>(&one, 1), mode);
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

        // v1.4.0: each segment's solver is the Pulsim in-house LU
        // (`Backend::Auto` → Pulsim since v1.3.0). This gives
        // `refactor_parametric` access to `partial_refactor` on the
        // per-mask cached factor; the legacy two-arg
        // `make_default_solver(state_size, Backend::Auto)` call is
        // bit-identical to the v1.3.0 result on real-scalar SPD
        // matrices. The no-arg `make_default_solver()` overload
        // (Eigen baseline) is now kept only for the showcase /
        // sanity-check tests that pin the Eigen behaviour for
        // paper-comparison purposes.
        auto solver = sparse::make_default_solver(
            pool_.state_size(graph_), sparse::Backend::Auto);
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

    /// Dispatch a single parametric update to the correct pool
    /// mutator based on the stored device kind. v1.4.0 supports
    /// Resistor, Inductor, Capacitor, VoltageSource. Other kinds
    /// raise std::out_of_range — caller is expected to either avoid
    /// the update or rebuild the cache from scratch via
    /// `build_lazy(dt_)`.
    void apply_parametric_update_(const ParametricUpdate& upd) {
        const auto kind = pool_.kind_of(upd.branch_id);
        using SK = DevicePool::StoredKind;
        switch (kind) {
            case SK::Resistor:
                pool_.update_resistor_R(upd.branch_id, upd.new_value);
                break;
            case SK::Inductor:
                pool_.update_inductor_L(upd.branch_id, upd.new_value);
                break;
            case SK::Capacitor:
                pool_.update_capacitor_C(upd.branch_id, upd.new_value);
                break;
            case SK::VoltageSource:
                pool_.update_voltage_source_V(upd.branch_id,
                                                upd.new_value);
                break;
            default:
                throw std::out_of_range(std::format(
                    "PwlStateSpaceCache::refactor_parametric: "
                    "branch {} has device kind {} which is not "
                    "supported by the v1.4.0 parametric refactor "
                    "pipeline (Resistor / Inductor / Capacitor / "
                    "VoltageSource only). Rebuild the cache via "
                    "build_lazy(dt) after updating this device.",
                    upd.branch_id, static_cast<int>(kind)));
        }
    }

    /// Re-assemble + refactor one (mask, segment) for the parametric
    /// case. Updates BOTH the primary segment (if cached) AND the
    /// rank-1 sliding solver (if initialized at this mask) — the
    /// two paths share a logical mask but have independent solver
    /// instances. Returns true if every applicable target took the
    /// path-based update; false if any target fell back to fresh
    /// `factorize()`. Throws std::runtime_error on fresh-factorize
    /// numerical singularity (no recovery — the parameter values are
    /// genuinely problematic).
    [[nodiscard]] bool refactor_one_parametric_(
        const topology::SwitchStateMask& mask,
        std::span<const Index> affected_cols) {
        // We may update BOTH a primary segment AND the rank-1
        // sliding solver if the mask matches both. Re-assemble
        // (J, b) once per target so refactor_solver_'s `std::move`
        // of new_J doesn't fight between paths.
        bool any_target_existed = false;
        bool any_path_succeeded = false;

        auto it = segments_.find(mask);
        if (it != segments_.end()) {
            any_target_existed = true;
            sparse::Matrix new_J;
            Vector new_b;
            assemble_segment(graph_, pool_, mask, dt_, new_J, new_b);
            sparse::compress_in_place(new_J);
            const bool ok = refactor_solver_(
                it->second.solver.get(),
                it->second.J, it->second.b_constant,
                new_J, new_b, affected_cols);
            any_path_succeeded = any_path_succeeded || ok;
        }
        if (rank1_initialized_ && mask == rank1_mask_) {
            any_target_existed = true;
            sparse::Matrix new_J;
            Vector new_b;
            assemble_segment(graph_, pool_, mask, dt_, new_J, new_b);
            sparse::compress_in_place(new_J);
            // The rank-1 sliding solver doesn't cache J (it
            // re-assembles every solve_rank1 call), but its
            // factor + b_constant are stateful and must be
            // refreshed here so any subsequent solve_rank1 sees
            // the post-update pool values.
            sparse::Matrix dummy_old_J;
            const bool ok = refactor_solver_(
                rank1_solver_.get(),
                dummy_old_J,
                rank1_b_constant_,
                new_J, new_b, affected_cols);
            any_path_succeeded = any_path_succeeded || ok;
        }
        if (!any_target_existed) {
            // No cached factor for this mask anywhere. Count as
            // fallback so the result invariant still holds; the
            // next solve(mask) will lazily build a fresh segment
            // from the post-update pool.
            return false;
        }
        return any_path_succeeded;
    }

    /// Common path-vs-factorize logic shared by primary and rank-1
    /// segments. Mutates `seg_J` + `seg_b` to the new values on
    /// return regardless of which path was taken.
    [[nodiscard]] bool refactor_solver_(
        sparse::DirectSolver* solver,
        sparse::Matrix& seg_J, Vector& seg_b,
        sparse::Matrix& new_J, Vector& new_b,
        std::span<const Index> affected_cols) {
        const Index n = static_cast<Index>(pool_.state_size(graph_));
        const bool can_try_partial =
            solver->supports_partial_refactor();

        // Decide whether to attempt the path-based update. Same
        // ratio gate as the multi-bit case: if the union path
        // covers > MAX_PATH_LENGTH_RATIO of n, the path-based
        // update wouldn't be cheaper than fresh factorize.
        bool try_path = can_try_partial && !affected_cols.empty();
        if (try_path) {
            const Index path_len =
                solver->partial_refactor_count_path(affected_cols);
            const Real ratio = n > 0
                ? (static_cast<Real>(path_len) / static_cast<Real>(n))
                : Real{1};
            try_path = (ratio <= sparse::MAX_PATH_LENGTH_RATIO);
        }

        bool refactored_via_partial = false;
        if (try_path) {
            refactored_via_partial =
                solver->partial_refactor(new_J, affected_cols);
        }

        if (!refactored_via_partial) {
            if (!solver->factorize(new_J)) {
                throw std::runtime_error(
                    "PwlStateSpaceCache::refactor_parametric: full "
                    "factorize fallback failed (numerically singular). "
                    "Check the updated parameter values.");
            }
        }

        // Mutate the cached segment's J + b in place so subsequent
        // solve() calls return the post-update result.
        seg_J = std::move(new_J);
        seg_b = std::move(new_b);
        return refactored_via_partial;
    }

    /// Compute the DEDUPED list of MNA column indices that change
    /// between two switch masks. For each switch bit that differs, the
    /// affected columns come from
    /// `DevicePool::columns_affected_by_switch(sw_idx, graph)` (the
    /// switch's two terminal nodes, skipping ground anchors). Multiple
    /// switches sharing a node produce a single deduplicated entry.
    ///
    /// v1.4.0 (openspec/changes/add-generalised-path-refactor Part A):
    /// dedup is now done HERE rather than left to the solver, because
    /// the solver's `partial_refactor_count_path()` query needs a
    /// canonical input to give a meaningful answer to the
    /// `MAX_PATH_LENGTH_RATIO` gate. Dedup at the call site keeps the
    /// solver's API surface narrow (it just walks the set it's given).
    ///
    /// Consumed by `DirectSolver::partial_refactor` overrides that
    /// implement path-based re-elimination. Backends without
    /// `partial_refactor` support return false; `solve_rank1`'s
    /// fallback to full `factorize` engages transparently.
    [[nodiscard]] std::vector<Index> compute_changed_columns_(
        const topology::SwitchStateMask& prev_mask,
        const topology::SwitchStateMask& curr_mask) const {
        const std::uint64_t diff = prev_mask.bits() ^ curr_mask.bits();
        if (diff == 0) {
            return {};
        }
        // Use a set for in-place dedup. Switches sharing a node (common
        // in half-bridge / full-bridge topologies where both legs
        // anchor to the same midpoint) would otherwise produce
        // duplicate column entries — fine for the solver's path-union
        // (its in_path bitmap dedupes anyway) but bad for the count
        // query which counts unique paths.
        std::set<Index> uniq;
        Size switch_idx = 0;
        for (Index b_id = 0; b_id < graph_.num_branches(); ++b_id) {
            const auto& branch = graph_.branch(b_id);
            if (branch.kind == topology::BranchKind::Switch) {
                if (((diff >> switch_idx) & 1ULL) != 0ULL) {
                    if (branch.from >= 0) uniq.insert(branch.from);
                    if (branch.to   >= 0) uniq.insert(branch.to);
                }
                ++switch_idx;
            }
        }
        return std::vector<Index>(uniq.begin(), uniq.end());
    }

    const topology::Graph& graph_;
    DevicePool& pool_;
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
    mutable std::atomic<std::uint64_t> multi_bit_rank1_hits_{0};
    mutable std::atomic<std::uint64_t> full_refactor_hits_{0};
    mutable std::atomic<std::uint64_t> fallbacks_{0};
};

}  // namespace pulsim::pwl
