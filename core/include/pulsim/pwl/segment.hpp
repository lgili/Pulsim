#pragma once

// =============================================================================
// Pulsim — Layer 4: PwlSegment (per-state cached record)
// =============================================================================
//
// `pulsim-v2-pwl-state-space-cache` Phase 2.
//
// One PwlSegment per switch combination. Holds:
//   * The assembled MNA matrix `J` (kept for diagnostics / future
//     dt-rebuild)
//   * The constant RHS `b_constant` (voltage-source `-V` terms)
//   * A pre-factorized DirectSolver (analyze + factorize done at
//     build, ready for solve)
//   * The state-vector size N + M (for sanity-checking callers)
//
// Move-only: the unique_ptr<DirectSolver> makes copy nonsensical.
// The cache stores segments in std::unordered_map and uses
// std::move when populating.

#include "pulsim/numeric/dense.hpp"
#include "pulsim/numeric/types.hpp"
#include "pulsim/sparse/matrix.hpp"
#include "pulsim/sparse/solver.hpp"

#include <atomic>
#include <memory>

namespace pulsim::pwl {

struct PwlSegment {
    sparse::Matrix J;                                  // assembled MNA matrix
    Vector b_constant;                                  // -V from sources
    std::unique_ptr<sparse::DirectSolver> solver;      // pre-factorized
    Size state_size = 0;                                // N + M

    // v2.0 Phase 1 — LRU recency tick for the lazy segment cache's
    // byte-budget eviction. `mutable`: cache HITS touch it through
    // the const lookup path without structural mutation.
    //
    // ATOMIC because that touch happens on a const path that the
    // Python bindings run with the GIL released, so two threads
    // sharing a warm cache can hit it concurrently. The LRU only
    // needs approximate recency, so relaxed ordering is enough —
    // the point is to not have a data race (adversarial-review
    // finding F3/F2). Hand-written copy/move: std::atomic is
    // neither copyable nor movable, and PwlSegment must stay
    // move-constructible for the flat_map.
    mutable std::atomic<std::uint64_t> lru_tick{0};

    [[nodiscard]] std::uint64_t tick() const noexcept {
        return lru_tick.load(std::memory_order_relaxed);
    }
    void touch(std::uint64_t t) const noexcept {
        lru_tick.store(t, std::memory_order_relaxed);
    }

    /// Estimated resident bytes of this segment (assembled J +
    /// RHS + the solver's numeric factors). Budget arithmetic
    /// only — not an allocator audit.
    [[nodiscard]] std::size_t approx_bytes() const noexcept {
        const auto nnz  = static_cast<std::size_t>(J.nonZeros());
        const auto cols = static_cast<std::size_t>(J.cols());
        std::size_t total =
            nnz * (sizeof(Real) + sizeof(sparse::Matrix::StorageIndex)) +
            (cols + 1) * sizeof(sparse::Matrix::StorageIndex) +
            static_cast<std::size_t>(b_constant.size()) * sizeof(Real);
        if (solver) {
            total += solver->factor_bytes();
        }
        return total;
    }

    PwlSegment() = default;
    PwlSegment(PwlSegment&& o) noexcept
        : J{std::move(o.J)},
          b_constant{std::move(o.b_constant)},
          solver{std::move(o.solver)},
          state_size{o.state_size},
          lru_tick{o.tick()} {}
    PwlSegment& operator=(PwlSegment&& o) noexcept {
        if (this != &o) {
            J           = std::move(o.J);
            b_constant  = std::move(o.b_constant);
            solver      = std::move(o.solver);
            state_size  = o.state_size;
            touch(o.tick());
        }
        return *this;
    }
    PwlSegment(const PwlSegment&) = delete;
    PwlSegment& operator=(const PwlSegment&) = delete;
};

static_assert(std::is_move_constructible_v<PwlSegment>);
static_assert(!std::is_copy_constructible_v<PwlSegment>,
              "PwlSegment must be move-only — holds a unique_ptr<DirectSolver>");

}  // namespace pulsim::pwl
