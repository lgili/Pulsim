#pragma once

// =============================================================================
// Pulsim v2 — Layer 4: PwlSegment (per-state cached record)
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

#include <memory>

namespace pulsim::pwl {

struct PwlSegment {
    sparse::Matrix J;                                  // assembled MNA matrix
    Vector b_constant;                                  // -V from sources
    std::unique_ptr<sparse::DirectSolver> solver;      // pre-factorized
    Size state_size = 0;                                // N + M

    PwlSegment() = default;
    PwlSegment(PwlSegment&&) noexcept = default;
    PwlSegment& operator=(PwlSegment&&) noexcept = default;
    PwlSegment(const PwlSegment&) = delete;
    PwlSegment& operator=(const PwlSegment&) = delete;
};

static_assert(std::is_move_constructible_v<PwlSegment>);
static_assert(!std::is_copy_constructible_v<PwlSegment>,
              "PwlSegment must be move-only — holds a unique_ptr<DirectSolver>");

}  // namespace pulsim::pwl
