#pragma once

// =============================================================================
// Pulsim — Layer 0: numeric scalar + index types
// =============================================================================
//
// `bootstrap-pulsim-v2-kernel` Phase 1. The two-page-summary version:
//
//   * `Real` is the working floating-point precision. Default `double`,
//     overridable at build time via `-DPULSIM_V2_REAL_TYPE=float` for
//     embedded / FPGA HIL targets that need single precision.
//   * `Index` is fixed at `std::int32_t`. Signed (negative sentinels
//     for ground / "no such device"), exactly 4 bytes (cache-friendly
//     vs 8 B, and matches what every direct sparse solver expects
//     natively). `2^31 − 1 ≈ 2 G nodes` is plenty.
//   * `Size` is `std::size_t`. Container sizes only; never used as a
//     node / branch ID.
//
// These are the SAME aliases every higher layer uses. Changing them is
// a coordinated rebuild of the entire v2 tree. They are also the only
// numeric types this file exposes — vector / matrix aliases live in
// `numeric/dense.hpp`, concepts live in `numeric/concepts.hpp`.

#include <cstdint>
#include <cstddef>
#include <limits>

namespace pulsim {

// -----------------------------------------------------------------------------
// Real — the simulator's working floating-point precision.
//
// Default `double`. To build a single-precision Pulsim (e.g. for an
// embedded HIL target), pass `-DPULSIM_V2_REAL_TYPE=float` at CMake
// configure time and the entire v2 tree compiles in float.
//
// Layer-2 device-model templates accept any floating-point type
// (so AD scalars and float can be used in the same template path);
// Layer 4 + Layer 5 work in `Real` for numeric stability.
// -----------------------------------------------------------------------------
#ifdef PULSIM_V2_REAL_TYPE
using Real = PULSIM_V2_REAL_TYPE;
#else
using Real = double;
#endif

// -----------------------------------------------------------------------------
// Index — node / branch / matrix-index type.
//
// `std::int32_t`:
//   * 4 B fits twice as many indices per cache line as 8 B
//   * matches int32 index arrays expected by every direct sparse solver
//     (KLU, UMFPACK, MKL Pardiso, Eigen SparseLU when templated on int32)
//   * signed allows `-1` as a sentinel for ground / "no such device"
//     without burning bit-31
//   * 2^31 − 1 ≈ 2 billion nodes is plenty for any conceivable
//     power-electronics circuit (large industrial inverter ≤ 1000 nodes)
// -----------------------------------------------------------------------------
using Index = std::int32_t;

// Container sizes and iteration counts. Distinct from `Index` to keep
// node-ID semantics separate from "how many things are in this vector".
using Size = std::size_t;

// -----------------------------------------------------------------------------
// Compile-time sentinels.
//
// `kGround` and `kInvalidIndex` are the SAME value (`-1`). They're aliased
// rather than collapsed so that consumers can express intent:
//   - `Index n = kGround;`           // "this device's pin is the ground rail"
//   - `Index found = kInvalidIndex;` // "lookup failed"
// Same bit-pattern, different semantic — readers know which case they're in.
// -----------------------------------------------------------------------------
inline constexpr Index kInvalidIndex = Index{-1};
inline constexpr Index kGround       = Index{-1};

// -----------------------------------------------------------------------------
// Static-asserts that lock in the type contract at compile time.
//
// If a future change accidentally widens Index to 8 B (e.g. someone
// switches to int64_t), the build breaks LOUDLY at this line rather than
// silently halving cache density on every consumer.
// -----------------------------------------------------------------------------
static_assert(sizeof(Index) == 4,
              "pulsim::Index must be exactly 4 bytes (int32). Layer 0 "
              "design relies on this for cache density and solver compat.");
static_assert(std::numeric_limits<Index>::is_signed,
              "pulsim::Index must be signed so kGround = -1 works.");
static_assert(std::numeric_limits<Index>::max() >= 2'147'483'647,
              "pulsim::Index must hold at least 2^31 − 1.");

}  // namespace pulsim
