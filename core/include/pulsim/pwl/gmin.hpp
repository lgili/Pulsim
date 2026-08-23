#pragma once

// =============================================================================
// Pulsim — Layer 4: gmin conductance floor + gmin stepping
// =============================================================================
//
// v2.0 Phase 2 (B.2), closing audit finding `no-gmin-infrastructure`.
//
// TWO DIFFERENT JOBS, ONE CONDUCTANCE.
//
//   * The FLOOR is a permanent, tiny conductance (1 pS) from every
//     non-ground node to ground, present in every DC assembly. It
//     exists to keep the factorization well-pivoted when a node is
//     connected to the rest of the circuit only through something
//     that is nearly an open circuit — every diode in a bridge
//     reverse-biased at 1e-15 S, a MOSFET below threshold, an open
//     switch at g_off. Those matrices are not singular, they are
//     merely appalling, and LU on them produces a solution with no
//     significant digits left.
//
//   * STEPPING is a homotopy. When the DC solve fails anyway, start
//     from a conductance so large the circuit is trivially solvable
//     (every node effectively shorted to ground through 10 mS),
//     then walk it back down by decades, warm-starting each solve
//     from the last. Newton only ever has to cross one decade of
//     nonlinearity at a time. This is the standard SPICE recovery
//     and it is what turns "Newton failed, good luck" into an
//     answer.
//
// WHY THE FLOOR DOES NOT REPLACE THE TOPOLOGY DIAGNOSTICS.
//
// A conductance to ground on every node would also make a
// structurally floating node solvable — the empty MNA column gets a
// diagonal, LU succeeds, and the user is handed v = 0 for a node
// that has no defined voltage at all. That is precisely the silent
// wrong answer Phase 1 taught the kernel to name and Phase 2's
// preflight taught the builder to repair. So the floor is stamped
// only AFTER a structural probe of the un-augmented matrix: if a
// node row is empty on its own merits, the named error still wins.
// gmin is for conditioning; `preflight.hpp` is for topology; they
// must not cover for each other.
//
// WHERE THE FLOOR IS *NOT* APPLIED. The transient does not need it:
// at dt > 0 every capacitor stamps 2C/dt on its terminals, which is
// a far larger diagonal than gmin would add, and the PWL cache
// factorizes each switch mask once and reuses it for millions of
// steps — a perturbation there would be paid forever for a benefit
// that only exists at DC, where capacitors are open.
//
// SIZING. floor = 1e-12 S is SPICE's GMIN. On a 12 V node that is
// 12 pA of leakage; against the 1e-9 S that `preflight.hpp` inserts
// for a genuinely unreferenced subnet it is three orders down, so
// the two never compete for the same job. `start = 1e-2` S with ten
// decades of ramp lands exactly on the floor.

#include "pulsim/numeric/dense.hpp"
#include "pulsim/numeric/types.hpp"
#include "pulsim/sparse/matrix.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace pulsim::pwl {

/// SPICE's GMIN. The conductance the DC solvers stamp node-to-ground
/// unless told otherwise.
inline constexpr Real kDefaultGmin = Real{1e-12};

/// Knobs for the conductance floor and the stepping homotopy.
struct GminConfig {
    /// Conductance (S) from every non-ground node to ground, stamped
    /// into every DC assembly. Zero disables the floor entirely.
    Real floor = kDefaultGmin;

    /// First rung of the stepping ramp (S). Must exceed `floor` for
    /// stepping to have anything to walk down.
    Real start = Real{1e-2};

    /// Number of rungs between `start` and `floor`, inclusive of
    /// `start`. A final solve at `floor` always follows.
    Size steps = Size{10};

    /// Newton budget per rung. Rungs are meant to be easy — a rung
    /// that needs 50 iterations means the ramp is too coarse.
    Size max_newton_iters = Size{50};
    Real tol_dx  = Real{1e-9};
    Real tol_res = Real{1e-9};

    /// Globalization inside each rung. OFF by default: the homotopy
    /// IS the globalization, and line search / Levenberg-Marquardt
    /// on top of an already-easy rung damps the step so hard that
    /// Newton stalls short of tolerance instead of converging
    /// quadratically. Turn them on only for a rung that genuinely
    /// will not close.
    bool enable_line_search = false;
    bool enable_lm = false;

    /// Reject a returned operating point whose residual in the
    /// UN-augmented system exceeds this. Catches the case where the
    /// floor turned out to be load-bearing, i.e. the answer depends
    /// on a conductance the user never put in the circuit.
    Real max_unaugmented_residual = Real{1e-6};
};

/// Add `g` siemens from every non-ground node to ground.
///
/// MNA row layout is `[v_0 .. v_{N-1} | i_src.. | i_L..]` and ground
/// is `kGround = -1`, which owns no row (`stamping::node_is_active`
/// rejects it), so rows `[0, num_nodes)` are exactly the non-ground
/// node voltages. Adding `g` to `J(i, i)` over that range is
/// therefore *exactly* a resistor of 1/g from node i to ground —
/// nothing else.
///
/// The branch-current rows are deliberately untouched. A diagonal
/// term on a voltage-source constraint row is not a conductance to
/// ground; it is a fictitious resistance in series with the source,
/// which changes the circuit rather than conditioning it.
inline void stamp_gmin(sparse::Matrix& J, Index num_nodes,
                        Real g) {
    if (!(g > Real{0})) {
        return;
    }
    const Index n = std::min<Index>(num_nodes,
                                     static_cast<Index>(J.rows()));
    for (Index i = 0; i < n; ++i) {
        J.coeffRef(i, i) += g;
    }
}

/// The conductance ramp, largest first, terminating exactly on
/// `cfg.floor` (or on 0 when the floor is disabled).
///
/// Geometric in `g`, i.e. uniform in log10 — the parameter Newton
/// actually feels, because a decade of conductance is a decade of
/// how hard the nonlinear devices are being clamped.
[[nodiscard]] inline std::vector<Real> gmin_ramp(
    const GminConfig& cfg) {
    std::vector<Real> ramp;
    const Real last = std::max(cfg.floor, Real{0});
    if (!(cfg.start > Real{0}) || cfg.steps == 0 ||
        !(cfg.start > last)) {
        ramp.push_back(last);
        return ramp;
    }
    // Walk `steps` rungs from start down to (but excluding) the
    // terminal value, then append the terminal value itself.
    const Real lo = std::max(last, cfg.start * Real{1e-18});
    const Real log_start = std::log10(cfg.start);
    const Real log_lo    = std::log10(lo);
    const Real span      = log_start - log_lo;
    ramp.reserve(cfg.steps + 1);
    for (Size k = 0; k < cfg.steps; ++k) {
        const Real frac = static_cast<Real>(k) /
                            static_cast<Real>(cfg.steps);
        ramp.push_back(std::pow(Real{10}, log_start - span * frac));
    }
    ramp.push_back(last);
    return ramp;
}

}  // namespace pulsim::pwl
