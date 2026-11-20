#pragma once

// =============================================================================
// Pulsim — lossless transmission line, Bergeron (method of characteristics)
// =============================================================================
//
// openspec add-realistic-passives section 5, DECISIONS D18. A two-conductor
// line of surge impedance Z0 and one-way delay tau between port a
// (a, a_ref) and port b (b, b_ref). With i_a, i_b the currents INTO the
// line at each port and the outgoing waves w_a = v_a + Z0·i_a,
// w_b = v_b + Z0·i_b, the line is exactly
//
//     v_a(t) − Z0·i_a(t) = w_b(t − tau)
//     v_b(t) − Z0·i_b(t) = w_a(t − tau)
//
// i.e. each port is a voltage source equal to the other end's wave tau ago
// behind Z0. Two branches (one per port), each with its own current
// unknown (the source rows of the state vector); the history lives in
// pwl::LineHistory (pwl/transmission_line_history.hpp). At DC (and in the
// AC sweep at omega = 0) it is a wire: v_a = v_b, i_a = −i_b. A series
// resistance is a builder composite (R/4, tau/2, R/2, tau/2, R/4).

#include "pulsim/numeric/types.hpp"
#include "pulsim/topology/graph.hpp"

#include <cmath>
#include <stdexcept>

namespace pulsim::models {

struct TransmissionLine {
    struct Params {
        Real Z0 = Real{0};    //!< surge impedance [Ω]
        Real delay = Real{0}; //!< one-way delay tau [s]
    };

    static constexpr topology::BranchKind kind = topology::BranchKind::Source;
    static constexpr bool is_linear = true;

    static void validate(const Params& p) {
        if (!(p.Z0 > Real{0}) || !std::isfinite(p.Z0)) {
            throw std::invalid_argument(
                "TransmissionLine: Z0 must be positive and finite (the surge "
                "impedance in ohms)");
        }
        if (!(p.delay > Real{0}) || !std::isfinite(p.delay)) {
            throw std::invalid_argument(
                "TransmissionLine: delay must be positive and finite (the "
                "one-way travel time in seconds)");
        }
    }
};

}  // namespace pulsim::models
