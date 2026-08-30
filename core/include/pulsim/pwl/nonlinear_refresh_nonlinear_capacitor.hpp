#pragma once

// =============================================================================
// Pulsim — Newton stamp for the charge-based nonlinear capacitor
// =============================================================================
//
// Phase 4, audit C.1. The trapezoidal companion written on CHARGE:
//
//     i(v) = (2/h)·(Q(v) − Q_n) − i_n
//     ∂i/∂v = (2/h)·C(v)
//
// stamped in the ordinary two-terminal pattern. Writing it on Q
// rather than on C·v is what conserves charge exactly no matter
// how sharply C(v) varies — and charge is what decides whether a
// half-bridge reaches ZVS within its dead time.

#include "pulsim/models/nonlinear_capacitor.hpp"
#include "pulsim/numeric/types.hpp"
#include "pulsim/pwl/nonlinear_capacitor_history.hpp"
#include "pulsim/sparse/matrix.hpp"
#include "pulsim/stamping/branch_coord.hpp"

#include <cmath>

namespace pulsim::pwl {

/// ADDS the Coss contributions to an already-zeroed (J_nl, f_nl).
/// Returns the largest |i| stamped, for the caller's residual
/// report.
inline Real refresh_nonlinear_capacitors(
    const Vector& x,
    sparse::Matrix& J_nl,
    Vector& f_nl,
    const NonlinearCapacitorHistory& history,
    Real h) {
    Real max_i = Real{0};
    if (history.empty() || !(h > Real{0})) {
        return max_i;
    }
    const Real two_over_h = Real{2} / h;
    for (const auto& e : history.entries()) {
        const Real v = stamping::read_node_voltage(x, e.from)
                       - stamping::read_node_voltage(x, e.to);
        const Real q =
            models::NonlinearCapacitor::charge(e.params, v);
        const Real c =
            models::NonlinearCapacitor::capacitance(e.params, v);
        const Real i = two_over_h * (q - e.q_prev) - e.i_prev;
        const Real g = two_over_h * c;

        const bool a = stamping::node_is_active(e.from);
        const bool b = stamping::node_is_active(e.to);
        if (a) {
            f_nl[e.from] += i;
            J_nl.coeffRef(e.from, e.from) += g;
        }
        if (b) {
            f_nl[e.to] -= i;
            J_nl.coeffRef(e.to, e.to) += g;
        }
        if (a && b) {
            J_nl.coeffRef(e.from, e.to) -= g;
            J_nl.coeffRef(e.to, e.from) -= g;
        }
        max_i = std::max(max_i, std::abs(i));
    }
    return max_i;
}

}  // namespace pulsim::pwl
