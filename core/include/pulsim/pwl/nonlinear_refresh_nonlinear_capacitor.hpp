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
#include <stdexcept>
#include <vector>

namespace pulsim::pwl {

/// Which integration rule the stamp is serving.
///
/// TR-BDF2's second stage uses a different derivative, so the
/// CHARGE history term differs — while the CONDUCTANCE does not,
/// because c1/h = 2/(gamma*h) is the identity the whole method
/// rests on. Stamping the trapezoidal history in a BDF2 stage
/// would therefore look plausible (right matrix, right sparsity,
/// converging Newton) and be wrong, which is why this is an
/// explicit mode rather than a default.
enum class CossStage {
    Trapezoidal,   //!< i = (2/h)(Q(v) - Q_n) - i_n
    Bdf2Stage2,    //!< i = (c1 Q(v) + c2 Q_gamma + c3 Q_n)/h
};

/// ADDS the Coss contributions to an already-zeroed (J_nl, f_nl).
/// Returns the largest |i| stamped, for the caller's residual
/// report.
///
/// For `Bdf2Stage2`, `h` is the FULL step (not gamma*h) and
/// `q_gamma` holds Q(v) at the stage point, one per device in
/// `history.entries()` order.
inline Real refresh_nonlinear_capacitors(
    const Vector& x,
    sparse::Matrix& J_nl,
    Vector& f_nl,
    const NonlinearCapacitorHistory& history,
    Real h,
    CossStage stage = CossStage::Trapezoidal,
    const std::vector<Real>* q_gamma = nullptr) {
    Real max_i = Real{0};
    if (history.empty() || !(h > Real{0})) {
        return max_i;
    }
    const Real two_over_h = Real{2} / h;
    const Real c1 = Real{2} + std::sqrt(Real{2});
    const Real gam = Real{2} - std::sqrt(Real{2});
    const Real rho = (Real{1} - gam) / gam;
    const Real c2 = -(Real{1} + rho) / (Real{1} - gam);
    const Real c3 = Real{1} / std::sqrt(Real{2});
    const bool bdf2 = (stage == CossStage::Bdf2Stage2);
    if (bdf2 && (q_gamma == nullptr
                  || q_gamma->size() != history.entries().size())) {
        throw std::invalid_argument(
            "refresh_nonlinear_capacitors: the BDF2 stage needs "
            "Q at the stage point, one per device");
    }
    Size idx = 0;
    for (const auto& e : history.entries()) {
        const Real v = stamping::read_node_voltage(x, e.from)
                       - stamping::read_node_voltage(x, e.to);
        const Real q =
            models::NonlinearCapacitor::charge(e.params, v);
        const Real c =
            models::NonlinearCapacitor::capacitance(e.params, v);
        const Real i =
            bdf2 ? (c1 * q + c2 * (*q_gamma)[idx] + c3 * e.q_prev)
                       / h
                 : two_over_h * (q - e.q_prev) - e.i_prev;
        // Same conductance either way: c1/h == 2/(gamma*h).
        const Real g = bdf2 ? c1 * c / h : two_over_h * c;
        ++idx;

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
