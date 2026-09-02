#pragma once

// =============================================================================
// Pulsim — Newton stamp for the Lauritzen–Mattsson diode
// =============================================================================
//
// The charge state q_M is eliminated before Newton ever sees it.
// `LauritzenDiodeHistory::begin_step(h)` reduces the trapezoidal
// rule to q_M = K0 + K1·q_E(v), so the branch is an ordinary
// two-terminal nonlinearity:
//
//     i(v)  = [ (1 − K1)·q_E(v) − K0 ] / T_M  +  v·G_min
//     di/dv = (1 − K1)/T_M · dq_E/dv          +  G_min
//
// with dq_E/dv taken from the same AD evaluation as q_E, so the
// Jacobian cannot drift from the residual.
//
// The coefficients are refreshed HERE, from the history, on every
// call. K0 and K1 embed `h`, and a variable-step engine retries a
// rejected step at a different one — so an API that computed them
// once per step could stamp the wrong interval while Newton
// converged perfectly happily. Recomputing costs a few flops per
// device and makes that impossible.

#include "pulsim/ad/ad_scalar.hpp"
#include "pulsim/models/lauritzen_diode.hpp"
#include "pulsim/numeric/types.hpp"
#include "pulsim/pwl/lauritzen_diode_history.hpp"
#include "pulsim/sparse/matrix.hpp"
#include "pulsim/stamping/branch_coord.hpp"

#include <array>
#include <cmath>

namespace pulsim::pwl {

/// Stamp every Lauritzen diode. Returns the largest |i| stamped,
/// which the Newton loop uses as its residual indicator.
inline Real refresh_lauritzen_diodes(
    const Vector& x,
    sparse::Matrix& J_nl,
    Vector& f_nl,
    LauritzenDiodeHistory& history,
    Real h) {
    Real max_i = Real{0};
    if (history.empty() || !(h > Real{0})) {
        return max_i;
    }
    history.begin_step(h);
    for (const auto& e : history.entries()) {
        const Real v = stamping::read_node_voltage(x, e.from)
                       - stamping::read_node_voltage(x, e.to);

        // q_E and dq_E/dv from one AD evaluation.
        using AD = ad::ADRealN<1>;
        const AD v_ad{v, std::array<Real, 1>{Real{1}}};
        const AD q_e_ad =
            models::LauritzenDiode::junction_charge<AD>(v_ad,
                                                         e.params);
        const Real q_e = q_e_ad.value();
        const Real dq_e_dv = q_e_ad.deriv(0);

        const Real scale = (Real{1} - e.k1) / e.params.T_M;
        const Real i = scale * q_e - e.k0 / e.params.T_M
                       + v * e.params.G_min;
        const Real g = scale * dq_e_dv + e.params.G_min;

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
