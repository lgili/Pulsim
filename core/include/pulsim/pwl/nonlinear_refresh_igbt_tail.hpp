#pragma once

// =============================================================================
// Pulsim — Newton stamp for the IGBT turn-off tail
// =============================================================================
//
// The charge state Q is eliminated before Newton sees it. With
// Q = K0 + K1·i_ss(v) from the history:
//
//     i_C   = (1 − k)·i_ss(v) + Q/tau
//           = i_ss(v)·[(1 − k) + K1/tau] + K0/tau
//
// THIS PASS STAMPS THE DIFFERENCE, not the total. The ordinary
// IGBT refresh already stamps i_ss for every IGBT including these,
// and the refresh passes are additive by design, so this one adds
//
//     i_delta   = (scale − 1)·i_ss(v) + K0/tau
//     di/dv     = (scale − 1)·di_ss/dv,
//     scale − 1 = K1/tau − k
//
// which composes with the existing pass instead of having to
// exclude branches from it. It is also self-checking: in
// equilibrium Q = k·i_ss·tau and f = 0 give
// K0/tau = k·i_ss/den and K1/tau = h·k/(2·tau·den) with
// den = 1 + h/(2·tau), so i_delta collapses to exactly zero and
// the device reduces to its steady-state law — which is the
// property that keeps the DC curve untouched.

#include "pulsim/ad/ad_scalar.hpp"
#include "pulsim/models/igbt_level1.hpp"
#include "pulsim/numeric/types.hpp"
#include "pulsim/pwl/igbt_tail_history.hpp"
#include "pulsim/sparse/matrix.hpp"
#include "pulsim/stamping/branch_coord.hpp"

#include <array>
#include <cmath>

namespace pulsim::pwl {

/// Stamp every IGBT that carries a turn-off tail. Returns the
/// largest |i| stamped, for the Newton residual indicator.
inline Real refresh_igbt_tails(
    const Vector& x,
    sparse::Matrix& J_nl,
    Vector& f_nl,
    IgbtTailHistory& history,
    Real h,
    TrBdf2Stage stage = TrBdf2Stage::Trapezoidal) {
    Real max_i = Real{0};
    if (history.empty() || !(h > Real{0})) {
        return max_i;
    }
    history.begin_step(h, stage);
    for (const auto& e : history.entries()) {
        // i_ss and its three partials from one AD evaluation, so
        // the Jacobian cannot drift from the residual.
        using AD = ad::ADRealN<3>;
        const AD v[3] = {
            AD{stamping::read_node_voltage(x, e.collector),
               std::array<Real, 3>{Real{1}, Real{0}, Real{0}}},
            AD{stamping::read_node_voltage(x, e.emitter),
               std::array<Real, 3>{Real{0}, Real{1}, Real{0}}},
            AD{stamping::read_node_voltage(x, e.gate),
               std::array<Real, 3>{Real{0}, Real{0}, Real{1}}},
        };
        const AD i_ss_ad =
            models::IgbtLevel1::steady_state_current<AD>(v,
                                                          e.params);

        // scale - 1: this pass is a delta on the plain stamp.
        const Real scale = e.k1 / e.params.tau_tail
                           - e.params.k_tail;
        const Real i = scale * i_ss_ad.value()
                       + e.k0 / e.params.tau_tail;

        const Index n[3] = {e.collector, e.emitter, e.gate};
        // Current leaves the collector and enters the emitter;
        // the gate carries none (ideal gate), but it still
        // contributes Jacobian columns.
        if (stamping::node_is_active(e.collector)) {
            f_nl[e.collector] += i;
        }
        if (stamping::node_is_active(e.emitter)) {
            f_nl[e.emitter] -= i;
        }
        for (Size c = 0; c < 3; ++c) {
            const Real g = scale * i_ss_ad.deriv(c);
            if (!stamping::node_is_active(n[c])) {
                continue;
            }
            if (stamping::node_is_active(e.collector)) {
                J_nl.coeffRef(e.collector, n[c]) += g;
            }
            if (stamping::node_is_active(e.emitter)) {
                J_nl.coeffRef(e.emitter, n[c]) -= g;
            }
        }
        max_i = std::max(max_i, std::abs(i));
    }
    return max_i;
}

}  // namespace pulsim::pwl
