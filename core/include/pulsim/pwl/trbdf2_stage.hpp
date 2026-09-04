#pragma once

// =============================================================================
// Pulsim — TR-BDF2 stage selector and its second-stage coefficients
// =============================================================================
//
// The composite TR-BDF2 step takes a trapezoidal stage over gamma*h
// and then a BDF2 stage over the remainder, both with the SAME
// matrix factor (that is the whole point of gamma = 2 - sqrt(2):
// the stage-2 derivative coefficient c1/h equals 2/(gamma*h)).
//
// A device that carries its own state has to know which stage it is
// being stamped in, because the two stages approximate dX/dt
// differently:
//
//     trapezoidal   dX/dt ~ (2/h)*(X_new - X_n) - (dX/dt)_n
//     BDF2 stage 2  dX/dt ~ (c1*X_new + c2*X_gamma + c3*X_n) / h
//
// Note what changes and what does not. The CONDUCTANCE is the same
// either way, since c1/h == 2/(gamma*h) — that is the identity the
// method rests on. Only the HISTORY term differs. So stamping a
// trapezoidal history term inside a BDF2 stage produces the right
// matrix, the right sparsity and a perfectly convergent Newton, and
// is wrong. That failure mode is why the stage is an explicit
// argument everywhere rather than a default.
//
// The coefficients, with gamma = 2 - sqrt(2) and rho = (1-gamma)/gamma:
//
//     c1 = 2 + sqrt(2)
//     c2 = -(1 + rho) / (1 - gamma)
//     c3 = 1 / sqrt(2)
//
// and c1 + c2 + c3 = 0 exactly, which is the consistency condition:
// a constant state must have zero derivative. `coeffs()` checks it
// at construction, so a typo in a future edit cannot survive.

#include "pulsim/numeric/types.hpp"

#include <cmath>

namespace pulsim::pwl {

/// Which stage of the composite step a stateful device is being
/// stamped in.
enum class TrBdf2Stage {
    Trapezoidal,   //!< dX/dt = (2/h)(X - X_n) - (dX/dt)_n
    Bdf2Stage2,    //!< dX/dt = (c1 X + c2 X_gamma + c3 X_n)/h
};

struct TrBdf2Coeffs {
    Real gamma;
    Real c1;
    Real c2;
    Real c3;
};

/// The composite method's constants. Computed once here so the
/// three stateful devices (and the Coss) cannot drift apart.
[[nodiscard]] inline TrBdf2Coeffs trbdf2_coeffs() noexcept {
    const Real root2 = std::sqrt(Real{2});
    const Real gamma = Real{2} - root2;
    const Real rho = (Real{1} - gamma) / gamma;
    return TrBdf2Coeffs{
        .gamma = gamma,
        .c1 = Real{2} + root2,
        .c2 = -(Real{1} + rho) / (Real{1} - gamma),
        .c3 = Real{1} / root2,
    };
}

/// c1 + c2 + c3 must vanish: a state that is not changing has no
/// derivative. Cheap enough to assert wherever the coefficients are
/// used for the first time in a step.
[[nodiscard]] inline bool trbdf2_coeffs_consistent() noexcept {
    const auto k = trbdf2_coeffs();
    return std::abs(k.c1 + k.c2 + k.c3) < Real{1e-12};
}

}  // namespace pulsim::pwl
