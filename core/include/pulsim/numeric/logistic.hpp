#pragma once

// =============================================================================
// Pulsim — the logistic sigmoid, evaluated so it cannot overflow
// =============================================================================
//
// v2.0 Phase 2, closing a loud-but-avoidable failure that ended runs
// on ordinary circuits.
//
// THE BUG. Every smooth device model in Pulsim blends its regions
// with a logistic, written the textbook way:
//
//     alpha = 1 / (1 + exp(-kappa * u))
//
// For u sufficiently negative the exponent is a large POSITIVE
// number, and `std::exp` overflows to +inf. The VALUE survives that
// — 1/(1+inf) is 0, which is the right answer — but the DERIVATIVE
// does not. Forward-mode AD (`ad::exp`) propagates
// `d = exp(x) * dx`, so `d` is inf too, and the reciprocal's
// derivative is inf/inf = NaN. One NaN in the Jacobian defeats
// Levenberg-Marquardt at every lambda, and the run dies with
//
//     solve_with_newton (LM): factor failed at λ = 1e+09
//
// The threshold is exactly `kappa * |u| > 709`, the double-precision
// exp limit. At the default kappa = 20 that is 35 V of reverse bias
// — an ordinary mains rectifier reaches it in the first half-cycle.
// It is a property of how the formula is WRITTEN, not of the
// circuit, the model, or the time step: no smaller dt reduces the
// reverse voltage a diode stands off, which is set by the source.
//
// THE FIX. Evaluate the same function with the exponent's sign
// forced non-positive:
//
//     u >= 0:   1 / (1 + exp(-u))
//     u <  0:   e / (1 + e),  e = exp(u)
//
// The two branches are algebraically identical — divide the second
// through by e — and both keep `exp` in (0, 1], where neither the
// value nor its derivative can overflow. They agree exactly at u = 0
// (1/2, slope 1/4), so the piecewise definition is smooth, and the
// derivative each branch produces under AD is the true
// alpha*(1 - alpha) in both cases.
//
// This is the same job SPICE's `pnjlim` does for a real Shockley
// junction: keep the exponential's argument inside the range the
// arithmetic can represent. Doing it in the formula rather than by
// limiting the Newton step is stronger — there is no iterate from
// which it can fail.

#include "pulsim/numeric/concepts.hpp"
#include "pulsim/numeric/types.hpp"

#include <cmath>

namespace pulsim::numeric {

/// The logistic `1 / (1 + exp(-z))`, overflow-free for every finite
/// `z` and for every scalar type — `Real` or a forward-AD scalar.
///
/// The branch is on the VALUE of `z`, which is safe precisely
/// because the two expressions are the same function: an AD scalar
/// crossing zero sees no discontinuity in either the value or the
/// derivative.
template <FloatingPoint S>
[[nodiscard]] inline S logistic(const S& z) noexcept {
    using std::exp;
    if (z >= Real{0}) {
        return S{1} / (S{1} + exp(-z));
    }
    const S e = exp(z);
    return e / (S{1} + e);
}

}  // namespace pulsim::numeric
