#pragma once

// =============================================================================
// Pulsim — excess (anomalous) core loss as a Newton law
// =============================================================================
//
// add-permeance-magnetic-domain 4.2. Bertotti's excess loss density
// k_x·|dB/dt|^1.5 of a core segment (area A, length l) is, in the
// gyrator–capacitor network, a branch whose MMF drop F [A] depends on the
// flux rate u = dΦ/dt [V] through it:
//
//     F = c·sign(u)·|u|^½,       c = k_x·l/√A     [A/V^½]
//
// That law has an infinite slope at u = 0 (not a Newton law: the project's
// laws are smooth and monotone to 1e-12). The design's regularisation:
//
//     F(u)  = c·u/(u² + ε²)^¼
//     F'(u) = c·(u²/2 + ε²)/(u² + ε²)^(5/4)  > 0   for every u
//
// analytic on the real line, odd, strictly increasing, linear with slope
// c/√ε inside |u| ≲ ε, and the square-root law for |u| ≫ ε. Under a
// sinusoidal flux rate of amplitude Û the mean power differs from the ideal
// law's c·Û^1.5·Γ(5/4)/(√π·Γ(7/4)) by the relative amount
//     δ = (3/4)·(ε/Û)² + O((ε/Û)^2.5)
// (the leading term is ε²/4·mean|u|^-½ over mean|u|^1.5, and
// Γ(1/4)Γ(7/4) / (Γ(3/4)Γ(5/4)) = 3), so ε ≤ Û·√(4δ/3) keeps the power
// within δ of the ideal law: ε = 1.15e-3·Û for 1e-6.
//
// The element is the DUAL seen through a unit winding (DECISIONS 12): on
// the winding's electrical node the flux rate is a voltage v = u and the
// MMF a current i = F, so the kernel kind is a voltage-controlled
// conductance i = F(v) between two electrical nodes — a static two-terminal
// Newton law like the Shockley diode, whose `current` body both device
// paths evaluate. (u²+ε²)^¼ is taken as sqrt(sqrt(·)): correctly rounded
// operations only, the same bits on every platform.

#include "pulsim/numeric/concepts.hpp"
#include "pulsim/numeric/types.hpp"
#include "pulsim/topology/graph.hpp"

#include <cmath>
#include <stdexcept>

namespace pulsim::models {

struct ExcessLoss {
    struct Params {
        Real c   = Real{0};   //!< excess-loss coefficient [A/V^½]
        Real eps = Real{0};   //!< regularisation rate ε [V] (flux rate)
    };

    static constexpr topology::BranchKind kind =
        topology::BranchKind::Nonlinear;
    static constexpr Size num_terminals = 2;
    static constexpr bool is_linear = false;

    static void validate(const Params& p) {
        if (!(p.c > Real{0}) || !std::isfinite(p.c)) {
            throw std::invalid_argument(
                "ExcessLoss: c must be positive and finite (the coefficient "
                "of F = c·sign(u)·|u|^0.5, in A/V^0.5)");
        }
        // floor 1e-150: below ~1.5e-162 eps² underflows and F(0) = 0/0
        if (!(p.eps >= Real{1e-150}) || !std::isfinite(p.eps)) {
            throw std::invalid_argument(
                "ExcessLoss: eps must be positive and finite, >= 1e-150 (the "
                "flux rate [V] below which the law is linear; eps <= 1.15e-3 x "
                "the rated flux-rate amplitude keeps the sine loss within 1e-6 "
                "of the square-root law)");
        }
    }

    /// MMF F [A] for the flux rate v[0] − v[1] [V] (the current from
    /// terminal 0 to terminal 1 of the electrical dual).
    template <numeric::FloatingPoint S, class P = Params>
    [[nodiscard]] static S current(const S* v, const P& p) noexcept {
        using std::sqrt;
        const S u = v[0] - v[1];
        const S r = sqrt(sqrt(u * u + p.eps * p.eps));
        return p.c * u / r;
    }

    /// dF/du in closed form (tests and the small-signal resistance).
    [[nodiscard]] static Real slope(const Params& p, Real u) noexcept {
        const Real s = u * u + p.eps * p.eps;
        return p.c * (u * u / Real{2} + p.eps * p.eps) / (s * std::sqrt(std::sqrt(s)));
    }
};

}  // namespace pulsim::models
