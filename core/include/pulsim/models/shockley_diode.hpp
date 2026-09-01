#pragma once

// =============================================================================
// Pulsim — exponential (Shockley) diode
// =============================================================================
//
// v2.0 Phase 4, audit C.1. Pulsim had two diodes and neither was
// exponential:
//
//   * `SwitchedDiode` — binary PWL, one conductance per state.
//   * `IdealDiode`    — a sigmoid blended onto a STRAIGHT LINE,
//                       i = (v − V_F0)/R_d above the knee.
//
// Both fix the forward drop by construction. A real junction does
// not: V_F rises about 60 mV per decade of current, so the same
// device that drops 0.53 V at 1 mA drops 0.77 V at 10 A. Over the
// decades a converter actually spans between light and full load,
// a fixed-drop model is wrong by a couple of hundred millivolts in
// the one quantity that sets conduction loss — and wrong in
// OPPOSITE directions at the two ends, which is an error no single
// fitted V_F0 can remove.
//
// THE LAW.
//
//     i = I_S · (exp(v / (n·V_T)) − 1) + v·G_min
//
// with V_T = k·T/q (25.85 mV at 300 K). `G_min` is SPICE's
// parallel conductance: it keeps a reverse-biased junction from
// leaving its node with no path to ground, which is a singular
// matrix rather than a wrong answer.
//
// KEEPING THE EXPONENT FINITE. Newton's first trial step on a
// rectifier easily proposes v = 50 V across a junction, and
// exp(50/0.02585) = e^1934 is +inf; the Jacobian is +inf too, the
// update is NaN, and the run is over. SPICE handles this with
// `pnjlim`, which limits the per-iteration voltage STEP — the
// converged answer still sits on the true exponential. Pulsim's
// Newton loop is device-agnostic and offers no per-device limiter
// hook, so the same protection is built into the law:
//
//     above v_lim, continue the curve by its own tangent
//         i(v) = i(v_lim) + g(v_lim)·(v − v_lim)
//
// C¹ at the join, monotone, and finite for any voltage at all, so
// Newton always gets a descent direction pointing back toward the
// solution.
//
// WHERE v_lim GOES IS THE WHOLE DESIGN. It must be far above any
// current a real circuit carries, or the model silently becomes a
// resistor in its working range. SPICE's own `vcrit` — the
// voltage where the exponential's curvature stops being
// resolvable — is NOT usable for this, because it sits at only
// ≈ 18 mA for I_S = 1e-12: a tangent from there gives a 1.41 Ω
// device, which at 10 A reports 14.7 V instead of 0.77 V. (That
// mistake was made and measured here before this comment was
// written.) `v_lim` is therefore set from a CURRENT ceiling,
// `i_lim`, defaulting to 1e6 A — four orders past the largest
// converter anyone simulates — so the tangent region is
// unreachable by any converged solution and exists purely to keep
// Newton's excursions finite.
//
// NO SERIES RESISTANCE PARAMETER, ON PURPOSE. `R_S` cannot be
// folded into a two-terminal law without either an internal node
// or an inner iteration, and the obvious inner iteration
// (v_j ← v − i(v_j)·R_S) is NOT a contraction where it matters:
// at 10 A the junction's own dynamic resistance is 2.6 mΩ, so the
// map's gain is R_S/r_j ≈ 4 and it diverges. Put an ordinary
// resistor in series instead — it is exact, and the existing
// branch machinery already solves it.

#include "pulsim/numeric/concepts.hpp"
#include "pulsim/numeric/types.hpp"
#include "pulsim/topology/graph.hpp"

#include <cmath>
#include <stdexcept>

namespace pulsim::models {

/// Exponential p-n junction.
struct ShockleyDiode {
    struct Params {
        Real I_S   = Real{1e-12};    //!< saturation current [A]
        Real n     = Real{1.0};      //!< emission (ideality) coeff
        Real V_T   = Real{0.025852}; //!< thermal voltage kT/q [V]
        Real G_min = Real{1e-12};    //!< parallel conductance [S]
        /// Current above which the exponential is continued by
        /// its own tangent, purely so Newton cannot overflow.
        /// Must sit far above the circuit's real currents; see
        /// the header for what happens when it does not.
        Real i_lim = Real{1e6};      //!< [A]
        /// Reverse breakdown knee [V], as a POSITIVE magnitude.
        /// 0 disables it — the junction then blocks forever,
        /// which is what a rectifier model wants. Set it for a
        /// Zener or a TVS.
        Real BV    = Real{0};
    };

    static constexpr topology::BranchKind kind =
        topology::BranchKind::Nonlinear;
    static constexpr Size num_terminals = 2;
    static constexpr bool is_linear = false;

    static void validate(const Params& p) {
        if (!(p.I_S > Real{0})) {
            throw std::invalid_argument(
                "ShockleyDiode: I_S must be positive — it is the "
                "prefactor of the exponential, and at zero the "
                "device is an open circuit with no reverse "
                "leakage either, which leaves a singular node");
        }
        if (!(p.n > Real{0})) {
            throw std::invalid_argument(
                "ShockleyDiode: n (emission coefficient) must be "
                "positive; physical values are 1 to 2");
        }
        if (!(p.V_T > Real{0})) {
            throw std::invalid_argument(
                "ShockleyDiode: V_T must be positive (it is kT/q, "
                "25.85 mV at 300 K)");
        }
        if (p.G_min < Real{0}) {
            throw std::invalid_argument(
                "ShockleyDiode: G_min must be >= 0");
        }
        if (!(p.i_lim > Real{0})) {
            throw std::invalid_argument(
                "ShockleyDiode: i_lim must be positive — it is "
                "the current above which the exponential is "
                "continued linearly so Newton cannot overflow, "
                "and it must sit far above any current the "
                "circuit actually carries");
        }
        if (p.BV < Real{0}) {
            throw std::invalid_argument(
                "ShockleyDiode: BV is a magnitude, so it must be "
                ">= 0 (0 disables breakdown). Give 5.1 for a "
                "5.1 V Zener, not -5.1");
        }
    }

    /// Thermal voltage kT/q at a junction temperature [K].
    ///
    /// NOTE, because it is the opposite of what everyone
    /// remembers: raising V_T ALONE raises the forward drop, it
    /// does not lower it. At a fixed current and fixed I_S,
    /// V_F = n·V_T·ln(i/I_S), so V_T and V_F move together —
    /// measured here, 0.710 V at 25 °C becomes 0.948 V at 125 °C.
    /// A real diode's well-known NEGATIVE temperature coefficient
    /// comes from I_S, which roughly doubles every 10 °C and
    /// overwhelms the kT/q term. So a temperature sweep must
    /// scale BOTH: pair this with `saturation_current_at`.
    [[nodiscard]] static Real thermal_voltage(Real T_kelvin)
        noexcept {
        constexpr Real k_over_q = Real{8.617333262e-5};  // [V/K]
        return k_over_q * T_kelvin;
    }

    /// I_S at temperature `T`, given its value at `T_ref` —
    /// SPICE's law:
    ///
    ///   I_S(T) = I_S(T_ref)·(T/T_ref)^(XTI/n)
    ///            · exp( E_g·q/(n·k) · (1/T_ref − 1/T) )
    ///
    /// with XTI = 3 (the saturation-current temperature exponent)
    /// and E_g = 1.11 eV for silicon. This is the term that makes
    /// a hot diode drop LESS; see `thermal_voltage`.
    [[nodiscard]] static Real saturation_current_at(
        Real I_S_ref, Real T_kelvin,
        Real T_ref_kelvin = Real{300.15},
        Real n = Real{1.0},
        Real E_g_eV = Real{1.11},
        Real XTI = Real{3.0}) noexcept {
        constexpr Real k_over_q = Real{8.617333262e-5};  // [V/K]
        const Real r = T_kelvin / T_ref_kelvin;
        return I_S_ref * std::pow(r, XTI / n)
               * std::exp(E_g_eV / (n * k_over_q)
                          * (Real{1} / T_ref_kelvin
                             - Real{1} / T_kelvin));
    }

    /// Junction voltage at which the tangent continuation starts,
    /// i.e. where the exponential reaches `i_lim`.
    [[nodiscard]] static Real v_lim(const Params& p) noexcept {
        const Real vte = p.n * p.V_T;
        return vte * std::log(p.i_lim / p.I_S + Real{1});
    }

    /// The forward voltage that carries `i` amps — the closed-form
    /// inverse, useful for sizing and for tests.
    [[nodiscard]] static Real voltage_for_current(
        const Params& p, Real i) noexcept {
        const Real vte = p.n * p.V_T;
        return vte * std::log(i / p.I_S + Real{1});
    }

    /// Current from terminal 0 (anode) to terminal 1 (cathode).
    ///
    /// Templated on `numeric::FloatingPoint S`; instantiates for
    /// `Real` (forward) and `ADRealN<2>` (Newton Jacobian).
    template <numeric::FloatingPoint S>
    [[nodiscard]] static S current(const S* v, const Params& p)
        noexcept {
        const S vd = v[0] - v[1];
        S i = limited_exp_branch_(vd, p);

        // Reverse breakdown, mirrored through the same limiter so
        // it cannot overflow either. Disabled at BV = 0.
        if (p.BV > Real{0}) {
            const S over = S{-p.BV} - vd;   // > 0 past the knee
            i = i - limited_exp_branch_(over, p);
        }

        return i + vd * p.G_min;
    }

private:
    /// I_S·(exp(u/vte) − 1), continued by its tangent above
    /// `v_lim` so it is finite for every u.
    template <numeric::FloatingPoint S>
    [[nodiscard]] static S limited_exp_branch_(const S& u,
                                                const Params& p)
        noexcept {
        using std::exp;
        const Real vte = p.n * p.V_T;
        const Real vl  = v_lim(p);
        if (u > S{vl}) {
            const Real e_l = std::exp(vl / vte);
            const Real i_l = p.I_S * (e_l - Real{1});
            const Real g_l = p.I_S * e_l / vte;
            return S{i_l} + S{g_l} * (u - S{vl});
        }
        return p.I_S * (exp(u / S{vte}) - S{1});
    }
};

}  // namespace pulsim::models
