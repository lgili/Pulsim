#pragma once

// =============================================================================
// Pulsim — Lauritzen–Mattsson diode: charge control and reverse recovery
// =============================================================================
//
// v2.0 Phase 4, audit C.1 ("Semicondutores nível-2", crítico).
// Every diode Pulsim had was STATIC I-V — the PWL `SwitchedDiode`,
// the smooth-blend `IdealDiode`, and the exponential
// `ShockleyDiode` all compute current from the present voltage
// alone. A static law cannot recover, because recovery is stored
// charge leaving the device, and a static law stores nothing.
//
// Measured on a double-pulse test (400 V rail, 20 A clamped
// inductive load, 50 nH commutation loop, low-side switch turning
// on):
//
//     diode              reverse peak     Q_rr
//     add_diode (PWL)      0.00000 A       0
//     Shockley             0.00000 A       0
//
// The 20 A commutates straight to zero. A real 600 V fast-recovery
// Si diode at that di/dt sweeps out several microcoulombs first,
// peaking 15–30 A NEGATIVE — and that current flows through the
// turning-on SWITCH, where it usually dominates turn-on loss. A
// simulator that reports zero is not slightly optimistic about
// hard-switched efficiency; it is silent about the largest term.
//
// THE MODEL (Lauritzen & Mattsson, 1991). Charge is the state:
//
//     i    = (q_E − q_M) / T_M
//     dq_M = (q_E − q_M) / T_M − q_M / tau
//     dt
//
// with the junction charge set by the usual exponential
//
//     q_E(v) = tau · I_S · (exp(v / (n·V_T)) − 1)
//
// `tau` is the carrier lifetime (how fast stored charge
// recombines) and `T_M` the transit time. In steady state
// dq_M/dt = 0 gives i = q_E/(tau + T_M), i.e. an ordinary Shockley
// characteristic with I_S scaled by tau/(tau + T_M) — so the DC
// curve is unchanged and only the DYNAMICS are new.
//
// Recovery falls straight out. Force the current negative and
// q_E collapses with the voltage, but q_M cannot: it can only
// decay at rate 1/tau. So i = (q_E − q_M)/T_M ≈ −q_M/T_M, a large
// reverse current that dies as the charge drains. `tau` sets how
// long, `T_M` how hard.
//
// WHY THE STAMP IS CHEAP. Trapezoidal on q_M, with
// A = 1/T_M + 1/tau and f = q_E/T_M − A·q_M:
//
//     q_M^{n+1}·(1 + hA/2) = q_M^n + (h/2)·f^n
//                            + (h / (2·T_M))·q_E^{n+1}
//
// so q_M^{n+1} = K0 + K1·q_E(v) is AFFINE in q_E — no inner
// iteration, no extra unknown, no internal node:
//
//     i(v)   = [ (1 − K1)·q_E(v) − K0 ] / T_M
//     di/dv  = (1 − K1)/T_M · dq_E/dv
//
// K0 and K1 depend only on the history and the step, so they are
// computed once per step and the Newton loop sees an ordinary
// two-terminal nonlinear branch.
//
// The exponential carries the same tangent continuation as
// `ShockleyDiode` — see that header for why `v_lim` is set from a
// current ceiling rather than from SPICE's `vcrit`.

#include "pulsim/models/shockley_diode.hpp"
#include "pulsim/numeric/concepts.hpp"
#include "pulsim/numeric/types.hpp"
#include "pulsim/topology/graph.hpp"

#include <cmath>
#include <stdexcept>

namespace pulsim::models {

/// Charge-controlled p-n junction with reverse recovery.
struct LauritzenDiode {
    struct Params {
        Real I_S   = Real{1e-12};    //!< saturation current [A]
        Real n     = Real{1.0};      //!< emission coefficient
        Real V_T   = Real{0.025852}; //!< thermal voltage kT/q [V]
        /// Carrier lifetime [s]. Sets HOW LONG recovery lasts.
        /// Fast-recovery Si: 10–100 ns. Standard rectifier:
        /// microseconds. A Schottky has no stored charge at all —
        /// use `ShockleyDiode` for one, not a tiny `tau` here.
        Real tau   = Real{1e-7};
        /// Transit time [s]. Sets HOW HARD the reverse peak is,
        /// and must stay well below `tau`.
        Real T_M   = Real{1e-8};
        Real G_min = Real{1e-12};    //!< parallel conductance [S]
        /// Current ceiling for the exponential's tangent
        /// continuation; see `ShockleyDiode`.
        Real i_lim = Real{1e6};      //!< [A]
    };

    static constexpr topology::BranchKind kind =
        topology::BranchKind::Nonlinear;
    static constexpr Size num_terminals = 2;
    static constexpr bool is_linear = false;

    static void validate(const Params& p) {
        if (!(p.I_S > Real{0})) {
            throw std::invalid_argument(
                "LauritzenDiode: I_S must be positive");
        }
        if (!(p.n > Real{0})) {
            throw std::invalid_argument(
                "LauritzenDiode: n must be positive; physical "
                "values are 1 to 2");
        }
        if (!(p.V_T > Real{0})) {
            throw std::invalid_argument(
                "LauritzenDiode: V_T must be positive (kT/q, "
                "25.85 mV at 300 K)");
        }
        if (!(p.tau > Real{0})) {
            throw std::invalid_argument(
                "LauritzenDiode: tau (carrier lifetime) must be "
                "positive — it is the whole point of this model. "
                "A device with no stored charge is a Schottky; "
                "use add_shockley_diode for that instead of "
                "driving tau to zero here");
        }
        if (!(p.T_M > Real{0})) {
            throw std::invalid_argument(
                "LauritzenDiode: T_M (transit time) must be "
                "positive");
        }
        if (!(p.T_M < p.tau)) {
            throw std::invalid_argument(
                "LauritzenDiode: T_M must be smaller than tau. "
                "The transit time is how fast injected charge "
                "crosses the base and the lifetime is how long it "
                "survives there; T_M >= tau describes no real "
                "device, and it makes the steady-state current "
                "I_S*tau/(tau+T_M) collapse so the DC curve stops "
                "matching the I_S you asked for");
        }
        if (p.G_min < Real{0}) {
            throw std::invalid_argument(
                "LauritzenDiode: G_min must be >= 0");
        }
        if (!(p.i_lim > Real{0})) {
            throw std::invalid_argument(
                "LauritzenDiode: i_lim must be positive");
        }
    }

    /// The Shockley parameters this model's junction charge uses.
    [[nodiscard]] static ShockleyDiode::Params junction(
        const Params& p) noexcept {
        ShockleyDiode::Params s;
        s.I_S   = p.I_S;
        s.n     = p.n;
        s.V_T   = p.V_T;
        s.G_min = Real{0};     // added once, at the branch level
        s.i_lim = p.i_lim;
        s.BV    = Real{0};
        return s;
    }

    /// Junction charge q_E(v) = tau · i_shockley(v) [C].
    ///
    /// Templated so Newton gets dq_E/dv from the same expression.
    template <numeric::FloatingPoint S>
    [[nodiscard]] static S junction_charge(const S& v,
                                            const Params& p)
        noexcept {
        const S vv[2] = {v, S{0}};
        return S{p.tau} * ShockleyDiode::current<S>(vv,
                                                     junction(p));
    }

    /// Steady-state current at `v` — what the DC curve is, and
    /// what a static Shockley with I_S·tau/(tau+T_M) would give.
    [[nodiscard]] static Real steady_state_current(
        Real v, const Params& p) noexcept {
        return junction_charge<Real>(v, p) / (p.tau + p.T_M);
    }

    /// Stored charge at a steady forward current `i` [C]. This is
    /// the charge recovery has to remove, so it is the number that
    /// predicts Q_rr.
    [[nodiscard]] static Real stored_charge_at_current(
        Real i, const Params& p) noexcept {
        return i * p.tau;
    }
};

}  // namespace pulsim::models
