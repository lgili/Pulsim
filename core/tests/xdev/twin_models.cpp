// External-device twins of two built-in laws (add-external-device-abi).
// ILLUSTRATIONS for the tests: synthetic parameter defaults, not real parts.
// Built as a separate shared library and loaded at run time; written with
// the C++ helper (pulsim/xdev/adapter.hpp), so the Jacobian comes from AD
// inside this library and only doubles cross the C boundary.

#include "pulsim/xdev/adapter.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>

namespace {
template <std::size_t R>
using Pattern = pulsim::xdev::adapter::Pattern<R>;

// The saturable inductor: pins {from, to}, internal row = the branch
// current i; f = {i, −i, v_from − v_to}, q = {0, 0, −λ(i)},
// λ(i) = L_res·i + (L_0 − L_res)·I_sat·atan(i / I_sat).
struct SatIndTwin {
    static constexpr const char* kName = "satind_twin";
    static constexpr std::uint32_t n_pins = 2, n_internal = 1;
    static constexpr std::uint32_t kClaims =
        PULSIM_XDEV_CLAIM_SMOOTH | PULSIM_XDEV_CLAIM_RECIPROCAL;
    static constexpr std::array<const char*, 2> kPins{"from", "to"};
    static constexpr std::array<pulsim_xdev_field, 3> kFields{{
        {"L_0", "H", 1e-3, 1e-12, 1e3},
        {"I_sat", "A", 1.0, 1e-9, 1e6},
        {"L_residual", "H", 1e-4, 0.0, 1e3},
    }};
    static constexpr Pattern<3> kF{{{false, false, true}, {false, false, true}, {true, true, false}}};
    static constexpr Pattern<3> kQ{{{false, false, false}, {false, false, false}, {false, false, true}}};
    static constexpr std::array<double, 3> kProbeLo{-10, -10, -20}, kProbeHi{10, 10, 20};
    static constexpr std::array<pulsim_xdev_monotone, 1> kMonotone{{{0, 2}}};  // i rises with i
    template <class S>
    static void eval(const double* p, const std::array<S, 3>& x, std::array<S, 3>& f,
                     std::array<S, 3>& q) {
        using std::atan;
        const double L0 = p[0], Is = p[1], Lr = p[2];
        f[0] = x[2];
        f[1] = -x[2];
        f[2] = x[0] - x[1];
        q[2] = -(Lr * x[2] + (L0 - Lr) * Is * atan(x[2] / Is));
    }
    static int validate(const double* p, char* msg, std::uint32_t len) {
        if (p[2] > p[0]) {
            std::snprintf(msg, len, "L_residual (%g H) above L_0 (%g H)", p[2], p[0]);
            return 1;
        }
        return 0;
    }
};

// The Shockley diode of models/shockley_diode.hpp, same operation order:
// exponential continued by its tangent above v_lim, optional breakdown, G_min.
struct ShockleyTwin {
    static constexpr const char* kName = "shockley_twin";
    static constexpr std::uint32_t n_pins = 2, n_internal = 0;
    static constexpr std::uint32_t kClaims =
        PULSIM_XDEV_CLAIM_SMOOTH | PULSIM_XDEV_CLAIM_PASSIVE;
    static constexpr std::array<const char*, 2> kPins{"anode", "cathode"};
    static constexpr std::array<pulsim_xdev_field, 6> kFields{{
        {"I_S", "A", 1e-12, 1e-30, 1.0},
        {"n", "", 1.0, 0.1, 10.0},
        {"V_T", "V", 0.025852, 1e-3, 1.0},
        {"G_min", "S", 1e-12, 0.0, 1.0},
        {"i_lim", "A", 1e6, 1e-6, 1e12},
        {"BV", "V", 0.0, 0.0, 1e5},
    }};
    static constexpr Pattern<2> kF{{{true, true}, {true, true}}};
    static constexpr Pattern<2> kQ{};
    static constexpr std::array<double, 2> kProbeLo{-1, -1}, kProbeHi{1, 1};
    static constexpr std::array<pulsim_xdev_monotone, 1> kMonotone{{{0, 0}}};

    template <class S>
    static S branch(const S& u, const double* p) {
        using std::exp;
        using std::log;
        const double vte = p[1] * p[2];
        const double vl = vte * log(p[4] / p[0] + 1.0);
        if (u > S{vl}) {
            const double e_l = exp(vl / vte);
            const double i_l = p[0] * (e_l - 1.0);
            const double g_l = p[0] * e_l / vte;
            return S{i_l} + S{g_l} * (u - S{vl});
        }
        return p[0] * (exp(u / S{vte}) - S{1});
    }
    template <class S>
    static void eval(const double* p, const std::array<S, 2>& x, std::array<S, 2>& f,
                     std::array<S, 2>& q) {
        const S vd = x[0] - x[1];
        S i = branch(vd, p);
        if (p[5] > 0.0) {
            const S over = S{-p[5]} - vd;
            i = i - branch(over, p);
        }
        i = i + vd * p[3];
        f[0] = i;
        f[1] = -i;
        q[0] = S{0};
        q[1] = S{0};
    }
};

// The same saturable inductor with a HAND Jacobian (the adapter's eval_jac
// path): λ and dλ/di from one tangent, as the built-in law's eval_jac.
// df in pattern order (0,2) (1,2) (2,0) (2,1); dq (2,2).
struct SatIndHand : SatIndTwin {
    static constexpr const char* kName = "satind_hand";
    static void eval_jac(const double* p, const double* x, double* f, double* q, double* df,
                         double* dq) {
        using AD1 = pulsim::ad::ADRealN<1>;
        using std::atan;
        const double L0 = p[0], Is = p[1], Lr = p[2];
        const double i = x[2];
        f[0] = i;
        f[1] = -i;
        f[2] = x[0] - x[1];
        const AD1 u{i, {1.0}};
        const AD1 lam = Lr * u + (L0 - Lr) * Is * atan(u / Is);
        q[0] = 0;
        q[1] = 0;
        q[2] = -lam.value();
        df[0] = 1;
        df[1] = -1;
        df[2] = 1;
        df[3] = -1;
        dq[0] = -lam.deriv(0);
    }
};
}  // namespace

PULSIM_XDEV_MODELS(SatIndTwin, ShockleyTwin, SatIndHand)
