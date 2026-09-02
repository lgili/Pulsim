#pragma once

// =============================================================================
// Pulsim — MNA-native PMSM: implicit, flux-state, full L(θ)
// =============================================================================
//
// v2.0 Phase 4, audit C.3 ("Máquinas nativas no MNA", alto,
// breaking). The Python PMSM/BLDC/IM are OBSERVERS: the back-EMF is
// computed from the previous step's (θ, ω) and injected through
// `b_extra_fn`, the mechanics are forward-Euler, and the stator is
// three ordinary inductors carrying ONE average inductance. Three
// consequences, each measured before this file was written:
//
//   * one-step-lag coupling is first order in dt — on an open-loop
//     PMSM the settled speed drifts −0.37 % at the tests' own
//     dt = 5e-5, −2 % at 2e-4, −8 % at 5e-4;
//   * the average L erases saliency from the electrical dynamics:
//     with Ld = 1 mH, Lq = 3 mH a locked-rotor step gives τ_d = 4 ms
//     instead of 2 (+100 %) and τ_q = 4 ms instead of 6 (−33 %), so
//     an IPM's phase currents are simply wrong, and there is no
//     L(θ) harmonic content and no HFI at all;
//   * nothing that needs the map to be exactly linear in the MNA
//     state — `steady_state`, `sampled_ac`, the TR-BDF2 router —
//     can see a machine that lives in a Python closure.
//
// THIS MODEL puts the machine IN the matrix.
//
//   Electrical, abc frame, flux linkage as the state:
//
//       λ_abc = L(θ_e) · i_abc + λ_pm(θ_e)
//       v_k   = R_s · i_k + dλ_k/dt            k = a, b, c
//
//   with the full θ-dependent inductance matrix
//
//       L(θ_e) = T(θ_e)⁻¹ · diag(L_d, L_q, L_0) · T(θ_e)
//
//   built from the amplitude-invariant Clarke–Park transform T, so
//   the dq inductances the datasheet gives are reproduced EXACTLY
//   and every L(θ) harmonic an IPM has is present. Derivatives in θ
//   come from dP/dθ and d²P/dθ² of the rotation block — closed
//   form, no finite differences, and no hand-derived cos(2θ) table
//   to get a sign wrong in.
//
//   Each phase is a branch with a current unknown (it piggy-backs
//   on the inductor numbering, exactly like the saturable
//   inductor), and its trapezoidal constraint row is stamped per
//   Newton iteration with λ evaluated at the CURRENT iterate — the
//   θ transformation is stamped per iteration, as the audit asks,
//   so there is no lag anywhere.
//
//   Mechanics as NODES. ω and θ are node voltages of a small
//   mechanical circuit the builder adds alongside the stator:
//
//       J · dω/dt = T_em − T_load − B·ω      capacitor J on the ω node
//       dθ/dt     = ω                         capacitor 1 F on the θ node
//
//   The capacitors are ordinary linear ones, so the mechanics get
//   the same trapezoidal companion as everything else — "mecânica
//   trapezoidal no mesmo solve" — and only the two couplings are
//   nonlinear: T_em(i_abc, θ) injected into the ω node, and ω
//   injected into the θ node. This is the same trick the thermal
//   network uses to live in the MNA, and it means a speed-dependent
//   load is just another element on the ω node.
//
//   Torque from co-energy, which is what makes the model
//   self-consistent with its own inductance matrix:
//
//       T_em = pp · [ (dλ_pm/dθ_e)ᵀ i  +  ½ iᵀ (dL/dθ_e) i ]
//
//   The second term IS the reluctance torque; it appears because
//   L(θ) is in the matrix, not because it was added on.
//
// CONVENTIONS are the FOC chain's, matched to the existing observer
// so the two can be compared in tests: d-axis on the PM flux,
// q-axis 90° ahead in the direction of rotation, phase offsets
// (0, −120°, +120°) for (a, b, c), λ_pm,a = ψ_pm cos θ_e so that
// e_a = −ψ_pm·pp·ω·sin θ_e.

#include "pulsim/numeric/types.hpp"
#include "pulsim/topology/graph.hpp"

#include <array>
#include <cmath>
#include <stdexcept>

namespace pulsim::models {

struct PmsmMna {
    struct Params {
        Real R_s        = Real{0.5};    //!< stator resistance [Ω]
        Real L_d        = Real{2e-3};   //!< d-axis inductance [H]
        Real L_q        = Real{2e-3};   //!< q-axis inductance [H]
        /// Zero-sequence inductance [H]. With a star point the
        /// zero-sequence current is fixed by KCL, so this only sets
        /// the conditioning of the neutral row; the default is the
        /// dq average, which is always well-scaled.
        Real L_0        = Real{0};
        Real psi_pm     = Real{0.05};   //!< PM flux linkage [Wb]
        Real pole_pairs = Real{4};      //!< pp
        Real T_load     = Real{0};      //!< constant load torque [N·m]
    };

    static constexpr topology::BranchKind kind =
        topology::BranchKind::Nonlinear;
    static constexpr Size num_terminals = 2;
    static constexpr bool is_linear = false;

    static void validate(const Params& p) {
        if (!(p.R_s >= Real{0})) {
            throw std::invalid_argument("PmsmMna: R_s must be >= 0");
        }
        if (!(p.L_d > Real{0}) || !(p.L_q > Real{0})) {
            throw std::invalid_argument(
                "PmsmMna: L_d and L_q must be positive");
        }
        if (p.L_0 < Real{0}) {
            throw std::invalid_argument(
                "PmsmMna: L_0 must be >= 0 (0 selects the dq average)");
        }
        if (!(p.psi_pm >= Real{0})) {
            throw std::invalid_argument(
                "PmsmMna: psi_pm must be >= 0");
        }
        if (!(p.pole_pairs >= Real{1}) ||
            p.pole_pairs != std::floor(p.pole_pairs)) {
            throw std::invalid_argument(
                "PmsmMna: pole_pairs must be a positive integer");
        }
    }

    [[nodiscard]] static Real L_zero(const Params& p) noexcept {
        return p.L_0 > Real{0} ? p.L_0
                               : Real{0.5} * (p.L_d + p.L_q);
    }

    using Mat3 = std::array<std::array<Real, 3>, 3>;
    using Vec3 = std::array<Real, 3>;

    /// Phase offsets (a, b, c) = (0, −120°, +120°).
    [[nodiscard]] static constexpr Real phase_offset(Size k) noexcept {
        constexpr Real two_pi_3 = Real{2.0943951023931953};
        return k == 0 ? Real{0} : (k == 1 ? -two_pi_3 : two_pi_3);
    }

    /// Amplitude-invariant Clarke, row-major, and its inverse.
    [[nodiscard]] static Mat3 clarke() noexcept {
        const Real s = Real{0.8660254037844386};   // √3/2
        const Real c = Real{2} / Real{3};
        return {{{c, -c / 2, -c / 2},
                 {Real{0}, c * s, -c * s},
                 {c / 2, c / 2, c / 2}}};
    }
    [[nodiscard]] static Mat3 clarke_inv() noexcept {
        const Real s = Real{0.8660254037844386};
        return {{{Real{1}, Real{0}, Real{1}},
                 {-Real{0.5}, s, Real{1}},
                 {-Real{0.5}, -s, Real{1}}}};
    }

    /// Park rotation block and its first two θ-derivatives.
    ///   P(θ) = [[cos, sin, 0], [−sin, cos, 0], [0, 0, 1]]
    [[nodiscard]] static Mat3 park(Real th, int deriv) noexcept {
        const Real c = std::cos(th), s = std::sin(th);
        switch (deriv) {
            case 0:  return {{{c, s, 0}, {-s, c, 0}, {0, 0, 1}}};
            case 1:  return {{{-s, c, 0}, {-c, -s, 0}, {0, 0, 0}}};
            default: return {{{-c, -s, 0}, {s, -c, 0}, {0, 0, 0}}};
        }
    }

    [[nodiscard]] static Mat3 mul(const Mat3& A, const Mat3& B) noexcept {
        Mat3 C{};
        for (Size i = 0; i < 3; ++i)
            for (Size j = 0; j < 3; ++j) {
                Real acc = 0;
                for (Size k = 0; k < 3; ++k) acc += A[i][k] * B[k][j];
                C[i][j] = acc;
            }
        return C;
    }
    [[nodiscard]] static Mat3 transpose(const Mat3& A) noexcept {
        Mat3 T{};
        for (Size i = 0; i < 3; ++i)
            for (Size j = 0; j < 3; ++j) T[i][j] = A[j][i];
        return T;
    }
    [[nodiscard]] static Mat3 diag_scale(const Mat3& A, const Vec3& d)
        noexcept {
        Mat3 C = A;
        for (Size i = 0; i < 3; ++i)
            for (Size j = 0; j < 3; ++j) C[i][j] *= d[j];
        return C;   // A · diag(d)
    }

    /// L(θ_e), dL/dθ_e, d²L/dθ_e² — all from
    ///   L = C⁻¹ · Pᵀ(θ) · D · P(θ) · C
    /// (with T = P·C and, for the amplitude-invariant Clarke used
    /// here, T⁻¹ = C⁻¹·Pᵀ since P is a rotation).
    struct Inductance {
        Mat3 L, dL, d2L;
    };
    [[nodiscard]] static Inductance inductance(const Params& p,
                                               Real theta_e) noexcept {
        const Vec3 d{p.L_d, p.L_q, L_zero(p)};
        const Mat3 C = clarke(), Ci = clarke_inv();
        const Mat3 P0 = park(theta_e, 0), P1 = park(theta_e, 1),
                   P2 = park(theta_e, 2);
        // Q(θ) = Pᵀ D P ;  Q' = P1ᵀ D P0 + P0ᵀ D P1 ;
        // Q'' = P2ᵀ D P0 + 2 P1ᵀ D P1 + P0ᵀ D P2
        auto Q = [&](const Mat3& A, const Mat3& B) {
            return mul(diag_scale(transpose(A), d), B);
        };
        const Mat3 Q0 = Q(P0, P0);
        Mat3 Q1 = Q(P1, P0), Q1b = Q(P0, P1);
        Mat3 Q2 = Q(P2, P0), Q2b = Q(P1, P1), Q2c = Q(P0, P2);
        for (Size i = 0; i < 3; ++i)
            for (Size j = 0; j < 3; ++j) {
                Q1[i][j] += Q1b[i][j];
                Q2[i][j] += Real{2} * Q2b[i][j] + Q2c[i][j];
            }
        Inductance out;
        out.L   = mul(mul(Ci, Q0), C);
        out.dL  = mul(mul(Ci, Q1), C);
        out.d2L = mul(mul(Ci, Q2), C);
        return out;
    }

    /// PM flux linkage per phase and its θ_e-derivatives.
    [[nodiscard]] static Vec3 lambda_pm(const Params& p, Real theta_e,
                                        int deriv) noexcept {
        Vec3 out{};
        for (Size k = 0; k < 3; ++k) {
            const Real a = theta_e + phase_offset(k);
            out[k] = deriv == 0 ? p.psi_pm * std::cos(a)
                   : deriv == 1 ? -p.psi_pm * std::sin(a)
                                : -p.psi_pm * std::cos(a);
        }
        return out;
    }

    /// Electromagnetic torque from co-energy, plus its partials.
    ///   T = pp·[ λ_pm'ᵀ i + ½ iᵀ L' i ]
    ///   ∂T/∂i_j   = pp·[ λ_pm'_j + (L' i)_j ]
    ///   ∂T/∂θ_m   = pp²·[ λ_pm''ᵀ i + ½ iᵀ L'' i ]
    struct Torque {
        Real T;
        Vec3 dT_di;
        Real dT_dtheta_m;
    };
    [[nodiscard]] static Torque torque(const Params& p, Real theta_e,
                                       const Vec3& i) noexcept {
        const auto ind = inductance(p, theta_e);
        const Vec3 lp1 = lambda_pm(p, theta_e, 1);
        const Vec3 lp2 = lambda_pm(p, theta_e, 2);
        Torque t{};
        Real quad1 = 0, quad2 = 0, lin1 = 0, lin2 = 0;
        for (Size a = 0; a < 3; ++a) {
            Real row1 = 0, row2 = 0;
            for (Size b = 0; b < 3; ++b) {
                row1 += ind.dL[a][b] * i[b];
                row2 += ind.d2L[a][b] * i[b];
            }
            quad1 += i[a] * row1;
            quad2 += i[a] * row2;
            lin1 += lp1[a] * i[a];
            lin2 += lp2[a] * i[a];
            t.dT_di[a] = p.pole_pairs * (lp1[a] + row1);
        }
        t.T = p.pole_pairs * (lin1 + Real{0.5} * quad1);
        t.dT_dtheta_m = p.pole_pairs * p.pole_pairs
                        * (lin2 + Real{0.5} * quad2);
        return t;
    }
};

}  // namespace pulsim::models
