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

#include "pulsim/models/flux_map_dq.hpp"
#include "pulsim/numeric/types.hpp"
#include "pulsim/topology/graph.hpp"

#include <array>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <vector>

namespace pulsim::models {

struct PmsmMna {
    /// One term of the PM-flux series λ_pm,a(θ_e) = Σ ψ_h·cos(h·θ_e + φ_h)
    /// (add-mna-native-machines group 3; phases b, c at θ_e ∓ 2π/3).
    struct PmHarmonic {
        Real h;     //!< harmonic order (positive integer)
        Real psi;   //!< amplitude [Wb]
        Real phi;   //!< phase [rad]
    };
    using Harmonics = std::shared_ptr<const std::vector<PmHarmonic>>;

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
        /// Non-sinusoidal PM flux; null = ψ_pm·cos(θ_e) (psi_pm is then
        /// the only PM parameter read).
        Harmonics harmonics{};
        /// Saturating, cross-coupled dq law λ_dq = ∇W'(i_d, i_q) (group 4);
        /// null = the constant L_d, L_q, ψ_pm above. With a map, L_d, L_q
        /// and psi_pm are the map's values at the origin (the builder
        /// checks it) and are read only for L_0's default and the
        /// held-mechanics DC point at ω0 = 0.
        std::shared_ptr<const FluxMapDQ> flux_map{};
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
        if (!(p.psi_pm >= Real{0}) && !p.flux_map) {   // a map carries its own
            throw std::invalid_argument(
                "PmsmMna: psi_pm must be >= 0");
        }
        if (!(p.pole_pairs >= Real{1}) ||
            p.pole_pairs != std::floor(p.pole_pairs)) {
            throw std::invalid_argument(
                "PmsmMna: pole_pairs must be a positive integer");
        }
        if (p.harmonics) {
            if (p.harmonics->empty()) {
                throw std::invalid_argument(
                    "PmsmMna: harmonics, when given, must not be empty");
            }
            for (const auto& t : *p.harmonics) {
                if (!(t.h >= Real{1}) || t.h != std::floor(t.h) ||
                    !std::isfinite(t.psi) || !std::isfinite(t.phi)) {
                    throw std::invalid_argument(
                        "PmsmMna: each harmonic needs an integer order "
                        "h >= 1 and finite psi, phi");
                }
            }
        }
        if (p.flux_map) {
            if (p.harmonics) {
                throw std::invalid_argument(
                    "PmsmMna: a flux map carries the PM flux in lambda_d; "
                    "it cannot be combined with PM harmonics");
            }
            p.flux_map->validate();
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
        if (p.harmonics) {
            // d/dθ of ψ·cos(h·a + φ): −h·ψ·sin(...), then −h²·ψ·cos(...).
            // h = 1, φ = 0 evaluates the same expressions as below.
            for (Size k = 0; k < 3; ++k) {
                const Real a = theta_e + phase_offset(k);
                Real acc = 0;
                for (const auto& t : *p.harmonics) {
                    const Real arg = t.h * a + t.phi;
                    acc += deriv == 0 ? t.psi * std::cos(arg)
                         : deriv == 1 ? -(t.h * t.psi) * std::sin(arg)
                                      : -(t.h * t.h * t.psi) * std::cos(arg);
                }
                out[k] = acc;
            }
            return out;
        }
        for (Size k = 0; k < 3; ++k) {
            const Real a = theta_e + phase_offset(k);
            out[k] = deriv == 0 ? p.psi_pm * std::cos(a)
                   : deriv == 1 ? -p.psi_pm * std::sin(a)
                                : -p.psi_pm * std::cos(a);
        }
        return out;
    }

    /// λ(θ_e, i) − λ(θ0_e, i0) without cancellation. `dth_e` = θ_e − θ0_e
    /// and `di` = i − i0 are passed in (node-voltage differences, exact
    /// when the two are close). The flux row multiplies this by 2/h, so
    /// its round-off must scale with the increment, not with |λ|
    /// (eps·(2/h)·ψ_pm exceeds Newton's 1e-9 V at h = 10 ns, ψ = 0.05):
    ///   L(θ) − L(θ0): the dq block Pᵀ·diag(L_d, L_q)·P is
    ///     Σ·I + Δ·[[cos 2θ, sin 2θ], [sin 2θ, −cos 2θ]],  Δ = (L_d − L_q)/2,
    ///   and cos 2θ − cos 2θ0 = −2 sin(θ + θ0) sin(θ − θ0),
    ///       sin 2θ − sin 2θ0 = +2 cos(θ + θ0) sin(θ − θ0);
    ///   λ_pm: cos(h·a + φ) − cos(h·a0 + φ) = −2 sin(h(a + a0)/2 + φ)·sin(h(a − a0)/2).
    /// `L` must be inductance(p, th_e).L (passed in: the stamp has it).
    [[nodiscard]] static Vec3 lambda_increment(const Params& p, const Mat3& L,
                                               Real th_e, Real th0_e,
                                               Real dth_e, const Vec3& i0,
                                               const Vec3& di) noexcept {
        Vec3 out{};
        for (Size k = 0; k < 3; ++k) {
            Real acc = 0;
            for (Size j = 0; j < 3; ++j) acc += L[k][j] * di[j];
            out[k] = acc;
        }
        // [L(θ) − L(θ0)]·i0 = C⁻¹·ΔQ·(C·i0); only the dq block moves.
        const Real half_diff = Real{0.5} * (p.L_d - p.L_q);
        if (half_diff != Real{0}) {
            const Mat3 C = clarke(), Ci = clarke_inv();
            const Real a0 = C[0][0] * i0[0] + C[0][1] * i0[1] + C[0][2] * i0[2];
            const Real b0 = C[1][0] * i0[0] + C[1][1] * i0[1] + C[1][2] * i0[2];
            const Real s1 = std::sin(dth_e), sum = th_e + th0_e;
            const Real dc = Real{-2} * std::sin(sum) * s1;
            const Real ds = Real{2} * std::cos(sum) * s1;
            const Real qa = half_diff * (dc * a0 + ds * b0);
            const Real qb = half_diff * (ds * a0 - dc * b0);
            for (Size k = 0; k < 3; ++k) out[k] += Ci[k][0] * qa + Ci[k][1] * qb;
        }
        auto pm = [&](Real h, Real psi, Real phi) {
            const Real sh = std::sin(h * dth_e / Real{2});
            for (Size k = 0; k < 3; ++k) {
                const Real mid = h * (sum_offset(th_e, th0_e, k)) / Real{2} + phi;
                out[k] += Real{-2} * psi * std::sin(mid) * sh;
            }
        };
        if (p.harmonics) {
            for (const auto& t : *p.harmonics) pm(t.h, t.psi, t.phi);
        } else {
            pm(Real{1}, p.psi_pm, Real{0});
        }
        return out;
    }
    /// a + a0 for phase k (each angle carries the phase offset).
    [[nodiscard]] static Real sum_offset(Real th_e, Real th0_e, Size k) noexcept {
        return th_e + th0_e + Real{2} * phase_offset(k);
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
    // ---------------------------------------------------------------
    // Flux-map law (group 4). λ_abc = T⁻¹(θ_e)·[∇W'(i_d, i_q), L_0·i_0],
    // i_dq0 = T(θ_e)·i, T = P(θ_e)·C. With Hb = blockdiag(∇²W', L_0):
    //   ∂λ/∂i   = T⁻¹·Hb·T
    //   ∂λ/∂θ_e = C⁻¹·P'ᵀ·λ_dq0 + T⁻¹·Hb·(P'·C·i),  P'·C·i = (i_q, −i_d, 0)
    // Torque = ∂/∂θ_m of the abc co-energy (3/2)·W'(T·i):
    //   T_em = (3/2)·pp·(λ_d·i_q − λ_q·i_d)
    // (the constant-L machine's pp·[λ_pm'ᵀi + ½ iᵀL'i] for the linear map).
    struct FluxMapState {
        Vec3 lam{};          //!< λ_abc
        Mat3 dlam_di{};      //!< ∂λ_abc/∂i_abc
        Vec3 dlam_dth{};     //!< ∂λ_abc/∂θ_e
        Real T = 0;          //!< T_em [N·m]
        Vec3 dT_di{};
        Real dT_dtheta_m = 0;
        Real i_d = 0, i_q = 0, lam_d = 0, lam_q = 0;
    };
    [[nodiscard]] static FluxMapState flux_map_state(const Params& p, Real theta_e,
                                                     const Vec3& i) {
        const auto& fm = *p.flux_map;
        const Real L0 = L_zero(p);
        const Mat3 C = clarke(), Ci = clarke_inv();
        const Mat3 P0 = park(theta_e, 0), P1 = park(theta_e, 1);
        const Mat3 T = mul(P0, C), Ti = mul(Ci, transpose(P0));
        const Mat3 Ci_P1t = mul(Ci, transpose(P1));
        Vec3 x{};
        for (Size r = 0; r < 3; ++r)
            for (Size j = 0; j < 3; ++j) x[r] += T[r][j] * i[j];
        const auto e = fm.eval(x[0], x[1]);
        const Vec3 ldq{e.gd, e.gq, L0 * x[2]};
        const Mat3 Hb{{{e.hdd, e.hdq, 0}, {e.hdq, e.hqq, 0}, {0, 0, L0}}};
        const Vec3 rot{x[1], -x[0], Real{0}};
        Vec3 Hrot{};
        for (Size r = 0; r < 3; ++r)
            for (Size s = 0; s < 3; ++s) Hrot[r] += Hb[r][s] * rot[s];
        FluxMapState out;
        out.dlam_di = mul(mul(Ti, Hb), T);
        for (Size k = 0; k < 3; ++k)
            for (Size r = 0; r < 3; ++r) {
                out.lam[k] += Ti[k][r] * ldq[r];
                out.dlam_dth[k] += Ci_P1t[k][r] * ldq[r] + Ti[k][r] * Hrot[r];
            }
        const Real pp = p.pole_pairs, k15 = Real{1.5} * pp;
        out.T = k15 * (e.gd * x[1] - e.gq * x[0]);
        const Real td = k15 * (e.hdd * x[1] - e.hdq * x[0] - e.gq);
        const Real tq = k15 * (e.hdq * x[1] + e.gd - e.hqq * x[0]);
        for (Size j = 0; j < 3; ++j) out.dT_di[j] = td * T[0][j] + tq * T[1][j];
        out.dT_dtheta_m = pp * (td * x[1] - tq * x[0]);
        out.i_d = x[0]; out.i_q = x[1]; out.lam_d = e.gd; out.lam_q = e.gq;
        return out;
    }

    /// λ(θ_e, i) − λ(θ0_e, i0) for the flux map, in increment form (round-off
    /// ∝ the increments, as `lambda_increment`):
    ///   = C⁻¹·(P(θ) − P(θ0))ᵀ·λ_dq0(x0) + T⁻¹(θ)·(λ_dq0(x) − λ_dq0(x0)),
    ///   x − x0 = P(θ)·C·Δi + (P(θ) − P(θ0))·C·i0,
    /// with P(θ) − P(θ0) from the half-angle identities and the gradient
    /// difference from `FluxMapDQ::grad_increment`.
    [[nodiscard]] static Vec3 flux_map_increment(const Params& p, Real th_e,
                                                 Real th0_e, Real dth_e,
                                                 const Vec3& i0, const Vec3& di) {
        const auto& fm = *p.flux_map;
        const Real L0 = L_zero(p);
        const Mat3 C = clarke(), Ci = clarke_inv();
        const Mat3 P = park(th_e, 0), P0 = park(th0_e, 0);
        const Real sh = std::sin(Real{0.5} * dth_e), sg = Real{0.5} * (th_e + th0_e);
        const Real dc = Real{-2} * std::sin(sg) * sh, ds = Real{2} * std::cos(sg) * sh;
        const Mat3 dP{{{dc, ds, 0}, {-ds, dc, 0}, {0, 0, 0}}};
        Vec3 ci0{}, cdi{};
        for (Size r = 0; r < 3; ++r)
            for (Size j = 0; j < 3; ++j) {
                ci0[r] += C[r][j] * i0[j];
                cdi[r] += C[r][j] * di[j];
            }
        Vec3 x0{}, dx{};
        for (Size r = 0; r < 3; ++r)
            for (Size s = 0; s < 3; ++s) {
                x0[r] += P0[r][s] * ci0[s];
                dx[r] += P[r][s] * cdi[s] + dP[r][s] * ci0[s];
            }
        const auto e0 = fm.eval(x0[0], x0[1]);
        const auto dg = fm.grad_increment(x0[0], x0[1], dx[0], dx[1]);
        const Vec3 l0{e0.gd, e0.gq, L0 * x0[2]};
        const Vec3 dl{dg[0], dg[1], L0 * dx[2]};
        const Mat3 Ti = mul(Ci, transpose(P));
        Vec3 out{};
        for (Size k = 0; k < 3; ++k)
            for (Size r = 0; r < 3; ++r) {
                Real dPt_l0 = 0;   // (dPᵀ·λ0)_r
                for (Size s = 0; s < 3; ++s) dPt_l0 += dP[s][r] * l0[s];
                out[k] += Ci[k][r] * dPt_l0 + Ti[k][r] * dl[r];
            }
        return out;
    }
};

}  // namespace pulsim::models
