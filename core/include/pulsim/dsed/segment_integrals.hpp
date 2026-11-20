#pragma once

// =============================================================================
// Pulsim — the exact engine's segment integrals for recorders (release finding
// F-B2-1, openspec/changes/update-number-changing-release/fixes/
// exact-buck-integrals.md)
// =============================================================================
//
// Between two events the exact engine's state is its mode's closed form. In
// the augmented energy coordinates of period_integrals.hpp (x̄ = (x, 1),
// x̃ = d∘x̄, As = d·[[A, b], [0, 0]]·d⁻¹ = V·diag(μ)·V⁻¹, z = V⁻¹ x̃0) a probe
// that is a polynomial of degree ≤ 2 in its readings, each reading affine in
// the state (v(), i() of a state column, a resistor, a PWL switch or diode
// under the step's mask, a constant current source) or the product of two
// (p()), is a quadratic form of x̄:
//
//   degree ≤ 1   y(s) = l·x̄(s) = Σ_i α_i e^{μ_i s},           α = r ∘ ζ,  r = Vᵀ(l/d)
//   degree 2     y(s) = x̄ᵀQx̄ = Σ_ij β_ij e^{(μ_i+μ_j) s},     β_ij = B_ij ζ_i ζ_j,  B = Vᵀ(D⁻¹QD⁻¹)V
//
// with ζ = z ∘ e^{μ s0} the modal coordinates at the piece's start s0. Over
// [s0, s0 + L], with the ONE kernel of period_integrals.hpp
// (detail::phi1_kernel / phi1_pair_kernel: ∫₀ᴸ e^{νs} ds = L·φ₁(νL)):
//
//   ∫y  = Σ_i α_i Lφ₁(μ_i L)                   ∫y² = Σ_ij α_i α_j Lφ₁((μ_i+μ_j)L)
//   ∫y  = Σ_ij β_ij Lφ₁((μ_i+μ_j)L)            ∫y² = Σ_pq γ_p γ_q Lφ₁((σ_p+σ_q)L)
//
// (degree 2: p, q run over the pairs i ≤ j, σ_p = μ_i + μ_j, γ_p = β_ii or
// 2β_ij). No quadrature and no output grid: the integrals are the
// trajectory's. A zero eigenvalue (the augmented constant, an integrator)
// gives φ₁(0) = 1 exactly; φ₁ keeps relative error near u at every argument
// (expm1 form), so |μL| = 1e-12 loses nothing to cancellation.
//
// Conditioning. A modal sum over products of p modal expansions carries
// u·κ₂(Ṽ)^p·Σ|terms| (measured on a near-defective pair, δ/a = 1e-3: ∫y
// 6e-13, ∫y² 8e-10, a degree-2 ∫y² 6e-6): p = 1 for the state, 2 for a
// degree-1 ∫y² and a degree-2 ∫y, 4 for a degree-2 ∫y². Each is taken modal
// only under the propagator's bar, u·κ^p ≤ 1e-10. Otherwise (a near-
// defective pair: critical damping, a degenerate cluster of paralleled
// banks) the matrix form, with no eigenvectors:
//
//   Φ = e^{As L},  I = ∫₀ᴸ Φ ds,  W_Q = ∫₀ᴸ Φᵀ Q Φ ds
//   (Taylor at h = L/2^k with ‖As‖₁ h ≤ 1/4, then k doublings
//    Φ(2h) = Φ(h)², I(2h) = I(h) + Φ(h)I(h), W(2h) = W(h) + Φ(h)ᵀW(h)Φ(h))
//
//   degree ≤ 1: ∫y = l̃·I x̃,  ∫y² = x̃ᵀ W_{l̃l̃ᵀ} x̃;   degree 2: ∫y = x̃ᵀ W_Q̃ x̃
//
// where every term is a positive-time propagation (no e^{−As t}): stiff modes
// cannot overflow, and the error is a few u·k of the terms. A degree-2
// probe's ∫y² there (u·κ⁴ > 1e-10) is the same Gramian one level up: w =
// x̃⊗x̃ obeys w' = (As⊗I + I⊗As)w and y = vec(Q̃)·w, so ∫y² = w0ᵀW₄w0 with
// W₄ the Gramian of vec(Q̃)vec(Q̃)ᵀ, dimension (n + 1)², formed for
// n + 1 ≤ 12; past that the piece's ∫y² is the recorder's line (its ∫y
// stays closed), counted.
//
// Every step is CHECKED: its closed form's end must be the engine's end state
// (1e-9 in the energy norm of x̃), so a step that is not its mode's closed
// form (a numeric step, a corrected state) is never integrated as one. (Not
// per probe against the recorder's samples: a reading like v(in, sw) = 12 −
// 11.999999999 carries the sample's own cancellation, 1e-6 relative.) A piece with no closed form (that check; a probe that is
// not a polynomial of degree ≤ 2 of such readings; time-varying sources,
// whose input is not the mode's constant b) is the recorder's line and is
// COUNTED (`n_line`, first reason kept): simulate() warns by name.
// A pure observer: nothing here moves a state; with no recorder nothing is built.

#include "pulsim/dsed/period_integrals.hpp"
#include "pulsim/ir/expr.hpp"
#include "pulsim/pwl/pwl_branch_law.hpp"
#include "pulsim/record/loop_feed.hpp"
#include "pulsim/record/recorder.hpp"

#include <array>
#include <cmath>
#include <complex>
#include <limits>
#include <map>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace pulsim::dsed {

/// A probe as a polynomial of degree ≤ 2 in x̄ = (x, 1).
struct SegForm {
    int deg = -1;     ///< 0, 1, 2; −1: not such a polynomial
    Vector l;         ///< deg ≤ 1: y = l·x̄ (a constant is l[n])
    DenseMatrix Q;    ///< deg 2: y = x̄ᵀQx̄, symmetric (linear terms folded in through x̄_n = 1)
};

namespace seg {

using Cplx = std::complex<Real>;

[[nodiscard]] inline SegForm constant(Index n1, Real c) {
    SegForm f;
    f.deg = 0;
    f.l = Vector::Zero(n1);
    f.l[n1 - 1] = c;
    return f;
}
[[nodiscard]] inline SegForm linear(Vector l) {
    SegForm f;
    f.deg = 1;
    f.l = std::move(l);
    return f;
}
[[nodiscard]] inline DenseMatrix quadratic_of(const SegForm& f) {
    if (f.deg == 2) return f.Q;
    const Index n1 = f.l.size();
    DenseMatrix Q = DenseMatrix::Zero(n1, n1);
    Q.col(n1 - 1) += Real{0.5} * f.l;
    Q.row(n1 - 1) += Real{0.5} * f.l.transpose();
    return Q;
}
[[nodiscard]] inline SegForm scaled(SegForm f, Real k) {
    if (f.deg == 2) f.Q *= k;
    else if (f.deg >= 0) f.l *= k;
    return f;
}
/// a + k·b
[[nodiscard]] inline SegForm sum(const SegForm& a, const SegForm& b, Real k) {
    if (a.deg < 0 || b.deg < 0) return {};
    SegForm f;
    f.deg = std::max(a.deg, b.deg);
    if (f.deg <= 1) {
        f.l = a.l + k * b.l;
        f.deg = std::max(a.deg, b.deg);
    } else {
        f.Q = quadratic_of(a) + k * quadratic_of(b);
    }
    return f;
}
[[nodiscard]] inline SegForm product(const SegForm& a, const SegForm& b) {
    if (a.deg < 0 || b.deg < 0) return {};
    if (a.deg == 0) return scaled(b, a.l[a.l.size() - 1]);
    if (b.deg == 0) return scaled(a, b.l[b.l.size() - 1]);
    if (a.deg == 1 && b.deg == 1) {
        SegForm f;
        f.deg = 2;
        f.Q = Real{0.5} * (a.l * b.l.transpose() + b.l * a.l.transpose());
        return f;
    }
    return {};
}

/// The probe's program (ir::Expr, Param k = atom k) over its atoms' forms:
/// constants, +, −, ×, ÷ by a constant, ^0 / ^1 / ^2; anything else (abs,
/// min, a quotient of states …) is not a polynomial of degree ≤ 2 (deg −1).
[[nodiscard]] inline SegForm compose(std::span<const ir::Expr::Instr> prog,
                                     std::span<const SegForm> atoms, Index n1) {
    using Op = ir::Expr::Op;
    std::vector<SegForm> st;
    st.reserve(prog.size());
    for (const auto& in : prog) {
        switch (in.op) {
        case Op::Const: st.push_back(constant(n1, in.value)); break;
        case Op::Param:
            st.push_back(in.index < atoms.size() ? atoms[in.index] : SegForm{});
            break;
        case Op::Neg:
            if (st.empty()) return {};
            st.back() = scaled(std::move(st.back()), Real{-1});
            break;
        case Op::Add:
        case Op::Sub:
        case Op::Mul:
        case Op::Div:
        case Op::Pow:
        case Op::Min:
        case Op::Max:
        case Op::Atan2: {
            if (st.size() < 2) return {};
            const SegForm b = std::move(st.back());
            st.pop_back();
            SegForm& a = st.back();
            const bool bc = b.deg == 0;
            const Real cb = bc ? b.l[n1 - 1] : Real{0};
            if (in.op == Op::Add) a = sum(a, b, Real{1});
            else if (in.op == Op::Sub) a = sum(a, b, Real{-1});
            else if (in.op == Op::Mul) a = product(a, b);
            else if (in.op == Op::Div) a = bc && cb != Real{0} ? scaled(std::move(a), Real{1} / cb) : SegForm{};
            else if (in.op == Op::Pow && bc && a.deg == 0) a = constant(n1, std::pow(a.l[n1 - 1], cb));
            else if (in.op == Op::Pow && bc && cb == Real{0} && a.deg >= 0) a = constant(n1, Real{1});
            else if (in.op == Op::Pow && bc && cb == Real{1}) {}
            else if (in.op == Op::Pow && bc && cb == Real{2}) a = product(a, a);
            else a = SegForm{};
            break;
        }
        default:   // a function of a state: no polynomial
            if (st.empty()) return {};
            st.back() = SegForm{};
        }
    }
    return st.size() == 1 ? st.back() : SegForm{};
}

/// Φ = e^{A L}, I = ∫₀ᴸ Φ and, per Q in qs, W = ∫₀ᴸ Φᵀ Q Φ (see the header):
/// a Taylor start at ‖A‖₁·L/2^k ≤ 1/4 and k doublings.
inline void gram_doubling(const DenseMatrix& A, Real L, std::span<const DenseMatrix* const> qs,
                          DenseMatrix& Phi, DenseMatrix& I, std::vector<DenseMatrix>& W) {
    const Index n = A.rows();
    const Real na = n > 0 ? A.cwiseAbs().colwise().sum().maxCoeff() : Real{0};
    int k = 0;
    if (na * L > Real{0.25}) k = static_cast<int>(std::ceil(std::log2(na * L / Real{0.25})));
    const Real h = std::ldexp(L, -k);
    const DenseMatrix Ah = A * h;
    // (1/4)^j / j! < 1e-30 at j = 24; the W series' operator is ≤ 2‖Ah‖ ≤ 1/2
    constexpr int kTerms = 24;
    DenseMatrix T = DenseMatrix::Identity(n, n);   // Ah^j / j!
    Phi = T;
    I = h * T;
    for (int j = 1; j <= kTerms; ++j) {
        T = (T * Ah) / static_cast<Real>(j);
        Phi += T;
        I += (h / static_cast<Real>(j + 1)) * T;
    }
    W.resize(qs.size());
    for (std::size_t q = 0; q < qs.size(); ++q) {
        DenseMatrix M = *qs[q];   // h^j L^j(Q) / j!, L(X) = AᵀX + XA
        W[q] = h * M;
        for (int j = 1; j <= kTerms; ++j) {
            M = (Ah.transpose() * M + M * Ah) / static_cast<Real>(j);
            W[q] += (h / static_cast<Real>(j + 1)) * M;
        }
    }
    for (int d = 0; d < k; ++d) {
        for (auto& w : W) w += Phi.transpose() * w * Phi;
        I += Phi * I;
        Phi = Phi * Phi;
    }
}

}  // namespace seg

/// One mode's closed forms of a set of probe forms (in x̄): the modal
/// coefficients, or the matrix form without a modal basis; the kernels of
/// the last two segment lengths are kept (the step's, and a clipped piece's).
class SegmentMode {
public:
    SegmentMode(detail::PeriodMode e, const std::vector<SegForm>& forms) : e_{std::move(e)} {
        const Index n1 = e_.n + 1;
        const Vector di = e_.d.cwiseInverse();
        deg_.resize(forms.size());
        r_.resize(forms.size());
        B_.resize(forms.size());
        lt_.resize(forms.size());
        Qt_.resize(forms.size());
        for (std::size_t g = 0; g < forms.size(); ++g) {
            const auto& f = forms[g];
            deg_[g] = f.deg;
            if (f.deg < 0) continue;
            if (f.deg <= 1 && f.l.size() != n1) { deg_[g] = -1; continue; }
            if (f.deg == 2 && f.Q.rows() != n1) { deg_[g] = -1; continue; }
            if (f.deg <= 1) {
                lt_[g] = f.l.cwiseProduct(di);
                if (e_.modal) r_[g] = e_.V.transpose() * lt_[g].template cast<seg::Cplx>();
            } else {
                Qt_[g] = di.asDiagonal() * f.Q * di.asDiagonal();
                if (e_.modal) {
                    B_[g] = e_.V.transpose() * Qt_[g].template cast<seg::Cplx>() * e_.V;
                }
            }
        }
    }

    [[nodiscard]] const detail::PeriodMode& mode() const noexcept { return e_; }
    [[nodiscard]] Index n() const noexcept { return e_.n; }

    /// The step's start x0 (reduced): its modal coordinates, or x̃0.
    void start(const Vector& x0) {
        xt0_ = e_.d.cwiseProduct(detail::augment(x0));
        if (e_.modal) z_ = e_.Vi * xt0_.template cast<seg::Cplx>();
    }

    /// The closed form's end after L against the engine's end x1, in the
    /// energy norm of the augmented coordinates: ‖x̃(L) − x̃1‖ / (‖x̃0‖ + ‖x̃1‖).
    [[nodiscard]] Real end_gap(Real L, const Vector& x1) {
        const Vector xt1 = e_.d.cwiseProduct(detail::augment(x1));
        Vector xe;
        if (e_.modal) {
            const Kern& k = kern_of_(L);
            xe = (e_.V * z_.cwiseProduct(k.E)).real();
        } else {
            xe = gram_of_(L).Phi * xt0_;
        }
        return (xe - xt1).norm() / (xt0_.norm() + xt1.norm());
    }

    /// Form g over [s0, s0 + L] from the start: out = {∫y, ∫y², y(s0),
    /// y(s0 + L)}, mag = the magnitudes of the terms summed for out[2],
    /// out[3]. False (why set): no closed form for it.
    bool integrate(std::size_t g, Real s0, Real L, Real* out, Real* mag, const char** why) {
        const int dg = deg_[g];
        if (dg < 0) { *why = "not a polynomial of degree <= 2 in readings affine in the state"; return false; }
        if (dg == 0) {
            const Real c = lt_[g][e_.n];
            out[0] = c * L; out[1] = c * c * L; out[2] = out[3] = c;
            mag[0] = mag[1] = std::abs(c);
            return true;
        }
        // the modal sums of a product of p modal expansions carry u·κ^p:
        // degree 1 ∫y² and degree 2 ∫y are p = 2, degree 2 ∫y² is p = 4
        if (dg == 1) {
            if (modal_ok_(2)) return modal_(g, dg, s0, L, out, mag, true);
            return matrix_(g, dg, s0, L, out, mag);
        }
        if (modal_ok_(4)) return modal_(g, dg, s0, L, out, mag, true);
        if (e_.n + 1 <= kKronMax) return matrix_(g, dg, s0, L, out, mag);
        // too large for the Kronecker square: ∫y closed, ∫y² the line
        const bool ok = modal_ok_(2) ? modal_(g, dg, s0, L, out, mag, false)
                                     : matrix_(g, dg, s0, L, out, mag);
        if (ok) *why = kNoFourth;
        return ok;
    }

    /// The largest n + 1 whose degree-2 ∫y² takes the Kronecker matrix form
    /// (a Gramian of dimension (n + 1)²).
    static constexpr Index kKronMax = 12;
    /// out[1] when the square has no closed form here (the recorder takes
    /// its line for ∫y² only), and the reason integrate() then gives.
    static constexpr const char* kNoFourth =
        "the square of a degree-2 probe on a mode with more than 11 states whose eigenbasis is "
        "too ill-conditioned for it (u*kappa^4 > 1e-10)";

private:
    detail::PeriodMode e_;
    std::vector<int> deg_;
    std::vector<Eigen::VectorXcd> r_;
    std::vector<Eigen::MatrixXcd> B_;
    std::vector<Vector> lt_;
    std::vector<DenseMatrix> Qt_;
    Vector xt0_;
    Eigen::VectorXcd z_, zeta_, alpha_;
    Eigen::MatrixXcd beta_;

    struct Kern {
        Real L = std::numeric_limits<Real>::quiet_NaN();
        Eigen::VectorXcd p1, E;
        Eigen::MatrixXcd P2, P4;
        bool have4 = false;
    };
    std::array<Kern, 2> kern_;
    std::size_t kern_last_ = 0;

    Kern& kern_of_(Real L) {
        for (std::size_t s = 0; s < 2; ++s) {
            if (kern_[s].L == L) { kern_last_ = s; return kern_[s]; }
        }
        const std::size_t s = 1 - kern_last_;
        Kern& k = kern_[s];
        k.L = L;
        k.have4 = false;
        detail::phi1_kernel(e_.mu, L, k.p1);
        detail::phi1_pair_kernel(e_.mu, L, k.P2);
        k.E = (e_.mu * L).array().exp().matrix();
        kern_last_ = s;
        return k;
    }
    // the quartic kernel over the pairs p = (i <= j): Lφ₁((σ_p + σ_q) L)
    void quartic_(Kern& k) {
        if (k.have4) return;
        const Index n1 = e_.mu.size();
        const Index M = n1 * (n1 + 1) / 2;
        Eigen::VectorXcd sig(M);
        for (Index i = 0, p = 0; i < n1; ++i) {
            for (Index j = i; j < n1; ++j) sig[p++] = e_.mu[i] + e_.mu[j];
        }
        detail::phi1_pair_kernel(sig, k.L, k.P4);
        k.have4 = true;
    }

    [[nodiscard]] bool modal_ok_(int p) const noexcept {
        return e_.modal && kExactRoundoff * std::pow(e_.kappa, p) <= kExactTierBound;
    }

    bool modal_(std::size_t g, int dg, Real s0, Real L, Real* out, Real* mag, bool sq) {
        const Index n1 = e_.n + 1;
        const Eigen::VectorXcd* zp = &z_;
        if (s0 != Real{0}) {
            zeta_ = z_.cwiseProduct((e_.mu * s0).array().exp().matrix());
            zp = &zeta_;
        }
        Kern& k = kern_of_(L);
        if (dg == 1) {
            alpha_ = r_[g].cwiseProduct(*zp);
            out[0] = alpha_.cwiseProduct(k.p1).sum().real();
            out[1] = (alpha_.transpose() * k.P2 * alpha_)(0).real();
            out[2] = alpha_.sum().real();
            out[3] = alpha_.cwiseProduct(k.E).sum().real();
            mag[0] = alpha_.cwiseAbs().sum();
            mag[1] = alpha_.cwiseProduct(k.E).cwiseAbs().sum();
            return true;
        }
        beta_ = B_[g].cwiseProduct(*zp * zp->transpose());
        out[0] = beta_.cwiseProduct(k.P2).sum().real();
        out[2] = beta_.sum().real();
        const Eigen::MatrixXcd EE = k.E * k.E.transpose();
        out[3] = beta_.cwiseProduct(EE).sum().real();
        mag[0] = beta_.cwiseAbs().sum();
        mag[1] = beta_.cwiseProduct(EE).cwiseAbs().sum();
        if (!sq) {
            out[1] = std::numeric_limits<Real>::quiet_NaN();
            return true;
        }
        quartic_(k);
        const Index M = n1 * (n1 + 1) / 2;
        Eigen::VectorXcd gam(M);
        for (Index i = 0, p = 0; i < n1; ++i) {
            for (Index j = i; j < n1; ++j) gam[p++] = i == j ? beta_(i, i) : beta_(i, j) + beta_(j, i);
        }
        out[1] = (gam.transpose() * k.P4 * gam)(0).real();
        return true;
    }

    // the matrix form (no modal basis): degree 1, and degree 2's ∫y
    struct Gram {
        Real L = std::numeric_limits<Real>::quiet_NaN();
        DenseMatrix Phi, I;
        std::vector<DenseMatrix> W;   // per form g (degree 1: l̃l̃ᵀ; degree 2: Q̃), sized lazily
        std::vector<DenseMatrix> W4;  // degree 2: the Kronecker Gramian of vec(Q̃)vec(Q̃)ᵀ
        std::vector<char> have;
    };
    std::array<Gram, 2> gram_;
    std::size_t gram_last_ = 0;
    DenseMatrix phi_s0_;

    Gram& gram_of_(Real L) {
        for (std::size_t s = 0; s < 2; ++s) {
            if (gram_[s].L == L) { gram_last_ = s; return gram_[s]; }
        }
        const std::size_t s = 1 - gram_last_;
        Gram& gm = gram_[s];
        gm.L = L;
        std::vector<DenseMatrix> none;
        seg::gram_doubling(e_.As, L, {}, gm.Phi, gm.I, none);
        gm.W.assign(deg_.size(), DenseMatrix{});
        gm.W4.assign(deg_.size(), DenseMatrix{});
        gm.have.assign(deg_.size(), 0);
        gram_last_ = s;
        return gm;
    }

    bool matrix_(std::size_t g, int dg, Real s0, Real L, Real* out, Real* mag) {
        Vector xs = xt0_;
        if (s0 != Real{0}) {
            std::vector<DenseMatrix> none;
            DenseMatrix I0;
            seg::gram_doubling(e_.As, s0, {}, phi_s0_, I0, none);
            xs = phi_s0_ * xt0_;
        }
        Gram& gm = gram_of_(L);
        if (!gm.have[g]) {
            const DenseMatrix Q = dg == 1 ? DenseMatrix(lt_[g] * lt_[g].transpose()) : Qt_[g];
            const DenseMatrix* qs[1] = {&Q};
            DenseMatrix Phi, I;
            std::vector<DenseMatrix> W;
            seg::gram_doubling(e_.As, L, qs, Phi, I, W);
            gm.W[g] = std::move(W[0]);
            if (dg == 2 && e_.n + 1 <= kKronMax) {
                // w = x̃ ⊗ x̃: w' = (As ⊗ I + I ⊗ As) w, y = vec(Q̃)·w, so
                // ∫y² = w0ᵀ W4 w0 with W4 the Gramian of vec(Q̃)vec(Q̃)ᵀ
                const Index n1 = e_.n + 1;
                const DenseMatrix I1 = DenseMatrix::Identity(n1, n1);
                DenseMatrix AK(n1 * n1, n1 * n1);
                for (Index i = 0; i < n1; ++i) {
                    for (Index j = 0; j < n1; ++j) {
                        AK.block(i * n1, j * n1, n1, n1) = e_.As(i, j) * I1;
                        if (i == j) AK.block(i * n1, j * n1, n1, n1) += e_.As;
                    }
                }
                const Eigen::Map<const Vector> q(Q.data(), n1 * n1);
                const DenseMatrix QK = q * q.transpose();
                const DenseMatrix* qk[1] = {&QK};
                seg::gram_doubling(AK, L, qk, Phi, I, W);
                gm.W4[g] = std::move(W[0]);
            }
            gm.have[g] = 1;
        }
        const Vector xe = gm.Phi * xs;
        if (dg == 2) {
            const DenseMatrix& Q = Qt_[g];
            out[0] = xs.dot(gm.W[g] * xs);
            if (gm.W4[g].size() > 0) {
                Vector w(xs.size() * xs.size());
                for (Index i = 0; i < xs.size(); ++i) w.segment(i * xs.size(), xs.size()) = xs[i] * xs;
                out[1] = w.dot(gm.W4[g] * w);
            } else {
                out[1] = std::numeric_limits<Real>::quiet_NaN();
            }
            out[2] = xs.dot(Q * xs);
            out[3] = xe.dot(Q * xe);
            mag[0] = (Q.cwiseAbs() * xs.cwiseAbs()).dot(xs.cwiseAbs());
            mag[1] = (Q.cwiseAbs() * xe.cwiseAbs()).dot(xe.cwiseAbs());
            return true;
        }
        const Vector& l = lt_[g];
        out[0] = l.dot(gm.I * xs);
        out[1] = xs.dot(gm.W[g] * xs);
        out[2] = l.dot(xs);
        out[3] = l.dot(xe);
        mag[0] = l.cwiseProduct(xs).cwiseAbs().sum();
        mag[1] = l.cwiseProduct(xe).cwiseAbs().sum();
        return true;
    }
};

/// The recorders' segment source on the exact engine (record::SegmentSource):
/// the readings of a record::LoopFeed as forms, the probes composed per mode,
/// one accepted step at a time (`begin` before the feed samples it).
template <class AdapterT>
class ExactSegmentSource final : public record::SegmentSource {
public:
    using MaskT = topology::SwitchStateMask;

    ExactSegmentSource(AdapterT& ad, const record::LoopFeed& feed) : ad_{ad} {
        if (ad.has_dynamic_sources()) {
            off_ = "the circuit has time-varying sources (a step's input is not its mode's "
                   "constant b)";
        }
        using L = record::DeviceCurrent::Law;
        const auto obs_of = [](const record::LinearReading& r) {
            PeriodObservable o;
            for (std::size_t k = 0; k < r.cols.size(); ++k) {
                o.w.emplace_back(r.cols[k], r.coef[k] / r.div);
            }
            return o;
        };
        for (const auto& r : feed.readings()) {
            Reading rd;
            if (r.dev.law == L::None) {
                rd.deg = r.product ? 2 : 1;
                rd.a = add_(obs_of(r.a));
                if (r.product) rd.b = add_(obs_of(r.b));
            } else if (r.dev.law == L::Pwl) {
                const auto& law = r.dev.pwl;
                PeriodObservable i;
                if (law.var >= 0) {
                    i.w.emplace_back(law.var, Real{1});
                } else {
                    if (law.from >= 0) i.w.emplace_back(law.from, Real{1});
                    if (law.to >= 0) i.w.emplace_back(law.to, Real{-1});
                    i.bit = static_cast<int>(law.bit);
                    i.g_on = law.g_on;
                    i.g_off = law.g_off;
                    i.v_on = law.diode ? law.v_f : Real{0};
                }
                rd.deg = r.dev.power ? 2 : 1;
                rd.a = add_(std::move(i));
                if (r.dev.power) rd.b = add_(obs_of(r.a));
            } else if (r.dev.law == L::Constant) {
                rd.c = ad.pool().current_source_params(r.dev.branch).I;
                rd.deg = r.dev.power ? 1 : 0;
                if (r.dev.power) rd.a = add_(obs_of(r.a));
            }   // else: a capacitor's companion, a Newton law, a waveform (deg −1)
            readings_.push_back(rd);
        }
        for (const auto& rec : feed.recorders()) {
            base_.push_back(probes_.size());
            for (Size p = 0; p < rec.n_probes(); ++p) {
                const auto& pr = rec.probes()[p];
                const auto at = rec.atom_readings(p);
                probes_.push_back({std::vector<ir::Expr::Instr>(pr.program().begin(),
                                                                pr.program().end()),
                                   std::vector<Size>(at.begin(), at.end()), pr.text()});
            }
        }
        stamp_.assign(probes_.size(), 0);
        ok_.assign(probes_.size(), 0);
        full_.assign(probes_.size(), {});
        pwhy_.assign(probes_.size(), nullptr);
    }

    /// The accepted step [t0, t1] of mode m from the reduced state x0 (the
    /// scheduler's start of it: the last step's end or an event's right
    /// limit) to x1 (its end).
    void begin(Real t0, Real t1, const Vector& x0, const Vector& x1, const MaskT& m) {
        t0_ = t0;
        t1_ = t1;
        x0_ = x0;
        x1_ = &x1;
        m_ = m;
        step_ok_ = -1;
        sm_ = nullptr;
        ++step_;
        active_ = true;
    }
    void end() noexcept {
        active_ = false;
        x1_ = nullptr;
    }

    bool piece(Size rec, Size p, Real ta, Real tb, Real /*ya*/, Real /*yb*/, Real lo, Real hi,
               Real* out) noexcept override {
        try {
            return piece_(rec, p, ta, tb, lo, hi, out);
        } catch (const std::exception&) {
            return line_("the closed form could not be built for this mode");
        }
    }

    [[nodiscard]] Size n_closed() const noexcept { return n_closed_; }
    [[nodiscard]] Size n_line() const noexcept { return n_line_; }
    [[nodiscard]] const std::string& why() const noexcept { return why_; }

private:
    // the step's closed-form end against the engine's, energy norm (the
    // propagators agree to u·κ; a numeric or corrected step is far off)
    static constexpr Real kEndGap = Real{1e-9};

    struct Reading {
        int deg = -1;
        std::size_t a = 0, b = 0;   // observables (deg 1: a; deg 2: a × b)
        Real c = 1;                 // deg 0: the value; deg ≤ 1: a's factor
    };
    struct ProbeProg {
        std::vector<ir::Expr::Instr> prog;
        std::vector<Size> atoms;   // their readings
        std::string text;
    };

    AdapterT& ad_;
    std::string off_;
    std::vector<PeriodObservable> obs_;
    std::vector<Reading> readings_;
    std::vector<ProbeProg> probes_;
    std::vector<Size> base_;
    std::map<MaskT, std::unique_ptr<SegmentMode>> modes_;
    SegmentMode* sm_ = nullptr;
    Real t0_ = 0, t1_ = 0;
    Vector x0_;
    const Vector* x1_ = nullptr;
    int step_ok_ = -1;   // the step's end check: −1 not yet made
    MaskT m_{0};
    std::size_t step_ = 0;
    bool active_ = false;
    std::vector<std::size_t> stamp_;
    std::vector<char> ok_;
    std::vector<std::array<Real, 4>> full_;
    std::vector<const char*> pwhy_;
    Size n_closed_ = 0, n_line_ = 0;
    std::string why_;

    std::size_t add_(PeriodObservable o) {
        obs_.push_back(std::move(o));
        return obs_.size() - 1;
    }

    bool line_(const std::string& why) noexcept {
        ++n_line_;
        if (why_.empty()) why_ = why;
        return false;
    }

    SegmentMode& mode_() {
        if (sm_ != nullptr) return *sm_;
        auto it = modes_.find(m_);
        if (it == modes_.end()) {
            const MaskT keep = ad_.current_mask();
            auto e = detail::make_period_mode(ad_, m_, obs_, true);
            ad_.set_mask(keep);
            const Index n1 = e.n + 1;
            std::vector<SegForm> rf(readings_.size());
            for (std::size_t j = 0; j < readings_.size(); ++j) {
                const auto& rd = readings_[j];
                const auto row = [&](std::size_t o) -> Vector {
                    return e.C.row(static_cast<Index>(o)).transpose();
                };
                if (rd.deg == 0) rf[j] = seg::constant(n1, rd.c);
                else if (rd.deg == 1) rf[j] = seg::linear(rd.c * row(rd.a));
                else if (rd.deg == 2) rf[j] = seg::product(seg::linear(row(rd.a)), seg::linear(row(rd.b)));
            }
            std::vector<SegForm> forms;
            forms.reserve(probes_.size());
            std::vector<SegForm> at;
            for (const auto& pp : probes_) {
                at.clear();
                for (const Size j : pp.atoms) at.push_back(j < rf.size() ? rf[j] : SegForm{});
                forms.push_back(seg::compose(pp.prog, at, n1));
            }
            it = modes_.emplace(m_, std::make_unique<SegmentMode>(std::move(e), forms)).first;
        }
        sm_ = it->second.get();
        if (x0_.size() != sm_->n()) throw std::runtime_error("state size");
        sm_->start(x0_);
        return *sm_;
    }

    bool piece_(Size rec, Size p, Real ta, Real tb, Real lo, Real hi, Real* out) {
        if (!off_.empty()) return line_(off_);
        if (!active_ || ta != t0_ || tb != t1_) {
            return line_("a recorder piece that is not one accepted step of the engine");
        }
        const std::size_t g = base_.at(rec) + p;
        SegmentMode& sm = mode_();
        if (step_ok_ < 0) {   // the step is its mode's closed form: its end is the engine's
            step_ok_ = x1_ != nullptr && sm.end_gap(t1_ - t0_, *x1_) <= kEndGap ? 1 : 0;
        }
        if (step_ok_ == 0) {
            return line_("a step whose end is not its mode's closed form (a numeric step or "
                         "a corrected state)");
        }
        if (stamp_[g] != step_) {
            stamp_[g] = step_;
            Real mag[2];
            const char* why = nullptr;
            ok_[g] = sm.integrate(g, Real{0}, t1_ - t0_, full_[g].data(), mag, &why) ? 1 : 0;
            pwhy_[g] = why;
        }
        if (!ok_[g]) {
            return line_(std::string("probe '") + probes_[g].text + "': " +
                         (pwhy_[g] != nullptr ? pwhy_[g] : "no closed form"));
        }
        if (lo == ta && hi == tb) {
            for (int k = 0; k < 4; ++k) out[k] = full_[g][static_cast<std::size_t>(k)];
        } else {
            Real mag[2];
            const char* why = nullptr;
            if (!sm.integrate(g, lo - ta, hi - lo, out, mag, &why)) {
                return line_(std::string("probe '") + probes_[g].text + "': " + why);
            }
        }
        if (std::isnan(out[1])) {   // ∫y closed, ∫y² the recorder's line: counted
            line_(std::string("probe '") + probes_[g].text + "': " + SegmentMode::kNoFourth);
            return true;
        }
        ++n_closed_;
        return true;
    }
};

}  // namespace pulsim::dsed
