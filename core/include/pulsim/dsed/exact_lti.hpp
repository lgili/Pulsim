#pragma once

// =============================================================================
// Pulsim — exact stepping of an LTI mode
// =============================================================================
//
// Between events a PWL circuit with DC (or sine-augmented) sources is
// LTI, ẋ = A·x + b, and its trajectory has a closed form valid for ANY
// h — no stability region, no truncation error, no step controller:
//
//   z = V⁻¹x,  w = V⁻¹b
//   z(t+h) = e^{Λh}·z(t) + h·φ₁(Λh)·w,     φ₁(s) = (eˢ − 1)/s
//   x(t+h) = V·z(t+h)                       (real part)
//
// add-exact-propagator-strategy (D1, D2): the propagator states its own
// accuracy per mask, in energy coordinates x̃ = D·x (D = diag(√C, √L)),
// where a lossless network has a skew Ã and κ₂ measures physics, not
// units:
//
//   tier 1  eigendecomposition, accepted when u·κ₂(Ṽ) ≤ 1e-10 AND its
//           first-order propagator error over the run horizon,
//           ‖Ṽ(G∘Ṽ⁻¹R)Ṽ⁻¹‖_F, max over a halving grid of t ≤ H, is ≤ 1e-10
//           (pair_quality); pairs that miss are
//           refined (refine_pairs), then the balanced pairs tried
//   tier 2  Padé [13/13] scaling and squaring of [[Ãh, b̃h],[0,0]]
//           (Higham 2005), squared as e^X − I (expm_aug), cached per
//           h, accepted when its estimated error (pade_defect: two
//           independent scalings, and a u-relative perturbation of the
//           matrix), max over h = h_max·2⁻ᵏ, is ≤ 1e-10. It needs no
//           eigenpairs: degenerate clusters and stiff modes take it.
//   tier 3  neither: no exact step for this mask (`valid == false`);
//           the scheduler takes its numeric path and the reason is
//           in the run's exact report.
//
// A basis is never accepted on invertibility alone: at exact critical
// damping `eig` returns an "invertible" basis with κ ≈ 1e9 and the
// modal formula is 1e-6 off (probes/dsed_baseline.py).

#include "pulsim/numeric/dense.hpp"
#include "pulsim/numeric/types.hpp"

#include <Eigen/Eigenvalues>
#include <Eigen/SVD>
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <limits>
#include <numbers>
#include <string>
#include <utility>
#include <vector>

namespace pulsim::dsed {

/// Unit roundoff u of the D2 bounds.
inline constexpr Real kExactRoundoff = std::numeric_limits<Real>::epsilon();
/// The accepted error level of tiers 1 and 2 (`u·κ`, `u·‖Ah‖₁`).
inline constexpr Real kExactTierBound = Real{1e-10};

/// The propagator of one mode.
struct ExactLTI {
    bool valid = false;        // tier 1 or 2: an exact step exists
    int tier = 3;
    Real kappa = std::numeric_limits<Real>::infinity();  // κ₂(Ṽ)
    Real norm_A1 = Real{0};    // ‖Ã‖₁ (energy coordinates)
    std::string reason;        // why not tier 1 (empty on tier 1)
    Real defect = std::numeric_limits<Real>::quiet_NaN();    // pair_quality of the kept pairs:
    Real backward = std::numeric_limits<Real>::quiet_NaN();  // first-order error, row-wise backward error / u
    int refined = 0;           // refinement iterations the kept pairs took
    bool balanced = false;     // the kept pairs are the balanced matrix's
    Real pade_defect = std::numeric_limits<Real>::quiet_NaN();  // tier 2's error estimate (when judged)

    // Tier 1, physical coordinates: V = D⁻¹Ṽ, V⁻¹ = Ṽ⁻¹D, split into
    // real and imaginary parts so a step multiplies real vectors only.
    DenseMatrix V_re, V_im, Vinv_re, Vinv_im;
    Eigen::VectorXcd lambda;   // eigenvalues (every tier, when eig ran)
    Eigen::VectorXcd w;        // V⁻¹b, cached

    // Tier 2 (energy coordinates).
    DenseMatrix As;
    Vector bs;
    Vector d;                  // D's diagonal
    mutable std::vector<std::pair<Real, DenseMatrix>> pade_cache;
    mutable std::size_t n_pade = 0;       // Padé exponentials evaluated
    mutable std::size_t n_pade_hits = 0;  // cache hits

    // Workspaces: a step allocates nothing.
    mutable Vector z_re, z_im, zn_re, zn_im, y, c1_re, c1_im;
};

/// eˢ − 1 for complex s without cancellation at any |s|.
[[nodiscard]] inline std::complex<Real> expm1c(std::complex<Real> s) {
    const Real a = s.real();
    const Real b = s.imag();
    const Real sh = std::sin(b / Real{2});
    return {std::expm1(a) * std::cos(b) - Real{2} * sh * sh,
            std::exp(a) * std::sin(b)};
}

/// φ₁(s) = (eˢ − 1)/s, relative error near u for every s (D1).
[[nodiscard]] inline std::complex<Real> phi1(std::complex<Real> s) {
    if (s == std::complex<Real>{0}) {
        return Real{1};
    }
    return expm1c(s) / s;
}

/// φ₂(s) = (eˢ − 1 − s)/s², relative error near u for every s: the
/// Taylor series below |s| = 1/2 (where eˢ − 1 − s cancels), the
/// closed form above.
[[nodiscard]] inline std::complex<Real> phi2(std::complex<Real> s) {
    if (std::abs(s) < Real{0.5}) {
        std::complex<Real> term{Real{0.5}};   // s^k/(k+2)!, k = 0
        std::complex<Real> sum = term;
        for (int k = 1; k < 30; ++k) {
            term *= s / static_cast<Real>(k + 2);
            sum += term;
            if (std::abs(term) <= std::numeric_limits<Real>::epsilon() * std::abs(sum)) break;
        }
        return sum;
    }
    return (expm1c(s) - s) / (s * s);
}

namespace detail {
inline std::string fmt_e(Real v) {
    char buf[32];
    std::snprintf(buf, sizeof buf, "%.3e", v);
    return buf;
}
}  // namespace detail

/// Parlett–Reinsch balancing (LAPACK dgebal's scaling step, no permutation):
/// `M ← S⁻¹·M·S` with S diagonal powers of 2 (exact), so every row and column
/// of the result have comparable 1-norms; returns S's diagonal. Eigen's
/// EigenSolver does not balance, and its backward error is ε·‖M‖ in every
/// entry: a g_off-limited winding current (a 4e14 /s row) put that error into
/// the slow modes, and a flyback's output drifted to 1.3e-3 V where the exact
/// solution of the same matrix is −9.2e-8 V. A balanced matrix is unchanged
/// (S = I), so its propagator is bit-identical. make_exact_lti tries the
/// balanced eigenpairs only where the plain ones fail pair_quality.
[[nodiscard]] inline Vector balance_in_place(DenseMatrix& M) {
    const auto n = M.rows();
    Vector s = Vector::Ones(n);
    constexpr Real radix = Real{2};
    for (bool done = false; !done;) {
        done = true;
        for (Eigen::Index i = 0; i < n; ++i) {
            // off-diagonal sums taken directly: sum − |M(i,i)| is cancellation
            // noise when the couplings are below ε·|M(i,i)|
            Real c0 = Real{0}, r = Real{0};
            for (Eigen::Index j = 0; j < n; ++j) {
                if (j != i) { c0 += std::abs(M(j, i)); r += std::abs(M(i, j)); }
            }
            if (!(c0 > Real{0}) || !(r > Real{0}) || !std::isfinite(c0) || !std::isfinite(r)) continue;
            Real c = c0, f = Real{1};
            while (c < r / radix) { f *= radix; c *= radix * radix; }
            while (c > r * radix) { f /= radix; c /= radix * radix; }
            if ((c + r) / f < Real{0.95} * (c0 + r)) {
                done = false;
                s[i] *= f;
                M.row(i) /= f;
                M.col(i) *= f;
            }
        }
    }
    return s;
}

/// κ₂ of a basis (∞ when singular).
[[nodiscard]] inline Real kappa2(const Eigen::MatrixXcd& V) {
    Eigen::JacobiSVD<Eigen::MatrixXcd> svd(V);
    const auto& sv = svd.singularValues();
    const auto n = sv.size();
    return sv(n - 1) > Real{0} ? sv(0) / sv(n - 1) : std::numeric_limits<Real>::infinity();
}

/// How good a set of eigenpairs (Ṽ, Λ) of Ã is: they are exact for Ã + ΔA,
/// ΔA = −R·Ṽ⁻¹, R = ÃṼ − ṼΛ.
///  - `defect`: the first-order error of the propagator over `H` per unit of
///    state (energy coordinates, Frobenius ≥ 2-norm), from the Fréchet
///    derivative of the exponential in that basis (Daleckii–Krein):
///      e^{(Ã+ΔA)H} − e^{ÃH} ≈ −Ṽ·(G ∘ E)·Ṽ⁻¹,   E = Ṽ⁻¹R,
///      G_jk = (e^{λ_j H} − e^{λ_k H})/(λ_j − λ_k) = H·e^{λ_hi H}·φ₁((λ_lo − λ_hi)H)
///    (λ_hi the one with the larger real part). A fast pair's large absolute
///    residual is damped by its own G row; an eigenvalue error of a slow or
///    conserved mode grows as H. It matched an mpmath exponential of the same
///    matrix to three digits on the flyback and phase-shift bridge modes.
///  - `backward`: max_i ‖ΔA_i‖/‖Ã_i‖, the row-wise backward error. Ã's rows
///    are Schur complements carrying their own rounding u·‖Ã_i‖ (a row of
///    ±1.4e14 entries fixes a near-conserved bridge mode only to ~0.03 /s:
///    its eigenvalue in the stored matrix is +0.031), so no method resolves a
///    mode better than that from this matrix. Reported only: such a mode is
///    refused, not called exact (review of slice 5).
/// R = Ã·Ṽ − Ṽ·Λ with every sum compensated (error-free products through
/// fma, TwoSum accumulation), rounded once: a row of ±1e15 entries leaves
/// u·1e15 of noise in a plain product, which drowns the residual of a slow
/// pair and stalls refine_pairs at hundreds of u.
[[nodiscard]] inline Eigen::MatrixXcd residual_compensated(const DenseMatrix& As,
                                                           const Eigen::MatrixXcd& V,
                                                           const Eigen::VectorXcd& lam) {
    const auto n = V.rows();
    const auto m = V.cols();
    Eigen::MatrixXcd R(n, m);
    // s + c += a·b, the product and the sum both error-free
    const auto acc = [](Real& s, Real& c, Real a, Real b) {
        const Real p = a * b;
        const Real pe = std::fma(a, b, -p);
        const Real t = s + p;
        const Real bp = t - s;
        c += (s - (t - bp)) + (p - bp) + pe;
        s = t;
    };
    for (Eigen::Index k = 0; k < m; ++k) {
        const Real lr = lam[k].real(), li = lam[k].imag();
        for (Eigen::Index i = 0; i < n; ++i) {
            Real sr = 0, cr = 0, si = 0, ci = 0;
            for (Eigen::Index l = 0; l < n; ++l) {
                const Real a = As(i, l);
                if (a == Real{0}) continue;
                acc(sr, cr, a, V(l, k).real());
                acc(si, ci, a, V(l, k).imag());
            }
            const Real vr = V(i, k).real(), vi = V(i, k).imag();
            acc(sr, cr, -vr, lr);   // − v·λ, (vr + i·vi)(lr + i·li)
            acc(sr, cr, vi, li);
            acc(si, ci, -vr, li);
            acc(si, ci, -vi, lr);
            R(i, k) = {sr + cr, si + ci};
        }
    }
    return R;
}

struct PairQuality {
    Real defect = std::numeric_limits<Real>::infinity();
    Real backward = std::numeric_limits<Real>::infinity();
    /// ≤ 1: tier 1 may use the pairs (defect ≤ kExactTierBound).
    [[nodiscard]] Real score() const;
};
[[nodiscard]] inline PairQuality pair_quality(const DenseMatrix& As, const Eigen::MatrixXcd& V,
                                              const Eigen::VectorXcd& lam, Real H) {
    const auto n = V.cols();
    const Eigen::MatrixXcd W = V.fullPivLu().inverse();
    const Eigen::MatrixXcd R = residual_compensated(As, V, lam);
    const Eigen::MatrixXcd E0 = W * R;
    // The propagator is used over every segment length up to H, and
    // G_jk(t) of a decaying pair peaks at t ≈ 1/|Re λ| ≪ H (review of slice
    // 5): the defect is the largest over t = H·2^-k down to 1/(8‖Ã‖₁), where
    // a halving grid sees at least 0.93 of each peak.
    const Real na = As.cwiseAbs().colwise().sum().maxCoeff();
    const Real t_lo = std::max(na > Real{0} ? Real{0.125} / na : H, H * Real{0x1p-60});
    Real d = Real{0};
    Eigen::MatrixXcd E(n, n);
    for (Real t = H; ; t *= Real{0.5}) {
        for (Eigen::Index j = 0; j < n; ++j) {
            for (Eigen::Index k = 0; k < n; ++k) {
                const bool jhi = lam[j].real() >= lam[k].real();
                const std::complex<Real> hi = jhi ? lam[j] : lam[k];
                const std::complex<Real> lo = jhi ? lam[k] : lam[j];
                E(j, k) = E0(j, k) * (t * std::exp(hi * t) * phi1((lo - hi) * t));
            }
        }
        const Real dt = (V * E * W).norm();
        d = std::isfinite(dt) ? std::max(d, dt) : std::numeric_limits<Real>::infinity();
        if (!(t > t_lo) || !std::isfinite(d)) break;
    }
    PairQuality q;
    if (std::isfinite(d)) q.defect = d;
    const Eigen::MatrixXcd dA = R * W;
    Real b = Real{0};
    for (Eigen::Index i = 0; i < n; ++i) {
        const Real ri = As.row(i).norm();
        const Real ei = dA.row(i).norm();
        if (ei > Real{0}) b = std::max(b, ri > Real{0} ? ei / ri : std::numeric_limits<Real>::infinity());
    }
    if (std::isfinite(b)) q.backward = b;
    return q;
}

// The backward error is reported, not a pass (review of slice 5): pairs
// within 64 u of a row whose norm is 1e15 still move a slow mode by 0.03 /s,
// a propagator 5e-5 off per 2 ms; the exact engine refuses that mode.
inline Real PairQuality::score() const { return defect / kExactTierBound; }

/// One Newton step on every eigenpair from E = Ṽ⁻¹(ÃṼ − ṼΛ):
/// λ_j += E_jj, v_k += Σ_{j≠k} E_jk/(λ_k − λ_j)·v_j between pairs whose gap
/// is wide (|λ_k − λ_j| > 1e3·|E_jk|); pairs closer than that form a
/// cluster whose block Λ_c + E_cc is diagonalised instead. Up to three steps while the defect is above
/// `target`; a step is kept when it lowers the score, or the defect at an
/// equal score (near a conserved mode the defect sits on the matrix's own
/// rounding and wanders while the backward error falls).
inline int refine_pairs(const DenseMatrix& As, Eigen::MatrixXcd& V, Eigen::VectorXcd& lam,
                        Real H, PairQuality& q, Real target) {
    const auto n = V.cols();
    int used = 0;
    for (int it = 0; it < 3 && q.defect > target; ++it) {
        const Eigen::MatrixXcd W = V.fullPivLu().inverse();
        const Eigen::MatrixXcd E = W * residual_compensated(As, V, lam);
        // clusters: pairs too close for a Newton step (|gap| ≤ 1e3·|E_jk|)
        std::vector<Eigen::Index> root(static_cast<std::size_t>(n));
        for (Eigen::Index k = 0; k < n; ++k) root[static_cast<std::size_t>(k)] = k;
        const auto find = [&](Eigen::Index a) {
            while (root[static_cast<std::size_t>(a)] != a) a = root[static_cast<std::size_t>(a)];
            return a;
        };
        const auto close = [&](Eigen::Index j, Eigen::Index k) {
            const Real g = std::abs(lam[k] - lam[j]);
            return g <= Real{1e3} * std::max(std::abs(E(j, k)), std::abs(E(k, j)));
        };
        for (Eigen::Index k = 0; k < n; ++k) {
            for (Eigen::Index j = 0; j < k; ++j) {
                if (close(j, k)) {
                    const Eigen::Index a = find(j), c = find(k);
                    if (a != c) root[static_cast<std::size_t>(std::max(a, c))] = std::min(a, c);
                }
            }
        }
        Eigen::MatrixXcd C = Eigen::MatrixXcd::Identity(n, n);
        Eigen::VectorXcd l2 = lam;
        for (Eigen::Index k = 0; k < n; ++k) {
            l2[k] += E(k, k);
            for (Eigen::Index j = 0; j < n; ++j) {
                if (j != k && find(j) != find(k)) C(j, k) = E(j, k) / (lam[k] - lam[j]);
            }
        }
        Eigen::MatrixXcd V2 = V * C;
        // a cluster: Ã restricted to its columns is Λ_c + E_cc, not diagonal;
        // diagonalise that small block (five identical rectifier bridges: a
        // λ ≈ −100 cluster whose coupling, ignored, put 2.3e-6 per 6 ms into
        // the propagator)
        for (Eigen::Index r = 0; r < n; ++r) {
            if (find(r) != r) continue;
            std::vector<Eigen::Index> idx;
            for (Eigen::Index k = 0; k < n; ++k) if (find(k) == r) idx.push_back(k);
            if (idx.size() < 2) continue;
            const auto m = static_cast<Eigen::Index>(idx.size());
            Eigen::MatrixXcd M(m, m);
            for (Eigen::Index a = 0; a < m; ++a)
                for (Eigen::Index c = 0; c < m; ++c)
                    M(a, c) = E(idx[a], idx[c]) + (a == c ? lam[idx[a]] : std::complex<Real>{0});
            Eigen::ComplexEigenSolver<Eigen::MatrixXcd> ce(M);
            // a defective block (a noise-split repeated eigenvalue) keeps its pairs
            if (ce.info() != Eigen::Success || !(kappa2(ce.eigenvectors()) <= Real{1e4})) continue;
            Eigen::MatrixXcd Vc(n, m);
            for (Eigen::Index a = 0; a < m; ++a) Vc.col(a) = V2.col(idx[a]);
            const Eigen::MatrixXcd Vr = Vc * ce.eigenvectors();
            for (Eigen::Index a = 0; a < m; ++a) {
                V2.col(idx[a]) = Vr.col(a);
                l2[idx[a]] = ce.eigenvalues()[a];
            }
        }
        V2.colwise().normalize();
        const PairQuality q2 = pair_quality(As, V2, l2, H);
        const bool better = q2.score() < q.score()
                            || (q2.score() == q.score() && q2.defect < q.defect);
        if (!better) break;
        V = std::move(V2);
        lam = std::move(l2);
        q = q2;
        ++used;
    }
    return used;
}

/// θ₁₃ of Higham 2005: ‖X‖₁ ≤ θ₁₃ needs no squaring.
inline constexpr Real kPadeTheta13 = 5.371920351148152;

/// e^M by scaling and squaring of the [13/13] Padé approximant (Higham
/// 2005, θ₁₃ = 5.37), squared in the form B = e^X − I, (I + B)² =
/// I + (2B + B²). Squaring e^X itself keeps the slow part of a stiff
/// matrix as I + 1e-11·(…) and amplifies that rounding at every one of
/// the ~23 squarings: on a paralleled-bridge mode (‖Ãh‖₁ = 2.5e7)
/// Eigen's `.exp()` was 3.0e-9 off a 60-digit exponential, this form
/// 4.3e-11 (macOS arm64, Eigen 5; DECISIONS, exact engine 7).
[[nodiscard]] inline DenseMatrix expm_aug(const DenseMatrix& M) {
    constexpr Real c[] = {64764752532480000.0, 32382376266240000.0, 7771770303897600.0,
                          1187353796428800.0, 129060195264000.0, 10559470521600.0,
                          670442572800.0, 33522128640.0, 1323241920.0, 40840800.0,
                          960960.0, 16380.0, 182.0, 1.0};
    const auto n = M.rows();
    const Real nrm = n == 0 ? Real{0} : M.cwiseAbs().colwise().sum().maxCoeff();
    if (!std::isfinite(nrm)) {
        return DenseMatrix::Constant(n, n, std::numeric_limits<Real>::quiet_NaN());
    }
    int s = 0;
    if (nrm > kPadeTheta13) {
        std::frexp(nrm / kPadeTheta13, &s);   // nrm/θ ≤ 2^s, exact (no libm rounding)
    }
    const DenseMatrix X = M * std::ldexp(Real{1}, -s);
    const DenseMatrix I = DenseMatrix::Identity(n, n);
    const DenseMatrix X2 = X * X;
    const DenseMatrix X4 = X2 * X2;
    const DenseMatrix X6 = X4 * X2;
    const DenseMatrix Uin = c[13] * X6 + c[11] * X4 + c[9] * X2;
    const DenseMatrix U = X * (X6 * Uin + c[7] * X6 + c[5] * X4 + c[3] * X2 + c[1] * I);
    const DenseMatrix Vin = c[12] * X6 + c[10] * X4 + c[8] * X2;
    const DenseMatrix V = X6 * Vin + c[6] * X6 + c[4] * X4 + c[2] * X2 + c[0] * I;
    DenseMatrix B = (V - U).partialPivLu().solve(Real{2} * U);   // r(X) − I
    for (int k = 0; k < s; ++k) {
        B = (Real{2} * B + B * B).eval();
    }
    B.diagonal().array() += Real{1};
    return B;
}

/// Tier 2's error estimate at step h, for M = [[Ã, b̃],[0, 0]]: the
/// larger of (1) the disagreement of e^{Mh} with (e^{Mh/3})³, two
/// scalings whose rounding is independent, and (2) the change of e^{Mh}
/// under u-relative perturbations of M's entries (two fixed sign
/// patterns, the larger): an estimate of what the stored matrix's own
/// rounding does to e^{Mh}. As ‖ΔE‖_F / max(1, ‖E‖_F). An estimate, not
/// a bound: (1) alone missed a stiff near-conserved mode by 6× (its two
/// scalings share the rounding of the cancelling rows), and a single
/// sign pattern moved 37× under a reordering of the states (a cancelling
/// pair given equal signs); the larger of all three was 1.3–10× above a
/// 60-digit error on every probe (1.8× at the least).
[[nodiscard]] inline Real pade_defect_at(const DenseMatrix& M, Real h) {
    const DenseMatrix E = expm_aug(M * h);
    const DenseMatrix E3 = expm_aug(M * (h / Real{3}));
    Real d = (E - E3 * E3 * E3).norm();
    for (int pattern = 0; pattern < 2; ++pattern) {
        DenseMatrix P = M;
        for (Eigen::Index j = 0; j < P.cols(); ++j) {
            for (Eigen::Index i = 0; i < P.rows(); ++i) {
                const bool plus = pattern == 0 ? (7 * i + 13 * j) % 5 < 3 : (i + j) % 2 == 0;
                P(i, j) *= Real{1} + (plus ? kExactRoundoff : -kExactRoundoff);
            }
        }
        d = std::max(d, (E - expm_aug(P * h)).norm());
    }
    d /= std::max(Real{1}, E.norm());
    return std::isfinite(d) ? d : std::numeric_limits<Real>::infinity();
}

/// The largest pade_defect_at over h = h_max·2⁻ᵏ (one length per
/// squaring count; steps between grid points are estimated by their
/// neighbours, not bounded), down to the first h that needs no squaring.
/// Stops at the first miss. Judged on the constant forcing column b̃: the
/// per-step level + ramp columns of exact_advance_affine_into are not
/// judged separately (their exponential shares the state block).
[[nodiscard]] inline Real pade_defect(const DenseMatrix& As, const Vector& bs, Real h_max) {
    const auto n = As.rows();
    DenseMatrix M = DenseMatrix::Zero(n + 1, n + 1);
    M.topLeftCorner(n, n) = As;
    M.topRightCorner(n, 1) = bs;
    const Real nrm = M.cwiseAbs().colwise().sum().maxCoeff();
    Real worst = Real{0};
    Real h = h_max;
    for (int k = 0; k < 128; ++k, h *= Real{0.5}) {
        worst = std::max(worst, pade_defect_at(M, h));
        if (!(worst <= kExactTierBound) || nrm * h <= kPadeTheta13) {
            break;
        }
    }
    return worst;
}

/// Build the propagator of ẋ = A·x + b. `scale` is D's diagonal (√m
/// per state, empty for the identity); `h_max` is the longest step the
/// caller will take, which tier 2's bound needs; `horizon` is the run's
/// length, over which tier 1's propagator defect is stated (≥ h_max).
[[nodiscard]] inline ExactLTI make_exact_lti(const DenseMatrix& A,
                                             const Vector& b,
                                             const Vector& scale = Vector{},
                                             Real h_max = Real{1e-5},
                                             Real horizon = Real{0}) {
    ExactLTI out;
    const auto n = A.rows();
    if (n == 0) {
        // a static mode: nothing to propagate, the step is the identity
        // on an empty state (the samples are the mode's algebraic solve)
        out.kappa = Real{1};
        out.tier = 1;
        out.valid = true;
        return out;
    }
    out.d = scale.size() == n ? scale : Vector::Ones(n);
    const Vector dinv = out.d.cwiseInverse();
    out.As = out.d.asDiagonal() * A * dinv.asDiagonal();
    out.bs = out.d.cwiseProduct(b);
    out.norm_A1 = out.As.cwiseAbs().colwise().sum().maxCoeff();

    // add-exact-propagator-strategy D2 (review of slice 4): the pairs are
    // judged by their own first-order propagator error over the horizon, not
    // by κ₂ alone. Eigen's EigenSolver does not balance and its backward error
    // is ε·‖Ã‖ in every entry: a g_off-limited winding row (4e14 /s) put that
    // error into a slow mode (flyback output 1.3e-3 V for −9.2e-8 V), and a
    // phase-shift bridge's conserved mode came out at λ = −0.0985 /s for
    // 1.2e-13 (6.3e-4 per ms). Balancing helps some matrices and hurts others
    // (the scaling back, up to 4e6, amplified the bridge's error). So: the plain
    // pairs, refined while their defect is above kFine; the balanced pairs,
    // refined, only when the plain ones miss both bars, and kept only when
    // their score is under half the plain one's. A mode whose best pairs miss
    // both bars gets no tier-1 step (typed in `reason`).
    const Real H = std::max(h_max, horizon);
    constexpr Real kFine = Real{1e-12};   // below this, keep the pairs as computed
    Eigen::EigenSolver<DenseMatrix> es(out.As);
    bool ok = es.info() == Eigen::Success;
    Eigen::MatrixXcd Vt;
    PairQuality q;
    if (ok) {
        out.lambda = es.eigenvalues();
        Vt = es.eigenvectors();
        q = pair_quality(out.As, Vt, out.lambda, H);
        if (q.defect > kFine) {
            out.refined = refine_pairs(out.As, Vt, out.lambda, H, q, kFine);
        }
    }
    if (!ok || q.score() > Real{1}) {
        DenseMatrix Ab = out.As;
        const Vector sb = balance_in_place(Ab);
        Eigen::EigenSolver<DenseMatrix> eb;
        if ((sb.array() != Real{1}).any() && eb.compute(Ab).info() == Eigen::Success) {
            Eigen::MatrixXcd Vb = sb.cast<std::complex<Real>>().asDiagonal() * eb.eigenvectors();
            Vb.colwise().normalize();
            Eigen::VectorXcd lb = eb.eigenvalues();
            PairQuality qb = pair_quality(out.As, Vb, lb, H);
            const int rb = qb.defect > kFine ? refine_pairs(out.As, Vb, lb, H, qb, kFine) : 0;
            if (!ok || qb.score() < Real{0.5} * q.score()) {
                Vt = std::move(Vb);
                out.lambda = std::move(lb);
                q = qb;
                out.refined = rb;
                out.balanced = true;
                ok = true;
            }
        }
    }
    out.defect = q.defect;
    out.backward = q.backward / kExactRoundoff;
    if (ok) out.kappa = kappa2(Vt);   // unit columns (Eigen's, or normalised above)
    if (ok) {
        if (kExactRoundoff * out.kappa <= kExactTierBound && q.score() <= Real{1}) {
            const Eigen::MatrixXcd Vt_inv = Vt.fullPivLu().inverse();
            const Eigen::MatrixXcd V = dinv.asDiagonal() * Vt;
            const Eigen::MatrixXcd V_inv = Vt_inv * out.d.asDiagonal();
            out.V_re = V.real();
            out.V_im = V.imag();
            out.Vinv_re = V_inv.real();
            out.Vinv_im = V_inv.imag();
            out.w = V_inv * b.cast<std::complex<Real>>();
            out.tier = 1;
            out.valid = true;
            out.z_re.resize(n);
            out.z_im.resize(n);
            out.zn_re.resize(n);
            out.zn_im.resize(n);
            return out;
        }
        out.reason = kExactRoundoff * out.kappa > kExactTierBound
            ? "kappa2(V) = " + detail::fmt_e(out.kappa) + " > 1e-10/u (near-defective eigenbasis)"
            : "propagator defect = " + detail::fmt_e(out.defect)
                  + " > 1e-10 over the run horizon (inaccurate eigenpairs"
                  + (out.backward <= Real{64} ? ": an ill-conditioned eigenvalue cluster" : "")
                  + "; backward error = " + detail::fmt_e(out.backward) + " u)";
    } else {
        out.reason = "eigensolver failed";
    }
    out.pade_defect = pade_defect(out.As, out.bs, h_max);
    if (out.pade_defect <= kExactTierBound) {
        out.tier = 2;
        out.valid = true;
        out.y.resize(n);
        return out;
    }
    out.reason += "; Pade defect = " + detail::fmt_e(out.pade_defect)
                  + " > 1e-10 at ||A*h||_1 <= " + detail::fmt_e(out.norm_A1 * h_max)
                  + ": no exact step for this mode";
    return out;
}

/// Back-compatible overload: b = 0 (pass b to exact_advance instead).
[[nodiscard]] inline ExactLTI make_exact_lti(const DenseMatrix& A) {
    return make_exact_lti(A, Vector::Zero(A.rows()));
}

/// The tier-2 exponential of [[Ã, b̃],[0, 0]]·h.
[[nodiscard]] inline DenseMatrix pade_compute(const ExactLTI& e,
                                              const Vector& bs, Real h) {
    const auto n = e.As.rows();
    DenseMatrix M = DenseMatrix::Zero(n + 1, n + 1);
    M.topLeftCorner(n, n) = e.As * h;
    M.topRightCorner(n, 1) = bs * h;
    ++e.n_pade;
    return expm_aug(M);
}

/// Same, cached per h (the main step repeats h = dt_max); least
/// recently used out, so partial steps evict the main step's only when
/// more than kPadeCacheSize − 1 other lengths cycle between two uses.
[[nodiscard]] inline const DenseMatrix& pade_exp(const ExactLTI& e, Real h) {
    for (std::size_t i = 0; i < e.pade_cache.size(); ++i) {
        if (e.pade_cache[i].first == h) {
            ++e.n_pade_hits;
            std::rotate(e.pade_cache.begin() + static_cast<std::ptrdiff_t>(i),
                        e.pade_cache.begin() + static_cast<std::ptrdiff_t>(i) + 1,
                        e.pade_cache.end());
            return e.pade_cache.back().second;
        }
    }
    constexpr std::size_t kPadeCacheSize = 64;   // event scans take up to n_scan lengths per step
    if (e.pade_cache.size() >= kPadeCacheSize) {
        e.pade_cache.erase(e.pade_cache.begin());
    }
    e.pade_cache.emplace_back(h, pade_compute(e, e.bs, h));
    return e.pade_cache.back().second;
}

/// Advance `x` by `h` into `out` (same size; no allocation after the
/// first call on tier 1).
inline void exact_advance_into(const ExactLTI& e, const Vector& x, Real h,
                               Vector& out) {
    const auto n = x.size();
    if (e.tier == 2) {
        const DenseMatrix& E = pade_exp(e, h);
        e.y.noalias() = E.topLeftCorner(n, n) * e.d.cwiseProduct(x);
        out = (e.y + E.topRightCorner(n, 1)).cwiseQuotient(e.d);
        return;
    }
    e.z_re.noalias() = e.Vinv_re * x;
    e.z_im.noalias() = e.Vinv_im * x;
    for (Eigen::Index i = 0; i < n; ++i) {
        const std::complex<Real> s = e.lambda[i] * h;
        // exp(s) of a strongly decayed mode underflows to 0: exact.
        const std::complex<Real> zi =
            std::exp(s) * std::complex<Real>{e.z_re[i], e.z_im[i]}
            + h * phi1(s) * e.w[i];
        e.zn_re[i] = zi.real();
        e.zn_im[i] = zi.imag();
    }
    out.noalias() = e.V_re * e.zn_re;
    out.noalias() -= e.V_im * e.zn_im;
}

/// Advance `x` by `h` under ẋ = A·x + b + c0 + c1·τ (τ ∈ [0, h], b the
/// vector the propagator was built with): a source that is level + ramp
/// over the step (add-exact-propagator-strategy D14),
///   x(h) = e^{Ah}x + h·φ₁(Ah)(b + c0) + h²·φ₂(Ah)·c1.
/// Tier 2: the exponential of [[Ãh, (b̃ + c̃0)h, c̃1h²],[0,0,0],[0,1,0]]
/// (states 1 and τ/h), not cached (c0, c1 change at every corner).
inline void exact_advance_affine_into(const ExactLTI& e, const Vector& x, Real h,
                                      const Vector& c0, const Vector& c1,
                                      Vector& out) {
    const auto n = x.size();
    if (e.tier == 2) {
        DenseMatrix M = DenseMatrix::Zero(n + 2, n + 2);
        M.topLeftCorner(n, n) = e.As * h;
        M.block(0, n, n, 1) = (e.bs + e.d.cwiseProduct(c0)) * h;
        M.block(0, n + 1, n, 1) = e.d.cwiseProduct(c1) * (h * h);
        M(n + 1, n) = Real{1};
        ++e.n_pade;
        const DenseMatrix E = expm_aug(M);
        e.y.noalias() = E.topLeftCorner(n, n) * e.d.cwiseProduct(x);
        out = (e.y + E.block(0, n, n, 1)).cwiseQuotient(e.d);
        return;
    }
    e.z_re.noalias() = e.Vinv_re * x;
    e.z_im.noalias() = e.Vinv_im * x;
    e.zn_re.noalias() = e.Vinv_re * c0;   // V⁻¹c0 (re, im), then V⁻¹c1
    e.zn_im.noalias() = e.Vinv_im * c0;
    e.c1_re.noalias() = e.Vinv_re * c1;
    e.c1_im.noalias() = e.Vinv_im * c1;
    for (Eigen::Index i = 0; i < n; ++i) {
        const std::complex<Real> s = e.lambda[i] * h;
        const std::complex<Real> w0 = e.w[i] + std::complex<Real>{e.zn_re[i], e.zn_im[i]};
        const std::complex<Real> w1{e.c1_re[i], e.c1_im[i]};
        const std::complex<Real> zi =
            std::exp(s) * std::complex<Real>{e.z_re[i], e.z_im[i]}
            + h * phi1(s) * w0 + (h * h) * phi2(s) * w1;
        e.zn_re[i] = zi.real();
        e.zn_im[i] = zi.imag();
    }
    out.noalias() = e.V_re * e.zn_re;
    out.noalias() -= e.V_im * e.zn_im;
}

/// Advance `x` by `h` under ẋ = A·x + b with the b the propagator was
/// built with.
[[nodiscard]] inline Vector exact_advance(const ExactLTI& e, const Vector& x,
                                          Real h) {
    Vector out(x.size());
    exact_advance_into(e, x, h, out);
    return out;
}

/// Advance `x` by `h` under ẋ = A·x + b for an explicit b (tests and
/// the b = 0 overload of make_exact_lti; not the hot path).
[[nodiscard]] inline Vector exact_advance(const ExactLTI& e, const Vector& x,
                                          const Vector& b, Real h) {
    const auto n = x.size();
    if (e.tier == 2) {
        const DenseMatrix E = pade_compute(e, e.d.cwiseProduct(b), h);
        const Vector y = E.topLeftCorner(n, n) * e.d.cwiseProduct(x)
                         + E.topRightCorner(n, 1);
        return y.cwiseQuotient(e.d);
    }
    const Eigen::MatrixXcd Vinv =
        e.Vinv_re.cast<std::complex<Real>>()
        + std::complex<Real>{0, 1} * e.Vinv_im.cast<std::complex<Real>>();
    const Eigen::VectorXcd w = Vinv * b.cast<std::complex<Real>>();
    const Eigen::VectorXcd z = Vinv * x.cast<std::complex<Real>>();
    Eigen::VectorXcd zn(n);
    for (Eigen::Index i = 0; i < n; ++i) {
        const std::complex<Real> s = e.lambda[i] * h;
        zn[i] = std::exp(s) * z[i] + h * phi1(s) * w[i];
    }
    return e.V_re * zn.real() - e.V_im * zn.imag();
}

/// Scan step for event location (D3): an eighth of the period of the
/// fastest under-damped mode (|Re λ| ≤ |Im λ|), +∞ when there is none.
/// shortcut: every under-damped mode counts, not only those the
/// predicate can see (cᵀV ≠ 0) — over-scans resonant circuits; upgrade
/// when predicates become affine rows (task 3.2).
[[nodiscard]] inline Real exact_scan_step(const ExactLTI& e) {
    Real w_max = Real{0};
    for (Eigen::Index i = 0; i < e.lambda.size(); ++i) {
        const Real wi = std::abs(e.lambda[i].imag());
        if (wi > w_max && std::abs(e.lambda[i].real()) <= wi) {
            w_max = wi;
        }
    }
    return w_max > Real{0}
        ? std::numbers::pi_v<Real> / (Real{4} * w_max)
        : std::numeric_limits<Real>::infinity();
}

/// Instants after an event at which a mode too fast for the output step
/// `h` is sampled: t_e + s·√2^k from s_lo = max(1/(8ρ_max), 1e-5·h) while
/// below min(30/ρ_min, h), ρ the decay rates −Re λ > 8/h. Empty when no
/// mode decays faster than 8/h, or when the slowest such mode is gone
/// (30/ρ_min) before 1e-5·h. A linear interpolant through these exact
/// samples is within a few per cent of the jump of the true exponential,
/// where one step of h spread the whole transient as a ramp.
inline void exact_fast_samples(const ExactLTI& e, Real t_e, Real h,
                               std::vector<Real>& out) {
    out.clear();
    if (!e.valid || !(h > Real{0})) return;
    Real r_lo = std::numeric_limits<Real>::infinity(), r_hi = Real{0};
    for (Eigen::Index i = 0; i < e.lambda.size(); ++i) {
        const Real r = -e.lambda[i].real();
        if (r > Real{8} / h) {
            r_lo = std::min(r_lo, r);
            r_hi = std::max(r_hi, r);
        }
    }
    if (r_hi == Real{0}) return;
    const Real s_hi = std::min(Real{30} / r_lo, h);
    for (Real s = std::max(Real{0.125} / r_hi, Real{1e-5} * h); s < s_hi;
         s *= std::numbers::sqrt2_v<Real>) {
        out.push_back(t_e + s);
    }
}

/// The settle length of a mode for decisions at an event: the parasitic
/// cluster is the modes whose time constant is below kParasiticFraction of
/// the run's resolved time scale `t_res`; δ = kSettleTau time constants of
/// the slowest of them (e^-40 ≈ 4e-18). 0 when the cluster is empty.
inline constexpr Real kParasiticFraction = Real{1e-6};
inline constexpr Real kSettleTau = Real{40};
[[nodiscard]] inline Real exact_settle_length(const ExactLTI& e, Real t_res) {
    if (!e.valid || !(t_res > Real{0})) return Real{0};
    const Real r_cut = Real{1} / (kParasiticFraction * t_res);
    Real r_lo = std::numeric_limits<Real>::infinity();
    for (Eigen::Index i = 0; i < e.lambda.size(); ++i) {
        const Real r = -e.lambda[i].real();
        if (r > r_cut) r_lo = std::min(r_lo, r);
    }
    return std::isfinite(r_lo) ? kSettleTau / r_lo : Real{0};
}

/// Slow-manifold projection (event_projection.hpp) with the cached
/// tier-1 basis: modes with Re λ < −fast_threshold go to their
/// quasi-static value −w/λ. False when tier 1 is not available.
inline bool exact_project_slow(const ExactLTI& e, Vector& x,
                               Real fast_threshold) {
    if (e.tier != 1 || !(fast_threshold > Real{0})) {
        return false;
    }
    const auto n = x.size();
    bool any_fast = false;
    for (Eigen::Index i = 0; i < n; ++i) {
        any_fast = any_fast || e.lambda[i].real() < -fast_threshold;
    }
    if (!any_fast) {
        return true;    // identity
    }
    e.z_re.noalias() = e.Vinv_re * x;
    e.z_im.noalias() = e.Vinv_im * x;
    for (Eigen::Index i = 0; i < n; ++i) {
        if (e.lambda[i].real() < -fast_threshold) {
            const std::complex<Real> zq = -e.w[i] / e.lambda[i];
            e.z_re[i] = zq.real();
            e.z_im[i] = zq.imag();
        }
    }
    x.noalias() = e.V_re * e.z_re;
    x.noalias() -= e.V_im * e.z_im;
    return true;
}

}  // namespace pulsim::dsed
