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
//   tier 1  eigendecomposition, accepted when u·κ₂(Ṽ) ≤ 1e-10
//   tier 2  Padé [13/13] scaling and squaring of [[Ãh, b̃h],[0,0]]
//           (Higham 2005, Eigen's MatrixFunctions), cached per h,
//           accepted when u·‖Ã·h_max‖₁ ≤ 1e-10
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
#include <unsupported/Eigen/MatrixFunctions>

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
    mutable Vector z_re, z_im, zn_re, zn_im, y;
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

namespace detail {
inline std::string fmt_e(Real v) {
    char buf[32];
    std::snprintf(buf, sizeof buf, "%.3e", v);
    return buf;
}
}  // namespace detail

/// Build the propagator of ẋ = A·x + b. `scale` is D's diagonal (√m
/// per state, empty for the identity); `h_max` is the longest step the
/// caller will take, which tier 2's bound needs.
[[nodiscard]] inline ExactLTI make_exact_lti(const DenseMatrix& A,
                                             const Vector& b,
                                             const Vector& scale = Vector{},
                                             Real h_max = Real{1e-5}) {
    ExactLTI out;
    const auto n = A.rows();
    if (n == 0) {
        out.reason = "no state";
        return out;
    }
    out.d = scale.size() == n ? scale : Vector::Ones(n);
    const Vector dinv = out.d.cwiseInverse();
    out.As = out.d.asDiagonal() * A * dinv.asDiagonal();
    out.bs = out.d.cwiseProduct(b);
    out.norm_A1 = out.As.cwiseAbs().colwise().sum().maxCoeff();

    Eigen::EigenSolver<DenseMatrix> es(out.As);
    if (es.info() == Eigen::Success) {
        out.lambda = es.eigenvalues();
        const Eigen::MatrixXcd Vt = es.eigenvectors();
        Eigen::JacobiSVD<Eigen::MatrixXcd> svd(Vt);
        const auto& sv = svd.singularValues();
        out.kappa = sv(n - 1) > Real{0}
            ? sv(0) / sv(n - 1)
            : std::numeric_limits<Real>::infinity();
        if (kExactRoundoff * out.kappa <= kExactTierBound) {
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
        out.reason = "kappa2(V) = " + detail::fmt_e(out.kappa)
                     + " > 1e-10/u (near-defective eigenbasis)";
    } else {
        out.reason = "eigensolver failed";
    }
    const Real nah = out.norm_A1 * h_max;
    if (kExactRoundoff * nah <= kExactTierBound) {
        out.tier = 2;
        out.valid = true;
        out.y.resize(n);
        return out;
    }
    out.reason += "; ||A*h||_1 = " + detail::fmt_e(nah)
                  + " > 1e-10/u (stiff): no exact step for this mode";
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
    return M.exp();
}

/// Same, cached per h (the main step repeats h = dt_max).
[[nodiscard]] inline const DenseMatrix& pade_exp(const ExactLTI& e, Real h) {
    for (const auto& [hh, E] : e.pade_cache) {
        if (hh == h) {
            ++e.n_pade_hits;
            return E;
        }
    }
    constexpr std::size_t kPadeCacheSize = 16;
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
