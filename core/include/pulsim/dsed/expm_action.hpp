// add-exact-propagator-strategy D13 / tasks 15.1-15.2: the action of the
// exponential on the sparse matrix of a mode, and the per-mode route rule
// (dense propagator, action, or hand-over to the implicit strategy).
//
// The action is the truncated Taylor series with scaling of Al-Mohy &
// Higham (2011), "Computing the action of the matrix exponential", Alg. 3.2,
// with the shift mu = trace(A)/n and the theta_m table of the double
// precision tolerance 2^-53. Shortcut: (m, s) are chosen from ||A||_1 (the
// alpha_1 bound) and not from the sharper d_p = ||A^p||^(1/p) estimates, so
// a highly non-normal mode takes up to ~2x the matvecs of scipy's
// expm_multiply (2492 there, 5610 here on the 1 ns snubber ladder); upgrade
// with a normest1 of A^p when such a mode must run on the action.
#pragma once

#include "pulsim/numeric/types.hpp"

#include <Eigen/Sparse>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <string>

namespace pulsim::dsed {

using SparseA = Eigen::SparseMatrix<Real, Eigen::ColMajor>;

/// theta_m for m = 1..55 at tolerance 2^-53 (Al-Mohy & Higham 2011, Table
/// A.3; the values scipy's expm_multiply uses), zero where not tabulated.
inline constexpr std::array<Real, 56> kExpmActionTheta = [] {
    std::array<Real, 56> t{};
    t[1] = 2.29e-16; t[2] = 2.58e-8; t[3] = 1.39e-5; t[4] = 3.40e-4; t[5] = 2.40e-3;
    t[6] = 9.07e-3; t[7] = 2.38e-2; t[8] = 5.00e-2; t[9] = 8.96e-2; t[10] = 1.44e-1;
    t[11] = 2.14e-1; t[12] = 3.00e-1; t[13] = 4.00e-1; t[14] = 5.14e-1; t[15] = 6.41e-1;
    t[16] = 7.81e-1; t[17] = 9.31e-1; t[18] = 1.09; t[19] = 1.26; t[20] = 1.44;
    t[21] = 1.62; t[22] = 1.82; t[23] = 2.01; t[24] = 2.22; t[25] = 2.43;
    t[26] = 2.64; t[27] = 2.86; t[28] = 3.08; t[29] = 3.31; t[30] = 3.54;
    t[35] = 4.7; t[40] = 6.0; t[45] = 7.2; t[50] = 8.5; t[55] = 9.9;
    return t;
}();

/// The Taylor degree m and the scaling s for ||A·h||_1 = `norm` (after the
/// shift): argmin m·ceil(norm / theta_m), m <= 55. norm = 0: (0, 1).
struct ExpmActionPlan {
    int m = 0;
    long s = 1;
    [[nodiscard]] long matvecs() const noexcept { return static_cast<long>(m) * s; }
};

[[nodiscard]] inline ExpmActionPlan expm_action_plan(Real norm) {
    ExpmActionPlan best;
    if (!(norm > Real{0})) return best;
    Real cost = std::numeric_limits<Real>::infinity();
    for (int m = 1; m <= 55; ++m) {
        const Real th = kExpmActionTheta[static_cast<std::size_t>(m)];
        if (th <= Real{0}) continue;
        const Real s = std::max(Real{1}, std::ceil(norm / th));
        if (m * s < cost) {
            cost = m * s;
            best = {m, static_cast<long>(s)};
        }
    }
    return best;
}

/// ||A||_1 of a sparse matrix (max column sum).
[[nodiscard]] inline Real sparse_norm1(const SparseA& A) {
    Real n1 = 0;
    for (Eigen::Index k = 0; k < A.outerSize(); ++k) {
        Real c = 0;
        for (SparseA::InnerIterator it(A, k); it; ++it) c += std::abs(it.value());
        n1 = std::max(n1, c);
    }
    return n1;
}

/// The augmented matrix [[A, b], [0, 0]] of ẋ = A·x + b (the constant
/// source a state at 1), compressed.
[[nodiscard]] inline SparseA augment_affine(const SparseA& A, const Vector& b) {
    const Eigen::Index n = A.rows();
    std::vector<Eigen::Triplet<Real>> tr;
    tr.reserve(static_cast<std::size_t>(A.nonZeros() + n));
    for (Eigen::Index k = 0; k < A.outerSize(); ++k)
        for (SparseA::InnerIterator it(A, k); it; ++it)
            tr.emplace_back(it.row(), it.col(), it.value());
    for (Eigen::Index i = 0; i < n; ++i)
        if (b[i] != Real{0}) tr.emplace_back(i, n, b[i]);
    SparseA M(n + 1, n + 1);
    M.setFromTriplets(tr.begin(), tr.end());   // compressed: never refilled
    return M;
}

/// The action e^{A·h}·v of one sparse mode (tasks 15.1). The workspaces
/// are the caller's (sized on the first call); `matvecs` counts the sparse
/// products this call took.
struct ExpmAction {
    SparseA A;            // the mode's matrix (augmented with its sources)
    Real mu = 0;          // trace(A)/n, the shift
    Real norm1 = 0;       // ||A - mu I||_1
    mutable Vector b, f;  // workspaces
    mutable std::size_t matvecs = 0;

    explicit ExpmAction(SparseA M) : A(std::move(M)) {
        A.makeCompressed();
        const Eigen::Index n = A.rows();
        if (n > 0) mu = A.diagonal().sum() / static_cast<Real>(n);
        // ||A - mu I||_1 without forming it: each column's diagonal moves by -mu
        for (Eigen::Index k = 0; k < A.outerSize(); ++k) {
            Real c = 0;
            bool diag = false;
            for (SparseA::InnerIterator it(A, k); it; ++it) {
                if (it.row() == k) { c += std::abs(it.value() - mu); diag = true; }
                else c += std::abs(it.value());
            }
            if (!diag) c += std::abs(mu);
            norm1 = std::max(norm1, c);
        }
    }

    [[nodiscard]] ExpmActionPlan plan(Real h) const { return expm_action_plan(norm1 * std::abs(h)); }

    /// out = e^{A·h}·v (out distinct from v).
    void apply(const Vector& v, Real h, Vector& out) const {
        const ExpmActionPlan p = plan(h);
        constexpr Real tol = Real{1.1102230246251565e-16};   // 2^-53
        out = v;
        if (p.m == 0) {
            out *= std::exp(mu * h);
            return;
        }
        b = v;
        const Real eta = std::exp(mu * h / static_cast<Real>(p.s));
        for (long i = 0; i < p.s; ++i) {
            Real c1 = b.cwiseAbs().maxCoeff();
            for (int j = 1; j <= p.m; ++j) {
                const Real coeff = h / (static_cast<Real>(p.s) * static_cast<Real>(j));
                f.noalias() = A * b;
                ++matvecs;
                b = coeff * (f - mu * b);
                const Real c2 = b.cwiseAbs().maxCoeff();
                out += b;
                if (c1 + c2 <= tol * out.cwiseAbs().maxCoeff()) break;
                c1 = c2;
            }
            out *= eta;
            b = out;
        }
    }
};

/// tasks 15.2 (D13): the route of one mode at its first visit.
struct ExactRoute {
    std::string kernel;     // "dense" | "action" | "implicit"
    std::string reason;     // "" | "cost" (the hand-over)
    Real dense_setup = 0;   // flops: 2 n^3 (6 + log2 ||A h||_1)
    Real dense_step = 0;    // flops: 2 n^2
    Real action_step = 0;   // dense-equivalent flops: 2 nnz m s (the plan's matvecs) x weight
    long matvecs = 0;       // the plan's matvecs per step
};

/// The constants of D13's rule (15.4 calibrates them).
struct ExactRouteCosts {
    long m_max = 200;                              // matvecs per step the action may take
    // 15.4: seconds per action flop (2 nnz per matvec) over seconds per dense flop
    // (2 n^2 per step), C++ on the ladder n = 201..2001 (Release, arm64, Eigen 5): the
    // matvec's O(n) vector work and sparse indexing; measured 4.4e-10..8.5e-10 s against
    // 4.7e-11..8.5e-11 s, ratio 6..17 (DECISIONS phase 3; the [.calib] test case)
    Real action_flop_weight = 8;
    std::size_t dense_budget = std::size_t{1} << 30; // bytes: 48 n^2 cached (V, V^-1, Phi ...)
};

/// D13: dense when its setup is repaid over the steps expected in the mode
/// and it fits the budget; else the action when its plan is within m_max;
/// else the implicit hand-over (reason 'cost'). n, nnz of the augmented
/// matrix; N_steps the steps expected in the mode.
[[nodiscard]] inline ExactRoute choose_exact_route(Real n, Real nnz, Real norm_Ah1, Real N_steps,
                                                   const ExactRouteCosts& c = {}) {
    ExactRoute r;
    const ExpmActionPlan p = expm_action_plan(norm_Ah1);
    r.matvecs = p.matvecs();
    r.dense_setup = Real{2} * n * n * n * (Real{6} + std::max(Real{0}, std::log2(norm_Ah1)));
    r.dense_step = Real{2} * n * n;
    r.action_step = Real{2} * nnz * static_cast<Real>(r.matvecs) * c.action_flop_weight;
    const bool fits = Real{48} * n * n <= static_cast<Real>(c.dense_budget);
    if (fits && r.dense_setup + N_steps * r.dense_step <= N_steps * r.action_step) {
        r.kernel = "dense";
    } else if (r.matvecs <= c.m_max) {
        r.kernel = "action";
    } else {
        r.kernel = "implicit";
        r.reason = "cost";
    }
    return r;
}

}  // namespace pulsim::dsed
