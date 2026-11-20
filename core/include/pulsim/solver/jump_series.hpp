// SPDX-License-Identifier: MIT
//
// The whole jump of an impulsive event as a VECTOR series (the restart rule's
// consistent start, run_transient.hpp `consistent_start`).
//
// The probe's backward-Euler stages of γh on one factor map x_k = T·x_{k−1} + f.
// Their second differences D_k = Δ_k − Δ_{k+1} obey D_{k+1} = T·D_k, and a mode of
// factor ρ and amplitude c gives D_k = −c(1 − ρ)²ρ^{k−1}: its whole jump is
// −c = Σ_k k·D_k = D_1/(1 − ρ)². Over all modes J = (I − T)⁻²·D_1 on the fast
// subspace. On the Krylov space span{D_1 … D_m}, T acts as the m×m matrix H whose
// columns are the shift (T·D_j = D_{j+1}) and, for the last, the least-squares
// coefficients of D_{m+1}; then J = [D_1 … D_m]·(I − H)⁻²·e_1. m modes need
// D_1 … D_{m+1} (m + 2 stages). J lies in span{D_k}: every linear invariant each
// stage keeps (a floating node set's charge, an inductor loop's flux) J keeps to
// rounding, whatever the fit's accuracy.
//
// This file: the coefficients g (J = [D_1 … D_m]·g) from the weighted D's.
#pragma once

#include "pulsim/numeric/dense.hpp"

#include <algorithm>
#include <cmath>
#include <complex>

namespace pulsim::solver {

/// Least-squares fit of column m of `dw` (D_{m+1}, weighted) on columns
/// 0..m−1 (D_1 … D_m). Returns the squared residual; `c` the coefficients.
inline Real jump_series_fit(const DenseMatrix& dw, Index m, Vector& c) {
    const auto v = dw.leftCols(m);
    c = v.colPivHouseholderQr().solve(dw.col(m));
    return (v * c - dw.col(m)).squaredNorm();
}

/// g with J = [D_1 … D_m]·g, from the fit `c` of D_{m+1}: (I − H)⁻²·e_1, H the
/// companion matrix (subdiagonal ones, last column c). Each Ritz value ρ of H is
/// held as the scalar rule holds a: a real one in [0, ρ_max], a complex one
/// (a ringing fast mode) at modulus ≤ ρ_max. False if no finite g.
inline bool jump_series_coeffs(const Vector& c, Real rho_max, Vector& g) {
    const Index m = c.size();
    DenseMatrix h = DenseMatrix::Zero(m, m);
    for (Index j = 0; j + 1 < m; ++j) h(j + 1, j) = Real{1};
    h.col(m - 1) = c;
    const DenseMatrix id = DenseMatrix::Identity(m, m);
    Vector e1 = Vector::Zero(m);
    e1[0] = Real{1};
    Eigen::EigenSolver<DenseMatrix> es(h);
    bool held = es.info() != Eigen::Success;
    using Cx = std::complex<Real>;
    Eigen::Matrix<Cx, Eigen::Dynamic, 1> lam = es.eigenvalues();
    if (!held) {
        for (Index i = 0; i < m; ++i) {
            const Cx l = lam[i];
            Cx lh = l;
            if (l.imag() == Real{0}) {
                lh = Cx(std::clamp(l.real(), Real{0}, rho_max), Real{0});
            } else if (std::abs(l) > rho_max) {
                lh = l * (rho_max / std::abs(l));
            }
            if (lh != l) held = true;
            lam[i] = lh;
        }
    }
    if (!held) {   // every Ritz value inside the rule's range: (I − H)⁻² e1 directly
        const auto lu = (id - h).partialPivLu();
        g = lu.solve(lu.solve(e1));
    } else if (es.info() == Eigen::Success) {
        const Eigen::Matrix<Cx, Eigen::Dynamic, Eigen::Dynamic> y = es.eigenvectors();
        const auto ylu = y.fullPivLu();
        if (!ylu.isInvertible()) return false;
        Eigen::Matrix<Cx, Eigen::Dynamic, 1> a = ylu.solve(e1.cast<Cx>());
        for (Index i = 0; i < m; ++i) {
            const Cx one_minus = Cx(Real{1}) - lam[i];
            a[i] /= one_minus * one_minus;
        }
        g = (y * a).real();
    } else {
        return false;
    }
    return g.allFinite();
}

}  // namespace pulsim::solver
