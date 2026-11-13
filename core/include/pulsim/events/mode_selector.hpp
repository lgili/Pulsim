#pragma once

// =============================================================================
// Pulsim — natural diode commutation as a complementarity problem
// =============================================================================
//
// add-complementarity-mode-selection, tasks 2 and 3 (design D1-D5, D8).
//
// Every PWL diode is an ideal element in series with V_f and R_on, with
// the leak g_off from anode to cathode (D1):
//
//     0 <= i_d  ⟂  w = V_f + R_on·i_d − v_ak >= 0,   i_term = i_d + g_off·v_ak
//
// At an event the diode ports of the backward-Euler companion network of
// a step h_e (state held) obey an affine relation; the mode right after
// the event is the solution of ONE linear complementarity problem on it
// (D2). This header holds the three dense, engine-free pieces:
//
//   * LemkeSolver   — Lemke's complementary pivoting, covering vector 1,
//                     lexicographic ratio test (D4). Finite for every LCP;
//                     on a copositive-plus (PSD) matrix a secondary ray
//                     proves that no solution exists (D3). A line-by-line
//                     port of the prototype (probes/lcp_core.py::lemke),
//                     including its tie tolerances, so pivot counts match.
//                     No heap allocation per call once the size is seen.
//   * enumerate_lcp — the 2^k oracle / non-passive fallback (D3), k <= 12.
//   * ModeSelector  — the port LCP in the previous mode's ("hybrid")
//                     coordinates from the k-port (Z, u) of the previous
//                     mode's own companion network, circuit-scale
//                     tolerances, tie widening (D5), hysteresis ports (D8).
//
// The engine supplies (Z, u) through its own factor (see
// NativeCircuitBuilderAdapter::diode_port_problem): u are the port
// voltages of the previous mode's companion solution, Z the driving-point
// matrix of that network (v = u − Z·e for a current e drawn from each
// anode into its cathode by an external element).

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include <Eigen/Dense>

#include "pulsim/numeric/dense.hpp"
#include "pulsim/numeric/types.hpp"

namespace pulsim::events {

enum class LcpStatus : int {
    Trivial = 0,       // q >= 0: z = 0 (the previous mode is consistent)
    Solution = 1,
    SecondaryRay = 2,  // no solution on a copositive-plus matrix
    PivotCap = 3,      // bug trap; lexicographic Lemke is finite
};

[[nodiscard]] inline const char* to_string(LcpStatus s) noexcept {
    switch (s) {
    case LcpStatus::Trivial: return "trivial";
    case LcpStatus::Solution: return "solution";
    case LcpStatus::SecondaryRay: return "secondary ray";
    case LcpStatus::PivotCap: return "pivot cap (bug)";
    }
    return "?";
}

struct LcpOutcome {
    LcpStatus status = LcpStatus::Trivial;
    int pivots = 0;
};

/// Lemke's method with the lexicographic ratio test: solve
/// w = M·z + q, z, w >= 0, zᵀw = 0.
class LemkeSolver {
public:
    LcpOutcome solve(const DenseMatrix& M, const Vector& q, Vector& z,
                     Vector& w) {
        using EI = Eigen::Index;
        const EI n = q.size();
        z.setZero(n);
        if (n == 0 || q.minCoeff() >= Real{0}) {
            w = q;
            return {LcpStatus::Trivial, 0};
        }
        const int max_pivots = 10 * static_cast<int>((n + 1) * (n + 1));
        // tableau columns: w (0..n-1), z (n..2n-1), z0 (2n)
        T_.resize(n, 2 * n + 1);
        T_.leftCols(n).setIdentity();
        T_.middleCols(n, n) = -M;
        T_.col(2 * n).setConstant(Real{-1});
        b_ = q;
        basis_.resize(static_cast<std::size_t>(n));
        for (EI i = 0; i < n; ++i) basis_[static_cast<std::size_t>(i)] = i;
        key_.resize(n + 1);
        best_.resize(n + 1);
        const Real scale = std::max({Real{1}, M.cwiseAbs().maxCoeff(),
                                     q.cwiseAbs().maxCoeff()});

        // first pivot: z0 enters; the lexicographically smallest row of
        // (b_i, B⁻¹_i) among the minimum b leaves
        EI col = 2 * n;
        const Real m = b_.minCoeff();
        EI r = -1;
        for (EI i = 0; i < n; ++i) {
            if (b_[i] > m + Real{1e-14} * scale) continue;
            if (r < 0) { r = i; continue; }
            for (EI c = 0; c <= n; ++c) {
                const Real d = row_key_raw_(i, c) - row_key_raw_(r, c);
                if (std::abs(d) > Real{1e-14}) {
                    if (d < 0) r = i;
                    break;
                }
            }
        }
        EI leaving = basis_[static_cast<std::size_t>(r)];
        pivot_(r, col);
        basis_[static_cast<std::size_t>(r)] = col;
        int pivots = 1;
        EI entering = leaving < n ? leaving + n : leaving - n;
        while (true) {
            if (pivots > max_pivots) return {LcpStatus::PivotCap, pivots};
            Real dmax = Real{1};
            for (EI i = 0; i < n; ++i) dmax = std::max(dmax, std::abs(T_(i, entering)));
            const Real dtol = Real{1e-12} * dmax;
            Real mr = std::numeric_limits<Real>::infinity();
            bool any = false;
            for (EI i = 0; i < n; ++i) {
                if (T_(i, entering) > dtol) {
                    any = true;
                    mr = std::min(mr, b_[i] / T_(i, entering));
                }
            }
            if (!any) return {LcpStatus::SecondaryRay, pivots};
            // ties of the minimum ratio; z0's row wins, else lexicographic
            EI first_tie = -1, z0_row = -1;
            int n_ties = 0;
            for (EI i = 0; i < n; ++i) {
                if (T_(i, entering) <= dtol) continue;
                const Real ratio = b_[i] / T_(i, entering);
                if (ratio <= mr + Real{1e-13} * (Real{1} + std::abs(mr))) {
                    if (first_tie < 0) first_tie = i;
                    ++n_ties;
                    if (basis_[static_cast<std::size_t>(i)] == 2 * n && z0_row < 0) z0_row = i;
                }
            }
            if (z0_row >= 0) {
                r = z0_row;
            } else if (n_ties == 1) {
                r = first_tie;
            } else {
                r = lexmin_(entering, dtol, mr);
            }
            leaving = basis_[static_cast<std::size_t>(r)];
            pivot_(r, entering);
            basis_[static_cast<std::size_t>(r)] = entering;
            ++pivots;
            if (leaving == 2 * n) break;
            entering = leaving < n ? leaving + n : leaving - n;
        }
        w.setZero(n);
        for (EI i = 0; i < n; ++i) {
            const EI v = basis_[static_cast<std::size_t>(i)];
            if (v < n) w[v] = b_[i];
            else if (v < 2 * n) z[v - n] = b_[i];
        }
        return {LcpStatus::Solution, pivots};
    }

private:
    [[nodiscard]] Real row_key_raw_(Eigen::Index i, Eigen::Index c) const {
        return c == 0 ? b_[i] : T_(i, c - 1);
    }

    void pivot_(Eigen::Index r, Eigen::Index col) {
        const Real p = T_(r, col);
        T_.row(r) /= p;
        b_[r] /= p;
        for (Eigen::Index i = 0; i < T_.rows(); ++i) {
            if (i == r) continue;
            const Real f = T_(i, col);
            if (f != Real{0}) {
                T_.row(i) -= f * T_.row(r);
                b_[i] -= f * b_[r];
            }
        }
        T_.col(col).setZero();
        T_(r, col) = Real{1};
    }

    // lexicographic minimum of (b_i, B⁻¹_i) / d_i over the tied rows
    [[nodiscard]] Eigen::Index lexmin_(Eigen::Index col, Real dtol, Real mr) {
        const Eigen::Index n = T_.rows();
        Eigen::Index bi = -1;
        for (Eigen::Index i = 0; i < n; ++i) {
            const Real d = T_(i, col);
            if (d <= dtol) continue;
            if (b_[i] / d > mr + Real{1e-13} * (Real{1} + std::abs(mr))) continue;
            key_[0] = b_[i] / d;
            for (Eigen::Index c = 0; c < n; ++c) key_[c + 1] = T_(i, c) / d;
            if (bi < 0) { best_ = key_; bi = i; continue; }
            for (Eigen::Index c = 0; c <= n; ++c) {
                const Real dd = key_[c] - best_[c];
                if (std::abs(dd) > Real{1e-13} * (Real{1} + std::abs(best_[c]))) {
                    if (dd < 0) { best_ = key_; bi = i; }
                    break;
                }
            }
        }
        return bi;
    }

    DenseMatrix T_;
    Vector b_, key_, best_;
    std::vector<Eigen::Index> basis_;
};

/// Every complementary solution by the 2^k principal systems (k <= 12):
/// the oracle and the non-passive fallback of D3. Returns the number of
/// solutions; (z, w) is the one with the fewest basic z (fewest flips
/// from the coordinates' own mode), ties by port order.
inline int enumerate_lcp(const DenseMatrix& M, const Vector& q, Vector& z,
                         Vector& w, Real tol = Real{1e-8}) {
    const Eigen::Index k = q.size();
    if (k > 12) return -1;
    const Real sc = std::max(Real{1}, k ? q.cwiseAbs().maxCoeff() : Real{0});
    int count = 0;
    int best_pop = std::numeric_limits<int>::max();
    Vector zc(k), wc(k);
    for (std::uint32_t mask = 0; mask < (1u << k); ++mask) {
        std::vector<Eigen::Index> a;
        for (Eigen::Index j = 0; j < k; ++j) if (mask >> j & 1u) a.push_back(j);
        zc.setZero();
        if (!a.empty()) {
            const auto na = static_cast<Eigen::Index>(a.size());
            DenseMatrix Maa(na, na);
            Vector qa(na);
            for (Eigen::Index r = 0; r < na; ++r) {
                qa[r] = q[a[static_cast<std::size_t>(r)]];
                for (Eigen::Index c = 0; c < na; ++c)
                    Maa(r, c) = M(a[static_cast<std::size_t>(r)], a[static_cast<std::size_t>(c)]);
            }
            const Vector za = Maa.completeOrthogonalDecomposition().solve(-qa);
            if ((Maa * za + qa).cwiseAbs().maxCoeff() > tol * sc) continue;
            for (Eigen::Index r = 0; r < na; ++r) zc[a[static_cast<std::size_t>(r)]] = za[r];
        }
        wc = M * zc + q;
        if ((k && zc.minCoeff() < -tol * sc) || (k && wc.minCoeff() < -tol * sc)) continue;
        ++count;
        const int pop = static_cast<int>(a.size());
        if (pop < best_pop) { best_pop = pop; z = zc; w = wc; }
    }
    return count;
}

/// One diode port in the conductance form (D1, R_on = 1/g_on).
struct DiodePort {
    Real R_on;
    Real g_off;
    Real V_f;
    Real V_th;   // > V_f: hysteresis (D8); <= V_f: no meaning (D1)
};

/// The answer for one event (D11).
struct ModeSelection {
    std::vector<char> on;     // per port: 1 conducting after the event
    LcpStatus status = LcpStatus::Trivial;
    int pivots = 0;           // first LCP
    int pivots_total = 0;     // with tie re-solves and hysteresis re-solves
    int lcps = 0;             // LCPs solved for this event
    int ties = 0;             // still tied after widening: decided OFF
    int widenings = 0;        // re-solves at a widened step
    int released = 0;         // hysteresis ports released (D8)
    int first_pass_ties = 0;  // ports tied at h_e (before widening)
    std::vector<Real> tie_currents;   // |i| of each port tied at h_e (A)
};

/// Circuit scales for the absolute tolerances of D4.
struct CircuitScale {
    Real vscale = Real{1};    // volts (1 V floor)
    Real iscale = Real{1};    // amperes
    Real c_max = Real{0};     // largest capacitance (F): σ·C_max term of tol_i
};

class ModeSelector {
public:
    /// Tolerance coefficient of D4 (tol_w = tol·vscale).
    Real tol = Real{1e-9};

    /// The port LCP in the previous mode's coordinates: z_j = w_j for a
    /// port ON in the previous mode, i_j for one OFF; the complement
    /// ŵ = H·z + q is i_j (ON before) or w_j (OFF before). (Z, u): the
    /// k-port of the previous mode's companion network.
    static void hybrid_lcp(const DenseMatrix& Z, const Vector& u,
                           const std::vector<DiodePort>& ports,
                           const std::vector<char>& prev_on,
                           DenseMatrix& H, Vector& q) {
        const Eigen::Index k = u.size();
        // e = Da·z + Γ·v (excess current over the previous stamp),
        // v = (I + ZΓ)⁻¹(u − Z·Da·z), ŵ = A·z + B·v + c
        Vector Da(k), Gm(k), A(k), B(k), c(k);
        for (Eigen::Index j = 0; j < k; ++j) {
            const auto& p = ports[static_cast<std::size_t>(j)];
            if (prev_on[static_cast<std::size_t>(j)]) {
                Da[j] = Real{1} / p.R_on; Gm[j] = p.g_off;
                A[j] = Real{1} / p.R_on; B[j] = Real{1} / p.R_on;
                c[j] = -p.V_f / p.R_on;
            } else {
                Da[j] = Real{1}; Gm[j] = Real{0};
                A[j] = p.R_on; B[j] = Real{-1}; c[j] = p.V_f;
            }
        }
        DenseMatrix IZG = Z * Gm.asDiagonal();
        IZG.diagonal().array() += Real{1};
        const Eigen::PartialPivLU<DenseMatrix> F(IZG);
        const DenseMatrix FZ = F.solve(Z);
        H = -(B.asDiagonal() * FZ * Da.asDiagonal());
        H.diagonal() += A;
        q = B.asDiagonal() * F.solve(u) + c;
    }

    /// Select the mode after an event. `port_at(h, Z, u)` fills the
    /// k-port of the previous mode's BE companion network of step h
    /// (called at h_e, and at 100·h_e, 1e4·h_e on ties).
    template <class PortFn>
    ModeSelection select(const std::vector<DiodePort>& ports,
                         const std::vector<char>& prev_on, PortFn&& port_at,
                         Real h_e, const CircuitScale& sc) {
        const auto k = static_cast<Eigen::Index>(ports.size());
        ModeSelection out;
        out.on.assign(ports.size(), 0);
        if (k == 0) return out;
        st_.assign(ports.size(), 0);
        const Real factors[3] = {Real{1}, Real{100}, Real{1e4}};
        for (int pass = 0; pass < 3; ++pass) {
            if (pass > 0) {
                bool tie = false;
                for (int s : st_) tie = tie || s < 0;
                if (!tie) break;
                ++out.widenings;
            }
            const Real h = h_e * factors[pass];
            port_at(h, Z_, u_);
            hybrid_lcp(Z_, u_, ports, prev_on, H_, q_);
            int piv = 0;
            const LcpStatus status = solve_with_hysteresis_(ports, prev_on, sc, h, piv, out.released);
            ++out.lcps;
            if (pass == 0) out.pivots = piv;
            out.pivots_total += piv;
            const bool failed = status == LcpStatus::SecondaryRay || status == LcpStatus::PivotCap;
            if (pass == 0) out.status = status;
            if (failed && pass == 0) return out;
            if (failed) break;  // a failed widening keeps the h_e decision; its ties stay OFF
            for (Eigen::Index j = 0; j < k; ++j) {
                auto& s = st_[static_cast<std::size_t>(j)];
                if (pass == 0 && cls_[static_cast<std::size_t>(j)] < 0) {
                    ++out.first_pass_ties;
                    out.tie_currents.push_back(std::abs(i_[j]));
                }
                if (pass == 0 || s < 0) s = cls_[static_cast<std::size_t>(j)];
            }
        }
        for (Eigen::Index j = 0; j < k; ++j) {
            const int s = st_[static_cast<std::size_t>(j)];
            if (s < 0) ++out.ties;
            out.on[static_cast<std::size_t>(j)] = s > 0 ? 1 : 0;
        }
        return out;
    }

    /// Physical port currents (A) and gaps (V) of the last LCP solved.
    [[nodiscard]] const Vector& last_current() const noexcept { return i_; }
    [[nodiscard]] const Vector& last_gap() const noexcept { return wg_; }

private:
    // Solve the LCP of (H_, q_) with OFF hysteresis ports held OFF,
    // releasing each held port whose voltage reaches V_th (D8, bounded:
    // the held set only shrinks), then classify every port into cls_.
    LcpStatus solve_with_hysteresis_(const std::vector<DiodePort>& ports,
                                     const std::vector<char>& prev_on,
                                     const CircuitScale& sc, Real h,
                                     int& pivots, int& released) {
        const auto k = static_cast<Eigen::Index>(ports.size());
        held_.assign(ports.size(), 0);
        for (Eigen::Index j = 0; j < k; ++j) {
            const auto& p = ports[static_cast<std::size_t>(j)];
            held_[static_cast<std::size_t>(j)] =
                (!prev_on[static_cast<std::size_t>(j)] && p.V_th > p.V_f) ? 1 : 0;
        }
        // impedance scale: currents are compared in volts (prototype)
        const Real zs = sc.vscale / sc.iscale;
        LcpStatus status = LcpStatus::Trivial;
        while (true) {
            free_.clear();
            for (Eigen::Index j = 0; j < k; ++j)
                if (!held_[static_cast<std::size_t>(j)]) free_.push_back(j);
            const auto nf = static_cast<Eigen::Index>(free_.size());
            Hs_.resize(nf, nf);
            qs_.resize(nf);
            for (Eigen::Index r = 0; r < nf; ++r) {
                const Eigen::Index jr = free_[static_cast<std::size_t>(r)];
                // ŵ of an ON-before port is a current; z of an OFF-before port too
                const Real sr = prev_on[static_cast<std::size_t>(jr)] ? zs : Real{1};
                qs_[r] = sr * q_[jr];
                for (Eigen::Index c = 0; c < nf; ++c) {
                    const Eigen::Index jc = free_[static_cast<std::size_t>(c)];
                    const Real scol = prev_on[static_cast<std::size_t>(jc)] ? Real{1} : zs;
                    Hs_(r, c) = sr * H_(jr, jc) / scol;
                }
            }
            const LcpOutcome o = lemke_.solve(Hs_, qs_, zf_, wf_);
            pivots += o.pivots;
            status = o.status;
            if (status == LcpStatus::SecondaryRay || status == LcpStatus::PivotCap)
                return status;
            // back to physical coordinates; held ports: z = i = 0
            zfull_.setZero(k);
            for (Eigen::Index r = 0; r < nf; ++r) {
                const Eigen::Index j = free_[static_cast<std::size_t>(r)];
                zfull_[j] = prev_on[static_cast<std::size_t>(j)] ? zf_[r] : zf_[r] / zs;
            }
            // held ports: their complement from the relation; free ports:
            // Lemke's own (exactly complementary) values
            what_ = H_ * zfull_ + q_;
            for (Eigen::Index r = 0; r < nf; ++r) {
                const Eigen::Index j = free_[static_cast<std::size_t>(r)];
                what_[j] = prev_on[static_cast<std::size_t>(j)] ? wf_[r] / zs : wf_[r];
            }
            i_.resize(k);
            wg_.resize(k);
            for (Eigen::Index j = 0; j < k; ++j) {
                if (prev_on[static_cast<std::size_t>(j)]) { wg_[j] = zfull_[j]; i_[j] = what_[j]; }
                else { i_[j] = zfull_[j]; wg_[j] = what_[j]; }
            }
            bool release = false;
            for (Eigen::Index j = 0; j < k; ++j) {
                const auto& p = ports[static_cast<std::size_t>(j)];
                // held OFF: i = 0, v = V_f − w
                if (held_[static_cast<std::size_t>(j)] && p.V_f - wg_[j] >= p.V_th) {
                    held_[static_cast<std::size_t>(j)] = 0;
                    ++released;
                    release = true;
                }
            }
            if (!release) break;
        }
        const Real tol_w = tol * sc.vscale;
        const Real tol_i = tol * sc.iscale + tol_w * sc.c_max / h;
        cls_.assign(ports.size(), 0);
        for (Eigen::Index j = 0; j < k; ++j) {
            int s;
            if (held_[static_cast<std::size_t>(j)] || wg_[j] > tol_w) s = 0;
            else if (i_[j] > tol_i) s = 1;
            else s = -1;
            cls_[static_cast<std::size_t>(j)] = s;
        }
        return status;
    }

    LemkeSolver lemke_;
    DenseMatrix Z_, H_, Hs_;
    Vector u_, q_, qs_, zf_, wf_, zfull_, what_, i_, wg_;
    std::vector<int> st_, cls_;
    std::vector<char> held_;
    std::vector<Eigen::Index> free_;
};

}  // namespace pulsim::events
