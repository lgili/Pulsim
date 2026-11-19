#pragma once

// =============================================================================
// Pulsim — registration checks of an external device (the contract, enforced)
// =============================================================================
//
// Run once per instance before a circuit accepts it. Each refusal names the
// device, the model, the property and the numbers (xdev::ContractError).
//
//   always: buffers written within their declared sizes (canaries); finite
//           outputs; eval pure (bit-identical on a repeat); KCL (pin f and
//           pin q sum to zero); declared Jacobian = central differences at
//           17 probe points; the Jacobian is the derivative BETWEEN probes
//           too (secant against the mean tangent along a 2 001-point line
//           per unknown, plus a geometric run to ±1e-15 of the box around
//           0); model state written within n_state and finite.
//   claims: SMOOTH — no Jacobian jump along the lines; PASSIVE — Σ f_p·v_p
//           ≥ 0 at every probe and line point; RECIPROCAL — dq/dx symmetric
//           in the pin block and in the internal block; monotone pairs —
//           f[row] non-decreasing in x[col] along the line, to rounding.
//
// What it cannot do: certify the ~1e-12 smoothness a Newton tolerance
// needs, or anything outside the declared probe box.

#include "pulsim/xdev/host.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

namespace pulsim::xdev {

namespace detail_checks {
inline constexpr std::size_t kCanary = 4;
// a NaN with a payload no arithmetic produces
inline double canary() {
    const std::uint64_t bits = 0x7ff4dead5eedbeefULL;
    double d;
    std::memcpy(&d, &bits, sizeof d);
    return d;
}
inline bool is_canary(double v) {
    std::uint64_t b;
    std::memcpy(&b, &v, sizeof b);
    return b == 0x7ff4dead5eedbeefULL;
}
}  // namespace detail_checks

inline void check_contract(const Instance& in) {
    namespace dc = detail_checks;
    const auto& d = in.model().d();
    const std::size_t R = in.rows(), NP = d.n_pins, NS = d.n_state;
    const auto num = &Instance::num;
    const std::vector<Real> state0(NS, Real{0});
    // every eval below also proves x, p and state come back unwritten
    struct Guard {
        const Instance& in;
        explicit Guard(const Instance& i) : in(i) { in.guard_inputs_ = true; }
        ~Guard() { in.guard_inputs_ = false; }
    } const guard(in);

    // ---- guarded evaluation: sizes and finiteness --------------------------
    auto guarded = [&](const std::vector<Real>& x, EvalOut& o) {
        std::vector<Real> f(R + dc::kCanary, dc::canary()), q(R + dc::kCanary, dc::canary()),
            df(d.nnz_f + dc::kCanary, dc::canary()), dq(d.nnz_q + dc::kCanary, dc::canary());
        const pulsim_xdev_ctx c = in.ctx_(state0.data());
        if (in.call_(c, x.data(), f.data(), q.data(), df.data(), dq.data()) != PULSIM_XDEV_OK) {
            in.refuse_("eval returned an error status at the probe x = " + Instance::vec(x.data(), R));
        }
        auto tail = [&](const char* w, const std::vector<Real>& v, std::size_t n) {
            for (std::size_t k = n; k < v.size(); ++k)
                if (!dc::is_canary(v[k])) {
                    in.refuse_(std::string("eval wrote past the end of ") + w + " (declared size "
                               + std::to_string(n) + ")");
                }
            for (std::size_t k = 0; k < n; ++k)
                if (dc::is_canary(v[k])) {
                    in.refuse_(std::string("eval left ") + w + "[" + std::to_string(k)
                               + "] unwritten (declared size " + std::to_string(n) + ")");
                }
        };
        tail("f", f, R);
        tail("q", q, R);
        tail("df", df, d.nnz_f);
        tail("dq", dq, d.nnz_q);
        in.size(o);
        in.eval(x.data(), state0.data(), o);  // finiteness (one comparison) + names
        auto same = [&](const char* w, const std::vector<Real>& a, const std::vector<Real>& b) {
            if (std::memcmp(a.data(), b.data(), b.size() * sizeof(Real)) != 0) {
                in.refuse_(std::string("eval is not pure: two calls at x = ")
                           + Instance::vec(x.data(), R) + " returned different " + w
                           + " (hidden state or uninitialised memory)");
            }
        };
        same("f", f, o.f);
        same("q", q, o.q);
        same("df", df, o.df);
        same("dq", dq, o.dq);
    };

    auto kcl = [&](const std::vector<Real>& x, const EvalOut& o) {
        for (const auto* w : {"f", "q"}) {
            const auto& v = w[0] == 'f' ? o.f : o.q;
            Real s = 0, a = 0;
            for (std::size_t p = 0; p < NP; ++p) {
                s += v[p];
                a += std::abs(v[p]);
            }
            if (!(std::abs(s) <= Real{1e-12} * a + Real{1e-300})) {
                in.refuse_(std::string("the pin rows' ") + w + " sum to " + num(s)
                           + ", not 0, at x = " + Instance::vec(x.data(), R)
                           + ": the device creates or destroys charge (KCL)");
            }
        }
    };
    auto passive = [&](const std::vector<Real>& x, const EvalOut& o) {
        if (!(d.claims & PULSIM_XDEV_CLAIM_PASSIVE)) return;
        Real p = 0, a = 0;
        for (std::size_t k = 0; k < NP; ++k) {
            p += o.f[k] * x[k];
            a += std::abs(o.f[k] * x[k]);
        }
        if (!(p >= -Real{1e-12} * a - Real{1e-300})) {
            in.refuse_("claims PASSIVE but absorbs " + num(p) + " W (< 0) at x = "
                       + Instance::vec(x.data(), R));
        }
    };
    auto reciprocal = [&](const std::vector<Real>& x, const EvalOut& o) {
        if (!(d.claims & PULSIM_XDEV_CLAIM_RECIPROCAL)) return;
        const auto J = in.dense_(o.dq, d.q_rows, d.q_cols, d.nnz_q);
        Real s = 0;
        for (Real v : J) s = std::max(s, std::abs(v));
        for (std::size_t i = 0; i < R; ++i)
            for (std::size_t j = i + 1; j < R; ++j) {
                if ((i < NP) != (j < NP)) continue;  // pin and internal blocks
                if (!(std::abs(J[i * R + j] - J[j * R + i]) <= Real{1e-9} * s)) {
                    in.refuse_("claims RECIPROCAL but dq[" + std::to_string(i) + "]/dx["
                               + std::to_string(j) + "] = " + num(J[i * R + j]) + " while dq["
                               + std::to_string(j) + "]/dx[" + std::to_string(i) + "] = "
                               + num(J[j * R + i]) + " at x = " + Instance::vec(x.data(), R));
                }
            }
    };

    // ---- probe points: the centre and 16 points of a fixed sequence --------
    std::vector<Real> centre(R);
    for (std::size_t k = 0; k < R; ++k) centre[k] = (d.probe_lo[k] + d.probe_hi[k]) / 2;
    std::uint64_t s = 0x9e3779b97f4a7c15ULL;  // fixed: the check is reproducible
    auto uniform = [&s]() {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        return static_cast<Real>(s >> 11) * (Real{1} / static_cast<Real>(1ULL << 53));
    };
    EvalOut o;
    std::vector<std::vector<Real>> probes;
    for (int k = 0; k < 17; ++k) {
        std::vector<Real> x(R);
        for (std::size_t c = 0; c < R; ++c) {
            x[c] = k == 0 ? centre[c]
                          : d.probe_lo[c] + uniform() * (d.probe_hi[c] - d.probe_lo[c]);
        }
        guarded(x, o);
        kcl(x, o);
        passive(x, o);
        reciprocal(x, o);
        probes.push_back(std::move(x));
    }

    // ---- lines: the Jacobian between probes, smoothness, monotonicity -----
    for (std::size_t c = 0; c < R; ++c) {
        const Real lo = d.probe_lo[c], hi = d.probe_hi[c];
        std::vector<Real> pts;
        for (int k = 0; k <= 2000; ++k) pts.push_back(lo + (hi - lo) * k / 2000);
        if (lo < 0 && hi > 0) {
            const Real w = std::max(-lo, hi);
            for (int e = 0; e <= 75; ++e) {   // ±w·10^(−e/5) down to ±1e-15·w
                const Real v = w * std::pow(Real{10}, -static_cast<Real>(e) / 5);
                if (v < hi) pts.push_back(v);
                if (-v > lo) pts.push_back(-v);
            }
            pts.push_back(0);
        }
        std::sort(pts.begin(), pts.end());
        pts.erase(std::unique(pts.begin(), pts.end()), pts.end());

        std::vector<Real> x = centre;
        EvalOut a, b;
        in.size(a);
        in.size(b);
        x[c] = pts[0];
        in.eval(x.data(), state0.data(), a);
        auto Ja = in.dense_(a.df, d.f_rows, d.f_cols, d.nnz_f);
        auto Qa = in.dense_(a.dq, d.q_rows, d.q_cols, d.nnz_q);
        // scale of each row's derivative in x[c] over the whole line
        std::vector<Real> fs(R, 0), qs(R, 0);
        for (std::size_t k = 1; k < pts.size(); ++k) {
            const Real xa = pts[k - 1], xb = pts[k];
            x[c] = xb;
            in.eval(x.data(), state0.data(), b);
            passive(x, b);
            const auto Jb = in.dense_(b.df, d.f_rows, d.f_cols, d.nnz_f);
            const auto Qb = in.dense_(b.dq, d.q_rows, d.q_cols, d.nnz_q);
            for (std::size_t r = 0; r < R; ++r) {
                fs[r] = std::max({fs[r], std::abs(Ja[r * R + c]), std::abs(Jb[r * R + c])});
                qs[r] = std::max({qs[r], std::abs(Qa[r * R + c]), std::abs(Qb[r * R + c])});
            }
            for (std::uint32_t m = 0; m < d.n_monotone; ++m) {
                const auto& mc = d.monotone[m];
                if (mc.col != c) continue;
                const Real tol = 4 * std::numeric_limits<Real>::epsilon()
                                 * (std::abs(a.f[mc.row]) + std::abs(b.f[mc.row]));
                if (!(b.f[mc.row] - a.f[mc.row] >= -tol) || !(Jb[mc.row * R + c] >= 0)) {
                    in.refuse_("claims f[" + std::to_string(mc.row) + "] non-decreasing in x["
                               + std::to_string(c) + "] but f goes " + num(a.f[mc.row]) + " -> "
                               + num(b.f[mc.row]) + " (slope " + num(Jb[mc.row * R + c])
                               + ") between x[" + std::to_string(c) + "] = " + num(xa) + " and "
                               + num(xb));
                }
            }
            a = b;
            Ja = Jb;
            Qa = Qb;
        }
        // second pass with the line scales known: secant against tangents
        x[c] = pts[0];
        in.eval(x.data(), state0.data(), a);
        Ja = in.dense_(a.df, d.f_rows, d.f_cols, d.nnz_f);
        Qa = in.dense_(a.dq, d.q_rows, d.q_cols, d.nnz_q);
        for (std::size_t k = 1; k < pts.size(); ++k) {
            const Real xa = pts[k - 1], xb = pts[k];
            x[c] = xb;
            in.eval(x.data(), state0.data(), b);
            const auto Jb = in.dense_(b.df, d.f_rows, d.f_cols, d.nnz_f);
            const auto Qb = in.dense_(b.dq, d.q_rows, d.q_cols, d.nnz_q);
            for (std::size_t r = 0; r < R; ++r) {
                for (int w = 0; w < 2; ++w) {
                    const Real va = w ? a.q[r] : a.f[r], vb = w ? b.q[r] : b.f[r];
                    const Real ta = w ? Qa[r * R + c] : Ja[r * R + c];
                    const Real tb = w ? Qb[r * R + c] : Jb[r * R + c];
                    const Real sc = w ? qs[r] : fs[r];
                    const char* nm = w ? "q" : "f";
                    const bool jump = std::abs(tb - ta) > Real{0.05} * sc;
                    if (jump && (d.claims & PULSIM_XDEV_CLAIM_SMOOTH)) {
                        in.refuse_(std::string("claims SMOOTH but d") + nm + "[" + std::to_string(r)
                                   + "]/dx[" + std::to_string(c) + "] jumps " + num(ta) + " -> "
                                   + num(tb) + " between x[" + std::to_string(c) + "] = "
                                   + num(xa) + " and " + num(xb) + " (not C1)");
                    }
                    if (jump) continue;  // a declared kink: the secant says nothing
                    const Real secant = (vb - va) / (xb - xa);
                    // rounding of the secant: f's size over the interval width
                    const Real round = 8 * std::numeric_limits<Real>::epsilon()
                                       * std::max(std::abs(va), std::abs(vb)) / (xb - xa);
                    if (!(std::abs(secant - (ta + tb) / 2) <= Real{1e-3} * sc + round + Real{1e-300})) {
                        in.refuse_(std::string("between the probes, d") + nm + "["
                                   + std::to_string(r) + "]/dx[" + std::to_string(c)
                                   + "] is not the derivative of " + nm + ": secant " + num(secant)
                                   + " against declared " + num(ta) + " .. " + num(tb)
                                   + " on x[" + std::to_string(c) + "] in [" + num(xa) + ", "
                                   + num(xb) + "]"
                                   + (ta == 0 && tb == 0 ? "; the entry may be missing from the declared pattern" : ""));
                    }
                }
            }
            a = b;
            Ja = Jb;
            Qa = Qb;
        }
    }

    // ---- off-axis lines: two diagonals and four fixed-seed directions -----
    // The axis lines all pass through the centre's other coordinates; a
    // Jacobian wrong only where several unknowns move together is caught
    // here: the directional derivative J·w against the secant of f and q
    // along x(τ) = centre + τ·w, τ in [-1, 1] (inside the box), and SMOOTH
    // against jumps of J·w.
    if (R > 1) {
        std::vector<std::vector<Real>> dirs(2, std::vector<Real>(R));
        for (std::size_t k = 0; k < R; ++k) {
            dirs[0][k] = 1;
            dirs[1][k] = k % 2 ? -1 : 1;
        }
        for (int n = 0; n < 4; ++n) {
            std::vector<Real> u(R);
            for (std::size_t k = 0; k < R; ++k) u[k] = 2 * uniform() - 1;
            dirs.push_back(std::move(u));
        }
        std::vector<Real> F, Q, JF, JQ, w(R), x(R), tau;
        for (std::size_t n = 0; n < dirs.size(); ++n) {
            // as fine as the axis lines: a sum of unknowns (v_a - v_b along
            // [1, -1]) moves no faster per point than one unknown on its axis
            Real l1 = 0;
            for (Real v : dirs[n]) l1 += std::abs(v);
            const int kN = static_cast<int>(std::ceil(2000 * std::max(l1, Real{1})));
            F.assign((kN + 1) * R, 0);
            Q.assign((kN + 1) * R, 0);
            tau.assign(kN + 1, 0);
            for (std::size_t k = 0; k < R; ++k) w[k] = (d.probe_hi[k] - d.probe_lo[k]) / 2 * dirs[n][k];
            JF.assign((kN + 1) * R, 0);
            JQ.assign((kN + 1) * R, 0);
            for (int i = 0; i <= kN; ++i) {
                tau[i] = Real{-1} + Real{2} * i / kN;
                for (std::size_t k = 0; k < R; ++k) x[k] = centre[k] + tau[i] * w[k];
                in.eval(x.data(), state0.data(), o);
                for (std::size_t r = 0; r < R; ++r) {
                    F[i * R + r] = o.f[r];
                    Q[i * R + r] = o.q[r];
                }
                for (std::uint32_t k = 0; k < d.nnz_f; ++k) JF[i * R + d.f_rows[k]] += o.df[k] * w[d.f_cols[k]];
                for (std::uint32_t k = 0; k < d.nnz_q; ++k) JQ[i * R + d.q_rows[k]] += o.dq[k] * w[d.q_cols[k]];
            }
            for (std::size_t r = 0; r < R; ++r) {
                for (int wq = 0; wq < 2; ++wq) {
                    const auto& V = wq ? Q : F;
                    const auto& T = wq ? JQ : JF;
                    const char* nm = wq ? "q" : "f";
                    Real sc = 0;
                    for (int i = 0; i <= kN; ++i) sc = std::max(sc, std::abs(T[i * R + r]));
                    for (int i = 1; i <= kN; ++i) {
                        const Real ta = T[(i - 1) * R + r], tb = T[i * R + r];
                        const Real va = V[(i - 1) * R + r], vb = V[i * R + r];
                        const bool jump = std::abs(tb - ta) > Real{0.05} * sc;
                        if (jump && (d.claims & PULSIM_XDEV_CLAIM_SMOOTH)) {
                            in.refuse_(std::string("claims SMOOTH but the derivative of ") + nm + "["
                                       + std::to_string(r) + "] along direction "
                                       + Instance::vec(w.data(), R) + " jumps " + num(ta) + " -> "
                                       + num(tb) + " at tau = " + num(tau[i]) + " (not C1)");
                        }
                        if (jump) continue;
                        const Real dt = tau[i] - tau[i - 1];
                        const Real secant = (vb - va) / dt;
                        const Real round = 8 * std::numeric_limits<Real>::epsilon()
                                           * std::max(std::abs(va), std::abs(vb)) / dt;
                        // mean value theorem: a C1 law's secant is a tangent inside the
                        // interval, so it lies between ta and tb (where f' is monotone
                        // over one step); the midpoint rule alone is O(step) wrong at a
                        // C1-not-C2 joint (the Shockley tangent continuation at v_lim)
                        const Real slack = Real{1e-3} * sc + round + Real{1e-300};
                        if (!(secant >= std::min(ta, tb) - slack && secant <= std::max(ta, tb) + slack)) {
                            in.refuse_(std::string("off the axes, the declared Jacobian is not the "
                                                   "derivative of ") + nm + "[" + std::to_string(r)
                                       + "]: along x = centre + tau*" + Instance::vec(w.data(), R)
                                       + " the secant is " + num(secant) + " against J*w = " + num(ta)
                                       + " .. " + num(tb) + " on tau in [" + num(tau[i - 1]) + ", "
                                       + num(tau[i]) + "]");
                        }
                    }
                }
            }
        }
    }

    // ---- the declared Jacobian at the probes (after the lines, so a claim
    // the lines refute is named before the Jacobian it breaks) -------------
    for (const auto& x : probes) {
        in.eval(x.data(), state0.data(), o);
        in.check_jacobian_at(x.data(), state0.data(), o, "registration check");
    }

    // ---- model state: written within n_state, finite -----------------------
    if (d.accept) {
        std::vector<Real> st(NS + dc::kCanary, dc::canary());
        for (std::size_t k = 0; k < NS; ++k) st[k] = 0;
        const pulsim_xdev_ctx c = in.ctx_(st.data());
        const std::vector<Real> x0 = centre, p0 = in.params();
        if (d.init && d.init(&c, centre.data(), st.data()) != PULSIM_XDEV_OK) {
            in.refuse_("init returned an error status at the probe centre");
        }
        if (d.accept(&c, centre.data(), Real{1e-6}, st.data()) != PULSIM_XDEV_OK) {
            in.refuse_("accept returned an error status at the probe centre");
        }
        if (centre != x0 || in.params() != p0) {
            in.refuse_("init/accept wrote through a const input (x or ctx->p)");
        }
        for (std::size_t k = 0; k < st.size(); ++k) {
            if (k >= NS && !dc::is_canary(st[k])) {
                in.refuse_("init/accept wrote past the declared n_state = " + std::to_string(NS));
            }
            if (k < NS && !std::isfinite(st[k])) {
                in.refuse_("accept left state[" + std::to_string(k) + "] = " + num(st[k]));
            }
        }
        // the Jacobian with a state the model wrote itself (not only zeros)
        for (const auto& x : probes) {
            in.eval(x.data(), st.data(), o);
            in.check_jacobian_at(x.data(), st.data(), o, "registration check (accepted state)");
        }
    }
}

}  // namespace pulsim::xdev
