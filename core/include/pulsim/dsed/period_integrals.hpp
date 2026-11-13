#pragma once

// =============================================================================
// Pulsim — one period of the exact engine as a map, with its integrals
// =============================================================================
//
// add-design-optimization-workflow, kernel list item 3. An optimiser
// evaluates a converter by its periodic steady state and a few period
// integrals; doing that in Python held the GIL for 87 % of an
// evaluation. This header post-processes ONE period run of the dsed
// exact stepper (PEDSimulator, events located, no event projection, so
// the state is continuous) into:
//
//   * the period map: x(T), the event log, the masks;
//   * M = dP/dx0 = Π S_k·Φ_k, Φ = e^{A·h} of each step,
//     S = I + (f⁺ − f⁻)·∇gᵀ / (∇g·f⁻) at a diode event, S = I at a gate
//     edge (its instant does not depend on x);
//   * for affine observables y = c(mask)·x̄, x̄ = (x, 1): ∫y, ∫y², ∫ya·yb
//     and the extrema with their instants, in closed form per step.
//
// The closed forms are those of pulsim/sensitivity.py (_kpi, _extrema).
// In energy coordinates the augmented Ab = [[A, b], [0, 0]] is
// As = V·diag(μ)·V⁻¹; with z = V⁻¹(d∘x̄0) and α = ((c/d)·V) ∘ z,
// y(s) = Σ α_i e^{μ_i s} (real), so
//
//   ∫₀ʰ y = Σ α_i·h·φ₁(μ_i h),   ∫₀ʰ ya·yb = Σ_ik αa_i·αb_k·h·φ₁((μ_i+μ_k)h).
//
// No divided difference appears: repeated eigenvalues with a sound basis
// are exact. A near-defective basis (u·κ₂(V) > 1e-10, the propagator's
// tier-1 bound) has no modal form: Φ then comes from Padé, and the
// integrals are DECLINED (`integrals_ok == false`, reason named) — the
// caller computes its KPIs another way, never from an approximation.

#include "pulsim/dsed/event_predictor.hpp"
#include "pulsim/dsed/exact_lti.hpp"
#include "pulsim/dsed/scheduler.hpp"
#include "pulsim/models/switched_diode.hpp"
#include "pulsim/numeric/dense.hpp"
#include "pulsim/numeric/types.hpp"
#include "pulsim/pwl/diode_event_state.hpp"
#include "pulsim/pwl/row_names.hpp"
#include "pulsim/topology/switch_state.hpp"

#include <Eigen/Eigenvalues>
#include <Eigen/SVD>
#include <unsupported/Eigen/MatrixFunctions>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <limits>
#include <numbers>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace pulsim::dsed {

/// y = w·x_full; with `bit >= 0` y = g(bit)·(w·x_full − v_on·[bit on]),
/// g = g_on / g_off (a switch or PWL diode current; v_on = V_f).
struct PeriodObservable {
    std::vector<std::pair<Index, Real>> w;
    int bit = -1;
    Real g_on = Real{1};
    Real g_off = Real{1};
    Real v_on = Real{0};
    bool extrema = false;
};

struct PeriodObservableStats {
    Real int1 = Real{0};   // ∫ y dt over the period
    Real int2 = Real{0};   // ∫ y² dt
    Real peak = -std::numeric_limits<Real>::infinity();
    Real t_peak = Real{0};
    Real valley = std::numeric_limits<Real>::infinity();
    Real t_valley = Real{0};
};

template <class MaskT>
struct PeriodEvent {
    Real t;
    std::string name;
    int type;
    MaskT old_mask;
    MaskT new_mask;
    Vector x;   // the state at the event (continuous: no projection)
};

template <class MaskT>
struct PeriodMapResult {
    Vector x_end;
    DenseMatrix monodromy;      // empty when not asked or on a grazing event
    std::string grazing;        // non-empty: an event tangent to its surface
    std::vector<PeriodEvent<MaskT>> events;
    MaskT first_mask{0};        // mask of the first step (after t = 0 events)
    MaskT last_mask{0};         // mask of the last step (before t = T events)
    Vector scale;               // max |x_i| over the samples
    std::size_t n_steps = 0;
    std::size_t n_exact_steps = 0;
    bool integrals_ok = false;
    std::string integrals_reason;
    std::vector<PeriodObservableStats> stats;
    std::vector<Real> cross;    // ∫ ya·yb per requested pair
    DenseMatrix y_left;         // (n_events, n_obs): under the old mask
    DenseMatrix y_right;        // (n_events, n_obs): under the new mask
    Vector y_start;             // right limit at 0 (first_mask)
    Vector y_end;               // left limit at T (last_mask)
};

/// The diodes of a builder circuit, as the scheduler commutes them.
struct PeriodDiodes {
    topology::SwitchStateMask owned{0};
    std::vector<SchedulerDiode> list;
};

/// The two commutation predicates per PWL diode, the SAME census and
/// predicates run_ped_native_builder registers (python/dsed_bindings.cpp).
/// shortcut: duplicated from that binding (being rewritten by another
/// change); fold the two into this helper once it settles.
template <class AdapterT>
[[nodiscard]] PeriodDiodes add_diode_predicates(const AdapterT& adapter,
                                                EventPredictor& pred) {
    PeriodDiodes out;
    pwl::DiodeEventState census(adapter.graph(), adapter.pool());
    if (census.num_diodes() == 0) {
        return out;
    }
    out.owned = census.diode_owned_bits();
    for (const auto& d : census.entries()) {
        std::string label = pwl::branch_label(adapter.graph(), d.branch_id);
        const AdapterT* ad = &adapter;
        const Index from = d.from;
        const Index to = d.to;
        const auto prm = d.params();
        for (const bool on : {false, true}) {
            const auto type = on ? PredicateType::DiodeOff
                                 : PredicateType::DiodeOn;
            pred.add_predicate(EventPredicate{
                .name = label + (on ? " turns OFF" : " turns ON"),
                .fn = [ad, from, to, prm, on](Real t, const Vector& x) {
                    return models::SwitchedDiode::crossing_signal(
                        on, ad->node_pair_voltage(from, to, t, x), prm);
                },
                .type = type,
                .priority = default_priority(type),
                .aux_index = static_cast<int>(d.switch_idx),
                .required_bit = on ? 1 : 0,
            });
        }
        out.list.push_back(SchedulerDiode{d.switch_idx, d.from, d.to, d.g_on,
                                          d.g_off, d.V_th, d.V_f,
                                          std::move(label)});
    }
    return out;
}

namespace detail {

using Cplx = std::complex<Real>;

/// One mask: its LTI data, observable rows and modal basis.
struct PeriodMode {
    Index n = 0;
    DenseMatrix A;
    Vector b;
    Vector d;            // energy scale of x̄ (n + 1; 1 for the constant)
    DenseMatrix As;      // d·Ab·d⁻¹
    DenseMatrix Rx;      // x_full = Rx·x + rc
    Vector rc;
    bool modal = false;
    std::string reason;  // why no modal form
    Eigen::VectorXcd mu;
    Eigen::MatrixXcd V, Vi;
    DenseMatrix C;       // (n_obs, n + 1) observable rows
    Eigen::MatrixXcd Rm; // (n_obs, n + 1) = (C·d⁻¹)·V
};

inline std::string fmt(Real v) {
    char buf[32];
    std::snprintf(buf, sizeof buf, "%.6g", v);
    return buf;
}

template <class AdapterT, class MaskT>
[[nodiscard]] PeriodMode make_period_mode(
    AdapterT& ad, const MaskT& mask,
    const std::vector<PeriodObservable>& obs) {
    ad.set_mask(mask);
    PeriodMode e;
    e.A = ad.A_matrix();
    e.n = e.A.rows();
    const Index n = e.n;
    const Index n1 = n + 1;
    e.b = ad.b_vector(Real{0});
    e.d = Vector::Ones(n1);
    e.d.head(n) = ad.energy_scale();
    e.Rx = ad.recover_from_state();
    e.rc = ad.recover_const();
    const Vector di = e.d.cwiseInverse();
    e.As = DenseMatrix::Zero(n1, n1);
    e.As.topLeftCorner(n, n) =
        e.d.head(n).asDiagonal() * e.A * di.head(n).asDiagonal();
    e.As.topRightCorner(n, 1) = e.d.head(n).cwiseProduct(e.b);

    // Graded order, largest rows first (sensitivity.py _Mode): with a
    // 5e12 1/s idle-mode row last, QR put a -425 1/s eigenvalue 2.7e-4
    // off (u·||A||).
    std::vector<Index> ord(static_cast<std::size_t>(n1));
    std::iota(ord.begin(), ord.end(), Index{0});
    const Vector rows = e.As.cwiseAbs().rowwise().sum();
    std::stable_sort(ord.begin(), ord.end(),
                     [&](Index a, Index c) { return rows[a] > rows[c]; });
    DenseMatrix Ap(n1, n1);
    for (Index i = 0; i < n1; ++i) {
        for (Index j = 0; j < n1; ++j) {
            Ap(i, j) = e.As(ord[static_cast<std::size_t>(i)],
                            ord[static_cast<std::size_t>(j)]);
        }
    }
    Eigen::EigenSolver<DenseMatrix> es(Ap);
    if (es.info() == Eigen::Success) {
        const Eigen::MatrixXcd Vp = es.eigenvectors();
        e.V.resize(n1, n1);
        for (Index i = 0; i < n1; ++i) {
            e.V.row(ord[static_cast<std::size_t>(i)]) = Vp.row(i);
        }
        e.mu = es.eigenvalues();
        Eigen::JacobiSVD<Eigen::MatrixXcd> svd(e.V);
        const auto& sv = svd.singularValues();
        const Real kappa = sv(n1 - 1) > Real{0}
            ? sv(0) / sv(n1 - 1) : std::numeric_limits<Real>::infinity();
        if (kExactRoundoff * kappa <= kExactTierBound) {
            e.Vi = e.V.fullPivLu().inverse();
            e.modal = true;
        } else {
            e.reason = "kappa2(V) = " + fmt(kappa)
                       + " > 1e-10/u (near-defective eigenbasis, e.g. "
                         "critical damping or a singular A)";
        }
    } else {
        e.reason = "eigensolver failed";
    }

    const auto n_obs = static_cast<Index>(obs.size());
    e.C = DenseMatrix::Zero(n_obs, n1);
    for (Index j = 0; j < n_obs; ++j) {
        const auto& o = obs[static_cast<std::size_t>(j)];
        for (const auto& [idx, coef] : o.w) {
            e.C.row(j).head(n) += coef * e.Rx.row(idx);
            e.C(j, n) += coef * e.rc[idx];
        }
        if (o.bit >= 0) {
            const bool on = mask.get(static_cast<Size>(o.bit));
            const Real g = on ? o.g_on : o.g_off;
            e.C.row(j) *= g;
            if (on) {
                e.C(j, n) -= g * o.v_on;
            }
        }
    }
    if (e.modal && n_obs > 0) {
        e.Rm = (e.C * di.asDiagonal()).cast<Cplx>() * e.V;
    }
    return e;
}

/// e^{A·h} (top-left block of e^{Ab·h}).
[[nodiscard]] inline DenseMatrix period_phi(const PeriodMode& e, Real h) {
    const Index n = e.n;
    DenseMatrix Es;
    if (e.modal) {
        const Eigen::VectorXcd ex = (e.mu * h).array().exp().matrix();
        Es = (e.V * ex.asDiagonal() * e.Vi).real();
    } else {
        Es = (e.As * h).exp();
    }
    const Vector di = e.d.cwiseInverse();
    return di.head(n).asDiagonal() * Es.topLeftCorner(n, n)
           * e.d.head(n).asDiagonal();
}

/// Search grid of one step [0, h] for the stationary points of a sum
/// of exponentials: K uniform cells plus h/2, h/4, ... down to about
/// 1/(2 max|Re mu|), so a fast mode's extremum near the step start is
/// bracketed (a uniform grid of 16 cells missed a 1.707 peak at 92 ns
/// of a 10 us step and reported 0.5). Mirrored by
/// pulsim.sensitivity._extrema_grid; keep the two equal.
inline std::vector<Real> extrema_grid(Real h, long long K, Real re_max) {
    std::vector<Real> g;
    g.reserve(static_cast<std::size_t>(K) + 64);
    for (long long c = 0; c <= K; ++c) {
        g.push_back(h * static_cast<Real>(c) / static_cast<Real>(K));
    }
    if (h * re_max > Real{16}) {
        const int M = std::min(60, static_cast<int>(
            std::ceil(std::log2(h * re_max / Real{2}))));
        for (int m = 1; m <= M; ++m) {
            g.push_back(std::ldexp(h, -m));
        }
    }
    std::sort(g.begin(), g.end());
    g.erase(std::unique(g.begin(), g.end()), g.end());
    return g;
}

inline Vector augment(const Vector& x) {
    Vector xb(x.size() + 1);
    xb.head(x.size()) = x;
    xb[x.size()] = Real{1};
    return xb;
}

}  // namespace detail

/// The period map, its monodromy and the observables' integrals from a
/// run of one period (`res`, from `x0` at t = 0 to `T`, no projection).
/// `diodes` is what add_diode_predicates returned for the run.
template <class AdapterT, class MaskT>
[[nodiscard]] PeriodMapResult<MaskT> period_map_from_run(
    AdapterT& ad, const PEDResult<MaskT>& res, const PeriodDiodes& diodes,
    const std::vector<PeriodObservable>& obs,
    const std::vector<std::pair<int, int>>& pairs, bool want_monodromy) {
    using detail::Cplx;
    using detail::PeriodMode;
    PeriodMapResult<MaskT> out;
    const std::size_t N = res.times.size();
    if (N == 0 || res.step_masks.size() != N) {
        throw std::runtime_error("period map: the run stored no samples");
    }
    const Real T = res.times.back() - res.times.front();
    const auto n_obs = static_cast<Index>(obs.size());
    out.n_steps = res.n_accept;
    out.n_exact_steps = res.n_exact_steps;
    out.stats.assign(obs.size(), PeriodObservableStats{});
    out.cross.assign(pairs.size(), Real{0});

    std::unordered_map<MaskT, PeriodMode> modes;
    auto mode_of = [&](const MaskT& m) -> const PeriodMode& {
        auto it = modes.find(m);
        if (it == modes.end()) {
            it = modes.emplace(m, detail::make_period_mode(ad, m, obs)).first;
        }
        return it->second;
    };

    // Left limit at sample k: the pre-event state where an event fired.
    std::vector<const Vector*> left(N);
    for (std::size_t k = 0; k < N; ++k) {
        left[k] = &res.states[k];
    }
    for (std::size_t j = 0;
         j < res.pre_event_index.size() && j < res.pre_event_states.size();
         ++j) {
        left[res.pre_event_index[j]] = &res.pre_event_states[j];
    }
    const Index n = res.states.front().size();
    out.scale = Vector::Zero(n);
    for (std::size_t k = 0; k < N; ++k) {
        out.scale = out.scale.cwiseMax(res.states[k].cwiseAbs());
        out.scale = out.scale.cwiseMax(left[k]->cwiseAbs());
    }
    out.x_end = res.states.back();
    out.first_mask = res.step_masks[N > 1 ? 1 : 0];
    out.last_mask = res.step_masks[N - 1];

    out.integrals_ok = n_obs > 0;
    if (out.integrals_ok && res.n_exact_steps < res.n_accept) {
        out.integrals_ok = false;
        out.integrals_reason = std::to_string(res.n_accept - res.n_exact_steps)
            + " step(s) of the run were numeric, not the closed form of "
              "their mode";
    }
    auto decline = [&](std::string why) {
        if (out.integrals_ok) {
            out.integrals_ok = false;
            out.integrals_reason = std::move(why);
        }
    };
    auto y_of = [&](const PeriodMode& e, const Vector& x) -> Vector {
        return n_obs > 0 ? Vector(e.C * detail::augment(x)) : Vector{};
    };

    DenseMatrix J;
    if (want_monodromy) {
        J = DenseMatrix::Identity(n, n);
    }
    std::vector<std::vector<Real>> yl, yr;
    std::size_t ev = 0;
    const Real t_tol = Real{4} * kExactRoundoff
                       * std::max(std::abs(res.times.back()), T);

    for (std::size_t k = 0; k < N; ++k) {
        const Real tk = res.times[k];
        // ---- the events that fired at this sample, in log order ----
        while (ev < res.event_log.size()
               && res.event_log[ev].t <= tk + t_tol) {
            const auto& er = res.event_log[ev++];
            const Vector& x = *left[k];
            const PeriodMode& eo = mode_of(er.old_mask);
            const PeriodMode& en = mode_of(er.new_mask);
            out.events.push_back(PeriodEvent<MaskT>{
                er.t, er.name, static_cast<int>(er.type), er.old_mask,
                er.new_mask, x});
            if (n_obs > 0) {
                const Vector a = y_of(eo, x);
                const Vector c = y_of(en, x);
                yl.emplace_back(a.data(), a.data() + a.size());
                yr.emplace_back(c.data(), c.data() + c.size());
            }
            const bool diode_event = er.type == PredicateType::DiodeOn
                                     || er.type == PredicateType::DiodeOff;
            if (!diode_event || J.size() == 0) {
                continue;
            }
            // ∇g: the branch-voltage row, under the old mask, of the diode
            // whose predicate fired (by its name: the logged new_mask is
            // the mask after the zero-time settle, which may flip other
            // diodes too, or revert this one)
            const SchedulerDiode* dd = nullptr;
            for (const auto& cand : diodes.list) {
                if (er.name == cand.label + " turns ON"
                    || er.name == cand.label + " turns OFF") {
                    dd = &cand;
                    break;
                }
            }
            if (dd == nullptr) {
                throw std::runtime_error(
                    "period map: diode event '" + er.name
                    + "' names no diode of the circuit");
            }
            if (er.old_mask == er.new_mask) {
                continue;   // the settle kept the mode: f+ = f-, S = I
            }
            Vector grad = Vector::Zero(n);
            if (dd->from >= 0) grad += eo.Rx.row(dd->from).transpose();
            if (dd->to >= 0) grad -= eo.Rx.row(dd->to).transpose();
            const Vector fm = eo.A * x + eo.b;
            const Vector fp = en.A * x + en.b;
            const Real den = grad.dot(fm);
            if (std::abs(den) <= Real{1e-12} * grad.norm() * fm.norm()) {
                out.grazing = "grazing event: " + er.name + " at t = "
                    + detail::fmt(er.t / T) + " T is tangent to its "
                      "switching surface (grad(g).f = 0), where the "
                      "period map is not differentiable.";
                J.resize(0, 0);
                continue;
            }
            J += ((fp - fm) / den) * (grad.transpose() * J);
        }
        if (k + 1 == N) {
            break;
        }
        // ---- the step (t_k, t_{k+1}] under its stamped mask ----
        const Real h = res.times[k + 1] - tk;
        const PeriodMode& e = mode_of(res.step_masks[k + 1]);
        if (J.size() != 0) {
            J = detail::period_phi(e, h) * J;
        }
        if (!out.integrals_ok || !(h > Real{0})) {
            continue;
        }
        if (!e.modal) {
            decline("no modal form for a mode of the period: " + e.reason);
            continue;
        }
        const Vector xb = detail::augment(res.states[k]);
        const Eigen::VectorXcd z =
            e.Vi * e.d.cwiseProduct(xb).template cast<Cplx>();
        // the closed form must reproduce the engine's step end
        {
            const Eigen::VectorXcd ex = (e.mu * h).array().exp().matrix();
            const Vector xe = (e.V * ex.cwiseProduct(z)).real()
                                  .cwiseQuotient(e.d).head(n);
            const Vector& xr = *left[k + 1];
            for (Index i = 0; i < n; ++i) {
                if (std::abs(xe[i] - xr[i])
                    > Real{1e-9} * (out.scale[i] + std::abs(xr[i]))
                          + Real{1e-300}) {
                    decline("the closed form of the step at t = "
                            + detail::fmt(tk) + " misses the engine's "
                              "state by " + detail::fmt(std::abs(xe[i] - xr[i]))
                            + " (state " + std::to_string(i) + ")");
                    break;
                }
            }
            if (!out.integrals_ok) {
                continue;
            }
        }
        const Index n1 = n + 1;
        Eigen::VectorXcd p1(n1);
        for (Index i = 0; i < n1; ++i) {
            p1[i] = h * phi1(e.mu[i] * h);
        }
        Eigen::MatrixXcd P2(n1, n1);
        for (Index i = 0; i < n1; ++i) {
            for (Index j = i; j < n1; ++j) {
                P2(i, j) = P2(j, i) = h * phi1((e.mu[i] + e.mu[j]) * h);
            }
        }
        Eigen::MatrixXcd alpha(n1, n_obs);
        for (Index j = 0; j < n_obs; ++j) {
            alpha.col(j) = e.Rm.row(j).transpose().cwiseProduct(z);
            auto& st = out.stats[static_cast<std::size_t>(j)];
            st.int1 += (alpha.col(j).transpose() * p1)(0).real();
            st.int2 += (alpha.col(j).transpose() * P2 * alpha.col(j))(0).real();
        }
        for (std::size_t q = 0; q < pairs.size(); ++q) {
            out.cross[q] += (alpha.col(pairs[q].first).transpose() * P2
                             * alpha.col(pairs[q].second))(0).real();
        }
        // ---- extrema: ends, then stationary points of y ----
        Real wmax = Real{0};
        Real rmax = Real{0};
        for (Index i = 0; i < n1; ++i) {
            wmax = std::max(wmax, std::abs(e.mu[i].imag()));
            rmax = std::max(rmax, std::abs(e.mu[i].real()));
        }
        const auto K = static_cast<long long>(std::max(
            Real{16}, std::ceil(h * wmax * Real{8}
                                / (Real{2} * std::numbers::pi_v<Real>))));
        const Vector y0 = y_of(e, res.states[k]);
        const Vector y1 = y_of(e, *left[k + 1]);
        std::vector<Real> grid;   // built on the first observable searched
        for (Index j = 0; j < n_obs; ++j) {
            if (!obs[static_cast<std::size_t>(j)].extrema) {
                continue;
            }
            auto& st = out.stats[static_cast<std::size_t>(j)];
            auto cand = [&](Real y, Real t) {
                if (y > st.peak) { st.peak = y; st.t_peak = t; }
                if (y < st.valley) { st.valley = y; st.t_valley = t; }
            };
            cand(y0[j], tk);
            cand(y1[j], res.times[k + 1]);
            if (K > (1LL << 16)) {
                decline("an observable rings " + std::to_string(K / 8)
                        + " times in one step: its extrema are not searched");
                break;
            }
            const Eigen::VectorXcd a = alpha.col(j);
            auto dy = [&](Real s) {
                Real v = Real{0};
                for (Index i = 0; i < n1; ++i) {
                    v += (a[i] * e.mu[i] * std::exp(e.mu[i] * s)).real();
                }
                return v;
            };
            auto yv = [&](Real s) {
                Real v = Real{0};
                for (Index i = 0; i < n1; ++i) {
                    v += (a[i] * std::exp(e.mu[i] * s)).real();
                }
                return v;
            };
            if (grid.empty()) {
                grid = detail::extrema_grid(h, K, rmax);
            }
            Real s_lo = grid.front();
            Real d_lo = dy(s_lo);
            for (std::size_t c = 1; c < grid.size(); ++c) {
                const Real s_hi = grid[c];
                const Real d_hi = dy(s_hi);
                if ((d_lo > 0 && d_hi < 0) || (d_lo < 0 && d_hi > 0)) {
                    // bisection on y' to float resolution
                    Real lo = s_lo, hi = s_hi;
                    const bool up = d_lo > 0;
                    for (int it = 0; it < 200; ++it) {
                        const Real mid = Real{0.5} * (lo + hi);
                        if (!(mid > lo && mid < hi)) break;
                        if ((dy(mid) > 0) == up) lo = mid; else hi = mid;
                    }
                    const Real s = Real{0.5} * (lo + hi);
                    cand(yv(s), tk + s);
                }
                s_lo = s_hi;
                d_lo = d_hi;
            }
        }
    }
    if (ev != res.event_log.size()) {
        throw std::runtime_error(
            "period map: " + std::to_string(res.event_log.size() - ev)
            + " logged event(s) match no stored sample");
    }
    if (J.size() != 0) {
        out.monodromy = std::move(J);
    }
    if (n_obs > 0) {
        const PeriodMode& e0 = mode_of(out.first_mask);
        out.y_start = y_of(e0, res.states.front());
        const PeriodMode& e1 = mode_of(out.last_mask);
        out.y_end = y_of(e1, *left[N - 1]);
        out.y_left.resize(static_cast<Index>(yl.size()), n_obs);
        out.y_right.resize(static_cast<Index>(yr.size()), n_obs);
        for (std::size_t i = 0; i < yl.size(); ++i) {
            for (Index j = 0; j < n_obs; ++j) {
                out.y_left(static_cast<Index>(i), j) =
                    yl[i][static_cast<std::size_t>(j)];
                out.y_right(static_cast<Index>(i), j) =
                    yr[i][static_cast<std::size_t>(j)];
            }
        }
    }
    if (!out.integrals_ok && out.integrals_reason.empty() && n_obs > 0) {
        out.integrals_reason = "declined";
    }
    return out;
}

}  // namespace pulsim::dsed
