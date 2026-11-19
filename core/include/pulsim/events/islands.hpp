#pragma once

// =============================================================================
// events/islands.hpp — floating islands of a diode port problem and the exact
// imposition of their KCL sums (add-complementarity-mode-selection DECISIONS
// #30 on dsed, F1 on pwl). Engine-neutral: the caller supplies the solves.
//
// An island is a set of nodes joined by strong elements (capacitors,
// inductors, sources, ON switches / diodes, resistors above `weak_g`) and
// tied to the rest only through weak conductances (OFF switches / diodes,
// resistors at or below `weak_g`). A factor rounds such an island's common
// mode to ε·(internal current)/G_boundary. Summed over the island's rows,
// every internal stamp (C/h included) cancels; the sum `g·s` built from the
// boundary branches alone is imposed exactly after each solve J·s = f:
//
//     σ = g·s − Σ_S f,   s −= w·σ/(g·w),   w = J⁻¹·e_ref
//
// Several islands are corrected jointly, K·α = σ with K(m, n) = g_m·w_n: K is
// the identity in exact arithmetic (disjoint islands), but behind a 1e7 Ω tie
// its off-diagonal rounding left two buses joined by 1 MΩ 6.5e-9 V off when
// corrected one after another; jointly ≤ 1.1e-14 V against a double-double
// reference (DECISIONS F1). A failed joint factor falls back to the
// sequential correction.
//
// No islands (empty list) when the circuit holds a device kind the rule does
// not classify (controlled sources, transformers, Newton devices).
// =============================================================================

#include "pulsim/numeric/dense.hpp"
#include "pulsim/numeric/types.hpp"
#include "pulsim/pwl/device_pool.hpp"
#include "pulsim/topology/graph.hpp"

#include <algorithm>
#include <cmath>
#include <unordered_map>
#include <utility>
#include <vector>

#include <Eigen/Dense>

namespace pulsim::events {

struct Island {
    Index ref = -1;                          ///< the correction's injection row (an island node)
    std::vector<std::pair<Index, Real>> g;   ///< (column, Σ boundary conductance), sorted
    std::vector<char> in;                    ///< in[node]: the node belongs to the island
    std::vector<Index> nodes;                ///< the island's nodes, ascending
};

/// The floating islands under `mask` (one bit per switch-kind branch, in
/// branch order). `weak_g` (S): resistors at or below it do not join an
/// island — 1e-3 on pwl (DECISIONS F1), 1e-6 on dsed (#30).
template <class Mask>
[[nodiscard]] std::vector<Island> find_islands(const topology::Graph& graph,
                                               const pwl::DevicePool& pool,
                                               const Mask& mask, Real weak_g) {
    std::vector<Island> out;
    const Index nn = graph.num_nodes();
    std::vector<Index> up(static_cast<std::size_t>(nn + 1));   // union-find; nn = ground
    for (Index i = 0; i <= nn; ++i) up[static_cast<std::size_t>(i)] = i;
    auto root = [&](Index i) {
        Index r = i < 0 ? nn : i;
        while (up[static_cast<std::size_t>(r)] != r) r = up[static_cast<std::size_t>(r)];
        return r;
    };
    auto join = [&](Index a, Index b) { up[static_cast<std::size_t>(root(a))] = root(b); };
    struct Weak { Index a, b; Real g; };
    std::vector<Weak> weak;
    using K = pwl::DevicePool::StoredKind;
    std::size_t sw = 0;
    for (Index b = 0; b < graph.num_branches(); ++b) {
        const auto& br = graph.branch(b);
        if (!pool.is_registered(b)) return out;
        const K kind = pool.kind_of(b);
        if (br.kind == topology::BranchKind::Switch) {
            const bool on = mask.get(sw++);
            if (kind != K::Diode && kind != K::Switch) return out;
            if (on) join(br.from, br.to);
            else weak.push_back({br.from, br.to, kind == K::Diode
                ? pool.diode_params(b).g_off : pool.switch_g_off(b)});
            continue;
        }
        switch (kind) {
        case K::Resistor: {
            const Real g = pool.resistor_params(b).G;
            if (std::abs(g) <= weak_g) weak.push_back({br.from, br.to, g});
            else join(br.from, br.to);
            break;
        }
        case K::Capacitor: case K::Inductor: case K::VoltageSource:
        case K::PWMVoltageSource: case K::SineVoltageSource: case K::PulseVoltageSource:
        case K::CurrentSource: case K::SineCurrentSource: case K::PulseCurrentSource:
        case K::PwlCurrentSource:
            // strong, or a source: no source sits on an island's boundary,
            // so its terms cancel in the KCL sum too
            join(br.from, br.to);
            break;
        default:
            return out;
        }
    }
    std::vector<Index> rt(static_cast<std::size_t>(nn));
    std::unordered_map<Index, Index> size;
    for (Index i = 0; i < nn; ++i) ++size[rt[static_cast<std::size_t>(i)] = root(i)];
    const Index gr = root(-1);
    std::unordered_map<Index, std::size_t> slot;
    for (Index i = 0; i < nn; ++i) {
        const Index r = rt[static_cast<std::size_t>(i)];
        if (r == gr || size[r] < 2) continue;   // grounded, or nothing internal to cancel
        auto [s, fresh] = slot.try_emplace(r, out.size());
        if (fresh) {
            out.emplace_back();
            out.back().ref = i;
            out.back().in.assign(static_cast<std::size_t>(nn), 0);
        }
        out[s->second].in[static_cast<std::size_t>(i)] = 1;
        out[s->second].nodes.push_back(i);
    }
    for (auto& isl : out) {
        std::unordered_map<Index, Real> col;
        for (const auto& w : weak) {
            const bool ia = w.a >= 0 && isl.in[static_cast<std::size_t>(w.a)];
            const bool ib = w.b >= 0 && isl.in[static_cast<std::size_t>(w.b)];
            if (ia == ib) continue;   // internal, or not touching the island
            const Index i = ia ? w.a : w.b, o = ia ? w.b : w.a;
            col[i] += w.g;
            if (o >= 0) col[o] -= w.g;
        }
        isl.g.assign(col.begin(), col.end());
        std::sort(isl.g.begin(), isl.g.end());   // a fixed summation order
    }
    std::erase_if(out, [](const Island& isl) { return isl.g.empty(); });
    return out;
}

/// g·s: the island's KCL sum of a solution s.
[[nodiscard]] inline Real island_dot(const Island& isl, const Vector& s) {
    Real a = Real{0};
    for (const auto& [j, g] : isl.g) a += g * s[j];
    return a;
}

/// Σ over the island's rows of a (+ b), summed exactly (Neumaier): the rows
/// are O(C/h·v) and cancel to the island's net.
[[nodiscard]] inline Real island_exact_sum(const Island& isl, const Vector& a,
                                           const Vector* b = nullptr) {
    Real s = Real{0}, c = Real{0};
    auto add = [&](Real v) {
        const Real t = s + v;
        c += std::abs(s) >= std::abs(v) ? (s - t) + v : (v - t) + s;
        s = t;
    };
    for (const Index i : isl.nodes) {
        add(a[i]);
        if (b) add((*b)[i]);
    }
    return s + c;
}

/// Σ_S of −E_j, E_j the unit current injected into `anode` and drawn from
/// `cathode` (ground: −1).
[[nodiscard]] inline Real island_unit_sum(const Island& isl, Index anode, Index cathode) {
    return Real((cathode >= 0 && isl.in[static_cast<std::size_t>(cathode)]) ? 1 : 0)
         - Real((anode >= 0 && isl.in[static_cast<std::size_t>(anode)]) ? 1 : 0);
}

/// The correction of one (mask, matrix): `prepare` once, then `fix` after
/// every solve J·s = f on that matrix.
class IslandCorrection {
public:
    /// `solve_ref(ref, w)` fills w = J⁻¹·e_ref for an island's reference row.
    template <class SolveRef>
    void prepare(const std::vector<Island>& isl, SolveRef&& solve_ref) {
        ws_.resize(isl.size());
        kap_.resize(isl.size());
        for (std::size_t m = 0; m < isl.size(); ++m) {
            solve_ref(isl[m].ref, ws_[m]);
            kap_[m] = island_dot(isl[m], ws_[m]);
        }
        joint_ok_ = false;
        if (isl.size() > 1) {
            const auto k = static_cast<Index>(isl.size());
            DenseMatrix K(k, k);
            for (Index m = 0; m < k; ++m)
                for (Index n = 0; n < k; ++n)
                    K(m, n) = island_dot(isl[static_cast<std::size_t>(m)],
                                         ws_[static_cast<std::size_t>(n)]);
            if (K.allFinite()) {
                joint_lu_.compute(K);
                joint_ok_ = std::isfinite(joint_lu_.rcond()) && joint_lu_.rcond() > Real{0};
            }
        }
    }

    /// Impose each island's KCL sum on s, a solution of J·s = f with
    /// Σ_S f = src(island); `base` (optional) is added to s inside the sum
    /// (a solve for an increment δ = s − base).
    template <class Src>
    void fix(const std::vector<Island>& isl, Vector& s, Src&& src,
             const Vector* base = nullptr) {
        auto sigma = [&](std::size_t m) {
            Real sig = island_dot(isl[m], s) - src(isl[m]);
            if (base) sig += island_dot(isl[m], *base);
            return sig;
        };
        if (isl.size() > 1 && joint_ok_) {
            Vector sig(static_cast<Index>(isl.size()));
            for (std::size_t m = 0; m < isl.size(); ++m) sig[static_cast<Index>(m)] = sigma(m);
            const Vector a = joint_lu_.solve(sig);
            for (std::size_t m = 0; m < isl.size(); ++m) s -= ws_[m] * a[static_cast<Index>(m)];
            fixes_ += isl.size();
            return;
        }
        for (std::size_t m = 0; m < isl.size(); ++m) {
            const Real sig = sigma(m);
            if (kap_[m] != Real{0} && std::isfinite(kap_[m])) {
                s -= ws_[m] * (sig / kap_[m]);
                ++fixes_;
            }
        }
    }

    /// Islands corrected so far (one per island and fix).
    [[nodiscard]] Size fixes() const noexcept { return fixes_; }

private:
    std::vector<Vector> ws_;
    std::vector<Real> kap_;
    Eigen::PartialPivLU<DenseMatrix> joint_lu_;
    bool joint_ok_ = false;
    Size fixes_ = 0;
};

}  // namespace pulsim::events
