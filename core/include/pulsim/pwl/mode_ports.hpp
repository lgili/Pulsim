#pragma once

// =============================================================================
// pwl/mode_ports.hpp — the diode k-port of one fixed-step decision site
// (add-complementarity-mode-selection D2, tasks 4.2 / 4b.3 / 6.3 on `pwl`).
//
// A pwl decision site (a trapezoidal step, a restart stage at γ·h on the
// factor of 2γh, a split sub-step at dt₂, the static step) solves
// J(mask, dt)·s = −(b_c + b_extra). Cutting every diode out as a port gives
// v = u − Z·e (e: current drawn from each anode into its cathode) under the
// site's starting mask.
//
//   * every pass (`site`): the site's own system, solved through the cache's
//     factor of that (mask, dt) — the very solve the settle's first round
//     makes (DECISIONS F1). The selector's 100·h / 1e4·h tie passes get the
//     same model: on pwl the next decision site is one step away
//     (DECISIONS F2).
//
// Floating islands (DECISIONS F1, the dsed adapter's DECISIONS #30 idea,
// re-implemented here): nodes joined by strong elements and tied to the rest
// only through OFF switches / diodes and resistors of ≤ 1 mS. The factor
// rounds such an island's common mode to ε·(internal current)/G_boundary
// (1e-5 V on a 1 mF bus at dt = 10 µs behind 1 MΩ). The island's KCL sum —
// every internal stamp, C/dt included, cancels — is imposed exactly after
// each solve: σ = g·s − Σ_S f, s −= w·σ/(g·w), w = J⁻¹e_ref; one more
// back-substitution per island and site. Disabled for a circuit with a
// device kind the rule does not classify.
//
// The selector's judge reads the engine's own (uncorrected) solve, so on an
// island a mode the corrected model calls consistent can fail the judge by
// the rounding of F1 (counted as `inconsistent`). A current source joins
// its terminals, so a bus fed by one to ground is not seen as an island
// (accuracy only: no correction is made, none is wrong).
//
// Conductance-form diodes only (R_on = 1/g_on > 0): a branch-form diode is
// reported through `branch_form_diode()` and handled by the caller.
// =============================================================================

#include "pulsim/events/mode_refusal.hpp"
#include "pulsim/events/mode_selector.hpp"
#include "pulsim/models/switched_diode.hpp"
#include "pulsim/numeric/dense.hpp"
#include "pulsim/numeric/types.hpp"
#include "pulsim/pwl/assemble.hpp"
#include "pulsim/pwl/cache.hpp"
#include "pulsim/pwl/device_pool.hpp"
#include "pulsim/pwl/diode_event_state.hpp"
#include "pulsim/stamping/branch_coord.hpp"
#include "pulsim/topology/graph.hpp"
#include "pulsim/topology/switch_state.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace pulsim::pwl {

class PwlModePorts {
public:
    /// Resistors at or below this conductance do not join an island (ties,
    /// bleeders, the builder's 1 GΩ regularisers): DECISIONS F1.
    static constexpr Real kWeakResistor = Real{1e-3};

    PwlModePorts(const PwlStateSpaceCache& cache, const topology::Graph& graph,
                 const DevicePool& pool, const DiodeEventState& diodes)
        : cache_{cache}, graph_{graph}, pool_{pool} {
        for (const auto& d : diodes.entries()) {
            if (d.var >= 0 && branch_form_.empty()) {
                branch_form_ = std::string(graph.branch_name(d.branch_id));
            }
            nodes_.emplace_back(d.from, d.to);
            ports_.push_back(events::DiodePort{
                d.var >= 0 ? Real{0} : Real{1} / d.g_on, d.g_off, d.V_f,
                d.V_th});
        }
    }

    /// First branch-form diode (empty: every diode is a resistor pair).
    [[nodiscard]] const std::string& branch_form_diode() const noexcept {
        return branch_form_;
    }
    [[nodiscard]] const std::vector<events::DiodePort>& ports() const noexcept {
        return ports_;
    }
    [[nodiscard]] Size solves() const noexcept { return solves_; }
    /// Islands corrected so far (one per island and port problem).
    [[nodiscard]] Size island_corrections() const noexcept { return island_fixes_; }

    /// Pass 0: the k-port of the site's own system J(mask, dt)·s =
    /// −(b_c + b_extra) — the cache's direct solve of that (mask, dt), then
    /// each floating island's KCL imposed exactly. dt = 0: static.
    void site(const topology::SwitchStateMask& mask, Real dt,
              const Vector& b_extra, DenseMatrix& Z, Vector& u) {
        const auto& isl = islands_(mask);
        const Vector& b_c = bconst_(mask);
        cache_.solve_at(mask, dt, b_extra, sol_);
        auto solve = [&](const Vector& rhs, Vector& out) {
            cache_.solve_at(mask, dt, rhs, out);
        };
        prepare_(isl, b_c, solve);
        // Σ_S f, f = −(b_c + b_extra): the exact sum of the given rows.
        fix_(isl, sol_, [&](const Island& s) { return -exact_sum_(s, b_c, &b_extra); });
        ports_of_(sol_, u);
        Z.resize(static_cast<Index>(nodes_.size()),
                 static_cast<Index>(nodes_.size()));
        for (std::size_t j = 0; j < nodes_.size(); ++j) {
            unit_rhs_(b_c, j, rhs_);   // J·s = −E_j
            cache_.solve_at(mask, dt, rhs_, sol_);
            fix_(isl, sol_, [&](const Island& s) { return unit_sum_(s, j); });
            column_(sol_, j, Z);
        }
        solves_ += nodes_.size() + 1 + isl.size();
    }

    /// Circuit scales of the LCP tolerances (design D4), as the dsed
    /// adapter's `port_scale_full`: vscale = largest node voltage / source /
    /// drop (1 V floor); iscale = largest branch current, current source or
    /// vscale·G of a resistor; c_max = largest capacitance.
    [[nodiscard]] events::CircuitScale scale(const Vector& x) const {
        const Index nn = graph_.num_nodes();
        events::CircuitScale sc;
        Real vs = Real{1}, is = Real{1e-12}, gmax = Real{0};
        for (Index i = 0; i < static_cast<Index>(x.size()); ++i) {
            if (i < nn) vs = std::max(vs, std::abs(x[i]));
            else is = std::max(is, std::abs(x[i]));
        }
        using K = DevicePool::StoredKind;
        for (Index b = 0; b < graph_.num_branches(); ++b) {
            if (!pool_.is_registered(b)) continue;
            switch (pool_.kind_of(b)) {
            case K::VoltageSource:
                vs = std::max(vs, std::abs(pool_.voltage_source_params(b).V));
                break;
            case K::SineVoltageSource: {
                const auto& p = pool_.sine_voltage_source_params(b);
                vs = std::max(vs, std::abs(p.v_dc) + std::abs(p.v_amplitude));
                break;
            }
            case K::PWMVoltageSource: {
                const auto& p = pool_.pwm_voltage_source_params(b);
                vs = std::max({vs, std::abs(p.v_high), std::abs(p.v_low)});
                break;
            }
            case K::Diode: {
                const auto& p = pool_.diode_params(b);
                vs = std::max({vs, std::abs(p.V_f), std::abs(p.V_th)});
                break;
            }
            case K::CurrentSource:
                is = std::max(is, std::abs(pool_.current_source_params(b).I));
                break;
            case K::Resistor:
                gmax = std::max(gmax, std::abs(pool_.resistor_params(b).G));
                break;
            case K::Capacitor:
                sc.c_max = std::max(sc.c_max, pool_.capacitor_params(b).C);
                break;
            default:
                break;
            }
        }
        sc.vscale = vs;
        sc.iscale = std::max(is, vs * gmax);
        return sc;
    }

private:
    /// A floating island under one mask: `g` is its KCL sum as a row over
    /// the unknowns, from the boundary branches alone.
    struct Island {
        Index ref = -1;                          // the correction's injection row
        std::vector<std::pair<Index, Real>> g;   // (column, Σ boundary conductance)
        std::vector<char> in;                    // in[node]
        std::vector<Index> nodes;
    };

    /// Per-mask memo bound: a long run on many diodes visits many masks;
    /// past this many the memo starts over (only b_c and the island lists
    /// are kept, both cheap to rebuild).
    static constexpr std::size_t kMemoMasks = 256;

    const Vector& bconst_(const topology::SwitchStateMask& mask) {
        auto it = bconst_cache_.find(mask);
        if (it == bconst_cache_.end()) {
            if (bconst_cache_.size() >= kMemoMasks) bconst_cache_.clear();
            sparse::Matrix G, C;
            Vector b;
            assemble_segment_split(graph_, pool_, mask, G, C, b);
            it = bconst_cache_.emplace(mask, std::move(b)).first;
        }
        return it->second;
    }

    /// The islands of `mask` (empty when the circuit holds a device kind
    /// the rule does not classify, or none floats). Ported from the dsed
    /// adapter's `port_islands_of_` (c4255226); kept separate because that
    /// one is a private member of the dsed adapter (unify in Phase 3).
    const std::vector<Island>& islands_(const topology::SwitchStateMask& mask) {
        auto it = islands_cache_.find(mask);
        if (it != islands_cache_.end()) return it->second;
        std::vector<Island> out;
        const Index nn = graph_.num_nodes();
        std::vector<Index> up(static_cast<std::size_t>(nn + 1));   // nn = ground
        for (Index i = 0; i <= nn; ++i) up[static_cast<std::size_t>(i)] = i;
        auto root = [&](Index i) {
            Index r = i < 0 ? nn : i;
            while (up[static_cast<std::size_t>(r)] != r) r = up[static_cast<std::size_t>(r)];
            return r;
        };
        auto join = [&](Index a, Index b) { up[static_cast<std::size_t>(root(a))] = root(b); };
        struct Weak { Index a, b; Real g; };
        std::vector<Weak> weak;
        using K = DevicePool::StoredKind;
        bool known = true;
        Size sw = 0;
        for (Index b = 0; b < graph_.num_branches() && known; ++b) {
            const auto& br = graph_.branch(b);
            if (!pool_.is_registered(b)) { known = false; break; }
            const K kind = pool_.kind_of(b);
            if (br.kind == topology::BranchKind::Switch) {
                const bool on = mask.get(sw++);
                if (kind == K::Diode || kind == K::Switch) {
                    if (on) join(br.from, br.to);
                    else weak.push_back({br.from, br.to, kind == K::Diode
                        ? pool_.diode_params(b).g_off : pool_.switch_g_off(b)});
                } else {
                    known = false;
                }
                continue;
            }
            switch (kind) {
            case K::Resistor: {
                const Real g = pool_.resistor_params(b).G;
                if (std::abs(g) <= kWeakResistor) weak.push_back({br.from, br.to, g});
                else join(br.from, br.to);
                break;
            }
            case K::Capacitor: case K::Inductor: case K::VoltageSource:
            case K::PWMVoltageSource: case K::SineVoltageSource: case K::PulseVoltageSource:
            case K::CurrentSource: case K::SineCurrentSource: case K::PulseCurrentSource:
            case K::PwlCurrentSource:
                // strong, or a source: no source sits on an island's boundary
                join(br.from, br.to);
                break;
            default:
                known = false;
            }
        }
        if (known) {
            std::vector<Index> rt(static_cast<std::size_t>(nn));
            std::unordered_map<Index, Index> size;
            for (Index i = 0; i < nn; ++i) ++size[rt[static_cast<std::size_t>(i)] = root(i)];
            const Index gr = root(-1);
            std::unordered_map<Index, std::size_t> slot;
            for (Index i = 0; i < nn; ++i) {
                const Index r = rt[static_cast<std::size_t>(i)];
                if (r == gr || size[r] < 2) continue;   // grounded, or nothing internal
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
        }
        if (islands_cache_.size() >= kMemoMasks) islands_cache_.clear();
        return islands_cache_.emplace(mask, std::move(out)).first->second;
    }

    /// w = J⁻¹·e_ref and κ = g·w per island, for the current (mask, dt).
    template <class Solve>
    void prepare_(const std::vector<Island>& isl, const Vector& b_c, Solve&& solve) {
        ws_.resize(isl.size());
        kap_.resize(isl.size());
        for (std::size_t m = 0; m < isl.size(); ++m) {
            rhs_ = -b_c;
            rhs_[isl[m].ref] -= Real{1};          // J·w = e_ref
            solve(rhs_, ws_[m]);
            kap_[m] = dot_(isl[m], ws_[m]);
        }
        // Several islands: K(m, n) = g_m·w_n is the identity in exact
        // arithmetic (disjoint islands, g_m sums J's rows over S_m), but
        // behind a 1e7 Ω tie the off-diagonal rounding moved two buses
        // joined by 1 MΩ by 6.5e-9 V when corrected one after another.
        joint_ok_ = false;
        if (isl.size() > 1) {
            const auto k = static_cast<Index>(isl.size());
            DenseMatrix K(k, k);
            for (Index m = 0; m < k; ++m)
                for (Index n = 0; n < k; ++n)
                    K(m, n) = dot_(isl[static_cast<std::size_t>(m)],
                                   ws_[static_cast<std::size_t>(n)]);
            if (K.allFinite()) {
                joint_lu_.compute(K);
                joint_ok_ = std::isfinite(joint_lu_.rcond()) && joint_lu_.rcond() > Real{0};
            }
        }
    }

    /// Impose each island's KCL sum on s, a solution of J·s = f with
    /// Σ_S f = src(island).
    template <class Src>
    void fix_(const std::vector<Island>& isl, Vector& s, Src&& src) {
        if (isl.size() > 1 && joint_ok_) {   // else: one after another, as dsed (#30)
            Vector sig(static_cast<Index>(isl.size()));
            for (std::size_t m = 0; m < isl.size(); ++m)
                sig[static_cast<Index>(m)] = dot_(isl[m], s) - src(isl[m]);
            const Vector a = joint_lu_.solve(sig);
            for (std::size_t m = 0; m < isl.size(); ++m) s -= ws_[m] * a[static_cast<Index>(m)];
            island_fixes_ += isl.size();
            return;
        }
        for (std::size_t m = 0; m < isl.size(); ++m) {
            const Real sig = dot_(isl[m], s) - src(isl[m]);
            if (kap_[m] != Real{0} && std::isfinite(kap_[m])) {
                s -= ws_[m] * (sig / kap_[m]);
                ++island_fixes_;
            }
        }
    }

    static Real dot_(const Island& isl, const Vector& s) {
        Real a = Real{0};
        for (const auto& [j, g] : isl.g) a += g * s[j];
        return a;
    }

    /// Σ over the island's rows of b_c (+ b_extra), summed exactly
    /// (Neumaier): the rows are O(C/dt·v) and cancel to the island's net.
    static Real exact_sum_(const Island& isl, const Vector& b_c, const Vector* b2) {
        Real s = Real{0}, c = Real{0};
        auto add = [&](Real v) {
            const Real t = s + v;
            c += std::abs(s) >= std::abs(v) ? (s - t) + v : (v - t) + s;
            s = t;
        };
        for (const Index i : isl.nodes) {
            add(b_c[i]);
            if (b2) add((*b2)[i]);
        }
        return s + c;
    }

    /// Σ_S of −E_j (the unit injection of port j).
    [[nodiscard]] Real unit_sum_(const Island& isl, std::size_t j) const {
        const auto [a, c] = nodes_[j];
        return Real((c >= 0 && isl.in[static_cast<std::size_t>(c)]) ? 1 : 0)
             - Real((a >= 0 && isl.in[static_cast<std::size_t>(a)]) ? 1 : 0);
    }

    [[nodiscard]] Real vp_(const Vector& s, std::size_t j) const {
        const auto [a, c] = nodes_[j];
        return stamping::read_node_voltage(s, a) -
               stamping::read_node_voltage(s, c);
    }

    void ports_of_(const Vector& s, Vector& u) const {
        u.resize(static_cast<Index>(nodes_.size()));
        for (std::size_t j = 0; j < nodes_.size(); ++j)
            u[static_cast<Index>(j)] = vp_(s, j);
    }

    /// rhs with which solve(·) gives J·s = −E_j: −b_c + E_j (the cache adds b_c).
    void unit_rhs_(const Vector& b_c, std::size_t j, Vector& rhs) const {
        rhs = -b_c;
        const auto [a, c] = nodes_[j];
        if (stamping::node_is_active(a)) rhs[a] += Real{1};
        if (stamping::node_is_active(c)) rhs[c] -= Real{1};
    }

    void column_(const Vector& s, std::size_t j, DenseMatrix& Z) const {
        for (std::size_t i = 0; i < nodes_.size(); ++i)
            Z(static_cast<Index>(i), static_cast<Index>(j)) = -vp_(s, i);
    }

    const PwlStateSpaceCache& cache_;
    const topology::Graph& graph_;
    const DevicePool& pool_;
    std::vector<std::pair<Index, Index>> nodes_;
    std::vector<events::DiodePort> ports_;
    std::string branch_form_;
    std::unordered_map<topology::SwitchStateMask, Vector> bconst_cache_;
    std::unordered_map<topology::SwitchStateMask, std::vector<Island>> islands_cache_;
    std::vector<Vector> ws_;
    std::vector<Real> kap_;
    Eigen::PartialPivLU<DenseMatrix> joint_lu_;
    bool joint_ok_ = false;
    Vector rhs_, sol_;
    Size solves_ = 0, island_fixes_ = 0;
};

/// Structural passivity at the diode ports (design D3), the dsed adapter's
/// rule: a VCVS or a negative R / C / L is non-passive (named); an ideal
/// transformer is passive but not reciprocal. shortcut: the coupled-inductor
/// PSD test of the dsed adapter is not repeated here; a non-PSD coupling
/// reads passive (Lemke then refuses at a ray instead of enumerating) —
/// upgrade by sharing that test when a coupled circuit runs 'lcp' on pwl.
[[nodiscard]] inline events::PassivityClass passivity_of(
    const topology::Graph& graph, const DevicePool& pool) {
    using K = DevicePool::StoredKind;
    events::PassivityClass out;
    for (Index b = 0; b < graph.num_branches(); ++b) {
        if (!pool.is_registered(b)) continue;
        const K kind = pool.kind_of(b);
        const bool neg =
            (kind == K::Resistor && pool.resistor_params(b).G < Real{0})
            || (kind == K::Capacitor && pool.capacitor_params(b).C < Real{0})
            || (kind == K::Inductor && pool.inductor_params(b).L < Real{0});
        if (kind == K::VCVS || neg)
            return {std::string(graph.branch_name(b)),
                    std::string(stored_kind_name(kind)), {}, {}};
        if (kind == K::IdealTransformer && out.nonreciprocal_device.empty()) {
            out.nonreciprocal_device = std::string(graph.branch_name(b));
            out.nonreciprocal_kind = std::string(stored_kind_name(kind));
        }
    }
    return out;
}

/// The engine's own law on a solved state (the judge of D6 step 4 on
/// `pwl`): 1 when every diode obeys the state `on[j]` it was solved under
/// — ON: i = g_on·(v − V_f) ≥ −tol·iscale; OFF: v − V_on ≤ tol·vscale,
/// V_on = SwitchedDiode::turn_on_voltage. `worst` (volts: an ON
/// violation as −i/g_on) and `worst_j` name the worst violation.
inline int judge_mode(const DiodeEventState& diodes, const std::vector<char>& on,
                      const Vector& x, const events::CircuitScale& sc, Real tol,
                      Real* worst = nullptr, int* worst_j = nullptr) {
    const auto& es = diodes.entries();
    if (on.size() != es.size())
        throw std::logic_error("pwl::judge_mode: the mode has " + std::to_string(on.size()) +
                               " entries for " + std::to_string(es.size()) + " diodes");
    Real w = Real{0};
    int wj = -1;
    bool ok = true;
    for (std::size_t j = 0; j < es.size(); ++j) {
        const auto& d = es[j];
        const auto p = d.params();
        const Real v = stamping::read_node_voltage(x, d.from) -
                       stamping::read_node_voltage(x, d.to);
        Real viol;
        bool bad;
        if (on[j]) {
            const Real i = models::SwitchedDiode::branch_current(true, v, p);
            bad = -i > tol * sc.iscale;
            viol = -i / p.g_on;
        } else {
            viol = v - models::SwitchedDiode::turn_on_voltage(p);
            bad = viol > tol * sc.vscale;
        }
        if (bad) ok = false;
        if (viol > w) { w = viol; wj = static_cast<int>(j); }
    }
    if (worst) *worst = w;
    if (worst_j) *worst_j = wj;
    return ok ? 1 : 0;
}

}  // namespace pulsim::pwl
