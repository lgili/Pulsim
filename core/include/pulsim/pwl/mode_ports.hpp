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
// Branch-form diodes (add-branch-form-ideal-switch 6.1) are ports with their
// own R_on; an ideal one (R_on = 0) has no ON conductance to cut, so the
// caller poses the site with it OFF (`ideal(j)`: the selector's OFF
// coordinates) and the LCP reads its crossing signals directly: the port
// current i_s when it conducts, the gap w = V_f − v when it blocks.
// =============================================================================

#include "pulsim/events/islands.hpp"
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
            // a branch-form diode's own R_on (0: ideal, posed in OFF
            // coordinates by the selector, cut out of the site's mask)
            ports_.push_back(events::DiodePort{
                d.var >= 0 ? pool.branch_R_on(d.branch_id) : Real{1} / d.g_on,
                d.g_off, d.V_f, d.V_th});
        }
    }

    /// First branch-form diode (empty: every diode is a resistor pair).
    [[nodiscard]] const std::string& branch_form_diode() const noexcept {
        return branch_form_;
    }
    [[nodiscard]] const std::vector<events::DiodePort>& ports() const noexcept {
        return ports_;
    }
    /// Port j is an ideal diode (R_on = 0): posed OFF in the site's mask.
    [[nodiscard]] bool ideal(std::size_t j) const noexcept {
        return ports_[j].R_on == Real{0};
    }
    [[nodiscard]] Size solves() const noexcept { return solves_; }
    /// Islands corrected so far (one per island and port problem).
    [[nodiscard]] Size island_corrections() const noexcept { return corr_.fixes(); }

    /// Pass 0: the k-port of the site's own system J(mask, dt)·s =
    /// −(b_c + b_extra) — the cache's direct solve of that (mask, dt), then
    /// each floating island's KCL imposed exactly. dt = 0: static.
    void site(const topology::SwitchStateMask& mask, Real dt,
              const Vector& b_extra, DenseMatrix& Z, Vector& u) {
        const auto& isl = islands_(mask);
        const Vector& b_c = bconst_(mask);
        cache_.solve_at(mask, dt, b_extra, sol_);
        corr_.prepare(isl, [&](Index ref, Vector& w) {
            rhs_ = -b_c;
            rhs_[ref] -= Real{1};          // J·w = e_ref (the cache adds b_c)
            cache_.solve_at(mask, dt, rhs_, w);
        });
        // Σ_S f, f = −(b_c + b_extra): the exact sum of the given rows.
        corr_.fix(isl, sol_, [&](const events::Island& s) {
            return -events::island_exact_sum(s, b_c, &b_extra);
        });
        ports_of_(sol_, u);
        Z.resize(static_cast<Index>(nodes_.size()),
                 static_cast<Index>(nodes_.size()));
        for (std::size_t j = 0; j < nodes_.size(); ++j) {
            unit_rhs_(b_c, j, rhs_);   // J·s = −E_j
            cache_.solve_at(mask, dt, rhs_, sol_);
            corr_.fix(isl, sol_, [&](const events::Island& s) {
                return events::island_unit_sum(s, nodes_[j].first, nodes_[j].second);
            });
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

    /// The islands of `mask` (events/islands.hpp, written for dsed too): empty
    /// when the circuit holds a device kind the rule does not classify.
    const std::vector<events::Island>& islands_(const topology::SwitchStateMask& mask) {
        auto it = islands_cache_.find(mask);
        if (it != islands_cache_.end()) return it->second;
        if (islands_cache_.size() >= kMemoMasks) islands_cache_.clear();
        return islands_cache_
            .emplace(mask, events::find_islands(graph_, pool_, mask, kWeakResistor))
            .first->second;
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
    std::unordered_map<topology::SwitchStateMask, std::vector<events::Island>> islands_cache_;
    events::IslandCorrection corr_;
    Vector rhs_, sol_;
    Size solves_ = 0;
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
        // the law's own reading: v_a − v_k, or V_f + i_s for an ON
        // branch-form diode (g_on = 1 there: i is the solved current)
        const Real v = d.law_voltage(x, on[j] != 0);
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

/// The LCP of one decision site, engine-neutral (add-complementarity-mode-
/// selection task 4.2; add-branch-form-ideal-switch 6.1). The site solves
/// J(mask, for_dt)·s = −(b_c + b_extra) through `cache.solve_at` — pwl's
/// trapezoidal step, a restart stage on the factor of 2γh, a split sub-step,
/// the static step (for_dt = 0); trbdf2 calls it with its own stage's
/// (mask, dt, b_extra). `mask_of(on)` overlays diode bits `on` (diode order)
/// on the site's gates; `prev_on` the bits the site starts from; `x0` the
/// state at its start (the circuit scales); `h_site` the step the tie
/// tolerance divides by. The selector poses every pass on the site's own
/// (Z, u) (DECISIONS F2); an ideal diode is cut out of it (OFF coordinates);
/// the judge reads the engine's own solve of each candidate.
/// `Z`, `u`, `xj`, `coords`: caller scratch (no allocation once sized).
template <class MaskOf>
[[nodiscard]] events::ModeSelection select_site(
    events::ModeSelector& sel, PwlModePorts& ports, const PwlStateSpaceCache& cache,
    const DiodeEventState& diodes, MaskOf&& mask_of, const std::vector<char>& prev_on,
    const Vector& x0, Real for_dt, Real h_site, const Vector& b_extra,
    DenseMatrix& Z, Vector& u, Vector& xj, std::vector<char>& coords) {
    coords.assign(prev_on.size(), 0);
    for (std::size_t j = 0; j < prev_on.size(); ++j)
        coords[j] = ports.ideal(j) ? 0 : prev_on[j];
    const auto mask0 = mask_of(coords);
    const auto sc = ports.scale(x0);
    bool have_site = false;
    auto port_at = [&](Real, DenseMatrix& Zp, Vector& up) {
        if (!have_site) {
            ports.site(mask0, for_dt, b_extra, Z, u);
            have_site = true;
        }
        Zp = Z;
        up = u;
    };
    auto judge = [&](const std::vector<char>& on) {
        cache.solve_at(mask_of(on), for_dt, b_extra, xj);
        return judge_mode(diodes, on, xj, sc, sel.tol);
    };
    return sel.select(ports.ports(), prev_on, port_at, h_site, sc, judge);
}

}  // namespace pulsim::pwl
