#pragma once

// =============================================================================
// pwl/mode_ports.hpp — the diode k-port of one fixed-step decision site
// (add-complementarity-mode-selection D2, tasks 4.2 / 4b.3 / 6.3 on `pwl`).
//
// A pwl decision site (a trapezoidal step, a restart stage at γ·h, a split
// sub-step) solves J(mask, dt)·s = −(b_c + b_extra). Cutting every diode out
// as a port gives v = u − Z·e (e: current drawn from each anode into its
// cathode) under the site's starting mask. The port model is written in
// INCREMENT form from the step-start state x0: J·δ = −(b_c + b_extra + J·x0),
// u = v(x0) + v(δ) — the dsed adapter's history form lost the answer to
// rounding on a floating bus (C/h·x ≫ the answer).
//
//   * pass 0 (`site`): the site's own discretisation, through the cache's
//     factor of that (mask, dt) — the one the settle's first solve used.
//   * widenings (`widened`, the selector's 100·h and 1e4·h tie passes):
//     backward Euler over h from x0, J = G + C/(2h) (the trapezoidal
//     C_coef = 2C, −2L at dt = 2h), in increment form J·δ = −(b_c + b_src +
//     G·x0) where the C/h·x0 terms cancel analytically. Dense LU of its own:
//     ties are rare, and the cache's event table (one entry per mask, refactored
//     on a dt change) is left alone so the engine's own factor counts and
//     solves are unchanged (DECISIONS #22).
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
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace pulsim::pwl {

class PwlModePorts {
public:
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

    /// Pass 0: the k-port of the site's own system J(mask, dt)·s =
    /// −(b_c + b_extra), around the step-start state x0. dt = 0: static.
    void site(const topology::SwitchStateMask& mask, Real dt,
              const Vector& b_extra, const Vector& x0, DenseMatrix& Z,
              Vector& u) {
        const Mats& m = mats_(mask);
        rhs_ = b_extra + m.G * x0;
        if (dt > Real{0}) rhs_ += (m.C * x0) / dt;
        cache_.solve_at(mask, dt, rhs_, sol_);
        ports_from_(x0, sol_, u);
        Z.resize(static_cast<Index>(nodes_.size()),
                 static_cast<Index>(nodes_.size()));
        for (std::size_t j = 0; j < nodes_.size(); ++j) {
            unit_rhs_(m.b, j, rhs_);   // J·s = −E_j
            cache_.solve_at(mask, dt, rhs_, sol_);
            column_(sol_, j, Z);
        }
        solves_ += nodes_.size() + 1;
    }

    /// Widening: backward Euler over h from x0, sources b_src (the site's
    /// source and callback rows, without the companion history).
    void widened(const topology::SwitchStateMask& mask, Real h,
                 const Vector& b_src, const Vector& x0, DenseMatrix& Z,
                 Vector& u) {
        const Mats& m = mats_(mask);
        DenseMatrix J = DenseMatrix(m.G);
        if (h > Real{0}) J += DenseMatrix(m.C) / (Real{2} * h);
        const Eigen::PartialPivLU<DenseMatrix> lu(J);
        rhs_ = -(m.b + b_src + m.G * x0);
        sol_ = lu.solve(rhs_);
        ports_from_(x0, sol_, u);
        Z.resize(static_cast<Index>(nodes_.size()),
                 static_cast<Index>(nodes_.size()));
        for (std::size_t j = 0; j < nodes_.size(); ++j) {
            unit_rhs_(m.b, j, rhs_);
            rhs_ = -(m.b + rhs_);      // J·s = −E_j
            sol_ = lu.solve(rhs_);
            column_(sol_, j, Z);
        }
        solves_ += nodes_.size() + 1;
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
            K kind;
            try {
                kind = pool_.kind_of(b);
            } catch (const std::out_of_range&) {
                continue;
            }
            switch (kind) {
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
    struct Mats {
        sparse::Matrix G, C;
        Vector b;
    };

    const Mats& mats_(const topology::SwitchStateMask& mask) {
        auto it = mats_cache_.find(mask);
        if (it == mats_cache_.end()) {
            Mats m;
            assemble_segment_split(graph_, pool_, mask, m.G, m.C, m.b);
            it = mats_cache_.emplace(mask, std::move(m)).first;
        }
        return it->second;
    }

    [[nodiscard]] Real vp_(const Vector& s, std::size_t j) const {
        const auto [a, c] = nodes_[j];
        return stamping::read_node_voltage(s, a) -
               stamping::read_node_voltage(s, c);
    }

    void ports_from_(const Vector& x0, const Vector& delta, Vector& u) const {
        u.resize(static_cast<Index>(nodes_.size()));
        for (std::size_t j = 0; j < nodes_.size(); ++j)
            u[static_cast<Index>(j)] = vp_(x0, j) + vp_(delta, j);
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
    std::unordered_map<topology::SwitchStateMask, Mats> mats_cache_;
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
