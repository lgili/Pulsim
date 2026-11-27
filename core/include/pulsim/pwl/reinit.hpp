#pragma once

// =============================================================================
// add-branch-form-ideal-switch D4: the re-initialisation record, shared by the
// implicit engines (pwl's run_transient, run_transient_trbdf2).
// =============================================================================
//
// A mode can make states dependent (an inductor cut-set opened by ideal
// elements, a capacitor loop closed by them). Both implicit engines realise
// the charge / flux conserving projection with three SDIRK(2,2) steps; this
// header owns what they share: when a committed step entered such a mode
// impulsively, and the record of the explicit projection (DECISIONS #10):
//
//   lambda = (K x- - d) / sum(s_j^2 / M_j),   dW = lambda (K x- - d) / 2 >= 0
//
// with d = K x(t+h) (the restart leaves the constraint exact). Moved here from
// run_transient.hpp unchanged (the fixed engine's numbers are bit-identical).

#include "pulsim/numeric/dense.hpp"
#include "pulsim/pwl/branch_structure.hpp"
#include "pulsim/pwl/device_pool.hpp"
#include "pulsim/solver/result.hpp"
#include "pulsim/solver/unsupported.hpp"
#include "pulsim/stamping/branch_coord.hpp"
#include "pulsim/topology/graph.hpp"
#include "pulsim/topology/switch_state.hpp"

#include <algorithm>
#include <cmath>
#include <format>
#include <string>
#include <vector>

namespace pulsim::pwl {

/// False when the state `xs` is NOT in the cut-set `d` (sum s_j i_j = 0
/// without current sources): a solve whose mask names the cut-set but whose
/// diode flip already undid it ("flip at the step's end" rule), so no
/// impulse happened. Loops: always true.
[[nodiscard]] inline bool in_dependent_mode(const topology::Graph& graph,
                                            const DevicePool& pool,
                                            const DependentStructure& d,
                                            const Vector& xs) {
    if (d.kind != DependentStructure::Kind::CutSet || !d.sources.empty()) {
        return true;
    }
    Real k = 0, scale = 0;
    for (Size j = 0; j < d.states.size(); ++j) {
        const Real i = xs[pool.branch_var_id_for_inductor(d.states[j], graph)];
        k += static_cast<Real>(d.signs[j]) * i;
        scale += std::abs(i);
    }
    return std::abs(k) <= Real{1e-9} * scale;
}

/// The last committed mode and the D4 records made against it.
class ReinitTracker {
public:
    ReinitTracker(const topology::Graph& graph, const DevicePool& pool,
                  bool active)
        : graph_{graph}, pool_{pool} {
        if (!active) return;
        long s = 0;
        for (Index b = 0; b < graph.num_branches(); ++b) {
            const bool sw = graph.branch(b).kind == topology::BranchKind::Switch;
            slot_.push_back(sw ? s++ : -1L);
        }
    }

    /// True when a gated (non-diode) switch bit differs between the last
    /// committed mode and `now`.
    [[nodiscard]] bool gated_change(const topology::SwitchStateMask& now) const {
        bool gated = false;
        if (!known_) return gated;
        for (Index el = 0; el < graph_.num_branches(); ++el) {
            const long s = slot_[static_cast<std::size_t>(el)];
            if (s < 0 || !pool_.is_registered(el) ||
                pool_.kind_of(el) == DevicePool::StoredKind::Diode) {
                continue;
            }
            gated = gated ||
                now.get(static_cast<Size>(s)) != prev_.get(static_cast<Size>(s));
        }
        return gated;
    }

    /// Dependent structures of the mode `now` that the last committed mode
    /// did not have, entered by a gated change anywhere (a gate can force a
    /// diode into the mode: charge pump, review M3) or at the start. A diode
    /// commutating on its own is consistent (it crossed its threshold); with
    /// `gated_only = false` those entries are returned too.
    [[nodiscard]] std::vector<const DependentStructure*>
    entering(BranchStructureChecker& structure, const topology::SwitchStateMask& now,
             bool gated_only = true) {
        std::vector<const DependentStructure*> out;
        const auto& ms = structure.check(now, true);
        const bool gated_step = gated_change(now);
        const ModeStructure* before = known_ ? &structure.check(prev_, true) : nullptr;
        for (const auto& d : ms.dependent) {
            bool old = false;
            if (before != nullptr) {
                for (const auto& o : before->dependent) {
                    old = old || (o.kind == d.kind && o.states == d.states &&
                                  o.elements == d.elements);
                }
            }
            if (old || (gated_only && !(gated_step || !known_))) continue;
            out.push_back(&d);
        }
        return out;
    }

    /// After a committed step solved under `stamped` from `x_prev` (at
    /// `t_prev`) to `x`: record each dependent structure the step entered
    /// impulsively (the explicit projection), or raise on 'error'.
    /// `x_prev` / `t_prev`: the state and time the mode was entered from (the
    /// fixed engine's event split passes the split instant's left limit).
    void record(BranchStructureChecker& structure,
                const topology::SwitchStateMask& stamped, const Vector& x_prev,
                const Vector& x, Real t_prev, long sample, bool on_impulse_error,
                const char* engine, solver::SimulationResult& result) {
        const auto ent = entering(structure, stamped);
        const auto& deps = structure.check(stamped, true).dependent;
        std::vector<char> grouped(deps.size(), 0);
        for (const DependentStructure* dp : ent) {
            const auto& d = *dp;
            // shortcut: a natural diode commutation in the same step as an
            // unrelated gate edge records its small residual too; upgrade
            // by locating the crossing.
            if (!in_dependent_mode(graph_, pool_, d, x)) continue;
            const bool cut = d.kind == DependentStructure::Kind::CutSet;
            // Structures of the mode that share a state with d constrain its
            // projection too (an ideal clamp diode across C2 when S closes
            // C1 -S- C2: C2 is held, C1 dumps all of its charge — 50 nJ, not
            // the free share's 2.381 nJ): one joint record for the group.
            const auto di = static_cast<Size>(dp - deps.data());
            if (di < deps.size() && grouped[di]) continue;
            std::vector<Size> grp;
            if (di < deps.size()) {
                grp.push_back(di);
                grouped[di] = 1;
                for (Size q = 0; q < grp.size(); ++q) {
                    for (Size o = 0; o < deps.size(); ++o) {
                        if (grouped[o] || deps[o].kind != d.kind) continue;
                        bool shares = false;
                        for (const Index st : deps[grp[q]].states) {
                            shares = shares || std::find(deps[o].states.begin(), deps[o].states.end(),
                                                         st) != deps[o].states.end();
                        }
                        if (shares) {
                            grouped[o] = 1;
                            grp.push_back(o);
                        }
                    }
                }
            }
            if (grp.size() > 1) {
                record_group_(deps, grp, d, cut, stamped, x_prev, x, t_prev, sample,
                              on_impulse_error, engine, result);
                continue;
            }
            std::vector<Real> xa, xb, mm;
            Real kx0 = 0, kx1 = 0, inv_m = 0, scale = 0;
            for (Size j = 0; j < d.states.size(); ++j) {
                const Index b = d.states[j];
                Real a = 0, c = 0, m = 0;
                if (cut) {
                    const Index v = pool_.branch_var_id_for_inductor(b, graph_);
                    a = x_prev[v];
                    c = x[v];
                    m = pool_.inductor_params(b).L;
                } else {
                    const auto& br = graph_.branch(b);
                    a = stamping::read_node_voltage(x_prev, br.from) -
                        stamping::read_node_voltage(x_prev, br.to);
                    c = stamping::read_node_voltage(x, br.from) -
                        stamping::read_node_voltage(x, br.to);
                    m = pool_.capacitor_params(b).C;
                }
                xa.push_back(a);
                xb.push_back(c);
                mm.push_back(m);
                scale += std::abs(a) + std::abs(c);
                kx0 += static_cast<Real>(d.signs[j]) * a;
                kx1 += static_cast<Real>(d.signs[j]) * c;
                inv_m += Real{1} / m;
            }
            // d = K x(t+h): the restart leaves the constraint exact.
            const Real r = kx0 - kx1;
            // Already consistent (start, resume, equal capacitors):
            // round-off is not an impulse (review M1).
            if (std::abs(r) <= Real{1e-12} * scale) continue;
            solver::ReinitRecord rec;
            rec.t = t_prev;
            rec.kind = cut ? "cut_set" : "capacitor_loop";
            for (const Index el : d.elements) {
                rec.devices.emplace_back(graph_.branch_name(el));
            }
            for (const Index b : d.states) {
                rec.states.emplace_back(graph_.branch_name(b));
            }
            rec.lambda = r / inv_m;
            rec.dW = Real{0.5} * rec.lambda * r;
            for (Size j = 0; j < xa.size(); ++j) {
                const Real proj = xa[j] -
                    static_cast<Real>(d.signs[j]) * rec.lambda / mm[j];
                rec.projection_gap = std::max(rec.projection_gap,
                                              std::abs(proj - xb[j]));
            }
            rec.sample = sample;
            rec.cause = cause_(d, stamped);
            if (on_impulse_error) {
                std::string sts;
                for (const auto& s : rec.states) {
                    sts += (sts.empty() ? "" : ", ") + s;
                }
                throw UnsupportedError(
                    engine, "impulse", rec.devices.front(),
                    std::string(stored_kind_name(pool_.kind_of(d.elements.front()))),
                    std::format(
                        "on_impulse='error': at t = {} '{}' makes {} "
                        "dependent ({}); re-initialising them moves "
                        "{:.6g} {} and removes {:.6g} J.",
                        t_prev, rec.devices.front(), sts,
                        cut ? "an inductor cut-set" : "a capacitor loop",
                        rec.lambda, cut ? "V*s" : "C", rec.dW));
            }
            result.reinit_records.push_back(std::move(rec));
        }
        prev_ = stamped;
        known_ = true;
    }

    /// A committed state whose step entered no new dependent structure
    /// (an event landing, a committed probe): the mode is remembered only.
    void note(const topology::SwitchStateMask& m) {
        prev_ = m;
        known_ = true;
    }

private:
    /// The minimum-energy projection of x_prev onto every constraint of the
    /// group (rows K, residual r = K x− − K x+, λ = (K M⁻¹ Kᵀ)⁻¹ r, dW =
    /// λ·r / 2), recorded once for the entering structure `d`.
    void record_group_(const std::vector<DependentStructure>& deps, const std::vector<Size>& grp,
                       const DependentStructure& d, bool cut,
                       const topology::SwitchStateMask& stamped, const Vector& x_prev,
                       const Vector& x, Real t_prev, long sample, bool on_impulse_error,
                       const char* engine, solver::SimulationResult& result) {
        std::vector<Index> st;
        for (const Size g : grp)
            for (const Index b : deps[g].states)
                if (std::find(st.begin(), st.end(), b) == st.end()) st.push_back(b);
        const auto nu = static_cast<Index>(st.size()), ng = static_cast<Index>(grp.size());
        Vector xa(nu), xb(nu), mi(nu);
        Real scale = 0;
        for (Index j = 0; j < nu; ++j) {
            const Index b = st[static_cast<Size>(j)];
            if (cut) {
                const Index v = pool_.branch_var_id_for_inductor(b, graph_);
                xa[j] = x_prev[v];
                xb[j] = x[v];
                mi[j] = Real{1} / pool_.inductor_params(b).L;
            } else {
                const auto& br = graph_.branch(b);
                xa[j] = stamping::read_node_voltage(x_prev, br.from) -
                        stamping::read_node_voltage(x_prev, br.to);
                xb[j] = stamping::read_node_voltage(x, br.from) -
                        stamping::read_node_voltage(x, br.to);
                mi[j] = Real{1} / pool_.capacitor_params(b).C;
            }
            scale += std::abs(xa[j]) + std::abs(xb[j]);
        }
        DenseMatrix k = DenseMatrix::Zero(ng, nu);
        Index row_d = 0;
        for (Index g = 0; g < ng; ++g) {
            const auto& ds = deps[grp[static_cast<Size>(g)]];
            if (&ds == &d) row_d = g;
            for (Size j = 0; j < ds.states.size(); ++j) {
                const auto c = std::find(st.begin(), st.end(), ds.states[j]) - st.begin();
                k(g, static_cast<Index>(c)) += static_cast<Real>(ds.signs[j]);
            }
        }
        const Vector r = k * (xa - xb);
        if (r.cwiseAbs().maxCoeff() <= Real{1e-12} * scale) return;   // consistent already
        const DenseMatrix a = k * mi.asDiagonal() * k.transpose();
        const Vector lam = a.completeOrthogonalDecomposition().solve(r);   // redundant rows: least norm
        if (!lam.allFinite()) return;
        solver::ReinitRecord rec;
        rec.t = t_prev;
        rec.kind = cut ? "cut_set" : "capacitor_loop";
        for (const Size g : grp)
            for (const Index el : deps[g].elements) {
                const std::string nm(graph_.branch_name(el));
                if (std::find(rec.devices.begin(), rec.devices.end(), nm) == rec.devices.end())
                    rec.devices.push_back(nm);
            }
        for (const Index b : st) rec.states.emplace_back(graph_.branch_name(b));
        rec.lambda = lam[row_d];
        rec.dW = Real{0.5} * lam.dot(r);
        const Vector proj = xa - mi.cwiseProduct(k.transpose() * lam);
        rec.projection_gap = (proj - xb).cwiseAbs().maxCoeff();
        rec.sample = sample;
        rec.cause = cause_(d, stamped);
        if (on_impulse_error) {
            std::string sts;
            for (const auto& s : rec.states) sts += (sts.empty() ? "" : ", ") + s;
            throw UnsupportedError(
                engine, "impulse", rec.devices.front(),
                std::string(stored_kind_name(pool_.kind_of(d.elements.front()))),
                std::format("on_impulse='error': at t = {} '{}' makes {} "
                            "dependent ({}); re-initialising them moves "
                            "{:.6g} {} and removes {:.6g} J.",
                            t_prev, rec.devices.front(), sts,
                            cut ? "an inductor cut-set" : "a capacitor loop",
                            rec.lambda, cut ? "V*s" : "C", rec.dW));
        }
        result.reinit_records.push_back(std::move(rec));
    }

    /// The element of `d` whose bit changed since the last committed mode,
    /// a gated one first (the gate that entered the mode), else a diode of
    /// `d` that changed, else d's first element. Only `d`'s own elements:
    /// a switch toggling elsewhere on the same step is not the cause.
    [[nodiscard]] std::string cause_(const DependentStructure& d,
                                     const topology::SwitchStateMask& now) const {
        auto changed = [&](Index el, bool gated) {
            const long s = slot_[static_cast<std::size_t>(el)];
            return known_ && s >= 0 && pool_.is_registered(el) &&
                   (pool_.kind_of(el) != DevicePool::StoredKind::Diode) == gated &&
                   now.get(static_cast<Size>(s)) != prev_.get(static_cast<Size>(s));
        };
        for (const bool gated : {true, false})
            for (const Index el : d.elements)
                if (changed(el, gated)) return std::string(graph_.branch_name(el));
        return d.elements.empty() ? std::string{}
                                  : std::string(graph_.branch_name(d.elements.front()));
    }

    const topology::Graph& graph_;
    const DevicePool& pool_;
    std::vector<long> slot_;
    topology::SwitchStateMask prev_{Size{0}};
    bool known_ = false;
};

}  // namespace pulsim::pwl
