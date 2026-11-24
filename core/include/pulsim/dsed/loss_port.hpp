#pragma once

// =============================================================================
// Pulsim — the exact engine's loss port (add-in-kernel-event-losses 3.2)
// =============================================================================
//
// The same solver::LossTracker the pwl and trbdf2 engines use, fed from the
// event-driven scheduler's step hooks (events/step_hooks.hpp):
//
//   on_start    the run's start state and mask (LossTracker::begin);
//   on_event    every event once it is final, with its left limit (old mask)
//               and right limit (settled, projected): the switching energy by
//               the reading rule (LossTracker::event), then the impulsive
//               energy of the event's re-initialisation records, booked to
//               the commutating device (LossTracker::impulses);
//   on_accept   every accepted step [t0, t1] of one mode: the conduction
//               energy ∫v·i as the EXACT integral of the device's two affine
//               observables along the mode's closed form (period_integrals.hpp:
//               Σ αv_i αi_k h φ₁((μ_i + μ_k) h) in the augmented modal basis).
//
// A step that is not exact is never approximated: no modal basis, or a closed
// form that misses the engine's step end by more than 1e-9, refuses the run by
// name (feature "device_losses"). Refused before the run, by name: a V_on
// conduction table (not a quadratic form), time-varying sources (the closed
// form takes the mode's constant input).
//
// The junction temperature loop (a `junction=` / `heat_node=` deposit node;
// DECISIONS session 25): the heat is a current into the deposit node, and the
// thermal network is states of the same reduced vector, so each accepted step
// adds to its end state the network's EXACT response to the step's heat:
//
//   conduction  p(s) = Σ v_k(s) i_k(s) along the mode's closed form, a sum of
//               exponentials in the augmented modal basis; its response is
//               Δx = ∫₀ʰ e^{A(h−s)} b_n p(s) ds
//                  = D⁻¹ V [ w_j Σ_il a_i c_l h·E(σ_il h, μ_j h) ],
//               w = V⁻¹ D b_n, σ_il = μ_i + μ_l, E(a, b) = (eᵃ − eᵇ)/(a − b);
//   an event    its switching / impulsive energy E is an impulse at its
//               instant: Δx = b_n E on the right limit (an event booked at a
//               step's start is propagated over the step: D⁻¹ V (e^{μh} ⊙ w) E).
//
// The scheduler applies Δx after the step's hooks (PEDSimulator::
// set_state_kick); the heat reaches the network when the step is accepted, so
// LossTracker's ledger counts it injected then (take_heat: nothing pending).
// Temperature enters only the tables' T_j, read at each step's start (the
// tracker's own rule). Refused by name: a deposit node an algebraic unknown
// reads instantly (no capacitance on the heat's path).
// A resumed run restores the tracker's state (pickle field 17) in place of begin;
// the thermal states are states of the snapshot.
// With no loss port nothing here is built: the run is unchanged.

#include "pulsim/dsed/period_integrals.hpp"
#include "pulsim/events/step_hooks.hpp"
#include "pulsim/pwl/pwl_branch_law.hpp"
#include "pulsim/solver/loss_tracker.hpp"
#include "pulsim/solver/unsupported.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace pulsim::dsed {

template <class AdapterT>
class ExactLossPort {
public:
    using MaskT = topology::SwitchStateMask;

    explicit ExactLossPort(AdapterT& ad) : ad_{ad}, lt_{ad.graph(), ad.pool(), "dsed"} {
        const auto& g = ad.graph();
        const auto& pool = ad.pool();
        for (const auto& s : pool.device_losses()) {
            if (!s.deposit_node().empty()) {   // LossTracker checked the node exists
                Index node = -1;
                for (Index n = 0; n < g.num_nodes(); ++n) {
                    if (g.node(n).name == s.deposit_node()) node = n;
                }
                std::size_t slot = 0;
                while (slot < hnodes_.size() && hnodes_[slot] != node) ++slot;
                if (slot == hnodes_.size()) {
                    hnodes_.push_back(node);
                    hnames_.push_back(s.device);
                }
                if (s.conduction == thermal::ConductionHeat::Electrical) {
                    hdev_.emplace_back(names_.size(), slot);
                }
            }
            if (s.v_on) {
                refuse_(s.device, "has a V_on conduction table: its heat is not a "
                                  "quadratic form of the state, so the exact engine "
                                  "has no exact integral for it");
            }
            if (ad.has_dynamic_sources()) {
                refuse_(s.device, "is in a circuit with time-varying sources: the "
                                  "conduction integral takes each mode's constant input");
            }
            Index br = -1;
            for (Index b = 0; b < g.num_branches(); ++b) {
                if (g.branch_name(b) == s.device) br = b;
            }
            const auto law = pwl::PwlBranchLaw::of(g, pool, br);
            PeriodObservable v;
            if (law.from >= 0) v.w.emplace_back(law.from, Real{1});
            if (law.to >= 0) v.w.emplace_back(law.to, Real{-1});
            PeriodObservable i;
            if (law.var >= 0) {
                i.w.emplace_back(law.var, Real{1});
            } else {
                i.w = v.w;
                i.bit = static_cast<int>(law.bit);
                i.g_on = law.g_on;
                i.g_off = law.g_off;
                i.v_on = law.diode ? law.v_f : Real{0};
            }
            obs_.push_back(std::move(v));
            obs_.push_back(std::move(i));
            names_.push_back(s.device);
        }
        // every switch-kind branch by name: the impulse's commutating device
        for (Index b = 0; b < g.num_branches(); ++b) {
            using K = pwl::DevicePool::StoredKind;
            if (g.branch(b).kind != topology::BranchKind::Switch || !pool.is_registered(b)
                || (pool.kind_of(b) != K::Switch && pool.kind_of(b) != K::Diode)) continue;
            const auto law = pwl::PwlBranchLaw::of(g, pool, b);
            bits_[std::string(g.branch_name(b))] = {law.bit, !law.diode};
        }
        e_.resize(names_.size());
    }

    [[nodiscard]] bool active() const noexcept { return lt_.active(); }

    void attach(events::StepHooks& h) {
        h.on_start([this](const Vector& x0, const MaskT& m0) {
            if (!resumed_) lt_.begin(full_(Real{0}, x0, m0), m0);
            n_seen_ = ad_.reinit_log().size();   // the start's records: not a commutation (as pwl)
            x_last_ = x0;
            started_ = true;
        });
        h.on_event([this](Real t, const Vector& xl, const MaskT& ml, const Vector& xr,
                          const MaskT& mr) {
            if (!started_) return;   // t = 0 ticks run before the first sample
            // booked once the step that ENDS at t is accepted: the scheduler
            // fires an event inside the step it ends, before that step's hook,
            // and the tracker's next accept_step must see the step's own mask
            pend_.push_back({t, full_(t, xl, ml), ml, full_(t, xr, mr), mr, xr, {}});
            impulses_of_(ml, mr, pend_.back().imp);
        });
        h.on_accept([this](Real t0, Real t1, const Vector& x1, const MaskT& m) {
            flush_(t0);   // an event at the step's start (a zero-length gate edge)
            if (hnodes_.empty()) {
                energies_(t0, t1, x_last_, x1, m);
                lt_.accept_step(t0, t1, full_(t1, x1, m), m, nullptr, Real{0.5}, Real{0},
                                Real{0.5}, e_.data());
                x_last_ = x1;
                flush_(t1);
                return;
            }
            // the junction loop: the heat of the events at t0, then of the step
            const auto start = lt_.take_heat();
            energies_(t0, t1, x_last_, x1, m);
            Vector dx = step_heat_(m, t1 - t0, start);
            lt_.accept_step(t0, t1, full_(t1, x1 + dx, m), m, nullptr, Real{0.5}, Real{0},
                            Real{0.5}, e_.data());
            // the step's queued heat: its conduction part is in dx; the rest (a
            // flip no event reported: none expected) is an impulse at t1
            auto rest = lt_.take_heat();
            for (const auto& [k, slot] : hdev_) {
                for (auto& [node, e] : rest) {
                    if (node == hnodes_[slot]) e -= e_[k];
                }
            }
            impulse_(m, rest, dx);
            kick_l_ = dx;            // both limits at t1
            kick_r_ = dx;
            x_last_ = x1;
            flush_(t1);              // the events at t1 (x_last_: their right limit)
            impulse_(ad_.current_mask(), lt_.take_heat(), kick_r_);   // on the right limit
            x_last_ += kick_r_;      // as the scheduler corrects its state
            kicked_ = true;
        });
    }

    /// The thermal correction of the step just accepted (PEDSimulator::
    /// set_state_kick): the right limit, and the pre-event record when an
    /// event ended the step. False when there is none (no deposit node).
    bool kick(Vector& x, Vector* x_left) {
        if (!kicked_) return false;
        kicked_ = false;
        x += kick_r_;
        if (x_left != nullptr) *x_left += kick_l_;
        return true;
    }
    [[nodiscard]] bool has_heat() const noexcept { return !hnodes_.empty(); }

    /// A resumed run continues the snapshot's ledgers (pickle field 17): the
    /// tracker's state replaces `begin` at the run's start.
    void restore(const std::vector<Real>& s) {
        lt_.restore(s);
        resumed_ = true;
    }
    /// The tracker's state at the run's end (every event booked first).
    [[nodiscard]] std::vector<Real> save() {
        flush_(std::numeric_limits<Real>::infinity());
        return lt_.save();
    }

    template <class Out>
    void write(Out& out) {
        flush_(std::numeric_limits<Real>::infinity());   // events at t_end
        lt_.write(out);
    }

private:
    [[noreturn]] static void refuse_(const std::string& dev, const std::string& why) {
        throw UnsupportedError(
            "dsed", "device_losses", dev, "",
            "engine 'dsed' cannot honour the loss port of '" + dev + "' (set_device_losses): it "
            + why + ". Use engine='pwl' or 'trbdf2'.");
    }

    /// The full MNA vector of x under mask m (the hooks run after the event
    /// moved the adapter to the new mask; a left limit is read under the old).
    [[nodiscard]] Vector full_(Real t, const Vector& x, const MaskT& m) {
        if (ad_.current_mask() == m) return ad_.recover_full(t, x);
        const MaskT keep = ad_.current_mask();
        ad_.set_mask(m);
        Vector f = ad_.recover_full(t, x);
        ad_.set_mask(keep);
        return f;
    }

    /// e_[k] = ∫ v_k i_k over [t0, t1] along the mode `m` from x0, exact.
    void energies_(Real t0, Real t1, const Vector& x0, const Vector& x1, const MaskT& m) {
        const Real h = t1 - t0;
        if (!(h > Real{0})) {
            std::fill(e_.begin(), e_.end(), Real{0});
            return;
        }
        const auto& e = mode_of_(m);
        if (!e.modal) {
            refuse_(names_.front(), "has no exact conduction integral on a mode at t = "
                    + detail::fmt(t0) + " s (" + e.reason + ")");
        }
        const Index n = e.n, n1 = n + 1;
        z_ = e.Vi * e.d.cwiseProduct(detail::augment(x0)).template cast<detail::Cplx>();
        const Eigen::VectorXcd& z = z_;
        // the closed form must reproduce the engine's step end (the start was right)
        const Eigen::VectorXcd ex = (e.mu * h).array().exp().matrix();
        const Eigen::VectorXcd terms = ex.cwiseProduct(z);
        const Vector xe = (e.V * terms).real().cwiseQuotient(e.d).head(n);
        // the modal sum's own rounding: its terms cancel to the state (a start
        // from rest sums steady-state-sized terms to nanovolts)
        const Vector mag = (e.V.cwiseAbs() * terms.cwiseAbs()).cwiseQuotient(e.d).head(n);
        // a state that is exactly 0 (by symmetry) still carries the other
        // modes' rounding, u·‖x‖ (a VSI's phase current at rest next to a 25 V
        // thermal state read 4e-34): that is its floor
        const Real floor = (n > 0
                            ? std::numeric_limits<Real>::epsilon()
                              * (x1.cwiseAbs().maxCoeff() + mag.maxCoeff())
                            : Real{0}) + Real{1e-300};
        for (Index i = 0; i < n; ++i) {
            const Real sc = std::abs(x0[i]) + std::abs(x1[i]) + mag[i] + floor;
            if (std::abs(xe[i] - x1[i]) > Real{1e-9} * sc) {
                refuse_(names_.front(), "has no exact conduction integral on the step at t = "
                        + detail::fmt(t0) + " s: its closed form misses the engine's state "
                        + std::to_string(i) + " by " + detail::fmt(std::abs(xe[i] - x1[i])));
            }
        }
        P2_.resize(n1, n1);
        for (Index i = 0; i < n1; ++i) {
            for (Index j = i; j < n1; ++j) {
                P2_(i, j) = P2_(j, i) = h * phi1((e.mu[i] + e.mu[j]) * h);
            }
        }
        for (std::size_t k = 0; k < e_.size(); ++k) {
            const auto r = static_cast<Index>(2 * k);
            const Eigen::VectorXcd av = e.Rm.row(r).transpose().cwiseProduct(z);
            const Eigen::VectorXcd ai = e.Rm.row(r + 1).transpose().cwiseProduct(z);
            e_[k] = (av.transpose() * P2_ * ai)(0).real();
        }
    }

    /// The mode's augmented modal form (built once per mask).
    const detail::PeriodMode& mode_of_(const MaskT& m) {
        auto it = modes_.find(m);
        if (it == modes_.end()) {
            const MaskT keep = ad_.current_mask();
            auto e = detail::make_period_mode(ad_, m, obs_, true);
            ad_.set_mask(keep);
            it = modes_.emplace(m, std::move(e)).first;
        }
        return it->second;
    }

    /// Per deposit node, the mode's reduced input of a unit current into it
    /// (b) and, with a modal form, its modal coordinates w = V⁻¹ D [b; 0].
    struct HeatIn {
        std::vector<Vector> b;
        std::vector<Eigen::VectorXcd> w;
    };
    const HeatIn& heat_in_(const MaskT& m) {
        auto it = heat_.find(m);
        if (it != heat_.end()) return it->second;
        const auto& e = mode_of_(m);
        const MaskT keep = ad_.current_mask();
        ad_.set_mask(m);
        HeatIn hi;
        const Real scale = std::max(Real{1}, ad_.recover_from_b_scale());
        for (std::size_t s = 0; s < hnodes_.size(); ++s) {
            auto [b, alg] = ad_.node_current_input(hnodes_[s]);
            if (alg > Real{1e-12} * scale) {
                ad_.set_mask(keep);
                refuse_(hnames_[s], "deposits its heat into a node an algebraic unknown reads "
                                    "instantly (a resistive path with no thermal capacitance), "
                                    "so the heat's response is not a state's; give each node of "
                                    "the thermal network a capacitor");
            }
            if (e.modal) {
                Vector ba = Vector::Zero(e.n + 1);
                ba.head(e.n) = b;
                hi.w.push_back(e.Vi * e.d.cwiseProduct(ba).template cast<detail::Cplx>());
            }
            hi.b.push_back(std::move(b));
        }
        ad_.set_mask(keep);
        return heat_.emplace(m, std::move(hi)).first->second;
    }

    /// The deposit nodes' slot of MNA node `node`.
    [[nodiscard]] std::size_t slot_(Index node) const {
        std::size_t s = 0;
        while (s < hnodes_.size() && hnodes_[s] != node) ++s;
        return s;
    }

    /// Heat delivered at an instant: Δx += b_n E under mode m.
    void impulse_(const MaskT& m, const std::vector<std::pair<Index, Real>>& heat, Vector& dx) {
        if (heat.empty()) return;
        const auto& hi = heat_in_(m);
        for (const auto& [node, E] : heat) {
            if (E != Real{0}) dx += hi.b[slot_(node)] * E;
        }
    }

    /// The thermal response at the step's end to the step's conduction heat
    /// (the devices' exact v·i along the mode, energies_ set z_) and to the
    /// events booked at its start (impulses at t0, propagated over the step).
    Vector step_heat_(const MaskT& m, Real h, const std::vector<std::pair<Index, Real>>& start) {
        Vector dx = Vector::Zero(x_last_.size());
        if (!(h > Real{0})) {   // no step to propagate over: the impulses as they are
            impulse_(m, start, dx);
            return dx;
        }
        const auto& e = mode_of_(m);   // modal (energies_ refused it otherwise)
        const auto& hi = heat_in_(m);
        const Index n = e.n, n1 = n + 1;
        using C = detail::Cplx;
        Eigen::VectorXcd g = Eigen::VectorXcd::Zero(n1);
        const Eigen::VectorXcd ex = (e.mu * h).array().exp().matrix();
        for (const auto& [node, E] : start) {
            if (E != Real{0}) g += E * ex.cwiseProduct(hi.w[slot_(node)]);
        }
        // ∫₀ʰ e^{μ_j (h−s)} e^{σ s} ds = h E(σh, μ_j h), the exponent kept ≤ the larger
        const auto kernel = [h](C a, C b) {
            return a.real() >= b.real() ? h * std::exp(a) * phi1(b - a)
                                        : h * std::exp(b) * phi1(a - b);
        };
        for (std::size_t s = 0; s < hnodes_.size(); ++s) {
            Eigen::MatrixXcd Q = Eigen::MatrixXcd::Zero(n1, n1);
            bool any = false;
            for (const auto& [k, sl] : hdev_) {
                if (sl != s) continue;
                const auto r = static_cast<Index>(2 * k);
                const Eigen::VectorXcd a = e.Rm.row(r).transpose().cwiseProduct(z_);
                const Eigen::VectorXcd c = e.Rm.row(r + 1).transpose().cwiseProduct(z_);
                Q.noalias() += a * c.transpose();
                any = true;
            }
            if (!any) continue;
            const auto& w = hi.w[s];
            for (Index j = 0; j < n1; ++j) {
                if (w[j] == C{0}) continue;
                const C bj = e.mu[j] * h;
                C acc{0};
                for (Index i = 0; i < n1; ++i) {
                    for (Index l = 0; l < n1; ++l) {
                        if (Q(i, l) != C{0}) acc += Q(i, l) * kernel((e.mu[i] + e.mu[l]) * h, bj);
                    }
                }
                g[j] += w[j] * acc;
            }
        }
        dx += (e.V * g).real().head(n).cwiseQuotient(e.d.head(n));
        return dx;
    }

    /// Book the pending events at or before t, in order.
    void flush_(Real t) {
        std::size_t k = 0;
        for (; k < pend_.size() && pend_[k].t <= t; ++k) {
            auto& e = pend_[k];
            lt_.event(e.t, e.xl, e.ml, e.xr, e.mr);
            imp_.insert(imp_.end(), e.imp.begin(), e.imp.end());
            lt_.impulses(imp_);
            x_last_ = e.xr_red;   // the next step starts from the event's right limit
        }
        pend_.erase(pend_.begin(), pend_.begin() + static_cast<std::ptrdiff_t>(k));
    }

    struct Impulse {
        std::string cause;
        Real dW = 0;
    };

    /// The records the event appended, each booked to its commutating
    /// device: the gated switch of its structure that changed, else any
    /// element of it that changed, else its first element (the fixed
    /// engine's `cause`, solver/result.hpp).
    void impulses_of_(const MaskT& ml, const MaskT& mr, std::vector<Impulse>& out) {
        const auto& log = ad_.reinit_log();
        for (; n_seen_ < log.size(); ++n_seen_) {
            const auto& r = log[n_seen_];
            std::string cause, changed;
            for (const auto& dev : r.devices) {
                const auto b = bits_.find(dev);
                if (b == bits_.end()) continue;
                const auto bit = static_cast<Size>(b->second.first);
                if (ml.get(bit) == mr.get(bit)) continue;
                if (b->second.second) { cause = dev; break; }
                if (changed.empty()) changed = dev;
            }
            if (cause.empty()) cause = !changed.empty() ? changed
                                       : r.devices.empty() ? std::string{} : r.devices.front();
            out.push_back({cause, r.dW});
        }
    }

    struct Pending {
        Real t;
        Vector xl;
        MaskT ml;
        Vector xr;
        MaskT mr;
        Vector xr_red;   // the right limit in the scheduler's (reduced) state
        std::vector<Impulse> imp;
    };

    AdapterT& ad_;
    solver::LossTracker lt_;
    std::vector<PeriodObservable> obs_;          // (v, i) per tracked device
    std::vector<std::string> names_;
    std::map<std::string, std::pair<long, bool>> bits_;   // name -> (bit, gated switch)
    std::map<MaskT, detail::PeriodMode> modes_;
    std::vector<Real> e_;
    std::vector<Impulse> imp_;   // every impulse booked (LossTracker counts them)
    std::vector<Pending> pend_;
    std::size_t n_seen_ = 0;
    Eigen::MatrixXcd P2_;
    Vector x_last_;   // the next step's start: the last step's end or an event's right limit
    bool started_ = false;
    bool resumed_ = false;
    // the junction loop
    std::vector<Index> hnodes_;                               // distinct deposit nodes (MNA)
    std::vector<std::string> hnames_;                         // a device depositing at each
    std::vector<std::pair<std::size_t, std::size_t>> hdev_;   // (device k, its node's slot): conduction heat
    std::map<MaskT, HeatIn> heat_;
    Eigen::VectorXcd z_;      // the step's modal start (energies_)
    Vector kick_l_, kick_r_;  // the step's thermal correction: pre-event record, right limit
    bool kicked_ = false;
};

}  // namespace pulsim::dsed
