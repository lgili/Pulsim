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
// name (feature "device_losses"). Refused before the run, by name: a junction
// or heat node (the temperature loop: the exact step has no input for the
// heat; DECISIONS session 23), a V_on conduction table (not a quadratic form),
// time-varying sources (the closed form takes the mode's constant input).
// A resumed run restores the tracker's state (pickle field 17) in place of begin.
// With no loss port nothing here is built: the run is unchanged.

#include "pulsim/dsed/period_integrals.hpp"
#include "pulsim/events/step_hooks.hpp"
#include "pulsim/pwl/pwl_branch_law.hpp"
#include "pulsim/solver/loss_tracker.hpp"
#include "pulsim/solver/unsupported.hpp"

#include <cmath>
#include <limits>
#include <map>
#include <string>
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
            if (!s.deposit_node().empty()) {
                refuse_(s.device, "deposits its heat into node '" + s.deposit_node()
                        + "': the exact engine's step has no input for the junction "
                          "heat (the temperature loop runs on engine='pwl' and "
                          "'trbdf2'); a fixed Tj= runs here");
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
            energies_(t0, t1, x_last_, x1, m);
            lt_.accept_step(t0, t1, full_(t1, x1, m), m, nullptr, Real{0.5}, Real{0},
                            Real{0.5}, e_.data());
            x_last_ = x1;
            flush_(t1);
        });
    }

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
        auto it = modes_.find(m);
        if (it == modes_.end()) {
            const MaskT keep = ad_.current_mask();
            auto e = detail::make_period_mode(ad_, m, obs_, true);
            ad_.set_mask(keep);
            it = modes_.emplace(m, std::move(e)).first;
        }
        const auto& e = it->second;
        if (!e.modal) {
            refuse_(names_.front(), "has no exact conduction integral on a mode at t = "
                    + detail::fmt(t0) + " s (" + e.reason + ")");
        }
        const Index n = e.n, n1 = n + 1;
        const Eigen::VectorXcd z =
            e.Vi * e.d.cwiseProduct(detail::augment(x0)).template cast<detail::Cplx>();
        // the closed form must reproduce the engine's step end (the start was right)
        const Eigen::VectorXcd ex = (e.mu * h).array().exp().matrix();
        const Eigen::VectorXcd terms = ex.cwiseProduct(z);
        const Vector xe = (e.V * terms).real().cwiseQuotient(e.d).head(n);
        // the modal sum's own rounding: its terms cancel to the state (a start
        // from rest sums steady-state-sized terms to nanovolts)
        const Vector mag = (e.V.cwiseAbs() * terms.cwiseAbs()).cwiseQuotient(e.d).head(n);
        for (Index i = 0; i < n; ++i) {
            const Real sc = std::abs(x0[i]) + std::abs(x1[i]) + mag[i] + Real{1e-300};
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
};

}  // namespace pulsim::dsed
