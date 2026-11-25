#pragma once

// =============================================================================
// Pulsim — modulators: a carrier PWM reading a duty register, and an edge
// list (add-unified-event-queue D3, task 6.1)
// =============================================================================
//
// Gate sources in the shape of events/gate_source.hpp (owned_bits, apply,
// next_edge, resolution, check); a `GateSet` composes them by switch index.
// Light on purpose: the control graph (blockchain/chain.hpp) includes it.

#include "pulsim/numeric/pwm_phase.hpp"
#include "pulsim/numeric/types.hpp"
#include "pulsim/sources/pwm_leg.hpp"
#include "pulsim/topology/switch_state.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace pulsim::events {

enum class LatchPolicy : std::uint8_t { Immediate, AtCarrierStart };

/// A duty a controller writes and a modulator reads. `AtCarrierStart`
/// latches a write at the first start, strictly after the write, of the
/// carrier (`T`, `phase`) the register is given.
struct DutyRegister {
    Real value = Real{0};
    LatchPolicy latch = LatchPolicy::Immediate;
    Real T = Real{0};        // AtCarrierStart: the latching carrier
    Real phase = Real{0};
    Real shadow = Real{0};
    Real latch_k = Real{0};  // carrier period index from which shadow holds
    bool pending = false;

    /// `t` is the write's SCHEDULED instant (a tick handler passes
    /// `TickInfo::scheduled_t`): an engine may land a tick slightly early,
    /// and the landed time could fall in the previous carrier period.
    void write(Real v, Real t) {
        if (latch == LatchPolicy::Immediate) {
            value = v;
            return;
        }
        if (!(T > Real{0})) {
            throw std::invalid_argument(
                "DutyRegister: AtCarrierStart needs the carrier period T");
        }
        // Period index with the carrier's own snapping, so the latch and
        // the modulator agree on which period an instant belongs to.
        const Real k = pwm_phase::cycle_position(t, phase, T).k;
        if (pending && k >= latch_k) value = shadow;   // the older write latched
        shadow = v;
        latch_k = k + Real{1};
        pending = true;
    }
    /// The duty in force at t.
    [[nodiscard]] Real read(Real t) const noexcept {
        if (!pending) return value;
        return pwm_phase::cycle_position(t, phase, T).k >= latch_k ? shadow
                                                                    : value;
    }
};

/// One switch (`ls` = -1) or a complementary leg with dead time, driven by
/// a sawtooth carrier of period `T` starting at `phase`.
struct CarrierPwm {
    Real T;
    Real phase = Real{0};
    const DutyRegister* duty;
    Index hs;
    Index ls = -1;
    Real dead_time = Real{0};

    [[nodiscard]] std::vector<Index> owned_bits() const {
        return ls < 0 ? std::vector<Index>{hs} : std::vector<Index>{hs, ls};
    }
    [[nodiscard]] Real resolution() const noexcept { return Real{0}; }

    [[nodiscard]] Real duty_at(Real t) const { return duty->read(t); }

    /// Refuses what would silently misbehave: no register, T <= 0, a leg
    /// on one bit, and an AtCarrierStart register latching on another
    /// carrier (its latch would fall between this carrier's edges).
    /// The register must outlive every schedule built from this source.
    void check(const std::string& name) const {
        const auto bad = [&](const char* why) {
            throw std::invalid_argument("CarrierPwm '" + name + "': " + why);
        };
        if (duty == nullptr) bad("no DutyRegister");
        if (!(T > Real{0})) bad("carrier period T must be > 0");
        if (hs == ls) bad("hs and ls are the same switch bit");
        if (duty->latch == LatchPolicy::AtCarrierStart
            && (duty->T != T || duty->phase != phase)) {
            bad("its AtCarrierStart DutyRegister latches on a different "
                "carrier (T and phase must equal the modulator's)");
        }
    }

    void apply(Real t, topology::SwitchStateMask& m) const {
        const Real d = duty_at(t);
        if (std::isnan(d)) {
            // A NaN duty (a diverged controller) holds the switch OFF, as the
            // graph gate did before this move: `!reached` is true for NaN.
            m.set(static_cast<Size>(hs), false);
            if (ls >= 0) m.set(static_cast<Size>(ls), false);
            return;
        }
        const auto p = pwm_phase::cycle_position(t, phase, T);
        if (ls < 0) {
            m.set(static_cast<Size>(hs), !pwm_phase::reached(p, d));
            return;
        }
        const auto th = thresholds(d);
        m.set(static_cast<Size>(hs), false);
        m.set(static_cast<Size>(ls), false);
        sources::detail::stamp_leg(m, sources::detail::pair_state(p, th),
                                   static_cast<Size>(hs),
                                   static_cast<Size>(ls));
    }

    /// How this modulator's edges move with the duty: {rising, falling}
    /// d(edge instant)/d(duty). A single-switch leg turns ON at the period
    /// start phase + kT (does not move) and OFF where the carrier position
    /// reaches d, at phase + kT + d*T (moves by T). A dead-time pair: NaN
    /// (its thresholds are not derived here). Used by the sensitivity of
    /// a closed loop (the edge-time term of the event jump).
    [[nodiscard]] std::array<Real, 2> edge_duty_derivative() const noexcept {
        if (ls >= 0) {
            const Real nan = std::numeric_limits<Real>::quiet_NaN();
            return {nan, nan};
        }
        return {Real{0}, T};
    }

    [[nodiscard]] Real next_edge(Real t) const {
        const Real d = duty_at(t);
        if (ls < 0) {
            const std::array<Real, 1> thr{d};
            return pwm_phase::next_constant_crossing(t, phase, T, thr);
        }
        const auto th = thresholds(d);
        const std::array<Real, 3> thr{th.hs_end, th.ls_start, th.ls_end};
        return pwm_phase::next_constant_crossing(t, phase, T, thr);
    }

private:
    // make_dead_time_pwm_pair_fn's thresholds, computed the same way.
    [[nodiscard]] sources::detail::PairThresholds thresholds(Real d) const {
        const Real dt_eff =
            std::max(Real{0}, std::min(dead_time, T * Real{0.5}));
        const Real dd = dt_eff / T;
        return {d - dd, d, Real{1} - dd};
    }
};

/// Gate edges given as a list of (t, bit, on): bit `bit` takes `on` at t
/// and holds it until its next edge; OFF before its first. Edges at one
/// instant on one bit: the last in list order wins. Sorted (stably) and
/// indexed at construction, the only allocation.
struct EdgeList {
    struct Edge {
        Real t;
        Index bit;
        bool on;
    };

    EdgeList() = default;
    explicit EdgeList(std::vector<Edge> edges) {
        std::stable_sort(edges.begin(), edges.end(),
                         [](const Edge& a, const Edge& b) { return a.t < b.t; });
        for (const auto& e : edges) {
            if (std::find(bits_.begin(), bits_.end(), e.bit) == bits_.end()) {
                bits_.push_back(e.bit);
            }
        }
        std::sort(bits_.begin(), bits_.end());
        per_bit_.resize(bits_.size());
        for (const auto& e : edges) {
            const auto j = static_cast<Size>(
                std::lower_bound(bits_.begin(), bits_.end(), e.bit) - bits_.begin());
            per_bit_[j].t.push_back(e.t);
            per_bit_[j].on.push_back(e.on ? 1 : 0);
            times_.push_back(e.t);
        }
    }

    [[nodiscard]] std::vector<Index> owned_bits() const { return bits_; }
    [[nodiscard]] Real resolution() const noexcept { return Real{0}; }

    /// Refuses a non-finite instant and a negative bit.
    void check(const std::string& name) const {
        for (const Real t : times_) {
            if (!std::isfinite(t)) {
                throw std::invalid_argument("EdgeList '" + name
                                            + "': an edge time is not finite");
            }
        }
        if (!bits_.empty() && bits_.front() < 0) {
            throw std::invalid_argument("EdgeList '" + name
                                        + "': a switch bit is negative");
        }
    }

    void apply(Real t, topology::SwitchStateMask& m) const {
        for (Size j = 0; j < bits_.size(); ++j) {
            const auto& b = per_bit_[j];
            const auto k = std::upper_bound(b.t.begin(), b.t.end(), t) - b.t.begin();
            m.set(static_cast<Size>(bits_[j]), k > 0 && b.on[static_cast<Size>(k - 1)] != 0);
        }
    }

    [[nodiscard]] Real next_edge(Real t) const {
        const auto it = std::upper_bound(times_.begin(), times_.end(), t);
        return it == times_.end() ? std::numeric_limits<Real>::infinity() : *it;
    }

private:
    struct BitEdges {
        std::vector<Real> t;
        std::vector<char> on;
    };
    std::vector<Index> bits_;
    std::vector<BitEdges> per_bit_;
    std::vector<Real> times_;   // every edge, sorted
};

}  // namespace pulsim::events
