#pragma once

// =============================================================================
// Pulsim — gate sources composed by switch index, and the carrier PWM
// modulator with a duty register (add-unified-event-queue D3, tasks 2.2/6.1)
// =============================================================================
//
// A gate source owns a set of switch bits, writes only those bits of a
// mask (`apply`), and states its next edge strictly after t (`next_edge`,
// +inf when none; a no-op edge is allowed, a missed one is not). A
// `GateSet` composes sources BY INDEX: two sources owning one bit is an
// error naming the bit and both sources, and a bit no source owns stays
// OFF (there is no `mask | mask` and no default-mask lambda). The set is
// a `SwitchSchedule` for the engines (`to_schedule`), so every engine
// lands its edges through the event queue as it does today's helpers'.
//
// `CarrierPwm` is a sawtooth carrier compared with a duty held in a
// `DutyRegister` (one switch, or a complementary leg with dead time). Its
// arithmetic is that of `make_pwm_switch_fn` / `make_dead_time_pwm_pair_fn`
// (pwm_phase, the same thresholds), so for a constant duty its masks and
// edges are the helpers' exactly. A tick writes the register; the edge it
// stated before the write is stale, so the engine invalidates the source
// (trbdf2: `EventQueue::invalidate`; pwl: the observer call resets its edge
// cache). `LatchPolicy::Immediate` takes a written duty at once (what a
// closure reading the controller's variable does today);
// `AtCarrierStart` holds it until the next carrier period starts.
//
// shortcut: sawtooth carrier only (no triangle); add a Carrier enum when a
// centre-aligned modulator is needed.

#include "pulsim/numeric/pwm_phase.hpp"
#include "pulsim/numeric/types.hpp"
#include "pulsim/sources/switch_schedule.hpp"
#include "pulsim/topology/switch_state.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
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

    void apply(Real t, topology::SwitchStateMask& m) const {
        const Real d = duty_at(t);
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

/// Today's helpers as a gate source: the schedule's mask, restricted to
/// the bits it is declared to own; its edges unchanged.
struct ScheduleGate {
    sources::SwitchSchedule sched;
    std::vector<Index> bits;

    [[nodiscard]] std::vector<Index> owned_bits() const { return bits; }
    [[nodiscard]] Real resolution() const noexcept { return Real{0}; }
    void apply(Real t, topology::SwitchStateMask& m) const {
        const auto s = sched(t);
        for (const Index b : bits) {
            m.set(static_cast<Size>(b), s.get(static_cast<Size>(b)));
        }
    }
    [[nodiscard]] Real next_edge(Real t) const {
        return sched.has_edges() ? sched.next_edge_after(t)
                                 : std::numeric_limits<Real>::infinity();
    }
};

/// Gate sources composed by switch index (D3).
class GateSet {
public:
    explicit GateSet(Size num_switches)
        : n_(num_switches), owner_(num_switches, -1) {}

    /// Adds a source under `name`; refuses a bit outside the circuit or
    /// one another source owns, naming both sources.
    template <class G>
    void add(G gate, std::string name) {
        const auto bits = gate.owned_bits();
        const int id = static_cast<int>(names_.size());
        for (const Index b : bits) {
            if (b < 0 || static_cast<Size>(b) >= n_) {
                throw std::invalid_argument(
                    "GateSet: gate source '" + name + "' drives switch bit "
                    + std::to_string(b) + ", outside the circuit's "
                    + std::to_string(n_) + " switches");
            }
            if (owner_[static_cast<Size>(b)] >= 0) {
                throw std::invalid_argument(
                    "GateSet: switch bit " + std::to_string(b)
                    + " is driven by two gate sources, '"
                    + names_[static_cast<Size>(owner_[static_cast<Size>(b)])]
                    + "' and '" + name + "'");
            }
        }
        for (const Index b : bits) owner_[static_cast<Size>(b)] = id;
        names_.push_back(std::move(name));
        exact_ = exact_ && gate.resolution() == Real{0};
        auto g = std::make_shared<const G>(std::move(gate));
        apply_.push_back([g](Real t, topology::SwitchStateMask& m) {
            g->apply(t, m);
        });
        next_.push_back([g](Real t) { return g->next_edge(t); });
    }

    [[nodiscard]] topology::SwitchStateMask mask(Real t) const {
        topology::SwitchStateMask m(n_);
        for (const auto& a : apply_) a(t, m);
        return m;
    }
    [[nodiscard]] Real next_edge(Real t) const {
        Real best = std::numeric_limits<Real>::infinity();
        for (const auto& f : next_) best = std::min(best, f(t));
        return best;
    }
    /// Index of the source owning bit b, -1 for an undriven switch.
    [[nodiscard]] int owner(Size b) const { return owner_.at(b); }

    /// The set as an engine schedule (edges only when every source is
    /// exact; an inexact one is located by the engine's sampling).
    [[nodiscard]] sources::SwitchSchedule to_schedule() const {
        auto self = std::make_shared<const GateSet>(*this);
        std::function<Real(Real)> ne;
        if (exact_) ne = [self](Real t) { return self->next_edge(t); };
        return sources::SwitchSchedule{
            [self](Real t) { return self->mask(t); }, std::move(ne)};
    }

private:
    Size n_;
    std::vector<int> owner_;
    std::vector<std::string> names_;
    std::vector<std::function<void(Real, topology::SwitchStateMask&)>> apply_;
    std::vector<std::function<Real(Real)>> next_;
    bool exact_ = true;
};

}  // namespace pulsim::events
