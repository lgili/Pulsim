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
// The modulators (`CarrierPwm`, `DutyRegister`, `EdgeList`) live in
// events/modulator.hpp. `CarrierPwm` is a sawtooth carrier compared with a duty held in a
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

#include "pulsim/events/modulator.hpp"
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

/// Today's helpers as a gate source: the schedule's mask, restricted to
/// the bits it is declared to own; its edges unchanged.
struct ScheduleGate {
    sources::SwitchSchedule sched;
    std::vector<Index> bits;

    [[nodiscard]] std::vector<Index> owned_bits() const { return bits; }
    // No edge oracle: not exact, so the set states no edges either.
    [[nodiscard]] Real resolution() const noexcept {
        return sched.has_edges() ? Real{0}
                                 : std::numeric_limits<Real>::infinity();
    }
    void check(const std::string&) const {}
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
        gate.check(name);
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
