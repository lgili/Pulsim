// add-unified-event-queue group 3 on the exact engine (PEDSimulator):
// control ticks from the run's event queue, monitors that never perturb,
// and the step hooks' commutation callback (the loss tracker's task 3.1).

#include <catch2/catch_test_macros.hpp>

#include "pulsim/builder/circuit_builder.hpp"
#include "pulsim/dsed/builder_adapter.hpp"
#include "pulsim/dsed/scheduler.hpp"
#include "pulsim/events/event_queue.hpp"
#include "pulsim/events/step_hooks.hpp"
#include "pulsim/models/switched_diode.hpp"
#include "pulsim/pwl/cache.hpp"
#include "pulsim/pwl/diode_event_state.hpp"
#include "pulsim/pwl/row_names.hpp"
#include "pulsim/topology/switch_state.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

using namespace pulsim;
using namespace pulsim::dsed;

namespace {

// A PWM on one bit with declared edges; a tick writes `duty`.
struct Pwm {
    Size n;
    Index bit;
    Real T;
    Real* duty;
    [[nodiscard]] topology::SwitchStateMask operator()(Real t) const {
        topology::SwitchStateMask m(n);
        const Real u = t / T - std::floor(t / T);
        m.set(static_cast<Size>(bit), u < *duty);
        return m;
    }
    [[nodiscard]] Real next_edge_after(Real t) const {
        const Real k = std::floor(t / T);
        for (const Real c : {(k + *duty) * T, (k + 1) * T, (k + 1 + *duty) * T}) {
            if (c > t) return c;
        }
        return (k + 2) * T;
    }
};

struct Run {
    PEDResult<topology::SwitchStateMask> res;
    std::vector<events::TickInfo> ticks;
    std::vector<Real> tick_t;
    std::vector<std::uint64_t> counts;
    Size commutations = 0;
    std::vector<Real> commutation_t;
};

// Buck: 24 V, switch, PWL freewheel diode, 1 mH, 47 uF || 10 ohm.
Run run_buck(Real t_end, std::vector<events::Rate> rates, std::vector<bool> monitor,
             bool hooks_on, bool write_duty) {
    builder::CircuitBuilder b;
    b.add_voltage_source("Vin", "in", "gnd", 24.0);
    b.add_switch("Q", "in", "sw", 1e2, 1e-9);
    b.add_diode("D", "gnd", "sw", 1e2, 1e-9);
    b.add_inductor("L", "sw", "out", 1e-3);
    b.add_capacitor("C", "out", "gnd", 47e-6);
    b.add_resistor("R", "out", "gnd", 10.0);
    pwl::PwlStateSpaceCache cache{b.graph(), b.pool()};
    using Adapter = NativeCircuitBuilderAdapter<>;
    Adapter ad{b.graph(), b.pool(), cache};
    EventPredictor pred;
    std::vector<SchedulerDiode> diodes;
    pwl::DiodeEventState census(ad.graph(), ad.pool());
    const topology::SwitchStateMask owned = census.diode_owned_bits();
    for (const auto& d : census.entries()) {
        std::string label = pwl::branch_label(ad.graph(), d.branch_id);
        const Adapter* a = &ad;
        const Index from = d.from;
        const Index to = d.to;
        const auto prm = d.params();
        for (const bool on : {false, true}) {
            const auto type = on ? PredicateType::DiodeOff : PredicateType::DiodeOn;
            pred.add_predicate(EventPredicate{
                .name = label + (on ? " turns OFF" : " turns ON"),
                .fn = [a, from, to, prm, on](Real t, const Vector& x) {
                    return models::SwitchedDiode::crossing_signal(
                        on, a->node_pair_voltage(from, to, t, x), prm);
                },
                .type = type,
                .priority = default_priority(type),
                .aux_index = static_cast<int>(d.switch_idx),
                .required_bit = on ? 1 : 0,
            });
        }
        diodes.push_back(SchedulerDiode{d.switch_idx, d.from, d.to, d.g_on, d.g_off,
                                        d.V_th, d.V_f, std::move(label)});
    }
    const auto n_sw = static_cast<Size>(ad.graph().num_switches());
    Real duty = 0.5;
    Pwm pwm{n_sw, b.switch_index_of("Q"), 1e-5, &duty};
    PEDSimulator<Adapter, Pwm> sim(ad, pwm, PIController{1e-6, 1e-9}, std::move(pred),
                                   Real{1e-9}, Real{1e-5}, 1);
    sim.enable_diode_commutation(owned, topology::SwitchStateMask(n_sw), std::move(diodes));
    sim.set_event_projection(true);
    Run out;
    events::StepHooks hooks;
    if (hooks_on) {
        hooks.on_commutation([&](const events::Commutation& c) {
            out.commutation_t.push_back(c.t);
        });
        sim.set_hooks(&hooks);
    }
    if (!rates.empty()) {
        events::TickPlan plan;
        plan.rates = std::move(rates);
        plan.monitor = std::move(monitor);
        plan.fn = [&](std::uint32_t j, const events::TickInfo& info, Real t, const Vector&) {
            if (j == 0) {
                out.ticks.push_back(info);
                out.tick_t.push_back(t);
            }
            if (write_duty && j == 0) duty = info.k % 2 == 0 ? 0.3 : 0.6;
        };
        sim.set_ticks(std::move(plan));
    }
    ad.set_mask(pwm(0.0));
    out.res = sim.simulate(Vector::Zero(ad.n_state()), t_end);
    out.counts = sim.tick_counts();
    out.commutations = hooks.commutations();
    return out;
}

bool same_bytes(const std::vector<Real>& a, const std::vector<Real>& b) {
    return a.size() == b.size()
        && (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(Real)) == 0);
}

bool same_states(const std::vector<Vector>& a, const std::vector<Vector>& b) {
    if (a.size() != b.size()) return false;
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (a[i].size() != b[i].size()
            || std::memcmp(a[i].data(), b[i].data(), sizeof(Real) * a[i].size()) != 0) {
            return false;
        }
    }
    return true;
}

}  // namespace

TEST_CASE("Exact engine: control ticks come from the queue at k*T, ceil(span/T) of them",
          "[v2][dsed][event_queue]") {
    const Real T = 1e-4 / 3.0;
    const Run r = run_buck(1e-3, {events::Rate{T}}, {}, false, true);
    REQUIRE(r.ticks.size() == 30);                  // ceil(1e-3 / T)
    REQUIRE(r.counts == std::vector<std::uint64_t>{30});
    for (std::size_t k = 0; k < r.ticks.size(); ++k) {
        CHECK(r.ticks[k].k == k);
        // integer tick arithmetic: (0 + 0) + k*T, exactly
        CHECK(r.ticks[k].scheduled_t == Real{0} + static_cast<Real>(k) * T);
        // the step lands on it to the engine's time tolerance
        CHECK(std::abs(r.tick_t[k] - r.ticks[k].scheduled_t) <= 1e-12);
    }
}

TEST_CASE("Exact engine: a monitor changes nothing the run computes",
          "[v2][dsed][event_queue]") {
    const Run ref = run_buck(1e-3, {}, {}, false, false);
    const Run mon = run_buck(1e-3, {events::Rate{1e-4 / 3.0}}, {true}, false, false);
    REQUIRE(mon.counts == std::vector<std::uint64_t>{30});
    CHECK(same_bytes(mon.res.times, ref.res.times));
    CHECK(same_states(mon.res.states, ref.res.states));
    // control: the same rate as a controller lands its ticks
    const Run ctl = run_buck(1e-3, {events::Rate{1e-4 / 3.0}}, {false}, false, false);
    CHECK_FALSE(same_bytes(ctl.res.times, ref.res.times));
}

TEST_CASE("Exact engine: on_commutation fires at every change of the integrated mask",
          "[v2][dsed][event_queue]") {
    const Run r = run_buck(1e-3, {events::Rate{1e-5}}, {}, true, true);
    // the result's own record: the mask each recorded step was integrated under
    const auto& m = r.res.step_masks;
    REQUIRE(m.size() == r.res.times.size());
    Size changes = 0;
    std::vector<Real> t_changes;
    for (std::size_t i = 1; i < m.size(); ++i) {
        if (!(m[i] == m[i - 1])) {
            ++changes;
            t_changes.push_back(r.res.times[i - 1]);
        }
    }
    REQUIRE(changes > 100);           // gate edges and diode turn-offs over 100 periods
    CHECK(r.commutations == changes);
    CHECK(r.commutation_t == t_changes);
}
