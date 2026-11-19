// add-exact-propagator-strategy session 24: snapshot and resume on the exact
// engine. ExactResumeState round-trips through its flat payload (refusing a
// truncated, padded or foreign one), and a buck split at a gate edge or
// mid-interval, with control ticks on the queue, continues the uninterrupted
// run (states at the common sample instants within 1e-12 of the scale, the
// tick counts summing to the uninterrupted run's).

#include <catch2/catch_test_macros.hpp>

#include "pulsim/builder/circuit_builder.hpp"
#include "pulsim/dsed/builder_adapter.hpp"
#include "pulsim/dsed/scheduler.hpp"
#include "pulsim/dsed/scheduler_auto.hpp"
#include "pulsim/events/event_queue.hpp"
#include "pulsim/models/switched_diode.hpp"
#include "pulsim/pwl/cache.hpp"
#include "pulsim/pwl/diode_event_state.hpp"
#include "pulsim/pwl/row_names.hpp"
#include "pulsim/topology/switch_state.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

using namespace pulsim;
using namespace pulsim::dsed;

namespace {

using Mask = topology::SwitchStateMask;
using Resume = ExactResumeState<Mask>;

struct Pwm {
    Size n;
    Index bit;
    Real T;
    [[nodiscard]] Mask operator()(Real t) const {
        Mask m(n);
        const Real u = t / T - std::floor(t / T);
        m.set(static_cast<Size>(bit), u < Real{0.5});
        return m;
    }
    [[nodiscard]] Real next_edge_after(Real t) const {
        const Real k = std::floor(t / T);
        for (const Real c : {(k + Real{0.5}) * T, (k + 1) * T, (k + Real{1.5}) * T}) {
            if (c > t) return c;
        }
        return (k + 2) * T;
    }
};

struct Out {
    PEDResult<Mask> res;
    Resume end;
    std::vector<std::uint64_t> ticks, next_k;
    std::vector<Real> origin;
};

// Buck (24 V, PWL freewheel diode, 100 uH, 47 uF || R) at 100 kHz, D = 0.5, a
// 30 kHz control tick (its instants fall between the gate edges).
Out run(Real r_load, Real t_end, const Resume* rs = nullptr, std::vector<Real> origin = {},
        std::vector<std::uint64_t> k0 = {}) {
    builder::CircuitBuilder b;
    b.add_voltage_source("Vin", "in", "gnd", 24.0);
    b.add_switch("Q", "in", "sw", 1e3, 1e-9);
    b.add_diode("D", "gnd", "sw", 1e3, 1e-9);
    b.add_inductor("L", "sw", "out", r_load > 50 ? 10e-6 : 100e-6);
    b.add_capacitor("C", "out", "gnd", 47e-6);
    b.add_resistor("R", "out", "gnd", r_load);
    pwl::PwlStateSpaceCache cache{b.graph(), b.pool()};
    using Adapter = NativeCircuitBuilderAdapter<>;
    Adapter ad{b.graph(), b.pool(), cache};
    EventPredictor pred;
    std::vector<SchedulerDiode> diodes;
    pwl::DiodeEventState census(ad.graph(), ad.pool());
    const Mask owned = census.diode_owned_bits();
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
    Pwm pwm{n_sw, b.switch_index_of("Q"), 1e-5};
    PEDSimulator<Adapter, Pwm> sim(ad, pwm, PIController{1e-6, 1e-9}, std::move(pred),
                                   Real{1e-9}, Real{1e-5}, 1);
    sim.enable_diode_commutation(owned, Mask(n_sw), std::move(diodes));
    sim.set_event_projection(true);
    events::TickPlan plan;
    events::Rate rate{Real{1} / Real{30e3}, Real{0}};
    if (!origin.empty()) rate.origin = origin[0];
    if (!k0.empty()) rate.k0 = k0[0];
    plan.rates = {rate};
    plan.fn = [](std::uint32_t, const events::TickInfo&, Real, const Vector&) {};
    sim.set_ticks(std::move(plan));
    ad.set_mask(rs != nullptr ? rs->mask : pwm(0.0));
    Out o;
    o.res = sim.simulate(Vector::Zero(ad.n_state()), t_end, rs);
    o.end = sim.final_state();
    o.ticks = sim.tick_counts();
    o.origin = sim.tick_origins();
    o.next_k = sim.tick_next_k();
    return o;
}

Real max_abs(const PEDResult<Mask>& r) {
    Real m = 0;
    for (const auto& x : r.states) m = std::max(m, x.cwiseAbs().maxCoeff());
    return m;
}

// The states at the sample instants both runs hold after t_split: their
// largest difference, and how many instants there were.
std::pair<Real, Size> compare(const Out& u, const Out& r, Real t_split) {
    std::map<Real, const Vector*> at;
    for (std::size_t i = 0; i < u.res.times.size(); ++i) {
        if (u.res.times[i] >= t_split) at[u.res.times[i]] = &u.res.states[i];
    }
    Real d = 0;
    Size n = 0;
    for (std::size_t i = 0; i < r.res.times.size(); ++i) {
        const auto it = at.find(r.res.times[i]);
        if (it == at.end()) continue;
        d = std::max(d, (*it->second - r.res.states[i]).cwiseAbs().maxCoeff());
        ++n;
    }
    return {d, n};
}

}  // namespace

TEST_CASE("ExactResumeState round-trips its payload and refuses a corrupt one",
          "[dsed][resume]") {
    Resume s;
    s.t = 1.25e-4;
    s.x = Vector::LinSpaced(3, -1.0, 2.0);
    s.last_time_event_t = 1.2e-4;
    s.last_fired_index = 3;
    s.last_fired_t = 1.21e-4;
    s.h = 2e-7;
    s.err_prev = 0.3;
    s.step_phase = 2;
    s.fast_t = {1.26e-4, 1.3e-4};
    s.burst_count = {1, 0};
    s.burst_t0 = {1.21e-4, 0.0};
    const auto v = s.flatten();
    const auto b = Resume::unflatten(v);
    CHECK(b.t == s.t);
    CHECK(b.x == s.x);
    CHECK(b.last_fired_index == 3);
    CHECK(b.fast_t == s.fast_t);
    CHECK(b.burst_count == s.burst_count);
    CHECK(b.burst_t0 == s.burst_t0);
    CHECK(b.step_phase == 2);
    auto cut = v;
    cut.pop_back();
    CHECK_THROWS_AS(Resume::unflatten(cut), std::invalid_argument);
    auto pad = v;
    pad.push_back(0.0);
    CHECK_THROWS_AS(Resume::unflatten(pad), std::invalid_argument);
    auto ver = v;
    ver[0] = 2.0;
    CHECK_THROWS_AS(Resume::unflatten(ver), std::invalid_argument);
    auto nan = v;   // last_fired_index (after version, t, n, x): a NaN is refused, never cast
    nan[3 + 3 + 1] = std::numeric_limits<Real>::quiet_NaN();
    CHECK_THROWS_AS(Resume::unflatten(nan), std::invalid_argument);
}

TEST_CASE("AutoResumeState round-trips its payload and refuses a corrupt or foreign one",
          "[dsed][resume]") {
    using Auto = dsed::AutoResumeState<topology::SwitchStateMask>;
    Auto s;
    s.t = 1.25e-4;
    s.x = Vector::LinSpaced(3, -1.0, 2.0);
    s.h_rk45 = 2e-7;
    s.err_prev = 0.3;
    s.step_phase = 2;
    s.fast_t = {1.26e-4, 1.3e-4};
    s.last_exact = false;
    s.bdf2_history = true;
    s.bdf2_prev_t = 1.24e-4;
    s.bdf2_prev = Vector::LinSpaced(3, 0.5, 1.5);
    const auto v = s.flatten();
    const auto b = Auto::unflatten(v);
    CHECK(b.t == s.t);
    CHECK(b.x == s.x);
    CHECK(b.h_rk45 == s.h_rk45);
    CHECK(b.err_prev == s.err_prev);
    CHECK(b.step_phase == 2);
    CHECK(b.fast_t == s.fast_t);
    CHECK(!b.last_exact);
    CHECK(b.bdf2_history);
    CHECK(b.bdf2_prev_t == s.bdf2_prev_t);
    CHECK(b.bdf2_prev == s.bdf2_prev);
    auto cut = v;
    cut.pop_back();
    CHECK_THROWS_AS(Auto::unflatten(cut), std::invalid_argument);
    auto pad = v;
    pad.push_back(0.0);
    CHECK_THROWS_AS(Auto::unflatten(pad), std::invalid_argument);
    auto nan = v;   // the state count (after tag, t): a NaN is refused, never cast
    nan[2] = std::numeric_limits<Real>::quiet_NaN();
    CHECK_THROWS_AS(Auto::unflatten(nan), std::invalid_argument);
    // each integrator reads only its own payload (the first value tells them apart)
    CHECK_THROWS_AS(Resume::unflatten(v), std::invalid_argument);
    Resume e;
    e.x = Vector::Zero(2);
    CHECK_THROWS_AS(Auto::unflatten(e.flatten()), std::invalid_argument);
}

TEST_CASE("A buck split at a gate edge or mid-interval continues the uninterrupted run",
          "[dsed][resume]") {
    for (const Real r_load : {10.0, 200.0}) {           // CCM, DCM
        const Out u = run(r_load, 2e-4);
        for (const Real ts : {1.0e-4, 1.023e-4}) {       // an edge; mid-interval
            const Out a = run(r_load, ts);
            REQUIRE(a.end.t == ts);
            const Out r = run(r_load, 2e-4, &a.end, a.origin, a.next_k);
            REQUIRE(r.res.times.front() == a.end.t);
            const auto [d, n] = compare(u, r, ts);
            CHECK(n * 2 >= r.res.times.size());
            CHECK(d <= 1e-12 * max_abs(u.res));
            CHECK((u.end.x - r.end.x).cwiseAbs().maxCoeff() <= 1e-12 * max_abs(u.res));
            CHECK(a.ticks[0] + r.ticks[0] == u.ticks[0]);   // no tick lost or repeated
        }
    }
}
