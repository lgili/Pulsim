// add-unified-event-queue tasks 2.2 / 6.1: gate sources composed by switch
// index, and the carrier PWM modulator with a duty register.

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

#include "pulsim/events/gate_source.hpp"
#include "pulsim/sources/dead_time_pwm_pair_fn.hpp"
#include "pulsim/sources/pwm_switch_fn.hpp"

using pulsim::Index;
using pulsim::Real;
using pulsim::Size;
using namespace pulsim::events;
using pulsim::topology::SwitchStateMask;

namespace {
// Grid points, edges and their neighbours: where an off-by-one would show.
std::vector<Real> probe_times(Real T, Real d) {
    std::vector<Real> ts;
    for (int k = 0; k < 2000; ++k) ts.push_back(k * 1e-7);
    for (int k = 0; k < 30; ++k) {
        for (const Real f : {0.0, d, 1.0 - 0.02, d - 0.02}) {
            const Real e = (k + f) * T;
            ts.push_back(e);
            ts.push_back(std::nextafter(e, 0.0));
            ts.push_back(std::nextafter(e, 1.0));
        }
    }
    return ts;
}
}  // namespace

TEST_CASE("CarrierPwm equals make_pwm_switch_fn for a constant duty",
          "[event_queue][gate_source]") {
    const Real f = 1e5, T = 1.0 / f;
    for (const Real d : {0.0, 0.25, 0.5, 0.731, 1.0}) {
        DutyRegister reg{d};
        const CarrierPwm g{T, 0.0, &reg, 1};
        const auto ref = pulsim::sources::make_pwm_switch_fn(f, d, 1, 3);
        std::size_t bad = 0;
        for (const Real t : probe_times(T, d)) {
            SwitchStateMask m(3);
            g.apply(t, m);
            bad += !(m == ref(t)) || g.next_edge(t) != ref.next_edge_after(t);
        }
        INFO("duty " << d);
        CHECK(bad == 0);
    }
}

TEST_CASE("CarrierPwm leg with dead time equals make_dead_time_pwm_pair_fn",
          "[event_queue][gate_source]") {
    const Real f = 1e5, T = 1.0 / f;
    for (const Real d : {0.1, 0.5, 0.9}) {
        DutyRegister reg{d};
        const CarrierPwm g{T, 0.0, &reg, 0, 2, 1e-7};
        const auto ref =
            pulsim::sources::make_dead_time_pwm_pair_fn(f, d, 0, 2, 3, 1e-7);
        std::size_t bad = 0;
        for (const Real t : probe_times(T, d)) {
            SwitchStateMask m(3);
            g.apply(t, m);
            bad += !(m == ref(t)) || g.next_edge(t) != ref.next_edge_after(t);
        }
        INFO("duty " << d);
        CHECK(bad == 0);
    }
}

TEST_CASE("a written duty moves the edges: immediate and at carrier start",
          "[event_queue][gate_source]") {
    const Real T = 1e-5;
    DutyRegister now{0.3};
    const CarrierPwm a{T, 0.0, &now, 0};
    CHECK(a.next_edge(0.0) == 0.3 * T);
    now.write(0.6, 0.0);
    CHECK(a.next_edge(0.0) == 0.6 * T);

    DutyRegister held{0.3, LatchPolicy::AtCarrierStart, T};
    const CarrierPwm b{T, 0.0, &held, 0};
    held.write(0.6, 0.1 * T);              // mid-period: held until T
    SwitchStateMask m(1);
    b.apply(0.45 * T, m);
    CHECK(!m.get(0));                      // old duty 0.3 still in force
    CHECK(b.next_edge(0.45 * T) == T);     // the period start latches
    b.apply(T + 0.45 * T, m);
    CHECK(m.get(0));                       // new duty 0.6 from T
    held.write(0.2, 1.5 * T);              // a later write: 0.6 kept until 2T
    CHECK(held.read(1.9 * T) == 0.6);
    CHECK(held.read(2 * T) == 0.2);
    DutyRegister bad{0.0, LatchPolicy::AtCarrierStart};
    CHECK_THROWS_AS(bad.write(0.5, 0.0), std::invalid_argument);
    GateSet set(2);
    DutyRegister other{0.3, LatchPolicy::AtCarrierStart, 2 * T};
    CHECK_THROWS_AS(set.add(CarrierPwm{T, 0.0, &other, 0}, "x"),
                    std::invalid_argument);   // latches on another carrier
    CHECK_THROWS_AS(set.add(CarrierPwm{T, 0.0, nullptr, 0}, "x"),
                    std::invalid_argument);
    CHECK_THROWS_AS(set.add(CarrierPwm{T, 0.0, &now, 1, 1}, "x"),
                    std::invalid_argument);
    CHECK(set.owner(0) == -1);
}

TEST_CASE("GateSet composes by index and refuses a doubly driven switch",
          "[event_queue][gate_source]") {
    const Real T = 1e-5;
    DutyRegister d1{0.3}, d2{0.6};
    GateSet set(4);
    set.add(CarrierPwm{T, 0.0, &d1, 0}, "pwm_a");
    set.add(ScheduleGate{pulsim::sources::make_pwm_switch_fn(5e4, 0.5, 2, 4),
                         {2}}, "helper_b");
    try {
        set.add(CarrierPwm{T, 0.0, &d2, 1, 2}, "leg_c");
        FAIL("overlap not refused");
    } catch (const std::invalid_argument& e) {
        const std::string w = e.what();
        CHECK(w.find("bit 2") != std::string::npos);
        CHECK(w.find("helper_b") != std::string::npos);
        CHECK(w.find("leg_c") != std::string::npos);
    }
    CHECK_THROWS_AS(set.add(CarrierPwm{T, 0.0, &d2, 7}, "far"),
                    std::invalid_argument);
    CHECK(set.owner(1) == -1);             // the refused leg took nothing
    CHECK(set.owner(3) == -1);
    const auto sched = set.to_schedule();
    REQUIRE(sched.has_edges());
    GateSet opaque(1);   // a schedule without an edge oracle is not exact
    opaque.add(ScheduleGate{pulsim::sources::SwitchSchedule{
                   [](Real) { return SwitchStateMask(1); }}, {0}}, "op");
    CHECK(!opaque.to_schedule().has_edges());
    for (int k = 0; k < 400; ++k) {
        const Real t = k * 1e-7;
        const auto m = sched(t);
        const auto ra = pulsim::sources::make_pwm_switch_fn(1e5, 0.3, 0, 4)(t);
        const auto rb = pulsim::sources::make_pwm_switch_fn(5e4, 0.5, 2, 4)(t);
        CHECK(m.get(0) == ra.get(0));
        CHECK(m.get(2) == rb.get(2));
        CHECK(!m.get(1));
        CHECK(!m.get(3));
        CHECK(sched.next_edge_after(t)
              == std::min(CarrierPwm{T, 0.0, &d1, 0}.next_edge(t),
                          pulsim::sources::make_pwm_switch_fn(5e4, 0.5, 2, 4)
                              .next_edge_after(t)));
    }
}

TEST_CASE("EdgeList: a bit holds its last edge, edges come out sorted",
          "[event_queue][gate_source]") {
    const EdgeList g{{{3e-6, 0, false}, {1e-6, 0, true}, {2e-6, 1, true},
                      {3e-6, 0, true}}};
    g.check("e");
    CHECK(g.owned_bits() == std::vector<Index>{0, 1});
    SwitchStateMask m(2);
    g.apply(0.0, m);
    CHECK((!m.get(0) && !m.get(1)));
    g.apply(1e-6, m);
    CHECK((m.get(0) && !m.get(1)));
    g.apply(3e-6, m);
    CHECK((m.get(0) && m.get(1)));      // the last edge of an instant wins
    CHECK(g.next_edge(0.0) == 1e-6);
    CHECK(g.next_edge(2e-6) == 3e-6);
    CHECK(std::isinf(g.next_edge(3e-6)));
    GateSet set(1);
    CHECK_THROWS_AS(set.add(EdgeList{{{1e-6, 2, true}}}, "far"),
                    std::invalid_argument);
    CHECK_THROWS_AS(set.add(EdgeList{{{std::numeric_limits<Real>::quiet_NaN(), 0,
                                       true}}}, "nan"),
                    std::invalid_argument);
}
