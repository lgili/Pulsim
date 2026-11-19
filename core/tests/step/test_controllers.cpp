// refactor-unified-transient-stepper task 5.1: the controllers' state as
// explicit, serialisable objects (core/include/pulsim/solver/step/controllers.hpp).

#include <catch2/catch_test_macros.hpp>

#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include "pulsim/solver/step/controllers.hpp"

using pulsim::Real;
using pulsim::Size;
namespace step = pulsim::step;

TEST_CASE("LteController: save/load round-trips every slot bit-exactly",
          "[step]") {
    step::LteController c;
    c.h = 3.141592653589793e-7;
    c.h_mem_gate[0] = 1e-9 / 3;
    c.h_mem_gate[1] = 0.0;
    c.pending_gate_slot = 1;
    c.h_mem_diode = {0.0, 2.5e-8, 1.0 / 7.0, 0.0};
    c.diode_slot = 2;
    c.gate_parity = 1;
    const step::ControllerState s = c.save();
    CHECK(s.kind == step::ControllerState::Kind::Lte);

    step::LteController d;
    d.load(s);
    CHECK(d.h == c.h);
    CHECK(d.h_mem_gate[0] == c.h_mem_gate[0]);
    CHECK(d.h_mem_gate[1] == c.h_mem_gate[1]);
    CHECK(d.pending_gate_slot == 1);
    CHECK(d.h_mem_diode == c.h_mem_diode);
    CHECK(d.diode_slot == 2);
    CHECK(d.gate_parity == 1);

    step::LteController none;   // the "no slot" sentinels survive too
    none.h = 1e-6;
    step::LteController e;
    e.diode_slot = 7;
    e.pending_gate_slot = 0;
    e.load(none.save());
    CHECK(e.diode_slot == Size(-1));
    CHECK(e.pending_gate_slot == -1);
    CHECK(e.h_mem_diode.empty());
}

TEST_CASE("LteController: refuses a state it did not write", "[step]") {
    step::LteController c;
    step::ControllerState s;
    s.kind = step::ControllerState::Kind::Fixed;
    s.v = {0.0, 0.0, 0.0, 0.0, 0.0};
    CHECK_THROWS_AS(c.load(s), std::invalid_argument);
    CHECK_THROWS_AS(c.load(step::ControllerState{}), std::invalid_argument);
    // slots out of range, a NaN: refused, and nothing is changed
    step::LteController ok;
    ok.h = 1e-6;
    ok.h_mem_diode = {0.0, 0.0};
    const auto good = ok.save();
    for (const auto& [i, v] : std::vector<std::pair<std::size_t, Real>>{
             {3, 2.0}, {4, 2.0}, {4, 0.5}, {5, 3.0}, {0, 0.0},
             {1, std::numeric_limits<Real>::quiet_NaN()}}) {
        auto b = good;
        b.v[i] = v;
        c.h = 7.0;
        CHECK_THROWS_AS(c.load(b), std::invalid_argument);
        CHECK(c.h == 7.0);
    }
    CHECK_NOTHROW(c.load(good));
}

TEST_CASE("LteController: factor laws", "[step]") {
    // reject: never grows, never below 0.1
    CHECK(step::LteController::reject_factor(1.0001, false) <= 0.9);
    CHECK(step::LteController::reject_factor(1e12, false) == 0.1);
    // accept: bounded growth 5x, the error floor 1e-4
    CHECK(step::LteController::accept_factor(0.0, false) == 5.0);
    CHECK(step::LteController::accept_factor(1e-4, true)
          == step::LteController::accept_factor(0.0, true));
    CHECK(step::LteController::accept_factor(1.0, false) == 0.9);
}

TEST_CASE("FixedClock: grid times from an integer counter", "[step]") {
    // runtime operands on both sides: a folded constant could contract
    // differently from the runtime expression
    volatile Real t0 = 1e-3, dt = 1e-7;
    volatile pulsim::Size k = 10;
    const step::FixedClock c{t0, dt};
    const Real expect = t0 + static_cast<Real>(k) * dt;
    CHECK(c.at(0) == t0);
    CHECK(c.at(k) == expect);
}
