// refactor-unified-transient-stepper task 5.1: the controllers' state as
// explicit, serialisable objects (core/include/pulsim/solver/step/controllers.hpp).

#include <catch2/catch_test_macros.hpp>

#include <stdexcept>

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

    step::LteController none;   // the "no slot" sentinels survive too
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
    const step::FixedClock c{1e-3, 1e-7};
    CHECK(c.at(0) == 1e-3);
    CHECK(c.at(10) == 1e-3 + static_cast<Real>(10) * 1e-7);
}
