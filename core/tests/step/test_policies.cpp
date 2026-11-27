// refactor-unified-transient-stepper task 5.2: the fixed engine's policies.
#include <catch2/catch_test_macros.hpp>

#include "pulsim/solver/step/policies.hpp"

namespace step = pulsim::step;

TEST_CASE("HalvingLadder: 2^h sub-steps of dt / 2^h, cap clamped at 20", "[step]") {
    STATIC_CHECK(step::HalvingLadder::cap(3) == 3);
    STATIC_CHECK(step::HalvingLadder::cap(1000) == 20);
    STATIC_CHECK(step::HalvingLadder::n_sub(4) == 16);
    CHECK(step::HalvingLadder::sub_dt(1e-6, 2) == 1e-6 / 4.0);
}

TEST_CASE("GuardPolicy: freeze, then clamp; 0 turns each off", "[step]") {
    auto r = step::GuardPolicy::apply(1.0, 500.0, 0.0, 0.0);
    CHECK((r.i == 500.0 && !r.froze && !r.clamped));
    r = step::GuardPolicy::apply(1.0, 500.0, 100.0, 0.0);
    CHECK((r.i == 1.0 && r.froze && !r.clamped));
    r = step::GuardPolicy::apply(1.0, 50.0, 100.0, 20.0);
    CHECK((r.i == 20.0 && !r.froze && r.clamped));
    r = step::GuardPolicy::apply(-30.0, -60.0, 100.0, 20.0);
    CHECK((r.i == -20.0 && !r.froze && r.clamped));
    // the freeze restores the pre-step current, which the clamp still bounds
    r = step::GuardPolicy::apply(-30.0, 500.0, 100.0, 20.0);
    CHECK((r.i == -20.0 && r.froze && r.clamped));
}
