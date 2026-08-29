// =============================================================================
// MMC — the GGJ Thevenin arm, pure math
// =============================================================================
//
// v2.0 Phase 3, "obra №2". These tests pin the aggregation against
// INDEPENDENT references — closed forms of the trapezoidal rule, not
// re-derivations of the arm's own equations. The crown test (an
// explicit chain of real switches and capacitors versus the
// aggregated arm, through the whole pwl engine) lives in
// python/tests/test_mmc_thevenin.py.

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "pulsim/mmc/thevenin_arm.hpp"

#include <cmath>

using namespace pulsim;
using namespace pulsim::mmc;
using Catch::Approx;

TEST_CASE("Thevenin arm: constant current ramps every cap exactly",
          "[v2][mmc][thevenin]") {
    // All N inserted, constant i: the trapezoidal update is
    // Δv = (dt/2C)(i + i) = dt·i/C per step, exactly — integer
    // arithmetic in disguise.
    ThevArmParams p;
    p.n_sm = 4;
    p.c_sm = 5e-3;
    p.r_on = 1e-3;
    p.dt = 1e-5;
    p.v_c_init = 100.0;
    ThevArm arm(p);

    const Real i = 50.0;
    (void)arm.pre_step(0.0, 4);          // arm the insertion set
    for (int k = 0; k < 1000; ++k) {
        (void)arm.pre_step(i, 4);
    }
    // First pre_step(i,·) back-solves with i_prev = 0 (half ramp),
    // the remaining 999 add a full dt·i/C each.
    const Real dv_full = p.dt * i / p.c_sm;
    const Real expect = 100.0 + 0.5 * dv_full + 999.0 * dv_full;
    for (Size s = 0; s < p.n_sm; ++s) {
        REQUIRE(arm.v_c()[s] == Approx(expect).epsilon(1e-12));
    }

    // End-of-run bookkeeping: finalize_step folds in exactly one
    // more full step — back-solve only, NO re-selection (a phantom
    // selection's trailing half-steps are the bug this method
    // exists to avoid).
    arm.finalize_step(i);
    for (Size s = 0; s < p.n_sm; ++s) {
        REQUIRE(arm.v_c()[s]
                 == Approx(expect + dv_full).epsilon(1e-12));
    }
}

TEST_CASE("Thevenin arm: RC discharge matches the trap closed form",
          "[v2][mmc][thevenin]") {
    // One SM, always inserted, shorted through R_load: each step
    // solves i_k = −V_eq/(R_load + R_eq) and back-solves the cap.
    // The trapezoidal discretisation of an RC discharge has the
    // closed form v_k = v_0 · ((1 − a)/(1 + a))^k with
    // a = dt/(2·R_total·C) — an INDEPENDENT formula to land on.
    ThevArmParams p;
    p.n_sm = 1;
    p.c_sm = 2e-3;
    p.r_on = 1e-3;
    p.dt = 1e-5;
    p.v_c_init = 48.0;
    ThevArm arm(p);

    const Real r_load = 0.5;
    Real i_prev = 0.0;
    auto st = arm.pre_step(i_prev, 1);
    for (int k = 0; k < 2000; ++k) {
        const Real i = -st.v_eq / (r_load + st.r_eq);
        st = arm.pre_step(i, 1);
        i_prev = i;
    }
    // Closed form of the companion recursion with the INCONSISTENT
    // start i_0 = 0 (the physical switch-on): defining
    // u_k = v_k + R_c·i_k, the recursion gives u_k = q·u_{k−1} with
    // q = (R_t − R_c)/(R_t + R_c) and
    // v_k = u_{k−1}·R_t/(R_t + R_c) — one factor of R_t/(R_t+R_c)
    // from the first half-step, then pure q decay. (The naive
    // v_0·q^k assumes the ODE-consistent i_0 = −v_0/R_t.)
    const Real r_t = r_load + p.r_on;
    const Real r_c = arm.r_c();
    const Real q = (r_t - r_c) / (r_t + r_c);
    const Real ref =
        48.0 * (r_t / (r_t + r_c)) * std::pow(q, 1999);
    REQUIRE(arm.v_c()[0] == Approx(ref).epsilon(1e-9));
}

TEST_CASE("Thevenin arm: the stamp is the series sum, and the flag "
          "fires only on insertion changes",
          "[v2][mmc][thevenin]") {
    ThevArmParams p;
    p.n_sm = 3;
    p.c_sm = 1e-3;
    p.r_on = 2e-3;
    p.dt = 2e-5;
    p.v_c_init = 10.0;
    ThevArm arm(p);
    const Real r_c = p.dt / (2.0 * p.c_sm);

    auto s1 = arm.pre_step(0.0, 2);
    REQUIRE(s1.r_eq == Approx(3 * 2e-3 + 2 * r_c));
    REQUIRE(s1.v_eq == Approx(2 * 10.0));   // fresh histories
    REQUIRE(s1.r_changed);                   // first stamp

    auto s2 = arm.pre_step(1.0, 2);
    REQUIRE_FALSE(s2.r_changed);             // same count → no refactor
    auto s3 = arm.pre_step(1.0, 3);
    REQUIRE(s3.r_changed);
    REQUIRE(s3.r_eq == Approx(3 * 2e-3 + 3 * r_c));
}

TEST_CASE("Thevenin arm: balancing pulls the spread together",
          "[v2][mmc][thevenin]") {
    // Deliberately unbalanced start; alternating-sign current with
    // sort-and-select must not let the spread grow, and after many
    // cycles it should be far below the initial 8 V.
    ThevArmParams p;
    p.n_sm = 8;
    p.c_sm = 2e-3;
    p.r_on = 1e-3;
    p.dt = 1e-5;
    p.v_c_init = 100.0;
    ThevArm arm(p);
    // A deliberate pre-charge spread. (The first attempt tried to
    // create the skew by driving n_on = 1 — and the balancer
    // round-robined the insertion and kept the arm perfectly
    // balanced, which is its job. Hence the setter.)
    for (Size i = 0; i < p.n_sm; ++i) {
        arm.set_v_c(i, 100.0 + Real(i) * 1.0);   // 7 V spread
    }
    auto spread = [&] {
        Real lo = arm.v_c()[0], hi = lo;
        for (Real v : arm.v_c()) {
            lo = std::min(lo, v);
            hi = std::max(hi, v);
        }
        return hi - lo;
    };
    const Real before = spread();
    REQUIRE(before > 1.0);                   // genuinely skewed

    // Alternate in BLOCKS of 50 steps (a 1 kHz ripple at dt=1e-5).
    // The first version flipped the sign EVERY step — a square wave
    // at Nyquist — and the spread exploded to 600 V: sort-and-select
    // predicts the coming step's direction from the PREVIOUS
    // current, and a per-step flip makes that prediction wrong 100%
    // of the time, turning the balancer into an anti-balancer. Real
    // arm currents are continuous, so the prediction is right at
    // every sample except the crossings; the block drive models
    // that.
    for (int k = 0; k < 4000; ++k) {
        const Real i = ((k / 50) % 2 == 0) ? 60.0 : -60.0;
        (void)arm.pre_step(i, 4);
    }
    REQUIRE(spread() < 0.2 * before);
}

TEST_CASE("Thevenin arm: bypass takes the trailing half-step, then "
          "clears the companion history",
          "[v2][mmc][thevenin]") {
    // Engine-consistent commutation (header, COMMUTATION
    // CONVENTION): the step where the SM leaves the inserted set,
    // its explicit twin's capacitor still sits behind the trap
    // companion while the series switch opens — the solve drives
    // i_C → 0 and the update leaves v += R_c·i_C⁻ behind. The
    // aggregated cap must take the same half-step, and THEN start
    // any later re-insertion from a fresh companion — replaying the
    // stale i_C would inject phantom charge every cycle.
    ThevArmParams p;
    p.n_sm = 1;
    p.c_sm = 1e-3;
    p.r_on = 1e-3;
    p.dt = 1e-5;
    p.v_c_init = 50.0;
    ThevArm arm(p);
    const Real r_c = arm.r_c();     // 0.005 ohm

    (void)arm.pre_step(0.0, 1);
    (void)arm.pre_step(100.0, 0);   // finalise at 100 A, then bypass
    // Finalisation of the conducting step: +r_c·(100 + 0) = 0.5 V.
    // Trailing half-step on the way out:   +r_c·100       = 0.5 V.
    const Real v_after = arm.v_c()[0];
    REQUIRE(v_after == Approx(50.0 + r_c * 100.0 + r_c * 100.0));
    (void)arm.pre_step(37.0, 1);    // bypassed: 37 A skips the cap
    REQUIRE(arm.v_c()[0] == Approx(v_after));   // untouched
    auto st = arm.pre_step(0.0, 1); // re-inserted stamp
    // V_eq must be v_C alone — no r_c·i_stale term.
    REQUIRE(st.v_eq == Approx(arm.v_c()[0] + r_c * 0.0));
}

TEST_CASE("Thevenin arm: constant-count membership swap restamps "
          "V_eq with r_changed = false",
          "[v2][mmc][thevenin]") {
    // The refactor flag is COUNT-based (R_eq depends only on n_on),
    // but V_eq depends on WHICH capacitors are in — a current sign
    // flip swaps the selection at constant count, and the stamp
    // must follow while r_changed stays false (no refactor needed,
    // but the b_extra injection must change).
    ThevArmParams p;
    p.n_sm = 2;
    p.c_sm = 1e-3;
    p.r_on = 1e-3;
    p.dt = 1e-5;
    p.v_c_init = 0.0;
    ThevArm arm(p);
    const Real r_c = arm.r_c();
    arm.set_v_c(0, 10.0);
    arm.set_v_c(1, 20.0);

    auto s1 = arm.pre_step(0.0, 1);      // charging: lowest → SM0
    REQUIRE(s1.v_eq == Approx(10.0));
    REQUIRE(arm.inserted()[0] == 1);

    auto s2 = arm.pre_step(-5.0, 1);     // discharging: highest → SM1
    // Same count → no refactor; V_eq must still move to SM1.
    REQUIRE_FALSE(s2.r_changed);
    REQUIRE(s2.r_eq == Approx(s1.r_eq));
    REQUIRE(arm.inserted()[1] == 1);
    REQUIRE(s2.v_eq == Approx(20.0));
    // The leaving SM0 finalised at −5 A and took its trailing
    // half-step: 10 + r_c·(−5) + r_c·(−5).
    REQUIRE(arm.v_c()[0] == Approx(10.0 - 2.0 * r_c * 5.0));
}

TEST_CASE("Thevenin arm: degenerate parameters refuse loudly",
          "[v2][mmc][thevenin]") {
    ThevArmParams p;
    p.n_sm = 0;
    p.c_sm = 1e-3;
    p.dt = 1e-5;
    REQUIRE_THROWS_AS(ThevArm(p), std::invalid_argument);
    p.n_sm = 4;
    p.c_sm = 0.0;
    REQUIRE_THROWS_AS(ThevArm(p), std::invalid_argument);
    p.c_sm = 1e-3;
    ThevArm ok(p);
    REQUIRE_THROWS_AS(ok.pre_step(0.0, 5), std::invalid_argument);
}
