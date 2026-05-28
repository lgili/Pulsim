// =============================================================================
// PED Gate 2 — PIController unit tests
// =============================================================================

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "pulsim/dsed/step_controller.hpp"

using pulsim::Real;
using pulsim::Vector;
using pulsim::dsed::PIController;

TEST_CASE("PIController accepts a small-error step + grows h",
           "[dsed][step_controller]") {
    PIController c;
    Vector x_old(2);
    x_old << Real{1}, Real{2};
    Vector x_new = x_old;
    Vector err(2);
    err << Real{1e-10}, Real{1e-10};  // well below atol=1e-9

    auto [ok, h_next] = c.accept(err, x_old, x_new, Real{1e-7});
    REQUIRE(ok);
    REQUIRE(h_next > Real{1e-7});  // grew
    REQUIRE(h_next <= Real{0.9 * 5.0 * 1e-7} + Real{1e-15});  // capped by rho_max=5
}

TEST_CASE("PIController rejects a too-large error + shrinks h",
           "[dsed][step_controller]") {
    PIController c;
    Vector x_old(2);
    x_old << Real{1}, Real{1};
    Vector x_new = x_old;
    Vector err(2);
    err << Real{1.0}, Real{1.0};  // huge — guaranteed reject

    auto [ok, h_next] = c.accept(err, x_old, x_new, Real{1e-7});
    REQUIRE_FALSE(ok);
    REQUIRE(h_next < Real{1e-7});  // shrunk
}

TEST_CASE("PIController err_norm normalises by atol/rtol scale",
           "[dsed][step_controller]") {
    PIController c{
        Real{1e-6},  // rtol
        Real{1e-9},  // atol
    };
    Vector x_old(2);
    x_old << Real{0}, Real{0};  // zero state → sc = atol = 1e-9
    Vector x_new = x_old;
    Vector err(2);
    err << Real{1e-9}, Real{1e-9};  // exactly one tolerance unit

    const Real e = c.err_norm(err, x_old, x_new);
    REQUIRE(e == Catch::Approx(Real{1.0}).epsilon(Real{1e-12}));
}

TEST_CASE("PIController throws after max_rejects consecutive rejections",
           "[dsed][step_controller]") {
    PIController c{
        Real{1e-6}, Real{1e-9},
        Real{0.7}, Real{0.3},
        Real{0.1}, Real{5.0},
        Real{0.9},
        2  // max_rejects = 2
    };
    Vector x_old(1), x_new(1);
    x_old << Real{1};
    x_new << Real{1};
    Vector err(1);
    err << Real{1.0};

    auto r1 = c.accept(err, x_old, x_new, Real{1e-7});
    REQUIRE_FALSE(r1.first);
    auto r2 = c.accept(err, x_old, x_new, Real{1e-7});
    REQUIRE_FALSE(r2.first);
    REQUIRE_THROWS_AS(c.accept(err, x_old, x_new, Real{1e-7}),
                      std::runtime_error);
}

TEST_CASE("PIController reset clears wind-up", "[dsed][step_controller]") {
    PIController c;
    Vector x_old(1), x_new(1), err(1);
    x_old << Real{1};
    x_new << Real{1};
    err << Real{1e-10};
    // Accept a few steps to populate err_prev
    for (int i = 0; i < 3; ++i) {
        auto r = c.accept(err, x_old, x_new, Real{1e-7});
        (void)r;  // discard
    }
    REQUIRE(c.err_prev() != Real{1});  // moved off default

    c.reset();
    REQUIRE(c.err_prev() == Real{1});
    REQUIRE(c.reject_streak() == 0);
}
