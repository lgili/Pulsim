// =============================================================================
// refactor-unify-robustness-policy — Phase 1 tests
// =============================================================================

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "pulsim/v1/robustness_profile.hpp"

#include <stdexcept>

using namespace pulsim::v1;

TEST_CASE("RobustnessProfile::for_tier produces tier-distinct knob bundles",
          "[v1][robustness][primitive]") {
    const auto agg = RobustnessProfile::for_tier(RobustnessTier::Aggressive);
    const auto std = RobustnessProfile::for_tier(RobustnessTier::Standard);
    const auto str = RobustnessProfile::for_tier(RobustnessTier::Strict);

    CHECK(agg.tier == RobustnessTier::Aggressive);
    CHECK(std.tier == RobustnessTier::Standard);
    CHECK(str.tier == RobustnessTier::Strict);

    // Newton iteration budget: aggressive < standard < strict.
    CHECK(agg.newton_max_iters < std.newton_max_iters);
    CHECK(std.newton_max_iters < str.newton_max_iters);

    // Tolerance: aggressive looser than strict.
    CHECK(agg.newton_tol_residual > str.newton_tol_residual);

    // Recovery aids: only Strict enables homotopy + source stepping +
    // pseudo-transient.
    CHECK_FALSE(agg.newton_use_homotopy);
    CHECK_FALSE(std.newton_use_homotopy);
    CHECK(str.newton_use_homotopy);
    CHECK(str.enable_source_stepping);
    CHECK(str.enable_pseudo_transient);

    // DAE fallback: Aggressive opts out, Standard + Strict allow.
    CHECK_FALSE(agg.allow_dae_fallback);
    CHECK(std.allow_dae_fallback);
    CHECK(str.allow_dae_fallback);
}

TEST_CASE("RobustnessProfile to_string round-trips through parse_robustness_tier",
          "[v1][robustness][parse]") {
    for (const auto t : {RobustnessTier::Aggressive,
                          RobustnessTier::Standard,
                          RobustnessTier::Strict}) {
        const auto label = to_string(t);
        const auto parsed = parse_robustness_tier(label);
        CHECK(parsed == t);
    }
}

TEST_CASE("parse_robustness_tier rejects unknown strings",
          "[v1][robustness][parse][validation]") {
    CHECK_THROWS_AS(parse_robustness_tier("turbo"),  std::invalid_argument);
    CHECK_THROWS_AS(parse_robustness_tier(""),       std::invalid_argument);
    CHECK_THROWS_AS(parse_robustness_tier("STRICT"), std::invalid_argument);
}
