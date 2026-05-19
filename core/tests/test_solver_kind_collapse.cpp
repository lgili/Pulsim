// simplify-and-harden-numerical-surface — Phase 8 tests.
//
// Verifies the new public-facing `LinearSolverKind::{Auto, Direct,
// Iterative}` values and `DCStrategy::Override` correctly resolve
// to concrete engines / strategies via the auto-selector.
//
// Phase 8 is ADDITIVE — the 6 concrete `LinearSolverKind` values
// (SparseLU, EnhancedSparseLU, KLU, GMRES, BiCGSTAB, CG) and the 5
// concrete `DCStrategy` values (Direct, GminStepping, SourceStepping,
// PseudoTransient, Homotopy) all stay working. The new abstract
// values just promote a smaller, friendlier user surface.

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "pulsim/v1/high_performance.hpp"
#include "pulsim/v1/numerical/dc_strategy.hpp"

using namespace pulsim::v1;
using Catch::Approx;

TEST_CASE("LinearSolverKind: Auto resolves to Direct on small systems",
          "[linear_solver][phase8][auto]") {
    // Below the 5000-unknown threshold → Direct (then KLU).
    const auto resolved =
        resolve_linear_solver_kind(LinearSolverKind::Auto, /*N=*/100);
    CHECK(resolved == LinearSolverKind::KLU);
}

TEST_CASE("LinearSolverKind: Auto resolves to Iterative on large systems",
          "[linear_solver][phase8][auto]") {
    // Above the threshold → Iterative (then GMRES).
    const auto resolved =
        resolve_linear_solver_kind(LinearSolverKind::Auto, /*N=*/10000);
    CHECK(resolved == LinearSolverKind::GMRES);
}

TEST_CASE("LinearSolverKind: Direct → KLU regardless of size",
          "[linear_solver][phase8][direct]") {
    CHECK(resolve_linear_solver_kind(LinearSolverKind::Direct, 100)
          == LinearSolverKind::KLU);
    CHECK(resolve_linear_solver_kind(LinearSolverKind::Direct, 100000)
          == LinearSolverKind::KLU);
}

TEST_CASE("LinearSolverKind: Iterative → GMRES regardless of size",
          "[linear_solver][phase8][iterative]") {
    CHECK(resolve_linear_solver_kind(LinearSolverKind::Iterative, 100)
          == LinearSolverKind::GMRES);
    CHECK(resolve_linear_solver_kind(LinearSolverKind::Iterative, 100000)
          == LinearSolverKind::GMRES);
}

TEST_CASE("LinearSolverKind: concrete engines pass through unchanged",
          "[linear_solver][phase8][concrete]") {
    for (auto kind : {LinearSolverKind::SparseLU,
                       LinearSolverKind::EnhancedSparseLU,
                       LinearSolverKind::KLU,
                       LinearSolverKind::GMRES,
                       LinearSolverKind::BiCGSTAB,
                       LinearSolverKind::CG}) {
        CHECK(resolve_linear_solver_kind(kind, 100) == kind);
        CHECK(resolve_linear_solver_kind(kind, 100000) == kind);
    }
}

TEST_CASE("LinearSolverKind: is_direct_solver classifies new values correctly",
          "[linear_solver][phase8][is_direct]") {
    CHECK(is_direct_solver(LinearSolverKind::Auto));      // Direct-leaning default
    CHECK(is_direct_solver(LinearSolverKind::Direct));
    CHECK_FALSE(is_direct_solver(LinearSolverKind::Iterative));

    CHECK(is_direct_solver(LinearSolverKind::SparseLU));
    CHECK(is_direct_solver(LinearSolverKind::EnhancedSparseLU));
    CHECK(is_direct_solver(LinearSolverKind::KLU));
    CHECK_FALSE(is_direct_solver(LinearSolverKind::GMRES));
    CHECK_FALSE(is_direct_solver(LinearSolverKind::BiCGSTAB));
    CHECK_FALSE(is_direct_solver(LinearSolverKind::CG));
}

TEST_CASE("DCStrategy: Override stores strategy_override on config",
          "[dc][phase8][override]") {
    DCConvergenceConfig cfg{};
    cfg.strategy = DCStrategy::Override;
    cfg.strategy_override = DCStrategy::PseudoTransient;

    CHECK(cfg.strategy == DCStrategy::Override);
    CHECK(cfg.strategy_override == DCStrategy::PseudoTransient);
}

TEST_CASE("DCStrategy: Auto + Override are the two recommended values",
          "[dc][phase8][public_surface]") {
    // Pin the documented contract: Auto and Override are the
    // user-facing values; the 5 concrete strategies stay supported
    // but should not be the documented recommendation.
    DCConvergenceConfig cfg{};
    CHECK(cfg.strategy == DCStrategy::Auto);  // default

    // strategy_override defaults to a concrete value (Direct).
    CHECK(cfg.strategy_override == DCStrategy::Direct);
}
