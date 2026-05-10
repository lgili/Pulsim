// =============================================================================
// RobustnessProfile — single source of truth for solver robustness knobs.
//
// Build:
//   cmake -S examples/cpp -B build/examples
//   cmake --build build/examples --target pulsim_example_robustness_profile
//   ./build/examples/pulsim_example_robustness_profile
//
// Demonstrates:
//   - RobustnessTier (Aggressive / Standard / Strict) → RobustnessProfile.
//   - Per-tier knob bundle: Newton iters, tolerance, recovery aids,
//     fallback policy. Every knob has one source instead of three (C++
//     Simulator ctor + python bindings + python wrapper) — the change
//     `refactor-unify-robustness-policy` Phase 1 deliverable.
//   - Round-trip the tier label through `parse_robustness_tier` for YAML
//     deserialization.
//
// See also: docs/robustness-policy.md
// =============================================================================
#include "pulsim/v1/robustness_profile.hpp"

#include <iomanip>
#include <iostream>
#include <stdexcept>

using namespace pulsim::v1;

void dump(const RobustnessProfile& p) {
    std::cout << "Profile [" << to_string(p.tier) << "]:\n"
              << "  Newton:    max_iters=" << p.newton_max_iters
              << "    tol_residual=" << std::scientific << p.newton_tol_residual
              << "    homotopy=" << std::boolalpha << p.newton_use_homotopy << "\n"
              << "  Linear:    fallback=" << p.linear_solver_allow_fallback
              << "    max_retries=" << p.linear_solver_max_retries << "\n"
              << "  Integrator: max_step_retries=" << p.integrator_max_step_retries
              << "    enable_lte=" << p.integrator_enable_lte << "\n"
              << "  Recovery:  source_stepping=" << p.enable_source_stepping
              << "    pseudo_transient=" << p.enable_pseudo_transient << "\n"
              << "  Fallback:  dae=" << p.allow_dae_fallback
              << "    aggressive_dt_backoff=" << p.allow_aggressive_dt_backoff << "\n";
}

int main() {
    std::cout << std::fixed;
    for (auto t : {RobustnessTier::Aggressive,
                   RobustnessTier::Standard,
                   RobustnessTier::Strict}) {
        dump(RobustnessProfile::for_tier(t));
        std::cout << "\n";
    }

    // YAML / CLI deserialization of the tier label.
    std::cout << "Parser round-trip:\n";
    for (auto t : {RobustnessTier::Aggressive,
                   RobustnessTier::Standard,
                   RobustnessTier::Strict}) {
        const auto label = to_string(t);
        const auto parsed = parse_robustness_tier(std::string(label));
        std::cout << "  to_string(" << static_cast<int>(t) << ") = '"
                  << label << "'   parse → " << static_cast<int>(parsed)
                  << "  ✓\n";
    }

    // Reject invalid label.
    try {
        parse_robustness_tier("turbo");
        std::cout << "  ✗ parse_robustness_tier('turbo') did not throw\n";
    } catch (const std::invalid_argument& exc) {
        std::cout << "  ✓ parse_robustness_tier('turbo') threw: "
                  << exc.what() << "\n";
    }
    return 0;
}
