#pragma once

#include "pulsim/v1/numeric_types.hpp"

#include <cstdint>
#include <string>
#include <string_view>

namespace pulsim::v1 {

// =============================================================================
// refactor-unify-robustness-policy — Phase 1
// =============================================================================
//
// Single source of truth for the simulator's robustness knob bundle.
// Today the same defaults are decided in three places:
//   - C++ Simulator constructor (per circuit analysis)
//   - python/bindings.cpp (`apply_robust_*_defaults` helpers)
//   - python/pulsim/__init__.py (`_tune_*_for_robust` Python helpers)
//
// `RobustnessProfile` is the canonical struct. Each tier maps to a
// well-defined set of knob values; circuits that need extra care
// (PWL with VCSwitch, deeply nonlinear AD, etc.) bias toward the
// stricter tiers. The downstream config sites consume the profile
// rather than re-deriving knobs from scratch.

/// Robustness tier — three discrete levels covering the cost / safety
/// trade-off. Most production simulations use `Standard`; `Aggressive`
/// is for benchmarks aiming for raw throughput; `Strict` is for new
/// circuits or convergence debugging where the safety net matters
/// more than the wall-clock.
enum class RobustnessTier : std::uint8_t {
    Aggressive = 0,   ///< maximum throughput, minimum safety nets
    Standard   = 1,   ///< production default — balanced
    Strict     = 2,   ///< maximum safety nets, slowest
};

[[nodiscard]] constexpr std::string_view to_string(RobustnessTier t) noexcept {
    switch (t) {
        case RobustnessTier::Aggressive: return "aggressive";
        case RobustnessTier::Standard:   return "standard";
        case RobustnessTier::Strict:     return "strict";
    }
    return "standard";
}

/// Canonical knob bundle. Each field maps onto a named `SimulationOptions`
/// (or sub-config) entry; the consumer applies the bundle via
/// `apply_to(SimulationOptions&)` (in the `.cpp` follow-up that wires
/// the actual mutators) or pulls individual knobs by hand for
/// fine-grained overrides.
struct RobustnessProfile {
    RobustnessTier tier = RobustnessTier::Standard;

    // Newton solver
    int  newton_max_iters         = 30;
    Real newton_tol_residual      = 1e-8;
    Real newton_tol_step          = 1e-9;
    bool newton_use_homotopy      = false;

    // Linear solver
    bool linear_solver_allow_fallback = true;
    int  linear_solver_max_retries     = 3;

    // Integrator
    int  integrator_max_step_retries   = 6;
    bool integrator_enable_lte         = true;

    // Recovery / convergence aids
    Real gmin_initial                  = 0.0;
    Real gmin_max                      = 1e-3;
    bool enable_source_stepping        = false;
    bool enable_pseudo_transient       = false;

    // Fallback policy
    bool allow_dae_fallback            = true;
    bool allow_aggressive_dt_backoff   = true;

    /// Construct the canonical profile for a given tier. Per-tier knob
    /// values come from the long-form documentation in
    /// `docs/robustness-policy.md` — same numbers the existing
    /// `apply_robust_*_defaults` helpers landed on, just consolidated
    /// into one source.
    [[nodiscard]] static constexpr RobustnessProfile for_tier(RobustnessTier t) noexcept {
        RobustnessProfile p;
        p.tier = t;
        switch (t) {
            case RobustnessTier::Aggressive:
                p.newton_max_iters             = 15;
                p.newton_tol_residual          = 1e-6;
                p.newton_tol_step              = 1e-7;
                p.newton_use_homotopy          = false;
                p.linear_solver_allow_fallback = false;
                p.linear_solver_max_retries    = 1;
                p.integrator_max_step_retries  = 2;
                p.integrator_enable_lte        = false;
                p.gmin_initial                 = 0.0;
                p.gmin_max                     = 1e-9;
                p.enable_source_stepping       = false;
                p.enable_pseudo_transient      = false;
                p.allow_dae_fallback           = false;
                p.allow_aggressive_dt_backoff  = false;
                return p;

            case RobustnessTier::Standard:
                // Match the existing simulator defaults — production
                // sweet spot for power-electronics circuits.
                return p;

            case RobustnessTier::Strict:
                p.newton_max_iters             = 80;
                p.newton_tol_residual          = 1e-10;
                p.newton_tol_step              = 1e-11;
                p.newton_use_homotopy          = true;
                p.linear_solver_allow_fallback = true;
                p.linear_solver_max_retries    = 8;
                p.integrator_max_step_retries  = 16;
                p.integrator_enable_lte        = true;
                p.gmin_initial                 = 1e-12;
                p.gmin_max                     = 1e-2;
                p.enable_source_stepping       = true;
                p.enable_pseudo_transient      = true;
                p.allow_dae_fallback           = true;
                p.allow_aggressive_dt_backoff  = true;
                return p;
        }
        return p;
    }
};

/// Parse a tier string from YAML / CLI: `"aggressive" | "standard" |
/// "strict"`. Throws `std::invalid_argument` on unknown values.
[[nodiscard]] inline RobustnessTier parse_robustness_tier(std::string_view s) {
    if (s == "aggressive") return RobustnessTier::Aggressive;
    if (s == "standard")   return RobustnessTier::Standard;
    if (s == "strict")     return RobustnessTier::Strict;
    throw std::invalid_argument(
        "Unknown robustness tier '" + std::string{s} +
        "' (expected aggressive | standard | strict)");
}

}  // namespace pulsim::v1
