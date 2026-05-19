#pragma once

// simplify-and-harden-numerical-surface — Phase 2.
//
// Numerical configuration preset enum + `SimulationOptions::from_preset(...)`
// factory. Replaces the previous "raw SimulationOptions{} vs
// make_robust_options() vs make_*_options() vs hand-tune 50 fields" split
// with a single named choice.
//
// Four presets:
//
//   - Auto         (default — picks Robust today; will evolve)
//   - Fast         (PWL Ideal + Trapezoidal + KLU + fixed step,
//                   pure-switching topologies)
//   - Robust       (TRBDF2 + KLU + variable + stiffness + retries,
//                   motor / mixed-domain / nonlinear circuits)
//   - HighFidelity (TRBDF2 + tight LTE + small dt_max, parity-validation
//                   runs against PLECS / PSIM / SPICE)
//
// Per-preset materialization tables live in `apply_preset_inplace()`
// inside `simulation.hpp` (kept there so the factory has the full
// SimulationOptions type definition in scope). This header just owns
// the enum + the public `apply_preset()` declaration.

#include "pulsim/v1/numeric_types.hpp"

#include <cstdint>
#include <string_view>

namespace pulsim::v1 {

/// Numerical configuration preset — single named choice that materializes
/// every numerical knob (integrator, linear solver, timestep controller,
/// DC strategy, stiffness, Newton tuning) into a coherent profile.
///
/// 95% of users should pick one of these instead of hand-tuning
/// `SimulationOptions` field-by-field. Use:
///
///     SimulationOptions opts =
///         SimulationOptions::from_preset(Preset::Auto, /*dt=*/1e-6,
///                                         /*tstop=*/1e-3);
///
/// then override only the fields that genuinely differ for your circuit.
enum class Preset : std::uint8_t {
    /// Default — currently maps to `Robust`. Tracks the production
    /// recommendation as it evolves; new users who don't know which
    /// preset to pick should pick `Auto`.
    Auto = 0,

    /// Pure-switching topologies (buck, boost, full-bridge, basic
    /// 3φ VSI) — PWL Ideal switching + Trapezoidal + KLU + fixed
    /// step. Smallest overhead per step, expects no nonlinear devices
    /// requiring Newton inner-loop convergence.
    Fast = 1,

    /// Motor drives, mixed-domain circuits, magnetics with saturation,
    /// thermal feedback — TRBDF2 + KLU + variable step + stiffness
    /// detection + 12-step retry budget. Handles every production
    /// topology Pulsim ships benchmarks for.
    Robust = 2,

    /// Parity-validation runs (vs PLECS / PSIM / SPICE / ngspice) —
    /// TRBDF2 + step-doubling LTE + tight tolerances + small `dt_max`.
    /// 3-10x slower than `Robust` but produces bit-comparable
    /// waveforms suitable for golden-CSV regression tests.
    HighFidelity = 3,
};

/// Human-readable name for the preset (useful for logging / telemetry).
[[nodiscard]] constexpr std::string_view to_string(Preset p) noexcept {
    switch (p) {
        case Preset::Auto:         return "Auto";
        case Preset::Fast:         return "Fast";
        case Preset::Robust:       return "Robust";
        case Preset::HighFidelity: return "HighFidelity";
    }
    return "Auto";
}

/// Parse a preset string from YAML or CLI (case-insensitive). Accepts
/// the canonical forms `auto`, `fast`, `robust`, `high_fidelity` plus
/// the common variants `high-fidelity` and `highfidelity`.
///
/// Returns `Preset::Auto` for unknown strings. Callers that want strict
/// validation should compare the input against the expected canonical
/// form before calling.
[[nodiscard]] constexpr Preset parse_preset_or_auto(
        std::string_view s) noexcept {
    // Manual case-insensitive comparison (constexpr-friendly, no
    // std::tolower allocation cost).
    auto eq_ci = [](std::string_view a, std::string_view b) {
        if (a.size() != b.size()) return false;
        for (std::size_t i = 0; i < a.size(); ++i) {
            const char ca = a[i];
            const char cb = b[i];
            const char la = (ca >= 'A' && ca <= 'Z') ? static_cast<char>(ca + 32) : ca;
            const char lb = (cb >= 'A' && cb <= 'Z') ? static_cast<char>(cb + 32) : cb;
            if (la != lb) return false;
        }
        return true;
    };
    if (eq_ci(s, "auto"))           return Preset::Auto;
    if (eq_ci(s, "fast"))           return Preset::Fast;
    if (eq_ci(s, "robust"))         return Preset::Robust;
    if (eq_ci(s, "high_fidelity"))  return Preset::HighFidelity;
    if (eq_ci(s, "high-fidelity"))  return Preset::HighFidelity;
    if (eq_ci(s, "highfidelity"))   return Preset::HighFidelity;
    return Preset::Auto;
}

}  // namespace pulsim::v1
