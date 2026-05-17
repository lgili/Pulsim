#pragma once

// =============================================================================
// Auto-parasitics: pre-flight topology analysis + automatic snubber sizing
// =============================================================================
// PulsimCore 0.10.0a12 — `boost-pfc-auto-parasitics` change.
//
// Problem
// -------
// Power-electronics users who build a boost / buck-boost / flyback converter
// in PWL Ideal mode hit a wall: the switch's parasitic output capacitance
// `C_oss` is responsible for setting the V_sw rise-time during the inductor
// commutation, and Tustin (the underlying trapezoidal integrator) has zero
// numerical damping on a pure L-C tank. Result: a 100 µH boost L with the
// default C_oss = 10 nF rings to ±500 V around the bus rail.
//
// The closed-form V_sw overshoot for a switch fed by an inductor with peak
// current I_peak in PWL Ideal mode is:
//
//     V_overshoot = I_peak · √(L / C_oss)
//
// Users don't (and shouldn't) have to know this. The runtime should detect
// the topology, size C_oss automatically so the predicted overshoot is
// bounded, and — when even the largest reasonable C_oss can't make PWL
// Ideal feasible (high f_sw + small duty-off window) — fall back to the
// smooth Shichman-Hodges (Behavioral) path that handles the LC dynamics
// without an artificial damping budget.
//
// This header provides the topology walker, sizing math, and report types.
// `RuntimeCircuit::auto_configure_parasitics()` is the entry point users
// actually call (directly or via `SimulationOptions::auto_parasitics`).
//
// Math notes
// ----------
// The closed-form sizing rules here mirror `python/pulsim/snubber.py`. Both
// are derived from the same physics so users get the same recommendation
// regardless of which surface they call it from.
//
//   recommend_C_oss(L, I_peak, V_bus, max_overshoot_frac):
//       V_overshoot_max = max_overshoot_frac · V_bus
//       C_oss           = (I_peak / V_overshoot_max)² · L
//
//   feasibility check (when f_sw provided):
//       t_rise = C_oss · V_bus / I_peak     (linear approx)
//       t_off  = duty_off / f_sw
//       feasible := (t_rise < t_off)
//
//   switching_loss_estimate(C, V, f_sw) = ½·C·V²·f_sw  (hard-switched)

#include "pulsim/v1/device_base.hpp"  // for Real / Scalar
#include <cmath>
#include <string>
#include <vector>
#include <sstream>
#include <iomanip>

namespace pulsim::v1 {

// =============================================================================
// Public types
// =============================================================================

/// One detected (Inductor → Switch) topological adjacency.
/// The `severity` field summarizes whether the current configuration is OK,
/// borderline, or numerically infeasible — independent of whether the
/// auto-configurer chose to mutate the device.
struct TopologyIssue {
    std::string switch_name;        // Name of MOSFET / IGBT / IdealDiode
    std::string inductor_name;      // Name of the Inductor in series

    Real L_henry            = 0.0;  // Inductance (H)
    Real I_peak_estimate    = 0.0;  // A — from initial_current heuristic
    Real V_bus_estimate     = 0.0;  // V — from cap initial_voltage heuristic
    Real predicted_overshoot = 0.0; // V above clamp rail
    Real current_C_oss      = 0.0;  // F — as configured before this analysis

    enum class Severity : std::uint8_t {
        Info,      // Predicted overshoot < 20 % of V_bus → no action needed
        Warning,   // 20 % ≤ overshoot < 100 % → C_oss bump recommended
        Critical   // Overshoot ≥ 100 % of V_bus → PWL Ideal infeasible
    };
    Severity severity = Severity::Info;
};

/// Single mutation the auto-configurer applied (or would apply when called
/// in `dry_run=true` mode). The `device_name` is the device being modified.
struct ParasiticAction {
    std::string device_name;
    enum class Kind : std::uint8_t {
        None,             // No change (e.g. user already set C_oss)
        SetCoss,          // Sized C_oss = new_C_oss F across switch
        DropToBehavioral, // Switched mode → SwitchingMode::Behavioral
        AddRCSnubber      // Future: add external R-C across switch
    };
    Kind kind = Kind::None;
    Real new_C_oss   = 0.0;
    Real snub_R      = 0.0;
    Real snub_C      = 0.0;
    std::string rationale;            // Why we made this choice
};

/// Aggregate report returned by `auto_configure_parasitics`.
struct TopologyReport {
    std::vector<TopologyIssue> issues;
    std::vector<ParasiticAction> actions;
    bool ran_pre_simulation = false;  // v1 always false; v2 can set true
    std::string summary;              // Human-readable multi-line text

    [[nodiscard]] std::size_t num_critical() const noexcept {
        std::size_t n = 0;
        for (const auto& it : issues) {
            if (it.severity == TopologyIssue::Severity::Critical) ++n;
        }
        return n;
    }
    [[nodiscard]] std::size_t num_actions() const noexcept {
        std::size_t n = 0;
        for (const auto& a : actions) {
            if (a.kind != ParasiticAction::Kind::None) ++n;
        }
        return n;
    }
};

/// Configuration for `RuntimeCircuit::auto_configure_parasitics`.
struct AutoParasiticsOptions {
    /// Master enable. `false` = behave as if this feature didn't exist.
    /// Default ON so cold-start users get convergent boost circuits out of
    /// the box; opt-out is a single field assignment for power users who
    /// want explicit control.
    bool enabled = true;

    /// Tolerated V_sw overshoot as fraction of V_bus. 0.5 = 50 % over the
    /// clamp rail is the practical limit before we drop to Behavioral mode.
    /// Values < 0.5 produce C_oss that can't charge to V_bus inside the
    /// switching period (boost transfer breaks) → infeasible verdict.
    Real max_overshoot_frac = 0.5;

    /// Set to write a one-block summary to stderr after configuration.
    /// Default ON because the user explicitly asked for "tell me what you
    /// did" — silent reconfiguration is a debugging nightmare.
    bool verbose = true;

    /// Skip devices that already have user-set parasitics. A MOSFET with
    /// `Params::C_oss > 0` or a non-Auto `switching_mode_` is considered
    /// "user knows what they're doing" and the analyzer never touches it.
    bool respect_user_overrides = true;

    /// Default operating-point estimates when the circuit doesn't have
    /// pre-charged initial conditions on the inductor or output cap. These
    /// are intentionally conservative (= small C_oss, low loss budget)
    /// because they're the fallback for cold-start; if the cold-start
    /// then ramps a larger current, the user can still see the runtime
    /// V_sw overshoot in their results and rerun with `max_overshoot_frac`
    /// tightened.
    Real fallback_I_peak  = 5.0;     // A
    Real fallback_V_bus   = 100.0;   // V
};

// =============================================================================
// Closed-form sizing math (port of python/pulsim/snubber.py)
// =============================================================================

/// V_overshoot = |I_peak| · √(L / C). Inputs must be > 0.
[[nodiscard]] inline Real predict_overshoot(Real L, Real C, Real I_peak) noexcept {
    if (L <= Real{0} || C <= Real{0}) return Real{0};
    return std::abs(I_peak) * std::sqrt(L / C);
}

/// Recommend C_oss for a switch fed by inductor L carrying I_peak into
/// V_bus, with `max_overshoot_frac · V_bus` tolerated overshoot.
[[nodiscard]] inline Real recommend_C_oss(Real L, Real I_peak,
                                           Real V_bus,
                                           Real max_overshoot_frac) noexcept {
    if (L <= Real{0} || I_peak <= Real{0} || V_bus <= Real{0}) return Real{0};
    if (max_overshoot_frac <= Real{0}) return Real{0};
    const Real V_overshoot_max = max_overshoot_frac * V_bus;
    return (I_peak / V_overshoot_max) * (I_peak / V_overshoot_max) * L;
}

/// Per-cycle "C_oss switching loss" for a hard-switched cell: ½·C·V²·f_sw.
[[nodiscard]] inline Real coss_switching_loss(Real C, Real V, Real f_sw) noexcept {
    if (C <= Real{0} || V <= Real{0} || f_sw <= Real{0}) return Real{0};
    return Real{0.5} * C * V * V * f_sw;
}

/// Feasibility check: t_rise (linear, constant I_peak) < OFF interval.
[[nodiscard]] inline bool coss_feasible_in_period(Real C, Real V_bus,
                                                    Real I_peak,
                                                    Real duty_off,
                                                    Real f_sw) noexcept {
    if (f_sw <= Real{0} || I_peak <= Real{0}) return true;
    const Real t_rise = C * V_bus / I_peak;
    const Real t_off  = duty_off / f_sw;
    return t_rise < t_off;
}

// =============================================================================
// Summary formatter (used by `auto_configure_parasitics`)
// =============================================================================

/// Render a TopologyReport into a multi-line human-readable string. The
/// summary is also assigned to `report.summary` by the caller. Lives here
/// so it stays decoupled from any specific stream type.
[[nodiscard]] inline std::string format_topology_report(const TopologyReport& r) {
    std::ostringstream os;
    os << "[pulsim] auto-parasitics: ";
    if (r.issues.empty()) {
        os << "no L-switch adjacencies detected — nothing to configure.";
        return os.str();
    }
    os << r.issues.size() << " switch-inductor pair(s) detected, "
       << r.num_actions() << " action(s) applied"
       << (r.num_critical() ? " (some critical)\n" : "\n");

    for (std::size_t i = 0; i < r.issues.size(); ++i) {
        const auto& it = r.issues[i];
        os << "  ["
           << (it.severity == TopologyIssue::Severity::Info     ? "info"
             : it.severity == TopologyIssue::Severity::Warning  ? "warn"
                                                                 : "CRIT")
           << "] " << it.switch_name << " ← " << it.inductor_name
           << "  L=" << std::fixed << std::setprecision(1) << (it.L_henry*1e6)
           << "µH  I_peak=" << std::setprecision(2) << it.I_peak_estimate << "A"
           << "  V_bus=" << std::setprecision(0) << it.V_bus_estimate << "V"
           << "  V_overshoot=" << std::setprecision(0) << it.predicted_overshoot << "V"
           << " (" << std::setprecision(0)
                   << (it.V_bus_estimate > 0
                       ? it.predicted_overshoot / it.V_bus_estimate * 100.0
                       : 0.0) << "%)"
           << '\n';
        if (i < r.actions.size()) {
            const auto& a = r.actions[i];
            os << "         action: ";
            switch (a.kind) {
                case ParasiticAction::Kind::None:
                    os << "none (user override respected)"; break;
                case ParasiticAction::Kind::SetCoss:
                    os << "C_oss = " << std::setprecision(1)
                       << (a.new_C_oss*1e9) << " nF"; break;
                case ParasiticAction::Kind::DropToBehavioral:
                    os << "→ Behavioral mode (PWL Ideal infeasible)"; break;
                case ParasiticAction::Kind::AddRCSnubber:
                    os << "+RC snubber R="
                       << std::setprecision(1) << a.snub_R << "Ω, C="
                       << (a.snub_C*1e9) << "nF"; break;
            }
            if (!a.rationale.empty()) os << " — " << a.rationale;
            os << '\n';
        }
    }
    return os.str();
}

}  // namespace pulsim::v1
