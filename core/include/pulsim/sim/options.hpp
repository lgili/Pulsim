#pragma once

// add-kernel-simulation-object: the run options and THE option x engine
// table.
//
// `kRunOptionTable` is the one declaration of every option a run takes
// (the keywords of `pulsim.simulate()`): which engines honour it, which
// value means "not requested", and the mechanism the others lack. The
// capability check (`option_refusals`), the routing of engine='auto'
// (`fixed_only`) and the Python surface (`pulsim.option_capabilities()`,
// simulate()'s refusals) all read it, so an option cannot be honoured on
// one engine and silently ignored on another: an engine outside its
// `engines` refuses it by name.
//
// `Defaults` is the one table of default values. `RunOptions` leaves
// every engine-dependent value as an empty optional ("not set" is never
// a sentinel); `to_simulation_options` / `to_trbdf2_options` resolve it
// for today's two drivers, exactly as simulate() did.

#include "pulsim/numeric/types.hpp"
#include "pulsim/pwl/device_pool.hpp"
#include "pulsim/solver/options.hpp"
#include "pulsim/solver/trbdf2_transient.hpp"
#include "pulsim/solver/unsupported.hpp"

#include <array>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace pulsim::sim {

using pwl::Engine;

/// The engines a run can land on, as table bits.
inline constexpr std::uint8_t kPwl = pwl::engine_bit(Engine::FixedTrap);
inline constexpr std::uint8_t kTrBdf2 = pwl::engine_bit(Engine::TrBdf2);
inline constexpr std::uint8_t kExact = pwl::engine_bit(Engine::Dsed);
inline constexpr std::uint8_t kEvery = kPwl | kTrBdf2 | kExact;
inline constexpr std::array<Engine, 3> kRunEngines{
    Engine::FixedTrap, Engine::TrBdf2, Engine::Dsed};

/// Which value of an option means "not requested" (so no engine is asked
/// to honour it). `Always`: every engine honours every value.
enum class Unset : std::uint8_t {
    Always,      // honoured everywhere; no value is a request to refuse
    None,        // None / nullopt
    Falsy,       // None, False, 0, empty
    NoneOrOne,   // store_every: 1 records every step
    NoneOrZero,  // max_dt_halvings: 0 asks for no retry, which no engine lacks
    Legacy,      // mode_selection: None / 'legacy' is every engine's settle
    Integrator,  // None; on pwl also 'kernel' / 'default' (its own integrator)
};

struct OptionSpec {
    std::string_view name;     // the simulate() keyword
    std::uint8_t engines;      // the engines that honour it
    Unset unset;
    std::string_view default_text;  // documented default (resolved: Defaults)
    std::string_view why;      // the mechanism the other engines lack
    [[nodiscard]] constexpr bool honoured_by(Engine e) const noexcept {
        return (engines & pwl::engine_bit(e)) != 0;
    }
};

/// The option x engine table. Order = simulate()'s signature order (it is
/// the order refusals are listed in). Python reads it through
/// `pulsim._pulsim.run_option_table()`; a test asserts its names equal
/// simulate()'s parameters.
inline constexpr std::array<OptionSpec, 47> kRunOptionTable{{
    {"engine", kEvery, Unset::Always, "'auto'", ""},
    {"rtol", kTrBdf2 | kExact, Unset::None, "1e-5 (trbdf2), 1e-6 (exact)",
     "rtol bounds a variable step's error estimate; a fixed step has none"},
    {"atol", kTrBdf2 | kExact, Unset::None, "1e-8 (trbdf2), 1e-9 (exact)",
     "atol bounds a variable step's error estimate; a fixed step has none"},
    {"dt_init", kTrBdf2 | kExact, Unset::None,
     "kernel's choice (trbdf2), 1e-9 (exact)",
     "dt_init is a variable-step controller's first step"},
    {"integrator", kExact, Unset::Integrator, "'auto' (exact)",
     "integrator selects an exact-engine integrator"},
    {"stiffness_threshold", kExact, Unset::None, "10.0",
     "stiffness_threshold tunes the exact engine's RK45/BDF2 dispatch"},
    {"h_bdf2", kExact, Unset::None, "1e-6",
     "h_bdf2 is the exact engine's BDF2 segment step"},
    {"t_start", kEvery, Unset::Always, "0.0", ""},
    {"switch_fn", kEvery, Unset::Always, "None", ""},
    {"b_extra_fn", kEvery, Unset::Always, "None", ""},
    {"step_observer", kPwl | kTrBdf2, Unset::None, "None",
     "step_observer runs a Python controller between steps"},
    {"start_from_dc_op", kPwl, Unset::Falsy, "False",
     "start_from_dc_op is a fixed-step entry point"},
    {"enable_nonlinear_refresh", kPwl | kTrBdf2, Unset::None,
     "None (stamp the Newton devices when there are any)",
     "enable_nonlinear_refresh governs a Newton solve the exact engine does "
     "not run"},
    {"max_newton_iterations", kPwl | kTrBdf2, Unset::None, "50",
     "max_newton_iterations bounds a Newton solve the exact engine does not "
     "run"},
    {"max_event_iterations", kPwl | kTrBdf2, Unset::None, "16",
     "max_event_iterations bounds a diode event iteration the exact engine "
     "does not run"},
    {"strict_event_iterations", kPwl, Unset::Falsy, "False",
     "strict_event_iterations belongs to the fixed-step diode iteration"},
    {"max_dt_halvings", kPwl, Unset::NoneOrZero, "6",
     "max_dt_halvings re-solves at sub-dt against a full-dt companion"},
    {"voltage_sanity_factor", kEvery, Unset::Always, "100", ""},
    {"store_every", kPwl, Unset::NoneOrOne, "1",
     "store_every decimates a uniform grid"},
    {"auto_regularize", kEvery, Unset::Always, "True", ""},
    {"tol_newton_dx", kPwl | kTrBdf2, Unset::None, "1e-9",
     "tol_newton_dx bounds a Newton solve the exact engine does not run"},
    {"tol_newton_res", kPwl | kTrBdf2, Unset::None, "1e-9",
     "tol_newton_res bounds a Newton solve the exact engine does not run"},
    {"enable_newton_line_search", kPwl | kTrBdf2, Unset::Falsy, "False",
     "enable_newton_line_search changes a Newton solve the exact engine "
     "does not run"},
    {"enable_newton_lm", kPwl | kTrBdf2, Unset::Falsy, "False",
     "enable_newton_lm changes a Newton solve the exact engine does not run"},
    {"newton_roundoff_budget", kPwl | kTrBdf2, Unset::None, "130",
     "newton_roundoff_budget bounds a Newton roundoff acceptance the exact "
     "engine does not make"},
    {"enable_substep_state_correction", kPwl, Unset::Falsy, "False",
     "enable_substep_state_correction is the fixed-step commutation split"},
    {"discontinuity_restart", kPwl, Unset::Falsy, "False",
     "discontinuity_restart is the fixed-step engine's restart rule"},
    {"resolution_diagnostics", kPwl, Unset::Falsy, "False",
     "resolution_diagnostics estimates a fixed step's truncation"},
    {"on_impulse", kEvery, Unset::Always, "'record'", ""},
    {"inductor_freeze_di_max", kPwl, Unset::None, "None (off)",
     "inductor_freeze_di_max is a fixed-step post-solve guard"},
    {"inductor_abs_clamp", kPwl, Unset::None, "None (off)",
     "inductor_abs_clamp is a fixed-step post-solve guard"},
    {"progress", kPwl | kExact, Unset::Falsy, "False",
     "progress has no per-step hook here"},
    {"initial_state", kEvery, Unset::Always, "None (the circuit's ICs)", ""},
    {"should_continue", kPwl | kTrBdf2, Unset::None, "None",
     "should_continue is a cancellation hook the exact engine does not poll"},
    {"closed_loops", kPwl | kTrBdf2, Unset::Falsy, "None",
     "closed_loops run Python controllers between steps"},
    {"live_stream", kPwl, Unset::None, "None",
     "live_stream needs the fixed-step kernel's per-step ring push"},
    {"mmc_arms", kPwl, Unset::Falsy, "None",
     "mmc_arms drives the fixed-step observer/b_extra pair"},
    {"on_cache", kPwl, Unset::None, "None",
     "on_cache refactors the fixed-step engine's PWL cache mid-run; the "
     "other engines have no such cache, so the hook would never fire and "
     "whatever it was closing (an electro-thermal loop, usually) would "
     "silently stay open"},
    {"controller_period", kPwl | kTrBdf2, Unset::None, "None",
     "controller_period is the cadence of a step_observer"},
    {"resume_from", kPwl | kTrBdf2, Unset::None, "None",
     "resume_from needs a snapshot the exact engine does not keep"},
    {"solver", kEvery, Unset::Always, "None", ""},
    {"sensitivities", kExact, Unset::None, "None",
     "sensitivities are computed on the exact path only"},
    {"mode_selection", kPwl | kExact, Unset::Legacy, "None ('legacy')",
     "mode_selection='shadow' / 'lcp' needs a settle hook this engine lacks"},
    {"allow_implicit_segments", kExact, Unset::None, "None",
     "allow_implicit_segments governs the exact engine's numeric segments"},
    // simulate()'s signature also has `dt` and `t_end` (every engine; the
    // run window) — rows so the table covers every parameter.
    {"t_end", kEvery, Unset::Always, "required", ""},
    {"dt", kEvery, Unset::Always,
     "None (pwl: required; trbdf2: kernel ceiling; exact: 1e-5)", ""},
    // Not a keyword: the circuit feature the router reads (C blocks).
    {"c_blocks", kPwl | kTrBdf2, Unset::Falsy, "",
     "C blocks sample on a fixed dt grid"},
}};

[[nodiscard]] constexpr const OptionSpec* find_option(std::string_view name) noexcept {
    for (const auto& o : kRunOptionTable) {
        if (o.name == name) return &o;
    }
    return nullptr;
}

/// THE default values. Every resolved default of a run is one of these.
struct Defaults {
    // Variable-step TR-BDF2 on the MNA (engine 'trbdf2' / 'auto').
    static constexpr Real rtol_variable = 1e-5;
    static constexpr Real atol_variable = 1e-8;
    // Event-driven exact engine ('exact' / 'dsed').
    static constexpr Real rtol_exact = 1e-6;
    static constexpr Real atol_exact = 1e-9;
    static constexpr Real dt_init_exact = 1e-9;
    static constexpr Real dt_max_exact = 1e-5;
    static constexpr Real stiffness_threshold = 10.0;
    static constexpr Real h_bdf2 = 1e-6;
    // Newton (pwl, trbdf2).
    static constexpr Size max_newton_iterations = 50;
    static constexpr Real tol_newton_dx = 1e-9;
    static constexpr Real tol_newton_res = 1e-9;
    static constexpr Size newton_roundoff_budget = 130;
    // Events and the fixed step.
    static constexpr Size max_event_iterations = 16;
    static constexpr Size max_dt_halvings = 6;
    static constexpr Size store_every = 1;
    static constexpr Real voltage_sanity_factor = 100.0;
};

enum class EngineChoice : std::uint8_t { Auto, Pwl, TrBdf2, Exact };
enum class ModeSelection : std::uint8_t { Legacy, Shadow, Lcp };
enum class OnImpulse : std::uint8_t { Record, Warn, Error };

/// The kernel-level run options: one field per simulate() keyword the
/// kernel reads. Python-composed attachments (gate sources, observers,
/// controllers, streams) are not options; `Simulation` takes them.
struct RunOptions {
    EngineChoice engine = EngineChoice::Auto;
    Real t_start = Real{0};
    Real t_end = Real{0};
    /// pwl: the fixed step (required); trbdf2 / exact: the step ceiling.
    std::optional<Real> dt;

    // Variable step (trbdf2, exact).
    std::optional<Real> rtol, atol, dt_init;
    // Exact engine only.
    std::optional<Real> stiffness_threshold, h_bdf2;

    // Newton (pwl, trbdf2).
    std::optional<bool> enable_nonlinear_refresh;
    std::optional<Size> max_newton_iterations;
    std::optional<Real> tol_newton_dx, tol_newton_res;
    std::optional<bool> enable_newton_line_search, enable_newton_lm;
    std::optional<Size> newton_roundoff_budget;

    // Events.
    std::optional<Size> max_event_iterations;
    bool strict_event_iterations = false;
    ModeSelection mode_selection = ModeSelection::Legacy;
    OnImpulse on_impulse = OnImpulse::Record;
    /// A/B switch of the fixed engine's mode guard (not a user option).
    bool mode_guard = true;

    // Fixed step (pwl).
    std::optional<Size> max_dt_halvings;
    std::optional<Size> store_every;
    bool start_from_dc_op = false;
    std::optional<bool> enable_substep_state_correction;
    std::optional<bool> discontinuity_restart;
    std::optional<bool> resolution_diagnostics;
    std::optional<Real> inductor_freeze_di_max, inductor_abs_clamp;

    // Every engine.
    bool auto_regularize = true;
    std::optional<Real> voltage_sanity_factor;
};

/// A run option that is invalid whatever the engine.
class OptionsError : public std::invalid_argument {
public:
    OptionsError(std::string option, const std::string& message)
        : std::invalid_argument(message), option_{std::move(option)} {}
    [[nodiscard]] const std::string& option() const noexcept { return option_; }
private:
    std::string option_;
};

/// The table rows `o` requests (an option set to a value some engine may
/// not honour), in table order.
[[nodiscard]] inline std::vector<std::string_view>
requested_options(const RunOptions& o) {
    std::vector<std::string_view> r;
    auto on = [&](bool hit, std::string_view n) { if (hit) r.push_back(n); };
    on(o.rtol.has_value(), "rtol");
    on(o.atol.has_value(), "atol");
    on(o.dt_init.has_value(), "dt_init");
    on(o.stiffness_threshold.has_value(), "stiffness_threshold");
    on(o.h_bdf2.has_value(), "h_bdf2");
    on(o.start_from_dc_op, "start_from_dc_op");
    on(o.enable_nonlinear_refresh.has_value(), "enable_nonlinear_refresh");
    on(o.max_newton_iterations.has_value(), "max_newton_iterations");
    on(o.max_event_iterations.has_value(), "max_event_iterations");
    on(o.strict_event_iterations, "strict_event_iterations");
    on(o.max_dt_halvings.value_or(0) != 0, "max_dt_halvings");
    on(o.store_every.value_or(1) != 1, "store_every");
    on(o.tol_newton_dx.has_value(), "tol_newton_dx");
    on(o.tol_newton_res.has_value(), "tol_newton_res");
    on(o.enable_newton_line_search.value_or(false), "enable_newton_line_search");
    on(o.enable_newton_lm.value_or(false), "enable_newton_lm");
    on(o.newton_roundoff_budget.has_value(), "newton_roundoff_budget");
    on(o.enable_substep_state_correction.value_or(false),
       "enable_substep_state_correction");
    on(o.discontinuity_restart.value_or(false), "discontinuity_restart");
    on(o.resolution_diagnostics.value_or(false), "resolution_diagnostics");
    on(o.inductor_freeze_di_max.has_value(), "inductor_freeze_di_max");
    on(o.inductor_abs_clamp.has_value(), "inductor_abs_clamp");
    on(o.mode_selection != ModeSelection::Legacy, "mode_selection");
    // Keep table order (the order refusals are listed in).
    std::vector<std::string_view> ordered;
    for (const auto& spec : kRunOptionTable) {
        for (auto n : r) {
            if (n == spec.name) { ordered.push_back(n); break; }
        }
    }
    return ordered;
}

/// The requested options engine `e` does not honour, in table order.
[[nodiscard]] inline std::vector<std::string_view>
option_refusals(const std::vector<std::string_view>& requested, Engine e) {
    std::vector<std::string_view> out;
    for (auto n : requested) {
        const auto* spec = find_option(n);
        if (spec != nullptr && !spec->honoured_by(e)) out.push_back(n);
    }
    return out;
}

// ---- Resolution for today's drivers (exactly what simulate() did) ----------

/// The fixed-step driver's options. Throws OptionsError on a value no
/// engine accepts (a zero step, store_every = 0, a zero Newton budget).
[[nodiscard]] inline solver::SimulationOptions
to_simulation_options(const RunOptions& o) {
    solver::SimulationOptions s;
    s.t_start = o.t_start;
    s.t_end = o.t_end;
    s.dt = o.dt.value_or(Real{0});
    if (o.store_every) {
        if (*o.store_every < 1) {
            throw OptionsError("store_every",
                "simulate(store_every=0): must be >= 1 (1 records every step)");
        }
        s.store_every = *o.store_every;
    }
    if (o.max_newton_iterations) {
        if (*o.max_newton_iterations < 1) {
            throw OptionsError("max_newton_iterations",
                "simulate(max_newton_iterations=0): must be >= 1 (None keeps "
                "the default, 50)");
        }
        s.max_newton_iterations = *o.max_newton_iterations;
    }
    if (o.max_event_iterations) s.max_event_iterations = *o.max_event_iterations;
    s.strict_event_iterations = o.strict_event_iterations;
    if (o.max_dt_halvings) s.max_dt_halvings = *o.max_dt_halvings;
    if (o.tol_newton_dx) s.tol_newton_dx = *o.tol_newton_dx;
    if (o.tol_newton_res) s.tol_newton_res = *o.tol_newton_res;
    if (o.enable_newton_line_search) {
        s.enable_newton_line_search = *o.enable_newton_line_search;
    }
    if (o.enable_newton_lm) s.enable_newton_lm = *o.enable_newton_lm;
    if (o.newton_roundoff_budget) {
        s.newton_roundoff_budget = *o.newton_roundoff_budget;
    }
    if (o.enable_substep_state_correction) {
        s.enable_substep_state_correction = *o.enable_substep_state_correction;
    }
    if (o.discontinuity_restart) s.discontinuity_restart = *o.discontinuity_restart;
    s.on_impulse_error = o.on_impulse == OnImpulse::Error;
    s.mode_selection = static_cast<int>(o.mode_selection);
    s.mode_guard = o.mode_guard;
    if (o.resolution_diagnostics) {
        s.resolution_diagnostics = *o.resolution_diagnostics;
    }
    // 0 is the documented OFF of both guards (and the driver's default).
    if (o.inductor_freeze_di_max && *o.inductor_freeze_di_max > Real{0}) {
        s.inductor_freeze_di_max = *o.inductor_freeze_di_max;
    }
    if (o.inductor_abs_clamp && *o.inductor_abs_clamp > Real{0}) {
        s.inductor_abs_clamp = *o.inductor_abs_clamp;
    }
    return s;
}

/// The variable-step driver's options (tick periods are attachments).
[[nodiscard]] inline solver::TrBdf2Options
to_trbdf2_options(const RunOptions& o) {
    solver::TrBdf2Options s;
    s.t_start = o.t_start;
    s.t_end = o.t_end;
    s.rtol = o.rtol.value_or(Defaults::rtol_variable);
    s.atol = o.atol.value_or(Defaults::atol_variable);
    // 0 = the kernel picks (fastest periodic source, narrowest pulse).
    s.h_init = o.dt_init.value_or(Real{0});
    s.h_max = o.dt.value_or(Real{0});
    if (o.max_event_iterations) {
        // 0 turns the fixed engine's diode iteration off; the variable
        // step's event localisation has no off switch (it used to be
        // dropped here in silence and 16 applied).
        if (*o.max_event_iterations == 0) {
            throw UnsupportedError("trbdf2", "max_event_iterations", "", "",
                "simulate(engine='trbdf2', max_event_iterations=0): 0 turns "
                "the fixed-step diode iteration off; the variable-step "
                "engine's event localisation has no off switch. Pass a "
                "count >= 1, or engine='pwl' with a dt.");
        }
        s.max_event_iterations = *o.max_event_iterations;
    }
    if (o.max_newton_iterations) {
        if (*o.max_newton_iterations < 1) {
            throw OptionsError("max_newton_iterations",
                "simulate(max_newton_iterations=0): must be >= 1 (None keeps "
                "the default, 50)");
        }
        s.max_newton_iterations = *o.max_newton_iterations;
    }
    // A value, as on the fixed engine (0 used to mean "default" here).
    if (o.tol_newton_dx) s.tol_newton_dx = *o.tol_newton_dx;
    if (o.tol_newton_res) s.tol_newton_res = *o.tol_newton_res;
    s.enable_newton_line_search = o.enable_newton_line_search.value_or(false);
    s.enable_newton_lm = o.enable_newton_lm.value_or(false);
    if (o.newton_roundoff_budget) {
        s.newton_roundoff_budget = *o.newton_roundoff_budget;
    }
    s.on_impulse_error = o.on_impulse == OnImpulse::Error;
    return s;
}

}  // namespace pulsim::sim
