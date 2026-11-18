#pragma once

// add-kernel-simulation-object: the kernel object that IS a simulation.
//
//   sim::Simulation s{circuit, options};   // plans: preflight, route,
//                                          // capability check, refresh
//   s.set_switch_fn(...); s.add_tick(T, f); // run-time attachments
//   auto r = s.run();                       // or run_until(t) + snapshot()
//
// A facade over today's drivers (`run_transient`, `run_transient_trbdf2`):
// it owns no numerics. What it owns is what used to exist only in
// `pulsim.simulate()`: the preflight on a private copy of the circuit,
// engine routing with recorded reasons, the option x engine refusals of
// `kRunOptionTable`, the device capability check, the Newton device
// stamps (no caller-supplied refresh), the cache, the undriven-switch
// default and the voltage-sanity check. The exact (event-driven) engine is
// composed in Python today; `Simulation` refuses it by name.

#include "pulsim/ir/circuit.hpp"
#include "pulsim/pwl/cache.hpp"
#include "pulsim/pwl/diode_event_state.hpp"
#include "pulsim/pwl/nonlinear_refresh_mosfet_level1.hpp"
#include "pulsim/sim/options.hpp"
#include "pulsim/solver/run_transient.hpp"
#include "pulsim/solver/trbdf2_transient.hpp"
#include "pulsim/solver/unsupported.hpp"
#include "pulsim/solver/voltage_sanity.hpp"

#include <format>
#include <functional>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace pulsim::sim {

enum class Severity : std::uint8_t { Info, Warning, Error };

/// A non-error message of a run, in order of occurrence.
struct Diagnostic {
    Severity severity = Severity::Warning;
    std::string code;     // stable: "preflight_ties", "undriven_switches", ...
    std::string message;
    std::string device;
    std::optional<Real> t;
};

/// What the constructor decided.
struct RunPlan {
    Engine engine = Engine::FixedTrap;
    EngineChoice asked = EngineChoice::Auto;
    std::vector<std::string> route_reasons;
    std::vector<ir::Tie> ties;          // inserted by the preflight
    std::vector<std::string> undriven;  // controlled switches nothing drives
};

/// The static Newton chain of a transient driver. `nullopt`: the driver
/// stamps the Newton devices itself when the pool has any; `true` forces
/// the chain; `false` on a circuit with a static Newton device is refused
/// (it would simulate it as an open circuit). The bindings call this too.
[[nodiscard]] inline pwl::NonlinearRefreshFn transient_refresh(
    const topology::Graph& graph, const pwl::DevicePool& pool,
    std::optional<bool> enable, std::string_view engine) {
    using pwl::SolveClass;
    if (!enable.has_value()) return {};
    if (*enable) {
        return pool.requires_static_refresh()
                   ? pwl::make_combined_diode_mosfet_refresh()
                   : pwl::NonlinearRefreshFn{};
    }
    for (Index b = 0; b < graph.num_branches(); ++b) {
        if (!pool.is_registered(b)
            || pwl::kind_caps(pool.kind_of(b)).cls != SolveClass::NewtonStatic) {
            continue;
        }
        const std::string name(graph.branch_name(b));
        throw UnsupportedError(std::string(engine), "enable_nonlinear_refresh",
            name, std::string(pwl::stored_kind_name(pool.kind_of(b))),
            std::string(engine) + ": enable_nonlinear_refresh=False would "
            "simulate '" + name + "' (a Newton device: smooth / Shockley "
            "diode, MOSFET, IGBT) as an open circuit. Pass None (the "
            "default) or True.");
    }
    return {};
}

/// Controlled switches (the bits no diode owns): what a gate source drives.
[[nodiscard]] inline std::vector<Size> controlled_switches(
    const topology::Graph& graph, const pwl::DevicePool& pool) {
    pwl::DiodeEventState diodes(graph, pool);
    const auto owned = diodes.diode_owned_bits();
    std::vector<Size> out;
    for (Size i = 0; i < graph.num_switches(); ++i) {
        if (!owned.get(i)) out.push_back(i);
    }
    return out;
}

class Simulation {
public:
    using TickFn = std::function<void(Real, const Vector&)>;

    /// Plans the run; throws `OptionsError`, `UnsupportedError` or
    /// `std::invalid_argument` before anything is solved.
    Simulation(ir::Circuit circuit, RunOptions options)
        : circuit_(std::move(circuit)), opts_(std::move(options)) {
        plan_.asked = opts_.engine;
        if (!(opts_.t_end > opts_.t_start)) {
            throw OptionsError("t_end", std::format(
                "Simulation: t_end ({}) must be greater than t_start ({})",
                opts_.t_end, opts_.t_start));
        }
        if (opts_.dt && !(*opts_.dt > Real{0})) {
            throw OptionsError("dt", std::format(
                "Simulation: dt must be positive, got {}", *opts_.dt));
        }
        if (!circuit_.controls().empty()) {
            throw UnsupportedError(engine_label(opts_.engine), "c_blocks", "", "",
                "Simulation: the circuit declares C blocks / graph-driven "
                "sources, which pulsim.simulate() composes in Python; the "
                "kernel object does not run them yet.");
        }
        // ---- Preflight on the run's own copy (the caller's circuit is
        //      never changed).
        if (opts_.auto_regularize) {
            auto reg = circuit_.with_preflight(pwl::PreflightOptions{});
            if (!reg.ties.empty()) {
                plan_.ties = reg.ties;
                diags_.push_back({Severity::Warning, "preflight_ties",
                                  reg.report.summary(), "", std::nullopt});
                circuit_ = std::move(reg.circuit);
            }
        }
        builder_.emplace(circuit_.to_builder());
        route_();
        const auto e = plan_.engine;
        // ---- Capability check: every refusal, not only the first.
        const auto refused = option_refusals(requested_options(opts_), e);
        if (!refused.empty()) {
            std::string names;
            for (auto n : refused) {
                if (!names.empty()) names += ", ";
                names += std::string(n);
            }
            std::string why;
            for (auto n : refused) {
                why += "; " + std::string(find_option(n)->why);
            }
            throw UnsupportedError(std::string(pwl::engine_name(e)),
                std::string(refused.front()), "", "",
                std::format("Simulation(engine='{}'): {} cannot be honoured "
                            "here{}.", pwl::engine_name(e), names, why));
        }
        if (const auto b = find_unsupported(graph(), pool(), e)) {
            throw unsupported_device(graph(), pool(), e, *b, std::format(
                "Simulation: engine '{}' cannot simulate '{}' ({})",
                pwl::engine_name(e), graph().branch_name(*b),
                pwl::stored_kind_name(pool().kind_of(*b))));
        }
        require_branch_form_supported(graph(), pool(), e);
        refresh_ = transient_refresh(graph(), pool(),
                                     opts_.enable_nonlinear_refresh,
                                     pwl::engine_name(e));
        t_cursor_ = opts_.t_start;
    }

    // ---- Attachments (things that are not part of the circuit) ----------
    void set_switch_fn(solver::SwitchScheduleFn fn,
                       std::function<Real(Real)> next_edge = {}) {
        switch_fn_ = std::move(fn);
        next_edge_ = std::move(next_edge);
        plan_.undriven.clear();
    }
    void set_source_term(solver::BExtraFn fn,
                         std::function<Real(Real)> next_jump = {}) {
        b_extra_ = std::move(fn);
        next_jump_ = std::move(next_jump);
    }
    /// Every step (pwl only: the variable-step engine has no grid to ride).
    void set_step_observer(solver::StepObserverFn fn) {
        if (fn && plan_.engine != Engine::FixedTrap) {
            throw UnsupportedError(std::string(pwl::engine_name(plan_.engine)),
                "step_observer", "", "",
                "Simulation: an every-step observer needs the fixed grid; add "
                "it as a tick with its period (add_tick) on this engine.");
        }
        observer_ = std::move(fn);
    }
    /// A control tick at t_start + k*period (pwl: the first grid point at
    /// or after it; trbdf2: landed exactly).
    void add_tick(Real period, TickFn fn) {
        if (!(period > Real{0})) {
            throw OptionsError("tick", std::format(
                "Simulation.add_tick: period must be positive, got {}", period));
        }
        tick_periods_.push_back(period);
        ticks_.push_back(std::move(fn));
    }
    void set_cancel(solver::ShouldContinueFn fn) { cancel_ = std::move(fn); }
    void set_initial_state(Vector x) { x0_ = std::move(x); }

    // ---- Queries -----------------------------------------------------------
    [[nodiscard]] const RunPlan& plan() const noexcept { return plan_; }
    [[nodiscard]] const RunOptions& options() const noexcept { return opts_; }
    /// The circuit the run solves (the caller's, plus any preflight ties).
    [[nodiscard]] const ir::Circuit& circuit() const noexcept { return circuit_; }
    [[nodiscard]] const builder::CircuitBuilder& builder() const noexcept {
        return *builder_;
    }
    [[nodiscard]] std::span<const Diagnostic> diagnostics() const noexcept {
        return diags_;
    }
    [[nodiscard]] const solver::TrBdf2Stats& trbdf2_stats() const noexcept {
        return stats_;
    }
    /// The state after the last completed run call (invalid before one).
    [[nodiscard]] const solver::SolverSnapshot& snapshot() const noexcept {
        return snapshot_;
    }
    /// Continue from `s`: the next run call starts at `s.t` with its state.
    void restore(const solver::SolverSnapshot& s) {
        if (!s.valid) {
            throw OptionsError("snapshot",
                "Simulation.restore: the snapshot is not valid (it came from a "
                "run that produced none)");
        }
        snapshot_ = s;
        t_cursor_ = s.t;
    }

    // ---- Running -------------------------------------------------------------
    solver::SimulationResult run() { return run_until(opts_.t_end); }

    /// Runs from where the last call (or `restore`) ended to `t`. On the
    /// fixed engine the continuation is the exact snapshot resume, so a
    /// split run equals a continuous one when `t` is on the step grid.
    solver::SimulationResult run_until(Real t) {
        if (!(t > t_cursor_) || t > opts_.t_end) {
            throw OptionsError("t", std::format(
                "Simulation.run_until({}): must be after {} and not after "
                "t_end = {}", t, t_cursor_, opts_.t_end));
        }
        const solver::SolverSnapshot* resume =
            snapshot_.valid ? &snapshot_ : nullptr;
        const Vector* x0 = (resume == nullptr && x0_) ? &*x0_ : nullptr;
        const auto& g = graph();
        const auto& p = pool();
        if (!switch_fn_ && plan_.asked == EngineChoice::Auto
            && plan_.engine == Engine::TrBdf2
            && !controlled_switches(g, p).empty()) {
            // simulate()'s rule: the two engines' undriven-switch defaults
            // differ, so 'auto' does not pick one for the caller.
            throw UnsupportedError("trbdf2", "switch_fn", "", "",
                "Simulation: controlled switch(es) with no switch_fn (the two "
                "engines' undriven-switch defaults differ). Attach a "
                "switch_fn, or set dt for the fixed-step engine.");
        }
        auto switch_fn = switch_fn_ ? switch_fn_ : default_switch_fn_();
        pwl::PwlStateSpaceCache cache(g, p);
        cache.share_symbolic(circuit_.symbolic_plan());
        solver::SimulationResult res;
        if (plan_.engine == Engine::FixedTrap) {
            auto so = to_simulation_options(opts_);
            so.t_start = t_cursor_;
            so.t_end = t;
            cache.build_lazy(so.dt);
            events::TickPlan tp;
            for (const Real T : tick_periods_) {
                tp.rates.push_back(events::Rate{T, Real{0}, opts_.t_start});
            }
            tp.fn = [this](std::uint32_t j, const events::TickInfo&, Real tt,
                           const Vector& x) { ticks_[j](tt, x); };
            res = solver::run_transient(cache, g, p, so, switch_fn, b_extra_,
                    opts_.start_from_dc_op && resume == nullptr, refresh_,
                    observer_, x0, cancel_, resume, next_jump_, next_edge_,
                    ticks_.empty() ? nullptr : &tp);
        } else {
            auto to = to_trbdf2_options(opts_);
            to.t_start = t_cursor_;
            to.t_end = t;
            to.observer_periods = tick_periods_;
            cache.build_lazy(to.h_max > Real{0} ? to.h_max
                                                : (t - t_cursor_) / Real{1000});
            std::function<void(Size, Real, const Vector&)> tick_fn;
            if (!ticks_.empty()) {
                tick_fn = [this](Size j, Real tt, const Vector& x) {
                    ticks_[j](tt, x);
                };
            }
            std::optional<Vector> x0o;
            if (x0 != nullptr) x0o = *x0;
            res = solver::run_transient_trbdf2(cache, g, p, to, switch_fn,
                    b_extra_, x0o, &stats_, next_edge_, cancel_, {}, refresh_,
                    resume, tick_fn);
        }
        snapshot_ = res.final_snapshot;
        t_cursor_ = t;
        check_voltage_(res);
        return res;
    }

private:
    [[nodiscard]] const topology::Graph& graph() const { return builder_->graph(); }
    [[nodiscard]] const pwl::DevicePool& pool() const { return builder_->pool(); }

    static std::string engine_label(EngineChoice c) {
        switch (c) {
            case EngineChoice::Pwl: return "pwl";
            case EngineChoice::TrBdf2: return "trbdf2";
            case EngineChoice::Exact: return "dsed";
            case EngineChoice::Auto: break;
        }
        return "auto";
    }

    /// Engine routing (simulate()'s rule): an explicit dt requests the
    /// fixed step; without one the variable step serves when nothing
    /// blocks it, else the run is refused with the reasons.
    void route_() {
        switch (opts_.engine) {
            case EngineChoice::Exact:
                throw UnsupportedError("dsed", "engine", "", "",
                    "Simulation: the exact (event-driven) engine is composed "
                    "in Python today; use pulsim.simulate(engine='exact'). "
                    "The kernel object runs 'pwl' and 'trbdf2'.");
            case EngineChoice::Pwl:
                if (!opts_.dt) {
                    throw OptionsError("dt",
                        "Simulation(engine='pwl'): a positive dt is required");
                }
                plan_.engine = Engine::FixedTrap;
                return;
            case EngineChoice::TrBdf2:
                plan_.engine = Engine::TrBdf2;
                return;
            case EngineChoice::Auto:
                break;
        }
        std::vector<std::string> why;
        for (auto n : option_refusals(requested_options(opts_), Engine::TrBdf2)) {
            why.emplace_back(find_option(n)->why);
        }
        if (find_unsupported(graph(), pool(), Engine::TrBdf2)) {
            why.emplace_back("a device kind the variable-step engine refuses");
        }
        if (opts_.dt) {
            if (why.empty()) {
                why.emplace_back("an explicit dt was given, which requests a "
                                 "fixed step");
            }
            plan_.route_reasons = std::move(why);
            plan_.engine = Engine::FixedTrap;
            return;
        }
        if (!why.empty()) {
            std::string all;
            for (const auto& w : why) all += (all.empty() ? "" : "; ") + w;
            throw UnsupportedError("trbdf2", "", "", "", std::format(
                "Simulation: this circuit cannot use the variable-step engine "
                "— {}. The fixed-step engine can run it, but needs a step: "
                "set dt.", all));
        }
        plan_.engine = Engine::TrBdf2;
    }

    /// No gate source: the engines' defaults (pwl all-CLOSED with a
    /// diagnostic when controlled switches exist; trbdf2 all-OPEN).
    solver::SwitchScheduleFn default_switch_fn_() {
        const auto n = graph().num_switches();
        topology::SwitchStateMask m(n);
        if (plan_.engine == Engine::FixedTrap) {
            const auto ctl = controlled_switches(graph(), pool());
            if (!ctl.empty() && plan_.undriven.empty()) {
                std::string idx;
                for (auto i : ctl) {
                    idx += (idx.empty() ? "" : ", ") + std::to_string(i);
                    plan_.undriven.push_back(switch_name_(i));
                }
                diags_.push_back({Severity::Warning, "undriven_switches",
                    std::format("no switch_fn was given, but the circuit has "
                                "{} controlled switch(es) (indices [{}]). They "
                                "default to ALL CLOSED.", ctl.size(), idx),
                    "", std::nullopt});
            }
            for (Size i = 0; i < n; ++i) m.set(i, true);
        }
        return [m](Real) { return m; };
    }

    [[nodiscard]] std::string switch_name_(Size i) const {
        Size k = 0;
        for (Index b = 0; b < graph().num_branches(); ++b) {
            if (graph().branch(b).kind != topology::BranchKind::Switch) continue;
            if (k++ == i) return std::string(graph().branch_name(b));
        }
        return {};
    }

    void check_voltage_(const solver::SimulationResult& res) {
        const auto v = solver::find_implausible_voltage(
            graph(), pool(), res,
            opts_.voltage_sanity_factor.value_or(Defaults::voltage_sanity_factor));
        if (v.node >= 0) {
            diags_.push_back({Severity::Warning, "implausible_voltage",
                              solver::describe(graph(), pool(), v),
                              std::string(graph().node(v.node).name),
                              std::nullopt});
        }
    }

    ir::Circuit circuit_;
    RunOptions opts_;
    std::optional<builder::CircuitBuilder> builder_;
    RunPlan plan_;
    pwl::NonlinearRefreshFn refresh_;
    std::vector<Diagnostic> diags_;
    solver::TrBdf2Stats stats_;
    solver::SolverSnapshot snapshot_;
    Real t_cursor_ = Real{0};
    std::optional<Vector> x0_;
    solver::SwitchScheduleFn switch_fn_;
    std::function<Real(Real)> next_edge_;
    solver::BExtraFn b_extra_;
    std::function<Real(Real)> next_jump_;
    solver::StepObserverFn observer_;
    solver::ShouldContinueFn cancel_;
    std::vector<Real> tick_periods_;
    std::vector<TickFn> ticks_;
};

}  // namespace pulsim::sim
