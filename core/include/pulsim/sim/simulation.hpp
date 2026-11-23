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
#include "pulsim/sim/diagnostic.hpp"
#include "pulsim/sim/options.hpp"
#include "pulsim/solver/run_transient.hpp"
#include "pulsim/solver/trbdf2_transient.hpp"
#include "pulsim/solver/unsupported.hpp"
#include "pulsim/solver/voltage_sanity.hpp"
#include "pulsim/streaming/live_ring.hpp"

#include <format>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace pulsim::sim {

/// What the constructor decided.
struct RunPlan {
    Engine engine = Engine::FixedTrap;
    EngineChoice asked = EngineChoice::Auto;
    std::vector<std::string> route_reasons;
    std::vector<ir::Tie> ties;          // inserted by the preflight
    std::vector<std::string> undriven;  // controlled switches nothing drives
};

/// The static Newton chain of a transient driver. `nullopt` / `true`: the
/// driver stamps the Newton devices itself (pwl::static_device_refresh);
/// `false` on a circuit with a static Newton device is refused (it would
/// simulate it as an open circuit). The bindings call this too.
[[nodiscard]] inline pwl::NonlinearRefreshFn transient_refresh(
    const topology::Graph& graph, const pwl::DevicePool& pool,
    std::optional<bool> enable, std::string_view engine) {
    using pwl::SolveClass;
    if (!enable.has_value() || *enable) return {};
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
    /// `controls_composed`: the caller drives the circuit's C blocks /
    /// graph-driven sources through the attachments (pulsim.simulate()
    /// composes them in Python); otherwise they are refused.
    Simulation(ir::Circuit circuit, RunOptions options,
               bool controls_composed = false)
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
        if (!controls_composed && !circuit_.controls().empty()) {
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
        // the exact engine's sensitivities are simulate()'s post-pass, not
        // this kernel run's: refused here rather than silently dropped
        if (!opts_.sensitivities.empty() && e == Engine::Dsed) {
            throw UnsupportedError("dsed", "sensitivities", "", "",
                "Simulation(engine='dsed'): sensitivities on the exact engine "
                "are computed by pulsim.simulate(); this run computes them on "
                "'pwl' and 'trbdf2'.");
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
        // Value-dependent refusals (store_every = 0, max_event_iterations
        // = 0 on trbdf2, ...) surface here, before anything runs.
        if (e == Engine::FixedTrap) {
            (void)to_simulation_options(opts_);
        } else {
            (void)to_trbdf2_options(opts_);
        }
        t_cursor_ = opts_.t_start;
    }

    // The cache refers to the builder's graph and pool: not copyable or
    // movable.
    Simulation(const Simulation&) = delete;
    Simulation& operator=(const Simulation&) = delete;
    Simulation(Simulation&&) = delete;
    Simulation& operator=(Simulation&&) = delete;

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
    /// `offset` shifts the grid: ticks at t_start + offset + k*period.
    /// `monitor`: a `Kind::Monitor` (trbdf2 neither lands on it nor
    /// re-samples the gate after it; pwl ticks it on its grid as any tick).
    void add_tick(Real period, TickFn fn, Real offset = Real{0},
                  bool monitor = false) {
        if (!(period > Real{0})) {
            throw OptionsError("tick", std::format(
                "Simulation.add_tick: period must be positive, got {}", period));
        }
        tick_periods_.push_back(period);
        tick_offsets_.push_back(offset);
        tick_monitors_.push_back(monitor ? 1 : 0);
        ticks_.push_back(std::move(fn));
    }
    /// pwl: runs on every tick instant before the ticks (a composed control
    /// graph's state update).
    void set_tick_before(solver::StepObserverFn fn) { tick_before_ = std::move(fn); }
    /// pwl: push each step's (t, x) into a live ring; its stop flag cancels.
    void set_live_ring(std::shared_ptr<streaming::LiveRing> ring) {
        live_ring_ = std::move(ring);
    }
    void set_cancel(solver::ShouldContinueFn fn) { cancel_ = std::move(fn); }
    /// Accepted-step / event / device-state listeners for both engines (the
    /// recorders' in-loop feed). Not owned: it must outlive every run.
    void set_hooks(events::StepHooks* hooks) noexcept { hooks_ = hooks; }
    void set_initial_state(Vector x) { x0_ = std::move(x); }

    // ---- Queries -----------------------------------------------------------
    [[nodiscard]] const RunPlan& plan() const noexcept { return plan_; }
    [[nodiscard]] const RunOptions& options() const noexcept { return opts_; }
    /// The circuit the run solves (the caller's, plus any preflight ties).
    [[nodiscard]] const ir::Circuit& circuit() const noexcept { return circuit_; }
    [[nodiscard]] const ::pulsim::builder::CircuitBuilder& builder() const noexcept {
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
    /// Continue from `s`: the next run call starts at `s.t` with its state
    /// (`at_snapshot_time = false`: at the options' t_start, as
    /// pulsim.simulate(resume_from=, t_start=) does).
    void restore(const solver::SolverSnapshot& s, bool at_snapshot_time = true) {
        if (!s.valid) {
            throw OptionsError("snapshot",
                "Simulation.restore: the snapshot is not valid (it came from a "
                "run that produced none)");
        }
        if (s.controller.kind == step::ControllerState::Kind::Exact) {
            // the exact engine's snapshot (dsed::ExactResumeState): its state has
            // no companion history for these engines to continue from
            throw UnsupportedError(engine_label(opts_.engine), "resume_from", "", "",
                "Simulation.restore: the snapshot was written by the exact engine, "
                "whose state has no companion history (i_C, v_L) to continue a "
                "trapezoidal step from; resume it with engine='dsed'");
        }
        snapshot_ = s;
        t_cursor_ = at_snapshot_time ? s.t : opts_.t_start;
        // simulate(resume_from=, start_from_dc_op=True) hands the driver
        // both, as it always has; a plain restore continues the snapshot.
        dc_with_resume_ = !at_snapshot_time;
    }

    /// The run's PWL cache, built on first use and kept across run calls
    /// (a hook may refactor parameters in it before the run).
    pwl::PwlStateSpaceCache& cache() {
        if (!cache_) {
            cache_.emplace(builder_->graph(), builder_->pool());
            cache_->share_symbolic(circuit_.symbolic_plan());
            // trbdf2 solves at its own step; the lazy build needs only a
            // positive dt to mark the cache dynamic (simulate()'s rule).
            const Real h_max = opts_.dt.value_or(Real{0});
            cache_->build_lazy(plan_.engine == Engine::FixedTrap ? *opts_.dt
                               : h_max > Real{0} ? h_max
                               : (opts_.t_end - opts_.t_start) / Real{1000});
        }
        return *cache_;
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
        // The initial state seeds the first run call (with a restored
        // snapshot too, as the drivers take both); continuations ignore it.
        const Vector* x0 = (!ran_ && x0_) ? &*x0_ : nullptr;
        const auto& g = graph();
        auto& p = builder_->pool();  // the cache stamps device state into it
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
        auto& cache = this->cache();
        // The first run call starts the grid's ticks at the run's t0 (NaN);
        // a continuation keeps the original run's grid.
        const Real origin = t_cursor_ == opts_.t_start
                ? std::numeric_limits<Real>::quiet_NaN() : opts_.t_start;
        // A snapshot with one tick grid per tick source (the run that wrote
        // it had the same ticks) keeps that grid and its next index:
        // simulate()'s _resume_tick_origins / _resume_tick_next_k rule.
        std::vector<Real> origins;
        std::vector<std::uint64_t> next_k;
        if (resume != nullptr && !ticks_.empty()
            && resume->tick_origin.size() == ticks_.size()) {
            origins = resume->tick_origin;
            if (resume->tick_next_k.size() == ticks_.size()) {
                next_k = resume->tick_next_k;
            }
        }
        const bool dc_op = opts_.start_from_dc_op && !ran_
                && (resume == nullptr || dc_with_resume_);
        solver::SimulationResult res;
        if (plan_.engine == Engine::FixedTrap) {
            auto so = to_simulation_options(opts_);
            so.t_start = t_cursor_;
            so.t_end = t;
            events::TickPlan tp;
            for (std::size_t j = 0; j < tick_periods_.size(); ++j) {
                events::Rate r{tick_periods_[j], tick_offsets_[j],
                               origins.empty() ? origin : origins[j]};
                if (!next_k.empty()) r.k0 = next_k[j];
                tp.rates.push_back(r);
            }
            tp.fn = [this](std::uint32_t j, const events::TickInfo&, Real tt,
                           const Vector& x) { ticks_[j](tt, x); };
            tp.before = tick_before_;
            auto observer = observer_;
            auto cancel = cancel_;
            if (live_ring_) {  // atomics only: no GIL in the ring's part
                observer = [ring = live_ring_, user = observer_](
                               Real tt, const Vector& x) {
                    if (user) user(tt, x);
                    if (ring->should_decimate()) {
                        ring->push(tt, x.data(), static_cast<std::size_t>(x.size()));
                    }
                };
                cancel = [ring = live_ring_, user = cancel_]() {
                    return !ring->stopped() && (!user || user());
                };
            }
            res = solver::run_transient(cache, g, p, so, switch_fn, b_extra_,
                    dc_op, refresh_, observer, x0, cancel, resume, next_jump_,
                    next_edge_, ticks_.empty() ? nullptr : &tp, hooks_);
        } else {
            auto to = to_trbdf2_options(opts_);
            to.t_start = t_cursor_;
            to.t_end = t;
            to.observer_periods = tick_periods_;
            to.observer_offsets = tick_offsets_;
            to.observer_origin = origin;
            to.observer_monitors = tick_monitors_;
            to.observer_origins = std::move(origins);
            to.observer_next_k = std::move(next_k);
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
                    resume, tick_fn, hooks_);
        }
        ran_ = true;  // only a completed call: a retry after a throw starts over
        snapshot_ = res.final_snapshot;
        t_cursor_ = t;
        check_voltage_(res);
        if (opts_.on_impulse == OnImpulse::Warn && !res.reinit_records.empty()) {
            const auto& r0 = res.reinit_records.front();
            diags_.push_back({Severity::Warning, "impulse", std::format(
                "{} impulsive re-initialisation(s); first at t = {} ({}, {} J "
                "removed). See result.reinit_records.",
                res.reinit_records.size(), r0.t, r0.kind, r0.dW),
                r0.devices.empty() ? std::string{} : r0.devices.front(),
                r0.t});
        }
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
    std::optional<::pulsim::builder::CircuitBuilder> builder_;
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
    std::vector<Real> tick_offsets_;
    std::vector<char> tick_monitors_;
    std::vector<TickFn> ticks_;
    solver::StepObserverFn tick_before_;
    std::shared_ptr<streaming::LiveRing> live_ring_;
    events::StepHooks* hooks_ = nullptr;
    std::optional<pwl::PwlStateSpaceCache> cache_;
    bool ran_ = false;
    bool dc_with_resume_ = false;
};

}  // namespace pulsim::sim
