#pragma once

// The typed refusal: an engine (or analysis) that cannot simulate a
// device kind, or cannot honour a requested feature, throws
// `UnsupportedError` instead of running a different circuit. It derives
// from std::invalid_argument, so every existing `catch (invalid_argument)`
// and Python `except ValueError` keeps catching it; the bindings surface
// it as `pulsim.UnsupportedError` with the four fields as attributes.

#include "pulsim/pwl/device_pool.hpp"
#include "pulsim/topology/graph.hpp"

#include <format>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>

namespace pulsim {

class UnsupportedError : public std::invalid_argument {
public:
    UnsupportedError(std::string engine, std::string feature,
                     std::string device, std::string kind,
                     const std::string& message)
        : std::invalid_argument(message),
          engine_{std::move(engine)}, feature_{std::move(feature)},
          device_{std::move(device)}, kind_{std::move(kind)} {}

    [[nodiscard]] const std::string& engine() const noexcept { return engine_; }
    [[nodiscard]] const std::string& feature() const noexcept { return feature_; }
    [[nodiscard]] const std::string& device() const noexcept { return device_; }
    [[nodiscard]] const std::string& kind() const noexcept { return kind_; }

private:
    std::string engine_, feature_, device_, kind_;
};

/// First branch whose device kind the table marks unsupported on `e`.
[[nodiscard]] inline std::optional<Index>
find_unsupported(const topology::Graph& graph, const pwl::DevicePool& pool,
                 pwl::Engine e) {
    for (Index b = 0; b < graph.num_branches(); ++b) {
        if (pool.is_registered(b)
            && !pwl::kind_caps(pool.kind_of(b)).supports(e)) {
            return b;
        }
    }
    return std::nullopt;
}

/// Refusal for branch `b` on engine `e`, with the caller's message.
[[nodiscard]] inline UnsupportedError
unsupported_device(const topology::Graph& graph, const pwl::DevicePool& pool,
                   pwl::Engine e, Index b, const std::string& message) {
    return UnsupportedError(std::string(pwl::engine_name(e)), "",
                            std::string(graph.branch_name(b)),
                            std::string(pwl::stored_kind_name(pool.kind_of(b))),
                            message);
}

/// add-branch-form-ideal-switch task 4.3: the first branch-form switch /
/// diode engine `e` does not run refuses as
/// `UnsupportedError(engine, 'branch_form', device, kind)`.
inline void require_branch_form_supported(const topology::Graph& graph,
                                          const pwl::DevicePool& pool,
                                          pwl::Engine e) {
    if (pool.num_branch_switches() == 0) return;
    for (Index b = 0; b < graph.num_branches(); ++b) {
        if (!pool.is_registered(b) || !pool.is_branch_form(b)) continue;
        const auto k = pool.kind_of(b);
        for (const auto& row : pwl::kBranchFormCaps) {
            if (row.kind != k || row.supports(e)) continue;
            throw UnsupportedError(
                std::string(pwl::engine_name(e)), "branch_form",
                std::string(graph.branch_name(b)), std::string(row.name),
                std::format(
                    "engine '{}' cannot simulate '{}', a {} in branch form "
                    "(add_ideal_switch / add_diode(form='branch')): it would "
                    "run a different circuit. The branch form runs on "
                    "engine='pwl'{}; or use the conductance form "
                    "(add_switch / add_diode with g_on, g_off).",
                    pwl::engine_name(e), graph.branch_name(b), row.name,
                    k == pwl::DevicePool::StoredKind::Switch
                        ? " and engine='trbdf2'" : ""));
        }
    }
    if (e != pwl::Engine::TrBdf2) return;
    // Task 5.5 (refused on TR-BDF2 in this slice): its event restart is
    // not the SDIRK x3 re-initialisation, so a mode with dependent states
    // rings (7.0e10 V measured). Static over-approximation: g_off = 0
    // anywhere, or an R_on = 0 element that closes a loop with voltage
    // sources, capacitors or other R_on = 0 elements (all switches closed).
    const Index n = graph.num_nodes() + 1;
    std::vector<Index> par(static_cast<std::size_t>(n));
    for (Index i = 0; i < n; ++i) par[i] = i;
    auto find = [&](Index a) {
        while (par[a] != a) { par[a] = par[par[a]]; a = par[a]; }
        return a;
    };
    auto nd = [&](Index v) { return v < 0 ? n - 1 : v; };
    using K = pwl::DevicePool::StoredKind;
    for (Index b = 0; b < graph.num_branches(); ++b) {
        if (!pool.is_registered(b) || pool.is_branch_form(b)) continue;
        const auto k = pool.kind_of(b);
        if (k == K::Capacitor || k == K::VoltageSource ||
            k == K::PWMVoltageSource || k == K::SineVoltageSource ||
            k == K::PulseVoltageSource || k == K::VCVS || k == K::CCVS ||
            k == K::IdealTransformer) {
            par[find(nd(graph.branch(b).from))] = find(nd(graph.branch(b).to));
        }
    }
    for (Index b = 0; b < graph.num_branches(); ++b) {
        if (!pool.is_registered(b) || !pool.is_branch_form(b)) continue;
        const Real g_off = pool.kind_of(b) == K::Diode
            ? pool.diode_params(b).g_off : pool.switch_g_off(b);
        bool dep = g_off == Real{0};
        if (!dep && pool.branch_R_on(b) == Real{0}) {
            const Index a = find(nd(graph.branch(b).from));
            const Index c = find(nd(graph.branch(b).to));
            dep = a == c;
            par[a] = c;
        }
        if (!dep) continue;
        throw UnsupportedError(
            "trbdf2", "dependent_states", std::string(graph.branch_name(b)),
            std::string(pwl::stored_kind_name(pool.kind_of(b))),
            std::format(
                "engine 'trbdf2' cannot simulate '{}' (branch form, {}): "
                "with it a mode can make capacitor voltages or inductor "
                "currents dependent (an impulse), whose re-initialisation "
                "(three SDIRK steps) is implemented on engine='pwl' only. "
                "Use engine='pwl', or g_off > 0 and R_on > 0.",
                graph.branch_name(b),
                g_off == Real{0} ? "g_off = 0" : "R_on = 0 in a loop of "
                "capacitors / voltage sources"));
    }
}

/// Throw `UnsupportedError` for the first branch engine `e` cannot run.
inline void require_supported(const topology::Graph& graph,
                              const pwl::DevicePool& pool, pwl::Engine e) {
    require_branch_form_supported(graph, pool, e);
    const auto b = find_unsupported(graph, pool, e);
    if (!b) return;
    const auto kind = pwl::stored_kind_name(pool.kind_of(*b));
    throw unsupported_device(graph, pool, e, *b, std::format(
        "engine '{}' cannot simulate branch '{}', a {} ({}): it would run "
        "a different circuit. Engines that simulate it: see "
        "pulsim.engine_capabilities(); engine='pwl' runs every kind.",
        pwl::engine_name(e), graph.branch_name(*b), kind,
        pwl::kSolveClassNames[static_cast<std::size_t>(
            pwl::kind_caps(pool.kind_of(*b)).cls)]));
}

}  // namespace pulsim
