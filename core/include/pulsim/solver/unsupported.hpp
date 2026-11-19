#pragma once

// The typed refusal: an engine (or analysis) that cannot simulate a
// device kind, or cannot honour a requested feature, throws
// `UnsupportedError` instead of running a different circuit. It derives
// from std::invalid_argument, so every existing `catch (invalid_argument)`
// and Python `except ValueError` keeps catching it; the bindings surface
// it as `pulsim.UnsupportedError` with the four fields as attributes.

#include "pulsim/pwl/device_pool.hpp"
#include "pulsim/sim/error.hpp"
#include "pulsim/topology/graph.hpp"

#include <format>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>

namespace pulsim {

class UnsupportedError : public std::invalid_argument, public PulsimError {
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
            if (e == pwl::Engine::Bdf1) {
                throw UnsupportedError(
                    "bdf1", "branch_form", std::string(graph.branch_name(b)),
                    std::string(row.name),
                    std::format(
                        "engine 'bdf1' cannot simulate '{}', a {} in branch "
                        "form: bdf1 is retired (no event iteration, so the "
                        "device's state would stay frozen at what switch_fn "
                        "returns). Use engine='pwl' (the fixed-step "
                        "trapezoidal engine) or engine='trbdf2'.",
                        graph.branch_name(b), row.name));
            }
            throw UnsupportedError(
                std::string(pwl::engine_name(e)), "branch_form",
                std::string(graph.branch_name(b)), std::string(row.name),
                std::format(
                    "engine '{}' cannot simulate '{}', a {} in branch form "
                    "(add_ideal_switch / add_diode(form='branch')): it would "
                    "run a different circuit. The branch form runs on "
                    "engine='pwl' and engine='trbdf2'; or use the "
                    "conductance form (add_switch / add_diode with g_on, "
                    "g_off).",
                    pwl::engine_name(e), graph.branch_name(b), row.name));
        }
    }
}

/// add-in-kernel-event-losses: the event-loss hook is in engine='pwl' and
/// 'trbdf2' (solver/loss_tracker.hpp); every other engine refuses a circuit
/// with a loss port rather than run with the junction temperature held
/// constant. One line at each engine entry.
inline void refuse_device_losses(const pwl::DevicePool& pool,
                                 std::string_view engine) {
    if (pool.device_losses().empty()) return;
    const auto& s = pool.device_losses().front();
    throw UnsupportedError(
        std::string(engine), "device_losses", s.device, "",
        std::format(
            "engine '{}' cannot honour the loss port of '{}' "
            "(set_device_losses): the commutation energy and the junction "
            "temperature are read inside the time loops of engine='pwl', "
            "'trbdf2' and the event-driven run of 'dsed' only. Use one of "
            "them, or post-process with "
            "device_loss_summary.", engine, s.device));
}

/// add-pwl-igbt-and-thyristor: the first gated device (IGBT with reverse
/// blocking, GTO, thyristor, TRIAC) engine `e` does not run refuses as
/// `UnsupportedError(engine, 'gated_diode', device, 'Diode')`.
inline void require_gated_supported(const topology::Graph& graph,
                                    const pwl::DevicePool& pool,
                                    pwl::Engine e) {
    if (pool.num_gated_diodes() == 0) return;
    for (Index b = 0; b < graph.num_branches(); ++b) {
        if (!pool.is_registered(b)
            || pool.kind_of(b) != pwl::DevicePool::StoredKind::Diode) continue;
        const auto gate = pool.diode_params(b).gate;
        for (const auto& row : pwl::kGatedCaps) {
            if (row.gate != gate || row.supports(e)) continue;
            throw UnsupportedError(
                std::string(pwl::engine_name(e)), "gated_diode",
                std::string(graph.branch_name(b)), "Diode",
                std::format(
                    "engine '{}' cannot simulate '{}', a gated PWL device "
                    "({}: IGBT with reverse blocking, GTO, thyristor or "
                    "TRIAC): it would run a different circuit. Use "
                    "engine='pwl' or engine='trbdf2'.",
                    pwl::engine_name(e), graph.branch_name(b), row.name));
        }
    }
}

/// Throw `UnsupportedError` for the first branch engine `e` cannot run.
/// `loss_port`: the caller runs the exact engine's loss tracking (the dsed
/// event-driven run, python/dsed_bindings.cpp); every other dsed caller
/// (period maps, steady state, design batches) still refuses the port.
inline void require_supported(const topology::Graph& graph,
                              const pwl::DevicePool& pool, pwl::Engine e,
                              bool loss_port = false) {
    require_branch_form_supported(graph, pool, e);
    if (e != pwl::Engine::FixedTrap && e != pwl::Engine::TrBdf2
        && !(loss_port && e == pwl::Engine::Dsed))
        refuse_device_losses(pool, pwl::engine_name(e));
    require_gated_supported(graph, pool, e);
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
