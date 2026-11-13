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

/// Throw `UnsupportedError` for the first branch engine `e` cannot run.
inline void require_supported(const topology::Graph& graph,
                              const pwl::DevicePool& pool, pwl::Engine e) {
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
