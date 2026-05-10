#pragma once

#include "pulsim/v1/runtime_circuit.hpp"

#include <functional>
#include <map>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace pulsim::v1::templates {

// =============================================================================
// add-converter-templates — template registry (Phase 1)
// =============================================================================
//
// A converter template is a parametric Circuit fragment: the user
// supplies high-level design intent (Vin, Vout, fsw, load) and the
// template expands into a fully-wired Circuit with appropriately-sized
// passives, switching device, control source, and load.
//
// Templates are registered globally by name ("buck", "boost", etc.).
// The expansion function takes a `ConverterParameters` map (string →
// double, accepting the design-intent fields) and returns a `Circuit`
// plus a record of which auto-design defaults the expansion ended up
// applying — useful for reproducibility and post-hoc inspection.
//
// Header-only design: each template ships its own
// `<topology>_template.hpp` that calls `register_converter_template`
// at static-initialization time via a `static const` registrar.

using ConverterParameters = std::unordered_map<std::string, Real>;

/// Result of expanding a template — the assembled Circuit plus the
/// auto-designed parameters that were filled in (useful for telemetry /
/// reproducibility / docs).
struct ConverterExpansion {
    Circuit circuit;
    /// Effective parameters after auto-design — includes user inputs
    /// plus any defaults the template inferred (e.g. an inductor value
    /// chosen to bound the current ripple at 30 %).
    std::unordered_map<std::string, Real> resolved_parameters;
    /// Human-readable notes on each auto-design decision, keyed by the
    /// parameter name. Empty for parameters the user supplied.
    std::unordered_map<std::string, std::string> design_notes;
    /// Topology label ("buck", "boost", ...). Echoed back for
    /// downstream telemetry / logging.
    std::string topology;
};

using ConverterExpander =
    std::function<ConverterExpansion(const ConverterParameters&)>;

/// Singleton registry. Header-only so each template's static
/// registrar reaches it at translation-unit-load time.
class ConverterRegistry {
public:
    static ConverterRegistry& instance() {
        static ConverterRegistry r;
        return r;
    }

    /// Register a named template expander. Calling twice with the same
    /// name overwrites — useful when a downstream library wants to
    /// override the default expansion.
    void register_template(std::string name, ConverterExpander expander) {
        expanders_[std::move(name)] = std::move(expander);
    }

    /// Expand the named template with the given parameters.
    /// Throws std::invalid_argument if the topology isn't registered
    /// (with a "did you mean" suggestion based on the closest registered
    /// name when one is within edit distance ≤ 2).
    [[nodiscard]] ConverterExpansion expand(
        const std::string& topology,
        const ConverterParameters& params) const {
        const auto it = expanders_.find(topology);
        if (it == expanders_.end()) {
            const auto suggestion = closest_match_(topology);
            std::string msg =
                "ConverterRegistry: unknown topology '" + topology + "'";
            if (!suggestion.empty()) {
                msg += " (did you mean '" + suggestion + "'?)";
            }
            msg += ". Available topologies: ";
            bool first = true;
            for (const auto& [name, _] : expanders_) {
                if (!first) msg += ", ";
                msg += name;
                first = false;
            }
            throw std::invalid_argument(msg);
        }
        return it->second(params);
    }

    [[nodiscard]] std::vector<std::string> registered_topologies() const {
        std::vector<std::string> names;
        names.reserve(expanders_.size());
        for (const auto& [name, _] : expanders_) {
            names.push_back(name);
        }
        return names;
    }

    [[nodiscard]] bool has_template(const std::string& name) const {
        return expanders_.count(name) > 0;
    }

private:
    std::map<std::string, ConverterExpander> expanders_;

    [[nodiscard]] std::string closest_match_(const std::string& query) const {
        // Tiny Levenshtein for "did you mean" suggestions. We're matching
        // ≤ 30 templates; O(N · |query|²) is cheap.
        std::string best;
        std::size_t best_dist = query.size() + 1;
        for (const auto& [name, _] : expanders_) {
            const std::size_t d = edit_distance_(query, name);
            if (d < best_dist) {
                best_dist = d;
                best = name;
            }
        }
        // Only suggest when within edit distance 2 (typical typo).
        return (best_dist <= 2) ? best : std::string{};
    }

    [[nodiscard]] static std::size_t edit_distance_(const std::string& a,
                                                     const std::string& b) {
        const std::size_t m = a.size();
        const std::size_t n = b.size();
        std::vector<std::vector<std::size_t>> dp(
            m + 1, std::vector<std::size_t>(n + 1));
        for (std::size_t i = 0; i <= m; ++i) dp[i][0] = i;
        for (std::size_t j = 0; j <= n; ++j) dp[0][j] = j;
        for (std::size_t i = 1; i <= m; ++i) {
            for (std::size_t j = 1; j <= n; ++j) {
                const std::size_t cost = (a[i - 1] == b[j - 1]) ? 0u : 1u;
                dp[i][j] = std::min(
                    {dp[i - 1][j] + 1,
                     dp[i][j - 1] + 1,
                     dp[i - 1][j - 1] + cost});
            }
        }
        return dp[m][n];
    }
};

/// Convenience: look up an optional parameter with a default fallback.
[[nodiscard]] inline Real param_or(const ConverterParameters& params,
                                    const std::string& name,
                                    Real fallback) {
    const auto it = params.find(name);
    return (it == params.end()) ? fallback : it->second;
}

/// Convenience: require a parameter — throws with a helpful message if
/// the user didn't supply one.
[[nodiscard]] inline Real require_param(const ConverterParameters& params,
                                          const std::string& name,
                                          const std::string& topology) {
    const auto it = params.find(name);
    if (it == params.end()) {
        throw std::invalid_argument(
            "ConverterTemplate '" + topology + "': missing required parameter '"
            + name + "'");
    }
    return it->second;
}

}  // namespace pulsim::v1::templates
