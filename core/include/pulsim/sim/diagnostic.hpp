#pragma once

// add-kernel-simulation-object: a run's non-error messages (warnings the
// Python layer emits as `warnings.warn`, kept as data on the kernel side).

#include "pulsim/numeric/types.hpp"

#include <cstdint>
#include <optional>
#include <string>

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

}  // namespace pulsim::sim
