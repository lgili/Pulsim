#pragma once

// A simulation engine gave up on its own account: the diode settle did
// not converge, a diode chattered, the integrator stopped making
// progress, the step cap was reached, or a step went non-finite. Not a
// refusal (UnsupportedError names a feature the engine declines); this
// is a run the engine accepted and could not finish.
//
// The base stays std::runtime_error, so every existing catch and every
// Python `except RuntimeError` keeps working; the Python translator to
// pulsim.ConvergenceError is registered where the bindings define it.

#include <stdexcept>
#include <string>
#include <utility>

#include "pulsim/numeric/types.hpp"

namespace pulsim {

enum class ConvergenceKind { Settle, Chatter, NoProgress, MaxSteps, NaN };

[[nodiscard]] inline const char* to_string(ConvergenceKind k) noexcept {
    switch (k) {
        case ConvergenceKind::Settle: return "settle";
        case ConvergenceKind::Chatter: return "chatter";
        case ConvergenceKind::NoProgress: return "no_progress";
        case ConvergenceKind::MaxSteps: return "max_steps";
        case ConvergenceKind::NaN: return "nan";
    }
    return "unknown";
}

class ConvergenceFailure : public std::runtime_error {
public:
    ConvergenceFailure(ConvergenceKind kind, Real time, std::string device,
                       const std::string& message)
        : std::runtime_error(message), kind_{kind}, time_{time},
          device_{std::move(device)} {}

    [[nodiscard]] ConvergenceKind kind() const noexcept { return kind_; }
    [[nodiscard]] const char* kind_name() const noexcept { return to_string(kind_); }
    /// Simulated time of the failure (s).
    [[nodiscard]] Real time() const noexcept { return time_; }
    /// The device that failed, when one is known ("" otherwise).
    [[nodiscard]] const std::string& device() const noexcept { return device_; }

private:
    ConvergenceKind kind_;
    Real time_;
    std::string device_;
};

}  // namespace pulsim
