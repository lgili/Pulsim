#pragma once

// add-kernel-simulation-object (9.2): the mix-in base of every pulsim
// error. Each error keeps its std base (std::invalid_argument,
// std::runtime_error), so existing catches still work; `catch (const
// pulsim::PulsimError&)` catches them all. The Python translators map the
// classes onto `pulsim.PulsimError` subclasses.

#include <exception>

namespace pulsim {

class PulsimError {
public:
    virtual ~PulsimError() = default;
    /// The std::exception message of the concrete error.
    [[nodiscard]] const char* message() const noexcept {
        const auto* e = dynamic_cast<const std::exception*>(this);
        return e != nullptr ? e->what() : "";
    }

protected:
    PulsimError() = default;
    PulsimError(const PulsimError&) = default;
    PulsimError& operator=(const PulsimError&) = default;
};

}  // namespace pulsim
