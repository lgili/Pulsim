#pragma once

// add-kernel-simulation-object (9.2): the mix-in base of every pulsim
// error. Each error keeps its std base (std::invalid_argument,
// std::runtime_error), so existing catches still work; `catch (const
// pulsim::PulsimError&)` catches them all. The Python translators map the
// classes onto `pulsim.PulsimError` subclasses.

#include <exception>
#include <stdexcept>
#include <string>
#include <utility>

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

/// An engine broke one of its own invariants (e.g. a sample time below the
/// previous one): a bug in pulsim, raised instead of returning a corrupt
/// result. Python: `pulsim.InternalError` (a RuntimeError).
class InternalError : public std::logic_error, public PulsimError {
public:
    using std::logic_error::logic_error;
};

/// The circuit itself is wrong (a pin on the wrong domain, a magnetic
/// network with no reference, a tie on a node the circuit lacks). Names the
/// device and / or node it is about (empty when not one). Python:
/// `pulsim.NetlistError` (a ValueError).
class NetlistError : public std::invalid_argument, public PulsimError {
public:
    NetlistError(std::string device, std::string node, const std::string& what)
        : std::invalid_argument(what), device_(std::move(device)),
          node_(std::move(node)) {}
    [[nodiscard]] const std::string& device() const noexcept { return device_; }
    [[nodiscard]] const std::string& node() const noexcept { return node_; }

private:
    std::string device_;
    std::string node_;
};

/// A system matrix is structurally or numerically singular; the message
/// carries the row names `explain_singular` localised. A runtime_error, as
/// those sites threw before. A transient wraps it in SimulationAborted,
/// which then reports `singular()`. Python: `pulsim.SingularError` (a
/// SimulationAborted).
class SingularError : public std::runtime_error, public PulsimError {
public:
    using std::runtime_error::runtime_error;
};

/// True while the exception being handled is a SingularError (SimulationAborted
/// reads it in the engines' catch blocks, so those blocks need no edit).
[[nodiscard]] inline bool singular_in_flight() noexcept {
    const auto p = std::current_exception();
    if (!p) return false;
    try {
        std::rethrow_exception(p);
    } catch (const SingularError&) {
        return true;
    } catch (...) {
        return false;
    }
}

}  // namespace pulsim
