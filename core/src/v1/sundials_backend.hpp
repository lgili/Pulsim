#pragma once

#include "pulsim/v1/simulation.hpp"

namespace pulsim::v1 {

// Runtime SUNDIALS backend entrypoint used by Simulator transient flow.
[[nodiscard]] SimulationResult run_sundials_backend(
    Circuit& circuit,
    const SimulationOptions& options,
    const Vector& x0,
    SimulationCallback callback,
    EventCallback event_callback,
    SimulationControl* control,
    bool escalated_from_native);

}  // namespace pulsim::v1

