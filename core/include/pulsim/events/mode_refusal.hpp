#pragma once

// add-complementarity-mode-selection D3 / task 5: the named refusals of
// the mode selection, as `pulsim::UnsupportedError(engine, feature,
// device, kind)` (Python: `pulsim.UnsupportedError`, a ValueError).
//
//   feature 'consistent_mode'   — no conduction state of the diodes is
//                                 consistent at t (a secondary ray on a
//                                 passive circuit, or an enumeration
//                                 with no complementary mode); device:
//                                 the diodes, kind 'Diode'.
//   feature 'non_passive_ports' — non-passive circuit, Lemke ended at a
//                                 ray and more than 12 ports are free;
//                                 device / kind: the element that makes
//                                 the circuit non-passive.

#include <cstdio>
#include <string>
#include <vector>

#include "pulsim/events/mode_selector.hpp"
#include "pulsim/solver/unsupported.hpp"

namespace pulsim::events {

/// What makes a circuit non-passive at its diode ports (D3); empty
/// `device` for a passive one.
struct PassivityClass {
    std::string device;
    std::string kind;
    [[nodiscard]] bool passive() const noexcept { return device.empty(); }
};

/// Throw the named refusal for a failed selection; no-op on a solution.
inline void raise_if_refused(const ModeSelection& sel, const std::string& engine,
                             Real t, const std::vector<std::string>& diodes,
                             const PassivityClass& cls) {
    if (sel.status == LcpStatus::Solution || sel.status == LcpStatus::Trivial) return;
    char tb[32];
    std::snprintf(tb, sizeof tb, "%.9e", t);
    if (sel.status == LcpStatus::Unenumerable) {
        throw UnsupportedError(
            engine, "non_passive_ports", cls.device, cls.kind,
            "mode selection at t = " + std::string(tb) + " s: the circuit is not passive at "
            "its diode ports (" + cls.kind + " '" + cls.device + "'), Lemke's method ended "
            "at a secondary ray and " + std::to_string(diodes.size()) + " diodes exceed the "
            "enumeration limit of 12. Reduce the number of diodes the controlled element "
            "couples, or use a passive equivalent.");
    }
    std::string names;
    for (const auto& d : diodes) names += (names.empty() ? "" : ", ") + d;
    throw UnsupportedError(
        engine, "consistent_mode", names, "Diode",
        "mode selection at t = " + std::string(tb) + " s: no conduction state of the diodes ("
        + names + ") is consistent (" + to_string(sel.status) + "). The circuit forces "
        "a current no diode configuration can carry (a current source into blocking "
        "diodes, an inductor with no path): add the missing path.");
}

}  // namespace pulsim::events
