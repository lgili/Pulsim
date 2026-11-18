#pragma once

// refactor-unified-transient-stepper, task 5.1 (design D6): the step-size
// controllers' state as explicit objects.
//
// `LteController` is the variable-step engine's step memory, moved out of
// `run_transient_trbdf2` as data: the proposal `h`, the post-gate-edge
// memory per edge parity, the post-diode-event SDIRK restart memory per
// (diode, direction), and the two factor laws. The driver still owns the
// decisions (when to reject, what to remember); the object is what a
// snapshot must carry for a variable-step resume to continue with the same
// next step (`save` / `load`, `ControllerState`).
//
// `FixedClock` is the fixed engine's grid: t = t0 + k·dt from an integer
// counter, never accumulated.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "pulsim/numeric/types.hpp"

namespace pulsim::step {

/// A controller's state, flat and serialisable (the snapshot's controller
/// block). `kind` says which controller wrote it; `v` is that
/// controller's layout (see its `save`). Empty = no controller state:
/// the resumed controller starts from its initial step.
struct ControllerState {
    enum class Kind : std::uint8_t { None = 0, Fixed = 1, Lte = 2 };
    Kind kind = Kind::None;
    std::vector<Real> v;
    [[nodiscard]] bool empty() const noexcept { return kind == Kind::None; }
};

/// The fixed engine's grid point k: t0 + k·dt (one expression, as the
/// loop always wrote it, so the rounding of every grid time is unchanged).
struct FixedClock {
    Real t0 = Real{0};
    Real dt = Real{0};
    [[nodiscard]] Real at(Size k) const noexcept {
        return t0 + static_cast<Real>(k) * dt;
    }
};

/// The TR-BDF2 engine's step memory (moved from trbdf2_transient.hpp).
struct LteController {
    /// The next step's proposal (before event clamping).
    Real h = Real{0};
    /// Post-gate-edge restart memory, one slot per edge parity
    /// (0 = none yet), and the slot the current post-edge step fills.
    Real h_mem_gate[2] = {Real{0}, Real{0}};
    int pending_gate_slot = -1;
    /// SDIRK restart size per (diode, direction): slot 2·d + is_on;
    /// `diode_slot` is the slot the pending restart step fills.
    std::vector<Real> h_mem_diode;
    Size diode_slot = Size(-1);

    /// Factor on a rejected step: elementary controller, exponent
    /// −1/3 (TR-BDF2, order 2) or −1/2 (the embedded SDIRK(2,2) estimate).
    [[nodiscard]] static Real reject_factor(Real err, bool sdirk) noexcept {
        return std::clamp(
            Real{0.9} * std::pow(err, sdirk ? Real{-0.5}
                                            : Real{-1.0 / 3.0}),
            Real{0.1}, Real{0.9});
    }
    /// Factor on an accepted step (err floored at 1e-4: growth ≤ 5×).
    [[nodiscard]] static Real accept_factor(Real err, bool sdirk) noexcept {
        const Real e_used = std::max(err, Real{1e-4});
        return std::clamp(
            Real{0.9}
                * std::pow(e_used, sdirk ? Real{-0.5}
                                         : Real{-1.0 / 3.0}),
            Real{0.2}, Real{5.0});
    }

    /// Layout: [h, h_mem_gate[0], h_mem_gate[1], pending_gate_slot,
    /// diode_slot (−1 = none), h_mem_diode...].
    [[nodiscard]] ControllerState save() const {
        ControllerState s;
        s.kind = ControllerState::Kind::Lte;
        s.v.reserve(5 + h_mem_diode.size());
        s.v.push_back(h);
        s.v.push_back(h_mem_gate[0]);
        s.v.push_back(h_mem_gate[1]);
        s.v.push_back(static_cast<Real>(pending_gate_slot));
        s.v.push_back(diode_slot == Size(-1) ? Real{-1}
                                             : static_cast<Real>(diode_slot));
        s.v.insert(s.v.end(), h_mem_diode.begin(), h_mem_diode.end());
        return s;
    }
    /// Restores what `save` wrote; refuses another controller's state.
    void load(const ControllerState& s) {
        if (s.kind != ControllerState::Kind::Lte || s.v.size() < 5) {
            throw std::invalid_argument(
                "LteController::load: not an LTE controller state (kind "
                + std::to_string(static_cast<int>(s.kind)) + ", "
                + std::to_string(s.v.size()) + " values)");
        }
        h = s.v[0];
        h_mem_gate[0] = s.v[1];
        h_mem_gate[1] = s.v[2];
        pending_gate_slot = static_cast<int>(s.v[3]);
        diode_slot = s.v[4] < Real{0} ? Size(-1) : static_cast<Size>(s.v[4]);
        h_mem_diode.assign(s.v.begin() + 5, s.v.end());
    }
};

}  // namespace pulsim::step
