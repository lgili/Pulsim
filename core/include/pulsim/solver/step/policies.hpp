#pragma once

// refactor-unified-transient-stepper, task group 5 (design D2): the
// fixed engine's step policies as types. Each is the decision the driver
// made, with the expression it made it with (so the same bits); the loop
// that applies it (rollback, sub-steps, records) stays in the driver.
// The LTE controller's shrink-and-repeat law is `LteController::
// reject_factor` (controllers.hpp); IterateAtStepEnd and Localize are not
// here yet (DECISIONS 25).

#include <algorithm>

#include "pulsim/numeric/types.hpp"

namespace pulsim::step {

/// `max_dt_halvings`: a failed fixed step is retried as 2^h sub-steps of
/// dt / 2^h, h = 1 … cap. The cap is clamped at the point of use: the
/// option is unchecked by SimulationOptions::valid(), `Size{1} << h` is
/// undefined for h >= 64, and 2^20 sub-steps is already a hang, not a
/// recovery.
struct HalvingLadder {
    static constexpr Size kMaxHalvings = Size{20};
    [[nodiscard]] static constexpr Size cap(Size requested) noexcept {
        return std::min(requested, kMaxHalvings);
    }
    [[nodiscard]] static constexpr Size n_sub(Size h) noexcept { return Size{1} << h; }
    [[nodiscard]] static constexpr Real sub_dt(Real dt, Size h) noexcept {
        return dt / static_cast<Real>(n_sub(h));
    }
};

/// The inductor guards (`inductor_freeze_di_max`, `inductor_abs_clamp`;
/// 0 = off): the freeze snaps a step-to-step jump back to the pre-step
/// current, then the clamp bounds |i|. Returns the current to keep.
struct GuardPolicy {
    struct Result {
        Real i;
        bool froze;
        bool clamped;
    };
    [[nodiscard]] static constexpr Result apply(Real i_prev_step, Real i_new,
                                                Real freeze_di_max,
                                                Real abs_clamp) noexcept {
        Result r{i_new, false, false};
        if (freeze_di_max > Real{0}) {
            const Real di = r.i - i_prev_step;
            const Real adi = di < Real{0} ? -di : di;
            if (adi > freeze_di_max) {
                r.i = i_prev_step;
                r.froze = true;
            }
        }
        if (abs_clamp > Real{0}) {
            const Real lim = abs_clamp;
            if (r.i > lim) { r.i = lim; r.clamped = true; }
            if (r.i < -lim) { r.i = -lim; r.clamped = true; }
        }
        return r;
    }
};

/// The fixed engine's restart rule after a discontinuity: that step and
/// the next two are Alexander SDIRK(2,2) steps (γ written as the engine
/// always wrote it; DECISIONS 4). `None` is the run-time option off
/// (`restart_rule` false: no step owed).
struct RestartPolicy {
    static constexpr Size kSteps = Size{3};
    static constexpr Real kGamma = Real{1} - Real{0.70710678118654752440};
};

}  // namespace pulsim::step
