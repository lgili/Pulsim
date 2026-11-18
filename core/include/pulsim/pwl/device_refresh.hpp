#pragma once

// =============================================================================
// Pulsim — the kernel-built Newton device refresh
// =============================================================================
//
// add-unified-device-interface step 2. Both transient drivers used to build
// the Newton refresh as a chain of five `std::function` wrappers, written
// twice, each capturing the previous one and its own copy of the step size.
// Two defects came from that shape: a wrapper that captured the RAW base
// callback silently deleted every stamp before it (owner note
// nonlinear-refresh-wrapper-chain), and TR-BDF2 probe solves that did not set
// the captured step divided by zero (trbdf2-probe-solves-set-the-device-step).
//
// Here the refresh is ONE function: the static stamps (diode / MOSFET / IGBT,
// or the caller's deprecated override), then one pass per stateful history
// in the fixed `StatefulHistories` order, all reading ONE step context. A
// solve sets the context; no stamp captures a step of its own.

#include "pulsim/numeric/types.hpp"
#include "pulsim/pwl/device_pool.hpp"
#include "pulsim/pwl/nonlinear_refresh_igbt_tail.hpp"
#include "pulsim/pwl/nonlinear_refresh_lauritzen_diode.hpp"
#include "pulsim/pwl/nonlinear_refresh_mosfet_level1.hpp"
#include "pulsim/pwl/nonlinear_refresh_nonlinear_capacitor.hpp"
#include "pulsim/pwl/nonlinear_refresh_pmsm_mna.hpp"
#include "pulsim/pwl/nonlinear_refresh_saturable_inductor.hpp"
#include "pulsim/pwl/nonlinear_solve.hpp"
#include "pulsim/pwl/stateful_histories.hpp"
#include "pulsim/pwl/trbdf2_stage.hpp"

#include <algorithm>

namespace pulsim::pwl {

/// The step every device stamp of the next solve integrates over: the
/// interval `h` it advances and the rule (`stage`) of its time formula.
/// Owned by the driver, set before every solve, read by reference.
struct StepCtx {
    Real h = Real{0};
    TrBdf2Stage stage = TrBdf2Stage::Trapezoidal;
};

[[nodiscard]] inline CossStage coss_stage_of(TrBdf2Stage s) noexcept {
    switch (s) {
    case TrBdf2Stage::BackwardEuler: return CossStage::BackwardEuler;
    case TrBdf2Stage::Bdf2Stage2:    return CossStage::Bdf2Stage2;
    default:                         return CossStage::Trapezoidal;
    }
}

/// The static Newton stamps: the caller's override when one is given
/// (deprecated), else the kernel's standard chain when the pool has a
/// static Newton device, else none.
[[nodiscard]] inline NonlinearRefreshFn static_device_refresh(
    const DevicePool& pool, const NonlinearRefreshFn& override_fn) {
    if (override_fn || !pool.requires_static_refresh()) return override_fn;
    return make_combined_diode_mosfet_refresh();
}

/// The whole Newton refresh: `base` (from `static_device_refresh`), then
/// every stateful history. Empty when there is nothing to stamp, so a
/// linear circuit keeps its linear path. `hs` and `ctx` are captured by
/// reference and must outlive the returned function.
/// `sat_increment_form`: the saturable inductor's flux residual in the
/// closed-form increment form (TR-BDF2) or the direct form (fixed step).
[[nodiscard]] inline NonlinearRefreshFn build_device_refresh(
    NonlinearRefreshFn base, StatefulHistories& hs, const StepCtx& ctx,
    bool sat_increment_form) {
    if (!hs.any()) return base;
    return [base = std::move(base), &hs, &ctx, sat_increment_form](
               const Vector& x, sparse::Matrix& J_nl, Vector& f_nl,
               const topology::Graph& g, const DevicePool& pool) -> Real {
        Real m = Real{0};
        if (base) {
            m = base(x, J_nl, f_nl, g, pool);
        } else {
            sparse::zero_keep_pattern(J_nl);
            f_nl.setZero();
        }
        if (!hs.coss.empty()) {
            m = std::max(m, refresh_nonlinear_capacitors(
                                x, J_nl, f_nl, hs.coss, ctx.h,
                                coss_stage_of(ctx.stage),
                                &hs.coss.q_gamma()));
        }
        if (!hs.laur.empty()) {
            m = std::max(m, refresh_lauritzen_diodes(x, J_nl, f_nl, hs.laur,
                                                     ctx.h, ctx.stage));
        }
        if (!hs.pmsm.empty()) {
            m = std::max(m, refresh_pmsm_mna(x, J_nl, f_nl, hs.pmsm, ctx.h,
                                             ctx.stage));
        }
        if (!hs.tail.empty()) {
            m = std::max(m, refresh_igbt_tails(x, J_nl, f_nl, hs.tail,
                                               ctx.h, ctx.stage));
        }
        for (const auto& e : hs.sat.entries()) {
            m = std::max(m, stamp_saturable_inductor(e, x, J_nl, f_nl, ctx.h,
                                                     ctx.stage,
                                                     sat_increment_form));
        }
        return m;
    };
}

}  // namespace pulsim::pwl
