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
#include <concepts>
#include <optional>

namespace pulsim::pwl {

/// The step every device stamp of the next solve integrates over: the
/// interval `h` it advances and the rule (`stage`) of its time formula.
/// Owned by the driver, set before every solve, read by reference.
/// `increment_form`: the saturable inductor's flux residual in the
/// closed-form increment form (TR-BDF2) or the direct form (fixed step);
/// an engine constant, set by `build_device_refresh` (the drivers assign
/// `{h, stage}` per solve).
struct StepCtx {
    Real h = Real{0};
    TrBdf2Stage stage = TrBdf2Stage::Trapezoidal;
    bool increment_form = false;
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

/// The refresh of a DC operating point (no stateful pass: DC has no
/// step). `enable` unset means "decide from the pool": the standard chain
/// whenever the pool has any Newton device; false solves them as open
/// circuits (the caller asked for it).
[[nodiscard]] inline NonlinearRefreshFn dc_device_refresh(
    const DevicePool& pool, std::optional<bool> enable) {
    const bool want =
        enable.has_value() ? *enable : pool.has_nonlinear_devices();
    if (!want) return {};
    return make_combined_diode_mosfet_refresh();
}

// The Newton stamp of each stateful kind, reading the step ONLY from `c`.
inline Real stamp_kind(NonlinearCapacitorHistory& k, const Vector& x,
                       sparse::Matrix& J, Vector& f, const StepCtx& c) {
    return refresh_nonlinear_capacitors(x, J, f, k, c.h,
                                        coss_stage_of(c.stage),
                                        &k.q_gamma());
}
inline Real stamp_kind(LauritzenDiodeHistory& k, const Vector& x,
                       sparse::Matrix& J, Vector& f, const StepCtx& c) {
    return refresh_lauritzen_diodes(x, J, f, k, c.h, c.stage);
}
inline Real stamp_kind(PmsmMnaHistory& k, const Vector& x,
                       sparse::Matrix& J, Vector& f, const StepCtx& c) {
    return refresh_pmsm_mna(x, J, f, k, c.h, c.stage);
}
inline Real stamp_kind(IgbtTailHistory& k, const Vector& x,
                       sparse::Matrix& J, Vector& f, const StepCtx& c) {
    return refresh_igbt_tails(x, J, f, k, c.h, c.stage);
}
inline Real stamp_kind(SaturableInductorHistory& k, const Vector& x,
                       sparse::Matrix& J, Vector& f, const StepCtx& c) {
    Real m = Real{0};
    for (const auto& e : k.entries()) {
        m = std::max(m, stamp_saturable_inductor(e, x, J, f, c.h, c.stage,
                                                 c.increment_form));
    }
    return m;
}

namespace detail {
template <class List> struct AllStamp;
template <class... Ks> struct AllStamp<device::TypeList<Ks...>> {
    static constexpr bool value =
        (requires(Ks& k, const Vector& x, sparse::Matrix& J, Vector& f,
                  const StepCtx& c) {
            { stamp_kind(k, x, J, f, c) } -> std::same_as<Real>;
        } && ...);
};
}  // namespace detail
static_assert(detail::AllStamp<StatefulKinds>::value,
              "every StatefulKinds entry needs a stamp_kind overload");

/// The whole Newton refresh: `base` (from `static_device_refresh`), then
/// every stateful kind in list order. Empty when there is nothing to
/// stamp, so a linear circuit keeps its linear path. `hs` and `ctx` are
/// captured by reference and must outlive the returned function.
[[nodiscard]] inline NonlinearRefreshFn build_device_refresh(
    NonlinearRefreshFn base, StatefulHistories& hs, const StepCtx& ctx,
    bool increment_form) {
    if (!hs.any()) return base;
    return [base = std::move(base), &hs, &ctx, increment_form](
               const Vector& x, sparse::Matrix& J_nl, Vector& f_nl,
               const topology::Graph& g, const DevicePool& pool) -> Real {
        Real m = Real{0};
        if (base) {
            m = base(x, J_nl, f_nl, g, pool);
        } else {
            sparse::zero_keep_pattern(J_nl);
            f_nl.setZero();
        }
        StepCtx c = ctx;
        c.increment_form = increment_form;
        hs.for_each([&](auto& k) {
            if (!k.empty()) m = std::max(m, stamp_kind(k, x, J_nl, f_nl, c));
        });
        return m;
    };
}

}  // namespace pulsim::pwl
