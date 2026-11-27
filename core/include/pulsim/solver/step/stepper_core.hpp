#pragma once

// refactor-unified-transient-stepper, task groups 2 and 3 (design D1, D3):
// what both MNA drivers do the same way to the run's state, as one copy.
//
// - `solve_stage`: the single solve point. It tells the stateful devices
//   which stage they stamp (the step context from the stage table), then
//   runs the engine's linear-or-Newton dispatch. The dispatch itself, the
//   right-hand side and the counters stay the engine's (DECISIONS 22): the
//   two engines differ there on purpose (the fixed engine's memoised
//   nominal segment and restart-factor count, TR-BDF2's linear restart of a
//   failed first Newton solve, the right-hand-side association of
//   DECISIONS 7).
// - `commit`, `snapshot_into`, `restore`, `seed`: the committed past of a
//   run is two objects (the linear companions' history and the stateful
//   devices' histories); each function is exactly the adjacent pair of
//   calls the drivers made, in the same order: linear first (DECISIONS 23).
//
// Everything is forced inline: these sit in the step loop, where an
// out-of-line call shows on the instructions-per-step gate (DECISIONS 14,
// 16).

#include "pulsim/numeric/types.hpp"
#include "pulsim/pwl/device_refresh.hpp"   // StepCtx
#include "pulsim/pwl/trbdf2_stage.hpp"
#include "pulsim/solver/step/integrators.hpp"   // StageSpec

#include <utility>

#if defined(__GNUC__) || defined(__clang__)
#define PULSIM_STEP_INLINE [[gnu::always_inline]] inline
#else
#define PULSIM_STEP_INLINE inline
#endif

namespace pulsim::step {

/// The one place a solve is asked for: `ctx` is told the stage (and the
/// time it ends at, the external devices' ctx->t), then `dispatch()` solves
/// and returns whether it converged.
template <class Dispatch>
PULSIM_STEP_INLINE bool solve_stage(pwl::StepCtx& ctx, const StageSpec& stage,
                                    Real t_end, Dispatch&& dispatch) {
    ctx = stage.ctx();
    ctx.t = t_end;
    return std::forward<Dispatch>(dispatch)();
}

/// The linear-or-Newton choice of a solve point: `linear()` cannot fail,
/// `newton()` returns whether it converged.
template <class Linear, class Newton>
PULSIM_STEP_INLINE bool dispatch_solve(bool nonlinear, Linear&& linear,
                                       Newton&& newton) {
    if (!nonlinear) {
        std::forward<Linear>(linear)();
        return true;
    }
    return std::forward<Newton>(newton)();
}

/// Accept `x` as the state after a step of `h` integrated with `formula`
/// (trapezoidal or backward Euler).
template <class History, class Devices>
PULSIM_STEP_INLINE void commit(History& history, Devices& devices, const Vector& x,
                               Real h, pwl::TrBdf2Stage formula) {
    if (formula == pwl::TrBdf2Stage::BackwardEuler) {
        history.update_from_state_backward_euler(x, h);
    } else {
        history.update_from_state(x, h);
    }
    devices.commit(x, h, formula);
}

/// The rollback copy of both histories, into the caller's buffers (which
/// keep their capacity), and its restore.
template <class History, class Devices, class HistorySnap, class DevicesSnap>
PULSIM_STEP_INLINE void snapshot_into(const History& history, const Devices& devices,
                                      HistorySnap& h, DevicesSnap& d) {
    history.snapshot_into(h);
    devices.snapshot_into(d);
}
template <class History, class Devices, class HistorySnap, class DevicesSnap>
PULSIM_STEP_INLINE void restore(History& history, Devices& devices,
                                const HistorySnap& h, const DevicesSnap& d) {
    history.restore(h);
    devices.restore(d);
}

/// Seed both from a consistent state (a DC point or a given state).
template <class History, class Devices>
PULSIM_STEP_INLINE void seed(History& history, Devices& devices, const Vector& x) {
    history.seed_from_dc_op(x);
    devices.seed_from_dc_op(x);
}

}  // namespace pulsim::step
