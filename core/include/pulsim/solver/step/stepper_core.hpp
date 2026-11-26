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

}  // namespace pulsim::step
