#pragma once

// =============================================================================
// Pulsim — PED engine: Dormand-Prince RK45 (DOPRI5) with FSAL.
// =============================================================================
//
// Reference: Hairer, Norsett, Wanner, *Solving ODE I: Nonstiff
// Problems*, Springer 1993/2008, §II.5 (Butcher tableau p. 178).
//
// 7-stage, order 5(4) embedded RK with FSAL (First-Same-As-Last).
// On accepted steps, k7 of step n becomes k1 of step n+1 →
// 6 RHS evaluations per accepted step, 7 per rejected.
//
// Default integrator in Pulsim's PED engine, selected by the
// Gate 0 integrator rule when events_per_step > 0.2 OR
// stiffness < 50 (see `notes/DSED_FOUNDATIONS.md` §13.6).
//
// Direct port of the Gate 1 Python prototype (`prototype/dsed/
// rk45_dormand_prince.py`); validated to RMSE = 0.0057 % on buck
// CCM, 17× better than the Gate 1A target of 0.1 %.

#include <functional>
#include <optional>
#include <utility>

#include "pulsim/numeric/dense.hpp"
#include "pulsim/numeric/types.hpp"

namespace pulsim::dsed {

namespace dopri5 {

// --- Butcher tableau coefficients (Hairer/Norsett/Wanner 1993, p. 178) ---
inline constexpr Real C2 = Real{1} / Real{5};
inline constexpr Real C3 = Real{3} / Real{10};
inline constexpr Real C4 = Real{4} / Real{5};
inline constexpr Real C5 = Real{8} / Real{9};
inline constexpr Real C6 = Real{1};
inline constexpr Real C7 = Real{1};

inline constexpr Real A21 = Real{1} / Real{5};

inline constexpr Real A31 = Real{3} / Real{40};
inline constexpr Real A32 = Real{9} / Real{40};

inline constexpr Real A41 = Real{44} / Real{45};
inline constexpr Real A42 = -Real{56} / Real{15};
inline constexpr Real A43 = Real{32} / Real{9};

inline constexpr Real A51 = Real{19372} / Real{6561};
inline constexpr Real A52 = -Real{25360} / Real{2187};
inline constexpr Real A53 = Real{64448} / Real{6561};
inline constexpr Real A54 = -Real{212} / Real{729};

inline constexpr Real A61 = Real{9017} / Real{3168};
inline constexpr Real A62 = -Real{355} / Real{33};
inline constexpr Real A63 = Real{46732} / Real{5247};
inline constexpr Real A64 = Real{49} / Real{176};
inline constexpr Real A65 = -Real{5103} / Real{18656};

// b coefficients (5th-order solution)
inline constexpr Real B1 = Real{35} / Real{384};
inline constexpr Real B3 = Real{500} / Real{1113};
inline constexpr Real B4 = Real{125} / Real{192};
inline constexpr Real B5 = -Real{2187} / Real{6784};
inline constexpr Real B6 = Real{11} / Real{84};

// (b - bhat) coefficients — embedded error estimate
inline constexpr Real E1 = Real{71} / Real{57600};
inline constexpr Real E3 = -Real{71} / Real{16695};
inline constexpr Real E4 = Real{71} / Real{1920};
inline constexpr Real E5 = -Real{17253} / Real{339200};
inline constexpr Real E6 = Real{22} / Real{525};
inline constexpr Real E7 = -Real{1} / Real{40};

}  // namespace dopri5

/// Persistent FSAL state between accepted DOPRI5 steps.
///
/// On rejection, call ``invalidate()`` to force k1 recompute.
/// On topology change (event), do the same — f discontinuously
/// changes so the cached k1 is stale.
struct RK45State {
    std::optional<Vector> k1;

    [[nodiscard]] bool fsal_valid() const noexcept { return k1.has_value(); }
    void invalidate() noexcept { k1.reset(); }
};

/// RHS function signature: ``f(t, x) -> dx/dt``.
using RhsFn = std::function<Vector(Real, const Vector&)>;

/// One DOPRI5 step.
///
/// Returns ``(x_new, err)`` where ``err`` is the componentwise
/// embedded error estimate ``h · (b - bhat) · k``.
///
/// Mutates ``state.k1`` to ``k7`` for next-step FSAL reuse.
/// Cost: 6 RHS evals on accepted steps (FSAL); 7 on rejected
/// (k1 must be recomputed since FSAL was invalidated).
template <class F>
[[nodiscard]] std::pair<Vector, Vector> step(F&& f,
                                                Real t,
                                                const Vector& x,
                                                Real h,
                                                RK45State& state) {
    using namespace dopri5;

    // Stage 1: FSAL if available, else compute fresh
    const Vector k1 = state.fsal_valid()
                       ? *state.k1
                       : Vector{f(t, x)};

    // Stages 2-7
    const Vector k2 = f(t + C2 * h, x + h * (A21 * k1));
    const Vector k3 = f(t + C3 * h, x + h * (A31 * k1 + A32 * k2));
    const Vector k4 = f(t + C4 * h, x + h * (A41 * k1 + A42 * k2 + A43 * k3));
    const Vector k5 = f(t + C5 * h, x + h * (A51 * k1 + A52 * k2 + A53 * k3 + A54 * k4));
    const Vector k6 = f(t + C6 * h, x + h * (A61 * k1 + A62 * k2 + A63 * k3 + A64 * k4 + A65 * k5));

    // 5th-order solution
    Vector x_new = x + h * (B1 * k1 + B3 * k3 + B4 * k4 + B5 * k5 + B6 * k6);

    // FSAL stage — k7 = f(t+h, x_new) → reused as k1 next step
    Vector k7 = f(t + C7 * h, x_new);

    // Embedded error estimate
    Vector err = h * (E1 * k1 + E3 * k3 + E4 * k4 + E5 * k5
                       + E6 * k6 + E7 * k7);

    // Stash for next step (FSAL)
    state.k1 = std::move(k7);

    return {std::move(x_new), std::move(err)};
}

}  // namespace pulsim::dsed
