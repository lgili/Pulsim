#pragma once

// =============================================================================
// Pulsim — Layer 4 V17: Saturable-inductor Newton refresh
// =============================================================================
//
// Per-Newton-iteration stamping for `SaturableInductor` branches.
//
// THE STATE IS THE FLUX. The device obeys v = dλ/dt with
// λ(i) = ∫₀ⁱ L(u) du, so the trapezoidal rule is
//
//   λ(i_new) − λ(i_old) = (h/2)·(v_new + v_old)
//
// and the constraint-row residual (J·x + f = 0 form) is
//
//   R_row = (v_from − v_to) + V_L_old
//           − (2/h)·(λ(i_new) − λ_old) = 0
//
// This used to write the flux difference as L(i_new)·(i_new − i_old)
// — a right-endpoint rectangle rule, exact only while L is constant
// across the step. That is not a symmetric error: the stamp solves
// for Δi given the voltage, so ascending (L(i_new) smallest) Δi came
// out too large and descending (L(i_new) largest) |Δi| came out too
// small. Both push the current outward, so the error RECTIFIED. On a
// zero-mean 1 kHz sine at five thousand steps per cycle the DC
// current climbed 63.6 → 145.5 A over 400 cycles on a device with
// I_sat = 5 A, first order in h and unbounded in time, while a
// linear inductor in the same circuit held to 7e-15 A per cycle.
//
// The Jacobian gets SIMPLER, not harder:
//   ∂R/∂v_from  = +1
//   ∂R/∂v_to    = −1
//   ∂R/∂i_L_new = −(2/h)·L(i_new)
// The old form carried an extra −(2/h)·(i_new − i_old)·dL/di term.
// It is gone because it only ever existed as the derivative of the
// wrong expression; dλ/di IS L(i), exactly.
//
// TR-BDF2 SECOND STAGE:
//
//   R_row = (v_from − v_to)
//           − (c1·λ(i_new) + c2·λ_γ + c3·λ_old)/h = 0
//   ∂R/∂i_L_new = −(c1/h)·L(i_new)
//
// one-sided, so V_L_old is absent rather than rescaled. Since
// c1/h == 2/(γh), the conductance is identical to the trapezoidal
// stage's — which is what lets one factor serve both, and why
// stamping the wrong history term here would converge and lie.
//
// KCL contributions at from/to (current flows i_L from `from` to `to`):
//   f[from] += i_L_new        J[from, i_L_row] += 1
//   f[to]   -= i_L_new        J[to,   i_L_row] -= 1

#include "pulsim/models/device_model.hpp"
#include "pulsim/models/saturable_inductor.hpp"
#include "pulsim/numeric/dense.hpp"
#include "pulsim/numeric/types.hpp"
#include "pulsim/pwl/device_pool.hpp"
#include "pulsim/pwl/nonlinear_solve.hpp"
#include "pulsim/pwl/saturable_inductor_history.hpp"
#include "pulsim/pwl/trbdf2_stage.hpp"
#include "pulsim/sparse/matrix.hpp"
#include "pulsim/stamping/branch_coord.hpp"
#include "pulsim/topology/graph.hpp"

#include <algorithm>
#include <cmath>

namespace pulsim::pwl {

/// Stamp ONE saturable inductor. Both the stand-alone refresh and the
/// combined one call this; they used to carry byte-identical copies of
/// the body, which is how two stamps of the same device drift apart.
inline Real stamp_saturable_inductor(
    const SaturableInductorHistory::Entry& e,
    const Vector& x,
    sparse::Matrix& J_nl,
    Vector& f_nl,
    Real h,
    TrBdf2Stage stage) {
    const Real v_from = stamping::read_node_voltage(x, e.from);
    const Real v_to   = stamping::read_node_voltage(x, e.to);
    const Real i_L_new = x[e.branch_var_id];

    // λ(i) and its exact derivative L(i) under the entry's law —
    // closed form for the atan and table laws, an integration of the
    // Jiles-Atherton ODE from the step-start (H_n, M_n) for the
    // hysteretic one. Same residual, same Jacobian shape.
    const auto [lambda_new, L_eff] =
        SaturableInductorHistory::flux_and_inductance(e, i_L_new, stage);

    const auto kc = trbdf2_coeffs();
    const bool bdf2 = (stage == TrBdf2Stage::Bdf2Stage2);
    // c1/h in the BDF2 stage, 2/h in the trapezoidal one; equal when
    // h is that stage's own step, which is the whole point of γ.
    const Real d_coef = bdf2 ? kc.c1 / h : Real{2} / h;
    // Trapezoidal carries the previous step's voltage; BDF2 is
    // one-sided and carries none of it.
    const Real hist =
        bdf2 ? (kc.c2 * e.lambda_gamma + kc.c3 * e.lambda_old) / h
             : -e.V_L_old - (Real{2} / h) * e.lambda_old;

    const Real R_row = (v_from - v_to) - d_coef * lambda_new - hist;

    const bool from_active = stamping::node_is_active(e.from);
    const bool to_active   = stamping::node_is_active(e.to);

    if (from_active) f_nl[e.from] += i_L_new;
    if (to_active)   f_nl[e.to]   -= i_L_new;
    f_nl[e.branch_var_id] += R_row;

    if (from_active) {
        J_nl.coeffRef(e.branch_var_id, e.from) += Real{1};
        J_nl.coeffRef(e.from, e.branch_var_id) += Real{1};
    }
    if (to_active) {
        J_nl.coeffRef(e.branch_var_id, e.to)   -= Real{1};
        J_nl.coeffRef(e.to, e.branch_var_id)   -= Real{1};
    }
    J_nl.coeffRef(e.branch_var_id, e.branch_var_id) += -d_coef * L_eff;

    return std::abs(i_L_new);
}

/// Stamp the SaturableInductor contribution. Reads i_L_old
/// and V_L_old from `history`. Clears J_nl / f_nl first (so
/// this is a standalone refresh; for composition see the
/// combined refresh below). Returns max(|i_L_new|).
inline Real refresh_saturable_inductors(
    const Vector& x,
    sparse::Matrix& J_nl,
    Vector& f_nl,
    const topology::Graph& /*graph*/,
    const DevicePool& /*pool*/,
    const SaturableInductorHistory& history,
    Real dt,
    TrBdf2Stage stage = TrBdf2Stage::Trapezoidal) {
    if (J_nl.rows() > 0) J_nl.setZero();
    if (f_nl.size() > 0) f_nl.setZero();
    Real max_abs_i = Real{0};
    for (const auto& e : history.entries()) {
        max_abs_i = std::max(
            max_abs_i,
            stamp_saturable_inductor(e, x, J_nl, f_nl, dt, stage));
    }
    return max_abs_i;
}

/// NOTE ON A FUNCTION THAT USED TO LIVE HERE.
///
/// `make_combined_nonlinear_refresh(history, dt)` sat below this
/// point: 228 lines that re-implemented diode / MOSFET / IGBT
/// stamping and then appended a byte-for-byte duplicate of the
/// saturable block above. It had ZERO call sites anywhere in core,
/// python, tests, benchmarks or examples — every real caller uses
/// `make_combined_diode_mosfet_refresh()` instead — so the device's
/// physics existed in THREE places of which exactly one ran (the
/// live one was hand-inlined inside `run_transient`).
///
/// That is how two stamps of one device drift apart, and it nearly
/// mattered here: a change applied only to the two copies in this
/// file would have altered nothing at runtime while leaving the
/// documentation describing a formula the simulator no longer used.
/// Both are gone now — one `stamp_saturable_inductor` above, called
/// from both engines.

}  // namespace pulsim::pwl
