#pragma once

// =============================================================================
// Pulsim v2 — Layer 11 V18: Steinmetz / iGSE core-loss estimator
// =============================================================================
//
// Post-process analysis: given a simulation result and the
// inductor / transformer-winding branch's current history,
// estimate the core power loss using the Steinmetz formula:
//
//     P_v = K · f^α · B^β       [W/m³]
//
// where:
//   K, α, β: material-specific Steinmetz parameters (vendor
//            datasheets typically tabulate these for common
//            ferrite cores).
//   f      : excitation frequency [Hz]
//   B      : peak flux density during one cycle [T]
//
// B = N·i / (A_e · l_e)   (for an inductor, with N turns,
//                          A_e = effective core area,
//                          l_e = effective magnetic path length)
//
// We don't track flux density directly — for V0 this estimator
// takes the user-supplied (N, A_e, l_e) plus the recorded
// current history, computes B_peak over one cycle, then
// applies Steinmetz.
//
// Output: total volumetric loss [W/m³]. Multiply by core
// volume to get total core loss in watts.

#include "pulsim/numeric/types.hpp"
#include "pulsim/solver/result.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace pulsim::analysis {

struct CoreLossParams {
    // Steinmetz coefficients (from ferrite datasheet).
    Real K     = Real{0};       // [W/m³] base coefficient
    Real alpha = Real{1.5};     // frequency exponent
    Real beta  = Real{2.5};     // flux-density exponent
    // Magnetic geometry.
    Real N     = Real{1};       // turns
    Real A_e   = Real{1e-4};    // [m²] effective core area
    Real l_e   = Real{1e-2};    // [m] effective magnetic path
    // Volume for total loss.
    Real V_core = Real{1e-6};   // [m³] core volume
};

struct CoreLossResult {
    Real B_peak = Real{0};      // [T]
    Real f_est  = Real{0};      // [Hz] estimated cycle freq
    Real P_v    = Real{0};      // [W/m³] volumetric loss
    Real P_total = Real{0};     // [W] total core loss
};

/// Estimate core loss from a simulation result. Reads
/// branch-current history at `state_idx` over the time window
/// [t_start, t_end] (use the entire result if range omitted).
///
/// `state_idx` is the row in `result.states[k]` of the
/// inductor's branch-current variable — obtain from
/// `pool.branch_var_id_for_inductor(branch_id, graph)`.
///
/// Frequency estimation: count zero-crossings of (i − mean(i))
/// across the sample window to derive cycles, then f = cycles / Δt.
[[nodiscard]] inline CoreLossResult estimate_steinmetz(
    const solver::SimulationResult& result,
    Size state_idx,
    const CoreLossParams& p,
    Size k_start = 0,
    Size k_end   = 0) {
    if (k_end == 0 || k_end > result.num_steps()) {
        k_end = result.num_steps();
    }
    if (k_start >= k_end) {
        throw std::invalid_argument(
            "estimate_steinmetz: empty window");
    }
    if (k_end - k_start < 4) {
        throw std::invalid_argument(
            "estimate_steinmetz: window too small");
    }

    // 1. Find i_peak and i_mean in the window.
    Real i_min = result.states[k_start][state_idx];
    Real i_max = i_min;
    Real i_sum = Real{0};
    for (Size k = k_start; k < k_end; ++k) {
        const Real i = result.states[k][state_idx];
        i_min = std::min(i_min, i);
        i_max = std::max(i_max, i);
        i_sum += i;
    }
    const Real n = static_cast<Real>(k_end - k_start);
    const Real i_mean = i_sum / n;
    const Real i_ripple_pk = std::max(
        std::abs(i_max - i_mean),
        std::abs(i_mean - i_min));

    // 2. B_peak = N · i_peak / (A_e). (For ferrite cores
    //    Ampere's law gives H = N·i/l_e and B = μ·H; we
    //    approximate B_peak using the AC-ripple peak, which
    //    drives core loss.)
    CoreLossResult r;
    r.B_peak = p.N * i_ripple_pk / p.A_e;

    // 3. Frequency: count zero crossings of (i − i_mean).
    Size crossings = 0;
    Real prev = result.states[k_start][state_idx] - i_mean;
    for (Size k = k_start + 1; k < k_end; ++k) {
        const Real curr =
            result.states[k][state_idx] - i_mean;
        if ((prev < 0 && curr >= 0) ||
            (prev > 0 && curr <= 0)) {
            ++crossings;
        }
        prev = curr;
    }
    const Real t_window =
        result.times[k_end - 1] - result.times[k_start];
    if (t_window > Real{0} && crossings >= 2) {
        // Each cycle has 2 zero crossings.
        const Real cycles =
            static_cast<Real>(crossings) / Real{2};
        r.f_est = cycles / t_window;
    } else {
        r.f_est = Real{0};
    }

    // 4. Steinmetz volumetric loss.
    if (r.f_est > Real{0} && r.B_peak > Real{0}) {
        r.P_v = p.K *
                std::pow(r.f_est, p.alpha) *
                std::pow(r.B_peak, p.beta);
    } else {
        r.P_v = Real{0};
    }
    r.P_total = r.P_v * p.V_core;
    return r;
}

}  // namespace pulsim::analysis
