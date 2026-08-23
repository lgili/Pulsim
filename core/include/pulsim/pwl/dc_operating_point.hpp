#pragma once

// =============================================================================
// Pulsim — Layer 4: the DC operating point, resolved
// =============================================================================
//
// v2.0 Phase 2 (B.2).
//
// "The DC operating point" is not one solve. Getting it right means
// three things at once, and until now only `run_transient` did all
// three — every other entry point did a subset and reported the
// answer with the same confidence:
//
//   1. NONLINEAR DEVICES MUST BE STAMPED. `dc_assemble` skips
//      `BranchKind::Nonlinear` as an open circuit, so a bare
//      `compute_dc_op` on a 5 V source through 1 kΩ into a diode
//      answers 5.000 V at the anode. The truth is 0.700 V.
//   2. PWL DIODE STATES MUST BE RESOLVED. A diode's on/off bit is an
//      input to the matrix and an output of the solve, so the pair
//      has to be iterated to consistency. Solving once with every
//      diode assumed OFF answers the question for a different
//      circuit.
//   3. THE SOLVE MUST BE ALLOWED TO FAIL AND RECOVER. A stiff
//      operating point is normal, not exceptional; the cascade in
//      `dc_strategy.hpp` is what turns it into an answer.
//
// This header is those three, once, for everyone: `run_transient`'s
// pre-charge, the BDF1 bootstrap, and the public Python entry point
// all route through `compute_dc_operating_point`.

#include "pulsim/numeric/dense.hpp"
#include "pulsim/numeric/types.hpp"
#include "pulsim/pwl/dc_strategy.hpp"
#include "pulsim/pwl/device_pool.hpp"
#include "pulsim/pwl/diode_event_state.hpp"
#include "pulsim/pwl/gmin.hpp"
#include "pulsim/pwl/nonlinear_solve.hpp"
#include "pulsim/topology/graph.hpp"
#include "pulsim/topology/switch_state.hpp"

#include <format>
#include <stdexcept>
#include <string>
#include <utility>

namespace pulsim::pwl {

/// Knobs for a full operating-point resolution.
struct DCOperatingPointOptions {
    Real t_eval = Real{0};

    /// Diode-consistency rounds. Each round re-solves with the diode
    /// bits the previous round implied; zero or one means "trust the
    /// mask you were given".
    Size max_event_iterations = Size{16};

    Size max_newton_iters = Size{50};
    Real tol_dx  = Real{1e-9};
    Real tol_res = Real{1e-9};
    bool enable_line_search = false;
    bool enable_lm = false;

    GminConfig gmin{};

    /// When the direct solve fails, walk rungs 2-4 of the cascade
    /// (gmin stepping → source stepping → pseudo-transient) instead
    /// of giving up. Turning this off is how you ask for the raw
    /// diagnostic rather than an answer.
    bool enable_cascade = true;
};

/// The resolved operating point, plus what it took to get there.
struct DCOperatingPoint {
    Vector x;
    topology::SwitchStateMask mask{0};  //!< switch + diode state solved at
    DCSolveReport report;
    Size event_iterations = Size{0};
};

/// Solve the DC operating point of `graph`, honouring nonlinear
/// devices, iterating PWL diode states to consistency, and falling
/// through the DC cascade when the direct solve fails.
///
/// `diodes` may be null for a circuit with no PWL diodes. When it is
/// supplied, it is UPDATED IN PLACE to the resolved state — callers
/// that go on to run a transient want exactly that, so the run starts
/// from the same diode configuration the operating point implies.
///
/// `refresh` is the nonlinear stamping chain. Pass the RAW static
/// device chain (diodes / MOSFET / IGBT) and not a trap-companion
/// wrapper: companion stamps carry a 1/dt, which has no meaning in a
/// DC system.
[[nodiscard]] inline DCOperatingPoint compute_dc_operating_point(
    const topology::Graph& graph,
    const DevicePool& pool,
    const topology::SwitchStateMask& user_mask,
    const NonlinearRefreshFn& refresh = {},
    const DCOperatingPointOptions& opts = {},
    DiodeEventState* diodes = nullptr,
    const char* who = "compute_dc_operating_point") {

    const bool has_diodes =
        diodes != nullptr && diodes->num_diodes() > 0;
    const topology::SwitchStateMask diode_owned =
        has_diodes ? diodes->diode_owned_bits()
                    : topology::SwitchStateMask(0);

    DCOperatingPoint out;
    out.mask = user_mask;

    const Size max_rounds =
        opts.max_event_iterations > 0 ? opts.max_event_iterations
                                       : Size{1};
    bool flipped = false;
    Size iters = 0;

    do {
        topology::SwitchStateMask mask = user_mask;
        if (has_diodes) {
            mask = mask.overlay(diodes->current_diode_mask(),
                                 diode_owned);
        }
        out.mask = mask;

        try {
            out.x = refresh
                ? compute_dc_op_newton(
                      graph, pool, mask, refresh, opts.t_eval,
                      opts.max_newton_iters, opts.tol_dx,
                      opts.tol_res, opts.enable_line_search,
                      opts.enable_lm, opts.gmin.floor)
                : compute_dc_op(graph, pool, mask, opts.t_eval,
                                 opts.gmin.floor);
            out.report.strategy = DCStrategy::Naive;
            out.report.rungs_attempted = Size{1};
            out.report.final_gmin = opts.gmin.floor;
        } catch (const std::exception& direct_failed) {
            if (!opts.enable_cascade) {
                throw;
            }
            // The direct solve is rung 1. A stiff operating point is
            // exactly what the rest of the cascade exists for, so
            // walk it before handing the user an error.
            try {
                out.x = compute_dc_op_with_strategy(
                    graph, pool, mask, DCStrategy::Auto, opts.t_eval,
                    PseudoTransientConfig{}, SourceSteppingConfig{},
                    analysis::ShouldContinueFn{}, refresh, opts.gmin,
                    &out.report);
            } catch (const std::exception& cascade_failed) {
                throw std::runtime_error(std::format(
                    "{}: no DC operating point. The direct solve "
                    "failed with: {}\nThe fallback cascade then "
                    "reported: {}",
                    who, direct_failed.what(),
                    cascade_failed.what()));
            }
        }

        flipped = has_diodes && diodes->update_from_state(out.x);
        ++iters;
    } while (flipped && iters < max_rounds);

    out.event_iterations = iters;
    if (flipped) {
        throw std::runtime_error(std::format(
            "{}: the PWL diode states never settled — {} rounds of "
            "re-solving still flip at least one diode. The circuit "
            "has no consistent DC diode configuration (a latch, or "
            "two diodes fighting over the same node); raise "
            "max_event_iterations, or start the run from zero "
            "instead of from a DC operating point.",
            who, max_rounds));
    }
    return out;
}

}  // namespace pulsim::pwl
