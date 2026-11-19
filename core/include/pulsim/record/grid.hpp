// SPDX-License-Identifier: MIT
//
// add-probes-and-recorders D3: what grid a result's rows sit on, and the
// run's engine-independent counters. Filled by the engine adapters (the
// Python finalisation step today); counters an engine does not report
// stay zero.
#pragma once

#include "pulsim/numeric/types.hpp"

#include <cstdint>
#include <string>

namespace pulsim::record {

/// Uniform: rows on the fixed step grid (pwl). Accepted: one row per
/// accepted variable step plus event rows (trbdf2, exact). Requested:
/// rows at caller-chosen times (t_eval).
enum class GridKind : std::uint8_t { Uniform, Accepted, Requested };

struct RunStats {
    std::string engine;           ///< "pwl" | "trbdf2" | "dsed"; empty: not filled
    Size n_steps = 0, n_accept = 0, n_reject = 0, n_solves = 0, n_factorizations = 0;
    Size n_gate_events = 0, n_state_events = 0, n_mode_events = 0, n_ctrl_ticks = 0;
    Size n_newton_iterations = 0, n_newton_retries = 0, n_forced_accepts = 0;
    Size n_chatter_breaks = 0, n_gate_undersampled = 0;
    Size n_rk45_steps = 0, n_bdf2_steps = 0, n_exact_steps = 0;  ///< exact engine's integrators
    Real gate_pulse_min = 0, gate_sample_spacing_max = 0;
    Real cpu_time_seconds = 0;
};

}  // namespace pulsim::record
