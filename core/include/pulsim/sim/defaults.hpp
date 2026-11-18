#pragma once

// add-kernel-simulation-object: THE default values of a run. The driver
// option structs (solver::SimulationOptions, solver::TrBdf2Options), the
// kernel RunOptions resolution, the exact engine's Python dispatcher
// (through pulsim._pulsim.run_option_defaults()) and simulate() all read
// them here; a test asserts they agree.

#include "pulsim/numeric/types.hpp"

namespace pulsim::sim {

/// THE default values. Every resolved default of a run is one of these.
struct Defaults {
    // Variable-step TR-BDF2 on the MNA (engine 'trbdf2' / 'auto').
    static constexpr Real rtol_variable = 1e-5;
    static constexpr Real atol_variable = 1e-8;
    // Event-driven exact engine ('exact' / 'dsed').
    static constexpr Real rtol_exact = 1e-6;
    static constexpr Real atol_exact = 1e-9;
    static constexpr Real dt_init_exact = 1e-9;
    static constexpr Real dt_max_exact = 1e-5;
    static constexpr Real stiffness_threshold = 10.0;
    static constexpr Real h_bdf2 = 1e-6;
    // Newton (pwl, trbdf2).
    static constexpr Size max_newton_iterations = 50;
    static constexpr Real tol_newton_dx = 1e-9;
    static constexpr Real tol_newton_res = 1e-9;
    static constexpr Size newton_roundoff_budget = 130;
    // Events and the fixed step.
    static constexpr Size max_event_iterations = 16;
    static constexpr Size max_dt_halvings = 6;
    static constexpr Size store_every = 1;
    static constexpr Real voltage_sanity_factor = 100.0;
    /// control_tick='state': a split edge moves by at most this many dt
    /// (G30; the driver's own default is 0.01).
    static constexpr Real split_snap_state = 1e-6;
};

}  // namespace pulsim::sim
