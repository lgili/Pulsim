#pragma once

// add-kernel-simulation-object: THE default values of a run. The driver
// option structs (solver::SimulationOptions, solver::TrBdf2Options), the
// kernel RunOptions resolution, the exact engine's Python dispatcher
// (through pulsim._pulsim.run_option_defaults()) and simulate() all read
// them here; a test asserts they agree.

#include "pulsim/device/device_path.hpp"
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

    // update-number-changing-release: the flipped defaults of a run (the
    // old value of each stays reachable by an explicit option: RunOptions
    // discontinuity_restart = false, control_tick_state = false,
    // device_path = Legacy). mode_selection 'auto' is resolved by simulate()
    // (the kernel has no 'auto' rule: RunOptions keeps Legacy = the settle).
    static constexpr bool discontinuity_restart = true;
    static constexpr bool substep_state_correction = true;
    static constexpr bool control_tick_state = true;
    static constexpr device::DevicePath device_path = device::DevicePath::Interface;
};

}  // namespace pulsim::sim
