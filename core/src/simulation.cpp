#include "pulsim/simulation.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>

namespace pulsim {

// =============================================================================
// SwitchingLossTable implementation
// =============================================================================

namespace {
// Helper function for 2D bilinear interpolation
Real bilinear_interpolate(const std::vector<Real>& x_vals, const std::vector<Real>& y_vals,
                          const std::vector<Real>& data, Real x, Real y) {
    if (x_vals.empty() || y_vals.empty() || data.empty()) {
        return 0.0;
    }

    // Find x indices
    size_t nx = x_vals.size();
    size_t ny = y_vals.size();

    // Clamp x to range
    Real x_clamped = std::max(x_vals.front(), std::min(x, x_vals.back()));
    Real y_clamped = std::max(y_vals.front(), std::min(y, y_vals.back()));

    // Find lower indices
    size_t ix = 0;
    for (size_t i = 0; i < nx - 1; ++i) {
        if (x_vals[i + 1] >= x_clamped) {
            ix = i;
            break;
        }
        ix = i;
    }

    size_t iy = 0;
    for (size_t i = 0; i < ny - 1; ++i) {
        if (y_vals[i + 1] >= y_clamped) {
            iy = i;
            break;
        }
        iy = i;
    }

    // Handle edge cases
    size_t ix1 = std::min(ix + 1, nx - 1);
    size_t iy1 = std::min(iy + 1, ny - 1);

    // Get the four corner values
    Real q00 = data[iy * nx + ix];
    Real q10 = data[iy * nx + ix1];
    Real q01 = data[iy1 * nx + ix];
    Real q11 = data[iy1 * nx + ix1];

    // Calculate interpolation weights
    Real tx = (ix == ix1) ? 0.0 : (x_clamped - x_vals[ix]) / (x_vals[ix1] - x_vals[ix]);
    Real ty = (iy == iy1) ? 0.0 : (y_clamped - y_vals[iy]) / (y_vals[iy1] - y_vals[iy]);

    // Bilinear interpolation
    Real val = q00 * (1 - tx) * (1 - ty) +
               q10 * tx * (1 - ty) +
               q01 * (1 - tx) * ty +
               q11 * tx * ty;

    return val;
}
}  // namespace

Real SwitchingLossTable::interpolate_eon(Real voltage, Real current) const {
    return bilinear_interpolate(voltages, currents, eon_data, voltage, current);
}

Real SwitchingLossTable::interpolate_eoff(Real voltage, Real current) const {
    return bilinear_interpolate(voltages, currents, eoff_data, voltage, current);
}

Real SwitchingLossTable::interpolate_err(Real voltage, Real current) const {
    return bilinear_interpolate(voltages, currents, err_data, voltage, current);
}

// =============================================================================
// Simulator implementation
// =============================================================================

Simulator::Simulator(const Circuit& circuit, const SimulationOptions& options)
    : circuit_(circuit)
    , options_(options)
    , assembler_(circuit)
    , newton_solver_()
{
    NewtonSolver::Options newton_opts;
    newton_opts.max_iterations = options.max_newton_iterations;
    newton_opts.abstol = options.abstol;
    newton_opts.reltol = options.reltol;
    newton_opts.damping = options.damping_factor;
    newton_opts.auto_damping = true;
    newton_solver_.set_options(newton_opts);
}

NewtonResult Simulator::dc_operating_point() {
    Index n = circuit_.total_variables();

    // Initial guess: all zeros
    Vector x0 = Vector::Zero(n);

    // For DC analysis: capacitors open, inductors shorted
    // Need to iterate to allow switches to stabilize based on control voltages
    auto system_func = [this](const Vector& x, Vector& f, SparseMatrix& J) {
        // Update switch states based on current solution
        assembler_.update_switch_states(x, 0.0);

        // Assemble DC system
        SparseMatrix G;
        Vector b;
        assembler_.assemble_dc(G, b);

        // Add nonlinear contributions if any
        if (assembler_.has_nonlinear()) {
            SparseMatrix J_nl;
            Vector f_nl;
            assembler_.assemble_nonlinear(J_nl, f_nl, x);
            G += J_nl;
            b += f_nl;
        }

        // f(x) = G*x - b
        f = G * x - b;
        J = G;
    };

    return newton_solver_.solve(x0, system_func);
}

void Simulator::build_system(const Vector& x, Vector& f, SparseMatrix& J,
                             Real time, Real dt, const Vector& x_prev) {
    // Assemble transient system with companion models
    Vector b;
    assembler_.assemble_transient(J, b, x_prev, dt);

    // Update source values for current time
    assembler_.evaluate_sources(b, time);

    // Add nonlinear contributions if any
    if (assembler_.has_nonlinear()) {
        SparseMatrix J_nl;
        Vector f_nl;
        assembler_.assemble_nonlinear(J_nl, f_nl, x);
        J += J_nl;
        b += f_nl;
    }

    // f(x) = J*x - b
    f = J * x - b;
}

NewtonResult Simulator::step(Real time, Real dt, const Vector& x_prev) {
    auto system_func = [this, time, dt, &x_prev](const Vector& x, Vector& f, SparseMatrix& J) {
        build_system(x, f, J, time, dt, x_prev);
    };

    // Use previous solution as initial guess
    return newton_solver_.solve(x_prev, system_func);
}

SimulationResult Simulator::run_transient() {
    return run_transient(nullptr);
}

SimulationResult Simulator::run_transient(SimulationCallback callback) {
    return run_transient(callback, nullptr);
}

SimulationResult Simulator::run_transient(SimulationCallback callback, EventCallback event_callback,
                                          SimulationControl* control) {
    SimulationResult result;
    auto start_time = std::chrono::high_resolution_clock::now();

    Index n = circuit_.total_variables();

    // Reset power losses
    power_losses_ = PowerLosses{};

    // Build signal names
    for (Index i = 0; i < n; ++i) {
        result.signal_names.push_back(circuit_.signal_name(i));
    }

    // Get initial state
    Vector x;
    if (options_.use_ic) {
        // Use specified initial conditions (zeros for now)
        x = Vector::Zero(n);
    } else {
        // Compute DC operating point
        auto dc_result = dc_operating_point();
        if (dc_result.status != SolverStatus::Success) {
            result.final_status = dc_result.status;
            result.error_message = "DC operating point failed: " + dc_result.error_message;
            return result;
        }
        x = dc_result.x;
        result.newton_iterations_total += dc_result.iterations;
    }

    // Initialize switch states based on initial solution
    assembler_.update_switch_states(x, options_.tstart);

    // Store initial state
    Real time = options_.tstart;
    result.time.push_back(time);
    result.data.push_back(x);

    if (callback) {
        callback(time, x);
    }

    // Time stepping loop
    Real dt = options_.dt;
    int step_count = 0;

    while (time < options_.tstop) {
        if (control) {
            if (control->should_stop()) {
                result.final_status = SolverStatus::Success;
                result.error_message = "Simulation stopped by user";
                break;
            }

            while (control->should_pause() && !control->should_stop()) {
                control->wait_until_resumed();
            }

            if (control->should_stop()) {
                result.final_status = SolverStatus::Success;
                result.error_message = "Simulation stopped by user";
                break;
            }
        }

        // Don't overshoot tstop
        if (time + dt > options_.tstop) {
            dt = options_.tstop - time;
        }

        Real next_time = time + dt;

        // Take a step
        auto step_result = step(next_time, dt, x);

        if (step_result.status != SolverStatus::Success) {
            // Try with smaller timestep
            bool converged = false;
            Real dt_try = dt * 0.5;

            while (dt_try >= options_.dtmin && !converged) {
                next_time = time + dt_try;
                step_result = step(next_time, dt_try, x);

                if (step_result.status == SolverStatus::Success) {
                    converged = true;
                    dt = dt_try;  // Use this timestep going forward
                } else {
                    dt_try *= 0.5;
                }
            }

            if (!converged) {
                result.final_status = step_result.status;
                result.error_message = "Simulation failed at t=" + std::to_string(time) +
                                      ": " + step_result.error_message;
                break;
            }
        }

        result.newton_iterations_total += step_result.iterations;

        // Check for switch events
        if (assembler_.check_switch_events(step_result.x)) {
            // Event detected - find exact time using bisection
            Real t_event;
            Vector x_event;
            if (find_event_time(time, next_time, x, t_event, x_event)) {
                // Record event
                for (const auto& comp : circuit_.components()) {
                    if (comp.type() != ComponentType::Switch) continue;

                    const SwitchState* state = assembler_.find_switch_state(comp.name());
                    if (!state) continue;

                    const auto& params = std::get<SwitchParams>(comp.params());

                    // Get control voltage
                    Index n_ctrl_pos = circuit_.node_index(comp.nodes()[2]);
                    Index n_ctrl_neg = circuit_.node_index(comp.nodes()[3]);
                    Real v_ctrl = 0.0;
                    if (n_ctrl_pos >= 0) v_ctrl += x_event(n_ctrl_pos);
                    if (n_ctrl_neg >= 0) v_ctrl -= x_event(n_ctrl_neg);

                    bool would_close = v_ctrl > params.vth;
                    if (would_close != state->is_closed) {
                        // Get switch voltage and current
                        Index n1 = circuit_.node_index(comp.nodes()[0]);
                        Index n2 = circuit_.node_index(comp.nodes()[1]);
                        Real v_switch = 0.0;
                        if (n1 >= 0) v_switch += x_event(n1);
                        if (n2 >= 0) v_switch -= x_event(n2);
                        Real R = state->is_closed ? params.ron : params.roff;
                        Real i_switch = v_switch / R;

                        // Calculate switching loss (already accumulated in the function)
                        calculate_switching_loss(comp, *state, v_switch, i_switch, would_close);

                        // Record event in result
                        SimulationEvent sim_event;
                        sim_event.time = t_event;
                        sim_event.type = would_close ? SimulationEventType::SwitchClose
                                                     : SimulationEventType::SwitchOpen;
                        sim_event.component = comp.name();
                        sim_event.description = comp.name() + (would_close ? " closed" : " opened");
                        sim_event.value1 = v_switch;
                        sim_event.value2 = i_switch;
                        result.events.push_back(sim_event);

                        // Fire event callback
                        if (event_callback) {
                            SwitchEvent event;
                            event.switch_name = comp.name();
                            event.time = t_event;
                            event.new_state = would_close;
                            event.voltage = v_switch;
                            event.current = i_switch;
                            event_callback(event);
                        }
                    }
                }

                // Update switch states at event time
                assembler_.update_switch_states(x_event, t_event);

                // Re-simulate from event time to next_time with updated switch states
                Real dt_remaining = next_time - t_event;
                if (dt_remaining > options_.dtmin) {
                    step_result = step(next_time, dt_remaining, x_event);
                } else {
                    step_result.x = x_event;
                }
            } else {
                // Bisection failed, just update states
                assembler_.update_switch_states(step_result.x, next_time);
            }
        } else {
            // No events, update states normally
            assembler_.update_switch_states(step_result.x, next_time);
        }

        // Accumulate conduction losses
        accumulate_conduction_losses(step_result.x, dt);

        // Track diode states for reverse recovery detection
        update_diode_states(step_result.x, x, dt);

        // Update state
        x = step_result.x;
        time = next_time;
        step_count++;

        // Decimation and rolling buffer storage
        bool should_store = (options_.streaming_decimation <= 1) ||
                           (step_count % options_.streaming_decimation == 0);

        if (should_store) {
            // Rolling buffer: remove oldest point if at capacity
            if (options_.streaming_rolling_buffer &&
                static_cast<int64_t>(result.time.size()) >= options_.streaming_max_points) {
                result.time.erase(result.time.begin());
                result.data.erase(result.data.begin());
            }
            result.time.push_back(time);
            result.data.push_back(x);
        }

        // Callback is separate from storage (invoked every step for real-time updates)
        if (callback) {
            callback(time, x);
        }

        // Adaptive timestep: increase if converged quickly
        if (step_result.iterations < 5 && dt < options_.dtmax) {
            dt = std::min(dt * 1.2, options_.dtmax);
        }
    }

    result.total_steps = step_count;
    result.final_status = SolverStatus::Success;

    auto end_time = std::chrono::high_resolution_clock::now();
    result.total_time_seconds = std::chrono::duration<double>(end_time - start_time).count();

    // Compute average newton iterations
    if (step_count > 0) {
        result.average_newton_iterations = static_cast<double>(result.newton_iterations_total) / step_count;
    }

    // Populate solver info
    result.solver_info.method = options_.integration_method;
    result.solver_info.abstol = options_.abstol;
    result.solver_info.reltol = options_.reltol;
    result.solver_info.adaptive_timestep = options_.adaptive_timestep;

    return result;
}

SimulationResult Simulator::run_transient_with_progress(
    SimulationCallback callback,
    EventCallback event_callback,
    SimulationControl* control,
    const ProgressCallbackConfig& progress_config) {

    SimulationResult result;
    auto start_time = std::chrono::high_resolution_clock::now();
    auto last_progress_time = start_time;

    Index n = circuit_.total_variables();

    // Reset power losses and performance counters
    power_losses_ = PowerLosses{};
    int convergence_failures = 0;
    int timestep_reductions = 0;

    // Build signal names and signal info
    for (Index i = 0; i < n; ++i) {
        result.signal_names.push_back(circuit_.signal_name(i));

        // Build signal info
        SignalInfo info;
        info.name = circuit_.signal_name(i);
        if (i < circuit_.node_count()) {
            info.type = "voltage";
            info.unit = "V";
            info.nodes.push_back(circuit_.node_name(i));
        } else {
            info.type = "current";
            info.unit = "A";
        }
        result.signal_info.push_back(info);
    }

    // Get initial state
    Vector x;
    if (options_.use_ic) {
        x = Vector::Zero(n);
    } else {
        auto dc_result = dc_operating_point();
        if (dc_result.status != SolverStatus::Success) {
            result.final_status = dc_result.status;
            result.error_message = "DC operating point failed: " + dc_result.error_message;
            return result;
        }
        x = dc_result.x;
        result.newton_iterations_total += dc_result.iterations;
    }

    // Initialize switch states
    assembler_.update_switch_states(x, options_.tstart);

    // Store initial state
    Real time = options_.tstart;
    result.time.push_back(time);
    result.data.push_back(x);

    if (callback) {
        callback(time, x);
    }

    // Estimate total steps
    Real total_sim_time = options_.tstop - options_.tstart;
    int64_t estimated_steps = static_cast<int64_t>(total_sim_time / options_.dt);

    // Time stepping loop
    Real dt = options_.dt;
    int step_count = 0;
    int steps_since_progress = 0;

    while (time < options_.tstop) {
        // Handle pause/stop via control
        if (control) {
            if (control->should_stop()) {
                result.final_status = SolverStatus::Success;
                result.error_message = "Simulation stopped by user";
                break;
            }
            while (control->should_pause() && !control->should_stop()) {
                control->wait_until_resumed();
            }
            if (control->should_stop()) {
                result.final_status = SolverStatus::Success;
                result.error_message = "Simulation stopped by user";
                break;
            }
        }

        // Don't overshoot tstop
        if (time + dt > options_.tstop) {
            dt = options_.tstop - time;
        }

        Real next_time = time + dt;
        auto step_result = step(next_time, dt, x);

        if (step_result.status != SolverStatus::Success) {
            // Try with smaller timestep
            bool converged = false;
            Real dt_try = dt * 0.5;
            timestep_reductions++;

            while (dt_try >= options_.dtmin && !converged) {
                next_time = time + dt_try;
                step_result = step(next_time, dt_try, x);

                if (step_result.status == SolverStatus::Success) {
                    converged = true;
                    dt = dt_try;
                } else {
                    dt_try *= 0.5;
                    timestep_reductions++;
                }
            }

            if (!converged) {
                result.final_status = step_result.status;
                result.error_message = "Simulation failed at t=" + std::to_string(time) +
                                      ": " + step_result.error_message;
                break;
            }
        }

        result.newton_iterations_total += step_result.iterations;

        // Track convergence issues
        if (step_result.iterations > 10) {
            convergence_failures++;
        }

        // Check for switch events
        if (assembler_.check_switch_events(step_result.x)) {
            Real t_event;
            Vector x_event;
            if (find_event_time(time, next_time, x, t_event, x_event)) {
                for (const auto& comp : circuit_.components()) {
                    if (comp.type() != ComponentType::Switch) continue;

                    const SwitchState* state = assembler_.find_switch_state(comp.name());
                    if (!state) continue;

                    const auto& params = std::get<SwitchParams>(comp.params());
                    Index n_ctrl_pos = circuit_.node_index(comp.nodes()[2]);
                    Index n_ctrl_neg = circuit_.node_index(comp.nodes()[3]);
                    Real v_ctrl = 0.0;
                    if (n_ctrl_pos >= 0) v_ctrl += x_event(n_ctrl_pos);
                    if (n_ctrl_neg >= 0) v_ctrl -= x_event(n_ctrl_neg);

                    bool would_close = v_ctrl > params.vth;
                    if (would_close != state->is_closed) {
                        Index n1 = circuit_.node_index(comp.nodes()[0]);
                        Index n2 = circuit_.node_index(comp.nodes()[1]);
                        Real v_switch = 0.0;
                        if (n1 >= 0) v_switch += x_event(n1);
                        if (n2 >= 0) v_switch -= x_event(n2);
                        Real R = state->is_closed ? params.ron : params.roff;
                        Real i_switch = v_switch / R;

                        calculate_switching_loss(comp, *state, v_switch, i_switch, would_close);

                        // Record event in result
                        SimulationEvent sim_event;
                        sim_event.time = t_event;
                        sim_event.type = would_close ? SimulationEventType::SwitchClose
                                                     : SimulationEventType::SwitchOpen;
                        sim_event.component = comp.name();
                        sim_event.description = comp.name() + (would_close ? " closed" : " opened");
                        sim_event.value1 = v_switch;
                        sim_event.value2 = i_switch;
                        result.events.push_back(sim_event);

                        if (event_callback) {
                            SwitchEvent event;
                            event.switch_name = comp.name();
                            event.time = t_event;
                            event.new_state = would_close;
                            event.voltage = v_switch;
                            event.current = i_switch;
                            event_callback(event);
                        }
                    }
                }

                assembler_.update_switch_states(x_event, t_event);
                Real dt_remaining = next_time - t_event;
                if (dt_remaining > options_.dtmin) {
                    step_result = step(next_time, dt_remaining, x_event);
                } else {
                    step_result.x = x_event;
                }
            } else {
                assembler_.update_switch_states(step_result.x, next_time);
            }
        } else {
            assembler_.update_switch_states(step_result.x, next_time);
        }

        accumulate_conduction_losses(step_result.x, dt);
        update_diode_states(step_result.x, x, dt);

        x = step_result.x;
        time = next_time;
        step_count++;
        steps_since_progress++;

        // Decimation and rolling buffer storage
        bool should_store = (options_.streaming_decimation <= 1) ||
                           (step_count % options_.streaming_decimation == 0);

        if (should_store) {
            // Rolling buffer: remove oldest point if at capacity
            if (options_.streaming_rolling_buffer &&
                static_cast<int64_t>(result.time.size()) >= options_.streaming_max_points) {
                result.time.erase(result.time.begin());
                result.data.erase(result.data.begin());
            }
            result.time.push_back(time);
            result.data.push_back(x);
        }

        // Callback is separate from storage (invoked every step for real-time updates)
        if (callback) {
            callback(time, x);
        }

        // Progress callback with throttling
        if (progress_config.callback) {
            auto now = std::chrono::high_resolution_clock::now();
            double ms_since_last = std::chrono::duration<double, std::milli>(now - last_progress_time).count();

            if (ms_since_last >= progress_config.min_interval_ms ||
                steps_since_progress >= progress_config.min_steps) {

                SimulationProgress progress;
                progress.current_time = time;
                progress.total_time = options_.tstop;
                progress.progress_percent = 100.0 * (time - options_.tstart) / total_sim_time;
                progress.steps_completed = step_count;
                progress.total_steps_estimate = estimated_steps;
                progress.newton_iterations = step_result.iterations;
                progress.convergence_warning = (step_result.iterations > 10);
                progress.elapsed_seconds = std::chrono::duration<double>(now - start_time).count();

                // Estimate remaining time
                if (progress.progress_percent > 0.1) {
                    double rate = progress.elapsed_seconds / (progress.progress_percent / 100.0);
                    progress.estimated_remaining_seconds = rate * (1.0 - progress.progress_percent / 100.0);
                }

                progress_config.callback(progress);
                last_progress_time = now;
                steps_since_progress = 0;
            }
        }

        // Adaptive timestep
        if (step_result.iterations < 5 && dt < options_.dtmax) {
            dt = std::min(dt * 1.2, options_.dtmax);
        }
    }

    result.total_steps = step_count;
    result.final_status = SolverStatus::Success;
    result.convergence_failures = convergence_failures;
    result.timestep_reductions = timestep_reductions;

    auto end_time = std::chrono::high_resolution_clock::now();
    result.total_time_seconds = std::chrono::duration<double>(end_time - start_time).count();

    if (step_count > 0) {
        result.average_newton_iterations = static_cast<double>(result.newton_iterations_total) / step_count;
    }

    result.solver_info.method = options_.integration_method;
    result.solver_info.abstol = options_.abstol;
    result.solver_info.reltol = options_.reltol;
    result.solver_info.adaptive_timestep = options_.adaptive_timestep;

    return result;
}

bool Simulator::find_event_time(Real t_start, Real t_end, const Vector& x_start,
                                Real& t_event, Vector& x_event) {
    // Bisection to find event time
    const int max_bisections = 10;
    const Real tol = options_.dtmin;

    Real t_lo = t_start;
    Real t_hi = t_end;
    Vector x_lo = x_start;
    Vector x_hi;

    // Initial step to t_hi
    auto result_hi = step(t_hi, t_hi - t_lo, x_lo);
    if (result_hi.status != SolverStatus::Success) {
        return false;
    }
    x_hi = result_hi.x;

    for (int i = 0; i < max_bisections && (t_hi - t_lo) > tol; ++i) {
        Real t_mid = 0.5 * (t_lo + t_hi);
        auto result_mid = step(t_mid, t_mid - t_lo, x_lo);
        if (result_mid.status != SolverStatus::Success) {
            // Can't converge at mid, try closer to t_lo
            t_hi = t_mid;
            continue;
        }

        if (assembler_.check_switch_events(result_mid.x)) {
            // Event is between t_lo and t_mid
            t_hi = t_mid;
            x_hi = result_mid.x;
        } else {
            // Event is between t_mid and t_hi
            t_lo = t_mid;
            x_lo = result_mid.x;
        }
    }

    t_event = t_hi;
    x_event = x_hi;
    return true;
}

Real Simulator::calculate_switching_loss(const Component& comp, const SwitchState& /*state*/,
                                         Real voltage, Real current, bool turning_on) {
    // Check if we have a lookup table for this device
    auto table_it = switching_loss_tables_.find(comp.name());
    if (table_it != switching_loss_tables_.end()) {
        const auto& table = table_it->second;
        Real abs_voltage = std::abs(voltage);
        Real abs_current = std::abs(current);

        Real loss = 0.0;
        if (turning_on && table.has_eon()) {
            loss = table.interpolate_eon(abs_voltage, abs_current);
            power_losses_.turn_on_loss += loss;
        } else if (!turning_on && table.has_eoff()) {
            loss = table.interpolate_eoff(abs_voltage, abs_current);
            power_losses_.turn_off_loss += loss;
        }

        // Track per-device loss
        power_losses_.device_switching_loss[comp.name()] += loss;
        return loss;
    }

    // Fallback to simple model if no lookup table
    const auto& params = std::get<SwitchParams>(comp.params());

    // Simple switching loss model: E_sw = 0.5 * V * I * t_sw
    // where t_sw is the switching time (approximated by Ron for now)
    // This is a simplified model - real losses depend on switching waveforms

    Real t_sw = params.ron * 1e-6;  // Rough approximation of switching time
    Real loss = 0.5 * std::abs(voltage * current) * t_sw;

    if (turning_on) {
        power_losses_.turn_on_loss += loss;
    } else {
        power_losses_.turn_off_loss += loss;
    }

    power_losses_.device_switching_loss[comp.name()] += loss;
    return loss;
}

void Simulator::set_switching_loss_table(const std::string& device_name, const SwitchingLossTable& table) {
    switching_loss_tables_[device_name] = table;
}

Real Simulator::calculate_reverse_recovery_loss(const Component& diode_comp, Real current, Real di_dt) {
    // Reverse recovery loss occurs when a diode switches from conducting to blocking
    // The stored charge (Qrr) must be removed, causing a reverse current spike
    //
    // Simple model: Err = 0.5 * Qrr * Vr
    // where Qrr = Irr * trr / 2 (triangular approximation)
    // and Irr is peak reverse current, trr is reverse recovery time

    const auto& params = std::get<DiodeParams>(diode_comp.params());

    // Check if we have a lookup table for this diode
    auto table_it = switching_loss_tables_.find(diode_comp.name());
    if (table_it != switching_loss_tables_.end() && table_it->second.has_err()) {
        // Use lookup table
        Real abs_current = std::abs(current);
        // For reverse recovery, voltage is typically the blocking voltage
        // We use the breakdown voltage as a proxy for the blocking voltage
        Real abs_voltage = params.bv;
        Real loss = table_it->second.interpolate_err(abs_voltage, abs_current);
        power_losses_.reverse_recovery_loss += loss;
        power_losses_.device_switching_loss[diode_comp.name()] += loss;
        return loss;
    }

    // Simplified model based on transit time (tt) which is related to stored charge
    // Qrr ≈ I_f * tt where I_f is forward current before turn-off
    // Irr ≈ sqrt(2 * Qrr * di/dt)
    // trr ≈ sqrt(2 * Qrr / (di/dt))
    // Err ≈ 0.5 * Qrr * Vr ≈ 0.5 * I_f * tt * Vr

    if (params.tt <= 0.0) {
        // No transit time specified, assume negligible reverse recovery
        return 0.0;
    }

    Real Qrr = std::abs(current) * params.tt;
    Real Vr = params.bv;  // Use breakdown voltage as blocking voltage estimate

    // More accurate Err model considering di/dt
    // Err = Qrr * Vr * factor
    // where factor depends on snappiness of diode (0.25 to 0.5 typical)
    Real snappiness_factor = 0.35;  // Moderate snappiness

    Real loss = Qrr * Vr * snappiness_factor;

    // Scale by di/dt effect (faster switching = more loss)
    if (di_dt > 0) {
        Real di_dt_ref = 100e6;  // Reference di/dt of 100 A/µs
        Real di_dt_factor = std::sqrt(std::abs(di_dt) / di_dt_ref);
        di_dt_factor = std::min(di_dt_factor, 2.0);  // Cap at 2x
        loss *= di_dt_factor;
    }

    power_losses_.reverse_recovery_loss += loss;
    power_losses_.device_switching_loss[diode_comp.name()] += loss;
    return loss;
}

void Simulator::update_diode_states(const Vector& x, const Vector& /*x_prev*/, Real dt) {
    for (const auto& comp : circuit_.components()) {
        if (comp.type() != ComponentType::Diode) continue;

        Index n_anode = circuit_.node_index(comp.nodes()[0]);
        Index n_cathode = circuit_.node_index(comp.nodes()[1]);

        // Calculate diode current (approximate from voltage and conductance)
        Real Vd = 0.0;
        if (n_anode >= 0) Vd += x(n_anode);
        if (n_cathode >= 0) Vd -= x(n_cathode);

        // Note: x_prev voltages could be used for more accurate di/dt calculation
        // but are currently computed from current values

        const auto& params = std::get<DiodeParams>(comp.params());

        // Estimate current using diode equation
        Real Id = 0.0;
        if (params.ideal) {
            constexpr Real Gon = 1e3;
            constexpr Real Goff = 1e-9;
            Id = (Vd > 0) ? Gon * Vd : Goff * Vd;
        } else {
            Real Vt = params.vt;
            Real Is = params.is;
            Real n = params.n;
            Real Vd_limited = std::min(Vd, 40.0 * n * Vt);
            Id = Is * (std::exp(Vd_limited / (n * Vt)) - 1.0);
        }

        // Get or create diode state
        auto& state = diode_states_[comp.name()];
        bool is_conducting = (Id > 1e-6);  // Threshold for "conducting"

        // Detect reverse recovery event: was conducting, now blocking
        if (state.was_conducting && !is_conducting && state.prev_current > 1e-6) {
            // Calculate di/dt
            Real di_dt = (Id - state.prev_current) / dt;

            // Calculate reverse recovery loss
            calculate_reverse_recovery_loss(comp, state.prev_current, di_dt);
        }

        // Update state
        state.prev_current = Id;
        state.was_conducting = is_conducting;
    }
}

void Simulator::accumulate_conduction_losses(const Vector& x, Real dt) {
    for (const auto& comp : circuit_.components()) {
        Real p_cond = 0.0;

        if (comp.type() == ComponentType::Switch) {
            const SwitchState* state = assembler_.find_switch_state(comp.name());
            if (!state || !state->is_closed) continue;  // Only closed switches have conduction loss

            const auto& params = std::get<SwitchParams>(comp.params());

            // Get switch voltage
            Index n1 = circuit_.node_index(comp.nodes()[0]);
            Index n2 = circuit_.node_index(comp.nodes()[1]);
            Real v_switch = 0.0;
            if (n1 >= 0) v_switch += x(n1);
            if (n2 >= 0) v_switch -= x(n2);

            // Current through switch
            Real i_switch = v_switch / params.ron;

            // Conduction loss: P = I^2 * Ron
            p_cond = i_switch * i_switch * params.ron;
        }
        else if (comp.type() == ComponentType::Diode) {
            const auto& params = std::get<DiodeParams>(comp.params());

            Index n_anode = circuit_.node_index(comp.nodes()[0]);
            Index n_cathode = circuit_.node_index(comp.nodes()[1]);

            Real Vd = 0.0;
            if (n_anode >= 0) Vd += x(n_anode);
            if (n_cathode >= 0) Vd -= x(n_cathode);

            // Only accumulate loss when conducting (Vd > 0)
            if (Vd > 0) {
                Real Id = 0.0;
                if (params.ideal) {
                    constexpr Real Gon = 1e3;
                    Id = Gon * Vd;
                } else {
                    Real Vd_limited = std::min(Vd, 40.0 * params.n * params.vt);
                    Id = params.is * (std::exp(Vd_limited / (params.n * params.vt)) - 1.0);
                }
                // Conduction loss: P = Vd * Id
                p_cond = Vd * Id;
            }
        }
        else if (comp.type() == ComponentType::MOSFET) {
            const auto& params = std::get<MOSFETParams>(comp.params());

            Index n_drain = circuit_.node_index(comp.nodes()[0]);
            Index n_gate = circuit_.node_index(comp.nodes()[1]);
            Index n_source = circuit_.node_index(comp.nodes()[2]);

            Real Vd = (n_drain >= 0) ? x(n_drain) : 0.0;
            Real Vg = (n_gate >= 0) ? x(n_gate) : 0.0;
            Real Vs = (n_source >= 0) ? x(n_source) : 0.0;

            Real Vgs = Vg - Vs;
            Real Vds = Vd - Vs;

            // Handle PMOS
            Real sign = (params.type == MOSFETType::NMOS) ? 1.0 : -1.0;
            Vgs *= sign;
            Vds *= sign;

            // Only ON when Vgs > Vth
            if (Vgs > params.vth) {
                Real Id = 0.0;
                if (params.rds_on > 0) {
                    // Simple switch model
                    Id = Vds / params.rds_on;
                    p_cond = Id * Id * params.rds_on;
                } else {
                    // Level 1 model
                    Real Kp = params.kp_effective();
                    Real Vov = Vgs - params.vth;
                    if (Vds < Vov) {
                        // Linear region
                        Id = Kp * (Vov * Vds - 0.5 * Vds * Vds);
                    } else {
                        // Saturation
                        Id = 0.5 * Kp * Vov * Vov * (1.0 + params.lambda * Vds);
                    }
                    p_cond = std::abs(Vds * Id);
                }
            }
        }
        else if (comp.type() == ComponentType::IGBT) {
            const auto& params = std::get<IGBTParams>(comp.params());

            Index n_collector = circuit_.node_index(comp.nodes()[0]);
            Index n_gate = circuit_.node_index(comp.nodes()[1]);
            Index n_emitter = circuit_.node_index(comp.nodes()[2]);

            Real Vc = (n_collector >= 0) ? x(n_collector) : 0.0;
            Real Vg = (n_gate >= 0) ? x(n_gate) : 0.0;
            Real Ve = (n_emitter >= 0) ? x(n_emitter) : 0.0;

            Real Vge = Vg - Ve;
            Real Vce = Vc - Ve;

            // Only ON when Vge > Vth and Vce > 0
            if (Vge > params.vth && Vce > 0) {
                Real Ic = 0.0;
                if (Vce > params.vce_sat) {
                    Ic = (Vce - params.vce_sat) / params.rce_on;
                } else {
                    Real alpha = Vce / params.vce_sat;
                    Ic = alpha * Vce / params.rce_on;
                }
                // IGBT loss includes Vce_sat drop + I²R
                p_cond = params.vce_sat * Ic + Ic * Ic * params.rce_on;
            }
        }
        else {
            continue;  // Skip other component types
        }

        // Accumulate energy loss
        Real energy = p_cond * dt;
        power_losses_.conduction_loss += energy;
        power_losses_.device_conduction_loss[comp.name()] += energy;
    }
}

EfficiencyResult Simulator::calculate_efficiency(const SimulationResult& result,
                                                  const std::vector<std::string>& load_nodes,
                                                  const std::vector<std::string>& source_names) const {
    EfficiencyResult eff;

    if (result.time.size() < 2) {
        return eff;
    }

    Real total_time = result.time.back() - result.time.front();
    if (total_time <= 0) {
        return eff;
    }

    // Get node indices for loads
    std::vector<Index> load_indices;
    for (const auto& node_name : load_nodes) {
        try {
            Index idx = circuit_.node_index(node_name);
            if (idx >= 0) {
                load_indices.push_back(idx);
            }
        } catch (...) {
            // Node not found, skip
        }
    }

    // Get branch indices for sources
    std::vector<std::pair<Index, Index>> source_info;  // (node_pos_idx, branch_idx)
    for (const auto& source_name : source_names) {
        const Component* comp = circuit_.find_component(source_name);
        if (comp && comp->type() == ComponentType::VoltageSource) {
            Index n_pos = circuit_.node_index(comp->nodes()[0]);
            // Find branch index for this source
            Index branch_idx = circuit_.node_count();
            for (Index i = 0; i < static_cast<Index>(circuit_.components().size()); ++i) {
                const auto& c = circuit_.components()[i];
                if (c.name() == source_name) {
                    break;
                }
                if (c.type() == ComponentType::VoltageSource || c.type() == ComponentType::Inductor) {
                    branch_idx++;
                }
            }
            source_info.push_back({n_pos, branch_idx});
        }
    }

    // Integrate power over time
    Real input_energy = 0.0;

    for (size_t i = 1; i < result.time.size(); ++i) {
        Real dt = result.time[i] - result.time[i - 1];
        const Vector& x = result.data[i];

        // Input power from sources: P = V * I
        for (const auto& [n_pos, branch_idx] : source_info) {
            Real V = 0.0;
            if (n_pos >= 0 && n_pos < x.size()) {
                V = x(n_pos);
            }
            Real I = 0.0;
            if (branch_idx < x.size()) {
                I = x(branch_idx);  // Current flowing out of positive terminal
            }
            // Power delivered by source (positive when sourcing)
            Real P_in = V * (-I);  // Negative because current convention
            if (P_in > 0) {
                input_energy += P_in * dt;
            }
        }

        // Output power at load nodes
        // This is a simplified calculation - assumes loads are resistors to ground
        // For more accurate calculation, user should specify load components
        for (Index load_idx : load_indices) {
            if (load_idx >= 0 && load_idx < x.size()) {
                Real V_load = x(load_idx);
                // Estimate power - this is application-specific
                // For now, we skip this since it requires knowing load impedance
                (void)V_load;
            }
        }
    }

    // Use loss-based efficiency calculation
    // Efficiency = (Input - Losses) / Input
    eff.input_energy = input_energy;
    eff.loss_energy = power_losses_.total_loss();
    eff.output_energy = input_energy - eff.loss_energy;

    if (input_energy > 0) {
        eff.efficiency = eff.output_energy / input_energy;
        eff.efficiency = std::max(0.0, std::min(1.0, eff.efficiency));  // Clamp to [0, 1]
    }

    eff.average_input_power = input_energy / total_time;
    eff.average_output_power = eff.output_energy / total_time;
    eff.average_loss_power = eff.loss_energy / total_time;

    return eff;
}

SimulationResult simulate(const Circuit& circuit, const SimulationOptions& options) {
    Simulator sim(circuit, options);
    return sim.run_transient();
}

}  // namespace pulsim
