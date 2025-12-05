#include "spicelab/simulation.hpp"
#include <chrono>
#include <iostream>

namespace spicelab {

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

                        // Calculate switching loss
                        Real sw_loss = calculate_switching_loss(comp, *state, v_switch, i_switch, would_close);
                        power_losses_.switching_loss += sw_loss;

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

        // Update state
        x = step_result.x;
        time = next_time;
        step_count++;

        // Store result
        result.time.push_back(time);
        result.data.push_back(x);

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
    const auto& params = std::get<SwitchParams>(comp.params());

    // Simple switching loss model: E_sw = 0.5 * V * I * t_sw
    // where t_sw is the switching time (approximated by Ron for now)
    // This is a simplified model - real losses depend on switching waveforms

    Real t_sw = params.ron * 1e-6;  // Rough approximation of switching time
    if (turning_on) {
        // Turn-on loss: current rises while voltage falls
        return 0.5 * std::abs(voltage * current) * t_sw;
    } else {
        // Turn-off loss: voltage rises while current falls
        return 0.5 * std::abs(voltage * current) * t_sw;
    }
}

void Simulator::accumulate_conduction_losses(const Vector& x, Real dt) {
    for (const auto& comp : circuit_.components()) {
        if (comp.type() != ComponentType::Switch) continue;

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

        // Conduction loss: P = I^2 * Ron, Energy = P * dt
        Real p_cond = i_switch * i_switch * params.ron;
        power_losses_.conduction_loss += p_cond * dt;
    }
}

SimulationResult simulate(const Circuit& circuit, const SimulationOptions& options) {
    Simulator sim(circuit, options);
    return sim.run_transient();
}

}  // namespace spicelab
