#include "pulsim/v1/simulation.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>

namespace pulsim::v1 {

namespace {
constexpr int kMaxBisections = 12;
}

Simulator::Simulator(Circuit& circuit, const SimulationOptions& options)
    : circuit_(circuit)
    , options_(options)
    , newton_solver_(options_.newton_options)
    , timestep_controller_(options_.timestep_config)
    , lte_estimator_(options_.lte_config)
    , bdf_controller_(options_.bdf_config) {

    options_.newton_options.num_nodes = circuit_.num_nodes();
    options_.newton_options.num_branches = circuit_.num_branches();
    newton_solver_.set_options(options_.newton_options);
    newton_solver_.linear_solver().set_config(options_.linear_solver);

    // Ensure timestep controller limits align with simulation options
    auto cfg = options_.timestep_config;
    cfg.dt_min = options_.dt_min;
    cfg.dt_max = options_.dt_max;
    if (cfg.dt_initial <= 0) {
        cfg.dt_initial = options_.dt;
    }
    timestep_controller_ = AdvancedTimestepController(cfg);

    // Build device index map and switch monitors
    const auto& devices = circuit_.devices();
    const auto& conns = circuit_.connections();
    device_index_.clear();
    switch_monitors_.clear();

    for (std::size_t i = 0; i < devices.size(); ++i) {
        const auto& conn = conns[i];
        device_index_[conn.name] = i;

        if (const auto* sw = std::get_if<VoltageControlledSwitch>(&devices[i])) {
            if (conn.nodes.size() >= 3) {
                SwitchMonitor monitor;
                monitor.name = conn.name;
                monitor.ctrl = conn.nodes[0];
                monitor.t1 = conn.nodes[1];
                monitor.t2 = conn.nodes[2];
                monitor.v_threshold = sw->v_threshold();
                monitor.was_on = false;
                switch_monitors_.push_back(monitor);
            }
        }
    }

    initialize_loss_tracking();

    for (const auto& [name, energy] : options_.switching_energy) {
        set_switching_energy(name, energy);
    }
}

void Simulator::initialize_loss_tracking() {
    const auto& devices = circuit_.devices();
    loss_states_.assign(devices.size(), DeviceLossState{});
    switching_energy_.assign(devices.size(), std::nullopt);
    diode_conducting_.assign(devices.size(), false);
}

void Simulator::set_switching_energy(const std::string& device_name, const SwitchingEnergy& energy) {
    auto it = device_index_.find(device_name);
    if (it == device_index_.end()) return;
    switching_energy_[it->second] = energy;
}

DCAnalysisResult Simulator::dc_operating_point() {
    // Large timestep to emulate DC for dynamic elements
    circuit_.set_timestep(1e6);
    circuit_.set_integration_order(1);

    auto system_func = [this](const Vector& x, Vector& f, SparseMatrix& J) {
        circuit_.assemble_jacobian(J, f, x);
    };

    Vector x0 = Vector::Zero(circuit_.system_size());

    DCConvergenceSolver<RuntimeLinearSolver> solver(options_.dc_config);
    solver.set_linear_solver_config(options_.linear_solver);
    return solver.solve(x0, circuit_.num_nodes(), circuit_.num_branches(), system_func, nullptr);
}

NewtonResult Simulator::solve_step(Real t_next, Real dt, const Vector& x_prev) {
    circuit_.set_current_time(t_next);
    circuit_.set_timestep(dt);
    if (options_.enable_bdf_order_control) {
        circuit_.set_integration_order(std::clamp(bdf_controller_.current_order(), 1, 2));
    } else {
        circuit_.set_integration_order(2);
    }

    auto system_func = [this](const Vector& x, Vector& f, SparseMatrix& J) {
        circuit_.assemble_jacobian(J, f, x);
    };

    return newton_solver_.solve(x_prev, system_func);
}

bool Simulator::find_switch_event_time(const SwitchMonitor& sw,
                                       Real t_start, Real t_end,
                                       const Vector& x_start,
                                       Real& t_event, Vector& x_event) {
    if (t_end <= t_start) return false;

    Real t_lo = t_start;
    Real t_hi = t_end;
    Vector x_lo = x_start;
    Vector x_hi;

    // Initial solve at t_hi
    auto result_hi = solve_step(t_hi, t_hi - t_lo, x_lo);
    if (result_hi.status != SolverStatus::Success) {
        return false;
    }
    x_hi = result_hi.solution;

    auto ctrl_value = [&](const Vector& x) -> Real {
        return (sw.ctrl >= 0) ? x[sw.ctrl] : 0.0;
    };

    Real v_lo = ctrl_value(x_lo) - sw.v_threshold;

    if (v_lo == 0.0) {
        t_event = t_lo;
        x_event = x_lo;
        return true;
    }

    for (int i = 0; i < kMaxBisections && (t_hi - t_lo) > options_.dt_min; ++i) {
        Real t_mid = 0.5 * (t_lo + t_hi);
        auto result_mid = solve_step(t_mid, t_mid - t_lo, x_lo);
        if (result_mid.status != SolverStatus::Success) {
            t_hi = t_mid;
            continue;
        }

        Vector x_mid = result_mid.solution;
        Real v_mid = ctrl_value(x_mid) - sw.v_threshold;

        if ((v_lo > 0 && v_mid > 0) || (v_lo < 0 && v_mid < 0)) {
            t_lo = t_mid;
            x_lo = x_mid;
            v_lo = v_mid;
        } else {
            t_hi = t_mid;
            x_hi = x_mid;
        }
    }

    t_event = t_hi;
    x_event = x_hi;
    return true;
}

void Simulator::record_switch_event(const SwitchMonitor& sw, Real time,
                                    const Vector& x_state, bool new_state,
                                    SimulationResult& result, EventCallback event_callback) {
    // Compute switch voltage and current
    Real v_switch = 0.0;
    if (sw.t1 >= 0) v_switch += x_state[sw.t1];
    if (sw.t2 >= 0) v_switch -= x_state[sw.t2];

    // Use discrete g_on/g_off based on state for loss estimates
    Real g_on = 1e3;
    Real g_off = 1e-9;

    const auto& devices = circuit_.devices();
    auto idx_it = device_index_.find(sw.name);
    if (idx_it != device_index_.end()) {
        const auto* dev = std::get_if<VoltageControlledSwitch>(&devices[idx_it->second]);
        if (dev) {
            g_on = dev->g_on();
            g_off = dev->g_off();
        }
    }

    Real g = new_state ? g_on : g_off;
    Real i_switch = g * v_switch;

    SimulationEvent evt;
    evt.time = time;
    evt.type = new_state ? SimulationEventType::SwitchOn : SimulationEventType::SwitchOff;
    evt.component = sw.name;
    evt.description = sw.name + (new_state ? " on" : " off");
    evt.value1 = v_switch;
    evt.value2 = i_switch;
    result.events.push_back(evt);

    if (event_callback) {
        SwitchEvent se;
        se.switch_name = sw.name;
        se.time = time;
        se.new_state = new_state;
        se.voltage = v_switch;
        se.current = i_switch;
        event_callback(se);
    }

    // Accumulate switching energy if configured
    auto it = device_index_.find(sw.name);
    if (it != device_index_.end()) {
        const auto& energy_opt = switching_energy_[it->second];
        if (energy_opt) {
            Real e = new_state ? energy_opt->eon : energy_opt->eoff;
            accumulate_switching_loss(sw.name, new_state, e);
        }
    }
}

void Simulator::accumulate_switching_loss(const std::string& name, bool turning_on, Real energy) {
    if (energy <= 0.0) return;
    auto it = device_index_.find(name);
    if (it == device_index_.end()) return;

    auto& state = loss_states_[it->second];
    state.accumulator.add_switching_event(energy);
    if (turning_on) {
        state.switching_energy.turn_on += energy;
    } else {
        state.switching_energy.turn_off += energy;
    }
}

void Simulator::accumulate_reverse_recovery_loss(const std::string& name, Real energy) {
    if (energy <= 0.0) return;
    auto it = device_index_.find(name);
    if (it == device_index_.end()) return;

    auto& state = loss_states_[it->second];
    state.accumulator.add_switching_event(energy);
    state.switching_energy.reverse_recovery += energy;
}

void Simulator::accumulate_conduction_losses(const Vector& x, Real dt) {
    if (!options_.enable_losses) return;

    const auto& devices = circuit_.devices();
    const auto& conns = circuit_.connections();

    for (std::size_t i = 0; i < devices.size(); ++i) {
        const auto& conn = conns[i];
        Real p_cond = 0.0;

        std::visit([&](const auto& dev) {
            using T = std::decay_t<decltype(dev)>;

            auto node_voltage = [&](Index n) -> Real {
                return (n >= 0) ? x[n] : 0.0;
            };

            if constexpr (std::is_same_v<T, Resistor>) {
                Real v = node_voltage(conn.nodes[0]) - node_voltage(conn.nodes[1]);
                p_cond = (v * v) / dev.resistance();
            }
            else if constexpr (std::is_same_v<T, IdealSwitch>) {
                Real g = dev.is_closed() ? dev.g_on() : dev.g_off();
                Real v = node_voltage(conn.nodes[0]) - node_voltage(conn.nodes[1]);
                Real i = g * v;
                p_cond = std::abs(v * i);
            }
            else if constexpr (std::is_same_v<T, VoltageControlledSwitch>) {
                Real v_ctrl = node_voltage(conn.nodes[0]);
                bool on = v_ctrl > dev.v_threshold();
                Real g = on ? dev.g_on() : dev.g_off();
                Real v = node_voltage(conn.nodes[1]) - node_voltage(conn.nodes[2]);
                Real i = g * v;
                p_cond = std::abs(v * i);
            }
            else if constexpr (std::is_same_v<T, IdealDiode>) {
                Real v = node_voltage(conn.nodes[0]) - node_voltage(conn.nodes[1]);
                Real g = dev.is_conducting() ? dev.g_on() : dev.g_off();
                Real i = g * v;
                p_cond = std::max<Real>(0.0, v * i);

                bool conducting = dev.is_conducting();
                if (diode_conducting_[i] && !conducting) {
                    const auto& energy_opt = switching_energy_[i];
                    if (energy_opt && energy_opt->err > 0.0) {
                        accumulate_reverse_recovery_loss(conn.name, energy_opt->err);
                    }
                }
                diode_conducting_[i] = conducting;
            }
            else if constexpr (std::is_same_v<T, MOSFET>) {
                Real vg = node_voltage(conn.nodes[0]);
                Real vd = node_voltage(conn.nodes[1]);
                Real vs = node_voltage(conn.nodes[2]);
                auto params = dev.params();

                Real sign = params.is_nmos ? 1.0 : -1.0;
                Real vgs = sign * (vg - vs);
                Real vds = sign * (vd - vs);

                Real id = 0.0;
                if (vgs <= params.vth) {
                    id = params.g_off * vds;
                } else if (vds < vgs - params.vth) {
                    Real vov = vgs - params.vth;
                    id = params.kp * (vov * vds - 0.5 * vds * vds) * (1.0 + params.lambda * vds);
                } else {
                    Real vov = vgs - params.vth;
                    id = 0.5 * params.kp * vov * vov * (1.0 + params.lambda * vds);
                }

                id *= sign;
                p_cond = std::abs((vd - vs) * id);
            }
            else if constexpr (std::is_same_v<T, IGBT>) {
                Real vg = node_voltage(conn.nodes[0]);
                Real vc = node_voltage(conn.nodes[1]);
                Real ve = node_voltage(conn.nodes[2]);
                auto params = dev.params();

                Real vge = vg - ve;
                Real vce = vc - ve;
                bool on = (vge > params.vth) && (vce > 0);
                Real g = on ? params.g_on : params.g_off;
                Real i = g * vce;
                p_cond = std::abs(vce * i);
            }
        }, devices[i]);

        if (p_cond > 0.0) {
            auto& state = loss_states_[i];
            state.accumulator.add_sample(p_cond, dt);
            state.peak_power = std::max(state.peak_power, p_cond);
        }
    }
}

void Simulator::finalize_loss_summary(SimulationResult& result) {
    if (!options_.enable_losses) return;

    SystemLossSummary summary;
    const auto& conns = circuit_.connections();

    Real duration = 0.0;
    if (result.time.size() >= 2) {
        duration = result.time.back() - result.time.front();
    }

    for (std::size_t i = 0; i < loss_states_.size(); ++i) {
        const auto& state = loss_states_[i];
        if (state.accumulator.num_samples() == 0 &&
            state.accumulator.switching_energy() == 0.0) {
            continue;
        }

        LossResult res;
        res.device_name = conns[i].name;
        res.total_energy = state.accumulator.total_energy();
        res.average_power = duration > 0 ? res.total_energy / duration : 0.0;
        res.peak_power = state.peak_power;

        Real conduction_energy = state.accumulator.conduction_energy();
        Real switching_energy = state.accumulator.switching_energy();

        res.breakdown.conduction = duration > 0 ? conduction_energy / duration : 0.0;
        res.breakdown.turn_on = duration > 0 ? state.switching_energy.turn_on / duration : 0.0;
        res.breakdown.turn_off = duration > 0 ? state.switching_energy.turn_off / duration : 0.0;
        res.breakdown.reverse_recovery = duration > 0 ? state.switching_energy.reverse_recovery / duration : 0.0;

        if (res.breakdown.turn_on == 0.0 &&
            res.breakdown.turn_off == 0.0 &&
            res.breakdown.reverse_recovery == 0.0) {
            res.breakdown.turn_on = duration > 0 ? switching_energy / duration : 0.0;
        }

        summary.device_losses.push_back(res);
    }

    summary.compute_totals();
    result.loss_summary = summary;
}

SimulationResult Simulator::run_transient(SimulationCallback callback,
                                          EventCallback event_callback,
                                          SimulationControl* control) {
    auto dc = dc_operating_point();
    if (!dc.success) {
        SimulationResult result;
        result.success = false;
        result.final_status = dc.newton_result.status;
        result.message = "DC operating point failed: " + dc.message;
        result.linear_solver_telemetry = dc.linear_solver_telemetry;
        return result;
    }

    return run_transient(dc.newton_result.solution, callback, event_callback, control);
}

SimulationResult Simulator::run_transient(const Vector& x0,
                                          SimulationCallback callback,
                                          EventCallback event_callback,
                                          SimulationControl* control) {
    SimulationResult result;
    auto start_time = std::chrono::high_resolution_clock::now();

    initialize_loss_tracking();
    for (const auto& [name, energy] : options_.switching_energy) {
        set_switching_energy(name, energy);
    }
    lte_estimator_.reset();
    if (options_.enable_bdf_order_control) {
        bdf_controller_.set_order(std::clamp(options_.bdf_config.initial_order, 1, 2));
    }

    Real t = options_.tstart;
    Real dt = options_.dt;
    Vector x = x0;

    int rejection_streak = 0;
    int high_iter_streak = 0;
    int stiffness_cooldown = 0;

    circuit_.set_current_time(t);
    circuit_.set_timestep(dt);
    circuit_.set_integration_order(options_.enable_bdf_order_control ?
        std::clamp(bdf_controller_.current_order(), 1, 2) : 2);

    circuit_.update_history(x, true);

    result.time.push_back(t);
    result.states.push_back(x);

    if (callback) {
        callback(t, x);
    }

    for (auto& sw : switch_monitors_) {
        Real v_ctrl = (sw.ctrl >= 0) ? x[sw.ctrl] : 0.0;
        sw.was_on = v_ctrl > sw.v_threshold;
    }

    while (t < options_.tstop) {
        if (control) {
            if (control->should_stop()) {
                result.message = "Simulation stopped by user";
                break;
            }
            while (control->should_pause() && !control->should_stop()) {
                control->wait_until_resumed();
            }
            if (control->should_stop()) {
                result.message = "Simulation stopped by user";
                break;
            }
        }

        if (t + dt > options_.tstop) {
            dt = options_.tstop - t;
            if (dt < options_.dt_min * 0.1) {
                break;
            }
        }

        bool accepted = false;
        int retries = 0;
        NewtonResult step_result;
        Real dt_used = dt;

        while (!accepted && retries <= options_.max_step_retries) {
            if (options_.stiffness_config.enable && stiffness_cooldown > 0) {
                dt = std::max(options_.dt_min, dt * options_.stiffness_config.dt_backoff);
                if (options_.enable_bdf_order_control) {
                    bdf_controller_.set_order(
                        std::min(bdf_controller_.current_order(), options_.stiffness_config.max_bdf_order));
                }
            }

            Real t_next = t + dt;
            dt_used = dt;

            step_result = solve_step(t_next, dt_used, x);

            if (step_result.status != SolverStatus::Success) {
                dt = std::max(options_.dt_min, dt * 0.5);
                result.timestep_rejections++;
                retries++;
                rejection_streak++;
                high_iter_streak = 0;
                if (options_.stiffness_config.enable &&
                    rejection_streak >= options_.stiffness_config.rejection_streak_threshold) {
                    stiffness_cooldown = options_.stiffness_config.cooldown_steps;
                }
                if (options_.enable_bdf_order_control) {
                    bdf_controller_.reduce_on_failure();
                }
                continue;
            }

            Real lte = -1.0;
            if (options_.adaptive_timestep && lte_estimator_.has_sufficient_history()) {
                lte = lte_estimator_.compute(step_result.solution,
                                             circuit_.num_nodes(),
                                             circuit_.num_branches());
            }

            if (options_.adaptive_timestep && lte >= 0.0) {
                auto decision = timestep_controller_.compute_combined(
                    lte, step_result.iterations,
                    options_.enable_bdf_order_control ? bdf_controller_.current_order() : 2);

                if (!decision.accepted) {
                    dt = decision.dt_new;
                    result.timestep_rejections++;
                    retries++;
                    rejection_streak++;
                    high_iter_streak = 0;
                    if (options_.stiffness_config.enable &&
                        rejection_streak >= options_.stiffness_config.rejection_streak_threshold) {
                        stiffness_cooldown = options_.stiffness_config.cooldown_steps;
                    }
                    continue;
                }

                dt = decision.dt_new;
            }

            // Event-aligned step splitting for hard switching edges
            if (options_.enable_events && dt_used > options_.dt_min * 1.01) {
                bool split_for_event = false;
                for (auto& sw : switch_monitors_) {
                    Real v_now = (sw.ctrl >= 0) ? step_result.solution[sw.ctrl] : 0.0;
                    bool now_on = v_now > sw.v_threshold;
                    if (now_on != sw.was_on) {
                        Real t_event = t + dt_used;
                        Vector x_event = step_result.solution;
                        if (find_switch_event_time(sw, t, t + dt_used, x, t_event, x_event)) {
                            Real dt_event = t_event - t;
                            if (dt_event > options_.dt_min * 1.01 && dt_event < dt_used * 0.999) {
                                dt = std::max(options_.dt_min, dt_event);
                                retries++;
                                split_for_event = true;
                                break;
                            }
                        }
                    }
                }
                if (split_for_event) {
                    continue;
                }
            }

            accepted = true;
        }

        if (!accepted) {
            result.success = false;
            result.final_status = step_result.status;
            result.message = "Transient failed at t=" + std::to_string(t + dt_used) +
                             ": " + step_result.error_message;
            break;
        }

        result.newton_iterations_total += step_result.iterations;
        rejection_streak = 0;

        if (options_.stiffness_config.enable) {
            if (step_result.iterations >= options_.stiffness_config.newton_iter_threshold) {
                high_iter_streak++;
            } else {
                high_iter_streak = 0;
            }

            if (high_iter_streak >= options_.stiffness_config.newton_streak_threshold) {
                stiffness_cooldown = options_.stiffness_config.cooldown_steps;
            }

            if (options_.stiffness_config.monitor_conditioning) {
                auto telemetry = newton_solver_.linear_solver().telemetry();
                if (telemetry.last_error > options_.stiffness_config.conditioning_error_threshold) {
                    stiffness_cooldown = options_.stiffness_config.cooldown_steps;
                }
            }

            if (stiffness_cooldown > 0) {
                stiffness_cooldown--;
            }
        }

        if (options_.enable_events) {
            for (auto& sw : switch_monitors_) {
                Real v_now = (sw.ctrl >= 0) ? step_result.solution[sw.ctrl] : 0.0;
                bool now_on = v_now > sw.v_threshold;

                if (now_on != sw.was_on) {
                    Real t_event = t + dt_used;
                    Vector x_event = step_result.solution;

                    if (find_switch_event_time(sw, t, t + dt_used, x, t_event, x_event)) {
                        record_switch_event(sw, t_event, x_event, now_on, result, event_callback);
                    } else {
                        record_switch_event(sw, t + dt_used, step_result.solution, now_on, result, event_callback);
                    }

                    sw.was_on = now_on;
                }
            }
        }

        accumulate_conduction_losses(step_result.solution, dt_used);

        t += dt_used;
        x = step_result.solution;
        circuit_.update_history(x);

        if (options_.adaptive_timestep) {
            lte_estimator_.record_solution(x, t, dt_used);

            if (options_.enable_bdf_order_control && lte_estimator_.has_sufficient_history()) {
                Real lte = lte_estimator_.compute(x, circuit_.num_nodes(), circuit_.num_branches());
                auto decision = bdf_controller_.select_order(lte, options_.timestep_config.error_tolerance);
                if (decision.order_changed) {
                    circuit_.set_integration_order(std::clamp(decision.new_order, 1, 2));
                }
            }
        }

        result.time.push_back(t);
        result.states.push_back(x);

        if (callback) {
            callback(t, x);
        }

        result.total_steps++;
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    result.total_time_seconds = std::chrono::duration<double>(end_time - start_time).count();

    if (result.success) {
        result.final_status = SolverStatus::Success;
        if (result.message.empty()) {
            result.message = "Transient completed";
        }
    }

    finalize_loss_summary(result);

    result.linear_solver_telemetry = newton_solver_.linear_solver().telemetry();

    return result;
}

SimulationResult Simulator::run_transient_with_progress(
    SimulationCallback callback,
    EventCallback event_callback,
    SimulationControl* control,
    const ProgressCallbackConfig& progress_config) {

    auto start_time = std::chrono::high_resolution_clock::now();
    auto last_progress_time = start_time;
    int steps_since_progress = 0;
    int64_t steps_total = 0;

    auto wrapped_callback = [&](Real time, const Vector& state) {
        steps_total++;
        steps_since_progress++;

        if (callback) {
            callback(time, state);
        }

        if (!progress_config.callback) {
            return;
        }

        auto now = std::chrono::high_resolution_clock::now();
        double elapsed_ms = std::chrono::duration<double, std::milli>(now - last_progress_time).count();

        bool due_time = elapsed_ms >= progress_config.min_interval_ms;
        bool due_steps = steps_since_progress >= progress_config.min_steps;

        if (due_time && due_steps) {
            SimulationProgress progress;
            progress.current_time = time;
            progress.total_time = options_.tstop;
            progress.progress_percent = options_.tstop > 0
                ? 100.0 * progress.current_time / options_.tstop
                : 0.0;
            progress.steps_completed = steps_total;
            progress.elapsed_seconds = std::chrono::duration<double>(now - start_time).count();

            progress_config.callback(progress);

            last_progress_time = now;
            steps_since_progress = 0;
        }
    };

    SimulationResult result = run_transient(wrapped_callback, event_callback, control);

    if (progress_config.callback) {
        SimulationProgress progress;
        if (!result.time.empty()) {
            progress.current_time = result.time.back();
            progress.total_time = options_.tstop;
            progress.progress_percent = options_.tstop > 0
                ? 100.0 * progress.current_time / options_.tstop
                : 0.0;
            progress.steps_completed = steps_total;
        }
        progress.elapsed_seconds = result.total_time_seconds;
        progress_config.callback(progress);
    }

    return result;
}

}  // namespace pulsim::v1
