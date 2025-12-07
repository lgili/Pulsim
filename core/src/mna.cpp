#include "pulsim/mna.hpp"
#include <cmath>
#include <stdexcept>
#include <iostream>

namespace pulsim {

MNAAssembler::MNAAssembler(const Circuit& circuit)
    : circuit_(circuit) {

    // Assign branch indices for voltage sources, inductors, and transformers
    next_branch_idx_ = circuit_.node_count();
    for (const auto& comp : circuit_.components()) {
        if (comp.has_branch_current()) {
            branch_indices_[comp.name()] = next_branch_idx_++;
        }
        if (comp.type() == ComponentType::Transformer) {
            // Transformer has two branch currents
            branch_indices_[comp.name() + "_p"] = next_branch_idx_++;
            branch_indices_[comp.name() + "_s"] = next_branch_idx_++;
        }
        if (comp.type() == ComponentType::Diode ||
            comp.type() == ComponentType::MOSFET ||
            comp.type() == ComponentType::IGBT) {
            has_nonlinear_ = true;
        }
        // Initialize switch states
        if (comp.type() == ComponentType::Switch) {
            const auto& params = std::get<SwitchParams>(comp.params());
            SwitchState state;
            state.name = comp.name();
            state.is_closed = params.initial_state;
            state.last_control_voltage = 0.0;
            state.turn_on_time = -1.0;
            state.turn_off_time = -1.0;
            switch_states_.push_back(state);
        }
    }
}

Real MNAAssembler::evaluate_waveform(const Waveform& waveform, Real time) {
    return std::visit([time](const auto& w) -> Real {
        using T = std::decay_t<decltype(w)>;

        if constexpr (std::is_same_v<T, DCWaveform>) {
            return w.value;
        }
        else if constexpr (std::is_same_v<T, PulseWaveform>) {
            if (time < w.td) return w.v1;

            Real t = std::fmod(time - w.td, w.period);

            if (t < w.tr) {
                // Rising edge
                return w.v1 + (w.v2 - w.v1) * (t / w.tr);
            }
            t -= w.tr;

            if (t < w.pw) {
                // Pulse high
                return w.v2;
            }
            t -= w.pw;

            if (t < w.tf) {
                // Falling edge
                return w.v2 + (w.v1 - w.v2) * (t / w.tf);
            }

            // Pulse low
            return w.v1;
        }
        else if constexpr (std::is_same_v<T, SineWaveform>) {
            if (time < w.delay) return w.offset;
            Real t = time - w.delay;
            Real envelope = std::exp(-w.damping * t);
            return w.offset + w.amplitude * envelope * std::sin(2.0 * M_PI * w.frequency * t);
        }
        else if constexpr (std::is_same_v<T, PWLWaveform>) {
            if (w.points.empty()) return 0.0;
            if (time <= w.points.front().first) return w.points.front().second;
            if (time >= w.points.back().first) return w.points.back().second;

            // Linear interpolation
            for (size_t i = 1; i < w.points.size(); ++i) {
                if (time <= w.points[i].first) {
                    Real t0 = w.points[i-1].first;
                    Real t1 = w.points[i].first;
                    Real v0 = w.points[i-1].second;
                    Real v1 = w.points[i].second;
                    Real alpha = (time - t0) / (t1 - t0);
                    return v0 + alpha * (v1 - v0);
                }
            }
            return w.points.back().second;
        }
        else if constexpr (std::is_same_v<T, PWMWaveform>) {
            // PWM waveform with dead-time support
            // Dead-time is inserted at both rising and falling edges
            //
            // For non-complementary (high-side) signal:
            //   OFF during: [0, dead_time) and [t_on, t_on + dead_time)
            //   ON during:  [dead_time, t_on)
            //
            // For complementary (low-side) signal:
            //   The inverse, with same dead-time gaps
            //
            Real period = w.period();
            Real t_on_raw = w.t_on();  // Raw on-time without dead-time adjustment
            Real dt = w.dead_time;

            // Apply phase offset
            Real t = std::fmod(time + w.phase * period, period);
            if (t < 0) t += period;

            // Effective on/off times considering dead-time
            // Non-complementary (high-side):
            //   - Starts at dt (after dead-time)
            //   - Ends at t_on_raw - dt (before dead-time for falling edge)
            // Complementary (low-side):
            //   - Starts at t_on_raw + dt
            //   - Ends at period - dt

            bool is_on = false;

            if (!w.complementary) {
                // High-side: ON from dt to (t_on_raw - dt)
                // But we need t_on_raw > 2*dt for any on-time
                Real t_start = dt;
                Real t_end = t_on_raw;
                // Effective on-time reduced by dead-time at rising edge only
                // (dead-time at falling edge is handled by delaying complementary turn-on)
                if (t >= t_start && t < t_end) {
                    is_on = true;
                }
            } else {
                // Low-side (complementary): ON from (t_on_raw + dt) to (period - dt)
                // Dead-time after high-side turns off, and before high-side turns on
                Real t_start = t_on_raw + dt;
                Real t_end = period;
                // Only on if there's room for the low-side pulse
                if (t >= t_start && t < t_end) {
                    is_on = true;
                }
            }

            return is_on ? w.v_on : w.v_off;
        }
        else {
            return 0.0;
        }
    }, waveform);
}

void MNAAssembler::stamp_resistor(std::vector<Triplet>& triplets, Vector& /*b*/,
                                  const Component& comp) {
    const auto& params = std::get<ResistorParams>(comp.params());
    Real g = 1.0 / params.resistance;

    Index n1 = circuit_.node_index(comp.nodes()[0]);
    Index n2 = circuit_.node_index(comp.nodes()[1]);

    // Stamp: G[i,i] += g, G[j,j] += g, G[i,j] -= g, G[j,i] -= g
    if (n1 >= 0) {
        triplets.emplace_back(n1, n1, g);
        if (n2 >= 0) {
            triplets.emplace_back(n1, n2, -g);
        }
    }
    if (n2 >= 0) {
        triplets.emplace_back(n2, n2, g);
        if (n1 >= 0) {
            triplets.emplace_back(n2, n1, -g);
        }
    }
}

void MNAAssembler::stamp_capacitor_dc(std::vector<Triplet>& /*triplets*/, Vector& /*b*/,
                                      const Component& /*comp*/) {
    // For DC analysis, capacitor is open circuit (no stamp needed)
}

void MNAAssembler::stamp_capacitor_transient(std::vector<Triplet>& triplets, Vector& b,
                                             const Component& comp,
                                             const Vector& x_prev, Real dt) {
    const auto& params = std::get<CapacitorParams>(comp.params());
    Real C = params.capacitance;

    Index n1 = circuit_.node_index(comp.nodes()[0]);
    Index n2 = circuit_.node_index(comp.nodes()[1]);

    // Backward Euler companion model:
    // i = C * dv/dt ≈ C * (v_n - v_{n-1}) / dt
    // Equivalent to a conductance Geq = C/dt in parallel with current source Ieq
    Real Geq = C / dt;

    // Previous voltage across capacitor
    Real v_prev = 0.0;
    if (n1 >= 0) v_prev += x_prev(n1);
    if (n2 >= 0) v_prev -= x_prev(n2);

    Real Ieq = Geq * v_prev;

    // Stamp conductance
    if (n1 >= 0) {
        triplets.emplace_back(n1, n1, Geq);
        b(n1) += Ieq;
        if (n2 >= 0) {
            triplets.emplace_back(n1, n2, -Geq);
        }
    }
    if (n2 >= 0) {
        triplets.emplace_back(n2, n2, Geq);
        b(n2) -= Ieq;
        if (n1 >= 0) {
            triplets.emplace_back(n2, n1, -Geq);
        }
    }
}

void MNAAssembler::stamp_inductor_dc(std::vector<Triplet>& triplets, Vector& /*b*/,
                                     const Component& comp, Index branch_idx) {
    // For DC analysis, inductor is short circuit
    // V = 0 across inductor, I is unknown
    Index n1 = circuit_.node_index(comp.nodes()[0]);
    Index n2 = circuit_.node_index(comp.nodes()[1]);

    // KCL: current leaves n1, enters n2
    if (n1 >= 0) {
        triplets.emplace_back(n1, branch_idx, 1.0);
        triplets.emplace_back(branch_idx, n1, 1.0);
    }
    if (n2 >= 0) {
        triplets.emplace_back(n2, branch_idx, -1.0);
        triplets.emplace_back(branch_idx, n2, -1.0);
    }
    // V(n1) - V(n2) = 0 for DC (short circuit)
}

void MNAAssembler::stamp_inductor_transient(std::vector<Triplet>& triplets, Vector& b,
                                            const Component& comp, Index branch_idx,
                                            const Vector& x_prev, Real dt) {
    const auto& params = std::get<InductorParams>(comp.params());
    Real L = params.inductance;

    Index n1 = circuit_.node_index(comp.nodes()[0]);
    Index n2 = circuit_.node_index(comp.nodes()[1]);

    // Backward Euler companion model:
    // v = L * di/dt ≈ L * (i_n - i_{n-1}) / dt
    // Equivalent to: v = Req * i + Veq
    // where Req = L/dt, Veq = -L/dt * i_{n-1}
    Real Req = L / dt;
    Real i_prev = x_prev(branch_idx);
    Real Veq = -Req * i_prev;

    // KCL stamps (current variable)
    if (n1 >= 0) {
        triplets.emplace_back(n1, branch_idx, 1.0);
        triplets.emplace_back(branch_idx, n1, 1.0);
    }
    if (n2 >= 0) {
        triplets.emplace_back(n2, branch_idx, -1.0);
        triplets.emplace_back(branch_idx, n2, -1.0);
    }

    // V(n1) - V(n2) = Req * I + Veq
    triplets.emplace_back(branch_idx, branch_idx, -Req);
    b(branch_idx) = Veq;
}

// =============================================================================
// Trapezoidal (GEAR-2) Integration Companion Models
// =============================================================================

void MNAAssembler::stamp_capacitor_trapezoidal(std::vector<Triplet>& triplets, Vector& b,
                                                const Component& comp,
                                                const DynamicHistory& history, Real dt) {
    const auto& params = std::get<CapacitorParams>(comp.params());
    Real C = params.capacitance;

    Index n1 = circuit_.node_index(comp.nodes()[0]);
    Index n2 = circuit_.node_index(comp.nodes()[1]);

    // Trapezoidal companion model:
    // i = C * dv/dt ≈ C * (v_n - v_{n-1}) / dt
    // With trapezoidal: i_n = (2C/dt) * v_n - (2C/dt) * v_{n-1} - i_{n-1}
    // Equivalent conductance: Geq = 2C/dt
    // Equivalent current source: Ieq = (2C/dt) * v_{n-1} + i_{n-1}
    //
    // For simplicity, we estimate i_{n-1} from v_{n-1} and v_{n-2}:
    // i_{n-1} ≈ C * (v_{n-1} - v_{n-2}) / dt_prev
    Real Geq = 2.0 * C / dt;

    // Previous voltage
    Real v_prev = 0.0;
    if (n1 >= 0) v_prev += history.x_prev(n1);
    if (n2 >= 0) v_prev -= history.x_prev(n2);

    Real i_prev = 0.0;
    if (history.has_prev2 && history.dt_prev > 0) {
        // Calculate previous current from voltage derivative
        Real v_prev2 = 0.0;
        if (n1 >= 0) v_prev2 += history.x_prev2(n1);
        if (n2 >= 0) v_prev2 -= history.x_prev2(n2);
        i_prev = C * (v_prev - v_prev2) / history.dt_prev;
    }

    Real Ieq = Geq * v_prev + i_prev;

    // Stamp conductance
    if (n1 >= 0) {
        triplets.emplace_back(n1, n1, Geq);
        b(n1) += Ieq;
        if (n2 >= 0) {
            triplets.emplace_back(n1, n2, -Geq);
        }
    }
    if (n2 >= 0) {
        triplets.emplace_back(n2, n2, Geq);
        b(n2) -= Ieq;
        if (n1 >= 0) {
            triplets.emplace_back(n2, n1, -Geq);
        }
    }
}

void MNAAssembler::stamp_capacitor_bdf2(std::vector<Triplet>& triplets, Vector& b,
                                         const Component& comp,
                                         const DynamicHistory& history, Real dt) {
    const auto& params = std::get<CapacitorParams>(comp.params());
    Real C = params.capacitance;

    Index n1 = circuit_.node_index(comp.nodes()[0]);
    Index n2 = circuit_.node_index(comp.nodes()[1]);

    // BDF2 companion model:
    // For constant step: i_n = (3C)/(2dt) * v_n - (4C)/(2dt) * v_{n-1} + (C)/(2dt) * v_{n-2}
    // Equivalent conductance: Geq = 3C/(2dt)
    // Equivalent current: Ieq = (4C)/(2dt) * v_{n-1} - (C)/(2dt) * v_{n-2}

    Real Geq, Ieq;

    Real v_prev = 0.0;
    if (n1 >= 0) v_prev += history.x_prev(n1);
    if (n2 >= 0) v_prev -= history.x_prev(n2);

    if (history.has_prev2) {
        Real v_prev2 = 0.0;
        if (n1 >= 0) v_prev2 += history.x_prev2(n1);
        if (n2 >= 0) v_prev2 -= history.x_prev2(n2);

        // BDF2 coefficients (constant step)
        Geq = 1.5 * C / dt;
        Ieq = (2.0 * C / dt) * v_prev - (0.5 * C / dt) * v_prev2;
    } else {
        // Fall back to Backward Euler for first step
        Geq = C / dt;
        Ieq = Geq * v_prev;
    }

    // Stamp conductance
    if (n1 >= 0) {
        triplets.emplace_back(n1, n1, Geq);
        b(n1) += Ieq;
        if (n2 >= 0) {
            triplets.emplace_back(n1, n2, -Geq);
        }
    }
    if (n2 >= 0) {
        triplets.emplace_back(n2, n2, Geq);
        b(n2) -= Ieq;
        if (n1 >= 0) {
            triplets.emplace_back(n2, n1, -Geq);
        }
    }
}

void MNAAssembler::stamp_inductor_trapezoidal(std::vector<Triplet>& triplets, Vector& b,
                                               const Component& comp, Index branch_idx,
                                               const DynamicHistory& history, Real dt) {
    const auto& params = std::get<InductorParams>(comp.params());
    Real L = params.inductance;

    Index n1 = circuit_.node_index(comp.nodes()[0]);
    Index n2 = circuit_.node_index(comp.nodes()[1]);

    // Trapezoidal companion model for inductor:
    // v = L * di/dt
    // With trapezoidal: v_n = (2L/dt) * i_n - (2L/dt) * i_{n-1} - v_{n-1}
    // Rearranging for MNA: V(n1) - V(n2) = Req * I + Veq
    // where Req = 2L/dt, Veq = -(2L/dt) * i_{n-1} - v_{n-1}

    Real Req = 2.0 * L / dt;
    Real i_prev = history.x_prev(branch_idx);

    // Previous voltage across inductor
    Real v_prev = 0.0;
    if (n1 >= 0) v_prev += history.x_prev(n1);
    if (n2 >= 0) v_prev -= history.x_prev(n2);

    Real Veq = -Req * i_prev - v_prev;

    // KCL stamps (current variable)
    if (n1 >= 0) {
        triplets.emplace_back(n1, branch_idx, 1.0);
        triplets.emplace_back(branch_idx, n1, 1.0);
    }
    if (n2 >= 0) {
        triplets.emplace_back(n2, branch_idx, -1.0);
        triplets.emplace_back(branch_idx, n2, -1.0);
    }

    // V(n1) - V(n2) = Req * I + Veq
    triplets.emplace_back(branch_idx, branch_idx, -Req);
    b(branch_idx) = Veq;
}

void MNAAssembler::stamp_inductor_bdf2(std::vector<Triplet>& triplets, Vector& b,
                                        const Component& comp, Index branch_idx,
                                        const DynamicHistory& history, Real dt) {
    const auto& params = std::get<InductorParams>(comp.params());
    Real L = params.inductance;

    Index n1 = circuit_.node_index(comp.nodes()[0]);
    Index n2 = circuit_.node_index(comp.nodes()[1]);

    Real Req, Veq;
    Real i_prev = history.x_prev(branch_idx);

    if (history.has_prev2) {
        // BDF2 coefficients (constant step)
        // i_n = (4/3) * i_{n-1} - (1/3) * i_{n-2} + (2/3) * (dt/L) * v_n
        // Rearranging: v_n = (3L)/(2dt) * i_n - (4L)/(2dt) * i_{n-1} + (L)/(2dt) * i_{n-2}
        Real i_prev2 = history.x_prev2(branch_idx);
        Req = 1.5 * L / dt;
        Veq = -(2.0 * L / dt) * i_prev + (0.5 * L / dt) * i_prev2;
    } else {
        // Fall back to Backward Euler for first step
        Req = L / dt;
        Veq = -Req * i_prev;
    }

    // KCL stamps (current variable)
    if (n1 >= 0) {
        triplets.emplace_back(n1, branch_idx, 1.0);
        triplets.emplace_back(branch_idx, n1, 1.0);
    }
    if (n2 >= 0) {
        triplets.emplace_back(n2, branch_idx, -1.0);
        triplets.emplace_back(branch_idx, n2, -1.0);
    }

    triplets.emplace_back(branch_idx, branch_idx, -Req);
    b(branch_idx) = Veq;
}

void MNAAssembler::stamp_voltage_source(std::vector<Triplet>& triplets, Vector& b,
                                        const Component& comp, Index branch_idx, Real time) {
    const auto& params = std::get<VoltageSourceParams>(comp.params());
    Real V = evaluate_waveform(params.waveform, time);

    Index n1 = circuit_.node_index(comp.nodes()[0]);  // positive
    Index n2 = circuit_.node_index(comp.nodes()[1]);  // negative

    // KCL: current leaves n1 (positive), enters n2 (negative)
    if (n1 >= 0) {
        triplets.emplace_back(n1, branch_idx, 1.0);
        triplets.emplace_back(branch_idx, n1, 1.0);
    }
    if (n2 >= 0) {
        triplets.emplace_back(n2, branch_idx, -1.0);
        triplets.emplace_back(branch_idx, n2, -1.0);
    }

    // V(n1) - V(n2) = V
    b(branch_idx) = V;
}

void MNAAssembler::stamp_current_source(Vector& b, const Component& comp, Real time) {
    const auto& params = std::get<CurrentSourceParams>(comp.params());
    Real I = evaluate_waveform(params.waveform, time);

    Index n1 = circuit_.node_index(comp.nodes()[0]);  // positive (current enters)
    Index n2 = circuit_.node_index(comp.nodes()[1]);  // negative (current leaves)

    // Current flows from n2 to n1 (into positive terminal)
    if (n1 >= 0) b(n1) += I;
    if (n2 >= 0) b(n2) -= I;
}

void MNAAssembler::stamp_diode(std::vector<Triplet>& triplets, Vector& f,
                               const Component& comp, const Vector& x) {
    const auto& params = std::get<DiodeParams>(comp.params());

    Index n_anode = circuit_.node_index(comp.nodes()[0]);
    Index n_cathode = circuit_.node_index(comp.nodes()[1]);

    // Voltage across diode
    Real Vd = 0.0;
    if (n_anode >= 0) Vd += x(n_anode);
    if (n_cathode >= 0) Vd -= x(n_cathode);

    Real Id, Gd;
    if (params.ideal) {
        // Ideal diode: piecewise linear model
        constexpr Real Gon = 1e3;   // On conductance
        constexpr Real Goff = 1e-9; // Off conductance

        if (Vd > 0) {
            Id = Gon * Vd;
            Gd = Gon;
        } else {
            Id = Goff * Vd;
            Gd = Goff;
        }
    } else {
        // Shockley equation: Id = Is * (exp(Vd/(n*Vt)) - 1)
        Real Vt = params.vt;
        Real Is = params.is;
        Real n = params.n;

        // Limit Vd to prevent overflow
        Real Vd_limited = std::min(Vd, 40.0 * n * Vt);

        Real exp_term = std::exp(Vd_limited / (n * Vt));
        Id = Is * (exp_term - 1.0);
        Gd = (Is / (n * Vt)) * exp_term;

        // Add minimum conductance for numerical stability
        Gd = std::max(Gd, 1e-12);
    }

    // Newton-Raphson linearization: I = Id + Gd * (V - Vd)
    // f = Id - Gd * Vd (equivalent current source)
    Real Ieq = Id - Gd * Vd;

    // Stamp Jacobian (conductance)
    if (n_anode >= 0) {
        triplets.emplace_back(n_anode, n_anode, Gd);
        f(n_anode) -= Ieq;  // Current out of anode
        if (n_cathode >= 0) {
            triplets.emplace_back(n_anode, n_cathode, -Gd);
        }
    }
    if (n_cathode >= 0) {
        triplets.emplace_back(n_cathode, n_cathode, Gd);
        f(n_cathode) += Ieq;  // Current into cathode
        if (n_anode >= 0) {
            triplets.emplace_back(n_cathode, n_anode, -Gd);
        }
    }
}

void MNAAssembler::stamp_diode_capacitance(std::vector<Triplet>& triplets, Vector& b,
                                           const Component& comp, const Vector& x,
                                           const Vector& x_prev, Real dt) {
    const auto& params = std::get<DiodeParams>(comp.params());

    // Skip if no junction capacitance
    if (params.cj0 <= 0.0 && params.tt <= 0.0) {
        return;
    }

    Index n_anode = circuit_.node_index(comp.nodes()[0]);
    Index n_cathode = circuit_.node_index(comp.nodes()[1]);

    // Current voltage across diode
    Real Vd = 0.0;
    if (n_anode >= 0) Vd += x(n_anode);
    if (n_cathode >= 0) Vd -= x(n_cathode);

    // Previous voltage
    Real Vd_prev = 0.0;
    if (n_anode >= 0) Vd_prev += x_prev(n_anode);
    if (n_cathode >= 0) Vd_prev -= x_prev(n_cathode);

    Real C_total = 0.0;

    // Junction (depletion) capacitance: Cj = Cj0 / (1 - Vd/Vj)^m
    // Valid for Vd < Vj (forward bias limited)
    if (params.cj0 > 0.0) {
        Real Vj = params.vj;
        Real m = params.m;

        // Limit voltage to avoid singularity at Vd = Vj
        Real Vd_eff = std::min(Vd, 0.9 * Vj);

        Real Cj;
        if (Vd_eff < 0.0) {
            // Reverse bias: standard formula
            Cj = params.cj0 / std::pow(1.0 - Vd_eff / Vj, m);
        } else {
            // Forward bias: linearize to avoid singularity
            // Use linear extrapolation from Vd = 0
            Real Cj0 = params.cj0;
            Real dCj_dV = Cj0 * m / Vj;  // Derivative at Vd = 0
            Cj = Cj0 + dCj_dV * Vd_eff;
        }
        C_total += Cj;
    }

    // Diffusion capacitance: Cd = tt * dId/dVd = tt * Gd
    // In forward bias, the diffusion capacitance dominates
    if (params.tt > 0.0 && !params.ideal) {
        Real Vt = params.vt;
        Real Is = params.is;
        Real n = params.n;
        Real Vd_limited = std::min(Vd, 40.0 * n * Vt);
        Real exp_term = std::exp(Vd_limited / (n * Vt));
        Real Gd = (Is / (n * Vt)) * exp_term;
        Real Cd = params.tt * Gd;
        C_total += Cd;
    }

    if (C_total <= 0.0) {
        return;
    }

    // Backward Euler companion model for capacitor
    Real Geq = C_total / dt;
    Real Ieq = Geq * Vd_prev;

    // Stamp conductance
    if (n_anode >= 0) {
        triplets.emplace_back(n_anode, n_anode, Geq);
        b(n_anode) += Ieq;
        if (n_cathode >= 0) {
            triplets.emplace_back(n_anode, n_cathode, -Geq);
        }
    }
    if (n_cathode >= 0) {
        triplets.emplace_back(n_cathode, n_cathode, Geq);
        b(n_cathode) -= Ieq;
        if (n_anode >= 0) {
            triplets.emplace_back(n_cathode, n_anode, -Geq);
        }
    }
}

void MNAAssembler::stamp_switch(std::vector<Triplet>& triplets, Vector& /*b*/,
                                const Component& comp, const SwitchState& state) {
    const auto& params = std::get<SwitchParams>(comp.params());

    Index n1 = circuit_.node_index(comp.nodes()[0]);
    Index n2 = circuit_.node_index(comp.nodes()[1]);

    // Switch is modeled as a variable resistor
    Real R = state.is_closed ? params.ron : params.roff;
    Real g = 1.0 / R;

    // Stamp conductance (same as resistor)
    if (n1 >= 0) {
        triplets.emplace_back(n1, n1, g);
        if (n2 >= 0) {
            triplets.emplace_back(n1, n2, -g);
        }
    }
    if (n2 >= 0) {
        triplets.emplace_back(n2, n2, g);
        if (n1 >= 0) {
            triplets.emplace_back(n2, n1, -g);
        }
    }
}

void MNAAssembler::stamp_mosfet(std::vector<Triplet>& triplets, Vector& f,
                                const Component& comp, const Vector& x) {
    const auto& params = std::get<MOSFETParams>(comp.params());

    Index n_drain = circuit_.node_index(comp.nodes()[0]);
    Index n_gate = circuit_.node_index(comp.nodes()[1]);
    Index n_source = circuit_.node_index(comp.nodes()[2]);

    // Get terminal voltages
    Real Vd = (n_drain >= 0) ? x(n_drain) : 0.0;
    Real Vg = (n_gate >= 0) ? x(n_gate) : 0.0;
    Real Vs = (n_source >= 0) ? x(n_source) : 0.0;

    Real Vgs = Vg - Vs;
    Real Vds = Vd - Vs;

    // Handle PMOS by flipping voltages
    Real sign = (params.type == MOSFETType::NMOS) ? 1.0 : -1.0;
    Vgs *= sign;
    Vds *= sign;
    Real Vth = params.vth;

    Real Id = 0.0;        // Drain current
    Real gm = 0.0;        // Transconductance dId/dVgs
    Real gds = 0.0;       // Output conductance dId/dVds

    // Simple switch model if rds_on is specified
    if (params.rds_on > 0) {
        if (Vgs > Vth) {
            // ON state
            Real g = 1.0 / params.rds_on;
            Id = g * Vds;
            gds = g;
            gm = 0.0;
        } else {
            // OFF state
            Real g = 1.0 / params.rds_off;
            Id = g * Vds;
            gds = g;
            gm = 0.0;
        }
    } else {
        // Level 1 (Shichman-Hodges) model
        Real Kp = params.kp_effective();  // Kp * W/L
        Real lambda = params.lambda;

        if (Vgs <= Vth) {
            // Cutoff region
            Id = 0.0;
            gm = 0.0;
            gds = 1e-12;  // Small leakage conductance
        } else if (Vds < Vgs - Vth) {
            // Linear (triode) region
            // Id = Kp * [(Vgs - Vth) * Vds - Vds^2/2] * (1 + lambda*Vds)
            Real Vov = Vgs - Vth;  // Overdrive voltage
            Id = Kp * (Vov * Vds - 0.5 * Vds * Vds) * (1.0 + lambda * Vds);
            gm = Kp * Vds * (1.0 + lambda * Vds);
            gds = Kp * (Vov - Vds) * (1.0 + lambda * Vds) +
                  Kp * (Vov * Vds - 0.5 * Vds * Vds) * lambda;
        } else {
            // Saturation region
            // Id = (Kp/2) * (Vgs - Vth)^2 * (1 + lambda*Vds)
            Real Vov = Vgs - Vth;
            Id = 0.5 * Kp * Vov * Vov * (1.0 + lambda * Vds);
            gm = Kp * Vov * (1.0 + lambda * Vds);
            gds = 0.5 * Kp * Vov * Vov * lambda;
        }

        // Ensure minimum conductance for numerical stability
        gds = std::max(gds, 1e-12);
    }

    // Apply sign for PMOS
    Id *= sign;

    // Newton-Raphson linearization
    // I_drain = Id + gm*(Vgs - Vgs0) + gds*(Vds - Vds0)
    // Equivalent current: Ieq = Id - gm*Vgs - gds*Vds

    Real Ieq = Id - gm * Vgs * sign - gds * Vds * sign;

    // Stamp into Jacobian
    // Current flows: out of drain, into source (for NMOS with positive Id)
    // Partial derivatives:
    //   dId/dVd = gds
    //   dId/dVg = gm
    //   dId/dVs = -gm - gds

    if (n_drain >= 0) {
        triplets.emplace_back(n_drain, n_drain, gds);
        f(n_drain) -= Ieq;  // Current out of drain

        if (n_gate >= 0) {
            triplets.emplace_back(n_drain, n_gate, gm * sign);
        }
        if (n_source >= 0) {
            triplets.emplace_back(n_drain, n_source, -gds - gm * sign);
        }
    }

    if (n_source >= 0) {
        triplets.emplace_back(n_source, n_source, gds + gm * sign);
        f(n_source) += Ieq;  // Current into source

        if (n_gate >= 0) {
            triplets.emplace_back(n_source, n_gate, -gm * sign);
        }
        if (n_drain >= 0) {
            triplets.emplace_back(n_source, n_drain, -gds);
        }
    }

    // Body diode (optional)
    if (params.body_diode) {
        // Diode from source to drain (for NMOS)
        Real Vdiode = (params.type == MOSFETType::NMOS) ? (Vs - Vd) : (Vd - Vs);
        Real Is = params.is_body;
        Real n = params.n_body;
        constexpr Real Vt = 0.026;  // Thermal voltage

        Real Vd_limited = std::min(Vdiode, 40.0 * n * Vt);
        Real exp_term = std::exp(Vd_limited / (n * Vt));
        Real Id_diode = Is * (exp_term - 1.0);
        Real Gd = (Is / (n * Vt)) * exp_term;
        Gd = std::max(Gd, 1e-12);

        Real Ieq_diode = Id_diode - Gd * Vdiode;

        // Stamp body diode (source to drain for NMOS)
        if (params.type == MOSFETType::NMOS) {
            if (n_source >= 0) {
                triplets.emplace_back(n_source, n_source, Gd);
                f(n_source) -= Ieq_diode;
                if (n_drain >= 0) {
                    triplets.emplace_back(n_source, n_drain, -Gd);
                }
            }
            if (n_drain >= 0) {
                triplets.emplace_back(n_drain, n_drain, Gd);
                f(n_drain) += Ieq_diode;
                if (n_source >= 0) {
                    triplets.emplace_back(n_drain, n_source, -Gd);
                }
            }
        } else {
            // PMOS: diode from drain to source
            if (n_drain >= 0) {
                triplets.emplace_back(n_drain, n_drain, Gd);
                f(n_drain) -= Ieq_diode;
                if (n_source >= 0) {
                    triplets.emplace_back(n_drain, n_source, -Gd);
                }
            }
            if (n_source >= 0) {
                triplets.emplace_back(n_source, n_source, Gd);
                f(n_source) += Ieq_diode;
                if (n_drain >= 0) {
                    triplets.emplace_back(n_source, n_drain, -Gd);
                }
            }
        }
    }
}

void MNAAssembler::stamp_mosfet_capacitances(std::vector<Triplet>& triplets, Vector& b,
                                             const Component& comp, const Vector& x_prev, Real dt) {
    const auto& params = std::get<MOSFETParams>(comp.params());

    // Skip if no capacitances are specified
    if (params.cgs <= 0.0 && params.cgd <= 0.0 && params.cds <= 0.0) {
        return;
    }

    Index n_drain = circuit_.node_index(comp.nodes()[0]);
    Index n_gate = circuit_.node_index(comp.nodes()[1]);
    Index n_source = circuit_.node_index(comp.nodes()[2]);

    // Get previous terminal voltages
    Real Vd_prev = (n_drain >= 0) ? x_prev(n_drain) : 0.0;
    Real Vg_prev = (n_gate >= 0) ? x_prev(n_gate) : 0.0;
    Real Vs_prev = (n_source >= 0) ? x_prev(n_source) : 0.0;

    // Gate-Source Capacitance (Cgs)
    if (params.cgs > 0.0) {
        Real Geq = params.cgs / dt;
        Real Vgs_prev = Vg_prev - Vs_prev;
        Real Ieq = Geq * Vgs_prev;

        // Stamp between gate and source
        if (n_gate >= 0) {
            triplets.emplace_back(n_gate, n_gate, Geq);
            b(n_gate) += Ieq;
            if (n_source >= 0) {
                triplets.emplace_back(n_gate, n_source, -Geq);
            }
        }
        if (n_source >= 0) {
            triplets.emplace_back(n_source, n_source, Geq);
            b(n_source) -= Ieq;
            if (n_gate >= 0) {
                triplets.emplace_back(n_source, n_gate, -Geq);
            }
        }
    }

    // Gate-Drain Capacitance (Cgd) - Miller capacitance
    if (params.cgd > 0.0) {
        Real Geq = params.cgd / dt;
        Real Vgd_prev = Vg_prev - Vd_prev;
        Real Ieq = Geq * Vgd_prev;

        // Stamp between gate and drain
        if (n_gate >= 0) {
            triplets.emplace_back(n_gate, n_gate, Geq);
            b(n_gate) += Ieq;
            if (n_drain >= 0) {
                triplets.emplace_back(n_gate, n_drain, -Geq);
            }
        }
        if (n_drain >= 0) {
            triplets.emplace_back(n_drain, n_drain, Geq);
            b(n_drain) -= Ieq;
            if (n_gate >= 0) {
                triplets.emplace_back(n_drain, n_gate, -Geq);
            }
        }
    }

    // Drain-Source Capacitance (Cds)
    if (params.cds > 0.0) {
        Real Geq = params.cds / dt;
        Real Vds_prev = Vd_prev - Vs_prev;
        Real Ieq = Geq * Vds_prev;

        // Stamp between drain and source
        if (n_drain >= 0) {
            triplets.emplace_back(n_drain, n_drain, Geq);
            b(n_drain) += Ieq;
            if (n_source >= 0) {
                triplets.emplace_back(n_drain, n_source, -Geq);
            }
        }
        if (n_source >= 0) {
            triplets.emplace_back(n_source, n_source, Geq);
            b(n_source) -= Ieq;
            if (n_drain >= 0) {
                triplets.emplace_back(n_source, n_drain, -Geq);
            }
        }
    }
}

void MNAAssembler::stamp_igbt(std::vector<Triplet>& triplets, Vector& f,
                              const Component& comp, const Vector& x) {
    const auto& params = std::get<IGBTParams>(comp.params());

    Index n_collector = circuit_.node_index(comp.nodes()[0]);
    Index n_gate = circuit_.node_index(comp.nodes()[1]);
    Index n_emitter = circuit_.node_index(comp.nodes()[2]);

    // Get terminal voltages
    Real Vc = (n_collector >= 0) ? x(n_collector) : 0.0;
    Real Vg = (n_gate >= 0) ? x(n_gate) : 0.0;
    Real Ve = (n_emitter >= 0) ? x(n_emitter) : 0.0;

    Real Vge = Vg - Ve;
    Real Vce = Vc - Ve;

    Real Ic = 0.0;      // Collector current
    Real gce = 0.0;     // Output conductance dIc/dVce
    Real gm = 0.0;      // Transconductance dIc/dVge

    // Simplified IGBT model:
    // - When Vge < Vth: OFF state (high resistance)
    // - When Vge >= Vth and Vce > Vce_sat: ON state with saturation voltage drop
    // - The on-state is modeled as: Ic = (Vce - Vce_sat) / Rce_on

    if (Vge < params.vth) {
        // OFF state - very high resistance
        Real g_off = 1.0 / params.rce_off;
        Ic = g_off * Vce;
        gce = g_off;
        gm = 0.0;
    } else {
        // ON state - model with saturation voltage drop
        // The IGBT conducts with a forward voltage drop Vce_sat
        // Current through the device: Ic = (Vce - Vce_sat) / Rce_on for Vce > Vce_sat
        // For Vce < Vce_sat, we model a smooth transition

        Real g_on = 1.0 / params.rce_on;

        if (Vce > params.vce_sat) {
            // Normal forward conduction
            Ic = (Vce - params.vce_sat) * g_on;
            gce = g_on;
            gm = 0.0;  // In this simplified model, gm only affects turn-on region
        } else if (Vce > 0) {
            // Transition region (0 < Vce < Vce_sat)
            // Use smooth interpolation to avoid numerical issues
            Real alpha = Vce / params.vce_sat;  // 0 to 1
            Real g_eff = g_on * alpha;  // Smoothly vary conductance
            Ic = g_eff * Vce;
            gce = g_on * (2.0 * alpha);  // Derivative includes dg_eff/dVce
        } else {
            // Vce <= 0: reverse blocking (unless body diode conducts)
            Real g_off = 1.0 / params.rce_off;
            Ic = g_off * Vce;
            gce = g_off;
            gm = 0.0;
        }
    }

    // Ensure minimum conductance for numerical stability
    gce = std::max(gce, 1e-12);

    // Newton-Raphson linearization
    // I = Ic + gm*(Vge - Vge0) + gce*(Vce - Vce0)
    // Equivalent current: Ieq = Ic - gm*Vge - gce*Vce
    Real Ieq = Ic - gm * Vge - gce * Vce;

    // Stamp into Jacobian
    // Current flows: out of collector, into emitter
    // Partial derivatives:
    //   dIc/dVc = gce
    //   dIc/dVg = gm
    //   dIc/dVe = -gce - gm

    if (n_collector >= 0) {
        triplets.emplace_back(n_collector, n_collector, gce);
        f(n_collector) -= Ieq;  // Current out of collector

        if (n_gate >= 0) {
            triplets.emplace_back(n_collector, n_gate, gm);
        }
        if (n_emitter >= 0) {
            triplets.emplace_back(n_collector, n_emitter, -gce - gm);
        }
    }

    if (n_emitter >= 0) {
        triplets.emplace_back(n_emitter, n_emitter, gce + gm);
        f(n_emitter) += Ieq;  // Current into emitter

        if (n_gate >= 0) {
            triplets.emplace_back(n_emitter, n_gate, -gm);
        }
        if (n_collector >= 0) {
            triplets.emplace_back(n_emitter, n_collector, -gce);
        }
    }

    // Anti-parallel (freewheeling) diode
    if (params.body_diode) {
        // Diode conducts when Vce < 0 (from emitter to collector)
        Real Vdiode = Ve - Vc;  // Forward voltage for diode
        Real Is = params.is_diode;
        Real n = params.n_diode;
        constexpr Real Vt = 0.026;  // Thermal voltage

        Real Vd_limited = std::min(Vdiode, 40.0 * n * Vt);
        Real exp_term = std::exp(Vd_limited / (n * Vt));
        Real Id_diode = Is * (exp_term - 1.0);
        Real Gd = (Is / (n * Vt)) * exp_term;
        Gd = std::max(Gd, 1e-12);

        Real Ieq_diode = Id_diode - Gd * Vdiode;

        // Stamp body diode (emitter to collector)
        if (n_emitter >= 0) {
            triplets.emplace_back(n_emitter, n_emitter, Gd);
            f(n_emitter) -= Ieq_diode;
            if (n_collector >= 0) {
                triplets.emplace_back(n_emitter, n_collector, -Gd);
            }
        }
        if (n_collector >= 0) {
            triplets.emplace_back(n_collector, n_collector, Gd);
            f(n_collector) += Ieq_diode;
            if (n_emitter >= 0) {
                triplets.emplace_back(n_collector, n_emitter, -Gd);
            }
        }
    }
}

void MNAAssembler::stamp_transformer_dc(std::vector<Triplet>& triplets, Vector& b,
                                        const Component& comp, Index branch_idx_p, Index branch_idx_s) {
    const auto& params = std::get<TransformerParams>(comp.params());

    Index n_p1 = circuit_.node_index(comp.nodes()[0]);
    Index n_p2 = circuit_.node_index(comp.nodes()[1]);
    Index n_s1 = circuit_.node_index(comp.nodes()[2]);
    Index n_s2 = circuit_.node_index(comp.nodes()[3]);

    Real n = params.turns_ratio;

    // Ideal transformer equations:
    // V1 = n * V2  (voltage relationship)
    // I1 = -I2/n  (current relationship, power conservation)
    //
    // Using MNA with branch currents I_p (primary) and I_s (secondary):
    // V_p1 - V_p2 = n * (V_s1 - V_s2)  ... but this couples primary/secondary
    //
    // Alternative formulation using gyrator-like approach:
    // V_p = n * V_s
    // n * I_p + I_s = 0

    // Primary current KCL
    if (n_p1 >= 0) {
        triplets.emplace_back(n_p1, branch_idx_p, 1.0);
        triplets.emplace_back(branch_idx_p, n_p1, 1.0);
    }
    if (n_p2 >= 0) {
        triplets.emplace_back(n_p2, branch_idx_p, -1.0);
        triplets.emplace_back(branch_idx_p, n_p2, -1.0);
    }

    // Secondary current KCL
    if (n_s1 >= 0) {
        triplets.emplace_back(n_s1, branch_idx_s, 1.0);
        triplets.emplace_back(branch_idx_s, n_s1, 1.0);
    }
    if (n_s2 >= 0) {
        triplets.emplace_back(n_s2, branch_idx_s, -1.0);
        triplets.emplace_back(branch_idx_s, n_s2, -1.0);
    }

    // Coupling equations:
    // V_p - n*V_s = 0  (row for primary branch)
    // n*I_p + I_s = 0  (row for secondary branch)

    // For DC ideal transformer, we set:
    // Row branch_idx_p: Vp1 - Vp2 - n*(Vs1 - Vs2) = 0
    // Row branch_idx_s: n*Ip + Is = 0

    // Already stamped Vp1 - Vp2 in KCL-like form above, now add coupling:
    // Primary voltage equation couples to secondary
    if (n_s1 >= 0) {
        triplets.emplace_back(branch_idx_p, n_s1, -n);
    }
    if (n_s2 >= 0) {
        triplets.emplace_back(branch_idx_p, n_s2, n);
    }

    // Current coupling: n*Ip + Is = 0
    triplets.emplace_back(branch_idx_s, branch_idx_p, n);
    triplets.emplace_back(branch_idx_s, branch_idx_s, 1.0);

    // Clear any voltage terms in secondary branch equation
    // (we override what was set in KCL-like stamps above for row branch_idx_s)
    // Actually, we need to be more careful here. Let me restructure.

    // For ideal transformer, we have 4 equations:
    // KCL at n_p1: I_p enters
    // KCL at n_p2: I_p leaves
    // KCL at n_s1: I_s enters
    // KCL at n_s2: I_s leaves
    // Branch eq for I_p: V_p1 - V_p2 - n*(V_s1 - V_s2) = 0
    // Branch eq for I_s: n*I_p + I_s = 0

    b(branch_idx_p) = 0.0;
    b(branch_idx_s) = 0.0;
}

void MNAAssembler::stamp_transformer_transient(std::vector<Triplet>& triplets, Vector& b,
                                               const Component& comp, Index branch_idx_p, Index branch_idx_s,
                                               const Vector& x_prev, Real dt) {
    const auto& params = std::get<TransformerParams>(comp.params());

    // For now, use the same as DC (ideal transformer)
    // TODO: Add magnetizing inductance model
    stamp_transformer_dc(triplets, b, comp, branch_idx_p, branch_idx_s);

    // If magnetizing inductance is specified, add it in parallel with primary
    if (params.lm > 0) {
        Index n_p1 = circuit_.node_index(comp.nodes()[0]);
        Index n_p2 = circuit_.node_index(comp.nodes()[1]);

        Real Lm = params.lm;
        Real Geq = dt / Lm;

        // Previous voltage across primary
        Real v_prev = 0.0;
        if (n_p1 >= 0) v_prev += x_prev(n_p1);
        if (n_p2 >= 0) v_prev -= x_prev(n_p2);

        // Previous magnetizing current (approximated)
        Real i_prev = x_prev(branch_idx_p);
        Real Ieq = i_prev + Geq * v_prev;

        // Stamp magnetizing inductance as shunt element
        if (n_p1 >= 0) {
            triplets.emplace_back(n_p1, n_p1, Geq);
            b(n_p1) += Ieq;
            if (n_p2 >= 0) {
                triplets.emplace_back(n_p1, n_p2, -Geq);
            }
        }
        if (n_p2 >= 0) {
            triplets.emplace_back(n_p2, n_p2, Geq);
            b(n_p2) -= Ieq;
            if (n_p1 >= 0) {
                triplets.emplace_back(n_p2, n_p1, -Geq);
            }
        }
    }
}

SwitchState* MNAAssembler::find_switch_state(const std::string& name) {
    for (auto& state : switch_states_) {
        if (state.name == name) {
            return &state;
        }
    }
    return nullptr;
}

const SwitchState* MNAAssembler::find_switch_state(const std::string& name) const {
    for (const auto& state : switch_states_) {
        if (state.name == name) {
            return &state;
        }
    }
    return nullptr;
}

void MNAAssembler::update_switch_states(const Vector& x, Real time) {
    for (const auto& comp : circuit_.components()) {
        if (comp.type() != ComponentType::Switch) continue;

        const auto& params = std::get<SwitchParams>(comp.params());
        SwitchState* state = find_switch_state(comp.name());
        if (!state) continue;

        // Get control voltage (nodes[2] - nodes[3])
        Index n_ctrl_pos = circuit_.node_index(comp.nodes()[2]);
        Index n_ctrl_neg = circuit_.node_index(comp.nodes()[3]);

        Real v_ctrl = 0.0;
        if (n_ctrl_pos >= 0) v_ctrl += x(n_ctrl_pos);
        if (n_ctrl_neg >= 0) v_ctrl -= x(n_ctrl_neg);

        bool was_closed = state->is_closed;
        state->last_control_voltage = v_ctrl;

        // Hysteresis-free comparison for now
        if (v_ctrl > params.vth && !state->is_closed) {
            state->is_closed = true;
            state->turn_on_time = time;
        } else if (v_ctrl <= params.vth && state->is_closed) {
            state->is_closed = false;
            state->turn_off_time = time;
        }

        // Track state change (for event detection)
        (void)was_closed;  // Could be used for event logging
    }
}

bool MNAAssembler::check_switch_events(const Vector& x) const {
    for (const auto& comp : circuit_.components()) {
        if (comp.type() != ComponentType::Switch) continue;

        const auto& params = std::get<SwitchParams>(comp.params());
        const SwitchState* state = find_switch_state(comp.name());
        if (!state) continue;

        // Get control voltage
        Index n_ctrl_pos = circuit_.node_index(comp.nodes()[2]);
        Index n_ctrl_neg = circuit_.node_index(comp.nodes()[3]);

        Real v_ctrl = 0.0;
        if (n_ctrl_pos >= 0) v_ctrl += x(n_ctrl_pos);
        if (n_ctrl_neg >= 0) v_ctrl -= x(n_ctrl_neg);

        // Check if state would change
        bool would_close = v_ctrl > params.vth;
        if (would_close != state->is_closed) {
            return true;  // Event detected
        }
    }
    return false;
}

void MNAAssembler::assemble_dc(SparseMatrix& G, Vector& b) {
    Index n = variable_count();
    std::vector<Triplet> triplets;
    b = Vector::Zero(n);

    for (const auto& comp : circuit_.components()) {
        switch (comp.type()) {
            case ComponentType::Resistor:
                stamp_resistor(triplets, b, comp);
                break;
            case ComponentType::Capacitor:
                stamp_capacitor_dc(triplets, b, comp);
                break;
            case ComponentType::Inductor:
                stamp_inductor_dc(triplets, b, comp, branch_indices_.at(comp.name()));
                break;
            case ComponentType::VoltageSource:
                stamp_voltage_source(triplets, b, comp, branch_indices_.at(comp.name()), 0.0);
                break;
            case ComponentType::CurrentSource:
                stamp_current_source(b, comp, 0.0);
                break;
            case ComponentType::Switch: {
                const SwitchState* state = find_switch_state(comp.name());
                if (state) {
                    stamp_switch(triplets, b, comp, *state);
                }
                break;
            }
            case ComponentType::Transformer: {
                Index branch_idx_p = branch_indices_.at(comp.name() + "_p");
                Index branch_idx_s = branch_indices_.at(comp.name() + "_s");
                stamp_transformer_dc(triplets, b, comp, branch_idx_p, branch_idx_s);
                break;
            }
            default:
                break;  // Other components handled in nonlinear assembly
        }
    }

    G.resize(n, n);
    G.setFromTriplets(triplets.begin(), triplets.end());
}

void MNAAssembler::assemble_transient(SparseMatrix& G, Vector& b,
                                      const Vector& x_prev, Real dt) {
    Index n = variable_count();
    std::vector<Triplet> triplets;
    b = Vector::Zero(n);

    for (const auto& comp : circuit_.components()) {
        switch (comp.type()) {
            case ComponentType::Resistor:
                stamp_resistor(triplets, b, comp);
                break;
            case ComponentType::Capacitor:
                stamp_capacitor_transient(triplets, b, comp, x_prev, dt);
                break;
            case ComponentType::Inductor:
                stamp_inductor_transient(triplets, b, comp,
                                        branch_indices_.at(comp.name()), x_prev, dt);
                break;
            case ComponentType::VoltageSource:
                // Time will be set in evaluate_sources
                stamp_voltage_source(triplets, b, comp, branch_indices_.at(comp.name()), 0.0);
                break;
            case ComponentType::CurrentSource:
                stamp_current_source(b, comp, 0.0);
                break;
            case ComponentType::Switch: {
                const SwitchState* state = find_switch_state(comp.name());
                if (state) {
                    stamp_switch(triplets, b, comp, *state);
                }
                break;
            }
            case ComponentType::Transformer: {
                Index branch_idx_p = branch_indices_.at(comp.name() + "_p");
                Index branch_idx_s = branch_indices_.at(comp.name() + "_s");
                stamp_transformer_transient(triplets, b, comp, branch_idx_p, branch_idx_s, x_prev, dt);
                break;
            }
            case ComponentType::Diode:
                // Stamp diode junction and diffusion capacitances
                stamp_diode_capacitance(triplets, b, comp, x_prev, x_prev, dt);
                break;
            case ComponentType::MOSFET:
                // Stamp MOSFET parasitic capacitances (Cgs, Cgd, Cds)
                stamp_mosfet_capacitances(triplets, b, comp, x_prev, dt);
                break;
            default:
                break;
        }
    }

    G.resize(n, n);
    G.setFromTriplets(triplets.begin(), triplets.end());
}

void MNAAssembler::assemble_transient(SparseMatrix& G, Vector& b,
                                      const DynamicHistory& history, Real dt,
                                      IntegrationMethod method) {
    Index n = variable_count();
    std::vector<Triplet> triplets;
    b = Vector::Zero(n);

    for (const auto& comp : circuit_.components()) {
        switch (comp.type()) {
            case ComponentType::Resistor:
                stamp_resistor(triplets, b, comp);
                break;
            case ComponentType::Capacitor:
                switch (method) {
                    case IntegrationMethod::Trapezoidal:
                    case IntegrationMethod::GEAR2:
                        stamp_capacitor_trapezoidal(triplets, b, comp, history, dt);
                        break;
                    case IntegrationMethod::BDF2:
                        stamp_capacitor_bdf2(triplets, b, comp, history, dt);
                        break;
                    case IntegrationMethod::BackwardEuler:
                    default:
                        stamp_capacitor_transient(triplets, b, comp, history.x_prev, dt);
                        break;
                }
                break;
            case ComponentType::Inductor:
                switch (method) {
                    case IntegrationMethod::Trapezoidal:
                    case IntegrationMethod::GEAR2:
                        stamp_inductor_trapezoidal(triplets, b, comp,
                                                   branch_indices_.at(comp.name()), history, dt);
                        break;
                    case IntegrationMethod::BDF2:
                        stamp_inductor_bdf2(triplets, b, comp,
                                            branch_indices_.at(comp.name()), history, dt);
                        break;
                    case IntegrationMethod::BackwardEuler:
                    default:
                        stamp_inductor_transient(triplets, b, comp,
                                                branch_indices_.at(comp.name()), history.x_prev, dt);
                        break;
                }
                break;
            case ComponentType::VoltageSource:
                stamp_voltage_source(triplets, b, comp, branch_indices_.at(comp.name()), 0.0);
                break;
            case ComponentType::CurrentSource:
                stamp_current_source(b, comp, 0.0);
                break;
            case ComponentType::Switch: {
                const SwitchState* state = find_switch_state(comp.name());
                if (state) {
                    stamp_switch(triplets, b, comp, *state);
                }
                break;
            }
            case ComponentType::Transformer: {
                Index branch_idx_p = branch_indices_.at(comp.name() + "_p");
                Index branch_idx_s = branch_indices_.at(comp.name() + "_s");
                stamp_transformer_transient(triplets, b, comp, branch_idx_p, branch_idx_s, history.x_prev, dt);
                break;
            }
            case ComponentType::Diode:
                stamp_diode_capacitance(triplets, b, comp, history.x_prev, history.x_prev, dt);
                break;
            case ComponentType::MOSFET:
                stamp_mosfet_capacitances(triplets, b, comp, history.x_prev, dt);
                break;
            default:
                break;
        }
    }

    G.resize(n, n);
    G.setFromTriplets(triplets.begin(), triplets.end());
}

void MNAAssembler::assemble_nonlinear(SparseMatrix& J, Vector& f,
                                      const Vector& x) {
    Index n = variable_count();
    std::vector<Triplet> triplets;
    f = Vector::Zero(n);

    for (const auto& comp : circuit_.components()) {
        if (comp.type() == ComponentType::Diode) {
            stamp_diode(triplets, f, comp, x);
        } else if (comp.type() == ComponentType::MOSFET) {
            stamp_mosfet(triplets, f, comp, x);
        } else if (comp.type() == ComponentType::IGBT) {
            stamp_igbt(triplets, f, comp, x);
        }
    }

    J.resize(n, n);
    J.setFromTriplets(triplets.begin(), triplets.end());
}

void MNAAssembler::evaluate_sources(Vector& b, Real time) {
    for (const auto& comp : circuit_.components()) {
        if (comp.type() == ComponentType::VoltageSource) {
            const auto& params = std::get<VoltageSourceParams>(comp.params());
            Real V = evaluate_waveform(params.waveform, time);
            Index branch_idx = branch_indices_.at(comp.name());
            b(branch_idx) = V;
        }
        else if (comp.type() == ComponentType::CurrentSource) {
            const auto& params = std::get<CurrentSourceParams>(comp.params());
            Real I = evaluate_waveform(params.waveform, time);
            Index n1 = circuit_.node_index(comp.nodes()[0]);
            Index n2 = circuit_.node_index(comp.nodes()[1]);
            if (n1 >= 0) b(n1) += I;
            if (n2 >= 0) b(n2) -= I;
        }
    }
}

}  // namespace pulsim
