#include "spicelab/mna.hpp"
#include <cmath>
#include <stdexcept>

namespace spicelab {

MNAAssembler::MNAAssembler(const Circuit& circuit)
    : circuit_(circuit) {

    // Assign branch indices for voltage sources and inductors
    next_branch_idx_ = circuit_.node_count();
    for (const auto& comp : circuit_.components()) {
        if (comp.has_branch_current()) {
            branch_indices_[comp.name()] = next_branch_idx_++;
        }
        if (comp.type() == ComponentType::Diode) {
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

}  // namespace spicelab
