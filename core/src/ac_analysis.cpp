#include "pulsim/ac_analysis.hpp"
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <iostream>

namespace pulsim {

// =============================================================================
// ACResult Implementation
// =============================================================================

Real ACResult::magnitude(size_t freq_idx, size_t signal_idx) const {
    if (freq_idx >= data.size() || signal_idx >= static_cast<size_t>(data[freq_idx].size())) {
        return 0.0;
    }
    return std::abs(data[freq_idx](signal_idx));
}

Real ACResult::phase_deg(size_t freq_idx, size_t signal_idx) const {
    if (freq_idx >= data.size() || signal_idx >= static_cast<size_t>(data[freq_idx].size())) {
        return 0.0;
    }
    return std::arg(data[freq_idx](signal_idx)) * 180.0 / M_PI;
}

Real ACResult::magnitude_db(size_t freq_idx, size_t signal_idx) const {
    Real mag = magnitude(freq_idx, signal_idx);
    if (mag <= 0) return -200.0;  // Floor for log
    return 20.0 * std::log10(mag);
}

Real ACResult::transfer_magnitude_db(size_t freq_idx, size_t output_idx, size_t input_idx) const {
    if (freq_idx >= data.size()) return -200.0;

    Complex out = data[freq_idx](output_idx);
    Complex in = data[freq_idx](input_idx);

    if (std::abs(in) < 1e-30) return -200.0;

    Real mag = std::abs(out / in);
    if (mag <= 0) return -200.0;
    return 20.0 * std::log10(mag);
}

Real ACResult::transfer_phase_deg(size_t freq_idx, size_t output_idx, size_t input_idx) const {
    if (freq_idx >= data.size()) return 0.0;

    Complex out = data[freq_idx](output_idx);
    Complex in = data[freq_idx](input_idx);

    if (std::abs(in) < 1e-30) return 0.0;

    return std::arg(out / in) * 180.0 / M_PI;
}

// =============================================================================
// ACOptions Implementation
// =============================================================================

std::vector<Real> ACOptions::generate_frequencies() const {
    std::vector<Real> freqs;

    switch (sweep_type) {
        case FrequencySweepType::Linear: {
            if (npoints < 1) return freqs;
            freqs.reserve(npoints);
            for (int i = 0; i < npoints; ++i) {
                Real f = fstart + i * (fstop - fstart) / (npoints - 1);
                freqs.push_back(f);
            }
            break;
        }

        case FrequencySweepType::Decade: {
            if (fstart <= 0 || fstop <= 0 || npoints < 1) return freqs;
            Real log_start = std::log10(fstart);
            Real log_stop = std::log10(fstop);
            int num_decades = static_cast<int>(std::ceil(log_stop - log_start));
            int total_points = num_decades * npoints + 1;
            freqs.reserve(total_points);

            for (Real log_f = log_start; log_f <= log_stop + 1e-10; log_f += 1.0 / npoints) {
                freqs.push_back(std::pow(10.0, log_f));
            }
            break;
        }

        case FrequencySweepType::Octave: {
            if (fstart <= 0 || fstop <= 0 || npoints < 1) return freqs;
            Real log2_start = std::log2(fstart);
            Real log2_stop = std::log2(fstop);
            int num_octaves = static_cast<int>(std::ceil(log2_stop - log2_start));
            int total_points = num_octaves * npoints + 1;
            freqs.reserve(total_points);

            for (Real log2_f = log2_start; log2_f <= log2_stop + 1e-10; log2_f += 1.0 / npoints) {
                freqs.push_back(std::pow(2.0, log2_f));
            }
            break;
        }

        case FrequencySweepType::List:
            freqs = frequency_list;
            break;
    }

    return freqs;
}

// =============================================================================
// ACAnalyzer Implementation
// =============================================================================

ACAnalyzer::ACAnalyzer(const Circuit& circuit)
    : circuit_(circuit)
{
}

void ACAnalyzer::set_operating_point(const Vector& x_op) {
    x_op_ = x_op;
    has_operating_point_ = true;
    model_valid_ = false;  // Invalidate cached model
}

void ACAnalyzer::build_linearized_model(SparseMatrix& G_lin, SparseMatrix& C_lin) {
    Index n = circuit_.total_variables();
    Index num_nodes = circuit_.node_count();

    std::vector<Triplet> G_triplets;
    std::vector<Triplet> C_triplets;

    G_triplets.reserve(n * 4);
    C_triplets.reserve(n * 2);

    Index branch_idx = num_nodes;

    for (const auto& comp : circuit_.components()) {
        switch (comp.type()) {
            case ComponentType::Resistor: {
                const auto& params = std::get<ResistorParams>(comp.params());
                Real G = 1.0 / params.resistance;

                Index n1 = circuit_.node_index(comp.nodes()[0]);
                Index n2 = circuit_.node_index(comp.nodes()[1]);

                if (n1 >= 0) G_triplets.emplace_back(n1, n1, G);
                if (n2 >= 0) G_triplets.emplace_back(n2, n2, G);
                if (n1 >= 0 && n2 >= 0) {
                    G_triplets.emplace_back(n1, n2, -G);
                    G_triplets.emplace_back(n2, n1, -G);
                }
                break;
            }

            case ComponentType::Capacitor:
                stamp_capacitor_ac(comp, C_triplets);
                break;

            case ComponentType::Inductor:
                stamp_inductor_ac(comp, G_triplets, C_triplets);
                branch_idx++;
                break;

            case ComponentType::VoltageSource: {
                // Voltage source: V_n1 - V_n2 = V_source
                // In AC analysis, we set up for AC source magnitude
                Index n1 = circuit_.node_index(comp.nodes()[0]);
                Index n2 = circuit_.node_index(comp.nodes()[1]);

                if (n1 >= 0) {
                    G_triplets.emplace_back(n1, branch_idx, 1.0);
                    G_triplets.emplace_back(branch_idx, n1, 1.0);
                }
                if (n2 >= 0) {
                    G_triplets.emplace_back(n2, branch_idx, -1.0);
                    G_triplets.emplace_back(branch_idx, n2, -1.0);
                }
                branch_idx++;
                break;
            }

            case ComponentType::CurrentSource:
                // Current sources appear in the RHS, not the matrix
                break;

            case ComponentType::Diode:
                linearize_diode(comp, G_triplets, C_triplets);
                break;

            case ComponentType::MOSFET:
                linearize_mosfet(comp, G_triplets, C_triplets);
                break;

            case ComponentType::IGBT:
                linearize_igbt(comp, G_triplets, C_triplets);
                break;

            case ComponentType::Switch: {
                // Switch: linearize as resistance at operating point
                const auto& params = std::get<SwitchParams>(comp.params());
                Index n1 = circuit_.node_index(comp.nodes()[0]);
                Index n2 = circuit_.node_index(comp.nodes()[1]);

                // Determine switch state from operating point
                bool is_closed = false;
                if (has_operating_point_) {
                    Index n_ctrl_pos = circuit_.node_index(comp.nodes()[2]);
                    Index n_ctrl_neg = circuit_.node_index(comp.nodes()[3]);
                    Real v_ctrl = 0.0;
                    if (n_ctrl_pos >= 0) v_ctrl += x_op_(n_ctrl_pos);
                    if (n_ctrl_neg >= 0) v_ctrl -= x_op_(n_ctrl_neg);
                    is_closed = v_ctrl > params.vth;
                }

                Real R = is_closed ? params.ron : params.roff;
                Real G = 1.0 / R;

                if (n1 >= 0) G_triplets.emplace_back(n1, n1, G);
                if (n2 >= 0) G_triplets.emplace_back(n2, n2, G);
                if (n1 >= 0 && n2 >= 0) {
                    G_triplets.emplace_back(n1, n2, -G);
                    G_triplets.emplace_back(n2, n1, -G);
                }
                break;
            }

            case ComponentType::Transformer: {
                // Ideal transformer: V1 = n * V2, I1 = -I2 / n
                // Modeled with coupled inductors for AC analysis
                const auto& params = std::get<TransformerParams>(comp.params());

                Index p1 = circuit_.node_index(comp.nodes()[0]);
                Index p2 = circuit_.node_index(comp.nodes()[1]);
                [[maybe_unused]] Index s1 = circuit_.node_index(comp.nodes()[2]);
                [[maybe_unused]] Index s2 = circuit_.node_index(comp.nodes()[3]);

                // For ideal transformer: add constraint equations
                // V_p1 - V_p2 = n * (V_s1 - V_s2)
                // This requires additional branch variables for currents

                // Simplified: model as coupled inductors with very high inductance
                // Or use gyrator model

                // For now, add magnetizing inductance as a simple inductor
                if (params.lm > 0) {
                    Real L = params.lm;
                    // Add L to C matrix as sL term
                    // Using branch variable for inductor current
                    Index br = branch_idx;

                    if (p1 >= 0) {
                        G_triplets.emplace_back(p1, br, 1.0);
                        G_triplets.emplace_back(br, p1, 1.0);
                    }
                    if (p2 >= 0) {
                        G_triplets.emplace_back(p2, br, -1.0);
                        G_triplets.emplace_back(br, p2, -1.0);
                    }
                    C_triplets.emplace_back(br, br, -L);
                    branch_idx++;
                }
                break;
            }

            default:
                break;
        }
    }

    // Build sparse matrices
    G_lin.resize(n, n);
    C_lin.resize(n, n);
    G_lin.setFromTriplets(G_triplets.begin(), G_triplets.end());
    C_lin.setFromTriplets(C_triplets.begin(), C_triplets.end());

    // Cache the model
    G_lin_ = G_lin;
    C_lin_ = C_lin;
    model_valid_ = true;
}

void ACAnalyzer::stamp_capacitor_ac(const Component& comp, std::vector<Triplet>& C_triplets) {
    const auto& params = std::get<CapacitorParams>(comp.params());
    Real C = params.capacitance;

    Index n1 = circuit_.node_index(comp.nodes()[0]);
    Index n2 = circuit_.node_index(comp.nodes()[1]);

    // Capacitor admittance: Y = sC
    // Contribution to C matrix
    if (n1 >= 0) C_triplets.emplace_back(n1, n1, C);
    if (n2 >= 0) C_triplets.emplace_back(n2, n2, C);
    if (n1 >= 0 && n2 >= 0) {
        C_triplets.emplace_back(n1, n2, -C);
        C_triplets.emplace_back(n2, n1, -C);
    }
}

void ACAnalyzer::stamp_inductor_ac(const Component& comp, std::vector<Triplet>& G_triplets,
                                   std::vector<Triplet>& C_triplets) {
    const auto& params = std::get<InductorParams>(comp.params());
    Real L = params.inductance;

    Index n1 = circuit_.node_index(comp.nodes()[0]);
    Index n2 = circuit_.node_index(comp.nodes()[1]);
    Index num_nodes = circuit_.node_count();

    // Find branch index for this inductor
    Index branch_idx = num_nodes;
    for (const auto& c : circuit_.components()) {
        if (c.name() == comp.name()) break;
        if (c.type() == ComponentType::VoltageSource || c.type() == ComponentType::Inductor) {
            branch_idx++;
        }
    }

    // Inductor impedance: Z = sL, Y = 1/(sL)
    // Using branch current formulation:
    // V_n1 - V_n2 = sL * I_L
    // KCL: I into n1 = I_L, I into n2 = -I_L

    if (n1 >= 0) {
        G_triplets.emplace_back(n1, branch_idx, 1.0);
        G_triplets.emplace_back(branch_idx, n1, 1.0);
    }
    if (n2 >= 0) {
        G_triplets.emplace_back(n2, branch_idx, -1.0);
        G_triplets.emplace_back(branch_idx, n2, -1.0);
    }
    // -sL * I_L + (V_n1 - V_n2) = 0
    // The L term goes into the C matrix
    C_triplets.emplace_back(branch_idx, branch_idx, -L);
}

void ACAnalyzer::linearize_diode(const Component& comp, std::vector<Triplet>& G_triplets,
                                 std::vector<Triplet>& C_triplets) {
    const auto& params = std::get<DiodeParams>(comp.params());

    Index n_anode = circuit_.node_index(comp.nodes()[0]);
    Index n_cathode = circuit_.node_index(comp.nodes()[1]);

    // Get operating point voltage
    Real Vd = 0.0;
    if (has_operating_point_) {
        if (n_anode >= 0) Vd += x_op_(n_anode);
        if (n_cathode >= 0) Vd -= x_op_(n_cathode);
    }

    // Small-signal conductance: gd = dI/dV at operating point
    Real gd = 1e-12;  // Minimum conductance
    if (!params.ideal) {
        Real Vt = params.vt;
        Real Is = params.is;
        Real n = params.n;
        Real Vd_limited = std::min(Vd, 40.0 * n * Vt);
        // I = Is * (exp(Vd/(n*Vt)) - 1)
        // gd = dI/dVd = Is/(n*Vt) * exp(Vd/(n*Vt))
        gd = Is / (n * Vt) * std::exp(Vd_limited / (n * Vt));
        gd = std::max(gd, 1e-12);  // Floor
    } else {
        // Ideal diode: high conductance if forward biased
        gd = (Vd > 0) ? 1e3 : 1e-12;
    }

    // Stamp small-signal conductance
    if (n_anode >= 0) G_triplets.emplace_back(n_anode, n_anode, gd);
    if (n_cathode >= 0) G_triplets.emplace_back(n_cathode, n_cathode, gd);
    if (n_anode >= 0 && n_cathode >= 0) {
        G_triplets.emplace_back(n_anode, n_cathode, -gd);
        G_triplets.emplace_back(n_cathode, n_anode, -gd);
    }

    // Junction capacitance (if specified)
    if (params.cj0 > 0) {
        // Cj = Cj0 / (1 - Vd/Vj)^m for Vd < Fc*Vj
        Real Fc = 0.5;  // Forward-bias coefficient
        Real Cj;
        if (Vd < Fc * params.vj) {
            Cj = params.cj0 / std::pow(1.0 - Vd / params.vj, params.m);
        } else {
            // Linear extrapolation for forward bias
            Cj = params.cj0 / std::pow(1.0 - Fc, params.m) *
                 (1.0 + params.m * (Vd - Fc * params.vj) / (params.vj * (1.0 - Fc)));
        }

        if (n_anode >= 0) C_triplets.emplace_back(n_anode, n_anode, Cj);
        if (n_cathode >= 0) C_triplets.emplace_back(n_cathode, n_cathode, Cj);
        if (n_anode >= 0 && n_cathode >= 0) {
            C_triplets.emplace_back(n_anode, n_cathode, -Cj);
            C_triplets.emplace_back(n_cathode, n_anode, -Cj);
        }
    }

    // Diffusion capacitance
    if (params.tt > 0 && Vd > 0) {
        // Cd = tt * gd
        Real Cd = params.tt * gd;
        if (n_anode >= 0) C_triplets.emplace_back(n_anode, n_anode, Cd);
        if (n_cathode >= 0) C_triplets.emplace_back(n_cathode, n_cathode, Cd);
        if (n_anode >= 0 && n_cathode >= 0) {
            C_triplets.emplace_back(n_anode, n_cathode, -Cd);
            C_triplets.emplace_back(n_cathode, n_anode, -Cd);
        }
    }
}

void ACAnalyzer::linearize_mosfet(const Component& comp, std::vector<Triplet>& G_triplets,
                                  std::vector<Triplet>& C_triplets) {
    const auto& params = std::get<MOSFETParams>(comp.params());

    Index n_drain = circuit_.node_index(comp.nodes()[0]);
    Index n_gate = circuit_.node_index(comp.nodes()[1]);
    Index n_source = circuit_.node_index(comp.nodes()[2]);

    // Get operating point voltages
    Real Vd = 0, Vg = 0, Vs = 0;
    if (has_operating_point_) {
        if (n_drain >= 0) Vd = x_op_(n_drain);
        if (n_gate >= 0) Vg = x_op_(n_gate);
        if (n_source >= 0) Vs = x_op_(n_source);
    }

    Real Vgs = Vg - Vs;
    Real Vds = Vd - Vs;

    // Handle PMOS
    Real sign = (params.type == MOSFETType::NMOS) ? 1.0 : -1.0;
    Vgs *= sign;
    Vds *= sign;

    // Small-signal parameters: gm (transconductance), gds (output conductance)
    Real gm = 0.0, gds = 1e-12;

    if (params.rds_on > 0) {
        // Simple switch model
        if (Vgs > params.vth) {
            gds = 1.0 / params.rds_on;
        } else {
            gds = 1.0 / params.rds_off;
        }
    } else {
        // Level 1 model
        Real Kp = params.kp_effective();
        Real Vov = Vgs - params.vth;

        if (Vov > 0) {
            if (Vds < Vov) {
                // Linear region
                // Id = Kp * (Vov * Vds - 0.5 * Vds^2)
                gm = Kp * Vds;  // dId/dVgs
                gds = Kp * (Vov - Vds);  // dId/dVds
            } else {
                // Saturation
                // Id = 0.5 * Kp * Vov^2 * (1 + lambda * Vds)
                gm = Kp * Vov * (1.0 + params.lambda * Vds);
                gds = 0.5 * Kp * Vov * Vov * params.lambda;
            }
        }
    }

    // Ensure minimum conductances
    gm = std::max(gm, 0.0);
    gds = std::max(gds, 1e-12);

    // Small-signal model (simplified):
    // id = gm * vgs + gds * vds
    // where vgs = vg - vs, vds = vd - vs

    // gm contribution: current from drain node proportional to gate-source voltage
    if (n_drain >= 0 && n_gate >= 0) {
        G_triplets.emplace_back(n_drain, n_gate, sign * gm);
    }
    if (n_drain >= 0 && n_source >= 0) {
        G_triplets.emplace_back(n_drain, n_source, -sign * gm);
    }
    if (n_source >= 0 && n_gate >= 0) {
        G_triplets.emplace_back(n_source, n_gate, -sign * gm);
    }
    if (n_source >= 0) {
        G_triplets.emplace_back(n_source, n_source, sign * gm);
    }

    // gds contribution: conductance between drain and source
    if (n_drain >= 0) G_triplets.emplace_back(n_drain, n_drain, gds);
    if (n_source >= 0) G_triplets.emplace_back(n_source, n_source, gds);
    if (n_drain >= 0 && n_source >= 0) {
        G_triplets.emplace_back(n_drain, n_source, -gds);
        G_triplets.emplace_back(n_source, n_drain, -gds);
    }

    // Parasitic capacitances
    if (params.cgs > 0 && n_gate >= 0 && n_source >= 0) {
        C_triplets.emplace_back(n_gate, n_gate, params.cgs);
        C_triplets.emplace_back(n_source, n_source, params.cgs);
        C_triplets.emplace_back(n_gate, n_source, -params.cgs);
        C_triplets.emplace_back(n_source, n_gate, -params.cgs);
    }
    if (params.cgd > 0 && n_gate >= 0 && n_drain >= 0) {
        C_triplets.emplace_back(n_gate, n_gate, params.cgd);
        C_triplets.emplace_back(n_drain, n_drain, params.cgd);
        C_triplets.emplace_back(n_gate, n_drain, -params.cgd);
        C_triplets.emplace_back(n_drain, n_gate, -params.cgd);
    }
    if (params.cds > 0 && n_drain >= 0 && n_source >= 0) {
        C_triplets.emplace_back(n_drain, n_drain, params.cds);
        C_triplets.emplace_back(n_source, n_source, params.cds);
        C_triplets.emplace_back(n_drain, n_source, -params.cds);
        C_triplets.emplace_back(n_source, n_drain, -params.cds);
    }
}

void ACAnalyzer::linearize_igbt(const Component& comp, std::vector<Triplet>& G_triplets,
                                std::vector<Triplet>& C_triplets) {
    const auto& params = std::get<IGBTParams>(comp.params());

    Index n_collector = circuit_.node_index(comp.nodes()[0]);
    Index n_gate = circuit_.node_index(comp.nodes()[1]);
    Index n_emitter = circuit_.node_index(comp.nodes()[2]);

    // Get operating point
    Real Vc = 0, Vg = 0, Ve = 0;
    if (has_operating_point_) {
        if (n_collector >= 0) Vc = x_op_(n_collector);
        if (n_gate >= 0) Vg = x_op_(n_gate);
        if (n_emitter >= 0) Ve = x_op_(n_emitter);
    }

    Real Vge = Vg - Ve;
    Real Vce = Vc - Ve;

    // Small-signal conductance
    Real gce = 1e-12;

    if (Vge > params.vth && Vce > 0) {
        // IGBT is on
        gce = 1.0 / params.rce_on;
    } else {
        gce = 1.0 / params.rce_off;
    }

    // Output conductance
    if (n_collector >= 0) G_triplets.emplace_back(n_collector, n_collector, gce);
    if (n_emitter >= 0) G_triplets.emplace_back(n_emitter, n_emitter, gce);
    if (n_collector >= 0 && n_emitter >= 0) {
        G_triplets.emplace_back(n_collector, n_emitter, -gce);
        G_triplets.emplace_back(n_emitter, n_collector, -gce);
    }

    // Input capacitance
    if (params.cies > 0 && n_gate >= 0 && n_emitter >= 0) {
        C_triplets.emplace_back(n_gate, n_gate, params.cies);
        C_triplets.emplace_back(n_emitter, n_emitter, params.cies);
        C_triplets.emplace_back(n_gate, n_emitter, -params.cies);
        C_triplets.emplace_back(n_emitter, n_gate, -params.cies);
    }
}

void ACAnalyzer::build_admittance_matrix(ComplexSparseMatrix& Y, Real frequency) {
    if (!model_valid_) {
        SparseMatrix G, C;
        build_linearized_model(G, C);
    }

    Real omega = 2.0 * M_PI * frequency;
    Complex jw(0.0, omega);

    Index n = G_lin_.rows();
    Y.resize(n, n);

    // Y = G + jw*C
    std::vector<Eigen::Triplet<Complex>> Y_triplets;
    Y_triplets.reserve(G_lin_.nonZeros() + C_lin_.nonZeros());

    // Add G terms
    for (int k = 0; k < G_lin_.outerSize(); ++k) {
        for (SparseMatrix::InnerIterator it(G_lin_, k); it; ++it) {
            Y_triplets.emplace_back(it.row(), it.col(), Complex(it.value(), 0.0));
        }
    }

    // Add jw*C terms
    for (int k = 0; k < C_lin_.outerSize(); ++k) {
        for (SparseMatrix::InnerIterator it(C_lin_, k); it; ++it) {
            Y_triplets.emplace_back(it.row(), it.col(), jw * it.value());
        }
    }

    Y.setFromTriplets(Y_triplets.begin(), Y_triplets.end());
}

Index ACAnalyzer::find_source_index(const std::string& source_name) const {
    Index branch_idx = circuit_.node_count();
    for (const auto& comp : circuit_.components()) {
        if (comp.name() == source_name) {
            if (comp.type() == ComponentType::VoltageSource ||
                comp.type() == ComponentType::Inductor) {
                return branch_idx;
            }
            return -1;  // Not a branch variable
        }
        if (comp.type() == ComponentType::VoltageSource ||
            comp.type() == ComponentType::Inductor) {
            branch_idx++;
        }
    }
    return -1;  // Not found
}

ACPoint ACAnalyzer::analyze_at_frequency(Real frequency) {
    ACPoint result;
    result.frequency = frequency;

    // Build admittance matrix
    ComplexSparseMatrix Y;
    build_admittance_matrix(Y, frequency);

    Index n = Y.rows();

    // Build excitation vector (AC sources)
    ComplexVector b = ComplexVector::Zero(n);

    Index branch_idx = circuit_.node_count();
    for (const auto& comp : circuit_.components()) {
        if (comp.type() == ComponentType::VoltageSource) {
            // For AC analysis, voltage sources contribute to RHS
            // Set AC magnitude to 1.0 for the first voltage source (as reference)
            // This can be customized later
            const auto& params = std::get<VoltageSourceParams>(comp.params());

            // Get DC value or sine amplitude as AC magnitude
            Real ac_mag = 1.0;
            if (std::holds_alternative<SineWaveform>(params.waveform)) {
                ac_mag = std::get<SineWaveform>(params.waveform).amplitude;
            } else if (std::holds_alternative<DCWaveform>(params.waveform)) {
                ac_mag = std::get<DCWaveform>(params.waveform).value;
            }

            b(branch_idx) = Complex(ac_mag, 0.0);
            branch_idx++;
        } else if (comp.type() == ComponentType::Inductor) {
            branch_idx++;
        } else if (comp.type() == ComponentType::CurrentSource) {
            // Current sources contribute directly to node equations
            const auto& params = std::get<CurrentSourceParams>(comp.params());
            Index n_pos = circuit_.node_index(comp.nodes()[0]);
            Index n_neg = circuit_.node_index(comp.nodes()[1]);

            Real ac_mag = 1.0;
            if (std::holds_alternative<SineWaveform>(params.waveform)) {
                ac_mag = std::get<SineWaveform>(params.waveform).amplitude;
            } else if (std::holds_alternative<DCWaveform>(params.waveform)) {
                ac_mag = std::get<DCWaveform>(params.waveform).value;
            }

            if (n_pos >= 0) b(n_pos) -= Complex(ac_mag, 0.0);
            if (n_neg >= 0) b(n_neg) += Complex(ac_mag, 0.0);
        }
    }

    // Solve Y * x = b
    // Use Eigen's SparseLU for complex matrices
    Eigen::SparseLU<ComplexSparseMatrix> solver;
    solver.analyzePattern(Y);
    solver.factorize(Y);

    if (solver.info() != Eigen::Success) {
        // Singular matrix
        result.response = ComplexVector::Zero(n);
        return result;
    }

    result.response = solver.solve(b);

    return result;
}

ACResult ACAnalyzer::analyze(const ACOptions& options) {
    ACResult result;

    // Generate frequency points
    result.frequencies = options.generate_frequencies();

    if (result.frequencies.empty()) {
        result.status = SolverStatus::NumericalError;
        result.error_message = "No frequencies generated";
        return result;
    }

    // Build signal names
    Index n = circuit_.total_variables();
    for (Index i = 0; i < n; ++i) {
        result.signal_names.push_back(circuit_.signal_name(i));
    }

    // Analyze at each frequency
    result.data.reserve(result.frequencies.size());

    for (Real freq : result.frequencies) {
        ACPoint point = analyze_at_frequency(freq);
        result.data.push_back(point.response);
    }

    result.status = SolverStatus::Success;
    return result;
}

// =============================================================================
// Bode Plot Functions
// =============================================================================

BodeData extract_bode_data(const ACResult& result, size_t output_idx, size_t input_idx) {
    BodeData bode;

    bode.frequencies.reserve(result.num_frequencies());
    bode.magnitude_db.reserve(result.num_frequencies());
    bode.phase_deg.reserve(result.num_frequencies());

    for (size_t i = 0; i < result.num_frequencies(); ++i) {
        bode.frequencies.push_back(result.frequencies[i]);
        bode.magnitude_db.push_back(result.transfer_magnitude_db(i, output_idx, input_idx));
        bode.phase_deg.push_back(result.transfer_phase_deg(i, output_idx, input_idx));
    }

    // Calculate stability margins
    calculate_stability_margins(bode);

    return bode;
}

BodeData extract_bode_data(const ACResult& result, const std::string& output_signal,
                           const std::string& input_signal) {
    // Find signal indices
    size_t output_idx = 0, input_idx = 0;
    bool found_output = false, found_input = false;

    for (size_t i = 0; i < result.signal_names.size(); ++i) {
        if (result.signal_names[i] == output_signal) {
            output_idx = i;
            found_output = true;
        }
        if (result.signal_names[i] == input_signal) {
            input_idx = i;
            found_input = true;
        }
    }

    if (!found_output || !found_input) {
        return BodeData{};  // Return empty data
    }

    return extract_bode_data(result, output_idx, input_idx);
}

void calculate_stability_margins(BodeData& bode) {
    if (bode.frequencies.size() < 2) return;

    // Find gain crossover frequency (where magnitude = 0 dB)
    for (size_t i = 1; i < bode.frequencies.size(); ++i) {
        Real m1 = bode.magnitude_db[i - 1];
        Real m2 = bode.magnitude_db[i];

        // Check for crossing 0 dB
        if ((m1 >= 0 && m2 < 0) || (m1 < 0 && m2 >= 0)) {
            // Linear interpolation to find crossing
            Real f1 = bode.frequencies[i - 1];
            Real f2 = bode.frequencies[i];
            Real t = (0.0 - m1) / (m2 - m1);
            bode.gain_crossover_freq = f1 + t * (f2 - f1);

            // Phase margin = phase at gain crossover + 180°
            Real p1 = bode.phase_deg[i - 1];
            Real p2 = bode.phase_deg[i];
            Real phase_at_crossover = p1 + t * (p2 - p1);
            bode.phase_margin_deg = phase_at_crossover + 180.0;
            break;
        }
    }

    // Find phase crossover frequency (where phase = -180°)
    for (size_t i = 1; i < bode.frequencies.size(); ++i) {
        Real p1 = bode.phase_deg[i - 1];
        Real p2 = bode.phase_deg[i];

        // Check for crossing -180°
        if ((p1 >= -180 && p2 < -180) || (p1 < -180 && p2 >= -180)) {
            // Linear interpolation
            Real f1 = bode.frequencies[i - 1];
            Real f2 = bode.frequencies[i];
            Real t = (-180.0 - p1) / (p2 - p1);
            bode.phase_crossover_freq = f1 + t * (f2 - f1);

            // Gain margin = -magnitude at phase crossover
            Real m1 = bode.magnitude_db[i - 1];
            Real m2 = bode.magnitude_db[i];
            Real mag_at_crossover = m1 + t * (m2 - m1);
            bode.gain_margin_db = -mag_at_crossover;
            break;
        }
    }
}

// =============================================================================
// Convenience Functions
// =============================================================================

ACResult ac_analysis(const Circuit& circuit, const ACOptions& options,
                     const Vector& operating_point) {
    ACAnalyzer analyzer(circuit);

    if (operating_point.size() > 0) {
        analyzer.set_operating_point(operating_point);
    }

    return analyzer.analyze(options);
}

}  // namespace pulsim
