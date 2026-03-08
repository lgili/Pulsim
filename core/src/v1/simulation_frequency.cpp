/**
 * @file simulation_frequency.cpp
 * @brief Frequency-domain AC sweep implementation for the v1 runtime.
 */

#include "pulsim/v1/simulation.hpp"

#include <Eigen/SparseLU>

#include <cmath>
#include <complex>
#include <limits>
#include <numbers>
#include <optional>
#include <sstream>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace pulsim::v1 {

namespace {

using Complex = std::complex<Real>;
using ComplexVector = Eigen::Matrix<Complex, Eigen::Dynamic, 1>;
using ComplexSparseMatrix = Eigen::SparseMatrix<Complex>;

[[nodiscard]] bool nearly_equal(Real lhs, Real rhs, Real eps = 1e-15) {
    const Real scale = std::max<Real>({Real{1.0}, std::abs(lhs), std::abs(rhs)});
    return std::abs(lhs - rhs) <= eps * scale;
}

[[nodiscard]] std::optional<Real> interpolate_crossing(
    const std::vector<Real>& x,
    const std::vector<Real>& y,
    Real target,
    std::optional<std::reference_wrapper<const std::vector<Real>>> coupled,
    Real* coupled_value) {
    if (x.size() < 2 || y.size() != x.size()) {
        return std::nullopt;
    }

    for (std::size_t i = 1; i < x.size(); ++i) {
        const Real y0 = y[i - 1] - target;
        const Real y1 = y[i] - target;

        if (nearly_equal(y[i - 1], target)) {
            if (coupled && coupled_value != nullptr) {
                const auto& c = coupled->get();
                if (c.size() == y.size()) {
                    *coupled_value = c[i - 1];
                }
            }
            return x[i - 1];
        }

        if ((y0 < 0.0 && y1 > 0.0) || (y0 > 0.0 && y1 < 0.0) || nearly_equal(y[i], target)) {
            const Real denom = y[i] - y[i - 1];
            const Real alpha = nearly_equal(denom, 0.0) ? 0.0 : (target - y[i - 1]) / denom;
            const Real x_cross = x[i - 1] + alpha * (x[i] - x[i - 1]);

            if (coupled && coupled_value != nullptr) {
                const auto& c = coupled->get();
                if (c.size() == y.size()) {
                    *coupled_value = c[i - 1] + alpha * (c[i] - c[i - 1]);
                }
            }
            return x_cross;
        }
    }

    return std::nullopt;
}

[[nodiscard]] std::vector<Real> unwrap_phase_deg(const std::vector<Real>& phase_deg) {
    if (phase_deg.empty()) {
        return {};
    }
    std::vector<Real> out = phase_deg;
    Real offset = 0.0;
    for (std::size_t i = 1; i < out.size(); ++i) {
        const Real delta = out[i] - out[i - 1];
        if (delta > 180.0) {
            offset -= 360.0;
        } else if (delta < -180.0) {
            offset += 360.0;
        }
        out[i] += offset;
    }
    return out;
}

[[nodiscard]] std::string_view diag_reason(SimulationDiagnosticCode code) {
    switch (code) {
        case SimulationDiagnosticCode::FrequencyInvalidConfiguration:
            return "frequency_invalid_configuration";
        case SimulationDiagnosticCode::FrequencyUnsupportedConfiguration:
            return "frequency_unsupported_configuration";
        case SimulationDiagnosticCode::FrequencySolverFailure:
            return "frequency_solver_failure";
        default:
            return "";
    }
}

[[nodiscard]] std::vector<Real> build_frequency_grid(const FrequencyAnalysisOptions& options) {
    std::vector<Real> freq;
    if (options.points <= 0) {
        return freq;
    }
    freq.resize(static_cast<std::size_t>(options.points));

    if (options.points == 1) {
        freq[0] = options.f_start_hz;
        return freq;
    }

    if (options.sweep_scale == FrequencySweepScale::Linear) {
        const Real step = (options.f_stop_hz - options.f_start_hz) /
            static_cast<Real>(options.points - 1);
        for (int i = 0; i < options.points; ++i) {
            freq[static_cast<std::size_t>(i)] = options.f_start_hz + step * static_cast<Real>(i);
        }
        return freq;
    }

    const Real log_start = std::log10(options.f_start_hz);
    const Real log_stop = std::log10(options.f_stop_hz);
    const Real step = (log_stop - log_start) / static_cast<Real>(options.points - 1);
    for (int i = 0; i < options.points; ++i) {
        freq[static_cast<std::size_t>(i)] = std::pow(Real{10.0}, log_start + step * static_cast<Real>(i));
    }
    return freq;
}

[[nodiscard]] std::optional<Index> resolve_node_index(const Circuit& circuit, const std::string& name) {
    if (name.empty()) {
        return std::nullopt;
    }
    try {
        return circuit.get_node(name);
    } catch (...) {
        return std::nullopt;
    }
}

[[nodiscard]] Complex node_voltage_complex(const ComplexVector& x, Index node) {
    if (node < 0) {
        return Complex{0.0, 0.0};
    }
    return x[node];
}

[[nodiscard]] std::string mode_result_name(FrequencyAnalysisMode mode) {
    switch (mode) {
        case FrequencyAnalysisMode::OpenLoopTransfer:
            return "open_loop_transfer";
        case FrequencyAnalysisMode::ClosedLoopTransfer:
            return "closed_loop_transfer";
        case FrequencyAnalysisMode::InputImpedance:
            return "input_impedance";
        case FrequencyAnalysisMode::OutputImpedance:
            return "output_impedance";
    }
    return "frequency_response";
}

[[nodiscard]] std::string_view anchor_mode_name(FrequencyAnchorMode mode) {
    switch (mode) {
        case FrequencyAnchorMode::DC:
            return "dc";
        case FrequencyAnchorMode::Periodic:
            return "periodic";
        case FrequencyAnchorMode::Averaged:
            return "averaged";
        case FrequencyAnchorMode::Auto:
            return "auto";
    }
    return "unknown";
}

[[nodiscard]] VirtualChannelMetadata make_frequency_metadata() {
    VirtualChannelMetadata metadata;
    metadata.component_type = "frequency_analysis";
    metadata.component_name = "frequency_axis";
    metadata.source_component = "sweep";
    metadata.domain = "frequency";
    metadata.unit = "Hz";
    return metadata;
}

[[nodiscard]] VirtualChannelMetadata make_response_metadata(const std::string& response_name,
                                                            const std::string& unit,
                                                            std::vector<Index> nodes) {
    VirtualChannelMetadata metadata;
    metadata.component_type = "frequency_analysis";
    metadata.component_name = response_name;
    metadata.source_component = "ac_sweep";
    metadata.domain = "frequency";
    metadata.unit = unit;
    metadata.nodes = std::move(nodes);
    return metadata;
}

void update_min_period(std::optional<Real>& current, Real candidate) {
    if (!(std::isfinite(candidate) && candidate > 0.0)) {
        return;
    }
    if (!current.has_value() || candidate < *current) {
        current = candidate;
    }
}

[[nodiscard]] std::optional<Real> infer_periodic_anchor_period(const Circuit& circuit) {
    std::optional<Real> inferred_period;
    const auto& devices = circuit.devices();
    for (const auto& device : devices) {
        std::visit([&](const auto& dev) {
            using T = std::decay_t<decltype(dev)>;
            if constexpr (std::is_same_v<T, PWMVoltageSource>) {
                update_min_period(inferred_period, dev.period());
            } else if constexpr (std::is_same_v<T, SineVoltageSource>) {
                const Real freq = dev.params().frequency;
                if (std::isfinite(freq) && freq > 0.0) {
                    update_min_period(inferred_period, 1.0 / freq);
                }
            } else if constexpr (std::is_same_v<T, PulseVoltageSource>) {
                update_min_period(inferred_period, dev.params().period);
            }
        }, device);
    }
    return inferred_period;
}

}  // namespace

FrequencyAnalysisResult Simulator::run_frequency_analysis(const FrequencyAnalysisOptions& requested) {
    FrequencyAnalysisResult result;
    result.mode = requested.mode;

    FrequencyAnalysisOptions options = requested;
    if (!options.enabled && options_.frequency_analysis.enabled) {
        options = options_.frequency_analysis;
        result.mode = options.mode;
    }

    auto fail = [&](SimulationDiagnosticCode code,
                    const std::string& message,
                    int failed_point_index = -1,
                    std::optional<Real> failed_frequency_hz = std::nullopt) {
        result.success = false;
        result.diagnostic = code;
        result.message = message;
        result.failed_point_index = failed_point_index;
        if (failed_frequency_hz.has_value()) {
            result.failed_frequency_hz = *failed_frequency_hz;
        }
        if (!diag_reason(code).empty()) {
            result.message += " [" + std::string(diag_reason(code)) + "]";
        }
        return result;
    };

    if (!options.enabled) {
        return fail(
            SimulationDiagnosticCode::FrequencyInvalidConfiguration,
            "Frequency analysis is disabled. Set simulation.frequency_analysis.enabled=true.");
    }

    if (!(std::isfinite(options.f_start_hz) && std::isfinite(options.f_stop_hz)) ||
        options.f_start_hz <= 0.0 || options.f_stop_hz < options.f_start_hz) {
        return fail(
            SimulationDiagnosticCode::FrequencyInvalidConfiguration,
            "Invalid sweep range: require finite f_start_hz > 0 and f_stop_hz >= f_start_hz.");
    }

    if (options.points < 2) {
        return fail(
            SimulationDiagnosticCode::FrequencyInvalidConfiguration,
            "Invalid sweep density: points must be >= 2.");
    }

    if (!(std::isfinite(options.injection_current_amplitude) &&
          std::abs(options.injection_current_amplitude) > 0.0)) {
        return fail(
            SimulationDiagnosticCode::FrequencyInvalidConfiguration,
            "Invalid injection_current_amplitude: must be finite and non-zero.");
    }

    if (options.perturbation_port.positive_node.empty()) {
        return fail(
            SimulationDiagnosticCode::FrequencyInvalidConfiguration,
            "frequency_analysis.perturbation.positive is required.");
    }

    if ((options.mode == FrequencyAnalysisMode::OpenLoopTransfer ||
         options.mode == FrequencyAnalysisMode::ClosedLoopTransfer ||
         options.mode == FrequencyAnalysisMode::OutputImpedance) &&
        options.output_port.positive_node.empty()) {
        return fail(
            SimulationDiagnosticCode::FrequencyInvalidConfiguration,
            "frequency_analysis.output.positive is required for selected mode.");
    }

    const auto in_pos = resolve_node_index(circuit_, options.perturbation_port.positive_node);
    const auto in_neg = resolve_node_index(circuit_, options.perturbation_port.negative_node);
    if (!in_pos.has_value() || !in_neg.has_value()) {
        return fail(
            SimulationDiagnosticCode::FrequencyInvalidConfiguration,
            "Invalid perturbation node names in frequency_analysis block.");
    }

    std::optional<Index> out_pos;
    std::optional<Index> out_neg;
    if (!options.output_port.positive_node.empty()) {
        out_pos = resolve_node_index(circuit_, options.output_port.positive_node);
        out_neg = resolve_node_index(circuit_, options.output_port.negative_node);
        if (!out_pos.has_value() || !out_neg.has_value()) {
            return fail(
                SimulationDiagnosticCode::FrequencyInvalidConfiguration,
                "Invalid output node names in frequency_analysis block.");
        }
    }

    const bool impedance_mode = options.mode == FrequencyAnalysisMode::InputImpedance ||
                                options.mode == FrequencyAnalysisMode::OutputImpedance;
    const std::string response_name = mode_result_name(options.mode);
    const std::vector<Index> input_nodes{*in_pos, *in_neg};
    const std::vector<Index> output_nodes = (out_pos.has_value() && out_neg.has_value())
        ? std::vector<Index>{*out_pos, *out_neg}
        : input_nodes;

    result.channel_metadata.emplace("frequency_hz", make_frequency_metadata());
    result.channel_metadata.emplace(
        "response_real",
        make_response_metadata(response_name + "_real", impedance_mode ? "ohm" : "ratio", output_nodes));
    result.channel_metadata.emplace(
        "response_imag",
        make_response_metadata(response_name + "_imag", impedance_mode ? "ohm" : "ratio", output_nodes));
    result.channel_metadata.emplace(
        "magnitude",
        make_response_metadata(response_name + "_magnitude", impedance_mode ? "ohm" : "ratio", output_nodes));
    result.channel_metadata.emplace(
        "magnitude_db",
        make_response_metadata(response_name + "_magnitude_db", "dB", output_nodes));
    result.channel_metadata.emplace(
        "phase_deg",
        make_response_metadata(response_name + "_phase", "deg", input_nodes));

    const auto& virtual_components = circuit_.virtual_components();
    for (const auto& component : virtual_components) {
        const bool passive_probe = component.type == "voltage_probe" ||
            component.type == "current_probe" ||
            component.type == "power_probe" ||
            component.type == "electrical_scope" ||
            component.type == "thermal_scope" ||
            component.type == "signal_mux" ||
            component.type == "signal_demux";
        if (!passive_probe) {
            return fail(
                SimulationDiagnosticCode::FrequencyUnsupportedConfiguration,
                "Frequency analysis currently supports probe/scope virtual components only.");
        }
    }

    const auto& devices = circuit_.devices();
    const auto& connections = circuit_.connections();

    std::unordered_map<std::size_t, int> source_branch_local;
    int voltage_source_count = 0;

    for (std::size_t i = 0; i < devices.size(); ++i) {
        bool supports = false;
        bool is_voltage_source = false;

        std::visit([&](const auto& dev) {
            using T = std::decay_t<decltype(dev)>;
            supports = std::is_same_v<T, Resistor> ||
                       std::is_same_v<T, Capacitor> ||
                       std::is_same_v<T, Inductor> ||
                       std::is_same_v<T, VoltageSource> ||
                       std::is_same_v<T, CurrentSource> ||
                       std::is_same_v<T, PWMVoltageSource> ||
                       std::is_same_v<T, SineVoltageSource> ||
                       std::is_same_v<T, PulseVoltageSource>;
            is_voltage_source = std::is_same_v<T, VoltageSource> ||
                                std::is_same_v<T, PWMVoltageSource> ||
                                std::is_same_v<T, SineVoltageSource> ||
                                std::is_same_v<T, PulseVoltageSource>;
        }, devices[i]);

        if (!supports) {
            return fail(
                SimulationDiagnosticCode::FrequencyUnsupportedConfiguration,
                "Frequency analysis currently supports linear/passive devices and independent sources only.");
        }

        if (is_voltage_source) {
            source_branch_local.emplace(i, voltage_source_count++);
        }
    }

    const std::optional<Real> inferred_anchor_period = infer_periodic_anchor_period(circuit_);
    if (options.anchor_mode == FrequencyAnchorMode::Auto) {
        result.anchor_mode_selected = inferred_anchor_period.has_value()
            ? FrequencyAnchorMode::Periodic
            : FrequencyAnchorMode::DC;
    } else {
        result.anchor_mode_selected = options.anchor_mode;
    }

    std::string anchor_policy_note;

    if (result.anchor_mode_selected == FrequencyAnchorMode::Periodic) {
        PeriodicSteadyStateOptions periodic_options = options_.periodic_options;
        if (!(std::isfinite(periodic_options.period) && periodic_options.period > 0.0) &&
            inferred_anchor_period.has_value()) {
            periodic_options.period = *inferred_anchor_period;
        }
        if (!(std::isfinite(periodic_options.period) && periodic_options.period > 0.0)) {
            return fail(
                SimulationDiagnosticCode::FrequencyUnsupportedConfiguration,
                "Unable to determine periodic anchor period. Configure simulation.shooting.period "
                "or provide at least one periodic source with finite period/frequency.");
        }
        if (!(std::isfinite(periodic_options.tolerance) && periodic_options.tolerance > 0.0)) {
            periodic_options.tolerance = PeriodicSteadyStateOptions{}.tolerance;
        }
        if (!(std::isfinite(periodic_options.relaxation) && periodic_options.relaxation > 0.0 &&
              periodic_options.relaxation <= 1.0)) {
            periodic_options.relaxation = PeriodicSteadyStateOptions{}.relaxation;
        }
        if (periodic_options.max_iterations <= 0) {
            periodic_options.max_iterations = PeriodicSteadyStateOptions{}.max_iterations;
        }

        const auto periodic_result = run_periodic_shooting(periodic_options);
        if (!periodic_result.success) {
            // Deterministic auto fallback: periodic -> averaged when periodic
            // pre-anchoring fails but averaged linearization is still admissible.
            if (options.anchor_mode == FrequencyAnchorMode::Auto) {
                result.anchor_mode_selected = FrequencyAnchorMode::Averaged;
                anchor_policy_note =
                    "auto fallback applied: periodic pre-anchor failed and averaged anchor was selected";
            } else {
                return fail(
                    SimulationDiagnosticCode::FrequencySolverFailure,
                    "Periodic anchor failed before AC sweep: " + periodic_result.message);
            }
        }
    }

    const int n_nodes = static_cast<int>(circuit_.num_nodes());
    const int n_total = n_nodes + voltage_source_count;

    if (n_total <= 0) {
        return fail(
            SimulationDiagnosticCode::FrequencyInvalidConfiguration,
            "Circuit must contain at least one solvable node for frequency analysis.");
    }

    result.frequency_hz = build_frequency_grid(options);
    result.response_real.reserve(result.frequency_hz.size());
    result.response_imag.reserve(result.frequency_hz.size());
    result.magnitude.reserve(result.frequency_hz.size());
    result.magnitude_db.reserve(result.frequency_hz.size());
    result.phase_deg.reserve(result.frequency_hz.size());

    const Complex i_inj{options.injection_current_amplitude, 0.0};

    for (std::size_t point_index = 0; point_index < result.frequency_hz.size(); ++point_index) {
        const Real f_hz = result.frequency_hz[point_index];
        const Real omega = Real{2.0} * std::numbers::pi_v<Real> * f_hz;
        if (!(std::isfinite(omega) && omega > 0.0)) {
            return fail(
                SimulationDiagnosticCode::FrequencyInvalidConfiguration,
                "Non-positive angular frequency encountered in sweep.",
                static_cast<int>(point_index),
                f_hz);
        }

        std::vector<Eigen::Triplet<Complex>> triplets;
        triplets.reserve(static_cast<std::size_t>(devices.size() * 8 + 16));
        ComplexVector b = ComplexVector::Zero(n_total);

        auto stamp_admittance = [&](Complex y, Index n1, Index n2) {
            if (n1 >= 0) {
                triplets.emplace_back(n1, n1, y);
                if (n2 >= 0) {
                    triplets.emplace_back(n1, n2, -y);
                }
            }
            if (n2 >= 0) {
                triplets.emplace_back(n2, n2, y);
                if (n1 >= 0) {
                    triplets.emplace_back(n2, n1, -y);
                }
            }
        };

        auto stamp_voltage_source = [&](Index npos, Index nneg, int local_branch, Complex value) {
            const int br = n_nodes + local_branch;
            if (npos >= 0) {
                triplets.emplace_back(npos, br, Complex{1.0, 0.0});
                triplets.emplace_back(br, npos, Complex{1.0, 0.0});
            }
            if (nneg >= 0) {
                triplets.emplace_back(nneg, br, Complex{-1.0, 0.0});
                triplets.emplace_back(br, nneg, Complex{-1.0, 0.0});
            }
            b[br] += value;
        };

        auto stamp_current_source = [&](Index npos, Index nneg, Complex value) {
            if (npos >= 0) {
                b[npos] -= value;
            }
            if (nneg >= 0) {
                b[nneg] += value;
            }
        };

        // AC perturbation current source.
        stamp_current_source(*in_pos, *in_neg, i_inj);

        for (std::size_t i = 0; i < devices.size(); ++i) {
            const auto& conn = connections[i];
            std::visit([&](const auto& dev) {
                using T = std::decay_t<decltype(dev)>;
                if constexpr (std::is_same_v<T, Resistor>) {
                    const Real R = dev.resistance();
                    if (!(std::isfinite(R) && R > 0.0)) {
                        return;
                    }
                    stamp_admittance(Complex{1.0 / R, 0.0}, conn.nodes[0], conn.nodes[1]);
                } else if constexpr (std::is_same_v<T, Capacitor>) {
                    const Real C = dev.capacitance();
                    if (!(std::isfinite(C) && C >= 0.0)) {
                        return;
                    }
                    const Complex y{0.0, omega * C};
                    stamp_admittance(y, conn.nodes[0], conn.nodes[1]);
                } else if constexpr (std::is_same_v<T, Inductor>) {
                    const Real L = dev.inductance();
                    if (!(std::isfinite(L) && L > 0.0)) {
                        return;
                    }
                    const Complex y{0.0, -1.0 / (omega * L)};
                    stamp_admittance(y, conn.nodes[0], conn.nodes[1]);
                } else if constexpr (std::is_same_v<T, VoltageSource> ||
                                     std::is_same_v<T, PWMVoltageSource> ||
                                     std::is_same_v<T, SineVoltageSource> ||
                                     std::is_same_v<T, PulseVoltageSource>) {
                    const auto it = source_branch_local.find(i);
                    if (it != source_branch_local.end()) {
                        // Independent voltage sources are AC-grounded for small-signal solve.
                        stamp_voltage_source(conn.nodes[0], conn.nodes[1], it->second, Complex{0.0, 0.0});
                    }
                } else if constexpr (std::is_same_v<T, CurrentSource>) {
                    // Independent current sources are zeroed in small-signal solve.
                    (void)dev;
                }
            }, devices[i]);
        }

        ComplexSparseMatrix A(n_total, n_total);
        A.setFromTriplets(triplets.begin(), triplets.end());

        Eigen::SparseLU<ComplexSparseMatrix> solver;
        solver.analyzePattern(A);
        solver.factorize(A);
        if (solver.info() != Eigen::Success) {
            return fail(
                SimulationDiagnosticCode::FrequencySolverFailure,
                "Failed to factorize AC matrix at frequency " + std::to_string(f_hz) + " Hz.",
                static_cast<int>(point_index),
                f_hz);
        }

        ComplexVector x = solver.solve(b);
        if (solver.info() != Eigen::Success) {
            return fail(
                SimulationDiagnosticCode::FrequencySolverFailure,
                "Failed to solve AC system at frequency " + std::to_string(f_hz) + " Hz.",
                static_cast<int>(point_index),
                f_hz);
        }

        const Complex v_in = node_voltage_complex(x, *in_pos) - node_voltage_complex(x, *in_neg);
        const Complex v_out = (out_pos.has_value() && out_neg.has_value())
            ? node_voltage_complex(x, *out_pos) - node_voltage_complex(x, *out_neg)
            : v_in;

        Complex response{0.0, 0.0};
        if (options.mode == FrequencyAnalysisMode::InputImpedance) {
            response = v_in / i_inj;
        } else if (options.mode == FrequencyAnalysisMode::OutputImpedance) {
            response = v_out / i_inj;
        } else {
            if (std::abs(v_in) <= 1e-18) {
                return fail(
                    SimulationDiagnosticCode::FrequencySolverFailure,
                    "Input perturbation voltage is near zero; transfer ratio is undefined.",
                    static_cast<int>(point_index),
                    f_hz);
            }
            response = v_out / v_in;
        }

        const Real mag = std::abs(response);
        const Real mag_db = (mag > 0.0)
            ? Real{20.0} * std::log10(mag)
            : -std::numeric_limits<Real>::infinity();
        const Real phase = std::atan2(response.imag(), response.real()) *
            (Real{180.0} / std::numbers::pi_v<Real>);

        result.response_real.push_back(response.real());
        result.response_imag.push_back(response.imag());
        result.magnitude.push_back(mag);
        result.magnitude_db.push_back(mag_db);
        result.phase_deg.push_back(phase);
    }

    const bool transfer_mode = options.mode == FrequencyAnalysisMode::OpenLoopTransfer ||
                               options.mode == FrequencyAnalysisMode::ClosedLoopTransfer;
    if (transfer_mode) {
        result.gain_crossover_reason = FrequencyMetricUndefinedReason::NoGainCrossover;
        result.phase_margin_reason = FrequencyMetricUndefinedReason::NoGainCrossover;
        result.phase_crossover_reason = FrequencyMetricUndefinedReason::NoPhaseCrossover;
        result.gain_margin_reason = FrequencyMetricUndefinedReason::NoPhaseCrossover;

        const std::vector<Real> phase_unwrapped = unwrap_phase_deg(result.phase_deg);

        Real phase_at_gc = std::numeric_limits<Real>::quiet_NaN();
        if (const auto gc = interpolate_crossing(
                result.frequency_hz,
                result.magnitude_db,
                0.0,
                std::cref(phase_unwrapped),
                &phase_at_gc);
            gc.has_value()) {
            result.gain_crossover_hz = *gc;
            result.phase_margin_deg = Real{180.0} + phase_at_gc;
            result.gain_crossover_reason = FrequencyMetricUndefinedReason::None;
            result.phase_margin_reason = FrequencyMetricUndefinedReason::None;
        }

        Real mag_at_pc = std::numeric_limits<Real>::quiet_NaN();
        if (const auto pc = interpolate_crossing(
                result.frequency_hz,
                phase_unwrapped,
                -180.0,
                std::cref(result.magnitude_db),
                &mag_at_pc);
            pc.has_value()) {
            result.phase_crossover_hz = *pc;
            result.gain_margin_db = -mag_at_pc;
            result.phase_crossover_reason = FrequencyMetricUndefinedReason::None;
            result.gain_margin_reason = FrequencyMetricUndefinedReason::None;
        }
    }

    result.success = true;
    result.diagnostic = SimulationDiagnosticCode::None;
    result.message = "Frequency analysis completed (anchor=" +
        std::string(anchor_mode_name(result.anchor_mode_selected)) + ")";
    if (!anchor_policy_note.empty()) {
        result.message += "; " + anchor_policy_note;
    }
    return result;
}

}  // namespace pulsim::v1
