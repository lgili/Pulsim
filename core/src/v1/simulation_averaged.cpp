/**
 * @file simulation_averaged.cpp
 * @brief Averaged-converter transient runtime path for basic non-isolated DC-DC topologies.
 */

#include "pulsim/v1/simulation.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace pulsim::v1 {

namespace {

struct AveragedModelBinding {
    Real vin = 0.0;
    Real inductance = 0.0;
    Real capacitance = 0.0;
    Real load_resistance = 0.0;
    Index vin_node = -1;
    Index output_node = -1;
    Index inductor_branch = -1;
};

struct AveragedDerivatives {
    Real d_iL = 0.0;
    Real d_vOut = 0.0;
};

struct DcmTargets {
    Real iL_target = 0.0;
    Real output_current_ratio = 0.0;
};

[[nodiscard]] bool nearly_same(Real lhs, Real rhs, Real eps = 1e-12) {
    const Real scale = std::max<Real>({Real{1.0}, std::abs(lhs), std::abs(rhs)});
    return std::abs(lhs - rhs) <= eps * scale;
}

[[nodiscard]] std::optional<std::size_t> find_component_index(
    const Circuit& circuit,
    std::string_view component_name) {
    if (component_name.empty()) {
        return std::nullopt;
    }
    const auto& conns = circuit.connections();
    for (std::size_t i = 0; i < conns.size(); ++i) {
        if (conns[i].name == component_name) {
            return i;
        }
    }
    return std::nullopt;
}

[[nodiscard]] std::string topology_name(AveragedConverterTopology topology) {
    switch (topology) {
        case AveragedConverterTopology::Buck:
            return "buck";
        case AveragedConverterTopology::Boost:
            return "boost";
        case AveragedConverterTopology::BuckBoost:
            return "buck_boost";
    }
    return "unknown";
}

[[nodiscard]] std::optional<AveragedModelBinding> build_converter_binding(
    const Circuit& circuit,
    const AveragedConverterOptions& options,
    std::string* error_message) {
    AveragedModelBinding binding;
    const auto& devices = circuit.devices();
    const auto& conns = circuit.connections();

    const auto fail = [&](std::string message) -> std::optional<AveragedModelBinding> {
        if (error_message != nullptr) {
            *error_message = std::move(message);
        }
        return std::nullopt;
    };

    const auto vin_idx = find_component_index(circuit, options.vin_source);
    if (!vin_idx.has_value()) {
        return fail("Missing averaged_converter.vin_source component: '" + options.vin_source + "'");
    }
    const auto* vin_source = std::get_if<VoltageSource>(&devices[*vin_idx]);
    if (vin_source == nullptr) {
        return fail("averaged_converter.vin_source must reference a voltage_source");
    }
    const auto& vin_conn = conns[*vin_idx];
    if (vin_conn.nodes.size() < 2) {
        return fail("vin_source must have two nodes");
    }
    const Index vin_pos = vin_conn.nodes[0];
    const Index vin_neg = vin_conn.nodes[1];
    if (vin_pos >= 0 && vin_neg < 0) {
        binding.vin = vin_source->voltage();
        binding.vin_node = vin_pos;
    } else if (vin_pos < 0 && vin_neg >= 0) {
        binding.vin = -vin_source->voltage();
        binding.vin_node = vin_neg;
    } else {
        return fail("averaged_converter.vin_source must be referenced to ground");
    }

    const auto inductor_idx = find_component_index(circuit, options.inductor);
    if (!inductor_idx.has_value()) {
        return fail("Missing averaged_converter.inductor component: '" + options.inductor + "'");
    }
    const auto* inductor = std::get_if<Inductor>(&devices[*inductor_idx]);
    if (inductor == nullptr) {
        return fail("averaged_converter.inductor must reference an inductor");
    }
    binding.inductance = inductor->inductance();
    binding.inductor_branch = conns[*inductor_idx].branch_index;
    if (binding.inductor_branch < 0) {
        return fail("Mapped inductor does not expose a branch state index");
    }

    const auto capacitor_idx = find_component_index(circuit, options.capacitor);
    if (!capacitor_idx.has_value()) {
        return fail("Missing averaged_converter.capacitor component: '" + options.capacitor + "'");
    }
    const auto* capacitor = std::get_if<Capacitor>(&devices[*capacitor_idx]);
    if (capacitor == nullptr) {
        return fail("averaged_converter.capacitor must reference a capacitor");
    }
    binding.capacitance = capacitor->capacitance();

    const auto load_idx = find_component_index(circuit, options.load_resistor);
    if (!load_idx.has_value()) {
        return fail(
            "Missing averaged_converter.load_resistor component: '" + options.load_resistor + "'");
    }
    const auto* load = std::get_if<Resistor>(&devices[*load_idx]);
    if (load == nullptr) {
        return fail("averaged_converter.load_resistor must reference a resistor");
    }
    binding.load_resistance = load->resistance();

    try {
        binding.output_node = circuit.get_node(options.output_node);
    } catch (...) {
        return fail("Invalid averaged_converter.output_node: '" + options.output_node + "'");
    }

    if (!(std::isfinite(binding.inductance) && binding.inductance > 0.0)) {
        return fail("Mapped inductor value must be finite and > 0");
    }
    if (!(std::isfinite(binding.capacitance) && binding.capacitance > 0.0)) {
        return fail("Mapped capacitor value must be finite and > 0");
    }
    if (!(std::isfinite(binding.load_resistance) && binding.load_resistance > 0.0)) {
        return fail("Mapped load_resistor value must be finite and > 0");
    }
    if (!std::isfinite(binding.vin)) {
        return fail("Mapped vin_source value must be finite");
    }

    return binding;
}

[[nodiscard]] VirtualChannelMetadata make_averaged_channel_meta(
    std::string component_name,
    std::string unit,
    std::vector<Index> nodes) {
    VirtualChannelMetadata meta;
    meta.component_type = "averaged_converter";
    meta.component_name = std::move(component_name);
    meta.source_component = "averaged_state";
    meta.domain = "time";
    meta.unit = std::move(unit);
    meta.nodes = std::move(nodes);
    return meta;
}

[[nodiscard]] std::optional<AveragedDerivatives> compute_ccm_derivatives(
    AveragedConverterTopology topology,
    const AveragedModelBinding& binding,
    Real duty,
    Real iL,
    Real vOut) {
    AveragedDerivatives out;
    switch (topology) {
        case AveragedConverterTopology::Buck:
            out.d_iL = (duty * binding.vin - vOut) / binding.inductance;
            out.d_vOut = (iL - vOut / binding.load_resistance) / binding.capacitance;
            break;
        case AveragedConverterTopology::Boost:
            out.d_iL = (binding.vin - (1.0 - duty) * vOut) / binding.inductance;
            out.d_vOut =
                ((1.0 - duty) * iL - vOut / binding.load_resistance) / binding.capacitance;
            break;
        case AveragedConverterTopology::BuckBoost:
            out.d_iL = (duty * binding.vin + (1.0 - duty) * vOut) / binding.inductance;
            out.d_vOut =
                (-(1.0 - duty) * iL - vOut / binding.load_resistance) / binding.capacitance;
            break;
    }

    if (!(std::isfinite(out.d_iL) && std::isfinite(out.d_vOut))) {
        return std::nullopt;
    }
    return out;
}

[[nodiscard]] std::optional<DcmTargets> compute_dcm_targets(
    AveragedConverterTopology topology,
    const AveragedModelBinding& binding,
    Real duty,
    Real switching_frequency_hz) {
    if (!(std::isfinite(switching_frequency_hz) && switching_frequency_hz > 0.0)) {
        return std::nullopt;
    }

    const Real k = 2.0 * binding.inductance * switching_frequency_hz / binding.load_resistance;
    if (!(std::isfinite(k) && k > 0.0)) {
        return std::nullopt;
    }

    const Real vin_abs = std::abs(binding.vin);
    const Real vin_sign = std::copysign(1.0, binding.vin);
    const Real duty_sq = duty * duty;
    const Real eps = 1e-15;

    Real m = 0.0;
    Real v_target = 0.0;
    switch (topology) {
        case AveragedConverterTopology::Buck:
            m = 0.5 * (std::sqrt(k * k + 4.0 * duty_sq) - k);
            v_target = vin_sign * (m * vin_abs);
            break;
        case AveragedConverterTopology::Boost:
            m = 0.5 * (1.0 + std::sqrt(1.0 + 4.0 * duty_sq / k));
            v_target = vin_sign * (m * vin_abs);
            break;
        case AveragedConverterTopology::BuckBoost:
            m = duty / std::sqrt(k);
            v_target = -vin_sign * (m * vin_abs);
            break;
    }

    DcmTargets targets{};
    if (topology == AveragedConverterTopology::Buck) {
        targets.iL_target = v_target / binding.load_resistance;
        targets.output_current_ratio = 1.0;
    } else {
        if (vin_abs <= eps) {
            targets.iL_target = 0.0;
        } else {
            const Real p_out_target = (v_target * v_target) / binding.load_resistance;
            targets.iL_target = vin_sign * (p_out_target / vin_abs);
        }
        if (std::abs(v_target) <= eps) {
            targets.output_current_ratio = 0.0;
        } else {
            // Ratio i_out / i_L from ideal power transfer in DCM for boost-derived stages.
            targets.output_current_ratio = binding.vin / v_target;
        }
    }

    if (!(std::isfinite(targets.iL_target) && std::isfinite(targets.output_current_ratio))) {
        return std::nullopt;
    }
    return targets;
}

[[nodiscard]] std::optional<AveragedDerivatives> compute_dcm_derivatives(
    AveragedConverterTopology topology,
    const AveragedModelBinding& binding,
    Real duty,
    Real switching_frequency_hz,
    Real dt,
    Real iL,
    Real vOut) {
    const auto targets = compute_dcm_targets(topology, binding, duty, switching_frequency_hz);
    if (!targets.has_value()) {
        return std::nullopt;
    }

    const Real tau_i = std::max(dt, 1.0 / switching_frequency_hz);
    if (!(std::isfinite(tau_i) && tau_i > 0.0)) {
        return std::nullopt;
    }

    AveragedDerivatives out;
    out.d_iL = (targets->iL_target - iL) / tau_i;
    const Real i_out = targets->output_current_ratio * iL;
    out.d_vOut = (i_out - vOut / binding.load_resistance) / binding.capacitance;

    if (!(std::isfinite(out.d_iL) && std::isfinite(out.d_vOut))) {
        return std::nullopt;
    }
    return out;
}

[[nodiscard]] bool uses_ccm_equations(const AveragedConverterOptions& options, Real iL) {
    switch (options.operating_mode) {
        case AveragedOperatingMode::CCM:
            return true;
        case AveragedOperatingMode::DCM:
            return false;
        case AveragedOperatingMode::Auto:
            return iL >= options.ccm_current_threshold;
    }
    return true;
}

}  // namespace

SimulationResult Simulator::run_transient_averaged_impl(const Vector& x0,
                                                        SimulationCallback callback,
                                                        EventCallback /*event_callback*/,
                                                        SimulationControl* control) {
    SimulationResult result;
    const auto started = std::chrono::high_resolution_clock::now();
    const auto fail = [&](SimulationDiagnosticCode code, const std::string& msg) {
        result.success = false;
        result.final_status = SolverStatus::NumericalError;
        result.diagnostic = code;
        result.message = msg;
        result.backend_telemetry.selected_backend = "native";
        result.backend_telemetry.solver_family = "averaged_converter";
        result.backend_telemetry.formulation_mode = "averaged";
        return result;
    };

    const auto& avg = options_.averaged_converter;
    const std::string topology = topology_name(avg.topology);
    if (!avg.enabled) {
        return fail(
            SimulationDiagnosticCode::AveragedInvalidConfiguration,
            "Averaged converter mode is disabled");
    }
    if (!(std::isfinite(avg.duty_min) && std::isfinite(avg.duty_max) &&
          std::isfinite(avg.duty) && avg.duty_min >= 0.0 && avg.duty_max <= 1.0 &&
          avg.duty_min <= avg.duty_max &&
          avg.duty >= avg.duty_min &&
          avg.duty <= avg.duty_max)) {
        return fail(
            SimulationDiagnosticCode::AveragedInvalidConfiguration,
            "Invalid averaged duty bounds: require finite "
            "0 <= duty_min <= duty <= duty_max <= 1");
    }
    if (!(std::isfinite(avg.ccm_current_threshold) && avg.ccm_current_threshold >= 0.0)) {
        return fail(
            SimulationDiagnosticCode::AveragedInvalidConfiguration,
            "Invalid averaged_converter.ccm_current_threshold: must be finite and >= 0");
    }
    if (!(std::isfinite(avg.switching_frequency_hz) && avg.switching_frequency_hz > 0.0)) {
        return fail(
            SimulationDiagnosticCode::AveragedInvalidConfiguration,
            "Invalid averaged_converter.switching_frequency_hz: must be finite and > 0");
    }

    std::string binding_error;
    const auto binding = build_converter_binding(circuit_, avg, &binding_error);
    if (!binding.has_value()) {
        return fail(SimulationDiagnosticCode::AveragedInvalidConfiguration, binding_error);
    }

    if (!(std::isfinite(options_.dt) && options_.dt > 0.0)) {
        return fail(
            SimulationDiagnosticCode::AveragedInvalidConfiguration,
            "Invalid timestep for averaged mode: simulation.dt must be finite and > 0");
    }

    if (!(std::isfinite(options_.tstart) && std::isfinite(options_.tstop) &&
          options_.tstop >= options_.tstart)) {
        return fail(
            SimulationDiagnosticCode::AveragedInvalidConfiguration,
            "Invalid time window for averaged mode");
    }

    Vector x = Vector::Zero(circuit_.system_size());
    if (x0.size() == x.size() && x0.allFinite()) {
        x = x0;
    }

    const Real duty = std::clamp(avg.duty, avg.duty_min, avg.duty_max);
    if (!std::isfinite(duty)) {
        return fail(
            SimulationDiagnosticCode::AveragedInvalidConfiguration,
            "Invalid averaged duty configuration");
    }

    Real iL = std::isfinite(avg.initial_inductor_current)
        ? avg.initial_inductor_current
        : x[binding->inductor_branch];
    Real vOut = std::isfinite(avg.initial_output_voltage)
        ? avg.initial_output_voltage
        : x[binding->output_node];

    x[binding->inductor_branch] = iL;
    x[binding->output_node] = vOut;
    if (binding->vin_node >= 0) {
        x[binding->vin_node] = binding->vin;
    }

    const std::string iL_channel = "Iavg(" + avg.inductor + ")";
    const std::string vOut_channel = "Vavg(" + avg.output_node + ")";
    const std::string duty_channel = "Davg";

    result.virtual_channel_metadata.emplace(
        iL_channel,
        make_averaged_channel_meta(avg.inductor, "A", {binding->inductor_branch}));
    result.virtual_channel_metadata.emplace(
        vOut_channel,
        make_averaged_channel_meta(avg.output_node, "V", {binding->output_node}));
    result.virtual_channel_metadata.emplace(
        duty_channel,
        make_averaged_channel_meta("duty", "ratio", {}));

    auto& iL_series = result.virtual_channels[iL_channel];
    auto& vOut_series = result.virtual_channels[vOut_channel];
    auto& duty_series = result.virtual_channels[duty_channel];

    const Real dt = options_.dt;
    const Real tstop = options_.tstop;
    const std::size_t max_steps = 5'000'000;

    Real t = options_.tstart;
    std::size_t steps = 0;
    bool warned_out_of_envelope = false;
    const bool enforce_ccm_envelope = (avg.operating_mode == AveragedOperatingMode::CCM);

    while (t <= tstop || nearly_same(t, tstop)) {
        if (control != nullptr) {
            if (control->should_stop()) {
                result.success = false;
                result.final_status = SolverStatus::NumericalError;
                result.diagnostic = SimulationDiagnosticCode::UserStopRequested;
                result.message = "Simulation stopped by user";
                break;
            }
            while (control->should_pause()) {
                control->wait_until_resumed();
            }
        }

        x[binding->inductor_branch] = iL;
        x[binding->output_node] = vOut;
        if (binding->vin_node >= 0) {
            x[binding->vin_node] = binding->vin;
        }

        result.time.push_back(t);
        result.states.push_back(x);
        iL_series.push_back(iL);
        vOut_series.push_back(vOut);
        duty_series.push_back(duty);
        if (callback) {
            callback(t, x);
        }

        if (t >= tstop && !nearly_same(t, tstop)) {
            break;
        }

        std::optional<AveragedDerivatives> derivatives;
        if (uses_ccm_equations(avg, iL)) {
            derivatives = compute_ccm_derivatives(avg.topology, *binding, duty, iL, vOut);
        } else {
            derivatives = compute_dcm_derivatives(
                avg.topology,
                *binding,
                duty,
                avg.switching_frequency_hz,
                dt,
                iL,
                vOut);
        }

        if (!derivatives.has_value()) {
            return fail(
                SimulationDiagnosticCode::AveragedSolverFailure,
                "Non-finite derivative detected in averaged " + topology + " integration");
        }

        iL += derivatives->d_iL * dt;
        vOut += derivatives->d_vOut * dt;
        t += dt;
        ++steps;

        if (steps > max_steps) {
            return fail(
                SimulationDiagnosticCode::AveragedSolverFailure,
                "Averaged simulation exceeded maximum step budget");
        }

        if (enforce_ccm_envelope && iL < avg.ccm_current_threshold) {
            if (avg.envelope_policy == AveragedEnvelopePolicy::Strict) {
                return fail(
                    SimulationDiagnosticCode::AveragedOutOfEnvelope,
                    "Averaged " + topology + " run left CCM envelope (iL < ccm_current_threshold)");
            }
            warned_out_of_envelope = true;
        }
    }

    if (result.diagnostic == SimulationDiagnosticCode::UserStopRequested) {
        result.backend_telemetry.failure_reason = "user_stop_requested";
        return result;
    }

    result.success = true;
    result.final_status = SolverStatus::Success;
    result.diagnostic = SimulationDiagnosticCode::None;
    result.message = warned_out_of_envelope
        ? "Averaged " + topology + " simulation completed with out-of-envelope warning"
        : "Averaged " + topology + " simulation completed";
    result.total_steps = static_cast<int>(result.time.size());
    result.newton_iterations_total = 0;
    result.timestep_rejections = 0;
    result.backend_telemetry.selected_backend = "native";
    result.backend_telemetry.solver_family = "averaged_converter";
    result.backend_telemetry.formulation_mode = "averaged";
    result.backend_telemetry.function_evaluations = static_cast<int>(result.time.size());
    result.backend_telemetry.failure_reason.clear();

    const auto elapsed = std::chrono::high_resolution_clock::now() - started;
    result.total_time_seconds = std::chrono::duration<double>(elapsed).count();
    return result;
}

}  // namespace pulsim::v1
