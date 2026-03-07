/**
 * @file simulation.hpp
 * @brief Public declarations for pulsim/v1/simulation.hpp.
 */

#pragma once

#include "pulsim/v1/runtime_circuit.hpp"
#include "pulsim/v1/solver.hpp"
#include "pulsim/v1/high_performance.hpp"
#include "pulsim/v1/convergence_aids.hpp"
#include "pulsim/v1/integration.hpp"
#include "pulsim/v1/extensions.hpp"
#include "pulsim/v1/losses.hpp"
#include "pulsim/v1/transient_services.hpp"
#include "pulsim/simulation_control.hpp"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace pulsim::v1 {

using ::pulsim::SimulationControl;
using ::pulsim::ProgressCallbackConfig;
using ::pulsim::SimulationProgress;

// Streaming callback for transient simulation
using SimulationCallback = std::function<void(Real time, const Vector& state)>;

// Switch event callback
struct SwitchEvent {
    std::string switch_name;
    Real time = 0.0;
    bool new_state = false;  // true = on, false = off
    Real voltage = 0.0;
    Real current = 0.0;
};
using EventCallback = std::function<void(const SwitchEvent& event)>;

enum class SimulationEventType {
    SwitchOn,
    SwitchOff,
    ConvergenceWarning,
    TimestepChange
};

struct SimulationEvent {
    Real time = 0.0;
    SimulationEventType type = SimulationEventType::SwitchOn;
    std::string component;
    std::string description;
    Real value1 = 0.0;
    Real value2 = 0.0;
};

// Optional switching energy model (J per event)
struct SwitchingEnergy {
    Real eon = 0.0;
    Real eoff = 0.0;
    Real err = 0.0;
};

/**
 * @brief 3D datasheet surface for switching energies vs (I, V, Tj).
 *
 * Table layout is row-major in `(current, voltage, temperature)` order:
 * `index = ((i_current * N_voltage) + i_voltage) * N_temperature + i_temperature`.
 */
struct SwitchingEnergySurface3D {
    std::vector<Real> current_axis;
    std::vector<Real> voltage_axis;
    std::vector<Real> temperature_axis;
    std::vector<Real> eon_table;
    std::vector<Real> eoff_table;
    std::vector<Real> err_table;

    [[nodiscard]] bool has_eon() const { return !eon_table.empty(); }
    [[nodiscard]] bool has_eoff() const { return !eoff_table.empty(); }
    [[nodiscard]] bool has_err() const { return !err_table.empty(); }

    [[nodiscard]] std::size_t expected_table_size() const {
        if (current_axis.empty() || voltage_axis.empty() || temperature_axis.empty()) {
            return 0;
        }
        return current_axis.size() * voltage_axis.size() * temperature_axis.size();
    }

    [[nodiscard]] bool valid_shape() const {
        const std::size_t n = expected_table_size();
        if (n == 0) {
            return false;
        }
        auto table_ok = [n](const std::vector<Real>& table) {
            return table.empty() || table.size() == n;
        };
        return table_ok(eon_table) && table_ok(eoff_table) && table_ok(err_table);
    }

    [[nodiscard]] Real evaluate_eon(Real current, Real voltage, Real temperature) const {
        return evaluate_table(eon_table, current, voltage, temperature);
    }

    [[nodiscard]] Real evaluate_eoff(Real current, Real voltage, Real temperature) const {
        return evaluate_table(eoff_table, current, voltage, temperature);
    }

    [[nodiscard]] Real evaluate_err(Real current, Real voltage, Real temperature) const {
        return evaluate_table(err_table, current, voltage, temperature);
    }

private:
    struct AxisBracket {
        std::size_t i0 = 0;
        std::size_t i1 = 0;
        Real w = 0.0;
    };

    [[nodiscard]] static AxisBracket bracket_axis(const std::vector<Real>& axis, Real x) {
        AxisBracket bracket{};
        if (axis.empty()) {
            return bracket;
        }
        if (axis.size() == 1 || x <= axis.front()) {
            bracket.i0 = 0;
            bracket.i1 = 0;
            bracket.w = 0.0;
            return bracket;
        }
        if (x >= axis.back()) {
            const std::size_t last = axis.size() - 1;
            bracket.i0 = last;
            bracket.i1 = last;
            bracket.w = 0.0;
            return bracket;
        }

        const auto upper = std::upper_bound(axis.begin(), axis.end(), x);
        const std::size_t i1 = static_cast<std::size_t>(std::distance(axis.begin(), upper));
        const std::size_t i0 = i1 - 1;
        const Real a0 = axis[i0];
        const Real a1 = axis[i1];
        const Real denom = a1 - a0;
        bracket.i0 = i0;
        bracket.i1 = i1;
        bracket.w = denom > 0.0 ? std::clamp((x - a0) / denom, Real{0.0}, Real{1.0}) : 0.0;
        return bracket;
    }

    [[nodiscard]] std::size_t flat_index(std::size_t i_current,
                                         std::size_t i_voltage,
                                         std::size_t i_temperature) const {
        return ((i_current * voltage_axis.size()) + i_voltage) * temperature_axis.size() + i_temperature;
    }

    [[nodiscard]] Real evaluate_table(const std::vector<Real>& table,
                                      Real current,
                                      Real voltage,
                                      Real temperature) const {
        if (!valid_shape() || table.empty()) {
            return 0.0;
        }

        const AxisBracket ic = bracket_axis(current_axis, current);
        const AxisBracket iv = bracket_axis(voltage_axis, voltage);
        const AxisBracket it = bracket_axis(temperature_axis, temperature);

        const auto value_at = [&](std::size_t i, std::size_t v, std::size_t t) -> Real {
            return table[flat_index(i, v, t)];
        };

        const Real c000 = value_at(ic.i0, iv.i0, it.i0);
        const Real c001 = value_at(ic.i0, iv.i0, it.i1);
        const Real c010 = value_at(ic.i0, iv.i1, it.i0);
        const Real c011 = value_at(ic.i0, iv.i1, it.i1);
        const Real c100 = value_at(ic.i1, iv.i0, it.i0);
        const Real c101 = value_at(ic.i1, iv.i0, it.i1);
        const Real c110 = value_at(ic.i1, iv.i1, it.i0);
        const Real c111 = value_at(ic.i1, iv.i1, it.i1);

        const Real wi = ic.w;
        const Real wv = iv.w;
        const Real wt = it.w;

        const Real c00 = c000 * (1.0 - wi) + c100 * wi;
        const Real c01 = c001 * (1.0 - wi) + c101 * wi;
        const Real c10 = c010 * (1.0 - wi) + c110 * wi;
        const Real c11 = c011 * (1.0 - wi) + c111 * wi;
        const Real c0 = c00 * (1.0 - wv) + c10 * wv;
        const Real c1 = c01 * (1.0 - wv) + c11 * wv;
        return c0 * (1.0 - wt) + c1 * wt;
    }
};

struct StiffnessConfig {
    bool enable = true;
    int rejection_streak_threshold = 2;
    int newton_iter_threshold = 40;
    int newton_streak_threshold = 2;
    int cooldown_steps = 3;
    Real dt_backoff = 0.5;
    int max_bdf_order = 1;
    bool monitor_conditioning = true;
    Real conditioning_error_threshold = 1e-6;
    bool switch_integrator = true;
    Integrator stiff_integrator = Integrator::BDF1;
};

enum class FallbackReasonCode {
    NewtonFailure,
    LTERejection,
    EventSplit,
    StiffnessBackoff,
    TransientGminEscalation,
    MaxRetriesExceeded
};

struct FallbackTraceEntry {
    int step_index = 0;
    int retry_index = 0;
    Real time = 0.0;
    Real dt = 0.0;
    FallbackReasonCode reason = FallbackReasonCode::NewtonFailure;
    SolverStatus solver_status = SolverStatus::Success;
    std::string action;
};

struct FallbackPolicyOptions {
    bool trace_retries = true;
    bool enable_transient_gmin = true;
    int gmin_retry_threshold = 2;
    Real gmin_initial = 1e-9;
    Real gmin_max = 1e-3;
    Real gmin_growth = 10.0;
};

struct ModelRegularizationOptions {
    bool enable_auto = false;
    bool apply_only_in_recovery = true;
    int retry_threshold = 2;
    int max_escalations = 4;
    Real escalation_factor = 2.0;

    Real mosfet_kp_max = 8.0;
    Real mosfet_g_off_min = 1e-7;
    Real diode_g_on_max = 300.0;
    Real diode_g_off_min = 1e-9;
    Real igbt_g_on_max = 5e3;
    Real igbt_g_off_min = 1e-9;
    Real switch_g_on_max = 5e5;
    Real switch_g_off_min = 1e-9;
    Real vcswitch_g_on_max = 5e5;
    Real vcswitch_g_off_min = 1e-9;
};

struct BackendTelemetry {
    std::string requested_backend = "native";
    std::string selected_backend = "native";
    std::string solver_family = "native";
    std::string formulation_mode = "native";
    int function_evaluations = 0;
    int jacobian_evaluations = 0;
    int nonlinear_iterations = 0;
    int nonlinear_convergence_failures = 0;
    int error_test_failures = 0;
    int escalation_count = 0;
    int reinitialization_count = 0;
    int backend_recovery_count = 0;
    int state_space_primary_steps = 0;
    int dae_fallback_steps = 0;
    int segment_non_admissible_steps = 0;
    int segment_model_cache_hits = 0;
    int segment_model_cache_misses = 0;
    int linear_factor_cache_hits = 0;
    int linear_factor_cache_misses = 0;
    int linear_factor_cache_invalidations = 0;
    std::string linear_factor_cache_last_invalidation_reason;
    int reserved_output_samples = 0;
    int time_series_reallocations = 0;
    int state_series_reallocations = 0;
    int virtual_channel_reallocations = 0;
    int equation_assemble_system_calls = 0;
    int equation_assemble_residual_calls = 0;
    double equation_assemble_system_time_seconds = 0.0;
    double equation_assemble_residual_time_seconds = 0.0;
    int model_regularization_events = 0;
    int model_regularization_last_changed = 0;
    double model_regularization_last_intensity = 0.0;
    std::string failure_reason;
};

enum class SimulationDiagnosticCode {
    None,
    DcOperatingPointFailure,
    InvalidInitialState,
    InvalidTimeWindow,
    InvalidTimestep,
    InvalidThermalConfiguration,
    UserStopRequested,
    TransientStepFailure,
    PeriodicInvalidPeriod,
    PeriodicInvalidInitialState,
    PeriodicCycleFailure,
    PeriodicNoConvergence,
    HarmonicInvalidPeriod,
    HarmonicInvalidInitialState,
    HarmonicDifferentiationFailure,
    HarmonicSolverFailure
};

enum class FormulationMode {
    ProjectedWrapper,
    Direct
};

enum class ControlUpdateMode {
    Auto,
    Continuous,
    Discrete
};

struct PeriodicSteadyStateOptions {
    Real period = 0.0;
    int max_iterations = 20;
    Real tolerance = 1e-6;
    Real relaxation = 0.5;
    bool store_last_transient = true;
};

struct HarmonicBalanceOptions {
    Real period = 0.0;
    int num_samples = 32;
    int max_iterations = 25;
    Real tolerance = 1e-6;
    Real relaxation = 1.0;
    bool initialize_from_transient = true;
};

enum class ThermalCouplingPolicy {
    LossOnly,
    LossWithTemperatureScaling
};

enum class ThermalNetworkKind {
    SingleRC,
    Foster,
    Cauer
};

struct ThermalCouplingOptions {
    bool enable = false;
    Real ambient = 25.0;
    ThermalCouplingPolicy policy = ThermalCouplingPolicy::LossWithTemperatureScaling;
    Real default_rth = 1.0;
    Real default_cth = 0.1;
};

struct ThermalDeviceConfig {
    bool enabled = true;
    ThermalNetworkKind network_kind = ThermalNetworkKind::SingleRC;
    Real rth = 1.0;
    Real cth = 0.1;
    std::vector<Real> stage_rth;
    std::vector<Real> stage_cth;
    Real temp_init = 25.0;
    Real temp_ref = 25.0;
    Real alpha = 0.004;
};

struct DeviceThermalTelemetry {
    std::string device_name;
    bool enabled = false;
    Real final_temperature = 25.0;
    Real peak_temperature = 25.0;
    Real average_temperature = 25.0;
};

struct ThermalSummary {
    bool enabled = false;
    Real ambient = 25.0;
    Real max_temperature = 25.0;
    std::vector<DeviceThermalTelemetry> device_temperatures;
};

struct ComponentElectrothermalTelemetry {
    std::string component_name;
    bool thermal_enabled = false;

    Real conduction = 0.0;
    Real turn_on = 0.0;
    Real turn_off = 0.0;
    Real reverse_recovery = 0.0;
    Real total_loss = 0.0;
    Real total_energy = 0.0;
    Real average_power = 0.0;
    Real peak_power = 0.0;

    Real final_temperature = 25.0;
    Real peak_temperature = 25.0;
    Real average_temperature = 25.0;
};

struct SimulationOptions {
    // Time parameters
    Real tstart = 0.0;
    Real tstop = 1e-3;
    Real dt = 1e-6;
    Real dt_min = 1e-12;
    Real dt_max = 1e-3;

    // Solver options
    NewtonOptions newton_options{};
    DCConvergenceConfig dc_config{};
    LinearSolverStackConfig linear_solver = LinearSolverStackConfig::defaults();

    // Adaptive timestep + LTE
    bool adaptive_timestep = true;
    // Canonical timestep mode selection. When explicitly set through runtime/YAML
    // surfaces, this field defines fixed vs variable semantics.
    TransientStepMode step_mode = TransientStepMode::Variable;
    bool step_mode_explicit = false;
    // Global control update policy for mixed-domain blocks (PI/PID and related loops).
    ControlUpdateMode control_mode = ControlUpdateMode::Auto;
    // Explicit control sample interval in seconds when mode is Discrete.
    // Non-positive values disable discrete sampling.
    Real control_sample_time = 0.0;
    AdvancedTimestepConfig timestep_config = AdvancedTimestepConfig::for_power_electronics();
    RichardsonLTEConfig lte_config = RichardsonLTEConfig::defaults();
    FormulationMode formulation_mode = FormulationMode::ProjectedWrapper;
    bool direct_formulation_fallback = true;

    // Integration method selection
    Integrator integrator = Integrator::Trapezoidal;

    // BDF order control (currently supports order 1/2)
    bool enable_bdf_order_control = false;
    BDFOrderConfig bdf_config = BDFOrderConfig::defaults();

    // Stiffness handling
    StiffnessConfig stiffness_config{};

    // Periodic steady-state options
    bool enable_periodic_shooting = false;
    PeriodicSteadyStateOptions periodic_options{};
    bool enable_harmonic_balance = false;
    HarmonicBalanceOptions harmonic_balance{};

    // Events & losses
    bool enable_events = true;
    bool enable_losses = true;
    std::unordered_map<std::string, SwitchingEnergy> switching_energy;
    std::unordered_map<std::string, SwitchingEnergySurface3D> switching_energy_surfaces;
    ThermalCouplingOptions thermal{};
    std::unordered_map<std::string, ThermalDeviceConfig> thermal_devices;

    // Convergence fallback
    GminConfig gmin_fallback{};
    int max_step_retries = 6;
    FallbackPolicyOptions fallback_policy{};
    ModelRegularizationOptions model_regularization{};
};

struct SimulationResult {
    std::vector<Real> time;
    std::vector<Vector> states;
    std::vector<SimulationEvent> events;
    std::vector<std::string> mixed_domain_phase_order;
    std::unordered_map<std::string, std::vector<Real>> virtual_channels;
    std::unordered_map<std::string, VirtualChannelMetadata> virtual_channel_metadata;

    bool success = true;
    SolverStatus final_status = SolverStatus::Success;
    SimulationDiagnosticCode diagnostic = SimulationDiagnosticCode::None;
    std::string message;

    int total_steps = 0;
    int newton_iterations_total = 0;
    int timestep_rejections = 0;
    double total_time_seconds = 0.0;

    LinearSolverTelemetry linear_solver_telemetry;
    std::vector<FallbackTraceEntry> fallback_trace;
    BackendTelemetry backend_telemetry;

    SystemLossSummary loss_summary;
    ThermalSummary thermal_summary;
    std::vector<ComponentElectrothermalTelemetry> component_electrothermal;
};

struct PeriodicSteadyStateResult {
    bool success = false;
    int iterations = 0;
    Real residual_norm = 0.0;
    SimulationDiagnosticCode diagnostic = SimulationDiagnosticCode::None;
    Vector steady_state;
    SimulationResult last_cycle;
    std::string message;
};

struct HarmonicBalanceResult {
    bool success = false;
    int iterations = 0;
    Real residual_norm = 0.0;
    SimulationDiagnosticCode diagnostic = SimulationDiagnosticCode::None;
    Vector solution;
    std::vector<Real> sample_times;
    std::string message;
};

class Simulator {
public:
    explicit Simulator(Circuit& circuit, const SimulationOptions& options = {});

    // DC operating point (robust solver)
    [[nodiscard]] DCAnalysisResult dc_operating_point();

    // Transient simulation (DC start unless x0 provided)
    [[nodiscard]] SimulationResult run_transient(
        SimulationCallback callback = nullptr,
        EventCallback event_callback = nullptr,
        SimulationControl* control = nullptr);

    [[nodiscard]] SimulationResult run_transient(
        const Vector& x0,
        SimulationCallback callback = nullptr,
        EventCallback event_callback = nullptr,
        SimulationControl* control = nullptr);

    [[nodiscard]] SimulationResult run_transient_with_progress(
        SimulationCallback callback,
        EventCallback event_callback,
        SimulationControl* control,
        const ProgressCallbackConfig& progress_config);

    // Periodic steady-state (shooting method)
    [[nodiscard]] PeriodicSteadyStateResult run_periodic_shooting(
        const PeriodicSteadyStateOptions& options = {});

    [[nodiscard]] PeriodicSteadyStateResult run_periodic_shooting(
        const Vector& x0,
        const PeriodicSteadyStateOptions& options = {});

    // Harmonic balance (collocation with spectral differentiation)
    [[nodiscard]] HarmonicBalanceResult run_harmonic_balance(
        const HarmonicBalanceOptions& options = {});

    [[nodiscard]] HarmonicBalanceResult run_harmonic_balance(
        const Vector& x0,
        const HarmonicBalanceOptions& options = {});

    // Loss model attachment (optional)
    void set_switching_energy(const std::string& device_name, const SwitchingEnergy& energy);
    void set_switching_energy_surface(
        const std::string& device_name,
        const SwitchingEnergySurface3D& surface);

    // Accessors
    [[nodiscard]] const SimulationOptions& options() const { return options_; }
    void set_options(const SimulationOptions& options) { options_ = options; }
    [[nodiscard]] const TransientServiceRegistry& transient_services() const {
        return transient_services_;
    }
    [[nodiscard]] const ExtensionRegistry& extension_registry() const {
        return kernel_extension_registry();
    }

private:
    enum class StepSolvePath {
        SegmentPrimary,
        DaeFallback
    };

    struct SwitchMonitor {
        std::string name;
        Index ctrl = -1;
        Index t1 = -1;
        Index t2 = -1;
        Real v_threshold = 0.0;
        bool was_on = false;
    };

    struct ForcedSwitchMonitor {
        std::string name;
        std::size_t device_index = 0;
        Index t1 = -1;
        Index t2 = -1;
        std::optional<bool> was_forced_on;
    };

    [[nodiscard]] SimulationResult run_transient_native_impl(
        const Vector& x0,
        SimulationCallback callback,
        EventCallback event_callback,
        SimulationControl* control);

    NewtonResult solve_step(Real t_next, Real dt, const Vector& x_prev);
    NewtonResult solve_trbdf2_step(Real t_next, Real dt, const Vector& x_prev);
    NewtonResult solve_sdirk2_step(Real t_next, Real dt, const Vector& x_prev, Integrator method);
    bool find_switch_event_time(const SwitchMonitor& sw,
                                Real t_start, Real t_end,
                                const Vector& x_start,
                                Real& t_event, Vector& x_event);

    void record_switch_event(const SwitchMonitor& sw, Real time,
                             const Vector& x_state, bool new_state,
                             SimulationResult& result, EventCallback event_callback);
    [[nodiscard]] Real resolve_switching_event_energy(
        const std::string& name,
        bool turning_on,
        Real voltage,
        Real current) const;

    void accumulate_conduction_losses(const Vector& x, Real dt);
    void accumulate_switching_loss(const std::string& name, bool turning_on, Real energy);
    void accumulate_reverse_recovery_loss(const std::string& name, Real energy);

    void initialize_loss_tracking();
    void finalize_loss_summary(SimulationResult& result);
    void initialize_thermal_tracking();
    void update_thermal_state(Real dt);
    void finalize_thermal_summary(SimulationResult& result);
    void finalize_component_electrothermal(SimulationResult& result);
    void validate_electrothermal_consistency(SimulationResult& result);
    void record_fallback_event(SimulationResult& result,
                               int step_index,
                               int retry_index,
                               Real time,
                               Real dt,
                               FallbackReasonCode reason,
                               SolverStatus solver_status,
                               const std::string& action);

    Circuit& circuit_;
    SimulationOptions options_;
    NewtonRaphsonSolver<RuntimeLinearSolver> newton_solver_;

    std::vector<SwitchMonitor> switch_monitors_;
    std::vector<ForcedSwitchMonitor> forced_switch_monitors_;
    std::unordered_map<std::string, std::size_t> device_index_;
    Real transient_gmin_ = 0.0;

    AdvancedTimestepController timestep_controller_;
    AdaptiveLTEEstimator lte_estimator_;
    BDFOrderController bdf_controller_;
    TransientServiceRegistry transient_services_;
    StepSolvePath last_step_solve_path_ = StepSolvePath::DaeFallback;
    std::string last_step_solve_reason_ = "init";
    bool last_step_segment_cache_hit_ = false;
    bool last_step_segment_attempted_ = false;
    bool last_step_linear_factor_cache_hit_ = false;
    bool last_step_linear_factor_cache_miss_ = false;
    std::string last_step_linear_factor_cache_invalidation_reason_;
    bool segment_primary_disabled_for_run_ = false;
    std::uint64_t direct_assemble_system_calls_ = 0;
    std::uint64_t direct_assemble_residual_calls_ = 0;
    double direct_assemble_system_time_seconds_ = 0.0;
    double direct_assemble_residual_time_seconds_ = 0.0;
};

}  // namespace pulsim::v1
