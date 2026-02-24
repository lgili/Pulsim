#pragma once

#include "pulsim/v1/runtime_circuit.hpp"
#include "pulsim/v1/solver.hpp"
#include "pulsim/v1/high_performance.hpp"
#include "pulsim/v1/convergence_aids.hpp"
#include "pulsim/v1/integration.hpp"
#include "pulsim/v1/losses.hpp"
#include "pulsim/v1/transient_services.hpp"
#include "pulsim/simulation_control.hpp"

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
    MaxRetriesExceeded,
    BackendEscalation,
    BackendFailure
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

    // Backend escalation policy (Native -> SUNDIALS in Auto mode)
    bool enable_backend_escalation = true;
    int backend_escalation_threshold = 1;

    // Hybrid policy (SUNDIALS recovery window -> try Native again)
    bool enable_native_reentry = false;
    Real sundials_recovery_window = 0.0;
};

enum class TransientBackendMode {
    Native,
    SundialsOnly,
    Auto
};

enum class SundialsSolverFamily {
    IDA,
    CVODE,
    ARKODE
};

enum class SundialsFormulationMode {
    ProjectedWrapper,
    Direct
};

struct SundialsBackendOptions {
    bool enabled = false;
    SundialsSolverFamily family = SundialsSolverFamily::IDA;
    SundialsFormulationMode formulation = SundialsFormulationMode::ProjectedWrapper;
    bool allow_formulation_fallback = true;
    Real rel_tol = 1e-6;
    Real abs_tol = 1e-9;
    int max_steps = 100000;
    int max_nonlinear_iterations = 8;
    bool use_jacobian = true;
    bool reuse_linear_solver = true;
};

struct BackendTelemetry {
    std::string requested_backend = "native";
    std::string selected_backend = "native";
    std::string solver_family = "native";
    std::string formulation_mode = "projected_wrapper";
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
    int equation_assemble_system_calls = 0;
    int equation_assemble_residual_calls = 0;
    double equation_assemble_system_time_seconds = 0.0;
    double equation_assemble_residual_time_seconds = 0.0;
    bool sundials_compiled = false;
    bool sundials_used = false;
    std::string failure_reason;
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

struct ThermalCouplingOptions {
    bool enable = false;
    Real ambient = 25.0;
    ThermalCouplingPolicy policy = ThermalCouplingPolicy::LossWithTemperatureScaling;
    Real default_rth = 1.0;
    Real default_cth = 0.1;
};

struct ThermalDeviceConfig {
    bool enabled = true;
    Real rth = 1.0;
    Real cth = 0.1;
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
    AdvancedTimestepConfig timestep_config = AdvancedTimestepConfig::for_power_electronics();
    RichardsonLTEConfig lte_config = RichardsonLTEConfig::defaults();

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
    ThermalCouplingOptions thermal{};
    std::unordered_map<std::string, ThermalDeviceConfig> thermal_devices;

    // Convergence fallback
    GminConfig gmin_fallback{};
    int max_step_retries = 6;
    FallbackPolicyOptions fallback_policy{};

    // Transient backend selection
    TransientBackendMode transient_backend = TransientBackendMode::Native;
    SundialsBackendOptions sundials{};
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
};

struct PeriodicSteadyStateResult {
    bool success = false;
    int iterations = 0;
    Real residual_norm = 0.0;
    Vector steady_state;
    SimulationResult last_cycle;
    std::string message;
};

struct HarmonicBalanceResult {
    bool success = false;
    int iterations = 0;
    Real residual_norm = 0.0;
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

    // Accessors
    [[nodiscard]] const SimulationOptions& options() const { return options_; }
    void set_options(const SimulationOptions& options) { options_ = options; }
    [[nodiscard]] const TransientServiceRegistry& transient_services() const {
        return transient_services_;
    }

private:
    enum class StepSolvePath {
        SegmentPrimary,
        DaeFallback
    };

    struct DeviceLossState {
        LossAccumulator accumulator;
        LossBreakdown switching_energy{};  // Use fields as energy buckets (J)
        Real peak_power = 0.0;
    };

    struct DeviceThermalState {
        bool enabled = false;
        ThermalDeviceConfig config{};
        Real temperature = 25.0;
        Real peak_temperature = 25.0;
        Real sum_temperature = 0.0;
        int samples = 0;
    };

    struct SwitchMonitor {
        std::string name;
        Index ctrl = -1;
        Index t1 = -1;
        Index t2 = -1;
        Real v_threshold = 0.0;
        bool was_on = false;
    };

    [[nodiscard]] SimulationResult run_transient_native_impl(
        const Vector& x0,
        SimulationCallback callback,
        EventCallback event_callback,
        SimulationControl* control);

    [[nodiscard]] SimulationResult run_transient_sundials_impl(
        const Vector& x0,
        SimulationCallback callback,
        EventCallback event_callback,
        SimulationControl* control,
        bool escalated_from_native);

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

    void accumulate_conduction_losses(const Vector& x, Real dt);
    void accumulate_switching_loss(const std::string& name, bool turning_on, Real energy);
    void accumulate_reverse_recovery_loss(const std::string& name, Real energy);

    void initialize_loss_tracking();
    void finalize_loss_summary(SimulationResult& result);
    void initialize_thermal_tracking();
    [[nodiscard]] Real thermal_scale_factor(std::size_t device_index) const;
    void update_thermal_state(Real dt);
    void finalize_thermal_summary(SimulationResult& result);
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
    std::vector<DeviceLossState> loss_states_;
    std::vector<std::optional<SwitchingEnergy>> switching_energy_;
    std::vector<bool> diode_conducting_;
    std::unordered_map<std::string, std::size_t> device_index_;
    std::vector<DeviceThermalState> thermal_states_;
    std::vector<Real> last_device_power_;
    Real transient_gmin_ = 0.0;

    AdvancedTimestepController timestep_controller_;
    AdaptiveLTEEstimator lte_estimator_;
    BDFOrderController bdf_controller_;
    TransientServiceRegistry transient_services_;
    StepSolvePath last_step_solve_path_ = StepSolvePath::DaeFallback;
    std::string last_step_solve_reason_ = "init";
    bool last_step_segment_cache_hit_ = false;
    bool last_step_linear_factor_cache_hit_ = false;
    bool last_step_linear_factor_cache_miss_ = false;
    bool segment_primary_disabled_for_run_ = false;
    std::uint64_t direct_assemble_system_calls_ = 0;
    std::uint64_t direct_assemble_residual_calls_ = 0;
    double direct_assemble_system_time_seconds_ = 0.0;
    double direct_assemble_residual_time_seconds_ = 0.0;
};

}  // namespace pulsim::v1
