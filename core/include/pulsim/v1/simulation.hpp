#pragma once

#include "pulsim/v1/runtime_circuit.hpp"
#include "pulsim/v1/solver.hpp"
#include "pulsim/v1/high_performance.hpp"
#include "pulsim/v1/convergence_aids.hpp"
#include "pulsim/v1/integration.hpp"
#include "pulsim/v1/extensions.hpp"
#include "pulsim/v1/losses.hpp"
#include "pulsim/v1/transient_services.hpp"
#include "pulsim/v1/frequency_analysis.hpp"
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
    // refactor-pwl-switching-engine, Phase 6: PWL event-scheduler telemetry.
    // `pwl_topology_transitions` counts accepted-step boundaries where the
    // PWL switch bitmask changed. `pwl_event_commutations` is the total number
    // of individual device commutations committed (a single step that flips
    // two switches contributes 2 commutations and 1 transition).
    int pwl_topology_transitions = 0;
    int pwl_event_commutations = 0;
    int segment_model_cache_hits = 0;
    int segment_model_cache_misses = 0;
    int linear_factor_cache_hits = 0;
    int linear_factor_cache_misses = 0;
    int linear_factor_cache_invalidations = 0;
    std::string linear_factor_cache_last_invalidation_reason;
    /// Phase 2 of `refactor-linear-solver-cache`: count of cache misses
    /// where the symbolic factor (sparsity-pattern analyze) was reused —
    /// i.e. only `factorize` ran. Strict subset of
    /// `linear_factor_cache_misses`.
    int symbolic_factor_cache_hits = 0;
    /// Phase 5 of `refactor-linear-solver-cache`: per-reason invalidation
    /// counters. Sum of all `_*` counters equals
    /// `linear_factor_cache_invalidations`. The typed last-reason field
    /// mirrors `linear_factor_cache_last_invalidation_reason` for
    /// consumers that prefer a discriminated value over string parsing.
    int linear_factor_cache_invalidations_topology_changed = 0;
    int linear_factor_cache_invalidations_stamp_param_changed = 0;
    int linear_factor_cache_invalidations_gmin_escalated = 0;
    int linear_factor_cache_invalidations_source_stepping_active = 0;
    int linear_factor_cache_invalidations_numeric_instability = 0;
    int linear_factor_cache_invalidations_manual_invalidate = 0;
    CacheInvalidationReason linear_factor_cache_last_invalidation_reason_typed =
        CacheInvalidationReason::None;
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
    AdvancedTimestepConfig timestep_config = AdvancedTimestepConfig::for_power_electronics();
    RichardsonLTEConfig lte_config = RichardsonLTEConfig::defaults();
    FormulationMode formulation_mode = FormulationMode::ProjectedWrapper;
    bool direct_formulation_fallback = true;

    // Integration method selection
    Integrator integrator = Integrator::Trapezoidal;

    // PWL switching mode default (refactor-pwl-switching-engine, Phase 5).
    // Threads through to Circuit::set_default_switching_mode() at simulator
    // construction. `Auto` resolves to `Behavioral` for backward compat.
    SwitchingMode switching_mode = SwitchingMode::Auto;

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
    ModelRegularizationOptions model_regularization{};

    /// Phase 7 of `add-frequency-domain-analysis`: declarative analyses
    /// loaded from the YAML `analysis:` array. The user runs them via the
    /// existing `Simulator::run_ac_sweep` / `Simulator::run_fra` methods,
    /// iterating over these vectors. Order within each list matches YAML
    /// input order; running them sequentially against a single `Simulator`
    /// instance reuses the DC operating point automatically (the
    /// per-sweep `dc_operating_point` call is idempotent for a fixed
    /// circuit, so 7.4's "shared DC OP" contract is satisfied without
    /// special plumbing).
    std::vector<AcSweepOptions> ac_sweeps;
    std::vector<FraOptions>     fra_sweeps;
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

// Phase 1/2/3 frequency-analysis types (`LinearSystem`, `AcSweepScale`,
// `AcSweepOptions`, `AcMeasurement`, `AcSweepResult`, `FraOptions`,
// `FraMeasurement`, `FraResult`) live in
// `pulsim/v1/frequency_analysis.hpp`, included above. Extracted there so
// `SimulationOptions` can hold `std::vector<AcSweepOptions>` /
// `std::vector<FraOptions>` for the YAML `analysis:` array (Phase 7).

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

    /// Phase 1 of `add-frequency-domain-analysis`: linearize the circuit
    /// around `(x_op, t_op)` into descriptor state-space form
    /// `E·dx/dt = A·x + B·u`, `y = C·x + D·u`. The result feeds AC sweep
    /// (Phase 2) and downstream eigenvalue / observer-design tools.
    ///
    /// Today PWL-admissible circuits return `method == "piecewise_linear_segment"`
    /// directly from the segment engine's state-space machinery. Circuits with
    /// Behavioral-mode devices populate `failure_reason = "non_admissible_..."`
    /// — Behavioral linearization (AD-derived or finite-difference) is Phase 1.2.
    [[nodiscard]] LinearSystem linearize_around(const Vector& x_op, Real t_op);

    /// Phase 2 of `add-frequency-domain-analysis`: AC small-signal sweep.
    /// Linearizes around `(x_op, t_op)` (or runs DC OP first when
    /// `options.use_dc_op = true`), constructs the per-frequency complex
    /// pencil `K(ω) = jω·E - A`, and returns Bode data
    /// `H(jω) = (jωE - A)⁻¹·B` for each requested measurement node.
    ///
    /// The sparsity pattern of `K(ω)` is constant across the frequency
    /// sweep — `analyzePattern` runs once and `factorize` runs once per
    /// frequency, mirroring the Phase-3 cache architecture from
    /// `refactor-linear-solver-cache`.
    [[nodiscard]] AcSweepResult run_ac_sweep(const AcSweepOptions& options);

    /// Phase 3 of `add-frequency-domain-analysis`: Frequency Response
    /// Analysis. Per-frequency: configures `Circuit::set_ac_perturbation`
    /// to overlay `ε·sin(2π·f·t + φ)` on the named source, runs a transient
    /// for `n_cycles` periods at `samples_per_cycle` samples each,
    /// captures the measurement node's time series, discards the first
    /// `discard_cycles`, and DFTs the remainder at the perturbation
    /// frequency via Goertzel. Returns Bode magnitude / phase for each
    /// measurement node.
    [[nodiscard]] FraResult run_fra(const FraOptions& options);

    // Loss model attachment (optional)
    void set_switching_energy(const std::string& device_name, const SwitchingEnergy& energy);

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

    void accumulate_conduction_losses(const Vector& x, Real dt);
    void accumulate_switching_loss(const std::string& name, bool turning_on, Real energy);
    void accumulate_reverse_recovery_loss(const std::string& name, Real energy);

    void initialize_loss_tracking();
    void finalize_loss_summary(SimulationResult& result);
    void initialize_thermal_tracking();
    void update_thermal_state(Real dt);
    void finalize_thermal_summary(SimulationResult& result);
    void finalize_component_electrothermal(SimulationResult& result);
    void record_fallback_event(SimulationResult& result,
                               int step_index,
                               int retry_index,
                               Real time,
                               Real dt,
                               FallbackReasonCode reason,
                               SolverStatus solver_status,
                               const std::string& action);

    // ── Helpers extracted from run_transient_native_impl ──────────────────────
    // Collects segment/cache path telemetry from the last solve_step() call.
    void collect_step_solve_telemetry(SimulationResult& result);

    // Updates stiffness-detection counters and integrator switching after a
    // step is accepted. Mutates high_iter_streak, stiffness_cooldown, and
    // using_stiff_integrator via out-parameters; reads options_ / circuit_.
    void apply_post_accept_stiffness_update(const NewtonResult& step_result,
                                            Integrator base_integrator,
                                            int& high_iter_streak,
                                            int& stiffness_cooldown,
                                            bool& using_stiff_integrator);

    // Detects switch-state transitions in an accepted step and records events.
    void process_accepted_step_events(Real t, Real dt_used,
                                      const Vector& x_prev,
                                      const NewtonResult& step_result,
                                      SimulationResult& result,
                                      EventCallback event_callback);

    // Collects equation-assembler and Newton counters into result after the
    // transient loop completes (called once, just before return).
    void finalize_transient_telemetry(SimulationResult& result);
    // ─────────────────────────────────────────────────────────────────────────

    Circuit& circuit_;
    SimulationOptions options_;
    NewtonRaphsonSolver<RuntimeLinearSolver> newton_solver_;

    std::vector<SwitchMonitor> switch_monitors_;
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
    bool last_step_symbolic_factor_cache_hit_ = false;
    std::string last_step_linear_factor_cache_invalidation_reason_;
    CacheInvalidationReason last_step_linear_factor_cache_invalidation_reason_typed_ =
        CacheInvalidationReason::None;
    bool segment_primary_disabled_for_run_ = false;
    std::uint64_t direct_assemble_system_calls_ = 0;
    std::uint64_t direct_assemble_residual_calls_ = 0;
    double direct_assemble_system_time_seconds_ = 0.0;
    double direct_assemble_residual_time_seconds_ = 0.0;
};

}  // namespace pulsim::v1
