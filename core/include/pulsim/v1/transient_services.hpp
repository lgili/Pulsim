#pragma once

#include "pulsim/v1/runtime_circuit.hpp"
#include "pulsim/v1/solver.hpp"
#include "pulsim/v1/high_performance.hpp"
#include "pulsim/v1/losses.hpp"

#include <concepts>
#include <cstdint>
#include <limits>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

namespace pulsim::v1 {

struct SimulationOptions;

enum class TransientStepMode {
    Fixed,
    Variable
};

struct TransientStepRequest {
    TransientStepMode mode = TransientStepMode::Variable;
    Real t_now = 0.0;
    Real t_target = 0.0;
    Real dt_candidate = 0.0;
    Real dt_min = 0.0;
    Real pwm_boundary_time = std::numeric_limits<Real>::quiet_NaN();
    Real dead_time_boundary_time = std::numeric_limits<Real>::quiet_NaN();
    Real threshold_crossing_time = std::numeric_limits<Real>::quiet_NaN();
    int retry_index = 0;
    int max_retries = 0;
    bool event_adjacent = false;
};

enum class SegmentSolvePath {
    StateSpacePrimary,
    DaeFallback
};

struct EquationAssemblerTelemetry {
    std::uint64_t system_calls = 0;
    std::uint64_t residual_calls = 0;
    double system_time_seconds = 0.0;
    double residual_time_seconds = 0.0;
};

struct SegmentLinearStateSpace {
    SparseMatrix E;
    SparseMatrix A;
    SparseMatrix B;
    Vector c;
    Vector u;
};

struct SegmentModel {
    bool admissible = false;
    std::uint64_t topology_signature = 0;
    Real t_now = 0.0;
    Real t_target = 0.0;
    Real dt = 0.0;
    bool cache_hit = false;
    std::string classification;
    std::shared_ptr<const SegmentLinearStateSpace> linear_model;
};

struct SegmentStepOutcome {
    NewtonResult result;
    SegmentSolvePath path = SegmentSolvePath::DaeFallback;
    bool requires_fallback = true;
    bool linear_factor_cache_hit = false;
    bool linear_factor_cache_miss = false;
    std::string reason;
};

enum class RecoveryStage {
    None,
    DtBackoff,
    GlobalizationEscalation,
    StiffProfile,
    Regularization,
    Abort
};

struct RecoveryDecision {
    RecoveryStage stage = RecoveryStage::None;
    Real next_dt = 0.0;
    bool abort = false;
    std::string reason;
};

class EquationAssemblerService {
public:
    virtual ~EquationAssemblerService() = default;

    virtual void assemble_system(const Vector& x,
                                 Real t_next,
                                 Real dt,
                                 SparseMatrix& jacobian,
                                 Vector& residual) = 0;

    virtual void assemble_residual(const Vector& x,
                                   Real t_next,
                                   Real dt,
                                   Vector& residual) = 0;

    virtual void set_transient_gmin(Real gmin) = 0;
    [[nodiscard]] virtual Real transient_gmin() const = 0;
    [[nodiscard]] virtual EquationAssemblerTelemetry telemetry() const = 0;
    virtual void reset_telemetry() = 0;
};

class SegmentModelService {
public:
    virtual ~SegmentModelService() = default;

    [[nodiscard]] virtual SegmentModel build_model(
        const Vector& x_now,
        const TransientStepRequest& request) const = 0;
};

class SegmentStepperService {
public:
    virtual ~SegmentStepperService() = default;

    [[nodiscard]] virtual SegmentStepOutcome try_advance(
        const SegmentModel& model,
        const Vector& x_now,
        const TransientStepRequest& request) = 0;
};

class NonlinearSolveService {
public:
    virtual ~NonlinearSolveService() = default;

    [[nodiscard]] virtual NewtonResult solve(const Vector& x_guess,
                                             Real t_next,
                                             Real dt) = 0;

    [[nodiscard]] virtual const NewtonOptions& options() const = 0;
    virtual void set_options(const NewtonOptions& options) = 0;
};

class LinearSolveService {
public:
    virtual ~LinearSolveService() = default;

    [[nodiscard]] virtual RuntimeLinearSolver& solver() = 0;
    [[nodiscard]] virtual const LinearSolverStackConfig& config() const = 0;
};

class EventSchedulerService {
public:
    virtual ~EventSchedulerService() = default;

    [[nodiscard]] virtual Real next_segment_target(const TransientStepRequest& request,
                                                   Real t_stop) const = 0;
};

class RecoveryManagerService {
public:
    virtual ~RecoveryManagerService() = default;

    [[nodiscard]] virtual RecoveryDecision on_step_failure(
        const TransientStepRequest& request) const = 0;
};

class TelemetryCollectorService {
public:
    virtual ~TelemetryCollectorService() = default;

    virtual void on_step_attempt(const TransientStepRequest& request) = 0;
    virtual void on_step_accept(Real t_next, const Vector& state) = 0;
    virtual void on_step_reject(const RecoveryDecision& decision) = 0;
};

struct ThermalDeviceSummaryEntry {
    std::string device_name;
    bool enabled = false;
    Real final_temperature = 25.0;
    Real peak_temperature = 25.0;
    Real average_temperature = 25.0;
};

struct ThermalServiceSummary {
    bool enabled = false;
    Real ambient = 25.0;
    Real max_temperature = 25.0;
    std::vector<ThermalDeviceSummaryEntry> device_temperatures;
};

class LossService {
public:
    virtual ~LossService() = default;

    virtual void reset() = 0;
    virtual void commit_switching_event(std::string_view name,
                                        bool turning_on,
                                        Real energy) = 0;
    virtual void commit_reverse_recovery_event(std::string_view name,
                                               Real energy) = 0;
    virtual void commit_accepted_segment(const Vector& x,
                                         Real dt,
                                         std::span<const Real> thermal_scale) = 0;
    [[nodiscard]] virtual std::span<const Real> last_device_power() const = 0;
    [[nodiscard]] virtual SystemLossSummary finalize(Real duration) const = 0;
};

class ThermalService {
public:
    virtual ~ThermalService() = default;

    virtual void reset() = 0;
    [[nodiscard]] virtual Real thermal_scale_factor(std::size_t device_index) const = 0;
    [[nodiscard]] virtual std::span<const Real> thermal_scale_vector() const = 0;
    virtual void commit_accepted_segment(Real dt,
                                         std::span<const Real> device_power) = 0;
    [[nodiscard]] virtual ThermalServiceSummary finalize() const = 0;
};

struct TransientServiceRegistry {
    std::shared_ptr<EquationAssemblerService> equation_assembler;
    std::shared_ptr<SegmentModelService> segment_model;
    std::shared_ptr<SegmentStepperService> segment_stepper;
    std::shared_ptr<NonlinearSolveService> nonlinear_solve;
    std::shared_ptr<LinearSolveService> linear_solve;
    std::shared_ptr<EventSchedulerService> event_scheduler;
    std::shared_ptr<RecoveryManagerService> recovery_manager;
    std::shared_ptr<TelemetryCollectorService> telemetry_collector;
    std::shared_ptr<LossService> loss_service;
    std::shared_ptr<ThermalService> thermal_service;

    bool supports_fixed_mode = true;
    bool supports_variable_mode = true;

    [[nodiscard]] bool complete() const {
        return equation_assembler && segment_model && segment_stepper &&
               nonlinear_solve && linear_solve &&
               event_scheduler && recovery_manager && telemetry_collector &&
               loss_service && thermal_service;
    }

    [[nodiscard]] bool supports_mode(TransientStepMode mode) const {
        if (mode == TransientStepMode::Fixed) {
            return supports_fixed_mode;
        }
        return supports_variable_mode;
    }
};

template<typename Builder>
concept TransientServiceBuilder = requires(
    Builder& builder,
    Circuit& circuit,
    const SimulationOptions& options,
    NewtonRaphsonSolver<RuntimeLinearSolver>& newton_solver,
    std::shared_ptr<EquationAssemblerService> equation_assembler,
    std::shared_ptr<LinearSolveService> linear_solve) {
    { builder.supports_fixed_mode(options) } -> std::convertible_to<bool>;
    { builder.supports_variable_mode(options) } -> std::convertible_to<bool>;
    {
        builder.make_equation_assembler(circuit, options)
    } -> std::same_as<std::shared_ptr<EquationAssemblerService>>;
    {
        builder.make_nonlinear_solve(newton_solver, equation_assembler)
    } -> std::same_as<std::shared_ptr<NonlinearSolveService>>;
    {
        builder.make_segment_model(circuit, equation_assembler)
    } -> std::same_as<std::shared_ptr<SegmentModelService>>;
    {
        builder.make_linear_solve(newton_solver.linear_solver())
    } -> std::same_as<std::shared_ptr<LinearSolveService>>;
    {
        builder.make_segment_stepper(equation_assembler, linear_solve)
    } -> std::same_as<std::shared_ptr<SegmentStepperService>>;
    {
        builder.make_event_scheduler(circuit, options)
    } -> std::same_as<std::shared_ptr<EventSchedulerService>>;
    {
        builder.make_recovery_manager(options)
    } -> std::same_as<std::shared_ptr<RecoveryManagerService>>;
    {
        builder.make_telemetry_collector()
    } -> std::same_as<std::shared_ptr<TelemetryCollectorService>>;
    {
        builder.make_loss_service(circuit, options)
    } -> std::same_as<std::shared_ptr<LossService>>;
    {
        builder.make_thermal_service(circuit, options)
    } -> std::same_as<std::shared_ptr<ThermalService>>;
};

template<typename Builder>
requires TransientServiceBuilder<std::remove_cvref_t<Builder>>
[[nodiscard]] inline TransientServiceRegistry make_transient_service_registry(
    Circuit& circuit,
    const SimulationOptions& options,
    NewtonRaphsonSolver<RuntimeLinearSolver>& newton_solver,
    Builder&& builder) {

    auto& registry_builder = builder;

    TransientServiceRegistry registry;
    registry.supports_fixed_mode = static_cast<bool>(registry_builder.supports_fixed_mode(options));
    registry.supports_variable_mode = static_cast<bool>(registry_builder.supports_variable_mode(options));

    registry.equation_assembler = registry_builder.make_equation_assembler(circuit, options);
    registry.nonlinear_solve =
        registry_builder.make_nonlinear_solve(newton_solver, registry.equation_assembler);
    registry.segment_model =
        registry_builder.make_segment_model(circuit, registry.equation_assembler);
    registry.linear_solve =
        registry_builder.make_linear_solve(newton_solver.linear_solver());
    registry.segment_stepper =
        registry_builder.make_segment_stepper(registry.equation_assembler, registry.linear_solve);
    registry.event_scheduler = registry_builder.make_event_scheduler(circuit, options);
    registry.recovery_manager = registry_builder.make_recovery_manager(options);
    registry.telemetry_collector = registry_builder.make_telemetry_collector();
    registry.loss_service = registry_builder.make_loss_service(circuit, options);
    registry.thermal_service = registry_builder.make_thermal_service(circuit, options);
    return registry;
}

[[nodiscard]] TransientServiceRegistry make_default_transient_service_registry(
    Circuit& circuit,
    const SimulationOptions& options,
    NewtonRaphsonSolver<RuntimeLinearSolver>& newton_solver);

}  // namespace pulsim::v1
