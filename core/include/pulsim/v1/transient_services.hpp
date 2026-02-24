#pragma once

#include "pulsim/v1/runtime_circuit.hpp"
#include "pulsim/v1/solver.hpp"
#include "pulsim/v1/high_performance.hpp"

#include <cstdint>
#include <limits>
#include <memory>
#include <string>

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

struct TransientServiceRegistry {
    std::shared_ptr<EquationAssemblerService> equation_assembler;
    std::shared_ptr<SegmentModelService> segment_model;
    std::shared_ptr<SegmentStepperService> segment_stepper;
    std::shared_ptr<NonlinearSolveService> nonlinear_solve;
    std::shared_ptr<LinearSolveService> linear_solve;
    std::shared_ptr<EventSchedulerService> event_scheduler;
    std::shared_ptr<RecoveryManagerService> recovery_manager;
    std::shared_ptr<TelemetryCollectorService> telemetry_collector;

    bool supports_fixed_mode = true;
    bool supports_variable_mode = true;

    [[nodiscard]] bool complete() const {
        return equation_assembler && segment_model && segment_stepper &&
               nonlinear_solve && linear_solve &&
               event_scheduler && recovery_manager && telemetry_collector;
    }

    [[nodiscard]] bool supports_mode(TransientStepMode mode) const {
        if (mode == TransientStepMode::Fixed) {
            return supports_fixed_mode;
        }
        return supports_variable_mode;
    }
};

[[nodiscard]] TransientServiceRegistry make_default_transient_service_registry(
    Circuit& circuit,
    const SimulationOptions& options,
    NewtonRaphsonSolver<RuntimeLinearSolver>& newton_solver);

}  // namespace pulsim::v1
