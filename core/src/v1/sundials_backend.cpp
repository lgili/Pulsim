#include "sundials_backend.hpp"

#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <limits>
#include <numbers>
#include <sstream>
#include <string>
#include <type_traits>

#ifdef PULSIM_HAS_SUNDIALS
#include <arkode/arkode_arkstep.h>
#include <cvode/cvode.h>
#include <cvode/cvode_ls.h>
#include <ida/ida.h>
#include <ida/ida_ls.h>
#include <nvector/nvector_serial.h>
#include <sunlinsol/sunlinsol_spgmr.h>
#include <sundials/sundials_context.h>
#endif

namespace pulsim::v1 {

namespace {

[[nodiscard]] std::string solver_family_to_string(SundialsSolverFamily family) {
    switch (family) {
        case SundialsSolverFamily::IDA:
            return "ida";
        case SundialsSolverFamily::CVODE:
            return "cvode";
        case SundialsSolverFamily::ARKODE:
            return "arkode";
    }
    return "ida";
}

[[nodiscard]] std::string formulation_mode_to_string(SundialsFormulationMode mode) {
    switch (mode) {
        case SundialsFormulationMode::ProjectedWrapper:
            return "projected_wrapper";
        case SundialsFormulationMode::Direct:
            return "direct";
    }
    return "projected_wrapper";
}

#ifdef PULSIM_HAS_SUNDIALS

[[nodiscard]] Real clamp_positive(Real value, Real fallback) {
    if (!std::isfinite(value) || value <= 0.0) {
        return fallback;
    }
    return value;
}

[[nodiscard]] SolverStatus map_failure_status(int flag) {
    if (flag >= 0) {
        return SolverStatus::Success;
    }
#if defined(CV_TOO_MUCH_WORK) && defined(IDA_TOO_MUCH_WORK) && defined(ARK_TOO_MUCH_WORK)
    if (flag == CV_TOO_MUCH_WORK || flag == IDA_TOO_MUCH_WORK || flag == ARK_TOO_MUCH_WORK) {
        return SolverStatus::MaxIterationsReached;
    }
#endif
#if defined(CV_CONV_FAILURE) && defined(IDA_CONV_FAIL) && defined(ARK_CONV_FAIL)
    if (flag == CV_CONV_FAILURE || flag == IDA_CONV_FAIL || flag == ARK_CONV_FAIL) {
        return SolverStatus::Diverged;
    }
#endif
#if defined(CV_ERR_FAILURE) && defined(IDA_ERR_FAIL) && defined(ARK_ERR_FAILURE)
    if (flag == CV_ERR_FAILURE || flag == IDA_ERR_FAIL || flag == ARK_ERR_FAILURE) {
        return SolverStatus::NumericalError;
    }
#endif
    return SolverStatus::NumericalError;
}

class SharedImplicitStepperCore {
public:
    SharedImplicitStepperCore(const Circuit& source_circuit, const SimulationOptions& options)
        : circuit_(source_circuit)
        , options_(options)
        , newton_solver_(options_.newton_options)
        , transient_services_(make_default_transient_service_registry(circuit_, options_, newton_solver_)) {
        newton_solver_.linear_solver().set_config(options_.linear_solver);
        circuit_.set_integration_method(Integrator::BDF1);
    }

    void initialize_history(const Vector& x0, Real t0, Real dt0) {
        const Real dt_safe = std::max(dt0, options_.dt_min);
        circuit_.set_current_time(t0);
        circuit_.set_timestep(dt_safe);
        circuit_.set_integration_method(Integrator::BDF1);
        circuit_.update_history(x0, true);
    }

    void accept_state(const Vector& state, Real t, Real dt) {
        const Real dt_safe = std::max(dt, options_.dt_min);
        circuit_.set_current_time(t);
        circuit_.set_timestep(dt_safe);
        circuit_.set_integration_method(Integrator::BDF1);
        circuit_.update_history(state, true);
    }

    [[nodiscard]] NewtonResult solve_shared_step(Real t_next, Real dt, const Vector& x_guess) {
        const Real dt_safe = std::max(dt, options_.dt_min);
        circuit_.set_current_time(t_next);
        circuit_.set_timestep(dt_safe);
        circuit_.set_integration_method(Integrator::BDF1);
        circuit_.set_integration_order(1);
        transient_services_.equation_assembler->set_transient_gmin(0.0);

        TransientStepRequest request;
        request.mode = TransientStepMode::Variable;
        request.t_now = t_next - dt_safe;
        request.t_target = t_next;
        request.dt_candidate = dt_safe;
        request.dt_min = options_.dt_min;
        request.retry_index = 0;
        request.max_retries = 1;
        request.event_adjacent = false;

        const auto segment_model = transient_services_.segment_model->build_model(x_guess, request);
        const auto segment_outcome =
            transient_services_.segment_stepper->try_advance(segment_model, x_guess, request);
        if (!segment_outcome.requires_fallback) {
            return segment_outcome.result;
        }
        return transient_services_.nonlinear_solve->solve(x_guess, t_next, dt_safe);
    }

    [[nodiscard]] const SimulationOptions& options() const {
        return options_;
    }

private:
    Circuit circuit_;
    SimulationOptions options_;
    NewtonRaphsonSolver<RuntimeLinearSolver> newton_solver_;
    TransientServiceRegistry transient_services_;
};

class ProjectedImplicitStepper {
public:
    ProjectedImplicitStepper(const Circuit& source_circuit, const SimulationOptions& options)
        : shared_(source_circuit, options) {}

    void initialize_history(const Vector& x0, Real t0, Real dt0) {
        shared_.initialize_history(x0, t0, dt0);
    }

    void accept_state(const Vector& state, Real t, Real dt) {
        shared_.accept_state(state, t, dt);
    }

    [[nodiscard]] bool project_rhs(Real t, const Vector& state, Vector& rhs, std::string& error) {
        const Real dt = projection_dt();
        const Real t_next = t + dt;
        const NewtonResult step = shared_.solve_shared_step(t_next, dt, state);
        if (step.status == SolverStatus::Success) {
            rhs = (step.solution - state) / dt;
            return true;
        }
        error = step.error_message.empty() ? "projection Newton failed" : step.error_message;
        return false;
    }

private:
    [[nodiscard]] Real projection_dt() const {
        const auto& options = shared_.options();
        const Real dt_floor = std::max(options.dt_min, Real{1e-12});
        const Real dt_guess = clamp_positive(options.dt, dt_floor);
        const Real dt_ceiling = std::max(options.dt_max, dt_floor);
        return std::clamp(dt_guess, dt_floor, dt_ceiling);
    }

    SharedImplicitStepperCore shared_;
};

class DirectResidualAssembler {
public:
    explicit DirectResidualAssembler(const Circuit& source_circuit)
        : circuit_(source_circuit) {}

    void initialize_state(const Vector& x0, Real t0, Real dt0) {
        const Real dt_safe = std::max(dt0, Real{1e-15});
        circuit_.set_current_time(t0);
        circuit_.set_timestep(dt_safe);
        circuit_.set_integration_method(Integrator::BDF1);
        circuit_.update_history(x0, true);
    }

    void accept_state(const Vector& state, Real t, Real dt) {
        const Real dt_safe = std::max(dt, Real{1e-15});
        circuit_.set_current_time(t);
        circuit_.set_timestep(dt_safe);
        circuit_.set_integration_method(Integrator::BDF1);
        circuit_.update_history(state, true);
    }

    [[nodiscard]] bool compute_ida_residual(Real t,
                                            const Vector& state,
                                            const Vector& state_dot,
                                            Vector& residual,
                                            std::string& error) const {
        const Index n = circuit_.system_size();
        const Index nodes = circuit_.num_nodes();
        const Index branches = circuit_.num_branches();
        if (state.size() != n || state_dot.size() != n) {
            error = "direct residual dimension mismatch";
            return false;
        }

        const Real* dot_ptr = state_dot.data();
        circuit_.assemble_residual_hb(
            residual,
            state,
            t,
            std::span<const Real>(dot_ptr, static_cast<std::size_t>(std::max<Index>(nodes, 0))),
            std::span<const Real>(
                dot_ptr + std::max<Index>(nodes, 0),
                static_cast<std::size_t>(std::max<Index>(branches, 0))));
        if (!residual.allFinite()) {
            error = "direct residual produced non-finite values";
            return false;
        }
        return true;
    }

private:
    Circuit circuit_;
};

class DirectRhsStepper {
public:
    DirectRhsStepper(const Circuit& source_circuit, const SimulationOptions& options)
        : shared_(source_circuit, options) {}

    void initialize_history(const Vector& x0, Real t0, Real dt0) {
        const Real dt_safe = std::max(dt0, shared_.options().dt_min);
        shared_.initialize_history(x0, t0, dt_safe);
        last_eval_time_ = t0;
        last_step_dt_ = dt_safe;
    }

    void accept_state(const Vector& state, Real t, Real dt) {
        const Real dt_safe = std::max(dt, shared_.options().dt_min);
        shared_.accept_state(state, t, dt_safe);
        last_step_dt_ = dt_safe;
    }

    [[nodiscard]] bool rhs_from_implicit_step(Real t, const Vector& state, Vector& rhs, std::string& error) {
        const Real dt = projection_dt(t);
        const Real t_next = t + dt;
        const NewtonResult step = shared_.solve_shared_step(t_next, dt, state);
        if (step.status == SolverStatus::Success) {
            rhs = (step.solution - state) / dt;
            last_eval_time_ = t;
            return true;
        }
        last_eval_time_ = t;
        error = step.error_message.empty() ? "direct rhs Newton failed" : step.error_message;
        return false;
    }

private:
    [[nodiscard]] Real projection_dt(Real t_now) const {
        const auto& options = shared_.options();
        const Real dt_floor = std::max(options.dt_min, Real{1e-12});
        const Real dt_ceiling = std::max(options.dt_max, dt_floor);
        Real dt_guess = clamp_positive(last_step_dt_, clamp_positive(options.dt * 0.5, dt_floor));
        if (std::isfinite(last_eval_time_)) {
            const Real eval_dt = std::abs(t_now - last_eval_time_);
            if (std::isfinite(eval_dt) && eval_dt > dt_floor * 0.1) {
                dt_guess = eval_dt;
            }
        }
        return std::clamp(dt_guess, dt_floor, dt_ceiling);
    }

    SharedImplicitStepperCore shared_;
    Real last_eval_time_ = std::numeric_limits<Real>::quiet_NaN();
    Real last_step_dt_ = std::numeric_limits<Real>::quiet_NaN();
};

struct SundialsUserData {
    ProjectedImplicitStepper* projected_stepper = nullptr;
    DirectResidualAssembler* direct_assembler = nullptr;
    DirectRhsStepper* direct_rhs_stepper = nullptr;
    bool use_direct_formulation = false;
    std::size_t dimension = 0;
    std::string last_projection_error;
};

[[nodiscard]] bool build_ida_differential_id(const Circuit& circuit, N_Vector id) {
    auto* id_data = N_VGetArrayPointer(id);
    if (!id_data) {
        return false;
    }

    const Index n = circuit.system_size();
    for (Index i = 0; i < n; ++i) {
        id_data[static_cast<std::size_t>(i)] = SUN_RCONST(0.0);
    }

    const auto& devices = circuit.devices();
    const auto& conns = circuit.connections();
    for (std::size_t i = 0; i < devices.size() && i < conns.size(); ++i) {
        const auto& conn = conns[i];
        std::visit([&](const auto& dev) {
            using T = std::decay_t<decltype(dev)>;
            if constexpr (std::is_same_v<T, Capacitor>) {
                if (!conn.nodes.empty() && conn.nodes[0] >= 0 && conn.nodes[0] < n) {
                    id_data[static_cast<std::size_t>(conn.nodes[0])] = SUN_RCONST(1.0);
                }
                if (conn.nodes.size() > 1 && conn.nodes[1] >= 0 && conn.nodes[1] < n) {
                    id_data[static_cast<std::size_t>(conn.nodes[1])] = SUN_RCONST(1.0);
                }
            } else if constexpr (std::is_same_v<T, Inductor>) {
                if (conn.branch_index >= 0 && conn.branch_index < n) {
                    id_data[static_cast<std::size_t>(conn.branch_index)] = SUN_RCONST(1.0);
                }
            }
        }, devices[i]);
    }
    return true;
}

[[nodiscard]] bool load_state(N_Vector y, std::size_t n, Vector& state) {
    state.resize(static_cast<Eigen::Index>(n));
    auto* data = N_VGetArrayPointer(y);
    if (!data) {
        return false;
    }
    for (std::size_t i = 0; i < n; ++i) {
        state[static_cast<Eigen::Index>(i)] = static_cast<Real>(data[i]);
    }
    return true;
}

[[nodiscard]] bool store_state(N_Vector y, const Vector& state) {
    auto* data = N_VGetArrayPointer(y);
    if (!data) {
        return false;
    }
    const std::size_t n = static_cast<std::size_t>(state.size());
    for (std::size_t i = 0; i < n; ++i) {
        data[i] = static_cast<sunrealtype>(state[static_cast<Eigen::Index>(i)]);
    }
    return true;
}

[[nodiscard]] int projected_rhs_callback(sunrealtype t, N_Vector y, N_Vector ydot, void* user_data) {
    auto* data = static_cast<SundialsUserData*>(user_data);
    if (!data || !data->projected_stepper) {
        return -1;
    }

    Vector state;
    if (!load_state(y, data->dimension, state)) {
        data->last_projection_error = "failed to read N_Vector state";
        return -1;
    }

    Vector rhs;
    if (!data->projected_stepper->project_rhs(static_cast<Real>(t), state, rhs, data->last_projection_error)) {
        // Recoverable failure: let SUNDIALS shrink the step and retry.
        return 1;
    }

    auto* rhs_data = N_VGetArrayPointer(ydot);
    if (!rhs_data) {
        data->last_projection_error = "failed to write N_Vector derivative";
        return -1;
    }

    for (std::size_t i = 0; i < data->dimension; ++i) {
        rhs_data[i] = static_cast<sunrealtype>(rhs[static_cast<Eigen::Index>(i)]);
    }
    return 0;
}

[[nodiscard]] int direct_rhs_callback(sunrealtype t, N_Vector y, N_Vector ydot, void* user_data) {
    auto* data = static_cast<SundialsUserData*>(user_data);
    if (!data || !data->direct_rhs_stepper) {
        return -1;
    }

    Vector state;
    if (!load_state(y, data->dimension, state)) {
        data->last_projection_error = "failed to read N_Vector state";
        return -1;
    }

    Vector rhs;
    if (!data->direct_rhs_stepper->rhs_from_implicit_step(
            static_cast<Real>(t), state, rhs, data->last_projection_error)) {
        // Recoverable failure: let solver retry with smaller step.
        return 1;
    }

    auto* rhs_data = N_VGetArrayPointer(ydot);
    if (!rhs_data) {
        data->last_projection_error = "failed to write N_Vector derivative";
        return -1;
    }

    for (std::size_t i = 0; i < data->dimension; ++i) {
        rhs_data[i] = static_cast<sunrealtype>(rhs[static_cast<Eigen::Index>(i)]);
    }
    return 0;
}

[[nodiscard]] int ida_projected_residual(sunrealtype t,
                                         N_Vector y,
                                         N_Vector yp,
                                         N_Vector residual,
                                         void* user_data) {
    auto* data = static_cast<SundialsUserData*>(user_data);
    if (!data || !data->projected_stepper) {
        return -1;
    }

    Vector state;
    if (!load_state(y, data->dimension, state)) {
        data->last_projection_error = "failed to read N_Vector state";
        return -1;
    }

    Vector rhs;
    if (!data->projected_stepper->project_rhs(static_cast<Real>(t), state, rhs, data->last_projection_error)) {
        // Recoverable failure: let IDA retry with tighter internal step control.
        return 1;
    }

    auto* yp_data = N_VGetArrayPointer(yp);
    auto* r_data = N_VGetArrayPointer(residual);
    if (!yp_data || !r_data) {
        data->last_projection_error = "failed to access IDA vectors";
        return -1;
    }

    for (std::size_t i = 0; i < data->dimension; ++i) {
        r_data[i] = yp_data[i] - static_cast<sunrealtype>(rhs[static_cast<Eigen::Index>(i)]);
    }
    return 0;
}

[[nodiscard]] int ida_direct_residual(sunrealtype t,
                                      N_Vector y,
                                      N_Vector yp,
                                      N_Vector residual,
                                      void* user_data) {
    auto* data = static_cast<SundialsUserData*>(user_data);
    if (!data || !data->direct_assembler) {
        return -1;
    }

    Vector state;
    if (!load_state(y, data->dimension, state)) {
        data->last_projection_error = "failed to read N_Vector state";
        return -1;
    }

    Vector state_dot;
    if (!load_state(yp, data->dimension, state_dot)) {
        data->last_projection_error = "failed to read N_Vector derivative";
        return -1;
    }

    Vector f;
    if (!data->direct_assembler->compute_ida_residual(
            static_cast<Real>(t), state, state_dot, f, data->last_projection_error)) {
        // Recoverable failure: allow IDA to shrink and retry.
        return 1;
    }

    auto* r_data = N_VGetArrayPointer(residual);
    if (!r_data) {
        data->last_projection_error = "failed to access IDA residual vector";
        return -1;
    }
    for (std::size_t i = 0; i < data->dimension; ++i) {
        r_data[i] = static_cast<sunrealtype>(f[static_cast<Eigen::Index>(i)]);
    }
    return 0;
}

[[nodiscard]] int ida_direct_jtimes_setup(sunrealtype /*tt*/,
                                          N_Vector /*yy*/,
                                          N_Vector /*yp*/,
                                          N_Vector /*rr*/,
                                          sunrealtype /*c_j*/,
                                          void* /*user_data*/) {
    return 0;
}

[[nodiscard]] int ida_direct_jtimes(sunrealtype tt,
                                    N_Vector yy,
                                    N_Vector yp,
                                    N_Vector rr,
                                    N_Vector v,
                                    N_Vector Jv,
                                    sunrealtype c_j,
                                    void* user_data,
                                    N_Vector /*tmp1*/,
                                    N_Vector /*tmp2*/) {
    auto* data = static_cast<SundialsUserData*>(user_data);
    if (!data || !data->direct_assembler) {
        return -1;
    }

    Vector y_state;
    Vector yp_state;
    Vector direction;
    Vector residual_base;
    if (!load_state(yy, data->dimension, y_state) ||
        !load_state(yp, data->dimension, yp_state) ||
        !load_state(v, data->dimension, direction) ||
        !load_state(rr, data->dimension, residual_base)) {
        data->last_projection_error = "failed to read IDA vectors for J*v";
        return -1;
    }

    const Real v_norm = direction.norm();
    if (!(v_norm > 0.0) || !std::isfinite(v_norm)) {
        auto* out = N_VGetArrayPointer(Jv);
        if (!out) {
            data->last_projection_error = "failed to write IDA J*v vector";
            return -1;
        }
        for (std::size_t i = 0; i < data->dimension; ++i) {
            out[i] = SUN_RCONST(0.0);
        }
        return 0;
    }

    const Real y_norm = y_state.norm();
    const Real eps = std::sqrt(std::numeric_limits<Real>::epsilon());
    const Real sigma = std::max<Real>(eps * (1.0 + y_norm) / v_norm, 1e-12);

    Vector y_pert = y_state + sigma * direction;
    Vector yp_pert = yp_state + (sigma * static_cast<Real>(c_j)) * direction;

    Vector residual_pert;
    if (!data->direct_assembler->compute_ida_residual(
            static_cast<Real>(tt), y_pert, yp_pert, residual_pert, data->last_projection_error)) {
        return 1;
    }

    auto* out = N_VGetArrayPointer(Jv);
    if (!out) {
        data->last_projection_error = "failed to access IDA J*v output vector";
        return -1;
    }
    for (std::size_t i = 0; i < data->dimension; ++i) {
        out[i] = static_cast<sunrealtype>(
            (residual_pert[static_cast<Eigen::Index>(i)] -
             residual_base[static_cast<Eigen::Index>(i)]) /
            sigma);
    }
    return 0;
}

void configure_common_solver_limits(void* solver_mem,
                                    SundialsSolverFamily family,
                                    const SimulationOptions& options) {
    const Real dt_floor = std::max(options.dt_min, Real{1e-15});
    const Real dt_ceiling = std::max(options.dt_max, dt_floor);
    // Use a smaller startup step to reduce failures near discontinuities
    // (e.g. PWM edges / switch transitions).
    const Real dt0 = std::clamp(
        clamp_positive(options.dt * 0.25, std::max(dt_floor, Real{1e-12})),
        std::max(dt_floor * 10.0, Real{1e-12}),
        dt_ceiling);
    // Allow solver-level contraction below user dt_min during difficult solves.
    const Real min_step = std::max<Real>(dt_floor * 0.01, Real{1e-15});
    const long int max_steps = static_cast<long int>(std::max(options.sundials.max_steps, 1000));
    const int max_nl = std::max(options.sundials.max_nonlinear_iterations, 2);
    const int max_err_test_fails = 50;

    switch (family) {
        case SundialsSolverFamily::IDA:
            IDASetInitStep(solver_mem, dt0);
            IDASetMinStep(solver_mem, min_step);
            IDASetMaxStep(solver_mem, dt_ceiling);
            IDASetMaxNumSteps(solver_mem, max_steps);
            IDASetMaxNonlinIters(solver_mem, max_nl);
            IDASetMaxErrTestFails(solver_mem, max_err_test_fails);
            break;
        case SundialsSolverFamily::CVODE:
            CVodeSetInitStep(solver_mem, dt0);
            CVodeSetMinStep(solver_mem, min_step);
            CVodeSetMaxStep(solver_mem, dt_ceiling);
            CVodeSetMaxNumSteps(solver_mem, max_steps);
            CVodeSetMaxNonlinIters(solver_mem, max_nl);
            CVodeSetMaxErrTestFails(solver_mem, max_err_test_fails);
            break;
        case SundialsSolverFamily::ARKODE:
            ARKStepSetInitStep(solver_mem, dt0);
            ARKStepSetMinStep(solver_mem, min_step);
            ARKStepSetMaxStep(solver_mem, dt_ceiling);
            ARKStepSetMaxNumSteps(solver_mem, max_steps);
            ARKStepSetMaxNonlinIters(solver_mem, max_nl);
            ARKStepSetMaxErrTestFails(solver_mem, max_err_test_fails);
            break;
    }
}

[[nodiscard]] int saturating_int(long long value) {
    if (value > static_cast<long long>(INT_MAX)) {
        return INT_MAX;
    }
    if (value < static_cast<long long>(INT_MIN)) {
        return INT_MIN;
    }
    return static_cast<int>(value);
}

struct SolverCounters {
    long int steps = 0;
    long int nonlinear_iters = 0;
    long int function_evals = 0;
    long int jacobian_evals = 0;
    long int nonlinear_conv_fails = 0;
    long int error_test_fails = 0;
};

[[nodiscard]] bool read_solver_counters(void* solver_mem,
                                        SundialsSolverFamily family,
                                        SolverCounters& counters) {
    counters = {};
    int step_flag = 0;
    int nonlinear_flag = 0;

    auto read_optional = [](int flag, long int* value) {
        if (flag < 0 && value) {
            *value = 0;
        }
    };

    switch (family) {
        case SundialsSolverFamily::IDA:
            step_flag = IDAGetNumSteps(solver_mem, &counters.steps);
            nonlinear_flag = IDAGetNumNonlinSolvIters(solver_mem, &counters.nonlinear_iters);
            read_optional(IDAGetNumResEvals(solver_mem, &counters.function_evals), &counters.function_evals);
            read_optional(IDAGetNumJacEvals(solver_mem, &counters.jacobian_evals), &counters.jacobian_evals);
            read_optional(IDAGetNumNonlinSolvConvFails(solver_mem, &counters.nonlinear_conv_fails),
                          &counters.nonlinear_conv_fails);
            read_optional(IDAGetNumErrTestFails(solver_mem, &counters.error_test_fails),
                          &counters.error_test_fails);
            break;
        case SundialsSolverFamily::CVODE:
            step_flag = CVodeGetNumSteps(solver_mem, &counters.steps);
            nonlinear_flag = CVodeGetNumNonlinSolvIters(solver_mem, &counters.nonlinear_iters);
            read_optional(CVodeGetNumRhsEvals(solver_mem, &counters.function_evals), &counters.function_evals);
            read_optional(CVodeGetNumJacEvals(solver_mem, &counters.jacobian_evals), &counters.jacobian_evals);
            read_optional(CVodeGetNumNonlinSolvConvFails(solver_mem, &counters.nonlinear_conv_fails),
                          &counters.nonlinear_conv_fails);
            read_optional(CVodeGetNumErrTestFails(solver_mem, &counters.error_test_fails),
                          &counters.error_test_fails);
            break;
        case SundialsSolverFamily::ARKODE: {
            step_flag = ARKStepGetNumSteps(solver_mem, &counters.steps);
            nonlinear_flag = ARKStepGetNumNonlinSolvIters(solver_mem, &counters.nonlinear_iters);
            long int nfe = 0;
            long int nfi = 0;
            if (ARKStepGetNumRhsEvals(solver_mem, &nfe, &nfi) >= 0) {
                counters.function_evals = nfe + nfi;
            }
            read_optional(ARKStepGetNumJacEvals(solver_mem, &counters.jacobian_evals),
                          &counters.jacobian_evals);
            read_optional(ARKStepGetNumNonlinSolvConvFails(solver_mem, &counters.nonlinear_conv_fails),
                          &counters.nonlinear_conv_fails);
            read_optional(ARKStepGetNumErrTestFails(solver_mem, &counters.error_test_fails),
                          &counters.error_test_fails);
            break;
        }
    }
    return step_flag >= 0 && nonlinear_flag >= 0;
}

struct SundialsSwitchMonitor {
    std::string name;
    Index ctrl = -1;
    Index t1 = -1;
    Index t2 = -1;
    Real v_threshold = 0.0;
    Real g_on = 1e3;
    Real g_off = 1e-9;
    bool was_on = false;
};

[[nodiscard]] std::vector<SundialsSwitchMonitor> build_switch_monitors(
    const Circuit& circuit,
    const Vector& x0) {
    std::vector<SundialsSwitchMonitor> monitors;
    const auto& devices = circuit.devices();
    const auto& conns = circuit.connections();
    monitors.reserve(devices.size());

    for (std::size_t i = 0; i < devices.size() && i < conns.size(); ++i) {
        const auto* sw = std::get_if<VoltageControlledSwitch>(&devices[i]);
        if (!sw || conns[i].nodes.size() < 3) {
            continue;
        }

        SundialsSwitchMonitor monitor;
        monitor.name = conns[i].name;
        monitor.ctrl = conns[i].nodes[0];
        monitor.t1 = conns[i].nodes[1];
        monitor.t2 = conns[i].nodes[2];
        monitor.v_threshold = sw->v_threshold();
        monitor.g_on = sw->g_on();
        monitor.g_off = sw->g_off();
        if (monitor.ctrl >= 0 && monitor.ctrl < x0.size()) {
            monitor.was_on = x0[monitor.ctrl] > monitor.v_threshold;
        }
        monitors.push_back(std::move(monitor));
    }
    return monitors;
}

[[nodiscard]] bool estimate_switch_event_time(
    const SundialsSwitchMonitor& sw,
    Real t0,
    Real t1,
    const Vector& x0,
    const Vector& x1,
    Real& t_event,
    Vector& x_event) {
    if (t1 <= t0) {
        return false;
    }
    if (sw.ctrl < 0 || sw.ctrl >= x0.size() || sw.ctrl >= x1.size()) {
        return false;
    }

    const Real v0 = x0[sw.ctrl] - sw.v_threshold;
    const Real v1 = x1[sw.ctrl] - sw.v_threshold;

    if (!std::isfinite(v0) || !std::isfinite(v1)) {
        return false;
    }

    Real alpha = 1.0;
    const Real dv = v1 - v0;
    if (std::abs(dv) > 1e-15) {
        alpha = std::clamp((0.0 - v0) / dv, Real{0.0}, Real{1.0});
    }

    t_event = t0 + alpha * (t1 - t0);
    x_event = x0 + alpha * (x1 - x0);
    return true;
}

void record_switch_event_sample(const SundialsSwitchMonitor& sw,
                                Real t_event,
                                const Vector& x_state,
                                bool new_state,
                                SimulationResult& result,
                                EventCallback event_callback) {
    Real v_switch = 0.0;
    if (sw.t1 >= 0 && sw.t1 < x_state.size()) {
        v_switch += x_state[sw.t1];
    }
    if (sw.t2 >= 0 && sw.t2 < x_state.size()) {
        v_switch -= x_state[sw.t2];
    }

    const Real conductance = new_state ? sw.g_on : sw.g_off;
    const Real i_switch = conductance * v_switch;

    SimulationEvent evt;
    evt.time = t_event;
    evt.type = new_state ? SimulationEventType::SwitchOn : SimulationEventType::SwitchOff;
    evt.component = sw.name;
    evt.description = sw.name + (new_state ? " on" : " off");
    evt.value1 = v_switch;
    evt.value2 = i_switch;
    result.events.push_back(std::move(evt));

    if (event_callback) {
        SwitchEvent cb;
        cb.switch_name = sw.name;
        cb.time = t_event;
        cb.new_state = new_state;
        cb.voltage = v_switch;
        cb.current = i_switch;
        event_callback(cb);
    }
}

[[nodiscard]] Real next_periodic_boundary_time(Real t_now,
                                               Real period,
                                               Real phase_shift,
                                               Real boundary_in_period,
                                               Real min_gap) {
    if (!(period > 0.0) || !std::isfinite(period)) {
        return std::numeric_limits<Real>::infinity();
    }

    const Real b = std::clamp(boundary_in_period, Real{0.0}, period);
    const Real t_shifted = t_now + phase_shift;
    const Real cycles = std::floor((t_shifted - b) / period);
    Real candidate = (cycles + 1.0) * period + b - phase_shift;

    const Real target_min = t_now + std::max(min_gap, Real{1e-15});
    if (candidate <= target_min) {
        const Real delta = target_min - candidate;
        const Real jumps = std::ceil(delta / period);
        candidate += std::max<Real>(jumps, 1.0) * period;
    }
    return candidate;
}

void append_unique_boundary(std::vector<Real>& boundaries, Real value, Real period) {
    if (!std::isfinite(value)) {
        return;
    }
    const Real clamped = std::clamp(value, Real{0.0}, period);
    const Real tol = std::max<Real>(period * 1e-12, Real{1e-15});
    for (Real existing : boundaries) {
        if (std::abs(existing - clamped) <= tol) {
            return;
        }
    }
    boundaries.push_back(clamped);
}

[[nodiscard]] Real next_pwm_boundary_time(const Circuit& circuit,
                                          Real t_now,
                                          Real dt_min,
                                          std::string* source_name = nullptr) {
    Real next_time = std::numeric_limits<Real>::infinity();
    std::string next_name;
    const Real min_gap = std::max(std::max(std::abs(t_now) * 1e-14, dt_min * 0.01), Real{1e-15});
    const auto& devices = circuit.devices();
    const auto& conns = circuit.connections();

    for (std::size_t i = 0; i < devices.size() && i < conns.size(); ++i) {
        const auto* pwm = std::get_if<PWMVoltageSource>(&devices[i]);
        if (!pwm) {
            continue;
        }

        const auto& params = pwm->params();
        if (!(params.frequency > 0.0) || !std::isfinite(params.frequency)) {
            continue;
        }

        const Real period = 1.0 / params.frequency;
        if (!(period > 0.0) || !std::isfinite(period)) {
            continue;
        }

        const Real duty = std::clamp(pwm->duty_at(t_now), Real{0.0}, Real{1.0});
        Real t_on = duty * period - params.dead_time;
        t_on = std::clamp(t_on, Real{0.0}, period);

        std::vector<Real> boundaries;
        boundaries.reserve(5);
        append_unique_boundary(boundaries, 0.0, period);
        append_unique_boundary(boundaries, t_on, period);
        if (params.rise_time > 0.0) {
            append_unique_boundary(boundaries, params.rise_time, period);
        }
        if (params.fall_time > 0.0) {
            append_unique_boundary(boundaries, t_on + params.fall_time, period);
        }

        const Real phase_shift = params.phase / (2.0 * std::numbers::pi_v<Real>) * period;
        for (Real boundary : boundaries) {
            const Real candidate = next_periodic_boundary_time(
                t_now, period, phase_shift, boundary, min_gap);
            if (candidate < next_time) {
                next_time = candidate;
                next_name = conns[i].name;
            }
        }
    }

    if (source_name && std::isfinite(next_time)) {
        *source_name = next_name;
    }
    return next_time;
}

void record_pwm_boundary_event(const std::string& source_name,
                               Real t_event,
                               SimulationResult& result) {
    SimulationEvent evt;
    evt.time = t_event;
    evt.type = SimulationEventType::TimestepChange;
    evt.component = source_name;
    evt.description = "pwm_boundary";
    evt.value1 = t_event;
    evt.value2 = 0.0;
    result.events.push_back(std::move(evt));
}

[[nodiscard]] bool refresh_ida_derivative(sunrealtype t,
                                          N_Vector y,
                                          N_Vector yp,
                                          SundialsUserData& user_data,
                                          bool use_direct_formulation,
                                          std::string& error) {
    if (!yp) {
        error = "missing IDA derivative vector";
        return false;
    }
    if (use_direct_formulation) {
        N_VConst(SUN_RCONST(0.0), yp);
        return true;
    }
    if (projected_rhs_callback(t, y, yp, &user_data) != 0) {
        error = user_data.last_projection_error.empty()
            ? "failed to compute IDA initial derivative"
            : user_data.last_projection_error;
        return false;
    }
    return true;
}

[[nodiscard]] int enforce_ida_consistent_ic(void* solver_mem,
                                            sunrealtype t,
                                            const SimulationOptions& options) {
    const Real dt_floor = std::max(options.dt_min, Real{1e-12});
    const sunrealtype tout1 = t + static_cast<sunrealtype>(std::max(dt_floor * 10.0, Real{1e-11}));
    return IDACalcIC(solver_mem, IDA_YA_YDP_INIT, tout1);
}

[[nodiscard]] int reinitialize_solver(void* solver_mem,
                                      SundialsSolverFamily family,
                                      sunrealtype t,
                                      N_Vector y,
                                      N_Vector yp,
                                      SundialsUserData& user_data,
                                      bool use_direct_ida,
                                      N_Vector ida_id,
                                      const SimulationOptions& options) {
    int flag = 0;
    switch (family) {
        case SundialsSolverFamily::IDA: {
            if (use_direct_ida) {
                if (!yp) {
                    return -1;
                }
            } else if (projected_rhs_callback(t, y, yp, &user_data) != 0) {
                return -1;
            }
            flag = IDAReInit(solver_mem, t, y, yp);
            if (flag == 0 && use_direct_ida && ida_id) {
                flag = IDASetId(solver_mem, ida_id);
            }
            if (flag == 0 && use_direct_ida) {
                const int ic_flag = enforce_ida_consistent_ic(solver_mem, t, options);
                if (ic_flag < 0) {
                    flag = ic_flag;
                }
            }
            break;
        }
        case SundialsSolverFamily::CVODE:
            flag = CVodeReInit(solver_mem, t, y);
            break;
        case SundialsSolverFamily::ARKODE:
            flag = ARKStepReset(solver_mem, t, y);
            break;
    }
    if (flag == 0) {
        configure_common_solver_limits(solver_mem, family, options);
    }
    return flag;
}

#endif  // PULSIM_HAS_SUNDIALS

}  // namespace

SimulationResult run_sundials_backend(Circuit& circuit,
                                      const SimulationOptions& options,
                                      const Vector& x0,
                                      SimulationCallback callback,
                                      EventCallback event_callback,
                                      SimulationControl* control,
                                      bool escalated_from_native) {
    SimulationResult result;
    result.backend_telemetry.selected_backend = "sundials";
    result.backend_telemetry.solver_family = solver_family_to_string(options.sundials.family);
    result.backend_telemetry.formulation_mode =
        formulation_mode_to_string(options.sundials.formulation);
    result.backend_telemetry.sundials_compiled =
#ifdef PULSIM_HAS_SUNDIALS
        true;
#else
        false;
#endif
    result.backend_telemetry.sundials_used = false;
    result.backend_telemetry.escalation_count = escalated_from_native ? 1 : 0;
    result.backend_telemetry.reinitialization_count = 0;

    if (!options.sundials.enabled) {
        result.success = false;
        result.final_status = SolverStatus::MaxIterationsReached;
        result.message = "SUNDIALS backend requested but disabled in simulation.sundials.enabled";
        result.backend_telemetry.failure_reason = "sundials_backend_disabled";
        return result;
    }

    const bool direct_requested = options.sundials.formulation == SundialsFormulationMode::Direct;
    const bool direct_supported =
        options.sundials.family == SundialsSolverFamily::IDA ||
        options.sundials.family == SundialsSolverFamily::CVODE ||
        options.sundials.family == SundialsSolverFamily::ARKODE;
    if (direct_requested && !direct_supported) {
        // Keep deterministic behavior if a future/unknown family is requested.
        result.backend_telemetry.formulation_mode = "direct_requested_projected_wrapper";
    }

#ifndef PULSIM_HAS_SUNDIALS
    (void)circuit;
    (void)options;
    (void)x0;
    (void)callback;
    (void)event_callback;
    (void)control;
    (void)escalated_from_native;
    result.success = false;
    result.final_status = SolverStatus::MaxIterationsReached;
    result.message = "SUNDIALS backend requested but binary was built without SUNDIALS support";
    result.backend_telemetry.failure_reason = "sundials_not_compiled";
    return result;
#else
    auto wall_start = std::chrono::high_resolution_clock::now();
    const bool use_direct_ida =
        direct_requested && options.sundials.family == SundialsSolverFamily::IDA;
    const bool use_direct_rhs =
        direct_requested &&
        (options.sundials.family == SundialsSolverFamily::CVODE ||
         options.sundials.family == SundialsSolverFamily::ARKODE);

    const std::size_t n = static_cast<std::size_t>(x0.size());
    if (n == 0) {
        result.success = false;
        result.final_status = SolverStatus::NumericalError;
        result.message = "SUNDIALS backend received empty initial state";
        result.backend_telemetry.failure_reason = "invalid_initial_state";
        return result;
    }

    SUNContext sunctx = nullptr;
    if (SUNContext_Create(nullptr, &sunctx) != 0 || sunctx == nullptr) {
        result.success = false;
        result.final_status = SolverStatus::NumericalError;
        result.message = "SUNDIALS context creation failed";
        result.backend_telemetry.failure_reason = "sundials_context_create_failed";
        return result;
    }

    N_Vector y = N_VNew_Serial(static_cast<sunindextype>(n), sunctx);
    N_Vector yp = nullptr;
    N_Vector ida_id = nullptr;
    SUNLinearSolver linear_solver = nullptr;
    void* solver_mem = nullptr;

    auto cleanup = [&]() {
        if (solver_mem) {
            switch (options.sundials.family) {
                case SundialsSolverFamily::IDA:
                    IDAFree(&solver_mem);
                    break;
                case SundialsSolverFamily::CVODE:
                    CVodeFree(&solver_mem);
                    break;
                case SundialsSolverFamily::ARKODE:
                    ARKStepFree(&solver_mem);
                    break;
            }
        }
        if (linear_solver) {
            SUNLinSolFree(linear_solver);
        }
        if (yp) {
            N_VDestroy(yp);
        }
        if (ida_id) {
            N_VDestroy(ida_id);
        }
        if (y) {
            N_VDestroy(y);
        }
        if (sunctx) {
            SUNContext_Free(&sunctx);
        }
    };

    if (!y) {
        cleanup();
        result.success = false;
        result.final_status = SolverStatus::NumericalError;
        result.message = "SUNDIALS state vector allocation failed";
        result.backend_telemetry.failure_reason = "sundials_vector_allocation_failed";
        return result;
    }

    if (!store_state(y, x0)) {
        cleanup();
        result.success = false;
        result.final_status = SolverStatus::NumericalError;
        result.message = "Failed to initialize SUNDIALS state vector";
        result.backend_telemetry.failure_reason = "sundials_vector_init_failed";
        return result;
    }

    ProjectedImplicitStepper projection(circuit, options);
    DirectResidualAssembler direct_assembler(circuit);
    DirectRhsStepper direct_rhs_stepper(circuit, options);
    if (use_direct_ida) {
        direct_assembler.initialize_state(x0, options.tstart, std::max(options.dt, options.dt_min));
    } else if (use_direct_rhs) {
        direct_rhs_stepper.initialize_history(x0, options.tstart, std::max(options.dt, options.dt_min));
    } else {
        projection.initialize_history(x0, options.tstart, std::max(options.dt, options.dt_min));
    }
    auto switch_monitors = build_switch_monitors(circuit, x0);

    SundialsUserData user_data;
    user_data.projected_stepper = &projection;
    user_data.direct_assembler = &direct_assembler;
    user_data.direct_rhs_stepper = &direct_rhs_stepper;
    user_data.use_direct_formulation = use_direct_ida || use_direct_rhs;
    user_data.dimension = n;

    const Real rel_tol = clamp_positive(options.sundials.rel_tol, 1e-6);
    const Real abs_tol = clamp_positive(options.sundials.abs_tol, 1e-9);

    int init_flag = -1;
    switch (options.sundials.family) {
        case SundialsSolverFamily::IDA: {
            yp = N_VNew_Serial(static_cast<sunindextype>(n), sunctx);
            if (!yp) {
                cleanup();
                result.success = false;
                result.final_status = SolverStatus::NumericalError;
                result.message = "SUNDIALS IDA derivative vector allocation failed";
                result.backend_telemetry.failure_reason = "sundials_vector_allocation_failed";
                return result;
            }
            std::string derivative_error;
            if (!refresh_ida_derivative(static_cast<sunrealtype>(options.tstart),
                                        y,
                                        yp,
                                        user_data,
                                        use_direct_ida,
                                        derivative_error)) {
                cleanup();
                result.success = false;
                result.final_status = SolverStatus::NumericalError;
                result.message = "SUNDIALS IDA derivative initialization failed: " + derivative_error;
                result.backend_telemetry.failure_reason =
                    use_direct_ida ? "sundials_direct_init_failed" : "sundials_projection_failed";
                return result;
            }
            solver_mem = IDACreate(sunctx);
            if (!solver_mem) {
                cleanup();
                result.success = false;
                result.final_status = SolverStatus::NumericalError;
                result.message = "SUNDIALS IDA allocation failed";
                result.backend_telemetry.failure_reason = "sundials_solver_allocation_failed";
                return result;
            }
            init_flag = IDAInit(
                solver_mem,
                use_direct_ida ? ida_direct_residual : ida_projected_residual,
                options.tstart,
                y,
                yp);
            if (init_flag == 0) init_flag = IDASStolerances(solver_mem, rel_tol, abs_tol);
            if (init_flag == 0) init_flag = IDASetUserData(solver_mem, &user_data);
            if (init_flag == 0) {
                linear_solver = SUNLinSol_SPGMR(y, SUN_PREC_NONE, 0, sunctx);
                if (linear_solver) {
                    init_flag = IDASetLinearSolver(solver_mem, linear_solver, nullptr);
                }
            }
            if (init_flag == 0 && use_direct_ida && options.sundials.use_jacobian) {
                init_flag = IDASetJacTimes(
                    solver_mem,
                    ida_direct_jtimes_setup,
                    ida_direct_jtimes);
            }
            if (init_flag == 0 && use_direct_ida) {
                ida_id = N_VNew_Serial(static_cast<sunindextype>(n), sunctx);
                if (!ida_id || !build_ida_differential_id(circuit, ida_id)) {
                    init_flag = -1;
                }
            }
            if (init_flag == 0 && use_direct_ida && ida_id) {
                init_flag = IDASetId(solver_mem, ida_id);
            }
            if (init_flag == 0 && use_direct_ida) {
                const int ic_flag = enforce_ida_consistent_ic(
                    solver_mem,
                    static_cast<sunrealtype>(options.tstart),
                    options);
                if (ic_flag < 0) {
                    init_flag = ic_flag;
                }
            }
            break;
        }
        case SundialsSolverFamily::CVODE: {
            solver_mem = CVodeCreate(CV_BDF, sunctx);
            if (!solver_mem) {
                cleanup();
                result.success = false;
                result.final_status = SolverStatus::NumericalError;
                result.message = "SUNDIALS CVODE allocation failed";
                result.backend_telemetry.failure_reason = "sundials_solver_allocation_failed";
                return result;
            }
            init_flag = CVodeInit(
                solver_mem,
                use_direct_rhs ? direct_rhs_callback : projected_rhs_callback,
                options.tstart,
                y);
            if (init_flag == 0) init_flag = CVodeSStolerances(solver_mem, rel_tol, abs_tol);
            if (init_flag == 0) init_flag = CVodeSetUserData(solver_mem, &user_data);
            if (init_flag == 0) {
                linear_solver = SUNLinSol_SPGMR(y, SUN_PREC_NONE, 0, sunctx);
                if (linear_solver) {
                    init_flag = CVodeSetLinearSolver(solver_mem, linear_solver, nullptr);
                }
            }
            break;
        }
        case SundialsSolverFamily::ARKODE: {
            solver_mem = ARKStepCreate(
                nullptr,
                use_direct_rhs ? direct_rhs_callback : projected_rhs_callback,
                options.tstart,
                y,
                sunctx);
            if (!solver_mem) {
                cleanup();
                result.success = false;
                result.final_status = SolverStatus::NumericalError;
                result.message = "SUNDIALS ARKODE allocation failed";
                result.backend_telemetry.failure_reason = "sundials_solver_allocation_failed";
                return result;
            }
            init_flag = ARKStepSStolerances(solver_mem, rel_tol, abs_tol);
            if (init_flag == 0) init_flag = ARKStepSetUserData(solver_mem, &user_data);
            if (init_flag == 0) {
                linear_solver = SUNLinSol_SPGMR(y, SUN_PREC_NONE, 0, sunctx);
                if (linear_solver) {
                    init_flag = ARKStepSetLinearSolver(solver_mem, linear_solver, nullptr);
                }
            }
            break;
        }
    }

    if (init_flag != 0) {
        cleanup();
        result.success = false;
        result.final_status = SolverStatus::NumericalError;
        std::ostringstream oss;
        oss << "SUNDIALS initialization failed (flag=" << init_flag << ")";
        if (!user_data.last_projection_error.empty()) {
            oss << ": " << user_data.last_projection_error;
        }
        result.message = oss.str();
        result.backend_telemetry.failure_reason = "sundials_init_failed";
        return result;
    }

    configure_common_solver_limits(solver_mem, options.sundials.family, options);

    long long accumulated_solver_steps = 0;
    long long accumulated_nonlinear_iters = 0;
    long long accumulated_function_evals = 0;
    long long accumulated_jacobian_evals = 0;
    long long accumulated_nonlinear_fails = 0;
    long long accumulated_error_test_fails = 0;
    auto flush_solver_counters = [&]() {
        SolverCounters counters;
        if (!read_solver_counters(solver_mem, options.sundials.family, counters)) {
            return;
        }
        if (counters.steps > 0) {
            accumulated_solver_steps += static_cast<long long>(counters.steps);
        }
        if (counters.nonlinear_iters > 0) {
            accumulated_nonlinear_iters += static_cast<long long>(counters.nonlinear_iters);
        }
        if (counters.function_evals > 0) {
            accumulated_function_evals += static_cast<long long>(counters.function_evals);
        }
        if (counters.jacobian_evals > 0) {
            accumulated_jacobian_evals += static_cast<long long>(counters.jacobian_evals);
        }
        if (counters.nonlinear_conv_fails > 0) {
            accumulated_nonlinear_fails += static_cast<long long>(counters.nonlinear_conv_fails);
        }
        if (counters.error_test_fails > 0) {
            accumulated_error_test_fails += static_cast<long long>(counters.error_test_fails);
        }
    };

    const Real output_dt = std::clamp(
        clamp_positive(options.dt, std::max(options.dt_min, Real{1e-12})),
        std::max(options.dt_min, Real{1e-12}),
        std::max(options.dt_max, std::max(options.dt_min, Real{1e-12})));
    const Real output_tol = std::max<Real>(
        output_dt * 1e-6,
        std::max(options.dt_min * 4.0, Real{1e-15}));
    Real next_output_time = options.tstart + output_dt;

    Real t = options.tstart;
    Vector state = x0;
    result.time.push_back(t);
    result.states.push_back(state);
    if (callback) {
        callback(t, state);
    }

    while (t < options.tstop) {
        if (control) {
            if (control->should_stop()) {
                result.message = "Simulation stopped by user";
                break;
            }
            while (control->should_pause() && !control->should_stop()) {
                control->wait_until_resumed();
            }
            if (control->should_stop()) {
                result.message = "Simulation stopped by user";
                break;
            }
        }

        const Real t_prev = t;
        const Vector state_prev = state;

        std::string pwm_boundary_source;
        Real segment_target = options.tstop;
        if (options.enable_events) {
            const Real t_pwm = next_pwm_boundary_time(circuit, t, options.dt_min, &pwm_boundary_source);
            if (std::isfinite(t_pwm) && t_pwm < segment_target) {
                segment_target = t_pwm;
            }
        }

        sunrealtype tret = static_cast<sunrealtype>(t);
        const sunrealtype tout = static_cast<sunrealtype>(segment_target);
        int flag = 0;
        switch (options.sundials.family) {
            case SundialsSolverFamily::IDA:
                flag = IDASolve(solver_mem,
                                tout,
                                &tret,
                                y,
                                yp,
                                IDA_ONE_STEP);
                break;
            case SundialsSolverFamily::CVODE:
                flag = CVode(solver_mem,
                             tout,
                             y,
                             &tret,
                             CV_ONE_STEP);
                break;
            case SundialsSolverFamily::ARKODE:
                flag = ARKStepEvolve(solver_mem,
                                     tout,
                                     y,
                                     &tret,
                                     ARK_ONE_STEP);
                break;
        }

        if (flag < 0) {
            result.success = false;
            result.final_status = map_failure_status(flag);
            std::ostringstream oss;
            oss << "SUNDIALS solve failed (flag=" << flag << ")";
            if (!user_data.last_projection_error.empty()) {
                oss << ": " << user_data.last_projection_error;
            }
            result.message = oss.str();
            result.backend_telemetry.failure_reason = "sundials_step_failed";
            break;
        }

        Vector new_state;
        if (!load_state(y, n, new_state)) {
            result.success = false;
            result.final_status = SolverStatus::NumericalError;
            result.message = "Failed to read SUNDIALS state vector";
            result.backend_telemetry.failure_reason = "sundials_vector_read_failed";
            break;
        }

        const Real t_next = static_cast<Real>(tret);
        const Real dt_step = std::max(t_next - t, options.dt_min);
        if (!std::isfinite(t_next) || t_next <= t) {
            result.success = false;
            result.final_status = SolverStatus::NumericalError;
            result.message = "SUNDIALS produced non-increasing timestep";
            result.backend_telemetry.failure_reason = "sundials_invalid_timestep";
            break;
        }

        const Real boundary_tol = std::max<Real>(options.dt_min * 4.0, std::abs(segment_target) * 1e-12 + 1e-15);
        const bool reached_pwm_boundary =
            segment_target < options.tstop &&
            t_next >= (segment_target - boundary_tol);

        bool reinitialize_after_step = reached_pwm_boundary;
        if (reached_pwm_boundary && options.enable_events && !pwm_boundary_source.empty()) {
            record_pwm_boundary_event(pwm_boundary_source, std::min(t_next, segment_target), result);
        }

        if (options.enable_events && !switch_monitors.empty()) {
            for (auto& sw : switch_monitors) {
                if (sw.ctrl < 0 || sw.ctrl >= state_prev.size() || sw.ctrl >= new_state.size()) {
                    continue;
                }
                const Real v_ctrl_now = new_state[sw.ctrl];
                const bool now_on = v_ctrl_now > sw.v_threshold;
                if (now_on == sw.was_on) {
                    continue;
                }

                Real t_event = t_next;
                Vector x_event = new_state;
                (void)estimate_switch_event_time(sw, t_prev, t_next, state_prev, new_state, t_event, x_event);
                record_switch_event_sample(sw, t_event, x_event, now_on, result, event_callback);
                sw.was_on = now_on;
                reinitialize_after_step = true;
            }
        }

        t = t_next;
        state = std::move(new_state);
        if (use_direct_ida) {
            direct_assembler.accept_state(state, t, dt_step);
        } else if (use_direct_rhs) {
            direct_rhs_stepper.accept_state(state, t, dt_step);
        } else {
            projection.accept_state(state, t, dt_step);
        }

        ++result.total_steps;
        const bool reached_output_time = t >= (next_output_time - output_tol);
        const bool reached_final_time = (options.tstop - t) <= output_tol;
        if (reinitialize_after_step || reached_output_time || reached_final_time) {
            result.time.push_back(t);
            result.states.push_back(state);
            if (callback) {
                callback(t, state);
            }
            while (next_output_time <= t + output_tol) {
                next_output_time += output_dt;
            }
        }

        if (reinitialize_after_step) {
            flush_solver_counters();
            const int reinit_flag = reinitialize_solver(solver_mem,
                                                        options.sundials.family,
                                                        static_cast<sunrealtype>(t),
                                                        y,
                                                        yp,
                                                        user_data,
                                                        use_direct_ida,
                                                        ida_id,
                                                        options);
            if (reinit_flag != 0) {
                result.success = false;
                result.final_status = map_failure_status(reinit_flag);
                std::ostringstream oss;
                oss << "SUNDIALS reinitialization failed (flag=" << reinit_flag << ")";
                if (!user_data.last_projection_error.empty()) {
                    oss << ": " << user_data.last_projection_error;
                }
                result.message = oss.str();
                result.backend_telemetry.failure_reason = "sundials_reinit_failed";
                break;
            }
            result.backend_telemetry.reinitialization_count++;
        }
    }

    flush_solver_counters();
    if (result.total_steps == 0 && accumulated_solver_steps > 0) {
        result.total_steps = saturating_int(accumulated_solver_steps);
    }
    if (accumulated_nonlinear_iters > 0) {
        result.newton_iterations_total = saturating_int(accumulated_nonlinear_iters);
    }
    result.backend_telemetry.function_evaluations = saturating_int(accumulated_function_evals);
    result.backend_telemetry.jacobian_evaluations = saturating_int(accumulated_jacobian_evals);
    result.backend_telemetry.nonlinear_iterations = saturating_int(accumulated_nonlinear_iters);
    result.backend_telemetry.nonlinear_convergence_failures = saturating_int(accumulated_nonlinear_fails);
    result.backend_telemetry.error_test_failures = saturating_int(accumulated_error_test_fails);

    if (result.final_status == SolverStatus::Success && result.success) {
        result.success = true;
        result.final_status = SolverStatus::Success;
    }

    result.backend_telemetry.sundials_used = result.success;
    if (!result.success && result.backend_telemetry.failure_reason.empty()) {
        result.backend_telemetry.failure_reason = "sundials_backend_failed";
    }

    auto wall_end = std::chrono::high_resolution_clock::now();
    result.total_time_seconds = std::chrono::duration<double>(wall_end - wall_start).count();
    if (result.message.empty()) {
        result.message = result.success ? "SUNDIALS transient completed" : "SUNDIALS transient failed";
    }

    cleanup();
    return result;
#endif
}

}  // namespace pulsim::v1
