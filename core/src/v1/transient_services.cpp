#include "pulsim/v1/transient_services.hpp"

#include "pulsim/v1/simulation.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <memory>
#include <unordered_map>

namespace pulsim::v1 {

namespace {

constexpr std::uint64_t kFnvOffset = 1469598103934665603ULL;
constexpr std::uint64_t kFnvPrime = 1099511628211ULL;

[[nodiscard]] inline bool nearly_same_value(Real a, Real b) {
    const Real scale = std::max<Real>({Real{1.0}, std::abs(a), std::abs(b)});
    return std::abs(a - b) <= scale * Real{1e-12};
}

inline void hash_mix(std::uint64_t& seed, std::uint64_t value) {
    seed ^= value;
    seed *= kFnvPrime;
}

class DefaultEquationAssemblerService final : public EquationAssemblerService {
public:
    DefaultEquationAssemblerService(Circuit& circuit, const SimulationOptions& options)
        : circuit_(circuit)
        , options_(options) {}

    void assemble_system(const Vector& x,
                         Real t_next,
                         Real dt,
                         SparseMatrix& jacobian,
                         Vector& residual) override {
        const auto start = std::chrono::steady_clock::now();
        prepare_state(t_next, dt);
        circuit_.assemble_jacobian(jacobian, residual, x);
        apply_transient_gmin(jacobian, residual, x);
        const auto end = std::chrono::steady_clock::now();
        telemetry_.system_calls += 1;
        telemetry_.system_time_seconds += std::chrono::duration<double>(end - start).count();
    }

    void assemble_residual(const Vector& x,
                           Real t_next,
                           Real dt,
                           Vector& residual) override {
        const auto start = std::chrono::steady_clock::now();
        prepare_state(t_next, dt);
        circuit_.assemble_residual(residual, x);
        apply_transient_gmin_residual(residual, x);
        const auto end = std::chrono::steady_clock::now();
        telemetry_.residual_calls += 1;
        telemetry_.residual_time_seconds += std::chrono::duration<double>(end - start).count();
    }

    void set_transient_gmin(Real gmin) override {
        transient_gmin_ = std::max<Real>(0.0, gmin);
    }

    [[nodiscard]] Real transient_gmin() const override {
        return transient_gmin_;
    }

    [[nodiscard]] EquationAssemblerTelemetry telemetry() const override {
        return telemetry_;
    }

    void reset_telemetry() override {
        telemetry_ = {};
    }

private:
    void prepare_state(Real t_next, Real dt) {
        const Real dt_safe = std::max(dt, options_.dt_min);
        if (state_cached_ &&
            nearly_same_value(t_next, cached_t_next_) &&
            nearly_same_value(dt_safe, cached_dt_)) {
            return;
        }
        circuit_.set_current_time(t_next);
        circuit_.set_timestep(dt_safe);
        cached_t_next_ = t_next;
        cached_dt_ = dt_safe;
        state_cached_ = true;
    }

    void apply_transient_gmin(SparseMatrix& jacobian, Vector& residual, const Vector& x) const {
        if (transient_gmin_ <= 0.0) {
            return;
        }
        for (Index i = 0; i < circuit_.num_nodes(); ++i) {
            jacobian.coeffRef(i, i) += transient_gmin_;
            residual[i] += transient_gmin_ * x[i];
        }
    }

    void apply_transient_gmin_residual(Vector& residual, const Vector& x) const {
        if (transient_gmin_ <= 0.0) {
            return;
        }
        for (Index i = 0; i < circuit_.num_nodes(); ++i) {
            residual[i] += transient_gmin_ * x[i];
        }
    }

    Circuit& circuit_;
    const SimulationOptions& options_;
    Real transient_gmin_ = 0.0;
    bool state_cached_ = false;
    Real cached_t_next_ = std::numeric_limits<Real>::quiet_NaN();
    Real cached_dt_ = std::numeric_limits<Real>::quiet_NaN();
    EquationAssemblerTelemetry telemetry_{};
};

class DefaultSegmentModelService final : public SegmentModelService {
public:
    DefaultSegmentModelService(Circuit& circuit,
                               std::shared_ptr<EquationAssemblerService> assembler)
        : circuit_(circuit)
        , assembler_(std::move(assembler)) {}

    [[nodiscard]] SegmentModel build_model(
        const Vector& x_now,
        const TransientStepRequest& request) const override {
        SegmentModel model;
        model.t_now = request.t_now;
        model.t_target = request.t_target;
        model.dt = request.dt_candidate;

        bool has_strong_nonlinearity = false;
        std::uint64_t signature = kFnvOffset;
        hash_mix(signature, static_cast<std::uint64_t>(circuit_.num_nodes()));
        hash_mix(signature, static_cast<std::uint64_t>(circuit_.num_branches()));
        hash_mix(signature, request.mode == TransientStepMode::Fixed ? 0xF1 : 0xA5);
        hash_mix(signature, request.event_adjacent ? 0xE1 : 0xE0);

        const auto& devices = circuit_.devices();
        const auto& conns = circuit_.connections();
        for (std::size_t i = 0; i < devices.size() && i < conns.size(); ++i) {
            hash_mix(signature, static_cast<std::uint64_t>(i + 1));
            const auto& conn = conns[i];
            std::visit([&](const auto& dev) {
                using T = std::decay_t<decltype(dev)>;
                if constexpr (std::is_same_v<T, VoltageControlledSwitch>) {
                    Real v_ctrl = 0.0;
                    if (!conn.nodes.empty() && conn.nodes[0] >= 0 &&
                        conn.nodes[0] < x_now.size()) {
                        v_ctrl = x_now[conn.nodes[0]];
                    }
                    hash_mix(signature, 0x10);
                    hash_mix(signature, v_ctrl > dev.v_threshold() ? 0x1 : 0x0);
                } else if constexpr (std::is_same_v<T, IdealSwitch>) {
                    hash_mix(signature, 0x11);
                    hash_mix(signature, dev.is_closed() ? 0x1 : 0x0);
                } else if constexpr (std::is_same_v<T, IdealDiode> ||
                                     std::is_same_v<T, MOSFET> ||
                                     std::is_same_v<T, IGBT>) {
                    has_strong_nonlinearity = true;
                    hash_mix(signature, 0x20);
                } else {
                    hash_mix(signature, 0x30);
                }
            }, devices[i]);
        }

        model.topology_signature = signature;
        model.admissible = !has_strong_nonlinearity;
        if (!model.admissible) {
            model.classification = "segment_not_admissible_nonlinear_device";
            return model;
        }

        const Real dt_safe = std::max(request.dt_candidate, request.dt_min);
        if (!(std::isfinite(dt_safe) && dt_safe > 0.0)) {
            model.classification = "segment_invalid_dt";
            return model;
        }

        SparseMatrix jacobian(x_now.size(), x_now.size());
        Vector residual = Vector::Zero(x_now.size());
        assembler_->assemble_system(x_now, request.t_target, dt_safe, jacobian, residual);
        if (!residual.allFinite() || jacobian.rows() != x_now.size() ||
            jacobian.cols() != x_now.size()) {
            model.classification = "segment_assembly_non_finite";
            return model;
        }

        auto linear_model = std::make_shared<SegmentLinearStateSpace>();
        linear_model->E = jacobian;
        linear_model->A = jacobian;
        linear_model->B.resize(x_now.size(), 0);
        linear_model->u.resize(0);
        linear_model->c = -residual;

        model.linear_model = std::move(linear_model);
        model.classification = "piecewise_linear_segment";

        auto cache_it = topology_cache_.find(signature);
        if (cache_it == topology_cache_.end()) {
            CacheEntry entry;
            entry.state_size = x_now.size();
            entry.nonzeros = jacobian.nonZeros();
            entry.build_count = 1;
            topology_cache_.emplace(signature, std::move(entry));
            model.cache_hit = false;
        } else {
            cache_it->second.state_size = x_now.size();
            cache_it->second.nonzeros = jacobian.nonZeros();
            cache_it->second.build_count += 1;
            model.cache_hit = true;
        }

        return model;
    }

private:
    struct CacheEntry {
        Index state_size = 0;
        Index nonzeros = 0;
        std::uint64_t build_count = 0;
    };

    Circuit& circuit_;
    std::shared_ptr<EquationAssemblerService> assembler_;
    mutable std::unordered_map<std::uint64_t, CacheEntry> topology_cache_;
};

class DefaultSegmentStepperService final : public SegmentStepperService {
public:
    DefaultSegmentStepperService(
        std::shared_ptr<EquationAssemblerService> assembler,
        std::shared_ptr<LinearSolveService> linear_solve)
        : assembler_(std::move(assembler))
        , linear_solve_(std::move(linear_solve)) {}

    [[nodiscard]] SegmentStepOutcome try_advance(
        const SegmentModel& model,
        const Vector& x_now,
        const TransientStepRequest& request) override {
        SegmentStepOutcome outcome;
        outcome.path = model.admissible
            ? SegmentSolvePath::StateSpacePrimary
            : SegmentSolvePath::DaeFallback;

        if (!model.admissible) {
            outcome.requires_fallback = true;
            outcome.reason = model.classification;
            return outcome;
        }

        if (!model.linear_model) {
            outcome.requires_fallback = true;
            outcome.reason = model.classification.empty()
                ? "segment_missing_linear_model"
                : model.classification;
            return outcome;
        }

        const Real dt_safe = std::max(request.dt_candidate, request.dt_min);
        if (!(std::isfinite(dt_safe) && dt_safe > 0.0)) {
            outcome.requires_fallback = true;
            outcome.reason = "segment_invalid_dt";
            return outcome;
        }

        const auto& linear_model = *model.linear_model;
        if (linear_model.E.rows() != x_now.size() ||
            linear_model.E.cols() != x_now.size() ||
            linear_model.A.rows() != x_now.size() ||
            linear_model.A.cols() != x_now.size() ||
            linear_model.c.size() != x_now.size()) {
            outcome.requires_fallback = true;
            outcome.reason = "segment_invalid_linear_model_dims";
            return outcome;
        }

        if (!linear_model.c.allFinite()) {
            outcome.requires_fallback = true;
            outcome.reason = "segment_residual_non_finite";
            return outcome;
        }

        Vector rhs = linear_model.A * x_now + linear_model.c;
        if (linear_model.B.cols() > 0) {
            if (linear_model.B.rows() != x_now.size() ||
                linear_model.B.cols() != linear_model.u.size()) {
                outcome.requires_fallback = true;
                outcome.reason = "segment_invalid_input_dims";
                return outcome;
            }
            rhs += linear_model.B * linear_model.u;
        }

        if (!rhs.allFinite()) {
            outcome.requires_fallback = true;
            outcome.reason = "segment_rhs_non_finite";
            return outcome;
        }

        auto& linear = linear_solve_->solver();
        if (!linear.analyze(linear_model.E) || !linear.factorize(linear_model.E)) {
            outcome.requires_fallback = true;
            outcome.reason = "segment_linear_factorization_failed";
            return outcome;
        }

        const auto x_next_result = linear.solve(rhs);
        if (!x_next_result) {
            outcome.requires_fallback = true;
            outcome.reason = "segment_linear_solve_failed";
            return outcome;
        }

        Vector x_next = x_next_result.value();
        if (!x_next.allFinite()) {
            outcome.requires_fallback = true;
            outcome.reason = "segment_solution_non_finite";
            return outcome;
        }

        Vector residual_next = Vector::Zero(x_now.size());
        assembler_->assemble_residual(x_next, request.t_target, dt_safe, residual_next);
        if (!residual_next.allFinite()) {
            outcome.requires_fallback = true;
            outcome.reason = "segment_post_residual_non_finite";
            return outcome;
        }

        const Real r0 = (-linear_model.c).lpNorm<Eigen::Infinity>();
        const Real r1 = residual_next.lpNorm<Eigen::Infinity>();
        const Real r0_safe = std::max<Real>(r0, 1e-16);
        if (r1 > r0_safe * 1.05) {
            outcome.requires_fallback = true;
            outcome.reason = "segment_residual_not_improved";
            return outcome;
        }

        outcome.result.solution = std::move(x_next);
        outcome.result.status = SolverStatus::Success;
        outcome.result.iterations = 1;
        outcome.result.final_residual = r1;
        outcome.result.final_weighted_error = (outcome.result.solution - x_now).lpNorm<Eigen::Infinity>();
        outcome.requires_fallback = false;
        outcome.reason = "state_space_linearized_step";
        return outcome;
    }

private:
    std::shared_ptr<EquationAssemblerService> assembler_;
    std::shared_ptr<LinearSolveService> linear_solve_;
};

class DefaultNonlinearSolveService final : public NonlinearSolveService {
public:
    DefaultNonlinearSolveService(NewtonRaphsonSolver<RuntimeLinearSolver>& solver,
                                 std::shared_ptr<EquationAssemblerService> assembler)
        : solver_(solver)
        , assembler_(std::move(assembler)) {}

    [[nodiscard]] NewtonResult solve(const Vector& x_guess,
                                     Real t_next,
                                     Real dt) override {
        auto system_func = [this, t_next, dt](const Vector& x, Vector& residual, SparseMatrix& jacobian) {
            assembler_->assemble_system(x, t_next, dt, jacobian, residual);
        };

        auto residual_func = [this, t_next, dt](const Vector& x, Vector& residual) {
            assembler_->assemble_residual(x, t_next, dt, residual);
        };

        return solver_.solve(x_guess, system_func, residual_func);
    }

    [[nodiscard]] const NewtonOptions& options() const override {
        return solver_.options();
    }

    void set_options(const NewtonOptions& options) override {
        solver_.set_options(options);
    }

private:
    NewtonRaphsonSolver<RuntimeLinearSolver>& solver_;
    std::shared_ptr<EquationAssemblerService> assembler_;
};

class DefaultLinearSolveService final : public LinearSolveService {
public:
    explicit DefaultLinearSolveService(RuntimeLinearSolver& solver)
        : solver_(solver) {}

    [[nodiscard]] RuntimeLinearSolver& solver() override {
        return solver_;
    }

    [[nodiscard]] const LinearSolverStackConfig& config() const override {
        return solver_.config();
    }

private:
    RuntimeLinearSolver& solver_;
};

class DefaultEventSchedulerService final : public EventSchedulerService {
public:
    [[nodiscard]] Real next_segment_target(const TransientStepRequest& request,
                                           Real t_stop) const override {
        if (request.t_target > request.t_now && request.t_target < t_stop) {
            return request.t_target;
        }
        return t_stop;
    }
};

class DefaultRecoveryManagerService final : public RecoveryManagerService {
public:
    explicit DefaultRecoveryManagerService(Real backoff)
        : backoff_(std::clamp(backoff, Real{0.1}, Real{0.95})) {}

    [[nodiscard]] RecoveryDecision on_step_failure(
        const TransientStepRequest& request) const override {
        RecoveryDecision decision;

        if (request.retry_index + 1 >= request.max_retries) {
            decision.stage = RecoveryStage::Abort;
            decision.abort = true;
            decision.next_dt = request.dt_candidate;
            decision.reason = "max_retries_exceeded";
            return decision;
        }

        decision.stage = RecoveryStage::DtBackoff;
        decision.next_dt = std::max(request.dt_candidate * backoff_, request.dt_min);
        decision.abort = false;
        decision.reason = "dt_backoff";
        return decision;
    }

private:
    Real backoff_;
};

class DefaultTelemetryCollectorService final : public TelemetryCollectorService {
public:
    void on_step_attempt(const TransientStepRequest& /*request*/) override {}
    void on_step_accept(Real /*t_next*/, const Vector& /*state*/) override {}
    void on_step_reject(const RecoveryDecision& /*decision*/) override {}
};

}  // namespace

TransientServiceRegistry make_default_transient_service_registry(
    Circuit& circuit,
    const SimulationOptions& options,
    NewtonRaphsonSolver<RuntimeLinearSolver>& newton_solver) {

    TransientServiceRegistry registry;
    registry.supports_fixed_mode = true;
    registry.supports_variable_mode = true;

    registry.equation_assembler =
        std::make_shared<DefaultEquationAssemblerService>(circuit, options);

    registry.nonlinear_solve =
        std::make_shared<DefaultNonlinearSolveService>(newton_solver,
                                                       registry.equation_assembler);

    registry.segment_model =
        std::make_shared<DefaultSegmentModelService>(circuit, registry.equation_assembler);

    registry.linear_solve =
        std::make_shared<DefaultLinearSolveService>(newton_solver.linear_solver());

    registry.segment_stepper =
        std::make_shared<DefaultSegmentStepperService>(
            registry.equation_assembler,
            registry.linear_solve);

    registry.event_scheduler =
        std::make_shared<DefaultEventSchedulerService>();

    registry.recovery_manager =
        std::make_shared<DefaultRecoveryManagerService>(options.stiffness_config.dt_backoff);

    registry.telemetry_collector =
        std::make_shared<DefaultTelemetryCollectorService>();

    return registry;
}

}  // namespace pulsim::v1
