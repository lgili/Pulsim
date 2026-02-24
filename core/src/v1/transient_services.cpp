#include "pulsim/v1/transient_services.hpp"

#include "pulsim/v1/simulation.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <functional>
#include <limits>
#include <memory>
#include <numbers>
#include <optional>
#include <unordered_map>
#include <vector>

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

[[nodiscard]] std::uint64_t hash_sparse_numeric_signature(const SparseMatrix& matrix) {
    std::uint64_t hash = kFnvOffset;
    hash_mix(hash, static_cast<std::uint64_t>(matrix.rows()));
    hash_mix(hash, static_cast<std::uint64_t>(matrix.cols()));
    hash_mix(hash, static_cast<std::uint64_t>(matrix.nonZeros()));
    for (Index col = 0; col < matrix.outerSize(); ++col) {
        for (SparseMatrix::InnerIterator it(matrix, col); it; ++it) {
            hash_mix(hash, static_cast<std::uint64_t>(it.row() + 1));
            hash_mix(hash, static_cast<std::uint64_t>(it.col() + 1));
            const auto value_hash = static_cast<std::uint64_t>(std::hash<Real>{}(it.value()));
            hash_mix(hash, value_hash);
        }
    }
    return hash;
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
        const std::uint64_t matrix_hash = hash_sparse_numeric_signature(linear_model.E);
        const bool can_reuse_factorization =
            factorization_valid_ &&
            cached_topology_signature_ == model.topology_signature &&
            cached_matrix_hash_ == matrix_hash &&
            cached_state_size_ == linear_model.E.rows();

        LinearSolveResult x_next_result = LinearSolveResult::failure("segment_linear_not_attempted");
        if (can_reuse_factorization) {
            x_next_result = linear.solve(rhs);
            if (x_next_result.has_value()) {
                outcome.linear_factor_cache_hit = true;
            } else {
                factorization_valid_ = false;
            }
        }

        if (!x_next_result.has_value()) {
            outcome.linear_factor_cache_miss = true;
            if (!linear.analyze(linear_model.E) || !linear.factorize(linear_model.E)) {
                factorization_valid_ = false;
                outcome.requires_fallback = true;
                outcome.reason = "segment_linear_factorization_failed";
                return outcome;
            }

            x_next_result = linear.solve(rhs);
            if (!x_next_result) {
                factorization_valid_ = false;
                outcome.requires_fallback = true;
                outcome.reason = "segment_linear_solve_failed";
                return outcome;
            }

            factorization_valid_ = true;
            cached_topology_signature_ = model.topology_signature;
            cached_matrix_hash_ = matrix_hash;
            cached_state_size_ = linear_model.E.rows();
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
        outcome.reason = outcome.linear_factor_cache_hit
            ? "state_space_linearized_step_cache_hit"
            : "state_space_linearized_step_cache_miss";
        return outcome;
    }

private:
    std::shared_ptr<EquationAssemblerService> assembler_;
    std::shared_ptr<LinearSolveService> linear_solve_;
    bool factorization_valid_ = false;
    std::uint64_t cached_topology_signature_ = 0;
    std::uint64_t cached_matrix_hash_ = 0;
    Index cached_state_size_ = 0;
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
    DefaultEventSchedulerService(const Circuit& circuit, const SimulationOptions& options)
        : circuit_(circuit)
        , options_(options) {
        if (!options_.enable_events) {
            return;
        }

        const auto& devices = circuit_.devices();
        for (std::size_t i = 0; i < devices.size(); ++i) {
            if (std::holds_alternative<PWMVoltageSource>(devices[i])) {
                pwm_source_indices_.push_back(i);
            } else if (std::holds_alternative<PulseVoltageSource>(devices[i])) {
                pulse_source_indices_.push_back(i);
            }
        }

        const auto& virtual_components = circuit_.virtual_components();
        for (std::size_t i = 0; i < virtual_components.size(); ++i) {
            if (virtual_components[i].type == "pwm_generator") {
                virtual_pwm_indices_.push_back(i);
            }
        }
    }

    [[nodiscard]] Real next_segment_target(const TransientStepRequest& request,
                                           Real t_stop) const override {
        const Real tol = time_tolerance(request.t_now, t_stop, request.dt_min);
        Real target = t_stop;

        auto consider = [&](Real candidate) {
            if (!std::isfinite(candidate)) {
                return;
            }
            if (candidate <= request.t_now + tol) {
                return;
            }
            if (candidate < target - tol) {
                target = candidate;
            }
        };

        if (request.t_target > request.t_now + tol) {
            consider(std::min(request.t_target, t_stop));
        }

        consider(request.pwm_boundary_time);
        consider(request.dead_time_boundary_time);
        consider(request.threshold_crossing_time);

        if (options_.enable_events) {
            if (const auto waveform_boundary =
                    next_waveform_boundary(request.t_now, target, request.dt_min)) {
                consider(*waveform_boundary);
            }
        }

        if (target > t_stop) {
            target = t_stop;
        }
        if (target <= request.t_now + tol) {
            return t_stop;
        }
        return target;
    }

private:
    [[nodiscard]] static Real time_tolerance(Real t_now, Real t_stop, Real dt_min) {
        const Real scale = std::max<Real>({Real{1.0}, std::abs(t_now), std::abs(t_stop)});
        const Real dt_floor = std::max<Real>(std::abs(dt_min), Real{1e-15});
        return std::max<Real>(scale * Real{1e-12}, dt_floor * Real{1e-3});
    }

    [[nodiscard]] static std::optional<Real> next_periodic_boundary(
        Real t_now,
        Real t_stop,
        Real dt_min,
        Real period,
        Real phase_shift,
        std::vector<Real> boundaries) {
        if (!(std::isfinite(period) && period > 0.0)) {
            return std::nullopt;
        }

        const Real tol = time_tolerance(t_now, t_stop, dt_min);
        if (boundaries.empty()) {
            boundaries.push_back(0.0);
        }
        boundaries.push_back(period);
        for (auto& boundary : boundaries) {
            if (!std::isfinite(boundary)) {
                boundary = 0.0;
            }
            boundary = std::clamp(boundary, Real{0.0}, period);
        }

        std::sort(boundaries.begin(), boundaries.end());
        boundaries.erase(std::unique(boundaries.begin(),
                                     boundaries.end(),
                                     [&](Real lhs, Real rhs) {
                                         return std::abs(lhs - rhs) <= tol;
                                     }),
                         boundaries.end());

        const Real t_phase = t_now + phase_shift;
        Real t_mod = std::fmod(t_phase, period);
        if (t_mod < 0.0) {
            t_mod += period;
        }

        Real best = std::numeric_limits<Real>::infinity();
        for (Real boundary : boundaries) {
            Real delta = boundary - t_mod;
            if (delta <= tol) {
                delta += period;
            }
            if (!(delta > tol)) {
                continue;
            }
            const Real candidate = t_now + delta;
            if (candidate <= t_stop + tol && candidate < best) {
                best = candidate;
            }
        }

        if (!std::isfinite(best)) {
            return std::nullopt;
        }
        return best;
    }

    [[nodiscard]] static std::optional<Real> next_pwm_boundary(
        const PWMVoltageSource& source,
        Real t_now,
        Real t_stop,
        Real dt_min) {
        const auto& params = source.params();
        if (!(std::isfinite(params.frequency) && params.frequency > 0.0)) {
            return std::nullopt;
        }

        const Real period = Real{1.0} / params.frequency;
        const Real duty = std::clamp(source.duty_at(t_now), Real{0.0}, Real{1.0});
        const Real dead_time = std::max<Real>(params.dead_time, Real{0.0});
        const Real t_on = std::clamp(duty * period - dead_time, Real{0.0}, period);

        std::vector<Real> boundaries{Real{0.0}, t_on};
        if (params.rise_time > 0.0) {
            boundaries.push_back(std::clamp(params.rise_time, Real{0.0}, period));
        }
        if (params.fall_time > 0.0) {
            boundaries.push_back(std::clamp(t_on + params.fall_time, Real{0.0}, period));
        }

        const Real phase_shift =
            params.phase / (Real{2.0} * std::numbers::pi_v<Real>) * period;

        return next_periodic_boundary(t_now, t_stop, dt_min, period, phase_shift, std::move(boundaries));
    }

    [[nodiscard]] static std::optional<Real> next_virtual_pwm_boundary(
        const VirtualComponent& component,
        Real t_now,
        Real t_stop,
        Real dt_min) {
        const auto get_numeric = [&](const std::string& key, Real fallback) {
            const auto it = component.numeric_params.find(key);
            return (it != component.numeric_params.end()) ? it->second : fallback;
        };

        const Real frequency = std::max<Real>(get_numeric("frequency", 1e3), 1.0);
        const Real period = Real{1.0} / frequency;
        Real duty = get_numeric("duty", 0.5);
        if (get_numeric("duty_from_input", 0.0) > 0.5) {
            duty = get_numeric("duty_offset", duty);
        }
        const Real duty_min = std::clamp(get_numeric("duty_min", 0.0), Real{0.0}, Real{1.0});
        const Real duty_max = std::clamp(get_numeric("duty_max", 1.0), Real{0.0}, Real{1.0});
        if (duty_min <= duty_max) {
            duty = std::clamp(duty, duty_min, duty_max);
        } else {
            duty = std::clamp(duty, duty_max, duty_min);
        }
        const Real dead_time = std::max<Real>(get_numeric("dead_time", 0.0), Real{0.0});
        const Real t_on = std::clamp(duty * period - dead_time, Real{0.0}, period);
        const Real phase_shift =
            get_numeric("phase", 0.0) / (Real{2.0} * std::numbers::pi_v<Real>) * period;

        std::vector<Real> boundaries{Real{0.0}, t_on};
        return next_periodic_boundary(t_now, t_stop, dt_min, period, phase_shift, std::move(boundaries));
    }

    [[nodiscard]] static std::optional<Real> next_pulse_boundary(
        const PulseVoltageSource& source,
        Real t_now,
        Real t_stop,
        Real dt_min) {
        const auto& params = source.params();
        const Real tol = time_tolerance(t_now, t_stop, dt_min);
        const Real t_rise = std::max<Real>(params.t_rise, 0.0);
        const Real t_width = std::max<Real>(params.t_width, 0.0);
        const Real t_fall = std::max<Real>(params.t_fall, 0.0);
        const std::array<Real, 4> edge_offsets{
            Real{0.0},
            t_rise,
            t_rise + t_width,
            t_rise + t_width + t_fall
        };

        auto consider = [&](Real candidate, Real& best) {
            if (!std::isfinite(candidate)) {
                return;
            }
            if (candidate <= t_now + tol) {
                return;
            }
            if (candidate <= t_stop + tol && candidate < best) {
                best = candidate;
            }
        };

        Real best = std::numeric_limits<Real>::infinity();
        if (!(params.period > 0.0 && std::isfinite(params.period))) {
            for (Real edge : edge_offsets) {
                consider(params.t_delay + edge, best);
            }
        } else {
            const Real period = params.period;
            const Real cycle_position = (t_now - params.t_delay) / period;
            const auto base_cycle = static_cast<long long>(std::floor(cycle_position));
            for (long long cycle = base_cycle - 1; cycle <= base_cycle + 2; ++cycle) {
                const Real cycle_start = params.t_delay + static_cast<Real>(cycle) * period;
                for (Real edge : edge_offsets) {
                    consider(cycle_start + edge, best);
                }
                consider(cycle_start + period, best);
            }
        }

        if (!std::isfinite(best)) {
            return std::nullopt;
        }
        return best;
    }

    [[nodiscard]] std::optional<Real> next_waveform_boundary(
        Real t_now,
        Real t_stop,
        Real dt_min) const {
        if (!(t_stop > t_now)) {
            return std::nullopt;
        }

        const auto& devices = circuit_.devices();
        const auto& virtual_components = circuit_.virtual_components();
        const Real tol = time_tolerance(t_now, t_stop, dt_min);
        Real best = std::numeric_limits<Real>::infinity();

        auto consider = [&](const std::optional<Real>& candidate) {
            if (!candidate.has_value()) {
                return;
            }
            if (!std::isfinite(*candidate)) {
                return;
            }
            if (*candidate <= t_now + tol) {
                return;
            }
            if (*candidate <= t_stop + tol && *candidate < best) {
                best = *candidate;
            }
        };

        for (std::size_t index : pwm_source_indices_) {
            if (index >= devices.size()) {
                continue;
            }
            if (const auto* source = std::get_if<PWMVoltageSource>(&devices[index])) {
                consider(next_pwm_boundary(*source, t_now, t_stop, dt_min));
            }
        }

        for (std::size_t index : pulse_source_indices_) {
            if (index >= devices.size()) {
                continue;
            }
            if (const auto* source = std::get_if<PulseVoltageSource>(&devices[index])) {
                consider(next_pulse_boundary(*source, t_now, t_stop, dt_min));
            }
        }

        for (std::size_t index : virtual_pwm_indices_) {
            if (index >= virtual_components.size()) {
                continue;
            }
            consider(next_virtual_pwm_boundary(virtual_components[index], t_now, t_stop, dt_min));
        }

        if (!std::isfinite(best)) {
            return std::nullopt;
        }
        return best;
    }

    const Circuit& circuit_;
    const SimulationOptions& options_;
    std::vector<std::size_t> pwm_source_indices_;
    std::vector<std::size_t> pulse_source_indices_;
    std::vector<std::size_t> virtual_pwm_indices_;
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

        const int failure_index = std::max(0, request.retry_index);
        if (failure_index == 0) {
            decision.stage = RecoveryStage::DtBackoff;
            decision.next_dt = std::max(request.dt_candidate * backoff_, request.dt_min);
            decision.abort = false;
            decision.reason = "recovery_stage_dt_backoff";
            return decision;
        }

        if (failure_index == 1) {
            decision.stage = RecoveryStage::GlobalizationEscalation;
            decision.next_dt = std::max(request.dt_candidate * std::min(backoff_, Real{0.85}),
                                        request.dt_min);
            decision.abort = false;
            decision.reason = "recovery_stage_globalization";
            return decision;
        }

        if (failure_index == 2) {
            decision.stage = RecoveryStage::StiffProfile;
            decision.next_dt = std::max(request.dt_candidate * std::min(backoff_, Real{0.75}),
                                        request.dt_min);
            decision.abort = false;
            decision.reason = "recovery_stage_stiff_profile";
            return decision;
        }

        decision.stage = RecoveryStage::Regularization;
        decision.next_dt = std::max(request.dt_candidate * std::min(backoff_, Real{0.65}),
                                    request.dt_min);
        decision.abort = false;
        decision.reason = "recovery_stage_regularization";
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
        std::make_shared<DefaultEventSchedulerService>(circuit, options);

    registry.recovery_manager =
        std::make_shared<DefaultRecoveryManagerService>(options.stiffness_config.dt_backoff);

    registry.telemetry_collector =
        std::make_shared<DefaultTelemetryCollectorService>();

    return registry;
}

}  // namespace pulsim::v1
