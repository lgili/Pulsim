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

class DefaultLossService final : public LossService {
public:
    DefaultLossService(Circuit& circuit, const SimulationOptions& options)
        : circuit_(circuit)
        , options_(options) {
        reset();
    }

    void reset() override {
        const auto& devices = circuit_.devices();
        const auto& conns = circuit_.connections();
        states_.assign(devices.size(), DeviceLossState{});
        switching_energy_.assign(devices.size(), std::nullopt);
        diode_conducting_.assign(devices.size(), false);
        last_device_power_.assign(devices.size(), 0.0);
        name_to_index_.clear();

        for (std::size_t i = 0; i < conns.size(); ++i) {
            const auto& name = conns[i].name;
            name_to_index_[name] = i;
            auto it = options_.switching_energy.find(name);
            if (it != options_.switching_energy.end()) {
                switching_energy_[i] = it->second;
            }
        }
    }

    void commit_switching_event(const std::string& name,
                                bool turning_on,
                                Real energy) override {
        if (!options_.enable_losses || energy <= 0.0) {
            return;
        }

        const auto index = index_for(name);
        if (!index.has_value()) {
            return;
        }

        auto& state = states_[*index];
        state.accumulator.add_switching_event(energy);
        if (turning_on) {
            state.switching_energy.turn_on += energy;
        } else {
            state.switching_energy.turn_off += energy;
        }
    }

    void commit_reverse_recovery_event(const std::string& name,
                                       Real energy) override {
        if (!options_.enable_losses || energy <= 0.0) {
            return;
        }

        const auto index = index_for(name);
        if (!index.has_value()) {
            return;
        }

        auto& state = states_[*index];
        state.accumulator.add_switching_event(energy);
        state.switching_energy.reverse_recovery += energy;
    }

    void commit_accepted_segment(const Vector& x,
                                 Real dt,
                                 const std::vector<Real>& thermal_scale) override {
        if (!options_.enable_losses || dt <= 0.0) {
            return;
        }

        const auto& devices = circuit_.devices();
        const auto& conns = circuit_.connections();

        if (last_device_power_.size() != devices.size()) {
            last_device_power_.assign(devices.size(), 0.0);
        } else {
            std::fill(last_device_power_.begin(), last_device_power_.end(), 0.0);
        }

        auto node_voltage = [&x](Index node) -> Real {
            return (node >= 0) ? x[node] : 0.0;
        };
        auto thermal_factor = [&thermal_scale](std::size_t device_index) -> Real {
            if (device_index >= thermal_scale.size()) {
                return 1.0;
            }
            const Real value = thermal_scale[device_index];
            if (!std::isfinite(value)) {
                return 1.0;
            }
            return std::clamp(value, Real{0.05}, Real{4.0});
        };

        for (std::size_t i = 0; i < devices.size(); ++i) {
            const auto& conn = conns[i];
            Real p_cond = 0.0;

            std::visit([&](const auto& dev) {
                using T = std::decay_t<decltype(dev)>;

                if constexpr (std::is_same_v<T, Resistor>) {
                    const Real v = node_voltage(conn.nodes[0]) - node_voltage(conn.nodes[1]);
                    p_cond = (v * v) / dev.resistance();
                } else if constexpr (std::is_same_v<T, IdealSwitch>) {
                    const Real g = dev.is_closed() ? dev.g_on() : dev.g_off();
                    const Real v = node_voltage(conn.nodes[0]) - node_voltage(conn.nodes[1]);
                    const Real i_dev = g * v;
                    p_cond = std::abs(v * i_dev);
                } else if constexpr (std::is_same_v<T, VoltageControlledSwitch>) {
                    const Real v_ctrl = node_voltage(conn.nodes[0]);
                    const bool on = v_ctrl > dev.v_threshold();
                    const Real g = on ? dev.g_on() : dev.g_off();
                    const Real v = node_voltage(conn.nodes[1]) - node_voltage(conn.nodes[2]);
                    const Real i_dev = g * v;
                    p_cond = std::abs(v * i_dev);
                } else if constexpr (std::is_same_v<T, IdealDiode>) {
                    const Real v = node_voltage(conn.nodes[0]) - node_voltage(conn.nodes[1]);
                    const Real g = dev.is_conducting() ? dev.g_on() : dev.g_off();
                    const Real i_dev = g * v;
                    p_cond = std::max<Real>(0.0, v * i_dev);

                    const bool conducting = dev.is_conducting();
                    if (diode_conducting_[i] && !conducting) {
                        const auto& energy_opt = switching_energy_[i];
                        if (energy_opt.has_value() && energy_opt->err > 0.0) {
                            commit_reverse_recovery_event(conn.name, energy_opt->err);
                        }
                    }
                    diode_conducting_[i] = conducting;
                } else if constexpr (std::is_same_v<T, MOSFET>) {
                    const Real vg = node_voltage(conn.nodes[0]);
                    const Real vd = node_voltage(conn.nodes[1]);
                    const Real vs = node_voltage(conn.nodes[2]);
                    const auto params = dev.params();

                    const Real sign = params.is_nmos ? 1.0 : -1.0;
                    const Real vgs = sign * (vg - vs);
                    const Real vds = sign * (vd - vs);

                    Real id = 0.0;
                    if (vgs <= params.vth) {
                        id = params.g_off * vds;
                    } else if (vds < vgs - params.vth) {
                        const Real vov = vgs - params.vth;
                        id = params.kp * (vov * vds - 0.5 * vds * vds) * (1.0 + params.lambda * vds);
                    } else {
                        const Real vov = vgs - params.vth;
                        id = 0.5 * params.kp * vov * vov * (1.0 + params.lambda * vds);
                    }

                    id *= sign;
                    p_cond = std::abs((vd - vs) * id);
                } else if constexpr (std::is_same_v<T, IGBT>) {
                    const Real vg = node_voltage(conn.nodes[0]);
                    const Real vc = node_voltage(conn.nodes[1]);
                    const Real ve = node_voltage(conn.nodes[2]);
                    const auto params = dev.params();

                    const Real vge = vg - ve;
                    const Real vce = vc - ve;
                    const bool on = (vge > params.vth) && (vce > 0);
                    const Real g = on ? params.g_on : params.g_off;
                    const Real i_dev = g * vce;
                    p_cond = std::abs(vce * i_dev);
                }
            }, devices[i]);

            p_cond *= thermal_factor(i);
            const Real p_clamped = std::max<Real>(0.0, p_cond);
            last_device_power_[i] = p_clamped;

            if (p_clamped > 0.0) {
                auto& state = states_[i];
                state.accumulator.add_sample(p_clamped, dt);
                state.peak_power = std::max(state.peak_power, p_clamped);
            }
        }
    }

    [[nodiscard]] const std::vector<Real>& last_device_power() const override {
        return last_device_power_;
    }

    [[nodiscard]] SystemLossSummary finalize(Real duration) const override {
        SystemLossSummary summary;
        if (!options_.enable_losses) {
            return summary;
        }

        const auto& conns = circuit_.connections();
        for (std::size_t i = 0; i < states_.size() && i < conns.size(); ++i) {
            const auto& state = states_[i];
            if (state.accumulator.num_samples() == 0 &&
                state.accumulator.switching_energy() == 0.0) {
                continue;
            }

            LossResult res;
            res.device_name = conns[i].name;
            res.total_energy = state.accumulator.total_energy();
            res.average_power = duration > 0.0 ? res.total_energy / duration : 0.0;
            res.peak_power = state.peak_power;

            const Real conduction_energy = state.accumulator.conduction_energy();
            const Real switching_energy = state.accumulator.switching_energy();
            res.breakdown.conduction = duration > 0.0 ? conduction_energy / duration : 0.0;
            res.breakdown.turn_on = duration > 0.0 ? state.switching_energy.turn_on / duration : 0.0;
            res.breakdown.turn_off = duration > 0.0 ? state.switching_energy.turn_off / duration : 0.0;
            res.breakdown.reverse_recovery = duration > 0.0
                ? state.switching_energy.reverse_recovery / duration
                : 0.0;

            if (res.breakdown.turn_on == 0.0 &&
                res.breakdown.turn_off == 0.0 &&
                res.breakdown.reverse_recovery == 0.0) {
                res.breakdown.turn_on = duration > 0.0 ? switching_energy / duration : 0.0;
            }

            summary.device_losses.push_back(std::move(res));
        }

        summary.compute_totals();
        return summary;
    }

private:
    struct DeviceLossState {
        LossAccumulator accumulator;
        LossBreakdown switching_energy{};
        Real peak_power = 0.0;
    };

    [[nodiscard]] std::optional<std::size_t> index_for(const std::string& name) const {
        const auto it = name_to_index_.find(name);
        if (it == name_to_index_.end()) {
            return std::nullopt;
        }
        return it->second;
    }

    Circuit& circuit_;
    const SimulationOptions& options_;
    std::vector<DeviceLossState> states_;
    std::vector<std::optional<SwitchingEnergy>> switching_energy_;
    std::vector<bool> diode_conducting_;
    std::unordered_map<std::string, std::size_t> name_to_index_;
    std::vector<Real> last_device_power_;
};

class DefaultThermalService final : public ThermalService {
public:
    DefaultThermalService(Circuit& circuit, const SimulationOptions& options)
        : circuit_(circuit)
        , options_(options) {
        reset();
    }

    void reset() override {
        const auto& devices = circuit_.devices();
        const auto& conns = circuit_.connections();
        states_.assign(devices.size(), DeviceThermalState{});
        thermal_scale_.assign(devices.size(), 1.0);

        if (!options_.thermal.enable) {
            circuit_.reset_device_temperature_scales();
            return;
        }

        for (std::size_t i = 0; i < devices.size(); ++i) {
            bool supports_thermal = false;
            std::visit([&](const auto& dev) {
                using T = std::decay_t<decltype(dev)>;
                supports_thermal = device_traits<T>::has_thermal_model;
            }, devices[i]);

            if (!supports_thermal) {
                continue;
            }

            auto& state = states_[i];
            state.enabled = true;
            state.config.enabled = true;
            state.config.rth = options_.thermal.default_rth;
            state.config.cth = options_.thermal.default_cth;
            state.config.temp_init = options_.thermal.ambient;
            state.config.temp_ref = options_.thermal.ambient;

            if (i < conns.size()) {
                const auto cfg_it = options_.thermal_devices.find(conns[i].name);
                if (cfg_it != options_.thermal_devices.end()) {
                    state.config = cfg_it->second;
                }
            }

            state.enabled = state.config.enabled;
            state.temperature = state.config.temp_init;
            state.peak_temperature = state.temperature;
            state.sum_temperature = 0.0;
            state.samples = 0;
        }

        refresh_scales();
    }

    [[nodiscard]] Real thermal_scale_factor(std::size_t device_index) const override {
        if (device_index >= thermal_scale_.size()) {
            return 1.0;
        }
        return thermal_scale_[device_index];
    }

    [[nodiscard]] const std::vector<Real>& thermal_scale_vector() const override {
        return thermal_scale_;
    }

    void commit_accepted_segment(Real dt,
                                 const std::vector<Real>& device_power) override {
        if (!options_.thermal.enable || dt <= 0.0) {
            return;
        }
        if (states_.size() != device_power.size()) {
            return;
        }

        for (std::size_t i = 0; i < states_.size(); ++i) {
            auto& state = states_[i];
            if (!state.enabled) {
                continue;
            }

            const Real power = std::max<Real>(0.0, device_power[i]);
            const Real ambient = options_.thermal.ambient;
            const Real rth = std::max<Real>(state.config.rth, 1e-12);
            const Real cth = state.config.cth;

            if (cth <= 0.0) {
                state.temperature = ambient + power * rth;
            } else {
                const Real tau = std::max<Real>(rth * cth, 1e-12);
                const Real delta = state.temperature - ambient;
                const Real delta_dot = (power * rth - delta) / tau;
                state.temperature = ambient + delta + dt * delta_dot;
            }

            state.peak_temperature = std::max(state.peak_temperature, state.temperature);
            state.sum_temperature += state.temperature;
            state.samples += 1;
        }

        refresh_scales();
    }

    [[nodiscard]] ThermalServiceSummary finalize() const override {
        ThermalServiceSummary summary;
        summary.enabled = options_.thermal.enable;
        summary.ambient = options_.thermal.ambient;
        summary.max_temperature = options_.thermal.ambient;

        if (!options_.thermal.enable) {
            return summary;
        }

        const auto& conns = circuit_.connections();
        for (std::size_t i = 0; i < states_.size() && i < conns.size(); ++i) {
            const auto& state = states_[i];
            if (!state.enabled) {
                continue;
            }

            ThermalDeviceSummaryEntry entry;
            entry.device_name = conns[i].name;
            entry.enabled = true;
            entry.final_temperature = state.temperature;
            entry.peak_temperature = state.peak_temperature;
            entry.average_temperature = (state.samples > 0)
                ? state.sum_temperature / static_cast<Real>(state.samples)
                : state.temperature;
            summary.max_temperature = std::max(summary.max_temperature, entry.peak_temperature);
            summary.device_temperatures.push_back(std::move(entry));
        }

        return summary;
    }

private:
    struct DeviceThermalState {
        bool enabled = false;
        ThermalDeviceConfig config{};
        Real temperature = 25.0;
        Real peak_temperature = 25.0;
        Real sum_temperature = 0.0;
        int samples = 0;
    };

    [[nodiscard]] Real compute_scale(const DeviceThermalState& state) const {
        if (!options_.thermal.enable ||
            options_.thermal.policy == ThermalCouplingPolicy::LossOnly ||
            !state.enabled) {
            return 1.0;
        }

        const Real raw = 1.0 + state.config.alpha * (state.temperature - state.config.temp_ref);
        return std::clamp(raw, Real{0.05}, Real{4.0});
    }

    void refresh_scales() {
        if (thermal_scale_.size() != states_.size()) {
            thermal_scale_.assign(states_.size(), 1.0);
        }

        for (std::size_t i = 0; i < states_.size(); ++i) {
            thermal_scale_[i] = compute_scale(states_[i]);
        }
        circuit_.set_device_temperature_scales(thermal_scale_);
    }

    Circuit& circuit_;
    const SimulationOptions& options_;
    std::vector<DeviceThermalState> states_;
    std::vector<Real> thermal_scale_;
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

    registry.thermal_service =
        std::make_shared<DefaultThermalService>(circuit, options);

    registry.loss_service =
        std::make_shared<DefaultLossService>(circuit, options);

    return registry;
}

}  // namespace pulsim::v1
