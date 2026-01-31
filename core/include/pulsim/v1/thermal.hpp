#pragma once

// =============================================================================
// PulsimCore - Thermal Simulation Module
// =============================================================================
// Provides lumped-element thermal modeling including:
// - Foster thermal networks (series of parallel RC stages)
// - Cauer thermal networks (ladder of series R with shunt C)
// - Junction temperature calculation
// - Electro-thermal coupling support
//
// Key equations:
// - Heat flow: Q = (T1 - T2) / Rth  [W]
// - Temperature rate: dT/dt = Q / Cth  [K/s]
// - Junction temp: Tj = Tamb + Ploss * Rth_ja
// - Zth(t) = Rth * (1 - exp(-t/tau)), where tau = Rth * Cth
// =============================================================================

#include "pulsim/v1/numeric_types.hpp"
#include <vector>
#include <string>
#include <cmath>
#include <stdexcept>
#include <numeric>

namespace pulsim::v1 {

// =============================================================================
// Foster Network Stage
// =============================================================================

/// Single stage of a Foster thermal network (parallel RC)
/// Zth_i(t) = Rth_i * (1 - exp(-t/tau_i))
struct FosterStage {
    Real Rth;   ///< Thermal resistance (K/W or C/W)
    Real tau;   ///< Time constant (s) = Rth * Cth

    /// Compute thermal capacitance from Rth and tau
    [[nodiscard]] Real Cth() const { return tau / Rth; }

    /// Thermal impedance at time t for step power
    [[nodiscard]] Real Zth(Real t) const {
        if (t <= 0) return 0.0;
        return Rth * (1.0 - std::exp(-t / tau));
    }

    /// Temperature rise for constant power P at time t
    [[nodiscard]] Real delta_T(Real P, Real t) const {
        return P * Zth(t);
    }
};

// =============================================================================
// Foster Thermal Network
// =============================================================================

/// Foster thermal network representation
/// Total: Zth(t) = sum_i { Rth_i * (1 - exp(-t/tau_i)) }
///
/// Typical use: datasheet provides (Rth_i, tau_i) pairs from Zth curve fitting
class FosterNetwork {
public:
    FosterNetwork() = default;

    /// Create from parallel stages
    explicit FosterNetwork(std::vector<FosterStage> stages, std::string name = "")
        : stages_(std::move(stages)), name_(std::move(name)) {
        compute_total_Rth();
    }

    /// Create from Rth and tau vectors
    FosterNetwork(const std::vector<Real>& Rth_values,
                  const std::vector<Real>& tau_values,
                  const std::string& name = "")
        : name_(name) {
        if (Rth_values.size() != tau_values.size()) {
            throw std::invalid_argument("Rth and tau vectors must have same size");
        }
        stages_.reserve(Rth_values.size());
        for (std::size_t i = 0; i < Rth_values.size(); ++i) {
            stages_.push_back({Rth_values[i], tau_values[i]});
        }
        compute_total_Rth();
    }

    /// Add a stage
    void add_stage(Real Rth, Real tau) {
        stages_.push_back({Rth, tau});
        total_Rth_ += Rth;
    }

    /// Number of stages
    [[nodiscard]] std::size_t num_stages() const { return stages_.size(); }

    /// Get stage by index
    [[nodiscard]] const FosterStage& stage(std::size_t i) const { return stages_.at(i); }

    /// Total thermal resistance (steady-state)
    [[nodiscard]] Real total_Rth() const { return total_Rth_; }

    /// Thermal impedance at time t
    [[nodiscard]] Real Zth(Real t) const {
        Real z = 0.0;
        for (const auto& s : stages_) {
            z += s.Zth(t);
        }
        return z;
    }

    /// Temperature rise for constant power P at time t
    [[nodiscard]] Real delta_T(Real P, Real t) const {
        return P * Zth(t);
    }

    /// Steady-state temperature rise for power P
    [[nodiscard]] Real delta_T_ss(Real P) const {
        return P * total_Rth_;
    }

    /// Generate Zth(t) curve
    [[nodiscard]] std::vector<std::pair<Real, Real>> Zth_curve(
        Real t_start, Real t_end, std::size_t num_points) const {
        std::vector<std::pair<Real, Real>> curve;
        curve.reserve(num_points);
        Real dt = (t_end - t_start) / static_cast<Real>(num_points - 1);
        for (std::size_t i = 0; i < num_points; ++i) {
            Real t = t_start + i * dt;
            curve.emplace_back(t, Zth(t));
        }
        return curve;
    }

    [[nodiscard]] const std::string& name() const { return name_; }

    /// Get all stages
    [[nodiscard]] const std::vector<FosterStage>& stages() const { return stages_; }

private:
    std::vector<FosterStage> stages_;
    Real total_Rth_ = 0.0;
    std::string name_;

    void compute_total_Rth() {
        total_Rth_ = 0.0;
        for (const auto& s : stages_) {
            total_Rth_ += s.Rth;
        }
    }
};

// =============================================================================
// Cauer Network Stage
// =============================================================================

/// Single stage of a Cauer thermal network (series R, shunt C)
/// Physically represents a thermal layer
struct CauerStage {
    Real Rth;   ///< Thermal resistance of this layer (K/W)
    Real Cth;   ///< Thermal capacitance of this layer (J/K)

    /// Time constant of this layer
    [[nodiscard]] Real tau() const { return Rth * Cth; }
};

// =============================================================================
// Cauer Thermal Network
// =============================================================================

/// Cauer thermal network representation (ladder network)
/// Physically meaningful: each stage represents a thermal layer
/// (e.g., junction-to-case, case-to-heatsink, heatsink-to-ambient)
class CauerNetwork {
public:
    CauerNetwork() = default;

    /// Create from stages
    explicit CauerNetwork(std::vector<CauerStage> stages, std::string name = "")
        : stages_(std::move(stages)), name_(std::move(name)) {
        compute_total_Rth();
    }

    /// Create from Rth and Cth vectors
    CauerNetwork(const std::vector<Real>& Rth_values,
                 const std::vector<Real>& Cth_values,
                 const std::string& name = "")
        : name_(name) {
        if (Rth_values.size() != Cth_values.size()) {
            throw std::invalid_argument("Rth and Cth vectors must have same size");
        }
        stages_.reserve(Rth_values.size());
        for (std::size_t i = 0; i < Rth_values.size(); ++i) {
            stages_.push_back({Rth_values[i], Cth_values[i]});
        }
        compute_total_Rth();
    }

    /// Add a stage (layer)
    void add_stage(Real Rth, Real Cth) {
        stages_.push_back({Rth, Cth});
        total_Rth_ += Rth;
    }

    /// Number of stages (thermal layers)
    [[nodiscard]] std::size_t num_stages() const { return stages_.size(); }

    /// Get stage by index (0 = junction side)
    [[nodiscard]] const CauerStage& stage(std::size_t i) const { return stages_.at(i); }

    /// Total thermal resistance (steady-state)
    [[nodiscard]] Real total_Rth() const { return total_Rth_; }

    /// Total thermal capacitance
    [[nodiscard]] Real total_Cth() const {
        Real c = 0.0;
        for (const auto& s : stages_) {
            c += s.Cth;
        }
        return c;
    }

    /// Steady-state temperature rise for power P
    [[nodiscard]] Real delta_T_ss(Real P) const {
        return P * total_Rth_;
    }

    [[nodiscard]] const std::string& name() const { return name_; }

    /// Get all stages
    [[nodiscard]] const std::vector<CauerStage>& stages() const { return stages_; }

private:
    std::vector<CauerStage> stages_;
    Real total_Rth_ = 0.0;
    std::string name_;

    void compute_total_Rth() {
        total_Rth_ = 0.0;
        for (const auto& s : stages_) {
            total_Rth_ += s.Rth;
        }
    }
};

// =============================================================================
// Thermal Node State
// =============================================================================

/// State of a thermal node during transient simulation
struct ThermalNodeState {
    Real temperature;           ///< Current temperature (K or C)
    std::vector<Real> stage_T;  ///< Per-stage temperature for Foster network
};

// =============================================================================
// Thermal Simulator
// =============================================================================

/// Thermal simulator using Foster network model
/// Simulates junction temperature transients given power loss waveform
class ThermalSimulator {
public:
    /// Create simulator with Foster network and ambient temperature
    ThermalSimulator(const FosterNetwork& network, Real T_ambient = 25.0)
        : network_(network), T_ambient_(T_ambient) {
        reset();
    }

    /// Reset to ambient temperature
    void reset() {
        state_.temperature = T_ambient_;
        state_.stage_T.assign(network_.num_stages(), 0.0);
        time_ = 0.0;
    }

    /// Set ambient temperature
    void set_ambient(Real T_amb) { T_ambient_ = T_amb; }

    /// Get ambient temperature
    [[nodiscard]] Real ambient() const { return T_ambient_; }

    /// Get current junction temperature
    [[nodiscard]] Real junction_temperature() const { return state_.temperature; }

    /// Alias for junction_temperature
    [[nodiscard]] Real Tj() const { return junction_temperature(); }

    /// Get current time
    [[nodiscard]] Real time() const { return time_; }

    /// Get thermal network
    [[nodiscard]] const FosterNetwork& network() const { return network_; }

    /// Step simulation with constant power for duration dt
    /// Uses exponential integration for accuracy
    void step(Real power, Real dt) {
        if (dt <= 0) return;

        // Update each Foster stage using exponential integration
        // dT_i/dt = (P * Rth_i - T_i) / tau_i
        // Solution: T_i(t+dt) = P*Rth_i + (T_i(t) - P*Rth_i) * exp(-dt/tau_i)

        Real total_delta_T = 0.0;
        for (std::size_t i = 0; i < network_.num_stages(); ++i) {
            const auto& stage = network_.stage(i);
            Real T_ss = power * stage.Rth;  // Steady-state for this stage
            Real exp_factor = std::exp(-dt / stage.tau);
            state_.stage_T[i] = T_ss + (state_.stage_T[i] - T_ss) * exp_factor;
            total_delta_T += state_.stage_T[i];
        }

        state_.temperature = T_ambient_ + total_delta_T;
        time_ += dt;
    }

    /// Compute steady-state temperature for given power
    [[nodiscard]] Real steady_state_temperature(Real power) const {
        return T_ambient_ + network_.delta_T_ss(power);
    }

    /// Run simulation for power waveform
    /// @param times Time points (must be sorted)
    /// @param powers Power values at each time point
    /// @return Temperature waveform
    [[nodiscard]] std::vector<Real> simulate(
        const std::vector<Real>& times,
        const std::vector<Real>& powers) {

        if (times.size() != powers.size()) {
            throw std::invalid_argument("times and powers must have same size");
        }
        if (times.empty()) {
            return {};
        }

        reset();
        std::vector<Real> temperatures;
        temperatures.reserve(times.size());

        // Initial temperature
        temperatures.push_back(state_.temperature);

        for (std::size_t i = 1; i < times.size(); ++i) {
            Real dt = times[i] - times[i-1];
            // Use power sample at interval end (zero-order hold from current sample)
            Real P = powers[i];
            step(P, dt);
            temperatures.push_back(state_.temperature);
        }

        return temperatures;
    }

    /// Compute Zth(t) step response curve
    [[nodiscard]] std::vector<std::pair<Real, Real>> Zth_curve(
        Real t_end, std::size_t num_points, Real P_step = 1.0) {

        reset();
        std::vector<std::pair<Real, Real>> curve;
        curve.reserve(num_points);

        Real dt = t_end / static_cast<Real>(num_points - 1);
        curve.emplace_back(0.0, 0.0);  // Zth(0) = 0

        for (std::size_t i = 1; i < num_points; ++i) {
            step(P_step, dt);
            Real Zth = (state_.temperature - T_ambient_) / P_step;
            curve.emplace_back(time_, Zth);
        }

        return curve;
    }

    /// Get per-stage temperature rises
    [[nodiscard]] const std::vector<Real>& stage_temperatures() const {
        return state_.stage_T;
    }

private:
    FosterNetwork network_;
    Real T_ambient_;
    Real time_ = 0.0;
    ThermalNodeState state_;
};

// =============================================================================
// Thermal Limit Monitor
// =============================================================================

/// Monitors junction temperature against limits
class ThermalLimitMonitor {
public:
    /// Create monitor with temperature limits
    ThermalLimitMonitor(Real T_warning = 125.0, Real T_max = 150.0)
        : T_warning_(T_warning), T_max_(T_max) {}

    /// Check temperature and return status
    /// @return 0 = OK, 1 = warning, 2 = exceeded max
    [[nodiscard]] int check(Real Tj) const {
        if (Tj >= T_max_) return 2;
        if (Tj >= T_warning_) return 1;
        return 0;
    }

    /// Check if temperature is OK
    [[nodiscard]] bool is_ok(Real Tj) const { return Tj < T_warning_; }

    /// Check if in warning zone
    [[nodiscard]] bool is_warning(Real Tj) const {
        return Tj >= T_warning_ && Tj < T_max_;
    }

    /// Check if maximum exceeded
    [[nodiscard]] bool is_exceeded(Real Tj) const { return Tj >= T_max_; }

    /// Get warning threshold
    [[nodiscard]] Real T_warning() const { return T_warning_; }

    /// Get maximum threshold
    [[nodiscard]] Real T_max() const { return T_max_; }

    /// Set thresholds
    void set_limits(Real T_warn, Real T_max) {
        T_warning_ = T_warn;
        T_max_ = T_max;
    }

private:
    Real T_warning_;
    Real T_max_;
};

// =============================================================================
// Thermal Result
// =============================================================================

/// Result of thermal simulation
struct ThermalResult {
    std::vector<Real> times;           ///< Time points
    std::vector<Real> temperatures;    ///< Junction temperatures
    std::vector<Real> powers;          ///< Power loss at each time
    Real T_max = 0.0;                  ///< Peak temperature
    Real T_avg = 0.0;                  ///< Average temperature
    Real t_max = 0.0;                  ///< Time of peak temperature
    bool exceeded_limit = false;       ///< True if T_max was exceeded
    std::string message;

    /// Compute statistics from waveforms
    void compute_stats() {
        if (temperatures.empty()) return;

        T_max = temperatures[0];
        t_max = times[0];
        Real sum = 0.0;

        for (std::size_t i = 0; i < temperatures.size(); ++i) {
            sum += temperatures[i];
            if (temperatures[i] > T_max) {
                T_max = temperatures[i];
                t_max = times[i];
            }
        }

        T_avg = sum / static_cast<Real>(temperatures.size());
    }
};

// =============================================================================
// Factory Functions
// =============================================================================

/// Create a typical 3-stage Foster network for MOSFET
/// Parameters from typical datasheet
inline FosterNetwork create_mosfet_thermal_model(
    Real Rth_jc,    ///< Junction-to-case thermal resistance (K/W)
    Real Rth_cs,    ///< Case-to-sink thermal resistance (K/W)
    Real Rth_sa,    ///< Sink-to-ambient thermal resistance (K/W)
    const std::string& name = "") {

    // Typical time constants for MOSFET
    // Junction: fast (ms range)
    // Case: medium (10s of ms)
    // Heatsink: slow (seconds)
    std::vector<FosterStage> stages = {
        {Rth_jc, 0.005},     // ~5ms for junction
        {Rth_cs, 0.050},     // ~50ms for case
        {Rth_sa, 2.0}        // ~2s for heatsink
    };

    return FosterNetwork(stages, name);
}

/// Create Foster network from 4-parameter datasheet model
/// Many datasheets provide Zth = sum_i { Ri * (1 - exp(-t/taui)) }
inline FosterNetwork create_from_datasheet_4param(
    Real R1, Real tau1,
    Real R2, Real tau2,
    Real R3, Real tau3,
    Real R4, Real tau4,
    const std::string& name = "") {

    return FosterNetwork(
        {R1, R2, R3, R4},
        {tau1, tau2, tau3, tau4},
        name
    );
}

/// Create simple single-stage thermal model
inline FosterNetwork create_simple_thermal_model(
    Real Rth_ja,        ///< Total junction-to-ambient (K/W)
    Real tau = 1.0,     ///< Time constant (s)
    const std::string& name = "") {

    return FosterNetwork({{Rth_ja, tau}}, name);
}

} // namespace pulsim::v1
