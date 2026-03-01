#pragma once

// =============================================================================
// PulsimCore v2 - Validation & Benchmarking Framework
// =============================================================================
// This header provides:
// - 6.1: Analytical solutions for RC/RL/RLC circuits
// - 6.2: SPICE comparison framework
// - 6.3: Performance benchmark utilities
// - 6.4: Regression testing infrastructure
// =============================================================================

#include "pulsim/v1/numeric_types.hpp"
#include <vector>
#include <string>
#include <functional>
#include <chrono>
#include <cmath>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <numeric>

namespace pulsim::v1 {

// =============================================================================
// 6.1: Analytical Solutions for Validation
// =============================================================================

/// Analytical solution for RC circuit step response
/// V_out(t) = V_final * (1 - exp(-t/tau)) + V_initial * exp(-t/tau)
struct RCAnalytical {
    Real R;          // Resistance (Ohms)
    Real C;          // Capacitance (Farads)
    Real V_initial;  // Initial capacitor voltage (V)
    Real V_final;    // Final voltage (V) - step input

    [[nodiscard]] Real tau() const { return R * C; }

    [[nodiscard]] Real voltage(Real t) const {
        Real exp_term = std::exp(-t / tau());
        return V_final * (1.0 - exp_term) + V_initial * exp_term;
    }

    [[nodiscard]] Real current(Real t) const {
        return (V_final - voltage(t)) / R;
    }

    /// Generate analytical waveform
    [[nodiscard]] std::vector<std::pair<Real, Real>> waveform(
        Real t_start, Real t_end, Real dt) const {
        std::vector<std::pair<Real, Real>> result;
        for (Real t = t_start; t <= t_end; t += dt) {
            result.emplace_back(t, voltage(t));
        }
        return result;
    }
};

/// Analytical solution for RL circuit step response
/// I(t) = I_final * (1 - exp(-t/tau)) + I_initial * exp(-t/tau)
struct RLAnalytical {
    Real R;          // Resistance (Ohms)
    Real L;          // Inductance (Henries)
    Real V_source;   // Source voltage (V)
    Real I_initial;  // Initial inductor current (A)

    [[nodiscard]] Real tau() const { return L / R; }
    [[nodiscard]] Real I_final() const { return V_source / R; }

    [[nodiscard]] Real current(Real t) const {
        Real exp_term = std::exp(-t / tau());
        return I_final() * (1.0 - exp_term) + I_initial * exp_term;
    }

    [[nodiscard]] Real voltage_R(Real t) const {
        return current(t) * R;
    }

    [[nodiscard]] Real voltage_L(Real t) const {
        return V_source - voltage_R(t);
    }

    [[nodiscard]] std::vector<std::pair<Real, Real>> waveform(
        Real t_start, Real t_end, Real dt) const {
        std::vector<std::pair<Real, Real>> result;
        for (Real t = t_start; t <= t_end; t += dt) {
            result.emplace_back(t, current(t));
        }
        return result;
    }
};

/// RLC circuit damping type
enum class RLCDamping {
    Underdamped,   // zeta < 1
    Critical,      // zeta = 1
    Overdamped     // zeta > 1
};

/// Analytical solution for series RLC circuit step response
struct RLCAnalytical {
    Real R;          // Resistance (Ohms)
    Real L;          // Inductance (Henries)
    Real C;          // Capacitance (Farads)
    Real V_source;   // Source voltage (V)
    Real V_initial;  // Initial capacitor voltage (V)
    Real I_initial;  // Initial inductor current (A)

    [[nodiscard]] Real omega_0() const { return 1.0 / std::sqrt(L * C); }
    [[nodiscard]] Real zeta() const { return R / (2.0 * std::sqrt(L / C)); }
    [[nodiscard]] Real alpha() const { return R / (2.0 * L); }

    [[nodiscard]] RLCDamping damping_type() const {
        Real z = zeta();
        if (z < 0.999) return RLCDamping::Underdamped;
        if (z > 1.001) return RLCDamping::Overdamped;
        return RLCDamping::Critical;
    }

    [[nodiscard]] Real voltage(Real t) const {
        Real z = zeta();
        Real w0 = omega_0();
        Real a = alpha();
        Real V_ss = V_source;  // Steady-state voltage

        if (z < 1.0) {
            // Underdamped
            Real wd = w0 * std::sqrt(1.0 - z * z);
            Real A = V_initial - V_ss;
            Real B = (a * A + I_initial / C) / wd;
            return V_ss + std::exp(-a * t) * (A * std::cos(wd * t) + B * std::sin(wd * t));
        } else if (z > 1.0) {
            // Overdamped
            Real s1 = -a + std::sqrt(a * a - w0 * w0);
            Real s2 = -a - std::sqrt(a * a - w0 * w0);
            Real A = ((V_initial - V_ss) * s2 - I_initial / C) / (s2 - s1);
            Real B = (V_initial - V_ss) - A;
            return V_ss + A * std::exp(s1 * t) + B * std::exp(s2 * t);
        } else {
            // Critically damped
            Real A = V_initial - V_ss;
            Real B = a * A + I_initial / C;
            return V_ss + (A + B * t) * std::exp(-a * t);
        }
    }

    [[nodiscard]] Real current(Real t) const {
        // Numerical derivative for simplicity
        Real dt = 1e-9;
        Real dV = voltage(t + dt) - voltage(t);
        return C * dV / dt;
    }

    [[nodiscard]] std::vector<std::pair<Real, Real>> waveform(
        Real t_start, Real t_end, Real dt) const {
        std::vector<std::pair<Real, Real>> result;
        for (Real t = t_start; t <= t_end; t += dt) {
            result.emplace_back(t, voltage(t));
        }
        return result;
    }
};

/// Ideal diode half-wave rectifier analytical solution
struct DiodeRectifierAnalytical {
    Real V_peak;     // Peak AC voltage (V)
    Real frequency;  // AC frequency (Hz)
    Real V_forward;  // Diode forward voltage drop (V)

    [[nodiscard]] Real omega() const { return 2.0 * M_PI * frequency; }

    [[nodiscard]] Real voltage_out(Real t) const {
        Real v_in = V_peak * std::sin(omega() * t);
        if (v_in > V_forward) {
            return v_in - V_forward;
        }
        return 0.0;
    }

    [[nodiscard]] std::vector<std::pair<Real, Real>> waveform(
        Real t_start, Real t_end, Real dt) const {
        std::vector<std::pair<Real, Real>> result;
        for (Real t = t_start; t <= t_end; t += dt) {
            result.emplace_back(t, voltage_out(t));
        }
        return result;
    }
};

// =============================================================================
// Validation Metrics
// =============================================================================

/// Validation result with error metrics
struct ValidationResult {
    std::string test_name;
    bool passed = false;
    Real max_error = 0.0;           // Maximum absolute error
    Real rms_error = 0.0;           // RMS error
    Real max_relative_error = 0.0;  // Maximum relative error (%)
    Real mean_error = 0.0;          // Mean absolute error
    std::size_t num_points = 0;
    Real error_threshold = 0.001;   // 0.1% default

    [[nodiscard]] std::string to_string() const {
        std::ostringstream ss;
        ss << std::fixed << std::setprecision(6);
        ss << "Test: " << test_name << "\n";
        ss << "  Status: " << (passed ? "PASSED" : "FAILED") << "\n";
        ss << "  Points: " << num_points << "\n";
        ss << "  Max Error: " << max_error << "\n";
        ss << "  RMS Error: " << rms_error << "\n";
        ss << "  Max Relative: " << (max_relative_error * 100) << "%\n";
        ss << "  Threshold: " << (error_threshold * 100) << "%\n";
        return ss.str();
    }
};

/// Compare simulated vs analytical waveforms
[[nodiscard]] inline ValidationResult compare_waveforms(
    const std::string& name,
    const std::vector<std::pair<Real, Real>>& simulated,
    const std::vector<std::pair<Real, Real>>& analytical,
    Real threshold = 0.001) {

    ValidationResult result;
    result.test_name = name;
    result.error_threshold = threshold;

    if (simulated.empty() || analytical.empty()) {
        result.passed = false;
        return result;
    }

    std::vector<Real> errors;
    std::vector<Real> rel_errors;

    // Find matching time points and compute errors
    for (const auto& [t_sim, v_sim] : simulated) {
        // Find closest analytical point
        Real min_dt = std::numeric_limits<Real>::max();
        Real v_ana = 0.0;

        for (const auto& [t_ana, v_a] : analytical) {
            Real dt = std::abs(t_sim - t_ana);
            if (dt < min_dt) {
                min_dt = dt;
                v_ana = v_a;
            }
        }

        Real abs_err = std::abs(v_sim - v_ana);
        errors.push_back(abs_err);

        Real rel_err = (std::abs(v_ana) > 1e-12) ?
                       abs_err / std::abs(v_ana) : abs_err;
        rel_errors.push_back(rel_err);
    }

    result.num_points = errors.size();
    result.max_error = *std::max_element(errors.begin(), errors.end());
    result.max_relative_error = *std::max_element(rel_errors.begin(), rel_errors.end());

    Real sum_sq = 0.0;
    Real sum = 0.0;
    for (Real e : errors) {
        sum_sq += e * e;
        sum += e;
    }
    result.rms_error = std::sqrt(sum_sq / errors.size());
    result.mean_error = sum / errors.size();

    result.passed = result.max_relative_error <= threshold;

    return result;
}

// =============================================================================
// 6.1.9: Metrics Export
// =============================================================================

/// Export validation results to CSV
[[nodiscard]] inline std::string export_validation_csv(
    const std::vector<ValidationResult>& results) {

    std::ostringstream ss;
    ss << "test_name,passed,num_points,max_error,rms_error,max_relative_pct,threshold_pct\n";

    for (const auto& r : results) {
        ss << r.test_name << ","
           << (r.passed ? "true" : "false") << ","
           << r.num_points << ","
           << std::scientific << std::setprecision(6)
           << r.max_error << ","
           << r.rms_error << ","
           << (r.max_relative_error * 100) << ","
           << (r.error_threshold * 100) << "\n";
    }

    return ss.str();
}

/// Export validation results to JSON
[[nodiscard]] inline std::string export_validation_json(
    const std::vector<ValidationResult>& results) {

    std::ostringstream ss;
    ss << "{\n  \"validation_results\": [\n";

    for (std::size_t i = 0; i < results.size(); ++i) {
        const auto& r = results[i];
        ss << "    {\n"
           << "      \"test_name\": \"" << r.test_name << "\",\n"
           << "      \"passed\": " << (r.passed ? "true" : "false") << ",\n"
           << "      \"num_points\": " << r.num_points << ",\n"
           << std::scientific << std::setprecision(6)
           << "      \"max_error\": " << r.max_error << ",\n"
           << "      \"rms_error\": " << r.rms_error << ",\n"
           << "      \"max_relative_pct\": " << (r.max_relative_error * 100) << ",\n"
           << "      \"threshold_pct\": " << (r.error_threshold * 100) << "\n"
           << "    }";
        if (i < results.size() - 1) ss << ",";
        ss << "\n";
    }

    ss << "  ]\n}\n";
    return ss.str();
}

// =============================================================================
// 6.2: SPICE Reference Comparison Framework
// =============================================================================

/// SPICE comparison configuration
struct SPICEComparisonConfig {
    std::string ngspice_path = "ngspice";
    std::string temp_dir = "/tmp/pulsim_spice";
    Real time_tolerance = 1e-9;     // Time alignment tolerance
    Real value_tolerance = 0.001;   // 0.1% value tolerance
    bool align_timesteps = true;
    bool save_artifacts = false;
    std::string artifact_dir = "./spice_artifacts";
};

/// SPICE comparison result
struct SPICEComparisonResult {
    std::string circuit_name;
    bool passed = false;
    Real rms_error = 0.0;
    Real max_error = 0.0;
    std::size_t num_points = 0;
    std::string deviation_notes;
    std::vector<std::pair<Real, Real>> pulsim_data;
    std::vector<std::pair<Real, Real>> ngspice_data;
};

/// SPICE netlist generator for comparison
class SPICENetlistGenerator {
public:
    /// Generate RC circuit netlist
    [[nodiscard]] static std::string rc_circuit(
        Real R, Real C, Real V_source, Real t_end, Real dt) {

        std::ostringstream ss;
        ss << "* RC Circuit for validation\n"
           << "V1 in 0 DC " << V_source << "\n"
           << "R1 in out " << R << "\n"
           << "C1 out 0 " << C << " IC=0\n"
           << ".tran " << dt << " " << t_end << " UIC\n"
           << ".print tran V(out)\n"
           << ".end\n";
        return ss.str();
    }

    /// Generate RL circuit netlist
    [[nodiscard]] static std::string rl_circuit(
        Real R, Real L, Real V_source, Real t_end, Real dt) {

        std::ostringstream ss;
        ss << "* RL Circuit for validation\n"
           << "V1 in 0 DC " << V_source << "\n"
           << "R1 in out " << R << "\n"
           << "L1 out 0 " << L << " IC=0\n"
           << ".tran " << dt << " " << t_end << " UIC\n"
           << ".print tran I(L1)\n"
           << ".end\n";
        return ss.str();
    }

    /// Generate RLC circuit netlist
    [[nodiscard]] static std::string rlc_circuit(
        Real R, Real L, Real C, Real V_source, Real t_end, Real dt) {

        std::ostringstream ss;
        ss << "* Series RLC Circuit for validation\n"
           << "V1 in 0 DC " << V_source << "\n"
           << "R1 in n1 " << R << "\n"
           << "L1 n1 n2 " << L << " IC=0\n"
           << "C1 n2 0 " << C << " IC=0\n"
           << ".tran " << dt << " " << t_end << " UIC\n"
           << ".print tran V(n2)\n"
           << ".end\n";
        return ss.str();
    }

    /// Generate half-wave rectifier netlist
    [[nodiscard]] static std::string rectifier_circuit(
        Real V_peak, Real freq, Real R_load, Real t_end, Real dt) {

        std::ostringstream ss;
        ss << "* Half-wave rectifier for validation\n"
           << "V1 in 0 SIN(0 " << V_peak << " " << freq << ")\n"
           << "D1 in out DMOD\n"
           << "R1 out 0 " << R_load << "\n"
           << ".model DMOD D(IS=1e-14 N=1)\n"
           << ".tran " << dt << " " << t_end << "\n"
           << ".print tran V(out)\n"
           << ".end\n";
        return ss.str();
    }
};

// =============================================================================
// 6.3: Performance Benchmark Framework
// =============================================================================

/// Benchmark timing result
struct BenchmarkTiming {
    std::string name;
    std::chrono::nanoseconds total_time{0};
    std::chrono::nanoseconds min_time{std::chrono::nanoseconds::max()};
    std::chrono::nanoseconds max_time{0};
    std::size_t iterations = 0;

    [[nodiscard]] double average_ms() const {
        if (iterations == 0) return 0.0;
        return std::chrono::duration<double, std::milli>(total_time).count() / iterations;
    }

    [[nodiscard]] double min_ms() const {
        return std::chrono::duration<double, std::milli>(min_time).count();
    }

    [[nodiscard]] double max_ms() const {
        return std::chrono::duration<double, std::milli>(max_time).count();
    }

    [[nodiscard]] double total_ms() const {
        return std::chrono::duration<double, std::milli>(total_time).count();
    }
};

/// Memory usage statistics for benchmarks (uses MemoryTracker from high_performance.hpp)
/// Note: Use MemoryTracker::instance().stats() to get actual memory stats

/// Complete benchmark result
struct BenchmarkResult {
    std::string circuit_name;
    std::size_t num_nodes = 0;
    std::size_t num_devices = 0;
    std::size_t num_timesteps = 0;
    BenchmarkTiming timing;
    MemoryStats memory;  // Uses MemoryStats from high_performance.hpp
    Real simulation_time = 0.0;     // Simulated time (s)

    [[nodiscard]] double timesteps_per_second() const {
        double ms = timing.average_ms();
        if (ms <= 0) return 0.0;
        return (num_timesteps * 1000.0) / ms;
    }

    [[nodiscard]] std::string to_string() const {
        std::ostringstream ss;
        ss << "Benchmark: " << circuit_name << "\n"
           << "  Nodes: " << num_nodes << ", Devices: " << num_devices << "\n"
           << "  Timesteps: " << num_timesteps << "\n"
           << "  Average: " << std::fixed << std::setprecision(3)
           << timing.average_ms() << " ms\n"
           << "  Min: " << timing.min_ms() << " ms, Max: " << timing.max_ms() << " ms\n"
           << "  Throughput: " << std::setprecision(0)
           << timesteps_per_second() << " steps/s\n"
           << "  Peak Memory: " << (memory.peak_allocated / 1024) << " KB\n";
        return ss.str();
    }
};

/// Benchmark runner
class BenchmarkRunner {
public:
    using BenchmarkFunc = std::function<void()>;

    explicit BenchmarkRunner(std::size_t warmup = 3, std::size_t iterations = 10)
        : warmup_(warmup), iterations_(iterations) {}

    /// Run a benchmark
    [[nodiscard]] BenchmarkTiming run(const std::string& name, BenchmarkFunc func) {
        BenchmarkTiming result;
        result.name = name;

        // Warmup runs
        for (std::size_t i = 0; i < warmup_; ++i) {
            func();
        }

        // Timed runs
        for (std::size_t i = 0; i < iterations_; ++i) {
            auto start = std::chrono::high_resolution_clock::now();
            func();
            auto end = std::chrono::high_resolution_clock::now();

            auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
            result.total_time += duration;
            result.min_time = std::min(result.min_time, duration);
            result.max_time = std::max(result.max_time, duration);
            result.iterations++;
        }

        return result;
    }

    void set_warmup(std::size_t w) { warmup_ = w; }
    void set_iterations(std::size_t i) { iterations_ = i; }

private:
    std::size_t warmup_;
    std::size_t iterations_;
};

/// Export benchmark results to CSV (6.3.7)
[[nodiscard]] inline std::string export_benchmark_csv(
    const std::vector<BenchmarkResult>& results) {

    std::ostringstream ss;
    ss << "circuit_name,num_nodes,num_devices,num_timesteps,avg_ms,min_ms,max_ms,peak_memory_kb,throughput\n";

    for (const auto& r : results) {
        ss << r.circuit_name << ","
           << r.num_nodes << ","
           << r.num_devices << ","
           << r.num_timesteps << ","
           << std::fixed << std::setprecision(3)
           << r.timing.average_ms() << ","
           << r.timing.min_ms() << ","
           << r.timing.max_ms() << ","
           << (r.memory.peak_allocated / 1024) << ","
           << std::setprecision(0) << r.timesteps_per_second() << "\n";
    }

    return ss.str();
}

/// Export benchmark results to JSON
[[nodiscard]] inline std::string export_benchmark_json(
    const std::vector<BenchmarkResult>& results) {

    std::ostringstream ss;
    ss << "{\n  \"benchmarks\": [\n";

    for (std::size_t i = 0; i < results.size(); ++i) {
        const auto& r = results[i];
        ss << "    {\n"
           << "      \"circuit_name\": \"" << r.circuit_name << "\",\n"
           << "      \"num_nodes\": " << r.num_nodes << ",\n"
           << "      \"num_devices\": " << r.num_devices << ",\n"
           << "      \"num_timesteps\": " << r.num_timesteps << ",\n"
           << std::fixed << std::setprecision(3)
           << "      \"avg_ms\": " << r.timing.average_ms() << ",\n"
           << "      \"min_ms\": " << r.timing.min_ms() << ",\n"
           << "      \"max_ms\": " << r.timing.max_ms() << ",\n"
           << "      \"peak_memory_kb\": " << (r.memory.peak_allocated / 1024) << ",\n"
           << std::setprecision(0)
           << "      \"throughput\": " << r.timesteps_per_second() << "\n"
           << "    }";
        if (i < results.size() - 1) ss << ",";
        ss << "\n";
    }

    ss << "  ]\n}\n";
    return ss.str();
}

// =============================================================================
// 6.3.8: Deterministic Benchmark Harness
// =============================================================================

/// Configuration for deterministic benchmarks
struct DeterministicBenchmarkConfig {
    std::uint64_t random_seed = 42;
    bool fixed_device_order = true;
    bool fixed_node_order = true;
    bool disable_parallel = false;
    std::size_t warmup_iterations = 3;
    std::size_t timed_iterations = 10;
};

/// Deterministic benchmark harness
class DeterministicBenchmarkHarness {
public:
    explicit DeterministicBenchmarkHarness(const DeterministicBenchmarkConfig& config = {})
        : config_(config), runner_(config.warmup_iterations, config.timed_iterations) {}

    /// Run benchmark with deterministic settings
    template<typename SimFunc>
    [[nodiscard]] BenchmarkResult run(
        const std::string& name,
        std::size_t num_nodes,
        std::size_t num_devices,
        std::size_t num_timesteps,
        SimFunc sim_func) {

        BenchmarkResult result;
        result.circuit_name = name;
        result.num_nodes = num_nodes;
        result.num_devices = num_devices;
        result.num_timesteps = num_timesteps;

        // Run with deterministic settings
        result.timing = runner_.run(name, [&]() {
            sim_func();
        });

        return result;
    }

    [[nodiscard]] const DeterministicBenchmarkConfig& config() const { return config_; }

private:
    DeterministicBenchmarkConfig config_;
    BenchmarkRunner runner_;
};

// =============================================================================
// 6.4: Regression Testing Infrastructure
// =============================================================================

/// Regression test result
struct RegressionTestResult {
    std::string test_name;
    bool passed = false;
    Real current_value = 0.0;
    Real baseline_value = 0.0;
    Real threshold = 0.1;  // 10% default
    std::string metric_type;  // "accuracy", "performance", "memory"

    [[nodiscard]] Real deviation() const {
        if (baseline_value == 0.0) return 0.0;
        return (current_value - baseline_value) / baseline_value;
    }

    [[nodiscard]] bool is_regression() const {
        // For accuracy: lower is better, so current > baseline is regression
        // For performance: lower is better (time), so current > baseline is regression
        // For memory: lower is better, so current > baseline is regression
        return deviation() > threshold;
    }
};

/// Regression baseline storage
struct RegressionBaseline {
    std::string name;
    Real accuracy_rms = 0.0;
    Real performance_ms = 0.0;
    std::size_t memory_bytes = 0;
    std::string commit_hash;
    std::string timestamp;
};

/// Regression test runner
class RegressionTester {
public:
    /// Load baselines from CSV
    void load_baselines(const std::string& csv_data) {
        std::istringstream ss(csv_data);
        std::string line;

        // Skip header
        std::getline(ss, line);

        while (std::getline(ss, line)) {
            std::istringstream ls(line);
            RegressionBaseline b;
            std::string token;

            std::getline(ls, b.name, ',');
            std::getline(ls, token, ','); b.accuracy_rms = std::stod(token);
            std::getline(ls, token, ','); b.performance_ms = std::stod(token);
            std::getline(ls, token, ','); b.memory_bytes = std::stoull(token);
            std::getline(ls, b.commit_hash, ',');
            std::getline(ls, b.timestamp, ',');

            baselines_[b.name] = b;
        }
    }

    /// Check accuracy regression
    [[nodiscard]] RegressionTestResult check_accuracy(
        const std::string& name, Real current_rms, Real threshold = 0.1) {

        RegressionTestResult result;
        result.test_name = name;
        result.metric_type = "accuracy";
        result.current_value = current_rms;
        result.threshold = threshold;

        auto it = baselines_.find(name);
        if (it != baselines_.end()) {
            result.baseline_value = it->second.accuracy_rms;
            result.passed = !result.is_regression();
        } else {
            // No baseline, pass by default (first run)
            result.passed = true;
        }

        return result;
    }

    /// Check performance regression
    [[nodiscard]] RegressionTestResult check_performance(
        const std::string& name, Real current_ms, Real threshold = 0.1) {

        RegressionTestResult result;
        result.test_name = name;
        result.metric_type = "performance";
        result.current_value = current_ms;
        result.threshold = threshold;

        auto it = baselines_.find(name);
        if (it != baselines_.end()) {
            result.baseline_value = it->second.performance_ms;
            result.passed = !result.is_regression();
        } else {
            result.passed = true;
        }

        return result;
    }

    /// Check memory regression
    [[nodiscard]] RegressionTestResult check_memory(
        const std::string& name, std::size_t current_bytes, Real threshold = 0.1) {

        RegressionTestResult result;
        result.test_name = name;
        result.metric_type = "memory";
        result.current_value = static_cast<Real>(current_bytes);
        result.threshold = threshold;

        auto it = baselines_.find(name);
        if (it != baselines_.end()) {
            result.baseline_value = static_cast<Real>(it->second.memory_bytes);
            result.passed = !result.is_regression();
        } else {
            result.passed = true;
        }

        return result;
    }

    /// Export current results as new baseline
    [[nodiscard]] std::string export_baseline_csv(
        const std::vector<BenchmarkResult>& benchmarks,
        const std::vector<ValidationResult>& validations,
        const std::string& commit_hash) const {

        std::ostringstream ss;
        ss << "name,accuracy_rms,performance_ms,memory_bytes,commit_hash,timestamp\n";

        auto now = std::chrono::system_clock::now();
        auto time = std::chrono::system_clock::to_time_t(now);
        std::string timestamp = std::ctime(&time);
        timestamp.pop_back();  // Remove newline

        for (const auto& b : benchmarks) {
            // Find matching validation
            Real rms = 0.0;
            for (const auto& v : validations) {
                if (v.test_name == b.circuit_name) {
                    rms = v.rms_error;
                    break;
                }
            }

            ss << b.circuit_name << ","
               << std::scientific << std::setprecision(6)
               << rms << ","
               << std::fixed << std::setprecision(3)
               << b.timing.average_ms() << ","
               << b.memory.peak_allocated << ","
               << commit_hash << ","
               << timestamp << "\n";
        }

        return ss.str();
    }

private:
    std::unordered_map<std::string, RegressionBaseline> baselines_;
};

// =============================================================================
// 6.4.5: Waveform Tolerance Envelope
// =============================================================================

/// Tolerance envelope for waveform comparison
struct ToleranceEnvelope {
    Real absolute_tolerance = 1e-6;
    Real relative_tolerance = 0.001;  // 0.1%
    Real time_tolerance = 1e-9;

    /// Check if a point is within the envelope
    [[nodiscard]] bool within_envelope(
        Real expected, Real actual, Real time_error = 0.0) const {

        if (std::abs(time_error) > time_tolerance) {
            return false;
        }

        Real tol = absolute_tolerance + relative_tolerance * std::abs(expected);
        return std::abs(actual - expected) <= tol;
    }
};

/// Waveform regression checker
class WaveformRegressionChecker {
public:
    explicit WaveformRegressionChecker(const ToleranceEnvelope& env = {})
        : envelope_(env) {}

    /// Check waveform against baseline
    [[nodiscard]] bool check(
        const std::vector<std::pair<Real, Real>>& baseline,
        const std::vector<std::pair<Real, Real>>& current,
        std::vector<std::size_t>* violations = nullptr) const {

        bool all_pass = true;

        for (std::size_t i = 0; i < current.size() && i < baseline.size(); ++i) {
            Real t_base = baseline[i].first;
            Real v_base = baseline[i].second;
            Real t_curr = current[i].first;
            Real v_curr = current[i].second;

            Real t_err = std::abs(t_curr - t_base);

            if (!envelope_.within_envelope(v_base, v_curr, t_err)) {
                all_pass = false;
                if (violations) {
                    violations->push_back(i);
                }
            }
        }

        return all_pass;
    }

    [[nodiscard]] const ToleranceEnvelope& envelope() const { return envelope_; }
    void set_envelope(const ToleranceEnvelope& env) { envelope_ = env; }

private:
    ToleranceEnvelope envelope_;
};

} // namespace pulsim::v1
