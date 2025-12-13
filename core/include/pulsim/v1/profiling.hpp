/**
 * @file profiling.hpp
 * @brief Performance profiling utilities for PulsimCore
 *
 * This file implements section 10.5 of the improve-convergence-algorithms spec:
 * - Profile hot paths
 * - Identify optimization opportunities
 * - Measure memory usage
 *
 * Usage:
 *   // Enable profiling in debug builds
 *   #define PULSIM_ENABLE_PROFILING
 *
 *   // Use scoped timers
 *   {
 *       PULSIM_PROFILE_SCOPE("dc_operating_point");
 *       // ... code to profile ...
 *   }
 *
 *   // Or manual timing
 *   auto& profiler = pulsim::v1::Profiler::instance();
 *   profiler.start("my_operation");
 *   // ... code ...
 *   profiler.stop("my_operation");
 *
 *   // Print report
 *   profiler.report();
 */

#pragma once

#include "pulsim/v1/numeric_types.hpp"
#include <chrono>
#include <string>
#include <unordered_map>
#include <vector>
#include <algorithm>
#include <iomanip>
#include <sstream>
#include <mutex>
#include <atomic>

namespace pulsim::v1 {

// =============================================================================
// Timer Utilities
// =============================================================================

/// High-resolution timer for profiling
class Timer {
public:
    using Clock = std::chrono::high_resolution_clock;
    using TimePoint = Clock::time_point;
    using Duration = std::chrono::nanoseconds;

    void start() {
        start_ = Clock::now();
        running_ = true;
    }

    void stop() {
        if (running_) {
            end_ = Clock::now();
            running_ = false;
        }
    }

    [[nodiscard]] Duration elapsed() const {
        if (running_) {
            return Clock::now() - start_;
        }
        return end_ - start_;
    }

    [[nodiscard]] double elapsed_us() const {
        return std::chrono::duration<double, std::micro>(elapsed()).count();
    }

    [[nodiscard]] double elapsed_ms() const {
        return std::chrono::duration<double, std::milli>(elapsed()).count();
    }

    [[nodiscard]] double elapsed_s() const {
        return std::chrono::duration<double>(elapsed()).count();
    }

    [[nodiscard]] bool is_running() const { return running_; }

private:
    TimePoint start_;
    TimePoint end_;
    bool running_ = false;
};

// =============================================================================
// Profile Entry
// =============================================================================

/// Statistics for a profiled operation
struct ProfileStats {
    std::string name;
    std::size_t call_count = 0;
    double total_time_us = 0.0;
    double min_time_us = std::numeric_limits<double>::max();
    double max_time_us = 0.0;
    double last_time_us = 0.0;

    [[nodiscard]] double avg_time_us() const {
        return call_count > 0 ? total_time_us / call_count : 0.0;
    }

    void record(double time_us) {
        ++call_count;
        total_time_us += time_us;
        min_time_us = std::min(min_time_us, time_us);
        max_time_us = std::max(max_time_us, time_us);
        last_time_us = time_us;
    }

    void reset() {
        call_count = 0;
        total_time_us = 0.0;
        min_time_us = std::numeric_limits<double>::max();
        max_time_us = 0.0;
        last_time_us = 0.0;
    }
};

// =============================================================================
// Profiler Singleton
// =============================================================================

/// Thread-safe profiler for measuring performance
class Profiler {
public:
    static Profiler& instance() {
        static Profiler profiler;
        return profiler;
    }

    /// Start timing an operation
    void start(const std::string& name) {
#ifdef PULSIM_ENABLE_PROFILING
        std::lock_guard<std::mutex> lock(mutex_);
        active_timers_[name].start();
#else
        (void)name;
#endif
    }

    /// Stop timing an operation and record the result
    void stop(const std::string& name) {
#ifdef PULSIM_ENABLE_PROFILING
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = active_timers_.find(name);
        if (it != active_timers_.end()) {
            it->second.stop();
            double elapsed = it->second.elapsed_us();

            auto& stats = stats_[name];
            stats.name = name;
            stats.record(elapsed);

            active_timers_.erase(it);
        }
#else
        (void)name;
#endif
    }

    /// Get statistics for an operation
    [[nodiscard]] ProfileStats get_stats(const std::string& name) const {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = stats_.find(name);
        if (it != stats_.end()) {
            return it->second;
        }
        return ProfileStats{name};
    }

    /// Get all statistics
    [[nodiscard]] std::vector<ProfileStats> all_stats() const {
        std::lock_guard<std::mutex> lock(mutex_);
        std::vector<ProfileStats> result;
        result.reserve(stats_.size());
        for (const auto& [name, stats] : stats_) {
            result.push_back(stats);
        }
        // Sort by total time descending
        std::sort(result.begin(), result.end(), [](const auto& a, const auto& b) {
            return a.total_time_us > b.total_time_us;
        });
        return result;
    }

    /// Generate a text report
    [[nodiscard]] std::string report() const {
        auto all = all_stats();
        if (all.empty()) {
            return "No profiling data collected.\n";
        }

        std::ostringstream oss;
        oss << std::fixed << std::setprecision(2);
        oss << "\n";
        oss << "=============================================================================\n";
        oss << "                        Performance Profile Report\n";
        oss << "=============================================================================\n\n";

        // Calculate total time for percentage
        double total_time = 0.0;
        for (const auto& s : all) {
            total_time += s.total_time_us;
        }

        // Header
        oss << std::left << std::setw(35) << "Operation"
            << std::right << std::setw(10) << "Calls"
            << std::setw(14) << "Total (ms)"
            << std::setw(10) << "Avg (us)"
            << std::setw(10) << "Min (us)"
            << std::setw(10) << "Max (us)"
            << std::setw(8) << "% Time"
            << "\n";
        oss << std::string(97, '-') << "\n";

        // Data rows
        for (const auto& s : all) {
            double pct = total_time > 0 ? (s.total_time_us / total_time) * 100.0 : 0.0;
            oss << std::left << std::setw(35) << s.name
                << std::right << std::setw(10) << s.call_count
                << std::setw(14) << (s.total_time_us / 1000.0)
                << std::setw(10) << s.avg_time_us()
                << std::setw(10) << s.min_time_us
                << std::setw(10) << s.max_time_us
                << std::setw(7) << pct << "%"
                << "\n";
        }

        oss << std::string(97, '-') << "\n";
        oss << std::left << std::setw(35) << "TOTAL"
            << std::right << std::setw(10) << ""
            << std::setw(14) << (total_time / 1000.0)
            << "\n\n";

        return oss.str();
    }

    /// Reset all statistics
    void reset() {
        std::lock_guard<std::mutex> lock(mutex_);
        stats_.clear();
        active_timers_.clear();
    }

    /// Check if profiling is enabled
    [[nodiscard]] static constexpr bool is_enabled() {
#ifdef PULSIM_ENABLE_PROFILING
        return true;
#else
        return false;
#endif
    }

private:
    Profiler() = default;
    mutable std::mutex mutex_;
    std::unordered_map<std::string, ProfileStats> stats_;
    std::unordered_map<std::string, Timer> active_timers_;
};

// =============================================================================
// Scoped Timer
// =============================================================================

/// RAII timer for automatic scope timing
class ScopedTimer {
public:
    explicit ScopedTimer(const std::string& name)
        : name_(name) {
        Profiler::instance().start(name_);
    }

    ~ScopedTimer() {
        Profiler::instance().stop(name_);
    }

    // Non-copyable, non-movable
    ScopedTimer(const ScopedTimer&) = delete;
    ScopedTimer& operator=(const ScopedTimer&) = delete;
    ScopedTimer(ScopedTimer&&) = delete;
    ScopedTimer& operator=(ScopedTimer&&) = delete;

private:
    std::string name_;
};

// =============================================================================
// Profiling Macros
// =============================================================================

#ifdef PULSIM_ENABLE_PROFILING
    #define PULSIM_PROFILE_SCOPE(name) \
        ::pulsim::v1::ScopedTimer _pulsim_timer_##__LINE__(name)

    #define PULSIM_PROFILE_FUNCTION() \
        ::pulsim::v1::ScopedTimer _pulsim_timer_##__LINE__(__FUNCTION__)

    #define PULSIM_PROFILE_START(name) \
        ::pulsim::v1::Profiler::instance().start(name)

    #define PULSIM_PROFILE_STOP(name) \
        ::pulsim::v1::Profiler::instance().stop(name)
#else
    #define PULSIM_PROFILE_SCOPE(name) (void)0
    #define PULSIM_PROFILE_FUNCTION() (void)0
    #define PULSIM_PROFILE_START(name) (void)0
    #define PULSIM_PROFILE_STOP(name) (void)0
#endif

// =============================================================================
// Hot Path Identification
// =============================================================================

/// Identifies hot paths based on profiling data
struct HotPathAnalysis {
    struct HotPath {
        std::string name;
        double total_time_us;
        double percentage;
        std::size_t call_count;
        std::string recommendation;
    };

    std::vector<HotPath> hot_paths;
    double total_time_us = 0.0;

    /// Analyze profiling data and identify hot paths
    static HotPathAnalysis analyze(double threshold_percent = 5.0) {
        HotPathAnalysis result;
        auto stats = Profiler::instance().all_stats();

        // Calculate total time
        for (const auto& s : stats) {
            result.total_time_us += s.total_time_us;
        }

        // Identify hot paths (operations taking > threshold% of time)
        for (const auto& s : stats) {
            double pct = result.total_time_us > 0
                ? (s.total_time_us / result.total_time_us) * 100.0
                : 0.0;

            if (pct >= threshold_percent) {
                HotPath hp;
                hp.name = s.name;
                hp.total_time_us = s.total_time_us;
                hp.percentage = pct;
                hp.call_count = s.call_count;
                hp.recommendation = generate_recommendation(s);
                result.hot_paths.push_back(hp);
            }
        }

        // Sort by percentage descending
        std::sort(result.hot_paths.begin(), result.hot_paths.end(),
            [](const auto& a, const auto& b) {
                return a.percentage > b.percentage;
            });

        return result;
    }

    /// Generate optimization recommendation based on stats
    static std::string generate_recommendation(const ProfileStats& stats) {
        std::string rec;

        // High call count with short avg time -> reduce call overhead
        if (stats.call_count > 10000 && stats.avg_time_us() < 1.0) {
            rec = "Consider inlining or batching calls";
        }
        // Long max time vs avg -> investigate outliers
        else if (stats.max_time_us > stats.avg_time_us() * 10) {
            rec = "Investigate performance outliers";
        }
        // Linear solver related
        else if (stats.name.find("solve") != std::string::npos ||
                 stats.name.find("factorize") != std::string::npos) {
            rec = "Consider KLU solver or symbolic reuse";
        }
        // Newton iteration related
        else if (stats.name.find("newton") != std::string::npos ||
                 stats.name.find("Newton") != std::string::npos) {
            rec = "Tune convergence tolerances or use damping";
        }
        // Assembly related
        else if (stats.name.find("assemble") != std::string::npos ||
                 stats.name.find("stamp") != std::string::npos) {
            rec = "Consider SoA layout or SIMD vectorization";
        }
        else {
            rec = "Profile further to identify bottleneck";
        }

        return rec;
    }

    /// Generate report string
    [[nodiscard]] std::string report() const {
        if (hot_paths.empty()) {
            return "No hot paths identified.\n";
        }

        std::ostringstream oss;
        oss << std::fixed << std::setprecision(2);
        oss << "\n";
        oss << "=============================================================================\n";
        oss << "                        Hot Path Analysis Report\n";
        oss << "=============================================================================\n\n";

        for (const auto& hp : hot_paths) {
            oss << "Hot Path: " << hp.name << "\n";
            oss << "  Time: " << (hp.total_time_us / 1000.0) << " ms (" << hp.percentage << "%)\n";
            oss << "  Calls: " << hp.call_count << "\n";
            oss << "  Recommendation: " << hp.recommendation << "\n\n";
        }

        return oss.str();
    }
};

// =============================================================================
// Operation Counter
// =============================================================================

/// Thread-safe counter for operation statistics
class OperationCounter {
public:
    static OperationCounter& instance() {
        static OperationCounter counter;
        return counter;
    }

    void increment(const std::string& name, std::size_t count = 1) {
#ifdef PULSIM_ENABLE_PROFILING
        std::lock_guard<std::mutex> lock(mutex_);
        counters_[name] += count;
#else
        (void)name;
        (void)count;
#endif
    }

    [[nodiscard]] std::size_t get(const std::string& name) const {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = counters_.find(name);
        return it != counters_.end() ? it->second : 0;
    }

    [[nodiscard]] std::unordered_map<std::string, std::size_t> all() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return counters_;
    }

    void reset() {
        std::lock_guard<std::mutex> lock(mutex_);
        counters_.clear();
    }

    [[nodiscard]] std::string report() const {
        auto counters = all();
        if (counters.empty()) {
            return "No operation counts recorded.\n";
        }

        std::ostringstream oss;
        oss << "\n";
        oss << "=============================================================================\n";
        oss << "                        Operation Counts\n";
        oss << "=============================================================================\n\n";

        // Sort by count descending
        std::vector<std::pair<std::string, std::size_t>> sorted(counters.begin(), counters.end());
        std::sort(sorted.begin(), sorted.end(),
            [](const auto& a, const auto& b) {
                return a.second > b.second;
            });

        oss << std::left << std::setw(40) << "Operation"
            << std::right << std::setw(15) << "Count"
            << "\n";
        oss << std::string(55, '-') << "\n";

        for (const auto& [name, count] : sorted) {
            oss << std::left << std::setw(40) << name
                << std::right << std::setw(15) << count
                << "\n";
        }

        return oss.str();
    }

private:
    OperationCounter() = default;
    mutable std::mutex mutex_;
    std::unordered_map<std::string, std::size_t> counters_;
};

#ifdef PULSIM_ENABLE_PROFILING
    #define PULSIM_COUNT_OP(name) \
        ::pulsim::v1::OperationCounter::instance().increment(name)

    #define PULSIM_COUNT_OP_N(name, n) \
        ::pulsim::v1::OperationCounter::instance().increment(name, n)
#else
    #define PULSIM_COUNT_OP(name) (void)0
    #define PULSIM_COUNT_OP_N(name, n) (void)0
#endif

// =============================================================================
// Simulation Metrics
// =============================================================================

/// Metrics collected during simulation
struct SimulationMetrics {
    // Timing
    double total_time_ms = 0.0;
    double dc_time_ms = 0.0;
    double transient_time_ms = 0.0;

    // Iteration counts
    std::size_t dc_iterations = 0;
    std::size_t total_newton_iterations = 0;
    std::size_t total_timesteps = 0;
    std::size_t accepted_timesteps = 0;
    std::size_t rejected_timesteps = 0;

    // Linear solver stats
    std::size_t linear_solves = 0;
    std::size_t symbolic_factorizations = 0;
    std::size_t numeric_factorizations = 0;

    // Convergence stats
    std::size_t convergence_failures = 0;
    std::size_t gmin_steps_used = 0;
    std::size_t source_steps_used = 0;

    // Event handling
    std::size_t switch_events = 0;
    std::size_t timestep_cuts = 0;

    // Memory
    std::size_t peak_memory_bytes = 0;
    std::size_t matrix_nonzeros = 0;

    /// Calculate derived metrics
    [[nodiscard]] double avg_newton_per_step() const {
        return total_timesteps > 0
            ? static_cast<double>(total_newton_iterations) / total_timesteps
            : 0.0;
    }

    [[nodiscard]] double timestep_acceptance_rate() const {
        std::size_t total = accepted_timesteps + rejected_timesteps;
        return total > 0
            ? static_cast<double>(accepted_timesteps) / total * 100.0
            : 0.0;
    }

    [[nodiscard]] double steps_per_second() const {
        return transient_time_ms > 0
            ? static_cast<double>(total_timesteps) / (transient_time_ms / 1000.0)
            : 0.0;
    }

    /// Generate report
    [[nodiscard]] std::string report() const {
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(2);
        oss << "\n";
        oss << "=============================================================================\n";
        oss << "                        Simulation Metrics Report\n";
        oss << "=============================================================================\n\n";

        oss << "Timing:\n";
        oss << "  Total time:        " << total_time_ms << " ms\n";
        oss << "  DC analysis:       " << dc_time_ms << " ms\n";
        oss << "  Transient:         " << transient_time_ms << " ms\n";
        oss << "  Steps/second:      " << steps_per_second() << "\n\n";

        oss << "Iterations:\n";
        oss << "  DC iterations:     " << dc_iterations << "\n";
        oss << "  Newton total:      " << total_newton_iterations << "\n";
        oss << "  Avg Newton/step:   " << avg_newton_per_step() << "\n\n";

        oss << "Timesteps:\n";
        oss << "  Total:             " << total_timesteps << "\n";
        oss << "  Accepted:          " << accepted_timesteps << "\n";
        oss << "  Rejected:          " << rejected_timesteps << "\n";
        oss << "  Acceptance rate:   " << timestep_acceptance_rate() << "%\n\n";

        oss << "Linear Solver:\n";
        oss << "  Solves:            " << linear_solves << "\n";
        oss << "  Symbolic factor:   " << symbolic_factorizations << "\n";
        oss << "  Numeric factor:    " << numeric_factorizations << "\n\n";

        oss << "Convergence:\n";
        oss << "  Failures:          " << convergence_failures << "\n";
        oss << "  GMIN steps:        " << gmin_steps_used << "\n";
        oss << "  Source steps:      " << source_steps_used << "\n\n";

        oss << "Events:\n";
        oss << "  Switch events:     " << switch_events << "\n";
        oss << "  Timestep cuts:     " << timestep_cuts << "\n\n";

        oss << "Memory:\n";
        oss << "  Peak memory:       " << (peak_memory_bytes / 1024.0) << " KB\n";
        oss << "  Matrix nnz:        " << matrix_nonzeros << "\n\n";

        return oss.str();
    }
};

/// Thread-local metrics collector
class MetricsCollector {
public:
    static MetricsCollector& instance() {
        static thread_local MetricsCollector collector;
        return collector;
    }

    SimulationMetrics& metrics() { return metrics_; }
    const SimulationMetrics& metrics() const { return metrics_; }

    void reset() {
        metrics_ = SimulationMetrics{};
    }

private:
    MetricsCollector() = default;
    SimulationMetrics metrics_;
};

// =============================================================================
// Profiling Report Generator
// =============================================================================

/// Generate a complete profiling report
[[nodiscard]] inline std::string generate_full_profiling_report() {
    std::ostringstream oss;

    oss << Profiler::instance().report();
    oss << HotPathAnalysis::analyze().report();
    oss << OperationCounter::instance().report();
    oss << MetricsCollector::instance().metrics().report();

    return oss.str();
}

/// Reset all profiling data
inline void reset_profiling() {
    Profiler::instance().reset();
    OperationCounter::instance().reset();
    MetricsCollector::instance().reset();
}

}  // namespace pulsim::v1
