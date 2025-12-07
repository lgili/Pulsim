#pragma once

#include <atomic>
#include <chrono>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace pulsim::api::grpc {

// =============================================================================
// Prometheus-Compatible Metrics
// =============================================================================

// Metric types
enum class MetricType {
    Counter,
    Gauge,
    Histogram,
    Summary
};

// Label set for metrics
using Labels = std::unordered_map<std::string, std::string>;

// Base metric class
class Metric {
public:
    virtual ~Metric() = default;
    virtual MetricType type() const = 0;
    virtual std::string name() const = 0;
    virtual std::string help() const = 0;
    virtual std::string format_prometheus() const = 0;
};

// =============================================================================
// Counter - Monotonically increasing value
// =============================================================================

class Counter : public Metric {
public:
    Counter(std::string name, std::string help, Labels labels = {});

    MetricType type() const override { return MetricType::Counter; }
    std::string name() const override { return name_; }
    std::string help() const override { return help_; }
    std::string format_prometheus() const override;

    void inc(double value = 1.0);
    double value() const { return value_.load(); }

    // With labels
    Counter& labels(const Labels& l);

private:
    std::string name_;
    std::string help_;
    Labels labels_;
    std::atomic<double> value_{0.0};

    std::unordered_map<std::string, std::unique_ptr<Counter>> children_;
    mutable std::mutex children_mutex_;
};

// =============================================================================
// Gauge - Value that can go up and down
// =============================================================================

class Gauge : public Metric {
public:
    Gauge(std::string name, std::string help, Labels labels = {});

    MetricType type() const override { return MetricType::Gauge; }
    std::string name() const override { return name_; }
    std::string help() const override { return help_; }
    std::string format_prometheus() const override;

    void set(double value);
    void inc(double value = 1.0);
    void dec(double value = 1.0);
    double value() const { return value_.load(); }

    // With labels
    Gauge& labels(const Labels& l);

    // Set to current time
    void set_to_current_time();

private:
    std::string name_;
    std::string help_;
    Labels labels_;
    std::atomic<double> value_{0.0};

    std::unordered_map<std::string, std::unique_ptr<Gauge>> children_;
    mutable std::mutex children_mutex_;
};

// =============================================================================
// Histogram - Distribution of values
// =============================================================================

class Histogram : public Metric {
public:
    // Default buckets for latency (seconds)
    static std::vector<double> default_buckets();
    // Exponential buckets
    static std::vector<double> exponential_buckets(double start, double factor, size_t count);
    // Linear buckets
    static std::vector<double> linear_buckets(double start, double width, size_t count);

    Histogram(std::string name, std::string help,
              std::vector<double> buckets = default_buckets(),
              Labels labels = {});

    MetricType type() const override { return MetricType::Histogram; }
    std::string name() const override { return name_; }
    std::string help() const override { return help_; }
    std::string format_prometheus() const override;

    void observe(double value);

    double sum() const { return sum_.load(); }
    uint64_t count() const { return count_.load(); }

    // With labels
    Histogram& labels(const Labels& l);

    // Timer helper
    class Timer {
    public:
        explicit Timer(Histogram& histogram);
        ~Timer();
        double elapsed_seconds() const;

    private:
        Histogram& histogram_;
        std::chrono::steady_clock::time_point start_;
    };

    Timer start_timer() { return Timer(*this); }

private:
    std::string name_;
    std::string help_;
    Labels labels_;
    std::vector<double> buckets_;
    std::unique_ptr<std::atomic<uint64_t>[]> bucket_counts_;
    size_t bucket_count_{0};
    std::atomic<double> sum_{0.0};
    std::atomic<uint64_t> count_{0};

    std::unordered_map<std::string, std::unique_ptr<Histogram>> children_;
    mutable std::mutex children_mutex_;
};

// =============================================================================
// Metrics Registry
// =============================================================================

class MetricsRegistry {
public:
    static MetricsRegistry& instance();

    // Register metrics
    Counter& counter(const std::string& name, const std::string& help);
    Gauge& gauge(const std::string& name, const std::string& help);
    Histogram& histogram(const std::string& name, const std::string& help,
                         std::vector<double> buckets = Histogram::default_buckets());

    // Get all metrics in Prometheus format
    std::string format_prometheus() const;

    // Clear all metrics (for testing)
    void clear();

private:
    MetricsRegistry() = default;

    std::unordered_map<std::string, std::unique_ptr<Metric>> metrics_;
    mutable std::mutex mutex_;
};

// =============================================================================
// Pulsim-Specific Metrics
// =============================================================================

class PulsimMetrics {
public:
    static PulsimMetrics& instance();

    // gRPC metrics
    Counter& grpc_requests_total() { return grpc_requests_total_; }
    Counter& grpc_errors_total() { return grpc_errors_total_; }
    Histogram& grpc_request_duration() { return grpc_request_duration_; }

    // Simulation metrics
    Counter& simulations_total() { return simulations_total_; }
    Counter& simulations_failed() { return simulations_failed_; }
    Histogram& simulation_duration() { return simulation_duration_; }
    Histogram& simulation_timesteps() { return simulation_timesteps_; }
    Histogram& newton_iterations() { return newton_iterations_; }

    // Job queue metrics
    Gauge& jobs_pending() { return jobs_pending_; }
    Gauge& jobs_running() { return jobs_running_; }
    Counter& jobs_completed() { return jobs_completed_; }
    Counter& jobs_failed() { return jobs_failed_; }
    Counter& jobs_cancelled() { return jobs_cancelled_; }
    Counter& jobs_timeout() { return jobs_timeout_; }
    Histogram& job_queue_time() { return job_queue_time_; }

    // Worker metrics
    Gauge& workers_active() { return workers_active_; }
    Gauge& workers_idle() { return workers_idle_; }

    // Resource metrics
    Gauge& memory_usage_bytes() { return memory_usage_bytes_; }
    Gauge& active_sessions() { return active_sessions_; }

    // User quota metrics
    Counter& quota_exceeded_total() { return quota_exceeded_total_; }

private:
    PulsimMetrics();

    Counter& grpc_requests_total_;
    Counter& grpc_errors_total_;
    Histogram& grpc_request_duration_;

    Counter& simulations_total_;
    Counter& simulations_failed_;
    Histogram& simulation_duration_;
    Histogram& simulation_timesteps_;
    Histogram& newton_iterations_;

    Gauge& jobs_pending_;
    Gauge& jobs_running_;
    Counter& jobs_completed_;
    Counter& jobs_failed_;
    Counter& jobs_cancelled_;
    Counter& jobs_timeout_;
    Histogram& job_queue_time_;

    Gauge& workers_active_;
    Gauge& workers_idle_;

    Gauge& memory_usage_bytes_;
    Gauge& active_sessions_;

    Counter& quota_exceeded_total_;
};

// =============================================================================
// HTTP Metrics Endpoint
// =============================================================================

class MetricsServer {
public:
    explicit MetricsServer(int port = 9090);
    ~MetricsServer();

    void start();
    void stop();

    int port() const { return port_; }
    bool is_running() const { return running_; }

private:
    int port_;
    std::atomic<bool> running_{false};
    std::unique_ptr<std::thread> server_thread_;
    class Impl;
    std::unique_ptr<Impl> impl_;
};

// =============================================================================
// Convenience Macros
// =============================================================================

#define PULSIM_METRICS ::pulsim::api::grpc::PulsimMetrics::instance()

}  // namespace pulsim::api::grpc
