#include "pulsim/api/grpc/metrics.hpp"
#include <httplib.h>
#include <algorithm>
#include <sstream>
#include <iomanip>

namespace pulsim::api::grpc {

// =============================================================================
// Helper Functions
// =============================================================================

namespace {

std::string format_labels(const Labels& labels) {
    if (labels.empty()) return "";

    std::ostringstream oss;
    oss << "{";
    bool first = true;
    for (const auto& [key, value] : labels) {
        if (!first) oss << ",";
        oss << key << "=\"" << value << "\"";
        first = false;
    }
    oss << "}";
    return oss.str();
}

std::string labels_to_key(const Labels& labels) {
    std::vector<std::string> pairs;
    for (const auto& [k, v] : labels) {
        pairs.push_back(k + "=" + v);
    }
    std::sort(pairs.begin(), pairs.end());
    std::ostringstream oss;
    for (const auto& p : pairs) {
        oss << p << ";";
    }
    return oss.str();
}

}  // namespace

// =============================================================================
// Counter Implementation
// =============================================================================

Counter::Counter(std::string name, std::string help, Labels labels)
    : name_(std::move(name))
    , help_(std::move(help))
    , labels_(std::move(labels))
{}

void Counter::inc(double value) {
    double current = value_.load();
    while (!value_.compare_exchange_weak(current, current + value)) {
        // Retry
    }
}

Counter& Counter::labels(const Labels& l) {
    std::string key = labels_to_key(l);

    std::lock_guard<std::mutex> lock(children_mutex_);
    auto it = children_.find(key);
    if (it != children_.end()) {
        return *it->second;
    }

    Labels combined = labels_;
    for (const auto& [k, v] : l) {
        combined[k] = v;
    }

    auto child = std::make_unique<Counter>(name_, help_, combined);
    auto& ref = *child;
    children_[key] = std::move(child);
    return ref;
}

std::string Counter::format_prometheus() const {
    std::ostringstream oss;
    oss << "# HELP " << name_ << " " << help_ << "\n";
    oss << "# TYPE " << name_ << " counter\n";

    if (children_.empty()) {
        oss << name_ << format_labels(labels_) << " " << value_.load() << "\n";
    } else {
        std::lock_guard<std::mutex> lock(children_mutex_);
        for (const auto& [key, child] : children_) {
            oss << name_ << format_labels(child->labels_) << " " << child->value_.load() << "\n";
        }
        if (value_.load() > 0) {
            oss << name_ << format_labels(labels_) << " " << value_.load() << "\n";
        }
    }

    return oss.str();
}

// =============================================================================
// Gauge Implementation
// =============================================================================

Gauge::Gauge(std::string name, std::string help, Labels labels)
    : name_(std::move(name))
    , help_(std::move(help))
    , labels_(std::move(labels))
{}

void Gauge::set(double value) {
    value_.store(value);
}

void Gauge::inc(double value) {
    double current = value_.load();
    while (!value_.compare_exchange_weak(current, current + value)) {
        // Retry
    }
}

void Gauge::dec(double value) {
    inc(-value);
}

void Gauge::set_to_current_time() {
    auto now = std::chrono::system_clock::now();
    auto epoch = now.time_since_epoch();
    auto seconds = std::chrono::duration_cast<std::chrono::seconds>(epoch);
    set(static_cast<double>(seconds.count()));
}

Gauge& Gauge::labels(const Labels& l) {
    std::string key = labels_to_key(l);

    std::lock_guard<std::mutex> lock(children_mutex_);
    auto it = children_.find(key);
    if (it != children_.end()) {
        return *it->second;
    }

    Labels combined = labels_;
    for (const auto& [k, v] : l) {
        combined[k] = v;
    }

    auto child = std::make_unique<Gauge>(name_, help_, combined);
    auto& ref = *child;
    children_[key] = std::move(child);
    return ref;
}

std::string Gauge::format_prometheus() const {
    std::ostringstream oss;
    oss << "# HELP " << name_ << " " << help_ << "\n";
    oss << "# TYPE " << name_ << " gauge\n";

    if (children_.empty()) {
        oss << name_ << format_labels(labels_) << " " << value_.load() << "\n";
    } else {
        std::lock_guard<std::mutex> lock(children_mutex_);
        for (const auto& [key, child] : children_) {
            oss << name_ << format_labels(child->labels_) << " " << child->value_.load() << "\n";
        }
    }

    return oss.str();
}

// =============================================================================
// Histogram Implementation
// =============================================================================

std::vector<double> Histogram::default_buckets() {
    return {0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0};
}

std::vector<double> Histogram::exponential_buckets(double start, double factor, size_t count) {
    std::vector<double> buckets;
    buckets.reserve(count);
    double value = start;
    for (size_t i = 0; i < count; ++i) {
        buckets.push_back(value);
        value *= factor;
    }
    return buckets;
}

std::vector<double> Histogram::linear_buckets(double start, double width, size_t count) {
    std::vector<double> buckets;
    buckets.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        buckets.push_back(start + i * width);
    }
    return buckets;
}

Histogram::Histogram(std::string name, std::string help,
                     std::vector<double> buckets, Labels labels)
    : name_(std::move(name))
    , help_(std::move(help))
    , labels_(std::move(labels))
    , buckets_(std::move(buckets))
{
    std::sort(buckets_.begin(), buckets_.end());
    bucket_count_ = buckets_.size() + 1;  // +1 for +Inf
    bucket_counts_ = std::make_unique<std::atomic<uint64_t>[]>(bucket_count_);
    for (size_t i = 0; i < bucket_count_; ++i) {
        bucket_counts_[i].store(0);
    }
}

void Histogram::observe(double value) {
    // Increment all buckets where value <= bucket bound
    for (size_t i = 0; i < buckets_.size(); ++i) {
        if (value <= buckets_[i]) {
            bucket_counts_[i]++;
        }
    }
    // +Inf bucket
    bucket_counts_[buckets_.size()]++;

    // Update sum and count
    double current_sum = sum_.load();
    while (!sum_.compare_exchange_weak(current_sum, current_sum + value)) {}
    count_++;
}

Histogram& Histogram::labels(const Labels& l) {
    std::string key = labels_to_key(l);

    std::lock_guard<std::mutex> lock(children_mutex_);
    auto it = children_.find(key);
    if (it != children_.end()) {
        return *it->second;
    }

    Labels combined = labels_;
    for (const auto& [k, v] : l) {
        combined[k] = v;
    }

    auto child = std::make_unique<Histogram>(name_, help_, buckets_, combined);
    auto& ref = *child;
    children_[key] = std::move(child);
    return ref;
}

std::string Histogram::format_prometheus() const {
    std::ostringstream oss;
    oss << std::setprecision(6);
    oss << "# HELP " << name_ << " " << help_ << "\n";
    oss << "# TYPE " << name_ << " histogram\n";

    auto format_one = [&](const Histogram& h) {
        std::string label_str = format_labels(h.labels_);
        std::string label_prefix = h.labels_.empty() ? "" : ",";

        // Bucket counts (cumulative)
        uint64_t cumulative = 0;
        for (size_t i = 0; i < h.buckets_.size(); ++i) {
            cumulative += h.bucket_counts_[i].load();
            oss << name_ << "_bucket{";
            if (!h.labels_.empty()) {
                bool first = true;
                for (const auto& [k, v] : h.labels_) {
                    if (!first) oss << ",";
                    oss << k << "=\"" << v << "\"";
                    first = false;
                }
                oss << ",";
            }
            oss << "le=\"" << h.buckets_[i] << "\"} " << cumulative << "\n";
        }
        // +Inf bucket
        cumulative += h.bucket_counts_[h.buckets_.size()].load();
        oss << name_ << "_bucket{";
        if (!h.labels_.empty()) {
            bool first = true;
            for (const auto& [k, v] : h.labels_) {
                if (!first) oss << ",";
                oss << k << "=\"" << v << "\"";
                first = false;
            }
            oss << ",";
        }
        oss << "le=\"+Inf\"} " << cumulative << "\n";

        // Sum and count
        oss << name_ << "_sum" << label_str << " " << h.sum_.load() << "\n";
        oss << name_ << "_count" << label_str << " " << h.count_.load() << "\n";
    };

    if (children_.empty()) {
        format_one(*this);
    } else {
        std::lock_guard<std::mutex> lock(children_mutex_);
        for (const auto& [key, child] : children_) {
            format_one(*child);
        }
    }

    return oss.str();
}

// Histogram Timer
Histogram::Timer::Timer(Histogram& histogram)
    : histogram_(histogram)
    , start_(std::chrono::steady_clock::now())
{}

Histogram::Timer::~Timer() {
    histogram_.observe(elapsed_seconds());
}

double Histogram::Timer::elapsed_seconds() const {
    auto now = std::chrono::steady_clock::now();
    return std::chrono::duration<double>(now - start_).count();
}

// =============================================================================
// MetricsRegistry Implementation
// =============================================================================

MetricsRegistry& MetricsRegistry::instance() {
    static MetricsRegistry registry;
    return registry;
}

Counter& MetricsRegistry::counter(const std::string& name, const std::string& help) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = metrics_.find(name);
    if (it != metrics_.end()) {
        return static_cast<Counter&>(*it->second);
    }

    auto metric = std::make_unique<Counter>(name, help);
    auto& ref = *metric;
    metrics_[name] = std::move(metric);
    return ref;
}

Gauge& MetricsRegistry::gauge(const std::string& name, const std::string& help) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = metrics_.find(name);
    if (it != metrics_.end()) {
        return static_cast<Gauge&>(*it->second);
    }

    auto metric = std::make_unique<Gauge>(name, help);
    auto& ref = *metric;
    metrics_[name] = std::move(metric);
    return ref;
}

Histogram& MetricsRegistry::histogram(const std::string& name, const std::string& help,
                                       std::vector<double> buckets) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = metrics_.find(name);
    if (it != metrics_.end()) {
        return static_cast<Histogram&>(*it->second);
    }

    auto metric = std::make_unique<Histogram>(name, help, std::move(buckets));
    auto& ref = *metric;
    metrics_[name] = std::move(metric);
    return ref;
}

std::string MetricsRegistry::format_prometheus() const {
    std::ostringstream oss;
    std::lock_guard<std::mutex> lock(mutex_);

    for (const auto& [name, metric] : metrics_) {
        oss << metric->format_prometheus() << "\n";
    }

    return oss.str();
}

void MetricsRegistry::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    metrics_.clear();
}

// =============================================================================
// PulsimMetrics Implementation
// =============================================================================

PulsimMetrics& PulsimMetrics::instance() {
    static PulsimMetrics metrics;
    return metrics;
}

PulsimMetrics::PulsimMetrics()
    : grpc_requests_total_(MetricsRegistry::instance().counter(
          "pulsim_grpc_requests_total", "Total number of gRPC requests"))
    , grpc_errors_total_(MetricsRegistry::instance().counter(
          "pulsim_grpc_errors_total", "Total number of gRPC errors"))
    , grpc_request_duration_(MetricsRegistry::instance().histogram(
          "pulsim_grpc_request_duration_seconds", "gRPC request latency in seconds"))
    , simulations_total_(MetricsRegistry::instance().counter(
          "pulsim_simulations_total", "Total number of simulations run"))
    , simulations_failed_(MetricsRegistry::instance().counter(
          "pulsim_simulations_failed_total", "Total number of failed simulations"))
    , simulation_duration_(MetricsRegistry::instance().histogram(
          "pulsim_simulation_duration_seconds", "Simulation execution time",
          Histogram::exponential_buckets(0.001, 2.0, 15)))
    , simulation_timesteps_(MetricsRegistry::instance().histogram(
          "pulsim_simulation_timesteps", "Number of timesteps per simulation",
          Histogram::exponential_buckets(100, 2.0, 15)))
    , newton_iterations_(MetricsRegistry::instance().histogram(
          "pulsim_newton_iterations", "Total Newton iterations per simulation",
          Histogram::exponential_buckets(10, 2.0, 15)))
    , jobs_pending_(MetricsRegistry::instance().gauge(
          "pulsim_jobs_pending", "Number of jobs waiting in queue"))
    , jobs_running_(MetricsRegistry::instance().gauge(
          "pulsim_jobs_running", "Number of jobs currently running"))
    , jobs_completed_(MetricsRegistry::instance().counter(
          "pulsim_jobs_completed_total", "Total number of completed jobs"))
    , jobs_failed_(MetricsRegistry::instance().counter(
          "pulsim_jobs_failed_total", "Total number of failed jobs"))
    , jobs_cancelled_(MetricsRegistry::instance().counter(
          "pulsim_jobs_cancelled_total", "Total number of cancelled jobs"))
    , jobs_timeout_(MetricsRegistry::instance().counter(
          "pulsim_jobs_timeout_total", "Total number of timed out jobs"))
    , job_queue_time_(MetricsRegistry::instance().histogram(
          "pulsim_job_queue_time_seconds", "Time spent in queue before execution",
          Histogram::exponential_buckets(0.01, 2.0, 12)))
    , workers_active_(MetricsRegistry::instance().gauge(
          "pulsim_workers_active", "Number of workers currently processing jobs"))
    , workers_idle_(MetricsRegistry::instance().gauge(
          "pulsim_workers_idle", "Number of idle workers"))
    , memory_usage_bytes_(MetricsRegistry::instance().gauge(
          "pulsim_memory_usage_bytes", "Current memory usage in bytes"))
    , active_sessions_(MetricsRegistry::instance().gauge(
          "pulsim_active_sessions", "Number of active simulation sessions"))
    , quota_exceeded_total_(MetricsRegistry::instance().counter(
          "pulsim_quota_exceeded_total", "Total number of quota exceeded events"))
{}

// =============================================================================
// MetricsServer Implementation
// =============================================================================

class MetricsServer::Impl {
public:
    httplib::Server server;
};

MetricsServer::MetricsServer(int port)
    : port_(port)
    , impl_(std::make_unique<Impl>())
{}

MetricsServer::~MetricsServer() {
    stop();
}

void MetricsServer::start() {
    if (running_) return;

    impl_->server.Get("/metrics", [](const httplib::Request&, httplib::Response& res) {
        res.set_content(MetricsRegistry::instance().format_prometheus(),
                       "text/plain; version=0.0.4; charset=utf-8");
    });

    impl_->server.Get("/health", [](const httplib::Request&, httplib::Response& res) {
        res.set_content("{\"status\":\"ok\"}", "application/json");
    });

    running_ = true;
    server_thread_ = std::make_unique<std::thread>([this]() {
        impl_->server.listen("0.0.0.0", port_);
    });
}

void MetricsServer::stop() {
    if (!running_) return;

    impl_->server.stop();
    running_ = false;

    if (server_thread_ && server_thread_->joinable()) {
        server_thread_->join();
    }
    server_thread_.reset();
}

}  // namespace pulsim::api::grpc
