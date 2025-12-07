#pragma once

#include "pulsim/circuit.hpp"
#include "pulsim/simulation.hpp"
#include "pulsim/types.hpp"
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <queue>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace pulsim::api::grpc {

// =============================================================================
// Job Priority and Status
// =============================================================================

enum class JobPriority {
    Low = 0,
    Normal = 1,
    High = 2,
    Critical = 3
};

enum class JobStatus {
    Queued,
    Running,
    Completed,
    Failed,
    Cancelled,
    Timeout
};

// =============================================================================
// Job Definition
// =============================================================================

struct SimulationJob {
    std::string job_id;
    std::string user_id;
    std::string session_id;
    Circuit circuit;
    SimulationOptions options;
    JobPriority priority = JobPriority::Normal;

    std::chrono::system_clock::time_point created_at;
    std::chrono::system_clock::time_point deadline;  // For timeout
    std::string callback_url;  // Optional webhook for completion

    // Resource hints
    size_t estimated_steps = 0;
    size_t memory_hint_mb = 0;
};

struct JobResult {
    std::string job_id;
    JobStatus status = JobStatus::Queued;
    SimulationResult simulation;
    std::string error_message;
    std::string worker_id;

    std::chrono::system_clock::time_point started_at;
    std::chrono::system_clock::time_point completed_at;
    double duration_seconds = 0.0;

    // Resource usage
    size_t peak_memory_mb = 0;
    size_t newton_iterations = 0;
    size_t timesteps = 0;
};

// =============================================================================
// User Quotas and Resource Limits
// =============================================================================

struct UserQuota {
    std::string user_id;

    // Limits
    size_t max_concurrent_jobs = 4;
    size_t max_queued_jobs = 100;
    size_t max_simulation_time_seconds = 3600;  // 1 hour
    size_t max_timesteps = 10000000;
    size_t max_memory_mb = 4096;

    // Rate limits
    size_t max_jobs_per_hour = 100;
    size_t max_jobs_per_day = 1000;

    // Current usage (protected by QuotaManager mutex)
    size_t current_running = 0;
    size_t current_queued = 0;
    size_t jobs_this_hour = 0;
    size_t jobs_today = 0;

    // Reset times
    std::chrono::system_clock::time_point hour_reset;
    std::chrono::system_clock::time_point day_reset;
};

class QuotaManager {
public:
    QuotaManager();

    // Check if user can submit a job
    bool can_submit(const std::string& user_id, std::string& reason);

    // Reserve quota for a job
    bool reserve(const std::string& user_id);

    // Release quota when job completes/fails
    void release(const std::string& user_id, bool was_running);

    // Update user quota settings
    void set_quota(const std::string& user_id, const UserQuota& quota);

    // Get user quota (creates default if not exists)
    UserQuota& get_quota(const std::string& user_id);

    // Get default quota for new users
    const UserQuota& default_quota() const { return default_quota_; }
    void set_default_quota(const UserQuota& quota) { default_quota_ = quota; }

    // Reset rate limits (called periodically)
    void reset_hourly_limits();
    void reset_daily_limits();

private:
    std::unordered_map<std::string, std::unique_ptr<UserQuota>> quotas_;
    UserQuota default_quota_;
    mutable std::mutex mutex_;
};

// =============================================================================
// Job Queue Interface (Abstract for different backends)
// =============================================================================

class IJobQueue {
public:
    virtual ~IJobQueue() = default;

    // Submit a job
    virtual std::string submit(SimulationJob job) = 0;

    // Get next job for processing (blocks if empty)
    virtual std::optional<SimulationJob> dequeue(
        const std::chrono::milliseconds& timeout = std::chrono::milliseconds(1000)) = 0;

    // Get job status
    virtual std::optional<JobStatus> get_status(const std::string& job_id) = 0;

    // Get job result
    virtual std::optional<JobResult> get_result(const std::string& job_id) = 0;

    // Update job status/result
    virtual void update_status(const std::string& job_id, JobStatus status) = 0;
    virtual void set_result(const std::string& job_id, const JobResult& result) = 0;

    // Cancel a job
    virtual bool cancel(const std::string& job_id) = 0;

    // Queue statistics
    virtual size_t pending_count() const = 0;
    virtual size_t running_count() const = 0;
    virtual size_t completed_count() const = 0;

    // List jobs
    virtual std::vector<std::string> list_pending(size_t limit = 100) = 0;
    virtual std::vector<std::string> list_running() = 0;
    virtual std::vector<std::string> list_by_user(const std::string& user_id, size_t limit = 100) = 0;
};

// =============================================================================
// In-Memory Job Queue (Default Implementation)
// =============================================================================

class MemoryJobQueue : public IJobQueue {
public:
    MemoryJobQueue();
    ~MemoryJobQueue() override;

    std::string submit(SimulationJob job) override;
    std::optional<SimulationJob> dequeue(const std::chrono::milliseconds& timeout) override;
    std::optional<JobStatus> get_status(const std::string& job_id) override;
    std::optional<JobResult> get_result(const std::string& job_id) override;
    void update_status(const std::string& job_id, JobStatus status) override;
    void set_result(const std::string& job_id, const JobResult& result) override;
    bool cancel(const std::string& job_id) override;

    size_t pending_count() const override;
    size_t running_count() const override;
    size_t completed_count() const override;

    std::vector<std::string> list_pending(size_t limit) override;
    std::vector<std::string> list_running() override;
    std::vector<std::string> list_by_user(const std::string& user_id, size_t limit) override;

    // Memory-specific: clear old completed jobs
    void clear_completed_older_than(std::chrono::seconds age);

private:
    struct JobComparator {
        bool operator()(const SimulationJob& a, const SimulationJob& b) const {
            return static_cast<int>(a.priority) < static_cast<int>(b.priority);
        }
    };

    std::priority_queue<SimulationJob, std::vector<SimulationJob>, JobComparator> pending_;
    std::unordered_map<std::string, SimulationJob> jobs_;
    std::unordered_map<std::string, JobStatus> status_;
    std::unordered_map<std::string, JobResult> results_;
    std::unordered_map<std::string, std::string> running_;  // job_id -> worker_id

    mutable std::mutex mutex_;
    std::condition_variable job_available_;
    std::atomic<size_t> job_counter_{0};
};

// =============================================================================
// Redis Job Queue (Optional - Compile with -DPULSIM_WITH_REDIS)
// =============================================================================

#ifdef PULSIM_WITH_REDIS

class RedisJobQueue : public IJobQueue {
public:
    struct Config {
        std::string host = "localhost";
        int port = 6379;
        std::string password;
        int db = 0;
        std::string key_prefix = "pulsim:";
        int connection_timeout_ms = 5000;
        int socket_timeout_ms = 5000;
    };

    explicit RedisJobQueue(const Config& config);
    ~RedisJobQueue() override;

    std::string submit(SimulationJob job) override;
    std::optional<SimulationJob> dequeue(const std::chrono::milliseconds& timeout) override;
    std::optional<JobStatus> get_status(const std::string& job_id) override;
    std::optional<JobResult> get_result(const std::string& job_id) override;
    void update_status(const std::string& job_id, JobStatus status) override;
    void set_result(const std::string& job_id, const JobResult& result) override;
    bool cancel(const std::string& job_id) override;

    size_t pending_count() const override;
    size_t running_count() const override;
    size_t completed_count() const override;

    std::vector<std::string> list_pending(size_t limit) override;
    std::vector<std::string> list_running() override;
    std::vector<std::string> list_by_user(const std::string& user_id, size_t limit) override;

    // Redis-specific: health check
    bool ping();

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

#endif  // PULSIM_WITH_REDIS

// =============================================================================
// Worker Pool
// =============================================================================

struct WorkerPoolConfig {
    size_t num_workers = 0;  // 0 = auto (hardware concurrency)
    size_t max_job_time_seconds = 3600;
    size_t heartbeat_interval_seconds = 30;
    bool enable_job_isolation = false;  // Run jobs in separate processes
};

class WorkerPool {
public:
    using Config = WorkerPoolConfig;
    using CompletionCallback = std::function<void(const JobResult&)>;

    WorkerPool(IJobQueue& queue, QuotaManager& quotas, const Config& config = Config{});
    ~WorkerPool();

    // Start/stop workers
    void start();
    void stop();
    bool is_running() const { return running_; }

    // Get worker count
    size_t worker_count() const { return workers_.size(); }
    size_t active_workers() const { return active_count_; }

    // Set completion callback
    void set_completion_callback(CompletionCallback callback) {
        completion_callback_ = std::move(callback);
    }

    // Worker statistics
    struct WorkerStats {
        std::string worker_id;
        size_t jobs_completed = 0;
        size_t jobs_failed = 0;
        double total_cpu_time = 0.0;
        std::string current_job_id;
        std::chrono::system_clock::time_point started_at;
    };

    std::vector<WorkerStats> get_worker_stats() const;

private:
    void worker_loop(size_t worker_index);
    void execute_job(SimulationJob& job, WorkerStats& stats);
    std::string generate_worker_id(size_t index);

    IJobQueue& queue_;
    QuotaManager& quotas_;
    Config config_;

    std::vector<std::thread> workers_;
    std::vector<WorkerStats> worker_stats_;
    std::atomic<bool> running_{false};
    std::atomic<size_t> active_count_{0};
    mutable std::mutex stats_mutex_;

    CompletionCallback completion_callback_;
};

// =============================================================================
// Job Queue Factory
// =============================================================================

enum class QueueBackend {
    Memory,
    Redis
};

struct JobQueueConfig {
    QueueBackend backend = QueueBackend::Memory;

#ifdef PULSIM_WITH_REDIS
    RedisJobQueue::Config redis;
#endif
};

std::unique_ptr<IJobQueue> create_job_queue(const JobQueueConfig& config);

}  // namespace pulsim::api::grpc
