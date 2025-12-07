#include "pulsim/api/grpc/job_queue.hpp"
#include "pulsim/api/grpc/metrics.hpp"
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <random>
#include <sstream>

namespace pulsim::api::grpc {

// =============================================================================
// QuotaManager Implementation
// =============================================================================

QuotaManager::QuotaManager() {
    // Set sensible defaults
    default_quota_.max_concurrent_jobs = 4;
    default_quota_.max_queued_jobs = 100;
    default_quota_.max_simulation_time_seconds = 3600;
    default_quota_.max_timesteps = 10000000;
    default_quota_.max_memory_mb = 4096;
    default_quota_.max_jobs_per_hour = 100;
    default_quota_.max_jobs_per_day = 1000;
}

bool QuotaManager::can_submit(const std::string& user_id, std::string& reason) {
    auto& quota = get_quota(user_id);

    // Check concurrent limit
    if (quota.current_running >= quota.max_concurrent_jobs) {
        reason = "Maximum concurrent jobs limit reached (" +
                 std::to_string(quota.max_concurrent_jobs) + ")";
        PULSIM_METRICS.quota_exceeded_total().inc();
        return false;
    }

    // Check queue limit
    if (quota.current_queued >= quota.max_queued_jobs) {
        reason = "Maximum queued jobs limit reached (" +
                 std::to_string(quota.max_queued_jobs) + ")";
        PULSIM_METRICS.quota_exceeded_total().inc();
        return false;
    }

    // Check hourly rate limit
    auto now = std::chrono::system_clock::now();
    if (now >= quota.hour_reset) {
        quota.jobs_this_hour = 0;
        quota.hour_reset = now + std::chrono::hours(1);
    }
    if (quota.jobs_this_hour >= quota.max_jobs_per_hour) {
        reason = "Hourly job limit reached (" +
                 std::to_string(quota.max_jobs_per_hour) + ")";
        PULSIM_METRICS.quota_exceeded_total().inc();
        return false;
    }

    // Check daily rate limit
    if (now >= quota.day_reset) {
        quota.jobs_today = 0;
        quota.day_reset = now + std::chrono::hours(24);
    }
    if (quota.jobs_today >= quota.max_jobs_per_day) {
        reason = "Daily job limit reached (" +
                 std::to_string(quota.max_jobs_per_day) + ")";
        PULSIM_METRICS.quota_exceeded_total().inc();
        return false;
    }

    return true;
}

bool QuotaManager::reserve(const std::string& user_id) {
    auto& quota = get_quota(user_id);
    quota.current_queued++;
    quota.jobs_this_hour++;
    quota.jobs_today++;
    return true;
}

void QuotaManager::release(const std::string& user_id, bool was_running) {
    auto& quota = get_quota(user_id);
    if (was_running) {
        if (quota.current_running > 0) quota.current_running--;
    } else {
        if (quota.current_queued > 0) quota.current_queued--;
    }
}

void QuotaManager::set_quota(const std::string& user_id, const UserQuota& quota) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = quotas_.find(user_id);
    if (it != quotas_.end()) {
        // Update existing quota (preserve usage counters)
        auto& existing = *it->second;
        existing.max_concurrent_jobs = quota.max_concurrent_jobs;
        existing.max_queued_jobs = quota.max_queued_jobs;
        existing.max_simulation_time_seconds = quota.max_simulation_time_seconds;
        existing.max_timesteps = quota.max_timesteps;
        existing.max_memory_mb = quota.max_memory_mb;
        existing.max_jobs_per_hour = quota.max_jobs_per_hour;
        existing.max_jobs_per_day = quota.max_jobs_per_day;
    } else {
        auto q = std::make_unique<UserQuota>(quota);
        q->user_id = user_id;
        auto now = std::chrono::system_clock::now();
        q->hour_reset = now + std::chrono::hours(1);
        q->day_reset = now + std::chrono::hours(24);
        quotas_[user_id] = std::move(q);
    }
}

UserQuota& QuotaManager::get_quota(const std::string& user_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = quotas_.find(user_id);
    if (it != quotas_.end()) {
        return *it->second;
    }

    // Create quota from defaults
    auto q = std::make_unique<UserQuota>(default_quota_);
    q->user_id = user_id;
    auto now = std::chrono::system_clock::now();
    q->hour_reset = now + std::chrono::hours(1);
    q->day_reset = now + std::chrono::hours(24);
    auto& ref = *q;
    quotas_[user_id] = std::move(q);
    return ref;
}

void QuotaManager::reset_hourly_limits() {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& [user_id, quota] : quotas_) {
        quota->jobs_this_hour = 0;
        quota->hour_reset = std::chrono::system_clock::now() + std::chrono::hours(1);
    }
}

void QuotaManager::reset_daily_limits() {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& [user_id, quota] : quotas_) {
        quota->jobs_today = 0;
        quota->day_reset = std::chrono::system_clock::now() + std::chrono::hours(24);
    }
}

// =============================================================================
// MemoryJobQueue Implementation
// =============================================================================

MemoryJobQueue::MemoryJobQueue() = default;
MemoryJobQueue::~MemoryJobQueue() = default;

std::string MemoryJobQueue::submit(SimulationJob job) {
    // Generate job ID if not provided
    if (job.job_id.empty()) {
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        size_t counter = ++job_counter_;

        std::ostringstream oss;
        oss << "job_" << std::put_time(std::localtime(&time_t), "%Y%m%d_%H%M%S")
            << "_" << std::setfill('0') << std::setw(6) << counter;
        job.job_id = oss.str();
    }

    job.created_at = std::chrono::system_clock::now();

    std::string job_id = job.job_id;

    {
        std::lock_guard<std::mutex> lock(mutex_);
        jobs_[job_id] = job;
        status_[job_id] = JobStatus::Queued;
        pending_.push(std::move(job));
    }

    job_available_.notify_one();
    PULSIM_METRICS.jobs_pending().inc();

    return job_id;
}

std::optional<SimulationJob> MemoryJobQueue::dequeue(const std::chrono::milliseconds& timeout) {
    std::unique_lock<std::mutex> lock(mutex_);

    if (!job_available_.wait_for(lock, timeout, [this] {
        return !pending_.empty();
    })) {
        return std::nullopt;  // Timeout
    }

    SimulationJob job = pending_.top();
    pending_.pop();

    status_[job.job_id] = JobStatus::Running;
    running_[job.job_id] = "";  // Worker ID set later

    PULSIM_METRICS.jobs_pending().dec();
    PULSIM_METRICS.jobs_running().inc();

    return job;
}

std::optional<JobStatus> MemoryJobQueue::get_status(const std::string& job_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = status_.find(job_id);
    if (it != status_.end()) {
        return it->second;
    }
    return std::nullopt;
}

std::optional<JobResult> MemoryJobQueue::get_result(const std::string& job_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = results_.find(job_id);
    if (it != results_.end()) {
        return it->second;
    }
    return std::nullopt;
}

void MemoryJobQueue::update_status(const std::string& job_id, JobStatus status) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = status_.find(job_id);
    if (it != status_.end()) {
        it->second = status;

        if (status != JobStatus::Running && status != JobStatus::Queued) {
            running_.erase(job_id);
        }
    }
}

void MemoryJobQueue::set_result(const std::string& job_id, const JobResult& result) {
    std::lock_guard<std::mutex> lock(mutex_);
    results_[job_id] = result;
    status_[job_id] = result.status;
    running_.erase(job_id);

    PULSIM_METRICS.jobs_running().dec();

    switch (result.status) {
        case JobStatus::Completed:
            PULSIM_METRICS.jobs_completed().inc();
            break;
        case JobStatus::Failed:
            PULSIM_METRICS.jobs_failed().inc();
            break;
        case JobStatus::Cancelled:
            PULSIM_METRICS.jobs_cancelled().inc();
            break;
        case JobStatus::Timeout:
            PULSIM_METRICS.jobs_timeout().inc();
            break;
        default:
            break;
    }
}

bool MemoryJobQueue::cancel(const std::string& job_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = status_.find(job_id);
    if (it != status_.end() && it->second == JobStatus::Queued) {
        it->second = JobStatus::Cancelled;

        JobResult result;
        result.job_id = job_id;
        result.status = JobStatus::Cancelled;
        result.error_message = "Job cancelled by user";
        result.completed_at = std::chrono::system_clock::now();
        results_[job_id] = result;

        PULSIM_METRICS.jobs_pending().dec();
        PULSIM_METRICS.jobs_cancelled().inc();
        return true;
    }
    return false;
}

size_t MemoryJobQueue::pending_count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return pending_.size();
}

size_t MemoryJobQueue::running_count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return running_.size();
}

size_t MemoryJobQueue::completed_count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    size_t count = 0;
    for (const auto& [id, status] : status_) {
        if (status == JobStatus::Completed || status == JobStatus::Failed ||
            status == JobStatus::Cancelled || status == JobStatus::Timeout) {
            ++count;
        }
    }
    return count;
}

std::vector<std::string> MemoryJobQueue::list_pending(size_t limit) {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<std::string> result;

    for (const auto& [id, status] : status_) {
        if (status == JobStatus::Queued) {
            result.push_back(id);
            if (result.size() >= limit) break;
        }
    }
    return result;
}

std::vector<std::string> MemoryJobQueue::list_running() {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<std::string> result;
    for (const auto& [id, worker] : running_) {
        result.push_back(id);
    }
    return result;
}

std::vector<std::string> MemoryJobQueue::list_by_user(const std::string& user_id, size_t limit) {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<std::string> result;

    for (const auto& [id, job] : jobs_) {
        if (job.user_id == user_id) {
            result.push_back(id);
            if (result.size() >= limit) break;
        }
    }
    return result;
}

void MemoryJobQueue::clear_completed_older_than(std::chrono::seconds age) {
    auto cutoff = std::chrono::system_clock::now() - age;

    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<std::string> to_remove;

    for (const auto& [id, result] : results_) {
        if (result.completed_at < cutoff) {
            to_remove.push_back(id);
        }
    }

    for (const auto& id : to_remove) {
        jobs_.erase(id);
        status_.erase(id);
        results_.erase(id);
    }
}

// =============================================================================
// WorkerPool Implementation
// =============================================================================

WorkerPool::WorkerPool(IJobQueue& queue, QuotaManager& quotas, const Config& config)
    : queue_(queue)
    , quotas_(quotas)
    , config_(config)
{
    if (config_.num_workers == 0) {
        config_.num_workers = std::thread::hardware_concurrency();
        if (config_.num_workers == 0) config_.num_workers = 4;
    }
}

WorkerPool::~WorkerPool() {
    stop();
}

void WorkerPool::start() {
    if (running_) return;

    running_ = true;
    workers_.reserve(config_.num_workers);
    worker_stats_.resize(config_.num_workers);

    for (size_t i = 0; i < config_.num_workers; ++i) {
        worker_stats_[i].worker_id = generate_worker_id(i);
        worker_stats_[i].started_at = std::chrono::system_clock::now();
        workers_.emplace_back(&WorkerPool::worker_loop, this, i);
    }

    PULSIM_METRICS.workers_idle().set(static_cast<double>(config_.num_workers));
}

void WorkerPool::stop() {
    if (!running_) return;

    running_ = false;

    for (auto& worker : workers_) {
        if (worker.joinable()) {
            worker.join();
        }
    }

    workers_.clear();
    worker_stats_.clear();

    PULSIM_METRICS.workers_active().set(0);
    PULSIM_METRICS.workers_idle().set(0);
}

std::string WorkerPool::generate_worker_id(size_t index) {
    std::ostringstream oss;
    oss << "worker_" << std::setfill('0') << std::setw(3) << index;
    return oss.str();
}

std::vector<WorkerPool::WorkerStats> WorkerPool::get_worker_stats() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    return worker_stats_;
}

void WorkerPool::worker_loop(size_t worker_index) {
    auto& stats = worker_stats_[worker_index];

    while (running_) {
        auto job_opt = queue_.dequeue(std::chrono::milliseconds(1000));

        if (!job_opt) {
            continue;  // Timeout, check if still running
        }

        auto& job = *job_opt;

        // Update stats
        {
            std::lock_guard<std::mutex> lock(stats_mutex_);
            stats.current_job_id = job.job_id;
        }

        ++active_count_;
        PULSIM_METRICS.workers_active().inc();
        PULSIM_METRICS.workers_idle().dec();

        // Move from queued to running for quota tracking
        quotas_.get_quota(job.user_id).current_queued--;
        quotas_.get_quota(job.user_id).current_running++;

        execute_job(job, stats);

        // Release running quota
        quotas_.release(job.user_id, true);

        --active_count_;
        PULSIM_METRICS.workers_active().dec();
        PULSIM_METRICS.workers_idle().inc();

        {
            std::lock_guard<std::mutex> lock(stats_mutex_);
            stats.current_job_id.clear();
        }
    }
}

void WorkerPool::execute_job(SimulationJob& job, WorkerStats& stats) {
    JobResult result;
    result.job_id = job.job_id;
    result.worker_id = stats.worker_id;
    result.started_at = std::chrono::system_clock::now();

    auto timer = PULSIM_METRICS.simulation_duration().start_timer();

    try {
        // Check timeout
        auto deadline = job.deadline;
        if (deadline == std::chrono::system_clock::time_point{}) {
            deadline = result.started_at +
                       std::chrono::seconds(config_.max_job_time_seconds);
        }

        // Run simulation
        Simulator sim(job.circuit, job.options);
        result.simulation = sim.run_transient();

        result.status = (result.simulation.final_status == SolverStatus::Success)
                        ? JobStatus::Completed : JobStatus::Failed;

        if (result.status == JobStatus::Failed) {
            result.error_message = result.simulation.error_message;
        }

        result.timesteps = result.simulation.total_steps;
        result.newton_iterations = result.simulation.newton_iterations_total;

        PULSIM_METRICS.simulations_total().inc();
        PULSIM_METRICS.simulation_timesteps().observe(static_cast<double>(result.timesteps));
        PULSIM_METRICS.newton_iterations().observe(static_cast<double>(result.newton_iterations));

        if (result.status == JobStatus::Failed) {
            PULSIM_METRICS.simulations_failed().inc();
            stats.jobs_failed++;
        } else {
            stats.jobs_completed++;
        }

    } catch (const std::exception& e) {
        result.status = JobStatus::Failed;
        result.error_message = e.what();
        PULSIM_METRICS.simulations_failed().inc();
        stats.jobs_failed++;
    }

    result.completed_at = std::chrono::system_clock::now();
    result.duration_seconds = std::chrono::duration<double>(
        result.completed_at - result.started_at).count();

    stats.total_cpu_time += result.duration_seconds;

    queue_.set_result(job.job_id, result);

    // Callback
    if (completion_callback_) {
        completion_callback_(result);
    }
}

// =============================================================================
// Job Queue Factory
// =============================================================================

std::unique_ptr<IJobQueue> create_job_queue(const JobQueueConfig& config) {
    switch (config.backend) {
        case QueueBackend::Memory:
            return std::make_unique<MemoryJobQueue>();

#ifdef PULSIM_WITH_REDIS
        case QueueBackend::Redis:
            return std::make_unique<RedisJobQueue>(config.redis);
#endif

        default:
            return std::make_unique<MemoryJobQueue>();
    }
}

}  // namespace pulsim::api::grpc
