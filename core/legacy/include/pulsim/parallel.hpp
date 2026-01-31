#pragma once

#include "pulsim/types.hpp"
#include "pulsim/circuit.hpp"
#include "pulsim/simulation.hpp"
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <future>
#include <functional>
#include <atomic>
#include <memory>
#include <vector>

namespace pulsim {

// =============================================================================
// Thread Pool for Parallel Execution
// =============================================================================

class ThreadPool {
public:
    explicit ThreadPool(size_t num_threads = 0);
    ~ThreadPool();

    // Non-copyable
    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    // Submit a task and get a future for the result
    template<typename F, typename... Args>
    auto submit(F&& f, Args&&... args) -> std::future<decltype(f(args...))>;

    // Get number of worker threads
    size_t size() const { return workers_.size(); }

    // Wait for all tasks to complete
    void wait_all();

    // Get number of pending tasks
    size_t pending_tasks() const;

    // Singleton access (for global thread pool)
    static ThreadPool& global();

private:
    std::vector<std::thread> workers_;
    std::queue<std::function<void()>> tasks_;

    mutable std::mutex queue_mutex_;
    std::condition_variable condition_;
    std::condition_variable finished_condition_;
    std::atomic<bool> stop_{false};
    std::atomic<size_t> active_tasks_{0};
};

// Template implementation
template<typename F, typename... Args>
auto ThreadPool::submit(F&& f, Args&&... args) -> std::future<decltype(f(args...))> {
    using return_type = decltype(f(args...));

    auto task = std::make_shared<std::packaged_task<return_type()>>(
        std::bind(std::forward<F>(f), std::forward<Args>(args)...)
    );

    std::future<return_type> result = task->get_future();
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        if (stop_) {
            throw std::runtime_error("Cannot submit task to stopped thread pool");
        }
        tasks_.emplace([task]() { (*task)(); });
    }
    condition_.notify_one();
    return result;
}

// =============================================================================
// Parallel Matrix Assembly
// =============================================================================

struct ParallelAssemblyOptions {
    size_t num_threads;          // 0 = auto (use hardware concurrency)
    size_t min_components_per_thread;  // Minimum components to parallelize

    ParallelAssemblyOptions()
        : num_threads(0)
        , min_components_per_thread(100) {}
};

// Parallel MNA matrix assembler
// Splits component processing across threads, then merges triplets
class ParallelMNAAssembler {
public:
    explicit ParallelMNAAssembler(const Circuit& circuit,
                                   const ParallelAssemblyOptions& opts = ParallelAssemblyOptions());

    // Parallel DC assembly
    void assemble_dc_parallel(SparseMatrix& G, Vector& b);

    // Parallel transient assembly
    void assemble_transient_parallel(SparseMatrix& G, Vector& b,
                                     const Vector& x_prev, Real dt);

    // Get assembly statistics
    size_t threads_used() const { return threads_used_; }
    double speedup() const { return speedup_; }

private:
    const Circuit& circuit_;
    ParallelAssemblyOptions options_;
    size_t threads_used_ = 1;
    double speedup_ = 1.0;
};

// =============================================================================
// SIMD-Optimized Device Evaluation
// =============================================================================

// SIMD configuration (compile-time detection)
struct SIMDInfo {
    static bool has_sse2();
    static bool has_avx();
    static bool has_avx2();
    static bool has_avx512();
    static const char* best_available();
};

// SIMD-optimized exponential for diode evaluation
// Processes multiple values at once
namespace simd {

// Batch exponential: exp(x) for array of values
void exp_batch(const Real* input, Real* output, size_t count);

// Batch diode current: Is * (exp(Vd / (n * Vt)) - 1)
void diode_current_batch(const Real* Vd, Real* Id, Real* Gd,
                         size_t count, Real Is, Real n, Real Vt);

// Batch MOSFET drain current (saturation region)
void mosfet_saturation_batch(const Real* Vgs, const Real* Vds,
                              Real* Id, Real* gm, Real* gds,
                              size_t count, Real Kp, Real Vth, Real lambda);

}  // namespace simd

// =============================================================================
// Parallel Parameter Sweeps
// =============================================================================

// Parameter to sweep
struct SweepParameter {
    std::string component_name;  // Component to modify
    std::string param_name;      // Parameter name (e.g., "resistance", "capacitance")
    std::vector<Real> values;    // Values to sweep

    // Linear sweep constructor
    static SweepParameter linear(const std::string& comp, const std::string& param,
                                  Real start, Real stop, size_t count);

    // Logarithmic sweep constructor
    static SweepParameter logarithmic(const std::string& comp, const std::string& param,
                                       Real start, Real stop, size_t count);

    // Explicit values constructor
    static SweepParameter list(const std::string& comp, const std::string& param,
                               const std::vector<Real>& values);
};

// Result of a single sweep point
struct SweepPointResult {
    std::vector<Real> parameter_values;  // The parameter values for this point
    SimulationResult simulation;         // Full simulation result
    bool success = false;
    std::string error_message;
};

// Full sweep result
struct ParameterSweepResult {
    std::vector<SweepParameter> parameters;
    std::vector<SweepPointResult> results;

    size_t total_points() const { return results.size(); }
    size_t successful_points() const;

    // Get result at specific index
    const SweepPointResult& at(size_t index) const { return results.at(index); }
};

// Parallel parameter sweep executor
class ParameterSweeper {
public:
    explicit ParameterSweeper(size_t num_threads = 0);

    // Run sweep with single parameter
    ParameterSweepResult sweep(const Circuit& base_circuit,
                                const SimulationOptions& options,
                                const SweepParameter& param);

    // Run sweep with multiple parameters (Cartesian product)
    ParameterSweepResult sweep(const Circuit& base_circuit,
                                const SimulationOptions& options,
                                const std::vector<SweepParameter>& params);

    // Progress callback
    using ProgressCallback = std::function<void(size_t completed, size_t total)>;
    void set_progress_callback(ProgressCallback callback) { progress_callback_ = callback; }

    // Cancel ongoing sweep
    void cancel() { cancelled_ = true; }
    bool is_cancelled() const { return cancelled_; }

private:
    // Modify circuit with parameter values
    Circuit apply_parameters(const Circuit& base,
                             const std::vector<SweepParameter>& params,
                             const std::vector<size_t>& indices);

    size_t num_threads_;
    std::atomic<bool> cancelled_{false};
    ProgressCallback progress_callback_;
};

// =============================================================================
// Job Queue for Batch Runs
// =============================================================================

// Job priority levels
enum class JobPriority {
    Low = 0,
    Normal = 1,
    High = 2,
    Critical = 3
};

// Job status
enum class JobStatus {
    Pending,
    Running,
    Completed,
    Failed,
    Cancelled
};

// A simulation job
struct SimulationJob {
    std::string id;              // Unique job identifier
    Circuit circuit;             // Circuit to simulate
    SimulationOptions options;   // Simulation options
    JobPriority priority = JobPriority::Normal;

    // Metadata
    std::string description;
    std::chrono::system_clock::time_point submitted_at;
};

// Job result
struct JobResult {
    std::string job_id;
    JobStatus status = JobStatus::Pending;
    SimulationResult simulation;
    std::string error_message;

    std::chrono::system_clock::time_point started_at;
    std::chrono::system_clock::time_point completed_at;
    double duration_seconds = 0.0;
};

// Job queue for managing batch simulations
class JobQueue {
public:
    explicit JobQueue(size_t num_workers = 0);
    ~JobQueue();

    // Non-copyable
    JobQueue(const JobQueue&) = delete;
    JobQueue& operator=(const JobQueue&) = delete;

    // Submit a job
    std::string submit(const SimulationJob& job);
    std::string submit(const Circuit& circuit, const SimulationOptions& options,
                       JobPriority priority = JobPriority::Normal);

    // Get job status
    JobStatus get_status(const std::string& job_id) const;

    // Get job result (blocks until complete if not ready)
    JobResult get_result(const std::string& job_id, bool wait = true);

    // Check if result is ready
    bool is_ready(const std::string& job_id) const;

    // Cancel a job
    bool cancel(const std::string& job_id);

    // Get queue statistics
    size_t pending_count() const;
    size_t running_count() const;
    size_t completed_count() const;
    size_t total_count() const;

    // Clear completed jobs
    void clear_completed();

    // Shutdown the queue
    void shutdown();

    // Job completion callback
    using CompletionCallback = std::function<void(const JobResult&)>;
    void set_completion_callback(CompletionCallback callback) { completion_callback_ = callback; }

private:
    void worker_loop();
    std::string generate_job_id();

    std::vector<std::thread> workers_;

    // Priority queue comparator
    struct JobComparator {
        bool operator()(const SimulationJob& a, const SimulationJob& b) const {
            return static_cast<int>(a.priority) < static_cast<int>(b.priority);
        }
    };

    std::priority_queue<SimulationJob, std::vector<SimulationJob>, JobComparator> pending_jobs_;
    std::unordered_map<std::string, JobResult> results_;
    std::unordered_map<std::string, JobStatus> job_status_;

    mutable std::mutex mutex_;
    std::condition_variable job_available_;
    std::condition_variable result_available_;
    std::atomic<bool> shutdown_{false};
    std::atomic<size_t> running_count_{0};
    std::atomic<size_t> job_counter_{0};

    CompletionCallback completion_callback_;
};

// =============================================================================
// Convenience Functions
// =============================================================================

// Get recommended number of threads for parallelization
size_t recommended_threads();

// Run simulations in parallel
std::vector<SimulationResult> run_parallel(
    const std::vector<std::pair<Circuit, SimulationOptions>>& simulations,
    size_t num_threads = 0);

}  // namespace pulsim
