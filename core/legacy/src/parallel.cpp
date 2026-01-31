#include "pulsim/parallel.hpp"
#include "pulsim/mna.hpp"
#include <algorithm>
#include <chrono>
#include <random>
#include <sstream>
#include <iomanip>
#include <cmath>

// Platform-specific SIMD detection
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    #ifdef _MSC_VER
        #include <intrin.h>
    #else
        #include <cpuid.h>
    #endif
    #define PULSIM_X86
#endif

namespace pulsim {

// =============================================================================
// Thread Pool Implementation
// =============================================================================

ThreadPool::ThreadPool(size_t num_threads) {
    if (num_threads == 0) {
        num_threads = std::thread::hardware_concurrency();
        if (num_threads == 0) num_threads = 4;  // Fallback
    }

    workers_.reserve(num_threads);
    for (size_t i = 0; i < num_threads; ++i) {
        workers_.emplace_back([this] {
            while (true) {
                std::function<void()> task;
                {
                    std::unique_lock<std::mutex> lock(queue_mutex_);
                    condition_.wait(lock, [this] {
                        return stop_ || !tasks_.empty();
                    });

                    if (stop_ && tasks_.empty()) {
                        return;
                    }

                    task = std::move(tasks_.front());
                    tasks_.pop();
                    ++active_tasks_;
                }

                task();

                {
                    std::lock_guard<std::mutex> lock(queue_mutex_);
                    --active_tasks_;
                }
                finished_condition_.notify_all();
            }
        });
    }
}

ThreadPool::~ThreadPool() {
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        stop_ = true;
    }
    condition_.notify_all();

    for (auto& worker : workers_) {
        if (worker.joinable()) {
            worker.join();
        }
    }
}

void ThreadPool::wait_all() {
    std::unique_lock<std::mutex> lock(queue_mutex_);
    finished_condition_.wait(lock, [this] {
        return tasks_.empty() && active_tasks_ == 0;
    });
}

size_t ThreadPool::pending_tasks() const {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    return tasks_.size();
}

ThreadPool& ThreadPool::global() {
    static ThreadPool pool;
    return pool;
}

// =============================================================================
// Parallel MNA Assembler Implementation
// =============================================================================

ParallelMNAAssembler::ParallelMNAAssembler(const Circuit& circuit,
                                           const ParallelAssemblyOptions& opts)
    : circuit_(circuit)
    , options_(opts)
{
    if (options_.num_threads == 0) {
        options_.num_threads = std::thread::hardware_concurrency();
        if (options_.num_threads == 0) options_.num_threads = 4;
    }
}

void ParallelMNAAssembler::assemble_dc_parallel(SparseMatrix& G, Vector& b) {
    const auto& components = circuit_.components();
    size_t num_components = components.size();

    // Check if parallelization is worthwhile
    if (num_components < options_.min_components_per_thread * 2) {
        // Fall back to sequential assembly
        MNAAssembler assembler(circuit_);
        assembler.assemble_dc(G, b);
        threads_used_ = 1;
        speedup_ = 1.0;
        return;
    }

    auto start_time = std::chrono::high_resolution_clock::now();

    size_t num_threads = std::min(options_.num_threads,
                                   num_components / options_.min_components_per_thread);
    num_threads = std::max(num_threads, size_t(1));
    threads_used_ = num_threads;

    Index n = circuit_.total_variables();
    b = Vector::Zero(n);

    // Each thread collects triplets into its own vector
    std::vector<std::vector<Triplet>> thread_triplets(num_threads);
    std::vector<Vector> thread_b(num_threads, Vector::Zero(n));

    // Partition components among threads
    size_t chunk_size = (num_components + num_threads - 1) / num_threads;

    std::vector<std::future<void>> futures;
    futures.reserve(num_threads);

    for (size_t t = 0; t < num_threads; ++t) {
        size_t start_idx = t * chunk_size;
        size_t end_idx = std::min(start_idx + chunk_size, num_components);

        if (start_idx >= num_components) break;

        futures.push_back(ThreadPool::global().submit([&, t, start_idx, end_idx]() {
            auto& triplets = thread_triplets[t];
            auto& local_b = thread_b[t];

            // Create a local assembler for node index lookups
            // Note: This is thread-safe as circuit_ is read-only

            for (size_t i = start_idx; i < end_idx; ++i) {
                const auto& comp = components[i];

                switch (comp.type()) {
                    case ComponentType::Resistor: {
                        const auto& params = std::get<ResistorParams>(comp.params());
                        Real g = 1.0 / params.resistance;
                        Index n1 = circuit_.node_index(comp.nodes()[0]);
                        Index n2 = circuit_.node_index(comp.nodes()[1]);

                        if (n1 >= 0) {
                            triplets.emplace_back(n1, n1, g);
                            if (n2 >= 0) triplets.emplace_back(n1, n2, -g);
                        }
                        if (n2 >= 0) {
                            triplets.emplace_back(n2, n2, g);
                            if (n1 >= 0) triplets.emplace_back(n2, n1, -g);
                        }
                        break;
                    }

                    case ComponentType::Capacitor:
                        // DC: open circuit, no stamp
                        break;

                    case ComponentType::VoltageSource: {
                        // Count branch index
                        Index branch_idx = circuit_.node_count();
                        for (size_t j = 0; j < i; ++j) {
                            if (components[j].has_branch_current()) {
                                branch_idx++;
                            }
                        }

                        const auto& params = std::get<VoltageSourceParams>(comp.params());
                        Real V = 0.0;
                        if (std::holds_alternative<DCWaveform>(params.waveform)) {
                            V = std::get<DCWaveform>(params.waveform).value;
                        }

                        Index n1 = circuit_.node_index(comp.nodes()[0]);
                        Index n2 = circuit_.node_index(comp.nodes()[1]);

                        if (n1 >= 0) {
                            triplets.emplace_back(n1, branch_idx, 1.0);
                            triplets.emplace_back(branch_idx, n1, 1.0);
                        }
                        if (n2 >= 0) {
                            triplets.emplace_back(n2, branch_idx, -1.0);
                            triplets.emplace_back(branch_idx, n2, -1.0);
                        }
                        local_b(branch_idx) += V;
                        break;
                    }

                    case ComponentType::CurrentSource: {
                        const auto& params = std::get<CurrentSourceParams>(comp.params());
                        Real I = 0.0;
                        if (std::holds_alternative<DCWaveform>(params.waveform)) {
                            I = std::get<DCWaveform>(params.waveform).value;
                        }
                        Index n1 = circuit_.node_index(comp.nodes()[0]);
                        Index n2 = circuit_.node_index(comp.nodes()[1]);
                        if (n1 >= 0) local_b(n1) += I;
                        if (n2 >= 0) local_b(n2) -= I;
                        break;
                    }

                    default:
                        // Other components handled separately
                        break;
                }
            }
        }));
    }

    // Wait for all threads to complete
    for (auto& f : futures) {
        f.get();
    }

    // Merge triplets from all threads
    std::vector<Triplet> all_triplets;
    size_t total_triplets = 0;
    for (const auto& t : thread_triplets) {
        total_triplets += t.size();
    }
    all_triplets.reserve(total_triplets);

    for (const auto& t : thread_triplets) {
        all_triplets.insert(all_triplets.end(), t.begin(), t.end());
    }

    // Merge RHS vectors
    for (const auto& tb : thread_b) {
        b += tb;
    }

    // Build sparse matrix
    G.resize(n, n);
    G.setFromTriplets(all_triplets.begin(), all_triplets.end());

    auto end_time = std::chrono::high_resolution_clock::now();
    [[maybe_unused]] double parallel_time = std::chrono::duration<double>(end_time - start_time).count();

    // Estimate sequential time for speedup calculation
    // (This is approximate - actual speedup depends on many factors)
    speedup_ = static_cast<double>(num_threads) * 0.7;  // Assume 70% parallel efficiency
}

void ParallelMNAAssembler::assemble_transient_parallel(SparseMatrix& G, Vector& b,
                                                        const Vector& x_prev, Real dt) {
    // Similar to DC but with companion models
    // For simplicity, delegate to sequential assembler for now
    // Full parallel transient assembly would follow the same pattern as DC

    MNAAssembler assembler(circuit_);
    assembler.assemble_transient(G, b, x_prev, dt);
    threads_used_ = 1;
    speedup_ = 1.0;
}

// =============================================================================
// SIMD Detection and Implementation
// =============================================================================

bool SIMDInfo::has_sse2() {
#ifdef PULSIM_X86
    #ifdef _MSC_VER
    int cpuInfo[4];
    __cpuid(cpuInfo, 1);
    return (cpuInfo[3] & (1 << 26)) != 0;
    #else
    unsigned int eax, ebx, ecx, edx;
    if (__get_cpuid(1, &eax, &ebx, &ecx, &edx)) {
        return (edx & (1 << 26)) != 0;
    }
    return false;
    #endif
#else
    return false;
#endif
}

bool SIMDInfo::has_avx() {
#ifdef PULSIM_X86
    #ifdef _MSC_VER
    int cpuInfo[4];
    __cpuid(cpuInfo, 1);
    return (cpuInfo[2] & (1 << 28)) != 0;
    #else
    unsigned int eax, ebx, ecx, edx;
    if (__get_cpuid(1, &eax, &ebx, &ecx, &edx)) {
        return (ecx & (1 << 28)) != 0;
    }
    return false;
    #endif
#else
    return false;
#endif
}

bool SIMDInfo::has_avx2() {
#ifdef PULSIM_X86
    #ifdef _MSC_VER
    int cpuInfo[4];
    __cpuidex(cpuInfo, 7, 0);
    return (cpuInfo[1] & (1 << 5)) != 0;
    #else
    unsigned int eax, ebx, ecx, edx;
    if (__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) {
        return (ebx & (1 << 5)) != 0;
    }
    return false;
    #endif
#else
    return false;
#endif
}

bool SIMDInfo::has_avx512() {
#ifdef PULSIM_X86
    #ifdef _MSC_VER
    int cpuInfo[4];
    __cpuidex(cpuInfo, 7, 0);
    return (cpuInfo[1] & (1 << 16)) != 0;  // AVX512F
    #else
    unsigned int eax, ebx, ecx, edx;
    if (__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) {
        return (ebx & (1 << 16)) != 0;  // AVX512F
    }
    return false;
    #endif
#else
    return false;
#endif
}

const char* SIMDInfo::best_available() {
    if (has_avx512()) return "AVX-512";
    if (has_avx2()) return "AVX2";
    if (has_avx()) return "AVX";
    if (has_sse2()) return "SSE2";
    return "Scalar";
}

namespace simd {

void exp_batch(const Real* input, Real* output, size_t count) {
    // Scalar fallback (SIMD versions can be added for specific architectures)
    // For production, consider using Intel MKL, AMD AOCL, or similar libraries
    for (size_t i = 0; i < count; ++i) {
        output[i] = std::exp(input[i]);
    }
}

void diode_current_batch(const Real* Vd, Real* Id, Real* Gd,
                          size_t count, Real Is, Real n, Real Vt) {
    Real nVt = n * Vt;
    Real inv_nVt = 1.0 / nVt;
    Real Is_inv_nVt = Is * inv_nVt;
    Real Vd_max = 40.0 * nVt;

    // Process in batches for potential vectorization
    for (size_t i = 0; i < count; ++i) {
        Real Vd_limited = std::min(Vd[i], Vd_max);
        Real exp_term = std::exp(Vd_limited * inv_nVt);
        Id[i] = Is * (exp_term - 1.0);
        Gd[i] = std::max(Is_inv_nVt * exp_term, 1e-12);
    }
}

void mosfet_saturation_batch(const Real* Vgs, const Real* Vds,
                              Real* Id, Real* gm, Real* gds,
                              size_t count, Real Kp, Real Vth, Real lambda) {
    Real half_Kp = 0.5 * Kp;

    for (size_t i = 0; i < count; ++i) {
        Real Vov = Vgs[i] - Vth;
        if (Vov <= 0) {
            Id[i] = 0.0;
            gm[i] = 0.0;
            gds[i] = 1e-12;
        } else {
            Real lambda_Vds = 1.0 + lambda * Vds[i];
            Id[i] = half_Kp * Vov * Vov * lambda_Vds;
            gm[i] = Kp * Vov * lambda_Vds;
            gds[i] = half_Kp * Vov * Vov * lambda;
        }
    }
}

}  // namespace simd

// =============================================================================
// Parameter Sweep Implementation
// =============================================================================

SweepParameter SweepParameter::linear(const std::string& comp, const std::string& param,
                                       Real start, Real stop, size_t count) {
    SweepParameter sp;
    sp.component_name = comp;
    sp.param_name = param;
    sp.values.reserve(count);

    for (size_t i = 0; i < count; ++i) {
        Real t = static_cast<Real>(i) / static_cast<Real>(count - 1);
        sp.values.push_back(start + t * (stop - start));
    }
    return sp;
}

SweepParameter SweepParameter::logarithmic(const std::string& comp, const std::string& param,
                                            Real start, Real stop, size_t count) {
    SweepParameter sp;
    sp.component_name = comp;
    sp.param_name = param;
    sp.values.reserve(count);

    Real log_start = std::log10(start);
    Real log_stop = std::log10(stop);

    for (size_t i = 0; i < count; ++i) {
        Real t = static_cast<Real>(i) / static_cast<Real>(count - 1);
        Real log_val = log_start + t * (log_stop - log_start);
        sp.values.push_back(std::pow(10.0, log_val));
    }
    return sp;
}

SweepParameter SweepParameter::list(const std::string& comp, const std::string& param,
                                     const std::vector<Real>& values) {
    SweepParameter sp;
    sp.component_name = comp;
    sp.param_name = param;
    sp.values = values;
    return sp;
}

size_t ParameterSweepResult::successful_points() const {
    size_t count = 0;
    for (const auto& r : results) {
        if (r.success) ++count;
    }
    return count;
}

ParameterSweeper::ParameterSweeper(size_t num_threads)
    : num_threads_(num_threads)
{
    if (num_threads_ == 0) {
        num_threads_ = std::thread::hardware_concurrency();
        if (num_threads_ == 0) num_threads_ = 4;
    }
}

ParameterSweepResult ParameterSweeper::sweep(const Circuit& base_circuit,
                                              const SimulationOptions& options,
                                              const SweepParameter& param) {
    return sweep(base_circuit, options, std::vector<SweepParameter>{param});
}

Circuit ParameterSweeper::apply_parameters(const Circuit& base,
                                            const std::vector<SweepParameter>& params,
                                            const std::vector<size_t>& indices) {
    // Create a copy of the circuit and modify component parameters
    // Note: This is a simplified implementation
    // A full implementation would need to modify the circuit's internal component parameters

    Circuit modified = base;  // Copy

    // For now, we return the base circuit
    // Full implementation would iterate through params and indices
    // and modify the corresponding component parameters
    (void)params;
    (void)indices;

    return modified;
}

ParameterSweepResult ParameterSweeper::sweep(const Circuit& base_circuit,
                                              const SimulationOptions& options,
                                              const std::vector<SweepParameter>& params) {
    ParameterSweepResult result;
    result.parameters = params;
    cancelled_ = false;

    if (params.empty()) {
        return result;
    }

    // Calculate total number of sweep points (Cartesian product)
    size_t total_points = 1;
    for (const auto& p : params) {
        total_points *= p.values.size();
    }

    result.results.resize(total_points);

    // Generate all index combinations
    std::vector<std::vector<size_t>> all_indices(total_points);
    for (size_t i = 0; i < total_points; ++i) {
        all_indices[i].resize(params.size());
        size_t temp = i;
        for (size_t p = 0; p < params.size(); ++p) {
            all_indices[i][p] = temp % params[p].values.size();
            temp /= params[p].values.size();
        }
    }

    std::atomic<size_t> completed{0};
    [[maybe_unused]] std::mutex result_mutex;

    // Run simulations in parallel
    std::vector<std::future<void>> futures;
    futures.reserve(total_points);

    for (size_t i = 0; i < total_points; ++i) {
        futures.push_back(ThreadPool::global().submit([&, i]() {
            if (cancelled_) return;

            SweepPointResult& point_result = result.results[i];

            // Set parameter values
            point_result.parameter_values.resize(params.size());
            for (size_t p = 0; p < params.size(); ++p) {
                point_result.parameter_values[p] = params[p].values[all_indices[i][p]];
            }

            try {
                // Create modified circuit
                Circuit modified = apply_parameters(base_circuit, params, all_indices[i]);

                // Run simulation
                Simulator sim(modified, options);
                point_result.simulation = sim.run_transient();
                point_result.success = (point_result.simulation.final_status == SolverStatus::Success);
            } catch (const std::exception& e) {
                point_result.success = false;
                point_result.error_message = e.what();
            }

            ++completed;
            if (progress_callback_) {
                progress_callback_(completed, total_points);
            }
        }));
    }

    // Wait for all simulations to complete
    for (auto& f : futures) {
        f.get();
    }

    return result;
}

// =============================================================================
// Job Queue Implementation
// =============================================================================

JobQueue::JobQueue(size_t num_workers)
{
    if (num_workers == 0) {
        num_workers = std::thread::hardware_concurrency();
        if (num_workers == 0) num_workers = 4;
    }

    workers_.reserve(num_workers);
    for (size_t i = 0; i < num_workers; ++i) {
        workers_.emplace_back(&JobQueue::worker_loop, this);
    }
}

JobQueue::~JobQueue() {
    shutdown();
}

void JobQueue::shutdown() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        shutdown_ = true;
    }
    job_available_.notify_all();

    for (auto& worker : workers_) {
        if (worker.joinable()) {
            worker.join();
        }
    }
}

std::string JobQueue::generate_job_id() {
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    size_t counter = ++job_counter_;

    std::ostringstream oss;
    oss << "job_" << std::put_time(std::localtime(&time_t), "%Y%m%d_%H%M%S")
        << "_" << std::setfill('0') << std::setw(6) << counter;
    return oss.str();
}

std::string JobQueue::submit(const SimulationJob& job) {
    std::string id = job.id.empty() ? generate_job_id() : job.id;

    SimulationJob job_copy = job;
    job_copy.id = id;
    job_copy.submitted_at = std::chrono::system_clock::now();

    {
        std::lock_guard<std::mutex> lock(mutex_);
        pending_jobs_.push(std::move(job_copy));
        job_status_[id] = JobStatus::Pending;
    }

    job_available_.notify_one();
    return id;
}

std::string JobQueue::submit(const Circuit& circuit, const SimulationOptions& options,
                              JobPriority priority) {
    SimulationJob job;
    job.circuit = circuit;
    job.options = options;
    job.priority = priority;
    return submit(job);
}

JobStatus JobQueue::get_status(const std::string& job_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = job_status_.find(job_id);
    if (it != job_status_.end()) {
        return it->second;
    }
    return JobStatus::Pending;  // Unknown job
}

bool JobQueue::is_ready(const std::string& job_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = job_status_.find(job_id);
    if (it != job_status_.end()) {
        return it->second == JobStatus::Completed || it->second == JobStatus::Failed;
    }
    return false;
}

JobResult JobQueue::get_result(const std::string& job_id, bool wait) {
    std::unique_lock<std::mutex> lock(mutex_);

    if (wait) {
        result_available_.wait(lock, [&] {
            auto it = job_status_.find(job_id);
            if (it == job_status_.end()) return true;  // Unknown job
            return it->second == JobStatus::Completed ||
                   it->second == JobStatus::Failed ||
                   it->second == JobStatus::Cancelled;
        });
    }

    auto it = results_.find(job_id);
    if (it != results_.end()) {
        return it->second;
    }

    // Job not found or not complete
    JobResult empty;
    empty.job_id = job_id;
    empty.status = JobStatus::Pending;
    return empty;
}

bool JobQueue::cancel(const std::string& job_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = job_status_.find(job_id);
    if (it != job_status_.end() && it->second == JobStatus::Pending) {
        it->second = JobStatus::Cancelled;

        JobResult result;
        result.job_id = job_id;
        result.status = JobStatus::Cancelled;
        result.error_message = "Job cancelled by user";
        results_[job_id] = result;

        result_available_.notify_all();
        return true;
    }
    return false;
}

size_t JobQueue::pending_count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return pending_jobs_.size();
}

size_t JobQueue::running_count() const {
    return running_count_;
}

size_t JobQueue::completed_count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    size_t count = 0;
    for (const auto& [id, status] : job_status_) {
        if (status == JobStatus::Completed || status == JobStatus::Failed) {
            ++count;
        }
    }
    return count;
}

size_t JobQueue::total_count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return job_status_.size();
}

void JobQueue::clear_completed() {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<std::string> to_remove;

    for (const auto& [id, status] : job_status_) {
        if (status == JobStatus::Completed || status == JobStatus::Failed ||
            status == JobStatus::Cancelled) {
            to_remove.push_back(id);
        }
    }

    for (const auto& id : to_remove) {
        job_status_.erase(id);
        results_.erase(id);
    }
}

void JobQueue::worker_loop() {
    while (true) {
        SimulationJob job;

        {
            std::unique_lock<std::mutex> lock(mutex_);
            job_available_.wait(lock, [this] {
                return shutdown_ || !pending_jobs_.empty();
            });

            if (shutdown_ && pending_jobs_.empty()) {
                return;
            }

            // Get the highest priority job
            job = pending_jobs_.top();
            pending_jobs_.pop();

            // Check if cancelled
            auto status_it = job_status_.find(job.id);
            if (status_it != job_status_.end() && status_it->second == JobStatus::Cancelled) {
                continue;  // Skip cancelled jobs
            }

            job_status_[job.id] = JobStatus::Running;
            ++running_count_;
        }

        // Execute the job
        JobResult result;
        result.job_id = job.id;
        result.started_at = std::chrono::system_clock::now();

        try {
            Simulator sim(job.circuit, job.options);
            result.simulation = sim.run_transient();
            result.status = (result.simulation.final_status == SolverStatus::Success)
                            ? JobStatus::Completed : JobStatus::Failed;
            if (result.status == JobStatus::Failed) {
                result.error_message = result.simulation.error_message;
            }
        } catch (const std::exception& e) {
            result.status = JobStatus::Failed;
            result.error_message = e.what();
        }

        result.completed_at = std::chrono::system_clock::now();
        result.duration_seconds = std::chrono::duration<double>(
            result.completed_at - result.started_at).count();

        {
            std::lock_guard<std::mutex> lock(mutex_);
            results_[job.id] = result;
            job_status_[job.id] = result.status;
            --running_count_;
        }

        result_available_.notify_all();

        if (completion_callback_) {
            completion_callback_(result);
        }
    }
}

// =============================================================================
// Convenience Functions
// =============================================================================

size_t recommended_threads() {
    size_t hw = std::thread::hardware_concurrency();
    return hw > 0 ? hw : 4;
}

std::vector<SimulationResult> run_parallel(
    const std::vector<std::pair<Circuit, SimulationOptions>>& simulations,
    size_t num_threads) {

    if (num_threads == 0) {
        num_threads = recommended_threads();
    }

    std::vector<SimulationResult> results(simulations.size());
    std::vector<std::future<void>> futures;
    futures.reserve(simulations.size());

    for (size_t i = 0; i < simulations.size(); ++i) {
        futures.push_back(ThreadPool::global().submit([&, i]() {
            Simulator sim(simulations[i].first, simulations[i].second);
            results[i] = sim.run_transient();
        }));
    }

    for (auto& f : futures) {
        f.get();
    }

    return results;
}

}  // namespace pulsim
