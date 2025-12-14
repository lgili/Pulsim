#pragma once

// =============================================================================
// PulsimCore v2 - High-Performance Components (Phase 4)
// =============================================================================
// This header provides:
// - 4.1: Enhanced linear solver policies with caching and reuse
// - 4.2: Advanced convergence policies (Armijo, Trust Region)
// - 4.3: Memory optimization (Arena allocator, memory pools)
// - 4.4: SIMD detection and optimization helpers
// - 4.5: Cache-friendly SoA data layouts
// =============================================================================

#include "pulsim/v1/numeric_types.hpp"
#include "pulsim/v1/solver.hpp"
#include <Eigen/SparseLU>
#include <memory>
#include <new>
#include <cstdlib>
#include <cstring>
#include <bit>

namespace pulsim::v1 {

// =============================================================================
// 4.1.4-4.1.6: Enhanced SparseLU with Caching and Reuse Detection
// =============================================================================

/// Configuration for linear solver behavior
struct LinearSolverConfig {
    Real pivot_tolerance = 1e-10;     // Pivot tolerance (4.1.6)
    bool reuse_symbolic = true;       // Reuse symbolic analysis (4.1.5)
    bool detect_pattern_change = true; // Detect sparsity pattern changes (4.1.4)
    bool deterministic_pivoting = true; // Use deterministic pivot selection (4.1.8)

    [[nodiscard]] static constexpr LinearSolverConfig defaults() {
        return LinearSolverConfig{};
    }
};

/// Enhanced SparseLU policy with symbolic analysis caching (4.1.4, 4.1.5, 4.1.6)
class EnhancedSparseLUPolicy {
public:
    explicit EnhancedSparseLUPolicy(const LinearSolverConfig& config = {})
        : config_(config) {}

    // Disable copy and move (Eigen::SparseLU is not copyable/movable)
    EnhancedSparseLUPolicy(const EnhancedSparseLUPolicy&) = delete;
    EnhancedSparseLUPolicy& operator=(const EnhancedSparseLUPolicy&) = delete;
    EnhancedSparseLUPolicy(EnhancedSparseLUPolicy&&) = delete;
    EnhancedSparseLUPolicy& operator=(EnhancedSparseLUPolicy&&) = delete;

    bool analyze(const SparseMatrix& A) {
        // Check if pattern changed (4.1.4)
        if (config_.detect_pattern_change && analyzed_) {
            if (!pattern_matches(A)) {
                // Pattern changed, need full re-analysis
                pattern_hash_ = compute_pattern_hash(A);
            } else {
                // Pattern unchanged, can skip symbolic analysis
                return true;
            }
        }

        solver_.analyzePattern(A);
        analyzed_ = true;
        pattern_hash_ = compute_pattern_hash(A);
        factorize_count_ = 0;
        return true;
    }

    bool factorize(const SparseMatrix& A) {
        // Reuse symbolic analysis if pattern unchanged (4.1.5)
        if (config_.reuse_symbolic && analyzed_ && pattern_matches(A)) {
            // Skip analyze, just factorize
        } else {
            analyze(A);
        }

        solver_.factorize(A);
        singular_ = (solver_.info() != Eigen::Success);
        factorized_ = !singular_;
        ++factorize_count_;
        return factorized_;
    }

    [[nodiscard]] LinearSolveResult solve(const Vector& b) {
        if (!factorized_) {
            return LinearSolveResult::failure("Matrix not factorized");
        }

        Vector x = solver_.solve(b);

        if (solver_.info() != Eigen::Success) {
            return LinearSolveResult::failure("Linear solve failed");
        }

        return LinearSolveResult::success(std::move(x));
    }

    [[nodiscard]] bool is_singular() const { return singular_; }

    // Statistics
    [[nodiscard]] std::size_t factorize_count() const { return factorize_count_; }
    [[nodiscard]] bool is_analyzed() const { return analyzed_; }

    void reset() {
        analyzed_ = false;
        factorized_ = false;
        singular_ = false;
        factorize_count_ = 0;
        pattern_hash_ = 0;
    }

    [[nodiscard]] const LinearSolverConfig& config() const { return config_; }
    void set_config(const LinearSolverConfig& cfg) { config_ = cfg; }

private:
    LinearSolverConfig config_;
    Eigen::SparseLU<SparseMatrix> solver_;
    bool analyzed_ = false;
    bool factorized_ = false;
    bool singular_ = false;
    std::size_t factorize_count_ = 0;
    std::size_t pattern_hash_ = 0;

    // Compute hash of sparsity pattern for change detection
    [[nodiscard]] std::size_t compute_pattern_hash(const SparseMatrix& A) const {
        std::size_t hash = A.rows() ^ (A.cols() << 16) ^ (A.nonZeros() << 32);

        // Hash the outer index pointer array
        const auto* outer = A.outerIndexPtr();
        for (Index i = 0; i <= A.outerSize(); ++i) {
            hash ^= static_cast<std::size_t>(outer[i]) << (i % 48);
        }

        // Hash sample of inner indices for faster comparison
        const auto* inner = A.innerIndexPtr();
        std::size_t step = std::max(std::size_t(1), static_cast<std::size_t>(A.nonZeros()) / 100);
        for (std::size_t i = 0; i < static_cast<std::size_t>(A.nonZeros()); i += step) {
            hash ^= static_cast<std::size_t>(inner[i]) << ((i / step) % 48);
        }

        return hash;
    }

    [[nodiscard]] bool pattern_matches(const SparseMatrix& A) const {
        return compute_pattern_hash(A) == pattern_hash_;
    }
};

// =============================================================================
// 4.1.3: KLU Policy (SuiteSparse KLU integration)
// =============================================================================

#ifdef PULSIM_HAS_KLU
#include <klu.h>
#endif

/// KLU solver policy - high-performance sparse LU for circuit simulation
/// Enable with PULSIM_HAS_KLU define (automatically set when SuiteSparse is found)
class KLUPolicy {
public:
    explicit KLUPolicy([[maybe_unused]] const LinearSolverConfig& config = {})
        : config_(config) {
#ifdef PULSIM_HAS_KLU
        klu_defaults(&klu_common_);
        klu_common_.btf = 1;  // Enable block triangular form
        klu_common_.scale = 1; // Scale matrix for better conditioning
#endif
    }

    ~KLUPolicy() {
#ifdef PULSIM_HAS_KLU
        cleanup();
#endif
    }

    // Disable copy (solvers contain non-copyable state)
    KLUPolicy(const KLUPolicy&) = delete;
    KLUPolicy& operator=(const KLUPolicy&) = delete;

    // Disable move as well (Eigen::SparseLU cannot be moved, and this simplifies lifetime management)
    KLUPolicy(KLUPolicy&&) = delete;
    KLUPolicy& operator=(KLUPolicy&&) = delete;

    bool analyze(const SparseMatrix& A) {
#ifdef PULSIM_HAS_KLU
        cleanup();
        n_ = static_cast<int>(A.rows());

        // Copy sparse matrix structure (assume input is already compressed)
        // If not compressed, we need to work with a copy
        if (!A.isCompressed()) {
            SparseMatrix A_compressed = A;
            A_compressed.makeCompressed();
            Ap_.assign(A_compressed.outerIndexPtr(), A_compressed.outerIndexPtr() + A_compressed.outerSize() + 1);
            Ai_.assign(A_compressed.innerIndexPtr(), A_compressed.innerIndexPtr() + A_compressed.nonZeros());
            Ax_.resize(static_cast<size_t>(A_compressed.nonZeros()));
        } else {
            Ap_.assign(A.outerIndexPtr(), A.outerIndexPtr() + A.outerSize() + 1);
            Ai_.assign(A.innerIndexPtr(), A.innerIndexPtr() + A.nonZeros());
            Ax_.resize(static_cast<size_t>(A.nonZeros()));
        }

        klu_symbolic_ = klu_analyze(n_, Ap_.data(), Ai_.data(), &klu_common_);
        return klu_symbolic_ != nullptr;
#else
        return fallback_.analyze(A);
#endif
    }

    bool factorize(const SparseMatrix& A) {
#ifdef PULSIM_HAS_KLU
        if (!klu_symbolic_) {
            if (!analyze(A)) return false;
        }

        // Copy values
        const Real* values = A.valuePtr();
        for (size_t i = 0; i < Ax_.size(); ++i) {
            Ax_[i] = values[i];
        }

        // Refactorize if we already have numeric factorization
        if (klu_numeric_) {
            int ok = klu_refactor(Ap_.data(), Ai_.data(), Ax_.data(),
                                   klu_symbolic_, klu_numeric_, &klu_common_);
            if (ok == 0) {
                // Refactor failed, try full factorization
                klu_free_numeric(&klu_numeric_, &klu_common_);
                klu_numeric_ = nullptr;
            } else {
                return true;
            }
        }

        // Full numeric factorization
        klu_numeric_ = klu_factor(Ap_.data(), Ai_.data(), Ax_.data(),
                                   klu_symbolic_, &klu_common_);
        return klu_numeric_ != nullptr;
#else
        return fallback_.factorize(A);
#endif
    }

    [[nodiscard]] LinearSolveResult solve(const Vector& b) {
#ifdef PULSIM_HAS_KLU
        if (!klu_symbolic_ || !klu_numeric_) {
            return LinearSolveResult::failure("KLU not factorized");
        }

        Vector x = b;  // KLU solves in-place
        int ok = klu_solve(klu_symbolic_, klu_numeric_, n_, 1,
                           x.data(), &klu_common_);
        if (ok == 0) {
            return LinearSolveResult::failure("KLU solve failed");
        }
        return LinearSolveResult::success(std::move(x));
#else
        return fallback_.solve(b);
#endif
    }

    [[nodiscard]] bool is_singular() const {
#ifdef PULSIM_HAS_KLU
        if (!klu_numeric_) return true;
        // Check reciprocal condition number
        klu_rcond(const_cast<klu_symbolic*>(klu_symbolic_),
                  const_cast<klu_numeric*>(klu_numeric_),
                  const_cast<klu_common*>(&klu_common_));
        return klu_common_.rcond < 1e-14;
#else
        return fallback_.is_singular();
#endif
    }

    [[nodiscard]] static bool is_available() {
#ifdef PULSIM_HAS_KLU
        return true;
#else
        return false;
#endif
    }

private:
#ifdef PULSIM_HAS_KLU
    void cleanup() {
        if (klu_numeric_) {
            klu_free_numeric(&klu_numeric_, &klu_common_);
            klu_numeric_ = nullptr;
        }
        if (klu_symbolic_) {
            klu_free_symbolic(&klu_symbolic_, &klu_common_);
            klu_symbolic_ = nullptr;
        }
    }

    LinearSolverConfig config_;
    klu_common klu_common_{};
    klu_symbolic* klu_symbolic_ = nullptr;
    klu_numeric* klu_numeric_ = nullptr;
    int n_ = 0;
    std::vector<int> Ap_;
    std::vector<int> Ai_;
    std::vector<double> Ax_;
#else
    LinearSolverConfig config_;
    EnhancedSparseLUPolicy fallback_;
#endif
};

// =============================================================================
// 4.1.7: Linear Solver Benchmarking
// =============================================================================

/// Benchmark result for linear solver
struct LinearSolverBenchmark {
    std::string solver_name;
    std::size_t matrix_size = 0;
    std::size_t nonzeros = 0;
    double analyze_time_us = 0.0;
    double factorize_time_us = 0.0;
    double solve_time_us = 0.0;
    double total_time_us = 0.0;
    bool success = false;
};

// =============================================================================
// 4.2.3: Armijo Line Search Policy
// =============================================================================

/// Configuration for Armijo line search
struct ArmijoConfig {
    Real c1 = 1e-4;           // Sufficient decrease parameter (typically 1e-4)
    Real rho = 0.5;           // Backtracking factor
    int max_backtracks = 20;  // Maximum backtracking iterations
    Real min_step = 1e-10;    // Minimum step size

    [[nodiscard]] static constexpr ArmijoConfig defaults() {
        return ArmijoConfig{};
    }
};

/// Armijo backtracking line search (4.2.3)
class ArmijoLineSearch {
public:
    explicit ArmijoLineSearch(const ArmijoConfig& config = {})
        : config_(config) {}

    /// Perform line search
    /// @param x Current point
    /// @param dx Search direction
    /// @param f_x Function value at x
    /// @param grad_f_x Gradient at x (or approximation)
    /// @param f Function to evaluate f(x + alpha * dx)
    /// @return Step size alpha
    template<typename Func>
    [[nodiscard]] Real search(
        const Vector& x,
        const Vector& dx,
        Real f_x,
        const Vector& grad_f_x,
        Func&& f) {

        Real alpha = 1.0;
        Real directional_derivative = grad_f_x.dot(dx);

        // If not a descent direction, return small step
        if (directional_derivative >= 0) {
            return config_.min_step;
        }

        for (int i = 0; i < config_.max_backtracks; ++i) {
            Vector x_new = x + alpha * dx;
            Real f_new = f(x_new);

            // Armijo condition: f(x + alpha*dx) <= f(x) + c1 * alpha * grad_f^T * dx
            if (f_new <= f_x + config_.c1 * alpha * directional_derivative) {
                backtracks_ = i;
                return alpha;
            }

            // Backtrack
            alpha *= config_.rho;

            if (alpha < config_.min_step) {
                backtracks_ = i + 1;
                return config_.min_step;
            }
        }

        backtracks_ = config_.max_backtracks;
        return alpha;
    }

    /// Simplified search using residual norm
    template<typename ResidualFunc>
    [[nodiscard]] Real search_residual(
        const Vector& x,
        const Vector& dx,
        Real residual_norm,
        ResidualFunc&& residual_func) {

        Real alpha = 1.0;

        for (int i = 0; i < config_.max_backtracks; ++i) {
            Vector x_new = x + alpha * dx;
            Real new_residual = residual_func(x_new);

            // Accept if residual decreased
            if (new_residual < residual_norm) {
                backtracks_ = i;
                return alpha;
            }

            alpha *= config_.rho;

            if (alpha < config_.min_step) {
                backtracks_ = i + 1;
                return config_.min_step;
            }
        }

        backtracks_ = config_.max_backtracks;
        return alpha;
    }

    [[nodiscard]] int last_backtracks() const { return backtracks_; }
    [[nodiscard]] const ArmijoConfig& config() const { return config_; }
    void set_config(const ArmijoConfig& cfg) { config_ = cfg; }

private:
    ArmijoConfig config_;
    int backtracks_ = 0;
};

// =============================================================================
// 4.2.4: Trust Region Policy
// =============================================================================

/// Configuration for trust region method
struct TrustRegionConfig {
    Real initial_radius = 1.0;    // Initial trust region radius
    Real max_radius = 100.0;      // Maximum radius
    Real min_radius = 1e-10;      // Minimum radius
    Real eta1 = 0.25;             // Accept threshold
    Real eta2 = 0.75;             // Good step threshold
    Real gamma1 = 0.25;           // Radius shrink factor
    Real gamma2 = 2.0;            // Radius grow factor

    [[nodiscard]] static constexpr TrustRegionConfig defaults() {
        return TrustRegionConfig{};
    }
};

/// Trust region result
struct TrustRegionResult {
    Vector step;
    Real predicted_reduction = 0.0;
    Real actual_reduction = 0.0;
    Real ratio = 0.0;
    bool accepted = false;
    Real new_radius = 0.0;
};

/// Trust region method for Newton solver (4.2.4)
class TrustRegionMethod {
public:
    explicit TrustRegionMethod(const TrustRegionConfig& config = {})
        : config_(config), radius_(config.initial_radius) {}

    /// Compute trust region step (Cauchy point or dogleg)
    /// @param grad Gradient (or negative residual)
    /// @param newton_step Full Newton step
    /// @return Constrained step within trust region
    [[nodiscard]] Vector compute_step(const Vector& grad, const Vector& newton_step) {
        Real newton_norm = newton_step.norm();

        // If Newton step is within trust region, use it
        if (newton_norm <= radius_) {
            return newton_step;
        }

        // Cauchy point: steepest descent step
        Real grad_norm = grad.norm();
        if (grad_norm < 1e-15) {
            return newton_step * (radius_ / newton_norm);
        }

        // Compute Cauchy point length
        Real cauchy_length = grad_norm * grad_norm / (grad.dot(newton_step) / newton_norm * grad_norm);
        cauchy_length = std::min(cauchy_length, radius_);

        Vector cauchy_step = -cauchy_length * grad / grad_norm;

        // If Cauchy point is at boundary, use it
        if (cauchy_step.norm() >= radius_ * 0.99) {
            return cauchy_step;
        }

        // Dogleg: interpolate between Cauchy and Newton
        Vector diff = newton_step - cauchy_step;
        Real a = diff.squaredNorm();
        Real b = 2.0 * cauchy_step.dot(diff);
        Real c = cauchy_step.squaredNorm() - radius_ * radius_;

        Real discriminant = b * b - 4.0 * a * c;
        if (discriminant < 0 || a < 1e-15) {
            return cauchy_step;
        }

        Real tau = (-b + std::sqrt(discriminant)) / (2.0 * a);
        tau = std::clamp(tau, 0.0, 1.0);

        return cauchy_step + tau * diff;
    }

    /// Update trust region based on actual vs predicted reduction
    TrustRegionResult update(
        Real f_old,
        Real f_new,
        Real predicted_reduction,
        const Vector& step) {

        TrustRegionResult result;
        result.step = step;
        result.predicted_reduction = predicted_reduction;
        result.actual_reduction = f_old - f_new;

        // Compute ratio
        if (std::abs(predicted_reduction) < 1e-15) {
            result.ratio = (result.actual_reduction >= 0) ? 1.0 : 0.0;
        } else {
            result.ratio = result.actual_reduction / predicted_reduction;
        }

        // Accept/reject step
        result.accepted = (result.ratio >= config_.eta1);

        // Update radius
        if (result.ratio < config_.eta1) {
            // Poor step, shrink radius
            radius_ *= config_.gamma1;
        } else if (result.ratio > config_.eta2 && step.norm() >= radius_ * 0.99) {
            // Good step at boundary, expand radius
            radius_ = std::min(config_.gamma2 * radius_, config_.max_radius);
        }

        // Enforce minimum radius
        radius_ = std::max(radius_, config_.min_radius);
        result.new_radius = radius_;

        return result;
    }

    [[nodiscard]] Real radius() const { return radius_; }
    void set_radius(Real r) { radius_ = std::clamp(r, config_.min_radius, config_.max_radius); }
    void reset() { radius_ = config_.initial_radius; }

    [[nodiscard]] const TrustRegionConfig& config() const { return config_; }
    void set_config(const TrustRegionConfig& cfg) { config_ = cfg; }

private:
    TrustRegionConfig config_;
    Real radius_;
};

// =============================================================================
// 4.3.1-4.3.2: Arena Allocator and Memory Pool
// =============================================================================

/// Simple arena allocator with bump allocation (4.3.1)
class ArenaAllocator {
public:
    explicit ArenaAllocator(std::size_t initial_size = 1024 * 1024)  // 1MB default
        : block_size_(initial_size) {
        allocate_block(initial_size);
    }

    ~ArenaAllocator() {
        for (auto* block : blocks_) {
            std::free(block);
        }
    }

    // Non-copyable, non-movable
    ArenaAllocator(const ArenaAllocator&) = delete;
    ArenaAllocator& operator=(const ArenaAllocator&) = delete;
    ArenaAllocator(ArenaAllocator&&) = delete;
    ArenaAllocator& operator=(ArenaAllocator&&) = delete;

    /// Allocate memory with specified alignment (4.3.3)
    [[nodiscard]] void* allocate(std::size_t size, std::size_t alignment = alignof(std::max_align_t)) {
        // Align current position
        std::size_t aligned_pos = (current_pos_ + alignment - 1) & ~(alignment - 1);

        if (aligned_pos + size > current_block_size_) {
            // Need new block
            std::size_t new_block_size = std::max(block_size_, size + alignment);
            allocate_block(new_block_size);
            aligned_pos = (current_pos_ + alignment - 1) & ~(alignment - 1);
        }

        void* ptr = current_block_ + aligned_pos;
        current_pos_ = aligned_pos + size;
        total_allocated_ += size;

#ifdef PULSIM_DEBUG_ALLOCATOR
        // Poison guard bytes (4.3.7)
        if (size >= 8) {
            std::memset(ptr, 0xCD, size);  // Uninitialized marker
        }
#endif

        return ptr;
    }

    /// Allocate and construct object
    template<typename T, typename... Args>
    [[nodiscard]] T* create(Args&&... args) {
        void* ptr = allocate(sizeof(T), alignof(T));
        return new (ptr) T(std::forward<Args>(args)...);
    }

    /// Allocate array with cache line alignment (4.3.2)
    template<typename T>
    [[nodiscard]] T* allocate_array(std::size_t count, std::size_t alignment = 64) {
        return static_cast<T*>(allocate(count * sizeof(T), alignment));
    }

    /// Reset arena (reuse memory without deallocating)
    void reset() {
        if (!blocks_.empty()) {
            current_block_ = blocks_[0];
            current_block_size_ = block_sizes_[0];
        }
        current_pos_ = 0;
        total_allocated_ = 0;
    }

    /// Clear arena (deallocate all but first block)
    void clear() {
        for (std::size_t i = 1; i < blocks_.size(); ++i) {
            std::free(blocks_[i]);
        }
        if (!blocks_.empty()) {
            blocks_.resize(1);
            block_sizes_.resize(1);
            current_block_ = blocks_[0];
            current_block_size_ = block_sizes_[0];
        }
        current_pos_ = 0;
        total_allocated_ = 0;
    }

    // Statistics (4.3.6)
    [[nodiscard]] std::size_t total_allocated() const { return total_allocated_; }
    [[nodiscard]] std::size_t block_count() const { return blocks_.size(); }
    [[nodiscard]] std::size_t total_capacity() const {
        std::size_t total = 0;
        for (auto size : block_sizes_) total += size;
        return total;
    }

private:
    void allocate_block(std::size_t size) {
        // Round size up to multiple of alignment (required by std::aligned_alloc)
        constexpr std::size_t alignment = 64;
        std::size_t aligned_size = (size + alignment - 1) & ~(alignment - 1);

        // Use aligned allocation for SIMD compatibility (4.3.3)
        void* block = std::aligned_alloc(alignment, aligned_size);
        if (!block) {
            throw std::bad_alloc();
        }

#ifdef PULSIM_DEBUG_ALLOCATOR
        // Poison entire block (4.3.7)
        std::memset(block, 0xAB, size);  // Dead memory marker
#endif

        blocks_.push_back(static_cast<char*>(block));
        block_sizes_.push_back(size);
        current_block_ = static_cast<char*>(block);
        current_block_size_ = size;
        current_pos_ = 0;
    }

    std::size_t block_size_;
    std::vector<char*> blocks_;
    std::vector<std::size_t> block_sizes_;
    char* current_block_ = nullptr;
    std::size_t current_block_size_ = 0;
    std::size_t current_pos_ = 0;
    std::size_t total_allocated_ = 0;
};

/// Per-simulation memory pool for workspace reuse (4.3.4)
class SimulationMemoryPool {
public:
    explicit SimulationMemoryPool(std::size_t estimated_size = 1024 * 1024)
        : arena_(estimated_size) {}

    /// Get workspace vector (reuses allocation across timesteps)
    [[nodiscard]] Vector& get_workspace_vector(std::size_t size, std::size_t id = 0) {
        auto key = std::make_pair(size, id);
        auto it = workspace_vectors_.find(key);
        if (it != workspace_vectors_.end()) {
            return it->second;
        }
        auto [inserted, _] = workspace_vectors_.emplace(key, Vector(size));
        return inserted->second;
    }

    /// Get workspace matrix (reuses allocation)
    [[nodiscard]] SparseMatrix& get_workspace_matrix(Index rows, Index cols, std::size_t id = 0) {
        auto key = std::make_tuple(rows, cols, id);
        auto it = workspace_matrices_.find(key);
        if (it != workspace_matrices_.end()) {
            return it->second;
        }
        auto [inserted, _] = workspace_matrices_.emplace(key, SparseMatrix(rows, cols));
        return inserted->second;
    }

    /// Reset for new timestep (keeps allocations, clears data)
    void reset_timestep() {
        // Vectors and matrices keep their allocations
        arena_.reset();
    }

    /// Clear all workspaces
    void clear() {
        workspace_vectors_.clear();
        workspace_matrices_.clear();
        arena_.clear();
    }

    [[nodiscard]] ArenaAllocator& arena() { return arena_; }

    // Statistics (4.3.6)
    [[nodiscard]] std::size_t vector_count() const { return workspace_vectors_.size(); }
    [[nodiscard]] std::size_t matrix_count() const { return workspace_matrices_.size(); }
    [[nodiscard]] std::size_t arena_allocated() const { return arena_.total_allocated(); }

private:
    ArenaAllocator arena_;
    std::map<std::pair<std::size_t, std::size_t>, Vector> workspace_vectors_;
    std::map<std::tuple<Index, Index, std::size_t>, SparseMatrix> workspace_matrices_;
};

// =============================================================================
// 4.3.6: Memory Usage Tracking
// =============================================================================

/// Memory usage statistics
struct MemoryStats {
    std::size_t peak_allocated = 0;
    std::size_t current_allocated = 0;
    std::size_t allocation_count = 0;
    std::size_t deallocation_count = 0;
    std::size_t arena_allocated = 0;
    std::size_t workspace_vectors = 0;
    std::size_t workspace_matrices = 0;
};

/// Memory tracker (4.3.6)
class MemoryTracker {
public:
    static MemoryTracker& instance() {
        static MemoryTracker tracker;
        return tracker;
    }

    void record_allocation(std::size_t size) {
        current_allocated_ += size;
        peak_allocated_ = std::max(peak_allocated_, current_allocated_);
        ++allocation_count_;
    }

    void record_deallocation(std::size_t size) {
        current_allocated_ -= std::min(current_allocated_, size);
        ++deallocation_count_;
    }

    [[nodiscard]] MemoryStats stats() const {
        MemoryStats s;
        s.peak_allocated = peak_allocated_;
        s.current_allocated = current_allocated_;
        s.allocation_count = allocation_count_;
        s.deallocation_count = deallocation_count_;
        return s;
    }

    void reset() {
        peak_allocated_ = 0;
        current_allocated_ = 0;
        allocation_count_ = 0;
        deallocation_count_ = 0;
    }

private:
    MemoryTracker() = default;
    std::size_t peak_allocated_ = 0;
    std::size_t current_allocated_ = 0;
    std::size_t allocation_count_ = 0;
    std::size_t deallocation_count_ = 0;
};

// =============================================================================
// 4.4.1: SIMD Capability Detection
// =============================================================================

/// SIMD instruction set levels
enum class SIMDLevel {
    None,
    SSE2,
    SSE4,
    AVX,
    AVX2,
    AVX512,
    NEON  // ARM
};

/// Detect SIMD capabilities at compile time (4.4.1)
[[nodiscard]] constexpr SIMDLevel detect_simd_level() noexcept {
#if defined(__AVX512F__)
    return SIMDLevel::AVX512;
#elif defined(__AVX2__)
    return SIMDLevel::AVX2;
#elif defined(__AVX__)
    return SIMDLevel::AVX;
#elif defined(__SSE4_1__) || defined(__SSE4_2__)
    return SIMDLevel::SSE4;
#elif defined(__SSE2__) || defined(_M_X64)
    return SIMDLevel::SSE2;
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
    return SIMDLevel::NEON;
#else
    return SIMDLevel::None;
#endif
}

/// Get SIMD level name
[[nodiscard]] constexpr const char* simd_level_name(SIMDLevel level) noexcept {
    switch (level) {
        case SIMDLevel::None: return "None";
        case SIMDLevel::SSE2: return "SSE2";
        case SIMDLevel::SSE4: return "SSE4";
        case SIMDLevel::AVX: return "AVX";
        case SIMDLevel::AVX2: return "AVX2";
        case SIMDLevel::AVX512: return "AVX-512";
        case SIMDLevel::NEON: return "NEON";
        default: return "Unknown";
    }
}

/// Get optimal vector width for current SIMD level
[[nodiscard]] constexpr std::size_t simd_vector_width() noexcept {
    switch (detect_simd_level()) {
        case SIMDLevel::AVX512: return 8;   // 512 bits / 64 bits
        case SIMDLevel::AVX2:
        case SIMDLevel::AVX: return 4;      // 256 bits / 64 bits
        case SIMDLevel::SSE4:
        case SIMDLevel::SSE2: return 2;     // 128 bits / 64 bits
        case SIMDLevel::NEON: return 2;     // 128 bits / 64 bits
        default: return 1;
    }
}

// Compile-time SIMD level
inline constexpr SIMDLevel current_simd_level = detect_simd_level();
inline constexpr std::size_t simd_width = simd_vector_width();

// =============================================================================
// 4.5.1-4.5.2: Structure of Arrays (SoA) Layout
// =============================================================================

/// Cache line size (typically 64 bytes)
inline constexpr std::size_t cache_line_size = 64;

/// Aligned array for SoA layout (4.5.1, 4.5.2)
template<typename T, std::size_t Alignment = cache_line_size>
class AlignedArray {
public:
    AlignedArray() = default;

    explicit AlignedArray(std::size_t size)
        : size_(size) {
        allocate(size);
    }

    ~AlignedArray() {
        deallocate();
    }

    // Move semantics
    AlignedArray(AlignedArray&& other) noexcept
        : data_(other.data_)
        , size_(other.size_)
        , capacity_(other.capacity_) {
        other.data_ = nullptr;
        other.size_ = 0;
        other.capacity_ = 0;
    }

    AlignedArray& operator=(AlignedArray&& other) noexcept {
        if (this != &other) {
            deallocate();
            data_ = other.data_;
            size_ = other.size_;
            capacity_ = other.capacity_;
            other.data_ = nullptr;
            other.size_ = 0;
            other.capacity_ = 0;
        }
        return *this;
    }

    // No copy
    AlignedArray(const AlignedArray&) = delete;
    AlignedArray& operator=(const AlignedArray&) = delete;

    void resize(std::size_t new_size) {
        if (new_size > capacity_) {
            // Round size up to multiple of alignment (required by std::aligned_alloc)
            std::size_t byte_size = new_size * sizeof(T);
            std::size_t aligned_byte_size = (byte_size + Alignment - 1) & ~(Alignment - 1);

            T* new_data = static_cast<T*>(std::aligned_alloc(Alignment, aligned_byte_size));
            if (!new_data) throw std::bad_alloc();

            if (data_) {
                std::memcpy(new_data, data_, size_ * sizeof(T));
                std::free(data_);
            }
            data_ = new_data;
            capacity_ = new_size;
        }
        size_ = new_size;
    }

    [[nodiscard]] T& operator[](std::size_t i) { return data_[i]; }
    [[nodiscard]] const T& operator[](std::size_t i) const { return data_[i]; }

    [[nodiscard]] T* data() { return data_; }
    [[nodiscard]] const T* data() const { return data_; }

    [[nodiscard]] std::size_t size() const { return size_; }
    [[nodiscard]] std::size_t capacity() const { return capacity_; }
    [[nodiscard]] bool empty() const { return size_ == 0; }

    [[nodiscard]] T* begin() { return data_; }
    [[nodiscard]] T* end() { return data_ + size_; }
    [[nodiscard]] const T* begin() const { return data_; }
    [[nodiscard]] const T* end() const { return data_ + size_; }

private:
    void allocate(std::size_t size) {
        if (size > 0) {
            // Round size up to multiple of alignment (required by std::aligned_alloc)
            std::size_t byte_size = size * sizeof(T);
            std::size_t aligned_byte_size = (byte_size + Alignment - 1) & ~(Alignment - 1);

            data_ = static_cast<T*>(std::aligned_alloc(Alignment, aligned_byte_size));
            if (!data_) throw std::bad_alloc();
            capacity_ = size;
        }
    }

    void deallocate() {
        if (data_) {
            std::free(data_);
            data_ = nullptr;
        }
        size_ = 0;
        capacity_ = 0;
    }

    T* data_ = nullptr;
    std::size_t size_ = 0;
    std::size_t capacity_ = 0;
};

/// SoA layout for device parameters (4.5.1)
/// Example: Instead of array of {R, node1, node2} structs,
/// store separate arrays for R[], node1[], node2[]
template<typename... Ts>
class SoALayout {
public:
    static constexpr std::size_t num_fields = sizeof...(Ts);

    SoALayout() = default;

    explicit SoALayout(std::size_t size) {
        resize(size);
    }

    void resize(std::size_t new_size) {
        size_ = new_size;
        resize_impl<0>(new_size);
    }

    template<std::size_t I>
    [[nodiscard]] auto& get() {
        return std::get<I>(arrays_);
    }

    template<std::size_t I>
    [[nodiscard]] const auto& get() const {
        return std::get<I>(arrays_);
    }

    [[nodiscard]] std::size_t size() const { return size_; }

private:
    template<std::size_t I>
    void resize_impl(std::size_t new_size) {
        if constexpr (I < num_fields) {
            std::get<I>(arrays_).resize(new_size);
            resize_impl<I + 1>(new_size);
        }
    }

    std::tuple<AlignedArray<Ts>...> arrays_;
    std::size_t size_ = 0;
};

/// Resistor data in SoA layout
struct ResistorSoA {
    AlignedArray<Real> resistance;
    AlignedArray<Index> node_pos;
    AlignedArray<Index> node_neg;

    void resize(std::size_t n) {
        resistance.resize(n);
        node_pos.resize(n);
        node_neg.resize(n);
    }

    [[nodiscard]] std::size_t size() const { return resistance.size(); }
};

/// Capacitor data in SoA layout
struct CapacitorSoA {
    AlignedArray<Real> capacitance;
    AlignedArray<Real> voltage;
    AlignedArray<Real> voltage_prev;
    AlignedArray<Real> current;
    AlignedArray<Real> current_prev;
    AlignedArray<Index> node_pos;
    AlignedArray<Index> node_neg;

    void resize(std::size_t n) {
        capacitance.resize(n);
        voltage.resize(n);
        voltage_prev.resize(n);
        current.resize(n);
        current_prev.resize(n);
        node_pos.resize(n);
        node_neg.resize(n);
    }

    [[nodiscard]] std::size_t size() const { return capacitance.size(); }
};

/// Inductor data in SoA layout
struct InductorSoA {
    AlignedArray<Real> inductance;
    AlignedArray<Real> current;
    AlignedArray<Real> current_prev;
    AlignedArray<Real> voltage;
    AlignedArray<Real> voltage_prev;
    AlignedArray<Index> node_pos;
    AlignedArray<Index> node_neg;
    AlignedArray<Index> branch_index;

    void resize(std::size_t n) {
        inductance.resize(n);
        current.resize(n);
        current_prev.resize(n);
        voltage.resize(n);
        voltage_prev.resize(n);
        node_pos.resize(n);
        node_neg.resize(n);
        branch_index.resize(n);
    }

    [[nodiscard]] std::size_t size() const { return inductance.size(); }
};

// =============================================================================
// Static Assertions
// =============================================================================

static_assert(LinearSolverPolicy<EnhancedSparseLUPolicy>);
static_assert(LinearSolverPolicy<KLUPolicy>);

} // namespace pulsim::v1
