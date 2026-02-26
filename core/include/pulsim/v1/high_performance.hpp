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
#include <Eigen/SparseCholesky>
#include <Eigen/IterativeLinearSolvers>
#include <unsupported/Eigen/IterativeSolvers>
#ifdef PULSIM_HAS_HYPRE
#include <mpi.h>
#include <HYPRE.h>
#include <HYPRE_IJ_mv.h>
#include <HYPRE_parcsr_ls.h>
#endif
#include <algorithm>
#include <chrono>
#include <memory>
#include <new>
#include <cstdlib>
#include <cstring>
#include <bit>
#include <optional>
#include <utility>
#include <vector>
#ifdef _MSC_VER
#include <malloc.h>
#endif

namespace pulsim::v1 {

namespace detail {

[[nodiscard]] inline void* aligned_allocate(std::size_t alignment, std::size_t size) {
    const std::size_t aligned_size = (size + alignment - 1) & ~(alignment - 1);
#if defined(_MSC_VER)
    return _aligned_malloc(aligned_size, alignment);
#else
    return std::aligned_alloc(alignment, aligned_size);
#endif
}

inline void aligned_deallocate(void* ptr) noexcept {
#if defined(_MSC_VER)
    _aligned_free(ptr);
#else
    std::free(ptr);
#endif
}

} // namespace detail

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

struct IterativeSolverConfig {
    int max_iterations = 200;
    Real tolerance = 1e-10;
    int restart = 30;  // GMRES restart

    enum class PreconditionerKind {
        None,
        Jacobi,
        ILU0,
        ILUT,
        AMG
    };

    PreconditionerKind preconditioner = PreconditionerKind::Jacobi;
    bool enable_scaling = false;
    Real scaling_floor = 1e-12;
    Real ilut_drop_tolerance = 1e-4;
    Real ilut_fill_factor = 10.0;

    [[nodiscard]] static constexpr IterativeSolverConfig defaults() {
        return IterativeSolverConfig{};
    }
};

// =============================================================================
// Runtime Linear Solver Selection
// =============================================================================

enum class LinearSolverKind {
    SparseLU,
    EnhancedSparseLU,
    KLU,
    GMRES,
    BiCGSTAB,
    CG
};

struct LinearSolverTelemetry {
    int total_solve_calls = 0;
    int total_analyze_calls = 0;
    int total_factorize_calls = 0;
    int total_iterations = 0;
    int total_fallbacks = 0;
    int last_iterations = 0;
    Real last_error = 0.0;
    double total_analyze_time_seconds = 0.0;
    double total_factorize_time_seconds = 0.0;
    double total_solve_time_seconds = 0.0;
    double last_analyze_time_seconds = 0.0;
    double last_factorize_time_seconds = 0.0;
    double last_solve_time_seconds = 0.0;
    std::optional<LinearSolverKind> last_solver;
    std::optional<IterativeSolverConfig::PreconditionerKind> last_preconditioner;
};

struct LinearSolverStackConfig {
    std::vector<LinearSolverKind> order { LinearSolverKind::SparseLU };
    std::vector<LinearSolverKind> fallback_order {};
    LinearSolverConfig direct_config = LinearSolverConfig::defaults();
    IterativeSolverConfig iterative_config = IterativeSolverConfig::defaults();
    bool allow_fallback = true;
    bool auto_select = true;
    int size_threshold = 2000;
    int nnz_threshold = 200000;
    Real diag_min_threshold = 1e-12;

    [[nodiscard]] static LinearSolverStackConfig defaults() {
        return {};
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

        [[nodiscard]] const LinearSolverConfig& config() const { return config_; }
        void set_config(const LinearSolverConfig& config) {
        config_ = config;
    #ifndef PULSIM_HAS_KLU
        fallback_.set_config(config);
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
// Iterative Linear Solver Policies
// =============================================================================

#ifdef PULSIM_HAS_HYPRE
class HypreAMGSolver {
public:
    enum class KrylovKind {
        GMRES,
        BiCGSTAB,
        CG
    };

    HypreAMGSolver() = default;
    HypreAMGSolver(const HypreAMGSolver&) = delete;
    HypreAMGSolver& operator=(const HypreAMGSolver&) = delete;
    ~HypreAMGSolver() { destroy(); }

    bool setup(const SparseMatrix& A, const IterativeSolverConfig& config, KrylovKind kind) {
        destroy();

        if (A.rows() != A.cols()) {
            return false;
        }

        ensure_mpi_env();
        int mpi_initialized = 0;
        MPI_Initialized(&mpi_initialized);
        if (!mpi_initialized) {
            MPI_Init(nullptr, nullptr);
        }

        kind_ = kind;
        config_ = config;
        n_ = static_cast<HYPRE_Int>(A.rows());
        A_cache_ = A;

        if (n_ <= 0) {
            return false;
        }

        HYPRE_IJMatrixCreate(MPI_COMM_SELF, 0, n_ - 1, 0, n_ - 1, &ij_matrix_);
        HYPRE_IJMatrixSetObjectType(ij_matrix_, HYPRE_PARCSR);
        HYPRE_IJMatrixInitialize(ij_matrix_);

        Eigen::SparseMatrix<Scalar, Eigen::RowMajor> Arow = A;
        std::vector<HYPRE_Int> cols;
        std::vector<Scalar> vals;
        cols.reserve(static_cast<std::size_t>(Arow.nonZeros() / std::max<HYPRE_Int>(n_, 1)));
        vals.reserve(cols.capacity());

        for (HYPRE_Int i = 0; i < n_; ++i) {
            cols.clear();
            vals.clear();
            for (Eigen::SparseMatrix<Scalar, Eigen::RowMajor>::InnerIterator it(Arow, i); it; ++it) {
                cols.push_back(static_cast<HYPRE_Int>(it.col()));
                vals.push_back(static_cast<Scalar>(it.value()));
            }
            HYPRE_Int ncols = static_cast<HYPRE_Int>(cols.size());
            if (ncols > 0) {
                HYPRE_IJMatrixSetValues(ij_matrix_, 1, &ncols, &i, cols.data(), vals.data());
            }
        }

        HYPRE_IJMatrixAssemble(ij_matrix_);
        HYPRE_IJMatrixGetObject(ij_matrix_, reinterpret_cast<void**>(&par_matrix_));

        if (!create_solver()) {
            return false;
        }
        ready_ = true;
        return true;
    }

    [[nodiscard]] std::optional<Vector> solve(const Vector& b) {
        if (!ready_ || b.size() != n_) {
            return std::nullopt;
        }

        HYPRE_IJVectorCreate(MPI_COMM_SELF, 0, n_ - 1, &ij_b_);
        HYPRE_IJVectorSetObjectType(ij_b_, HYPRE_PARCSR);
        HYPRE_IJVectorInitialize(ij_b_);

        HYPRE_IJVectorCreate(MPI_COMM_SELF, 0, n_ - 1, &ij_x_);
        HYPRE_IJVectorSetObjectType(ij_x_, HYPRE_PARCSR);
        HYPRE_IJVectorInitialize(ij_x_);

        std::vector<HYPRE_Int> indices(static_cast<std::size_t>(n_));
        std::vector<Scalar> bvals(static_cast<std::size_t>(n_));
        std::vector<Scalar> xvals(static_cast<std::size_t>(n_), Scalar(0));
        for (HYPRE_Int i = 0; i < n_; ++i) {
            indices[static_cast<std::size_t>(i)] = i;
            bvals[static_cast<std::size_t>(i)] = b[i];
        }

        HYPRE_IJVectorSetValues(ij_b_, n_, indices.data(), bvals.data());
        HYPRE_IJVectorSetValues(ij_x_, n_, indices.data(), xvals.data());
        HYPRE_IJVectorAssemble(ij_b_);
        HYPRE_IJVectorAssemble(ij_x_);
        HYPRE_IJVectorGetObject(ij_b_, reinterpret_cast<void**>(&par_b_));
        HYPRE_IJVectorGetObject(ij_x_, reinterpret_cast<void**>(&par_x_));

        bool ok = solve_internal();

        HYPRE_IJVectorGetValues(ij_x_, n_, indices.data(), xvals.data());
        Vector x = Vector::Zero(n_);
        for (HYPRE_Int i = 0; i < n_; ++i) {
            x[i] = xvals[static_cast<std::size_t>(i)];
        }

        if (!ok) {
            Vector r = A_cache_ * x - b;
            Real denom = std::max<Real>(b.norm(), Real(1.0));
            if (r.norm() <= config_.tolerance * denom) {
                ok = true;
            }
        }

        destroy_vectors();
        if (!ok) {
            return std::nullopt;
        }
        return x;
    }

    [[nodiscard]] int last_iterations() const { return static_cast<int>(last_iterations_); }
    [[nodiscard]] Real last_error() const { return last_error_; }

private:
    IterativeSolverConfig config_{};
    KrylovKind kind_ = KrylovKind::GMRES;
    bool ready_ = false;
    HYPRE_Int n_ = 0;

    HYPRE_IJMatrix ij_matrix_ = nullptr;
    HYPRE_ParCSRMatrix par_matrix_ = nullptr;
    HYPRE_IJVector ij_b_ = nullptr;
    HYPRE_IJVector ij_x_ = nullptr;
    HYPRE_ParVector par_b_ = nullptr;
    HYPRE_ParVector par_x_ = nullptr;
    HYPRE_Solver solver_ = nullptr;
    HYPRE_Solver precond_ = nullptr;
    HYPRE_Int last_iterations_ = 0;
    Real last_error_ = 0.0;
    SparseMatrix A_cache_;

    static void ensure_mpi_env() {
#if defined(_WIN32)
        if (std::getenv("OMPI_MCA_btl") == nullptr) {
            _putenv_s("OMPI_MCA_btl", "self");
        }
        if (std::getenv("OMPI_MCA_btl_base_exclude") == nullptr) {
            _putenv_s("OMPI_MCA_btl_base_exclude", "tcp,openib");
        }
#else
        if (std::getenv("OMPI_MCA_btl") == nullptr) {
            setenv("OMPI_MCA_btl", "self", 0);
        }
        if (std::getenv("OMPI_MCA_btl_base_exclude") == nullptr) {
            setenv("OMPI_MCA_btl_base_exclude", "tcp,openib", 0);
        }
#endif
    }

    void destroy_vectors() {
        if (ij_b_) {
            HYPRE_IJVectorDestroy(ij_b_);
            ij_b_ = nullptr;
            par_b_ = nullptr;
        }
        if (ij_x_) {
            HYPRE_IJVectorDestroy(ij_x_);
            ij_x_ = nullptr;
            par_x_ = nullptr;
        }
    }

    void destroy() {
        destroy_vectors();
        if (solver_) {
            switch (kind_) {
                case KrylovKind::GMRES:
                    HYPRE_ParCSRGMRESDestroy(solver_);
                    break;
                case KrylovKind::BiCGSTAB:
                    HYPRE_ParCSRBiCGSTABDestroy(solver_);
                    break;
                case KrylovKind::CG:
                    HYPRE_ParCSRPCGDestroy(solver_);
                    break;
            }
            solver_ = nullptr;
        }
        if (precond_) {
            HYPRE_BoomerAMGDestroy(precond_);
            precond_ = nullptr;
        }
        if (ij_matrix_) {
            HYPRE_IJMatrixDestroy(ij_matrix_);
            ij_matrix_ = nullptr;
            par_matrix_ = nullptr;
        }
        ready_ = false;
        n_ = 0;
    }

    bool create_solver() {
        switch (kind_) {
            case KrylovKind::GMRES:
                HYPRE_ParCSRGMRESCreate(MPI_COMM_SELF, &solver_);
                HYPRE_ParCSRGMRESSetTol(solver_, config_.tolerance);
                HYPRE_ParCSRGMRESSetMaxIter(solver_, config_.max_iterations);
                {
                    HYPRE_Int kdim = static_cast<HYPRE_Int>(config_.restart);
                    if (kdim <= 0) kdim = 30;
                    if (n_ > 0 && kdim > n_) kdim = n_;
                    HYPRE_ParCSRGMRESSetKDim(solver_, kdim);
                }
                HYPRE_ParCSRGMRESSetPrintLevel(solver_, 0);
                HYPRE_ParCSRGMRESSetLogging(solver_, 0);
                break;
            case KrylovKind::BiCGSTAB:
                HYPRE_ParCSRBiCGSTABCreate(MPI_COMM_SELF, &solver_);
                HYPRE_ParCSRBiCGSTABSetTol(solver_, config_.tolerance);
                HYPRE_ParCSRBiCGSTABSetMaxIter(solver_, config_.max_iterations);
                HYPRE_ParCSRBiCGSTABSetPrintLevel(solver_, 0);
                break;
            case KrylovKind::CG:
                HYPRE_ParCSRPCGCreate(MPI_COMM_SELF, &solver_);
                HYPRE_ParCSRPCGSetTol(solver_, config_.tolerance);
                HYPRE_ParCSRPCGSetMaxIter(solver_, config_.max_iterations);
                HYPRE_ParCSRPCGSetPrintLevel(solver_, 0);
                break;
        }

        if (!solver_) {
            return false;
        }

        HYPRE_BoomerAMGCreate(&precond_);
        if (!precond_) {
            return false;
        }
        HYPRE_BoomerAMGSetTol(precond_, 0.0);
        HYPRE_BoomerAMGSetMaxIter(precond_, 1);
        HYPRE_BoomerAMGSetPrintLevel(precond_, 0);

        switch (kind_) {
            case KrylovKind::GMRES:
                HYPRE_ParCSRGMRESSetPrecond(solver_, HYPRE_BoomerAMGSolve, HYPRE_BoomerAMGSetup, precond_);
                break;
            case KrylovKind::BiCGSTAB:
                HYPRE_ParCSRBiCGSTABSetPrecond(solver_, HYPRE_BoomerAMGSolve, HYPRE_BoomerAMGSetup, precond_);
                break;
            case KrylovKind::CG:
                HYPRE_ParCSRPCGSetPrecond(solver_, HYPRE_BoomerAMGSolve, HYPRE_BoomerAMGSetup, precond_);
                break;
        }

        return true;
    }

    bool solve_internal() {
        int status = 0;
        switch (kind_) {
            case KrylovKind::GMRES:
                HYPRE_ParCSRGMRESSetup(solver_, par_matrix_, par_b_, par_x_);
                status = HYPRE_ParCSRGMRESSolve(solver_, par_matrix_, par_b_, par_x_);
                if (status == 0) {
                    HYPRE_ParCSRGMRESGetNumIterations(solver_, &last_iterations_);
                    HYPRE_Real hypre_res = 0.0;
                    HYPRE_ParCSRGMRESGetFinalRelativeResidualNorm(solver_, &hypre_res);
                    last_error_ = static_cast<Real>(hypre_res);
                }
                break;
            case KrylovKind::BiCGSTAB:
                HYPRE_ParCSRBiCGSTABSetup(solver_, par_matrix_, par_b_, par_x_);
                status = HYPRE_ParCSRBiCGSTABSolve(solver_, par_matrix_, par_b_, par_x_);
                if (status == 0) {
                    HYPRE_ParCSRBiCGSTABGetNumIterations(solver_, &last_iterations_);
                    HYPRE_Real hypre_res = 0.0;
                    HYPRE_ParCSRBiCGSTABGetFinalRelativeResidualNorm(solver_, &hypre_res);
                    last_error_ = static_cast<Real>(hypre_res);
                }
                break;
            case KrylovKind::CG:
                HYPRE_ParCSRPCGSetup(solver_, par_matrix_, par_b_, par_x_);
                status = HYPRE_ParCSRPCGSolve(solver_, par_matrix_, par_b_, par_x_);
                if (status == 0) {
                    HYPRE_ParCSRPCGGetNumIterations(solver_, &last_iterations_);
                    HYPRE_Real hypre_res = 0.0;
                    HYPRE_ParCSRPCGGetFinalRelativeResidualNorm(solver_, &hypre_res);
                    last_error_ = static_cast<Real>(hypre_res);
                }
                break;
        }
        if (status == 0) {
            return true;
        }

        if (precond_) {
            HYPRE_BoomerAMGSetTol(precond_, config_.tolerance);
            HYPRE_BoomerAMGSetMaxIter(precond_, config_.max_iterations);
            int amg_status = HYPRE_BoomerAMGSetup(precond_, par_matrix_, par_b_, par_x_);
            if (amg_status == 0) {
                amg_status = HYPRE_BoomerAMGSolve(precond_, par_matrix_, par_b_, par_x_);
            }
            if (amg_status == 0) {
                HYPRE_BoomerAMGGetNumIterations(precond_, &last_iterations_);
                HYPRE_Real hypre_res = 0.0;
                HYPRE_BoomerAMGGetFinalRelativeResidualNorm(precond_, &hypre_res);
                last_error_ = static_cast<Real>(hypre_res);
                return true;
            }
        }

        return false;
    }
};
#endif

class GMRESPolicy {
public:
    explicit GMRESPolicy(const IterativeSolverConfig& config = {})
        : config_(config) {
        configure_all();
    }

    bool analyze(const SparseMatrix& A) {
        prepare_matrix(A);
        const SparseMatrix& target = scaled_ready_ ? scaled_matrix_ : A;
        computed_ = compute_with(target);
        return computed_;
    }

    bool factorize(const SparseMatrix& A) {
        prepare_matrix(A);
        const SparseMatrix& target = scaled_ready_ ? scaled_matrix_ : A;
        computed_ = compute_with(target);
        return computed_;
    }

    [[nodiscard]] LinearSolveResult solve(const Vector& b) {
        if (!computed_) {
            return LinearSolveResult::failure("GMRES: matrix not factorized");
        }
        Vector rhs = scaled_ready_ ? (row_scale_.array() * b.array()).matrix() : b;
        auto result = solve_with(rhs);
        if (!result) {
            return LinearSolveResult::failure("GMRES solve failed");
        }
        return LinearSolveResult::success(std::move(*result));
    }

    [[nodiscard]] bool is_singular() const { return !computed_; }

    [[nodiscard]] int last_iterations() const { return last_iterations_; }
    [[nodiscard]] Real last_error() const { return last_error_; }

    void set_config(const IterativeSolverConfig& config) {
        config_ = config;
        configure_all();
        computed_ = false;
    }

private:
    IterativeSolverConfig config_;
    Eigen::GMRES<SparseMatrix, Eigen::IdentityPreconditioner> solver_identity_;
    Eigen::GMRES<SparseMatrix, Eigen::DiagonalPreconditioner<Scalar>> solver_jacobi_;
    Eigen::GMRES<SparseMatrix, Eigen::IncompleteLUT<Scalar>> solver_ilu0_;
    Eigen::GMRES<SparseMatrix, Eigen::IncompleteLUT<Scalar>> solver_ilut_;
#ifdef PULSIM_HAS_HYPRE
    HypreAMGSolver hypre_amg_;
#endif
    bool computed_ = false;
    SparseMatrix matrix_storage_;
    SparseMatrix scaled_matrix_;
    Vector row_scale_;
    bool scaled_ready_ = false;
    int last_iterations_ = 0;
    Real last_error_ = 0.0;

    void configure_all() {
        solver_identity_.setMaxIterations(config_.max_iterations);
        solver_identity_.setTolerance(config_.tolerance);
        solver_identity_.set_restart(config_.restart);
        solver_jacobi_.setMaxIterations(config_.max_iterations);
        solver_jacobi_.setTolerance(config_.tolerance);
        solver_jacobi_.set_restart(config_.restart);
        solver_ilu0_.setMaxIterations(config_.max_iterations);
        solver_ilu0_.setTolerance(config_.tolerance);
        solver_ilu0_.set_restart(config_.restart);
        solver_ilu0_.preconditioner().setDroptol(0.0);
        solver_ilu0_.preconditioner().setFillfactor(1.0);
        solver_ilut_.setMaxIterations(config_.max_iterations);
        solver_ilut_.setTolerance(config_.tolerance);
        solver_ilut_.set_restart(config_.restart);
        solver_ilut_.preconditioner().setDroptol(config_.ilut_drop_tolerance);
        solver_ilut_.preconditioner().setFillfactor(config_.ilut_fill_factor);
    }

    void prepare_matrix(const SparseMatrix& A) {
        if (!config_.enable_scaling) {
            scaled_ready_ = false;
            row_scale_.resize(0);
            return;
        }

        row_scale_ = Vector::Ones(A.rows());
        Vector max_abs = Vector::Zero(A.rows());
        for (int col = 0; col < A.outerSize(); ++col) {
            for (SparseMatrix::InnerIterator it(A, col); it; ++it) {
                const Index row = it.row();
                const Real val = std::abs(static_cast<Real>(it.value()));
                if (val > max_abs[row]) {
                    max_abs[row] = val;
                }
            }
        }

        for (Index i = 0; i < max_abs.size(); ++i) {
            if (max_abs[i] > config_.scaling_floor) {
                row_scale_[i] = Real(1.0) / max_abs[i];
            }
        }

        scaled_matrix_ = A;
        for (int col = 0; col < scaled_matrix_.outerSize(); ++col) {
            for (SparseMatrix::InnerIterator it(scaled_matrix_, col); it; ++it) {
                it.valueRef() *= row_scale_[it.row()];
            }
        }
        scaled_ready_ = true;
    }

    bool compute_with(const SparseMatrix& A) {
        // Eigen iterative solvers can retain matrix references internally.
        // Keep a stable storage copy to avoid dangling-reference crashes when callers
        // provide temporaries or short-lived assembly buffers.
        matrix_storage_ = A;
        switch (config_.preconditioner) {
            case IterativeSolverConfig::PreconditionerKind::None:
                solver_identity_.compute(matrix_storage_);
                return solver_identity_.info() == Eigen::Success;
            case IterativeSolverConfig::PreconditionerKind::Jacobi:
                solver_jacobi_.compute(matrix_storage_);
                return solver_jacobi_.info() == Eigen::Success;
            case IterativeSolverConfig::PreconditionerKind::ILU0:
                solver_ilu0_.compute(matrix_storage_);
                return solver_ilu0_.info() == Eigen::Success;
            case IterativeSolverConfig::PreconditionerKind::ILUT:
                solver_ilut_.compute(matrix_storage_);
                return solver_ilut_.info() == Eigen::Success;
            case IterativeSolverConfig::PreconditionerKind::AMG:
#ifdef PULSIM_HAS_HYPRE
                return hypre_amg_.setup(matrix_storage_, config_, HypreAMGSolver::KrylovKind::GMRES);
#else
                return false;
#endif
        }
        return false;
    }

    [[nodiscard]] std::optional<Vector> solve_with(const Vector& b) {
        switch (config_.preconditioner) {
            case IterativeSolverConfig::PreconditionerKind::None: {
                Vector x = solver_identity_.solve(b);
                last_iterations_ = solver_identity_.iterations();
                last_error_ = solver_identity_.error();
                if (solver_identity_.info() != Eigen::Success) return std::nullopt;
                return x;
            }
            case IterativeSolverConfig::PreconditionerKind::Jacobi: {
                Vector x = solver_jacobi_.solve(b);
                last_iterations_ = solver_jacobi_.iterations();
                last_error_ = solver_jacobi_.error();
                if (solver_jacobi_.info() != Eigen::Success) return std::nullopt;
                return x;
            }
            case IterativeSolverConfig::PreconditionerKind::ILU0: {
                Vector x = solver_ilu0_.solve(b);
                last_iterations_ = solver_ilu0_.iterations();
                last_error_ = solver_ilu0_.error();
                if (solver_ilu0_.info() != Eigen::Success) return std::nullopt;
                return x;
            }
            case IterativeSolverConfig::PreconditionerKind::ILUT: {
                Vector x = solver_ilut_.solve(b);
                last_iterations_ = solver_ilut_.iterations();
                last_error_ = solver_ilut_.error();
                if (solver_ilut_.info() != Eigen::Success) return std::nullopt;
                return x;
            }
            case IterativeSolverConfig::PreconditionerKind::AMG: {
#ifdef PULSIM_HAS_HYPRE
                auto x = hypre_amg_.solve(b);
                if (!x) return std::nullopt;
                last_iterations_ = hypre_amg_.last_iterations();
                last_error_ = hypre_amg_.last_error();
                return *x;
#else
                break;
#endif
            }
        }
        return std::nullopt;
    }
};

class BiCGSTABPolicy {
public:
    explicit BiCGSTABPolicy(const IterativeSolverConfig& config = {})
        : config_(config) {
        configure_all();
    }

    bool analyze(const SparseMatrix& A) {
        prepare_matrix(A);
        const SparseMatrix& target = scaled_ready_ ? scaled_matrix_ : A;
        computed_ = compute_with(target);
        return computed_;
    }

    bool factorize(const SparseMatrix& A) {
        prepare_matrix(A);
        const SparseMatrix& target = scaled_ready_ ? scaled_matrix_ : A;
        computed_ = compute_with(target);
        return computed_;
    }

    [[nodiscard]] LinearSolveResult solve(const Vector& b) {
        if (!computed_) {
            return LinearSolveResult::failure("BiCGSTAB: matrix not factorized");
        }
        Vector rhs = scaled_ready_ ? (row_scale_.array() * b.array()).matrix() : b;
        auto result = solve_with(rhs);
        if (!result) {
            return LinearSolveResult::failure("BiCGSTAB solve failed");
        }
        return LinearSolveResult::success(std::move(*result));
    }

    [[nodiscard]] bool is_singular() const { return !computed_; }

    [[nodiscard]] int last_iterations() const { return last_iterations_; }
    [[nodiscard]] Real last_error() const { return last_error_; }

    void set_config(const IterativeSolverConfig& config) {
        config_ = config;
        configure_all();
        computed_ = false;
    }

private:
    IterativeSolverConfig config_;
    Eigen::BiCGSTAB<SparseMatrix, Eigen::IdentityPreconditioner> solver_identity_;
    Eigen::BiCGSTAB<SparseMatrix, Eigen::DiagonalPreconditioner<Scalar>> solver_jacobi_;
    Eigen::BiCGSTAB<SparseMatrix, Eigen::IncompleteLUT<Scalar>> solver_ilu0_;
    Eigen::BiCGSTAB<SparseMatrix, Eigen::IncompleteLUT<Scalar>> solver_ilut_;
#ifdef PULSIM_HAS_HYPRE
    HypreAMGSolver hypre_amg_;
#endif
    bool computed_ = false;
    SparseMatrix matrix_storage_;
    SparseMatrix scaled_matrix_;
    Vector row_scale_;
    bool scaled_ready_ = false;
    int last_iterations_ = 0;
    Real last_error_ = 0.0;

    void configure_all() {
        solver_identity_.setMaxIterations(config_.max_iterations);
        solver_identity_.setTolerance(config_.tolerance);
        solver_jacobi_.setMaxIterations(config_.max_iterations);
        solver_jacobi_.setTolerance(config_.tolerance);
        solver_ilu0_.setMaxIterations(config_.max_iterations);
        solver_ilu0_.setTolerance(config_.tolerance);
        solver_ilu0_.preconditioner().setDroptol(0.0);
        solver_ilu0_.preconditioner().setFillfactor(1.0);
        solver_ilut_.setMaxIterations(config_.max_iterations);
        solver_ilut_.setTolerance(config_.tolerance);
        solver_ilut_.preconditioner().setDroptol(config_.ilut_drop_tolerance);
        solver_ilut_.preconditioner().setFillfactor(config_.ilut_fill_factor);
    }

    void prepare_matrix(const SparseMatrix& A) {
        if (!config_.enable_scaling) {
            scaled_ready_ = false;
            row_scale_.resize(0);
            return;
        }

        row_scale_ = Vector::Ones(A.rows());
        Vector max_abs = Vector::Zero(A.rows());
        for (int col = 0; col < A.outerSize(); ++col) {
            for (SparseMatrix::InnerIterator it(A, col); it; ++it) {
                const Index row = it.row();
                const Real val = std::abs(static_cast<Real>(it.value()));
                if (val > max_abs[row]) {
                    max_abs[row] = val;
                }
            }
        }

        for (Index i = 0; i < max_abs.size(); ++i) {
            if (max_abs[i] > config_.scaling_floor) {
                row_scale_[i] = Real(1.0) / max_abs[i];
            }
        }

        scaled_matrix_ = A;
        for (int col = 0; col < scaled_matrix_.outerSize(); ++col) {
            for (SparseMatrix::InnerIterator it(scaled_matrix_, col); it; ++it) {
                it.valueRef() *= row_scale_[it.row()];
            }
        }
        scaled_ready_ = true;
    }

    bool compute_with(const SparseMatrix& A) {
        matrix_storage_ = A;
        switch (config_.preconditioner) {
            case IterativeSolverConfig::PreconditionerKind::None:
                solver_identity_.compute(matrix_storage_);
                return solver_identity_.info() == Eigen::Success;
            case IterativeSolverConfig::PreconditionerKind::Jacobi:
                solver_jacobi_.compute(matrix_storage_);
                return solver_jacobi_.info() == Eigen::Success;
            case IterativeSolverConfig::PreconditionerKind::ILU0:
                solver_ilu0_.compute(matrix_storage_);
                return solver_ilu0_.info() == Eigen::Success;
            case IterativeSolverConfig::PreconditionerKind::ILUT:
                solver_ilut_.compute(matrix_storage_);
                return solver_ilut_.info() == Eigen::Success;
            case IterativeSolverConfig::PreconditionerKind::AMG:
#ifdef PULSIM_HAS_HYPRE
                return hypre_amg_.setup(matrix_storage_, config_, HypreAMGSolver::KrylovKind::BiCGSTAB);
#else
                return false;
#endif
        }
        return false;
    }

    [[nodiscard]] std::optional<Vector> solve_with(const Vector& b) {
        switch (config_.preconditioner) {
            case IterativeSolverConfig::PreconditionerKind::None: {
                Vector x = solver_identity_.solve(b);
                last_iterations_ = solver_identity_.iterations();
                last_error_ = solver_identity_.error();
                if (solver_identity_.info() != Eigen::Success) return std::nullopt;
                return x;
            }
            case IterativeSolverConfig::PreconditionerKind::Jacobi: {
                Vector x = solver_jacobi_.solve(b);
                last_iterations_ = solver_jacobi_.iterations();
                last_error_ = solver_jacobi_.error();
                if (solver_jacobi_.info() != Eigen::Success) return std::nullopt;
                return x;
            }
            case IterativeSolverConfig::PreconditionerKind::ILU0: {
                Vector x = solver_ilu0_.solve(b);
                last_iterations_ = solver_ilu0_.iterations();
                last_error_ = solver_ilu0_.error();
                if (solver_ilu0_.info() != Eigen::Success) return std::nullopt;
                return x;
            }
            case IterativeSolverConfig::PreconditionerKind::ILUT: {
                Vector x = solver_ilut_.solve(b);
                last_iterations_ = solver_ilut_.iterations();
                last_error_ = solver_ilut_.error();
                if (solver_ilut_.info() != Eigen::Success) return std::nullopt;
                return x;
            }
            case IterativeSolverConfig::PreconditionerKind::AMG: {
#ifdef PULSIM_HAS_HYPRE
                auto x = hypre_amg_.solve(b);
                if (!x) return std::nullopt;
                last_iterations_ = hypre_amg_.last_iterations();
                last_error_ = hypre_amg_.last_error();
                return *x;
#else
                break;
#endif
            }
        }
        return std::nullopt;
    }
};

class ConjugateGradientPolicy {
public:
    explicit ConjugateGradientPolicy(const IterativeSolverConfig& config = {})
        : config_(config) {
        configure_all();
    }

    bool analyze(const SparseMatrix& A) {
        prepare_matrix(A);
        const SparseMatrix& target = scaled_ready_ ? scaled_matrix_ : A;
        computed_ = compute_with(target);
        return computed_;
    }

    bool factorize(const SparseMatrix& A) {
        prepare_matrix(A);
        const SparseMatrix& target = scaled_ready_ ? scaled_matrix_ : A;
        computed_ = compute_with(target);
        return computed_;
    }

    [[nodiscard]] LinearSolveResult solve(const Vector& b) {
        if (!computed_) {
            return LinearSolveResult::failure("CG: matrix not factorized");
        }
        Vector rhs = scaled_ready_ ? (row_scale_.array() * b.array()).matrix() : b;
        auto result = solve_with(rhs);
        if (!result) {
            return LinearSolveResult::failure("CG solve failed");
        }
        return LinearSolveResult::success(std::move(*result));
    }

    [[nodiscard]] bool is_singular() const { return !computed_; }

    [[nodiscard]] int last_iterations() const { return last_iterations_; }
    [[nodiscard]] Real last_error() const { return last_error_; }

    [[nodiscard]] bool is_spd(const SparseMatrix& A) const {
        Eigen::SimplicialLLT<SparseMatrix> llt;
        llt.compute(A);
        return llt.info() == Eigen::Success;
    }

    void set_config(const IterativeSolverConfig& config) {
        config_ = config;
        configure_all();
        computed_ = false;
    }

private:
    IterativeSolverConfig config_;
    Eigen::ConjugateGradient<SparseMatrix, Eigen::Lower|Eigen::Upper,
                             Eigen::IdentityPreconditioner> solver_identity_;
    Eigen::ConjugateGradient<SparseMatrix, Eigen::Lower|Eigen::Upper,
                             Eigen::DiagonalPreconditioner<Scalar>> solver_jacobi_;
    Eigen::ConjugateGradient<SparseMatrix, Eigen::Lower|Eigen::Upper,
                             Eigen::IncompleteLUT<Scalar>> solver_ilu0_;
    Eigen::ConjugateGradient<SparseMatrix, Eigen::Lower|Eigen::Upper,
                             Eigen::IncompleteLUT<Scalar>> solver_ilut_;
#ifdef PULSIM_HAS_HYPRE
    HypreAMGSolver hypre_amg_;
#endif
    bool computed_ = false;
    SparseMatrix matrix_storage_;
    SparseMatrix scaled_matrix_;
    Vector row_scale_;
    bool scaled_ready_ = false;
    int last_iterations_ = 0;
    Real last_error_ = 0.0;

    void configure_all() {
        solver_identity_.setMaxIterations(config_.max_iterations);
        solver_identity_.setTolerance(config_.tolerance);
        solver_jacobi_.setMaxIterations(config_.max_iterations);
        solver_jacobi_.setTolerance(config_.tolerance);
        solver_ilu0_.setMaxIterations(config_.max_iterations);
        solver_ilu0_.setTolerance(config_.tolerance);
        solver_ilu0_.preconditioner().setDroptol(0.0);
        solver_ilu0_.preconditioner().setFillfactor(1.0);
        solver_ilut_.setMaxIterations(config_.max_iterations);
        solver_ilut_.setTolerance(config_.tolerance);
        solver_ilut_.preconditioner().setDroptol(config_.ilut_drop_tolerance);
        solver_ilut_.preconditioner().setFillfactor(config_.ilut_fill_factor);
    }

    void prepare_matrix(const SparseMatrix& A) {
        if (!config_.enable_scaling) {
            scaled_ready_ = false;
            row_scale_.resize(0);
            return;
        }

        row_scale_ = Vector::Ones(A.rows());
        Vector max_abs = Vector::Zero(A.rows());
        for (int col = 0; col < A.outerSize(); ++col) {
            for (SparseMatrix::InnerIterator it(A, col); it; ++it) {
                const Index row = it.row();
                const Real val = std::abs(static_cast<Real>(it.value()));
                if (val > max_abs[row]) {
                    max_abs[row] = val;
                }
            }
        }

        for (Index i = 0; i < max_abs.size(); ++i) {
            if (max_abs[i] > config_.scaling_floor) {
                row_scale_[i] = Real(1.0) / max_abs[i];
            }
        }

        scaled_matrix_ = A;
        for (int col = 0; col < scaled_matrix_.outerSize(); ++col) {
            for (SparseMatrix::InnerIterator it(scaled_matrix_, col); it; ++it) {
                it.valueRef() *= row_scale_[it.row()];
            }
        }
        scaled_ready_ = true;
    }

    bool compute_with(const SparseMatrix& A) {
        matrix_storage_ = A;
        switch (config_.preconditioner) {
            case IterativeSolverConfig::PreconditionerKind::None:
                solver_identity_.compute(matrix_storage_);
                return solver_identity_.info() == Eigen::Success;
            case IterativeSolverConfig::PreconditionerKind::Jacobi:
                solver_jacobi_.compute(matrix_storage_);
                return solver_jacobi_.info() == Eigen::Success;
            case IterativeSolverConfig::PreconditionerKind::ILU0:
                solver_ilu0_.compute(matrix_storage_);
                return solver_ilu0_.info() == Eigen::Success;
            case IterativeSolverConfig::PreconditionerKind::ILUT:
                solver_ilut_.compute(matrix_storage_);
                return solver_ilut_.info() == Eigen::Success;
            case IterativeSolverConfig::PreconditionerKind::AMG:
#ifdef PULSIM_HAS_HYPRE
                return hypre_amg_.setup(matrix_storage_, config_, HypreAMGSolver::KrylovKind::CG);
#else
                return false;
#endif
        }
        return false;
    }

    [[nodiscard]] std::optional<Vector> solve_with(const Vector& b) {
        switch (config_.preconditioner) {
            case IterativeSolverConfig::PreconditionerKind::None: {
                Vector x = solver_identity_.solve(b);
                last_iterations_ = solver_identity_.iterations();
                last_error_ = solver_identity_.error();
                if (solver_identity_.info() != Eigen::Success) return std::nullopt;
                return x;
            }
            case IterativeSolverConfig::PreconditionerKind::Jacobi: {
                Vector x = solver_jacobi_.solve(b);
                last_iterations_ = solver_jacobi_.iterations();
                last_error_ = solver_jacobi_.error();
                if (solver_jacobi_.info() != Eigen::Success) return std::nullopt;
                return x;
            }
            case IterativeSolverConfig::PreconditionerKind::ILU0: {
                Vector x = solver_ilu0_.solve(b);
                last_iterations_ = solver_ilu0_.iterations();
                last_error_ = solver_ilu0_.error();
                if (solver_ilu0_.info() != Eigen::Success) return std::nullopt;
                return x;
            }
            case IterativeSolverConfig::PreconditionerKind::ILUT: {
                Vector x = solver_ilut_.solve(b);
                last_iterations_ = solver_ilut_.iterations();
                last_error_ = solver_ilut_.error();
                if (solver_ilut_.info() != Eigen::Success) return std::nullopt;
                return x;
            }
            case IterativeSolverConfig::PreconditionerKind::AMG: {
#ifdef PULSIM_HAS_HYPRE
                auto x = hypre_amg_.solve(b);
                if (!x) return std::nullopt;
                last_iterations_ = hypre_amg_.last_iterations();
                last_error_ = hypre_amg_.last_error();
                return *x;
#else
                break;
#endif
            }
        }
        return std::nullopt;
    }
};

// =============================================================================
// Runtime Linear Solver Stack (selection + deterministic fallback)
// =============================================================================

class RuntimeLinearSolver {
public:
    explicit RuntimeLinearSolver(const LinearSolverStackConfig& config = {})
        : config_(config),
          sparse_(std::make_unique<SparseLUPolicy>()),
          enhanced_(std::make_unique<EnhancedSparseLUPolicy>(config.direct_config)),
                    klu_(std::make_unique<KLUPolicy>(config.direct_config)),
                    gmres_(std::make_unique<GMRESPolicy>(config.iterative_config)),
                    bicgstab_(std::make_unique<BiCGSTABPolicy>(config.iterative_config)),
                    cg_(std::make_unique<ConjugateGradientPolicy>(config.iterative_config)) {}

    RuntimeLinearSolver(const RuntimeLinearSolver&) = delete;
    RuntimeLinearSolver& operator=(const RuntimeLinearSolver&) = delete;

    RuntimeLinearSolver(RuntimeLinearSolver&&) noexcept = default;
    RuntimeLinearSolver& operator=(RuntimeLinearSolver&&) noexcept = default;

    bool analyze(const SparseMatrix& A) {
        last_matrix_ = &A;
        active_kind_.reset();
        primary_order_ = build_order(A, config_.order);
        fallback_order_ = build_fallback_order(A);

        auto timed_analyze_with = [&](LinearSolverKind kind) {
            const auto start = std::chrono::steady_clock::now();
            const bool ok = analyze_with(kind, A);
            const auto end = std::chrono::steady_clock::now();
            const double dt = std::chrono::duration<double>(end - start).count();
            telemetry_.total_analyze_calls += 1;
            telemetry_.total_analyze_time_seconds += dt;
            telemetry_.last_analyze_time_seconds = dt;
            return ok;
        };

        for (std::size_t i = 0; i < primary_order_.size(); ++i) {
            auto kind = primary_order_[i];
            if (!is_available(kind)) continue;
            if (timed_analyze_with(kind)) {
                active_kind_ = kind;
                return true;
            }
        }
        return false;
    }

    bool factorize(const SparseMatrix& A) {
        last_matrix_ = &A;

        auto timed_factorize_with = [&](LinearSolverKind kind) {
            const auto start = std::chrono::steady_clock::now();
            const bool ok = factorize_with(kind, A);
            const auto end = std::chrono::steady_clock::now();
            const double dt = std::chrono::duration<double>(end - start).count();
            telemetry_.total_factorize_calls += 1;
            telemetry_.total_factorize_time_seconds += dt;
            telemetry_.last_factorize_time_seconds = dt;
            return ok;
        };

        if (active_kind_) {
            auto kind = *active_kind_;
            if (is_available(kind) && timed_factorize_with(kind)) {
                return true;
            }
        }

        if (primary_order_.empty()) {
            primary_order_ = build_order(A, config_.order);
        }
        if (fallback_order_.empty()) {
            fallback_order_ = build_fallback_order(A);
        }

        for (auto kind : primary_order_) {
            if (!is_available(kind)) continue;
            if (timed_factorize_with(kind)) {
                active_kind_ = kind;
                return true;
            }
            if (!config_.allow_fallback) return false;
        }

        if (!config_.allow_fallback) return false;

        for (auto kind : fallback_order_) {
            if (!is_available(kind)) continue;
            if (timed_factorize_with(kind)) {
                active_kind_ = kind;
                return true;
            }
        }
        return false;
    }

    [[nodiscard]] LinearSolveResult solve(const Vector& b) {
        if (!active_kind_) {
            return LinearSolveResult::failure("No active linear solver");
        }

        auto timed_solve_with = [&](LinearSolverKind kind) {
            const auto start = std::chrono::steady_clock::now();
            auto out = solve_with(kind, b);
            const auto end = std::chrono::steady_clock::now();
            const double dt = std::chrono::duration<double>(end - start).count();
            telemetry_.total_solve_time_seconds += dt;
            telemetry_.last_solve_time_seconds = dt;
            return out;
        };

        auto active_kind = *active_kind_;
        auto result = timed_solve_with(active_kind);
        update_telemetry(active_kind, result);
        if (result || !config_.allow_fallback || !last_matrix_) {
            return result;
        }

        if (primary_order_.empty()) {
            primary_order_ = build_order(*last_matrix_, config_.order);
        }
        if (fallback_order_.empty()) {
            fallback_order_ = build_fallback_order(*last_matrix_);
        }

        for (auto kind : primary_order_) {
            if (kind == active_kind) continue;
            if (!is_available(kind)) continue;

            const auto factor_start = std::chrono::steady_clock::now();
            const bool factor_ok = factorize_with(kind, *last_matrix_);
            const auto factor_end = std::chrono::steady_clock::now();
            const double factor_dt = std::chrono::duration<double>(factor_end - factor_start).count();
            telemetry_.total_factorize_calls += 1;
            telemetry_.total_factorize_time_seconds += factor_dt;
            telemetry_.last_factorize_time_seconds = factor_dt;
            if (!factor_ok) {
                if (!config_.allow_fallback) break;
                continue;
            }

            auto retry = timed_solve_with(kind);
            update_telemetry(kind, retry);
            if (retry) {
                active_kind_ = kind;
                telemetry_.total_fallbacks += 1;
                return retry;
            }
        }

        for (auto kind : fallback_order_) {
            if (kind == active_kind) continue;
            if (!is_available(kind)) continue;

            const auto factor_start = std::chrono::steady_clock::now();
            const bool factor_ok = factorize_with(kind, *last_matrix_);
            const auto factor_end = std::chrono::steady_clock::now();
            const double factor_dt = std::chrono::duration<double>(factor_end - factor_start).count();
            telemetry_.total_factorize_calls += 1;
            telemetry_.total_factorize_time_seconds += factor_dt;
            telemetry_.last_factorize_time_seconds = factor_dt;
            if (!factor_ok) {
                if (!config_.allow_fallback) break;
                continue;
            }

            auto retry = timed_solve_with(kind);
            update_telemetry(kind, retry);
            if (retry) {
                active_kind_ = kind;
                telemetry_.total_fallbacks += 1;
                return retry;
            }

            if (!config_.allow_fallback) break;
        }

        return result;
    }

    [[nodiscard]] bool is_singular() const {
        if (!active_kind_) return true;
        return is_singular_with(*active_kind_);
    }

    [[nodiscard]] const LinearSolverStackConfig& config() const { return config_; }
    void set_config(const LinearSolverStackConfig& config) {
        config_ = config;
        if (enhanced_) enhanced_->set_config(config_.direct_config);
        if (klu_) klu_->set_config(config_.direct_config);
        if (gmres_) gmres_->set_config(config_.iterative_config);
        if (bicgstab_) bicgstab_->set_config(config_.iterative_config);
        if (cg_) cg_->set_config(config_.iterative_config);
        active_kind_.reset();
        last_matrix_ = nullptr;
        primary_order_.clear();
        fallback_order_.clear();
        telemetry_ = {};
    }

    [[nodiscard]] std::optional<LinearSolverKind> active_kind() const {
        return active_kind_;
    }

    [[nodiscard]] LinearSolverTelemetry telemetry() const { return telemetry_; }

    void set_force_iterative(bool force) {
        force_iterative_ = force;
        primary_order_.clear();
        fallback_order_.clear();
        active_kind_.reset();
    }

    [[nodiscard]] bool force_iterative() const { return force_iterative_; }

private:
    LinearSolverStackConfig config_;
    std::unique_ptr<SparseLUPolicy> sparse_;
    std::unique_ptr<EnhancedSparseLUPolicy> enhanced_;
    std::unique_ptr<KLUPolicy> klu_;
    std::unique_ptr<GMRESPolicy> gmres_;
    std::unique_ptr<BiCGSTABPolicy> bicgstab_;
    std::unique_ptr<ConjugateGradientPolicy> cg_;
    std::optional<LinearSolverKind> active_kind_;
    const SparseMatrix* last_matrix_ = nullptr;
    std::vector<LinearSolverKind> primary_order_;
    std::vector<LinearSolverKind> fallback_order_;
    LinearSolverTelemetry telemetry_{};
    bool force_iterative_ = false;

    [[nodiscard]] bool is_available(LinearSolverKind kind) const {
        if (kind == LinearSolverKind::KLU) return KLUPolicy::is_available();
        return true;
    }

    [[nodiscard]] bool is_symmetric(const SparseMatrix& A, Real tol = 1e-12) const {
        return A.isApprox(A.transpose(), tol);
    }

    [[nodiscard]] bool allow_cg(const SparseMatrix& A) const {
        if (!cg_) return false;
        if (config_.iterative_config.enable_scaling) return false;
        if (config_.iterative_config.preconditioner == IterativeSolverConfig::PreconditionerKind::ILU0 ||
            config_.iterative_config.preconditioner == IterativeSolverConfig::PreconditionerKind::ILUT ||
            config_.iterative_config.preconditioner == IterativeSolverConfig::PreconditionerKind::AMG) {
            return false;
        }
        if (!is_symmetric(A)) return false;
        return cg_->is_spd(A);
    }

    [[nodiscard]] std::vector<LinearSolverKind> build_order(
        const SparseMatrix& A,
        const std::vector<LinearSolverKind>& base_order) const {

        if (!config_.auto_select || base_order.empty()) {
            std::vector<LinearSolverKind> order = base_order;
            order.erase(std::remove_if(order.begin(), order.end(), [&](LinearSolverKind kind) {
                return kind == LinearSolverKind::CG && !allow_cg(A);
            }), order.end());
            return order;
        }

        const int rows = static_cast<int>(A.rows());
        const int nnz = static_cast<int>(A.nonZeros());
        bool prefer_iterative = (rows >= config_.size_threshold) || (nnz >= config_.nnz_threshold);

        Real min_diag = std::numeric_limits<Real>::infinity();
        for (int k = 0; k < A.outerSize(); ++k) {
            for (SparseMatrix::InnerIterator it(A, k); it; ++it) {
                if (it.row() == it.col()) {
                    Real val = std::abs(static_cast<Real>(it.value()));
                    if (val < min_diag) min_diag = val;
                }
            }
        }

        if (min_diag < config_.diag_min_threshold) {
            prefer_iterative = false;  // ill-conditioned hint -> prefer direct
        }

        std::vector<LinearSolverKind> order = base_order;
        auto is_iterative = [](LinearSolverKind kind) {
            return kind == LinearSolverKind::GMRES ||
                   kind == LinearSolverKind::BiCGSTAB ||
                   kind == LinearSolverKind::CG;
        };

        if (force_iterative_) {
            std::vector<LinearSolverKind> filtered;
            filtered.reserve(order.size());
            for (auto kind : order) {
                if (is_iterative(kind)) {
                    filtered.push_back(kind);
                }
            }
            if (!filtered.empty()) {
                order = std::move(filtered);
            }
        }

        order.erase(std::remove_if(order.begin(), order.end(), [&](LinearSolverKind kind) {
            return kind == LinearSolverKind::CG && !allow_cg(A);
        }), order.end());

        if (prefer_iterative) {
            auto it = std::find_if(order.begin(), order.end(), is_iterative);
            if (it != order.end() && it != order.begin()) {
                LinearSolverKind chosen = *it;
                order.erase(it);
                order.insert(order.begin(), chosen);
            }
        } else {
            auto it = std::find_if(order.begin(), order.end(),
                                   [&](LinearSolverKind kind) { return !is_iterative(kind); });
            if (it != order.end() && it != order.begin()) {
                LinearSolverKind chosen = *it;
                order.erase(it);
                order.insert(order.begin(), chosen);
            }
        }

        return order;
    }

    [[nodiscard]] std::vector<LinearSolverKind> build_fallback_order(const SparseMatrix& A) const {
        if (!config_.fallback_order.empty()) {
            return build_order(A, config_.fallback_order);
        }
        return build_order(A, config_.order);
    }

    bool analyze_with(LinearSolverKind kind, const SparseMatrix& A) {
        switch (kind) {
            case LinearSolverKind::SparseLU:
                return sparse_->analyze(A);
            case LinearSolverKind::EnhancedSparseLU:
                return enhanced_->analyze(A);
            case LinearSolverKind::KLU:
                return klu_->analyze(A);
            case LinearSolverKind::GMRES:
                return gmres_->analyze(A);
            case LinearSolverKind::BiCGSTAB:
                return bicgstab_->analyze(A);
            case LinearSolverKind::CG:
                if (!allow_cg(A)) return false;
                return cg_->analyze(A);
        }
        return false;
    }

    bool factorize_with(LinearSolverKind kind, const SparseMatrix& A) {
        switch (kind) {
            case LinearSolverKind::SparseLU:
                return sparse_->factorize(A);
            case LinearSolverKind::EnhancedSparseLU:
                return enhanced_->factorize(A);
            case LinearSolverKind::KLU:
                return klu_->factorize(A);
            case LinearSolverKind::GMRES:
                return gmres_->factorize(A);
            case LinearSolverKind::BiCGSTAB:
                return bicgstab_->factorize(A);
            case LinearSolverKind::CG:
                if (!allow_cg(A)) return false;
                return cg_->factorize(A);
        }
        return false;
    }

    [[nodiscard]] LinearSolveResult solve_with(LinearSolverKind kind, const Vector& b) {
        switch (kind) {
            case LinearSolverKind::SparseLU:
                return sparse_->solve(b);
            case LinearSolverKind::EnhancedSparseLU:
                return enhanced_->solve(b);
            case LinearSolverKind::KLU:
                return klu_->solve(b);
            case LinearSolverKind::GMRES:
                return gmres_->solve(b);
            case LinearSolverKind::BiCGSTAB:
                return bicgstab_->solve(b);
            case LinearSolverKind::CG:
                return cg_->solve(b);
        }
        return LinearSolveResult::failure("Unknown linear solver");
    }

    [[nodiscard]] bool is_singular_with(LinearSolverKind kind) const {
        switch (kind) {
            case LinearSolverKind::SparseLU:
                return sparse_->is_singular();
            case LinearSolverKind::EnhancedSparseLU:
                return enhanced_->is_singular();
            case LinearSolverKind::KLU:
                return klu_->is_singular();
            case LinearSolverKind::GMRES:
                return gmres_->is_singular();
            case LinearSolverKind::BiCGSTAB:
                return bicgstab_->is_singular();
            case LinearSolverKind::CG:
                return cg_->is_singular();
        }
        return true;
    }

    void update_telemetry(LinearSolverKind kind, [[maybe_unused]] const LinearSolveResult& result) {
        telemetry_.total_solve_calls += 1;
        telemetry_.last_solver = kind;
        telemetry_.last_preconditioner.reset();

        int iterations = 0;
        Real error = 0.0;
        switch (kind) {
            case LinearSolverKind::GMRES:
                iterations = gmres_->last_iterations();
                error = gmres_->last_error();
                telemetry_.last_preconditioner = config_.iterative_config.preconditioner;
                break;
            case LinearSolverKind::BiCGSTAB:
                iterations = bicgstab_->last_iterations();
                error = bicgstab_->last_error();
                telemetry_.last_preconditioner = config_.iterative_config.preconditioner;
                break;
            case LinearSolverKind::CG:
                iterations = cg_->last_iterations();
                error = cg_->last_error();
                telemetry_.last_preconditioner = config_.iterative_config.preconditioner;
                break;
            default:
                break;
        }

        telemetry_.last_iterations = iterations;
        telemetry_.last_error = error;
        telemetry_.total_iterations += iterations;
    }
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
            detail::aligned_deallocate(block);
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
            detail::aligned_deallocate(blocks_[i]);
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
        // Keep arena blocks cache-line aligned for SIMD-friendly access.
        constexpr std::size_t alignment = 64;
        void* block = detail::aligned_allocate(alignment, size);
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
            std::size_t byte_size = new_size * sizeof(T);
            T* new_data = static_cast<T*>(detail::aligned_allocate(Alignment, byte_size));
            if (!new_data) throw std::bad_alloc();

            if (data_) {
                std::memcpy(new_data, data_, size_ * sizeof(T));
                detail::aligned_deallocate(data_);
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
            std::size_t byte_size = size * sizeof(T);
            data_ = static_cast<T*>(detail::aligned_allocate(Alignment, byte_size));
            if (!data_) throw std::bad_alloc();
            capacity_ = size;
        }
    }

    void deallocate() {
        if (data_) {
            detail::aligned_deallocate(data_);
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
