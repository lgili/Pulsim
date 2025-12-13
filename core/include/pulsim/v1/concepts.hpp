#pragma once

// =============================================================================
// PulsimCore v2 - C++23 Concepts for Device Models and Solvers
// =============================================================================
// This header defines concepts that constrain template parameters for:
// - Device models (resistors, capacitors, switches, etc.)
// - Linear solvers (SparseLU, KLU, etc.)
// - Integration methods (Backward Euler, Trapezoidal, BDF)
// - Convergence policies (line search, trust region, etc.)
// =============================================================================

#include <concepts>
#include <type_traits>
#include <optional>
#include <span>
#include <string>

namespace pulsim::v1 {

// =============================================================================
// Basic Type Concepts
// =============================================================================

/// Floating-point type for numerical computations
template<typename T>
concept FloatingPoint = std::floating_point<T>;

/// Signed integer type for indices
template<typename T>
concept SignedIndex = std::signed_integral<T>;

// =============================================================================
// Matrix/Vector Concepts
// =============================================================================

/// A type that can be indexed like a vector
template<typename V>
concept VectorLike = requires(V v, std::size_t i) {
    { v[i] } -> std::convertible_to<typename V::Scalar>;
    { v.size() } -> std::convertible_to<std::size_t>;
    { v.data() } -> std::same_as<typename V::Scalar*>;
};

/// A type that represents a sparse matrix
template<typename M>
concept SparseMatrixLike = requires(M m, std::size_t i, std::size_t j) {
    { m.rows() } -> std::convertible_to<std::size_t>;
    { m.cols() } -> std::convertible_to<std::size_t>;
    { m.nonZeros() } -> std::convertible_to<std::size_t>;
    { m.coeffRef(i, j) } -> std::convertible_to<typename M::Scalar&>;
};

// =============================================================================
// Device Model Concepts
// =============================================================================

/// Represents the state of a device for stamping
template<typename S>
concept DeviceState = requires(S s) {
    typename S::Scalar;
    { s.voltage() } -> std::convertible_to<typename S::Scalar>;
    { s.current() } -> std::convertible_to<typename S::Scalar>;
};

/// A device that can stamp into the MNA matrix
template<typename D>
concept StampableDevice = requires(D device) {
    typename D::Scalar;
    typename D::Params;

    // Must have pin count known at compile time
    { D::num_pins } -> std::convertible_to<std::size_t>;

    // Must have device type identifier
    { D::device_type } -> std::convertible_to<int>;
};

/// A device that contributes to the conductance matrix G
template<typename D, typename Matrix, typename Vector>
concept ConductanceDevice = StampableDevice<D> && requires(
    D device,
    Matrix& G,
    Vector& b,
    std::span<const std::size_t> nodes
) {
    { device.stamp_conductance(G, b, nodes) } -> std::same_as<void>;
};

/// A device that contributes to the Jacobian for Newton iteration
template<typename D, typename Matrix, typename Vector>
concept NonlinearDevice = StampableDevice<D> && requires(
    D device,
    Matrix& J,
    Vector& f,
    const Vector& x,
    std::span<const std::size_t> nodes
) {
    { device.stamp_jacobian(J, f, x, nodes) } -> std::same_as<void>;
};

/// A device with dynamic (time-varying) behavior
template<typename D>
concept DynamicDevice = StampableDevice<D> && requires(D device) {
    { device.has_dynamics() } -> std::convertible_to<bool>;
    { device.update_history() } -> std::same_as<void>;
};

/// A device that can report its Jacobian sparsity pattern at compile time
template<typename D>
concept StaticSparsityDevice = StampableDevice<D> && requires {
    { D::jacobian_pattern() } -> std::convertible_to<std::span<const std::pair<int, int>>>;
};

// =============================================================================
// Solver Concepts
// =============================================================================

/// A linear solver that can solve Ax = b
template<typename S, typename Matrix, typename Vector>
concept LinearSolver = requires(S solver, const Matrix& A, const Vector& b) {
    { solver.analyze_pattern(A) } -> std::same_as<void>;
    { solver.factorize(A) } -> std::convertible_to<bool>;
    { solver.solve(b) } -> std::convertible_to<Vector>;
    { solver.is_singular() } -> std::convertible_to<bool>;
};

/// A Newton solver for nonlinear systems
template<typename S, typename Vector, typename SystemFunc>
concept NewtonSolver = requires(S solver, const Vector& x0, SystemFunc f) {
    { solver.solve(x0, f) };  // Returns result type
    { solver.iterations() } -> std::convertible_to<int>;
    { solver.residual() } -> std::convertible_to<typename Vector::Scalar>;
};

// =============================================================================
// Integration Method Concepts
// =============================================================================

/// Coefficients for companion model integration
template<typename C>
concept CompanionCoeffs = requires(C c) {
    { c.alpha } -> std::convertible_to<double>;
    { c.beta } -> std::convertible_to<double>;
};

/// An integration method that can provide companion model coefficients
template<typename M>
concept IntegrationMethod = requires(M method, double dt, double dt_prev) {
    { method.coefficients(dt, dt_prev) } -> CompanionCoeffs;
    { method.order() } -> std::convertible_to<int>;
    { method.is_implicit() } -> std::convertible_to<bool>;
};

/// An integration method with local truncation error estimation
template<typename M, typename Vector>
concept ErrorEstimatingMethod = IntegrationMethod<M> && requires(
    M method,
    const Vector& y_high,
    const Vector& y_low
) {
    { method.estimate_error(y_high, y_low) } -> std::convertible_to<typename Vector::Scalar>;
};

// =============================================================================
// Convergence Policy Concepts
// =============================================================================

/// A policy that determines step acceptance
template<typename P, typename Scalar>
concept ConvergencePolicy = requires(P policy, Scalar residual, Scalar prev_residual) {
    { policy.is_converged(residual) } -> std::convertible_to<bool>;
    { policy.compute_damping(residual, prev_residual) } -> std::convertible_to<Scalar>;
};

/// A line search policy for Newton methods
template<typename P, typename Vector, typename SystemFunc>
concept LineSearchPolicy = requires(
    P policy,
    const Vector& x,
    const Vector& dx,
    const Vector& f,
    SystemFunc system
) {
    { policy.search(x, dx, f, system) } -> std::convertible_to<typename Vector::Scalar>;
};

// =============================================================================
// Result Types (portable alternative to std::expected)
// =============================================================================

/// Error types for solver operations
enum class SolverError {
    SingularMatrix,
    MaxIterationsReached,
    NumericalInstability,
    InvalidInput,
    MemoryAllocationFailed
};

/// Convert SolverError to string
[[nodiscard]] inline constexpr const char* to_string(SolverError err) noexcept {
    switch (err) {
        case SolverError::SingularMatrix: return "SingularMatrix";
        case SolverError::MaxIterationsReached: return "MaxIterationsReached";
        case SolverError::NumericalInstability: return "NumericalInstability";
        case SolverError::InvalidInput: return "InvalidInput";
        case SolverError::MemoryAllocationFailed: return "MemoryAllocationFailed";
        default: return "Unknown";
    }
}

/// Result type for operations that can fail (portable alternative to std::expected)
template<typename T>
struct Result {
    std::optional<T> value;
    SolverError error = SolverError::InvalidInput;

    [[nodiscard]] bool has_value() const { return value.has_value(); }
    [[nodiscard]] explicit operator bool() const { return has_value(); }
    [[nodiscard]] T& operator*() { return *value; }
    [[nodiscard]] const T& operator*() const { return *value; }
    [[nodiscard]] T* operator->() { return &*value; }
    [[nodiscard]] const T* operator->() const { return &*value; }

    static Result success(T v) {
        Result r;
        r.value = std::move(v);
        return r;
    }

    static Result failure(SolverError err) {
        Result r;
        r.error = err;
        return r;
    }
};

// =============================================================================
// Utility Concepts
// =============================================================================

/// A type that can be called with a progress update
template<typename F, typename Progress>
concept ProgressCallback = requires(F f, const Progress& p) {
    { f(p) } -> std::convertible_to<bool>;  // Returns false to abort
};

/// A type that provides circuit topology information
template<typename C>
concept CircuitTopology = requires(C circuit) {
    { circuit.num_nodes() } -> std::convertible_to<std::size_t>;
    { circuit.num_branches() } -> std::convertible_to<std::size_t>;
    { circuit.num_devices() } -> std::convertible_to<std::size_t>;
};

}  // namespace pulsim::v1
