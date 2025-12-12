#pragma once

// =============================================================================
// PulsimCore v2 - Expression Templates for Lazy Vector/Matrix Evaluation
// =============================================================================
// This header provides expression templates for efficient vector and matrix
// operations without creating temporaries. Operations are lazily evaluated
// only when the result is needed.
//
// Features:
// - Zero-overhead abstraction through expression templates
// - SIMD-optimized evaluation kernels
// - Fallback to Eigen expressions for validation
// - Support for mixed scalar types
// =============================================================================

#include "pulsim/v2/numeric_types.hpp"
#include <cstddef>
#include <type_traits>
#include <concepts>
#include <cmath>
#include <span>

// SIMD headers (platform-specific)
#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#define PULSIM_HAS_SSE 1
#if defined(__AVX__)
#define PULSIM_HAS_AVX 1
#endif
#if defined(__AVX2__)
#define PULSIM_HAS_AVX2 1
#endif
#if defined(__AVX512F__)
#define PULSIM_HAS_AVX512 1
#endif
#elif defined(__aarch64__) || defined(_M_ARM64)
#include <arm_neon.h>
#define PULSIM_HAS_NEON 1
#endif

// Optional Eigen fallback
#ifndef PULSIM_USE_EIGEN_EXPRESSIONS
#define PULSIM_USE_EIGEN_EXPRESSIONS 0
#endif

#if PULSIM_USE_EIGEN_EXPRESSIONS
#include <Eigen/Core>
#endif

namespace pulsim::v2 {

// =============================================================================
// 2.4.1: Expression Base Template
// =============================================================================

/// Tag types for expression operations
struct AddOp {};
struct SubOp {};
struct MulOp {};
struct DivOp {};
struct ScaleOp {};
struct NegateOp {};

// ExpressionMarker is defined in numeric_types.hpp

/// Concept for expression-like types (only our custom types, not Eigen)
template<typename T>
concept ExpressionLike = requires(const T& expr, std::size_t i) {
    { expr.size() } -> std::convertible_to<std::size_t>;
    { expr[i] } -> std::convertible_to<typename T::value_type>;
} && std::derived_from<T, ExpressionMarker>;

/// Concept for scalar types usable in expressions
template<typename T>
concept ScalarValue = std::is_arithmetic_v<T>;

/// Base class for all expressions using CRTP
template<typename Derived, typename T>
class ExpressionBase : public ExpressionMarker {
public:
    using value_type = T;

    /// Access element at index i
    [[nodiscard]] constexpr T operator[](std::size_t i) const {
        return static_cast<const Derived&>(*this).eval_at(i);
    }

    /// Get the size of the expression
    [[nodiscard]] constexpr std::size_t size() const {
        return static_cast<const Derived&>(*this).size();
    }

    /// Get derived reference (CRTP)
    [[nodiscard]] constexpr const Derived& derived() const {
        return static_cast<const Derived&>(*this);
    }

    /// Evaluate the full expression into a destination
    template<typename Dest>
    constexpr void eval_to(Dest& dest) const {
        const auto n = size();
        for (std::size_t i = 0; i < n; ++i) {
            dest[i] = (*this)[i];
        }
    }

    /// Evaluate to a new StaticVector
    template<std::size_t N>
    [[nodiscard]] constexpr StaticVector<T, N> eval() const {
        StaticVector<T, N> result;
        eval_to(result);
        return result;
    }

    /// Sum all elements
    [[nodiscard]] constexpr T sum() const {
        T s = T{0};
        const auto n = size();
        for (std::size_t i = 0; i < n; ++i) {
            s += (*this)[i];
        }
        return s;
    }

    /// Dot product with another expression
    template<ExpressionLike Other>
    [[nodiscard]] constexpr T dot(const Other& other) const {
        T s = T{0};
        const auto n = size();
        for (std::size_t i = 0; i < n; ++i) {
            s += (*this)[i] * other[i];
        }
        return s;
    }

    /// Squared L2 norm
    [[nodiscard]] constexpr T squared_norm() const {
        return dot(derived());
    }

    /// L2 norm
    [[nodiscard]] T norm() const {
        return std::sqrt(squared_norm());
    }

    /// Maximum absolute value
    [[nodiscard]] T max_abs() const {
        T m = T{0};
        const auto n = size();
        for (std::size_t i = 0; i < n; ++i) {
            T a = std::abs((*this)[i]);
            if (a > m) m = a;
        }
        return m;
    }
};

// =============================================================================
// 2.4.2: Binary Expression Template
// =============================================================================

/// Binary expression: Lhs op Rhs
template<typename Op, typename Lhs, typename Rhs>
class BinaryExpr : public ExpressionBase<BinaryExpr<Op, Lhs, Rhs>,
                                          typename Lhs::value_type> {
public:
    using value_type = typename Lhs::value_type;
    using lhs_type = Lhs;
    using rhs_type = Rhs;
    using op_type = Op;

    constexpr BinaryExpr(const Lhs& lhs, const Rhs& rhs)
        : lhs_(lhs), rhs_(rhs) {}

    [[nodiscard]] constexpr std::size_t size() const {
        return lhs_.size();
    }

    [[nodiscard]] constexpr value_type eval_at(std::size_t i) const {
        if constexpr (std::is_same_v<Op, AddOp>) {
            return lhs_[i] + rhs_[i];
        } else if constexpr (std::is_same_v<Op, SubOp>) {
            return lhs_[i] - rhs_[i];
        } else if constexpr (std::is_same_v<Op, MulOp>) {
            return lhs_[i] * rhs_[i];
        } else if constexpr (std::is_same_v<Op, DivOp>) {
            return lhs_[i] / rhs_[i];
        } else {
            static_assert(sizeof(Op) == 0, "Unknown binary operation");
        }
    }

    [[nodiscard]] constexpr const Lhs& lhs() const { return lhs_; }
    [[nodiscard]] constexpr const Rhs& rhs() const { return rhs_; }

private:
    const Lhs& lhs_;
    const Rhs& rhs_;
};

/// AddExpr alias
template<typename Lhs, typename Rhs>
using AddExpr = BinaryExpr<AddOp, Lhs, Rhs>;

/// SubExpr alias
template<typename Lhs, typename Rhs>
using SubExpr = BinaryExpr<SubOp, Lhs, Rhs>;

/// MulExpr alias (element-wise)
template<typename Lhs, typename Rhs>
using MulExpr = BinaryExpr<MulOp, Lhs, Rhs>;

/// DivExpr alias (element-wise)
template<typename Lhs, typename Rhs>
using DivExpr = BinaryExpr<DivOp, Lhs, Rhs>;

// =============================================================================
// Scalar Expression (for scalar * vector)
// =============================================================================

/// Wrapper to treat a scalar as a constant expression
template<typename T>
class ScalarExpr : public ExpressionBase<ScalarExpr<T>, T> {
public:
    using value_type = T;

    constexpr explicit ScalarExpr(T value, std::size_t size)
        : value_(value), size_(size) {}

    [[nodiscard]] constexpr std::size_t size() const { return size_; }
    [[nodiscard]] constexpr T eval_at(std::size_t) const { return value_; }

private:
    T value_;
    std::size_t size_;
};

// =============================================================================
// Scale Expression (scalar * expression)
// =============================================================================

template<typename Expr>
class ScaleExpr : public ExpressionBase<ScaleExpr<Expr>, typename Expr::value_type> {
public:
    using value_type = typename Expr::value_type;

    constexpr ScaleExpr(value_type scalar, const Expr& expr)
        : scalar_(scalar), expr_(expr) {}

    [[nodiscard]] constexpr std::size_t size() const { return expr_.size(); }

    [[nodiscard]] constexpr value_type eval_at(std::size_t i) const {
        return scalar_ * expr_[i];
    }

    [[nodiscard]] constexpr value_type scalar() const { return scalar_; }
    [[nodiscard]] constexpr const Expr& expr() const { return expr_; }

private:
    value_type scalar_;
    const Expr& expr_;
};

// =============================================================================
// Negate Expression
// =============================================================================

template<typename Expr>
class NegateExpr : public ExpressionBase<NegateExpr<Expr>, typename Expr::value_type> {
public:
    using value_type = typename Expr::value_type;

    constexpr explicit NegateExpr(const Expr& expr) : expr_(expr) {}

    [[nodiscard]] constexpr std::size_t size() const { return expr_.size(); }

    [[nodiscard]] constexpr value_type eval_at(std::size_t i) const {
        return -expr_[i];
    }

private:
    const Expr& expr_;
};

// =============================================================================
// Vector Wrapper (for using raw arrays/vectors as expressions)
// =============================================================================

/// Wrapper to make contiguous containers work as expressions
template<typename T>
class VectorRef : public ExpressionBase<VectorRef<T>, T> {
public:
    using value_type = T;

    constexpr VectorRef(const T* data, std::size_t size)
        : data_(data), size_(size) {}

    template<std::size_t N>
    constexpr explicit VectorRef(const StaticVector<T, N>& vec)
        : data_(vec.data()), size_(N) {}

    constexpr explicit VectorRef(std::span<const T> span)
        : data_(span.data()), size_(span.size()) {}

    [[nodiscard]] constexpr std::size_t size() const { return size_; }
    [[nodiscard]] constexpr T eval_at(std::size_t i) const { return data_[i]; }
    [[nodiscard]] constexpr const T* data() const { return data_; }

private:
    const T* data_;
    std::size_t size_;
};

// Deduction guide
template<typename T, std::size_t N>
VectorRef(const StaticVector<T, N>&) -> VectorRef<T>;

// =============================================================================
// Operator Overloads for Expressions
// =============================================================================

// Expression + Expression
template<typename E1, typename E2>
    requires ExpressionLike<E1> && ExpressionLike<E2>
[[nodiscard]] constexpr auto operator+(const E1& lhs, const E2& rhs) {
    return AddExpr<E1, E2>(lhs, rhs);
}

// Expression - Expression
template<typename E1, typename E2>
    requires ExpressionLike<E1> && ExpressionLike<E2>
[[nodiscard]] constexpr auto operator-(const E1& lhs, const E2& rhs) {
    return SubExpr<E1, E2>(lhs, rhs);
}

// Expression * Expression (element-wise)
template<typename E1, typename E2>
    requires ExpressionLike<E1> && ExpressionLike<E2>
[[nodiscard]] constexpr auto operator*(const E1& lhs, const E2& rhs) {
    return MulExpr<E1, E2>(lhs, rhs);
}

// Expression / Expression (element-wise)
template<typename E1, typename E2>
    requires ExpressionLike<E1> && ExpressionLike<E2>
[[nodiscard]] constexpr auto operator/(const E1& lhs, const E2& rhs) {
    return DivExpr<E1, E2>(lhs, rhs);
}

// Scalar * Expression
template<typename E, ScalarValue S>
    requires ExpressionLike<E>
[[nodiscard]] constexpr auto operator*(S scalar, const E& expr) {
    return ScaleExpr<E>(static_cast<typename E::value_type>(scalar), expr);
}

// Expression * Scalar
template<typename E, ScalarValue S>
    requires ExpressionLike<E>
[[nodiscard]] constexpr auto operator*(const E& expr, S scalar) {
    return ScaleExpr<E>(static_cast<typename E::value_type>(scalar), expr);
}

// Expression / Scalar
template<typename E, ScalarValue S>
    requires ExpressionLike<E>
[[nodiscard]] constexpr auto operator/(const E& expr, S scalar) {
    return ScaleExpr<E>(typename E::value_type{1} / static_cast<typename E::value_type>(scalar), expr);
}

// -Expression
template<typename E>
    requires ExpressionLike<E>
[[nodiscard]] constexpr auto operator-(const E& expr) {
    return NegateExpr<E>(expr);
}

// =============================================================================
// 2.4.3: Lazy Evaluation Helper
// =============================================================================

/// Evaluate any expression to a vector
template<typename E, std::size_t N>
    requires ExpressionLike<E>
[[nodiscard]] constexpr StaticVector<typename E::value_type, N> eval(const E& expr) {
    StaticVector<typename E::value_type, N> result;
    for (std::size_t i = 0; i < N; ++i) {
        result[i] = expr[i];
    }
    return result;
}

/// Evaluate expression into existing storage
template<typename E, typename Dest>
    requires ExpressionLike<E>
constexpr void eval_to(const E& expr, Dest& dest) {
    const auto n = expr.size();
    for (std::size_t i = 0; i < n; ++i) {
        dest[i] = expr[i];
    }
}

// =============================================================================
// 2.4.4: SIMD-Optimized Evaluation Kernels
// =============================================================================

namespace simd {

/// SIMD width in elements for double precision
#if defined(PULSIM_HAS_AVX512)
inline constexpr std::size_t double_width = 8;
#elif defined(PULSIM_HAS_AVX)
inline constexpr std::size_t double_width = 4;
#elif defined(PULSIM_HAS_SSE) || defined(PULSIM_HAS_NEON)
inline constexpr std::size_t double_width = 2;
#else
inline constexpr std::size_t double_width = 1;
#endif

/// SIMD width in elements for single precision
#if defined(PULSIM_HAS_AVX512)
inline constexpr std::size_t float_width = 16;
#elif defined(PULSIM_HAS_AVX)
inline constexpr std::size_t float_width = 8;
#elif defined(PULSIM_HAS_SSE) || defined(PULSIM_HAS_NEON)
inline constexpr std::size_t float_width = 4;
#else
inline constexpr std::size_t float_width = 1;
#endif

/// Check if type can use SIMD
template<typename T>
inline constexpr bool can_simd = (std::is_same_v<T, double> && double_width > 1) ||
                                  (std::is_same_v<T, float> && float_width > 1);

/// SIMD vector addition (double, AVX)
#if defined(PULSIM_HAS_AVX)
inline void add_avx(const double* a, const double* b, double* c, std::size_t n) {
    std::size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m256d va = _mm256_loadu_pd(a + i);
        __m256d vb = _mm256_loadu_pd(b + i);
        __m256d vc = _mm256_add_pd(va, vb);
        _mm256_storeu_pd(c + i, vc);
    }
    // Scalar fallback for remainder
    for (; i < n; ++i) {
        c[i] = a[i] + b[i];
    }
}

inline void sub_avx(const double* a, const double* b, double* c, std::size_t n) {
    std::size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m256d va = _mm256_loadu_pd(a + i);
        __m256d vb = _mm256_loadu_pd(b + i);
        __m256d vc = _mm256_sub_pd(va, vb);
        _mm256_storeu_pd(c + i, vc);
    }
    for (; i < n; ++i) {
        c[i] = a[i] - b[i];
    }
}

inline void scale_avx(double scalar, const double* a, double* c, std::size_t n) {
    std::size_t i = 0;
    __m256d vs = _mm256_set1_pd(scalar);
    for (; i + 4 <= n; i += 4) {
        __m256d va = _mm256_loadu_pd(a + i);
        __m256d vc = _mm256_mul_pd(vs, va);
        _mm256_storeu_pd(c + i, vc);
    }
    for (; i < n; ++i) {
        c[i] = scalar * a[i];
    }
}

inline double dot_avx(const double* a, const double* b, std::size_t n) {
    __m256d sum = _mm256_setzero_pd();
    std::size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m256d va = _mm256_loadu_pd(a + i);
        __m256d vb = _mm256_loadu_pd(b + i);
        sum = _mm256_fmadd_pd(va, vb, sum);
    }
    // Horizontal sum
    __m128d low = _mm256_castpd256_pd128(sum);
    __m128d high = _mm256_extractf128_pd(sum, 1);
    __m128d sum128 = _mm_add_pd(low, high);
    sum128 = _mm_hadd_pd(sum128, sum128);
    double result = _mm_cvtsd_f64(sum128);
    // Remainder
    for (; i < n; ++i) {
        result += a[i] * b[i];
    }
    return result;
}
#endif

/// SIMD vector addition (double, SSE)
#if defined(PULSIM_HAS_SSE) && !defined(PULSIM_HAS_AVX)
inline void add_sse(const double* a, const double* b, double* c, std::size_t n) {
    std::size_t i = 0;
    for (; i + 2 <= n; i += 2) {
        __m128d va = _mm_loadu_pd(a + i);
        __m128d vb = _mm_loadu_pd(b + i);
        __m128d vc = _mm_add_pd(va, vb);
        _mm_storeu_pd(c + i, vc);
    }
    for (; i < n; ++i) {
        c[i] = a[i] + b[i];
    }
}
#endif

/// SIMD vector addition (float, NEON)
#if defined(PULSIM_HAS_NEON)
inline void add_neon(const float* a, const float* b, float* c, std::size_t n) {
    std::size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        float32x4_t vc = vaddq_f32(va, vb);
        vst1q_f32(c + i, vc);
    }
    for (; i < n; ++i) {
        c[i] = a[i] + b[i];
    }
}

inline void add_neon(const double* a, const double* b, double* c, std::size_t n) {
    std::size_t i = 0;
    for (; i + 2 <= n; i += 2) {
        float64x2_t va = vld1q_f64(a + i);
        float64x2_t vb = vld1q_f64(b + i);
        float64x2_t vc = vaddq_f64(va, vb);
        vst1q_f64(c + i, vc);
    }
    for (; i < n; ++i) {
        c[i] = a[i] + b[i];
    }
}

inline void sub_neon(const double* a, const double* b, double* c, std::size_t n) {
    std::size_t i = 0;
    for (; i + 2 <= n; i += 2) {
        float64x2_t va = vld1q_f64(a + i);
        float64x2_t vb = vld1q_f64(b + i);
        float64x2_t vc = vsubq_f64(va, vb);
        vst1q_f64(c + i, vc);
    }
    for (; i < n; ++i) {
        c[i] = a[i] - b[i];
    }
}

inline void scale_neon(double scalar, const double* a, double* c, std::size_t n) {
    std::size_t i = 0;
    float64x2_t vs = vdupq_n_f64(scalar);
    for (; i + 2 <= n; i += 2) {
        float64x2_t va = vld1q_f64(a + i);
        float64x2_t vc = vmulq_f64(vs, va);
        vst1q_f64(c + i, vc);
    }
    for (; i < n; ++i) {
        c[i] = scalar * a[i];
    }
}

inline double dot_neon(const double* a, const double* b, std::size_t n) {
    float64x2_t sum = vdupq_n_f64(0.0);
    std::size_t i = 0;
    for (; i + 2 <= n; i += 2) {
        float64x2_t va = vld1q_f64(a + i);
        float64x2_t vb = vld1q_f64(b + i);
        sum = vfmaq_f64(sum, va, vb);
    }
    double result = vgetq_lane_f64(sum, 0) + vgetq_lane_f64(sum, 1);
    for (; i < n; ++i) {
        result += a[i] * b[i];
    }
    return result;
}
#endif

/// Scalar fallback implementations
inline void add_scalar(const double* a, const double* b, double* c, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) {
        c[i] = a[i] + b[i];
    }
}

inline void sub_scalar(const double* a, const double* b, double* c, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) {
        c[i] = a[i] - b[i];
    }
}

inline void scale_scalar(double scalar, const double* a, double* c, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) {
        c[i] = scalar * a[i];
    }
}

inline double dot_scalar(const double* a, const double* b, std::size_t n) {
    double sum = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

/// Dispatched add operation
inline void add(const double* a, const double* b, double* c, std::size_t n) {
#if defined(PULSIM_HAS_AVX)
    add_avx(a, b, c, n);
#elif defined(PULSIM_HAS_SSE)
    add_sse(a, b, c, n);
#elif defined(PULSIM_HAS_NEON)
    add_neon(a, b, c, n);
#else
    add_scalar(a, b, c, n);
#endif
}

inline void sub(const double* a, const double* b, double* c, std::size_t n) {
#if defined(PULSIM_HAS_AVX)
    sub_avx(a, b, c, n);
#elif defined(PULSIM_HAS_NEON)
    sub_neon(a, b, c, n);
#else
    sub_scalar(a, b, c, n);
#endif
}

inline void scale(double scalar, const double* a, double* c, std::size_t n) {
#if defined(PULSIM_HAS_AVX)
    scale_avx(scalar, a, c, n);
#elif defined(PULSIM_HAS_NEON)
    scale_neon(scalar, a, c, n);
#else
    scale_scalar(scalar, a, c, n);
#endif
}

inline double dot(const double* a, const double* b, std::size_t n) {
#if defined(PULSIM_HAS_AVX)
    return dot_avx(a, b, n);
#elif defined(PULSIM_HAS_NEON)
    return dot_neon(a, b, n);
#else
    return dot_scalar(a, b, n);
#endif
}

} // namespace simd

// =============================================================================
// SIMD-Optimized Expression Evaluation
// =============================================================================

/// Evaluate addition expression with SIMD
template<typename T, std::size_t N>
void eval_add_simd(const StaticVector<T, N>& lhs,
                   const StaticVector<T, N>& rhs,
                   StaticVector<T, N>& dest) {
    if constexpr (std::is_same_v<T, double>) {
        simd::add(lhs.data(), rhs.data(), dest.data(), N);
    } else {
        for (std::size_t i = 0; i < N; ++i) {
            dest[i] = lhs[i] + rhs[i];
        }
    }
}

template<typename T, std::size_t N>
void eval_sub_simd(const StaticVector<T, N>& lhs,
                   const StaticVector<T, N>& rhs,
                   StaticVector<T, N>& dest) {
    if constexpr (std::is_same_v<T, double>) {
        simd::sub(lhs.data(), rhs.data(), dest.data(), N);
    } else {
        for (std::size_t i = 0; i < N; ++i) {
            dest[i] = lhs[i] - rhs[i];
        }
    }
}

template<typename T, std::size_t N>
void eval_scale_simd(T scalar,
                     const StaticVector<T, N>& src,
                     StaticVector<T, N>& dest) {
    if constexpr (std::is_same_v<T, double>) {
        simd::scale(scalar, src.data(), dest.data(), N);
    } else {
        for (std::size_t i = 0; i < N; ++i) {
            dest[i] = scalar * src[i];
        }
    }
}

template<typename T, std::size_t N>
T dot_simd(const StaticVector<T, N>& a, const StaticVector<T, N>& b) {
    if constexpr (std::is_same_v<T, double>) {
        return simd::dot(a.data(), b.data(), N);
    } else {
        T sum = T{0};
        for (std::size_t i = 0; i < N; ++i) {
            sum += a[i] * b[i];
        }
        return sum;
    }
}

// =============================================================================
// 2.4.6: Eigen Fallback (for validation)
// =============================================================================

#if PULSIM_USE_EIGEN_EXPRESSIONS

namespace eigen_compat {

/// Convert StaticVector to Eigen::Vector
template<typename T, std::size_t N>
auto to_eigen(const StaticVector<T, N>& vec) {
    Eigen::Matrix<T, static_cast<int>(N), 1> result;
    for (std::size_t i = 0; i < N; ++i) {
        result(static_cast<int>(i)) = vec[i];
    }
    return result;
}

/// Convert Eigen::Vector to StaticVector
template<typename T, std::size_t N>
StaticVector<T, N> from_eigen(const Eigen::Matrix<T, static_cast<int>(N), 1>& vec) {
    StaticVector<T, N> result;
    for (std::size_t i = 0; i < N; ++i) {
        result[i] = vec(static_cast<int>(i));
    }
    return result;
}

/// Evaluate expression using Eigen (for validation)
template<typename E, std::size_t N>
    requires ExpressionLike<E>
auto eval_with_eigen(const E& expr) {
    using T = typename E::value_type;
    Eigen::Matrix<T, static_cast<int>(N), 1> result;
    for (std::size_t i = 0; i < N; ++i) {
        result(static_cast<int>(i)) = expr[i];
    }
    return result;
}

} // namespace eigen_compat

#endif // PULSIM_USE_EIGEN_EXPRESSIONS

// =============================================================================
// Expression Traits for Introspection
// =============================================================================

namespace detail {

/// Check if type is a binary expression
template<typename T>
struct is_binary_expr : std::false_type {};

template<typename Op, typename L, typename R>
struct is_binary_expr<BinaryExpr<Op, L, R>> : std::true_type {};

template<typename T>
inline constexpr bool is_binary_expr_v = is_binary_expr<T>::value;

/// Check if type is a scale expression
template<typename T>
struct is_scale_expr : std::false_type {};

template<typename E>
struct is_scale_expr<ScaleExpr<E>> : std::true_type {};

template<typename T>
inline constexpr bool is_scale_expr_v = is_scale_expr<T>::value;

/// Get expression depth (for optimization decisions)
template<typename T>
struct expr_depth {
    static constexpr std::size_t value = 1;
};

template<typename Op, typename L, typename R>
struct expr_depth<BinaryExpr<Op, L, R>> {
    static constexpr std::size_t value = 1 +
        std::max(expr_depth<L>::value, expr_depth<R>::value);
};

template<typename E>
struct expr_depth<ScaleExpr<E>> {
    static constexpr std::size_t value = 1 + expr_depth<E>::value;
};

template<typename E>
struct expr_depth<NegateExpr<E>> {
    static constexpr std::size_t value = 1 + expr_depth<E>::value;
};

template<typename T>
inline constexpr std::size_t expr_depth_v = expr_depth<T>::value;

} // namespace detail

// =============================================================================
// Static Assertions
// =============================================================================

namespace detail {

// Verify expression templates compile and work at compile time
static_assert([]() constexpr {
    StaticVector<double, 4> a{1.0, 2.0, 3.0, 4.0};
    StaticVector<double, 4> b{5.0, 6.0, 7.0, 8.0};
    auto expr = a + b;
    return expr[0] == 6.0 && expr[3] == 12.0;
}());

static_assert([]() constexpr {
    StaticVector<double, 4> a{1.0, 2.0, 3.0, 4.0};
    auto expr = 2.0 * a;
    return expr[0] == 2.0 && expr[3] == 8.0;
}());

static_assert([]() constexpr {
    StaticVector<double, 4> a{1.0, 2.0, 3.0, 4.0};
    StaticVector<double, 4> b{1.0, 2.0, 3.0, 4.0};
    auto expr = a - b;
    return expr[0] == 0.0 && expr[3] == 0.0;
}());

// Verify nested expressions work
static_assert([]() constexpr {
    StaticVector<double, 4> a{1.0, 2.0, 3.0, 4.0};
    StaticVector<double, 4> b{1.0, 1.0, 1.0, 1.0};
    auto expr = 2.0 * (a + b);  // nested expression
    return expr[0] == 4.0 && expr[3] == 10.0;
}());

// Verify expression depth tracking
static_assert(expr_depth_v<StaticVector<double, 4>> == 1);
static_assert(expr_depth_v<AddExpr<StaticVector<double, 4>, StaticVector<double, 4>>> == 2);

} // namespace detail

} // namespace pulsim::v2
