#pragma once

// =============================================================================
// PulsimCore v2 - Numeric Types Foundation
// =============================================================================
// This header provides the foundational numeric types for the simulation engine:
// - Configurable Real type (double/float) via template parameter
// - Index type with configurable width (32/64 bit)
// - StaticVector<T, N> for fixed-size vectors (SIMD-friendly)
// - StaticMatrix<T, Rows, Cols> for small matrices
// - SparsityPattern<N> for compile-time Jacobian patterns
// - Units type for compile-time dimensional analysis
// =============================================================================

#include <array>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <concepts>
#include <cmath>
#include <algorithm>
#include <initializer_list>

namespace pulsim::v2 {

// =============================================================================
// 2.1.1: Real Type Configuration
// =============================================================================

/// Precision policy for selecting floating-point type
enum class Precision {
    Single,  // float  - 32-bit, faster but less accurate
    Double   // double - 64-bit, standard precision
};

/// Type selector for Real based on precision policy
template<Precision P>
struct RealSelector {
    using type = double;
};

template<>
struct RealSelector<Precision::Single> {
    using type = float;
};

/// Template alias for Real type based on precision
template<Precision P>
using RealT = typename RealSelector<P>::type;

/// Concrete Real types
using RealD = RealT<Precision::Double>;
using RealS = RealT<Precision::Single>;

/// Default Real type (double precision) - non-template alias
using Real = RealD;

/// Concept for valid Real types
template<typename T>
concept RealType = std::floating_point<T>;

/// Traits for Real type characteristics
template<RealType T>
struct RealTraits {
    static constexpr T epsilon = std::numeric_limits<T>::epsilon();
    static constexpr T min_normal = std::numeric_limits<T>::min();
    static constexpr T max_value = std::numeric_limits<T>::max();
    static constexpr T infinity = std::numeric_limits<T>::infinity();
    static constexpr int digits = std::numeric_limits<T>::digits;
    static constexpr int digits10 = std::numeric_limits<T>::digits10;

    /// Recommended absolute tolerance for convergence
    static constexpr T default_abstol = (sizeof(T) == 4) ? T{1e-6} : T{1e-9};

    /// Recommended relative tolerance for convergence
    static constexpr T default_reltol = (sizeof(T) == 4) ? T{1e-4} : T{1e-3};
};

// =============================================================================
// 2.1.2: Index Type with Configurable Width
// =============================================================================

/// Index width policy
enum class IndexWidth {
    Narrow,  // int32_t - sufficient for most circuits (up to 2B nodes)
    Wide     // int64_t - for very large circuits
};

/// Type selector for Index based on width policy
template<IndexWidth W>
struct IndexSelector {
    using type = std::int32_t;
    using unsigned_type = std::uint32_t;
};

template<>
struct IndexSelector<IndexWidth::Wide> {
    using type = std::int64_t;
    using unsigned_type = std::uint64_t;
};

/// Template alias for Index type based on width
template<IndexWidth W>
using IndexT = typename IndexSelector<W>::type;

template<IndexWidth W>
using UIndexT = typename IndexSelector<W>::unsigned_type;

/// Concrete Index types
using Index32 = IndexT<IndexWidth::Narrow>;
using Index64 = IndexT<IndexWidth::Wide>;
using UIndex32 = UIndexT<IndexWidth::Narrow>;
using UIndex64 = UIndexT<IndexWidth::Wide>;

/// Default Index type (32-bit signed) - non-template alias
using Index = Index32;

/// Concept for valid Index types
template<typename T>
concept IndexType = std::signed_integral<T>;

/// Special index value for ground node
template<IndexType I>
inline constexpr I ground_node_v = I{-1};

/// Default ground node constant
inline constexpr Index ground_node = Index{-1};

// =============================================================================
// 2.1.3: StaticVector<T, N> - Fixed-Size Vector
// =============================================================================

/// Fixed-size vector for small, compile-time known sizes
/// Optimized for SIMD operations and cache efficiency
template<typename T, std::size_t N>
    requires (N > 0)
class StaticVector {
public:
    using value_type = T;
    using size_type = std::size_t;
    using reference = T&;
    using const_reference = const T&;
    using pointer = T*;
    using const_pointer = const T*;
    using iterator = T*;
    using const_iterator = const T*;

    static constexpr size_type static_size = N;

    // Constructors
    constexpr StaticVector() noexcept : data_{} {}

    constexpr explicit StaticVector(T value) noexcept {
        for (size_type i = 0; i < N; ++i) data_[i] = value;
    }

    constexpr StaticVector(std::initializer_list<T> init) noexcept {
        size_type i = 0;
        for (auto it = init.begin(); it != init.end() && i < N; ++it, ++i) {
            data_[i] = *it;
        }
        for (; i < N; ++i) data_[i] = T{};
    }

    template<typename... Args>
        requires (sizeof...(Args) == N && (std::convertible_to<Args, T> && ...))
    constexpr StaticVector(Args... args) noexcept : data_{static_cast<T>(args)...} {}

    // Element access
    [[nodiscard]] constexpr reference operator[](size_type i) noexcept { return data_[i]; }
    [[nodiscard]] constexpr const_reference operator[](size_type i) const noexcept { return data_[i]; }

    [[nodiscard]] constexpr reference at(size_type i) noexcept { return data_[i]; }
    [[nodiscard]] constexpr const_reference at(size_type i) const noexcept { return data_[i]; }

    [[nodiscard]] constexpr pointer data() noexcept { return data_.data(); }
    [[nodiscard]] constexpr const_pointer data() const noexcept { return data_.data(); }

    [[nodiscard]] constexpr reference front() noexcept { return data_[0]; }
    [[nodiscard]] constexpr const_reference front() const noexcept { return data_[0]; }

    [[nodiscard]] constexpr reference back() noexcept { return data_[N - 1]; }
    [[nodiscard]] constexpr const_reference back() const noexcept { return data_[N - 1]; }

    // Iterators
    [[nodiscard]] constexpr iterator begin() noexcept { return data_.data(); }
    [[nodiscard]] constexpr const_iterator begin() const noexcept { return data_.data(); }
    [[nodiscard]] constexpr const_iterator cbegin() const noexcept { return data_.data(); }

    [[nodiscard]] constexpr iterator end() noexcept { return data_.data() + N; }
    [[nodiscard]] constexpr const_iterator end() const noexcept { return data_.data() + N; }
    [[nodiscard]] constexpr const_iterator cend() const noexcept { return data_.data() + N; }

    // Capacity
    [[nodiscard]] static constexpr size_type size() noexcept { return N; }
    [[nodiscard]] static constexpr bool empty() noexcept { return N == 0; }

    // Operations
    constexpr void fill(T value) noexcept {
        for (size_type i = 0; i < N; ++i) data_[i] = value;
    }

    constexpr void swap(StaticVector& other) noexcept {
        std::swap(data_, other.data_);
    }

    // Arithmetic operations (element-wise)
    [[nodiscard]] constexpr StaticVector operator+(const StaticVector& rhs) const noexcept {
        StaticVector result;
        for (size_type i = 0; i < N; ++i) result[i] = data_[i] + rhs[i];
        return result;
    }

    [[nodiscard]] constexpr StaticVector operator-(const StaticVector& rhs) const noexcept {
        StaticVector result;
        for (size_type i = 0; i < N; ++i) result[i] = data_[i] - rhs[i];
        return result;
    }

    [[nodiscard]] constexpr StaticVector operator*(T scalar) const noexcept {
        StaticVector result;
        for (size_type i = 0; i < N; ++i) result[i] = data_[i] * scalar;
        return result;
    }

    [[nodiscard]] constexpr StaticVector operator/(T scalar) const noexcept {
        StaticVector result;
        for (size_type i = 0; i < N; ++i) result[i] = data_[i] / scalar;
        return result;
    }

    constexpr StaticVector& operator+=(const StaticVector& rhs) noexcept {
        for (size_type i = 0; i < N; ++i) data_[i] += rhs[i];
        return *this;
    }

    constexpr StaticVector& operator-=(const StaticVector& rhs) noexcept {
        for (size_type i = 0; i < N; ++i) data_[i] -= rhs[i];
        return *this;
    }

    constexpr StaticVector& operator*=(T scalar) noexcept {
        for (size_type i = 0; i < N; ++i) data_[i] *= scalar;
        return *this;
    }

    constexpr StaticVector& operator/=(T scalar) noexcept {
        for (size_type i = 0; i < N; ++i) data_[i] /= scalar;
        return *this;
    }

    [[nodiscard]] constexpr StaticVector operator-() const noexcept {
        StaticVector result;
        for (size_type i = 0; i < N; ++i) result[i] = -data_[i];
        return result;
    }

    // Vector operations
    [[nodiscard]] constexpr T dot(const StaticVector& rhs) const noexcept {
        T sum = T{};
        for (size_type i = 0; i < N; ++i) sum += data_[i] * rhs[i];
        return sum;
    }

    [[nodiscard]] constexpr T squared_norm() const noexcept {
        return dot(*this);
    }

    [[nodiscard]] T norm() const noexcept {
        return std::sqrt(squared_norm());
    }

    [[nodiscard]] StaticVector normalized() const noexcept {
        T n = norm();
        if (n > T{0}) return *this / n;
        return *this;
    }

    [[nodiscard]] constexpr T sum() const noexcept {
        T s = T{};
        for (size_type i = 0; i < N; ++i) s += data_[i];
        return s;
    }

    [[nodiscard]] constexpr T max_element() const noexcept {
        T m = data_[0];
        for (size_type i = 1; i < N; ++i) if (data_[i] > m) m = data_[i];
        return m;
    }

    [[nodiscard]] constexpr T min_element() const noexcept {
        T m = data_[0];
        for (size_type i = 1; i < N; ++i) if (data_[i] < m) m = data_[i];
        return m;
    }

    [[nodiscard]] T max_abs() const noexcept {
        T m = std::abs(data_[0]);
        for (size_type i = 1; i < N; ++i) {
            T a = std::abs(data_[i]);
            if (a > m) m = a;
        }
        return m;
    }

    // Comparison
    [[nodiscard]] constexpr bool operator==(const StaticVector& rhs) const noexcept {
        for (size_type i = 0; i < N; ++i) {
            if (data_[i] != rhs[i]) return false;
        }
        return true;
    }

private:
    alignas(16) std::array<T, N> data_;
};

// Scalar * Vector
template<typename T, std::size_t N>
[[nodiscard]] constexpr StaticVector<T, N> operator*(T scalar, const StaticVector<T, N>& vec) noexcept {
    return vec * scalar;
}

// Convenience type aliases
template<typename T> using Vec2 = StaticVector<T, 2>;
template<typename T> using Vec3 = StaticVector<T, 3>;
template<typename T> using Vec4 = StaticVector<T, 4>;

using Vec2d = Vec2<double>;
using Vec3d = Vec3<double>;
using Vec4d = Vec4<double>;
using Vec2f = Vec2<float>;
using Vec3f = Vec3<float>;
using Vec4f = Vec4<float>;

// =============================================================================
// 2.1.4: StaticMatrix<T, Rows, Cols> - Fixed-Size Matrix
// =============================================================================

/// Fixed-size matrix stored in row-major order
template<typename T, std::size_t Rows, std::size_t Cols>
    requires (Rows > 0 && Cols > 0)
class StaticMatrix {
public:
    using value_type = T;
    using size_type = std::size_t;
    using reference = T&;
    using const_reference = const T&;

    static constexpr size_type rows = Rows;
    static constexpr size_type cols = Cols;
    static constexpr size_type total_size = Rows * Cols;

    // Constructors
    constexpr StaticMatrix() noexcept : data_{} {}

    constexpr explicit StaticMatrix(T value) noexcept {
        for (size_type i = 0; i < total_size; ++i) data_[i] = value;
    }

    constexpr StaticMatrix(std::initializer_list<std::initializer_list<T>> init) noexcept {
        size_type r = 0;
        for (auto row_it = init.begin(); row_it != init.end() && r < Rows; ++row_it, ++r) {
            size_type c = 0;
            for (auto col_it = row_it->begin(); col_it != row_it->end() && c < Cols; ++col_it, ++c) {
                data_[r * Cols + c] = *col_it;
            }
            for (; c < Cols; ++c) data_[r * Cols + c] = T{};
        }
        for (; r < Rows; ++r) {
            for (size_type c = 0; c < Cols; ++c) data_[r * Cols + c] = T{};
        }
    }

    // Element access
    [[nodiscard]] constexpr reference operator()(size_type r, size_type c) noexcept {
        return data_[r * Cols + c];
    }

    [[nodiscard]] constexpr const_reference operator()(size_type r, size_type c) const noexcept {
        return data_[r * Cols + c];
    }

    [[nodiscard]] constexpr T* data() noexcept { return data_.data(); }
    [[nodiscard]] constexpr const T* data() const noexcept { return data_.data(); }

    // Row and column access
    [[nodiscard]] constexpr StaticVector<T, Cols> row(size_type r) const noexcept {
        StaticVector<T, Cols> result;
        for (size_type c = 0; c < Cols; ++c) result[c] = data_[r * Cols + c];
        return result;
    }

    [[nodiscard]] constexpr StaticVector<T, Rows> col(size_type c) const noexcept {
        StaticVector<T, Rows> result;
        for (size_type r = 0; r < Rows; ++r) result[r] = data_[r * Cols + c];
        return result;
    }

    constexpr void set_row(size_type r, const StaticVector<T, Cols>& v) noexcept {
        for (size_type c = 0; c < Cols; ++c) data_[r * Cols + c] = v[c];
    }

    constexpr void set_col(size_type c, const StaticVector<T, Rows>& v) noexcept {
        for (size_type r = 0; r < Rows; ++r) data_[r * Cols + c] = v[r];
    }

    // Operations
    constexpr void fill(T value) noexcept {
        for (size_type i = 0; i < total_size; ++i) data_[i] = value;
    }

    constexpr void set_zero() noexcept { fill(T{}); }

    /// Set diagonal to value (only for square matrices)
    constexpr void set_diagonal(T value) noexcept requires (Rows == Cols) {
        for (size_type i = 0; i < Rows; ++i) data_[i * Cols + i] = value;
    }

    /// Create identity matrix
    [[nodiscard]] static constexpr StaticMatrix identity() noexcept requires (Rows == Cols) {
        StaticMatrix result;
        for (size_type i = 0; i < Rows; ++i) result(i, i) = T{1};
        return result;
    }

    // Arithmetic operations
    [[nodiscard]] constexpr StaticMatrix operator+(const StaticMatrix& rhs) const noexcept {
        StaticMatrix result;
        for (size_type i = 0; i < total_size; ++i) result.data_[i] = data_[i] + rhs.data_[i];
        return result;
    }

    [[nodiscard]] constexpr StaticMatrix operator-(const StaticMatrix& rhs) const noexcept {
        StaticMatrix result;
        for (size_type i = 0; i < total_size; ++i) result.data_[i] = data_[i] - rhs.data_[i];
        return result;
    }

    [[nodiscard]] constexpr StaticMatrix operator*(T scalar) const noexcept {
        StaticMatrix result;
        for (size_type i = 0; i < total_size; ++i) result.data_[i] = data_[i] * scalar;
        return result;
    }

    constexpr StaticMatrix& operator+=(const StaticMatrix& rhs) noexcept {
        for (size_type i = 0; i < total_size; ++i) data_[i] += rhs.data_[i];
        return *this;
    }

    constexpr StaticMatrix& operator-=(const StaticMatrix& rhs) noexcept {
        for (size_type i = 0; i < total_size; ++i) data_[i] -= rhs.data_[i];
        return *this;
    }

    constexpr StaticMatrix& operator*=(T scalar) noexcept {
        for (size_type i = 0; i < total_size; ++i) data_[i] *= scalar;
        return *this;
    }

    // Matrix-vector multiplication: A * x
    [[nodiscard]] constexpr StaticVector<T, Rows> operator*(const StaticVector<T, Cols>& x) const noexcept {
        StaticVector<T, Rows> result;
        for (size_type r = 0; r < Rows; ++r) {
            T sum = T{};
            for (size_type c = 0; c < Cols; ++c) {
                sum += data_[r * Cols + c] * x[c];
            }
            result[r] = sum;
        }
        return result;
    }

    // Matrix-matrix multiplication
    template<std::size_t OtherCols>
    [[nodiscard]] constexpr StaticMatrix<T, Rows, OtherCols> operator*(
        const StaticMatrix<T, Cols, OtherCols>& rhs) const noexcept {
        StaticMatrix<T, Rows, OtherCols> result;
        for (size_type r = 0; r < Rows; ++r) {
            for (size_type c = 0; c < OtherCols; ++c) {
                T sum = T{};
                for (size_type k = 0; k < Cols; ++k) {
                    sum += data_[r * Cols + k] * rhs(k, c);
                }
                result(r, c) = sum;
            }
        }
        return result;
    }

    // Transpose
    [[nodiscard]] constexpr StaticMatrix<T, Cols, Rows> transpose() const noexcept {
        StaticMatrix<T, Cols, Rows> result;
        for (size_type r = 0; r < Rows; ++r) {
            for (size_type c = 0; c < Cols; ++c) {
                result(c, r) = data_[r * Cols + c];
            }
        }
        return result;
    }

    // Trace (only for square matrices)
    [[nodiscard]] constexpr T trace() const noexcept requires (Rows == Cols) {
        T sum = T{};
        for (size_type i = 0; i < Rows; ++i) sum += data_[i * Cols + i];
        return sum;
    }

    // Determinant for 2x2 and 3x3
    [[nodiscard]] constexpr T determinant() const noexcept requires (Rows == Cols && Rows <= 3) {
        if constexpr (Rows == 1) {
            return data_[0];
        } else if constexpr (Rows == 2) {
            return data_[0] * data_[3] - data_[1] * data_[2];
        } else { // Rows == 3
            return data_[0] * (data_[4] * data_[8] - data_[5] * data_[7])
                 - data_[1] * (data_[3] * data_[8] - data_[5] * data_[6])
                 + data_[2] * (data_[3] * data_[7] - data_[4] * data_[6]);
        }
    }

    // Inverse for 2x2
    [[nodiscard]] constexpr StaticMatrix inverse() const noexcept requires (Rows == 2 && Cols == 2) {
        T det = determinant();
        StaticMatrix result;
        result(0, 0) =  data_[3] / det;
        result(0, 1) = -data_[1] / det;
        result(1, 0) = -data_[2] / det;
        result(1, 1) =  data_[0] / det;
        return result;
    }

    // Frobenius norm
    [[nodiscard]] T frobenius_norm() const noexcept {
        T sum = T{};
        for (size_type i = 0; i < total_size; ++i) sum += data_[i] * data_[i];
        return std::sqrt(sum);
    }

    // Comparison
    [[nodiscard]] constexpr bool operator==(const StaticMatrix& rhs) const noexcept {
        for (size_type i = 0; i < total_size; ++i) {
            if (data_[i] != rhs.data_[i]) return false;
        }
        return true;
    }

private:
    alignas(16) std::array<T, Rows * Cols> data_;
};

// Scalar * Matrix
template<typename T, std::size_t Rows, std::size_t Cols>
[[nodiscard]] constexpr StaticMatrix<T, Rows, Cols> operator*(
    T scalar, const StaticMatrix<T, Rows, Cols>& mat) noexcept {
    return mat * scalar;
}

// Convenience type aliases
template<typename T> using Mat2 = StaticMatrix<T, 2, 2>;
template<typename T> using Mat3 = StaticMatrix<T, 3, 3>;
template<typename T> using Mat4 = StaticMatrix<T, 4, 4>;

using Mat2d = Mat2<double>;
using Mat3d = Mat3<double>;
using Mat4d = Mat4<double>;
using Mat2f = Mat2<float>;
using Mat3f = Mat3<float>;
using Mat4f = Mat4<float>;

// =============================================================================
// 2.1.5: SparsityPattern<N> - Compile-Time Sparsity Patterns
// =============================================================================

/// Entry in a sparsity pattern
struct PatternEntry {
    std::int32_t row;
    std::int32_t col;

    constexpr PatternEntry() noexcept : row(-1), col(-1) {}
    constexpr PatternEntry(std::int32_t r, std::int32_t c) noexcept : row(r), col(c) {}

    [[nodiscard]] constexpr bool operator==(const PatternEntry& other) const noexcept {
        return row == other.row && col == other.col;
    }

    [[nodiscard]] constexpr bool operator<(const PatternEntry& other) const noexcept {
        return (row < other.row) || (row == other.row && col < other.col);
    }

    [[nodiscard]] constexpr bool valid() const noexcept {
        return row >= 0 && col >= 0;
    }
};

/// Compile-time sparsity pattern for Jacobian matrices
template<std::size_t MaxEntries>
class SparsityPattern {
public:
    using size_type = std::size_t;

    constexpr SparsityPattern() noexcept : count_(0), entries_{} {}

    constexpr SparsityPattern(std::initializer_list<PatternEntry> init) noexcept : count_(0), entries_{} {
        for (auto it = init.begin(); it != init.end() && count_ < MaxEntries; ++it) {
            entries_[count_++] = *it;
        }
    }

    template<std::size_t N>
    constexpr SparsityPattern(const std::array<PatternEntry, N>& arr) noexcept : count_(N), entries_{} {
        static_assert(N <= MaxEntries, "Array too large for SparsityPattern");
        for (size_type i = 0; i < N; ++i) entries_[i] = arr[i];
    }

    // Add entry
    constexpr bool add(std::int32_t row, std::int32_t col) noexcept {
        if (count_ >= MaxEntries) return false;
        entries_[count_++] = PatternEntry{row, col};
        return true;
    }

    constexpr bool add(PatternEntry e) noexcept {
        if (count_ >= MaxEntries) return false;
        entries_[count_++] = e;
        return true;
    }

    // Access
    [[nodiscard]] constexpr const PatternEntry& operator[](size_type i) const noexcept {
        return entries_[i];
    }

    [[nodiscard]] constexpr size_type size() const noexcept { return count_; }
    [[nodiscard]] static constexpr size_type capacity() noexcept { return MaxEntries; }
    [[nodiscard]] constexpr bool empty() const noexcept { return count_ == 0; }

    // Iterators
    [[nodiscard]] constexpr const PatternEntry* begin() const noexcept { return entries_.data(); }
    [[nodiscard]] constexpr const PatternEntry* end() const noexcept { return entries_.data() + count_; }

    // Check if pattern contains an entry
    [[nodiscard]] constexpr bool contains(std::int32_t row, std::int32_t col) const noexcept {
        for (size_type i = 0; i < count_; ++i) {
            if (entries_[i].row == row && entries_[i].col == col) return true;
        }
        return false;
    }

    // Merge with another pattern
    template<std::size_t OtherMax>
    [[nodiscard]] constexpr SparsityPattern<MaxEntries + OtherMax> merge(
        const SparsityPattern<OtherMax>& other) const noexcept {
        SparsityPattern<MaxEntries + OtherMax> result;
        for (size_type i = 0; i < count_; ++i) result.add(entries_[i]);
        for (size_type i = 0; i < other.size(); ++i) {
            if (!result.contains(other[i].row, other[i].col)) {
                result.add(other[i]);
            }
        }
        return result;
    }

    // Get bounding box
    [[nodiscard]] constexpr std::pair<std::int32_t, std::int32_t> max_indices() const noexcept {
        std::int32_t max_r = 0, max_c = 0;
        for (size_type i = 0; i < count_; ++i) {
            if (entries_[i].row > max_r) max_r = entries_[i].row;
            if (entries_[i].col > max_c) max_c = entries_[i].col;
        }
        return {max_r, max_c};
    }

private:
    size_type count_;
    std::array<PatternEntry, MaxEntries> entries_;
};

/// Create a dense 2x2 pattern for two-terminal devices
[[nodiscard]] constexpr SparsityPattern<4> make_2x2_pattern() noexcept {
    return SparsityPattern<4>{{
        PatternEntry{0, 0}, PatternEntry{0, 1},
        PatternEntry{1, 0}, PatternEntry{1, 1}
    }};
}

/// Create a dense 3x3 pattern for three-terminal devices
[[nodiscard]] constexpr SparsityPattern<9> make_3x3_pattern() noexcept {
    return SparsityPattern<9>{{
        PatternEntry{0, 0}, PatternEntry{0, 1}, PatternEntry{0, 2},
        PatternEntry{1, 0}, PatternEntry{1, 1}, PatternEntry{1, 2},
        PatternEntry{2, 0}, PatternEntry{2, 1}, PatternEntry{2, 2}
    }};
}

// =============================================================================
// 2.1.6: Units Type for Dimensional Analysis
// =============================================================================

/// Compile-time dimensional analysis using phantom types
/// Dimensions: Length(m), Mass(kg), Time(s), Current(A), Temperature(K), Voltage(V), Power(W)
template<int Length, int Mass, int Time, int Current, int Temperature>
struct Dimensions {
    static constexpr int length = Length;
    static constexpr int mass = Mass;
    static constexpr int time = Time;
    static constexpr int current = Current;
    static constexpr int temperature = Temperature;
};

// Common dimension types
using Dimensionless = Dimensions<0, 0, 0, 0, 0>;
using DimLength     = Dimensions<1, 0, 0, 0, 0>;  // m
using DimMass       = Dimensions<0, 1, 0, 0, 0>;  // kg
using DimTime       = Dimensions<0, 0, 1, 0, 0>;  // s
using DimCurrent    = Dimensions<0, 0, 0, 1, 0>;  // A
using DimTemp       = Dimensions<0, 0, 0, 0, 1>;  // K

// Derived electrical dimensions
using DimVoltage    = Dimensions<2, 1, -3, -1, 0>;   // V = m²·kg·s⁻³·A⁻¹
using DimResistance = Dimensions<2, 1, -3, -2, 0>;   // Ω = m²·kg·s⁻³·A⁻²
using DimConductance= Dimensions<-2, -1, 3, 2, 0>;   // S = Ω⁻¹
using DimCapacitance= Dimensions<-2, -1, 4, 2, 0>;   // F = s⁴·A²·m⁻²·kg⁻¹
using DimInductance = Dimensions<2, 1, -2, -2, 0>;   // H = m²·kg·s⁻²·A⁻²
using DimPower      = Dimensions<2, 1, -3, 0, 0>;    // W = m²·kg·s⁻³
using DimEnergy     = Dimensions<2, 1, -2, 0, 0>;    // J = m²·kg·s⁻²
using DimFrequency  = Dimensions<0, 0, -1, 0, 0>;    // Hz = s⁻¹
using DimCharge     = Dimensions<0, 0, 1, 1, 0>;     // C = A·s

/// Quantity type with compile-time dimensional checking
template<typename T, typename Dim>
class Quantity {
public:
    using value_type = T;
    using dimension = Dim;

    constexpr Quantity() noexcept : value_(T{}) {}
    constexpr explicit Quantity(T value) noexcept : value_(value) {}

    [[nodiscard]] constexpr T value() const noexcept { return value_; }
    [[nodiscard]] constexpr T& value() noexcept { return value_; }

    // Same-dimension operations
    [[nodiscard]] constexpr Quantity operator+(const Quantity& rhs) const noexcept {
        return Quantity(value_ + rhs.value_);
    }

    [[nodiscard]] constexpr Quantity operator-(const Quantity& rhs) const noexcept {
        return Quantity(value_ - rhs.value_);
    }

    [[nodiscard]] constexpr Quantity operator-() const noexcept {
        return Quantity(-value_);
    }

    constexpr Quantity& operator+=(const Quantity& rhs) noexcept {
        value_ += rhs.value_;
        return *this;
    }

    constexpr Quantity& operator-=(const Quantity& rhs) noexcept {
        value_ -= rhs.value_;
        return *this;
    }

    // Scalar multiplication
    [[nodiscard]] constexpr Quantity operator*(T scalar) const noexcept {
        return Quantity(value_ * scalar);
    }

    [[nodiscard]] constexpr Quantity operator/(T scalar) const noexcept {
        return Quantity(value_ / scalar);
    }

    // Comparison
    [[nodiscard]] constexpr bool operator==(const Quantity& rhs) const noexcept {
        return value_ == rhs.value_;
    }

    [[nodiscard]] constexpr bool operator<(const Quantity& rhs) const noexcept {
        return value_ < rhs.value_;
    }

    [[nodiscard]] constexpr bool operator<=(const Quantity& rhs) const noexcept {
        return value_ <= rhs.value_;
    }

    [[nodiscard]] constexpr bool operator>(const Quantity& rhs) const noexcept {
        return value_ > rhs.value_;
    }

    [[nodiscard]] constexpr bool operator>=(const Quantity& rhs) const noexcept {
        return value_ >= rhs.value_;
    }

private:
    T value_;
};

// Scalar * Quantity
template<typename T, typename Dim>
[[nodiscard]] constexpr Quantity<T, Dim> operator*(T scalar, const Quantity<T, Dim>& q) noexcept {
    return q * scalar;
}

// Quantity * Quantity -> new dimensions
template<typename T, typename Dim1, typename Dim2>
[[nodiscard]] constexpr auto operator*(const Quantity<T, Dim1>& lhs, const Quantity<T, Dim2>& rhs) noexcept {
    using ResultDim = Dimensions<
        Dim1::length + Dim2::length,
        Dim1::mass + Dim2::mass,
        Dim1::time + Dim2::time,
        Dim1::current + Dim2::current,
        Dim1::temperature + Dim2::temperature
    >;
    return Quantity<T, ResultDim>(lhs.value() * rhs.value());
}

// Quantity / Quantity -> new dimensions
template<typename T, typename Dim1, typename Dim2>
[[nodiscard]] constexpr auto operator/(const Quantity<T, Dim1>& lhs, const Quantity<T, Dim2>& rhs) noexcept {
    using ResultDim = Dimensions<
        Dim1::length - Dim2::length,
        Dim1::mass - Dim2::mass,
        Dim1::time - Dim2::time,
        Dim1::current - Dim2::current,
        Dim1::temperature - Dim2::temperature
    >;
    return Quantity<T, ResultDim>(lhs.value() / rhs.value());
}

// Type aliases for common electrical quantities
template<typename T = double>
using Voltage = Quantity<T, DimVoltage>;

template<typename T = double>
using Current = Quantity<T, DimCurrent>;

template<typename T = double>
using Resistance = Quantity<T, DimResistance>;

template<typename T = double>
using Conductance = Quantity<T, DimConductance>;

template<typename T = double>
using Capacitance = Quantity<T, DimCapacitance>;

template<typename T = double>
using Inductance = Quantity<T, DimInductance>;

template<typename T = double>
using Power = Quantity<T, DimPower>;

template<typename T = double>
using Energy = Quantity<T, DimEnergy>;

template<typename T = double>
using Frequency = Quantity<T, DimFrequency>;

template<typename T = double>
using Time = Quantity<T, DimTime>;

template<typename T = double>
using Temperature = Quantity<T, DimTemp>;

// User-defined literals for quantities (in inline namespace)
inline namespace literals {
    // Voltage
    constexpr Voltage<double> operator""_V(long double v) { return Voltage<double>(static_cast<double>(v)); }
    constexpr Voltage<double> operator""_mV(long double v) { return Voltage<double>(static_cast<double>(v) * 1e-3); }
    constexpr Voltage<double> operator""_kV(long double v) { return Voltage<double>(static_cast<double>(v) * 1e3); }

    // Current
    constexpr Current<double> operator""_A(long double v) { return Current<double>(static_cast<double>(v)); }
    constexpr Current<double> operator""_mA(long double v) { return Current<double>(static_cast<double>(v) * 1e-3); }
    constexpr Current<double> operator""_uA(long double v) { return Current<double>(static_cast<double>(v) * 1e-6); }

    // Resistance
    constexpr Resistance<double> operator""_Ohm(long double v) { return Resistance<double>(static_cast<double>(v)); }
    constexpr Resistance<double> operator""_kOhm(long double v) { return Resistance<double>(static_cast<double>(v) * 1e3); }
    constexpr Resistance<double> operator""_MOhm(long double v) { return Resistance<double>(static_cast<double>(v) * 1e6); }

    // Capacitance
    constexpr Capacitance<double> operator""_F(long double v) { return Capacitance<double>(static_cast<double>(v)); }
    constexpr Capacitance<double> operator""_mF(long double v) { return Capacitance<double>(static_cast<double>(v) * 1e-3); }
    constexpr Capacitance<double> operator""_uF(long double v) { return Capacitance<double>(static_cast<double>(v) * 1e-6); }
    constexpr Capacitance<double> operator""_nF(long double v) { return Capacitance<double>(static_cast<double>(v) * 1e-9); }
    constexpr Capacitance<double> operator""_pF(long double v) { return Capacitance<double>(static_cast<double>(v) * 1e-12); }

    // Inductance
    constexpr Inductance<double> operator""_H(long double v) { return Inductance<double>(static_cast<double>(v)); }
    constexpr Inductance<double> operator""_mH(long double v) { return Inductance<double>(static_cast<double>(v) * 1e-3); }
    constexpr Inductance<double> operator""_uH(long double v) { return Inductance<double>(static_cast<double>(v) * 1e-6); }
    constexpr Inductance<double> operator""_nH(long double v) { return Inductance<double>(static_cast<double>(v) * 1e-9); }

    // Time
    constexpr Time<double> operator""_s(long double v) { return Time<double>(static_cast<double>(v)); }
    constexpr Time<double> operator""_ms(long double v) { return Time<double>(static_cast<double>(v) * 1e-3); }
    constexpr Time<double> operator""_us(long double v) { return Time<double>(static_cast<double>(v) * 1e-6); }
    constexpr Time<double> operator""_ns(long double v) { return Time<double>(static_cast<double>(v) * 1e-9); }

    // Frequency
    constexpr Frequency<double> operator""_Hz(long double v) { return Frequency<double>(static_cast<double>(v)); }
    constexpr Frequency<double> operator""_kHz(long double v) { return Frequency<double>(static_cast<double>(v) * 1e3); }
    constexpr Frequency<double> operator""_MHz(long double v) { return Frequency<double>(static_cast<double>(v) * 1e6); }
    constexpr Frequency<double> operator""_GHz(long double v) { return Frequency<double>(static_cast<double>(v) * 1e9); }

    // Power
    constexpr Power<double> operator""_W(long double v) { return Power<double>(static_cast<double>(v)); }
    constexpr Power<double> operator""_mW(long double v) { return Power<double>(static_cast<double>(v) * 1e-3); }
    constexpr Power<double> operator""_kW(long double v) { return Power<double>(static_cast<double>(v) * 1e3); }
}

// =============================================================================
// Static Assertions for Type System
// =============================================================================

namespace detail {
    // Verify Real type traits
    static_assert(RealTraits<double>::default_abstol == 1e-9);
    static_assert(RealTraits<float>::default_abstol == 1e-6f);

    // Verify StaticVector operations
    static_assert(StaticVector<int, 3>{1, 2, 3}.size() == 3);
    static_assert(StaticVector<int, 3>{1, 2, 3}[0] == 1);
    static_assert((StaticVector<int, 2>{1, 2} + StaticVector<int, 2>{3, 4})[0] == 4);
    static_assert(StaticVector<int, 3>{1, 2, 3}.dot(StaticVector<int, 3>{1, 1, 1}) == 6);

    // Verify StaticMatrix operations
    static_assert(StaticMatrix<int, 2, 2>::identity()(0, 0) == 1);
    static_assert(StaticMatrix<int, 2, 2>::identity()(0, 1) == 0);

    // Verify SparsityPattern
    static_assert(make_2x2_pattern().size() == 4);
    static_assert(make_3x3_pattern().size() == 9);

    // Verify dimensional analysis: V = I * R
    static_assert(std::is_same_v<
        decltype(std::declval<Current<>>() * std::declval<Resistance<>>()),
        Voltage<>
    >);

    // Verify dimensional analysis: P = V * I
    static_assert(std::is_same_v<
        decltype(std::declval<Voltage<>>() * std::declval<Current<>>()),
        Power<>
    >);
}

// =============================================================================
// 2.1.7: Normalization/Scaling Helpers for Mixed Units
// =============================================================================
// In MNA systems, we solve for both voltages (typically 0-1000V) and currents
// (typically 1e-9 to 1000A). This mismatch can cause conditioning issues.
// These helpers provide scaling/normalization to stabilize solvers.

/// Scaling factors for normalizing MNA variables
template<typename T = double>
struct ScalingFactors {
    T voltage_scale = T{1.0};    // Scale factor for voltages (default: 1V)
    T current_scale = T{1.0};    // Scale factor for currents (default: 1A)
    T time_scale = T{1.0};       // Scale factor for time (default: 1s)
    T conductance_scale = T{1.0}; // Derived: current_scale / voltage_scale

    /// Constructor with automatic conductance calculation
    constexpr ScalingFactors(T v_scale = T{1.0}, T i_scale = T{1.0}, T t_scale = T{1.0}) noexcept
        : voltage_scale(v_scale)
        , current_scale(i_scale)
        , time_scale(t_scale)
        , conductance_scale(i_scale / v_scale) {}

    /// Create scaling factors for typical power electronics (high voltage, low current)
    [[nodiscard]] static constexpr ScalingFactors power_electronics() noexcept {
        return ScalingFactors(T{100.0}, T{1.0}, T{1e-6});  // 100V, 1A, 1μs base
    }

    /// Create scaling factors for signal-level circuits (low voltage, low current)
    [[nodiscard]] static constexpr ScalingFactors signal_level() noexcept {
        return ScalingFactors(T{1.0}, T{1e-3}, T{1e-9});  // 1V, 1mA, 1ns base
    }

    /// Create scaling factors for high-power systems (high voltage, high current)
    [[nodiscard]] static constexpr ScalingFactors high_power() noexcept {
        return ScalingFactors(T{1000.0}, T{100.0}, T{1e-3});  // 1kV, 100A, 1ms base
    }

    /// Create scaling factors from circuit characteristics
    [[nodiscard]] static constexpr ScalingFactors from_circuit(
        T max_voltage, T max_current, T min_timestep) noexcept {
        // Use magnitude-based scaling
        T v_scale = max_voltage > T{0} ? max_voltage : T{1.0};
        T i_scale = max_current > T{0} ? max_current : T{1.0};
        T t_scale = min_timestep > T{0} ? min_timestep : T{1e-6};
        return ScalingFactors(v_scale, i_scale, t_scale);
    }
};

/// Variable normalizer for MNA systems
/// Converts between physical units and normalized (dimensionless) values
template<typename T = double>
class VariableNormalizer {
public:
    using value_type = T;

    constexpr VariableNormalizer() noexcept = default;
    constexpr explicit VariableNormalizer(const ScalingFactors<T>& factors) noexcept
        : factors_(factors) {}

    // Normalize (physical -> normalized)
    [[nodiscard]] constexpr T normalize_voltage(T v) const noexcept {
        return v / factors_.voltage_scale;
    }

    [[nodiscard]] constexpr T normalize_current(T i) const noexcept {
        return i / factors_.current_scale;
    }

    [[nodiscard]] constexpr T normalize_conductance(T g) const noexcept {
        return g / factors_.conductance_scale;
    }

    [[nodiscard]] constexpr T normalize_resistance(T r) const noexcept {
        return r * factors_.conductance_scale;
    }

    [[nodiscard]] constexpr T normalize_capacitance(T c) const noexcept {
        return c * factors_.voltage_scale / (factors_.current_scale * factors_.time_scale);
    }

    [[nodiscard]] constexpr T normalize_inductance(T l) const noexcept {
        return l * factors_.current_scale / (factors_.voltage_scale * factors_.time_scale);
    }

    [[nodiscard]] constexpr T normalize_time(T t) const noexcept {
        return t / factors_.time_scale;
    }

    // Denormalize (normalized -> physical)
    [[nodiscard]] constexpr T denormalize_voltage(T v_norm) const noexcept {
        return v_norm * factors_.voltage_scale;
    }

    [[nodiscard]] constexpr T denormalize_current(T i_norm) const noexcept {
        return i_norm * factors_.current_scale;
    }

    [[nodiscard]] constexpr T denormalize_conductance(T g_norm) const noexcept {
        return g_norm * factors_.conductance_scale;
    }

    [[nodiscard]] constexpr T denormalize_resistance(T r_norm) const noexcept {
        return r_norm / factors_.conductance_scale;
    }

    [[nodiscard]] constexpr T denormalize_time(T t_norm) const noexcept {
        return t_norm * factors_.time_scale;
    }

    /// Get the underlying scaling factors
    [[nodiscard]] constexpr const ScalingFactors<T>& factors() const noexcept {
        return factors_;
    }

    /// Check if normalization is active (not identity scaling)
    [[nodiscard]] constexpr bool is_active() const noexcept {
        return factors_.voltage_scale != T{1.0} ||
               factors_.current_scale != T{1.0} ||
               factors_.time_scale != T{1.0};
    }

private:
    ScalingFactors<T> factors_;
};

/// Weighted norm calculator for mixed voltage/current convergence checking
/// Uses different tolerances for voltages and currents based on typical magnitudes
template<typename T = double>
class WeightedNorm {
public:
    using value_type = T;

    /// Tolerance configuration
    struct Tolerances {
        T abstol_v = T{1e-9};   // Absolute tolerance for voltages (V)
        T reltol_v = T{1e-3};   // Relative tolerance for voltages
        T abstol_i = T{1e-12};  // Absolute tolerance for currents (A)
        T reltol_i = T{1e-3};   // Relative tolerance for currents
    };

    constexpr WeightedNorm() noexcept = default;
    constexpr explicit WeightedNorm(const Tolerances& tol) noexcept : tol_(tol) {}

    /// Compute weighted infinity norm for convergence check
    /// Returns max(|delta_v| / tol_v, |delta_i| / tol_i) across all variables
    template<typename VectorType>
    [[nodiscard]] T compute_inf_norm(
        const VectorType& delta,
        const VectorType& solution,
        Index num_nodes,
        Index num_branches) const {
        T max_norm = T{0};

        // Check voltage nodes
        for (Index i = 0; i < num_nodes; ++i) {
            T tol = tol_.abstol_v + tol_.reltol_v * std::abs(solution[i]);
            T norm = std::abs(delta[i]) / tol;
            max_norm = std::max(max_norm, norm);
        }

        // Check current branches
        for (Index i = num_nodes; i < num_nodes + num_branches; ++i) {
            T tol = tol_.abstol_i + tol_.reltol_i * std::abs(solution[i]);
            T norm = std::abs(delta[i]) / tol;
            max_norm = std::max(max_norm, norm);
        }

        return max_norm;
    }

    /// Check if solution has converged
    template<typename VectorType>
    [[nodiscard]] bool has_converged(
        const VectorType& delta,
        const VectorType& solution,
        Index num_nodes,
        Index num_branches) const {
        return compute_inf_norm(delta, solution, num_nodes, num_branches) <= T{1.0};
    }

    /// Get/set tolerances
    [[nodiscard]] constexpr const Tolerances& tolerances() const noexcept { return tol_; }
    constexpr void set_tolerances(const Tolerances& tol) noexcept { tol_ = tol; }

private:
    Tolerances tol_;
};

/// Per-unit (p.u.) system converter for power systems analysis
/// Useful for power electronics where values span many orders of magnitude
template<typename T = double>
class PerUnitSystem {
public:
    using value_type = T;

    /// Base quantities for per-unit conversion
    struct BaseQuantities {
        T S_base = T{1e3};      // Base apparent power (VA), default 1kVA
        T V_base = T{1.0};      // Base voltage (V)
        T f_base = T{50.0};     // Base frequency (Hz)

        // Derived base quantities
        [[nodiscard]] constexpr T I_base() const noexcept { return S_base / V_base; }
        [[nodiscard]] constexpr T Z_base() const noexcept { return V_base * V_base / S_base; }
        [[nodiscard]] constexpr T Y_base() const noexcept { return S_base / (V_base * V_base); }
        [[nodiscard]] constexpr T L_base() const noexcept { return Z_base() / (T{2.0} * T{3.14159265358979323846} * f_base); }
        [[nodiscard]] constexpr T C_base() const noexcept { return T{1.0} / (Z_base() * T{2.0} * T{3.14159265358979323846} * f_base); }
    };

    constexpr PerUnitSystem() noexcept = default;
    constexpr explicit PerUnitSystem(const BaseQuantities& base) noexcept : base_(base) {}

    // To per-unit
    [[nodiscard]] constexpr T to_pu_voltage(T v) const noexcept { return v / base_.V_base; }
    [[nodiscard]] constexpr T to_pu_current(T i) const noexcept { return i / base_.I_base(); }
    [[nodiscard]] constexpr T to_pu_power(T p) const noexcept { return p / base_.S_base; }
    [[nodiscard]] constexpr T to_pu_impedance(T z) const noexcept { return z / base_.Z_base(); }
    [[nodiscard]] constexpr T to_pu_admittance(T y) const noexcept { return y / base_.Y_base(); }

    // From per-unit
    [[nodiscard]] constexpr T from_pu_voltage(T v_pu) const noexcept { return v_pu * base_.V_base; }
    [[nodiscard]] constexpr T from_pu_current(T i_pu) const noexcept { return i_pu * base_.I_base(); }
    [[nodiscard]] constexpr T from_pu_power(T p_pu) const noexcept { return p_pu * base_.S_base; }
    [[nodiscard]] constexpr T from_pu_impedance(T z_pu) const noexcept { return z_pu * base_.Z_base(); }
    [[nodiscard]] constexpr T from_pu_admittance(T y_pu) const noexcept { return y_pu * base_.Y_base(); }

    [[nodiscard]] constexpr const BaseQuantities& base() const noexcept { return base_; }

private:
    BaseQuantities base_;
};

/// Default scaling factors instance
inline constexpr ScalingFactors<double> default_scaling{};

/// Default per-unit system for power electronics (1kVA, 400V, 50Hz)
inline constexpr PerUnitSystem<double>::BaseQuantities power_electronics_base{
    1000.0,  // 1kVA
    400.0,   // 400V (typical DC link)
    50.0     // 50Hz
};

}  // namespace pulsim::v2
