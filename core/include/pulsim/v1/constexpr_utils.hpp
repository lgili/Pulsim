#pragma once

// =============================================================================
// PulsimCore v2 - Compile-Time Utilities
// =============================================================================
// This header provides constexpr utilities for compile-time computations:
// - Constexpr math functions (sqrt, abs, pow, etc.)
// - Compile-time array utilities
// - Index sequence utilities for template metaprogramming
// - Compile-time string utilities
// =============================================================================

#include <array>
#include <cstddef>
#include <limits>
#include <type_traits>
#include <utility>

namespace pulsim::v1 {

// =============================================================================
// Constexpr Math Utilities
// =============================================================================

/// Constexpr absolute value
template<typename T>
[[nodiscard]] constexpr T cabs(T x) noexcept {
    return x < T{0} ? -x : x;
}

/// Constexpr sign function
template<typename T>
[[nodiscard]] constexpr T sign(T x) noexcept {
    return (T{0} < x) - (x < T{0});
}

/// Constexpr minimum
template<typename T>
[[nodiscard]] constexpr T cmin(T a, T b) noexcept {
    return a < b ? a : b;
}

/// Constexpr maximum
template<typename T>
[[nodiscard]] constexpr T cmax(T a, T b) noexcept {
    return a > b ? a : b;
}

/// Constexpr clamp
template<typename T>
[[nodiscard]] constexpr T cclamp(T x, T lo, T hi) noexcept {
    return cmax(lo, cmin(x, hi));
}

/// Constexpr square root (Newton-Raphson)
template<typename T>
[[nodiscard]] constexpr T csqrt(T x) noexcept {
    if (x < T{0}) return std::numeric_limits<T>::quiet_NaN();
    if (x == T{0}) return T{0};

    T guess = x;
    T prev = T{0};

    // Newton-Raphson iteration
    while (cabs(guess - prev) > std::numeric_limits<T>::epsilon() * guess) {
        prev = guess;
        guess = T{0.5} * (guess + x / guess);
    }
    return guess;
}

/// Constexpr power (integer exponent)
template<typename T>
[[nodiscard]] constexpr T cpow(T base, int exp) noexcept {
    if (exp == 0) return T{1};
    if (exp < 0) return T{1} / cpow(base, -exp);

    T result = T{1};
    while (exp > 0) {
        if (exp & 1) result *= base;
        base *= base;
        exp >>= 1;
    }
    return result;
}

/// Constexpr power (unsigned exponent) - more efficient
template<typename T, unsigned N>
[[nodiscard]] constexpr T cpow_n(T base) noexcept {
    if constexpr (N == 0) return T{1};
    else if constexpr (N == 1) return base;
    else if constexpr (N % 2 == 0) {
        T half = cpow_n<T, N / 2>(base);
        return half * half;
    } else {
        return base * cpow_n<T, N - 1>(base);
    }
}

/// Constexpr exponential approximation (Taylor series)
template<typename T>
[[nodiscard]] constexpr T cexp(T x) noexcept {
    // exp(x) = sum(x^n / n!) for n = 0 to infinity
    // Use enough terms for double precision
    constexpr int max_terms = 30;

    T sum = T{1};
    T term = T{1};

    for (int n = 1; n < max_terms; ++n) {
        term *= x / static_cast<T>(n);
        sum += term;
        if (cabs(term) < std::numeric_limits<T>::epsilon() * cabs(sum)) {
            break;
        }
    }
    return sum;
}

/// Constexpr natural logarithm approximation
template<typename T>
[[nodiscard]] constexpr T clog(T x) noexcept {
    if (x <= T{0}) return -std::numeric_limits<T>::infinity();
    if (x == T{1}) return T{0};

    // Use Newton-Raphson: find y such that exp(y) = x
    T y = x - T{1};  // Initial guess
    T prev = T{0};

    for (int i = 0; i < 100 && cabs(y - prev) > std::numeric_limits<T>::epsilon() * cabs(y); ++i) {
        prev = y;
        T ey = cexp(y);
        y = y - (ey - x) / ey;
    }
    return y;
}

// =============================================================================
// Physical Constants (constexpr)
// =============================================================================

namespace constants {
    /// Boltzmann constant (J/K)
    inline constexpr double k_B = 1.380649e-23;

    /// Elementary charge (C)
    inline constexpr double q_e = 1.602176634e-19;

    /// Thermal voltage at T=300K (V)
    inline constexpr double V_T_300K = k_B * 300.0 / q_e;

    /// Pi
    inline constexpr double pi = 3.14159265358979323846;

    /// Epsilon for floating-point comparisons
    inline constexpr double epsilon = 1e-12;

    /// Default absolute tolerance
    inline constexpr double default_abstol = 1e-9;

    /// Default relative tolerance
    inline constexpr double default_reltol = 1e-3;
}

/// Compute thermal voltage at temperature T (Kelvin)
[[nodiscard]] constexpr double thermal_voltage(double T) noexcept {
    return constants::k_B * T / constants::q_e;
}

// =============================================================================
// Compile-Time Array Utilities
// =============================================================================

/// Create array filled with a value
template<typename T, std::size_t N>
[[nodiscard]] constexpr std::array<T, N> make_filled_array(T value) noexcept {
    std::array<T, N> arr{};
    for (std::size_t i = 0; i < N; ++i) {
        arr[i] = value;
    }
    return arr;
}

/// Create array with indices [0, 1, 2, ..., N-1]
template<typename T, std::size_t N>
[[nodiscard]] constexpr std::array<T, N> make_iota_array() noexcept {
    std::array<T, N> arr{};
    for (std::size_t i = 0; i < N; ++i) {
        arr[i] = static_cast<T>(i);
    }
    return arr;
}

/// Transform array elements with a function
template<typename T, std::size_t N, typename F>
[[nodiscard]] constexpr std::array<T, N> transform_array(
    const std::array<T, N>& arr, F&& f) noexcept {
    std::array<T, N> result{};
    for (std::size_t i = 0; i < N; ++i) {
        result[i] = f(arr[i]);
    }
    return result;
}

/// Sum all elements in array
template<typename T, std::size_t N>
[[nodiscard]] constexpr T array_sum(const std::array<T, N>& arr) noexcept {
    T sum = T{0};
    for (std::size_t i = 0; i < N; ++i) {
        sum += arr[i];
    }
    return sum;
}

/// Inner product of two arrays
template<typename T, std::size_t N>
[[nodiscard]] constexpr T array_dot(
    const std::array<T, N>& a,
    const std::array<T, N>& b) noexcept {
    T sum = T{0};
    for (std::size_t i = 0; i < N; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

/// L2 norm of array
template<typename T, std::size_t N>
[[nodiscard]] constexpr T array_norm(const std::array<T, N>& arr) noexcept {
    return csqrt(array_dot(arr, arr));
}

/// Maximum element in array
template<typename T, std::size_t N>
[[nodiscard]] constexpr T array_max(const std::array<T, N>& arr) noexcept {
    static_assert(N > 0, "Array must not be empty");
    T max_val = arr[0];
    for (std::size_t i = 1; i < N; ++i) {
        if (arr[i] > max_val) max_val = arr[i];
    }
    return max_val;
}

/// Minimum element in array
template<typename T, std::size_t N>
[[nodiscard]] constexpr T array_min(const std::array<T, N>& arr) noexcept {
    static_assert(N > 0, "Array must not be empty");
    T min_val = arr[0];
    for (std::size_t i = 1; i < N; ++i) {
        if (arr[i] < min_val) min_val = arr[i];
    }
    return min_val;
}

// =============================================================================
// Index Sequence Utilities
// =============================================================================

/// Apply function to each index in a sequence
template<typename F, std::size_t... Is>
constexpr void for_each_index_impl(F&& f, std::index_sequence<Is...>) {
    (f(std::integral_constant<std::size_t, Is>{}), ...);
}

template<std::size_t N, typename F>
constexpr void for_each_index(F&& f) {
    for_each_index_impl(std::forward<F>(f), std::make_index_sequence<N>{});
}

/// Generate array using index
template<typename T, std::size_t N, typename F>
[[nodiscard]] constexpr std::array<T, N> generate_array(F&& f) {
    return [&f]<std::size_t... Is>(std::index_sequence<Is...>) {
        return std::array<T, N>{f(Is)...};
    }(std::make_index_sequence<N>{});
}

// =============================================================================
// Compile-Time Lookup Tables
// =============================================================================

/// Generate lookup table for a function
template<typename T, std::size_t N, typename F>
[[nodiscard]] constexpr std::array<T, N> make_lut(F&& f, T x_min, T x_max) {
    return generate_array<T, N>([&](std::size_t i) {
        T x = x_min + (x_max - x_min) * static_cast<T>(i) / static_cast<T>(N - 1);
        return f(x);
    });
}

/// Interpolate in a lookup table
template<typename T, std::size_t N>
[[nodiscard]] constexpr T lut_interpolate(
    const std::array<T, N>& lut, T x, T x_min, T x_max) noexcept {
    // Clamp x to range
    x = cclamp(x, x_min, x_max);

    // Find index
    T normalized = (x - x_min) / (x_max - x_min) * static_cast<T>(N - 1);
    std::size_t idx = static_cast<std::size_t>(normalized);
    if (idx >= N - 1) idx = N - 2;

    // Linear interpolation
    T t = normalized - static_cast<T>(idx);
    return lut[idx] * (T{1} - t) + lut[idx + 1] * t;
}

// =============================================================================
// Numerical Utilities
// =============================================================================

/// Check if two floating-point values are approximately equal
template<typename T>
[[nodiscard]] constexpr bool approx_equal(
    T a, T b,
    T rel_tol = static_cast<T>(constants::default_reltol),
    T abs_tol = static_cast<T>(constants::default_abstol)) noexcept {
    T diff = cabs(a - b);
    return diff <= abs_tol || diff <= rel_tol * cmax(cabs(a), cabs(b));
}

/// Safe division (returns fallback if denominator is near zero)
template<typename T>
[[nodiscard]] constexpr T safe_div(T num, T den, T fallback = T{0}) noexcept {
    if (cabs(den) < std::numeric_limits<T>::epsilon()) {
        return fallback;
    }
    return num / den;
}

/// Smooth step function (Hermite interpolation)
template<typename T>
[[nodiscard]] constexpr T smoothstep(T edge0, T edge1, T x) noexcept {
    x = cclamp((x - edge0) / (edge1 - edge0), T{0}, T{1});
    return x * x * (T{3} - T{2} * x);
}

/// Smoother step function (Ken Perlin's improved version)
template<typename T>
[[nodiscard]] constexpr T smootherstep(T edge0, T edge1, T x) noexcept {
    x = cclamp((x - edge0) / (edge1 - edge0), T{0}, T{1});
    return x * x * x * (x * (x * T{6} - T{15}) + T{10});
}

// =============================================================================
// Unit Conversion Utilities (constexpr)
// =============================================================================

namespace units {
    // SI prefixes
    inline constexpr double femto = 1e-15;
    inline constexpr double pico  = 1e-12;
    inline constexpr double nano  = 1e-9;
    inline constexpr double micro = 1e-6;
    inline constexpr double milli = 1e-3;
    inline constexpr double kilo  = 1e3;
    inline constexpr double mega  = 1e6;
    inline constexpr double giga  = 1e9;

    // Frequency/time conversions
    [[nodiscard]] constexpr double hz_to_rad(double hz) noexcept {
        return 2.0 * constants::pi * hz;
    }

    [[nodiscard]] constexpr double rad_to_hz(double rad) noexcept {
        return rad / (2.0 * constants::pi);
    }

    [[nodiscard]] constexpr double period_to_freq(double T) noexcept {
        return T > 0 ? 1.0 / T : 0.0;
    }

    // Decibel conversions
    [[nodiscard]] constexpr double linear_to_db(double x) noexcept {
        return 20.0 * clog(x) / clog(10.0);
    }

    [[nodiscard]] constexpr double db_to_linear(double db) noexcept {
        return cpow(10.0, static_cast<int>(db / 20));
    }
}

// =============================================================================
// Static Assertions for Math Functions
// =============================================================================

namespace detail {
    // Compile-time tests
    static_assert(cabs(-5) == 5, "cabs test failed");
    static_assert(cabs(5) == 5, "cabs test failed");
    static_assert(sign(-5) == -1, "sign test failed");
    static_assert(sign(5) == 1, "sign test failed");
    static_assert(cmin(3, 5) == 3, "cmin test failed");
    static_assert(cmax(3, 5) == 5, "cmax test failed");
    static_assert(cclamp(7, 0, 5) == 5, "cclamp test failed");
    static_assert(cpow(2, 3) == 8, "cpow test failed");
    static_assert(cpow_n<int, 4>(2) == 16, "cpow_n test failed");
    static_assert(approx_equal(csqrt(4.0), 2.0), "csqrt test failed");
}

}  // namespace pulsim::v1
