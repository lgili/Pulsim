#pragma once

// =============================================================================
// Pulsim v2 — Layer 2: forward-mode AD scalar
// =============================================================================
//
// `pulsim-v2-device-models-ad-driven` Phase 1.
//
// ADRealN<N> is a forward-mode automatic differentiation scalar that
// carries a value AND a fixed-size array of N partial derivatives.
// Every arithmetic operation propagates the derivatives via chain
// rule.
//
// The pivot move: every Layer 2 device model writes its math ONCE
// as a templated function on a `FloatingPoint` type. The same
// template instantiates for:
//   * `Real` (= double) — pure forward evaluation, used in tests
//     and the Layer 5 hot loop.
//   * `ADRealN<N>` — forward evaluation + Jacobian. Layer 3 uses
//     this to derive the matrix stamp WITHOUT a separate manual
//     implementation.
//
// The 4-place device-stamp duplication of v1 is structurally
// impossible in v2: there's only one function. The compiler
// generates the Jacobian.
//
// Memory layout: ADRealN<8> is 72 bytes (1 Real value + 8 Real
// derivs). Stack-allocated, trivially copyable, cache-friendly.
// Used by value, never by pointer.

#include "pulsim/v2/numeric/concepts.hpp"
#include "pulsim/v2/numeric/types.hpp"

#include <array>
#include <cmath>

namespace pulsim::v2::ad {

// -----------------------------------------------------------------------------
// ADRealN<N> — forward-mode AD scalar with N derivative slots.
// -----------------------------------------------------------------------------
template <Size N>
class ADRealN {
public:
    static constexpr Size num_inputs = N;

    // -------- Construction --------------------------------------------------

    /// Default-constructs to (value=0, derivs=all zero).
    constexpr ADRealN() noexcept : value_{0}, derivs_{} {}

    /// Construct from a Real constant — value = the real, derivs all
    /// zero. Represents a quantity with no input dependence.
    constexpr ADRealN(Real value) noexcept : value_{value}, derivs_{} {}

    /// Construct from explicit value + derivs (used by the seed
    /// helper and by arithmetic operators).
    constexpr ADRealN(Real value,
                       const std::array<Real, N>& derivs) noexcept
        : value_{value}, derivs_{derivs} {}

    // -------- Accessors -----------------------------------------------------

    [[nodiscard]] constexpr Real value() const noexcept { return value_; }
    [[nodiscard]] constexpr const std::array<Real, N>& derivatives() const noexcept {
        return derivs_;
    }
    [[nodiscard]] constexpr Real deriv(Size i) const noexcept {
        return derivs_[i];
    }

    // -------- Arithmetic (ad-vs-ad) -----------------------------------------

    friend constexpr ADRealN operator+(const ADRealN& a,
                                        const ADRealN& b) noexcept {
        std::array<Real, N> d{};
        for (Size i = 0; i < N; ++i) d[i] = a.derivs_[i] + b.derivs_[i];
        return ADRealN{a.value_ + b.value_, d};
    }

    friend constexpr ADRealN operator-(const ADRealN& a,
                                        const ADRealN& b) noexcept {
        std::array<Real, N> d{};
        for (Size i = 0; i < N; ++i) d[i] = a.derivs_[i] - b.derivs_[i];
        return ADRealN{a.value_ - b.value_, d};
    }

    friend constexpr ADRealN operator*(const ADRealN& a,
                                        const ADRealN& b) noexcept {
        // Product rule: d(a·b) = b·da + a·db
        std::array<Real, N> d{};
        for (Size i = 0; i < N; ++i) {
            d[i] = b.value_ * a.derivs_[i] + a.value_ * b.derivs_[i];
        }
        return ADRealN{a.value_ * b.value_, d};
    }

    friend constexpr ADRealN operator/(const ADRealN& a,
                                        const ADRealN& b) noexcept {
        // Quotient rule: d(a/b) = (b·da - a·db) / b²
        const Real bv2 = b.value_ * b.value_;
        std::array<Real, N> d{};
        for (Size i = 0; i < N; ++i) {
            d[i] = (b.value_ * a.derivs_[i] - a.value_ * b.derivs_[i]) / bv2;
        }
        return ADRealN{a.value_ / b.value_, d};
    }

    // -------- Arithmetic (ad-vs-real) ---------------------------------------

    friend constexpr ADRealN operator+(const ADRealN& a, Real b) noexcept {
        return ADRealN{a.value_ + b, a.derivs_};
    }
    friend constexpr ADRealN operator+(Real a, const ADRealN& b) noexcept {
        return b + a;
    }
    friend constexpr ADRealN operator-(const ADRealN& a, Real b) noexcept {
        return ADRealN{a.value_ - b, a.derivs_};
    }
    friend constexpr ADRealN operator-(Real a, const ADRealN& b) noexcept {
        std::array<Real, N> d{};
        for (Size i = 0; i < N; ++i) d[i] = -b.derivs_[i];
        return ADRealN{a - b.value_, d};
    }
    friend constexpr ADRealN operator*(const ADRealN& a, Real b) noexcept {
        std::array<Real, N> d{};
        for (Size i = 0; i < N; ++i) d[i] = a.derivs_[i] * b;
        return ADRealN{a.value_ * b, d};
    }
    friend constexpr ADRealN operator*(Real a, const ADRealN& b) noexcept {
        return b * a;
    }
    friend constexpr ADRealN operator/(const ADRealN& a, Real b) noexcept {
        std::array<Real, N> d{};
        for (Size i = 0; i < N; ++i) d[i] = a.derivs_[i] / b;
        return ADRealN{a.value_ / b, d};
    }
    friend constexpr ADRealN operator/(Real a, const ADRealN& b) noexcept {
        // d(a/b) = -a·db / b²
        const Real bv2 = b.value_ * b.value_;
        std::array<Real, N> d{};
        for (Size i = 0; i < N; ++i) d[i] = -a * b.derivs_[i] / bv2;
        return ADRealN{a / b.value_, d};
    }

    // -------- Unary minus ---------------------------------------------------

    friend constexpr ADRealN operator-(const ADRealN& a) noexcept {
        std::array<Real, N> d{};
        for (Size i = 0; i < N; ++i) d[i] = -a.derivs_[i];
        return ADRealN{-a.value_, d};
    }

    // -------- Compound assignment -------------------------------------------

    ADRealN& operator+=(const ADRealN& b) noexcept { return *this = *this + b; }
    ADRealN& operator-=(const ADRealN& b) noexcept { return *this = *this - b; }
    ADRealN& operator*=(const ADRealN& b) noexcept { return *this = *this * b; }
    ADRealN& operator/=(const ADRealN& b) noexcept { return *this = *this / b; }
    ADRealN& operator+=(Real b) noexcept { value_ += b; return *this; }
    ADRealN& operator-=(Real b) noexcept { value_ -= b; return *this; }
    ADRealN& operator*=(Real b) noexcept {
        value_ *= b;
        for (auto& d : derivs_) d *= b;
        return *this;
    }
    ADRealN& operator/=(Real b) noexcept {
        value_ /= b;
        for (auto& d : derivs_) d /= b;
        return *this;
    }

    // -------- Comparison (value only) ---------------------------------------
    //
    // Compares values only — derivative info plays no role in ordering.
    // Both ad-vs-ad and ad-vs-real overloads provided.

    friend constexpr bool operator==(const ADRealN& a, const ADRealN& b) noexcept { return a.value_ == b.value_; }
    friend constexpr bool operator!=(const ADRealN& a, const ADRealN& b) noexcept { return a.value_ != b.value_; }
    friend constexpr bool operator<(const ADRealN& a, const ADRealN& b)  noexcept { return a.value_ <  b.value_; }
    friend constexpr bool operator<=(const ADRealN& a, const ADRealN& b) noexcept { return a.value_ <= b.value_; }
    friend constexpr bool operator>(const ADRealN& a, const ADRealN& b)  noexcept { return a.value_ >  b.value_; }
    friend constexpr bool operator>=(const ADRealN& a, const ADRealN& b) noexcept { return a.value_ >= b.value_; }
    friend constexpr bool operator==(const ADRealN& a, Real b) noexcept { return a.value_ == b; }
    friend constexpr bool operator!=(const ADRealN& a, Real b) noexcept { return a.value_ != b; }
    friend constexpr bool operator<(const ADRealN& a, Real b)  noexcept { return a.value_ <  b; }
    friend constexpr bool operator<=(const ADRealN& a, Real b) noexcept { return a.value_ <= b; }
    friend constexpr bool operator>(const ADRealN& a, Real b)  noexcept { return a.value_ >  b; }
    friend constexpr bool operator>=(const ADRealN& a, Real b) noexcept { return a.value_ >= b; }

private:
    Real value_;
    std::array<Real, N> derivs_;
};

// -----------------------------------------------------------------------------
// Math functions
//
// Each math function computes the value AND propagates the chain-rule
// factor through every derivative slot. Standard library is used for
// the underlying scalar implementation.
// -----------------------------------------------------------------------------

template <Size N>
inline ADRealN<N> exp(const ADRealN<N>& x) noexcept {
    const Real ev = std::exp(x.value());
    std::array<Real, N> d{};
    for (Size i = 0; i < N; ++i) d[i] = ev * x.deriv(i);
    return ADRealN<N>{ev, d};
}

template <Size N>
inline ADRealN<N> log(const ADRealN<N>& x) noexcept {
    const Real lv = std::log(x.value());
    std::array<Real, N> d{};
    for (Size i = 0; i < N; ++i) d[i] = x.deriv(i) / x.value();
    return ADRealN<N>{lv, d};
}

template <Size N>
inline ADRealN<N> sqrt(const ADRealN<N>& x) noexcept {
    const Real sv = std::sqrt(x.value());
    const Real half_over_sv = Real{0.5} / sv;
    std::array<Real, N> d{};
    for (Size i = 0; i < N; ++i) d[i] = half_over_sv * x.deriv(i);
    return ADRealN<N>{sv, d};
}

template <Size N>
inline ADRealN<N> tanh(const ADRealN<N>& x) noexcept {
    const Real tv = std::tanh(x.value());
    const Real one_minus_tv2 = Real{1} - tv * tv;
    std::array<Real, N> d{};
    for (Size i = 0; i < N; ++i) d[i] = one_minus_tv2 * x.deriv(i);
    return ADRealN<N>{tv, d};
}

template <Size N>
inline ADRealN<N> sinh(const ADRealN<N>& x) noexcept {
    const Real cv = std::cosh(x.value());
    std::array<Real, N> d{};
    for (Size i = 0; i < N; ++i) d[i] = cv * x.deriv(i);
    return ADRealN<N>{std::sinh(x.value()), d};
}

template <Size N>
inline ADRealN<N> cosh(const ADRealN<N>& x) noexcept {
    const Real sv = std::sinh(x.value());
    std::array<Real, N> d{};
    for (Size i = 0; i < N; ++i) d[i] = sv * x.deriv(i);
    return ADRealN<N>{std::cosh(x.value()), d};
}

template <Size N>
inline ADRealN<N> abs(const ADRealN<N>& x) noexcept {
    // d|x|/dx = sign(x); at x=0 we use the right-derivative (+1).
    const Real sign = (x.value() < Real{0}) ? Real{-1} : Real{1};
    std::array<Real, N> d{};
    for (Size i = 0; i < N; ++i) d[i] = sign * x.deriv(i);
    return ADRealN<N>{std::abs(x.value()), d};
}

// -----------------------------------------------------------------------------
// seed<N>(v0, v1, ..., vN-1) — canonical input-seeding helper.
//
// Returns `std::array<ADRealN<N>, N>` where element i has value = v_i
// and derivs[j] = (i == j ? 1 : 0). This is the standard way to
// derive a Jacobian: seed each independent input, evaluate the
// function, read derivatives from the result.
//
// Implemented as an N=2 / N=3 / N=4 set of overloads — covers the
// common device topologies (2-terminal R/L/C/D, 3-terminal MOSFET/
// IGBT/BJT, 4-terminal transformer/PMSM). Larger N can be added
// when a device needs them.
// -----------------------------------------------------------------------------

template <Size N, typename... Ts>
[[nodiscard]] constexpr std::array<ADRealN<N>, N> seed_impl(Ts... vs) noexcept {
    static_assert(sizeof...(vs) == N,
                  "seed<N>(...) requires exactly N values");
    std::array<Real, N> values{static_cast<Real>(vs)...};
    std::array<ADRealN<N>, N> result{};
    for (Size i = 0; i < N; ++i) {
        std::array<Real, N> derivs{};
        derivs[i] = Real{1};
        result[i] = ADRealN<N>{values[i], derivs};
    }
    return result;
}

[[nodiscard]] constexpr std::array<ADRealN<2>, 2> seed2(Real v0, Real v1) noexcept {
    return seed_impl<2>(v0, v1);
}
[[nodiscard]] constexpr std::array<ADRealN<3>, 3> seed3(Real v0, Real v1, Real v2) noexcept {
    return seed_impl<3>(v0, v1, v2);
}
[[nodiscard]] constexpr std::array<ADRealN<4>, 4> seed4(Real v0, Real v1, Real v2, Real v3) noexcept {
    return seed_impl<4>(v0, v1, v2, v3);
}

}  // namespace pulsim::v2::ad

// -----------------------------------------------------------------------------
// Opt the AD scalar into the Layer-0 `FloatingPoint` concept.
//
// This is the customization-point trick: Layer 0's concepts.hpp
// declares `is_ad_scalar_v<T>` defaulting to `false`. Layer 2's
// AD scalar specializes it to `true` for every `ADRealN<N>`, which
// makes Layer 0's `numeric::FloatingPoint<ADRealN<N>>` evaluate to
// true. Device models templated on `FloatingPoint S` then accept
// both `Real` (forward eval) AND `ADRealN<N>` (Jacobian).
// -----------------------------------------------------------------------------
namespace pulsim::v2::numeric {
template <Size N>
inline constexpr bool is_ad_scalar_v<ad::ADRealN<N>> = true;
}  // namespace pulsim::v2::numeric
