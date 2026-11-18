// Locale-free text -> floating point, for the IR's YAML and expression readers.
#pragma once

#include <charconv>
#include <cmath>
#include <limits>
#include <locale>
#include <sstream>
#include <string>
#include <string_view>
#include <system_error>

namespace pulsim::numeric {

namespace detail {

template <class T>
concept HasFloatFromChars = requires(const char* p, T& v) { std::from_chars(p, p, v); };

/// from_chars' own grammar: [-] (digits [. digits] | . digits) [(e|E) [+-] digits].
inline bool is_decimal(std::string_view s) {
    std::size_t i = 0;
    if (i < s.size() && s[i] == '-') ++i;
    const auto digits = [&] {
        const std::size_t b = i;
        while (i < s.size() && s[i] >= '0' && s[i] <= '9') ++i;
        return i - b;
    };
    std::size_t mantissa = digits();
    if (i < s.size() && s[i] == '.') {
        ++i;
        mantissa += digits();
    }
    if (mantissa == 0) return false;
    if (i < s.size() && (s[i] == 'e' || s[i] == 'E')) {
        ++i;
        if (i < s.size() && (s[i] == '+' || s[i] == '-')) ++i;
        if (digits() == 0) return false;
    }
    return i == s.size();
}

inline bool iequals(std::string_view a, std::string_view b) {
    if (a.size() != b.size()) return false;
    for (std::size_t i = 0; i < a.size(); ++i) {
        const char c = (a[i] >= 'A' && a[i] <= 'Z') ? static_cast<char>(a[i] - 'A' + 'a') : a[i];
        if (c != b[i]) return false;
    }
    return true;
}

template <class T>
bool parse_real_stream(const char* first, const char* last, T& out) {
    std::string_view s(first, static_cast<std::size_t>(last - first));
    const bool neg = !s.empty() && s.front() == '-';
    const std::string_view mag = neg ? s.substr(1) : s;
    if (iequals(mag, "inf") || iequals(mag, "infinity")) {
        out = neg ? -std::numeric_limits<T>::infinity() : std::numeric_limits<T>::infinity();
        return true;
    }
    if (iequals(mag, "nan")) {
        out = std::numeric_limits<T>::quiet_NaN();
        return true;
    }
    if (!is_decimal(s)) return false;
    std::istringstream is{std::string(s)};
    is.imbue(std::locale::classic());
    T v{};
    // strtod reports ERANGE for a subnormal result but still returns it; a true
    // underflow (0) or overflow stays an error, as with from_chars.
    if (!(is >> v) && !(v != T{} && std::isfinite(v) && std::fabs(v) < std::numeric_limits<T>::min())) {
        return false;
    }
    out = v;
    return true;
}

}  // namespace detail

/// Parses the WHOLE range [first, last) as a number; false if it is not one.
/// libc++ before LLVM 20 has no floating-point std::from_chars: there (or with
/// PULSIM_PARSE_REAL_FORCE_STREAM) a classic-locale stream reads the same grammar.
template <class T>
[[nodiscard]] bool parse_real(const char* first, const char* last, T& out) {
#ifndef PULSIM_PARSE_REAL_FORCE_STREAM
    if constexpr (detail::HasFloatFromChars<T>) {
        const auto r = std::from_chars(first, last, out);
        return r.ec == std::errc{} && r.ptr == last;
    } else
#endif
    {
        return detail::parse_real_stream(first, last, out);
    }
}

}  // namespace pulsim::numeric
