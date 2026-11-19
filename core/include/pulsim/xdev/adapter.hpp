#pragma once

// =============================================================================
// Pulsim — model-side C++ helper for the external device ABI (NOT stable)
// =============================================================================
//
// Turns a law written like a built-in law (a template over the scalar, f and
// q per row) into the C table of pulsim_xdev.h, so a model author does not
// write C. The Jacobian comes from Pulsim's forward AD evaluated INSIDE the
// library; only doubles cross the boundary. This header compiles into the
// model library (C++20) and may change in any release; the library's binary
// contract is pulsim_xdev.h alone.
//
//   struct MyLaw {
//       static constexpr const char* kName = "my_law";
//       static constexpr std::uint32_t n_pins = 2, n_internal = 0;
//       static constexpr std::uint32_t kClaims = PULSIM_XDEV_CLAIM_SMOOTH;
//       static constexpr std::array<const char*, 2> kPins{"a", "k"};
//       static constexpr std::array<pulsim_xdev_field, 1> kFields{{{"G", "S", 1, 0, 1e3}}};
//       static constexpr Pattern<2> kF{...}, kQ{};   // structural entries
//       static constexpr std::array<double, 2> kProbeLo{...}, kProbeHi{...};
//       template <class S> static void eval(const double* p, const std::array<S, 2>& x,
//                                           std::array<S, 2>& f, std::array<S, 2>& q);
//   };
//   PULSIM_XDEV_MODELS(MyLaw)            // exports pulsim_xdev_get
//
// Optional members: `kMonotone` (std::array<pulsim_xdev_monotone, K>) and
// `static int validate(const double* p, char* msg, std::uint32_t len)`.

#include "pulsim/ad/ad_scalar.hpp"
#include "pulsim/xdev/pulsim_xdev.h"

#include <array>
#include <cstddef>
#include <cstdint>

namespace pulsim::xdev::adapter {

template <std::size_t R>
using Pattern = std::array<std::array<bool, R>, R>;

template <std::size_t R>
constexpr std::size_t count(const Pattern<R>& p) {
    std::size_t n = 0;
    for (const auto& row : p)
        for (bool b : row) n += b ? 1 : 0;
    return n;
}

template <std::size_t N, std::size_t R>
constexpr std::array<std::array<std::uint32_t, N>, 2> entries(const Pattern<R>& p) {
    std::array<std::array<std::uint32_t, N>, 2> out{};
    std::size_t k = 0;
    for (std::size_t r = 0; r < R; ++r)
        for (std::size_t c = 0; c < R; ++c)
            if (p[r][c]) {
                out[0][k] = static_cast<std::uint32_t>(r);
                out[1][k] = static_cast<std::uint32_t>(c);
                ++k;
            }
    return out;
}

template <class L>
struct Table {
    static constexpr std::size_t R = L::n_pins + L::n_internal;
    static constexpr std::size_t NF = count<R>(L::kF);
    static constexpr std::size_t NQ = count<R>(L::kQ);
    // a zero-length pattern still needs a valid pointer
    static constexpr auto kFE = entries<(NF > 0 ? NF : 1), R>(L::kF);
    static constexpr auto kQE = entries<(NQ > 0 ? NQ : 1), R>(L::kQ);

    static int32_t PULSIM_XDEV_CALL eval(const pulsim_xdev_ctx* c, const double* x,
                                         double* f, double* q, double* df,
                                         double* dq) noexcept {
        using AD = ad::ADRealN<R>;
        try {
            std::array<AD, R> xs{}, fo{}, qo{};
            for (std::size_t k = 0; k < R; ++k) {
                std::array<double, R> d{};
                d[k] = 1.0;
                xs[k] = AD{x[k], d};
            }
            L::template eval<AD>(c->p, xs, fo, qo);
            for (std::size_t r = 0; r < R; ++r) {
                f[r] = fo[r].value();
                q[r] = qo[r].value();
            }
            for (std::size_t k = 0; k < NF; ++k) df[k] = fo[kFE[0][k]].deriv(kFE[1][k]);
            for (std::size_t k = 0; k < NQ; ++k) dq[k] = qo[kQE[0][k]].deriv(kQE[1][k]);
        } catch (...) {
            return -1;  // no exception crosses the C boundary
        }
        return PULSIM_XDEV_OK;
    }

    static int32_t PULSIM_XDEV_CALL validate(const double* p, char* msg,
                                             uint32_t len) noexcept {
        if constexpr (requires { L::validate(p, msg, len); }) {
            try {
                return static_cast<int32_t>(L::validate(p, msg, len));
            } catch (...) {
                return -1;
            }
        } else {
            return PULSIM_XDEV_OK;
        }
    }

    static constexpr std::uint32_t n_mono() {
        if constexpr (requires { L::kMonotone; }) return L::kMonotone.size();
        else return 0;
    }
    static constexpr const pulsim_xdev_monotone* mono() {
        if constexpr (requires { L::kMonotone; }) return L::kMonotone.data();
        else return nullptr;
    }

    static inline const pulsim_xdev_descriptor desc{
        PULSIM_XDEV_ABI_VERSION,
        static_cast<std::uint32_t>(sizeof(pulsim_xdev_descriptor)),
        L::kName,
        static_cast<std::uint32_t>(L::n_pins),
        static_cast<std::uint32_t>(L::n_internal),
        static_cast<std::uint32_t>(L::kFields.size()),
        0u,
        static_cast<std::uint32_t>(NF),
        static_cast<std::uint32_t>(NQ),
        n_mono(),
        static_cast<std::uint32_t>(L::kClaims),
        L::kPins.data(),
        L::kFields.data(),
        kFE[0].data(), kFE[1].data(), kQE[0].data(), kQE[1].data(),
        L::kProbeLo.data(), L::kProbeHi.data(),
        mono(),
        &eval, &validate,
        nullptr, nullptr, nullptr, nullptr,
        {nullptr, nullptr, nullptr, nullptr},
    };
};

template <class... Ls>
const pulsim_xdev_descriptor* pick(std::uint32_t i) noexcept {
    static const pulsim_xdev_descriptor* const table[] = {&Table<Ls>::desc...};
    return i < sizeof...(Ls) ? table[i] : nullptr;
}

}  // namespace pulsim::xdev::adapter

/// Export the library's models, in this order, as `pulsim_xdev_get`.
#define PULSIM_XDEV_MODELS(...)                                                  \
    PULSIM_XDEV_EXTERN_C PULSIM_XDEV_EXPORT const pulsim_xdev_descriptor*       \
        PULSIM_XDEV_CALL pulsim_xdev_get(uint32_t index) {                       \
        return ::pulsim::xdev::adapter::pick<__VA_ARGS__>(index);                \
    }
