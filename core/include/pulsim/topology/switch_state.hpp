#pragma once

// =============================================================================
// Pulsim — Layer 1: SwitchStateMask (dynamic width, Phase 1)
// =============================================================================
//
// `pulsim-v2-topology-and-switch-enumeration` Phase 2; widened to a
// dynamic bitset in Phase 1 of the v2.0 kernel foundation.
//
// Bitmask over an ARBITRARY number of switches. Storage is a
// small-buffer optimised word array:
//   * ≤ 128 switches — two inline `std::uint64_t` words, no heap
//     allocation anywhere (covers every practical converter leg and
//     the entire pre-Phase-1 usage: 3φ inverters, NPC phases, CHB
//     stacks, MMC arms into the tens of submodules);
//   * > 128 switches — words spill into a `std::vector` (one
//     allocation per mask; large-MMC studies at hundreds of switches
//     are cache-bound elsewhere, so this is not a hot-loop cost).
//
// The v1.x class was a single `uint64_t` whose constructor THREW for
// more than 64 switches — a hard ceiling that made 120-switch MMC
// phases unrepresentable (audit findings switch-mask-64-cap /
// 64-switch-hard-ceiling, CONFIRMED). The public API was already
// bit-indexed, so this swap is source-compatible for every consumer
// that used get/set/flip/count/==/hash. The raw-word shortcuts
// (`bits()` / `set_bits()`) remain for ≤ 64 switches (Gray-code
// enumerator, tests, benches) and throw beyond — loudly, instead of
// silently truncating.
//
// Direct usability as `std::map` / `std::unordered_map` key for the
// Layer 4 PWL state-space cache: equality, total ordering, and a
// stable `hash()` are all part of the contract.

#include "pulsim/numeric/types.hpp"

#include <bit>
#include <cstdint>
#include <format>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

namespace pulsim::topology {

class SwitchStateMask {
public:
    /// Number of switches representable without heap allocation.
    static constexpr Size kInlineBits = 128;

    // -------------------------------------------------------------------------
    // Construction
    // -------------------------------------------------------------------------

    /// Build a mask of `num_switches` bits, all initialised to 0
    /// (every switch open). Any width is accepted; widths above
    /// `kInlineBits` allocate the word array on the heap.
    explicit SwitchStateMask(Size num_switches) : size_{num_switches} {
        nwords_ = words_for_(num_switches);
        if (nwords_ > kInlineWords) {
            ext_.assign(static_cast<std::size_t>(nwords_), 0ULL);
        }
    }

    SwitchStateMask() = default;
    SwitchStateMask(const SwitchStateMask&)            = default;
    SwitchStateMask(SwitchStateMask&&) noexcept        = default;
    SwitchStateMask& operator=(const SwitchStateMask&) = default;
    SwitchStateMask& operator=(SwitchStateMask&&) noexcept = default;

    // -------------------------------------------------------------------------
    // Bit ops
    // -------------------------------------------------------------------------

    [[nodiscard]] bool get(Size i) const noexcept {
        return ((wptr_()[word_of_(i)] >> bit_of_(i)) & 1ULL) != 0ULL;
    }

    void set(Size i, bool v) noexcept {
        const std::uint64_t one = (1ULL << bit_of_(i));
        if (v) wptr_()[word_of_(i)] |= one;
        else   wptr_()[word_of_(i)] &= ~one;
    }

    void flip(Size i) noexcept {
        wptr_()[word_of_(i)] ^= (1ULL << bit_of_(i));
    }

    [[nodiscard]] Size count() const noexcept {
        Size c = 0;
        const std::uint64_t* w = wptr_();
        for (Size k = 0; k < nwords_; ++k) {
            c += static_cast<Size>(std::popcount(w[k]));
        }
        return c;
    }

    [[nodiscard]] Size size() const noexcept { return size_; }

    // -------------------------------------------------------------------------
    // Word-level access (Phase 1) — the hot-path replacement for the
    // old raw-uint64 shortcuts. Layer 4's XOR delta detection and
    // Layer 5's diode-overlay merge iterate words instead of assuming
    // a single one.
    // -------------------------------------------------------------------------

    /// Number of 64-bit words backing this mask (0 for an empty mask).
    [[nodiscard]] Size num_words() const noexcept { return nwords_; }

    /// Read word `w` (0-based; word 0 holds switches 0..63).
    [[nodiscard]] std::uint64_t word(Size w) const noexcept {
        return wptr_()[w];
    }

    /// Number of differing bits between two same-width masks.
    [[nodiscard]] Size hamming_distance(
        const SwitchStateMask& o) const noexcept {
        Size c = 0;
        const std::uint64_t* a = wptr_();
        const std::uint64_t* b = o.wptr_();
        const Size nw = nwords_ < o.nwords_ ? nwords_ : o.nwords_;
        for (Size k = 0; k < nw; ++k) {
            c += static_cast<Size>(std::popcount(a[k] ^ b[k]));
        }
        return c;
    }

    /// Word-wise overlay merge (Layer 5 diode auto-state):
    /// result bit i = owned.get(i) ? overlay.get(i) : this->get(i).
    [[nodiscard]] SwitchStateMask overlay(
        const SwitchStateMask& over,
        const SwitchStateMask& owned) const {
        SwitchStateMask out(size_);
        std::uint64_t* ow = out.wptr_();
        const std::uint64_t* self = wptr_();
        const std::uint64_t* ovr  = over.wptr_();
        const std::uint64_t* own  = owned.wptr_();
        for (Size k = 0; k < nwords_; ++k) {
            ow[k] = (self[k] & ~own[k]) | (ovr[k] & own[k]);
        }
        return out;
    }

    // -------------------------------------------------------------------------
    // Legacy raw-uint64 shortcuts — valid only while the mask fits in
    // one word. The Gray-code enumerator, tests and benches use these
    // for ≤ 64-switch circuits; wider masks throw std::logic_error so
    // a stale caller fails LOUDLY instead of silently truncating
    // (the audit's silent-wrong-answer bar applies here too).
    // -------------------------------------------------------------------------

    [[nodiscard]] std::uint64_t bits() const {
        if (size_ > 64) {
            throw std::logic_error(std::format(
                "SwitchStateMask::bits(): mask has {} switches (> 64); "
                "use word(i)/num_words()/hamming_distance() for wide "
                "masks.", size_));
        }
        return nwords_ == 0 ? 0ULL : wptr_()[0];
    }

    void set_bits(std::uint64_t b) {
        if (size_ > 64) {
            throw std::logic_error(std::format(
                "SwitchStateMask::set_bits(): mask has {} switches "
                "(> 64); use set(i, v) for wide masks.", size_));
        }
        if (nwords_ > 0) wptr_()[0] = b;
    }

    // -------------------------------------------------------------------------
    // Equality, ordering, hash
    // -------------------------------------------------------------------------

    friend bool operator==(const SwitchStateMask& a,
                            const SwitchStateMask& b) noexcept {
        if (a.size_ != b.size_) return false;
        const std::uint64_t* aw = a.wptr_();
        const std::uint64_t* bw = b.wptr_();
        for (Size k = 0; k < a.nwords_; ++k) {
            if (aw[k] != bw[k]) return false;
        }
        return true;
    }

    friend bool operator!=(const SwitchStateMask& a,
                            const SwitchStateMask& b) noexcept {
        return !(a == b);
    }

    /// Lexicographic total order: smaller size first, then smaller bit
    /// pattern (most-significant word decides — matches the single-
    /// word ordering of the v1.x class for ≤ 64 switches). Used by
    /// `std::set<SwitchStateMask>`.
    friend bool operator<(const SwitchStateMask& a,
                           const SwitchStateMask& b) noexcept {
        if (a.size_ != b.size_) return a.size_ < b.size_;
        const std::uint64_t* aw = a.wptr_();
        const std::uint64_t* bw = b.wptr_();
        for (Size k = a.nwords_; k-- > 0;) {
            if (aw[k] != bw[k]) return aw[k] < bw[k];
        }
        return false;
    }

    /// Deterministic word-chained splitmix64 hash mixing the size and
    /// every backing word — no truncation at any width.
    [[nodiscard]] std::size_t hash() const noexcept {
        auto mix = [](std::uint64_t x) noexcept {
            x ^= x >> 30; x *= 0xbf58476d1ce4e5b9ULL;
            x ^= x >> 27; x *= 0x94d049bb133111ebULL;
            x ^= x >> 31;
            return x;
        };
        std::uint64_t h =
            mix(0x9e3779b97f4a7c15ULL ^ static_cast<std::uint64_t>(size_));
        const std::uint64_t* w = wptr_();
        for (Size k = 0; k < nwords_; ++k) {
            h = mix(h ^ w[k]);
        }
        return static_cast<std::size_t>(h);
    }

    // -------------------------------------------------------------------------
    // Diagnostics
    // -------------------------------------------------------------------------

    /// Binary representation in MSB-first format, e.g. "0b110010 N=6".
    [[nodiscard]] std::string to_string() const {
        std::string bin;
        bin.reserve(static_cast<std::size_t>(size_) + 8);
        bin += "0b";
        if (size_ == 0) bin += "0";
        else {
            for (Size i = size_; i-- > 0; ) {
                bin += (get(i) ? '1' : '0');
            }
        }
        bin += std::format(" N={}", size_);
        return bin;
    }

private:
    static constexpr Size kInlineWords = kInlineBits / 64;

    [[nodiscard]] static Size words_for_(Size nbits) noexcept {
        return (nbits + 63) / 64;
    }
    [[nodiscard]] static Size word_of_(Size i) noexcept { return i / 64; }
    [[nodiscard]] static Size bit_of_(Size i) noexcept { return i % 64; }

    [[nodiscard]] const std::uint64_t* wptr_() const noexcept {
        return ext_.empty() ? inline_ : ext_.data();
    }
    [[nodiscard]] std::uint64_t* wptr_() noexcept {
        return ext_.empty() ? inline_ : ext_.data();
    }

    Size size_   = 0;
    Size nwords_ = 0;
    std::uint64_t inline_[kInlineWords] = {0ULL, 0ULL};
    std::vector<std::uint64_t> ext_;   // engaged only above kInlineBits
};

}  // namespace pulsim::topology

// -----------------------------------------------------------------------------
// std::hash specialization so SwitchStateMask is usable as an
// unordered_map key directly.
// -----------------------------------------------------------------------------
namespace std {
template <>
struct hash<pulsim::topology::SwitchStateMask> {
    std::size_t operator()(
        const pulsim::topology::SwitchStateMask& m) const noexcept {
        return m.hash();
    }
};
}  // namespace std
