#pragma once

// =============================================================================
// Pulsim — Layer 1: SwitchStateMask
// =============================================================================
//
// `pulsim-v2-topology-and-switch-enumeration` Phase 2.
//
// Fixed-width bitmask over up to 64 switches, backed by a single
// `std::uint64_t`. Real-world coverage:
//   * 3φ inverter: 6 switches
//   * MMC arm (6 submodules): 12 switches
//   * 5-level NPC: 8-16 switches per phase
//   * 13-level cascaded H-bridge: 24-48 switches
//
// Past 64 the 2^N enumeration is itself the bottleneck — Layer 4
// needs lazy generation + on-demand caching, which is a follow-up.
// The public API is bit-indexed so a future swap to a dynamic
// bitset is non-breaking.
//
// Direct usability as `std::map` / `std::unordered_map` key for the
// Layer 4 PWL state-space cache: equality, total ordering, and a
// stable `hash()` are all part of the contract.

#include "pulsim/numeric/types.hpp"

#include <bit>
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <string>

namespace pulsim::topology {

class SwitchStateMask {
public:
    // -------------------------------------------------------------------------
    // Construction
    // -------------------------------------------------------------------------

    /// Build a mask of `num_switches` bits, all initialised to 0
    /// (every switch open). Throws `std::invalid_argument` if
    /// `num_switches > 64` — Layer 1 doesn't support wider masks yet.
    explicit SwitchStateMask(Size num_switches) : size_{num_switches} {
        if (num_switches > 64) {
            throw std::invalid_argument(
                "SwitchStateMask supports at most 64 switches; got " +
                std::to_string(num_switches) +
                ". Larger circuits need the (future) dynamic-bitset path.");
        }
    }

    SwitchStateMask() = default;

    // -------------------------------------------------------------------------
    // Bit ops
    // -------------------------------------------------------------------------

    [[nodiscard]] bool get(Size i) const noexcept {
        return ((bits_ >> i) & 1ULL) != 0ULL;
    }

    void set(Size i, bool v) noexcept {
        const std::uint64_t one_at_i = (1ULL << i);
        if (v) bits_ |= one_at_i;
        else   bits_ &= ~one_at_i;
    }

    void flip(Size i) noexcept {
        bits_ ^= (1ULL << i);
    }

    [[nodiscard]] Size count() const noexcept {
        return static_cast<Size>(std::popcount(bits_));
    }

    [[nodiscard]] Size size() const noexcept { return size_; }

    /// Raw backing bits — used by the Gray-code enumerator and by
    /// Layer 4 for fast XOR-based delta detection between adjacent
    /// states.
    [[nodiscard]] std::uint64_t bits() const noexcept { return bits_; }
    void set_bits(std::uint64_t b) noexcept { bits_ = b; }

    // -------------------------------------------------------------------------
    // Equality, ordering, hash
    // -------------------------------------------------------------------------

    friend bool operator==(const SwitchStateMask& a,
                           const SwitchStateMask& b) noexcept {
        return a.size_ == b.size_ && a.bits_ == b.bits_;
    }

    friend bool operator!=(const SwitchStateMask& a,
                           const SwitchStateMask& b) noexcept {
        return !(a == b);
    }

    /// Lexicographic total order: smaller size first, then smaller
    /// bit pattern. Used by `std::set<SwitchStateMask>`.
    friend bool operator<(const SwitchStateMask& a,
                          const SwitchStateMask& b) noexcept {
        if (a.size_ != b.size_) return a.size_ < b.size_;
        return a.bits_ < b.bits_;
    }

    /// Deterministic 64-bit-mixed hash. Mixes size and bits so masks
    /// of different sizes with coincidentally-equal bit patterns
    /// don't collide unnecessarily.
    [[nodiscard]] std::size_t hash() const noexcept {
        // splitmix64 on (size_ << 56) | bits_. Cheap, well-mixed.
        std::uint64_t x = (static_cast<std::uint64_t>(size_) << 56) ^ bits_;
        x ^= x >> 30; x *= 0xbf58476d1ce4e5b9ULL;
        x ^= x >> 27; x *= 0x94d049bb133111ebULL;
        x ^= x >> 31;
        return static_cast<std::size_t>(x);
    }

    // -------------------------------------------------------------------------
    // Diagnostics
    // -------------------------------------------------------------------------

    /// Binary representation in MSB-first format, e.g. "0b110010 N=6".
    [[nodiscard]] std::string to_string() const {
        std::string bin;
        bin.reserve(size_ + 4);
        bin += "0b";
        if (size_ == 0) bin += "0";
        else {
            for (Size i = size_; i-- > 0; ) {
                bin += (get(i) ? '1' : '0');
            }
        }
        bin += " N=" + std::to_string(size_);
        return bin;
    }

private:
    std::uint64_t bits_ = 0;
    Size size_ = 0;
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
