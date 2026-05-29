#pragma once

// =============================================================================
// Pulsim — Layer 1: Gray-code enumeration over switch states
// =============================================================================
//
// `pulsim-v2-topology-and-switch-enumeration` Phase 3.
//
// Enumerates all 2^N `SwitchStateMask` values in Gray-code order:
// every consecutive pair differs by EXACTLY one bit. This is the
// Layer 4-friendly order — re-factoring the system matrix after a
// single-bit switch flip can use Sherman-Morrison rank-1 update
// (O(n²)) instead of a full LU re-factor (O(n³)).
//
// Standard binary order would flip multiple bits between consecutive
// states (e.g. 3 → 4 flips three bits). Gray code (0, 1, 3, 2, 6,
// 7, 5, 4, ...) flips exactly one. Conversion is the textbook
// `state[i] = i XOR (i >> 1)`.
//
// Range-friendly: works with C++20 range-based for. Uses a sentinel
// end iterator (no need to store the full 2^N end mask in the
// iterator state).

#include "pulsim/numeric/types.hpp"
#include "pulsim/topology/switch_state.hpp"

#include <cstdint>
#include <stdexcept>

namespace pulsim::topology {

class SwitchStateEnumerator {
public:
    // -------------------------------------------------------------------------
    // Iterator types
    // -------------------------------------------------------------------------
    //
    // Forward iterator yielding `SwitchStateMask` by value. The
    // sentinel-based end means the iterator only stores its current
    // counter index and the bit-count N — small, hot-loop-friendly.
    // -------------------------------------------------------------------------
    class Iterator {
    public:
        using value_type = SwitchStateMask;
        using difference_type = std::ptrdiff_t;
        using iterator_category = std::forward_iterator_tag;

        Iterator() = default;
        Iterator(std::uint64_t counter, Size num_switches) noexcept
            : counter_{counter}, num_switches_{num_switches} {}

        [[nodiscard]] SwitchStateMask operator*() const noexcept {
            SwitchStateMask m(num_switches_);
            // Textbook binary-to-Gray conversion.
            m.set_bits(counter_ ^ (counter_ >> 1));
            return m;
        }

        Iterator& operator++() noexcept {
            ++counter_;
            return *this;
        }
        Iterator operator++(int) noexcept {
            Iterator copy = *this;
            ++(*this);
            return copy;
        }

        friend bool operator==(const Iterator& a, const Iterator& b) noexcept {
            return a.counter_ == b.counter_ &&
                   a.num_switches_ == b.num_switches_;
        }
        friend bool operator!=(const Iterator& a, const Iterator& b) noexcept {
            return !(a == b);
        }

    private:
        std::uint64_t counter_ = 0;
        Size num_switches_ = 0;
    };

    // -------------------------------------------------------------------------
    // Construction
    // -------------------------------------------------------------------------

    explicit SwitchStateEnumerator(Size num_switches)
        : num_switches_{num_switches}, num_states_{num_states_for(num_switches)} {
        // N >= 64 would need a 128-bit counter; the in-process Gray-code
        // path can only represent 2^63 masks, so it would SILENTLY
        // enumerate half the switch-state space and build an incomplete
        // cache. Fail loudly instead — large-N circuits must use Layer 4's
        // lazy `PwlStateSpaceCache` enumeration (which never materialises
        // the full 2^N set).
        if (num_switches >= 64) {
            throw std::invalid_argument(
                "SwitchStateEnumerator: num_switches >= 64 is unsupported by "
                "in-process Gray-code enumeration (2^64 needs a 128-bit "
                "counter). Use lazy PwlStateSpaceCache enumeration for "
                "large-N circuits.");
        }
    }

    [[nodiscard]] Iterator begin() const noexcept {
        return Iterator{0, num_switches_};
    }
    [[nodiscard]] Iterator end() const noexcept {
        return Iterator{num_states_, num_switches_};
    }

    /// Total number of states the enumeration will yield (= 2^N).
    /// Valid for N ≤ 63 only; the constructor throws for N ≥ 64 (a
    /// 128-bit counter would be required). Layer 4's lazy enumeration
    /// is the right answer past N ~ 20 anyway.
    [[nodiscard]] std::uint64_t num_states() const noexcept {
        return num_states_;
    }

    [[nodiscard]] Size num_switches() const noexcept {
        return num_switches_;
    }

private:
    static constexpr std::uint64_t num_states_for(Size n) noexcept {
        // N = 0 → 1 state (the empty mask). The N >= 64 branch is
        // defensive only: the constructor rejects N >= 64 before any
        // object is usable (it runs in the member-init list, so this
        // must stay noexcept and total). Layer 4's lazy path is the
        // supported route for large N; the in-process path here is for
        // ≤ 20-ish switches in practice.
        if (n == 0) return 1ULL;
        if (n >= 64) return 1ULL << 63;
        return 1ULL << n;
    }

    Size num_switches_;
    std::uint64_t num_states_;
};

/// Free function for natural call-site syntax:
///   for (auto mask : enumerate_switch_states(N)) { ... }
[[nodiscard]] inline SwitchStateEnumerator enumerate_switch_states(
    Size num_switches) {
    return SwitchStateEnumerator{num_switches};
}

}  // namespace pulsim::topology
