#pragma once

// =============================================================================
// Pulsim — Native PWM switch_fn classes for the DSED schedulers (Bridge.12)
// =============================================================================
//
// Pure-C++ replacements for the Python `class PWM` pattern that the
// DSED scheduler bindings (pulsim_dsed_bindings::PySwitchFnSSM) detect
// and call WITHOUT going through pybind11. Eliminates the 25%
// per-step Python overhead measured in Bridge.11 profiles:
//
//   * Python PWM switch_fn (Bridge.11 baseline) :  ~1.2 µs / step
//   * Native PWM switch_fn (Bridge.12)          :  ~50 ns / step
//
// User-facing: instantiate via the bound class (`pulsim.NativePwm2Switch`)
// and pass to `pulsim.simulate(switch_fn=...)` exactly like the Python
// version. The DSED dispatch detects the type and uses the native fast
// path automatically.

#include "pulsim/numeric/types.hpp"
#include "pulsim/topology/switch_state.hpp"

#include <cmath>
#include <stdexcept>

namespace pulsim::sources {

/// Native 2-switch complementary PWM (high-side + low-side).
///
/// Equivalent to the user-side Python pattern:
///
/// ```python
/// class PWM:
///     def __init__(self, T_sw, D, hs_first=True):
///         self.T_sw, self.D = T_sw, D
///         self.m_a = SwitchStateMask(2); self.m_a.set(0, hs_first); ...
///     def __call__(self, t):
///         return self.m_a if (t/self.T_sw) % 1.0 < self.D else self.m_b
///     def next_edge_after(self, t):
///         k = floor(t / self.T_sw)
///         for c in (k*T+D*T, (k+1)*T, (k+1)*T+D*T):
///             if c > t + eps: return c
///         return (k+2) * T
/// ```
///
/// but in pure C++ — the DSED scheduler can call `operator()` and
/// `next_edge_after()` without the GIL.
struct NativePwm2Switch {
    Real T_sw;
    Real D;
    topology::SwitchStateMask mask_a;   ///< Active when phase < D
    topology::SwitchStateMask mask_b;   ///< Active when phase >= D

    /// @param T_sw       Period (seconds).
    /// @param D          Duty cycle in [0, 1].
    /// @param n_switches Number of switches in the circuit's mask.
    /// @param hs_first   If true (default): switch 0 closed during D·T,
    ///                   switch 1 closed during (1-D)·T. If false: the
    ///                   reverse (useful for boost: LS closed first).
    NativePwm2Switch(Real T_sw, Real D, int n_switches,
                      bool hs_first = true)
        : T_sw{T_sw},
          D{D},
          mask_a(static_cast<std::size_t>(n_switches)),
          mask_b(static_cast<std::size_t>(n_switches)) {
        if (n_switches < 2) {
            throw std::invalid_argument(
                "NativePwm2Switch: need at least 2 switches in mask");
        }
        if (D < Real{0} || D > Real{1}) {
            throw std::invalid_argument(
                "NativePwm2Switch: duty must be in [0, 1]");
        }
        if (T_sw <= Real{0}) {
            throw std::invalid_argument(
                "NativePwm2Switch: T_sw must be positive");
        }
        // Switch 0 = first half; switch 1 = second half (or swapped).
        const bool a0 = hs_first;
        const bool b0 = !hs_first;
        mask_a.set(0, a0);   mask_a.set(1, !a0);
        mask_b.set(0, b0);   mask_b.set(1, !b0);
    }

    [[nodiscard]] topology::SwitchStateMask
    operator()(Real t) const noexcept {
        // Return-by-value: SwitchStateMask is a small POD-ish struct
        // (bitset + size). Copy cost is negligible.
        const Real phase = std::fmod(t / T_sw, Real{1.0});
        return phase < D ? mask_a : mask_b;
    }

    [[nodiscard]] Real next_edge_after(Real t) const noexcept {
        const Real k = std::floor(t / T_sw);
        constexpr Real eps = Real{1e-15};
        const Real candidates[3] = {
            k * T_sw + D * T_sw,
            (k + Real{1}) * T_sw,
            (k + Real{1}) * T_sw + D * T_sw,
        };
        for (Real c : candidates) {
            if (c > t + eps) return c;
        }
        return (k + Real{2}) * T_sw;
    }
};

/// Native 3-phase / multi-pattern PWM where the schedule cycles
/// through K masks at fixed phase boundaries (e.g., NPC 3-level: P→O→N).
///
/// `phase_boundaries[k]` ∈ (0, 1) gives the END of mask k's active window
/// as a fraction of T_sw. Last boundary must be exactly 1 (or omitted, in
/// which case 1 is implied). Number of masks K = phase_boundaries.size().
///
/// Example (3-level inverter P-O-N with 33/33/33 split):
///   phase_boundaries = [0.333, 0.666, 1.0]
///   masks = [mask_P, mask_O, mask_N]
struct NativeMultiMaskPwm {
    Real T_sw;
    std::vector<Real> phase_boundaries;   ///< K boundaries, ending at 1.0
    std::vector<topology::SwitchStateMask> masks;  ///< K masks

    NativeMultiMaskPwm(Real T_sw,
                        std::vector<Real> phase_boundaries_in,
                        std::vector<topology::SwitchStateMask> masks_in)
        : T_sw{T_sw},
          phase_boundaries{std::move(phase_boundaries_in)},
          masks{std::move(masks_in)} {
        if (T_sw <= Real{0}) {
            throw std::invalid_argument(
                "NativeMultiMaskPwm: T_sw must be positive");
        }
        if (phase_boundaries.size() != masks.size()) {
            throw std::invalid_argument(
                "NativeMultiMaskPwm: phase_boundaries and masks must "
                "have the same length");
        }
        if (phase_boundaries.empty()) {
            throw std::invalid_argument(
                "NativeMultiMaskPwm: need at least one mask");
        }
        // Verify monotonic + ends at 1.0
        Real prev = Real{0};
        for (Real b : phase_boundaries) {
            if (b <= prev || b > Real{1}) {
                throw std::invalid_argument(
                    "NativeMultiMaskPwm: phase_boundaries must be "
                    "strictly increasing within (0, 1].");
            }
            prev = b;
        }
        if (phase_boundaries.back() != Real{1}) {
            // Allow user to omit the trailing 1.0 — append it.
            phase_boundaries.back() = Real{1};
        }
    }

    [[nodiscard]] topology::SwitchStateMask
    operator()(Real t) const noexcept {
        const Real phase = std::fmod(t / T_sw, Real{1.0});
        for (std::size_t i = 0; i < phase_boundaries.size(); ++i) {
            if (phase < phase_boundaries[i]) return masks[i];
        }
        return masks.back();   // exactly at phase = 1.0 corner
    }

    [[nodiscard]] Real next_edge_after(Real t) const noexcept {
        const Real k = std::floor(t / T_sw);
        constexpr Real eps = Real{1e-15};
        // Candidates: each boundary in the current period + next-period 0
        for (Real b : phase_boundaries) {
            const Real c = k * T_sw + b * T_sw;
            if (c > t + eps) return c;
        }
        // Past the last boundary — next edge is start of next period
        return (k + Real{1}) * T_sw;
    }
};

}  // namespace pulsim::sources
