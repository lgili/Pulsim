#pragma once

// =============================================================================
// Pulsim — DSED bridge: native C++ CircuitBuilderAdapter (Bridge.11)
// =============================================================================
//
// Pure-C++ port of `python/pulsim/dsed/_builder_bridge.py`. Eliminates
// the GIL roundtrip on `A_matrix() / b_vector(t) / rhs(t, x)` hot-loop
// calls that the Python adapter pays per scheduler step.
//
// Architecture
// ------------
// Holds a non-owning reference to a `PwlStateSpaceCache` and lazily
// resolves the per-mask LTI state-space `(A, b_constant, B)` via the
// existing `cache.compute_lti_state_space(mask)` extractor (Bridge.3
// + 5.1b + 6). Once resolved, subsequent visits to the same mask hit
// an `unordered_map` lookup — no MNA assembly per step.
//
// Time-varying source overlay (Bridge.6) is computed natively via the
// `compute_sine_b_extra / compute_pwm_b_extra / compute_pulse_b_extra`
// helpers — same code path that `run_transient` uses internally, so
// PWL and DSED engines agree on source values bit-for-bit.
//
// Speedup target (buck CCM, 5 ms, 100 kHz, 1007 steps)
// ----------------------------------------------------
//   Python adapter + Python scheduler (Bridge.5)  : 60.8 µs / step
//   Python adapter + native scheduler (Bridge.10) : 22.2 µs / step
//   Native adapter + native scheduler (Bridge.11) : <expected ~2-5 µs / step>
//
// MaskT
// -----
// Templated over the mask type — the canonical instantiation uses
// `topology::SwitchStateMask` which already has a `std::hash`
// specialisation. Other mask types are accepted as long as they're
// hashable and copy-comparable.

#include "pulsim/numeric/dense.hpp"
#include "pulsim/numeric/types.hpp"
#include "pulsim/pwl/cache.hpp"
#include "pulsim/pwl/device_pool.hpp"
#include "pulsim/sources/pulse_b_extra.hpp"
#include "pulsim/sources/pwm_b_extra.hpp"
#include "pulsim/sources/sine_b_extra.hpp"
#include "pulsim/topology/graph.hpp"
#include "pulsim/topology/switch_state.hpp"

#include <cstdint>
#include <functional>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

// ADL hook so `PEDSimulatorAuto<NativeCircuitBuilderAdapter<SSM>,
// SwitchFn>` can call `mode_id_of(mask)` for SSM masks (the default
// `pulsim::dsed::mode_id_of<MaskT>` template only matches integral
// / enum types). Lives in the SSM's namespace so two-phase lookup
// picks it up at template instantiation.
namespace pulsim::topology {
inline int mode_id_of(const SwitchStateMask& m) noexcept {
    // SwitchStateMask already has std::hash specialised — reuse it
    // so semantically-equal masks map to the same mode id.
    return static_cast<int>(std::hash<SwitchStateMask>{}(m));
}
}  // namespace pulsim::topology

namespace pulsim::dsed {

/// Native C++ port of `python/pulsim/dsed/_builder_bridge.py`.
/// Satisfies the `HasLTIPerMode` concept the PED schedulers expect.
template <class MaskT = topology::SwitchStateMask>
class NativeCircuitBuilderAdapter {
public:
    /// Per-mask LTI snapshot — mirror of the Python `_mask_cache`
    /// tuple. Owns its matrices (cache_'s extractor returned by value).
    struct LTIEntry {
        DenseMatrix A;                          // n_state × n_state
        Vector       b_constant;                 // n_state — DC sources
        DenseMatrix  b_projection;               // n_state × n_mna
        std::vector<Index> state_row_indices;
        std::vector<bool>  state_is_cap;
    };

    /// @param graph        Circuit's topology graph (non-owning ref).
    /// @param pool         Device pool (non-owning ref).
    /// @param cache        Built PwlStateSpaceCache (non-owning ref).
    /// @param b_extra_fn   Optional user-supplied b_extra(t) callback
    ///                     (full-MNA size). Composes additively with
    ///                     the auto-detected sine/PWM/pulse overlays.
    NativeCircuitBuilderAdapter(
        const topology::Graph& graph,
        const pwl::DevicePool& pool,
        pwl::PwlStateSpaceCache& cache,
        std::function<Vector(Real)> b_extra_fn = nullptr)
        : graph_{graph},
          pool_{pool},
          cache_{cache},
          b_extra_fn_{std::move(b_extra_fn)},
          has_dynamic_sources_{detect_dynamic_sources_()} {}

    // ----- HasLTIPerMode contract --------------------------------

    [[nodiscard]] const DenseMatrix& A_matrix() const {
        ensure_current_resolved_();
        return current_entry_->A;
    }

    [[nodiscard]] Vector b_vector(Real t) const {
        ensure_current_resolved_();
        if (!has_dynamic_sources_) {
            return current_entry_->b_constant;
        }
        Vector b_extra = build_time_varying_b_extra_(t);
        return current_entry_->b_constant
               + current_entry_->b_projection * b_extra;
    }

    [[nodiscard]] Vector rhs(Real t, const Vector& x) const {
        ensure_current_resolved_();
        Vector r = current_entry_->A * x;
        if (!has_dynamic_sources_) {
            r += current_entry_->b_constant;
        } else {
            // Inline the b_vector logic to avoid building b_extra
            // twice if rhs() is called inside a tight RK45 stage.
            Vector b_extra = build_time_varying_b_extra_(t);
            r += current_entry_->b_constant
                + current_entry_->b_projection * b_extra;
        }
        return r;
    }

    [[nodiscard]] const MaskT& current_mask() const {
        if (!current_mask_set_) {
            throw std::runtime_error(
                "NativeCircuitBuilderAdapter: current_mask not set. "
                "The scheduler should call set_mask() before any "
                "rhs / A_matrix / b_vector query.");
        }
        return current_mask_;
    }

    void set_mask(const MaskT& m) {
        current_mask_ = m;
        current_mask_set_ = true;
        current_entry_ = nullptr;   // invalidate hot pointer
    }

    // ----- Diagnostics -------------------------------------------

    /// State-vector size for the currently-active mask. Triggers a
    /// mask resolution if not already cached.
    [[nodiscard]] int n_state() const {
        ensure_current_resolved_();
        return static_cast<int>(current_entry_->A.rows());
    }

    /// All-zero initial state (the dispatcher's default when the
    /// user doesn't supply `initial_state`).
    [[nodiscard]] Vector initial_state_zero() const {
        return Vector::Zero(n_state());
    }

    /// Whether the time-varying overlay path is enabled. Useful for
    /// tests + diagnostics; the hot loop reads this once at
    /// construction and branches inside b_vector / rhs.
    [[nodiscard]] bool has_dynamic_sources() const noexcept {
        return has_dynamic_sources_;
    }

    /// Number of distinct masks resolved so far. Useful for tests:
    /// after a full PED run, this should equal the number of
    /// gate-modes the converter visited.
    [[nodiscard]] std::size_t num_cached_masks() const noexcept {
        return mask_cache_.size();
    }

private:
    // -----------------------------------------------------------------

    /// Probe `compute_*_b_extra` at t = 0 and t = 1 µs. If neither
    /// is non-zero AND there's no user callback, the circuit is
    /// DC-only and the fast path in b_vector / rhs skips the
    /// B-projection product entirely.
    [[nodiscard]] bool detect_dynamic_sources_() const {
        if (b_extra_fn_) return true;
        Vector b0 = sources::compute_pwm_b_extra(pool_, graph_, Real{0});
        b0 += sources::compute_sine_b_extra(pool_, graph_, Real{0});
        b0 += sources::compute_pulse_b_extra(pool_, graph_, Real{0});
        if (b0.norm() != Real{0}) return true;
        Vector b1 = sources::compute_pwm_b_extra(pool_, graph_, Real{1e-6});
        b1 += sources::compute_sine_b_extra(pool_, graph_, Real{1e-6});
        b1 += sources::compute_pulse_b_extra(pool_, graph_, Real{1e-6});
        return b1.norm() != Real{0};
    }

    /// Sum sine + PWM + pulse + user-supplied contributions at time t.
    /// Always returns a full-MNA-size vector even when no dynamic
    /// sources are present (the caller decides whether to short-
    /// circuit via `has_dynamic_sources_`).
    [[nodiscard]] Vector build_time_varying_b_extra_(Real t) const {
        Vector b = sources::compute_pwm_b_extra(pool_, graph_, t);
        b += sources::compute_sine_b_extra(pool_, graph_, t);
        b += sources::compute_pulse_b_extra(pool_, graph_, t);
        if (b_extra_fn_) {
            b += b_extra_fn_(t);
        }
        return b;
    }

    /// Lazily resolve the per-mask LTI entry. Sets the hot pointer
    /// `current_entry_` so subsequent calls skip the map lookup.
    void ensure_current_resolved_() const {
        if (current_entry_) return;
        if (!current_mask_set_) {
            throw std::runtime_error(
                "NativeCircuitBuilderAdapter: current_mask not set");
        }
        auto it = mask_cache_.find(current_mask_);
        if (it == mask_cache_.end()) {
            auto lti = cache_.compute_lti_state_space(current_mask_);
            LTIEntry entry{
                std::move(lti.A),
                std::move(lti.b_constant),
                std::move(lti.b_projection),
                std::move(lti.state_row_indices),
                std::move(lti.state_is_cap),
            };
            auto [iter, inserted] = mask_cache_.emplace(
                current_mask_, std::move(entry));
            current_entry_ = &iter->second;
        } else {
            current_entry_ = &it->second;
        }
    }

    // -----------------------------------------------------------------

    const topology::Graph&    graph_;
    const pwl::DevicePool&    pool_;
    pwl::PwlStateSpaceCache&  cache_;
    std::function<Vector(Real)> b_extra_fn_;
    bool                       has_dynamic_sources_;

    MaskT  current_mask_{};
    bool   current_mask_set_ = false;

    // mutable: A_matrix / b_vector / rhs are conceptually const but
    // need to populate the lazy per-mask cache.
    mutable std::unordered_map<MaskT, LTIEntry> mask_cache_;
    mutable const LTIEntry* current_entry_ = nullptr;
};

}  // namespace pulsim::dsed
