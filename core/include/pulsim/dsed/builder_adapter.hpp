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
#include "pulsim/dsed/exact_lti.hpp"
#include "pulsim/pwl/cache.hpp"
#include "pulsim/pwl/device_pool.hpp"
#include "pulsim/sources/pulse_b_extra.hpp"
#include "pulsim/sources/pwm_b_extra.hpp"
#include "pulsim/sources/sine_b_extra.hpp"
#include "pulsim/topology/graph.hpp"
#include "pulsim/topology/switch_state.hpp"

#include <cstdint>
#include <numbers>
#include <functional>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

// Phase-0 fix #5: the old ADL hook here mapped SwitchStateMask →
// mode id by TRUNCATING std::hash to int32. Two distinct masks
// colliding in 32 bits silently poisoned StiffnessDetector's
// eigenvalue cache (wrong λ_max → wrong integrator, no diagnostic).
// The adapter now owns an injective, dense `mode_id()` (below) that
// `resolve_mode_id` in scheduler_auto.hpp prefers; the hash shim is
// deleted so any future stateless consumer fails to compile instead
// of aliasing silently.

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
        // v2.0 Phase 3 — the algebraic recovery map (see
        // ContinuousLTI): x_full = recover_from_state·x_s
        // + recover_const + recover_from_b·b_extra(t), original
        // MNA coordinates. What lets a diode predicate read v_D
        // out of the reduced state.
        DenseMatrix recover_from_state;         // n_mna × n_state
        Vector recover_const;                   // n_mna
        DenseMatrix recover_from_b;             // n_mna × n_mna
        // v2.0 Phase 3 items 2-3 — the mode's exact stepper (see
        // exact_lti.hpp). For a DC-driven circuit it decomposes A
        // itself. For a SINE-driven one (item 3) it decomposes the
        // AUGMENTED system: each distinct source frequency ω adds an
        // oscillator pair u = (sin ωt, cos ωt) with
        // u̇ = [[0,ω],[−ω,0]]·u, phases and amplitudes folded into
        // the coupling columns — the augmented system is autonomous
        // LTI, so a rectifier under mains drive steps exactly too.
        // `valid == false` (defective basis, e.g. driven exactly at
        // a circuit resonance) falls back to the numeric path.
        ExactLTI exact;
        Vector exact_b;                 // reduced, augmented-const b
        std::vector<Real> aug_omegas;   // one per oscillator pair
        bool exact_built = false;       // lazy — see item 4 note
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
          has_dynamic_sources_{detect_dynamic_sources_()},
          sine_only_sources_{detect_sine_only_()} {}

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

    /// v2.0 Phase 3 — v(from) − v(to) reconstructed from the reduced
    /// state under the CURRENT mask, via the algebraic recovery map.
    ///
    /// This is what a diode event predicate evaluates: the branch
    /// voltage of a device the reduction eliminated. Either node may
    /// be ground (kGround = −1), contributing exactly 0.
    ///
    /// Cost: two dense dot products of length n_state, plus — only
    /// when the circuit has time-varying sources — one b_extra build
    /// and two more of length n_mna. Illinois calls this ~30 times
    /// per located event; events are sparse, so correctness beats
    /// caching here.
    [[nodiscard]] Real node_pair_voltage(Index from, Index to,
                                           Real t,
                                           const Vector& x) const {
        ensure_current_resolved_();
        const LTIEntry& e = *current_entry_;
        Vector b_extra;
        if (has_dynamic_sources_) {
            b_extra = build_time_varying_b_extra_(t);
        }
        auto node_v = [&](Index n) -> Real {
            if (n < 0) {
                return Real{0};
            }
            Real v = e.recover_from_state.row(n).dot(x)
                     + e.recover_const[n];
            if (has_dynamic_sources_) {
                v += e.recover_from_b.row(n).dot(b_extra);
            }
            return v;
        };
        return node_v(from) - node_v(to);
    }

    /// v2.0 Phase 3 item 2 — the current mode's exact stepper, or
    /// nullptr when it does not apply (time-varying sources, or a
    /// defective eigenbasis for this mask). The scheduler treats
    /// nullptr as "integrate numerically", so falling back is
    /// always safe.
    /// Whether the CURRENT mask can be stepped exactly. Builds the
    /// stepper on first ask (cheap flag check afterwards).
    [[nodiscard]] bool has_exact_step() const {
        ensure_current_resolved_();
        ensure_exact_built_(*current_entry_);
        return current_entry_->exact.valid;
    }

    /// Advance the reduced state from absolute time `t` by `h`,
    /// exactly. For a sine-driven circuit the oscillator sub-state
    /// is rebuilt from `t` analytically on every call, so phase
    /// never drifts however many steps are taken.
    [[nodiscard]] Vector exact_advance_state(Real t, const Vector& x,
                                               Real h) const {
        ensure_current_resolved_();
        ensure_exact_built_(*current_entry_);
        const LTIEntry& e = *current_entry_;
        const auto n = x.size();
        const auto k =
            static_cast<Index>(e.aug_omegas.size());
        if (k == 0) {
            return exact_advance(e.exact, x, e.exact_b, h);
        }
        Vector y(n + 2 * k);
        y.head(n) = x;
        for (Index j = 0; j < k; ++j) {
            const Real w = e.aug_omegas[static_cast<Size>(j)];
            y[n + 2 * j]     = std::sin(w * t);
            y[n + 2 * j + 1] = std::cos(w * t);
        }
        return exact_advance(e.exact, y, e.exact_b, h).head(n);
    }

    /// v2.0 Phase 3 item 5 — the FULL MNA vector at time `t`,
    /// reconstructed from the reduced state under the CURRENT mask:
    /// node voltages first, then source and inductor branch
    /// currents, in exactly the pwl engine's row layout. This is
    /// what makes `result.v('sw_node')` — the most-probed waveform
    /// in power electronics — recoverable from a dsed run.
    [[nodiscard]] Vector recover_full(Real t, const Vector& x) const {
        ensure_current_resolved_();
        const LTIEntry& e = *current_entry_;
        Vector full = e.recover_from_state * x + e.recover_const;
        if (has_dynamic_sources_) {
            full += e.recover_from_b *
                    build_time_varying_b_extra_(t);
        }
        return full;
    }

    /// The graph / pool this adapter was built over — the diode
    /// predicate derivation (v2.0 Phase 3) runs its census on the
    /// same objects so the two engines cannot drift.
    [[nodiscard]] const topology::Graph& graph() const noexcept {
        return graph_;
    }
    [[nodiscard]] const pwl::DevicePool& pool() const noexcept {
        return pool_;
    }

    /// Injective, dense mode id for `mask` (Phase-0 fix #5).
    ///
    /// First-seen masks get sequential ids (0, 1, 2, …), interned in
    /// an exact-mask map — no hashing-to-int32 collision is possible,
    /// so StiffnessDetector's per-mode λ_max cache can never alias
    /// two different topologies. Preferred over the free-function
    /// `mode_id_of` by `resolve_mode_id` (scheduler_auto.hpp).
    [[nodiscard]] int mode_id(const MaskT& m) const {
        const auto next = static_cast<int>(mode_ids_.size());
        return mode_ids_.try_emplace(m, next).first->second;
    }

private:
    // -----------------------------------------------------------------

    /// STRUCTURAL detection of time-varying sources (Phase-0 fix #3).
    ///
    /// The previous implementation probed `compute_*_b_extra` at
    /// t = 0 and t = 1 µs and declared the circuit DC-only when both
    /// norms were zero. That is a correctness trap, not an
    /// optimisation: a pulse source with `delay > 1 µs` (zero at
    /// both probe instants) was classified as "no dynamic sources"
    /// and silently dropped from the ENTIRE run — the worst kind of
    /// wrong answer. Deciding structurally from the device pool has
    /// zero false negatives and is cheaper than two full b_extra
    /// builds.
    [[nodiscard]] bool detect_dynamic_sources_() const {
        if (b_extra_fn_) return true;
        using K = pwl::DevicePool::StoredKind;
        const auto n_branches = graph_.num_branches();
        for (Index b_id = 0; b_id < n_branches; ++b_id) {
            K kind;
            try {
                kind = pool_.kind_of(b_id);
            } catch (const std::out_of_range&) {
                // Branch present in the graph but not registered in
                // the pool (raw-graph construction paths). It cannot
                // contribute a time-varying b_extra overlay.
                continue;
            }
            if (kind == K::PWMVoltageSource ||
                kind == K::SineVoltageSource ||
                kind == K::PulseVoltageSource) {
                return true;
            }
        }
        return false;
    }

    /// Build the exact stepper on demand (v2.0 Phase 3 item 4).
    /// `exact.valid` stays false — with the flag set — when the
    /// circuit's sources rule it out or the eigenbasis is
    /// defective; callers fall back to the numeric path either way.
    void ensure_exact_built_(LTIEntry& entry) const {
        if (entry.exact_built) {
            return;
        }
        entry.exact_built = true;
        if (!has_dynamic_sources_) {
            entry.exact = make_exact_lti(entry.A);
            entry.exact_b = entry.b_constant;
        } else if (sine_only_sources_) {
            build_augmented_exact_(entry);
        }
        // PWM/pulse sources or a user b_extra_fn: b(t) is not a
        // finite sum of oscillators — the numeric path owns those.
    }

    /// True iff every time-varying contribution is a
    /// SineVoltageSource (no PWM, no pulse, no user callback) —
    /// the class the augmented-oscillator exact stepper covers.
    [[nodiscard]] bool detect_sine_only_() const {
        if (!has_dynamic_sources_ || b_extra_fn_) {
            return false;
        }
        using K = pwl::DevicePool::StoredKind;
        bool any_sine = false;
        for (Index b_id = 0; b_id < graph_.num_branches(); ++b_id) {
            K kind;
            try {
                kind = pool_.kind_of(b_id);
            } catch (const std::out_of_range&) {
                continue;
            }
            if (kind == K::PWMVoltageSource ||
                kind == K::PulseVoltageSource) {
                return false;
            }
            if (kind == K::SineVoltageSource) {
                any_sine = true;
            }
        }
        return any_sine;
    }

    /// Build the augmented exact stepper for a sine-driven mode.
    ///
    ///   y = [x; u₁; …; u_k],  u_j = (sin ω_j t, cos ω_j t)
    ///   ẏ = [[A, Q],[0, blkdiag(S_j)]]·y + [b̃₀; 0]
    ///
    /// where Q folds every source's amplitude AND phase into the
    /// (sin, cos) pair of its frequency:
    ///   amp·sin(ωt+φ) = amp·cosφ·sin(ωt) + amp·sinφ·cos(ωt),
    /// and b̃₀ carries the reduced DC part (b_constant plus each
    /// sine source's v_dc through its −1 branch-row stamp, matching
    /// compute_sine_b_extra's sign convention exactly).
    void build_augmented_exact_(LTIEntry& entry) const {
        using K = pwl::DevicePool::StoredKind;
        struct SineInfo {
            Index src_var;
            Real omega, amp, phase, v_dc;
        };
        std::vector<SineInfo> sines;
        for (Index b_id = 0; b_id < graph_.num_branches(); ++b_id) {
            const auto& branch = graph_.branch(b_id);
            if (branch.kind != topology::BranchKind::Source ||
                !pool_.is_registered(b_id)) {
                continue;
            }
            if (pool_.kind_of(b_id) != K::SineVoltageSource) {
                continue;
            }
            const auto& p = pool_.sine_voltage_source_params(b_id);
            if (p.frequency <= Real{0}) {
                continue;    // value_at degenerates to v_dc: DC-only
            }
            sines.push_back({
                pool_.branch_var_id_for_source(b_id, graph_),
                Real{2} * std::numbers::pi_v<Real> * p.frequency,
                p.v_amplitude, p.phase, p.v_dc});
        }

        const auto n = entry.A.rows();
        const auto n_mna =
            static_cast<Index>(entry.b_projection.cols());

        // DC part: v_dc of every sine source (frequency > 0 or not)
        // arrives through the same −1 stamp the overlay uses.
        Vector b_dc_mna = Vector::Zero(n_mna);
        for (Index b_id = 0; b_id < graph_.num_branches(); ++b_id) {
            const auto& branch = graph_.branch(b_id);
            if (branch.kind != topology::BranchKind::Source ||
                !pool_.is_registered(b_id)) {
                continue;
            }
            if (pool_.kind_of(b_id) != K::SineVoltageSource) {
                continue;
            }
            const auto& p = pool_.sine_voltage_source_params(b_id);
            b_dc_mna[pool_.branch_var_id_for_source(b_id, graph_)]
                += -p.v_dc;
        }

        // Distinct frequencies, amplitude+phase folded per source.
        std::vector<Real> omegas;
        std::vector<Vector> q_sin, q_cos;    // reduced coupling cols
        for (const auto& sn : sines) {
            Index j = kInvalidIndex;
            for (Size m = 0; m < omegas.size(); ++m) {
                if (omegas[m] == sn.omega) {
                    j = static_cast<Index>(m);
                    break;
                }
            }
            if (j == kInvalidIndex) {
                omegas.push_back(sn.omega);
                q_sin.push_back(Vector::Zero(n));
                q_cos.push_back(Vector::Zero(n));
                j = static_cast<Index>(omegas.size() - 1);
            }
            // −amp·sin(ωt+φ) on the source row, reduced through
            // b_projection.
            Vector stamp = Vector::Zero(n_mna);
            stamp[sn.src_var] = Real{-1};
            const Vector col = entry.b_projection * stamp;
            q_sin[static_cast<Size>(j)]
                += col * (sn.amp * std::cos(sn.phase));
            q_cos[static_cast<Size>(j)]
                += col * (sn.amp * std::sin(sn.phase));
        }

        const auto k = static_cast<Index>(omegas.size());
        DenseMatrix A_aug =
            DenseMatrix::Zero(n + 2 * k, n + 2 * k);
        A_aug.topLeftCorner(n, n) = entry.A;
        for (Index j = 0; j < k; ++j) {
            const Real w = omegas[static_cast<Size>(j)];
            A_aug.block(0, n + 2 * j, n, 1) =
                q_sin[static_cast<Size>(j)];
            A_aug.block(0, n + 2 * j + 1, n, 1) =
                q_cos[static_cast<Size>(j)];
            A_aug(n + 2 * j,     n + 2 * j + 1) =  w;   // ṡ =  w·c
            A_aug(n + 2 * j + 1, n + 2 * j)     = -w;   // ċ = −w·s
        }

        Vector b_aug = Vector::Zero(n + 2 * k);
        b_aug.head(n) =
            entry.b_constant + entry.b_projection * b_dc_mna;

        entry.exact = make_exact_lti(A_aug);
        entry.exact_b = std::move(b_aug);
        entry.aug_omegas = std::move(omegas);
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
                std::move(lti.recover_from_state),
                std::move(lti.recover_const),
                std::move(lti.recover_from_b),
            };
            // The exact stepper's eigendecomposition is built
            // LAZILY, on the first exact_advance_state call —
            // measured at N = 400 states it is 252 ms of the
            // 254 ms mask-resolve cost, and the Auto scheduler
            // resolves masks without ever stepping exactly.
            // (v2.0 Phase 3 item 4.)
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
    bool                       sine_only_sources_;

    MaskT  current_mask_{};
    bool   current_mask_set_ = false;

    // mutable: A_matrix / b_vector / rhs are conceptually const but
    // need to populate the lazy per-mask cache.
    mutable std::unordered_map<MaskT, LTIEntry> mask_cache_;
    // Phase-0 fix #5: exact-mask → dense sequential mode id intern
    // table backing mode_id(). Grows by #visited modes only.
    mutable std::unordered_map<MaskT, int> mode_ids_;
    mutable LTIEntry* current_entry_ = nullptr;
};

}  // namespace pulsim::dsed
