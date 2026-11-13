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
#include "pulsim/dsed/event_projection.hpp"
#include "pulsim/dsed/exact_lti.hpp"
#include "pulsim/events/mode_refusal.hpp"
#include "pulsim/events/mode_selector.hpp"
#include "pulsim/pwl/cache.hpp"
#include "pulsim/pwl/device_pool.hpp"
#include "pulsim/solver/unsupported.hpp"
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
        std::vector<Real> state_mass;   // C or L per state (energy scale)
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
          sine_only_sources_{detect_sine_only_()} {
        // The LTI extraction drops Newton devices; refuse them here so
        // every caller (not only simulate()) gets the typed refusal.
        require_supported(graph, pool, pwl::Engine::Dsed);
    }

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
        Vector out(x.size());
        exact_advance_state_into(t, x, h, out);
        return out;
    }

    /// Same, into `out` (sized like x, distinct from x): no heap
    /// allocation once the mode's workspaces exist.
    void exact_advance_state_into(Real t, const Vector& x, Real h,
                                  Vector& out) const {
        ensure_current_resolved_();
        ensure_exact_built_(*current_entry_);
        const LTIEntry& e = *current_entry_;
        const auto n = x.size();
        const auto k =
            static_cast<Index>(e.aug_omegas.size());
        if (k == 0) {
            exact_advance_into(e.exact, x, h, out);
            return;
        }
        aug_in_.resize(n + 2 * k);
        aug_out_.resize(n + 2 * k);
        aug_in_.head(n) = x;
        for (Index j = 0; j < k; ++j) {
            const Real w = e.aug_omegas[static_cast<Size>(j)];
            aug_in_[n + 2 * j]     = std::sin(w * t);
            aug_in_[n + 2 * j + 1] = std::cos(w * t);
        }
        exact_advance_into(e.exact, aug_in_, h, aug_out_);
        out = aug_out_.head(n);
    }

    /// Event-scan step of the current mode (exact_scan_step).
    [[nodiscard]] Real exact_scan_step() const {
        ensure_current_resolved_();
        ensure_exact_built_(*current_entry_);
        return pulsim::dsed::exact_scan_step(current_entry_->exact);
    }

    /// The tier-1 bound's longest step (D2); the scheduler sets its
    /// dt_max. Changing it rebuilds the propagators lazily.
    void set_max_step(Real h) {
        if (h == h_max_) {
            return;
        }
        h_max_ = h;
        for (auto& [m, e] : mask_cache_) {
            e.exact_built = false;
            e.exact = ExactLTI{};
        }
    }

    /// Consistent reinitialisation onto the current mode's slow
    /// manifold (event_projection.hpp), through the cached tier-1
    /// basis when there is one (no eigendecomposition per event).
    void project_slow(Real t, Vector& x, Real fast_threshold) const {
        ensure_current_resolved_();
        ensure_exact_built_(*current_entry_);
        const LTIEntry& e = *current_entry_;
        if (e.aug_omegas.empty()
            && exact_project_slow(e.exact, x, fast_threshold)) {
            return;
        }
        x = project_onto_slow_manifold(e.A, b_vector(t), x,
                                       fast_threshold);
    }

    /// D5 — builder initial conditions (`c0=` / `i0=` / set_initial,
    /// by branch id: CircuitBuilder::initial_conditions()) on the
    /// reduced state of the CURRENT mask. An inductor state takes its
    /// branch current. The capacitor states minimise
    /// Σ C_k·(v_k(x) − c0_k)² over the capacitors that declared a value
    /// (v_k = the branch voltage v_from − v_to, a floating capacitor
    /// included): exact when the declared values are consistent, the
    /// charge-conserving projection when a capacitor loop makes them
    /// inconsistent (1 µF at 10 V ∥ 3 µF at 2 V → 4 V); an undeclared
    /// capacitor is free (minimum-norm). Returns the state and the
    /// largest |v_k(x) − c0_k| (0 when consistent).
    [[nodiscard]] std::pair<Vector, Real> reduced_initial_state(
        const std::unordered_map<Index, Real>& ics) const {
        ensure_current_resolved_();
        const LTIEntry& e = *current_entry_;
        const auto n = e.A.rows();
        Vector xs = Vector::Zero(n);
        std::vector<Index> cap_states;
        for (Index j = 0; j < n; ++j) {
            if (e.state_is_cap[static_cast<Size>(j)]) {
                cap_states.push_back(j);
            }
        }
        struct CapRow { Index from, to; Real C, c0; };
        std::vector<CapRow> caps;
        for (const auto& [bid, value] : ics) {
            if (bid < 0 || bid >= graph_.num_branches()
                || !pool_.is_registered(bid)) {
                continue;
            }
            const auto k = pool_.kind_of(bid);
            if (k == pwl::DevicePool::StoredKind::Inductor) {
                const Index bv = pool_.branch_var_id_for_inductor(bid, graph_);
                for (Index j = 0; j < n; ++j) {
                    const auto ju = static_cast<Size>(j);
                    if (!e.state_is_cap[ju] && e.state_row_indices[ju] == bv) {
                        xs[j] = value;
                    }
                }
            } else if (k == pwl::DevicePool::StoredKind::Capacitor) {
                const auto& br = graph_.branch(bid);
                caps.push_back({br.from, br.to,
                                pool_.capacitor_params(bid).C, value});
            }
        }
        const auto m = static_cast<Index>(caps.size());
        const auto nc = static_cast<Index>(cap_states.size());
        if (m == 0 || nc == 0) {
            return {xs, Real{0}};
        }
        // v_k(x) is affine in x: v_k = c_k + Σ_j r_kj·x_j.
        DenseMatrix Mw(m, nc);
        Vector rw(m), c(m);
        for (Index k = 0; k < m; ++k) {
            const auto& cp = caps[static_cast<Size>(k)];
            c[k] = node_pair_voltage(cp.from, cp.to, Real{0}, xs);
        }
        for (Index jj = 0; jj < nc; ++jj) {
            Vector xp = xs;
            xp[cap_states[static_cast<Size>(jj)]] += Real{1};
            for (Index k = 0; k < m; ++k) {
                const auto& cp = caps[static_cast<Size>(k)];
                Mw(k, jj) = (node_pair_voltage(cp.from, cp.to, Real{0}, xp)
                             - c[k]) * std::sqrt(cp.C);
            }
        }
        for (Index k = 0; k < m; ++k) {
            const auto& cp = caps[static_cast<Size>(k)];
            rw[k] = (cp.c0 - c[k]) * std::sqrt(cp.C);
        }
        const Vector y = Mw.completeOrthogonalDecomposition().solve(rw);
        for (Index jj = 0; jj < nc; ++jj) {
            xs[cap_states[static_cast<Size>(jj)]] = y[jj];
        }
        Real adj = Real{0};
        for (const auto& cp : caps) {
            adj = std::max(adj, std::abs(
                node_pair_voltage(cp.from, cp.to, Real{0}, xs) - cp.c0));
        }
        return {xs, adj};
    }

    /// Largest |node voltage| at (t, x): the voltage scale of the
    /// scheduler's zero band.
    [[nodiscard]] Real max_node_voltage(Real t, const Vector& x) const {
        Real m = Real{0};
        for (Index n = 0; n < graph_.num_nodes(); ++n) {
            m = std::max(m, std::abs(node_pair_voltage(n, -1, t, x)));
        }
        return m;
    }

    /// The full MNA vector (recover_full) written into `out`.
    void recover_full_into(Real t, const Vector& x,
                           Eigen::Ref<Vector> out) const {
        ensure_current_resolved_();
        const LTIEntry& e = *current_entry_;
        out.noalias() = e.recover_from_state * x;
        out += e.recover_const;
        if (has_dynamic_sources_) {
            out.noalias() += e.recover_from_b *
                             build_time_varying_b_extra_(t);
        }
    }

    /// One row per visited mask: the propagator tier, κ₂ and the
    /// reason for any tier other than 1 (D10).
    struct ExactReportRow {
        MaskT mask;
        int tier;
        Real kappa;
        Real norm_Ah;
        std::string reason;
    };
    [[nodiscard]] std::vector<ExactReportRow> exact_report() const {
        std::vector<ExactReportRow> rows;
        for (const auto& [m, e] : mask_cache_) {
            if (e.exact_built) {
                rows.push_back({m, e.exact.tier, e.exact.kappa,
                                e.exact.norm_A1 * h_max_,
                                e.exact.reason});
            }
        }
        return rows;
    }
    /// Eigendecompositions built, Padé exponentials evaluated, Padé
    /// cache hits — summed over the visited masks.
    [[nodiscard]] std::size_t n_eig() const noexcept { return n_eig_; }
    [[nodiscard]] std::size_t n_pade() const {
        std::size_t s = 0;
        for (const auto& [m, e] : mask_cache_) s += e.exact.n_pade;
        return s;
    }
    [[nodiscard]] std::size_t n_pade_hits() const {
        std::size_t s = 0;
        for (const auto& [m, e] : mask_cache_) s += e.exact.n_pade_hits;
        return s;
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
    /// D's diagonal: √m per state, then `n_aug` unit entries for the
    /// sine oscillator sub-state.
    [[nodiscard]] static Vector energy_scale_(const LTIEntry& entry,
                                              Index n_aug) {
        const auto n = entry.A.rows();
        Vector d = Vector::Ones(n + n_aug);
        if (static_cast<Index>(entry.state_mass.size()) == n) {
            for (Index i = 0; i < n; ++i) {
                d[i] = std::sqrt(std::abs(
                    entry.state_mass[static_cast<Size>(i)]));
            }
        }
        return d;
    }

    void ensure_exact_built_(LTIEntry& entry) const {
        if (entry.exact_built) {
            return;
        }
        entry.exact_built = true;
        if (!has_dynamic_sources_) {
            entry.exact_b = entry.b_constant;
            entry.exact = make_exact_lti(entry.A, entry.exact_b,
                                         energy_scale_(entry, 0), h_max_);
            ++n_eig_;
        } else if (sine_only_sources_) {
            build_augmented_exact_(entry);
            ++n_eig_;
        } else {
            entry.exact.reason = b_extra_fn_
                ? "b_extra_fn: a user source function has no closed form"
                : "PWM / pulse source: not piecewise-constant events yet";
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

        entry.exact = make_exact_lti(A_aug, b_aug,
                                     energy_scale_(entry, 2 * k), h_max_);
        entry.exact_b = std::move(b_aug);
        entry.aug_omegas = std::move(omegas);
    }

public:
    /// add-complementarity-mode-selection D2 / task 3.1 — the k-port of
    /// the diode ports `ports` (anode, cathode node) in the backward-
    /// Euler companion network of step h under `mask`, from the state x
    /// held at t (read through the CURRENT mask's recovery map; only
    /// capacitor voltages and inductor currents enter). u: the port
    /// voltages of that network's solution; Z: v = u − Z·e for a current
    /// e drawn from each anode into its cathode. One factor per (mask, h)
    /// — the cache's event solver (`solve_at` at dt = 2h, since its
    /// companion is trapezoidal: G + C_coef/dt with C_coef = 2C, −2L) —
    /// and k + 1 back-substitutions.
    void diode_port_problem(const MaskT& mask, Real t, const Vector& x,
                            Real h,
                            const std::vector<std::pair<Index, Index>>& ports,
                            DenseMatrix& Z, Vector& u) const {
        ensure_current_resolved_();
        const LTIEntry& e = *current_entry_;
        Vector& xf = port_x_full_;
        xf = e.recover_from_state * x + e.recover_const;
        if (has_dynamic_sources_) {
            xf += e.recover_from_b * build_time_varying_b_extra_(t);
        }
        diode_port_problem_full(mask, t, xf, h, ports, Z, u);
    }

    /// Same, from a full-MNA vector at t⁻ (capacitor branch voltages and
    /// inductor currents are read from it; the rest is ignored).
    void diode_port_problem_full(
        const MaskT& mask, Real t, const Vector& xf, Real h,
        const std::vector<std::pair<Index, Index>>& ports, DenseMatrix& Z,
        Vector& u) const {
        if (!port_C_set_) {
            sparse::Matrix G;
            Vector b;
            pwl::assemble_segment_split(graph_, pool_, mask, G, port_C_, b);
            port_C_set_ = true;
        }
        auto it = port_b_.find(mask);
        if (it == port_b_.end()) {
            sparse::Matrix G, C;
            Vector b;
            pwl::assemble_segment_split(graph_, pool_, mask, G, C, b);
            it = port_b_.emplace(mask, std::move(b)).first;
        }
        const Vector& b_c = it->second;
        const Real dt = Real{2} * h;
        // BE history −(C_phys/h)·x(t⁻) = −(C_coef/(2h))·x(t⁻); sources at t + h
        Vector& rhs = port_rhs_;
        rhs = (Real{-1} / dt) * (port_C_ * xf);
        if (has_dynamic_sources_) {
            rhs += build_time_varying_b_extra_(t + h);
        }
        Vector& sol = port_sol_;
        cache_.solve_at(mask, dt, rhs, sol);
        const auto k = static_cast<Eigen::Index>(ports.size());
        auto vp = [&](const Vector& s, std::size_t j) {
            const auto [a, c] = ports[j];
            return (a >= 0 ? s[a] : Real{0}) - (c >= 0 ? s[c] : Real{0});
        };
        u.resize(k);
        Z.resize(k, k);
        for (Eigen::Index j = 0; j < k; ++j) u[j] = vp(sol, static_cast<std::size_t>(j));
        for (Eigen::Index j = 0; j < k; ++j) {
            // J·s = −(b_c + b_extra) with b_extra = E_j − b_c: J·s = −E_j
            rhs = -b_c;
            const auto [a, c] = ports[static_cast<std::size_t>(j)];
            if (a >= 0) rhs[a] += Real{1};
            if (c >= 0) rhs[c] -= Real{1};
            cache_.solve_at(mask, dt, rhs, sol);
            for (Eigen::Index i = 0; i < k; ++i)
                Z(i, j) = -vp(sol, static_cast<std::size_t>(i));
        }
        ++n_port_solves_;
    }

    /// Circuit scales of the LCP tolerances (design D4, prototype
    /// `_lcp_at`): vscale = largest source / drop / node-voltage
    /// magnitude (1 V floor), iscale = largest current-source, branch
    /// current (inductors, sources) or vscale·G of a resistor, c_max =
    /// largest capacitance. State read under the CURRENT mask.
    [[nodiscard]] events::CircuitScale port_scale(Real t,
                                                  const Vector& x) const {
        ensure_current_resolved_();
        const LTIEntry& e = *current_entry_;
        Vector xf = e.recover_from_state * x + e.recover_const;
        if (has_dynamic_sources_) {
            xf += e.recover_from_b * build_time_varying_b_extra_(t);
        }
        return port_scale_full(xf);
    }

    /// Same, from a full-MNA vector.
    [[nodiscard]] events::CircuitScale port_scale_full(const Vector& xf) const {
        const Index nn = graph_.num_nodes();
        events::CircuitScale sc;
        Real vs = Real{1}, is = Real{1e-12}, gmax = Real{0};
        for (Index i = 0; i < static_cast<Index>(xf.size()); ++i) {
            if (i < nn) vs = std::max(vs, std::abs(xf[i]));
            else is = std::max(is, std::abs(xf[i]));
        }
        using K = pwl::DevicePool::StoredKind;
        for (Index b = 0; b < graph_.num_branches(); ++b) {
            K kind;
            try {
                kind = pool_.kind_of(b);
            } catch (const std::out_of_range&) {
                continue;            // a branch without pool params
            }
            switch (kind) {
            case K::VoltageSource:
                vs = std::max(vs, std::abs(pool_.voltage_source_params(b).V));
                break;
            case K::SineVoltageSource: {
                const auto& p = pool_.sine_voltage_source_params(b);
                vs = std::max(vs, std::abs(p.v_dc) + std::abs(p.v_amplitude));
                break;
            }
            case K::PWMVoltageSource: {
                const auto& p = pool_.pwm_voltage_source_params(b);
                vs = std::max({vs, std::abs(p.v_high), std::abs(p.v_low)});
                break;
            }
            case K::Diode: {
                const auto& p = pool_.diode_params(b);
                vs = std::max({vs, std::abs(p.V_f), std::abs(p.V_th)});
                break;
            }
            case K::CurrentSource:
                is = std::max(is, std::abs(pool_.current_source_params(b).I));
                break;
            case K::Resistor:
                gmax = std::max(gmax, std::abs(pool_.resistor_params(b).G));
                break;
            case K::Capacitor:
                sc.c_max = std::max(sc.c_max, pool_.capacitor_params(b).C);
                break;
            default:
                break;
            }
        }
        sc.vscale = vs;
        sc.iscale = std::max(is, vs * gmax);
        return sc;
    }

    /// add-complementarity-mode-selection D3 / task 5.1: structural
    /// passivity at the diode ports. A VCVS (gain), or a resistor,
    /// capacitor or inductor with a negative value, makes the circuit
    /// non-passive; the first one found is named. The ideal transformer
    /// is passive (not reciprocal): Lemke keeps its guarantee.
    [[nodiscard]] events::PassivityClass passivity() const {
        using K = pwl::DevicePool::StoredKind;
        for (Index b = 0; b < graph_.num_branches(); ++b) {
            if (!pool_.is_registered(b)) continue;
            const K kind = pool_.kind_of(b);
            const bool neg =
                (kind == K::Resistor && pool_.resistor_params(b).G < Real{0})
                || (kind == K::Capacitor && pool_.capacitor_params(b).C < Real{0})
                || (kind == K::Inductor && pool_.inductor_params(b).L < Real{0});
            if (kind == K::VCVS || neg)
                return {std::string(graph_.branch_name(b)),
                        std::string(pwl::stored_kind_name(kind))};
        }
        return {};
    }

    /// Port-model k-port builds (each k + 1 back-substitutions).
    [[nodiscard]] std::size_t n_port_problems() const noexcept {
        return n_port_solves_;
    }

private:
    mutable std::unordered_map<MaskT, Vector> port_b_;
    mutable sparse::Matrix port_C_;
    mutable bool port_C_set_ = false;
    mutable Vector port_x_full_, port_rhs_, port_sol_;
    mutable std::size_t n_port_solves_ = 0;

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
            entry.state_mass = std::move(lti.state_mass);
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
    Real h_max_ = Real{1e-5};
    mutable std::size_t n_eig_ = 0;
    mutable Vector aug_in_, aug_out_;
};

}  // namespace pulsim::dsed
