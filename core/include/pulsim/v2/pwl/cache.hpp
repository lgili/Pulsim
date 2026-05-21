#pragma once

// =============================================================================
// Pulsim v2 — Layer 4: PwlStateSpaceCache (the PLECS-killer)
// =============================================================================
//
// `pulsim-v2-pwl-state-space-cache` Phase 4.
//
// THE architectural pivot. For each switch combination Layer 1
// enumerates, pre-build the per-segment MNA matrix (Layer 3
// stampers + Layer 2 device models) and pre-factorise via Layer 0's
// DirectSolver. Per-step hot loop: hash-map lookup + triangular
// solve. NO assemble, NO factorize, NO Newton iteration per step.
//
// Expected speedup vs v1: 10-50× on PE workloads. This is THE
// architectural reason v2 will compete with PLECS.
//
// V0 scope: static-only circuits (Resistor + VoltageSource +
// Switch). Capacitor/Inductor (trapezoidal companion + history) and
// nonlinear devices (per-segment Newton) are V1+ extensions.

#include "pulsim/v2/numeric/dense.hpp"
#include "pulsim/v2/numeric/types.hpp"
#include "pulsim/v2/pwl/assemble.hpp"
#include "pulsim/v2/pwl/device_pool.hpp"
#include "pulsim/v2/pwl/segment.hpp"
#include "pulsim/v2/sparse/matrix.hpp"
#include "pulsim/v2/sparse/solver.hpp"
#include "pulsim/v2/topology/enumerator.hpp"
#include "pulsim/v2/topology/graph.hpp"
#include "pulsim/v2/topology/switch_state.hpp"

#include <stdexcept>
#include <unordered_map>

namespace pulsim::v2::pwl {

class PwlStateSpaceCache {
public:
    PwlStateSpaceCache(const topology::Graph& graph,
                        const DevicePool& pool) noexcept
        : graph_{graph}, pool_{pool} {}

    /// Build all 2^N segments and pre-factorize each (V1 dt-aware).
    ///
    /// `dt > 0` enables dynamic-device stamping (Capacitor /
    /// Inductor trap companion). `dt = 0` (or the no-arg overload)
    /// skips dynamic devices and produces a V0-identical static
    /// build.
    ///
    /// Calling build(dt) twice with different dt CLEARs and
    /// rebuilds every factor (the matrix is dt-dependent).
    ///
    /// Time:   O(2^N · (assemble + analyze + factorize))
    /// Memory: O(2^N · (nnz(J) + nnz(L) + nnz(U)))
    void build(Real dt) {
        dt_ = dt;
        segments_.clear();
        const Size num_switches = graph_.num_switches();
        sparse::Matrix J;
        Vector b;
        for (auto mask :
             topology::enumerate_switch_states(num_switches)) {
            assemble_segment(graph_, pool_, mask, dt, J, b);
            sparse::compress_in_place(J);

            auto solver = sparse::make_default_solver();
            // analyze + factorize ONCE per segment. The result is
            // cached and reused across every simulation step that
            // visits this switch state.
            if (!solver->analyze(J)) {
                throw std::runtime_error(
                    "PwlStateSpaceCache::build: structurally singular "
                    "matrix for switch state " + mask.to_string());
            }
            if (!solver->factorize(J)) {
                throw std::runtime_error(
                    "PwlStateSpaceCache::build: numerically singular "
                    "matrix for switch state " + mask.to_string());
            }

            PwlSegment seg;
            seg.J = std::move(J);
            seg.b_constant = std::move(b);
            seg.solver = std::move(solver);
            seg.state_size = pool_.state_size(graph_);
            segments_.emplace(mask, std::move(seg));
        }
    }

    /// V0 backwards-compat overload — static-only build (dt = 0).
    /// Caps and Inductors are silently skipped.
    void build() { build(Real{0}); }

    /// Currently-built dt. Returns 0 for static-only builds.
    /// Layer 5's run_transient checks this against opts.dt and
    /// throws on mismatch.
    [[nodiscard]] Real dt() const noexcept { return dt_; }

    /// O(1) segment lookup. Throws if the mask wasn't built (which
    /// only happens if the caller passes a SwitchStateMask of the
    /// wrong size — every valid mask for the graph IS in the cache
    /// after build()).
    [[nodiscard]] const PwlSegment& lookup(
        const topology::SwitchStateMask& mask) const {
        const auto it = segments_.find(mask);
        if (it == segments_.end()) {
            throw std::out_of_range(
                "PwlStateSpaceCache::lookup: no segment built for "
                "mask " + mask.to_string());
        }
        return it->second;
    }

    /// HOT LOOP entry point. Look up the segment for `mask` and
    /// solve `J · x = -(b_constant + b_extra)` via the cached
    /// factor. `b_extra` lets Layer 5 inject time-varying source
    /// values or history terms in V1; in V0 it's typically zero.
    ///
    /// Newton's convention is solve(J, -f, dx). For Layer 4's
    /// static-only V0 path the residual at convergence is
    /// f = J·x + b_total = 0, so the solution is x = -(L^-1 ·
    /// (U^-1 · b_total)) = solver.solve(-b_total). We compute that
    /// directly: rhs = -(b_constant + b_extra), then solve.
    void solve(const topology::SwitchStateMask& mask,
                const Vector& b_extra,
                Vector& x) const {
        const PwlSegment& seg = lookup(mask);
        Vector rhs = -(seg.b_constant + b_extra);
        seg.solver->solve(rhs, x);
    }

    [[nodiscard]] Size num_segments() const noexcept {
        return segments_.size();
    }

private:
    const topology::Graph& graph_;
    const DevicePool& pool_;
    std::unordered_map<topology::SwitchStateMask, PwlSegment>
        segments_;
    Real dt_ = Real{0};   // 0 = static-only build (V0)
};

}  // namespace pulsim::v2::pwl
