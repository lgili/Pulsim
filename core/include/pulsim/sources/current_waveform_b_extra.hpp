#pragma once

// =============================================================================
// Pulsim — built-in sine / pulse / table current-source pass
// =============================================================================
//
// Sibling of `sine_b_extra.hpp`: walks the DevicePool for the waveform
// current sources and writes each one's instantaneous value into the
// two NODE rows of `b_extra` (a current source has no branch row),
// with the sign `stamp_current_source` uses for the DC source:
// out[from] −= i(t), out[to] += i(t) — the value leaves at `from`.
//
// `next_current_waveform_breakpoint` is the earliest declared corner
// (pulse edges, table knots) strictly after t, for an engine that must
// not step across one.

#include "pulsim/numeric/dense.hpp"
#include "pulsim/numeric/types.hpp"
#include "pulsim/pwl/device_pool.hpp"
#include "pulsim/topology/graph.hpp"

#include <algorithm>
#include <limits>

namespace pulsim::sources {

[[nodiscard]] inline bool is_current_waveform(
    pwl::DevicePool::StoredKind k) noexcept {
    using K = pwl::DevicePool::StoredKind;
    return k == K::SineCurrentSource || k == K::PulseCurrentSource ||
           k == K::PwlCurrentSource;
}

[[nodiscard]] inline bool has_current_waveform_sources(
    const pwl::DevicePool& pool, const topology::Graph& graph) {
    for (Index b_id = 0; b_id < graph.num_branches(); ++b_id) {
        if (pool.is_registered(b_id) &&
            is_current_waveform(pool.kind_of(b_id))) {
            return true;
        }
    }
    return false;
}

/// Fill `out` (resized to the state size and zeroed) with every
/// waveform current source's value at `t`.
inline void compute_current_waveform_b_extra(
    const pwl::DevicePool& pool, const topology::Graph& graph, Real t,
    Vector& out) {
    const auto n = static_cast<Index>(pool.state_size(graph));
    if (out.size() != n) {
        out = Vector::Zero(n);
    } else {
        out.setZero();
    }
    for (Index b_id = 0; b_id < graph.num_branches(); ++b_id) {
        if (!pool.is_registered(b_id) ||
            !is_current_waveform(pool.kind_of(b_id))) {
            continue;
        }
        const auto& br = graph.branch(b_id);
        const Real i = pool.current_waveform_value(b_id, t);
        if (br.from >= 0) out[br.from] -= i;
        if (br.to >= 0) out[br.to] += i;
    }
}

[[nodiscard]] inline Real next_current_waveform_breakpoint(
    const pwl::DevicePool& pool, const topology::Graph& graph, Real t) {
    Real best = std::numeric_limits<Real>::infinity();
    for (Index b_id = 0; b_id < graph.num_branches(); ++b_id) {
        if (pool.is_registered(b_id) &&
            is_current_waveform(pool.kind_of(b_id))) {
            best = std::min(best,
                            pool.current_waveform_next_breakpoint(b_id, t));
        }
    }
    return best;
}

}  // namespace pulsim::sources
