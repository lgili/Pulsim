#pragma once

// =============================================================================
// Pulsim — the stateful Newton device-kind contract and its registry
// =============================================================================
//
// add-unified-device-interface step 4 (first part). A stateful Newton kind
// (a device with memory stamped every Newton iteration: Coss, Lauritzen
// diode, IGBT tail, saturable inductor, MNA PMSM) is ONE class that models
// `StatefulKind`; the run holds one instance per kind of the list
// `pwl::StatefulKinds` and every driver operation — init, seed, commit,
// stage capture, restart extrapolation, derivative refresh, rollback,
// snapshot, the Newton stamp — is a fold over that list. A kind missing a
// member does not compile; a kind missing from the list is never stamped,
// which the conformance test names (test_engine_conformance.py).
//
// The checks below are what the list must satisfy as a whole: every
// snapshot block but the linear companion is claimed by exactly one kind,
// with the block's reals per instance, and every kind sits on a Newton row
// of the kind table (`kKindCaps`), so the engines without a Newton solve
// refuse it by name.

#include "pulsim/device/device_state.hpp"
#include "pulsim/numeric/dense.hpp"
#include "pulsim/numeric/types.hpp"
#include "pulsim/pwl/device_pool.hpp"
#include "pulsim/pwl/trbdf2_stage.hpp"
#include "pulsim/topology/graph.hpp"

#include <array>
#include <concepts>
#include <cstddef>
#include <tuple>
#include <vector>

namespace pulsim::device {

template <class... Ks> struct TypeList {
    static constexpr std::size_t size = sizeof...(Ks);
    using Tuple = std::tuple<Ks...>;
};

/// What one stateful Newton kind implements (the table in design.md,
/// "How to add a device kind"). `for_each_state(entry, f)` visits every
/// real the entry carries from step to step — the snapshot's reals AND the
/// stage scratch — so the in-run rollback restores exactly what the step
/// may have written.
template <class K>
concept StatefulKind =
    requires(K& k, const K& ck, typename K::Entry& e,
             const typename K::Entry& ce, const topology::Graph& g,
             const pwl::DevicePool& pool, const Vector& x, Real r,
             pwl::TrBdf2Stage st, const std::vector<Real>& flat) {
        { K::kBlock } -> std::convertible_to<StateBlock>;
        { K::kFlatPerDevice } -> std::convertible_to<std::size_t>;
        { K::kStoredKind } -> std::convertible_to<pwl::DevicePool::StoredKind>;
        k.init(g, pool);
        { ck.empty() } -> std::same_as<bool>;
        { ck.entries().size() } -> std::convertible_to<std::size_t>;
        { k.mutable_entries().size() } -> std::convertible_to<std::size_t>;
        k.seed_from_dc_op(x);
        k.commit(x, r, st);
        k.capture_gamma(x);
        k.extrapolate_previous(x, r);
        k.refresh_derivatives(x, r);
        { ck.to_flat() } -> std::same_as<std::vector<Real>>;
        k.from_flat(flat);
        K::for_each_state(e, [](auto&) {});
        K::for_each_state(ce, [](const auto&) {});
    };

namespace detail {
template <class List> struct ListCheck;
template <class... Ks> struct ListCheck<TypeList<Ks...>> {
    static constexpr bool kinds = (StatefulKind<Ks> && ...);
    static constexpr bool reals =
        ((kStateBlocks[static_cast<std::size_t>(Ks::kBlock)].reals_per_instance
          == Ks::kFlatPerDevice) && ...);
    static constexpr bool newton =
        (pwl::kind_caps(Ks::kStoredKind).newton() && ...);
    static constexpr bool blocks_claimed_once() {
        std::array<int, kNumStateBlocks> n{};
        ((++n[static_cast<std::size_t>(Ks::kBlock)]), ...);
        if (n[static_cast<std::size_t>(StateBlock::LinearCompanion)] != 0) return false;
        for (std::size_t b = 1; b < kNumStateBlocks; ++b) {
            if (n[b] != 1) return false;
        }
        return true;
    }
};
}  // namespace detail

/// Compile-time checks of a stateful-kind list; `static_assert` it where
/// the list is defined.
template <class List>
inline constexpr bool valid_stateful_list =
    detail::ListCheck<List>::kinds && detail::ListCheck<List>::reals &&
    detail::ListCheck<List>::newton &&
    detail::ListCheck<List>::blocks_claimed_once();

}  // namespace pulsim::device
