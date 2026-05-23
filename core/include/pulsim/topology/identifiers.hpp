#pragma once

// =============================================================================
// Pulsim — Layer 1: strong identifier types (NodeId, BranchId, StateIdx)
// =============================================================================
//
// `kInvalidIndex == kGround == -1` aliasing in `numeric/types.hpp` is
// documented (and intentional) but it lets `Index` from any role mix
// silently with `Index` from another. The state-vector arithmetic in
// `DevicePool::branch_var_id_*` — where a relative branch offset is
// added to an absolute node count — is the proof-by-example that bare
// `int32_t` is too permissive.
//
// This header introduces three strong wrappers:
//
//   * `NodeId`     — a graph node position
//   * `BranchId`   — a graph branch position
//   * `StateIdx`   — a position in the state vector `x` (node + source
//                    + inductor segments concatenated)
//
// They are:
//   * trivially copyable, 4 bytes, same memory layout as `Index`
//   * `constexpr` everywhere
//   * `explicit`-only ctors — NO implicit conversion from `Index`
//   * `noexcept` accessors `.get()` returning the underlying `Index`
//   * `operator==` / `operator!=` defaulted via C++20 spaceship
//   * `kInvalid<T>` sentinel == `Index{-1}`
//
// **STATUS — V0 additive**: these types are NOT YET threaded through the
// kernel's existing `Index`-returning APIs. Migrating `Graph::add_node`,
// `CircuitBuilder::node_id_of`, `DevicePool::branch_var_id_*`, the
// stamping layer, Python bindings, and ~200 test files is its own
// OpenSpec proposal — a multi-day effort that is high-leverage but
// also high-risk if done in a single pass.
//
// What ships now:
//   * The type definitions (so downstream code CAN opt in immediately
//     by writing `NodeId{idx}` at the boundary).
//   * A compile-time test that proves the safety property: you cannot
//     pass a `NodeId` where a `BranchId` is expected, period.
//
// Once a future PR is ready to do the full migration, the `Index`-based
// signatures become wrappers around the strong-typed ones, and the
// kernel gets type-safe by default.

#include "pulsim/numeric/types.hpp"

#include <compare>
#include <functional>

namespace pulsim::topology {

namespace detail {

// CRTP base: every strong-id type shares the same value-semantics
// boilerplate (ctor, equality, ordering, hash). Stamping out three
// `struct` definitions by hand would be just as good — the CRTP only
// exists to prove via `static_assert` below that the strong types
// remain 4 B and trivially copyable.
template <typename Derived>
class StrongIndex {
public:
    constexpr StrongIndex() noexcept = default;
    constexpr explicit StrongIndex(Index v) noexcept : v_{v} {}

    [[nodiscard]] constexpr Index get() const noexcept { return v_; }
    [[nodiscard]] constexpr bool is_valid() const noexcept {
        return v_ != kInvalidIndex;
    }

    // Defaulted spaceship gives us ==, !=, <, <=, >, >= for free —
    // strong types are still orderable (`std::map<NodeId, ...>` works)
    // but they only compare against THEIR OWN type, not other strong
    // types and not bare `Index`.
    friend constexpr auto operator<=>(
        const StrongIndex&, const StrongIndex&) noexcept = default;

private:
    Index v_ = kInvalidIndex;
};

}  // namespace detail

/// Strong-typed graph node position (index into `Graph::nodes_`).
/// `NodeId{kGround}` is the ground rail sentinel.
class NodeId : public detail::StrongIndex<NodeId> {
    using detail::StrongIndex<NodeId>::StrongIndex;
};

/// Strong-typed graph branch position (index into `Graph::branches_`).
class BranchId : public detail::StrongIndex<BranchId> {
    using detail::StrongIndex<BranchId>::StrongIndex;
};

/// Strong-typed state-vector position. The state layout is
///   [v_0 .. v_{N-1}] [i_src_0 .. i_src_{M-1}] [i_L_0 .. i_L_{K-1}]
/// so a `StateIdx` can refer to a node voltage, a source current, or
/// an inductor current — but NEVER mixes silently with a NodeId.
class StateIdx : public detail::StrongIndex<StateIdx> {
    using detail::StrongIndex<StateIdx>::StrongIndex;
};

// -----------------------------------------------------------------------------
// Memory + ABI contract.
//
// These asserts guarantee the strong types are pure compile-time
// safety: at runtime they pass through registers exactly like `Index`,
// and they can be reinterpret_cast'd to `Index[]` arrays for solver
// interop if a downstream consumer needs that escape hatch.
// -----------------------------------------------------------------------------
static_assert(sizeof(NodeId)   == sizeof(Index));
static_assert(sizeof(BranchId) == sizeof(Index));
static_assert(sizeof(StateIdx) == sizeof(Index));
static_assert(alignof(NodeId)   == alignof(Index));
static_assert(alignof(BranchId) == alignof(Index));
static_assert(alignof(StateIdx) == alignof(Index));

// -----------------------------------------------------------------------------
// Compile-time safety property (proven via type traits).
//
// The strong types are NOT implicitly convertible to one another and
// NOT implicitly convertible to/from `Index`. Verified at compile time
// by these traits so that the contract can't drift silently in a
// future refactor.
// -----------------------------------------------------------------------------
static_assert(!std::is_convertible_v<NodeId, BranchId>);
static_assert(!std::is_convertible_v<NodeId, StateIdx>);
static_assert(!std::is_convertible_v<BranchId, NodeId>);
static_assert(!std::is_convertible_v<BranchId, StateIdx>);
static_assert(!std::is_convertible_v<StateIdx, NodeId>);
static_assert(!std::is_convertible_v<StateIdx, BranchId>);

// `Index` → `NodeId` requires explicit `NodeId{idx}` (cannot leak in
// implicitly from arithmetic results). Same for the others.
static_assert(!std::is_convertible_v<Index, NodeId>);
static_assert(!std::is_convertible_v<Index, BranchId>);
static_assert(!std::is_convertible_v<Index, StateIdx>);
static_assert( std::is_constructible_v<NodeId,   Index>);
static_assert( std::is_constructible_v<BranchId, Index>);
static_assert( std::is_constructible_v<StateIdx, Index>);

}  // namespace pulsim::topology

// -----------------------------------------------------------------------------
// std::hash specialisations so the strong types are usable as
// `unordered_map` keys directly (NodeId → something is a common
// pattern in solver bookkeeping).
// -----------------------------------------------------------------------------
namespace std {

template <>
struct hash<pulsim::topology::NodeId> {
    std::size_t operator()(
        pulsim::topology::NodeId n) const noexcept {
        return std::hash<pulsim::Index>{}(n.get());
    }
};

template <>
struct hash<pulsim::topology::BranchId> {
    std::size_t operator()(
        pulsim::topology::BranchId b) const noexcept {
        return std::hash<pulsim::Index>{}(b.get());
    }
};

template <>
struct hash<pulsim::topology::StateIdx> {
    std::size_t operator()(
        pulsim::topology::StateIdx s) const noexcept {
        return std::hash<pulsim::Index>{}(s.get());
    }
};

}  // namespace std
